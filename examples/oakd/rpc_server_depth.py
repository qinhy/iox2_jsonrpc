from __future__ import annotations

import os, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from iox2_jsonrpc import EmptyParams, RpcModel

_THIS_DIR = Path(__file__).absolute().parent
for p in (_THIS_DIR, _THIS_DIR.parent, Path(os.path.dirname(os.path.dirname(_THIS_DIR.parent)))):
    if (s := str(p)) not in sys.path:
        sys.path.append(s)

from pcd_utils import DEFAULT_CALIBRATION, StereoRgbCalibration, read_image, stereo_rgb_to_colored_point_cloud  # noqa: E402
from pcd_dnn_utils import FastFoundationStereoDisparity, stereo_rgb_to_colored_point_cloud_dnn  # noqa: E402

Resolution = tuple[int, int]
ColorOrder = Literal["RGB", "BGR"]
DepthBackend = Literal["sgbm", "dnn"]
OutputFrame = Literal["left", "left_rectified"]
TranslationUnit = Literal["m", "cm", "mm"]
Matrix3x3 = tuple[tuple[float, float, float], ...]
Matrix4x4 = tuple[tuple[float, float, float, float], ...]
DistortionCoeffs = tuple[float, ...]

C = DEFAULT_CALIBRATION

def _res(v: Any) -> Resolution:
    return int(v[0]), int(v[1])


def _mat(v: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in v)


def _floats(v: Any) -> DistortionCoeffs:
    return tuple(float(x) for x in v)


def _field(key: str, fn: Any = _mat) -> Any:
    return Field(default=fn(C[key]))


def _dump(v: Any) -> dict[str, Any]:
    if v is None:
        return {}
    if isinstance(v, dict):
        return dict(v)
    for name in ("model_dump", "dict"):
        if hasattr(v, name):
            return getattr(v, name)()
    raise TypeError(f"Expected a dict/RpcModel-compatible object, got {type(v)!r}")


def _depth_stats(points_m: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if not points_m.size:
        return None, None, None
    z = np.asarray(points_m, dtype=np.float64)[:, 2]
    z = z[np.isfinite(z)]
    return (None, None, None) if not z.size else tuple(float(x) for x in (z.min(), z.max(), z.mean()))


class DepthBaseModel(RpcModel):
    service: Literal["serverDepth"] = "serverDepth"


class CameraInfoParams(DepthBaseModel):
    source_translation_unit: TranslationUnit = "cm"
    rgb_resolution: Resolution = _field("rgb_resolution", _res)
    left_resolution: Resolution = _field("left_resolution", _res)
    right_resolution: Resolution = _field("right_resolution", _res)
    rgb_intrinsics: Matrix3x3 = _field("rgb_intrinsics")
    left_intrinsics: Matrix3x3 = _field("left_intrinsics")
    right_intrinsics: Matrix3x3 = _field("right_intrinsics")
    left_to_right_extrinsics: Matrix4x4 = _field("left_to_right_extrinsics")
    left_to_rgb_extrinsics: Matrix4x4 = _field("left_to_rgb_extrinsics")
    rgb_distortion: DistortionCoeffs = _field("rgb_distortion", _floats)
    left_distortion: DistortionCoeffs = _field("left_distortion", _floats)
    right_distortion: DistortionCoeffs = _field("right_distortion", _floats)


class SetCameraInfoResult(DepthBaseModel):
    configured: bool
    source_translation_unit: TranslationUnit
    rgb_resolution: Resolution
    left_resolution: Resolution
    right_resolution: Resolution
    stereo_baseline_m: float
    stereo_baseline_cm: float


class BackendOverrides(BaseModel):
    backend: DepthBackend | None = None
    repo_dir: str | None = None
    model_path: str | None = None
    model_dir: str | None = None
    device: str | None = None
    valid_iters: int | None = None
    max_disp: int | None = None
    hiera: bool | None = None
    model_scale: float | None = None
    stereo_input_color_order: ColorOrder | None = None
    remove_invisible: bool | None = None

BACKEND_KEYS = BackendOverrides.model_fields.keys()

class BackendParams(BackendOverrides):
    backend: DepthBackend = "sgbm"
    device: str = "cuda"
    valid_iters: int = 8
    max_disp: int = 192
    hiera: bool = False
    model_scale: float = 1.0
    stereo_input_color_order: ColorOrder = "RGB"
    remove_invisible: bool = True


class BackendStatusResult(BackendParams):
    configured: bool = True
    predictor_loaded: bool = False


class ToPcdParams(BackendOverrides):
    left_path: str; right_path: str; rgb_path: str
    output_path: str = "colored_cloud.pcd"
    camera_calib: CameraInfoParams | None = None
    input_color_order: ColorOrder = "BGR"
    rgb_image_is_undistorted: bool = False
    alpha: float = 0.0
    max_depth_m: float | None = 10.0
    stride: int = Field(default=1, ge=1)
    output_frame: OutputFrame = "left"
    save_binary_pcd: bool = True
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5


class ToPcdResult(DepthBaseModel):
    backend: DepthBackend
    output_path: str
    point_count: int
    color_count: int
    pcd_size_bytes: int
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    depth_mean_m: float | None = None
    disparity_width: int | None = None
    disparity_height: int | None = None


@dataclass
class DepthController:
    service_name: str = "serverDepth"
    controller_name: str = "depth"
    camera_info_params: CameraInfoParams = field(default_factory=CameraInfoParams)
    backend_params: BackendParams = field(default_factory=BackendParams)
    _dnn_predictor: FastFoundationStereoDisparity | None = field(default=None, init=False, repr=False)
    _dnn_predictor_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)

    def _calibration(self, camera_info: CameraInfoParams | None = None) -> StereoRgbCalibration:
        data = _dump(camera_info or self.camera_info_params)
        data.pop("service", None)
        return StereoRgbCalibration.from_dict(data, source_translation_unit=data.pop("source_translation_unit", "cm"))

    def _camera_info_result(self, calib: StereoRgbCalibration) -> SetCameraInfoResult:
        keys = ("source_translation_unit", "rgb_resolution",
                "left_resolution", "right_resolution", "stereo_baseline_m", "stereo_baseline_cm")
        return SetCameraInfoResult(configured=True, **{k: getattr(calib, k) for k in keys})

    def _backend_result(self) -> BackendStatusResult:
        return BackendStatusResult(
            configured=True,
            predictor_loaded=self._dnn_predictor is not None,
            **{k: getattr(self.backend_params, k) for k in BACKEND_KEYS},
        )

    def _dnn_key(self, backend: BackendParams) -> tuple[Any, ...]:        
        DNN_KEY_FIELDS = ("repo_dir", "model_path", "model_dir", "device", "valid_iters", "max_disp", "hiera")
        vals = [getattr(backend, k) for k in DNN_KEY_FIELDS]
        vals[4], vals[5], vals[6] = int(vals[4]), int(vals[5]), bool(vals[6])
        return tuple(vals)

    def _get_dnn_predictor(self, backend: BackendParams) -> FastFoundationStereoDisparity:
        key = self._dnn_key(backend)
        if self._dnn_predictor is None or self._dnn_predictor_key != key:
            self._dnn_predictor = FastFoundationStereoDisparity(
                repo_dir=backend.repo_dir,
                model_path=backend.model_path,
                model_dir=backend.model_dir,
                device=backend.device,
                valid_iters=int(backend.valid_iters),
                max_disp=int(backend.max_disp),
                hiera=bool(backend.hiera),
            )
            self._dnn_predictor_key = key
        return self._dnn_predictor

    def _effective_backend(self, params: ToPcdParams) -> BackendParams:
        data = {k: getattr(self.backend_params, k) for k in BACKEND_KEYS}
        overrides = _dump(params)
        data.update({k: overrides[k] for k in BACKEND_KEYS if overrides.get(k) is not None})
        return BackendParams(**data)

    def set_camera_info(self, params: CameraInfoParams) -> SetCameraInfoResult:
        self.camera_info_params = params
        return self._camera_info_result(self._calibration(params))

    def camera_info(self, params: EmptyParams) -> SetCameraInfoResult:
        del params
        return self._camera_info_result(self._calibration())

    def set_backend(self, params: BackendParams) -> BackendStatusResult:
        old_key = self._dnn_key(self.backend_params)
        self.backend_params = params
        if old_key != self._dnn_key(params):
            self._dnn_predictor = self._dnn_predictor_key = None
        return self._backend_result()

    def backend(self, params: EmptyParams) -> BackendStatusResult:
        del params
        return self._backend_result()

    def to_pcd(self, params: ToPcdParams) -> ToPcdResult:
        output_path = Path(params.output_path)
        if output_path.suffix.lower() != ".pcd":
            raise ValueError(f"output_path must end with .pcd, got: {output_path}")

        backend = self._effective_backend(params)
        common = dict(
            left_image=read_image(params.left_path, color=False),
            right_image=read_image(params.right_path, color=False),
            rgb_image=read_image(params.rgb_path, color=True),
            calibration=self._calibration(params.camera_calib),
            output_path=output_path,
            min_disparity=float(params.min_disparity),
            max_depth_m=params.max_depth_m,
            stride=params.stride,
            output_frame=params.output_frame,
            save_binary_pcd=params.save_binary_pcd,
            input_color_order=params.input_color_order,
            alpha=params.alpha,
            rgb_image_is_undistorted=params.rgb_image_is_undistorted,
        )
        if backend.backend == "sgbm":
            cloud = stereo_rgb_to_colored_point_cloud(**common,
                num_disparities=params.num_disparities, block_size=params.block_size)
        elif backend.backend == "dnn":
            cloud = stereo_rgb_to_colored_point_cloud_dnn(**common,
                disparity_predictor=self._get_dnn_predictor(backend),
                model_scale=float(backend.model_scale),
                stereo_input_color_order=backend.stereo_input_color_order,
                remove_invisible=backend.remove_invisible,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend.backend}")

        depth_min_m, depth_max_m, depth_mean_m = _depth_stats(cloud.points_m)
        h, w = (None, None) if cloud.disparity is None else cloud.disparity.shape[:2]
        return ToPcdResult(
            backend=backend.backend,
            output_path=str(output_path),
            point_count=int(cloud.points_m.shape[0]),
            color_count=int(cloud.colors_rgb.shape[0]),
            pcd_size_bytes=int(output_path.stat().st_size),
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            depth_mean_m=depth_mean_m,
            disparity_width=w,
            disparity_height=h,
        )


def run_server() -> None:
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    Iox2JsonRpcServer(DepthController()).run_forever()


if __name__ == "__main__":
    run_server()
