from __future__ import annotations

import os, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from iox2_jsonrpc import EmptyParams, RpcModel

_THIS_DIR = Path(__file__).absolute().parent
for path in (
    _THIS_DIR,
    _THIS_DIR.parent,
    Path(os.path.dirname(os.path.dirname(_THIS_DIR.parent))),
):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.append(path_text)

from pcd_utils import DEFAULT_CALIBRATION, StereoRgbCalibration, read_image, stereo_rgb_to_colored_point_cloud  # noqa: E402
from pcd_dnn_utils import FastFoundationStereoDisparity, stereo_rgb_to_colored_point_cloud_dnn  # noqa: E402

Resolution = tuple[int, int]
ColorOrder = Literal["RGB", "BGR"]
DepthBackend = Literal["sgbm", "dnn"]
OutputFrame = Literal["left", "left_rectified"]
TranslationUnit = Literal["m", "cm", "mm"]
Matrix3x3 = tuple[tuple[float, float, float], ...]
Matrix4x4 = tuple[tuple[float, float, float, float], ...]
DistortionCoefficients = tuple[float, ...]

_DEFAULT_CALIBRATION = DEFAULT_CALIBRATION
_DNN_CACHE_KEY_FIELDS = ("repo_dir", "model_path", "model_dir", "device", "valid_iters", "max_disp", "hiera")


def _as_resolution(value: Any) -> Resolution:
    return int(value[0]), int(value[1])


def _as_matrix(value: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _as_float_tuple(value: Any) -> DistortionCoefficients:
    return tuple(float(item) for item in value)


def _default_calibration_field(key: str, converter: Any = _as_matrix) -> Any:
    return Field(default=converter(_DEFAULT_CALIBRATION[key]))


def _model_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)

    for method_name in ("model_dump", "dict"):
        if hasattr(value, method_name):
            return getattr(value, method_name)()

    raise TypeError(f"Expected a dict/RpcModel-compatible object, got {type(value)!r}")


def _model_field_names(model_type: type[BaseModel]) -> tuple[str, ...]:
    fields = getattr(model_type, "model_fields", None) or getattr(model_type, "__fields__", {})
    return tuple(fields.keys())


def _depth_statistics(points_m: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if not points_m.size:
        return None, None, None

    depth_values = np.asarray(points_m, dtype=np.float64)[:, 2]
    depth_values = depth_values[np.isfinite(depth_values)]
    if not depth_values.size:
        return None, None, None

    return tuple(float(value) for value in (depth_values.min(), depth_values.max(), depth_values.mean()))


def _read_image_or_npy(path: str | Path, *, color: bool) -> np.ndarray:
    image_path = Path(path).expanduser()

    if image_path.suffix.lower() == ".npy":
        array = np.load(image_path, allow_pickle=False)

        if color:
            if array.ndim != 3 or array.shape[2] < 3:
                raise ValueError(
                    f"Expected color .npy image with shape HxWx3 or HxWx4, "
                    f"got {array.shape} from {image_path}"
                )
            array = array[:, :, :3]
        else:
            if array.ndim == 2:
                pass
            elif array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]
            else:
                raise ValueError(
                    f"Expected grayscale .npy image with shape HxW or HxWx1, "
                    f"got {array.shape} from {image_path}"
                )

        return np.ascontiguousarray(array)

    return read_image(image_path, color=color)


def _save_cloud_npz(path: str | Path, cloud: Any) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "points_m": np.asarray(cloud.points_m),
        "colors_rgb": np.asarray(cloud.colors_rgb),
    }

    if cloud.disparity is not None:
        arrays["disparity"] = np.asarray(cloud.disparity)

    np.savez(output_path, **arrays)
    return output_path


class DepthBaseModel(RpcModel):
    service: Literal["serverDepth"] = "serverDepth"


class DepthCalibrationParams(DepthBaseModel):
    source_translation_unit: TranslationUnit = "cm"
    rgb_resolution: Resolution = _default_calibration_field("rgb_resolution", _as_resolution)
    left_resolution: Resolution = _default_calibration_field("left_resolution", _as_resolution)
    right_resolution: Resolution = _default_calibration_field("right_resolution", _as_resolution)
    rgb_intrinsics: Matrix3x3 = _default_calibration_field("rgb_intrinsics")
    left_intrinsics: Matrix3x3 = _default_calibration_field("left_intrinsics")
    right_intrinsics: Matrix3x3 = _default_calibration_field("right_intrinsics")
    left_to_right_extrinsics: Matrix4x4 = _default_calibration_field("left_to_right_extrinsics")
    left_to_rgb_extrinsics: Matrix4x4 = _default_calibration_field("left_to_rgb_extrinsics")
    rgb_distortion: DistortionCoefficients = _default_calibration_field("rgb_distortion", _as_float_tuple)
    left_distortion: DistortionCoefficients = _default_calibration_field("left_distortion", _as_float_tuple)
    right_distortion: DistortionCoefficients = _default_calibration_field("right_distortion", _as_float_tuple)


class SetDepthCalibrationResult(DepthBaseModel):
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


BACKEND_KEYS = _model_field_names(BackendOverrides)


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
    left_path: str
    right_path: str
    rgb_path: str
    output_path: str = "colored_cloud.pcd"
    calibration: DepthCalibrationParams | None = None
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
    size_bytes: int
    depth_min_m: float | None = None
    depth_max_m: float | None = None
    depth_mean_m: float | None = None
    disparity_width: int | None = None
    disparity_height: int | None = None


@dataclass
class DepthController:
    service_name: str = "serverDepth"
    controller_name: str = "depth"
    calibration_params: DepthCalibrationParams = field(default_factory=DepthCalibrationParams)
    backend_params: BackendParams = field(default_factory=BackendParams)
    _dnn_predictor: FastFoundationStereoDisparity | None = field(default=None, init=False, repr=False)
    _dnn_predictor_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)

    def _build_calibration(self, params: DepthCalibrationParams | None = None) -> StereoRgbCalibration:
        calibration_data = _model_to_dict(params or self.calibration_params)
        calibration_data.pop("service", None)
        translation_unit = calibration_data.pop("source_translation_unit", "cm")
        return StereoRgbCalibration.from_dict(calibration_data, source_translation_unit=translation_unit)

    def _calibration_result(self, calibration: StereoRgbCalibration) -> SetDepthCalibrationResult:
        result_fields = (
            "source_translation_unit",
            "rgb_resolution",
            "left_resolution",
            "right_resolution",
            "stereo_baseline_m",
            "stereo_baseline_cm",
        )
        return SetDepthCalibrationResult(
            configured=True,
            **{field_name: getattr(calibration, field_name) for field_name in result_fields},
        )

    def _backend_result(self) -> BackendStatusResult:
        return BackendStatusResult(
            configured=True,
            predictor_loaded=self._dnn_predictor is not None,
            **{key: getattr(self.backend_params, key) for key in BACKEND_KEYS},
        )

    def _dnn_cache_key(self, backend: BackendParams) -> tuple[Any, ...]:
        values = [getattr(backend, key) for key in _DNN_CACHE_KEY_FIELDS]
        values[4] = int(values[4])
        values[5] = int(values[5])
        values[6] = bool(values[6])
        return tuple(values)

    def _get_dnn_predictor(self, backend: BackendParams) -> FastFoundationStereoDisparity:
        cache_key = self._dnn_cache_key(backend)
        if self._dnn_predictor is None or self._dnn_predictor_key != cache_key:
            self._dnn_predictor = FastFoundationStereoDisparity(
                repo_dir=backend.repo_dir,
                model_path=backend.model_path,
                model_dir=backend.model_dir,
                device=backend.device,
                valid_iters=int(backend.valid_iters),
                max_disp=int(backend.max_disp),
                hiera=bool(backend.hiera),
            )
            self._dnn_predictor_key = cache_key

        return self._dnn_predictor

    def _effective_backend(self, params: ToPcdParams) -> BackendParams:
        backend_data = {key: getattr(self.backend_params, key) for key in BACKEND_KEYS}
        override_data = _model_to_dict(params)
        backend_data.update({key: override_data[key] for key in BACKEND_KEYS if override_data.get(key) is not None})
        return BackendParams(**backend_data)

    def set_calibration(self, params: DepthCalibrationParams) -> SetDepthCalibrationResult:
        self.calibration_params = params
        return self._calibration_result(self._build_calibration(params))

    def calibration(self, params: EmptyParams) -> SetDepthCalibrationResult:
        del params
        return self._calibration_result(self._build_calibration())

    def set_backend(self, params: BackendParams) -> BackendStatusResult:
        old_cache_key = self._dnn_cache_key(self.backend_params)
        self.backend_params = params
        if old_cache_key != self._dnn_cache_key(params):
            self._dnn_predictor = None
            self._dnn_predictor_key = None
        return self._backend_result()

    def backend(self, params: EmptyParams) -> BackendStatusResult:
        del params
        return self._backend_result()

    def to_pcd(self, params: ToPcdParams) -> ToPcdResult:
        output_path = Path(params.output_path).expanduser()
        output_suffix = output_path.suffix.lower()

        if output_suffix not in {".pcd", ".npz"}:
            raise ValueError(f"output_path must end with .pcd or .npz, got: {output_path}")

        backend = self._effective_backend(params)
        conversion_args = dict(
            left_image=_read_image_or_npy(params.left_path, color=False),
            right_image=_read_image_or_npy(params.right_path, color=False),
            rgb_image=_read_image_or_npy(params.rgb_path, color=True),
            calibration=self._build_calibration(params.calibration),
            output_path=output_path if output_suffix == ".pcd" else None,
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
            cloud = stereo_rgb_to_colored_point_cloud(
                **conversion_args,
                num_disparities=params.num_disparities,
                block_size=params.block_size,
            )
        elif backend.backend == "dnn":
            cloud = stereo_rgb_to_colored_point_cloud_dnn(
                **conversion_args,
                disparity_predictor=self._get_dnn_predictor(backend),
                model_scale=float(backend.model_scale),
                stereo_input_color_order=backend.stereo_input_color_order,
                remove_invisible=backend.remove_invisible,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend.backend}")

        if output_suffix == ".npz":
            _save_cloud_npz(output_path, cloud)

        depth_min_m, depth_max_m, depth_mean_m = _depth_statistics(cloud.points_m)
        disparity_height, disparity_width = (None, None) if cloud.disparity is None else cloud.disparity.shape[:2]

        return ToPcdResult(
            backend=backend.backend,
            output_path=str(output_path),
            point_count=int(cloud.points_m.shape[0]),
            color_count=int(cloud.colors_rgb.shape[0]),
            size_bytes=int(output_path.stat().st_size),
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            depth_mean_m=depth_mean_m,
            disparity_width=disparity_width,
            disparity_height=disparity_height,
        )


def run_server(controller_name: str = "depth") -> None:
    from iox2_jsonrpc.iceoryx import Iox2JsonRpcServer

    Iox2JsonRpcServer(DepthController(controller_name=controller_name)).run_forever()
