"""
pcd_dnn_utils.py

DNN-based stereo helpers that extend pcd_utils.py with a Fast-FoundationStereo
disparity backend.

This file intentionally does:

    from pcd_utils import *

so your existing pcd_utils.py API remains available, while this module adds:

    - FastFoundationStereoDisparity
    - compute_disparity_fast_foundationstereo()
    - stereo_rgb_to_colored_point_cloud_dnn()
    - write_fast_foundationstereo_intrinsic_file()

Typical usage:

    from pcd_dnn_utils import (
        read_image,
        FastFoundationStereoDisparity,
        stereo_rgb_to_colored_point_cloud_dnn,
    )

    left = read_image("left.png", color=False)
    right = read_image("right.png", color=False)
    rgb = read_image("rgb.png", color=True)  # cv2 -> BGR

    predictor = FastFoundationStereoDisparity(
        repo_dir="/path/to/Fast-FoundationStereo",
        model_path="/path/to/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth",
        valid_iters=8,
        max_disp=192,
    )

    cloud = stereo_rgb_to_colored_point_cloud_dnn(
        left,
        right,
        rgb,
        disparity_predictor=predictor,
        output_path="colored_cloud.pcd",
        input_color_order="BGR",
        max_depth_m=8.0,
    )

Requirements for this module:
    pip install numpy opencv-python torch pyyaml omegaconf imageio

You must also clone/install Fast-FoundationStereo and download its weights.
The wrapper calls the PyTorch checkpoint path used by scripts/run_demo.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import contextlib
import os
import sys
import warnings

import numpy as np

# Re-export your original utility API.
import pcd_utils as _pcd_utils
from pcd_utils import *  # noqa: F401,F403


ColorOrder = Literal["RGB", "BGR"]
DeviceLike = str | Any


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required. Install it with:\n"
            "    pip install opencv-python"
        ) from exc
    return cv2


def _require_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for Fast-FoundationStereo inference. Install a CUDA build "
            "matching your system, for example from the official PyTorch selector."
        ) from exc
    return torch


def _resolve_repo_dir(
    repo_dir: str | Path | None,
    model_path: str | Path | None,
) -> Path:
    """Find a local Fast-FoundationStereo checkout."""
    candidates: list[Path] = []

    if repo_dir is not None:
        candidates.append(Path(repo_dir).expanduser())

    env_repo = os.environ.get("FAST_FOUNDATIONSTEREO_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser())

    if model_path is not None:
        p = Path(model_path).expanduser().resolve()
        # Common layout:
        #   Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth
        for parent in [p.parent, *p.parents]:
            if parent.name == "Fast-FoundationStereo":
                candidates.append(parent)
                break
        # Also try two levels up from weights/<checkpoint-dir>/model.pth.
        if len(p.parents) >= 3:
            candidates.append(p.parents[2])

    candidates.append(Path.cwd())

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "scripts" / "run_demo.py").exists() and (candidate / "core").exists():
            return candidate

    msg = [
        "Could not locate a Fast-FoundationStereo checkout.",
        "Pass repo_dir='/path/to/Fast-FoundationStereo' or set FAST_FOUNDATIONSTEREO_REPO.",
        "Expected to find scripts/run_demo.py and core/ under repo_dir.",
    ]
    raise FileNotFoundError("\n".join(msg))


def _resolve_model_path(
    repo_dir: Path,
    model_path: str | Path | None,
    model_dir: str | Path | None,
) -> Path:
    """Resolve a Fast-FoundationStereo PyTorch checkpoint path."""
    if model_path is not None:
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = (repo_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Fast-FoundationStereo model_path does not exist: {path}")
        return path

    search_dirs: list[Path] = []

    if model_dir is not None:
        d = Path(model_dir).expanduser()
        if not d.is_absolute():
            d = (repo_dir / d).resolve()
        search_dirs.append(d)

    search_dirs.extend(
        [
            repo_dir / "weights" / "23-36-37",
            repo_dir / "weights",
        ]
    )

    patterns = [
        "model_best_bp2_serialize.pth",
        "model_best*.pth",
        "*.pth",
        "*.pt",
    ]

    for d in search_dirs:
        if d.is_file():
            return d.resolve()
        if not d.exists():
            continue
        for pattern in patterns:
            matches = sorted(d.glob(pattern))
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        "Could not find a Fast-FoundationStereo checkpoint. Pass model_path=... "
        "or model_dir=... explicitly."
    )


def _ensure_uint8_rgb_image(
    image: np.ndarray,
    *,
    input_color_order: ColorOrder = "RGB",
) -> np.ndarray:
    """Convert HxW/HxWx1/HxWx3 image to uint8 RGB HxWx3."""
    array = np.asarray(image)

    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] >= 3:
        array = array[:, :, :3]
        if input_color_order == "BGR":
            array = array[:, :, ::-1]
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.max()) <= 1.0:
            array = array * 255.0
        return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))

    if array.dtype == np.uint16:
        max_value = int(array.max())
        if max_value == 0:
            return np.zeros(array.shape, dtype=np.uint8)
        return np.ascontiguousarray(
            np.clip(array.astype(np.float32) * (255.0 / max_value), 0, 255).astype(np.uint8)
        )

    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _resize_pair_for_model(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    *,
    model_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if model_scale <= 0:
        raise ValueError(f"model_scale must be positive, got {model_scale}")

    if model_scale == 1.0:
        return left_rgb, right_rgb

    cv2 = _require_cv2()
    left_scaled = cv2.resize(
        left_rgb,
        dsize=None,
        fx=float(model_scale),
        fy=float(model_scale),
        interpolation=cv2.INTER_AREA if model_scale < 1.0 else cv2.INTER_LINEAR,
    )
    right_scaled = cv2.resize(
        right_rgb,
        dsize=(left_scaled.shape[1], left_scaled.shape[0]),
        interpolation=cv2.INTER_AREA if model_scale < 1.0 else cv2.INTER_LINEAR,
    )
    return left_scaled, right_scaled


def _restore_disparity_to_original_geometry(
    disparity_model_pixels: np.ndarray,
    *,
    original_hw: tuple[int, int],
    model_scale: float,
) -> np.ndarray:
    """Return disparity in original rectified-image pixel units.

    If the model ran on scale=s images, its disparity is in scaled pixels:
        d_scaled = s * d_original
    so after resizing to original H/W, divide by s.
    """
    cv2 = _require_cv2()

    disparity = np.asarray(disparity_model_pixels, dtype=np.float32)
    original_h, original_w = original_hw

    if disparity.shape[:2] != (original_h, original_w):
        disparity = cv2.resize(
            disparity,
            dsize=(original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

    if model_scale != 1.0:
        disparity = disparity / float(model_scale)

    disparity[~np.isfinite(disparity)] = np.nan
    disparity[disparity <= 0.0] = np.nan
    return disparity


@dataclass
class FastFoundationStereoDisparity:
    """PyTorch inference wrapper for NVlabs/Fast-FoundationStereo.

    This class is intentionally small: it loads the serialized model from the
    official repo and returns only a float32 disparity map in pixel units.

    Args:
        repo_dir:
            Local path to cloned Fast-FoundationStereo.
        model_path:
            Path to .pth checkpoint. Relative paths are resolved from repo_dir.
        model_dir:
            Optional checkpoint directory. Used only when model_path is not given.
        device:
            "cuda" is recommended. "cpu" may be unsupported/very slow for this repo.
        valid_iters:
            Number of refinement iterations. Lower is faster, higher is usually better.
        max_disp:
            Maximum disparity used by the model.
        hiera:
            Use hierarchical inference path if supported by the checkpoint.
        autocast:
            Enable CUDA AMP inference.
        amp_dtype:
            Optional torch dtype. If omitted, this wrapper tries Utils.AMP_DTYPE
            from the repo and falls back to torch.float16.
        optimize_build_volume:
            Forward kwarg used by the official demo. Kept configurable for repo changes.
    """

    repo_dir: str | Path | None = None
    model_path: str | Path | None = None
    model_dir: str | Path | None = None
    device: DeviceLike = "cuda"
    valid_iters: int = 8
    max_disp: int = 192
    hiera: bool = False
    autocast: bool = True
    amp_dtype: Any | None = None
    optimize_build_volume: str = "pytorch1"

    def __post_init__(self) -> None:
        self.repo_path = _resolve_repo_dir(self.repo_dir, self.model_path)
        self.checkpoint_path = _resolve_model_path(
            self.repo_path,
            self.model_path,
            self.model_dir,
        )

        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        torch = _require_torch()
        self.torch = torch

        try:
            from core.utils.utils import InputPadder  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Could not import core.utils.utils.InputPadder from Fast-FoundationStereo. "
                f"repo_dir appears to be: {self.repo_path}"
            ) from exc

        self.InputPadder = InputPadder

        if self.amp_dtype is None:
            try:
                from Utils import AMP_DTYPE  # type: ignore

                self.amp_dtype = AMP_DTYPE
            except Exception:
                self.amp_dtype = torch.float16

        if isinstance(self.device, str):
            if self.device == "cuda" and not torch.cuda.is_available():
                warnings.warn(
                    "device='cuda' requested but CUDA is not available. Falling back to CPU. "
                    "Fast-FoundationStereo is designed for NVIDIA GPU inference.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.device = "cpu"

        self.model = self._load_model()

    def _load_model(self):
        torch = self.torch

        # weights_only=False matches the official demo because the checkpoint is
        # a serialized model object rather than a simple state_dict.
        model = torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )

        # The official demo overrides these after loading the serialized model.
        if hasattr(model, "args"):
            try:
                model.args.valid_iters = int(self.valid_iters)
            except Exception:
                pass
            try:
                model.args.max_disp = int(self.max_disp)
            except Exception:
                pass

        model.to(self.device)
        model.eval()
        return model

    @property
    def device_type(self) -> str:
        if isinstance(self.device, str):
            return "cuda" if self.device.startswith("cuda") else self.device
        return getattr(self.device, "type", "cuda")

    def predict(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        *,
        input_color_order: ColorOrder = "RGB",
        model_scale: float = 1.0,
        remove_invisible: bool = True,
    ) -> np.ndarray:
        """Predict disparity for a rectified stereo pair.

        Args:
            left_rectified, right_rectified:
                Rectified/undistorted stereo images with horizontal epipolar lines.
            input_color_order:
                Color order of input images if they are 3-channel.
            model_scale:
                Optional scale applied before inference. Output is restored to the
                original image size and original pixel-disparity units.
                Example: model_scale=0.5 runs faster on 1280x800 input.
            remove_invisible:
                Mark pixels whose corresponding right-camera u coordinate would be
                outside the image as invalid NaN.

        Returns:
            HxW float32 disparity in original rectified-image pixel units.
            Invalid pixels are NaN.
        """
        torch = self.torch

        if left_rectified is None or right_rectified is None:
            raise ValueError("left_rectified and right_rectified are required")

        if np.asarray(left_rectified).shape[:2] != np.asarray(right_rectified).shape[:2]:
            raise ValueError(
                "left_rectified and right_rectified must have the same height/width, "
                f"got {np.asarray(left_rectified).shape[:2]} and {np.asarray(right_rectified).shape[:2]}"
            )

        original_h, original_w = np.asarray(left_rectified).shape[:2]

        left_rgb = _ensure_uint8_rgb_image(
            left_rectified,
            input_color_order=input_color_order,
        )
        right_rgb = _ensure_uint8_rgb_image(
            right_rectified,
            input_color_order=input_color_order,
        )

        left_model, right_model = _resize_pair_for_model(
            left_rgb,
            right_rgb,
            model_scale=float(model_scale),
        )

        h_model, w_model = left_model.shape[:2]

        img0 = (
            torch.as_tensor(left_model, device=self.device)
            .float()[None]
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        img1 = (
            torch.as_tensor(right_model, device=self.device)
            .float()[None]
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        use_amp = bool(self.autocast and self.device_type == "cuda")

        with torch.inference_mode():
            if use_amp:
                autocast_ctx = torch.amp.autocast(
                    "cuda",
                    enabled=True,
                    dtype=self.amp_dtype,
                )
            else:
                autocast_ctx = contextlib.nullcontext()

            with autocast_ctx:
                if self.hiera:
                    if not hasattr(self.model, "run_hierachical"):
                        raise AttributeError(
                            "This checkpoint/model does not expose run_hierachical(). "
                            "Set hiera=False."
                        )
                    disp = self.model.run_hierachical(
                        img0,
                        img1,
                        iters=int(self.valid_iters),
                        test_mode=True,
                        small_ratio=0.5,
                    )
                else:
                    try:
                        disp = self.model.forward(
                            img0,
                            img1,
                            iters=int(self.valid_iters),
                            test_mode=True,
                            optimize_build_volume=self.optimize_build_volume,
                        )
                    except TypeError:
                        # Some future/alternate checkpoints may not use optimize_build_volume.
                        disp = self.model.forward(
                            img0,
                            img1,
                            iters=int(self.valid_iters),
                            test_mode=True,
                        )

        disp = padder.unpad(disp.float())
        disp_np = disp.detach().cpu().numpy().reshape(h_model, w_model).astype(np.float32)
        disp_np = np.clip(disp_np, 0.0, None)

        if remove_invisible:
            yy, xx = np.meshgrid(
                np.arange(disp_np.shape[0], dtype=np.float32),
                np.arange(disp_np.shape[1], dtype=np.float32),
                indexing="ij",
            )
            del yy
            right_u = xx - disp_np
            disp_np[right_u < 0.0] = np.nan

        return _restore_disparity_to_original_geometry(
            disp_np,
            original_hw=(original_h, original_w),
            model_scale=float(model_scale),
        )


def compute_disparity_fast_foundationstereo(
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
    *,
    predictor: FastFoundationStereoDisparity | None = None,
    repo_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    device: DeviceLike = "cuda",
    valid_iters: int = 8,
    max_disp: int = 192,
    hiera: bool = False,
    model_scale: float = 1.0,
    input_color_order: ColorOrder = "RGB",
    remove_invisible: bool = True,
) -> np.ndarray:
    """Convenience function for one-off Fast-FoundationStereo disparity.

    For repeated/live use, create FastFoundationStereoDisparity once and pass it
    as predictor=... to avoid reloading the model every frame.
    """
    if predictor is None:
        predictor = FastFoundationStereoDisparity(
            repo_dir=repo_dir,
            model_path=model_path,
            model_dir=model_dir,
            device=device,
            valid_iters=valid_iters,
            max_disp=max_disp,
            hiera=hiera,
        )

    return predictor.predict(
        left_rectified,
        right_rectified,
        input_color_order=input_color_order,
        model_scale=model_scale,
        remove_invisible=remove_invisible,
    )


def stereo_rgb_to_colored_point_cloud_dnn(
    left_image: np.ndarray,
    right_image: np.ndarray,
    rgb_image: np.ndarray,
    *,
    calibration: StereoRgbCalibration | None = None,  # noqa: F405
    disparity_predictor: FastFoundationStereoDisparity | None = None,
    repo_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    device: DeviceLike = "cuda",
    valid_iters: int = 8,
    max_disp: int = 192,
    hiera: bool = False,
    model_scale: float = 1.0,
    output_path: str | Path | None = None,
    input_color_order: ColorOrder = "RGB",
    stereo_input_color_order: ColorOrder = "RGB",
    rgb_image_is_undistorted: bool = False,
    alpha: float = 0.0,
    min_disparity: float = 0.5,
    max_depth_m: float | None = 10.0,
    stride: int = 1,
    output_frame: Literal["left", "left_rectified"] = "left",
    save_binary_pcd: bool = True,
    remove_invisible: bool = True,
) -> ColoredPointCloud:  # noqa: F405
    """Full DNN pipeline: stereo + RGB -> colored PCD/PLY.

    This is the DNN sibling of pcd_utils.stereo_rgb_to_colored_point_cloud().
    It uses your existing pcd_utils rectification, reprojection, RGB colorization,
    and file saving, but swaps OpenCV SGBM for Fast-FoundationStereo.

    Args:
        left_image, right_image:
            Raw synchronized stereo images.
        rgb_image:
            Synchronized color image used for final point colors.
        calibration:
            Defaults to StereoRgbCalibration.default().
        disparity_predictor:
            Reusable FastFoundationStereoDisparity instance. Recommended for live use.
        repo_dir/model_path/model_dir/device/valid_iters/max_disp/hiera:
            Used only when disparity_predictor is not supplied.
        model_scale:
            Optional resize before DNN inference. The returned disparity is corrected
            back to original rectified-image geometry.
        input_color_order:
            Color order of rgb_image for final PCD colors. Use "BGR" for cv2.imread().
        stereo_input_color_order:
            Color order of stereo images if they are 3-channel.
        rgb_image_is_undistorted:
            Set True only if the RGB image has already been undistorted.
        max_depth_m:
            Reject farther points.
        stride:
            Keep every Nth pixel for smaller clouds.
        output_frame:
            "left" is recommended.
    """
    calibration = calibration or StereoRgbCalibration.default()  # noqa: F405

    left_rect, right_rect, rectification = rectify_stereo_pair(  # noqa: F405
        left_image,
        right_image,
        calibration,
        alpha=alpha,
    )

    if disparity_predictor is None:
        disparity_predictor = FastFoundationStereoDisparity(
            repo_dir=repo_dir,
            model_path=model_path,
            model_dir=model_dir,
            device=device,
            valid_iters=valid_iters,
            max_disp=max_disp,
            hiera=hiera,
        )

    disparity = disparity_predictor.predict(
        left_rect,
        right_rect,
        input_color_order=stereo_input_color_order,
        model_scale=model_scale,
        remove_invisible=remove_invisible,
    )

    points_rectified, _pixel_xy = disparity_to_points_rectified(  # noqa: F405
        disparity,
        rectification,
        min_disparity=float(min_disparity),
        max_depth_m=max_depth_m,
        stride=stride,
    )

    points, colors = colorize_points_from_rgb(  # noqa: F405
        points_rectified,
        rgb_image,
        calibration,
        rectification=rectification,
        points_frame="left_rectified",
        output_frame=output_frame,
        input_color_order=input_color_order,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
    )

    if output_path is not None:
        save_point_cloud(  # noqa: F405
            output_path,
            points,
            colors,
            binary_pcd=save_binary_pcd,
        )

    return ColoredPointCloud(  # noqa: F405
        points_m=points,
        colors_rgb=colors,
        disparity=disparity,
        rectification=rectification,
    )


def write_fast_foundationstereo_intrinsic_file(
    path: str | Path,
    rectification: StereoRectification,  # noqa: F405
) -> Path:
    """Write a K.txt compatible with Fast-FoundationStereo's demo script.

    The file format is:
        line 1: flattened 3x3 intrinsic matrix
        line 2: stereo baseline in meters

    This is useful when you want to run the official demo on already-rectified
    images produced by pcd_utils.rectify_stereo_pair().

    Note:
        Your pcd_dnn_utils pipeline does not need this file internally.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    K = np.asarray(rectification.P1[:3, :3], dtype=np.float64)
    fx = float(rectification.P1[0, 0])
    if fx == 0:
        raise ValueError("Invalid rectification.P1: fx is zero")

    # OpenCV stereoRectify convention: P2[0, 3] = -fx * baseline for horizontal stereo.
    baseline_m = abs(float(rectification.P2[0, 3]) / fx)

    with path.open("w", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.12g}" for v in K.reshape(-1)) + "\n")
        f.write(f"{baseline_m:.12g}\n")

    return path


_DNN_PUBLIC_NAMES = [
    "FastFoundationStereoDisparity",
    "compute_disparity_fast_foundationstereo",
    "stereo_rgb_to_colored_point_cloud_dnn",
    "write_fast_foundationstereo_intrinsic_file",
]

# Preserve pcd_utils.py's public API when users do:
#     from pcd_dnn_utils import *
__all__ = list(getattr(_pcd_utils, "__all__", [])) + _DNN_PUBLIC_NAMES



if __name__ == "__main__":
    # left = read_image("left.png", color=False)
    # right = read_image("right.png", color=False)
    # rgb = read_image("rgb.png", color=True)  # cv2 reads BGR

    # predictor = FastFoundationStereoDisparity(
    #     repo_dir="/path/to/Fast-FoundationStereo",
    #     model_path="/path/to/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth",
    #     valid_iters=8,
    #     max_disp=192,
    # )

    # cloud = stereo_rgb_to_colored_point_cloud_dnn(
    #     left,
    #     right,
    #     rgb,
    #     disparity_predictor=predictor,
    #     output_path="colored_cloud.pcd",
    #     input_color_order="BGR",
    #     model_scale=0.5,      # useful for 1280x800 speed
    #     max_depth_m=8.0,
    #     stride=1,
    # )
    pass