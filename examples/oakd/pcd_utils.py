"""
pcd_utils.py

Utilities for converting synchronized Luxonis/OAK-style stereo + RGB frames into
a colored point cloud.

Expected input:
    - left mono image
    - right mono image
    - RGB/color image captured at the same time
    - calibration dictionary like the one supplied in this chat

Main entry point:
    stereo_rgb_to_colored_point_cloud(...)

Coordinate convention:
    - Stereo disparity is reprojected into the rectified-left camera frame.
    - Before color projection, points are rotated back into the original-left
      frame, then transformed into the RGB camera frame using left_to_rgb_extrinsics.
    - Saved point clouds are in the original-left camera frame by default.

Dependencies:
    pip install numpy opencv-python

Optional:
    pip install open3d
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import json
import math
import warnings

import numpy as np


ColorOrder = Literal["RGB", "BGR"]
PointsFrame = Literal["left", "left_rectified"]
OutputFrame = Literal["left", "left_rectified"]


DEFAULT_CALIBRATION: dict[str, Any] = {
    "rgb_resolution": [4056, 3040],
    "left_resolution": [1280, 800],
    "right_resolution": [1280, 800],
    "stereo_translation_units_hint": (
        "Calibration extrinsics units from device; Luxonis stereo baseline "
        "override API uses centimeters."
    ),
    "rgb_intrinsics": [
        [2430.31884765625, 0.0, 2063.196044921875],
        [0.0, 2429.41748046875, 1490.1956787109375],
        [0.0, 0.0, 1.0],
    ],
    "left_intrinsics": [
        [570.8507690429688, 0.0, 653.754150390625],
        [0.0, 570.580810546875, 390.99169921875],
        [0.0, 0.0, 1.0],
    ],
    "right_intrinsics": [
        [567.8758544921875, 0.0, 655.560546875],
        [0.0, 567.7424926757812, 393.97039794921875],
        [0.0, 0.0, 1.0],
    ],
    "left_to_right_extrinsics": [
        [
            0.9998766183853149,
            0.0023975009098649025,
            -0.015519456937909126,
            -7.537897109985352,
        ],
        [
            -0.0024368134327232838,
            0.9999938607215881,
            -0.002514647087082267,
            0.09707357734441757,
        ],
        [
            0.01551333349198103,
            0.0025521547067910433,
            0.9998764395713806,
            -0.08006280660629272,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "left_to_rgb_extrinsics": [
        [
            0.999745786190033,
            -0.010174884460866451,
            -0.020119857043027878,
            -3.7557337284088135,
        ],
        [
            0.010090984404087067,
            0.9999399781227112,
            -0.004267154261469841,
            -0.004705727566033602,
        ],
        [
            0.02016206830739975,
            0.004063040018081665,
            0.9997884631156921,
            -0.04603101313114166,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "rgb_distortion": [
        11.808209419250488,
        11.02328872680664,
        0.0005683265044353902,
        -0.0014364976668730378,
        -1.831695795059204,
        11.769153594970703,
        14.672675132751465,
        -1.088363766670227,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.00932407472282648,
        -0.015433108434081078,
    ],
    "left_distortion": [
        5.454817771911621,
        1.694711446762085,
        8.319105836562812e-05,
        -5.4938958783168346e-05,
        0.029059873893857002,
        5.82321310043335,
        3.369436502456665,
        0.23804199695587158,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.004699581768363714,
        -0.0014164879685267806,
    ],
    "right_distortion": [
        5.091114521026611,
        1.5919005870819092,
        -7.720104804320727e-06,
        2.0027317077619955e-05,
        0.029687780886888504,
        5.4577412605285645,
        3.150493621826172,
        0.22900323569774628,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0026084419805556536,
        -0.002354657743126154,
    ],
}


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for stereo processing. Install it with:\n"
            "    pip install opencv-python"
        ) from exc
    return cv2


def _as_float_array(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array.copy()


def _resolution_wh(value: Any, *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must be [width, height]")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must contain positive values")
    return width, height


def _scale_intrinsics(
    camera_matrix: np.ndarray,
    calibration_resolution_wh: tuple[int, int],
    actual_resolution_wh: tuple[int, int],
) -> np.ndarray:
    """Scale fx/fy/cx/cy when an image is resized from calibration resolution."""
    calib_w, calib_h = calibration_resolution_wh
    actual_w, actual_h = actual_resolution_wh

    if (calib_w, calib_h) == (actual_w, actual_h):
        return camera_matrix.astype(np.float64, copy=True)

    sx = actual_w / float(calib_w)
    sy = actual_h / float(calib_h)

    scaled = camera_matrix.astype(np.float64, copy=True)
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()

    if image is None:
        raise ValueError("image is None")

    array = np.asarray(image)

    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 1:
        gray = array[:, :, 0]
    elif array.ndim == 3 and array.shape[2] >= 3:
        gray = cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {array.shape}")

    if gray.dtype == np.uint8:
        return gray

    if np.issubdtype(gray.dtype, np.floating):
        finite = gray[np.isfinite(gray)]
        if finite.size == 0:
            raise ValueError("image contains no finite values")
        max_value = float(finite.max())
        if max_value <= 1.0:
            gray = gray * 255.0
        return np.clip(gray, 0, 255).astype(np.uint8)

    if gray.dtype == np.uint16:
        # Common camera output format. Preserve contrast by scaling to 8-bit.
        max_value = int(gray.max())
        if max_value == 0:
            return np.zeros_like(gray, dtype=np.uint8)
        return np.clip(gray.astype(np.float32) * (255.0 / max_value), 0, 255).astype(np.uint8)

    return np.clip(gray, 0, 255).astype(np.uint8)


def _normalize_colors_uint8(colors: np.ndarray, *, input_color_order: ColorOrder = "RGB") -> np.ndarray:
    array = np.asarray(colors)

    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"colors must have shape Nx3, got {array.shape}")

    array = array[:, :3]

    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if input_color_order == "BGR":
        array = array[:, ::-1]

    return array


@dataclass(frozen=True)
class StereoRgbCalibration:
    """Calibration for left/right stereo cameras plus an RGB camera.

    All translation vectors stored in this object are in meters.
    """

    rgb_resolution: tuple[int, int]
    left_resolution: tuple[int, int]
    right_resolution: tuple[int, int]

    rgb_intrinsics: np.ndarray
    left_intrinsics: np.ndarray
    right_intrinsics: np.ndarray

    rgb_distortion: np.ndarray
    left_distortion: np.ndarray
    right_distortion: np.ndarray

    left_to_right: np.ndarray
    left_to_rgb: np.ndarray

    source_translation_unit: Literal["m", "cm", "mm"] = "cm"

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_translation_unit: Literal["m", "cm", "mm"] = "cm",
    ) -> "StereoRgbCalibration":
        scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[source_translation_unit]

        left_to_right = _as_float_array(
            data["left_to_right_extrinsics"],
            name="left_to_right_extrinsics",
            shape=(4, 4),
        )
        left_to_rgb = _as_float_array(
            data["left_to_rgb_extrinsics"],
            name="left_to_rgb_extrinsics",
            shape=(4, 4),
        )

        left_to_right[:3, 3] *= scale
        left_to_rgb[:3, 3] *= scale

        return cls(
            rgb_resolution=_resolution_wh(data["rgb_resolution"], name="rgb_resolution"),
            left_resolution=_resolution_wh(data["left_resolution"], name="left_resolution"),
            right_resolution=_resolution_wh(data["right_resolution"], name="right_resolution"),
            rgb_intrinsics=_as_float_array(data["rgb_intrinsics"], name="rgb_intrinsics", shape=(3, 3)),
            left_intrinsics=_as_float_array(data["left_intrinsics"], name="left_intrinsics", shape=(3, 3)),
            right_intrinsics=_as_float_array(data["right_intrinsics"], name="right_intrinsics", shape=(3, 3)),
            rgb_distortion=_as_float_array(data["rgb_distortion"], name="rgb_distortion").reshape(-1, 1),
            left_distortion=_as_float_array(data["left_distortion"], name="left_distortion").reshape(-1, 1),
            right_distortion=_as_float_array(data["right_distortion"], name="right_distortion").reshape(-1, 1),
            left_to_right=left_to_right,
            left_to_rgb=left_to_rgb,
            source_translation_unit=source_translation_unit,
        )

    @classmethod
    def default(cls) -> "StereoRgbCalibration":
        return cls.from_dict(DEFAULT_CALIBRATION, source_translation_unit="cm")

    @property
    def stereo_baseline_m(self) -> float:
        """Dominant stereo baseline in meters, from the left-to-right X translation."""
        return abs(float(self.left_to_right[0, 3]))

    @property
    def stereo_baseline_cm(self) -> float:
        return self.stereo_baseline_m * 100.0

    @property
    def stereo_translation_norm_m(self) -> float:
        return float(np.linalg.norm(self.left_to_right[:3, 3]))

    @property
    def left_to_right_rotation(self) -> np.ndarray:
        return self.left_to_right[:3, :3].copy()

    @property
    def left_to_right_translation_m(self) -> np.ndarray:
        return self.left_to_right[:3, 3].reshape(3, 1).copy()

    @property
    def left_to_rgb_rotation(self) -> np.ndarray:
        return self.left_to_rgb[:3, :3].copy()

    @property
    def left_to_rgb_translation_m(self) -> np.ndarray:
        return self.left_to_rgb[:3, 3].reshape(3, 1).copy()


@dataclass(frozen=True)
class StereoRectification:
    image_size: tuple[int, int]  # width, height
    left_map_x: np.ndarray
    left_map_y: np.ndarray
    right_map_x: np.ndarray
    right_map_y: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    valid_roi_left: tuple[int, int, int, int]
    valid_roi_right: tuple[int, int, int, int]


@dataclass(frozen=True)
class ColoredPointCloud:
    points_m: np.ndarray
    colors_rgb: np.ndarray
    disparity: np.ndarray | None = None
    rectification: StereoRectification | None = None


def load_calibration_json(
    path: str | Path,
    *,
    source_translation_unit: Literal["m", "cm", "mm"] = "cm",
) -> StereoRgbCalibration:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return StereoRgbCalibration.from_dict(data, source_translation_unit=source_translation_unit)


def make_stereo_rectification(
    calibration: StereoRgbCalibration,
    *,
    image_size: tuple[int, int] | None = None,
    alpha: float = 0.0,
    zero_disparity: bool = True,
) -> StereoRectification:
    """Build OpenCV rectification maps and Q matrix.

    Args:
        calibration: Stereo/RGB calibration.
        image_size: Input stereo image size as (width, height). If omitted,
            calibration.left_resolution is used.
        alpha: OpenCV stereoRectify crop value. 0 crops invalid edges, 1 keeps all pixels.
        zero_disparity: If True, aligns principal points after rectification.
    """
    cv2 = _require_cv2()

    image_size = image_size or calibration.left_resolution
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError(f"Invalid stereo image_size: {image_size}")

    if image_size != calibration.left_resolution:
        warnings.warn(
            "Stereo input size differs from calibration left_resolution. "
            "Intrinsics are being scaled; this assumes the image was resized, not cropped.",
            RuntimeWarning,
            stacklevel=2,
        )

    K_left = _scale_intrinsics(
        calibration.left_intrinsics,
        calibration.left_resolution,
        image_size,
    )
    K_right = _scale_intrinsics(
        calibration.right_intrinsics,
        calibration.right_resolution,
        image_size,
    )

    flags = cv2.CALIB_ZERO_DISPARITY if zero_disparity else 0

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        cameraMatrix1=K_left,
        distCoeffs1=calibration.left_distortion,
        cameraMatrix2=K_right,
        distCoeffs2=calibration.right_distortion,
        imageSize=image_size,
        R=calibration.left_to_right_rotation,
        T=calibration.left_to_right_translation_m,
        flags=flags,
        alpha=float(alpha),
    )

    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        K_left,
        calibration.left_distortion,
        R1,
        P1,
        image_size,
        cv2.CV_32FC1,
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        K_right,
        calibration.right_distortion,
        R2,
        P2,
        image_size,
        cv2.CV_32FC1,
    )

    return StereoRectification(
        image_size=image_size,
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        valid_roi_left=tuple(int(v) for v in roi_left),
        valid_roi_right=tuple(int(v) for v in roi_right),
    )


def rectify_stereo_pair(
    left_image: np.ndarray,
    right_image: np.ndarray,
    calibration: StereoRgbCalibration | None = None,
    *,
    rectification: StereoRectification | None = None,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, StereoRectification]:
    """Undistort and rectify left/right images."""
    cv2 = _require_cv2()

    if left_image is None or right_image is None:
        raise ValueError("left_image and right_image are required")

    if left_image.shape[:2] != right_image.shape[:2]:
        raise ValueError(
            f"left/right image sizes must match, got {left_image.shape[:2]} and {right_image.shape[:2]}"
        )

    h, w = left_image.shape[:2]

    if rectification is None:
        calibration = calibration or StereoRgbCalibration.default()
        rectification = make_stereo_rectification(
            calibration,
            image_size=(w, h),
            alpha=alpha,
        )

    left_rect = cv2.remap(
        left_image,
        rectification.left_map_x,
        rectification.left_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rect = cv2.remap(
        right_image,
        rectification.right_map_x,
        rectification.right_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return left_rect, right_rect, rectification


def compute_disparity_sgbm(
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
    *,
    min_disparity: int = 0,
    num_disparities: int = 128,
    block_size: int = 5,
    uniqueness_ratio: int = 8,
    speckle_window_size: int = 80,
    speckle_range: int = 2,
    disp12_max_diff: int = 1,
    pre_filter_cap: int = 31,
    invalid_to_nan: bool = True,
) -> np.ndarray:
    """Compute disparity in pixels using OpenCV StereoSGBM.

    Returns:
        float32 disparity image in pixels. Invalid values are NaN by default.

    Tuning hints:
        - Increase num_disparities for closer objects.
        - Increase block_size for smoother but less detailed output.
        - Use good texture/lighting; stereo needs visible texture.
    """
    cv2 = _require_cv2()

    left_gray = _to_gray_uint8(left_rectified)
    right_gray = _to_gray_uint8(right_rectified)

    if left_gray.shape != right_gray.shape:
        raise ValueError("Rectified left/right grayscale images must have the same shape")

    num_disparities = int(math.ceil(max(16, num_disparities) / 16.0) * 16)

    block_size = int(block_size)
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    channels = 1
    p1 = 8 * channels * block_size * block_size
    p2 = 32 * channels * block_size * block_size

    matcher = cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=int(disp12_max_diff),
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=int(speckle_window_size),
        speckleRange=int(speckle_range),
        preFilterCap=int(pre_filter_cap),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

    if invalid_to_nan:
        invalid = disparity <= float(min_disparity)
        disparity[invalid] = np.nan

    return disparity


def disparity_to_points_rectified(
    disparity: np.ndarray,
    rectification: StereoRectification,
    *,
    min_disparity: float = 0.5,
    max_depth_m: float | None = 10.0,
    stride: int = 1,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject disparity into 3D points in the rectified-left frame.

    Returns:
        points_rectified_m:
            Nx3 float64 points in meters.
        pixel_xy:
            Nx2 integer pixel coordinates in the rectified-left image.
    """
    cv2 = _require_cv2()

    disparity = np.asarray(disparity, dtype=np.float32)

    if disparity.ndim != 2:
        raise ValueError(f"disparity must be HxW, got {disparity.shape}")

    valid = np.isfinite(disparity) & (disparity > float(min_disparity))

    if mask is not None:
        mask_array = np.asarray(mask).astype(bool)
        if mask_array.shape != disparity.shape:
            raise ValueError(f"mask shape {mask_array.shape} does not match disparity shape {disparity.shape}")
        valid &= mask_array

    if stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid &= stride_mask

    safe_disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
    points_3d = cv2.reprojectImageTo3D(safe_disparity, rectification.Q)

    finite_points = np.isfinite(points_3d).all(axis=2)
    valid &= finite_points

    if max_depth_m is not None:
        z = points_3d[:, :, 2]
        valid &= z > 0.0
        valid &= z <= float(max_depth_m)
    else:
        valid &= points_3d[:, :, 2] > 0.0

    ys, xs = np.nonzero(valid)
    points = points_3d[ys, xs, :].astype(np.float64)
    pixel_xy = np.column_stack([xs, ys]).astype(np.int32)

    return points, pixel_xy


def rectified_left_to_original_left(
    points_rectified_m: np.ndarray,
    rectification: StereoRectification,
) -> np.ndarray:
    """Rotate points from rectified-left frame back into original-left frame."""
    points = np.asarray(points_rectified_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_rectified_m must be Nx3, got {points.shape}")

    # OpenCV rectification uses x_rectified = R1 * x_original.
    # Therefore x_original = R1.T * x_rectified.
    return points @ rectification.R1


def transform_points(
    points_m: np.ndarray,
    transform_4x4: np.ndarray,
) -> np.ndarray:
    """Apply a 4x4 rigid transform to Nx3 points."""
    points = np.asarray(points_m, dtype=np.float64)
    transform = np.asarray(transform_4x4, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_m must be Nx3, got {points.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"transform_4x4 must be 4x4, got {transform.shape}")

    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points_to_rgb_pixels(
    points_left_m: np.ndarray,
    rgb_image: np.ndarray,
    calibration: StereoRgbCalibration,
    *,
    rgb_image_is_undistorted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Project original-left-frame 3D points into the RGB image.

    Args:
        points_left_m: Nx3 points in the original-left camera frame.
        rgb_image: RGB or BGR image. Only shape is used here.
        calibration: Calibration object.
        rgb_image_is_undistorted:
            False means project with RGB distortion coefficients into a normal/raw
            RGB image. True means project with zero distortion, for an already
            undistorted RGB image.

    Returns:
        pixel_xy:
            Nx2 float64 projected pixel coordinates in the RGB image.
        points_rgb_m:
            Nx3 points transformed into the RGB camera frame.
    """
    cv2 = _require_cv2()

    points_left = np.asarray(points_left_m, dtype=np.float64)
    if points_left.ndim != 2 or points_left.shape[1] != 3:
        raise ValueError(f"points_left_m must be Nx3, got {points_left.shape}")

    if rgb_image is None or np.asarray(rgb_image).ndim < 2:
        raise ValueError("rgb_image must be an image array")

    rgb_h, rgb_w = np.asarray(rgb_image).shape[:2]
    K_rgb = _scale_intrinsics(
        calibration.rgb_intrinsics,
        calibration.rgb_resolution,
        (rgb_w, rgb_h),
    )

    dist_rgb = None if rgb_image_is_undistorted else calibration.rgb_distortion

    points_rgb = transform_points(points_left, calibration.left_to_rgb)

    # Points are already in the RGB camera coordinate frame, so rvec/tvec are zero.
    projected, _ = cv2.projectPoints(
        objectPoints=points_rgb.reshape(-1, 1, 3),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.zeros((3, 1), dtype=np.float64),
        cameraMatrix=K_rgb,
        distCoeffs=dist_rgb,
    )

    pixel_xy = projected.reshape(-1, 2).astype(np.float64)
    return pixel_xy, points_rgb


def sample_rgb_colors(
    rgb_image: np.ndarray,
    pixel_xy: np.ndarray,
    *,
    input_color_order: ColorOrder = "RGB",
    interpolation: Literal["nearest"] = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample RGB colors at projected pixel locations.

    Returns:
        colors_rgb_uint8:
            Mx3 uint8 colors.
        valid_mask:
            Boolean mask over the input pixel_xy array.
    """
    if interpolation != "nearest":
        raise NotImplementedError("Only nearest-neighbor color sampling is currently implemented")

    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"rgb_image must be HxWx3 or HxWx4, got {image.shape}")

    pixels = np.asarray(pixel_xy, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixel_xy must be Nx2, got {pixels.shape}")

    h, w = image.shape[:2]
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)

    valid = (
        np.isfinite(pixels).all(axis=1)
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )

    sampled = image[v[valid], u[valid], :3]
    colors_rgb = _normalize_colors_uint8(sampled, input_color_order=input_color_order)
    return colors_rgb, valid


def colorize_points_from_rgb(
    points_m: np.ndarray,
    rgb_image: np.ndarray,
    calibration: StereoRgbCalibration,
    *,
    rectification: StereoRectification | None = None,
    points_frame: PointsFrame = "left_rectified",
    output_frame: OutputFrame = "left",
    input_color_order: ColorOrder = "RGB",
    rgb_image_is_undistorted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Attach RGB colors to 3D points.

    Args:
        points_m:
            Nx3 points. By default these are expected to be in the rectified-left
            frame, as returned by disparity_to_points_rectified().
        rgb_image:
            Synchronized color image.
        calibration:
            Stereo/RGB calibration.
        rectification:
            Required when points_frame="left_rectified".
        points_frame:
            "left_rectified" or "left".
        output_frame:
            "left" saves/returns physical original-left frame points.
            "left_rectified" returns rectified-left frame points.
        input_color_order:
            Use "BGR" if rgb_image came from cv2.imread().
        rgb_image_is_undistorted:
            Set True only if the RGB image has already been undistorted.

    Returns:
        colored_points_m:
            Mx3 points after removing projections outside the RGB image.
        colors_rgb_uint8:
            Mx3 uint8 RGB colors.
    """
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_m must be Nx3, got {points.shape}")

    if points_frame == "left_rectified":
        if rectification is None:
            raise ValueError("rectification is required when points_frame='left_rectified'")
        points_left = rectified_left_to_original_left(points, rectification)
    elif points_frame == "left":
        points_left = points
    else:
        raise ValueError(f"Unsupported points_frame: {points_frame}")

    pixel_xy, points_rgb = project_points_to_rgb_pixels(
        points_left,
        rgb_image,
        calibration,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
    )

    in_front = points_rgb[:, 2] > 0.0
    pixel_xy_front = pixel_xy[in_front]

    colors_rgb, valid_in_front = sample_rgb_colors(
        rgb_image,
        pixel_xy_front,
        input_color_order=input_color_order,
    )

    valid = np.zeros(points.shape[0], dtype=bool)
    valid_indices_front = np.nonzero(in_front)[0]
    valid[valid_indices_front[valid_in_front]] = True

    if output_frame == "left":
        colored_points = points_left[valid]
    elif output_frame == "left_rectified":
        colored_points = points[valid]
    else:
        raise ValueError(f"Unsupported output_frame: {output_frame}")

    return colored_points.astype(np.float64), colors_rgb


def _pack_rgb_as_float32(colors_rgb_uint8: np.ndarray) -> np.ndarray:
    colors = _normalize_colors_uint8(colors_rgb_uint8, input_color_order="RGB")
    rgb_uint32 = (
        (colors[:, 0].astype(np.uint32) << 16)
        | (colors[:, 1].astype(np.uint32) << 8)
        | colors[:, 2].astype(np.uint32)
    )
    return rgb_uint32.astype("<u4").view("<f4")


def save_pcd(
    path: str | Path,
    points_m: np.ndarray,
    colors_rgb: np.ndarray,
    *,
    binary: bool = True,
) -> Path:
    """Save a colored point cloud as PCD.

    Uses the common PCL-compatible packed-float RGB field:
        FIELDS x y z rgb
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points_m, dtype=np.float32)
    colors = _normalize_colors_uint8(colors_rgb, input_color_order="RGB")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_m must be Nx3, got {points.shape}")
    if colors.shape[0] != points.shape[0]:
        raise ValueError(f"points/colors length mismatch: {points.shape[0]} vs {colors.shape[0]}")

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    n = int(points.shape[0])
    rgb_float = _pack_rgb_as_float32(colors)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    )

    if binary:
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("rgb", "<f4"),
            ]
        )
        structured = np.empty(n, dtype=dtype)
        structured["x"] = points[:, 0]
        structured["y"] = points[:, 1]
        structured["z"] = points[:, 2]
        structured["rgb"] = rgb_float

        with path.open("wb") as f:
            f.write(header.encode("ascii"))
            structured.tofile(f)
    else:
        with path.open("w", encoding="ascii") as f:
            f.write(header)
            for xyz, rgb in zip(points, rgb_float, strict=True):
                f.write(
                    f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                    f"{float(rgb):.9e}\n"
                )

    return path


def save_ply_ascii(
    path: str | Path,
    points_m: np.ndarray,
    colors_rgb: np.ndarray,
) -> Path:
    """Save a colored point cloud as ASCII PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points_m, dtype=np.float64)
    colors = _normalize_colors_uint8(colors_rgb, input_color_order="RGB")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_m must be Nx3, got {points.shape}")
    if colors.shape[0] != points.shape[0]:
        raise ValueError(f"points/colors length mismatch: {points.shape[0]} vs {colors.shape[0]}")

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    n = int(points.shape[0])
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for xyz, rgb in zip(points, colors, strict=True):
            f.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )

    return path


def save_point_cloud(
    path: str | Path,
    points_m: np.ndarray,
    colors_rgb: np.ndarray,
    *,
    binary_pcd: bool = True,
) -> Path:
    """Save .pcd or .ply based on the file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pcd":
        return save_pcd(path, points_m, colors_rgb, binary=binary_pcd)
    if suffix == ".ply":
        return save_ply_ascii(path, points_m, colors_rgb)

    raise ValueError("Unsupported point-cloud extension. Use .pcd or .ply")


def npz_to_pcd(
    input_npz: str | Path,
    output_path: str | Path,
    *,
    binary_pcd: bool = True,
) -> Path:
    input_npz = Path(input_npz)
    output_path = Path(output_path)

    if input_npz.suffix.lower() != ".npz":
        raise ValueError(f"input_npz must end with .npz, got: {input_npz}")

    if output_path.suffix.lower() not in {".pcd", ".ply"}:
        raise ValueError(f"output_path must end with .pcd or .ply, got: {output_path}")

    with np.load(input_npz, allow_pickle=False) as z:
        if "points_m" not in z:
            raise KeyError(f"{input_npz} does not contain 'points_m'")
        if "colors_rgb" not in z:
            raise KeyError(f"{input_npz} does not contain 'colors_rgb'")

        points_m = np.asarray(z["points_m"])
        colors_rgb = np.asarray(z["colors_rgb"])

    if points_m.ndim != 2 or points_m.shape[1] != 3:
        raise ValueError(f"points_m must have shape Nx3, got {points_m.shape}")

    if colors_rgb.ndim != 2 or colors_rgb.shape[1] < 3:
        raise ValueError(f"colors_rgb must have shape Nx3, got {colors_rgb.shape}")

    if points_m.shape[0] != colors_rgb.shape[0]:
        raise ValueError(
            f"points/colors length mismatch: "
            f"{points_m.shape[0]} points vs {colors_rgb.shape[0]} colors"
        )

    save_point_cloud(
        output_path,
        points_m,
        colors_rgb,
        binary_pcd=binary_pcd,
    )

    return output_path


def stereo_rgb_to_colored_point_cloud(
    left_image: np.ndarray,
    right_image: np.ndarray,
    rgb_image: np.ndarray,
    *,
    calibration: StereoRgbCalibration | None = None,
    output_path: str | Path | None = None,
    input_color_order: ColorOrder = "RGB",
    rgb_image_is_undistorted: bool = False,
    alpha: float = 0.0,
    min_disparity: int = 0,
    num_disparities: int = 128,
    block_size: int = 5,
    max_depth_m: float | None = 10.0,
    stride: int = 1,
    output_frame: OutputFrame = "left",
    save_binary_pcd: bool = True,
) -> ColoredPointCloud:
    """Full pipeline: stereo images + RGB image -> colored point cloud.

    Args:
        left_image, right_image:
            Synchronized stereo images.
        rgb_image:
            Synchronized color image.
        calibration:
            Defaults to the calibration embedded in this file.
        output_path:
            Optional .pcd or .ply path.
        input_color_order:
            "RGB" if rgb_image is RGB.
            "BGR" if rgb_image came from cv2.imread() or cv2.VideoCapture().
        rgb_image_is_undistorted:
            Set True only if you already undistorted the RGB image.
        alpha:
            Rectification crop factor. 0 crops invalid edges; 1 keeps all pixels.
        num_disparities:
            Stereo search width in pixels. Must be multiple of 16; this function
            rounds up automatically.
        block_size:
            SGBM matching window size. Odd integer, usually 3-9.
        max_depth_m:
            Remove points farther than this. Use None to disable.
        stride:
            Keep every Nth pixel in x/y for a smaller point cloud.
        output_frame:
            "left" is recommended. "left_rectified" is also available.

    Returns:
        ColoredPointCloud with points in meters and RGB colors as uint8.
    """
    calibration = calibration or StereoRgbCalibration.default()

    left_rect, right_rect, rectification = rectify_stereo_pair(
        left_image,
        right_image,
        calibration,
        alpha=alpha,
    )

    disparity = compute_disparity_sgbm(
        left_rect,
        right_rect,
        min_disparity=min_disparity,
        num_disparities=num_disparities,
        block_size=block_size,
    )

    points_rectified, _pixel_xy = disparity_to_points_rectified(
        disparity,
        rectification,
        min_disparity=max(0.5, float(min_disparity)),
        max_depth_m=max_depth_m,
        stride=stride,
    )

    points, colors = colorize_points_from_rgb(
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
        save_point_cloud(output_path, points, colors, binary_pcd=save_binary_pcd)

    return ColoredPointCloud(
        points_m=points,
        colors_rgb=colors,
        disparity=disparity,
        rectification=rectification,
    )


def read_image(path: str | Path, *, color: bool = True) -> np.ndarray:
    """Convenience image reader using OpenCV.

    Note:
        color=True returns BGR order because cv2.imread returns BGR.
        Pass input_color_order="BGR" when colorizing this image.
    """
    cv2 = _require_cv2()
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


__all__ = [
    "DEFAULT_CALIBRATION",
    "StereoRgbCalibration",
    "StereoRectification",
    "ColoredPointCloud",
    "load_calibration_json",
    "make_stereo_rectification",
    "rectify_stereo_pair",
    "compute_disparity_sgbm",
    "disparity_to_points_rectified",
    "rectified_left_to_original_left",
    "transform_points",
    "project_points_to_rgb_pixels",
    "sample_rgb_colors",
    "colorize_points_from_rgb",
    "save_pcd",
    "save_ply_ascii",
    "save_point_cloud",
    "stereo_rgb_to_colored_point_cloud",
    "read_image",
]


if __name__ == "__main__":
    # Example:
    #
    # left = read_image("left.png", color=False)
    # right = read_image("right.png", color=False)
    # color = read_image("rgb.png", color=True)  # cv2 gives BGR
    #
    # cloud = stereo_rgb_to_colored_point_cloud(
    #     left,
    #     right,
    #     color,
    #     output_path="colored_cloud.pcd",
    #     input_color_order="BGR",
    #     num_disparities=160,
    #     block_size=5,
    #     max_depth_m=8.0,
    #     stride=1,
    # )
    #
    # print(f"Saved {len(cloud.points_m):,} colored points")
    pass
