
from dataclasses import dataclass
from typing import Any
import zipfile
import numpy as np

from pathlib import Path
from typing import Any, Literal
import json
import numpy as np

from pcd_utils import *  # noqa: F401,F403


YoloOverlapPolicy = Literal["highest_confidence", "first", "none"]
YoloSegmentOutputFrame = Literal["rgb", "left"]


@dataclass
class YoloMasks:
    data: np.ndarray  # N x H x W, float/bool/uint8


@dataclass
class YoloBoxes:
    cls: np.ndarray   # N
    conf: np.ndarray  # N
    xyxy: np.ndarray  # N x 4


@dataclass
class YoloResult:
    masks: YoloMasks
    boxes: YoloBoxes
    names: dict[int, str]

    @staticmethod
    def make_simple_yolo_result(
        masks: np.ndarray,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        boxes_xyxy: np.ndarray,
        names: dict[int, str] | None = None,
    ):
        masks = np.asarray(masks)
        class_ids = np.asarray(class_ids, dtype=np.int32)
        confidences = np.asarray(confidences, dtype=np.float32)
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)

        if masks.ndim == 2:
            masks = masks[None, :, :]

        if masks.ndim != 3:
            raise ValueError(f"masks must be NxHxW, got {masks.shape}")

        n = masks.shape[0]

        if class_ids.shape[0] != n:
            raise ValueError(f"class_ids length {class_ids.shape[0]} != mask count {n}")

        if confidences.shape[0] != n:
            raise ValueError(f"confidences length {confidences.shape[0]} != mask count {n}")

        if boxes_xyxy.shape != (n, 4):
            raise ValueError(f"boxes_xyxy must be Nx4, got {boxes_xyxy.shape}")

        return YoloResult(
            masks=YoloMasks(data=masks),
            boxes=YoloBoxes(
                cls=class_ids,
                conf=confidences,
                xyxy=boxes_xyxy,
            ),
            names=names or {},
        )
    
    @staticmethod
    def from_ultralytics(result):
        masks = result.masks.data.detach().cpu().numpy()

        class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)

        names = getattr(result, "names", {}) or {}

        return YoloResult.make_simple_yolo_result(
            masks=masks,
            class_ids=class_ids,
            confidences=confidences,
            boxes_xyxy=boxes_xyxy,
            names=names,
        )

    @staticmethod
    def from_file(path: str | Path):
        """
        Load a YoloResult saved by YoloResult.to_file().
        """
        path = Path(path)

        with np.load(path, allow_pickle=False) as data:
            required = {
                "format_version",
                "masks_data",
                "boxes_cls",
                "boxes_conf",
                "boxes_xyxy",
                "names_keys",
                "names_values",
            }

            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Invalid YoloResult file. Missing keys: {sorted(missing)}")

            version = int(np.asarray(data["format_version"]).reshape(-1)[0])
            if version != 1:
                raise ValueError(f"Unsupported YoloResult file version: {version}")

            masks = data["masks_data"]
            class_ids = data["boxes_cls"]
            confidences = data["boxes_conf"]
            boxes_xyxy = data["boxes_xyxy"]

            names_keys = data["names_keys"]
            names_values = data["names_values"]

            if names_keys.shape[0] != names_values.shape[0]:
                raise ValueError(
                    f"Invalid names data: keys={names_keys.shape}, values={names_values.shape}"
                )

            names = {
                int(k): str(v)
                for k, v in zip(names_keys, names_values)
            }

        return YoloResult.make_simple_yolo_result(
            masks=masks,
            class_ids=class_ids,
            confidences=confidences,
            boxes_xyxy=boxes_xyxy,
            names=names,
        )


    def to_file(
        self,
        path: str | Path,
        *,
        compress: bool = False,
        compresslevel: int = 1,
    ) -> Path:
        """
        Save YoloResult to a fast .npz file.

        Parameters
        ----------
        path:
            Output file path. Should usually end in .npz.
        compress:
            False = fastest save/load, larger file.
            True = smaller file, especially for bool/uint8 masks.
        compresslevel:
            Zip compression level, only used when compress=True.
            1 is usually the best speed/size tradeoff.
        """
        path = Path(path)

        names = self.names or {}
        name_items = sorted((int(k), str(v)) for k, v in names.items())

        names_keys = np.asarray([k for k, _ in name_items], dtype=np.int32)
        names_values = np.asarray([v for _, v in name_items], dtype=np.str_)

        arrays = {
            "format_version": np.asarray(1, dtype=np.uint16),

            "masks_data": np.asarray(self.masks.data),

            "boxes_cls": np.asarray(self.boxes.cls),
            "boxes_conf": np.asarray(self.boxes.conf),
            "boxes_xyxy": np.asarray(self.boxes.xyxy),

            "names_keys": names_keys,
            "names_values": names_values,
        }

        compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        level = compresslevel if compress else None

        with zipfile.ZipFile(
            path,
            mode="w",
            compression=compression,
            compresslevel=level,
            allowZip64=True,
        ) as zf:
            for key, arr in arrays.items():
                with zf.open(f"{key}.npy", mode="w", force_zip64=True) as f:
                    np.lib.format.write_array(
                        f,
                        np.asarray(arr),
                        allow_pickle=False,
                    )

        return path



@dataclass(frozen=True)
class YoloSegment3D:
    instance_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy_rgb: tuple[float, float, float, float]

    points_m: np.ndarray       # Nx3, in output_frame
    colors_rgb: np.ndarray     # Nx3, uint8 RGB
    pixels_rgb: np.ndarray     # Nx2, integer RGB pixels: [u, v]

    mask_area_px: int
    centroid_m: np.ndarray     # 3
    aabb_min_m: np.ndarray     # 3
    aabb_max_m: np.ndarray     # 3
    output_frame: Literal["rgb", "left"]

    pcd_path: str | None = None
    pixels_path: str | None = None
    meta_path: str | None = None


@dataclass(frozen=True)
class YoloSegments3DResult:
    # Combined segmented cloud, ready for ROS PointCloud2
    points_m: np.ndarray          # Mx3
    colors_rgb: np.ndarray        # Mx3
    pixels_rgb: np.ndarray        # Mx2, RGB pixels [u, v]
    instance_ids: np.ndarray      # M
    class_ids: np.ndarray         # M
    confidences: np.ndarray       # M

    # RGB-sized instance-id image, ready for ROS Image
    instance_map: np.ndarray      # HxW uint32, 0 = background

    # Per-object metadata
    segments: list[YoloSegment3D]

    # Debug / optional downstream data
    depth_rgb_m: np.ndarray | None
    disparity: np.ndarray | None
    rectification: Any | None
    output_frame: Literal["rgb", "left"]


def _torch_to_numpy(x: Any) -> np.ndarray:
    """Accept torch tensors, numpy arrays, or list-like values."""
    if x is None:
        return np.asarray([])
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _safe_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _safe_filename_text(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


def _resize_yolo_mask_to_rgb(mask: np.ndarray, rgb_hw: tuple[int, int], threshold: float) -> np.ndarray:
    """
    Resize a YOLO mask to RGB image size.

    Important:
    This assumes the YOLO result corresponds to the same RGB image geometry.
    If you manually letterboxed/padded/resized before YOLO, undo that first.
    """
    rgb_h, rgb_w = rgb_hw
    m = np.asarray(mask, np.float32)

    if m.shape != (rgb_h, rgb_w):
        c = cv2()
        m = c.resize(m, (rgb_w, rgb_h), interpolation=c.INTER_NEAREST)

    return m > float(threshold)


def _segment_meta_dict(seg: YoloSegment3D) -> dict[str, Any]:
    return {
        "instance_id": int(seg.instance_id),
        "class_id": int(seg.class_id),
        "class_name": seg.class_name,
        "confidence": float(seg.confidence),
        "bbox_xyxy_rgb": [float(v) for v in seg.bbox_xyxy_rgb],
        "mask_area_px": int(seg.mask_area_px),
        "point_count": int(len(seg.points_m)),
        "centroid_m": seg.centroid_m.astype(float).tolist(),
        "aabb_min_m": seg.aabb_min_m.astype(float).tolist(),
        "aabb_max_m": seg.aabb_max_m.astype(float).tolist(),
        "output_frame": seg.output_frame,
        "pcd_path": seg.pcd_path,
        "pixels_path": seg.pixels_path,
        "meta_path": seg.meta_path,
    }


def yolo_result_to_pcd_segments(
    left_image,
    right_image,
    rgb_image,
    yolo_result,
    *,
    calibration=None,
    output_dir: str | Path | None = None,
    frame_name: str = "frame",
    input_color_order: ColorOrder = "RGB",
    rgb_image_is_undistorted: bool = False,
    output_frame: YoloSegmentOutputFrame = "rgb",

    # Stereo/depth parameters
    alpha: float = 0.0,
    min_disparity: float = 0,
    num_disparities: int = 128,
    block_size: int = 5,
    max_depth_m: float = 10.0,
    splat_px: int = 1,

    # YOLO/mask parameters
    mask_threshold: float = 0.5,
    overlap_policy: YoloOverlapPolicy = "highest_confidence",
    min_points: int = 30,

    # Saving
    save_pcd: bool = True,
    save_pixels: bool = True,
    save_meta: bool = True,
    save_binary_pcd: bool = True,
) -> YoloSegments3DResult:
    """
    Convert one Ultralytics YOLO-seg result into 3D PCD segments.

    Best use:
        result = model(rgb_image)[0]
        seg3d = yolo_result_to_pcd_segments(
            left, right, rgb_image, result,
            output_dir="segments_out",
            input_color_order="BGR",  # if rgb_image came from cv2.imread
            output_frame="rgb",
        )

    Returns arrays designed to be easy to publish later in ROS 2:
        seg3d.points_m
        seg3d.colors_rgb
        seg3d.instance_ids
        seg3d.class_ids
        seg3d.confidences
        seg3d.pixels_rgb
        seg3d.instance_map
        seg3d.segments
    """

    calibration = calibration or StereoRgbCalibration.default()
    rgb_arr = np.asarray(rgb_image)
    rgb_h, rgb_w = rgb_arr.shape[:2]

    masks_obj = getattr(yolo_result, "masks", None)
    boxes_obj = getattr(yolo_result, "boxes", None)

    if masks_obj is None or getattr(masks_obj, "data", None) is None:
        empty_i = np.empty((0,), np.int32)
        empty_f = np.empty((0,), np.float32)
        return YoloSegments3DResult(
            points_m=np.empty((0, 3), np.float64),
            colors_rgb=np.empty((0, 3), np.uint8),
            pixels_rgb=np.empty((0, 2), np.int32),
            instance_ids=empty_i,
            class_ids=empty_i,
            confidences=empty_f,
            instance_map=np.zeros((rgb_h, rgb_w), np.uint32),
            segments=[],
            depth_rgb_m=None,
            disparity=None,
            rectification=None,
            output_frame=output_frame,
        )

    masks = _torch_to_numpy(masks_obj.data)
    if masks.ndim == 2:
        masks = masks[None, :, :]

    n = int(masks.shape[0])
    if n == 0:
        empty_i = np.empty((0,), np.int32)
        empty_f = np.empty((0,), np.float32)
        return YoloSegments3DResult(
            points_m=np.empty((0, 3), np.float64),
            colors_rgb=np.empty((0, 3), np.uint8),
            pixels_rgb=np.empty((0, 2), np.int32),
            instance_ids=empty_i,
            class_ids=empty_i,
            confidences=empty_f,
            instance_map=np.zeros((rgb_h, rgb_w), np.uint32),
            segments=[],
            depth_rgb_m=None,
            disparity=None,
            rectification=None,
            output_frame=output_frame,
        )

    if boxes_obj is not None:
        class_ids_yolo = _torch_to_numpy(getattr(boxes_obj, "cls", None)).astype(np.int32).reshape(-1)
        confs_yolo = _torch_to_numpy(getattr(boxes_obj, "conf", None)).astype(np.float32).reshape(-1)
        bboxes_yolo = _torch_to_numpy(getattr(boxes_obj, "xyxy", None)).astype(np.float32).reshape(-1, 4)
    else:
        class_ids_yolo = np.zeros((n,), np.int32)
        confs_yolo = np.ones((n,), np.float32)
        bboxes_yolo = np.full((n, 4), np.nan, np.float32)

    class_ids_yolo = np.resize(class_ids_yolo, n)
    confs_yolo = np.resize(confs_yolo, n)
    if bboxes_yolo.shape[0] != n:
        bboxes_yolo = np.resize(bboxes_yolo, (n, 4))

    names = getattr(yolo_result, "names", None)

    # ------------------------------------------------------------------
    # 1. Compute RGB-aligned depth once.
    # ------------------------------------------------------------------
    left_rect, right_rect, rect = rectify_stereo_pair(
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

    points_rect, _xy_left_rect = disparity_to_points_rectified(
        disparity,
        rect,
        min_disparity=max(0.5, float(min_disparity)),
        max_depth_m=max_depth_m,
        stride=1,
    )

    if len(points_rect) == 0:
        instance_map = np.zeros((rgb_h, rgb_w), np.uint32)
        return YoloSegments3DResult(
            points_m=np.empty((0, 3), np.float64),
            colors_rgb=np.empty((0, 3), np.uint8),
            pixels_rgb=np.empty((0, 2), np.int32),
            instance_ids=np.empty((0,), np.int32),
            class_ids=np.empty((0,), np.int32),
            confidences=np.empty((0,), np.float32),
            instance_map=instance_map,
            segments=[],
            depth_rgb_m=np.full((rgb_h, rgb_w), np.nan, np.float64),
            disparity=disparity,
            rectification=rect,
            output_frame=output_frame,
        )

    points_left = rectified_left_to_original_left(points_rect, rect)

    depth_rgb_m, _valid_depth_rgb = points_left_to_rgb_depth(
        points_left,
        rgb_image,
        calibration,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
        splat_px=splat_px,
    )

    if not np.isfinite(depth_rgb_m).any():
        instance_map = np.zeros((rgb_h, rgb_w), np.uint32)
        return YoloSegments3DResult(
            points_m=np.empty((0, 3), np.float64),
            colors_rgb=np.empty((0, 3), np.uint8),
            pixels_rgb=np.empty((0, 2), np.int32),
            instance_ids=np.empty((0,), np.int32),
            class_ids=np.empty((0,), np.int32),
            confidences=np.empty((0,), np.float32),
            instance_map=instance_map,
            segments=[],
            depth_rgb_m=depth_rgb_m,
            disparity=disparity,
            rectification=rect,
            output_frame=output_frame,
        )

    points_rgb, xy_rgb = rgb_depth_to_points_rgb(
        depth_rgb_m,
        calibration,
        rgb_image_is_undistorted=rgb_image_is_undistorted,
    )

    x = xy_rgb[:, 0]
    y = xy_rgb[:, 1]

    colors_all = rgb8(rgb_arr[y, x, :3], input_color_order)

    if output_frame == "rgb":
        points_all = points_rgb
    elif output_frame == "left":
        points_all = transform_points(points_rgb, np.linalg.inv(calibration.left_to_rgb))
    else:
        raise ValueError("output_frame must be 'rgb' or 'left'")

    # ------------------------------------------------------------------
    # 2. Build RGB-sized instance map.
    # ------------------------------------------------------------------
    bool_masks: list[np.ndarray] = []
    for i in range(n):
        bool_masks.append(
            _resize_yolo_mask_to_rgb(
                masks[i],
                (rgb_h, rgb_w),
                mask_threshold,
            )
        )

    instance_map = np.zeros((rgb_h, rgb_w), np.uint32)

    if overlap_policy == "highest_confidence":
        score_map = np.full((rgb_h, rgb_w), -np.inf, np.float32)
        for i, m in enumerate(bool_masks):
            inst_id = i + 1
            conf = float(confs_yolo[i])
            update = m & (conf >= score_map)
            instance_map[update] = inst_id
            score_map[update] = conf

    elif overlap_policy == "first":
        for i, m in enumerate(bool_masks):
            inst_id = i + 1
            update = m & (instance_map == 0)
            instance_map[update] = inst_id

    elif overlap_policy == "none":
        # Non-exclusive mode allows duplicate 3D points across segments.
        # instance_map is only a visualization map in this mode.
        for i, m in enumerate(bool_masks):
            inst_id = i + 1
            update = m & (instance_map == 0)
            instance_map[update] = inst_id

    else:
        raise ValueError("overlap_policy must be 'highest_confidence', 'first', or 'none'")

    # ------------------------------------------------------------------
    # 3. Slice points into per-object segments.
    # ------------------------------------------------------------------
    output_dir_path = Path(output_dir) if output_dir is not None else None
    seg_dir = None
    if output_dir_path is not None:
        seg_dir = output_dir_path / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

    segments: list[YoloSegment3D] = []

    combined_points = []
    combined_colors = []
    combined_pixels = []
    combined_instance_ids = []
    combined_class_ids = []
    combined_confidences = []

    for i, m in enumerate(bool_masks):
        inst_id = i + 1
        class_id = int(class_ids_yolo[i])
        class_name = _safe_class_name(names, class_id)
        conf = float(confs_yolo[i])
        bbox = tuple(float(v) for v in bboxes_yolo[i].tolist())

        if overlap_policy == "none":
            keep = m[y, x]
        else:
            keep = instance_map[y, x] == inst_id

        seg_points = points_all[keep]
        seg_colors = colors_all[keep]
        seg_pixels = xy_rgb[keep]

        if len(seg_points) < int(min_points):
            continue

        centroid = np.nanmean(seg_points, axis=0)
        aabb_min = np.nanmin(seg_points, axis=0)
        aabb_max = np.nanmax(seg_points, axis=0)

        pcd_path = None
        pixels_path = None
        meta_path = None

        safe_name = _safe_filename_text(class_name)
        stem = f"{frame_name}_obj_{i:03d}_id{inst_id:03d}_{safe_name}_conf{conf:.2f}"

        if seg_dir is not None:
            if save_pcd:
                pcd_file = seg_dir / f"{stem}.pcd"
                save_point_cloud(
                    pcd_file,
                    seg_points,
                    seg_colors,
                    binary_pcd=save_binary_pcd,
                )
                pcd_path = str(pcd_file)

            if save_pixels:
                pixels_file = seg_dir / f"{stem}_pixels.npy"
                np.save(pixels_file, seg_pixels.astype(np.int32))
                pixels_path = str(pixels_file)

        seg = YoloSegment3D(
            instance_id=inst_id,
            class_id=class_id,
            class_name=class_name,
            confidence=conf,
            bbox_xyxy_rgb=bbox,
            points_m=seg_points.astype(np.float64),
            colors_rgb=seg_colors.astype(np.uint8),
            pixels_rgb=seg_pixels.astype(np.int32),
            mask_area_px=int(m.sum()),
            centroid_m=centroid.astype(np.float64),
            aabb_min_m=aabb_min.astype(np.float64),
            aabb_max_m=aabb_max.astype(np.float64),
            output_frame=output_frame,
            pcd_path=pcd_path,
            pixels_path=pixels_path,
            meta_path=None,
        )

        if seg_dir is not None and save_meta:
            meta_file = seg_dir / f"{stem}_meta.json"
            meta = _segment_meta_dict(seg)
            meta["meta_path"] = str(meta_file)
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            seg = YoloSegment3D(
                **{
                    **seg.__dict__,
                    "meta_path": str(meta_file),
                }
            )

        segments.append(seg)

        combined_points.append(seg.points_m)
        combined_colors.append(seg.colors_rgb)
        combined_pixels.append(seg.pixels_rgb)
        combined_instance_ids.append(
            np.full((len(seg.points_m),), inst_id, np.int32)
        )
        combined_class_ids.append(
            np.full((len(seg.points_m),), class_id, np.int32)
        )
        combined_confidences.append(
            np.full((len(seg.points_m),), conf, np.float32)
        )

    if combined_points:
        points_out = np.concatenate(combined_points, axis=0)
        colors_out = np.concatenate(combined_colors, axis=0)
        pixels_out = np.concatenate(combined_pixels, axis=0)
        instance_ids_out = np.concatenate(combined_instance_ids, axis=0)
        class_ids_out = np.concatenate(combined_class_ids, axis=0)
        confidences_out = np.concatenate(combined_confidences, axis=0)
    else:
        points_out = np.empty((0, 3), np.float64)
        colors_out = np.empty((0, 3), np.uint8)
        pixels_out = np.empty((0, 2), np.int32)
        instance_ids_out = np.empty((0,), np.int32)
        class_ids_out = np.empty((0,), np.int32)
        confidences_out = np.empty((0,), np.float32)

    return YoloSegments3DResult(
        points_m=points_out,
        colors_rgb=colors_out,
        pixels_rgb=pixels_out,
        instance_ids=instance_ids_out,
        class_ids=class_ids_out,
        confidences=confidences_out,
        instance_map=instance_map,
        segments=segments,
        depth_rgb_m=depth_rgb_m,
        disparity=disparity,
        rectification=rect,
        output_frame=output_frame,
    )

if __name__ == "__main__":
    # # Example:
    # # rgb image from cv2.imread is actually BGR
    # left = read_image("left.png", color=False)
    # right = read_image("right.png", color=False)
    # rgb = read_image("rgb.jpg", color=True)

    # # Ultralytics YOLO-seg result
    # result = model(rgb)[0]

    # seg3d = yolo_result_to_pcd_segments(
    #     left,
    #     right,
    #     rgb,
    #     result,
    #     output_dir="out/frame_000",
    #     frame_name="frame_000",
    #     input_color_order="BGR",
    #     output_frame="rgb",
    #     overlap_policy="highest_confidence",
    #     max_depth_m=2.0,
    #     min_points=50,
    # )
    pass
