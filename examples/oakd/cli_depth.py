from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

# Match the local import style used by cli_camera.py, but keep this CLI usable
# when it is placed next to rpc_server_depth.py.
_THIS_DIR = Path(__file__).absolute().parent
for _path in (
    _THIS_DIR,
    _THIS_DIR.parent,
    Path(os.path.dirname(os.path.dirname(_THIS_DIR.parent))),
):
    _path_s = str(_path)
    if _path_s not in sys.path:
        sys.path.append(_path_s)


try:  # noqa: E402
    from utils import configure_file_logging, print_json_result
except Exception:  # pragma: no cover - fallback for standalone use
    def configure_file_logging(*args: Any, **kwargs: Any) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def print_json_result(method: str, result: Any) -> None:
        print(f"\n=== {method} ===")
        print(json.dumps(result, indent=2, default=str))


TranslationUnit = Literal["m", "cm", "mm"]

_CAMERA_INFO_KEYS = {
    "service",
    "source_translation_unit",
    "rgb_resolution",
    "left_resolution",
    "right_resolution",
    "rgb_intrinsics",
    "left_intrinsics",
    "right_intrinsics",
    "left_to_right_extrinsics",
    "left_to_rgb_extrinsics",
    "rgb_distortion",
    "left_distortion",
    "right_distortion",
}


class DepthRpcApi:
    """Thin JSON-RPC client wrapper for the minimal depth converter methods."""

    def __init__(self, registry: Any, *, controller_name: str = "depth") -> None:
        from iox2_jsonrpc.iceoryx import Iox2RpcRegistry

        self.registry: Iox2RpcRegistry = registry
        self.controller_name = controller_name

    @classmethod
    def discover(cls, *, controller_name: str = "depth") -> "DepthRpcApi":
        from iox2_jsonrpc.iceoryx import Iox2RpcRegistry

        registry = Iox2RpcRegistry.discover_all()
        logging.info("\n=== Discovered JSON-RPC catalog ===")
        logging.info(json.dumps(registry.catalog(), indent=2, default=str))
        return cls(registry, controller_name=controller_name)

    def method(self, name: str) -> str:
        return f"{self.controller_name}.{name}"

    def call(self, name: str, params: dict[str, Any] | None = None, *, timeout_s: float = 300.0) -> Any:
        method = self.method(name)
        logging.info("\n=== %s ===", method)

        if params is None:
            return self.registry.call_unique(method, timeout_s=timeout_s)

        return self.registry.call_unique(method, params, timeout_s=timeout_s)

    def call_and_print(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float = 300.0,
    ) -> Any:
        result = self.call(name, params=params, timeout_s=timeout_s)
        print_json_result(self.method(name), result)
        return result

    def camera_info(self, *, timeout_s: float = 5.0) -> Any:
        return self.call("camera_info", timeout_s=timeout_s)

    def set_camera_info(self, params: dict[str, Any], *, timeout_s: float = 5.0) -> Any:
        return self.call("set_camera_info", params=params, timeout_s=timeout_s)

    def backend(self, *, timeout_s: float = 5.0) -> Any:
        return self.call("backend", timeout_s=timeout_s)

    def set_backend(self, params: dict[str, Any], *, timeout_s: float = 5.0) -> Any:
        return self.call("set_backend", params=params, timeout_s=timeout_s)

    def to_pcd(self, params: dict[str, Any], *, timeout_s: float = 300.0) -> Any:
        return self.call("to_pcd", params=params, timeout_s=timeout_s)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_camera_info_params(
    path: str | Path | None,
    *,
    source_translation_unit: TranslationUnit = "cm",
) -> dict[str, Any]:
    """Load a calibration JSON and keep only rpc_server_depth camera-info keys.

    Accepts either a raw calibration object or a wrapper like:
        {"camera_calib": {...}}
    Extra fields from rpc_server_camera.CameraCalibrationParams, such as
    stereo_baseline_cm and stereo_translation_units_hint, are ignored.
    """
    if path is None:
        return {"source_translation_unit": source_translation_unit}

    data = _read_json(path)
    if "camera_calib" in data and isinstance(data["camera_calib"], dict):
        data = data["camera_calib"]

    params = {key: value for key, value in data.items() if key in _CAMERA_INFO_KEYS}
    params["source_translation_unit"] = str(params.get("source_translation_unit") or source_translation_unit)
    return params


def _require_path_arg(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name)
    if not value:
        cli_name = "--" + name.replace("_path", "").replace("_", "-")
        raise SystemExit(f"{cli_name} is required for {args.mode}")
    return str(Path(value).expanduser())


def _selected_backend(args: argparse.Namespace, *, default_for_set: bool = False) -> str | None:
    if args.dnn:
        return "dnn"
    if args.backend is not None:
        return args.backend
    if default_for_set:
        return "sgbm"
    return None


def _backend_params_from_args(args: argparse.Namespace, *, default_for_set: bool = False) -> dict[str, Any]:
    backend = _selected_backend(args, default_for_set=default_for_set)
    params: dict[str, Any] = {}
    if backend is not None:
        params["backend"] = backend

    if backend == "dnn":
        params.update(
            {
                "repo_dir": str(Path(args.repo_dir).expanduser()) if args.repo_dir else None,
                "model_path": str(Path(args.model_path).expanduser()) if args.model_path else None,
                "model_dir": str(Path(args.model_dir).expanduser()) if args.model_dir else None,
                "device": args.device,
                "valid_iters": int(args.valid_iters),
                "max_disp": int(args.max_disp),
                "hiera": bool(args.hiera),
                "model_scale": float(args.model_scale),
                "stereo_input_color_order": args.stereo_input_color_order,
                "remove_invisible": not bool(args.keep_invisible),
            }
        )

    return params


def _to_pcd_params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "left_path": _require_path_arg(args, "left_path"),
        "right_path": _require_path_arg(args, "right_path"),
        "rgb_path": _require_path_arg(args, "rgb_path"),
        "output_path": str(Path(args.output_path).expanduser()),
        "input_color_order": args.input_color_order,
        "rgb_image_is_undistorted": bool(args.rgb_image_is_undistorted),
        "alpha": float(args.alpha),
        "min_disparity": int(args.min_disparity),
        "num_disparities": int(args.num_disparities),
        "block_size": int(args.block_size),
        "stride": int(args.stride),
        "output_frame": args.output_frame,
        "save_binary_pcd": not bool(args.ascii_pcd),
    }

    backend_params = _backend_params_from_args(args, default_for_set=False)
    params.update(backend_params)

    if args.max_depth_m is not None:
        params["max_depth_m"] = float(args.max_depth_m)
    else:
        params["max_depth_m"] = None

    if args.inline_calib:
        params["camera_calib"] = _load_camera_info_params(
            args.calib_json,
            source_translation_unit=args.source_translation_unit,
        )

    return params


def run_camera_info_client(*, timeout_s: float = 5.0) -> Any:
    configure_file_logging()
    api = DepthRpcApi.discover()
    return api.call_and_print("camera_info", timeout_s=timeout_s)


def run_set_camera_info_client(
    *,
    calib_json: str | Path | None = None,
    source_translation_unit: TranslationUnit = "cm",
    timeout_s: float = 5.0,
) -> Any:
    configure_file_logging()
    api = DepthRpcApi.discover()
    params = _load_camera_info_params(calib_json, source_translation_unit=source_translation_unit)
    return api.call_and_print("set_camera_info", params=params, timeout_s=timeout_s)


def run_backend_info_client(*, timeout_s: float = 5.0) -> Any:
    configure_file_logging()
    api = DepthRpcApi.discover()
    return api.call_and_print("backend", timeout_s=timeout_s)


def run_set_backend_client(args: argparse.Namespace) -> Any:
    configure_file_logging()
    api = DepthRpcApi.discover()
    params = _backend_params_from_args(args, default_for_set=True)
    return api.call_and_print("set_backend", params=params, timeout_s=args.timeout_s)


def run_to_pcd_client(args: argparse.Namespace) -> Any:
    configure_file_logging()
    api = DepthRpcApi.discover()

    if args.calib_json and not args.inline_calib:
        camera_info = _load_camera_info_params(
            args.calib_json,
            source_translation_unit=args.source_translation_unit,
        )
        api.call_and_print("set_camera_info", params=camera_info, timeout_s=args.timeout_s)

    params = _to_pcd_params_from_args(args)
    return api.call_and_print("to_pcd", params=params, timeout_s=args.timeout_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal RGB + stereo image paths -> PCD JSON-RPC CLI",
    )
    parser.add_argument(
        "mode",
        choices=[
            "server",
            "serve",
            "info",
            "camera-info",
            "set-info",
            "backend",
            "backend-info",
            "set-backend",
            "to-pcd",
            "convert",
            "client",
        ],
        help="Run the RPC server, inspect/set calibration/backend, or convert image paths to PCD.",
    )

    parser.add_argument("--left", "--left-path", dest="left_path", default=None, help="Left mono image path.")
    parser.add_argument("--right", "--right-path", dest="right_path", default=None, help="Right mono image path.")
    parser.add_argument("--rgb", "--rgb-path", dest="rgb_path", default=None, help="RGB/color image path.")
    parser.add_argument(
        "--output",
        "--output-path",
        dest="output_path",
        default="colored_cloud.pcd",
        help="Output .pcd path.",
    )

    parser.add_argument(
        "--calib-json",
        "--camera-calib-json",
        dest="calib_json",
        default=None,
        help="Optional camera calibration JSON. Used by set-info or before to-pcd.",
    )
    parser.add_argument(
        "--source-translation-unit",
        choices=["m", "cm", "mm"],
        default="cm",
        help="Unit for calibration extrinsic translations. Luxonis calibration is usually cm.",
    )
    parser.add_argument(
        "--inline-calib",
        action="store_true",
        help="Send --calib-json inside depth.to_pcd instead of first calling depth.set_camera_info.",
    )

    parser.add_argument(
        "--backend",
        choices=["sgbm", "dnn"],
        default=None,
        help="Depth backend. For to-pcd, omit this to use the server's depth.set_backend setting.",
    )
    parser.add_argument(
        "--dnn",
        action="store_true",
        help="Shortcut for --backend dnn.",
    )

    parser.add_argument("--input-color-order", choices=["RGB", "BGR"], default="BGR")
    parser.add_argument("--rgb-image-is-undistorted", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--num-disparities", type=int, default=128, help="SGBM only.")
    parser.add_argument("--block-size", type=int, default=5, help="SGBM only.")
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=10.0,
        help="Reject points farther than this many meters. Pass --max-depth-m -1 to disable.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output-frame", choices=["left", "left_rectified"], default="left")
    parser.add_argument("--ascii-pcd", action="store_true", help="Write ASCII PCD instead of binary PCD.")

    # Fast-FoundationStereo / pcd_dnn_utils options.
    parser.add_argument("--repo-dir", default=None, help="Fast-FoundationStereo repo directory. DNN only.")
    parser.add_argument("--model-path", default=None, help="Fast-FoundationStereo .pth checkpoint path. DNN only.")
    parser.add_argument("--model-dir", default=None, help="Directory containing the Fast-FoundationStereo checkpoint. DNN only.")
    parser.add_argument("--device", default="cuda", help="DNN device, usually cuda or cpu.")
    parser.add_argument("--valid-iters", type=int, default=8, help="DNN refinement iterations.")
    parser.add_argument("--max-disp", type=int, default=192, help="DNN maximum disparity.")
    parser.add_argument("--hiera", action="store_true", help="Use DNN hierarchical inference if the checkpoint supports it.")
    parser.add_argument("--model-scale", type=float, default=1.0, help="Resize stereo images before DNN inference.")
    parser.add_argument(
        "--stereo-input-color-order",
        choices=["RGB", "BGR"],
        default="RGB",
        help="Color order for 3-channel stereo inputs before DNN inference.",
    )
    parser.add_argument(
        "--keep-invisible",
        action="store_true",
        help="DNN only: keep pixels whose corresponding right-camera u coordinate falls outside the image.",
    )

    parser.add_argument("--timeout-s", type=float, default=300.0, help="RPC timeout in seconds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mode = args.mode.lower()

    if args.max_depth_m is not None and args.max_depth_m < 0:
        args.max_depth_m = None

    if mode in {"server", "serve"}:
        from rpc_server_depth import run_server

        run_server()
    elif mode in {"info", "camera-info"}:
        run_camera_info_client(timeout_s=args.timeout_s)
    elif mode == "set-info":
        run_set_camera_info_client(
            calib_json=args.calib_json,
            source_translation_unit=args.source_translation_unit,
            timeout_s=args.timeout_s,
        )
    elif mode in {"backend", "backend-info"}:
        run_backend_info_client(timeout_s=args.timeout_s)
    elif mode == "set-backend":
        run_set_backend_client(args)
    elif mode in {"to-pcd", "convert", "client"}:
        run_to_pcd_client(args)
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()

# Example:

# ```bash
# python cli_depth.py server
# ```

# Set backend once:

# ```bash
# python cli_depth.py set-backend --backend sgbm
# ```

# Or set DNN once:

# ```bash
# python cli_depth.py set-backend \
#   --dnn \
#   --repo-dir /path/to/Fast-FoundationStereo \
#   --model-path /path/to/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
#   --model-scale 0.5
# ```

# Check current backend:

# ```bash
# python cli_depth.py backend
# ```

# Then convert without specifying backend:

# ```bash
# python cli_depth.py to-pcd \
#   --left left.png \
#   --right right.png \
#   --rgb rgb.png \
#   --output colored_cloud.pcd
# ```

# You can still override for one call:

# ```bash
# python cli_depth.py to-pcd \
#   --backend sgbm \
#   --left left.png \
#   --right right.png \
#   --rgb rgb.png \
#   --output colored_cloud_sgbm.pcd
# ```
