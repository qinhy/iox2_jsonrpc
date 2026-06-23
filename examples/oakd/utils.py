from __future__ import annotations

import base64, json, logging
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any

import av
import cv2
import depthai as dai
import numpy as np

RGB_SOCKET = dai.CameraBoardSocket.CAM_A
LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C
CROP = dai.ImgResizeMode.CROP

__all__ = (
    "DepthAIH264Decoder", "FullRGBStereoH264Reader", "RGBStereoCompositor",
    "configure_file_logging", "debug_h264_msg", "decode_mjpeg_frame",
    "encode_frame_as_jpeg_base64", "make_full_rgb_encoded_pipeline",
    "make_full_rgb_h264_pipeline", "make_full_rgb_pipeline",
    "make_full_rgb_stereo_h264_synced_pipeline", "make_rgb_h264_pipeline",
    "make_rgb_pipeline", "make_rgb_stereo_combined_pipeline", "print_json_result",
    "save_base64_jpeg", "shorten_capture_result",
)


def configure_file_logging(log_file: str | Path = "client.log") -> None:
    """Configure simple file logging for CLI clients."""
    logging.basicConfig(filename=str(log_file), level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")


def encode_frame_as_jpeg_base64(frame: Any, jpeg_quality: int) -> str:
    """Encode an OpenCV frame as a base64 JPEG string."""
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok: raise RuntimeError("Failed to encode camera frame as JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def save_base64_jpeg(jpeg_base64: str, filename: str | Path) -> Path:
    """Decode a base64 JPEG string and save it to disk."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(jpeg_base64))
    return path


def shorten_capture_result(result: Any, *, output_dir: str | Path = ".") -> Any:
    """Save large JPEG payloads to disk and replace them with placeholders."""
    if not isinstance(result, dict): return result
    compact = dict(result)
    jpeg = compact.get("jpeg_base64")
    if jpeg:
        filename = Path(output_dir) / f"capture_{compact.get('frame_id')}.jpg"
        save_base64_jpeg(jpeg, filename)
        compact["jpeg_base64"] = f"<saved to {filename}, {len(jpeg)} base64 chars>"
    return compact


def print_json_result(title: str, result: Any) -> None:
    """Print and log a JSON-serializable RPC result."""
    text = json.dumps(result, indent=2, default=str)
    logging.info(text)
    print(f"\n=== {title} ===")
    print(text)


def _camera(pipeline: dai.Pipeline, socket: Any = RGB_SOCKET):
    return pipeline.create(dai.node.Camera).build(socket)


def _control_queue(camera: Any, label: str = "DepthAI") -> Any | None:
    try: return camera.inputControl.createInputQueue()
    except Exception:
        logging.info("%s inputControl queue is unavailable", label)
        return None


def _output_queue(output: Any, max_size: int, blocking: bool = False):
    return output.createOutputQueue(maxSize=int(max_size), blocking=blocking)


def _sized_output(camera: Any, *, fps: float, size: tuple[int, int], frame_type: Any):
    return camera.requestOutput(size=(int(size[0]), int(size[1])), type=frame_type,
                                resizeMode=CROP, fps=float(fps))


def _full_output(camera: Any, *, fps: float, frame_type: Any | None = None):
    kwargs = {"fps": float(fps), "useHighestResolution": True}
    if frame_type is not None: kwargs["type"] = frame_type
    return camera.requestFullResolutionOutput(**kwargs)


def _h264_profile():
    profiles = dai.VideoEncoderProperties.Profile
    return getattr(profiles, "H264_BASELINE", profiles.H264_MAIN)


def _align_h264_width(width: int) -> int:
    """H264 encoders prefer width aligned to 32 pixels."""
    return max(32, (int(width) // 32) * 32)


def _as_uint8_array(data: Any) -> np.ndarray:
    return np.frombuffer(data, np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else np.asarray(data, np.uint8)


def _as_bytes(data: Any) -> bytes:
    return data.tobytes() if hasattr(data, "tobytes") else bytes(data)


def _gray_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame


def _compose_rgb_stereo(rgb: np.ndarray, left: np.ndarray, right: np.ndarray,
                        *, stereo_height: int | None = None) -> np.ndarray:
    rgb_h, rgb_w = rgb.shape[:2]
    stereo_h = int(stereo_height if stereo_height is not None else rgb_h // 2)
    left = _gray_to_bgr(cv2.resize(left, (rgb_w // 2, stereo_h)))
    right = _gray_to_bgr(cv2.resize(right, (rgb_w // 2, stereo_h)))
    return np.ascontiguousarray(np.vstack([rgb, np.hstack([left, right])]))


def _make_h264_encoder(pipeline: dai.Pipeline, frame_out: Any, *, fps: float,
                       bitrate_kbps: int, keyframe_frequency: int | None = None,
                       profile: Any | None = None):
    encoder = pipeline.create(dai.node.VideoEncoder).build(
        frame_out, frameRate=float(fps), profile=profile or _h264_profile())
    encoder.setBitrateKbps(int(bitrate_kbps))
    encoder.setNumBFrames(0)
    encoder.setKeyframeFrequency(int(keyframe_frequency or max(1, int(float(fps)))))
    return encoder


def make_full_rgb_pipeline(device: dai.Device, *, fps: float, queue_max_size: int):
    """Build a full-resolution RGB DepthAI pipeline."""
    pipeline = dai.Pipeline(device)
    cam = _camera(pipeline)
    return pipeline, _output_queue(_full_output(cam, fps=fps), queue_max_size), _control_queue(cam)


def make_rgb_pipeline(device: dai.Device, *, fps: float, width: int, height: int,
                      queue_max_size: int):
    """Build a cropped BGR RGB pipeline at the requested size."""
    pipeline = dai.Pipeline(device)
    cam = _camera(pipeline)
    out = _sized_output(cam, fps=fps, size=(width, height), frame_type=dai.ImgFrame.Type.BGR888p)
    return pipeline, _output_queue(out, queue_max_size), _control_queue(cam)


class RGBStereoCompositor(dai.node.HostNode):
    """Host node that stacks RGB above side-by-side stereo mono frames."""

    def __init__(self):
        dai.node.HostNode.__init__(self)
        self.output = self.createOutput()
        self.width = self.height = self.stereo_height = 0

    def build(self, rgb_out: Any, left_out: Any, right_out: Any, *, width: int,
              height: int, stereo_height: int):
        self.width = int(width)
        self.height = int(height)
        self.stereo_height = int(stereo_height)
        self.link_args(rgb_out, left_out, right_out)
        return self

    def process(self, rgb_msg: Any, left_msg: Any, right_msg: Any) -> None:
        rgb = cv2.resize(rgb_msg.getCvFrame(), (self.width, self.height))
        final = _compose_rgb_stereo(rgb, left_msg.getCvFrame(), right_msg.getCvFrame(),
                                    stereo_height=self.stereo_height)
        out = dai.ImgFrame()
        out.setData(final)
        out.setWidth(final.shape[1])
        out.setHeight(final.shape[0])
        out.setType(dai.ImgFrame.Type.BGR888i)
        try:
            out.setTimestamp(rgb_msg.getTimestamp())
            out.setSequenceNum(rgb_msg.getSequenceNum())
        except Exception: pass
        self.output.send(out)


def make_rgb_stereo_combined_pipeline(device: dai.Device, *, fps: float, width: int,
                                      height: int, queue_max_size: int,
                                      stereo_height: int | None = None):
    width = int(width)
    height = int(height)
    stereo_height = int(stereo_height if stereo_height is not None else height // 2)
    pipeline = dai.Pipeline(device)
    cam_rgb, cam_left, cam_right = _camera(pipeline), _camera(pipeline, LEFT_SOCKET), _camera(pipeline, RIGHT_SOCKET)
    rgb = _sized_output(cam_rgb, fps=fps, size=(width, height), frame_type=dai.ImgFrame.Type.BGR888p)
    left = _sized_output(cam_left, fps=fps, size=(width // 2, stereo_height), frame_type=dai.ImgFrame.Type.GRAY8)
    right = _sized_output(cam_right, fps=fps, size=(width // 2, stereo_height), frame_type=dai.ImgFrame.Type.GRAY8)
    comp = pipeline.create(RGBStereoCompositor).build(
        rgb, left, right, width=width, height=height, stereo_height=stereo_height)
    return pipeline, _output_queue(comp.output, queue_max_size), _control_queue(cam_rgb)


def make_full_rgb_encoded_pipeline(device: dai.Device, *, fps: float, queue_max_size: int,
                                   profile=dai.VideoEncoderProperties.Profile.MJPEG,
                                   quality: int = 90):
    """Build a full-resolution RGB pipeline using on-device encoding."""
    pipeline = dai.Pipeline(device)
    cam = _camera(pipeline)
    rgb = _full_output(cam, fps=fps, frame_type=dai.ImgFrame.Type.NV12)
    encoder = pipeline.create(dai.node.VideoEncoder).build(rgb, frameRate=float(fps), profile=profile)
    if profile == dai.VideoEncoderProperties.Profile.MJPEG: encoder.setQuality(int(quality))
    return pipeline, _output_queue(encoder.out, queue_max_size), _control_queue(cam)


def decode_mjpeg_frame(encoded_msg: Any) -> np.ndarray:
    frame = cv2.imdecode(_as_uint8_array(encoded_msg.getData()), cv2.IMREAD_COLOR)
    if frame is None: raise RuntimeError("Failed to decode MJPEG frame")
    return frame


def make_full_rgb_h264_pipeline(device: dai.Device, *, fps: float,
                                queue_max_size: int = 30, bitrate_kbps: int = 4000):
    pipeline = dai.Pipeline(device)
    cam = _camera(pipeline)
    rgb = _full_output(cam, fps=fps, frame_type=dai.ImgFrame.Type.NV12)
    encoder = _make_h264_encoder(pipeline, rgb, fps=fps, bitrate_kbps=bitrate_kbps,
                                 profile=dai.VideoEncoderProperties.Profile.H264_MAIN)
    return pipeline, _output_queue(encoder.out, queue_max_size, True), _control_queue(cam)


def make_rgb_h264_pipeline(device: dai.Device, *, fps: float, width: int = 1920,
                           height: int = 1080, queue_max_size: int = 30,
                           bitrate_kbps: int = 6000):
    pipeline = dai.Pipeline(device)
    cam = _camera(pipeline)
    rgb = _sized_output(cam, fps=fps, size=(_align_h264_width(width), int(height)),
                        frame_type=dai.ImgFrame.Type.NV12)
    encoder = _make_h264_encoder(pipeline, rgb, fps=fps, bitrate_kbps=bitrate_kbps)
    return pipeline, _output_queue(encoder.out, queue_max_size, True), _control_queue(cam)


def _annex_b_nal_types(data: bytes) -> list[int]:
    types, i = [], 0
    while i < len(data) - 5:
        if data[i:i + 4] == b"\x00\x00\x00\x01": nal_start = i + 4
        elif data[i:i + 3] == b"\x00\x00\x01": nal_start = i + 3
        else:
            i += 1
            continue
        types.append(data[nal_start] & 0x1F)
        i = nal_start + 1
    return types


def debug_h264_msg(msg: Any, label: str = "h264") -> None:
    data = _as_bytes(msg.getData())
    print(f"{label}: {len(data)} bytes, first={data[:16].hex(' ')}")
    print(f"{label}: nal_types={_annex_b_nal_types(data)}")


class DepthAIH264Decoder:
    """Small low-latency PyAV decoder for DepthAI Annex-B H264 chunks."""

    def __init__(self, output_format: str = "bgr24"):
        self.codec = av.CodecContext.create("h264", "r")
        self.codec.thread_count = 1
        self.output_format = output_format

    def decode(self, encoded_msg: Any) -> list[np.ndarray]:
        chunk = _as_bytes(encoded_msg.getData())
        if not chunk: return []
        try: frames = self.codec.decode(av.Packet(chunk))
        except av.InvalidDataError: return []
        return [frame.to_ndarray(format=self.output_format) for frame in frames]

    def flush(self) -> list[np.ndarray]:
        try: frames = self.codec.decode(None)
        except Exception: return []
        return [frame.to_ndarray(format=self.output_format) for frame in frames]


def make_full_rgb_stereo_h264_synced_pipeline(
    device: dai.Device, *, fps: float, queue_max_size: int = 30,
    rgb_bitrate_kbps: int = 8000, mono_bitrate_kbps: int = 2000,
    stereo_sync_ms: float = 2.0,
):
    pipeline = dai.Pipeline(device)
    cam_rgb, cam_left, cam_right = _camera(pipeline), _camera(pipeline, LEFT_SOCKET), _camera(pipeline, RIGHT_SOCKET)
    rgb = _full_output(cam_rgb, fps=fps, frame_type=dai.ImgFrame.Type.NV12)
    left = _full_output(cam_left, fps=fps, frame_type=dai.ImgFrame.Type.YUV400p)
    right = _full_output(cam_right, fps=fps, frame_type=dai.ImgFrame.Type.YUV400p)

    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=float(stereo_sync_ms)))
    sync.setSyncAttempts(-1)
    left.link(sync.inputs["left"])
    right.link(sync.inputs["right"])
    demux = pipeline.create(dai.node.MessageDemux)
    sync.out.link(demux.input)

    rgb_enc = _make_h264_encoder(pipeline, rgb, fps=fps, bitrate_kbps=rgb_bitrate_kbps)
    left_enc = _make_h264_encoder(pipeline, demux.outputs["left"], fps=fps, bitrate_kbps=mono_bitrate_kbps)
    right_enc = _make_h264_encoder(pipeline, demux.outputs["right"], fps=fps, bitrate_kbps=mono_bitrate_kbps)
    queue_size = max(30, int(queue_max_size))
    return (pipeline, _output_queue(rgb_enc.out, queue_size), _output_queue(left_enc.out, queue_size),
            _output_queue(right_enc.out, queue_size), _control_queue(cam_rgb, "DepthAI RGB"))


class FullRGBStereoH264Reader:
    """Decode and compose synchronized full RGB plus left/right H264 streams."""

    def __init__(self, *, output_format: str = "bgr24", max_buffered_frames: int = 120):
        self.rgb_decoder = DepthAIH264Decoder(output_format=output_format)
        self.left_decoder = DepthAIH264Decoder(output_format="gray")
        self.right_decoder = DepthAIH264Decoder(output_format="gray")
        self.rgb_frames = deque(maxlen=max_buffered_frames)
        self.left_frames = deque(maxlen=max_buffered_frames)
        self.right_frames = deque(maxlen=max_buffered_frames)

    def decode_packets(self, rgb_packet: Any | None = None,
                       left_packet: Any | None = None,
                       right_packet: Any | None = None) -> None:
        """Decode packets and keep all decoded frames from each stream."""
        if rgb_packet is not None: self.rgb_frames.extend(self.rgb_decoder.decode(rgb_packet))
        if left_packet is not None: self.left_frames.extend(self.left_decoder.decode(left_packet))
        if right_packet is not None: self.right_frames.extend(self.right_decoder.decode(right_packet))

    def has_complete_frame_set(self) -> bool:
        return all((self.rgb_frames, self.left_frames, self.right_frames))

    def compose_next(self, *, stereo_height: int | None = None):
        """Consume and compose the next RGB/left/right frame set."""
        if not self.has_complete_frame_set(): return None
        return self._compose(self.rgb_frames.popleft(), self.left_frames.popleft(),
                             self.right_frames.popleft(), stereo_height=stereo_height)

    def compose_latest(self, *, stereo_height: int | None = None,
                       clear_old: bool = True) -> list[np.ndarray]:
        """Return newest raw RGB frames plus their RGB+stereo composite frames."""
        n = min(len(self.rgb_frames), len(self.left_frames), len(self.right_frames))
        if n <= 0: return []
        frames: list[np.ndarray] = []
        for rgb, left, right in zip(list(self.rgb_frames)[-n:], list(self.left_frames)[-n:], list(self.right_frames)[-n:]):
            frames.append(rgb)
            frames.append(self._compose(rgb, left, right, stereo_height=stereo_height))
        if clear_old: self.clear_buffers()
        return frames

    def _compose(self, rgb: np.ndarray, left: np.ndarray, right: np.ndarray,
                 *, stereo_height: int | None = None) -> np.ndarray:
        return _compose_rgb_stereo(rgb, left, right, stereo_height=stereo_height)

    def buffer_sizes(self) -> tuple[int, int, int]:
        return len(self.rgb_frames), len(self.left_frames), len(self.right_frames)

    def clear_buffers(self) -> None:
        self.rgb_frames.clear()
        self.left_frames.clear()
        self.right_frames.clear()

    def take_all_rgb_frames(self) -> list[np.ndarray]:
        frames = list(self.rgb_frames)
        self.rgb_frames.clear()
        return frames

    def compose_rgb_with_latest_stereo(self, *, stereo_height: int | None = None,
                                       clear_rgb: bool = True) -> list[np.ndarray]:
        """Compose every RGB frame with newest stereo, or return raw RGB if absent."""
        if not self.rgb_frames: return []
        rgb_list = list(self.rgb_frames)
        if clear_rgb: self.rgb_frames.clear()
        if not self.left_frames or not self.right_frames: return rgb_list
        latest_left, latest_right = self.left_frames[-1], self.right_frames[-1]
        self._keep_latest_stereo(latest_left, latest_right)
        return [self._compose(rgb, latest_left, latest_right, stereo_height=stereo_height)
                for rgb in rgb_list]

    def _keep_latest_stereo(self, left: np.ndarray, right: np.ndarray) -> None:
        self.left_frames.clear()
        self.right_frames.clear()
        self.left_frames.append(left)
        self.right_frames.append(right)
