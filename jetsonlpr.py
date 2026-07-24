#!/usr/bin/env python3
from __future__ import annotations

import base64
import logging
import json
import os
import platform
import queue
import re
import signal
import socket
import struct
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO


LOGGER = logging.getLogger("jetsonlpr")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ts_to_display(ts: str) -> str:
    dt = datetime.strptime(ts, "%Y%m%d_%H%M%S_%f")
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}_{dt.microsecond // 1000:03d}"


def put_latest(q: queue.Queue, item: object) -> None:
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        q.get_nowait()
    except queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except queue.Full:
        LOGGER.warning("Queue remained full; dropping newest item.")


def encode_jpg(image: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")
    return bytes(buf)


def cleanup_plate_string(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned.upper().replace(" ", "")


def first_last_match(left_right_len: int, a: str, b: str) -> bool:
    if len(a) < left_right_len or len(b) < left_right_len:
        return False
    return a[:left_right_len] == b[:left_right_len] or a[-left_right_len:] == b[-left_right_len:]


@dataclass
class CameraConfig:
    index: int
    purpose: str
    device_path: str
    width: int
    height: int
    fps: int
    detect_conf: float
    name: str = ""


@dataclass
class AppConfig:
    config_path: str
    discord_bot_token: str
    discord_user_ids: List[str]
    street_raw: str
    street_pretty: str
    city_raw: str
    city_pretty: str
    skip_lpr: set
    cameras: List[CameraConfig]
    model_path: str
    lpr_size: int
    bbox_pad_ratio: float
    edge_margin_px: int
    frame_jpeg_quality: int
    discord_frame_x: int
    discord_frame_y: int
    live_jpeg_quality: int
    live_width: int
    live_height: int
    socket_path: str
    output_root: Path
    all_plates_root: Path
    queue_maxsize: int
    group_window_sec: float
    group_idle_sec: float
    recent_window_sec: float
    plate_history_size: int
    parked_history_match_count: int
    stationary_ratio_threshold: float
    recent_approx_len: int
    history_approx_len: int
    motion_diff_threshold: float
    motion_changed_ratio_threshold: float
    min_plate_len: int
    max_plate_len: int
    reject_prefixes: List[str]
    pattern_bonus_regex: str
    correction_table_path: str
    enable_live_video: bool
    enable_discord: bool
    ocr_config: str
    ocr_ckpt: str
    ocr_device: str
    ocr_amp: bool
    ocr_topk_templates: int
    ocr_conf_gate: float
    ocr_char_topk: int
    ocr_beam: int
    ocr_gap_delta: float
    ocr_blur_gate: float


@dataclass
class PlateJob:
    camera_index: int
    camera_label: str
    ts: str
    base_name: str
    x1_orig: int
    y1_orig: int
    x2_orig: int
    y2_orig: int
    crop_jpg: bytes
    preview_jpg: bytes
    full_jpg: bytes
    conf: float
    arrival_mono: float = field(default_factory=time.monotonic)

    @property
    def width(self) -> int:
        return max(1, self.x2_orig - self.x1_orig)

    @property
    def center_x(self) -> float:
        return (self.x1_orig + self.x2_orig) / 2.0


@dataclass
class OcrCandidate:
    job: PlateJob
    raw_text: str
    cleaned_text: str
    final_plate: str
    confidence: float
    corrected: bool
    score: float = 0.0


@dataclass
class GroupBuffer:
    camera_index: int
    first_mono: float
    last_mono: float
    jobs: List[PlateJob] = field(default_factory=list)
    avg_center_x: float = 0.0
    avg_width: float = 1.0

    def add(self, job: PlateJob) -> None:
        if not self.jobs:
            self.jobs.append(job)
            self.avg_center_x = job.center_x
            self.avg_width = job.width
            self.first_mono = job.arrival_mono
            self.last_mono = job.arrival_mono
            return

        self.jobs.append(job)
        self.last_mono = job.arrival_mono
        n = float(len(self.jobs))
        self.avg_center_x = ((self.avg_center_x * (n - 1.0)) + job.center_x) / n
        self.avg_width = ((self.avg_width * (n - 1.0)) + job.width) / n


@dataclass
class NotificationEvent:
    text_plain: str
    text_discord: str
    image_bytes: bytes
    image_filename: str


class FrameStore:
    """
    Live-video compatibility store.

    This intentionally mirrors the original easylpr-client.py design where
    the browser stream reuses the most recently captured camera frames instead
    of opening the cameras a second time. In other words, `last_frame` is the
    in-memory equivalent of the old `last_frame[]` array and is updated only by
    the camera capture workers.
    """

    def __init__(self, num_cameras: int):
        self.last_frame: List[Optional[np.ndarray]] = [None] * num_cameras
        self.frame_lock: List[threading.Lock] = [threading.Lock() for _ in range(num_cameras)]

    def update(self, camera_index: int, frame: np.ndarray) -> None:
        with self.frame_lock[camera_index]:
            self.last_frame[camera_index] = frame.copy()

    def snapshot(self, camera_index: int) -> Optional[np.ndarray]:
        with self.frame_lock[camera_index]:
            frame = self.last_frame[camera_index]
            return None if frame is None else frame.copy()


class CorrectionTable:
    HASH_MASK = 0x0FFFFFFF

    def __init__(self, path: str):
        self.path = path
        self.table: Dict[str, List[Tuple[str, str]]] = {}
        if path and os.path.exists(path):
            self.table = self._load_table(path)
            LOGGER.info("Loaded %d correction buckets from %s", len(self.table), path)
        elif path:
            LOGGER.warning("Correction table not found: %s; continuing without it.", path)

    @classmethod
    def _hash_function(cls, input_string: str) -> int:
        value = 0
        for char in input_string:
            value = (value * 31 + ord(char)) & cls.HASH_MASK
        return value

    @staticmethod
    def _load_table(path: str) -> Dict[str, List[Tuple[str, str]]]:
        table: Dict[str, List[Tuple[str, str]]] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or ": " not in line:
                    continue
                hash_part, pairs_part = line.split(": ", 1)
                bucket = table.setdefault(hash_part.strip(), [])
                for pair in pairs_part.split(") ("):
                    cleaned = pair.replace("(", "").replace(")", "")
                    src_dst = [part.strip() for part in cleaned.split(", ")]
                    if len(src_dst) == 2:
                        bucket.append((src_dst[0], src_dst[1]))
        return table

    def correct(self, text: str) -> Tuple[str, bool]:
        if not text or not self.table:
            return text, False

        hash_key = f"{self._hash_function(text):08x}"
        for src, dst in self.table.get(hash_key, []):
            if src == text:
                return dst, True
        return text, False


class DiscordDMClient:
    API_BASE = "https://discord.com/api/v10"

    def __init__(self, token: str, user_ids: Sequence[str], enabled: bool):
        self.token = token.strip()
        self.user_ids = [user_id.strip() for user_id in user_ids if user_id.strip()]
        self.enabled = bool(enabled and self.token and self.user_ids)
        self.queue: queue.Queue = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.session = requests.Session()
        self.dm_channel_cache: Dict[str, str] = {}

    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("Discord notifications disabled.")
            return
        self.thread = threading.Thread(target=self._worker, name="discord", daemon=True)
        self.thread.start()
        LOGGER.info("Discord worker started for %d user(s).", len(self.user_ids))

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        self.session.close()

    def enqueue(self, event: NotificationEvent) -> None:
        if not self.enabled:
            return
        put_latest(self.queue, event)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bot {self.token}"}

    def _post_json(self, url: str, payload: Dict[str, object], timeout: float = 20.0) -> requests.Response:
        while True:
            resp = self.session.post(url, headers=self._headers(), json=payload, timeout=timeout)
            if resp.status_code != 429:
                return resp
            retry_after = 1.0
            try:
                retry_after = float(resp.json().get("retry_after", 1.0))
            except Exception:
                pass
            LOGGER.warning("Discord rate-limited on JSON POST; retrying in %.2fs", retry_after)
            time.sleep(retry_after)

    def _post_file(self, url: str, filename: str, image_bytes: bytes, timeout: float = 30.0) -> requests.Response:
        while True:
            files = {"files[0]": (filename, image_bytes, "image/jpeg")}
            resp = self.session.post(url, headers=self._headers(), files=files, timeout=timeout)
            if resp.status_code != 429:
                return resp
            retry_after = 1.0
            try:
                retry_after = float(resp.json().get("retry_after", 1.0))
            except Exception:
                pass
            LOGGER.warning("Discord rate-limited on file POST; retrying in %.2fs", retry_after)
            time.sleep(retry_after)

    def _get_dm_channel(self, user_id: str) -> str:
        cached = self.dm_channel_cache.get(user_id)
        if cached:
            return cached

        resp = self._post_json(
            f"{self.API_BASE}/users/@me/channels",
            {"recipient_id": user_id},
            timeout=15.0,
        )
        resp.raise_for_status()
        channel_id = resp.json()["id"]
        self.dm_channel_cache[user_id] = channel_id
        return channel_id

    def _send_to_user(self, user_id: str, event: NotificationEvent) -> None:
        channel_id = self._get_dm_channel(user_id)
        url = f"{self.API_BASE}/channels/{channel_id}/messages"

        resp = self._post_json(
            url,
            {
                "content": event.text_discord,
                "allowed_mentions": {"parse": []},
            },
            timeout=15.0,
        )
        if resp.status_code in (403, 404):
            self.dm_channel_cache.pop(user_id, None)
            channel_id = self._get_dm_channel(user_id)
            url = f"{self.API_BASE}/channels/{channel_id}/messages"
            resp = self._post_json(
                url,
                {
                    "content": event.text_discord,
                    "allowed_mentions": {"parse": []},
                },
                timeout=15.0,
            )
        resp.raise_for_status()

        resp = self._post_file(url, event.image_filename, event.image_bytes, timeout=30.0)
        if resp.status_code in (403, 404):
            self.dm_channel_cache.pop(user_id, None)
            channel_id = self._get_dm_channel(user_id)
            url = f"{self.API_BASE}/channels/{channel_id}/messages"
            resp = self._post_file(url, event.image_filename, event.image_bytes, timeout=30.0)
        resp.raise_for_status()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                event = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            for user_id in self.user_ids:
                try:
                    self._send_to_user(user_id, event)
                except Exception as exc:
                    LOGGER.exception("Discord send failed for user %s: %s", user_id, exc)


class CameraLiveVideoServer:
    def __init__(
        self,
        socket_path: str,
        frame_store: FrameStore,
        config: AppConfig,
        stop_event: threading.Event,
    ):
        self.socket_path = socket_path
        self.frame_store = frame_store
        self.config = config
        self.stop_event = stop_event
        self.server_sock: Optional[socket.socket] = None
        self.accept_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.socket_path:
            LOGGER.info("Live video server disabled because SOCKET_PATH is empty.")
            return

        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.socket_path)
        try:
            os.chmod(self.socket_path, 0o666)
        except OSError:
            pass
        self.server_sock.listen(5)
        self.accept_thread = threading.Thread(target=self._accept_loop, name="live-video", daemon=True)
        self.accept_thread.start()
        LOGGER.info("CameraLiveVideoServer listening on %s", self.socket_path)

    def stop(self) -> None:
        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except OSError:
                pass
        if self.accept_thread is not None:
            self.accept_thread.join(timeout=2.0)
        if self.socket_path and os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except OSError:
                pass

    def _accept_loop(self) -> None:
        assert self.server_sock is not None
        while not self.stop_event.is_set():
            try:
                conn, _ = self.server_sock.accept()
            except OSError:
                if self.stop_event.is_set():
                    return
                continue
            LOGGER.info("Live video client connected.")
            threading.Thread(
                target=self._handle_client,
                args=(conn,),
                name="live-video-client",
                daemon=True,
            ).start()

    def _recv_exact(self, conn: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = conn.recv(size - len(data))
            if not chunk:
                raise ConnectionError("socket closed while reading")
            data.extend(chunk)
        return bytes(data)

    def _handle_client(self, conn: socket.socket) -> None:
        streaming = False
        try:
            with conn:
                while not self.stop_event.is_set():
                    try:
                        length_data = self._recv_exact(conn, 4)
                        (msg_len,) = struct.unpack("!I", length_data)
                        if msg_len <= 0:
                            LOGGER.debug("Ignoring empty live-video command")
                            continue
                        data = self._recv_exact(conn, msg_len)
                        cmd = data.decode("utf-8")
                    except Exception:
                        return

                    if cmd == "START_LIVE_VIDEO":
                        streaming = True
                        LOGGER.info("Live video streaming started for client.")
                    elif cmd == "STOP_LIVE_VIDEO":
                        streaming = False
                        LOGGER.info("Live video streaming stopped for client.")
                    elif cmd == "GET_CAMERA_INFO":
                        if not self._send_camera_info(conn):
                            return
                    elif cmd == "SEND_MORE_V2" and streaming:
                        if not self._send_live_frames(conn, include_count=True):
                            return
                    elif cmd == "SEND_MORE" and streaming:
                        if not self._send_live_frames(conn, include_count=False):
                            return
                    else:
                        LOGGER.debug("Live video command ignored: %s", cmd)
        finally:
            LOGGER.info("Live video client disconnected.")

    def _send_camera_info(self, conn: socket.socket) -> bool:
        try:
            payload = {
                "camera_count": len(self.config.cameras),
                "cameras": [
                    {
                        "index": camera.index,
                        "purpose": camera.purpose,
                        "device_path": camera.device_path,
                        "label": camera.name or f"{platform.node()}-{camera.index}",
                    }
                    for camera in self.config.cameras
                ],
            }
            data = json.dumps(payload).encode("utf-8")
            conn.sendall(struct.pack("!I", len(data)))
            conn.sendall(data)
            return True
        except OSError:
            return False

    def _send_live_frames(self, conn: socket.socket, include_count: bool = False) -> bool:
        """
        Send one JPEG block per configured camera using the latest frames already
        captured by the camera worker threads. This intentionally mirrors the old
        CameraLiveVideoServer/last_frame[] design instead of re-capturing video.
        """
        # Important: this reuses the already captured camera frames kept in
        # FrameStore/last_frame. It does not re-open or re-capture from any
        # camera device for the browser stream.
        if include_count:
            try:
                conn.sendall(struct.pack("!I", len(self.config.cameras)))
            except OSError:
                return False

        for camera in self.config.cameras:
            frame = self.frame_store.snapshot(camera.index)
            if frame is None:
                try:
                    conn.sendall(struct.pack("!I", 0))
                except OSError:
                    return False
                continue

            frame = cv2.resize(frame, (self.config.live_width, self.config.live_height))
            try:
                jpeg = encode_jpg(frame, self.config.live_jpeg_quality)
                conn.sendall(struct.pack("!I", len(jpeg)))
                conn.sendall(jpeg)
            except OSError:
                return False
        return True


class NotificationFilter:
    def __init__(self, config: AppConfig, camera_cfg: CameraConfig):
        self.config = config
        self.camera_cfg = camera_cfg
        self.lock = threading.Lock()
        self.plate_history: Deque[str] = deque(maxlen=config.plate_history_size)
        self.recent_events: Deque[Tuple[float, str]] = deque()
        self.last_x1: Optional[int] = None
        self.last_x2: Optional[int] = None

    def _purge_recent(self, now_mono: float) -> None:
        while self.recent_events and (now_mono - self.recent_events[0][0]) > self.config.recent_window_sec:
            self.recent_events.popleft()

    def _same_position(self, x1: int, x2: int) -> bool:
        if self.last_x1 is None or self.last_x2 is None:
            return False

        a = abs(x1 - self.last_x1)
        b = abs(x2 - self.last_x2)
        d = min(a, b)
        a1 = abs(x2 - x1)
        b1 = abs(self.last_x2 - self.last_x1)
        d1 = max(a1, b1)
        dv = (a + b) / 2.0
        dv1 = (a1 + b1) / 2.0
        if d1 <= 0 or dv1 <= 0:
            return False
        ratio_d = d / d1
        ratio_dv = dv / dv1
        return ratio_d < self.config.stationary_ratio_threshold and ratio_dv < self.config.stationary_ratio_threshold

    def should_notify(self, plate: str, job: PlateJob) -> Tuple[bool, str]:
        now_mono = time.monotonic()
        with self.lock:
            if plate in self.config.skip_lpr:
                return False, "plate-in-skip-list"

            self._purge_recent(now_mono)

            same_position = self._same_position(job.x1_orig, job.x2_orig)
            self.last_x1 = job.x1_orig
            self.last_x2 = job.x2_orig

            if same_position:
                self.plate_history.append(plate)
                return False, "same-position-as-last-notified"

            history_match_count = sum(
                1 for old_plate in self.plate_history if first_last_match(self.config.history_approx_len, old_plate, plate)
            )
            self.plate_history.append(plate)
            if history_match_count > self.config.parked_history_match_count:
                return False, "parked-vehicle-history"

            for _, recent_plate in self.recent_events:
                if recent_plate == plate:
                    return False, "recent-exact-duplicate"
                if first_last_match(self.config.recent_approx_len, recent_plate, plate):
                    return False, "recent-approx-duplicate"

            self.recent_events.append((now_mono, plate))
            return True, "accepted"


class YOLODetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.lock = threading.Lock()

    def predict(self, frame_small: np.ndarray, conf: float, imgsz: int):
        with self.lock:
            return self.model.predict(frame_small, conf=conf, imgsz=imgsz, verbose=False)[0]


class OcrEngine:
    def __init__(self, config: AppConfig):
        try:
            from lpr_backend_service import PlateRecognizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing lpr_backend_service.py / PlateRecognizer. Place your existing OCR backend module beside this script or in PYTHONPATH."
            ) from exc

        self.recognizer = PlateRecognizer(
            config=config.ocr_config,
            ckpt=config.ocr_ckpt,
            device=config.ocr_device,
            amp=config.ocr_amp,
            topk_templates=config.ocr_topk_templates,
            conf_gate=config.ocr_conf_gate,
            char_topk=config.ocr_char_topk,
            beam=config.ocr_beam,
            gap_delta=config.ocr_gap_delta,
            blur_gate=config.ocr_blur_gate,
        )

    def recognize(self, crop_jpg: bytes) -> Tuple[str, float]:
        b64 = base64.b64encode(crop_jpg).decode("utf-8")
        result = self.recognizer.infer_b64(b64)
        v4 = result.get("v4", {})
        raw = str(v4.get("text_raw", ""))
        conf = float(v4.get("confidence", 0.0))
        return raw, conf


class OCRWorker:
    def __init__(
        self,
        config: AppConfig,
        candidate_queue: queue.Queue,
        notify_queue: queue.Queue,
        correction_table: CorrectionTable,
        filters: Dict[int, NotificationFilter],
        stop_event: threading.Event,
    ):
        self.config = config
        self.candidate_queue = candidate_queue
        self.notify_queue = notify_queue
        self.correction_table = correction_table
        self.filters = filters
        self.stop_event = stop_event
        self.hostname = platform.node()
        self.ocr: Optional[OcrEngine] = None
        self.ocr_last_init_attempt = 0.0
        self.thread = threading.Thread(target=self._run, name="ocr", daemon=True)
        self.groups_by_camera: Dict[int, List[GroupBuffer]] = {camera.index: [] for camera in config.cameras}
        self.pattern_bonus = re.compile(config.pattern_bonus_regex) if config.pattern_bonus_regex else None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.thread.join(timeout=5.0)


    def _ensure_ocr_ready(self) -> bool:
        if self.ocr is not None:
            return True

        now_mono = time.monotonic()
        if (now_mono - self.ocr_last_init_attempt) < 5.0:
            return False

        self.ocr_last_init_attempt = now_mono
        try:
            self.ocr = OcrEngine(self.config)
            LOGGER.info("OCR engine initialized successfully.")
            return True
        except Exception as exc:
            LOGGER.exception(
                "OCR engine initialization failed; live video and capture will keep running. Retrying in 5 seconds. Error: %s",
                exc,
            )
            self.ocr = None
            return False

    def _job_matches_group(self, job: PlateJob, group: GroupBuffer) -> bool:
        width = max(job.width, int(group.avg_width), 1)
        center_gap = abs(job.center_x - group.avg_center_x)
        if center_gap <= width * 1.25:
            return True

        group_x1 = int(group.avg_center_x - group.avg_width / 2.0)
        group_x2 = int(group.avg_center_x + group.avg_width / 2.0)
        inter = max(0, min(job.x2_orig, group_x2) - max(job.x1_orig, group_x1))
        min_w = max(1, min(job.width, group_x2 - group_x1))
        overlap = inter / float(min_w)
        return overlap >= 0.25

    def _get_or_create_group(self, job: PlateJob) -> GroupBuffer:
        groups = self.groups_by_camera[job.camera_index]
        best_group: Optional[GroupBuffer] = None
        best_distance = float("inf")

        for group in groups:
            if self._job_matches_group(job, group):
                distance = abs(job.center_x - group.avg_center_x)
                if distance < best_distance:
                    best_distance = distance
                    best_group = group

        if best_group is None:
            best_group = GroupBuffer(
                camera_index=job.camera_index,
                first_mono=job.arrival_mono,
                last_mono=job.arrival_mono,
            )
            groups.append(best_group)

        best_group.add(job)
        return best_group

    def _purge_and_flush_stale_groups(self, force_all: bool = False) -> None:
        now_mono = time.monotonic()
        for camera_index, groups in list(self.groups_by_camera.items()):
            keep: List[GroupBuffer] = []
            for group in groups:
                stale = force_all or ((now_mono - group.last_mono) >= self.config.group_idle_sec)
                old = force_all or ((now_mono - group.first_mono) >= self.config.group_window_sec)
                if stale or old:
                    self._process_group(group.jobs)
                else:
                    keep.append(group)
            self.groups_by_camera[camera_index] = keep

    def _length_score(self, plate: str) -> float:
        length = len(plate)
        if length == 7:
            return 3.0
        if length == 6:
            return 2.0
        if length >= 8:
            return 1.5
        if length == 5:
            return 1.0
        if length == 4:
            return 0.5
        return -5.0

    def _build_candidate(self, job: PlateJob) -> Optional[OcrCandidate]:
        if self.ocr is None:
            return None
        raw_text, confidence = self.ocr.recognize(job.crop_jpg)
        cleaned = cleanup_plate_string(raw_text)
        final_plate, corrected = self.correction_table.correct(cleaned)

        if not final_plate:
            return None
        if len(final_plate) < self.config.min_plate_len:
            return None
        if len(final_plate) > self.config.max_plate_len:
            return None
        for prefix in self.config.reject_prefixes:
            if prefix and final_plate.startswith(prefix):
                return None

        return OcrCandidate(
            job=job,
            raw_text=raw_text,
            cleaned_text=cleaned,
            final_plate=final_plate,
            confidence=confidence,
            corrected=corrected,
        )

    def _score_candidates(self, candidates: List[OcrCandidate]) -> None:
        counts = Counter(candidate.final_plate for candidate in candidates)
        mid = (len(candidates) - 1) / 2.0

        for index, candidate in enumerate(candidates):
            score = 0.0
            score += counts[candidate.final_plate] * 2.0
            score += candidate.confidence
            score += self._length_score(candidate.final_plate)
            score += 2.5 if candidate.corrected else 0.0
            if self.pattern_bonus and self.pattern_bonus.match(candidate.final_plate):
                score += 1.5
            if len(candidates) > 1:
                distance = abs(index - mid)
                norm = max(0.0, 1.0 - (distance / max(mid, 1.0)))
                score += 0.5 * norm
            candidate.score = score

    def _choose_best_candidate(self, candidates: List[OcrCandidate]) -> Optional[OcrCandidate]:
        if not candidates:
            return None
        self._score_candidates(candidates)
        candidates.sort(key=lambda c: (c.score, c.confidence), reverse=True)
        return candidates[0]

    def _save_notification_artifacts(self, candidate: OcrCandidate) -> None:
        day_dir = self.config.output_root / datetime.now().strftime("%Y.%m.%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / f"{candidate.job.base_name}_FRAME.jpg").write_bytes(candidate.job.full_jpg)
        (day_dir / f"{candidate.job.base_name}_FRAME_{self.config.discord_frame_x}x{self.config.discord_frame_y}.jpg").write_bytes(candidate.job.preview_jpg)
        (day_dir / f"{candidate.job.base_name}_PLATE.jpg").write_bytes(candidate.job.crop_jpg)

        all_dir = self.config.all_plates_root / datetime.now().strftime("%Y%m%d")
        all_dir.mkdir(parents=True, exist_ok=True)
        all_name = (
            f"{candidate.job.base_name}_OCR[{candidate.final_plate}]_"
            f"CONF[{candidate.confidence:.2f}]_CORR[{int(candidate.corrected)}].jpg"
        )
        (all_dir / all_name).write_bytes(candidate.job.crop_jpg)

    def _build_notification_text(self, candidate: OcrCandidate) -> Tuple[str, str]:
        display_ts = ts_to_display(candidate.job.ts)
        plain = (
            f"{display_ts} Plate[{candidate.final_plate}] detected at "
            f"{self.config.street_pretty} by CAM[{candidate.job.camera_label}]"
        )
        discord_text = (
            f"{display_ts} Plate[**`{candidate.final_plate}`**] detected at "
            f"{self.config.street_pretty} by CAM[**`{candidate.job.camera_label}`**]"
        )
        return plain, discord_text

    def _process_group(self, jobs: List[PlateJob]) -> None:
        if not jobs:
            return

        candidates: List[OcrCandidate] = []
        for job in jobs:
            try:
                candidate = self._build_candidate(job)
                if candidate is not None:
                    candidates.append(candidate)
            except Exception as exc:
                LOGGER.exception("OCR failed for %s: %s", job.base_name, exc)

        chosen = self._choose_best_candidate(candidates)
        if chosen is None:
            LOGGER.debug("No usable OCR result from %d candidate(s).", len(jobs))
            return

        allow, reason = self.filters[chosen.job.camera_index].should_notify(chosen.final_plate, chosen.job)
        plain, discord_text = self._build_notification_text(chosen)
        LOGGER.info("%s [%s]", plain, reason)
        if not allow:
            return

        self._save_notification_artifacts(chosen)
        event = NotificationEvent(
            text_plain=plain,
            text_discord=discord_text,
            image_bytes=chosen.job.preview_jpg,
            image_filename=f"{chosen.job.base_name}_FRAME_{self.config.discord_frame_x}x{self.config.discord_frame_y}.jpg",
        )
        put_latest(self.notify_queue, event)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if not self._ensure_ocr_ready():
                time.sleep(0.2)
                continue

            try:
                job = self.candidate_queue.get(timeout=0.2)
            except queue.Empty:
                self._purge_and_flush_stale_groups(force_all=False)
                continue

            self._get_or_create_group(job)
            self._purge_and_flush_stale_groups(force_all=False)

        self._purge_and_flush_stale_groups(force_all=True)


class NotificationDispatcher:
    def __init__(self, notify_queue: queue.Queue, discord_client: DiscordDMClient, stop_event: threading.Event):
        self.notify_queue = notify_queue
        self.discord_client = discord_client
        self.stop_event = stop_event
        self.thread = threading.Thread(target=self._run, name="notify", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                event = self.notify_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            LOGGER.info("Notify: %s", event.text_plain)
            self.discord_client.enqueue(event)


class CameraWorker:
    def __init__(
        self,
        config: AppConfig,
        camera_cfg: CameraConfig,
        detector: YOLODetector,
        candidate_queue: queue.Queue,
        frame_store: FrameStore,
        stop_event: threading.Event,
    ):
        self.config = config
        self.camera_cfg = camera_cfg
        self.detector = detector
        self.candidate_queue = candidate_queue
        self.frame_store = frame_store
        self.stop_event = stop_event
        self.thread = threading.Thread(target=self._run, name=f"cam-{camera_cfg.index}", daemon=True)
        self.hostname = platform.node()
        self.prev_frame: Optional[np.ndarray] = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.thread.join(timeout=3.0)

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.camera_cfg.device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera device {self.camera_cfg.device_path}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_cfg.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _expand_bbox(self, x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = int(width * self.config.bbox_pad_ratio)
        pad_y = int(height * self.config.bbox_pad_ratio)
        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(frame_w - 1, x2 + pad_x)
        y2p = min(frame_h - 1, y2 + pad_y)
        return x1p, y1p, x2p, y2p

    def _touches_edge(self, x1: int, x2: int, frame_w: int) -> bool:
        margin = self.config.edge_margin_px
        return x1 <= margin or x2 >= (frame_w - 1 - margin)

    def _is_moving(self, prev_frame: np.ndarray, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        prev_crop = prev_frame[y1:y2, x1:x2]
        curr_crop = frame[y1:y2, x1:x2]
        if prev_crop.size == 0 or curr_crop.size == 0:
            return False
        if prev_crop.shape != curr_crop.shape:
            return True

        prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = float(diff.mean())
        changed_ratio = float(np.count_nonzero(diff > 25)) / float(diff.size)
        return (
            mean_diff >= self.config.motion_diff_threshold
            or changed_ratio >= self.config.motion_changed_ratio_threshold
        )

    def _build_job(
        self,
        frame: np.ndarray,
        full_jpg: bytes,
        preview_jpg: bytes,
        ts: str,
        x1_orig: int,
        y1_orig: int,
        x2_orig: int,
        y2_orig: int,
        conf: float,
    ) -> PlateJob:
        frame_h, frame_w = frame.shape[:2]
        x1p, y1p, x2p, y2p = self._expand_bbox(x1_orig, y1_orig, x2_orig, y2_orig, frame_w, frame_h)
        crop = frame[y1p:y2p, x1p:x2p].copy()
        crop_jpg = encode_jpg(crop, self.config.frame_jpeg_quality)
        camera_label = self.camera_cfg.name or f"{self.hostname}-{self.camera_cfg.index}"
        base_name = (
            f"TS[{ts}]_ST[{self.config.street_raw}]_CITY[{self.config.city_raw}]_"
            f"CAM[{camera_label}]_X1X2[{x1_orig}_{x2_orig}_{self.camera_cfg.index}]"
        )
        return PlateJob(
            camera_index=self.camera_cfg.index,
            camera_label=camera_label,
            ts=ts,
            base_name=base_name,
            x1_orig=x1_orig,
            y1_orig=y1_orig,
            x2_orig=x2_orig,
            y2_orig=y2_orig,
            crop_jpg=crop_jpg,
            preview_jpg=preview_jpg,
            full_jpg=full_jpg,
            conf=conf,
        )

    def _extract_jobs(self, frame: np.ndarray) -> List[PlateJob]:
        frame_h, frame_w = frame.shape[:2]
        preview = cv2.resize(frame, (self.config.discord_frame_x, self.config.discord_frame_y))
        preview_jpg = encode_jpg(preview, self.config.frame_jpeg_quality)
        full_jpg = encode_jpg(frame, self.config.frame_jpeg_quality)
        ts = now_ts()

        frame_small = cv2.resize(frame, (self.config.lpr_size, self.config.lpr_size))
        results = self.detector.predict(frame_small, conf=self.camera_cfg.detect_conf, imgsz=self.config.lpr_size)
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().tolist()
        scale_x = frame_w / float(self.config.lpr_size)
        scale_y = frame_h / float(self.config.lpr_size)

        jobs: List[PlateJob] = []
        for idx in np.argsort(np.array(confs))[::-1]:
            x1s, y1s, x2s, y2s = xyxy[idx]
            x1 = max(0, int(x1s * scale_x))
            y1 = max(0, int(y1s * scale_y))
            x2 = min(frame_w - 1, int(x2s * scale_x))
            y2 = min(frame_h - 1, int(y2s * scale_y))
            if x2 <= x1 or y2 <= y1:
                continue

            x1p, y1p, x2p, y2p = self._expand_bbox(x1, y1, x2, y2, frame_w, frame_h)
            if self._touches_edge(x1p, x2p, frame_w):
                LOGGER.debug("Ignoring edge-cut plate on cam %d at [%d,%d]", self.camera_cfg.index, x1p, x2p)
                continue

            if self.prev_frame is not None and not self._is_moving(self.prev_frame, frame, (x1p, y1p, x2p, y2p)):
                continue

            jobs.append(self._build_job(frame, full_jpg, preview_jpg, ts, x1, y1, x2, y2, confs[idx]))

        return jobs

    def _run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        while not self.stop_event.is_set():
            if cap is None:
                try:
                    cap = self._open_capture()
                    LOGGER.info(
                        "Camera %d opened on %s at %dx%d %dfps",
                        self.camera_cfg.index,
                        self.camera_cfg.device_path,
                        self.camera_cfg.width,
                        self.camera_cfg.height,
                        self.camera_cfg.fps,
                    )
                except Exception as exc:
                    LOGGER.error("Camera %d open failed: %s", self.camera_cfg.index, exc)
                    time.sleep(2.0)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                LOGGER.warning("Camera %d read failed; reopening.", self.camera_cfg.index)
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(1.0)
                continue

            # Update the shared latest-frame buffer for live video.
            # Live streaming reads from this buffer and does not re-capture.
            self.frame_store.update(self.camera_cfg.index, frame)
            try:
                jobs = self._extract_jobs(frame)
                for job in jobs:
                    put_latest(self.candidate_queue, job)
            except Exception as exc:
                LOGGER.exception("Camera %d processing error: %s", self.camera_cfg.index, exc)

            self.prev_frame = frame.copy()

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def read_config_lines(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                value = value[1:-1]
            elif len(value) >= 2 and value[0] == "'" and value[-1] == "'":
                value = value[1:-1]
            values[key] = value
    return values


def as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    return default


def as_int(values: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(values.get(key, default))
    except Exception:
        return default


def as_positive_int(values: Dict[str, str], key: str, default: int) -> int:
    value = as_int(values, key, default)
    return value if value > 0 else default


def as_float(values: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(values.get(key, default))
    except Exception:
        return default


def parse_cameras(values: Dict[str, str], defaults: Dict[str, object]) -> List[CameraConfig]:
    indices = sorted(
        {
            int(match.group(1))
            for key in values
            for match in [re.fullmatch(r"(?:CAM|DEV)(\d+)", key)]
            if match
        }
    )

    cameras: List[CameraConfig] = []
    for index in indices:
        purpose = values.get(f"CAM{index}", "LPR").strip().upper() or "LPR"
        if purpose != "LPR":
            LOGGER.warning("Skipping camera %d because CAM%d=%s and only LPR is supported.", index, index, purpose)
            continue
        device_path = values.get(f"DEV{index}", "").strip()
        if not device_path:
            LOGGER.warning("Skipping camera %d because DEV%d is missing.", index, index)
            continue
        if not device_path.startswith("/dev/video"):
            LOGGER.warning("Skipping camera %d because DEV%d=%s is not a USB video device.", index, index, device_path)
            continue
        width = as_int(values, f"WIDTH{index}", int(defaults["frame_width"]))
        height = as_int(values, f"HEIGHT{index}", int(defaults["frame_height"]))
        fps = as_int(values, f"FPS{index}", int(defaults["frame_rate"]))
        detect_conf = as_float(values, f"CONF{index}", float(defaults["detect_conf"]))
        name = values.get(f"NAME{index}", "").strip()
        cameras.append(
            CameraConfig(
                index=index,
                purpose="LPR",
                device_path=device_path,
                width=width,
                height=height,
                fps=fps,
                detect_conf=detect_conf,
                name=name,
            )
        )
    return cameras


def load_config(path: str) -> AppConfig:
    values = read_config_lines(path)

    defaults: Dict[str, object] = {
        "frame_width": 1920,
        "frame_height": 1080,
        "frame_rate": 5,
        "detect_conf": 0.75,
    }

    street_raw = values.get("STREET", "Unknown-Street")
    city_raw = values.get("CITY", "Unknown-City")
    street_pretty = street_raw.replace("-", " ")
    city_pretty = city_raw.replace("-", " ")

    discord_user_ids = [item.strip() for item in values.get("DISCORD_USER_ID", "").split(",") if item.strip()]
    skip_lpr = {item.strip().upper() for item in values.get("SKIP_LPR", "").split(",") if item.strip()}
    cameras = parse_cameras(values, defaults)

    if not cameras:
        raise RuntimeError("No usable cameras found in config.txt. Add CAMx=\"LPR\" and DEVx=\"/dev/videoX\" entries.")

    return AppConfig(
        config_path=path,
        discord_bot_token=values.get("DISCORD_BOT_TOKEN", ""),
        discord_user_ids=discord_user_ids,
        street_raw=street_raw,
        street_pretty=street_pretty,
        city_raw=city_raw,
        city_pretty=city_pretty,
        skip_lpr=skip_lpr,
        cameras=cameras,
        model_path=values.get("MODEL_PATH", "best_lpr_800x800.pt"),
        lpr_size=as_int(values, "LPR_SIZE", 800),
        bbox_pad_ratio=as_float(values, "BBOX_PAD_RATIO", 0.15),
        edge_margin_px=as_int(values, "EDGE_MARGIN_PX", 0),
        frame_jpeg_quality=as_int(values, "FRAME_JPEG_QUALITY", 90),
        discord_frame_x=as_positive_int(values, "DISCORD_FRAME_X", 1024),
        discord_frame_y=as_positive_int(values, "DISCORD_FRAME_Y", 576),
        live_jpeg_quality=as_int(values, "LIVE_JPEG_QUALITY", 80),
        live_width=as_int(values, "LIVE_WIDTH", 640),
        live_height=as_int(values, "LIVE_HEIGHT", 480),
        socket_path=values.get("SOCKET_PATH", "/tmp/camera.sock"),
        output_root=Path(values.get("OUTPUT_DIR", "./video_frames")),
        all_plates_root=Path(values.get("ALL_PLATES_DIR", "./all-plates")),
        queue_maxsize=as_int(values, "QUEUE_MAXSIZE", 256),
        group_window_sec=as_float(values, "GROUP_WINDOW_SEC", 1.0),
        group_idle_sec=as_float(values, "GROUP_IDLE_SEC", 0.60),
        recent_window_sec=as_float(values, "RECENT_WINDOW_SEC", 10.0),
        plate_history_size=as_int(values, "PLATE_HISTORY_SIZE", 10),
        parked_history_match_count=as_int(values, "PARKED_HISTORY_MATCH_COUNT", 3),
        stationary_ratio_threshold=as_float(values, "STATIONARY_RATIO_THRESHOLD", 0.04),
        recent_approx_len=as_int(values, "RECENT_APPROX_LEN", 3),
        history_approx_len=as_int(values, "HISTORY_APPROX_LEN", 4),
        motion_diff_threshold=as_float(values, "MOTION_DIFF_THRESHOLD", 12.0),
        motion_changed_ratio_threshold=as_float(values, "MOTION_CHANGED_RATIO_THRESHOLD", 0.015),
        min_plate_len=as_int(values, "MIN_PLATE_LEN", 4),
        max_plate_len=as_int(values, "MAX_PLATE_LEN", 8),
        reject_prefixes=[item.strip().upper() for item in values.get("REJECT_PREFIXES", "125").split(",") if item.strip()],
        pattern_bonus_regex=values.get("PLATE_PATTERN_BONUS_REGEX", r"^[A-Z]{3}\d{4}$"),
        correction_table_path=values.get("LPR_TABLE", "lpr-table.txt"),
        enable_live_video=as_bool(values.get("ENABLE_LIVE_VIDEO", "1"), True),
        enable_discord=as_bool(values.get("ENABLE_DISCORD", "1"), True),
        ocr_config=values.get("OCR_CONFIG", "configs/us_multi_v4_20251003_ny7_norm.yaml"),
        ocr_ckpt=values.get("OCR_CKPT", "runs/T22.8K_V2.5K_125.ckpt"),
        ocr_device=values.get("OCR_DEVICE", "cuda"),
        ocr_amp=as_bool(values.get("OCR_AMP", "1"), True),
        ocr_topk_templates=as_int(values, "OCR_TOPK_TEMPLATES", 3),
        ocr_conf_gate=as_float(values, "OCR_CONF_GATE", 0.985),
        ocr_char_topk=as_int(values, "OCR_CHAR_TOPK", 3),
        ocr_beam=as_int(values, "OCR_BEAM", 3),
        ocr_gap_delta=as_float(values, "OCR_GAP_DELTA", 0.22),
        ocr_blur_gate=as_float(values, "OCR_BLUR_GATE", 80.0),
    )


class JetsonLprAllApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.stop_event = threading.Event()
        self.detector = YOLODetector(config.model_path)
        self.candidate_queue: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)
        self.notify_queue: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)
        self.frame_store = FrameStore(max(camera.index for camera in config.cameras) + 1)
        self.correction_table = CorrectionTable(config.correction_table_path)
        self.filters = {
            camera.index: NotificationFilter(config, camera)
            for camera in config.cameras
        }
        self.discord_client = DiscordDMClient(
            config.discord_bot_token,
            config.discord_user_ids,
            config.enable_discord,
        )
        self.notification_dispatcher = NotificationDispatcher(
            self.notify_queue,
            self.discord_client,
            self.stop_event,
        )
        self.ocr_worker = OCRWorker(
            config,
            self.candidate_queue,
            self.notify_queue,
            self.correction_table,
            self.filters,
            self.stop_event,
        )
        self.camera_workers = [
            CameraWorker(config, camera, self.detector, self.candidate_queue, self.frame_store, self.stop_event)
            for camera in config.cameras
        ]
        self.live_video_server = CameraLiveVideoServer(
            config.socket_path if config.enable_live_video else "",
            self.frame_store,
            config,
            self.stop_event,
        )

    def start(self) -> None:
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self.config.all_plates_root.mkdir(parents=True, exist_ok=True)
        self.live_video_server.start()
        for worker in self.camera_workers:
            worker.start()
        self.discord_client.start()
        self.notification_dispatcher.start()
        self.ocr_worker.start()
        LOGGER.info("jetsonlpr-all started with %d camera(s).", len(self.camera_workers))

    def stop(self) -> None:
        self.stop_event.set()
        for worker in self.camera_workers:
            worker.stop()
        self.ocr_worker.stop()
        self.notification_dispatcher.stop()
        self.live_video_server.stop()
        self.discord_client.stop()
        LOGGER.info("jetsonlpr-all stopped.")


def install_signal_handlers(app: JetsonLprAllApp) -> None:
    def _handler(signum, _frame):
        LOGGER.info("Received signal %s; shutting down.", signum)
        app.stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> int:
    setup_logging()
    config_path = os.environ.get("JETSONLPR_CONFIG", "config.txt")
    config = load_config(config_path)

    LOGGER.info("Loaded config from %s", config.config_path)
    LOGGER.info("Street=%s City=%s Cameras=%s", config.street_pretty, config.city_pretty, [camera.device_path for camera in config.cameras])
    LOGGER.info("Using LPR model: %s", config.model_path)
    LOGGER.info("Using OCR config=%s ckpt=%s", config.ocr_config, config.ocr_ckpt)

    app = JetsonLprAllApp(config)
    install_signal_handlers(app)
    app.start()

    try:
        while not app.stop_event.is_set():
            time.sleep(0.5)
    finally:
        app.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
