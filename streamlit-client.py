#!/usr/bin/env python3
import io
import inspect
from pathlib import Path
import os
import platform
import re
import socket
import struct
import time

from PIL import Image, ImageOps
import streamlit as st

CONFIG_PATH = os.environ.get("JETSONLPR_CONFIG", "config.txt")
DEFAULT_SOCKET_PATH = "/tmp/camera.sock"
USB_PREFIX = "/dev/video"
REFRESH_INTERVAL_SEC = 1.0
IMAGE_ROOT = Path(os.environ.get("EASYLPR_IMAGE_DIR", str(Path(__file__).resolve().parent / "video_frames"))).expanduser()
IMAGE_FILENAME_SUFFIX = "_FRAME_1024x576.jpg"



def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def read_config_lines(path: str) -> dict:
    values = {}
    if not path or not os.path.exists(path):
        return values

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


def get_socket_path(config_path: str) -> str:
    values = read_config_lines(config_path)
    return values.get("SOCKET_PATH", DEFAULT_SOCKET_PATH) or DEFAULT_SOCKET_PATH


def discover_usb_cameras(config_path: str):
    values = read_config_lines(config_path)
    hostname = platform.node()

    indices = sorted(
        {
            int(match.group(1))
            for key in values
            for match in [re.fullmatch(r"(?:CAM|DEV)(\d+)", key)]
            if match
        }
    )

    cameras = []
    for index in indices:
        purpose = values.get(f"CAM{index}", "LPR").strip().upper() or "LPR"
        device_path = values.get(f"DEV{index}", "").strip()
        if purpose != "LPR":
            continue
        if not device_path.startswith(USB_PREFIX):
            continue
        label = values.get(f"NAME{index}", "").strip() or f"{hostname}-{index}"
        cameras.append({"index": index, "label": label, "device_path": device_path})
    return cameras


def send_command(sock: socket.socket, cmd_str: str) -> None:
    cmd_bytes = cmd_str.encode("utf-8")
    sock.sendall(struct.pack("!I", len(cmd_bytes)) + cmd_bytes)


def read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("socket closed while reading data")
        data.extend(chunk)
    return bytes(data)


def read_frames(sock: socket.socket, count: int):
    frames = []
    for _ in range(count):
        length_data = read_exact(sock, 4)
        msg_len = struct.unpack("!I", length_data)[0]
        if msg_len <= 0:
            frames.append(None)
            continue

        img_bytes = read_exact(sock, msg_len)
        pil_im = Image.open(io.BytesIO(img_bytes))
        pil_im.load()
        frames.append(pil_im)
    return frames


def close_socket() -> None:
    sock = st.session_state.pop("sock", None)
    if sock is not None:
        socket_path = st.session_state.get("connected_socket_path") or DEFAULT_SOCKET_PATH
        try:
            sock.close()
        except Exception:
            pass
        print(f"[Streamlit] Disconnected from camera server at {socket_path}", flush=True)


def reset_connection_state() -> None:
    close_socket()
    st.session_state["streaming"] = False
    st.session_state["connected_socket_path"] = None


def ensure_connection(socket_path: str) -> None:
    current_path = st.session_state.get("connected_socket_path")
    if current_path != socket_path:
        reset_connection_state()

    if st.session_state.get("sock") is not None:
        return

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(socket_path)
    sock.settimeout(None)
    st.session_state["sock"] = sock
    st.session_state["connected_socket_path"] = socket_path
    print(f"[Streamlit] Connected to camera server at {socket_path}", flush=True)
    if st.session_state.get("streaming"):
        send_command(sock, "START_LIVE_VIDEO")


def render_frames(cameras, frames) -> None:
    if not cameras:
        st.warning("No USB LPR cameras were discovered in config.txt.")
        return

    if len(cameras) == 1:
        columns = st.columns(1)
    elif len(cameras) <= 4:
        columns = st.columns(2)
    else:
        columns = st.columns(3)

    for i, camera in enumerate(cameras):
        frame = frames[i] if i < len(frames) else None
        with columns[i % len(columns)]:
            st.markdown(
                f"<p style='font-size:20px; color:red;'>Camera {camera['label']}</p>",
                unsafe_allow_html=True,
            )
            if frame is None:
                st.info("No frame yet from this camera.")
            else:
                st.image(frame, use_column_width=True)



def image_dates():
    if not IMAGE_ROOT.is_dir():
        return []
    return sorted((p.name for p in IMAGE_ROOT.iterdir()
                   if p.is_dir() and re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", p.name)), reverse=True)


def image_files(date_name):
    folder = IMAGE_ROOT / date_name
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir()
                   if p.is_file() and p.name.endswith(IMAGE_FILENAME_SUFFIX)),
                  key=lambda p: p.name, reverse=True)


def image_label(path):
    """Show timestamp, street, and city from EasyLPR image filenames.

    Example:
      TS[20261025_103846_384789]_ST[123 John Doe Ave]_CITY[Jane-NY-12345]_...
      -> 10:38:46 | 123 John Doe Ave | Jane NY 12345

    Falls back to earlier filename formats if bracketed fields are absent.
    """
    stem = path.stem

    # Preferred EasyLPR format: named bracketed fields.
    fields = {name.upper(): value for name, value in
              re.findall(r"([A-Za-z][A-Za-z0-9]*)\[([^\]]*)\]", stem)}

    ts = fields.get("TS", "")
    match = re.fullmatch(r"(?P<date>\d{8})_(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})(?:_\d+)?", ts)
    if not match:
        # Support older filenames containing a date and a time.
        match = re.search(
            r"20\d{2}[-.]?\d{2}[-.]?\d{2}[_ T-]+(?P<h>\d{2})[:._-]?(?P<m>\d{2})[:._-]?(?P<s>\d{2})",
            stem,
        )
    if not match:
        match = re.search(
            r"(?<!\d)(?P<h>[01]\d|2[0-3])[:._-]?(?P<m>[0-5]\d)[:._-]?(?P<s>[0-5]\d)(?!\d)",
            stem,
        )

    clock = "Time unknown"
    if match:
        h, m, sec = (int(match.group(k)) for k in ("h", "m", "s"))
        if h < 24 and m < 60 and sec < 60:
            clock = f"{h:02d}:{m:02d}:{sec:02d}"

    street = fields.get("ST", "")
    city = fields.get("CITY", "")

    # Preserve backward compatibility with the previous filename parser.
    if not street or not city:
        if "__" in stem and len(stem.split("__")) >= 3:
            parts = stem.split("__")
            street = street or parts[-2]
            city = city or parts[-1]
        else:
            sm = re.search(r"(?:^|[_-])street[=_-](.+?)(?=[_-]city[=_-]|$)", stem, re.I)
            cm = re.search(r"(?:^|[_-])city[=_-](.+)$", stem, re.I)
            if sm and not street:
                street = sm.group(1)
            if cm and not city:
                city = cm.group(1)

    def display_field(value):
        return re.sub(r"\s+", " ", value.replace("-", " ").replace("_", " ")).strip() or "—"

    return f"{clock} | {display_field(street)} | {display_field(city)}"


def remember_history_date() -> None:
    """Keep the date selection independent of Streamlit widget lifecycle."""
    selected_date = st.session_state.get("_history_date_widget")
    if selected_date != st.session_state.get("history_saved_date"):
        st.session_state["history_saved_date"] = selected_date
        st.session_state["history_saved_image"] = None


def remember_history_image() -> None:
    """Keep the image selection when switching away from Image List."""
    selected_image = st.session_state.get("_history_image_widget")
    st.session_state["history_saved_image"] = (
        str(selected_image) if selected_image is not None else None
    )


def refresh_history() -> None:
    """Reload folder contents and select the newest date/image on request."""
    st.session_state.pop("history_dates_snapshot", None)
    st.session_state.pop("history_files_snapshot", None)
    st.session_state["history_saved_date"] = None
    st.session_state["history_saved_image"] = None
    st.session_state.pop("_history_date_widget", None)
    st.session_state.pop("_history_image_widget", None)


def render_image_list():
    # Keep a snapshot until Refresh is clicked: new captures don't unexpectedly
    # replace a user's current selection while they browse history.
    if "history_dates_snapshot" not in st.session_state:
        st.session_state["history_dates_snapshot"] = image_dates()
    if "history_files_snapshot" not in st.session_state:
        st.session_state["history_files_snapshot"] = {}

    left, right = st.columns([1, 2.5], gap="large")
    with left:
        st.button("Refresh Image List", on_click=refresh_history)
        st.subheader("Dates")
        dates = st.session_state["history_dates_snapshot"]
        if not dates:
            st.info(f"No date folders found under {IMAGE_ROOT}")
            return

        saved_date = st.session_state.get("history_saved_date")
        if saved_date not in dates:
            saved_date = dates[0]
            st.session_state["history_saved_date"] = saved_date
            st.session_state["history_saved_image"] = None

        # Widget state may be discarded when the user switches tabs. Rebuild it
        # from separate, durable session-state values on each return.
        st.session_state["_history_date_widget"] = saved_date
        with st.container(height=190, border=True):
            date = st.radio(
                "Dates", dates, key="_history_date_widget",
                label_visibility="collapsed", on_change=remember_history_date,
            )

        st.subheader("Vehicle Images")
        snapshots = st.session_state["history_files_snapshot"]
        if date not in snapshots:
            snapshots[date] = image_files(date)
        files = snapshots[date]
        if not files:
            st.info("No vehicle images for this date.")
            return

        saved_image = st.session_state.get("history_saved_image")
        file_by_path = {str(path): path for path in files}
        if saved_image not in file_by_path:
            saved_image = str(files[0])
            st.session_state["history_saved_image"] = saved_image

        st.session_state["_history_image_widget"] = file_by_path[saved_image]
        with st.container(height=390, border=True):
            selected = st.radio(
                "Vehicle Images", files, format_func=image_label,
                label_visibility="collapsed", key="_history_image_widget",
                on_change=remember_history_image,
            )
        st.caption(f"{len(files)} image(s)")

    with right:
        st.subheader("Image Preview")
        try:
            with Image.open(selected) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                filt = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                preview = ImageOps.pad(rgb, (1024, 576), method=filt, color=(24, 24, 24))
            st.image(preview, width=1024)
            st.caption(f"{date} | {image_label(selected)}")
        except (OSError, ValueError) as exc:
            st.error(f"Unable to load image: {exc}")


def select_tab():
    # Modern Streamlit supports stateful tabs; older versions use a tab-like
    # horizontal selector so the live stream won't force reruns in Image List.
    if "on_change" in inspect.signature(st.tabs).parameters:
        live, history = st.tabs(["Live Video", "Image List"], key="active_tab", on_change="rerun")
        return st.session_state.get("active_tab", "Live Video"), live, history
    view = st.radio("View", ["Live Video", "Image List"], horizontal=True,
                    label_visibility="collapsed", key="active_view")
    return view, st.container(), st.container()


def main() -> None:
    try:
        st.set_page_config(page_title="EasyLPR Live Video", layout="wide")
    except Exception:
        pass

    st.markdown(
        "<h1 style='text-align: left; color: green;'><b>EasyLPR Live Video</b></h1>",
        unsafe_allow_html=True,
    )

    if "streaming" not in st.session_state:
        st.session_state["streaming"] = False
    if "connected_socket_path" not in st.session_state:
        st.session_state["connected_socket_path"] = None

    cameras = discover_usb_cameras(CONFIG_PATH)
    socket_path = get_socket_path(CONFIG_PATH)

    st.sidebar.header("Connection")
    st.sidebar.text(f"Config: {CONFIG_PATH}")
    st.sidebar.text(f"Socket: {socket_path}")
    st.sidebar.text(f"USB LPR cameras: {len(cameras)}")
    for camera in cameras:
        st.sidebar.markdown(f"**{camera['label']}**")
        st.sidebar.caption(camera.get("device_path", ""))

    selected_tab, tab_live, tab_images = select_tab()
    if selected_tab != "Live Video" and st.session_state.get("streaming"):
        reset_connection_state()

    if selected_tab == "Live Video":
        try:
            ensure_connection(socket_path)
        except Exception as exc:
            reset_connection_state()
            with tab_live:
                st.error(f"Failed to connect to camera server at {socket_path}: {exc}")
            return
        sock = st.session_state["sock"]
        with tab_live:
            st.success(f"Connected to camera server at {socket_path} for {len(cameras)} camera(s)!")
            start_disabled = st.session_state["streaming"]
            stop_disabled = not st.session_state["streaming"]

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Start", disabled=start_disabled):
                    try:
                        send_command(sock, "START_LIVE_VIDEO")
                        st.session_state["streaming"] = True
                    except Exception as exc:
                        reset_connection_state()
                        st.error(f"Failed to start live video: {exc}")
                    rerun()

            with col2:
                if st.button("Stop", disabled=stop_disabled):
                    try:
                        send_command(sock, "STOP_LIVE_VIDEO")
                    except Exception:
                        pass
                    st.session_state["streaming"] = False
                    rerun()

            with col3:
                if st.button("Reconnect"):
                    reset_connection_state()
                    rerun()

            if st.session_state["streaming"]:
                try:
                    send_command(sock, "SEND_MORE")
                    frames = read_frames(sock, len(cameras))
                except Exception as exc:
                    reset_connection_state()
                    st.warning(f"Live video connection was lost: {exc}")
                    rerun()
                    return

                if frames:
                    render_frames(cameras, frames)
                    time.sleep(REFRESH_INTERVAL_SEC)
                    rerun()
                else:
                    st.warning("No frames received or streaming ended.")
                    st.session_state["streaming"] = False
                    rerun()
            else:
                render_frames(cameras, [None] * len(cameras))
    else:
        with tab_images:
            render_image_list()

if __name__ == "__main__":
    main()
