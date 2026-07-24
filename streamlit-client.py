#!/usr/bin/env python3
import io
import os
import platform
import re
import socket
import struct
import time

from PIL import Image
import streamlit as st

CONFIG_PATH = os.environ.get("JETSONLPR_CONFIG", "config.txt")
DEFAULT_SOCKET_PATH = "/tmp/camera.sock"
USB_PREFIX = "/dev/video"
REFRESH_INTERVAL_SEC = 1.0


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

    try:
        ensure_connection(socket_path)
    except Exception as exc:
        reset_connection_state()
        st.error(f"Failed to connect to camera server at {socket_path}: {exc}")
        return

    sock = st.session_state["sock"]

    st.sidebar.header("Connection")
    st.sidebar.text(f"Config: {CONFIG_PATH}")
    st.sidebar.text(f"Socket: {socket_path}")
    st.sidebar.text(f"USB LPR cameras: {len(cameras)}")
    for camera in cameras:
        st.sidebar.markdown(f"**{camera['label']}**")
        st.sidebar.caption(camera.get("device_path", ""))

    st.success(f"Connected to camera server at {socket_path} for {len(cameras)} camera(s)!")

    tab_live, = st.tabs(["Live Video"])
    with tab_live:
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


if __name__ == "__main__":
    main()
