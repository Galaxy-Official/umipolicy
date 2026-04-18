import socket
import threading
import time

import cv2
from flask import Flask, Response, jsonify
from werkzeug.serving import make_server


def camera_name_from_observation_key(camera_key: str) -> str:
    prefix = "observation.images."
    if not camera_key.startswith(prefix):
        raise ValueError(
            f"Quest stream camera key must start with '{prefix}' (got {camera_key!r})."
        )

    camera_name = camera_key[len(prefix) :]
    if not camera_name:
        raise ValueError("Quest stream camera key must include a camera name.")

    return camera_name


def detect_advertise_host(bind_host: str, advertise_host: str) -> str:
    if advertise_host and advertise_host != "auto":
        return advertise_host

    if bind_host not in {"0.0.0.0", "::", ""}:
        return bind_host

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            detected_host = sock.getsockname()[0]
            if detected_host and not detected_host.startswith("127."):
                return detected_host
    except OSError:
        pass

    try:
        detected_host = socket.gethostbyname(socket.gethostname())
        if detected_host:
            return detected_host
    except OSError:
        pass

    return "localhost"


class QuestCameraStreamer:
    def __init__(
        self,
        camera_key: str,
        bind_host: str = "0.0.0.0",
        port: int = 8090,
        advertise_host: str = "auto",
        fps: int = 15,
        jpeg_quality: int = 80,
    ):
        if fps <= 0:
            raise ValueError(f"Quest stream fps must be positive (got {fps}).")

        if not 1 <= jpeg_quality <= 100:
            raise ValueError(
                f"Quest stream JPEG quality must be in [1, 100] (got {jpeg_quality})."
            )

        self.camera_key = camera_key
        self.camera_name = camera_name_from_observation_key(camera_key)
        self.bind_host = bind_host
        self.port = port
        self.advertise_host = detect_advertise_host(bind_host, advertise_host)
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        self._condition = threading.Condition()
        self._latest_jpeg = None
        self._latest_frame_id = 0
        self._last_publish_t = 0.0
        self._is_stopped = False
        self._server = None
        self._server_thread = None

        self.app = Flask(__name__)
        self._register_routes()

    @property
    def stream_url(self) -> str:
        return f"http://{self.advertise_host}:{self.port}"

    def _register_routes(self):
        @self.app.route("/")
        def homepage():
            return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quest Side Camera</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #050505;
      --fg: #f5f5f5;
      --muted: #8f8f8f;
    }}
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      background: radial-gradient(circle at top, #1b1b1b, var(--bg) 55%);
      color: var(--fg);
      font-family: Helvetica, Arial, sans-serif;
    }}
    body {{
      display: grid;
      place-items: center;
    }}
    .frame {{
      width: 100vw;
      height: 100vh;
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    img {{
      max-width: 100vw;
      max-height: 100vh;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    .status {{
      position: fixed;
      left: 24px;
      bottom: 24px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.6);
      color: var(--muted);
      font-size: 18px;
      letter-spacing: 0.02em;
    }}
  </style>
</head>
<body>
  <div class="frame">
    <img id="stream" src="/stream.mjpg" alt="Quest side camera stream">
  </div>
  <div class="status" id="status">Waiting for {self.camera_name} camera frames...</div>
  <script>
    const img = document.getElementById("stream");
    const status = document.getElementById("status");
    img.addEventListener("load", () => {{
      status.textContent = "Live";
    }});
    img.addEventListener("error", () => {{
      status.textContent = "Reconnecting...";
      window.setTimeout(() => {{
        img.src = "/stream.mjpg?ts=" + Date.now();
      }}, 1000);
    }});
  </script>
</body>
</html>"""

        @self.app.route("/healthz")
        def healthz():
            return jsonify(
                {
                    "status": "ok",
                    "camera_key": self.camera_key,
                    "has_frame": self._latest_jpeg is not None,
                    "frame_id": self._latest_frame_id,
                    "stream_url": self.stream_url,
                }
            )

        @self.app.route("/stream.mjpg")
        def mjpeg_stream():
            return Response(
                self._generate_stream(),
                content_type="multipart/x-mixed-replace; boundary=frame",
            )

    def start(self):
        if self._server is not None:
            return

        self._server = make_server(self.bind_host, self.port, self.app, threaded=True)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

    def stop(self):
        with self._condition:
            self._is_stopped = True
            self._condition.notify_all()

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

        if self._server_thread is not None:
            self._server_thread.join(timeout=2)
            self._server_thread = None

    def publish_frame(self, frame) -> bool:
        now = time.monotonic()
        if now - self._last_publish_t < 1.0 / self.fps:
            return False

        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()

        if frame is None:
            return False

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Quest stream only supports HxWx3 color images (got shape {frame.shape})."
            )

        rgb_frame = frame.astype("uint8", copy=False)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return False

        with self._condition:
            self._latest_jpeg = encoded.tobytes()
            self._latest_frame_id += 1
            self._last_publish_t = now
            self._condition.notify_all()

        return True

    def _generate_stream(self):
        last_frame_id = 0
        while True:
            with self._condition:
                while (
                    not self._is_stopped
                    and (
                        self._latest_jpeg is None
                        or self._latest_frame_id == last_frame_id
                    )
                ):
                    self._condition.wait(timeout=1.0)

                if self._is_stopped:
                    break

                jpeg_bytes = self._latest_jpeg
                last_frame_id = self._latest_frame_id

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode()
                + jpeg_bytes
                + b"\r\n"
            )
