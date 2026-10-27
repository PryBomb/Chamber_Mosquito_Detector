"""
status_server.py - Lightweight LAN Status Server for MosquitoSentinel
======================================================================
Provides a read-only GET /api/status endpoint using Python's standard library.
Subclasses BaseHTTPRequestHandler (never SimpleHTTPRequestHandler) to prevent
accidental filesystem exposure.

Requirements & Invariants:
- Uses http.server from the Python standard library (zero third-party dependencies).
- Requires a secret token (X-Sentinel-Token) compared via hmac.compare_digest.
- If no token is configured, refuses to start.
- Silences standard request access logs.
- All handlers wrapped in try/except; client errors never affect the detector.
- Queries SQLite in read-only mode (file:...db?mode=ro) with cached row count.
"""

import os
import sys
import time
import json
import hmac
import sqlite3
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from socketserver import ThreadingMixIn
from typing import Optional, Tuple, Dict, Any


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server allowing concurrent non-blocking requests."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, detector_ref=None, token: str = ""):
        super().__init__(server_address, RequestHandlerClass)
        self.detector = detector_ref
        self.token = token
        self.is_serving = False
        self._cached_registry_count: Optional[int] = None
        self._last_registry_check_time: float = 0.0
        self._registry_lock = threading.Lock()

    def get_cached_registry_count(self, db_path: str, ttl_sec: float = 5.0) -> Optional[int]:
        """
        Safely reads row count from SQLite in read-only mode.
        If the file is locked, missing, or unreadable, returns the last cached value or None.
        Never writes and never blocks for more than 0.5s.
        """
        now = time.time()
        with self._registry_lock:
            if (now - self._last_registry_check_time < ttl_sec) and (self._cached_registry_count is not None):
                return self._cached_registry_count

            if not os.path.isfile(db_path):
                return self._cached_registry_count

            try:
                uri = f"file:{os.path.abspath(db_path)}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, timeout=0.5)
                cur = conn.cursor()
                cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='mosquito_registry'")
                if cur.fetchone()[0] == 0:
                    conn.close()
                    return self._cached_registry_count

                cur.execute("SELECT count(*) FROM mosquito_registry")
                row = cur.fetchone()
                conn.close()
                if row:
                    self._cached_registry_count = int(row[0])
                    self._last_registry_check_time = now
            except Exception:
                # Return stale cache on lock / error; do not fail the request
                pass

            return self._cached_registry_count


def _parse_preview_dimensions(params: Dict[str, list]) -> Tuple[int, int]:
    """Parses and clamps preview dimensions to 160..1280 (w) and 90..720 (h)."""
    try:
        w = int(params.get("width", params.get("w", [640]))[0])
        h = int(params.get("height", params.get("h", [360]))[0])
        w = max(160, min(1280, w))
        h = max(90, min(720, h))
    except (ValueError, IndexError, TypeError):
        w, h = 640, 360
    return w, h


class SentinelStatusHandler(BaseHTTPRequestHandler):
    """
    Handles read-only requests. Subclasses BaseHTTPRequestHandler to guarantee
    that no filesystem files can be served.
    """

    server: ThreadedHTTPServer
    _parse_preview_dimensions = staticmethod(_parse_preview_dimensions)

    def log_message(self, format, *args):
        """Silence per-request access logging to avoid terminal clutter."""
        pass

    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        try:
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "X-Sentinel-Token, Content-Type")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _check_auth(self) -> bool:
        """Compares X-Sentinel-Token using constant-time hmac.compare_digest."""
        token_hdr = self.headers.get("X-Sentinel-Token", "")
        if not token_hdr or not self.server.token:
            return False
        return hmac.compare_digest(token_hdr.strip(), self.server.token.strip())

    def do_OPTIONS(self):
        """CORS preflight handling."""
        try:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "X-Sentinel-Token, Content-Type")
            self.end_headers()
        except Exception:
            pass

    def _handle_frame_jpg(self, params: Dict[str, list]):
        if not self._check_auth():
            self._send_json_response(401, {"error": "Unauthorized", "message": "Missing or invalid X-Sentinel-Token."})
            return

        detector = self.server.detector
        if detector is None or not hasattr(detector, "get_latest_preview_jpeg"):
            self._send_json_response(503, {"error": "Service Unavailable", "message": "Preview stream not available."})
            return

        w, h = _parse_preview_dimensions(params)

        ret = detector.get_latest_preview_jpeg(width=w, height=h)
        if ret is None:
            self._send_json_response(503, {"error": "Service Unavailable", "message": "Waiting for camera frame."})
            return

        jpeg_bytes, ts, seq = ret
        try:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg_bytes)))
            self.send_header("X-Frame-Timestamp", f"{ts:.4f}")
            self.send_header("X-Frame-Index", str(seq))
            det_fps = getattr(detector, "fps", None)
            if det_fps is not None:
                self.send_header("X-Detector-FPS", f"{det_fps:.1f}")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "X-Frame-Timestamp, X-Frame-Index, X-Detector-FPS")
            self.end_headers()
            self.wfile.write(jpeg_bytes)
        except ConnectionError:
            pass

    def _handle_stream_mjpg(self, params: Dict[str, list]):
        if not self._check_auth():
            self._send_json_response(401, {"error": "Unauthorized", "message": "Missing or invalid X-Sentinel-Token."})
            return

        detector = self.server.detector
        if detector is None or not hasattr(detector, "get_latest_preview_jpeg"):
            self._send_json_response(503, {"error": "Service Unavailable", "message": "Preview stream not available."})
            return

        w, h = _parse_preview_dimensions(params)

        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            last_seq = -1
            while getattr(self.server, "is_serving", True):
                cur_seq = getattr(detector, "latest_preview_seq", None)
                if cur_seq is not None and cur_seq == last_seq:
                    time.sleep(0.02)
                    continue
                ret = detector.get_latest_preview_jpeg(width=w, height=h)
                if ret is None:
                    time.sleep(0.04)
                    continue
                jpeg_bytes, ts, seq = ret
                if seq == last_seq:
                    time.sleep(0.015)
                    continue
                last_seq = seq
                part_header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg_bytes)).encode("ascii") + b"\r\n"
                    b"X-Frame-Timestamp: " + f"{ts:.4f}".encode("ascii") + b"\r\n"
                    b"X-Frame-Index: " + str(seq).encode("ascii") + b"\r\n\r\n"
                )
                self.wfile.write(part_header + jpeg_bytes + b"\r\n")
                self.wfile.flush()
        except ConnectionError:
            pass

    def _handle_registry(self, params: Dict[str, list]):
        if not self._check_auth():
            self._send_json_response(401, {"error": "Unauthorized", "message": "Missing or invalid X-Sentinel-Token."})
            return

        detector = self.server.detector
        tracker = getattr(detector, "tracker", None) if detector else None
        db_path = getattr(detector, "db_path", None)
        if not db_path and tracker is not None:
            db_path = getattr(tracker, "db_path", None)
        if not db_path:
            import config
            db_path = getattr(config, "TRACKER_DB_PATH", "mosquito_registry.db")

        if not os.path.isfile(db_path):
            self._send_json_response(200, {"total": 0, "records": []})
            return

        try:
            limit = 500
            if "limit" in params:
                try:
                    limit = max(1, min(1000, int(params["limit"][0])))
                except (ValueError, IndexError):
                    pass

            date_filter = params.get("date", [None])[0]

            uri = f"file:{os.path.abspath(db_path)}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=1.0)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='mosquito_registry'")
            if cur.fetchone()[0] == 0:
                conn.close()
                self._send_json_response(200, {"total": 0, "records": []})
                return

            if date_filter:
                clean_date = date_filter.strip()
                cur.execute(
                    "SELECT id, mos_id, species, first_seen, last_seen, detect_count, "
                    "confidence, snapshot_path, verification_status "
                    "FROM mosquito_registry "
                    "WHERE first_seen LIKE ? OR last_seen LIKE ? "
                    "ORDER BY id DESC LIMIT ?",
                    (f"{clean_date}%", f"{clean_date}%", limit)
                )
            else:
                cur.execute(
                    "SELECT id, mos_id, species, first_seen, last_seen, detect_count, "
                    "confidence, snapshot_path, verification_status "
                    "FROM mosquito_registry "
                    "ORDER BY id DESC LIMIT ?",
                    (limit,)
                )

            rows = cur.fetchall()
            conn.close()

            records = [dict(r) for r in rows]
            self._send_json_response(200, {"total": len(records), "records": records})
        except Exception as e:
            self._send_json_response(500, {"error": "Database Error", "detail": "Could not read registry"})

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            params = parse_qs(parsed.query)

            if path == "/api/frame.jpg":
                self._handle_frame_jpg(params)
                return

            if path == "/api/stream.mjpg":
                self._handle_stream_mjpg(params)
                return

            if path == "/api/registry":
                self._handle_registry(params)
                return

            if path != "/api/status":
                self._send_json_response(404, {"error": "Not Found", "message": "Endpoint not found."})
                return

            if not self._check_auth():
                self._send_json_response(401, {"error": "Unauthorized", "message": "Missing or invalid X-Sentinel-Token."})
                return

            detector = self.server.detector
            if detector is None:
                self._send_json_response(503, {"error": "Service Unavailable", "message": "Detector reference not bound."})
                return

            # Compute uptime
            uptime_sec = None
            if getattr(detector, "start_time", None) is not None:
                uptime_sec = round(time.time() - detector.start_time, 2)

            # Age of last frame
            age_of_last_frame = None
            last_frame_t = getattr(detector, "last_frame_time", None)
            if last_frame_t is not None:
                age_of_last_frame = round(time.time() - last_frame_t, 3)

            # Tracker status via thread-safe lock
            active_track_ids = []
            active_tracks_summary = []
            tracker = getattr(detector, "tracker", None)
            if tracker is not None and hasattr(tracker, "get_status"):
                t_stat = tracker.get_status()
                active_track_ids = t_stat.get("active_ids", [])
                active_tracks_summary = t_stat.get("active_tracks", [])

            # Confirmed detections in the current frame (as of the last AI result)
            confirmed_in_frame = getattr(detector, "confirmed_detections_count", None)

            # Registry row count (cached read-only query)
            db_path = getattr(detector, "db_path", None)
            if not db_path and tracker is not None:
                db_path = getattr(tracker, "db_path", None)
            if not db_path:
                import config
                db_path = getattr(config, "TRACKER_DB_PATH", "mosquito_registry.db")

            registry_total = self.server.get_cached_registry_count(db_path)

            # Optional CUSUM statistical monitor status
            cusum_data = None
            cusum_mon = getattr(detector, "cusum_monitor", None)
            if cusum_mon is not None and hasattr(cusum_mon, "get_status"):
                try:
                    cusum_data = cusum_mon.get_status().to_dict()
                    if hasattr(cusum_mon, "get_alert_history"):
                        cusum_data["alert_history"] = cusum_mon.get_alert_history(limit=20)
                except Exception:
                    cusum_data = None

            status_payload = {
                "detector_running": bool(getattr(detector, "worker_running", False)),
                "uptime_sec": uptime_sec,
                "fps": round(detector.fps, 2) if getattr(detector, "fps", None) is not None else None,
                "ai_fps": round(detector.ai_fps, 2) if getattr(detector, "ai_fps", None) is not None else None,
                "age_of_last_frame_sec": age_of_last_frame,
                "mode": str(getattr(detector, "mode", "unknown")),
                "confirmed_detections_in_frame": confirmed_in_frame,
                "active_track_ids": active_track_ids,
                "active_tracks_summary": active_tracks_summary,
                "registry_total_count": registry_total,
                "cusum": cusum_data,
                "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self._send_json_response(200, status_payload)

        except Exception as e:
            # Bad or malformed requests must never crash the server or detector
            self._send_json_response(500, {"error": "Internal Server Error", "detail": str(e)})

    def do_POST(self):
        self._send_json_response(404, {"error": "Not Found", "message": "Method not allowed."})

    def do_PUT(self):
        self._send_json_response(404, {"error": "Not Found", "message": "Method not allowed."})

    def do_DELETE(self):
        self._send_json_response(404, {"error": "Not Found", "message": "Method not allowed."})


def load_sentinel_token(secret_path: Optional[str] = None) -> Optional[str]:
    """
    Loads token from environment variable SENTINEL_AUTH_TOKEN or secret_path JSON file.
    Returns None if no token is found or if it is empty.
    """
    env_token = os.environ.get("SENTINEL_AUTH_TOKEN", "").strip()
    if env_token:
        return env_token

    if secret_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        secret_path = os.path.join(base_dir, "sentinel_secret.json")

    if os.path.isfile(secret_path):
        try:
            with open(secret_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                token = str(data.get("token", "")).strip()
                if token:
                    return token
        except Exception as e:
            print(f"[WARNING] Could not read secret file {secret_path}: {e}")

    return None


def start_status_server(detector, host: str = "0.0.0.0", port: int = 8080,
                        secret_path: Optional[str] = None) -> Tuple[Optional[ThreadedHTTPServer], Optional[threading.Thread]]:
    """
    Starts the threaded status HTTP server in a daemon thread.
    Refuses to start if no token is configured.
    """
    token = load_sentinel_token(secret_path)
    if not token:
        print("[WARNING] Status server NOT started: No token found in SENTINEL_AUTH_TOKEN env or sentinel_secret.json.")
        print("          Configure a token to enable the remote GUI status server.")
        return None, None

    try:
        server = ThreadedHTTPServer((host, port), SentinelStatusHandler, detector_ref=detector, token=token)
    except Exception as e:
        print(f"[ERROR] Failed to bind status server to {host}:{port}: {e}")
        return None, None

    def _serve_loop():
        server.is_serving = True
        try:
            server.serve_forever(poll_interval=0.2)
        finally:
            server.is_serving = False

    thread = threading.Thread(target=_serve_loop, daemon=True, name="SentinelStatusServerThread")
    thread.start()
    print(f"[INFO] Sentinel Status Server running on http://{host}:{port}/api/status (token authentication active)")
    return server, thread


def stop_status_server(server: Optional[ThreadedHTTPServer], thread: Optional[threading.Thread]):
    """
    Cleanly shuts down and closes the status server.
    Only calls shutdown() if serve_forever is actually running, preventing deadlocks.
    """
    if server is None:
        return

    try:
        if getattr(server, "is_serving", False):
            server.shutdown()
        server.server_close()
    except Exception as e:
        print(f"[WARNING] Error while closing status server: {e}")

    if thread is not None and thread.is_alive():
        thread.join(timeout=3.0)
    print("[INFO] Sentinel Status Server stopped cleanly.")
