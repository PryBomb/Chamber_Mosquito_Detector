"""Isolated unit tests for Live Camera Streaming, FrameFetchWorker, and LiveDetectionPage.
Runs strictly in-memory without saving any media/sample files to disk.
Run: python -m unittest test_live_stream -v
"""
import os
import time
import socket
import threading
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QBuffer, QByteArray

from gui_app import (
    PYSIDE6_AVAILABLE,
    FrameFetchWorker,
    VideoViewport,
    LiveDetectionPage,
    MosquitoSentinelWindow,
)


def create_in_memory_jpeg(width: int = 1280, height: int = 720, color_rgb: int = 0x1F8A4C) -> bytes:
    """Creates a valid JPEG byte payload entirely in memory without writing to disk."""
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(color_rgb)
    buf = QBuffer()
    buf.open(QBuffer.ReadWrite)
    img.save(buf, "JPG")
    return bytes(buf.data())


class MockCameraHandler(BaseHTTPRequestHandler):
    """Local in-memory HTTP server simulating Pi camera /api/frame.jpg endpoint."""
    response_mode = "200_ok"  # "200_ok", "401_unauth", "404_not_found", "503_no_frame", "corrupt_jpeg"
    expected_token = "secret-token-123"
    valid_jpeg_bytes = b""
    downscaled_jpeg_bytes = b""
    frame_index_override = None

    def log_message(self, format, *args):
        pass  # silence request logs

    def do_GET(self):
        clean_path = self.path.split("?")[0]
        if clean_path not in ("/api/frame.jpg", "/api/stream.mjpg"):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        token_hdr = self.headers.get("X-Sentinel-Token", "")
        if self.expected_token and token_hdr != self.expected_token:
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"Unauthorized")
            return

        if clean_path == "/api/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--frame")
            self.end_headers()
            part = (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(self.valid_jpeg_bytes)).encode("ascii")
                + b"\r\n\r\n"
                + self.valid_jpeg_bytes
                + b"\r\n"
            )
            self.wfile.write(part)
            return

        if self.response_mode == "200_ok":
            # Return downscaled 640x360 frame if requested in query
            data = self.downscaled_jpeg_bytes if "640" in self.path and self.downscaled_jpeg_bytes else self.valid_jpeg_bytes
            seq_val = str(self.frame_index_override) if self.frame_index_override is not None else "101"
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Frame-Timestamp", f"{time.time() - 0.040:.4f}")
            self.send_header("X-Frame-Index", seq_val)
            self.send_header("X-Detector-FPS", "28.5")
            self.end_headers()
            self.wfile.write(data)
        elif self.response_mode == "404_not_found":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
        elif self.response_mode == "503_no_frame":
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"No frame yet")
        elif self.response_mode == "corrupt_jpeg":
            corrupt = b"NOT_A_JPEG_CORRUPTED_BYTES"
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(corrupt)))
            self.end_headers()
            self.wfile.write(corrupt)
        else:
            self.send_response(500)
            self.end_headers()


class TestLiveStreamIntegration(unittest.TestCase):
    """Tests FrameFetchWorker, VideoViewport, and LiveDetectionPage."""

    @classmethod
    def setUpClass(cls):
        if not PYSIDE6_AVAILABLE:
            raise unittest.SkipTest("PySide6 is not available")
        cls.app = QApplication.instance() or QApplication([])
        MockCameraHandler.valid_jpeg_bytes = create_in_memory_jpeg(1280, 720)
        MockCameraHandler.downscaled_jpeg_bytes = create_in_memory_jpeg(640, 360)

        # Start ephemeral local HTTP server
        cls.server = HTTPServer(("127.0.0.1", 0), MockCameraHandler)
        cls.port = cls.server.server_port
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        MockCameraHandler.response_mode = "200_ok"
        MockCameraHandler.expected_token = "secret-token-123"
        self.config = {
            "pi_host": "127.0.0.1",
            "pi_port": self.port,
            "token": "secret-token-123",
            "timeout_sec": 3.0,
            "poll_interval_sec": 1.0,
        }

    def test_video_viewport_rendering_and_scaling(self):
        """Test VideoViewport empty state, aspect-ratio scaling, and stale note."""
        vp = VideoViewport()
        vp.resize(640, 360)
        self.assertFalse(vp.has_frame())

        # Test empty state paint
        vp.set_error_message("Connecting…", "Waiting for camera.")
        vp.repaint()

        # Test valid frame paint
        pm = QPixmap()
        pm.loadFromData(MockCameraHandler.valid_jpeg_bytes)
        vp.set_frame(pm)
        self.assertTrue(vp.has_frame())
        vp.repaint()

        # Test stale overlay paint
        vp.set_stale(True, "Connection timed out")
        self.assertTrue(vp._is_stale)
        vp.repaint()

        # Clear frame
        vp.set_stale(False)
        self.assertFalse(vp._is_stale)

    def test_live_detection_page_initial_state(self):
        """Verify initial badges, labels, and disabled snapshot button."""
        page = LiveDetectionPage()
        page.show()
        self.assertEqual(page.badge.text(), "● Connecting…")
        self.assertIn("Resolution —", page.lbl_stream_info.text())
        self.assertFalse(page.btn_snap.isEnabled(), "Snapshot button must remain disabled")
        self.assertFalse(page.viewport.has_frame())
        page.deleteLater()

    def test_live_detection_page_frame_received(self):
        """Receiving a frame updates resolution, displays Stream rate — on first frame, and measured fps on second."""
        page = LiveDetectionPage()
        page.show()
        img = QImage()
        img.loadFromData(MockCameraHandler.valid_jpeg_bytes)

        # First arrival: fewer than two frame times, must show "Stream rate —"
        page.on_frame_received(img)
        self.assertEqual(page.badge.text(), "● Live")
        self.assertTrue(page.viewport.has_frame())
        self.assertIn("1280 × 720", page.lbl_stream_info.text())
        self.assertIn("Stream rate —", page.lbl_stream_info.text())

        # Second arrival: at least two frame times, displays measured rate
        time.sleep(0.01)
        page.on_frame_received(img)
        self.assertIn("Stream rate", page.lbl_stream_info.text())
        self.assertIn("fps (network fetch)", page.lbl_stream_info.text())
        page.deleteLater()

    def test_live_detection_page_stale_retention_on_transient_error(self):
        """On transient error, last good frame remains on screen with a stale note."""
        page = LiveDetectionPage()
        page.show()
        img = QImage()
        img.loadFromData(MockCameraHandler.valid_jpeg_bytes)
        page.on_frame_received(img)
        self.assertEqual(page.badge.text(), "● Live")

        # Transient timeout error occurs
        page.on_frame_error("timeout", "Connection timed out")
        self.assertEqual(page.badge.text(), "● Stale")
        self.assertTrue(page.viewport.has_frame(), "Last good frame must remain on screen")
        self.assertTrue(page.viewport._is_stale)
        self.assertIn("stale", page.lbl_stream_info.text().lower())

        # Frame resumes: recovers cleanly to Live
        page.on_frame_received(img)
        self.assertEqual(page.badge.text(), "● Live")
        self.assertFalse(page.viewport._is_stale)
        page.deleteLater()

    def test_live_detection_page_separate_error_messages(self):
        """Verify distinct error messages for 401, 404, 503, timeout, connection_refused, decode_error."""
        error_types = [
            ("unauthorized", "Unauthorized (401)", "Error"),
            ("not_found", "Endpoint Not Found (404)", "Error"),
            ("no_frame", "Waiting for first frame (503)", "Connecting…"),
            ("timeout", "Connection Timed Out", "Error"),
            ("connection_refused", "Cannot Connect to Raspberry Pi", "Error"),
            ("decode_error", "Picture Decode Error", "Error"),
        ]
        for err_type, expected_title, expected_badge in error_types:
            page = LiveDetectionPage()
            page.show()
            page.on_frame_error(err_type, "Test detail")
            self.assertEqual(page.viewport._error_title, expected_title)
            self.assertIn(expected_badge, page.badge.text())
            page.deleteLater()

    def _wait_until(self, condition, timeout: float = 3.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            self.app.processEvents()
            if condition():
                return True
            time.sleep(0.05)
        self.app.processEvents()
        return condition()

    def test_frame_fetch_worker_successful_fetch(self):
        """Worker fetches /api/frame.jpg and emits QImage and fetch duration."""
        worker = FrameFetchWorker(self.config)
        received = []

        def on_recv(image, sec):
            received.append((image, sec))

        worker.sig_frame_received.connect(on_recv)
        worker.resume()
        worker.start()

        self.assertTrue(self._wait_until(lambda: len(received) > 0, timeout=4.0))

        worker.stop()
        worker.wait(3000)

        self.assertGreaterEqual(len(received), 1)
        image, sec = received[0]
        self.assertIsInstance(image, QImage)
        self.assertFalse(image.isNull())
        self.assertEqual(image.width(), 1280)
        self.assertEqual(image.height(), 720)
        self.assertGreater(sec, 0.0)

    def test_frame_fetch_worker_unauthorized(self):
        """Worker emits unauthorized on missing or invalid token (401)."""
        bad_config = dict(self.config, token="wrong-token")
        worker = FrameFetchWorker(bad_config)
        errors = []
        worker.sig_frame_error.connect(lambda t, d: errors.append((t, d)))

        worker.resume()
        worker.start()
        self.assertTrue(self._wait_until(lambda: len(errors) > 0, timeout=4.0))
        worker.stop()
        worker.wait(3000)

        self.assertGreaterEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "unauthorized")

    def test_frame_fetch_worker_503_waiting_for_frame(self):
        """Worker emits no_frame when Pi server returns 503."""
        MockCameraHandler.response_mode = "503_no_frame"
        worker = FrameFetchWorker(self.config)
        errors = []
        worker.sig_frame_error.connect(lambda t, d: errors.append((t, d)))

        worker.resume()
        worker.start()
        self.assertTrue(self._wait_until(lambda: len(errors) > 0, timeout=4.0))
        worker.stop()
        worker.wait(3000)

        self.assertGreaterEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "no_frame")

    def test_frame_fetch_worker_decode_error(self):
        """Worker emits decode_error when received bytes are corrupted."""
        MockCameraHandler.response_mode = "corrupt_jpeg"
        worker = FrameFetchWorker(self.config)
        errors = []
        worker.sig_frame_error.connect(lambda t, d: errors.append((t, d)))

        worker.resume()
        worker.start()
        self.assertTrue(self._wait_until(lambda: len(errors) > 0, timeout=4.0))
        worker.stop()
        worker.wait(3000)

        self.assertGreaterEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "decode_error")

    def test_frame_fetch_worker_connection_refused(self):
        """Worker emits connection_refused when port has no listening server."""
        # Pick unused port
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        unused_port = sock.getsockname()[1]
        sock.close()

        refused_config = dict(self.config, pi_port=unused_port)
        worker = FrameFetchWorker(refused_config)
        errors = []
        worker.sig_frame_error.connect(lambda t, d: errors.append((t, d)))

        worker.resume()
        worker.start()
        self.assertTrue(self._wait_until(lambda: len(errors) > 0, timeout=4.0))
        worker.stop()
        worker.wait(3000)

        self.assertGreaterEqual(len(errors), 1)
        self.assertIn(errors[0][0], ("connection_refused", "timeout", "network_error"))

    def test_window_page_pause_resume_lifecycle(self):
        """Worker pauses on other pages and resumes only when Live Detection page is active."""
        window = MosquitoSentinelWindow(self.config)
        window.show()

        # Overview page (index 0): worker must be paused
        self.assertEqual(window.stack.currentIndex(), 0)
        self.assertTrue(window.frame_worker._paused)

        # Switch to Live Detection page (index 1): worker must resume
        window.switch_page(1)
        self.assertEqual(window.stack.currentIndex(), 1)
        self.assertFalse(window.frame_worker._paused)

        # Switch to Analytics page (index 3): worker must pause
        window.switch_page(3)
        self.assertEqual(window.stack.currentIndex(), 3)
        self.assertTrue(window.frame_worker._paused)

        # Clean window close stops both workers
        window.close()
        self.assertFalse(window.frame_worker._running)
        self.assertFalse(window.polling_worker._running)

    def test_frame_fetch_worker_diagnostics_and_latency(self):
        """Worker measures download time, decode time, frame age, and emits diagnostics."""
        worker = FrameFetchWorker(self.config)
        diags = []
        worker.sig_diagnostics_updated.connect(lambda d: diags.append(d))

        worker.resume()
        worker.start()
        self.assertTrue(self._wait_until(lambda: len(diags) > 0, timeout=4.0))
        worker.stop()
        worker.wait(3000)

        self.assertGreaterEqual(len(diags), 1)
        d = diags[0]
        self.assertIn("download_ms", d)
        self.assertIn("decode_ms", d)
        self.assertIn("total_fetch_ms", d)
        self.assertIn("frame_age_ms", d)
        self.assertIn("detector_fps", d)
        self.assertGreater(d["download_ms"], 0.0)
        self.assertGreater(d["decode_ms"], 0.0)
        self.assertEqual(d["detector_fps"], 28.5)

    def test_stale_frame_discarding(self):
        """Worker drops duplicate/outdated frame sequence numbers and emits sig_stale_dropped."""
        worker = FrameFetchWorker(self.config)
        dropped = []
        received = []
        worker.sig_stale_dropped.connect(lambda seq: dropped.append(seq))
        worker.sig_frame_received.connect(lambda img, sec: received.append(img))

        # First frame with index 101
        MockCameraHandler.frame_index_override = "101"
        worker.resume()
        worker.start()
        self.assertTrue(self._wait_until(lambda: len(received) >= 1, timeout=4.0))

        # Server returns old frame index 100 -> must be dropped as stale
        MockCameraHandler.frame_index_override = "100"
        self.assertTrue(self._wait_until(lambda: 100 in dropped, timeout=4.0))

        worker.stop()
        worker.wait(3000)

        self.assertGreaterEqual(len(dropped), 1)
        self.assertIn(100, dropped)
        MockCameraHandler.frame_index_override = None

    def test_preview_resolution_downscaling_640x360(self):
        """Worker requests 640x360 preview without altering detector resolution."""
        cfg_640 = dict(self.config, preview_width=640, preview_height=360)
        worker = FrameFetchWorker(cfg_640)
        received = []
        worker.sig_frame_received.connect(lambda img, sec: received.append(img))

        worker.resume()
        worker.start()
        self.assertTrue(self._wait_until(lambda: len(received) >= 1, timeout=4.0))
        worker.stop()
        worker.wait(3000)

        self.assertGreaterEqual(len(received), 1)
        img = received[0]
        self.assertEqual(img.width(), 640)
        self.assertEqual(img.height(), 360)

    def test_live_detection_page_rich_diagnostics_tooltip(self):
        """Diagnostics updates enrich label and tooltip with latency, FPS, and frame age."""
        page = LiveDetectionPage()
        page.show()
        img = QImage()
        img.loadFromData(MockCameraHandler.downscaled_jpeg_bytes)

        # Feed 2 frames to measure arrival rate
        page.on_frame_received(img)
        time.sleep(0.02)
        page.on_frame_received(img)

        # Feed detailed diagnostics
        page.on_diagnostics_updated({
            "download_ms": 12.4,
            "decode_ms": 2.1,
            "total_fetch_ms": 14.5,
            "frame_age_ms": 35.0,
            "interval_ms": 48.0,
            "detector_fps": 28.5,
            "dropped_stale": 2,
        })

        lbl_text = page.lbl_stream_info.text()
        tooltip_text = page.lbl_stream_info.toolTip()

        self.assertIn("640 × 360", lbl_text)
        self.assertIn("Pi 28.5 fps", lbl_text)
        self.assertIn("Latency", lbl_text)
        self.assertIn("Age 0.04s", lbl_text)
        self.assertIn("Arrival Interval", tooltip_text)
        self.assertIn("Stale / Duplicate Frames Discarded: 2", tooltip_text)
        page.deleteLater()

    def test_status_server_frame_and_mjpeg_endpoints(self):
        """Status server serves /api/frame.jpg with auth and downscaling, and /api/stream.mjpg."""
        from status_server import ThreadedHTTPServer, SentinelStatusHandler
        import urllib.request

        # Mock detector with get_latest_preview_jpeg
        class MockDetector:
            fps = 27.8
            def get_latest_preview_jpeg(self, width=640, height=360, quality=80):
                return MockCameraHandler.downscaled_jpeg_bytes, time.time() - 0.03, 55

        srv = ThreadedHTTPServer(("127.0.0.1", 0), SentinelStatusHandler, detector_ref=MockDetector(), token="test-tok")
        srv_port = srv.server_port
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        srv.is_serving = True

        try:
            # 1. GET without token -> 401
            req_unauth = urllib.request.Request(f"http://127.0.0.1:{srv_port}/api/frame.jpg")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_unauth)
            self.assertEqual(ctx.exception.code, 401)

            # 2. GET with token -> 200 image/jpeg with headers
            req_auth = urllib.request.Request(f"http://127.0.0.1:{srv_port}/api/frame.jpg?w=640&h=360")
            req_auth.add_header("X-Sentinel-Token", "test-tok")
            with urllib.request.urlopen(req_auth) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("Content-Type"), "image/jpeg")
                self.assertEqual(resp.headers.get("X-Frame-Index"), "55")
                self.assertIn("27.8", resp.headers.get("X-Detector-FPS"))
                body = resp.read()
                self.assertGreater(len(body), 100)

            # 3. GET /api/stream.mjpg with token -> 200 multipart
            req_stream = urllib.request.Request(f"http://127.0.0.1:{srv_port}/api/stream.mjpg")
            req_stream.add_header("X-Sentinel-Token", "test-tok")
            with urllib.request.urlopen(req_stream) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("multipart/x-mixed-replace", resp.headers.get("Content-Type"))
                chunk = resp.read(256)
                self.assertIn(b"--frame", chunk)

        finally:
            srv.is_serving = False
            srv.shutdown()
            srv.server_close()

    def test_status_server_oversized_dimensions_clamped(self):
        """Oversized width/height clamped to 1280x720, undersized to 160x90, invalid to 640x360."""
        from status_server import ThreadedHTTPServer, SentinelStatusHandler
        import urllib.request

        captured_sizes = []

        class MockDetector:
            latest_preview_seq = 1

            def get_latest_preview_jpeg(self, width=640, height=360, quality=80):
                captured_sizes.append((width, height))
                return MockCameraHandler.downscaled_jpeg_bytes, time.time(), 1

        srv = ThreadedHTTPServer(("127.0.0.1", 0), SentinelStatusHandler, detector_ref=MockDetector(), token="test-tok")
        srv_port = srv.server_port
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        srv.is_serving = True

        try:
            # 1. Oversized dimensions -> clamped to 1280x720
            req = urllib.request.Request(f"http://127.0.0.1:{srv_port}/api/frame.jpg?width=3840&height=2160")
            req.add_header("X-Sentinel-Token", "test-tok")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
            self.assertEqual(captured_sizes[-1], (1280, 720))

            # 2. Undersized dimensions -> clamped to 160x90
            req = urllib.request.Request(f"http://127.0.0.1:{srv_port}/api/frame.jpg?w=50&h=20")
            req.add_header("X-Sentinel-Token", "test-tok")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
            self.assertEqual(captured_sizes[-1], (160, 90))

            # 3. Invalid dimensions -> fallback to 640x360
            req = urllib.request.Request(f"http://127.0.0.1:{srv_port}/api/frame.jpg?w=invalid&h=bad")
            req.add_header("X-Sentinel-Token", "test-tok")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
            self.assertEqual(captured_sizes[-1], (640, 360))

        finally:
            srv.is_serving = False
            srv.shutdown()
            srv.server_close()

    def test_status_server_stream_does_not_call_encoder_when_seq_unchanged(self):
        """Stream does not call get_latest_preview_jpeg when sequence number has not changed."""
        from status_server import ThreadedHTTPServer, SentinelStatusHandler
        import urllib.request

        class MockSeqDetector:
            def __init__(self):
                self.latest_preview_seq = 100
                self.call_count = 0

            def get_latest_preview_jpeg(self, width=640, height=360, quality=80):
                self.call_count += 1
                return MockCameraHandler.downscaled_jpeg_bytes, time.time(), self.latest_preview_seq

        det = MockSeqDetector()
        srv = ThreadedHTTPServer(("127.0.0.1", 0), SentinelStatusHandler, detector_ref=det, token="test-tok")
        srv_port = srv.server_port
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        srv.is_serving = True

        try:
            req_stream = urllib.request.Request(f"http://127.0.0.1:{srv_port}/api/stream.mjpg")
            req_stream.add_header("X-Sentinel-Token", "test-tok")
            with urllib.request.urlopen(req_stream, timeout=2.0) as resp:
                chunk = resp.read(256)
                self.assertIn(b"--frame", chunk)
                self.assertEqual(det.call_count, 1)

                # Wait while seq remains 100: encoder should NOT be called again
                time.sleep(0.08)
                self.assertEqual(det.call_count, 1)

                # Advance sequence number
                det.latest_preview_seq = 101
                time.sleep(0.06)
                self.assertEqual(det.call_count, 2)

        finally:
            srv.is_serving = False
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
