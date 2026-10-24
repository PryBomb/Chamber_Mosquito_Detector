"""
gui_app.py - MosquitoSentinel Desktop GUI (Stage 5A: Client Integration)
========================================================================
A modern, scientific dark-themed desktop application for the
Chamber Mosquito Detection and Spatial Tracking System.

Stage 5A Architecture:
- Connects to the Pi status server via read-only GET /api/status.
- Background QThread (StatusPollingWorker) performs non-blocking HTTP polling.
- Zero GUI freezing during network latency or disconnection.
- Strictly displays live telemetry when connected; unknown/null fields display as N/A.
- Never hardcodes tokens in source; loads from client_config.json or env vars.
- Clean shutdown on window close.
"""

import sys
import os
import platform
import json
import time
import socket
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from PySide6.QtCore import Qt, QTimer, QSize, QThread, Signal, Slot
    from PySide6.QtGui import QFont, QColor, QIcon, QPainter, QBrush, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QStackedWidget,
        QPushButton,
        QLabel,
        QFrame,
        QScrollArea,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QSizePolicy,
        QSpacerItem,
    )
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Client Configuration Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_client_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads client networking configuration from client_config.json or environment.
    Keeps IP addresses, ports, and secret tokens outside of Git tracking.
    """
    cfg = {
        "pi_host": os.environ.get("SENTINEL_PI_HOST", "127.0.0.1"),
        "pi_port": int(os.environ.get("SENTINEL_PI_PORT", 8080)),
        "token": os.environ.get("SENTINEL_AUTH_TOKEN", ""),
        "poll_interval_sec": float(os.environ.get("SENTINEL_POLL_INTERVAL", 1.0)),
        "timeout_sec": 2.5
    }

    if config_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "client_config.json")

    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)
                if "pi_host" in user_data and str(user_data["pi_host"]).strip():
                    cfg["pi_host"] = str(user_data["pi_host"]).strip()
                if "pi_port" in user_data:
                    cfg["pi_port"] = int(user_data["pi_port"])
                if "token" in user_data:
                    cfg["token"] = str(user_data["token"]).strip()
                if "poll_interval_sec" in user_data:
                    cfg["poll_interval_sec"] = max(0.2, float(user_data["poll_interval_sec"]))
                if "timeout_sec" in user_data:
                    cfg["timeout_sec"] = max(0.5, float(user_data["timeout_sec"]))
        except Exception as e:
            print(f"[WARNING] Error reading {config_path}: {e}")

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2. Non-Blocking Background Status Polling Worker
# ─────────────────────────────────────────────────────────────────────────────
if PYSIDE6_AVAILABLE:
    class StatusPollingWorker(QThread):
        """
        Background worker that polls GET /api/status periodically.
        Emits Qt signals to update UI without blocking the main event loop.
        """
        sig_status_updated = Signal(dict)
        sig_connection_error = Signal(str, str)  # (error_type, description)

        def __init__(self, config: Dict[str, Any], parent=None):
            super().__init__(parent)
            self.config = config
            self._running = True

        def stop(self):
            """Signals the thread to terminate cleanly."""
            self._running = False

        def run(self):
            url = f"http://{self.config['pi_host']}:{self.config['pi_port']}/api/status"
            token = self.config.get("token", "")
            timeout = self.config.get("timeout_sec", 2.5)
            interval = self.config.get("poll_interval_sec", 1.0)

            while self._running:
                try:
                    req = urllib.request.Request(url, method="GET")
                    if token:
                        req.add_header("X-Sentinel-Token", token)

                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        if resp.status == 200:
                            raw = resp.read().decode("utf-8")
                            try:
                                data = json.loads(raw)
                                if isinstance(data, dict):
                                    self.sig_status_updated.emit(data)
                                else:
                                    self.sig_connection_error.emit("malformed_json", "Response is not a JSON object")
                            except json.JSONDecodeError as je:
                                self.sig_connection_error.emit("malformed_json", f"JSON parse error: {je}")
                        else:
                            self.sig_connection_error.emit("http_error", f"HTTP {resp.status}")

                except urllib.error.HTTPError as he:
                    if he.code == 401:
                        self.sig_connection_error.emit("unauthorized", "Unauthorized: X-Sentinel-Token invalid or missing")
                    elif he.code == 404:
                        self.sig_connection_error.emit("not_found", "Endpoint /api/status not found (404)")
                    else:
                        self.sig_connection_error.emit("http_error", f"HTTP {he.code}: {he.reason}")
                except urllib.error.URLError as ue:
                    reason_str = str(ue.reason)
                    if "timed out" in reason_str.lower() or "timeout" in reason_str.lower():
                        self.sig_connection_error.emit("timeout", f"Connection timed out: {self.config['pi_host']}:{self.config['pi_port']}")
                    elif "connectionrefused" in reason_str.lower() or "actively refused" in reason_str.lower():
                        self.sig_connection_error.emit("connection_refused", f"Cannot connect to {self.config['pi_host']}:{self.config['pi_port']}")
                    else:
                        self.sig_connection_error.emit("network_error", reason_str)
                except (TimeoutError, socket.timeout):
                    self.sig_connection_error.emit("timeout", f"Connection timed out: {self.config['pi_host']}:{self.config['pi_port']}")
                except Exception as ex:
                    self.sig_connection_error.emit("unknown_error", str(ex))

                # Sleep in small slices so stop() responds immediately
                elapsed = 0.0
                while self._running and elapsed < interval:
                    time.sleep(0.1)
                    elapsed += 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scientific Dark Theme Stylesheet (QSS)
# ─────────────────────────────────────────────────────────────────────────────
DARK_STYLE_SHEET = """
QMainWindow {
    background-color: #0d1117;
}

QWidget {
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
    font-size: 13px;
}

/* Sidebar Styling */
#sidebar {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

#brandTitle {
    font-size: 17px;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: 0.5px;
}

#brandSubtitle {
    font-size: 11px;
    color: #8b949e;
    margin-bottom: 8px;
}

/* Navigation Buttons */
QPushButton.nav-btn {
    text-align: left;
    padding: 10px 14px;
    border-radius: 6px;
    border: 1px solid transparent;
    background-color: transparent;
    color: #c9d1d9;
    font-size: 13px;
    font-weight: 500;
}

QPushButton.nav-btn:hover {
    background-color: #21262d;
    color: #f0f6fc;
}

QPushButton.nav-btn:checked, QPushButton.nav-btn.active {
    background-color: #1f2937;
    border: 1px solid #388bfd;
    color: #58a6ff;
    font-weight: 600;
}

/* Card Styling */
QFrame.metric-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
}

QFrame.metric-card:hover {
    border: 1px solid #58a6ff;
    background-color: #1c2128;
}

QLabel.card-title {
    font-size: 11px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QLabel.card-value {
    font-size: 24px;
    font-weight: 700;
    color: #f0f6fc;
    margin-top: 4px;
    margin-bottom: 2px;
}

QLabel.card-subtext {
    font-size: 11px;
    color: #7d8590;
}

/* Panel Containers */
QFrame.content-panel {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 18px;
}

QLabel.panel-heading {
    font-size: 15px;
    font-weight: 600;
    color: #f0f6fc;
    margin-bottom: 8px;
}

/* Tables */
QTableWidget {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    gridline-color: #21262d;
    color: #c9d1d9;
}

QTableWidget::item {
    padding: 6px 8px;
}

QTableWidget::item:selected {
    background-color: #1f2937;
    color: #58a6ff;
}

QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #30363d;
    font-weight: 600;
    font-size: 12px;
}

/* Action Buttons */
QPushButton.action-btn {
    background-color: #21262d;
    border: 1px solid #363b42;
    border-radius: 6px;
    padding: 7px 14px;
    color: #c9d1d9;
    font-weight: 500;
}

QPushButton.action-btn:hover {
    background-color: #30363d;
    border-color: #8b949e;
    color: #ffffff;
}

QPushButton.primary-btn {
    background-color: #238636;
    border: 1px solid #2ea043;
    border-radius: 6px;
    padding: 7px 16px;
    color: #ffffff;
    font-weight: 600;
}

QPushButton.primary-btn:hover {
    background-color: #2ea043;
}

/* Status Badges */
QLabel.status-badge {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #0d1117;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #30363d;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #484f58;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reusable Metric Card Widget
# ─────────────────────────────────────────────────────────────────────────────
if PYSIDE6_AVAILABLE:
    class MetricCard(QFrame):
        """Displays a key operational metric with title, main value, and subtitle."""

        def __init__(self, title: str, initial_value: str, subtitle: str, accent_color: str = "#58a6ff", parent=None):
            super().__init__(parent)
            self.setProperty("class", "metric-card")

            layout = QVBoxLayout(self)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(4)

            self.lbl_title = QLabel(title)
            self.lbl_title.setProperty("class", "card-title")
            self.lbl_title.setStyleSheet(f"color: {accent_color};")

            self.lbl_value = QLabel(initial_value)
            self.lbl_value.setProperty("class", "card-value")

            self.lbl_subtext = QLabel(subtitle)
            self.lbl_subtext.setProperty("class", "card-subtext")

            layout.addWidget(self.lbl_title)
            layout.addWidget(self.lbl_value)
            layout.addWidget(self.lbl_subtext)

        def update_value(self, value: str, subtitle: Optional[str] = None):
            self.lbl_value.setText(value)
            if subtitle is not None:
                self.lbl_subtext.setText(subtitle)

        def set_accent_color(self, color: str):
            self.lbl_title.setStyleSheet(f"color: {color};")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dashboard Overview Page (Connected to Status Telemetry)
# ─────────────────────────────────────────────────────────────────────────────
if PYSIDE6_AVAILABLE:
    class OverviewPage(QWidget):
        """Main dashboard showing live telemetry, metric cards, and active tracks table."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.last_success_time: Optional[float] = None
            self.is_stale = False
            self.init_ui()

        def init_ui(self):
            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(24, 20, 24, 20)
            main_layout.setSpacing(18)

            # Top Header Bar
            header_layout = QHBoxLayout()
            title_box = QVBoxLayout()
            title_box.setSpacing(2)

            page_title = QLabel("System Overview & Chamber Telemetry")
            page_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #f0f6fc;")
            page_subtitle = QLabel("Real-time telemetry polled from Raspberry Pi 4 detector service")
            page_subtitle.setStyleSheet("font-size: 12px; color: #8b949e;")

            title_box.addWidget(page_title)
            title_box.addWidget(page_subtitle)
            header_layout.addLayout(title_box)
            header_layout.addStretch()

            # Live Clock Label
            self.lbl_clock = QLabel()
            self.lbl_clock.setStyleSheet("font-size: 12px; color: #8b949e; background: #161b22; padding: 6px 12px; border-radius: 6px; border: 1px solid #30363d;")
            header_layout.addWidget(self.lbl_clock)
            main_layout.addLayout(header_layout)

            # Grid of 4 Metric Cards
            cards_grid = QGridLayout()
            cards_grid.setHorizontalSpacing(16)
            cards_grid.setVerticalSpacing(16)

            self.card_status = MetricCard(
                title="Detector Status",
                initial_value="CONNECTING...",
                subtitle="Polling GET /api/status",
                accent_color="#58a6ff"
            )
            self.card_fps = MetricCard(
                title="Stream & AI FPS",
                initial_value="N/A",
                subtitle="Display: N/A | AI: N/A",
                accent_color="#3fb950"
            )
            self.card_mosquitoes = MetricCard(
                title="Active Chamber Mosquitoes",
                initial_value="N/A",
                subtitle="Persistent Tracks (Unique MOS-IDs)",
                accent_color="#ff9500"
            )
            self.card_detections = MetricCard(
                title="Detections & Registry Total",
                initial_value="N/A",
                subtitle="Frame: N/A | Registry: N/A",
                accent_color="#ff3b5c"
            )

            cards_grid.addWidget(self.card_status, 0, 0)
            cards_grid.addWidget(self.card_fps, 0, 1)
            cards_grid.addWidget(self.card_mosquitoes, 0, 2)
            cards_grid.addWidget(self.card_detections, 0, 3)
            main_layout.addLayout(cards_grid)

            # Lower Panels
            lower_layout = QHBoxLayout()
            lower_layout.setSpacing(16)

            # Left Panel: Detailed Subsystem Telemetry
            telemetry_panel = QFrame()
            telemetry_panel.setProperty("class", "content-panel")
            telem_vbox = QVBoxLayout(telemetry_panel)

            lbl_subsys_title = QLabel("Chamber Subsystem Diagnostics")
            lbl_subsys_title.setProperty("class", "panel-heading")
            telem_vbox.addWidget(lbl_subsys_title)

            self.telem_labels = {}
            telemetry_rows = [
                ("detector_running", "Detector Engine", "N/A"),
                ("detector_mode", "Stage 1 Detection Mode", "N/A"),
                ("fps", "Display Stream Rate", "N/A"),
                ("ai_fps", "AI Worker Processing Rate", "N/A"),
                ("frame_age", "Last Camera Frame Age", "N/A"),
                ("confirmed_frame", "Confirmed Detections (Current Frame)", "N/A"),
                ("active_ids", "Active Tracked IDs", "N/A"),
                ("registry_total", "Historical SQLite Registry Rows", "N/A"),
            ]

            for key, label_text, default_val in telemetry_rows:
                row = QHBoxLayout()
                lbl_name = QLabel(label_text)
                lbl_name.setStyleSheet("font-size: 13px; color: #c9d1d9;")
                lbl_val = QLabel(default_val)
                lbl_val.setStyleSheet("font-size: 12px; color: #8b949e; font-weight: 500;")
                row.addWidget(lbl_name)
                row.addStretch()
                row.addWidget(lbl_val)
                telem_vbox.addLayout(row)
                self.telem_labels[key] = lbl_val

            telem_vbox.addStretch()
            lower_layout.addWidget(telemetry_panel, stretch=1)

            # Right Panel: Active Tracks Table (Live Chamber Mosquitoes)
            tracks_panel = QFrame()
            tracks_panel.setProperty("class", "content-panel")
            tracks_vbox = QVBoxLayout(tracks_panel)

            lbl_tracks_title = QLabel("Active Chamber Mosquitoes (Live Registry)")
            lbl_tracks_title.setProperty("class", "panel-heading")
            tracks_vbox.addWidget(lbl_tracks_title)

            self.tracks_table = QTableWidget(0, 4)
            self.tracks_table.setHorizontalHeaderLabels(["MOS ID", "Species", "Confidence", "Hit Count"])
            self.tracks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tracks_vbox.addWidget(self.tracks_table)

            lower_layout.addWidget(tracks_panel, stretch=1)
            main_layout.addLayout(lower_layout)
            main_layout.addStretch()

            # 1-second timer for clock and stale data check
            self.clock_timer = QTimer(self)
            self.clock_timer.timeout.connect(self.on_timer_tick)
            self.clock_timer.start(1000)
            self.on_timer_tick()

        def on_timer_tick(self):
            now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
            self.lbl_clock.setText(f"System Time: {now_str}")

            # Check for stale data if last success was > 3.5 seconds ago
            if self.last_success_time is not None:
                stale_delta = time.time() - self.last_success_time
                if stale_delta > 3.5:
                    self.is_stale = True
                    self.card_status.update_value("STALE", f"Data stale (no update for {stale_delta:.1f}s)")
                    self.card_status.set_accent_color("#ff9500")

        def update_telemetry(self, data: Dict[str, Any]):
            """Slot called when valid 200 OK data is received from the status server."""
            self.last_success_time = time.time()
            self.is_stale = False

            # Extract fields with safe defaults (null/missing -> N/A, never 0)
            running = data.get("detector_running")
            uptime = data.get("uptime_sec")
            fps = data.get("fps")
            ai_fps = data.get("ai_fps")
            frame_age = data.get("age_of_last_frame_sec")
            mode = data.get("mode")
            confirmed_frame = data.get("confirmed_detections_in_frame")
            active_ids = data.get("active_track_ids")
            active_tracks = data.get("active_tracks_summary")
            registry_total = data.get("registry_total_count")

            # 1. Update Card 1: Detector Status
            uptime_str = f"{uptime:.1f}s" if isinstance(uptime, (int, float)) else "N/A"
            if running is True:
                self.card_status.update_value("RUNNING", f"Mode: {str(mode).upper() if mode else 'N/A'} | Uptime: {uptime_str}")
                self.card_status.set_accent_color("#3fb950")
            elif running is False:
                self.card_status.update_value("STANDBY", "Detector loop is idle")
                self.card_status.set_accent_color("#8b949e")
            else:
                self.card_status.update_value("N/A", "Detector state unknown")
                self.card_status.set_accent_color("#8b949e")

            # 2. Update Card 2: FPS (Display & AI)
            fps_str = f"{fps:.1f} FPS" if isinstance(fps, (int, float)) else "N/A"
            ai_fps_str = f"{ai_fps:.1f} FPS" if isinstance(ai_fps, (int, float)) else "N/A"
            self.card_fps.update_value(fps_str, f"AI Worker: {ai_fps_str} (Frames processed)")

            # 3. Update Card 3: Active Tracked Mosquitoes (Separate Count)
            if isinstance(active_ids, list):
                active_count_str = f"{len(active_ids)} Active"
                ids_str = ", ".join(str(i) for i in active_ids) if active_ids else "None"
                self.card_mosquitoes.update_value(active_count_str, f"IDs: {ids_str}")
            else:
                active_count_str = "N/A"
                ids_str = "N/A"
                self.card_mosquitoes.update_value("N/A", "Active tracks unavailable")

            # 4. Update Card 4: Confirmed in Frame & Registry Total (Separate Counts)
            frame_str = str(int(confirmed_frame)) if isinstance(confirmed_frame, (int, float)) else "N/A"
            reg_str = f"{int(registry_total):,} rows" if isinstance(registry_total, (int, float)) else "N/A"
            self.card_detections.update_value(f"Frame: {frame_str}", f"Registry Total: {reg_str}")

            # 5. Update Detailed Telemetry Labels
            self.telem_labels["detector_running"].setText("RUNNING (Active)" if running is True else ("STANDBY" if running is False else "N/A"))
            self.telem_labels["detector_running"].setStyleSheet("color: #3fb950;" if running is True else "color: #8b949e;")
            self.telem_labels["detector_mode"].setText(str(mode).upper() if mode else "N/A")
            self.telem_labels["fps"].setText(fps_str)
            self.telem_labels["ai_fps"].setText(ai_fps_str)
            self.telem_labels["frame_age"].setText(f"{frame_age:.3f} s" if isinstance(frame_age, (int, float)) else "N/A")
            self.telem_labels["confirmed_frame"].setText(frame_str)
            self.telem_labels["active_ids"].setText(ids_str)
            self.telem_labels["registry_total"].setText(reg_str)

            # 6. Update Active Tracks Table
            self.tracks_table.setRowCount(0)
            if isinstance(active_tracks, list) and len(active_tracks) > 0:
                valid_tracks = [t for t in active_tracks if isinstance(t, dict)]
                self.tracks_table.setRowCount(len(valid_tracks))
                for row_idx, track in enumerate(valid_tracks):
                    mid = str(track.get("mos_id", "N/A"))
                    sp = str(track.get("species", "N/A"))
                    conf_val = track.get("confidence")
                    conf = f"{conf_val * 100:.1f}%" if isinstance(conf_val, (int, float)) else "N/A"
                    hits = str(track.get("hit_count", "N/A"))

                    for col_idx, text in enumerate([mid, sp, conf, hits]):
                        item = QTableWidgetItem(text)
                        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                        self.tracks_table.setItem(row_idx, col_idx, item)

        def set_error_state(self, err_type: str, description: str):
            """Slot called when polling encounters an error or disconnection."""
            self.is_stale = True

            if err_type == "unauthorized":
                self.card_status.update_value("UNAUTHORIZED", "X-Sentinel-Token missing or invalid")
                self.card_status.set_accent_color("#e3b341")
            elif err_type in ("connection_refused", "timeout"):
                self.card_status.update_value("OFFLINE", "Cannot connect to status server")
                self.card_status.set_accent_color("#f85149")
            elif err_type == "malformed_json":
                self.card_status.update_value("MALFORMED", "Invalid JSON from server")
                self.card_status.set_accent_color("#f85149")
            else:
                self.card_status.update_value("ERROR", description[:45])
                self.card_status.set_accent_color("#f85149")

            self.card_fps.update_value("N/A", "Display: N/A | AI: N/A")
            self.card_mosquitoes.update_value("N/A", "Cannot retrieve active tracks")
            self.card_detections.update_value("N/A", "Frame: N/A | Registry: N/A")

            for key in self.telem_labels:
                self.telem_labels[key].setText("N/A")
                self.telem_labels[key].setStyleSheet("color: #8b949e;")

            self.tracks_table.setRowCount(1)
            err_item = QTableWidgetItem(f"[{err_type.upper()}] {description}")
            err_item.setFlags(err_item.flags() ^ Qt.ItemIsEditable)
            self.tracks_table.setItem(0, 0, err_item)
            for c in range(1, 4):
                filler = QTableWidgetItem("-")
                filler.setFlags(filler.flags() ^ Qt.ItemIsEditable)
                self.tracks_table.setItem(0, c, filler)


    class LiveDetectionPlaceholder(QWidget):
        """Placeholder for Live Camera Feed."""

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 20, 24, 20)
            layout.setSpacing(14)

            header = QHBoxLayout()
            title = QLabel("Live Chamber Video Stream")
            title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f6fc;")
            header.addWidget(title)
            header.addStretch()

            btn_start = QPushButton("Stage 6: Video Streaming")
            btn_start.setProperty("class", "primary-btn")
            btn_start.setEnabled(False)
            header.addWidget(btn_start)
            layout.addLayout(header)

            viewport = QFrame()
            viewport.setStyleSheet("""
                background-color: #05070a;
                border: 2px dashed #30363d;
                border-radius: 8px;
            """)
            vp_layout = QVBoxLayout(viewport)
            vp_layout.setAlignment(Qt.AlignCenter)

            vp_label = QLabel("Chamber Video Stream (Stage 6)")
            vp_label.setStyleSheet("font-size: 18px; font-weight: 600; color: #58a6ff;")
            vp_sub = QLabel("Telemetry integration active (Stage 5A).\nLive video downscaling & streaming reserved for Stage 6.")
            vp_sub.setStyleSheet("font-size: 12px; color: #8b949e; text-align: center;")
            vp_sub.setAlignment(Qt.AlignCenter)

            vp_layout.addWidget(vp_label, alignment=Qt.AlignCenter)
            vp_layout.addWidget(vp_sub, alignment=Qt.AlignCenter)
            layout.addWidget(viewport, stretch=1)


    class HistoryPlaceholder(QWidget):
        """Placeholder for SQLite Registry Table."""

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 20, 24, 20)
            layout.setSpacing(14)

            header = QHBoxLayout()
            title = QLabel("Mosquito Registry & Detection History")
            title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f6fc;")
            header.addWidget(title)
            layout.addLayout(header)

            panel = QFrame()
            panel.setProperty("class", "content-panel")
            p_layout = QVBoxLayout(panel)
            p_layout.setAlignment(Qt.AlignCenter)
            lbl = QLabel("Historical Registry Explorer\n(Will query /api/registry once implemented)")
            lbl.setStyleSheet("font-size: 14px; color: #8b949e;")
            lbl.setAlignment(Qt.AlignCenter)
            p_layout.addWidget(lbl)
            layout.addWidget(panel, stretch=1)


    class AnalyticsPlaceholder(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 20, 24, 20)
            title = QLabel("Chamber Analytics & Metrics")
            title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f6fc;")
            layout.addWidget(title)

            panel = QFrame()
            panel.setProperty("class", "content-panel")
            p_layout = QVBoxLayout(panel)
            p_layout.setAlignment(Qt.AlignCenter)
            lbl = QLabel("Analytics & Visualization Charts Placeholder")
            lbl.setStyleSheet("font-size: 14px; color: #8b949e;")
            lbl.setAlignment(Qt.AlignCenter)
            p_layout.addWidget(lbl)
            layout.addWidget(panel, stretch=1)


    class CusumAlertsPage(QWidget):
        """
        CUSUM Activity Monitoring & Alerts Dashboard.
        Displays real-time statistical surveillance data, diurnal baselines,
        positive CUSUM scores (S_t^+), and historical activity alert events.
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.init_ui()

        def init_ui(self):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 20, 24, 20)
            layout.setSpacing(18)

            # Header
            header_layout = QHBoxLayout()
            title_box = QVBoxLayout()
            title_box.setSpacing(2)

            page_title = QLabel("CUSUM Activity Monitoring & Alerts")
            page_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #f0f6fc;")
            page_subtitle = QLabel("Statistical surveillance detecting sustained increases in observed chamber mosquito activity (30-min intervals)")
            page_subtitle.setStyleSheet("font-size: 12px; color: #8b949e;")

            title_box.addWidget(page_title)
            title_box.addWidget(page_subtitle)
            header_layout.addLayout(title_box)
            header_layout.addStretch()
            layout.addLayout(header_layout)

            # Grid of 4 Metric Cards
            cards_grid = QGridLayout()
            cards_grid.setHorizontalSpacing(16)
            cards_grid.setVerticalSpacing(16)

            self.card_state = MetricCard(
                title="CUSUM Surveillance State",
                initial_value="INSUFFICIENT DATA",
                subtitle="Awaiting CUSUM telemetry...",
                accent_color="#8b949e"
            )
            self.card_s_plus = MetricCard(
                title="Positive CUSUM Score (S+)",
                initial_value="S+ = N/A",
                subtitle="Threshold: h = 5.0 | Slack: k = 0.5",
                accent_color="#58a6ff"
            )
            self.card_activity = MetricCard(
                title="Current 30-min Activity (x_t)",
                initial_value="N/A",
                subtitle="Unique confirmed IDs in interval",
                accent_color="#ff9500"
            )
            self.card_baseline = MetricCard(
                title="Active Diurnal Baseline",
                initial_value="N/A",
                subtitle="Mean: N/A | Std: N/A",
                accent_color="#3fb950"
            )

            cards_grid.addWidget(self.card_state, 0, 0)
            cards_grid.addWidget(self.card_s_plus, 0, 1)
            cards_grid.addWidget(self.card_activity, 0, 2)
            cards_grid.addWidget(self.card_baseline, 0, 3)
            layout.addLayout(cards_grid)

            # Lower Panels
            lower_layout = QHBoxLayout()
            lower_layout.setSpacing(16)

            # Left Panel: Diagnostics & Parameters
            diag_panel = QFrame()
            diag_panel.setProperty("class", "content-panel")
            diag_vbox = QVBoxLayout(diag_panel)

            lbl_diag_title = QLabel("Diurnal Baselines & Decision Parameters")
            lbl_diag_title.setProperty("class", "panel-heading")
            diag_vbox.addWidget(lbl_diag_title)

            self.diag_labels = {}
            diag_rows = [
                ("baseline_status", "Baseline Status & Readiness", "N/A"),
                ("window_type", "Current Diurnal Window", "N/A"),
                ("threshold_h", "Alert Decision Threshold (h)", "5.00"),
                ("reference_k", "Reference Allowance Parameter (k)", "0.50"),
                ("min_std", "Minimum Std Floor (max(σ, 0.5))", "0.50"),
                ("last_alert_time", "Latest Alert Timestamp", "None"),
                ("last_alert_reason", "Latest Alert Description", "No active alerts"),
            ]

            for key, label_text, default_val in diag_rows:
                row = QHBoxLayout()
                lbl_name = QLabel(label_text)
                lbl_name.setStyleSheet("font-size: 13px; color: #c9d1d9;")
                lbl_val = QLabel(default_val)
                lbl_val.setStyleSheet("font-size: 12px; color: #8b949e; font-weight: 500;")
                row.addWidget(lbl_name)
                row.addStretch()
                row.addWidget(lbl_val)
                diag_vbox.addLayout(row)
                self.diag_labels[key] = lbl_val

            diag_vbox.addStretch()
            lower_layout.addWidget(diag_panel, stretch=1)

            # Right Panel: Alert History Table
            history_panel = QFrame()
            history_panel.setProperty("class", "content-panel")
            hist_vbox = QVBoxLayout(history_panel)

            lbl_hist_title = QLabel("Chamber Activity Alert History")
            lbl_hist_title.setProperty("class", "panel-heading")
            hist_vbox.addWidget(lbl_hist_title)

            self.alerts_table = QTableWidget(0, 4)
            self.alerts_table.setHorizontalHeaderLabels(["Timestamp", "Window", "Activity", "CUSUM S+"])
            self.alerts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            hist_vbox.addWidget(self.alerts_table)

            lower_layout.addWidget(history_panel, stretch=1)
            layout.addLayout(lower_layout)
            layout.addStretch()

        def update_cusum(self, cusum_data: Optional[Dict[str, Any]]):
            """Slot updating CUSUM display non-blockingly from status payload."""
            if not cusum_data or not isinstance(cusum_data, dict):
                self.card_state.update_value("INSUFFICIENT DATA", "Awaiting CUSUM telemetry...")
                self.card_state.set_accent_color("#8b949e")
                self.card_s_plus.update_value("S+ = N/A", "Threshold: h = 5.0 | Slack: k = 0.5")
                self.card_activity.update_value("N/A", "Unique confirmed IDs in interval")
                self.card_baseline.update_value("N/A", "Mean: N/A | Std: N/A")
                for k in self.diag_labels:
                    if k not in ("threshold_h", "reference_k", "min_std"):
                        self.diag_labels[k].setText("N/A")
                return

            state = str(cusum_data.get("state", "UNKNOWN")).upper()
            w_type = str(cusum_data.get("window_type", "N/A")).capitalize()
            activity = cusum_data.get("current_activity")
            s_plus = cusum_data.get("s_plus")
            thresh_h = cusum_data.get("threshold_h", 5.0)
            ref_k = cusum_data.get("reference_k", 0.5)
            b_mean = cusum_data.get("baseline_mean")
            b_std = cusum_data.get("baseline_std")
            b_n = cusum_data.get("baseline_n", 0)
            b_days = cusum_data.get("baseline_days", 0)
            b_days_req = cusum_data.get("baseline_days_required", 7)
            b_ready = bool(cusum_data.get("baseline_ready", False))
            alert_time = cusum_data.get("last_alert_time")
            alert_reason = cusum_data.get("last_alert_reason")

            # 1. Update State Card
            if state == "ALERT_TRIGGERED":
                self.card_state.update_value("ALERT TRIGGERED", f"Window: {w_type} (Sustained increase)")
                self.card_state.set_accent_color("#f85149")
            elif state == "MONITORING":
                self.card_state.update_value("MONITORING", f"Active Window: {w_type}")
                self.card_state.set_accent_color("#3fb950")
            elif state == "COLLECTING_BASELINE":
                self.card_state.update_value("COLLECTING", f"Baseline (Day {b_days}/{b_days_req}, N={b_n}/12)")
                self.card_state.set_accent_color("#e3b341")
            elif state == "OFFLINE":
                self.card_state.update_value("DETECTOR OFFLINE", "Observation collection paused")
                self.card_state.set_accent_color("#f85149")
            else:
                self.card_state.update_value("INSUFFICIENT DATA", f"Window: {w_type}")
                self.card_state.set_accent_color("#8b949e")

            # 2. Update CUSUM Score Card
            if s_plus is not None and isinstance(s_plus, (int, float)):
                self.card_s_plus.update_value(
                    f"S+ = {s_plus:.2f}",
                    f"Threshold: h = {thresh_h:.1f} | Slack: k = {ref_k:.1f}"
                )
                if s_plus >= thresh_h:
                    self.card_s_plus.set_accent_color("#f85149")
                else:
                    self.card_s_plus.set_accent_color("#58a6ff")
            else:
                self.card_s_plus.update_value("S+ = N/A", f"Threshold: h = {thresh_h:.1f} | Slack: k = {ref_k:.1f}")
                self.card_s_plus.set_accent_color("#58a6ff")

            # 3. Update Activity Card
            if activity is not None and isinstance(activity, (int, float)):
                self.card_activity.update_value(f"{int(activity)} IDs", f"Unique IDs in 30-min window")
            else:
                self.card_activity.update_value("N/A", "No valid interval observation")

            # 4. Update Baseline Card
            if b_mean is not None and b_std is not None and isinstance(b_mean, (int, float)):
                self.card_baseline.update_value(
                    f"μ₀: {b_mean:.2f} | σ₀: {b_std:.2f}",
                    f"N = {b_n} valid intervals ({'Ready' if b_ready else 'Collecting'})"
                )
            else:
                self.card_baseline.update_value("N/A", f"Collecting (Day {b_days}/{b_days_req}, N={b_n}/12)")

            # 5. Update Diagnostic Rows
            self.diag_labels["baseline_status"].setText(
                f"READY (7 Days, N = {b_n})" if b_ready else f"COLLECTING (Day {b_days}/{b_days_req}, N = {b_n}/12)"
            )
            self.diag_labels["baseline_status"].setStyleSheet("color: #3fb950;" if b_ready else "color: #e3b341;")
            self.diag_labels["window_type"].setText(w_type)
            self.diag_labels["threshold_h"].setText(f"{thresh_h:.2f}")
            self.diag_labels["reference_k"].setText(f"{ref_k:.2f}")
            self.diag_labels["last_alert_time"].setText(alert_time if alert_time else "None")
            self.diag_labels["last_alert_reason"].setText(alert_reason[:45] if alert_reason else "No active alerts")

            # 6. Update Alert History Table (deduplicated, bounded)
            alert_history = cusum_data.get("alert_history", []) if isinstance(cusum_data, dict) else []
            if isinstance(alert_history, list) and len(alert_history) > 0:
                valid_alerts = [r for r in alert_history if isinstance(r, dict)]
                self.alerts_table.setRowCount(len(valid_alerts))
                for row_idx, rec in enumerate(valid_alerts):
                    ts = str(rec.get("timestamp", "N/A"))
                    w_type_rec = str(rec.get("window_type", "N/A")).capitalize()
                    act_val = rec.get("activity_value")
                    act_str = f"{int(act_val)} IDs" if isinstance(act_val, (int, float)) else str(act_val or "N/A")
                    s_val = rec.get("s_plus")
                    s_str = f"{float(s_val):.2f}" if isinstance(s_val, (int, float)) else str(s_val or "N/A")

                    for col_idx, text in enumerate([ts, w_type_rec, act_str, s_str]):
                        item = QTableWidgetItem(text)
                        item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                        self.alerts_table.setItem(row_idx, col_idx, item)
            else:
                self.alerts_table.setRowCount(0)

        def set_offline_state(self, err_type: str, description: str):
            """Called when polling fails or detector goes offline."""
            self.card_state.update_value("DETECTOR OFFLINE", f"[{err_type.upper()}] Telemetry unavailable")
            self.card_state.set_accent_color("#f85149")
            self.card_s_plus.update_value("S+ = N/A", "Detector connection offline")
            self.card_activity.update_value("N/A", "Cannot retrieve activity observation")
            self.card_baseline.update_value("N/A", "Baseline monitoring paused")
            self.diag_labels["baseline_status"].setText("OFFLINE")
            self.diag_labels["baseline_status"].setStyleSheet("color: #f85149;")
            self.diag_labels["window_type"].setText("N/A")

            # Retain existing table if populated, or set informative single row if empty
            if self.alerts_table.rowCount() == 0:
                self.alerts_table.setRowCount(1)
                err_item = QTableWidgetItem(f"[{err_type.upper()}] {description}")
                err_item.setFlags(err_item.flags() ^ Qt.ItemIsEditable)
                self.alerts_table.setItem(0, 0, err_item)



    class SettingsPlaceholder(QWidget):
        def __init__(self, config: Dict[str, Any], parent=None):
            super().__init__(parent)
            self.config = config
            layout = QVBoxLayout(self)
            layout.setContentsMargins(24, 20, 24, 20)
            title = QLabel("Client & Chamber Settings")
            title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f6fc;")
            layout.addWidget(title)

            panel = QFrame()
            panel.setProperty("class", "content-panel")
            p_layout = QVBoxLayout(panel)
            token_masked = "***" if self.config.get("token") else "(None configured)"
            lbl = QLabel(
                "Client Network Configuration (Loaded from client_config.json):\n\n"
                f"• Target Pi Host: {self.config.get('pi_host', '127.0.0.1')}\n"
                f"• Target Pi Port: {self.config.get('pi_port', 8080)}\n"
                f"• Token Status: {token_masked}\n"
                f"• Polling Interval: {self.config.get('poll_interval_sec', 1.0):.1f} s\n"
                f"• Request Timeout: {self.config.get('timeout_sec', 2.5):.1f} s\n\n"
                "To adjust these settings, edit your local client_config.json file (untracked in Git)."
            )
            lbl.setStyleSheet("font-size: 13px; color: #c9d1d9; line-height: 1.6;")
            p_layout.addWidget(lbl)
            p_layout.addStretch()
            layout.addWidget(panel, stretch=1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Application Window
# ─────────────────────────────────────────────────────────────────────────────
if PYSIDE6_AVAILABLE:
    class MosquitoSentinelWindow(QMainWindow):
        """Main application window with navigation sidebar and telemetry integration."""

        def __init__(self, client_cfg: Optional[Dict[str, Any]] = None):
            super().__init__()
            self.client_config = client_cfg if client_cfg is not None else load_client_config()

            self.setWindowTitle("MosquitoSentinel - Chamber Surveillance & Classification")
            self.setMinimumSize(960, 620)
            self.resize(1220, 760)

            # Central Widget & Root Layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            root_layout = QHBoxLayout(central_widget)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)

            # Left Sidebar
            self.sidebar = QWidget()
            self.sidebar.setObjectName("sidebar")
            self.sidebar.setFixedWidth(240)
            sidebar_layout = QVBoxLayout(self.sidebar)
            sidebar_layout.setContentsMargins(16, 20, 16, 16)
            sidebar_layout.setSpacing(8)

            # Brand Header
            lbl_brand = QLabel("MosquitoSentinel")
            lbl_brand.setObjectName("brandTitle")
            lbl_sub = QLabel("Chamber Surveillance System")
            lbl_sub.setObjectName("brandSubtitle")
            sidebar_layout.addWidget(lbl_brand)
            sidebar_layout.addWidget(lbl_sub)

            # Separator Line
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #30363d; margin-bottom: 12px;")
            sidebar_layout.addWidget(sep)

            # Navigation Buttons
            self.nav_buttons = []
            nav_items = [
                ("Overview", 0),
                ("Live Detection", 1),
                ("Detection History", 2),
                ("Analytics", 3),
                ("Alerts", 4),
                ("Settings", 5),
            ]

            for text, idx in nav_items:
                btn = QPushButton(text)
                btn.setProperty("class", "nav-btn")
                btn.setCheckable(True)
                btn.clicked.connect(lambda checked, i=idx: self.switch_page(i))
                sidebar_layout.addWidget(btn)
                self.nav_buttons.append(btn)

            sidebar_layout.addStretch()

            # Sidebar Footer / Connection Status Badge
            footer_frame = QFrame()
            footer_frame.setStyleSheet("background: #0d1117; border-radius: 6px; padding: 10px; border: 1px solid #30363d;")
            footer_layout = QVBoxLayout(footer_frame)
            footer_layout.setContentsMargins(8, 8, 8, 8)
            footer_layout.setSpacing(4)

            self.lbl_conn_stat = QLabel("Status: Connecting...")
            self.lbl_conn_stat.setStyleSheet("font-size: 11px; font-weight: 600; color: #58a6ff;")

            self.lbl_server_info = QLabel(f"Server: {self.client_config['pi_host']}:{self.client_config['pi_port']}")
            self.lbl_server_info.setStyleSheet("font-size: 10px; color: #8b949e;")

            lbl_stage = QLabel("Stage 5A: Telemetry Polling")
            lbl_stage.setStyleSheet("font-size: 10px; color: #3fb950; font-weight: 500;")

            footer_layout.addWidget(self.lbl_conn_stat)
            footer_layout.addWidget(self.lbl_server_info)
            footer_layout.addWidget(lbl_stage)
            sidebar_layout.addWidget(footer_frame)

            root_layout.addWidget(self.sidebar)

            # Right Content Stack
            self.stack = QStackedWidget()
            self.overview_page = OverviewPage()
            self.live_page = LiveDetectionPlaceholder()
            self.history_page = HistoryPlaceholder()
            self.analytics_page = AnalyticsPlaceholder()
            self.alerts_page = CusumAlertsPage()
            self.settings_page = SettingsPlaceholder(self.client_config)

            self.stack.addWidget(self.overview_page)
            self.stack.addWidget(self.live_page)
            self.stack.addWidget(self.history_page)
            self.stack.addWidget(self.analytics_page)
            self.stack.addWidget(self.alerts_page)
            self.stack.addWidget(self.settings_page)

            root_layout.addWidget(self.stack, stretch=1)

            # Activate First Page
            self.switch_page(0)

            # Start Background Status Poller
            self.polling_worker = StatusPollingWorker(self.client_config)
            self.polling_worker.sig_status_updated.connect(self.on_status_updated)
            self.polling_worker.sig_connection_error.connect(self.on_connection_error)
            self.polling_worker.start()

        def switch_page(self, index: int):
            self.stack.setCurrentIndex(index)
            for idx, btn in enumerate(self.nav_buttons):
                btn.setChecked(idx == index)

        @Slot(dict)
        def on_status_updated(self, data: Dict[str, Any]):
            """Dispatches valid telemetry updates to OverviewPage, CusumAlertsPage, and status badges."""
            self.overview_page.update_telemetry(data)
            self.alerts_page.update_cusum(data.get("cusum"))
            host = self.client_config['pi_host']
            port = self.client_config['pi_port']
            self.lbl_conn_stat.setText(f"Status: Connected ({host}:{port})")
            self.lbl_conn_stat.setStyleSheet("font-size: 11px; font-weight: 600; color: #3fb950;")

        @Slot(str, str)
        def on_connection_error(self, err_type: str, description: str):
            """Dispatches error states to OverviewPage, CusumAlertsPage, and status badges."""
            self.overview_page.set_error_state(err_type, description)
            self.alerts_page.set_offline_state(err_type, description)
            if err_type == "unauthorized":
                self.lbl_conn_stat.setText("Status: Unauthorized (401)")
                self.lbl_conn_stat.setStyleSheet("font-size: 11px; font-weight: 600; color: #e3b341;")
            elif err_type in ("connection_refused", "timeout"):
                self.lbl_conn_stat.setText("Status: Offline (Disconnected)")
                self.lbl_conn_stat.setStyleSheet("font-size: 11px; font-weight: 600; color: #f85149;")
            elif err_type == "malformed_json":
                self.lbl_conn_stat.setText("Status: Error (Malformed JSON)")
                self.lbl_conn_stat.setStyleSheet("font-size: 11px; font-weight: 600; color: #f85149;")
            else:
                self.lbl_conn_stat.setText("Status: Connection Error")
                self.lbl_conn_stat.setStyleSheet("font-size: 11px; font-weight: 600; color: #f85149;")

        def closeEvent(self, event):
            """Clean shutdown hook: stops background polling thread safely."""
            if hasattr(self, 'polling_worker') and self.polling_worker is not None:
                self.polling_worker.stop()
                self.polling_worker.wait(3000)
            event.accept()


def main():
    if not PYSIDE6_AVAILABLE:
        print("[ERROR] PySide6 is not installed in the current Python environment.")
        print("Please install it with: pip install -r requirements-gui.txt")
        sys.exit(1)

    print("[INFO] Launching MosquitoSentinel Desktop GUI (Stage 5A)...")
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE_SHEET)
    window = MosquitoSentinelWindow()

    # Center window within primary screen's available geometry
    screen = app.primaryScreen()
    if screen is not None:
        screen_geo = screen.availableGeometry()
        geo = window.frameGeometry()
        geo.moveCenter(screen_geo.center())
        window.move(geo.topLeft())

    # Request foreground display and focus
    window.show()
    window.raise_()
    window.activateWindow()

    print("[INFO] MosquitoSentinel Window initialized successfully.")
    print("       Note: If Windows focus-stealing prevention places the window behind")
    print("       your terminal/IDE, click the MosquitoSentinel icon on your taskbar")
    print("       or press Alt+Tab to bring it forward.")
    print("       Press Ctrl+C in this terminal or close the window to exit.")

    # Periodic timer to allow Python interpreter to process Ctrl+C signals
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(500)

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n[INFO] Application closed cleanly by user (Ctrl+C).")
        sys.exit(0)


if __name__ == "__main__":
    main()
