"""
gui_app.py - MosquitoSentinel Desktop GUI (Stage 5A: Client Integration)
========================================================================
Light "frosted glass" medical-monitor redesign.

What changed vs. the dark version
- Light theme with a soft teal -> lavender gradient background.
- Translucent white "glass" cards and a floating glass sidebar.
- Status is shown with a banner + colored values (red = problem, green = ok).
- Empty tables show a friendly empty state instead of an error row.
- Polling worker now wakes instantly on "Retry connection".

Everything networking/telemetry related (client_config.json, X-Sentinel-Token,
GET /api/status, N/A for null fields, clean shutdown) behaves as before.

Note on "blur": Qt widgets cannot blur what is behind them, so the glass
effect is faked: the background is painted as a gradient with soft blurred
color blobs, and cards are semi-transparent white on top of it.
"""

import sys
import os
import json
import math
import time
import socket
import threading
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

try:
    from PySide6.QtCore import Qt, QTimer, QSize, QThread, Signal, Slot, QByteArray, QPointF, QRectF, QRect, QUrl, QDate
    from PySide6.QtGui import (QColor, QIcon, QPainter, QPen, QBrush, QPainterPath, QPixmap, QImage, QLinearGradient,
                               QRadialGradient, QDesktopServices)
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QStackedWidget, QPushButton, QLabel, QFrame, QTableWidget, QTableWidgetItem,
        QHeaderView, QComboBox, QScrollArea, QDateEdit, QAbstractItemView, QGraphicsDropShadowEffect,
        QDialog, QFileDialog, QMessageBox,
    )
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

try:
    from PySide6.QtSvg import QSvgRenderer
    SVG_AVAILABLE = True
except ImportError:
    SVG_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 0. Palette (light mode)
# ─────────────────────────────────────────────────────────────────────────────
INK = "#12282F"       # main text
SUB = "#4A6168"       # secondary text
MUTED = "#5E7379"     # "no data" values
TEAL = "#0E7C86"      # brand / primary
OK = "#1F8A4C"
WARN = "#B7791F"
BAD = "#B42318"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Client Configuration Loader (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_config.json")


def load_client_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads client networking configuration from client_config.json or environment."""
    cfg = {
        "pi_host": os.environ.get("SENTINEL_PI_HOST", "127.0.0.1"),
        "pi_port": int(os.environ.get("SENTINEL_PI_PORT", 8080)),
        "token": os.environ.get("SENTINEL_AUTH_TOKEN", ""),
        "poll_interval_sec": float(os.environ.get("SENTINEL_POLL_INTERVAL", 1.0)),
        "timeout_sec": 2.5,
    }
    if config_path is None:
        config_path = default_config_path()

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
        """Polls GET /api/status periodically; emits signals so the UI never blocks."""
        sig_status_updated = Signal(dict)
        sig_connection_error = Signal(str, str)  # (error_type, description)

        def __init__(self, config: Dict[str, Any], parent=None):
            super().__init__(parent)
            self.config = config
            self._running = True
            self._wake = threading.Event()

        def stop(self):
            self._running = False
            self._wake.set()

        def poll_now(self):
            """Skip the remaining wait and poll immediately (Retry button)."""
            self._wake.set()

        def run(self):
            url = f"http://{self.config['pi_host']}:{self.config['pi_port']}/api/status"
            token = self.config.get("token", "")
            timeout = self.config.get("timeout_sec", 2.5)
            interval = self.config.get("poll_interval_sec", 1.0)
            host_port = f"{self.config['pi_host']}:{self.config['pi_port']}"

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
                    low = reason_str.lower()
                    if "timed out" in low or "timeout" in low:
                        self.sig_connection_error.emit("timeout", f"Connection timed out: {host_port}")
                    elif "connectionrefused" in low or "actively refused" in low or "refused" in low:
                        self.sig_connection_error.emit("connection_refused", f"Cannot connect to {host_port}")
                    else:
                        self.sig_connection_error.emit("network_error", reason_str)
                except (TimeoutError, socket.timeout):
                    self.sig_connection_error.emit("timeout", f"Connection timed out: {host_port}")
                except Exception as ex:
                    self.sig_connection_error.emit("unknown_error", str(ex))

                # Wait for the next poll, but wake instantly on stop() / poll_now()
                self._wake.wait(interval)
                self._wake.clear()

    class FrameFetchWorker(QThread):
        """Fetches /api/frame.jpg with low latency; emits QImage and diagnostics without touching Qt widgets."""
        sig_frame_received = Signal(QImage, float)   # (image, fetch_duration_sec)
        sig_diagnostics_updated = Signal(dict)       # (diagnostics_dict)
        sig_stale_dropped = Signal(int)              # (dropped_seq)
        sig_frame_error = Signal(str, str)           # (error_type, description)

        def __init__(self, config: Dict[str, Any], parent=None):
            super().__init__(parent)
            self.config = config
            self._running = True
            self._paused = True
            self._wake = threading.Event()
            self._last_emitted_seq: int = -1
            self._last_arrival_time: float = 0.0
            self._dropped_stale_count: int = 0

        def pause(self):
            self._paused = True

        def resume(self):
            self._paused = False
            self._wake.set()

        def stop(self):
            self._running = False
            self._paused = False
            self._wake.set()

        def fetch_now(self):
            self._wake.set()

        def run(self):
            host = self.config.get("pi_host", "127.0.0.1")
            port = self.config.get("pi_port", 8080)
            token = self.config.get("token", "")
            timeout = float(self.config.get("timeout_sec", 2.0))
            # Responsive target interval: default 0.05s (~20 FPS) for smooth live preview
            interval = float(self.config.get("preview_interval_sec", 0.05))
            w = int(self.config.get("preview_width", 0))
            h = int(self.config.get("preview_height", 0))
            if w > 0 and h > 0:
                url = f"http://{host}:{port}/api/frame.jpg?width={w}&height={h}"
            else:
                url = f"http://{host}:{port}/api/frame.jpg"
            host_port = f"{host}:{port}"

            while self._running:
                if self._paused:
                    self._wake.wait(0.5)
                    self._wake.clear()
                    continue

                t0 = time.perf_counter()
                try:
                    req = urllib.request.Request(url, method="GET")
                    if token:
                        req.add_header("X-Sentinel-Token", token)
                    req.add_header("Cache-Control", "no-cache, no-store")
                    req.add_header("Pragma", "no-cache")

                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        status_code = resp.status
                        if status_code == 200:
                            raw_bytes = resp.read()
                            t_dl = time.perf_counter()
                            dl_ms = (t_dl - t0) * 1000.0

                            # Parse server diagnostics headers if provided
                            server_ts_hdr = resp.headers.get("X-Frame-Timestamp")
                            frame_seq_hdr = resp.headers.get("X-Frame-Index")
                            det_fps_hdr = resp.headers.get("X-Detector-FPS")

                            server_ts = float(server_ts_hdr) if server_ts_hdr else None
                            frame_seq = int(frame_seq_hdr) if frame_seq_hdr and frame_seq_hdr.isdigit() else None
                            det_fps = float(det_fps_hdr) if det_fps_hdr else None

                            t_dec_start = time.perf_counter()
                            image = QImage()
                            if image.loadFromData(raw_bytes) and not image.isNull():
                                t_dec_end = time.perf_counter()
                                dec_ms = (t_dec_end - t_dec_start) * 1000.0
                                total_fetch_ms = (t_dec_end - t0) * 1000.0
                                frame_age_ms = (time.time() - server_ts) * 1000.0 if server_ts is not None else None

                                # Discard stale / duplicate buffered frames (Requirement 2 & 3)
                                if frame_seq is not None and frame_seq <= self._last_emitted_seq:
                                    self._dropped_stale_count += 1
                                    self.sig_stale_dropped.emit(frame_seq)
                                    continue

                                if frame_seq is not None:
                                    self._last_emitted_seq = frame_seq

                                now_arr = time.time()
                                intv_ms = (now_arr - self._last_arrival_time) * 1000.0 if self._last_arrival_time > 0 else 0.0
                                self._last_arrival_time = now_arr

                                diag = {
                                    "download_ms": dl_ms,
                                    "decode_ms": dec_ms,
                                    "total_fetch_ms": total_fetch_ms,
                                    "frame_age_ms": frame_age_ms,
                                    "interval_ms": intv_ms,
                                    "frame_seq": frame_seq,
                                    "detector_fps": det_fps,
                                    "dropped_stale": self._dropped_stale_count,
                                    "width": image.width(),
                                    "height": image.height(),
                                }
                                self.sig_frame_received.emit(image, total_fetch_ms / 1000.0)
                                self.sig_diagnostics_updated.emit(diag)
                            else:
                                self.sig_frame_error.emit("decode_error", "Received picture failed to decode (invalid JPEG data)")
                        else:
                            self.sig_frame_error.emit("http_error", f"HTTP {status_code}")

                except urllib.error.HTTPError as he:
                    if he.code == 401:
                        self.sig_frame_error.emit("unauthorized", "Unauthorized: X-Sentinel-Token invalid or missing (401)")
                    elif he.code == 404:
                        self.sig_frame_error.emit("not_found", "Endpoint /api/frame.jpg not found (404)")
                    elif he.code == 503:
                        self.sig_frame_error.emit("no_frame", "Waiting for first frame from camera (503)")
                    else:
                        self.sig_frame_error.emit("http_error", f"HTTP {he.code}: {he.reason}")
                except urllib.error.URLError as ue:
                    reason_str = str(ue.reason)
                    low = reason_str.lower()
                    if "timed out" in low or "timeout" in low:
                        self.sig_frame_error.emit("timeout", f"Connection timed out: {host_port}")
                    elif "connectionrefused" in low or "actively refused" in low or "refused" in low:
                        self.sig_frame_error.emit("connection_refused", f"Cannot connect to {host_port}")
                    else:
                        self.sig_frame_error.emit("network_error", reason_str)
                except (TimeoutError, socket.timeout):
                    self.sig_frame_error.emit("timeout", f"Connection timed out: {host_port}")
                except Exception as ex:
                    self.sig_frame_error.emit("unknown_error", str(ex))

                if self._running and not self._paused:
                    elapsed = time.perf_counter() - t0
                    wait_time = max(0.005, interval - elapsed)
                    self._wake.wait(wait_time)
                    self._wake.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Analytics data store (pure Python, no Qt, read-only)
# ─────────────────────────────────────────────────────────────────────────────
SLOT = timedelta(minutes=30)
UNCONFIRMED_LABELS = {"", "unknown", "unconfirmed", "n/a", "na", "none", "null", "-", "—"}
CONF_BINS = [("0.00–0.49", 0.0, 0.5), ("0.50–0.69", 0.5, 0.7), ("0.70–0.89", 0.7, 0.9), ("0.90–1.00", 0.9, 1.0001)]


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def normalize_species(v) -> Optional[str]:
    """Returns the recorded species label, or None when it is missing/unconfirmed."""
    if v is None:
        return None
    s = str(v).strip()
    return None if (s.lower() in UNCONFIRMED_LABELS or "unconfirmed" in s.lower() or "unknown" in s.lower()) else s


def _to_naive_dt(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return datetime.now()
    if dt.tzinfo is not None:
        return dt.astimezone().replace(tzinfo=None)
    return dt


def floor_slot(dt: datetime) -> datetime:
    return dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)


def parse_ts(v) -> Optional[datetime]:
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        dt = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


class AnalyticsStore:
    """
    Collects analytics from the payloads of the EXISTING GET /api/status poll only.
    It never reads or writes the Pi's database/CSV and never recomputes CUSUM.

    - A mosquito ID is counted once per 30-min interval, and once per selected period.
    - A 30-min interval is "covered" once at least one successful poll happened in it.
      Uncovered intervals are unknown (None), never zero.
    - Species/confidence per ID: the most recent recorded value wins.
    """
    MAX_AGE = timedelta(days=7, hours=1)
    CUSUM_MIN_GAP = timedelta(seconds=20)

    def __init__(self, now: Optional[datetime] = None):
        self.started = _to_naive_dt(now)
        self.slot_ids: Dict[datetime, set] = {}
        self.covered: set = set()
        self.slot_window: Dict[datetime, str] = {}
        self.species: Dict[str, Optional[str]] = {}
        self.conf: Dict[str, float] = {}
        self.cusum_samples = []            # [(datetime, s_plus)]
        self.alerts: Dict[str, dict] = {}  # from cusum.alert_history
        self.threshold: Optional[float] = None
        self.cusum_state: Optional[str] = None
        self.cusum_seen = False            # payload ever contained a cusum object
        self.version = 0

    # ── ingest ──────────────────────────────────────────────────────────────
    def record(self, data: Dict[str, Any], now: Optional[datetime] = None):
        if not isinstance(data, dict):
            return
        now = _to_naive_dt(now)
        slot = floor_slot(now)
        changed = slot not in self.covered
        self.covered.add(slot)
        ids = self.slot_ids.setdefault(slot, set())

        tracks = data.get("active_tracks_summary")
        if isinstance(tracks, list):
            for t in tracks:
                if not isinstance(t, dict) or t.get("mos_id") is None:
                    continue
                mid = str(t["mos_id"])
                if mid not in ids:
                    ids.add(mid)
                    changed = True
                sp = normalize_species(t.get("species"))
                if sp is not None:
                    if self.species.get(mid) != sp:
                        self.species[mid] = sp
                        changed = True
                elif mid not in self.species:
                    self.species[mid] = None
                    changed = True
                c = t.get("confidence")
                if _is_num(c) and 0.0 <= c <= 1.0 and self.conf.get(mid) != float(c):
                    self.conf[mid] = float(c)
                    changed = True

        active = data.get("active_track_ids")
        if isinstance(active, list):
            for mid in active:
                if mid is None:
                    continue
                mid = str(mid)
                if mid not in ids:
                    ids.add(mid)
                    changed = True
                if mid not in self.species:
                    self.species[mid] = None

        cusum = data.get("cusum")
        if isinstance(cusum, dict):
            self.cusum_seen = True
            wt = cusum.get("window_type")
            if isinstance(wt, str) and wt.strip().lower() not in UNCONFIRMED_LABELS:
                if self.slot_window.get(slot) != wt.strip():
                    self.slot_window[slot] = wt.strip()
                    changed = True
            state = cusum.get("state")
            if isinstance(state, str) and state != self.cusum_state:
                self.cusum_state = state
                changed = True
            h = cusum.get("threshold_h")
            if _is_num(h) and h != self.threshold:
                self.threshold = float(h)
                changed = True
            s = cusum.get("s_plus")
            if _is_num(s) and (not self.cusum_samples or now - self.cusum_samples[-1][0] >= self.CUSUM_MIN_GAP):
                self.cusum_samples.append((now, float(s)))
                changed = True
            hist = cusum.get("alert_history")
            if isinstance(hist, list):
                for rec in hist:
                    if not isinstance(rec, dict):
                        continue
                    key = f"{rec.get('timestamp')}|{rec.get('window_type')}"
                    if key not in self.alerts:
                        sv = rec.get("s_plus")
                        self.alerts[key] = {"time": parse_ts(rec.get("timestamp")),
                                            "s_plus": float(sv) if _is_num(sv) else None,
                                            "window": rec.get("window_type")}
                        changed = True

        cutoff = now - self.MAX_AGE
        for old in [s for s in self.slot_ids if s < cutoff]:
            self.slot_ids.pop(old, None)
            self.covered.discard(old)
            self.slot_window.pop(old, None)
        while self.cusum_samples and self.cusum_samples[0][0] < cutoff:
            self.cusum_samples.pop(0)
        for k in [k for k, a in self.alerts.items() if a["time"] is not None and a["time"] < cutoff]:
            self.alerts.pop(k, None)
        if changed:
            self.version += 1

    # ── queries ─────────────────────────────────────────────────────────────
    def range_slots(self, hours: float, now: Optional[datetime] = None):
        end = floor_slot(_to_naive_dt(now))
        n = max(1, int(hours * 2))
        return [end - SLOT * (n - 1 - i) for i in range(n)]

    def series(self, hours: float, now: Optional[datetime] = None):
        """[(slot_start, unique_id_count or None if not observed)]"""
        return [(s, len(self.slot_ids.get(s, ())) if s in self.covered else None)
                for s in self.range_slots(hours, now)]

    def unique_ids(self, hours: float, now: Optional[datetime] = None) -> set:
        out = set()
        for s in self.range_slots(hours, now):
            if s in self.covered:
                out |= self.slot_ids.get(s, set())
        return out

    def species_breakdown(self, hours: float, now: Optional[datetime] = None, max_named: int = 5):
        """[(label, count, kind)] kind: 'named' | 'other' | 'unconfirmed'. Counts sum to the unique-ID total."""
        confirmed: Dict[str, int] = {}
        unconfirmed = 0
        for mid in self.unique_ids(hours, now):
            sp = self.species.get(mid)
            if sp is None:
                unconfirmed += 1
            else:
                confirmed[sp] = confirmed.get(sp, 0) + 1
        ranked = sorted(confirmed.items(), key=lambda kv: (-kv[1], kv[0]))
        out = [(n, c, "named") for n, c in ranked[:max_named]]
        rest = sum(c for _, c in ranked[max_named:])
        if rest:
            out.append(("Other species", rest, "other"))
        if unconfirmed:
            out.append(("Species unconfirmed", unconfirmed, "unconfirmed"))
        return out

    def dominant_species(self, hours: float, now: Optional[datetime] = None):
        """(label, count, total_ids) of the most frequent CONFIRMED species, or None."""
        ids = self.unique_ids(hours, now)
        counts: Dict[str, int] = {}
        for mid in ids:
            sp = self.species.get(mid)
            if sp is not None:
                counts[sp] = counts.get(sp, 0) + 1
        if not counts:
            return None
        label, n = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return label, n, len(ids)

    def peak(self, hours: float, now: Optional[datetime] = None):
        best = None
        for slot, v in self.series(hours, now):
            if v and (best is None or v > best[1]):
                best = (slot, v)
        return best

    def avg_confidence(self, hours: float, now: Optional[datetime] = None):
        """(mean 0-1, ids_with_score, total_ids) or None if no valid score."""
        ids = self.unique_ids(hours, now)
        vals = [self.conf[m] for m in ids if m in self.conf]
        return (sum(vals) / len(vals), len(vals), len(ids)) if vals else None

    def confidence_bins(self, hours: float, now: Optional[datetime] = None):
        vals = [self.conf[m] for m in self.unique_ids(hours, now) if m in self.conf]
        if not vals:
            return None
        return [(label, sum(1 for v in vals if lo <= v < hi)) for label, lo, hi in CONF_BINS]

    def window_counts(self, hours: float, now: Optional[datetime] = None):
        """Unique IDs per monitoring window as REPORTED by the detector (cusum.window_type)."""
        groups: Dict[str, set] = {}
        for s in self.range_slots(hours, now):
            wt = self.slot_window.get(s)
            if wt and s in self.covered:
                groups.setdefault(wt, set()).update(self.slot_ids.get(s, set()))
        return [(w, len(i)) for w, i in groups.items()] if groups else None

    def cusum_points(self, hours: float, now: Optional[datetime] = None):
        now = _to_naive_dt(now)
        cutoff = now - timedelta(hours=hours)
        pts = [(t, v) for t, v in self.cusum_samples if t >= cutoff]
        alerts = [a for a in self.alerts.values() if a["time"] is not None and cutoff <= a["time"] <= now]
        return pts, alerts


# ─────────────────────────────────────────────────────────────────────────────
# 3. Light Glass Stylesheet (QSS)
# ─────────────────────────────────────────────────────────────────────────────
LIGHT_STYLE_SHEET = f"""
QWidget {{
    color: {INK};
    font-family: "Plus Jakarta Sans", "Segoe UI", "Inter", "Noto Sans", Helvetica, Arial, sans-serif;
    font-size: 13px;
}}
QLabel {{ background: transparent; }}
QToolTip {{ background: #FFFFFF; color: {INK}; border: 1px solid #C9D6D8; padding: 4px; }}

/* Glass surfaces */
QFrame[class="glass"] {{
    background-color: rgba(255, 255, 255, 140);
    border: 1px solid rgba(255, 255, 255, 190);
    border-radius: 20px;
}}
QFrame[class="banner"] {{
    background-color: rgba(254, 236, 234, 215);
    border: 1px solid rgba(253, 162, 155, 255);
    border-radius: 18px;
}}
QFrame[class="inner"] {{
    background-color: rgba(255, 255, 255, 150);
    border: 1px solid rgba(18, 40, 47, 30);
    border-radius: 12px;
}}
QFrame[class="sep"] {{ background-color: rgba(18, 40, 47, 20); border: none; }}

/* Sidebar navigation */
QPushButton[class="nav"] {{
    text-align: left; padding: 0 14px; height: 46px; border: none; border-radius: 12px;
    background: transparent; color: #35505A; font-size: 14px; font-weight: 500;
}}
QPushButton[class="nav"]:hover {{ background-color: rgba(255, 255, 255, 160); }}
QPushButton[class="nav"]:checked {{ background-color: {TEAL}; color: #FFFFFF; font-weight: 600; }}

/* Buttons */
QPushButton[class="primary"] {{
    height: 40px; padding: 0 20px; border: none; border-radius: 12px;
    background-color: {TEAL}; color: #FFFFFF; font-weight: 600; font-size: 13px;
}}
QPushButton[class="primary"]:hover {{ background-color: #0A6A73; }}
QPushButton[class="danger"] {{
    height: 40px; padding: 0 20px; border: none; border-radius: 12px;
    background-color: {BAD}; color: #FFFFFF; font-weight: 600; font-size: 13px;
}}
QPushButton[class="danger"]:hover {{ background-color: #912018; }}
QPushButton[class="ghost"] {{
    height: 40px; padding: 0 18px; border: 1px solid rgba(18, 40, 47, 45); border-radius: 12px;
    background-color: rgba(255, 255, 255, 150); color: {INK}; font-weight: 600;
}}
QPushButton[class="ghost"]:hover {{ background-color: rgba(255, 255, 255, 220); }}
QPushButton[class="ghost"]:disabled {{ color: #8A9CA2; background-color: rgba(255, 255, 255, 90); }}

/* Inputs */
QDateEdit, QComboBox, QLineEdit {{
    height: 38px; padding: 0 12px; border-radius: 10px;
    background-color: rgba(255, 255, 255, 190); border: 1px solid rgba(18, 40, 47, 35);
}}
QDateEdit::drop-down {{ border: none; width: 26px; }}
QCalendarWidget QWidget {{ background: #FFFFFF; }}
QComboBox QAbstractItemView {{ background: #FFFFFF; selection-background-color: #DCEFF2; selection-color: {INK}; }}

/* Tables */
QTableWidget {{
    background: transparent; border: none; gridline-color: transparent; color: {INK};
}}
QTableWidget::item {{ padding: 8px; border-bottom: 1px solid rgba(18, 40, 47, 20); }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background-color: rgba(18, 40, 47, 18); color: #3B5159; padding: 10px 8px;
    border: none; font-weight: 600; font-size: 11px;
}}
QScrollBar:vertical {{ border: none; background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: rgba(18, 40, 47, 60); min-height: 20px; border-radius: 4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4. UI helpers & reusable widgets
# ─────────────────────────────────────────────────────────────────────────────
if PYSIDE6_AVAILABLE:
    def lbl(text="", size=13, weight=500, color=INK, wrap=False) -> QLabel:
        w = QLabel(text)
        w.setStyleSheet(f"font-size:{size}px; font-weight:{weight}; color:{color};")
        w.setWordWrap(wrap)
        return w

    def pill(text, bg, fg) -> QLabel:
        w = QLabel(text)
        w.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:12px; padding:5px 12px; font-size:12px; font-weight:600;")
        w.setAlignment(Qt.AlignCenter)
        return w

    def glass(radius_blur: int = 30) -> QFrame:
        f = QFrame()
        f.setProperty("class", "glass")
        shadow = QGraphicsDropShadowEffect(f)
        shadow.setBlurRadius(radius_blur)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(40, 80, 110, 28))
        f.setGraphicsEffect(shadow)
        return f

    def separator() -> QFrame:
        s = QFrame()
        s.setProperty("class", "sep")
        s.setFixedHeight(1)
        return s

    def add_info_rows(layout: QVBoxLayout, rows) -> Dict[str, QLabel]:
        """rows: [(key, label, default)] -> {key: value QLabel}. Rows split by hairlines."""
        out = {}
        for i, (key, text, default) in enumerate(rows):
            if i:
                layout.addWidget(separator())
            row = QHBoxLayout()
            row.setContentsMargins(0, 11, 0, 11)
            value = lbl(default, 13, 600, MUTED)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(lbl(text, 13, 500, INK))
            row.addStretch()
            row.addWidget(value)
            layout.addLayout(row)
            out[key] = value
        return out

    def set_value(label: QLabel, text: str, color: str = MUTED):
        label.setText(text)
        label.setStyleSheet(f"font-size:13px; font-weight:600; color:{color};")

    def page_header(title: str, subtitle: str, right: Optional[QWidget] = None) -> QHBoxLayout:
        h = QHBoxLayout()
        box = QVBoxLayout()
        box.setSpacing(3)
        box.addWidget(lbl(title, 26, 700, INK))
        box.addWidget(lbl(subtitle, 13, 400, SUB))
        h.addLayout(box)
        h.addStretch()
        if right is not None:
            h.addWidget(right, alignment=Qt.AlignTop)
        return h

    def svg_icon(paths: str, size: int = 18, idle: str = "#35505A") -> QIcon:
        """Stroke icon: dark when idle, white when the (checkable) button is checked."""
        icon = QIcon()
        if not SVG_AVAILABLE:
            return icon
        for color, state in ((idle, QIcon.Off), ("#FFFFFF", QIcon.On)):
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
                   f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')
            renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
            pm = QPixmap(size * 2, size * 2)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            renderer.render(p)
            p.end()
            icon.addPixmap(pm, QIcon.Normal, state)
        return icon

    NAV_ICONS = {
        "Overview": '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
        "Live Detection": '<path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>',
        "Detection History": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
        "Analytics": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
        "Alerts": '<path d="M18 8a6 6 0 10-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.7 21a2 2 0 01-3.4 0"/>',
        "Settings": '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>',
    }

    class GradientBackground(QWidget):
        """Paints the soft gradient + blurred color blobs that the glass cards sit on."""

        def paintEvent(self, _event):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            g = QLinearGradient(0, 0, w, h)
            g.setColorAt(0.0, QColor("#D9F1F2"))
            g.setColorAt(0.48, QColor("#E4EBFA"))
            g.setColorAt(1.0, QColor("#F6E6F1"))
            p.fillRect(self.rect(), g)
            for cx, cy, r, (R, G, B, A) in (
                (w * 0.05, h * 0.02, max(w, h) * 0.45, (127, 214, 216, 175)),
                (w * 0.98, h * 1.00, max(w, h) * 0.50, (199, 168, 240, 150)),
            ):
                rg = QRadialGradient(QPointF(cx, cy), r)
                rg.setColorAt(0.0, QColor(R, G, B, A))
                rg.setColorAt(1.0, QColor(R, G, B, 0))
                p.setPen(Qt.NoPen)
                p.setBrush(rg)
                p.drawEllipse(QPointF(cx, cy), r, r)

    class MetricCard(QFrame):
        """Glass card: muted title, big colored value, small subtext."""

        def __init__(self, title: str, initial_value: str, subtitle: str, accent_color: str = MUTED, parent=None,
                     icon_svg: Optional[str] = None, value_size: int = 24):
            super().__init__(parent)
            self._value_size = value_size
            self.setProperty("class", "glass")
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(30)
            shadow.setOffset(0, 8)
            shadow.setColor(QColor(40, 80, 110, 28))
            self.setGraphicsEffect(shadow)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(22, 20, 22, 20)
            layout.setSpacing(6)

            self.lbl_title = lbl(title, 13, 500, SUB)
            self.lbl_value = QLabel(initial_value)
            self.lbl_value.setWordWrap(True)
            self.lbl_subtext = lbl(subtitle, 12, 400, SUB, wrap=True)
            self.set_accent_color(accent_color)

            if icon_svg:
                head = QHBoxLayout()
                head.setSpacing(8)
                ic = QLabel()
                ic.setPixmap(svg_icon(icon_svg, 18, idle=TEAL).pixmap(QSize(18, 18)))
                head.addWidget(ic)
                head.addWidget(self.lbl_title)
                head.addStretch()
                layout.addLayout(head)
            else:
                layout.addWidget(self.lbl_title)
            layout.addWidget(self.lbl_value)
            layout.addWidget(self.lbl_subtext)
            layout.addStretch()

        def update_value(self, value: str, subtitle: Optional[str] = None):
            self.lbl_value.setText(value)
            if subtitle is not None:
                self.lbl_subtext.setText(subtitle)

        def set_accent_color(self, color: str):
            """The accent colors the big value (status at a glance)."""
            self.lbl_value.setStyleSheet(f"font-size:{self._value_size}px; font-weight:700; color:{color};")

    class TableWithEmpty(QWidget):
        """A styled table that swaps to a friendly empty state when it has no rows."""

        def __init__(self, headers, empty_title: str, empty_sub: str, parent=None):
            super().__init__(parent)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)

            self.table = QTableWidget(0, len(headers))
            self.table.setHorizontalHeaderLabels(headers)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.verticalHeader().setVisible(False)
            self.table.setShowGrid(False)
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.setSelectionMode(QAbstractItemView.NoSelection)

            empty = QWidget()
            e = QVBoxLayout(empty)
            e.setAlignment(Qt.AlignCenter)
            self.empty_title = lbl(empty_title, 15, 700, INK)
            self.empty_sub = lbl(empty_sub, 13, 400, SUB, wrap=True)
            self.empty_title.setAlignment(Qt.AlignCenter)
            self.empty_sub.setAlignment(Qt.AlignCenter)
            self.empty_sub.setMaximumWidth(320)
            e.addWidget(self.empty_title, alignment=Qt.AlignCenter)
            e.addWidget(self.empty_sub, alignment=Qt.AlignCenter)

            self.stack = QStackedWidget()
            self.stack.addWidget(empty)       # 0
            self.stack.addWidget(self.table)  # 1
            lay.addWidget(self.stack)
            self._header_only = QTableWidget(0, len(headers))  # keeps header visible in empty state
            self._header_only.setHorizontalHeaderLabels(headers)
            self._header_only.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self._header_only.verticalHeader().setVisible(False)
            self._header_only.setFixedHeight(self._header_only.horizontalHeader().height() + 2)
            self._header_only.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self._header_only.setSelectionMode(QAbstractItemView.NoSelection)
            lay.insertWidget(0, self._header_only)
            self.set_rows([])

        def set_empty_text(self, title: str, sub: str):
            self.empty_title.setText(title)
            self.empty_sub.setText(sub)

        def set_rows(self, rows):
            """rows: list of list[str]. Empty list -> empty state."""
            self.table.setRowCount(0)
            if not rows:
                self._header_only.setVisible(True)
                self.stack.setCurrentIndex(0)
                return
            self._header_only.setVisible(False)
            self.table.setRowCount(len(rows))
            for r, cols in enumerate(rows):
                for c, text in enumerate(cols):
                    self.table.setItem(r, c, QTableWidgetItem(text))
            self.stack.setCurrentIndex(1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pages
# ─────────────────────────────────────────────────────────────────────────────
if PYSIDE6_AVAILABLE:
    def clock_pill() -> (QFrame, QLabel):
        f = QFrame()
        f.setProperty("class", "glass")
        v = QVBoxLayout(f)
        v.setContentsMargins(16, 8, 16, 8)
        v.setSpacing(0)
        v.addWidget(lbl("System time", 11, 500, SUB), alignment=Qt.AlignRight)
        t = lbl("", 13, 600, INK)
        v.addWidget(t, alignment=Qt.AlignRight)
        return f, t

    class OverviewPage(QWidget):
        """Dashboard: offline banner, 4 status cards, chamber diagnostics, live tracks."""
        retry_requested = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.last_success_time: Optional[float] = None
            self.is_stale = False
            self.init_ui()

        def init_ui(self):
            root = QVBoxLayout(self)
            root.setContentsMargins(12, 8, 8, 8)
            root.setSpacing(18)

            clock_frame, self.lbl_clock = clock_pill()
            root.addLayout(page_header("Overview", "Live status from the Raspberry Pi 4 detector", clock_frame))

            # Banner (hidden while healthy)
            self.banner = QFrame()
            self.banner.setProperty("class", "banner")
            b = QHBoxLayout(self.banner)
            b.setContentsMargins(20, 14, 20, 14)
            b.setSpacing(16)
            icon = QLabel("!")
            icon.setFixedSize(40, 40)
            icon.setAlignment(Qt.AlignCenter)
            icon.setStyleSheet(f"background:#FBCFCA; color:#9A241B; border-radius:20px; font-size:20px; font-weight:800;")
            text = QVBoxLayout()
            text.setSpacing(2)
            self.banner_title = lbl("Detector offline", 15, 700, "#7A1C14")
            self.banner_msg = lbl("", 13, 400, "#7A271A", wrap=True)
            text.addWidget(self.banner_title)
            text.addWidget(self.banner_msg)
            btn_retry = QPushButton("Retry connection")
            btn_retry.setProperty("class", "danger")
            btn_retry.setCursor(Qt.PointingHandCursor)
            btn_retry.clicked.connect(self.retry_requested.emit)
            b.addWidget(icon)
            b.addLayout(text, stretch=1)
            b.addWidget(btn_retry)
            self.banner.hide()
            root.addWidget(self.banner)

            # Status cards
            grid = QGridLayout()
            grid.setHorizontalSpacing(16)
            self.card_status = MetricCard("Detector", "CONNECTING…", "Polling GET /api/status")
            self.card_fps = MetricCard("Camera & AI speed", "N/A", "Display: N/A | AI: N/A")
            self.card_mosquitoes = MetricCard("Mosquitoes in chamber", "N/A", "Persistent tracks (unique MOS-IDs)")
            self.card_detections = MetricCard("Detections & registry", "N/A", "Frame: N/A | Registry: N/A")
            for i, c in enumerate((self.card_status, self.card_fps, self.card_mosquitoes, self.card_detections)):
                grid.addWidget(c, 0, i)
                grid.setColumnStretch(i, 1)
            root.addLayout(grid)

            # Lower panels
            lower = QHBoxLayout()
            lower.setSpacing(16)

            health = glass()
            hv = QVBoxLayout(health)
            hv.setContentsMargins(24, 22, 24, 18)
            hv.addWidget(lbl("Chamber health", 16, 700))
            self.telem_labels = add_info_rows(hv, [
                ("detector_running", "Detector engine", "N/A"),
                ("detector_mode", "Detection mode", "N/A"),
                ("fps", "Display stream rate", "N/A"),
                ("ai_fps", "AI worker rate", "N/A"),
                ("frame_age", "Last camera frame age", "N/A"),
                ("confirmed_frame", "Confirmed in current frame", "N/A"),
                ("active_ids", "Active tracked IDs", "N/A"),
                ("registry_total", "Registry rows (SQLite)", "N/A"),
            ])
            hv.addStretch()
            lower.addWidget(health, stretch=1)

            tracks = glass()
            tv = QVBoxLayout(tracks)
            tv.setContentsMargins(24, 22, 24, 18)
            tv.addWidget(lbl("Mosquitoes in chamber", 16, 700))
            self.tracks = TableWithEmpty(["MOS ID", "Species", "Confidence", "Hits"],
                                         "Waiting for the detector",
                                         "Tracked mosquitoes will appear here once the connection is restored.")
            tv.addWidget(self.tracks, stretch=1)
            lower.addWidget(tracks, stretch=1)
            root.addLayout(lower, stretch=1)

            self.clock_timer = QTimer(self)
            self.clock_timer.timeout.connect(self.on_timer_tick)
            self.clock_timer.start(1000)
            self.on_timer_tick()

        def _show_banner(self, title: str, msg: str):
            self.banner_title.setText(title)
            self.banner_msg.setText(msg)
            self.banner.show()

        def on_timer_tick(self):
            self.lbl_clock.setText(datetime.now().strftime("%d %b %Y · %H:%M:%S"))
            if self.last_success_time is not None and not self.banner.isVisible():
                stale_delta = time.time() - self.last_success_time
                if stale_delta > 3.5:
                    self.is_stale = True
                    self.card_status.update_value("STALE", f"No update for {stale_delta:.1f}s")
                    self.card_status.set_accent_color(WARN)

        def update_telemetry(self, data: Dict[str, Any]):
            """Valid 200 OK payload received."""
            self.last_success_time = time.time()
            self.is_stale = False
            self.banner.hide()

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

            # Card 1: detector status
            uptime_str = f"{uptime:.1f}s" if isinstance(uptime, (int, float)) else "N/A"
            if running is True:
                self.card_status.update_value("RUNNING", f"Mode: {str(mode).upper() if mode else 'N/A'} | Uptime: {uptime_str}")
                self.card_status.set_accent_color(OK)
            elif running is False:
                self.card_status.update_value("STANDBY", "Detector loop is idle")
                self.card_status.set_accent_color(MUTED)
            else:
                self.card_status.update_value("N/A", "Detector state unknown")
                self.card_status.set_accent_color(MUTED)

            # Card 2: FPS
            fps_str = f"{fps:.1f} FPS" if isinstance(fps, (int, float)) else "N/A"
            ai_fps_str = f"{ai_fps:.1f} FPS" if isinstance(ai_fps, (int, float)) else "N/A"
            self.card_fps.update_value(fps_str, f"AI worker: {ai_fps_str}")
            self.card_fps.set_accent_color(INK if isinstance(fps, (int, float)) else MUTED)

            # Card 3: active tracks
            if isinstance(active_ids, list):
                ids_str = ", ".join(str(i) for i in active_ids) if active_ids else "None"
                self.card_mosquitoes.update_value(f"{len(active_ids)} active", f"IDs: {ids_str}")
                self.card_mosquitoes.set_accent_color(INK)
            else:
                ids_str = "N/A"
                self.card_mosquitoes.update_value("N/A", "Active tracks unavailable")
                self.card_mosquitoes.set_accent_color(MUTED)

            # Card 4: frame + registry
            frame_str = str(int(confirmed_frame)) if isinstance(confirmed_frame, (int, float)) else "N/A"
            reg_str = f"{int(registry_total):,} rows" if isinstance(registry_total, (int, float)) else "N/A"
            self.card_detections.update_value(f"Frame: {frame_str}", f"Registry total: {reg_str}")
            self.card_detections.set_accent_color(INK)

            # Health rows
            set_value(self.telem_labels["detector_running"],
                      "Running" if running is True else ("Standby" if running is False else "N/A"),
                      OK if running is True else MUTED)
            set_value(self.telem_labels["detector_mode"], str(mode).upper() if mode else "N/A", INK if mode else MUTED)
            set_value(self.telem_labels["fps"], fps_str, INK if fps_str != "N/A" else MUTED)
            set_value(self.telem_labels["ai_fps"], ai_fps_str, INK if ai_fps_str != "N/A" else MUTED)
            set_value(self.telem_labels["frame_age"],
                      f"{frame_age:.3f} s" if isinstance(frame_age, (int, float)) else "N/A",
                      INK if isinstance(frame_age, (int, float)) else MUTED)
            set_value(self.telem_labels["confirmed_frame"], frame_str, INK if frame_str != "N/A" else MUTED)
            set_value(self.telem_labels["active_ids"], ids_str, INK if ids_str != "N/A" else MUTED)
            set_value(self.telem_labels["registry_total"], reg_str, INK if reg_str != "N/A" else MUTED)

            # Tracks table
            rows = []
            if isinstance(active_tracks, list):
                for t in active_tracks:
                    if not isinstance(t, dict):
                        continue
                    conf_val = t.get("confidence")
                    conf = f"{conf_val * 100:.1f}%" if isinstance(conf_val, (int, float)) else "N/A"
                    rows.append([str(t.get("mos_id", "N/A")), str(t.get("species", "N/A")), conf, str(t.get("hit_count", "N/A"))])
            self.tracks.set_empty_text("No mosquitoes tracked", "The chamber is clear right now.")
            self.tracks.set_rows(rows)

        def set_error_state(self, err_type: str, description: str):
            """Polling failed or server unreachable."""
            self.is_stale = True
            if err_type == "unauthorized":
                self.card_status.update_value("UNAUTHORIZED", "X-Sentinel-Token missing or invalid")
                self.card_status.set_accent_color(WARN)
                self._show_banner("Access denied", "The detector rejected the access token. Check the token in client_config.json.")
            elif err_type in ("connection_refused", "timeout", "network_error"):
                self.card_status.update_value("OFFLINE", "Cannot connect to status server")
                self.card_status.set_accent_color(BAD)
                self._show_banner("Detector offline", "Can’t reach the Raspberry Pi. Chamber monitoring is paused until the connection returns.")
            elif err_type == "malformed_json":
                self.card_status.update_value("INVALID DATA", "Invalid JSON from server")
                self.card_status.set_accent_color(BAD)
                self._show_banner("Unexpected response", "The detector sent data this app can’t read.")
            else:
                self.card_status.update_value("ERROR", description[:45])
                self.card_status.set_accent_color(BAD)
                self._show_banner("Connection error", description)

            self.card_fps.update_value("N/A", "Display: N/A | AI: N/A")
            self.card_fps.set_accent_color(MUTED)
            self.card_mosquitoes.update_value("N/A", "Cannot retrieve active tracks")
            self.card_mosquitoes.set_accent_color(MUTED)
            self.card_detections.update_value("N/A", "Frame: N/A | Registry: N/A")
            self.card_detections.set_accent_color(MUTED)
            for label in self.telem_labels.values():
                set_value(label, "N/A", MUTED)
            self.tracks.set_empty_text("Waiting for the detector", "Tracked mosquitoes will appear here once the connection is restored.")
            self.tracks.set_rows([])

    class VideoViewport(QFrame):
        """Displays the Pi camera frame scaled to fit with aspect ratio kept, or status/error messages."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(320, 180)
            self._pixmap: Optional[QPixmap] = None
            self._is_stale: bool = False
            self._stale_note: str = ""
            self._error_title: str = "Connecting to video stream…"
            self._error_subtitle: str = "Waiting for live frame from http://<host>:<port>/api/frame.jpg."

        def set_frame(self, pixmap: QPixmap):
            self._pixmap = pixmap
            self._is_stale = False
            self._stale_note = ""
            self.update()

        def set_stale(self, is_stale: bool, note: str = ""):
            self._is_stale = is_stale
            self._stale_note = note
            self.update()

        def set_error_message(self, title: str, subtitle: str):
            self._error_title = title
            self._error_subtitle = subtitle
            self.update()

        def has_frame(self) -> bool:
            return self._pixmap is not None and not self._pixmap.isNull()

        def paintEvent(self, _event):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setRenderHint(QPainter.SmoothPixmapTransform)

            w, h = self.width(), self.height()
            rect = QRectF(0, 0, w, h)

            # Dark gradient background
            bg = QLinearGradient(0, 0, w, h)
            bg.setColorAt(0.0, QColor("#17313A"))
            bg.setColorAt(1.0, QColor("#25465A"))
            p.setPen(Qt.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(rect, 14, 14)

            if self.has_frame():
                pw, ph = self._pixmap.width(), self._pixmap.height()
                if pw > 0 and ph > 0:
                    p.save()
                    clip_path = QPainterPath()
                    clip_path.addRoundedRect(rect, 14, 14)
                    p.setClipPath(clip_path)

                    scale = min((w - 8) / pw, (h - 8) / ph)
                    dw, dh = int(pw * scale), int(ph * scale)
                    dx, dy = (w - dw) // 2, (h - dh) // 2
                    target_rect = QRect(dx, dy, dw, dh)

                    p.drawPixmap(target_rect, self._pixmap)
                    p.restore()

                    if self._is_stale:
                        p.save()
                        stale_text = f"⚠ Stale · {self._stale_note}" if self._stale_note else "⚠ Stale frame"
                        f = p.font()
                        f.setPointSize(10)
                        f.setBold(True)
                        p.setFont(f)
                        fm = p.fontMetrics()
                        tw = fm.horizontalAdvance(stale_text)
                        box_rect = QRectF(16, 16, tw + 24, 28)
                        p.setPen(QColor(183, 121, 31, 200))
                        p.setBrush(QColor(25, 40, 48, 220))
                        p.drawRoundedRect(box_rect, 8, 8)
                        p.setPen(QColor("#FBEFD5"))
                        p.drawText(box_rect, Qt.AlignCenter, stale_text)
                        p.restore()
            else:
                p.save()
                f_title = p.font()
                f_title.setPointSize(15)
                f_title.setBold(True)
                p.setFont(f_title)
                p.setPen(QColor("#E6F1F3"))
                p.drawText(QRectF(24, h // 2 - 32, w - 48, 30), Qt.AlignCenter, self._error_title)

                f_sub = p.font()
                f_sub.setPointSize(11)
                p.setFont(f_sub)
                p.setPen(QColor("#B8CDD2"))
                p.drawText(QRectF(32, h // 2 + 4, w - 64, 48), Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self._error_subtitle)
                p.restore()

    class ExpandedCameraDialog(QDialog):
        """Full-screen or expanded camera preview dialog."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("MosquitoSentinel - Live Camera Feed (Expanded)")
            self.setMinimumSize(800, 450)
            self.resize(1280, 720)
            self.setStyleSheet("background-color: #0E1B20;")

            layout = QVBoxLayout(self)
            layout.setContentsMargins(14, 12, 14, 14)
            layout.setSpacing(10)

            top_bar = QHBoxLayout()
            lbl_title = QLabel("Live Chamber Camera · Press ESC or click Close to return")
            lbl_title.setStyleSheet("color: #B8CDD2; font-size: 13px; font-weight: 600;")
            top_bar.addWidget(lbl_title)
            top_bar.addStretch()

            btn_close = QPushButton("✕ Exit Expanded View")
            btn_close.setProperty("class", "ghost")
            btn_close.setCursor(Qt.PointingHandCursor)
            btn_close.clicked.connect(self.accept)
            top_bar.addWidget(btn_close)
            layout.addLayout(top_bar)

            self.viewport = VideoViewport(self)
            layout.addWidget(self.viewport, stretch=1)

        def set_frame(self, pixmap: QPixmap):
            self.viewport.set_frame(pixmap)

        def set_stale(self, is_stale: bool, note: str = ""):
            self.viewport.set_stale(is_stale, note)

        def set_error_message(self, title: str, subtitle: str):
            self.viewport.set_error_message(title, subtitle)

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self.accept()
            else:
                super().keyPressEvent(event)

    class LiveDetectionPage(QWidget):
        """Live camera stream with mosquito tracking overlay, telemetry, and stream metrics."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._frame_times: List[float] = []
            self._last_frame_size: Optional[Tuple[int, int]] = None
            self._last_render_ms: float = 0.0
            self._latest_diag: Dict[str, Any] = {}
            self._current_stream_fps: float = 0.0
            self._expanded_dialog: Optional[ExpandedCameraDialog] = None

            root = QVBoxLayout(self)
            root.setContentsMargins(12, 8, 8, 8)
            root.setSpacing(18)

            self.badge = pill("● Connecting…", "#E8EEF0", MUTED)
            root.addLayout(page_header("Live Detection", "Chamber camera with real-time mosquito tracking", self.badge))

            card = glass()
            v = QVBoxLayout(card)
            v.setContentsMargins(14, 14, 14, 14)
            v.setSpacing(14)

            self.viewport = VideoViewport()
            v.addWidget(self.viewport, stretch=1)

            controls = QHBoxLayout()
            self.btn_snap = QPushButton("Snapshot")
            self.btn_snap.setProperty("class", "ghost")
            self.btn_snap.setCursor(Qt.PointingHandCursor)
            self.btn_snap.setEnabled(False)  # enabled once a valid frame is received
            self.btn_snap.clicked.connect(self._on_snapshot_clicked)
            controls.addWidget(self.btn_snap)
            controls.addStretch()

            self.lbl_stream_info = lbl("Resolution — · Frame rate —", 12, 400, SUB)
            controls.addWidget(self.lbl_stream_info)

            # Symbol-only fullscreen button on the right side
            self.btn_full = QPushButton()
            self.btn_full.setProperty("class", "ghost")
            self.btn_full.setFixedSize(40, 40)
            self.btn_full.setToolTip("Expand Camera")
            self.btn_full.setAccessibleName("Expand Camera")
            self.btn_full.setCursor(Qt.PointingHandCursor)
            self.btn_full.setIcon(svg_icon('<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>', 20))
            self.btn_full.setIconSize(QSize(20, 20))
            self.btn_full.setEnabled(False)
            self.btn_full.clicked.connect(self._on_expand_camera_clicked)
            controls.addWidget(self.btn_full)
            v.addLayout(controls)
            root.addWidget(card, stretch=1)

        def _on_snapshot_clicked(self):
            """Captures the current annotated camera frame and saves it with a unique timestamp."""
            if not self.viewport.has_frame() or self.viewport._pixmap is None:
                return

            try:
                import config
                captures_dir = getattr(config, "CAPTURES_DIR", "")
            except Exception:
                captures_dir = ""
            if not captures_dir:
                captures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")

            try:
                os.makedirs(captures_dir, exist_ok=True)
                now_dt = datetime.now()
                fname = f"snapshot_{now_dt.strftime('%Y%m%d_%H%M%S_%f')[:21]}.jpg"
                fpath = os.path.join(captures_dir, fname)
                saved = self.viewport._pixmap.save(fpath, "JPG")
                if saved and os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                    orig_text = "Snapshot"
                    self.btn_snap.setText("✓ Saved!")
                    QTimer.singleShot(2500, lambda: self.btn_snap.setText(orig_text))
                    print(f"[INFO] Camera snapshot saved: {fname}")
                else:
                    self.btn_snap.setText("⚠ Save failed")
                    QTimer.singleShot(2500, lambda: self.btn_snap.setText("Snapshot"))
            except Exception as e:
                print(f"[ERROR] Snapshot failed: {e}")
                self.btn_snap.setText("⚠ Error")
                QTimer.singleShot(2500, lambda: self.btn_snap.setText("Snapshot"))

        def _on_expand_camera_clicked(self):
            """Expands the camera preview into a dedicated larger view."""
            self._expanded_dialog = ExpandedCameraDialog(self)
            if self.viewport.has_frame() and self.viewport._pixmap:
                self._expanded_dialog.set_frame(self.viewport._pixmap)
            self._expanded_dialog.show()

        def update_badge(self, state: str):
            """Updates the top-right badge to Connecting / Live / Stale / Error."""
            styles = {
                "connecting": ("● Connecting…", "#E8EEF0", MUTED),
                "live": ("● Live", "#DDF3E4", OK),
                "stale": ("● Stale", "#FBEFD5", WARN),
                "error": ("● Error", "#FDE3E0", BAD),
            }
            text, bg, fg = styles.get(state.lower(), ("● " + state.title(), "#E8EEF0", MUTED))
            self.badge.setText(text)
            self.badge.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:12px; padding:5px 12px; font-size:12px; font-weight:600;")

        @Slot(QImage, float)
        def on_frame_received(self, image: QImage, *args, **kwargs):
            t_ren_start = time.perf_counter()
            now = time.time()
            self._frame_times.append(now)
            cutoff = now - 5.0
            while len(self._frame_times) > 1 and self._frame_times[0] < cutoff:
                self._frame_times.pop(0)

            pixmap = QPixmap.fromImage(image)
            self._last_frame_size = (pixmap.width(), pixmap.height())
            self.viewport.set_frame(pixmap)
            self.update_badge("live")
            self.btn_snap.setEnabled(True)
            self.btn_full.setEnabled(True)

            if getattr(self, "_expanded_dialog", None) is not None and self._expanded_dialog.isVisible():
                self._expanded_dialog.set_frame(pixmap)

            w, h = self._last_frame_size
            self._last_render_ms = (time.perf_counter() - t_ren_start) * 1000.0

            if len(self._frame_times) >= 2:
                dt = self._frame_times[-1] - self._frame_times[0]
                fps = (len(self._frame_times) - 1) / dt if dt > 0 else 0.0
                self._current_stream_fps = fps
                self._update_stream_info_label(w, h, fps)
            else:
                self.lbl_stream_info.setText(f"Resolution {w} × {h} · Stream rate —")

        @Slot(dict)
        def on_diagnostics_updated(self, diag: Dict[str, Any]):
            self._latest_diag = diag
            if len(self._frame_times) >= 2 and self._last_frame_size:
                w, h = self._last_frame_size
                self._update_stream_info_label(w, h, self._current_stream_fps)

        def _update_stream_info_label(self, w: int, h: int, fps: float):
            diag = getattr(self, "_latest_diag", {})
            dl_ms = diag.get("download_ms")
            dec_ms = diag.get("decode_ms")
            ren_ms = getattr(self, "_last_render_ms", 0.0)
            age_ms = diag.get("frame_age_ms")
            det_fps = diag.get("detector_fps")

            if dl_ms is not None:
                latency_parts = [f"dl {dl_ms:.0f}ms", f"dec {dec_ms or 0:.0f}ms"]
                if ren_ms > 0:
                    latency_parts.append(f"ren {ren_ms:.0f}ms")
                tot_lat = (dl_ms or 0) + (dec_ms or 0) + (ren_ms or 0)
                lat_str = f" · Latency {tot_lat:.0f}ms ({', '.join(latency_parts)})"
            else:
                tot_lat = ren_ms
                lat_str = " (network fetch)"
            age_str = f" · Age {age_ms/1000.0:.2f}s" if age_ms is not None else ""
            det_str = f" [Pi {det_fps:.1f} fps]" if det_fps else ""

            self.lbl_stream_info.setText(
                f"Resolution {w} × {h} · Stream rate {fps:.1f} fps{det_str}{lat_str}{age_str}"
            )

            drop_count = diag.get("dropped_stale", 0)
            intv_ms = diag.get("interval_ms", 0)
            tt = (
                f"Video Stream Diagnostics:\n"
                f"  • Preview Resolution: {w} × {h}\n"
                f"  • Laptop Stream FPS: {fps:.1f} fps (Arrival Interval: {intv_ms:.1f} ms)\n"
                f"  • Raspberry Pi Detector FPS: {det_fps if det_fps else 'N/A'}\n"
                f"  • JPEG Download Time: {dl_ms or 0:.1f} ms\n"
                f"  • Worker Decode Time: {dec_ms or 0:.1f} ms\n"
                f"  • GUI Render Time: {ren_ms:.1f} ms\n"
                f"  • Total Preview Latency: {tot_lat:.1f} ms\n"
                f"  • Frame Age (Sensor -> Screen): {age_ms/1000.0 if age_ms is not None else 'N/A'} s\n"
                f"  • Stale / Duplicate Frames Discarded: {drop_count}"
            )
            self.lbl_stream_info.setToolTip(tt)

        @Slot(str, str)
        def on_frame_error(self, err_type: str, description: str):
            error_messages = {
                "unauthorized": ("Unauthorized (401)", "Invalid or missing token. Check SENTINEL_AUTH_TOKEN or client_config.json."),
                "not_found": ("Endpoint Not Found (404)", "The camera endpoint /api/frame.jpg was not found on the server."),
                "no_frame": ("Waiting for first frame (503)", "The camera engine has not produced a frame yet. Live view will appear automatically."),
                "timeout": ("Connection Timed Out", "No response from Raspberry Pi within 3 seconds."),
                "connection_refused": ("Cannot Connect to Raspberry Pi", "Connection was refused. Ensure the detector server is running on the Pi."),
                "decode_error": ("Picture Decode Error", "Received image data could not be decoded as a valid JPEG."),
                "network_error": ("Network Error", description),
            }
            title, subtitle = error_messages.get(err_type, ("Connection Error", description))

            short_reasons = {
                "unauthorized": "Unauthorized (401)",
                "not_found": "Not Found (404)",
                "no_frame": "Waiting for frame (503)",
                "timeout": "Timed out",
                "connection_refused": "Connection refused",
                "decode_error": "Decode error",
                "network_error": "Network error",
            }
            short = short_reasons.get(err_type, "Connection error")

            if self.viewport.has_frame():
                self.viewport.set_stale(True, short)
                self.update_badge("stale")
                w, h = self._last_frame_size if self._last_frame_size else (0, 0)
                res_str = f"{w} × {h} (stale)" if w and h else "—"
                self.lbl_stream_info.setText(f"Resolution {res_str} · Stream rate 0.0 fps (stale)")
                if getattr(self, "_expanded_dialog", None) is not None and self._expanded_dialog.isVisible():
                    self._expanded_dialog.set_stale(True, short)
            else:
                self.btn_snap.setEnabled(False)
                self.viewport.set_error_message(title, subtitle)
                if err_type == "no_frame":
                    self.update_badge("connecting")
                else:
                    self.update_badge("error")
                self.lbl_stream_info.setText("Resolution — · Stream rate —")
                if getattr(self, "_expanded_dialog", None) is not None and self._expanded_dialog.isVisible():
                    self._expanded_dialog.set_error_message(title, subtitle)

    class HistoryPage(QWidget):
        """SQLite registry explorer and CSV exporter querying /api/registry or local database."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.records: List[Dict[str, Any]] = []
            self.client_config: Optional[Dict[str, Any]] = None
            self._filter_all = False

            root = QVBoxLayout(self)
            root.setContentsMargins(12, 8, 8, 8)
            root.setSpacing(18)
            root.addLayout(page_header("Detection History", "Registry of every mosquito the chamber has recorded"))

            bar = glass()
            h = QHBoxLayout(bar)
            h.setContentsMargins(18, 12, 18, 12)

            lbl_date = QLabel("Filter by date:")
            lbl_date.setStyleSheet(f"color:{SUB}; font-size:12px; font-weight:600;")
            h.addWidget(lbl_date)

            self.date_filter = QDateEdit(QDate.currentDate())
            self.date_filter.setCalendarPopup(True)
            self.date_filter.setDisplayFormat("dd MMM yyyy")
            self.date_filter.setFixedWidth(150)
            self.date_filter.dateChanged.connect(self._apply_filter)
            h.addWidget(self.date_filter)

            self.btn_show_all = QPushButton("Show All Dates")
            self.btn_show_all.setProperty("class", "ghost")
            self.btn_show_all.setCursor(Qt.PointingHandCursor)
            self.btn_show_all.clicked.connect(self._show_all_dates)
            h.addWidget(self.btn_show_all)

            self.lbl_record_count = QLabel("0 records")
            self.lbl_record_count.setStyleSheet(f"color:{SUB}; font-size:12px;")
            h.addWidget(self.lbl_record_count)

            h.addStretch()

            self.btn_export = QPushButton("Export CSV")
            self.btn_export.setProperty("class", "ghost")
            self.btn_export.setCursor(Qt.PointingHandCursor)
            self.btn_export.setEnabled(False)
            self.btn_export.clicked.connect(self._on_export_csv_clicked)
            h.addWidget(self.btn_export)
            root.addWidget(bar)

            panel = glass()
            v = QVBoxLayout(panel)
            v.setContentsMargins(24, 20, 24, 18)
            self.registry = TableWithEmpty(["Timestamp", "ID", "Species", "Confidence", "Hits"],
                                           "No detections to show",
                                           "No mosquito detections found in the chamber registry.")
            v.addWidget(self.registry)
            root.addWidget(panel, stretch=1)

        def set_client_config(self, cfg: Dict[str, Any]):
            self.client_config = cfg

        def _show_all_dates(self):
            self._filter_all = True
            self._render_rows()

        def _apply_filter(self):
            self._filter_all = False
            self._render_rows()

        def load_records(self, records: List[Dict[str, Any]]):
            """Loads raw registry records and renders rows under the active date filter."""
            self.records = records if isinstance(records, list) else []
            self.btn_export.setEnabled(len(self.records) > 0)
            self._render_rows()

        def _get_filtered_records(self) -> List[Dict[str, Any]]:
            if self._filter_all:
                return self.records

            qdate = self.date_filter.date()
            filter_str = f"{qdate.year():04d}-{qdate.month():02d}-{qdate.day():02d}"
            matching = []
            for r in self.records:
                ts = str(r.get("last_seen") or r.get("first_seen") or "")
                if ts.startswith(filter_str):
                    matching.append(r)
            return matching

        def _render_rows(self):
            filtered = self._get_filtered_records()
            rows = []
            for r in filtered:
                ts = str(r.get("last_seen") or r.get("first_seen") or "N/A")
                mid = str(r.get("mos_id", "N/A"))
                sp = str(r.get("species", "N/A"))
                c_val = r.get("confidence")
                conf = f"{float(c_val)*100:.1f}%" if _is_num(c_val) else "N/A"
                hits = str(r.get("detect_count", 1))
                rows.append([ts, mid, sp, conf, hits])

            if filtered:
                self.lbl_record_count.setText(f"{len(filtered)} records displayed")
                self.registry.set_rows(rows)
            else:
                if self.records:
                    qdate = self.date_filter.date()
                    d_str = f"{qdate.year():04d}-{qdate.month():02d}-{qdate.day():02d}"
                    self.lbl_record_count.setText(f"0 records on {d_str} ({len(self.records)} total)")
                    self.registry.set_empty_text(
                        f"No detections on {d_str}",
                        f"There are {len(self.records)} records on other dates. Click 'Show All Dates' to view all records."
                    )
                else:
                    self.lbl_record_count.setText("0 records")
                    self.registry.set_empty_text(
                        "No detections to show",
                        "No mosquito detections found in the chamber registry."
                    )
                self.registry.set_rows([])

        def fetch_registry(self, client_cfg: Optional[Dict[str, Any]] = None):
            """Fetches records from remote /api/registry, or falls back to local SQLite database."""
            cfg = client_cfg or self.client_config
            if cfg is None:
                cfg = load_client_config()

            records = None

            # 1. Try remote status server /api/registry
            pi_host = cfg.get("pi_host", "127.0.0.1")
            pi_port = cfg.get("pi_port", 8080)
            token = cfg.get("token", "")
            timeout = cfg.get("timeout_sec", 3.0)

            try:
                url = f"http://{pi_host}:{pi_port}/api/registry?limit=1000"
                req = urllib.request.Request(url)
                if token:
                    req.add_header("X-Sentinel-Token", token)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        records = data.get("records", [])
            except Exception:
                records = None

            # 2. If remote unreachable, query Supabase cloud database directly if enabled
            if records is None:
                try:
                    import config
                    if getattr(config, "SUPABASE_ENABLED", False) and getattr(config, "SUPABASE_URL", "") and getattr(config, "SUPABASE_KEY", ""):
                        from supabase import create_client
                        supa = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
                        res = supa.table(config.SUPABASE_TABLE).select("*").order("first_seen", desc=True).limit(1000).execute()
                        if res.data:
                            records = []
                            for r in res.data:
                                rec = dict(r)
                                rec["mos_id"] = r.get("id")
                                records.append(rec)
                except Exception as e:
                    records = None

            # 3. If still None, fallback to local SQLite if it exists
            if records is None:
                try:
                    import sqlite3
                    import config
                    db_path = getattr(config, "TRACKER_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mosquito_registry.db"))
                    if os.path.isfile(db_path):
                        uri = f"file:{os.path.abspath(db_path)}?mode=ro"
                        conn = sqlite3.connect(uri, uri=True, timeout=1.0)
                        conn.row_factory = sqlite3.Row
                        cur = conn.cursor()
                        cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='mosquito_registry'")
                        if cur.fetchone()[0] > 0:
                            cur.execute(
                                "SELECT id, mos_id, species, first_seen, last_seen, detect_count, "
                                "confidence, snapshot_path, verification_status "
                                "FROM mosquito_registry ORDER BY id DESC LIMIT 1000"
                            )
                            rows = cur.fetchall()
                            records = [dict(r) for r in rows]
                        conn.close()
                except Exception as e:
                    print(f"[WARNING] Local registry fallback error: {e}")

            if records is not None:
                self.load_records(records)

        def _on_export_csv_clicked(self):
            """Exports detection history records to user-selected CSV file."""
            records_to_export = self._get_filtered_records()
            if not records_to_export:
                records_to_export = self.records
            if not records_to_export:
                QMessageBox.information(self, "Export CSV", "No detection records available to export.")
                return

            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = os.path.join(os.path.expanduser("~"), f"mosquito_detection_history_{now_str}.csv")
            chosen_path, _ = QFileDialog.getSaveFileName(
                self, "Export Detection History", default_path, "CSV Files (*.csv);;All Files (*)"
            )
            if not chosen_path:
                return

            try:
                import csv
                with open(chosen_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Timestamp", "Mosquito_ID", "Species", "Confidence", "Hits",
                        "First_Seen", "Last_Seen", "Verification_Status", "Snapshot_File"
                    ])
                    for r in records_to_export:
                        conf_val = r.get("confidence")
                        conf_str = f"{float(conf_val)*100:.2f}%" if _is_num(conf_val) else str(conf_val or "")
                        writer.writerow([
                            r.get("last_seen") or r.get("first_seen") or "",
                            r.get("mos_id", ""),
                            r.get("species", ""),
                            conf_str,
                            r.get("detect_count", 1),
                            r.get("first_seen", ""),
                            r.get("last_seen", ""),
                            r.get("verification_status", "unverified"),
                            r.get("snapshot_path", "")
                        ])
                QMessageBox.information(
                    self, "Export Successful",
                    f"Successfully exported {len(records_to_export)} records to:\n{chosen_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Could not export CSV file:\n{e}")

    def nice_ceiling(v: float) -> float:
        if v <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(v))
        for m in (1, 2, 2.5, 5, 10):
            if v <= m * mag:
                return m * mag
        return 10 * mag

    class ChartBase(QWidget):
        """Shared empty/unavailable-state painting."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumHeight(260)
            self.message = "No data yet"

        def _paint_message(self, p: QPainter):
            p.setPen(QColor(SUB))
            p.drawText(self.rect().adjusted(24, 0, -24, 0), Qt.AlignCenter | Qt.TextWordWrap, self.message)

    class LineChart(ChartBase):
        """Smooth time-series line chart. Gaps (None / long breaks) are NOT drawn as zero."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.x0 = self.x1 = None
            self.points = []        # [(datetime, float|None)]
            self.markers = []       # [(datetime, float)]  alert events
            self.bands = []         # [(datetime, datetime)] unobserved spans
            self.threshold = None
            self.threshold_label = ""
            self.integer_y = True
            self.break_gap = None

        def set_message(self, text: str):
            self.points, self.markers, self.bands, self.threshold = [], [], [], None
            self.message = text
            self.update()

        def set_data(self, x0, x1, points, *, integer_y=True, threshold=None, threshold_label="",
                     markers=(), bands=(), break_gap=None):
            self.x0, self.x1 = x0, x1
            self.points, self.markers, self.bands = list(points), list(markers), list(bands)
            self.integer_y, self.threshold, self.threshold_label = integer_y, threshold, threshold_label
            self.break_gap = break_gap
            self.update()

        @staticmethod
        def _smooth(pts, lo, hi):
            clamp = lambda y: max(lo, min(hi, y))
            path = QPainterPath(pts[0])
            if len(pts) == 2:
                path.lineTo(pts[1])
                return path
            for i in range(len(pts) - 1):
                p0 = pts[i - 1] if i > 0 else pts[i]
                p1, p2 = pts[i], pts[i + 1]
                p3 = pts[i + 2] if i + 2 < len(pts) else p2
                c1 = QPointF(p1.x() + (p2.x() - p0.x()) / 6, clamp(p1.y() + (p2.y() - p0.y()) / 6))
                c2 = QPointF(p2.x() - (p3.x() - p1.x()) / 6, clamp(p2.y() - (p3.y() - p1.y()) / 6))
                path.cubicTo(c1, c2, p2)
            return path

        def paintEvent(self, _event):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            f = p.font()
            f.setPointSizeF(8.5)
            p.setFont(f)
            w, h = self.width(), self.height()
            vals = [v for _, v in self.points if v is not None]
            if self.x0 is None or (not vals and not self.markers):
                self._paint_message(p)
                return

            left, right, top, bottom = 44, 14, 14, 30
            pw, ph = w - left - right, h - top - bottom
            ys = vals + [m[1] for m in self.markers] + ([self.threshold] if self.threshold is not None else [])
            vmax = max(ys + [0.0])
            ymax = max(4, ((int(math.ceil(vmax)) + 3) // 4) * 4) if self.integer_y else nice_ceiling(vmax * 1.1)
            span = max(1.0, (self.x1 - self.x0).total_seconds())
            X = lambda t: left + (t - self.x0).total_seconds() / span * pw
            Y = lambda v: top + ph - max(0.0, v) / ymax * ph

            for a, b in self.bands:  # unobserved time: hatched, not zero
                xa, xb = X(max(a, self.x0)), X(min(b, self.x1))
                if xb > xa:
                    p.fillRect(QRectF(xa, top, xb - xa, ph), QBrush(QColor(18, 40, 47, 38), Qt.BDiagPattern))
            for i in range(5):
                y = top + ph - ph * i / 4
                p.setPen(QPen(QColor(18, 40, 47, 28), 1))
                p.drawLine(QPointF(left, y), QPointF(w - right, y))
                p.setPen(QColor(SUB))
                p.drawText(QRectF(0, y - 8, left - 6, 16), Qt.AlignRight | Qt.AlignVCenter,
                           str(int(ymax * i / 4)) if self.integer_y else f"{round(ymax * i / 4, 2):g}")
            fmt = "%H:%M" if span <= 36 * 3600 else "%a %d"
            for i in range(7):
                t = self.x0 + (self.x1 - self.x0) * i / 6
                p.setPen(QColor(SUB))
                p.drawText(QRectF(X(t) - 32, h - bottom + 8, 64, 16), Qt.AlignCenter, t.strftime(fmt))

            if self.threshold is not None:
                y = Y(self.threshold)
                p.setPen(QPen(QColor(BAD), 1.6, Qt.DashLine))
                p.drawLine(QPointF(left, y), QPointF(w - right, y))
                p.setPen(QColor(BAD))
                p.drawText(QRectF(w - right - 190, y - 17, 190, 14), Qt.AlignRight | Qt.AlignVCenter, self.threshold_label)

            # Split into contiguous segments
            segs, cur, prev_t = [], [], None
            for t, v in self.points:
                if v is None or (prev_t is not None and self.break_gap and t - prev_t > self.break_gap):
                    if cur:
                        segs.append(cur)
                    cur = []
                if v is not None:
                    cur.append((t, v))
                    prev_t = t
                else:
                    prev_t = None
            if cur:
                segs.append(cur)

            show_dots = len(vals) <= 60
            base = top + ph
            for seg in segs:
                pts = [QPointF(X(t), Y(v)) for t, v in seg]
                if len(pts) >= 2:
                    path = self._smooth(pts, top, base)
                    fill = QPainterPath(path)
                    fill.lineTo(pts[-1].x(), base)
                    fill.lineTo(pts[0].x(), base)
                    fill.closeSubpath()
                    g = QLinearGradient(0, top, 0, base)
                    g.setColorAt(0, QColor(14, 124, 134, 70))
                    g.setColorAt(1, QColor(14, 124, 134, 0))
                    p.setPen(Qt.NoPen)
                    p.fillPath(fill, g)
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(TEAL), 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    p.drawPath(path)
                if show_dots or len(pts) == 1:
                    p.setPen(QPen(QColor(TEAL), 2))
                    p.setBrush(QColor("#FFFFFF"))
                    for pt in pts:
                        p.drawEllipse(pt, 3.5, 3.5)

            for t, v in self.markers:  # alert events
                if self.x0 <= t <= self.x1:
                    p.setPen(QPen(QColor("#FFFFFF"), 2))
                    p.setBrush(QColor(BAD))
                    p.drawEllipse(QPointF(X(t), Y(v)), 6, 6)

    class BarChart(ChartBase):
        """Bar chart with value labels. A value of None means 'not observed' (not zero)."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.data = []

        def set_message(self, text: str):
            self.data = []
            self.message = text
            self.update()

        def set_data(self, data):
            self.data = list(data)
            self.update()

        def paintEvent(self, _event):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            f = p.font()
            f.setPointSizeF(8.5)
            p.setFont(f)
            w, h = self.width(), self.height()
            if not self.data:
                self._paint_message(p)
                return
            left, right, top, bottom = 34, 8, 22, 28
            pw, ph = w - left - right, h - top - bottom
            vmax = max([v for _, v in self.data if v is not None] + [0])
            ymax = max(4, ((int(vmax) + 3) // 4) * 4)
            for i in range(5):
                y = top + ph - ph * i / 4
                p.setPen(QPen(QColor(18, 40, 47, 28), 1))
                p.drawLine(QPointF(left, y), QPointF(w - right, y))
                p.setPen(QColor(SUB))
                p.drawText(QRectF(0, y - 8, left - 6, 16), Qt.AlignRight | Qt.AlignVCenter, str(int(ymax * i / 4)))
            n = len(self.data)
            slot = pw / n
            bw = min(slot * 0.55, 72)
            fm = p.fontMetrics()
            for i, (label, value) in enumerate(self.data):
                cx = left + i * slot + slot / 2
                x = cx - bw / 2
                if value is None:
                    p.setPen(QPen(QColor(18, 40, 47, 60), 1, Qt.DashLine))
                    p.setBrush(QBrush(QColor(18, 40, 47, 38), Qt.BDiagPattern))
                    p.drawRect(QRectF(x, top + ph - 40, bw, 40))
                    p.setPen(QColor(SUB))
                    p.drawText(QRectF(x - 10, top + ph - 60, bw + 20, 16), Qt.AlignCenter, "n/a")
                else:
                    bh = max(3.0, value / ymax * ph) if value > 0 else 2.0
                    rect = QRectF(x, top + ph - bh, bw, bh)
                    g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
                    g.setColorAt(0, QColor("#0E7C86"))
                    g.setColorAt(1, QColor(14, 124, 134, 120))
                    p.setPen(Qt.NoPen)
                    p.setBrush(g if value > 0 else QColor(18, 40, 47, 40))
                    p.drawRoundedRect(rect, 5, 5)
                    p.setPen(QColor(INK))
                    p.drawText(QRectF(x - 10, rect.top() - 18, bw + 20, 16), Qt.AlignCenter, str(int(value)))
                p.setPen(QColor(SUB))
                p.drawText(QRectF(cx - slot / 2, h - bottom + 8, slot, 16), Qt.AlignCenter,
                           fm.elidedText(str(label), Qt.ElideRight, int(slot) - 4))

    class DonutChart(ChartBase):
        """Doughnut chart; items are (name, count, hex_color). Legend shows counts and percentages."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.data = []

        def set_message(self, text: str):
            self.data = []
            self.message = text
            self.update()

        def set_data(self, items):
            self.data = [(n, v, c) for n, v, c in items if v > 0]
            self.update()

        def paintEvent(self, _event):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            total = sum(v for _, v, _ in self.data)
            if total <= 0:
                self._paint_message(p)
                return
            size = max(80.0, min(h - 16, w * 0.48))
            thick = size * 0.17
            ring = QRectF(8 + thick / 2, (h - size) / 2 + thick / 2, size - thick, size - thick)
            start = 90 * 16
            for name, value, color in self.data:
                span = -int(value / total * 360 * 16)
                p.setPen(QPen(QColor(color), thick, Qt.SolidLine, Qt.FlatCap))
                p.drawArc(ring, start, span + (16 if len(self.data) > 1 else 0))
                start += span
            f = p.font()
            f.setPointSizeF(20)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(INK))
            p.drawText(QRectF(ring.x(), ring.center().y() - 24, ring.width(), 30), Qt.AlignCenter, str(total))
            f.setPointSizeF(9)
            f.setBold(False)
            p.setFont(f)
            p.setPen(QColor(SUB))
            p.drawText(QRectF(ring.x(), ring.center().y() + 6, ring.width(), 18), Qt.AlignCenter, "unique IDs")
            f.setPointSizeF(9.5)
            p.setFont(f)
            lx = 8 + size + 24
            avail = int(w - lx - 24)
            ly = h / 2 - len(self.data) * 15
            fm = p.fontMetrics()
            for i, (name, value, color) in enumerate(self.data):
                y = ly + i * 30
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(color))
                p.drawRoundedRect(QRectF(lx, y + 5, 12, 12), 3, 3)
                p.setPen(QColor(INK))
                text = f"{name} · {value} ({value / total * 100:.0f}%)"
                p.drawText(QRectF(lx + 20, y, max(40, avail - 20), 22), Qt.AlignLeft | Qt.AlignVCenter,
                           fm.elidedText(text, Qt.ElideRight, max(40, avail - 20)))

    SPECIES_COLORS = ["#0E7C86", "#6B8FD6", "#9B7FD1", "#E0A458", "#7BC4A4"]
    OTHER_COLOR, UNCONFIRMED_COLOR = "#B7A6DE", "#9AA8AD"
    RANGES = [("Last 12 hours", 12), ("Last 24 hours", 24), ("Last 7 days", 168)]
    ICON_UNIQUE = '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="9"/>'
    ICON_SPECIES = '<path d="M11 20A7 7 0 019.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10z"/><path d="M2 21c0-3 1.9-5.5 6-7"/>'
    ICON_PEAK = '<path d="M3 17l6-6 4 4 8-8M15 7h6v6"/>'
    ICON_CONF = '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>'

    class AnalyticsPage(QWidget):
        """Surveillance analytics built ONLY from data the existing /api/status poll provides."""
        refresh_requested = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.store = AnalyticsStore()
            self.conn = "connecting"
            self._last_key = None

            outer = QVBoxLayout(self)
            outer.setContentsMargins(12, 8, 8, 8)
            outer.setSpacing(14)
            outer.addLayout(page_header("Analytics", "Mosquito surveillance trends for the chamber"))

            # Filter row
            bar = glass()
            bh = QHBoxLayout(bar)
            bh.setContentsMargins(18, 10, 18, 10)
            self.cmb_range = QComboBox()
            for text, hrs in RANGES:
                self.cmb_range.addItem(text, hrs)
            self.cmb_range.setFixedWidth(170)
            self.cmb_range.currentIndexChanged.connect(lambda _i: self.refresh(force=True))
            btn = QPushButton("Refresh")
            btn.setProperty("class", "ghost")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(self._on_refresh_clicked)
            self.lbl_updated = lbl("Not updated yet", 12, 500, SUB)
            self.lbl_note = lbl("", 12, 400, SUB, wrap=True)
            bh.addWidget(self.cmb_range)
            bh.addWidget(btn)
            bh.addSpacing(8)
            bh.addWidget(self.lbl_note, stretch=1)
            bh.addWidget(self.lbl_updated)
            outer.addWidget(bar)

            # Scrollable content
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setStyleSheet("QScrollArea{background:transparent;border:none;} "
                                 "QScrollArea > QWidget > QWidget{background:transparent;}")
            content = QWidget()
            col = QVBoxLayout(content)
            col.setContentsMargins(0, 0, 8, 8)
            col.setSpacing(16)

            grid = QGridLayout()
            grid.setHorizontalSpacing(16)
            self.card_unique = MetricCard("Unique mosquitoes", "—", "Distinct IDs", icon_svg=ICON_UNIQUE, value_size=22)
            self.card_species = MetricCard("Dominant species", "—", "Most frequent", icon_svg=ICON_SPECIES, value_size=22)
            self.card_peak = MetricCard("Peak activity", "—", "Busiest 30 min", icon_svg=ICON_PEAK, value_size=22)
            self.card_conf = MetricCard("Average confidence", "—", "Recorded scores", icon_svg=ICON_CONF, value_size=22)
            for i, c in enumerate((self.card_unique, self.card_species, self.card_peak, self.card_conf)):
                grid.addWidget(c, 0, i)
                grid.setColumnStretch(i, 1)
            col.addLayout(grid)

            # Middle row
            self.line_chart, self.donut = LineChart(), DonutChart()
            row2 = QHBoxLayout()
            row2.setSpacing(16)
            p1, self.sub_line, _ = self._panel("Mosquito Activity Over Time", "", self.line_chart)
            p2, self.sub_donut, _ = self._panel("Species Breakdown", "Unique IDs by recorded species label", self.donut)
            row2.addWidget(p1, stretch=3)
            row2.addWidget(p2, stretch=2)
            col.addLayout(row2)

            # Bottom row
            self.tod_chart, self.cusum_chart, self.conf_chart = BarChart(), LineChart(), BarChart()
            row3 = QHBoxLayout()
            row3.setSpacing(16)
            p3, self.sub_tod, _ = self._panel("Activity by Time of Day", "Unique IDs per monitoring window reported by the detector", self.tod_chart)
            p4, self.sub_cusum, self.cusum_pill = self._panel("CUSUM Activity Monitoring",
                                                              "Activity-monitoring signal (S+), not an outbreak confirmation",
                                                              self.cusum_chart, with_pill=True)
            self.conf_panel, _s, _p = self._panel("Confidence Distribution", "Recorded scores per unique ID", self.conf_chart)
            row3.addWidget(p3, stretch=3)
            row3.addWidget(p4, stretch=4)
            row3.addWidget(self.conf_panel, stretch=3)
            col.addLayout(row3)
            col.addStretch()
            scroll.setWidget(content)
            outer.addWidget(scroll, stretch=1)

            self._timer = QTimer(self)  # keeps the time axis moving
            self._timer.timeout.connect(self.refresh)
            self._timer.start(30000)
            self.refresh(force=True)

        def _panel(self, title, subtitle, chart, with_pill=False):
            panel = glass()
            v = QVBoxLayout(panel)
            v.setContentsMargins(22, 20, 22, 16)
            v.setSpacing(2)
            head = QHBoxLayout()
            head.addWidget(lbl(title, 16, 700))
            head.addStretch()
            pl = None
            if with_pill:
                pl = QLabel("")
                head.addWidget(pl)
            v.addLayout(head)
            sub = lbl(subtitle, 12, 400, SUB, wrap=True)
            v.addWidget(sub)
            v.addWidget(chart, stretch=1)
            return panel, sub, pl

        # ── data in ────────────────────────────────────────────────────────
        def record_status(self, data: Dict[str, Any]):
            try:
                self.store.record(data)
                self.refresh()
            except Exception as exc:  # analytics must never crash the app
                print(f"[WARNING] Analytics update skipped: {type(exc).__name__}: {exc}")

        def set_connection(self, state: str):
            mapping = {"connected": "connected", "unauthorized": "unauthorized",
                       "disconnected": "disconnected",
                       "connection_refused": "disconnected", "timeout": "disconnected",
                       "network_error": "disconnected", "malformed_json": "malformed"}
            new = mapping.get(state, "error")
            if new != self.conn:
                self.conn = new
                self.refresh()

        def _on_refresh_clicked(self):
            self.refresh_requested.emit()   # poll the Pi right away
            self.refresh(force=True)

        # ── rendering ──────────────────────────────────────────────────────
        def _no_data_message(self) -> str:
            return {
                "unauthorized": "Unauthorized. Check the access token in client_config.json.",
                "disconnected": "Disconnected from Raspberry Pi. No data has been collected yet.",
                "malformed": "The detector returned unreadable data. No analytics available yet.",
                "error": "Connection error. No analytics available yet.",
                "connecting": "Waiting for the first telemetry update…",
            }.get(self.conn, "Historical data unavailable for this period.")

        def _conn_reason(self) -> str:
            return {"unauthorized": "unauthorized (check the access token).",
                    "disconnected": "disconnected from the Raspberry Pi.",
                    "malformed": "the detector returned unreadable data.",
                    "error": "connection error.",
                    "connecting": "waiting for the first update."}.get(self.conn, "")

        def refresh(self, force: bool = False):
            hours = self.cmb_range.currentData() or 12
            now = datetime.now()
            key = (self.store.version, hours, self.conn, floor_slot(now))
            if not force and key == self._last_key:
                return
            self._last_key = key
            try:
                self._render(hours, now)
                self.lbl_updated.setText(f"Updated {now:%H:%M:%S}")
            except Exception as exc:
                print(f"[WARNING] Analytics render failed: {type(exc).__name__}: {exc}")

        def _render(self, hours: int, now: datetime):
            st = self.store
            range_text = dict((h, t) for t, h in RANGES)[hours].lower()
            series = st.series(hours, now)
            covered = any(v is not None for _, v in series)
            ids = st.unique_ids(hours, now)
            no_cov = self._no_data_message()
            no_det = "No detections recorded in this period." if covered else no_cov

            note = f"Live telemetry since {st.started:%d %b %H:%M}. Earlier history isn’t available from the Pi yet."
            if covered and self.conn in ("disconnected", "malformed", "error"):
                note += " Disconnected: showing data collected before the connection dropped."
            self.lbl_note.setText(note)

            # Cards
            if covered:
                self.card_unique.update_value(str(len(ids)), f"Distinct IDs · {range_text}")
                self.card_unique.set_accent_color(INK)
            else:
                self.card_unique.update_value("—", "No observation coverage")
                self.card_unique.set_accent_color(MUTED)
            dom = st.dominant_species(hours, now)
            if dom:
                self.card_species.update_value(dom[0], f"{dom[1]} of {dom[2]} unique IDs")
                self.card_species.set_accent_color(INK)
            elif ids:
                self.card_species.update_value("Unknown", "No confirmed species recorded")
                self.card_species.set_accent_color(MUTED)
            else:
                self.card_species.update_value("—", "No detections yet" if covered else "No observation coverage")
                self.card_species.set_accent_color(MUTED)
            peak = st.peak(hours, now)
            if peak:
                self.card_peak.update_value(f"{peak[0]:%H:%M}–{peak[0] + SLOT:%H:%M}", f"{peak[1]} unique IDs")
                self.card_peak.set_accent_color(INK)
            else:
                self.card_peak.update_value("—", "No detections yet" if covered else "No observation coverage")
                self.card_peak.set_accent_color(MUTED)
            avg = st.avg_confidence(hours, now)
            if avg:
                self.card_conf.update_value(f"{avg[0] * 100:.1f}%", f"{avg[1]} of {avg[2]} IDs have a score")
                self.card_conf.set_accent_color(INK)
            else:
                self.card_conf.update_value("N/A", "Confidence data not recorded." if ids else "No detections yet")
                self.card_conf.set_accent_color(MUTED)

            # Chart 1: activity over time (unobserved slots = hatched gaps, never zero)
            self.sub_line.setText(f"Unique IDs per 30 min · {range_text}" + ("" if ids or not covered else " · No detections recorded in this period."))
            if not covered:
                self.line_chart.set_message(no_cov)
            else:
                half = SLOT / 2
                pts = [(s + half, v) for s, v in series]
                bands = [(s, s + SLOT) for s, v in series if v is None]
                self.line_chart.set_data(series[0][0], series[-1][0] + SLOT, pts, bands=bands, integer_y=True)

            # Chart 2: species breakdown
            if not ids:
                self.donut.set_message(no_det)
            else:
                items, named = [], 0
                for name, count, kind in st.species_breakdown(hours, now):
                    if kind == "named":
                        color = SPECIES_COLORS[named % len(SPECIES_COLORS)]
                        named += 1
                    else:
                        color = OTHER_COLOR if kind == "other" else UNCONFIRMED_COLOR
                    items.append((name, count, color))
                self.donut.set_data(items)

            # Chart 3: time of day (windows exactly as reported by the detector)
            win = st.window_counts(hours, now)
            if not covered:
                self.tod_chart.set_message(no_cov)
            elif win is None:
                self.tod_chart.set_message("Time-of-day data unavailable: the detector isn’t reporting a monitoring window.")
            else:
                self.tod_chart.set_data([(w.replace("_", " ").title(), c) for w, c in win])

            # Chart 4: CUSUM (values straight from the detector; nothing recomputed here)
            pts_c, alerts = st.cusum_points(hours, now)
            self._update_cusum_pill()
            if not pts_c and not alerts:
                if self.conn != "connected":
                    self.cusum_chart.set_message(f"CUSUM data unavailable: {self._conn_reason()}")
                elif not st.cusum_seen:
                    self.cusum_chart.set_message("CUSUM data unavailable: the detector isn’t reporting CUSUM telemetry.")
                elif st.cusum_state:
                    self.cusum_chart.set_message(f"CUSUM score unavailable. Detector state: {st.cusum_state.replace('_', ' ').title()}.")
                else:
                    self.cusum_chart.set_message("CUSUM history unavailable for this period.")
            else:
                th = st.threshold
                markers = [(a["time"], a["s_plus"] if a["s_plus"] is not None else (th or 0.0)) for a in alerts]
                self.cusum_chart.set_data(now - timedelta(hours=hours), now, pts_c, integer_y=False,
                                          threshold=th, threshold_label=f"Alert threshold h = {th:g}" if th is not None else "",
                                          markers=markers, break_gap=timedelta(minutes=5))

            # Optional chart 5: only when real confidence scores exist
            bins = st.confidence_bins(hours, now)
            self.conf_panel.setVisible(bins is not None)
            if bins is not None:
                self.conf_chart.set_data(bins)

        def _update_cusum_pill(self):
            st = self.store
            colors = {"MONITORING": (OK, "#DDF3E4"), "ALERT_TRIGGERED": (BAD, "#FDE3E0"),
                      "COLLECTING_BASELINE": (WARN, "#FBEFD5"), "OFFLINE": (BAD, "#FDE3E0")}
            if st.cusum_state and self.conn == "connected":
                fg, bg = colors.get(st.cusum_state.upper(), (MUTED, "#E8EEF0"))
                text = f"State: {st.cusum_state.replace('_', ' ').title()}"
            elif st.cusum_state:
                fg, bg, text = MUTED, "#E8EEF0", f"Last state: {st.cusum_state.replace('_', ' ').title()}"
            else:
                fg, bg, text = MUTED, "#E8EEF0", "CUSUM data unavailable"
            self.cusum_pill.setText(text)
            self.cusum_pill.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:12px; padding:4px 12px; font-size:12px; font-weight:600;")

    class CusumAlertsPage(QWidget):
        """CUSUM activity monitoring: state cards, decision parameters, alert history."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.init_ui()

        def init_ui(self):
            root = QVBoxLayout(self)
            root.setContentsMargins(12, 8, 8, 8)
            root.setSpacing(18)
            root.addLayout(page_header("Activity Alerts",
                                       "CUSUM monitoring detects sustained rises in chamber activity (30-min intervals)"))

            grid = QGridLayout()
            grid.setHorizontalSpacing(16)
            self.card_state = MetricCard("Surveillance state", "INSUFFICIENT DATA", "Awaiting CUSUM telemetry…")
            self.card_s_plus = MetricCard("CUSUM score (S+)", "S+ = N/A", "Threshold: h = 5.0 | Slack: k = 0.5")
            self.card_activity = MetricCard("Activity, last 30 min", "N/A", "Unique confirmed IDs in interval")
            self.card_baseline = MetricCard("Daily baseline", "N/A", "Mean: N/A | Std: N/A")
            for i, c in enumerate((self.card_state, self.card_s_plus, self.card_activity, self.card_baseline)):
                grid.addWidget(c, 0, i)
                grid.setColumnStretch(i, 1)
            root.addLayout(grid)

            lower = QHBoxLayout()
            lower.setSpacing(16)

            diag = glass()
            dv = QVBoxLayout(diag)
            dv.setContentsMargins(24, 22, 24, 18)
            dv.addWidget(lbl("Detection settings", 16, 700))
            self.diag_labels = add_info_rows(dv, [
                ("baseline_status", "Baseline status", "N/A"),
                ("window_type", "Current diurnal window", "N/A"),
                ("threshold_h", "Alert threshold (h)", "5.00"),
                ("reference_k", "Allowance (k)", "0.50"),
                ("min_std", "Minimum std. floor, max(σ, 0.5)", "0.50"),
                ("last_alert_time", "Latest alert", "None"),
                ("last_alert_reason", "Alert description", "No active alerts"),
            ])
            for k in ("threshold_h", "reference_k", "min_std"):
                set_value(self.diag_labels[k], self.diag_labels[k].text(), INK)
            dv.addStretch()
            lower.addWidget(diag, stretch=1)

            hist = glass()
            hv = QVBoxLayout(hist)
            hv.setContentsMargins(24, 22, 24, 18)
            hv.addWidget(lbl("Alert history", 16, 700))
            self.alerts = TableWithEmpty(["Time", "Window", "Activity", "S+"],
                                         "No active alerts",
                                         "Alerts will be logged here when activity rises above baseline.")
            hv.addWidget(self.alerts, stretch=1)
            lower.addWidget(hist, stretch=1)
            root.addLayout(lower, stretch=1)

        def update_cusum(self, cusum_data: Optional[Dict[str, Any]]):
            """Update from the 'cusum' block of the status payload."""
            if not cusum_data or not isinstance(cusum_data, dict):
                self.card_state.update_value("INSUFFICIENT DATA", "Awaiting CUSUM telemetry…")
                self.card_state.set_accent_color(MUTED)
                self.card_s_plus.update_value("S+ = N/A", "Threshold: h = 5.0 | Slack: k = 0.5")
                self.card_s_plus.set_accent_color(MUTED)
                self.card_activity.update_value("N/A", "Unique confirmed IDs in interval")
                self.card_activity.set_accent_color(MUTED)
                self.card_baseline.update_value("N/A", "Mean: N/A | Std: N/A")
                self.card_baseline.set_accent_color(MUTED)
                for k in ("baseline_status", "window_type", "last_alert_time", "last_alert_reason"):
                    set_value(self.diag_labels[k], "N/A", MUTED)
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

            # State card
            if state == "ALERT_TRIGGERED":
                self.card_state.update_value("ALERT TRIGGERED", f"Window: {w_type} (sustained increase)")
                self.card_state.set_accent_color(BAD)
            elif state == "MONITORING":
                self.card_state.update_value("MONITORING", f"Active window: {w_type}")
                self.card_state.set_accent_color(OK)
            elif state == "COLLECTING_BASELINE":
                self.card_state.update_value("COLLECTING", f"Baseline (Day {b_days}/{b_days_req}, N={b_n}/12)")
                self.card_state.set_accent_color(WARN)
            elif state == "OFFLINE":
                self.card_state.update_value("DETECTOR OFFLINE", "Observation collection paused")
                self.card_state.set_accent_color(BAD)
            else:
                self.card_state.update_value("INSUFFICIENT DATA", f"Window: {w_type}")
                self.card_state.set_accent_color(MUTED)

            # S+ card
            if isinstance(s_plus, (int, float)):
                self.card_s_plus.update_value(f"S+ = {s_plus:.2f}", f"Threshold: h = {thresh_h:.1f} | Slack: k = {ref_k:.1f}")
                self.card_s_plus.set_accent_color(BAD if s_plus >= thresh_h else TEAL)
            else:
                self.card_s_plus.update_value("S+ = N/A", f"Threshold: h = {thresh_h:.1f} | Slack: k = {ref_k:.1f}")
                self.card_s_plus.set_accent_color(MUTED)

            # Activity card
            if isinstance(activity, (int, float)):
                self.card_activity.update_value(f"{int(activity)} IDs", "Unique IDs in 30-min window")
                self.card_activity.set_accent_color(INK)
            else:
                self.card_activity.update_value("N/A", "No valid interval observation")
                self.card_activity.set_accent_color(MUTED)

            # Baseline card
            if isinstance(b_mean, (int, float)) and b_std is not None:
                self.card_baseline.update_value(f"μ₀ {b_mean:.2f} · σ₀ {b_std:.2f}",
                                                f"N = {b_n} valid intervals ({'Ready' if b_ready else 'Collecting'})")
                self.card_baseline.set_accent_color(INK)
            else:
                self.card_baseline.update_value("N/A", f"Collecting (Day {b_days}/{b_days_req}, N={b_n}/12)")
                self.card_baseline.set_accent_color(MUTED)

            # Detail rows
            set_value(self.diag_labels["baseline_status"],
                      f"Ready (7 days, N = {b_n})" if b_ready else f"Collecting (Day {b_days}/{b_days_req}, N = {b_n}/12)",
                      OK if b_ready else WARN)
            set_value(self.diag_labels["window_type"], w_type, INK)
            set_value(self.diag_labels["threshold_h"], f"{thresh_h:.2f}", INK)
            set_value(self.diag_labels["reference_k"], f"{ref_k:.2f}", INK)
            set_value(self.diag_labels["last_alert_time"], alert_time if alert_time else "None", INK if alert_time else MUTED)
            set_value(self.diag_labels["last_alert_reason"],
                      alert_reason[:45] if alert_reason else "No active alerts", INK if alert_reason else MUTED)

            # Alert history
            rows = []
            history = cusum_data.get("alert_history", [])
            if isinstance(history, list):
                for rec in history:
                    if not isinstance(rec, dict):
                        continue
                    act = rec.get("activity_value")
                    s_val = rec.get("s_plus")
                    rows.append([
                        str(rec.get("timestamp", "N/A")),
                        str(rec.get("window_type", "N/A")).capitalize(),
                        f"{int(act)} IDs" if isinstance(act, (int, float)) else str(act or "N/A"),
                        f"{float(s_val):.2f}" if isinstance(s_val, (int, float)) else str(s_val or "N/A"),
                    ])
            self.alerts.set_rows(rows)

        def set_offline_state(self, err_type: str, description: str):
            """Polling failed or the detector went offline."""
            self.card_state.update_value("OFFLINE", "Telemetry unavailable")
            self.card_state.set_accent_color(BAD)
            self.card_s_plus.update_value("S+ = N/A", "Detector connection offline")
            self.card_s_plus.set_accent_color(MUTED)
            self.card_activity.update_value("N/A", "No observations")
            self.card_activity.set_accent_color(MUTED)
            self.card_baseline.update_value("PAUSED", "Waiting for detector")
            self.card_baseline.set_accent_color(MUTED)
            set_value(self.diag_labels["baseline_status"], "Offline", BAD)
            set_value(self.diag_labels["window_type"], "N/A", MUTED)
            # Keep any existing alert rows so history stays readable while offline.

    class SettingsPage(QWidget):
        """Read-only view of client_config.json."""

        def __init__(self, config: Dict[str, Any], parent=None):
            super().__init__(parent)
            self.config = config
            root = QVBoxLayout(self)
            root.setContentsMargins(12, 8, 8, 8)
            root.setSpacing(18)
            root.addLayout(page_header("Settings", "Connection to the Raspberry Pi detector"))

            panel = glass()
            panel.setMaximumWidth(760)
            v = QVBoxLayout(panel)
            v.setContentsMargins(24, 22, 24, 14)
            v.addWidget(lbl("Connection", 16, 700))
            token_set = bool(config.get("token"))
            rows = add_info_rows(v, [
                ("host", "Pi host", str(config.get("pi_host", "127.0.0.1"))),
                ("port", "Port", str(config.get("pi_port", 8080))),
                ("token", "Access token", "Configured (hidden)" if token_set else "Not configured"),
                ("poll", "Polling interval", f"{config.get('poll_interval_sec', 1.0):.1f} s"),
                ("timeout", "Request timeout", f"{config.get('timeout_sec', 2.5):.1f} s"),
            ])
            for k, w in rows.items():
                set_value(w, w.text(), INK if (k != "token" or token_set) else MUTED)

            note = glass()
            note.setMaximumWidth(760)
            n = QHBoxLayout(note)
            n.setContentsMargins(24, 18, 24, 18)
            box = QVBoxLayout()
            box.setSpacing(2)
            box.addWidget(lbl("Changing these values", 13, 700))
            self.lbl_note = lbl("Edit the local client_config.json file (untracked in Git), then restart the app.",
                                12, 400, SUB, wrap=True)
            box.addWidget(self.lbl_note)
            btn = QPushButton("Open config file")
            btn.setProperty("class", "ghost")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(self.open_config)
            n.addLayout(box, stretch=1)
            n.addWidget(btn)

            root.addWidget(panel)
            root.addWidget(note)
            root.addStretch()

        def open_config(self):
            path = default_config_path()
            if os.path.isfile(path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            else:
                self.lbl_note.setText(f"client_config.json not found. Create it next to gui_app.py: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Application Window
# ─────────────────────────────────────────────────────────────────────────────
if PYSIDE6_AVAILABLE:
    class MosquitoSentinelWindow(QMainWindow):
        """Glass sidebar + stacked pages + background telemetry polling."""

        def __init__(self, client_cfg: Optional[Dict[str, Any]] = None):
            super().__init__()
            self.client_config = client_cfg if client_cfg is not None else load_client_config()

            self.setWindowTitle("MosquitoSentinel - Chamber Surveillance & Classification")
            self.setMinimumSize(1100, 700)
            self.resize(1360, 840)

            central = GradientBackground()
            self.setCentralWidget(central)
            root = QHBoxLayout(central)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(12)

            root.addWidget(self._build_sidebar())

            # Pages
            self.stack = QStackedWidget()
            self.overview_page = OverviewPage()
            self.live_page = LiveDetectionPage()
            self.history_page = HistoryPage()
            self.analytics_page = AnalyticsPage()
            self.alerts_page = CusumAlertsPage()
            self.settings_page = SettingsPage(self.client_config)
            for page in (self.overview_page, self.live_page, self.history_page,
                         self.analytics_page, self.alerts_page, self.settings_page):
                self.stack.addWidget(page)
            root.addWidget(self.stack, stretch=1)

            # Background poller
            self.polling_worker = StatusPollingWorker(self.client_config)
            self.polling_worker.sig_status_updated.connect(self.on_status_updated)
            self.polling_worker.sig_connection_error.connect(self.on_connection_error)
            self.overview_page.retry_requested.connect(self.polling_worker.poll_now)
            self.analytics_page.refresh_requested.connect(self.polling_worker.poll_now)
            self.polling_worker.start()

            # Live camera frame worker (active only on Live Detection page)
            self.frame_worker = FrameFetchWorker(self.client_config)
            self.frame_worker.sig_frame_received.connect(self.live_page.on_frame_received)
            self.frame_worker.sig_diagnostics_updated.connect(self.live_page.on_diagnostics_updated)
            self.frame_worker.sig_frame_error.connect(self.live_page.on_frame_error)
            self.frame_worker.start()

            self.switch_page(0)

        def _build_sidebar(self) -> QFrame:
            sidebar = glass(36)
            sidebar.setFixedWidth(244)
            v = QVBoxLayout(sidebar)
            v.setContentsMargins(14, 22, 14, 16)
            v.setSpacing(4)

            # Brand
            brand = QHBoxLayout()
            brand.setContentsMargins(6, 0, 6, 0)
            brand.setSpacing(12)
            logo = QLabel("✚")
            logo.setFixedSize(38, 38)
            logo.setAlignment(Qt.AlignCenter)
            logo.setStyleSheet(f"background:{TEAL}; color:#FFFFFF; border-radius:11px; font-size:18px; font-weight:800;")
            names = QVBoxLayout()
            names.setSpacing(0)
            names.addWidget(lbl("MosquitoSentinel", 16, 700))
            names.addWidget(lbl("Chamber surveillance", 12, 400, SUB))
            brand.addWidget(logo)
            brand.addLayout(names)
            v.addLayout(brand)
            v.addSpacing(22)

            # Navigation
            self.nav_buttons = []
            for idx, name in enumerate(NAV_ICONS):
                btn = QPushButton(f"  {name}")
                btn.setProperty("class", "nav")
                btn.setCheckable(True)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setIcon(svg_icon(NAV_ICONS[name]))
                btn.setIconSize(QSize(18, 18))
                btn.clicked.connect(lambda _checked, i=idx: self.switch_page(i))
                v.addWidget(btn)
                self.nav_buttons.append(btn)
            v.addStretch()

            # Connection card
            card = QFrame()
            card.setProperty("class", "inner")
            c = QVBoxLayout(card)
            c.setContentsMargins(14, 12, 14, 12)
            c.setSpacing(4)
            self.lbl_conn_stat = lbl("● Connecting…", 13, 700, TEAL)
            self.lbl_server_info = lbl(f"Server {self.client_config['pi_host']}:{self.client_config['pi_port']}", 12, 400, SUB)
            c.addWidget(self.lbl_conn_stat)
            c.addWidget(self.lbl_server_info)
            c.addWidget(lbl("Telemetry polling · Stage 5A", 12, 400, SUB))
            v.addWidget(card)
            return sidebar

        def _set_conn(self, text: str, color: str):
            self.lbl_conn_stat.setText(f"● {text}")
            self.lbl_conn_stat.setStyleSheet(f"font-size:13px; font-weight:700; color:{color};")

        def switch_page(self, index: int):
            self.stack.setCurrentIndex(index)
            for idx, btn in enumerate(self.nav_buttons):
                btn.setChecked(idx == index)
            if getattr(self, "frame_worker", None) is not None:
                if index == 1:
                    self.frame_worker.resume()
                else:
                    self.frame_worker.pause()
            if index == 2 and getattr(self, "history_page", None) is not None:
                self.history_page.fetch_registry(self.client_config)

        @Slot(dict)
        def on_status_updated(self, data: Dict[str, Any]):
            self.overview_page.update_telemetry(data)
            self.alerts_page.update_cusum(data.get("cusum"))
            self.analytics_page.record_status(data)
            self.analytics_page.set_connection("connected")
            if self.stack.currentIndex() == 2 and getattr(self, "history_page", None) is not None:
                self.history_page.fetch_registry(self.client_config)
            self._set_conn("Connected", OK)

        @Slot(str, str)
        def on_connection_error(self, err_type: str, description: str):
            self.overview_page.set_error_state(err_type, description)
            self.alerts_page.set_offline_state(err_type, description)
            self.analytics_page.set_connection(err_type)
            if err_type == "unauthorized":
                self._set_conn("Unauthorized (401)", WARN)
            elif err_type in ("connection_refused", "timeout", "network_error"):
                self._set_conn("Disconnected", BAD)
            elif err_type == "malformed_json":
                self._set_conn("Invalid response", BAD)
            else:
                self._set_conn("Connection error", BAD)

        def closeEvent(self, event):
            """Stop background worker threads cleanly."""
            if getattr(self, "polling_worker", None) is not None:
                self.polling_worker.stop()
                self.polling_worker.wait(3000)
            if getattr(self, "frame_worker", None) is not None:
                self.frame_worker.stop()
                self.frame_worker.wait(3000)
            event.accept()


def main():
    if not PYSIDE6_AVAILABLE:
        print("[ERROR] PySide6 is not installed in the current Python environment.")
        print("Please install it with: pip install -r requirements-gui.txt")
        sys.exit(1)

    print("[INFO] Launching MosquitoSentinel Desktop GUI (Stage 5A, light glass theme)...")
    app = QApplication(sys.argv)
    app.setStyleSheet(LIGHT_STYLE_SHEET)
    window = MosquitoSentinelWindow()

    screen = app.primaryScreen()
    if screen is not None:
        geo = window.frameGeometry()
        geo.moveCenter(screen.availableGeometry().center())
        window.move(geo.topLeft())

    window.show()
    window.raise_()
    window.activateWindow()

    print("[INFO] MosquitoSentinel Window initialized successfully.")
    print("       Press Ctrl+C in this terminal or close the window to exit.")

    # Lets the Python interpreter process Ctrl+C while Qt's loop runs
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