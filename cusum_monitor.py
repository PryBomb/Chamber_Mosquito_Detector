"""
cusum_monitor.py - Cumulative Sum (CUSUM) Activity Monitor
============================================================
Statistical quality-control monitor for detecting sustained increases in
observed chamber mosquito activity in the MosquitoSentinel project.

Features & Mathematical Specification:
--------------------------------------
1. Time Window & Diurnal Stratification:
   - Evaluates observations over discrete intervals (default: 30 minutes).
   - Separates Morning (default 06:00-12:00) and Afternoon (12:00-18:00) baselines.
   - Nighttime / off-window intervals (18:00-06:00) are flagged and never folded
     into morning or afternoon baselines.

2. One-Sided Upper CUSUM Statistic:
   For observation x_t, baseline mean mu_0, baseline standard deviation sigma_0,
   and reference allowance k (default k = 0.5):
       z_t = (x_t - mu_0) / max(sigma_0, 0.5)
       S_t^+ = max(0, S_{t-1}^+ + z_t - k)

   An alert is triggered when S_t^+ >= h (default h = 5.0), provided that the
   baseline is ready and the observation is valid.

3. Activity Metric:
   - Metric: "interval_unique_confirmed_mosquito_ids"
   - Deduplication: Within each 30-minute interval, identical tracking IDs
     are counted once, regardless of frame count or detection hits.
   - Missing observations (e.g. detector stopped, communication loss) are
     flagged as MISSING with activity=None; they are NEVER converted to zero.
   - Zero observed mosquitoes during an active, fully-covered interval is a valid 0.

4. Production Safety:
   - Preserves all existing models, SQLite tables, CSV logs, and video loops.
   - Uses an isolated SQLite database (cusum_history.db) or in-memory DB for tests.

5. Baseline Lifecycle & Coverage Invariants:
   - Baseline Collection Start: Begins with the first valid interval observed within
     the respective diurnal window (morning 06:00-12:00 or afternoon 12:00-18:00).
   - Baseline Readiness Requirement: Baseline monitoring CANNOT freeze or become active
     until BOTH requirements are satisfied:
       a) At least 7 distinct calendar days of valid baseline data (baseline_days >= 7).
       b) At least 12 valid observation intervals (min_baseline_observations >= 12).
     The baseline CANNOT freeze after only 12 observations on Day 1.
   - Insufficient Coverage Days: Any interval with coverage_ratio < min_coverage_ratio (0.70)
     or where the detector was offline is marked is_valid = 0 and activity_count = NULL.
     It is NEVER treated as zero activity, does not contribute to baseline statistics,
     and does not count toward the distinct calendar day requirement. Baseline collection
     simply extends until 7 distinct calendar days with valid coverage are accumulated.
   - Diurnal Independence: Morning and afternoon baselines are calculated and tracked
     completely independently.
"""

import os
import sys
import math
import json
import sqlite3
import threading
from datetime import datetime, date, time as dt_time, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple


try:
    import config
except ImportError:
    config = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration Dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CusumConfig:
    """Configurable parameters for CUSUM statistical monitoring."""
    interval_minutes: int = getattr(config, "CUSUM_INTERVAL_MINUTES", 30) if config else 30
    baseline_days: int = getattr(config, "CUSUM_BASELINE_DAYS", 7) if config else 7
    min_baseline_observations: int = getattr(config, "CUSUM_MIN_BASELINE_OBSERVATIONS", 12) if config else 12
    k: float = getattr(config, "CUSUM_K", 0.5) if config else 0.5
    h: float = getattr(config, "CUSUM_H", 5.0) if config else 5.0
    min_std: float = getattr(config, "CUSUM_MIN_STD", 0.5) if config else 0.5
    morning_start_hour: int = getattr(config, "CUSUM_MORNING_START_HOUR", 6) if config else 6
    morning_end_hour: int = getattr(config, "CUSUM_MORNING_END_HOUR", 12) if config else 12
    afternoon_start_hour: int = getattr(config, "CUSUM_AFTERNOON_START_HOUR", 12) if config else 12
    afternoon_end_hour: int = getattr(config, "CUSUM_AFTERNOON_END_HOUR", 18) if config else 18
    min_coverage_ratio: float = getattr(config, "CUSUM_MIN_COVERAGE_RATIO", 0.70) if config else 0.70



# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema DDL for Dedicated Isolated CUSUM Storage
# ─────────────────────────────────────────────────────────────────────────────
_CUSUM_DDL = """
CREATE TABLE IF NOT EXISTS cusum_intervals (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    interval_start     TEXT    NOT NULL UNIQUE,
    interval_end       TEXT    NOT NULL,
    window_type        TEXT    NOT NULL,  -- 'morning', 'afternoon', 'night'
    activity_count     INTEGER,           -- NULL if invalid/missing
    unique_ids_json    TEXT    NOT NULL,  -- JSON list of distinct tracking IDs
    species_counts_json TEXT   NOT NULL,  -- JSON dict of species -> count
    coverage_ratio     REAL    NOT NULL,  -- Fraction of interval actively monitored (0.0 - 1.0)
    is_valid           INTEGER NOT NULL,  -- 1 if valid, 0 if missing/incomplete
    is_baseline        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cusum_interval_start ON cusum_intervals(interval_start);
CREATE INDEX IF NOT EXISTS idx_cusum_window_valid ON cusum_intervals(window_type, is_valid);

CREATE TABLE IF NOT EXISTS cusum_state (
    window_type        TEXT PRIMARY KEY,  -- 'morning' or 'afternoon'
    s_plus             REAL NOT NULL DEFAULT 0.0,
    in_alert           INTEGER NOT NULL DEFAULT 0,
    last_updated       TEXT,
    last_alert_time    TEXT,
    last_alert_reason  TEXT
);

CREATE TABLE IF NOT EXISTS cusum_alerts (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT    NOT NULL,
    window_type        TEXT    NOT NULL,
    activity_value     REAL    NOT NULL,
    baseline_mean      REAL    NOT NULL,
    baseline_std       REAL    NOT NULL,
    s_plus             REAL    NOT NULL,
    threshold_h        REAL    NOT NULL,
    reason             TEXT    NOT NULL,
    status             TEXT    NOT NULL  -- 'TRIGGERED', 'ONGOING', 'RESOLVED'
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. Status and Data Transfer Objects
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CusumStatus:
    state: str                 # "COLLECTING_BASELINE", "MONITORING", "ALERT_TRIGGERED", "INSUFFICIENT_DATA", "OFFLINE"
    window_type: str           # "morning", "afternoon", "night"
    current_activity: Optional[int]
    s_plus: float
    threshold_h: float
    reference_k: float
    baseline_mean: Optional[float]
    baseline_std: Optional[float]
    baseline_n: int
    baseline_days: int         # Number of distinct valid calendar days in baseline
    baseline_days_required: int # Required calendar days (e.g. 7)
    baseline_ready: bool
    last_alert_time: Optional[str]
    last_alert_reason: Optional[str]
    system_time: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main CusumMonitor Implementation
# ─────────────────────────────────────────────────────────────────────────────
class CusumMonitor:
    """
    Independent CUSUM monitoring engine.
    Thread-safe, robust against restarts, and completely decoupled from camera loops.
    """

    def __init__(self, db_path: Optional[str] = None, config: Optional[CusumConfig] = None):
        self.config = config or CusumConfig()
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, "cusum_history.db")
        else:
            self.db_path = db_path

        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript(_CUSUM_DDL)
                # Initialize state rows for diurnal windows
                for w_type in ("morning", "afternoon"):
                    conn.execute("""
                        INSERT OR IGNORE INTO cusum_state (window_type, s_plus, in_alert, last_updated)
                        VALUES (?, 0.0, 0, NULL)
                    """, (w_type,))
                conn.commit()
            finally:
                conn.close()

    def classify_window(self, dt: datetime) -> str:
        """Classifies a timestamp into 'morning', 'afternoon', or 'night'."""
        hour = dt.hour
        if self.config.morning_start_hour <= hour < self.config.morning_end_hour:
            return "morning"
        elif self.config.afternoon_start_hour <= hour < self.config.afternoon_end_hour:
            return "afternoon"
        else:
            return "night"

    def get_baseline_days(self, window_type: str) -> int:
        """Counts the distinct valid calendar days in the baseline for this window."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute("""
                    SELECT count(DISTINCT substr(interval_start, 1, 10)) as cnt
                    FROM cusum_intervals
                    WHERE window_type = ? AND is_valid = 1 AND is_baseline = 1 AND activity_count IS NOT NULL
                """, (window_type,)).fetchone()
                return int(row["cnt"]) if row else 0
            finally:
                conn.close()

    def get_baseline_stats(self, window_type: str) -> Tuple[bool, int, Optional[float], Optional[float]]:
        """
        Retrieves baseline readiness, sample count N, mean mu_0, and std sigma_0
        for the specified window type ('morning' or 'afternoon').

        Readiness invariant:
        Requires BOTH:
        1. At least self.config.baseline_days distinct calendar days of valid baseline data.
        2. At least self.config.min_baseline_observations valid observations.
        """
        if window_type not in ("morning", "afternoon"):
            return False, 0, None, None

        with self._lock:
            conn = self._get_connection()
            try:
                # Query valid baseline observations within the baseline period
                rows = conn.execute("""
                    SELECT activity_count, interval_start FROM cusum_intervals
                    WHERE window_type = ? AND is_valid = 1 AND is_baseline = 1 AND activity_count IS NOT NULL
                    ORDER BY interval_start ASC
                """, (window_type,)).fetchall()

                values = [r["activity_count"] for r in rows]
                n = len(values)
                distinct_dates = set(r["interval_start"][:10] for r in rows)
                num_days = len(distinct_dates)

                # Baseline is ready ONLY when both calendar-day and sample-count requirements are satisfied
                is_ready = bool((num_days >= self.config.baseline_days) and (n >= self.config.min_baseline_observations))

                if n < self.config.min_baseline_observations:
                    partial_mean = sum(values) / float(n) if n > 0 else None
                    return False, n, partial_mean, None

                mean = sum(values) / float(n)
                if n > 1:
                    variance = sum((x - mean) ** 2 for x in values) / float(n - 1)
                    std = math.sqrt(variance)
                else:
                    std = 0.0

                return is_ready, n, mean, std
            finally:
                conn.close()

    def ingest_interval(
        self,
        interval_start: datetime,
        interval_end: datetime,
        unique_mosquito_ids: List[str],
        species_counts: Optional[Dict[str, int]] = None,
        coverage_ratio: float = 1.0,
        detector_running: bool = True
    ) -> Dict[str, Any]:
        """
        Ingests a 30-minute surveillance interval.

        Rules:
        - Rejects overlapping/duplicate interval_start timestamps.
        - If detector was stopped or coverage < min_coverage_ratio, marked as MISSING/INVALID (activity=None).
        - If valid, activity_count = len(unique_mosquito_ids).
        - Updates CUSUM score S_t^+ if in morning/afternoon and baseline is ready.
        - Triggers alerts when S_t^+ >= h; handles ongoing vs new alerts cleanly.
        """
        start_str = interval_start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = interval_end.strftime("%Y-%m-%d %H:%M:%S")
        window_type = self.classify_window(interval_start)
        species_map = species_counts or {}

        # Distinct IDs deduplication
        deduped_ids = sorted(list(set(unique_mosquito_ids)))
        is_valid = bool(detector_running and (coverage_ratio >= self.config.min_coverage_ratio))
        activity_count = len(deduped_ids) if is_valid else None

        with self._lock:
            conn = self._get_connection()
            try:
                # 1. Prevent duplicate ingestion and handle incomplete interval restart upgrades
                existing = conn.execute(
                    "SELECT id, is_valid FROM cusum_intervals WHERE interval_start = ?", (start_str,)
                ).fetchone()
                if existing:
                    if existing["is_valid"] == 1:
                        return {
                            "status": "duplicate_ignored",
                            "interval_start": start_str,
                            "message": f"Interval starting at {start_str} has already been ingested."
                        }
                    elif is_valid:
                        # Upgrade previously incomplete/invalid interval to valid completed interval
                        conn.execute("DELETE FROM cusum_intervals WHERE id = ?", (existing["id"],))
                    else:
                        return {
                            "status": "duplicate_ignored",
                            "interval_start": start_str,
                            "message": f"Incomplete interval starting at {start_str} already recorded."
                        }

                # 2. Check baseline readiness before this interval
                cur_baseline_ready, b_n, b_mean, b_std = self.get_baseline_stats(window_type)
                is_baseline_sample = 1 if (window_type in ("morning", "afternoon") and not cur_baseline_ready and is_valid) else 0

                # 3. Insert interval record
                conn.execute("""
                    INSERT INTO cusum_intervals (
                        interval_start, interval_end, window_type, activity_count,
                        unique_ids_json, species_counts_json, coverage_ratio, is_valid, is_baseline
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    start_str, end_str, window_type, activity_count,
                    json.dumps(deduped_ids), json.dumps(species_map),
                    float(coverage_ratio), 1 if is_valid else 0, is_baseline_sample
                ))

                result_event = {
                    "interval_start": start_str,
                    "interval_end": end_str,
                    "window_type": window_type,
                    "is_valid": is_valid,
                    "activity_count": activity_count,
                    "s_plus": 0.0,
                    "alert_triggered": False,
                    "alert_status": "NONE"
                }

                # If nighttime, missing, or off-window: no CUSUM update
                if window_type == "night" or not is_valid:
                    conn.commit()
                    return result_event

                # Re-fetch baseline stats (in case this interval completed the baseline)
                baseline_ready, b_n, b_mean, b_std = self.get_baseline_stats(window_type)
                if not baseline_ready or b_mean is None or b_std is None:
                    # Still collecting baseline
                    conn.commit()
                    result_event["alert_status"] = "COLLECTING_BASELINE"
                    return result_event

                # 4. Perform CUSUM Calculation
                # Load current CUSUM state
                state_row = conn.execute(
                    "SELECT s_plus, in_alert FROM cusum_state WHERE window_type = ?", (window_type,)
                ).fetchone()
                prev_s = state_row["s_plus"] if state_row else 0.0
                prev_in_alert = bool(state_row["in_alert"]) if state_row else False

                denom = max(b_std, self.config.min_std)
                z_t = (activity_count - b_mean) / denom
                s_plus = max(0.0, prev_s + z_t - self.config.k)
                result_event["s_plus"] = round(s_plus, 4)

                # 5. Alert Determination
                new_in_alert = False
                alert_status = "NORMAL"

                if s_plus >= self.config.h:
                    new_in_alert = True
                    reason = (
                        f"Observed chamber activity ({activity_count} IDs) exceeded diurnal baseline "
                        f"(mean={b_mean:.2f}, std={b_std:.2f}). CUSUM score {s_plus:.2f} >= {self.config.h:.2f}."
                    )
                    if not prev_in_alert:
                        # New alert triggered
                        alert_status = "TRIGGERED"
                        conn.execute("""
                            INSERT INTO cusum_alerts (
                                timestamp, window_type, activity_value, baseline_mean, baseline_std,
                                s_plus, threshold_h, reason, status
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'TRIGGERED')
                        """, (start_str, window_type, activity_count, b_mean, b_std, s_plus, self.config.h, reason))
                        conn.execute("""
                            UPDATE cusum_state
                            SET last_alert_time = ?, last_alert_reason = ?
                            WHERE window_type = ?
                        """, (start_str, reason, window_type))
                        result_event["alert_triggered"] = True
                    else:
                        # Prolonged ongoing alert
                        alert_status = "ONGOING"
                        conn.execute("""
                            UPDATE cusum_alerts
                            SET s_plus = ?, activity_value = ?, status = 'ONGOING'
                            WHERE id = (SELECT MAX(id) FROM cusum_alerts WHERE window_type = ?)
                        """, (s_plus, activity_count, window_type))
                else:
                    new_in_alert = False
                    if prev_in_alert:
                        # Alert resolved
                        alert_status = "RESOLVED"
                        conn.execute("""
                            UPDATE cusum_alerts
                            SET status = 'RESOLVED'
                            WHERE id = (SELECT MAX(id) FROM cusum_alerts WHERE window_type = ?)
                        """, (window_type,))

                # 6. Persist CUSUM State
                conn.execute("""
                    UPDATE cusum_state
                    SET s_plus = ?, in_alert = ?, last_updated = ?
                    WHERE window_type = ?
                """, (s_plus, 1 if new_in_alert else 0, start_str, window_type))

                conn.commit()
                result_event["alert_status"] = alert_status
                return result_event

            finally:
                conn.close()

    def get_status(self, now: Optional[datetime] = None) -> CusumStatus:
        """
        Returns full monitoring status for the active time window.
        Safe to call from GUI worker without blocking.
        """
        cur_dt = now or datetime.now()
        window_type = self.classify_window(cur_dt)
        eval_window = window_type if window_type in ("morning", "afternoon") else "afternoon"

        with self._lock:
            conn = self._get_connection()
            try:
                # Load state
                state_row = conn.execute(
                    "SELECT s_plus, in_alert, last_alert_time, last_alert_reason FROM cusum_state WHERE window_type = ?",
                    (eval_window,)
                ).fetchone()

                s_plus = state_row["s_plus"] if state_row else 0.0
                in_alert = bool(state_row["in_alert"]) if state_row else False
                last_alert_time = state_row["last_alert_time"] if state_row else None
                last_alert_reason = state_row["last_alert_reason"] if state_row else None

                # Baseline statistics
                baseline_ready, b_n, b_mean, b_std = self.get_baseline_stats(eval_window)
                b_days = self.get_baseline_days(eval_window)

                # Latest observation
                latest_obs = conn.execute("""
                    SELECT activity_count, is_valid FROM cusum_intervals
                    ORDER BY id DESC LIMIT 1
                """).fetchone()

                latest_activity = latest_obs["activity_count"] if (latest_obs and latest_obs["is_valid"]) else None

                # Overall system state determination
                if in_alert:
                    state = "ALERT_TRIGGERED"
                elif not baseline_ready:
                    state = "COLLECTING_BASELINE" if b_n > 0 else "INSUFFICIENT_DATA"
                else:
                    state = "MONITORING"

                return CusumStatus(
                    state=state,
                    window_type=window_type,
                    current_activity=latest_activity,
                    s_plus=round(s_plus, 2),
                    threshold_h=self.config.h,
                    reference_k=self.config.k,
                    baseline_mean=round(b_mean, 2) if b_mean is not None else None,
                    baseline_std=round(b_std, 2) if b_std is not None else None,
                    baseline_n=b_n,
                    baseline_days=b_days,
                    baseline_days_required=self.config.baseline_days,
                    baseline_ready=baseline_ready,
                    last_alert_time=last_alert_time,
                    last_alert_reason=last_alert_reason,
                    system_time=cur_dt.strftime("%Y-%m-%d %H:%M:%S")
                )
            finally:
                conn.close()

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves historical alert records."""
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute("""
                    SELECT id, timestamp, window_type, activity_value, baseline_mean,
                           baseline_std, s_plus, threshold_h, reason, status
                    FROM cusum_alerts
                    ORDER BY id DESC LIMIT ?
                """, (limit,)).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def reset_alert(self, window_type: Optional[str] = None):
        """Manually acknowledges and resets CUSUM score and alert status."""
        windows = [window_type] if window_type else ["morning", "afternoon"]
        with self._lock:
            conn = self._get_connection()
            try:
                for w in windows:
                    conn.execute("""
                        UPDATE cusum_state
                        SET s_plus = 0.0, in_alert = 0
                        WHERE window_type = ?
                    """, (w,))
                    conn.execute("""
                        UPDATE cusum_alerts
                        SET status = 'RESOLVED'
                        WHERE window_type = ? AND status IN ('TRIGGERED', 'ONGOING')
                    """, (w,))
                conn.commit()
            finally:
                conn.close()

    def close(self):
        """Clean close hook."""
        pass
