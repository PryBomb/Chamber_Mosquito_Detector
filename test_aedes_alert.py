"""Automated test suite for Confirmed Aedes aegypti Threshold Alert System.

Validates all operational safeguards:
1. Qualification rules: species == "Aedes aegypti", confidence >= 0.80, hits >= 3.
2. Non-qualifying exclusions: albopictus, unconfirmed species, confidence < 0.80, hits < 3.
3. Deduplication: Count only unique active tracking IDs; repeated frames do not inflate count.
4. Lifecycle transitions:
   - < 3 qualifying -> NORMAL
   - >= 3 qualifying -> ALERT TRIGGERED
   - Drops below 3 -> returns to NORMAL
5. State-change history: History entry logged ONLY on status transitions, avoiding polling duplicates.
6. CUSUM deactivation & preservation: Active alerts use threshold system; CUSUM is disabled from active loop.
7. Status server API: GET /api/status provides real-time 'alert' telemetry block.
8. GUI Alert view: CusumAlertsPage properly renders alert cards, active table, and history table.
"""
import os
import sys
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import config
from mosquito_tracker_server import ActiveMosquitoTrack, MosquitoTracker
from status_server import ThreadedHTTPServer, SentinelStatusHandler
import urllib.request
import json
import threading
import tempfile
import shutil
import sqlite3


class TestAedesAlertQualification(unittest.TestCase):
    """Verifies that only unique active tracks meeting all 3 criteria qualify."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_tracker.db")
        self.tracker = MosquitoTracker(db_path=self.db_path)
        # Suppress background sync
        if getattr(self.tracker, "_sync", None):
            self.tracker._sync = None

    def tearDown(self):
        self.tracker.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _add_track(self, track_id: int, mos_id: str, species: str, conf: float, hits: int = 3):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        track = ActiveMosquitoTrack(
            track_id=track_id,
            mos_id=mos_id,
            species=species,
            confidence=conf,
            box=(10, 10, 50, 50),
            crop_bgr=None,
            now_str=now_str,
            hit_count=hits
        )
        self.tracker._active_tracks[track_id] = track
        return track

    def test_01_strictly_aedes_aegypti_qualifies(self):
        """Qualifies when species is exactly 'Aedes aegypti', conf >= 0.80, hits >= 3."""
        self._add_track(1, "MOS-001", "Aedes aegypti", 0.85, hits=3)
        status = self.tracker.get_status()
        alert = status["alert"]

        self.assertEqual(alert["qualifying_count"], 1)
        self.assertEqual(alert["qualifying_ids"], ["MOS-001"])
        self.assertEqual(alert["status"], "NORMAL")  # 1 < 3

    def test_02_aedes_albopictus_does_not_qualify(self):
        """Aedes albopictus must NOT qualify regardless of confidence."""
        self._add_track(1, "MOS-001", "Aedes albopictus", 0.95, hits=5)
        status = self.tracker.get_status()
        alert = status["alert"]

        self.assertEqual(alert["qualifying_count"], 0)
        self.assertEqual(alert["status"], "NORMAL")

    def test_03_unconfirmed_species_does_not_qualify(self):
        """Unconfirmed mosquito detections must NOT qualify."""
        self._add_track(1, "MOS-001", "Mosquito - species unconfirmed", 0.85, hits=5)
        status = self.tracker.get_status()
        alert = status["alert"]

        self.assertEqual(alert["qualifying_count"], 0)

    def test_04_confidence_below_threshold_does_not_qualify(self):
        """Confidence score < 0.80 (e.g. 0.79) must NOT qualify."""
        self._add_track(1, "MOS-001", "Aedes aegypti", 0.79, hits=5)
        status = self.tracker.get_status()
        alert = status["alert"]

        self.assertEqual(alert["qualifying_count"], 0)

    def test_05_hits_below_threshold_does_not_qualify(self):
        """Confirmation frames < 3 (e.g. hits=1 or 2) must NOT qualify."""
        self._add_track(1, "MOS-001", "Aedes aegypti", 0.90, hits=2)
        status = self.tracker.get_status()
        alert = status["alert"]

        self.assertEqual(alert["qualifying_count"], 0)

    def test_06_deduplication_by_unique_mos_id(self):
        """Multiple active tracks with identical mos_id are counted only once."""
        self._add_track(1, "MOS-001", "Aedes aegypti", 0.90, hits=3)
        self._add_track(2, "MOS-001", "Aedes aegypti", 0.92, hits=4)
        status = self.tracker.get_status()
        alert = status["alert"]

        self.assertEqual(alert["qualifying_count"], 1)
        self.assertEqual(alert["qualifying_ids"], ["MOS-001"])


class TestAedesAlertLifecycleAndTransitions(unittest.TestCase):
    """Verifies threshold triggering (>=3), return to NORMAL (<3), and history transition logging."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_lifecycle.db")
        self.tracker = MosquitoTracker(db_path=self.db_path)
        if getattr(self.tracker, "_sync", None):
            self.tracker._sync = None

    def tearDown(self):
        self.tracker.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _add_track(self, track_id: int, mos_id: str, species: str = "Aedes aegypti", conf: float = 0.85, hits: int = 3):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        track = ActiveMosquitoTrack(
            track_id=track_id,
            mos_id=mos_id,
            species=species,
            confidence=conf,
            box=(10, 10, 50, 50),
            crop_bgr=None,
            now_str=now_str,
            hit_count=hits
        )
        self.tracker._active_tracks[track_id] = track
        return track

    def test_01_two_qualifying_remains_normal(self):
        """2 qualifying mosquitoes do not trigger alert (status NORMAL, history empty)."""
        self._add_track(1, "MOS-001", hits=3)
        self._add_track(2, "MOS-002", hits=4)
        status = self.tracker.get_status()
        alert = status["alert"]

        self.assertEqual(alert["qualifying_count"], 2)
        self.assertEqual(alert["status"], "NORMAL")
        self.assertFalse(alert["alert_triggered"])
        self.assertEqual(len(alert["alert_history"]), 0)

    def test_02_three_qualifying_triggers_alert(self):
        """3 qualifying mosquitoes transition status to ALERT TRIGGERED and log history."""
        self._add_track(1, "MOS-001", hits=3)
        self._add_track(2, "MOS-002", hits=3)
        self._add_track(3, "MOS-003", hits=3)
        status = self.tracker.get_status()
        alert = status["alert"]

        self.assertEqual(alert["qualifying_count"], 3)
        self.assertEqual(alert["status"], "ALERT TRIGGERED")
        self.assertTrue(alert["alert_triggered"])
        self.assertIsNotNone(alert["last_alert_time"])
        self.assertEqual(len(alert["alert_history"]), 1)
        self.assertEqual(alert["alert_history"][0]["status"], "ALERT TRIGGERED")

    def test_03_no_duplicate_history_entries_during_repeated_polling(self):
        """Repeated get_status polling while alert is active does NOT add duplicate history entries."""
        self._add_track(1, "MOS-001", hits=3)
        self._add_track(2, "MOS-002", hits=3)
        self._add_track(3, "MOS-003", hits=3)

        # Poll 5 consecutive times
        for _ in range(5):
            st = self.tracker.get_status()
            self.assertEqual(len(st["alert"]["alert_history"]), 1)

    def test_04_count_dropping_below_three_returns_to_normal(self):
        """When qualifying count falls below 3, status returns to NORMAL and logs resolution entry."""
        # 1. Trigger alert with 3 mosquitoes
        self._add_track(1, "MOS-001", hits=3)
        self._add_track(2, "MOS-002", hits=3)
        self._add_track(3, "MOS-003", hits=3)
        st1 = self.tracker.get_status()
        self.assertEqual(st1["alert"]["status"], "ALERT TRIGGERED")

        # 2. One mosquito departs -> count is 2
        del self.tracker._active_tracks[3]
        st2 = self.tracker.get_status()
        self.assertEqual(st2["alert"]["status"], "NORMAL")
        self.assertFalse(st2["alert"]["alert_triggered"])

        # History should now contain exactly 2 events: TRIGGERED, then NORMAL
        history = self.tracker._alert_history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["status"], "ALERT TRIGGERED")
        self.assertEqual(history[1]["status"], "NORMAL")
        self.assertIn("fell to 2", history[1]["reason"])

        # Repeated polling at count 2 adds no further history entries
        for _ in range(3):
            st3 = self.tracker.get_status()
            self.assertEqual(len(st3["alert"]["alert_history"]), 2)

    def test_05_four_qualifying_scales_correctly(self):
        """A 4th qualifying mosquito keeps status ALERT TRIGGERED without duplicate transition."""
        self._add_track(1, "MOS-001", hits=3)
        self._add_track(2, "MOS-002", hits=3)
        self._add_track(3, "MOS-003", hits=3)
        self.tracker.get_status()

        self._add_track(4, "MOS-004", hits=3)
        st = self.tracker.get_status()
        self.assertEqual(st["alert"]["status"], "ALERT TRIGGERED")
        self.assertEqual(st["alert"]["qualifying_count"], 4)
        self.assertEqual(len(st["alert"]["alert_history"]), 1)


class TestCusumDeactivationAndStatusEndpoint(unittest.TestCase):
    """Verifies CUSUM deactivation from active loop and HTTP /api/status payload integrity."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_status.db")

        # Create Mock Detector with tracker
        class MockDetector:
            def __init__(self, db_path):
                self.db_path = db_path
                self.worker_running = True
                self.fps = 15.0
                self.ai_fps = 12.0
                self.mode = "chamber_surveillance"
                self.confirmed_detections_count = 3
                self.tracker = MosquitoTracker(db_path=db_path)
                if getattr(self.tracker, "_sync", None):
                    self.tracker._sync = None
                self.start_time = time.time()

                # Add 3 qualifying tracks
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for i in range(1, 4):
                    self.tracker._active_tracks[i] = ActiveMosquitoTrack(
                        track_id=i,
                        mos_id=f"MOS-00{i}",
                        species="Aedes aegypti",
                        confidence=0.88,
                        box=(10, 10, 50, 50),
                        crop_bgr=None,
                        now_str=now_str,
                        hit_count=3
                    )

        cls.detector = MockDetector(cls.db_path)
        cls.server = ThreadedHTTPServer(
            ("127.0.0.1", 0),
            SentinelStatusHandler,
            detector_ref=cls.detector,
            token="test-token"
        )
        cls.port = cls.server.server_port
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.server.is_serving = True

    @classmethod
    def tearDownClass(cls):
        cls.server.is_serving = False
        cls.server.shutdown()
        cls.server.server_close()
        cls.detector.tracker.close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_cusum_is_disabled_in_config(self):
        """CUSUM_ENABLED must be False in config while CUSUM_DB_PATH is preserved."""
        self.assertFalse(getattr(config, "CUSUM_ENABLED", True))
        self.assertTrue(hasattr(config, "CUSUM_DB_PATH"))
        self.assertTrue(hasattr(config, "CUSUM_INTERVAL_MINUTES"))

    def test_02_status_endpoint_returns_alert_payload_and_null_cusum(self):
        """GET /api/status returns 'alert' block and 'cusum' is null when disabled."""
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/status")
        req.add_header("X-Sentinel-Token", "test-token")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        self.assertIn("alert", data)
        alert = data["alert"]
        self.assertEqual(alert["status"], "ALERT TRIGGERED")
        self.assertEqual(alert["qualifying_count"], 3)
        self.assertEqual(alert["threshold"], 3)
        self.assertIsNone(data["cusum"])


class TestGUIAlertsPage(unittest.TestCase):
    """Verifies that the GUI Alerts page correctly consumes the alert payload."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_01_alerts_page_rendering(self):
        from gui_app import CusumAlertsPage

        page = CusumAlertsPage()
        # Verify initial state
        self.assertIn("NORMAL", page.card_state.lbl_value.text())

        # Update with alert triggered payload
        alert_payload = {
            "status": "ALERT TRIGGERED",
            "alert_triggered": True,
            "target_species": "Aedes aegypti",
            "threshold": 3,
            "min_confidence": 0.80,
            "min_confirm_frames": 3,
            "qualifying_count": 3,
            "qualifying_ids": ["MOS-001", "MOS-002", "MOS-003"],
            "qualifying_tracks": [
                {"mos_id": "MOS-001", "species": "Aedes aegypti", "confidence": 0.92, "hit_count": 5},
                {"mos_id": "MOS-002", "species": "Aedes aegypti", "confidence": 0.88, "hit_count": 3},
                {"mos_id": "MOS-003", "species": "Aedes aegypti", "confidence": 0.85, "hit_count": 4},
            ],
            "last_alert_time": "2026-10-28 17:00:00",
            "alert_history": [
                {"timestamp": "2026-10-28 17:00:00", "status": "ALERT TRIGGERED", "qualifying_count": 3, "reason": "3 confirmed Aedes aegypti detected"}
            ]
        }
        page.update_alert(alert_payload)

        self.assertEqual(page.card_state.lbl_value.text(), "ALERT TRIGGERED")
        self.assertEqual(page.card_count.lbl_value.text(), "3 qualifying")
        self.assertEqual(page.active_table.table.rowCount(), 3)
        self.assertEqual(page.history_table.table.rowCount(), 1)

        # Update back to NORMAL
        normal_payload = dict(alert_payload)
        normal_payload["status"] = "NORMAL"
        normal_payload["alert_triggered"] = False
        normal_payload["qualifying_count"] = 1
        normal_payload["qualifying_tracks"] = [
            {"mos_id": "MOS-001", "species": "Aedes aegypti", "confidence": 0.92, "hit_count": 5}
        ]
        page.update_alert(normal_payload)
        self.assertEqual(page.card_state.lbl_value.text(), "NORMAL")
        self.assertEqual(page.card_count.lbl_value.text(), "1 qualifying")
        self.assertEqual(page.active_table.table.rowCount(), 1)


class TestHardwareAlarmController(unittest.TestCase):
    """Verifies that HardwareAlarmController toggles safely in simulated and real environments."""

    def test_01_hardware_alarm_controller_lifecycle(self):
        from chamber_detector_server import HardwareAlarmController

        ctrl = HardwareAlarmController(pin=17, active_high=True, enabled=True)
        self.assertEqual(ctrl.pin, 17)
        self.assertTrue(ctrl.enabled)
        self.assertFalse(ctrl._current_state)

        # Trigger alarm
        ctrl.set_alarm(True)
        self.assertTrue(ctrl._current_state)

        # Deactivate alarm
        ctrl.set_alarm(False)
        self.assertFalse(ctrl._current_state)

        # Cleanup
        ctrl.cleanup()
        self.assertFalse(ctrl._current_state)


if __name__ == "__main__":
    unittest.main()
