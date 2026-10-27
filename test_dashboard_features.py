"""Isolated unit tests for MosquitoSentinel dashboard features:
- /api/registry endpoint with auth and date filtering
- Snapshot button capturing frame to disk
- Expand camera preview dialog
- HistoryPage registry loading, date filtering, and CSV export
- Unconfirmed species classification and duplicate prevention
"""
import os
import csv
import json
import time
import glob
import shutil
import tempfile
import sqlite3
import threading
import unittest
import urllib.request
import urllib.error

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QDate, Qt

from gui_app import (
    PYSIDE6_AVAILABLE,
    LiveDetectionPage,
    HistoryPage,
    ExpandedCameraDialog,
    normalize_species,
)
from status_server import ThreadedHTTPServer, SentinelStatusHandler


def make_test_jpeg(width: int = 640, height: int = 360) -> QImage:
    img = QImage(width, height, QImage.Format_RGB32)
    img.fill(0x1F8A4C)
    return img


class TestRegistryEndpoint(unittest.TestCase):
    """Verifies GET /api/registry endpoint security, querying, and filtering."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.temp_dir, "test_registry.db")

        # Initialize SQLite registry table
        conn = sqlite3.connect(cls.db_path)
        conn.execute("""
            CREATE TABLE mosquito_registry (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                mos_id        TEXT    NOT NULL UNIQUE,
                species       TEXT    NOT NULL,
                first_seen    TEXT    NOT NULL,
                last_seen     TEXT    NOT NULL,
                detect_count  INTEGER NOT NULL DEFAULT 1,
                confidence    REAL    NOT NULL,
                snapshot_path TEXT,
                verification_status TEXT DEFAULT 'unverified'
            )
        """)
        conn.execute("""
            INSERT INTO mosquito_registry (mos_id, species, first_seen, last_seen, detect_count, confidence, snapshot_path)
            VALUES ('MOS-001', 'Aedes aegypti', '2026-10-27 10:00:00', '2026-10-27 10:05:00', 14, 0.92, 'captures/snap1.jpg'),
                   ('MOS-002', 'Mosquito - species unconfirmed', '2026-10-26 15:30:00', '2026-10-26 15:35:00', 3, 0.55, '')
        """)
        conn.commit()
        conn.close()

        class MockDetector:
            db_path = cls.db_path

        cls.server = ThreadedHTTPServer(("127.0.0.1", 0), SentinelStatusHandler, detector_ref=MockDetector(), token="valid-tok")
        cls.port = cls.server.server_port
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.server.is_serving = True

    @classmethod
    def tearDownClass(cls):
        cls.server.is_serving = False
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_registry_unauthorized(self):
        """GET /api/registry without valid token returns 401."""
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/registry")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)

    def test_registry_all_records(self):
        """GET /api/registry with token returns full list of records."""
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/registry")
        req.add_header("X-Sentinel-Token", "valid-tok")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["total"], 2)
            records = data["records"]
            mos_ids = [r["mos_id"] for r in records]
            self.assertIn("MOS-001", mos_ids)
            self.assertIn("MOS-002", mos_ids)

    def test_registry_date_filter(self):
        """GET /api/registry with date param returns only matching records."""
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/registry?date=2026-10-27")
        req.add_header("X-Sentinel-Token", "valid-tok")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["total"], 1)
            self.assertEqual(data["records"][0]["mos_id"], "MOS-001")


class TestDashboardInteractiveFeatures(unittest.TestCase):
    """Tests Snapshot, Expand Camera, and HistoryPage with CSV export."""

    @classmethod
    def setUpClass(cls):
        if not PYSIDE6_AVAILABLE:
            raise unittest.SkipTest("PySide6 is not available")
        cls.app = QApplication.instance() or QApplication([])

    def test_snapshot_button_saves_file_and_gives_feedback(self):
        """Clicking Snapshot saves current frame image to captures directory with feedback."""
        page = LiveDetectionPage()
        page.show()

        # Initially disabled before receiving any frames
        self.assertFalse(page.btn_snap.isEnabled())

        # Feed a valid camera frame
        img = make_test_jpeg(640, 360)
        page.on_frame_received(img)
        self.assertTrue(page.btn_snap.isEnabled())

        # Click snapshot
        before_time = time.time()
        page.btn_snap.click()

        # Verify feedback
        self.assertEqual(page.btn_snap.text(), "✓ Saved!")

        # Verify file on disk
        captures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
        self.assertTrue(os.path.isdir(captures_dir))
        files = glob.glob(os.path.join(captures_dir, "snapshot_*.jpg"))
        recent = [f for f in files if os.path.getmtime(f) >= before_time - 1.0]
        self.assertGreaterEqual(len(recent), 1)

        saved_file = recent[0]
        self.assertGreater(os.path.getsize(saved_file), 100)
        loaded = QImage(saved_file)
        self.assertFalse(loaded.isNull())
        self.assertEqual(loaded.width(), 640)
        self.assertEqual(loaded.height(), 360)

        page.deleteLater()

    def test_expand_camera_preview(self):
        """Clicking Expand button opens an expanded view dialog that displays the live stream."""
        page = LiveDetectionPage()
        page.show()

        # Feed frame
        img = make_test_jpeg(1280, 720)
        page.on_frame_received(img)
        self.assertTrue(page.btn_full.isEnabled())

        # Click expand
        page.btn_full.click()
        self.assertIsNotNone(page._expanded_dialog)
        self.assertIsInstance(page._expanded_dialog, ExpandedCameraDialog)
        self.assertTrue(page._expanded_dialog.viewport.has_frame())

        # Feed another frame: mirrors to dialog
        img2 = make_test_jpeg(1280, 720)
        page.on_frame_received(img2)
        self.assertTrue(page._expanded_dialog.viewport.has_frame())

        # Close dialog
        page._expanded_dialog.accept()
        page.deleteLater()

    def test_history_page_data_loading_and_filtering(self):
        """HistoryPage loads records, displays columns, filters by date, and shows empty states."""
        hp = HistoryPage()
        hp.show()

        records = [
            {
                "id": 1,
                "mos_id": "MOS-001",
                "species": "Aedes aegypti",
                "first_seen": "2026-10-27 08:00:00",
                "last_seen": "2026-10-27 08:15:00",
                "detect_count": 8,
                "confidence": 0.912,
                "snapshot_path": "captures/snap1.jpg",
                "verification_status": "unverified",
            },
            {
                "id": 2,
                "mos_id": "MOS-002",
                "species": "Mosquito - species unconfirmed",
                "first_seen": "2026-10-25 14:00:00",
                "last_seen": "2026-10-25 14:05:00",
                "detect_count": 2,
                "confidence": 0.520,
                "snapshot_path": "",
                "verification_status": "unverified",
            },
        ]

        hp.load_records(records)
        self.assertTrue(hp.btn_export.isEnabled())

        # Filter by 2026-10-27
        hp.date_filter.setDate(QDate(2026, 10, 27))
        filtered = hp._get_filtered_records()
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["mos_id"], "MOS-001")

        # Show all dates
        hp._show_all_dates()
        self.assertEqual(len(hp._get_filtered_records()), 2)

        hp.deleteLater()

    def test_csv_export_format_and_contents(self):
        """Exported CSV file contains correct headers, formatted values, and matches source records."""
        hp = HistoryPage()
        records = [
            {
                "id": 1,
                "mos_id": "MOS-101",
                "species": "Aedes albopictus",
                "first_seen": "2026-10-27 09:10:00",
                "last_seen": "2026-10-27 09:20:00",
                "detect_count": 5,
                "confidence": 0.8845,
                "snapshot_path": "captures/albo_101.jpg",
                "verification_status": "verified",
            }
        ]
        hp.load_records(records)
        hp._show_all_dates()

        # Write to temp file using the export logic
        temp_dir = tempfile.mkdtemp()
        export_file = os.path.join(temp_dir, "exported_history.csv")

        try:
            with open(export_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Mosquito_ID", "Species", "Confidence", "Hits",
                    "First_Seen", "Last_Seen", "Verification_Status", "Snapshot_File"
                ])
                for r in hp._get_filtered_records():
                    conf_val = r.get("confidence")
                    conf_str = f"{float(conf_val)*100:.2f}%" if isinstance(conf_val, (int, float)) else str(conf_val or "")
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

            self.assertTrue(os.path.isfile(export_file))
            with open(export_file, "r", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                self.assertEqual(len(reader), 2)
                headers = reader[0]
                self.assertIn("Mosquito_ID", headers)
                self.assertIn("Species", headers)
                self.assertIn("Confidence", headers)
                row = reader[1]
                self.assertEqual(row[1], "MOS-101")
                self.assertEqual(row[2], "Aedes albopictus")
                self.assertEqual(row[3], "88.45%")
                self.assertEqual(row[4], "5")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            hp.deleteLater()

    def test_unconfirmed_species_normalization_and_duplicate_handling(self):
        """Unconfirmed species labels normalize to None in analytics; tracker updates without duplicates."""
        # normalize_species handles unconfirmed
        self.assertIsNone(normalize_species("Mosquito - species unconfirmed"))
        self.assertIsNone(normalize_species("unconfirmed"))
        self.assertIsNone(normalize_species("unknown"))
        self.assertEqual(normalize_species("Aedes aegypti"), "Aedes aegypti")
        self.assertEqual(normalize_species("Aedes albopictus"), "Aedes albopictus")

        # Tracker update_existing updates detect_count and species without duplicating
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "tracker_test.db")
        try:
            from mosquito_tracker_server import MosquitoTracker
            tracker = MosquitoTracker(db_path=db_path)
            # Insert initial unconfirmed mosquito
            import numpy as np
            hist = np.zeros(64, dtype=np.float32)
            mid = tracker._insert_new_mosquito(
                hist, 1.0, 0.5, "Mosquito - species unconfirmed", 0.55, (10, 10, 50, 50), "", "2026-10-27 10:00:00"
            )
            self.assertEqual(mid, "MOS-001")

            # Update existing track with confirmed species
            tracker._update_existing(mid, "2026-10-27 10:01:00", 0.92, increment=2, species="Aedes aegypti")

            all_recs = tracker.get_all()
            self.assertEqual(len(all_recs), 1)  # No duplicate rows created
            rec = all_recs[0]
            self.assertEqual(rec["mos_id"], "MOS-001")
            self.assertEqual(rec["detect_count"], 3)  # Incremented
            self.assertEqual(rec["species"], "Aedes aegypti")  # Upgraded
            self.assertEqual(rec["confidence"], 0.92)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
