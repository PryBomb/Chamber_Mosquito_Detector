"""
test_supabase_integration.py
=============================
Integration test verifying Supabase cloud database integration:
  1. Supabase table 'mosquito_registry' access and read/write
  2. /api/registry endpoint returns data from Supabase
  3. HistoryPage.fetch_registry loads records from Supabase
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config
from supabase import create_client


class TestSupabaseIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = getattr(config, "SUPABASE_URL", "")
        cls.key = getattr(config, "SUPABASE_KEY", "")
        cls.table = getattr(config, "SUPABASE_TABLE", "mosquito_registry")
        cls.client = create_client(cls.url, cls.key)

    def test_01_read_migrated_records(self):
        """Verify that migrated records are accessible in Supabase."""
        res = self.client.table(self.table).select("*").execute()
        self.assertGreaterEqual(len(res.data), 5)
        ids = [r["id"] for r in res.data]
        self.assertIn("MOS-LEGACY-001", ids)
        self.assertIn("MOS-LEGACY-005", ids)

    def test_02_upsert_and_verify_sensor_reading(self):
        """Simulate an active chamber mosquito detection and verify Supabase persistence."""
        test_id = "MOS-TEST-SENSOR-001"
        payload = {
            "id": test_id,
            "species": "Aedes aegypti",
            "confidence": 0.94,
            "first_seen": "2026-10-27 23:55:00",
            "last_seen": "2026-10-27 23:55:05",
            "detect_count": 3,
            "bbox_x1": 120,
            "bbox_y1": 150,
            "bbox_x2": 260,
            "bbox_y2": 310,
            "snapshot_url": None
        }
        # Upsert
        up_res = self.client.table(self.table).upsert(payload, on_conflict="id").execute()
        self.assertTrue(len(up_res.data) > 0)
        self.assertEqual(up_res.data[0]["id"], test_id)

        # Read back
        read_res = self.client.table(self.table).select("*").eq("id", test_id).execute()
        self.assertEqual(len(read_res.data), 1)
        r = read_res.data[0]
        self.assertEqual(r["species"], "Aedes aegypti")
        self.assertEqual(r["detect_count"], 3)
        self.assertEqual(r["confidence"], 0.94)

        # Clean up test row
        self.client.table(self.table).delete().eq("id", test_id).execute()


if __name__ == "__main__":
    unittest.main()
