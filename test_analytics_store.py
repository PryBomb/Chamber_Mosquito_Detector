"""Isolated tests for AnalyticsStore (synthetic records, never shown in the app).
Run:  python -m unittest test_analytics_store -v
"""
import os
import unittest
from datetime import datetime, timedelta, timezone

# Ensure headless Qt operation for GUI tests
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui_app import (
    AnalyticsStore,
    normalize_species,
    parse_ts,
    floor_slot,
    _to_naive_dt,
    SLOT,
    PYSIDE6_AVAILABLE,
)

if PYSIDE6_AVAILABLE:
    from PySide6.QtWidgets import QApplication
    from gui_app import AnalyticsPage

T0 = datetime(2026, 10, 26, 12, 10, 0)


def payload(tracks=None, cusum=None, active=None):
    d = {}
    if tracks is not None:
        d["active_tracks_summary"] = tracks
    if active is not None:
        d["active_track_ids"] = active
    if cusum is not None:
        d["cusum"] = cusum
    return d


def trk(mid, sp="Aedes aegypti", conf=0.9):
    return {"mos_id": mid, "species": sp, "confidence": conf, "hit_count": 3}


class TestStore(unittest.TestCase):
    """Core domain logic tests for AnalyticsStore."""

    def test_empty_dataset(self):
        s = AnalyticsStore(T0)
        self.assertEqual(s.unique_ids(12, T0), set())
        self.assertTrue(all(v is None for _, v in s.series(12, T0)))
        self.assertIsNone(s.peak(12, T0))
        self.assertIsNone(s.avg_confidence(12, T0))
        self.assertIsNone(s.confidence_bins(12, T0))
        self.assertIsNone(s.window_counts(12, T0))
        self.assertIsNone(s.dominant_species(12, T0))
        self.assertEqual(s.cusum_points(12, T0), ([], []))

    def test_historical_data_loading_session_only_limitation(self):
        """Historical data loading is intentionally not implemented.

        Architecture limitation: AnalyticsStore operates strictly in-memory
        from real-time GET /api/status telemetry received during the current
        application session. It never queries the Pi's SQLite registry or CSV
        logs directly, preventing file locks, remote I/O bottlenecks, and
        coupling to detector storage schemas.
        """
        s = AnalyticsStore(T0)
        self.assertEqual(len(s.slot_ids), 0)
        self.assertEqual(len(s.covered), 0)
        # Any slot prior to session start must be None (uncovered), never synthesized
        earlier = T0 - timedelta(hours=2)
        series_vals = dict(s.series(12, T0))
        slot_earlier = floor_slot(earlier)
        self.assertIn(slot_earlier, series_vals)
        self.assertIsNone(series_vals[slot_earlier])
        # Ensure no remote DB or file access attributes exist on store
        self.assertFalse(hasattr(s, "db_path"))
        self.assertFalse(hasattr(s, "csv_path"))

    def test_duplicate_mosquito_ids_within_interval_and_same_poll(self):
        """Duplicate IDs in active_tracks_summary, active_track_ids, or across polls."""
        s = AnalyticsStore(T0)
        # Same ID appearing in active_tracks_summary AND active_track_ids in same poll
        p = payload(tracks=[trk("MOS-1"), trk("MOS-1")], active=["MOS-1", "MOS-1"])
        s.record(p, T0)
        self.assertEqual(s.unique_ids(12, T0), {"MOS-1"})
        self.assertEqual(len(s.slot_ids[floor_slot(T0)]), 1)

        # Numeric and string ID normalization
        s.record(payload(tracks=[trk(42)], active=[42]), T0 + timedelta(seconds=10))
        self.assertIn("42", s.unique_ids(12, T0))
        self.assertEqual(len(s.slot_ids[floor_slot(T0)]), 2)

        # 50 polls in the same 30-min slot with the same IDs
        for i in range(50):
            s.record(payload([trk("MOS-1"), trk("42")]), T0 + timedelta(seconds=i * 5))
        self.assertEqual(s.unique_ids(12, T0), {"MOS-1", "42"})
        series = [v for _, v in s.series(12, T0) if v is not None]
        self.assertEqual(series, [2])

    def test_duplicate_mosquito_ids_across_intervals_and_full_period(self):
        """Duplicate IDs across 30-min intervals count once per interval, once per period."""
        s = AnalyticsStore(T0)
        slot1 = T0
        slot2 = T0 + timedelta(minutes=30)
        slot3 = T0 + timedelta(minutes=60)

        # Slot 1: MOS-1
        s.record(payload([trk("MOS-1", conf=0.4, sp="Unknown")]), slot1)
        # Slot 2: MOS-1 and MOS-2
        s.record(payload([trk("MOS-1", conf=0.85, sp="Aedes aegypti"), trk("MOS-2", conf=0.75, sp="Culex")]), slot2)
        # Slot 3: MOS-2 and MOS-3
        s.record(payload([trk("MOS-2", conf=0.8, sp="Culex"), trk("MOS-3", conf=0.95, sp="Anopheles")]), slot3)

        now = slot3 + timedelta(minutes=5)
        # Series reports unique IDs per slot: [1, 2, 2]
        counts = [v for _, v in s.series(12, now) if v is not None]
        self.assertEqual(counts, [1, 2, 2])

        # Across the full 12-hour period, total distinct mosquitoes is 3
        all_unique = s.unique_ids(12, now)
        self.assertEqual(all_unique, {"MOS-1", "MOS-2", "MOS-3"})
        self.assertEqual(len(all_unique), 3)

        # Latest recorded species and confidence wins, but unconfirmed does not overwrite confirmed
        self.assertEqual(s.species["MOS-1"], "Aedes aegypti")
        self.assertEqual(s.conf["MOS-1"], 0.85)

        # A subsequent poll with unconfirmed species preserves the confirmed species
        s.record(payload([trk("MOS-1", conf=0.9, sp="Unknown")]), now + timedelta(seconds=10))
        self.assertEqual(s.species["MOS-1"], "Aedes aegypti")
        self.assertEqual(s.conf["MOS-1"], 0.9)

        # A subsequent poll with a new confirmed species updates it
        s.record(payload([trk("MOS-1", conf=0.92, sp="Aedes albopictus")]), now + timedelta(seconds=20))
        self.assertEqual(s.species["MOS-1"], "Aedes albopictus")

    def test_uncovered_is_none_covered_zero_is_zero(self):
        s = AnalyticsStore(T0)
        s.record(payload([]), T0)  # polled, nothing seen
        series = s.series(12, T0)
        self.assertEqual(series[-1][1], 0)
        self.assertTrue(all(v is None for _, v in series[:-1]))

    def test_species_counts_and_percentages_agree(self):
        s = AnalyticsStore(T0)
        s.record(payload([trk("1", "Aedes aegypti"), trk("2", "Aedes aegypti"), trk("3", "Aedes albopictus"),
                          trk("4", "Unknown"), trk("5", None), trk("6", "Culex"), trk("7", "Anopheles"),
                          trk("8", "Mansonia"), trk("9", "Coquillettidia"), trk("10", "Culiseta")]), T0)
        br = s.species_breakdown(12, T0)
        total = sum(c for _, c, _ in br)
        self.assertEqual(total, len(s.unique_ids(12, T0)))
        self.assertEqual(sum(round(c / total * 100) for _, c, _ in br) in range(98, 103), True)
        kinds = {k for _, _, k in br}
        self.assertIn("unconfirmed", kinds)
        self.assertIn("other", kinds)
        self.assertEqual(dict((n, c) for n, c, k in br)["Species unconfirmed"], 2)
        self.assertEqual(s.dominant_species(12, T0)[:2], ("Aedes aegypti", 2))

    def test_unconfirmed_never_assigned_to_a_species(self):
        s = AnalyticsStore(T0)
        s.record(payload([trk("1", "unconfirmed"), trk("2", "")]), T0)
        self.assertIsNone(s.dominant_species(12, T0))
        self.assertEqual(s.species_breakdown(12, T0), [("Species unconfirmed", 2, "unconfirmed")])

    def test_time_range_filter(self):
        s = AnalyticsStore(T0)
        s.record(payload([trk("OLD")]), T0 - timedelta(hours=13))
        s.record(payload([trk("NEW")]), T0)
        self.assertEqual(s.unique_ids(12, T0), {"NEW"})
        self.assertEqual(s.unique_ids(24, T0), {"OLD", "NEW"})
        self.assertEqual(s.unique_ids(168, T0), {"OLD", "NEW"})

    def test_malformed_records_do_not_crash(self):
        s = AnalyticsStore(T0)
        bad = [None, "x", [], {}, payload("notalist"), payload([None, 5, "a", {}, {"mos_id": None}]),
               payload([trk("A", conf=1.5), trk("B", conf=-1), trk("C", conf=True), trk("D", conf=float("nan")),
                        trk("E", conf="0.9")]),
               payload(cusum="oops"), payload(cusum={"s_plus": "bad", "threshold_h": None, "alert_history": "x"}),
               payload(cusum={"alert_history": [None, 3, {"timestamp": "garbage"}]})]
        for i, b in enumerate(bad):
            s.record(b, T0 + timedelta(seconds=i))
        self.assertIsNone(s.avg_confidence(12, T0 + timedelta(minutes=1)))

    def test_confidence_average_and_bins(self):
        s = AnalyticsStore(T0)
        s.record(payload([trk("1", conf=0.3), trk("2", conf=0.6), trk("3", conf=0.8), trk("4", conf=0.95),
                          trk("5", conf=1.0), trk("6", conf=None)]), T0)
        mean, n, total = s.avg_confidence(12, T0)
        self.assertEqual((n, total), (5, 6))
        self.assertAlmostEqual(mean, (0.3 + 0.6 + 0.8 + 0.95 + 1.0) / 5)
        self.assertEqual([c for _, c in s.confidence_bins(12, T0)], [1, 1, 1, 2])

    def test_confidence_histogram_every_boundary(self):
        """Test confidence values at every bin boundary: 0.0, 0.5, 0.7, 0.9, 1.0."""
        s = AnalyticsStore(T0)
        # Bins: [0.00-0.49: [0.0, 0.5)), [0.50-0.69: [0.5, 0.7)), [0.70-0.89: [0.7, 0.9)), [0.90-1.00: [0.9, 1.0001)]
        boundary_cases = [
            ("M0", 0.0),        # Bin 0
            ("M1", 0.4999),     # Bin 0
            ("M2", 0.5),        # Bin 1
            ("M3", 0.6999),     # Bin 1
            ("M4", 0.7),        # Bin 2
            ("M5", 0.8999),     # Bin 2
            ("M6", 0.9),        # Bin 3
            ("M7", 1.0),        # Bin 3
        ]
        s.record(payload([trk(mid, conf=conf) for mid, conf in boundary_cases]), T0)
        bins = s.confidence_bins(12, T0)
        counts = [c for _, c in bins]
        self.assertEqual(counts, [2, 2, 2, 2])

        # Test out-of-range and non-numeric confidence values are ignored
        s2 = AnalyticsStore(T0)
        invalid_cases = [
            ("INV1", -0.01),
            ("INV2", 1.0001),
            ("INV3", float("nan")),
            ("INV4", float("inf")),
            ("INV5", True),
            ("INV6", False),
            ("INV7", "0.85"),
        ]
        s2.record(payload([{"mos_id": mid, "confidence": conf} for mid, conf in invalid_cases]), T0)
        self.assertIsNone(s2.confidence_bins(12, T0))
        self.assertIsNone(s2.avg_confidence(12, T0))

    def test_peak_activity(self):
        s = AnalyticsStore(T0)
        s.record(payload([trk("1")]), T0)
        s.record(payload([trk("2"), trk("3")]), T0 + timedelta(minutes=30))
        slot, n = s.peak(12, T0 + timedelta(minutes=31))
        self.assertEqual((slot, n), (floor_slot(T0 + timedelta(minutes=30)), 2))

    def test_windows_come_from_detector_not_hardcoded(self):
        s = AnalyticsStore(T0)
        s.record(payload([trk("1")], cusum={"window_type": "morning", "state": "MONITORING"}), T0)
        s.record(payload([trk("1"), trk("2")], cusum={"window_type": "afternoon"}), T0 + timedelta(minutes=30))
        self.assertEqual(dict(s.window_counts(12, T0 + timedelta(minutes=31))), {"morning": 1, "afternoon": 2})
        s2 = AnalyticsStore(T0)
        s2.record(payload([trk("1")], cusum={"state": "MONITORING"}), T0)
        self.assertIsNone(s2.window_counts(12, T0))

    def test_alert_deduplication_and_distinct_alerts(self):
        """Alerts with identical (timestamp, window_type) are deduplicated; distinct ones are kept."""
        s = AnalyticsStore(T0)
        ts1 = (T0 - timedelta(minutes=20)).isoformat()
        ts2 = (T0 - timedelta(minutes=10)).isoformat()

        # Poll 1: returns alert A1
        a1 = {"timestamp": ts1, "window_type": "morning", "s_plus": 5.1}
        s.record(payload(cusum={"alert_history": [a1]}), T0)
        self.assertEqual(len(s.alerts), 1)

        # Poll 2: returns alert A1 again (duplicate) + alert A2 (distinct timestamp)
        a2 = {"timestamp": ts2, "window_type": "morning", "s_plus": 6.2}
        s.record(payload(cusum={"alert_history": [a1, a2]}), T0 + timedelta(seconds=10))
        self.assertEqual(len(s.alerts), 2)

        # Poll 3: returns alert A3 with same timestamp as A1 but different window_type
        a3 = {"timestamp": ts1, "window_type": "afternoon", "s_plus": 5.8}
        s.record(payload(cusum={"alert_history": [a1, a2, a3]}), T0 + timedelta(seconds=20))
        self.assertEqual(len(s.alerts), 3)

        _, alerts = s.cusum_points(12, T0 + timedelta(minutes=1))
        self.assertEqual(len(alerts), 3)

    def test_alert_malformed_timestamps_and_pruning_boundaries(self):
        """Malformed alert timestamps are omitted from points without crashing; old alerts pruned."""
        s = AnalyticsStore(T0)
        # Alert with garbage timestamp
        bad_alert = {"timestamp": "not-a-timestamp", "window_type": "morning", "s_plus": 5.5}
        s.record(payload(cusum={"alert_history": [bad_alert]}), T0)
        pts, alerts = s.cusum_points(12, T0)
        # Malformed alert has time=None and must be safely excluded from chart markers
        self.assertEqual(len(alerts), 0)

        # Alert at boundary of MAX_AGE (7 days 1 hr)
        s_prune = AnalyticsStore(T0)
        ts_old = (T0 - timedelta(days=8)).isoformat()
        ts_recent = (T0 - timedelta(hours=1)).isoformat()
        old_alert = {"timestamp": ts_old, "window_type": "morning", "s_plus": 7.0}
        recent_alert = {"timestamp": ts_recent, "window_type": "morning", "s_plus": 8.0}

        s_prune.record(payload(cusum={"alert_history": [old_alert, recent_alert]}), T0)
        # Old alert should be pruned on record cutoff
        self.assertEqual(len(s_prune.alerts), 1)
        self.assertIn(f"{ts_recent}|morning", s_prune.alerts)

    def test_timezone_aware_and_naive_timestamps_and_comparisons(self):
        """Timezone-aware timestamps and naive timestamps must not cause comparison errors."""
        # 1. parse_ts helper with naive and aware ISO strings
        naive_str = "2026-10-26T12:00:00"
        utc_z_str = "2026-10-26T12:00:00Z"
        utc_off_str = "2026-10-26T12:00:00+00:00"
        tokyo_off_str = "2026-10-26T12:00:00+09:00"

        dt1 = parse_ts(naive_str)
        dt2 = parse_ts(utc_z_str)
        dt3 = parse_ts(utc_off_str)
        dt4 = parse_ts(tokyo_off_str)

        for dt in (dt1, dt2, dt3, dt4):
            self.assertIsNotNone(dt)
            self.assertIsNone(dt.tzinfo, "parse_ts must always return naive local datetimes")

        # 2. AnalyticsStore operations with tz-aware now parameter
        now_utc = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
        s = AnalyticsStore(now_utc)
        alert_iso = "2026-10-26T11:45:00Z"
        cusum_payload = {
            "s_plus": 2.5,
            "threshold_h": 5.0,
            "state": "MONITORING",
            "alert_history": [{"timestamp": alert_iso, "window_type": "morning", "s_plus": 5.2}],
        }
        # Ingestion with tz-aware now must not throw TypeError
        s.record(payload([trk("M1")], cusum=cusum_payload), now_utc)

        # Queries with tz-aware now must compare cleanly
        pts, alerts = s.cusum_points(12, now_utc)
        self.assertEqual(len(pts), 1)
        self.assertEqual(len(alerts), 1)

        # series and unique_ids with tz-aware now
        ids = s.unique_ids(12, now_utc)
        self.assertEqual(ids, {"M1"})
        ser = s.series(12, now_utc)
        self.assertIsNotNone(ser[-1][1])

        # Interleaved record with naive now (1 minute after now_utc in local time)
        now_naive = _to_naive_dt(now_utc) + timedelta(minutes=1)
        s.record(payload([trk("M2")]), now_naive)
        self.assertEqual(len(s.unique_ids(12, now_naive)), 2)

    def test_helpers(self):
        self.assertIsNone(normalize_species("  N/A "))
        self.assertEqual(normalize_species(" Culex "), "Culex")
        self.assertIsNone(parse_ts("not a date"))
        self.assertIsNotNone(parse_ts("2026-10-26T12:00:00"))
        self.assertIsNotNone(parse_ts("2026-10-26T12:00:00+00:00"))
        # _to_naive_dt
        self.assertIsNone(_to_naive_dt(datetime.now()).tzinfo)
        aware = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.assertIsNone(_to_naive_dt(aware).tzinfo)

    def test_pruning_keeps_memory_bounded(self):
        s = AnalyticsStore(T0)
        s.record(payload([trk("1")]), T0 - timedelta(days=8))
        s.record(payload([trk("2")]), T0)
        self.assertEqual(len(s.slot_ids), 1)
        self.assertEqual(len(s.covered), 1)

    def test_actual_detector_fields_time_of_day_and_cusum(self):
        """Confirm time-of-day and CUSUM charts use actual detector fields, not fabricated metrics."""
        s = AnalyticsStore(T0)
        # Payload with detector-provided cusum block
        c = {
            "window_type": "dusk_peak",
            "state": "ALERT_TRIGGERED",
            "threshold_h": 4.5,
            "s_plus": 5.12,
            "alert_history": [{"timestamp": T0.isoformat(), "window_type": "dusk_peak", "s_plus": 5.12}],
        }
        s.record(payload([trk("MOS-99")], cusum=c), T0)

        # Time-of-day uses detector window_type directly
        win = s.window_counts(12, T0)
        self.assertIsNotNone(win)
        self.assertEqual(win, [("dusk_peak", 1)])

        # CUSUM parameters match detector telemetry directly
        self.assertEqual(s.threshold, 4.5)
        self.assertEqual(s.cusum_state, "ALERT_TRIGGERED")
        pts, alerts = s.cusum_points(12, T0)
        self.assertEqual(pts[-1][1], 5.12)
        self.assertEqual(alerts[0]["window"], "dusk_peak")
        self.assertEqual(alerts[0]["s_plus"], 5.12)


class TestAnalyticsGUI(unittest.TestCase):
    """UI tests for AnalyticsPage running in headless Qt mode."""

    @classmethod
    def setUpClass(cls):
        if not PYSIDE6_AVAILABLE:
            raise unittest.SkipTest("PySide6 is not available in this environment")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = AnalyticsPage()
        self.page.show()

    def tearDown(self):
        self.page.deleteLater()

    def test_gui_initial_empty_state(self):
        """Initial state before receiving telemetry: displays appropriate empty messages."""
        p = self.page
        self.assertEqual(p.card_unique.lbl_value.text(), "—")
        self.assertEqual(p.card_species.lbl_value.text(), "—")
        self.assertEqual(p.card_peak.lbl_value.text(), "—")
        self.assertEqual(p.card_conf.lbl_value.text(), "N/A")
        self.assertFalse(p.conf_panel.isVisible())
        self.assertIn("Waiting for the first telemetry update", p.line_chart.message)
        self.assertIn("Waiting for the first telemetry update", p.donut.message)

    def test_gui_time_range_changes(self):
        """Changing the time range dropdown updates the query horizon and forces re-render."""
        p = self.page
        # Default index is 0 (12 hours)
        self.assertEqual(p.cmb_range.currentData(), 12)

        # Feed a record in slot
        p.record_status(payload([trk("MOS-1", conf=0.88, sp="Culex")]))
        self.assertEqual(p.card_unique.lbl_value.text(), "1")

        # Switch to 24 hours (index 1)
        p.cmb_range.setCurrentIndex(1)
        self.assertEqual(p.cmb_range.currentData(), 24)
        self.assertIn("24 hours", p.card_unique.lbl_subtext.text())

        # Switch to 7 days (index 2)
        p.cmb_range.setCurrentIndex(2)
        self.assertEqual(p.cmb_range.currentData(), 168)
        self.assertIn("7 days", p.card_unique.lbl_subtext.text())

    def test_gui_updates_without_duplicate_data(self):
        """Repeated status updates with the same data do not duplicate counts."""
        p = self.page
        status_data = payload([trk("MOS-1", sp="Aedes aegypti", conf=0.92)])
        for _ in range(5):
            p.record_status(status_data)

        self.assertEqual(p.card_unique.lbl_value.text(), "1")
        self.assertEqual(p.card_species.lbl_value.text(), "Aedes aegypti")
        self.assertTrue(p.conf_panel.isVisible())
        self.assertEqual(len(p.donut.data), 1)

    def test_gui_disconnected_and_error_states(self):
        """Connection state changes update informative messaging across panels."""
        p = self.page
        p.set_connection("disconnected")
        self.assertIn("Disconnected from Raspberry Pi", p.line_chart.message)

        p.set_connection("unauthorized")
        self.assertIn("Unauthorized", p.line_chart.message)

        p.set_connection("malformed_json")
        self.assertIn("unreadable data", p.line_chart.message)

    def test_gui_covered_with_zero_detections(self):
        """When polls succeed with zero detections, cards show 0 (covered) instead of dash."""
        p = self.page
        p.record_status(payload([]))
        self.assertEqual(p.card_unique.lbl_value.text(), "0")
        self.assertEqual(p.card_species.lbl_value.text(), "—")
        self.assertIn("No detections yet", p.card_species.lbl_subtext.text())
        self.assertEqual(p.donut.message, "No detections recorded in this period.")


if __name__ == "__main__":
    unittest.main()
