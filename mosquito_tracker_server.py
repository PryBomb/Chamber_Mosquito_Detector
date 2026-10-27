"""
mosquito_tracker.py
====================
Fast local SQLite registry for unique mosquito identification and duplicate
prevention, with optional background sync to Supabase.

Architecture:
  - SQLite (< 1ms) handles all live-camera writes so detection never lags.
  - A background thread drains a sync-queue to Supabase when available.
  - Every confirmed Aedes detection is compared against existing registry
    entries using colour histogram + morphology similarity before a new
    MOS-XXX ID is assigned.

Usage:
    from mosquito_tracker import MosquitoTracker
    tracker = MosquitoTracker()
    mos_id, status = tracker.register(crop_bgr, species, confidence, box, snapshot_path)
    # status: "new" | "duplicate" | "unverified"
"""

import os
import time
import sqlite3
import threading
import queue
import json
import csv
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

import config

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration constants (override via config.py if desired)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH            = getattr(config, "TRACKER_DB_PATH",    os.path.join(config.BASE_DIR, "mosquito_registry.db"))
COLOR_HIST_BINS    = getattr(config, "COLOR_HIST_BINS",    32)        # bins per channel (H & S in HSV)
SIMILARITY_HIGH    = getattr(config, "SIMILARITY_HIGH",    0.80)      # >= 80% -> duplicate
SIMILARITY_LOW     = getattr(config, "SIMILARITY_LOW",     0.55)      # 55-80% -> unverified
MORPH_WEIGHT       = getattr(config, "MORPH_WEIGHT",       0.35)      # fraction of score from morphology
COLOR_WEIGHT       = getattr(config, "COLOR_WEIGHT",       0.65)      # fraction of score from colour histogram
SUPABASE_ENABLED   = getattr(config, "SUPABASE_ENABLED",   False)     # flip to True in config.py
SUPABASE_URL       = getattr(config, "SUPABASE_URL",       "")
SUPABASE_KEY       = getattr(config, "SUPABASE_KEY",       "")
SUPABASE_TABLE     = getattr(config, "SUPABASE_TABLE",     "mosquito_registry")
SUPABASE_BUCKET    = getattr(config, "SUPABASE_BUCKET",    "mosquito-images")

# ─────────────────────────────────────────────────────────────────────────────
# Schema Migration & Payload Configuration
# ─────────────────────────────────────────────────────────────────────────────
_MIGRATION_COLUMNS = [
    ("verification_status", "TEXT NOT NULL DEFAULT 'unverified'"),
    ("model_prediction",    "TEXT"),
    ("verified_species",    "TEXT"),
]
_SUPABASE_EXCLUDED_KEYS = frozenset({"id", "color_hist", "verification_status", "model_prediction", "verified_species"})

# 2.  Schema
# ─────────────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS mosquito_registry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    mos_id        TEXT    NOT NULL UNIQUE,
    species       TEXT    NOT NULL,
    first_seen    TEXT    NOT NULL,
    last_seen     TEXT    NOT NULL,
    detect_count  INTEGER NOT NULL DEFAULT 1,
    confidence    REAL    NOT NULL,
    snapshot_path TEXT,
    bbox_x1       INTEGER, bbox_y1 INTEGER,
    bbox_x2       INTEGER, bbox_y2 INTEGER,
    color_hist    TEXT    NOT NULL,
    morph_ar      REAL,
    morph_area    REAL,
    synced              INTEGER NOT NULL DEFAULT 0,
    verification_status TEXT    NOT NULL DEFAULT 'unverified',
    model_prediction    TEXT,
    verified_species    TEXT
);
CREATE INDEX IF NOT EXISTS idx_mos_id ON mosquito_registry(mos_id);
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def _colour_hist(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [COLOR_HIST_BINS], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [COLOR_HIST_BINS], [0, 256])
    combined = np.concatenate([h_hist, s_hist], axis=0).flatten()
    total = combined.sum()
    if total > 0:
        combined /= total
    return combined.astype(np.float32)


def _morph_features(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0, 0.5
    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    ar   = float(bw) / max(float(bh), 1.0)
    norm = float(cv2.contourArea(c)) / max(float(bw * bh), 1.0)
    return ar, norm


def _similarity(hist_a, hist_b, ar_a, ar_b, norm_a, norm_b):
    bhat_dist = cv2.compareHist(hist_a.reshape(-1, 1), hist_b.reshape(-1, 1),
                                cv2.HISTCMP_BHATTACHARYYA)
    color_sim = max(0.0, 1.0 - bhat_dist)
    ar_diff   = abs(ar_a - ar_b)   / max(ar_a, ar_b, 0.01)
    norm_diff = abs(norm_a - norm_b)
    morph_sim = max(0.0, 1.0 - 0.5 * ar_diff - 0.5 * norm_diff)
    return COLOR_WEIGHT * color_sim + MORPH_WEIGHT * morph_sim


def _compute_iou(b1, b2):
    """Computes Intersection-over-Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / max(a1 + a2 - inter, 1)


def _centroid_distance(b1, b2):
    """Computes Euclidean distance between the centers of two bounding boxes."""
    c1x = (b1[0] + b1[2]) / 2.0
    c1y = (b1[1] + b1[3]) / 2.0
    c2x = (b2[0] + b2[2]) / 2.0
    c2y = (b2[1] + b2[3]) / 2.0
    return float(np.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2))


class ActiveMosquitoTrack:
    """Represents a live mosquito currently being tracked inside the chamber."""
    def __init__(self, track_id: int, mos_id: str, species: str, confidence: float,
                 box: Tuple[int, int, int, int], crop_bgr: Optional[np.ndarray], now_str: str):
        self.track_id = track_id
        self.mos_id = mos_id
        self.species = species
        self.confidence = float(confidence)
        self.box = box
        self.crop = crop_bgr
        self.first_seen = now_str
        self.last_seen = now_str
        self.last_seen_time = time.time()
        self.last_db_update = self.last_seen_time
        self.last_logged_time = self.last_seen_time
        self.missed_frames = 0
        self.hit_count = 1
        self.pending_db_hits = 0

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, box: Tuple[int, int, int, int], species: str, confidence: float,
               crop_bgr: Optional[np.ndarray], now_str: str):
        self.box = box
        if confidence > self.confidence or "unconfirmed" in self.species.lower():
            self.species = species
        self.confidence = max(self.confidence, float(confidence))
        if crop_bgr is not None and getattr(crop_bgr, "size", 0) > 0:
            self.crop = crop_bgr
        self.last_seen = now_str
        self.last_seen_time = time.time()
        self.missed_frames = 0
        self.hit_count += 1
        self.pending_db_hits += 1


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Supabase sync worker (background thread, non-blocking)
# ─────────────────────────────────────────────────────────────────────────────
class _SupabaseSyncWorker(threading.Thread):
    def __init__(self, db_path):
        super().__init__(daemon=True, name="SupabaseSyncWorker")
        self.db_path = db_path
        self.queue = queue.Queue()
        self._stop = threading.Event()

    def enqueue(self, row, snapshot_path=None):
        self.queue.put((row, snapshot_path))

    def stop(self):
        self._stop.set()

    def run(self):
        if not SUPABASE_ENABLED or not SUPABASE_URL or not SUPABASE_KEY:
            return
        try:
            from supabase import create_client
            supa = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"[Supabase] Could not connect: {e}")
            return

        while not self._stop.is_set():
            try:
                row, snapshot_path = self.queue.get(timeout=5)
            except queue.Empty:
                continue

            remote_url = None
            if snapshot_path and os.path.isfile(snapshot_path):
                try:
                    fname = os.path.basename(snapshot_path)
                    with open(snapshot_path, "rb") as f:
                        supa.storage.from_(SUPABASE_BUCKET).upload(
                            fname, f, {"content-type": "image/jpeg"}
                        )
                    remote_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{fname}"
                except Exception as e:
                    print(f"[Supabase] Image upload failed: {e}")

            mos_id = row.get("mos_id", row.get("id"))
            payload = {
                "id": mos_id,
                "species": row.get("species"),
                "confidence": float(row.get("confidence", 0.0)),
                "first_seen": row.get("first_seen"),
                "last_seen": row.get("last_seen"),
                "detect_count": int(row.get("detect_count", 1)),
                "bbox_x1": row.get("bbox_x1"),
                "bbox_y1": row.get("bbox_y1"),
                "bbox_x2": row.get("bbox_x2"),
                "bbox_y2": row.get("bbox_y2"),
            }
            if remote_url:
                payload["snapshot_url"] = remote_url
            try:
                supa.table(SUPABASE_TABLE).upsert(payload, on_conflict="id").execute()
                conn = sqlite3.connect(self.db_path)
                conn.execute("UPDATE mosquito_registry SET synced=1 WHERE mos_id=?", (mos_id,))
                conn.commit()
                conn.close()
                print(f"[Supabase] Synced {mos_id}")
            except Exception as e:
                print(f"[Supabase] Upsert failed for {mos_id}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MosquitoTracker -- main public class
# ─────────────────────────────────────────────────────────────────────────────
class MosquitoTracker:
    """
    Thread-safe mosquito ID registry backed by a local SQLite database.

    Public API:
        register(crop, species, confidence, box, snapshot_path)
            -> (mos_id: str, status: str)
               status in {"new", "duplicate", "unverified"}
        verify(mos_id, status, verified_species) -> (bool, str)
        get_all()   -> list of dicts
        get_count() -> int
        close()
    """

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.is_production_db = bool(os.path.abspath(db_path) == os.path.abspath(DB_PATH))
        self._lock   = threading.Lock()
        self._init_db()
        self._cache  = self._load_cache()
        if SUPABASE_ENABLED and self.is_production_db:
            self._sync = _SupabaseSyncWorker(db_path)
            self._sync.start()
        else:
            self._sync = None
        self._active_tracks = {}
        self._pending_tracks = []
        self._next_track_id = 1
        self.last_snapshot_time = 0.0
        print(f"[MosquitoTracker] SQLite registry ready -- {len(self._cache)} known mosquito(es).")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(_DDL)
            self._migrate_db(conn)
        finally:
            conn.close()

    def _migrate_db(self, conn):
        cur = conn.execute("PRAGMA table_info(mosquito_registry)")
        existing_cols = {row[1] for row in cur.fetchall()}
        missing = [(col, col_def) for col, col_def in _MIGRATION_COLUMNS if col not in existing_cols]
        if not missing:
            return

        print(f"[MosquitoTracker] Migrating schema: adding {len(missing)} missing column(s)...")
        old_isolation = conn.isolation_level
        conn.isolation_level = None  # Explicit transaction control for DDL across Python versions
        try:
            conn.execute("BEGIN IMMEDIATE")
            for col, col_def in missing:
                print(f"[MosquitoTracker]   -> ALTER TABLE mosquito_registry ADD COLUMN {col} {col_def}")
                conn.execute(f"ALTER TABLE mosquito_registry ADD COLUMN {col} {col_def}")
            conn.execute("COMMIT")
            print("[MosquitoTracker] Schema migration completed successfully.")
        except Exception as migration_err:
            if getattr(conn, "in_transaction", False):
                try:
                    conn.execute("ROLLBACK")
                except Exception as rollback_err:
                    print(f"[MosquitoTracker] Rollback failed: {rollback_err}")
            print(f"[MosquitoTracker] Migration failed: {migration_err}")
            raise RuntimeError(f"Database migration failed: {migration_err}") from migration_err
        finally:
            try:
                conn.isolation_level = old_isolation
            except Exception as iso_err:
                print(f"[MosquitoTracker] Failed to restore isolation level: {iso_err}")

    def _load_cache(self):
        cache = {}
        # If Supabase is enabled and running against the production DB, load records from Supabase into cache
        if SUPABASE_ENABLED and SUPABASE_URL and SUPABASE_KEY and getattr(self, "is_production_db", False):
            try:
                from supabase import create_client
                supa = create_client(SUPABASE_URL, SUPABASE_KEY)
                res = supa.table(SUPABASE_TABLE).select("*").execute()
                for r in res.data:
                    mos_id = r.get("id")
                    if mos_id:
                        cache[mos_id] = {
                            "mos_id": mos_id,
                            "species": r.get("species", "Mosquito - species unconfirmed"),
                            "first_seen": r.get("first_seen"),
                            "last_seen": r.get("last_seen"),
                            "detect_count": int(r.get("detect_count", 1)),
                            "confidence": float(r.get("confidence", 0.8)),
                            "color_hist": np.zeros(64, dtype=np.float32),
                            "morph_ar": 1.0,
                            "morph_area": 0.5,
                            "snapshot_path": r.get("snapshot_url", ""),
                        }
                if cache:
                    print(f"[MosquitoTracker] Loaded {len(cache)} record(s) from Supabase cloud database.")
            except Exception as e:
                print(f"[MosquitoTracker] Warning: could not load initial cache from Supabase ({e}); using SQLite.")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM mosquito_registry").fetchall()
        conn.close()
        for r in rows:
            d = dict(r)
            try:
                d["color_hist"] = np.array(json.loads(d["color_hist"]), dtype=np.float32)
            except Exception:
                d["color_hist"] = np.zeros(64, dtype=np.float32)
            cache[d["mos_id"]] = d
        return cache

    def _next_mos_id(self, conn):
        max_idx = 0
        row = conn.execute("SELECT MAX(id) FROM mosquito_registry").fetchone()
        if row and row[0]:
            max_idx = max(max_idx, int(row[0]))
        for mid in self._cache.keys():
            if mid.startswith("MOS-") and not mid.startswith("MOS-LEGACY-"):
                try:
                    num = int(mid.split("-")[1])
                    max_idx = max(max_idx, num)
                except Exception:
                    pass
            elif mid.startswith("MOS-LEGACY-"):
                try:
                    num = int(mid.split("-")[2])
                    max_idx = max(max_idx, num)
                except Exception:
                    pass
        next_idx = max_idx + 1
        return f"MOS-{next_idx:03d}"

    def _insert_new_mosquito(self, hist, ar, norm_area, species, confidence, box, snapshot_path, now):
        conn   = sqlite3.connect(self.db_path)
        mos_id = self._next_mos_id(conn)
        x1, y1, x2, y2 = box if box else (0, 0, 0, 0)
        row = {
            "mos_id":        mos_id,
            "species":       species,
            "first_seen":    now,
            "last_seen":     now,
            "detect_count":  1,
            "confidence":    round(confidence, 4),
            "snapshot_path": snapshot_path or "",
            "bbox_x1": x1, "bbox_y1": y1,
            "bbox_x2": x2, "bbox_y2": y2,
            "color_hist":    json.dumps(hist.tolist()),
            "morph_ar":      round(ar, 4),
            "morph_area":    round(norm_area, 4),
            "synced":              0,
            "verification_status": "unverified",
            "model_prediction":    species,
            "verified_species":    None,
        }
        conn.execute("""
            INSERT INTO mosquito_registry
              (mos_id, species, first_seen, last_seen, detect_count,
               confidence, snapshot_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
               color_hist, morph_ar, morph_area, synced,
               verification_status, model_prediction, verified_species)
            VALUES
              (:mos_id, :species, :first_seen, :last_seen, :detect_count,
               :confidence, :snapshot_path, :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
               :color_hist, :morph_ar, :morph_area, :synced,
               :verification_status, :model_prediction, :verified_species)
        """, row)
        conn.commit()
        conn.close()

        cache_entry               = dict(row)
        cache_entry["color_hist"] = hist
        self._cache[mos_id]       = cache_entry

        print(f"[MosquitoTracker] NEW  -> {mos_id}  ({species}  conf={confidence:.2f})")

        if SUPABASE_ENABLED and self._sync is not None:
            self._sync.enqueue(row, snapshot_path)

        return mos_id

    def _append_csv(self, timestamp, species, confidence, box, snapshot_filename, mos_id, status):
        log_file = getattr(config, 'LOG_FILE', 'detection_log.csv')
        file_exists = os.path.isfile(log_file)
        try:
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "species", "confidence",
                                     "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                                     "snapshot", "mos_id", "status"])
                x1, y1, x2, y2 = box if box else (0, 0, 0, 0)
                writer.writerow([timestamp, species, f"{confidence*100:.2f}%",
                                 x1, y1, x2, y2,
                                 snapshot_filename or "", mos_id, status])
        except Exception as e:
            print(f"[MosquitoTracker] CSV append error: {e}")

    def register(self, crop_bgr, species, confidence, box, snapshot_path=None):
        """
        Compare crop against known registry; assign/update ID accordingly.

        Returns:
            mos_id  -- assigned unique mosquito ID (e.g. "MOS-007")
            status  -- "new" | "duplicate" | "unverified"
        """
        hist          = _colour_hist(crop_bgr)
        ar, norm_area = _morph_features(crop_bgr)

        best_id    = None
        best_score = 0.0

        with self._lock:
            for mos_id, entry in self._cache.items():
                score = _similarity(
                    hist, entry["color_hist"],
                    ar,   entry.get("morph_ar") or 1.0,
                    norm_area, entry.get("morph_area") or 0.5,
                )
                if score > best_score:
                    best_score = score
                    best_id    = mos_id

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Duplicate
            if best_score >= SIMILARITY_HIGH and best_id is not None:
                self._update_existing(best_id, now, confidence)
                return best_id, "duplicate"

            # Unverified
            if SIMILARITY_LOW <= best_score < SIMILARITY_HIGH and best_id is not None:
                self._update_existing(best_id, now, confidence)
                return best_id, "unverified"

            # New mosquito
            mos_id = self._insert_new_mosquito(hist, ar, norm_area, species, confidence, box, snapshot_path, now)
            return mos_id, "new"

    def update_tracks(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Multi-mosquito spatial tracker for the chamber.
        Tracks each individual mosquito across frames and ensures simultaneous
        mosquitoes in the chamber receive unique, non-overlapping MOS-IDs.

        Args:
            detections: list of dicts with:
                - "box": (x1, y1, x2, y2)
                - "species": str
                - "confidence": float
                - "color": tuple (optional)
                - "crop": np.ndarray (optional)
            frame: full BGR frame (np.ndarray)

        Returns:
            list of dicts for all tracked mosquitoes in the current frame:
            [ {
                "box": (x1, y1, x2, y2),
                "mos_id": str,
                "species": str,
                "confidence": float,
                "status": "new" | "tracked",
                "color": tuple,
                "label": str,
                "snapshot_path": str
              }, ... ]
        """
        now_dt = datetime.now()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        now_time = time.time()

        max_dist = getattr(config, "TRACK_MAX_DISTANCE", 120)
        iou_thresh = getattr(config, "TRACK_IOU_THRESH", 0.15)
        max_missed = getattr(config, "TRACK_MAX_MISSED", 15)
        log_interval = getattr(config, "LOG_INTERVAL_SEC", 5.0)

        # Standardize detections
        sanitized_dets = []
        for d in detections:
            box = d.get("box") if isinstance(d, dict) else d[0]
            species = d.get("species", "Mosquito") if isinstance(d, dict) else d[1]
            conf = float(d.get("confidence", 0.5) if isinstance(d, dict) else d[2])
            color = d.get("color", (0, 0, 255)) if isinstance(d, dict) else (d[3] if len(d) > 3 else (0, 0, 255))
            crop = d.get("crop", None) if isinstance(d, dict) else (d[4] if len(d) > 4 else None)

            if (crop is None or getattr(crop, "size", 0) == 0) and frame is not None and getattr(frame, "size", 0) > 0:
                x1, y1, x2, y2 = box
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

            sanitized_dets.append({
                "box": box,
                "species": species,
                "confidence": conf,
                "color": color,
                "crop": crop
            })

        with self._lock:
            active_track_ids = list(self._active_tracks.keys())

            # Spatial matching between active tracks and detections
            candidate_matches = []
            for tid in active_track_ids:
                track = self._active_tracks[tid]
                for d_idx, det in enumerate(sanitized_dets):
                    dbox = det["box"]
                    iou = _compute_iou(track.box, dbox)
                    dist = _centroid_distance(track.box, dbox)
                    if iou >= iou_thresh or dist <= max_dist:
                        score = (iou * 2.0) + max(0.0, 1.0 - (dist / max(max_dist, 1.0)))
                        candidate_matches.append((score, tid, d_idx))

            # Greedy 1-to-1 matching (no detection or track shared)
            candidate_matches.sort(key=lambda x: x[0], reverse=True)
            matched_tracks = set()
            matched_dets = set()
            paired = []
            for score, tid, d_idx in candidate_matches:
                if tid not in matched_tracks and d_idx not in matched_dets:
                    matched_tracks.add(tid)
                    matched_dets.add(d_idx)
                    paired.append((tid, d_idx))

            results = [None] * len(sanitized_dets)

            # 1. Update matched active tracks
            for tid, d_idx in paired:
                track = self._active_tracks[tid]
                det = sanitized_dets[d_idx]
                box = det["box"]
                species = det["species"]
                conf = det["confidence"]
                color = det["color"]
                crop = det["crop"]

                track.update(box, species, conf, crop, now_str)

                # Periodic SQLite update
                if now_time - track.last_db_update >= 2.0:
                    inc = max(1, track.pending_db_hits)
                    self._update_existing(track.mos_id, now_str, conf, increment=inc, species=track.species)
                    track.pending_db_hits = 0
                    track.last_db_update = now_time

                # Periodic CSV logging
                if now_time - track.last_logged_time >= log_interval:
                    self._append_csv(now_str, track.species, conf, box, "", track.mos_id, "tracked")
                    track.last_logged_time = now_time

                label = f"{track.species}: {conf*100:.1f}%  [{track.mos_id}]"
                results[d_idx] = {
                    "box": box,
                    "mos_id": track.mos_id,
                    "species": track.species,
                    "confidence": conf,
                    "status": "tracked",
                    "color": color,
                    "label": label,
                    "snapshot_path": "",
                }

            # 2. Process unmatched detections (New mosquitoes in the chamber)
            active_mos_ids = {t.mos_id for t in self._active_tracks.values()}

            # --- Pending stage: confirm new detections before assigning IDs ---
            pending_needed = getattr(config, "TRACK_CONFIRM_FRAMES", 3)
            unmatched_idx = [i for i in range(len(sanitized_dets)) if i not in matched_dets]
            pend_candidates = []
            for p_i, pend in enumerate(self._pending_tracks):
                for d_idx in unmatched_idx:
                    dbox = sanitized_dets[d_idx]["box"]
                    iou = _compute_iou(pend["box"], dbox)
                    dist = _centroid_distance(pend["box"], dbox)
                    if iou >= iou_thresh or dist <= max_dist:
                        score = (iou * 2.0) + max(0.0, 1.0 - (dist / max(max_dist, 1.0)))
                        pend_candidates.append((score, p_i, d_idx))
            pend_candidates.sort(key=lambda x: x[0], reverse=True)
            used_pend = set()
            used_dets = set()
            new_pending = []
            confirmed_dets = set()
            for score, p_i, d_idx in pend_candidates:
                if p_i in used_pend or d_idx in used_dets:
                    continue
                used_pend.add(p_i)
                used_dets.add(d_idx)
                hits = self._pending_tracks[p_i]["hits"] + 1
                if hits >= pending_needed:
                    confirmed_dets.add(d_idx)
                else:
                    new_pending.append({"box": sanitized_dets[d_idx]["box"], "hits": hits})
            for d_idx in unmatched_idx:
                if d_idx not in used_dets:
                    new_pending.append({"box": sanitized_dets[d_idx]["box"], "hits": 1})
            self._pending_tracks = new_pending

            for d_idx, det in enumerate(sanitized_dets):
                if d_idx in matched_dets:
                    continue
                if d_idx not in confirmed_dets:
                    continue

                box = det["box"]
                species = det["species"]
                conf = det["confidence"]
                color = det["color"]
                crop = det["crop"]

                # Take snapshot with cooldown
                snap_path = ""
                snap_filename = ""
                if getattr(config, "AUTO_SAVE_SNAPSHOT", True) and (now_time - self.last_snapshot_time >= getattr(config, "SNAPSHOT_COOLDOWN_SEC", 2.0)):
                    clean_sp = species.replace(" ", "_")
                    snap_filename = f"{now_dt.strftime('%Y%m%d_%H%M%S')}_{clean_sp}_{int(conf*100)}pct.jpg"
                    captures_dir = getattr(config, "CAPTURES_DIR", "")
                    if captures_dir:
                        snap_path = os.path.join(captures_dir, snap_filename)
                        try:
                            if frame is not None and getattr(frame, "size", 0) > 0:
                                cv2.imwrite(snap_path, frame)
                                self.last_snapshot_time = now_time
                                print(f"[ALERT] Snapshot saved: {snap_filename}")
                        except Exception as e:
                            print(f"[MosquitoTracker] Snapshot error: {e}")
                            snap_path = ""
                            snap_filename = ""

                # Feature extraction
                if crop is not None and getattr(crop, "size", 0) > 0:
                    hist = _colour_hist(crop)
                    ar, norm_area = _morph_features(crop)
                else:
                    hist = np.zeros(COLOR_HIST_BINS * 2, dtype=np.float32)
                    ar, norm_area = 1.0, 0.5

                # Compare with cached mosquitoes not currently active in chamber
                best_id = None
                best_score = 0.0
                for mid, entry in self._cache.items():
                    if mid in active_mos_ids:
                        continue  # Must NEVER reuse an active mosquito's ID!
                    score = _similarity(hist, entry["color_hist"], ar, entry.get("morph_ar") or 1.0, norm_area, entry.get("morph_area") or 0.5)
                    if score > best_score:
                        best_score = score
                        best_id = mid

                if best_score >= SIMILARITY_HIGH and best_id is not None:
                    mos_id = best_id
                    status = "duplicate"
                    self._update_existing(mos_id, now_str, conf, species=species)
                else:
                    mos_id = self._insert_new_mosquito(hist, ar, norm_area, species, conf, box, snap_path, now_str)
                    status = "new"

                # Add to active tracks
                tid = self._next_track_id
                self._next_track_id += 1
                new_track = ActiveMosquitoTrack(tid, mos_id, species, conf, box, crop, now_str)
                new_track.last_logged_time = now_time
                self._active_tracks[tid] = new_track
                active_mos_ids.add(mos_id)

                # Log new detection to CSV
                self._append_csv(now_str, species, conf, box, snap_filename, mos_id, status)

                label = f"{species}: {conf*100:.1f}%  [{mos_id}]"
                results[d_idx] = {
                    "box": box,
                    "mos_id": mos_id,
                    "species": species,
                    "confidence": conf,
                    "status": status,
                    "color": color,
                    "label": label,
                    "snapshot_path": snap_path,
                }

            # 3. Update unmatched tracks (missed frames)
            for tid in active_track_ids:
                if tid not in matched_tracks:
                    track = self._active_tracks[tid]
                    track.missed_frames += 1
                    if track.missed_frames > max_missed:
                        del self._active_tracks[tid]

            return [r for r in results if r is not None]

    def _update_existing(self, mos_id, now, confidence, increment=1, species=None):
        conn = sqlite3.connect(self.db_path)
        if species and "unconfirmed" not in species.lower():
            conn.execute("""
                UPDATE mosquito_registry
                   SET last_seen    = ?,
                       detect_count = detect_count + ?,
                       confidence   = MAX(confidence, ?),
                       species      = CASE WHEN species LIKE '%unconfirmed%' THEN ? ELSE species END,
                       synced       = 0
                 WHERE mos_id = ?
            """, (now, increment, round(confidence, 4), species, mos_id))
        else:
            conn.execute("""
                UPDATE mosquito_registry
                   SET last_seen    = ?,
                       detect_count = detect_count + ?,
                       confidence   = MAX(confidence, ?),
                       synced       = 0
                 WHERE mos_id = ?
            """, (now, increment, round(confidence, 4), mos_id))
        conn.commit()
        conn.close()
        if mos_id in self._cache:
            self._cache[mos_id]["last_seen"]    = now
            self._cache[mos_id]["detect_count"] = self._cache[mos_id].get("detect_count", 1) + increment
            if species and "unconfirmed" not in species.lower():
                self._cache[mos_id]["species"] = species

        if SUPABASE_ENABLED and self._sync is not None:
            update_payload = {
                "mos_id": mos_id,
                "species": species or (self._cache.get(mos_id, {}).get("species")),
                "confidence": confidence,
                "first_seen": self._cache.get(mos_id, {}).get("first_seen", now),
                "last_seen": now,
                "detect_count": self._cache.get(mos_id, {}).get("detect_count", 1),
            }
            self._sync.enqueue(update_payload)

    def get_all(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT mos_id, species, first_seen, last_seen, detect_count, "
            "confidence, snapshot_path, synced, "
            "verification_status, model_prediction, verified_species "
            "FROM mosquito_registry ORDER BY id"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def verify(self, mos_id: str, status: str, verified_species: str):
        """
        Record independent reviewer confirmation or correction of a mosquito species.

        Semantics:
            - 'verified': Reviewer confirms the species. Requires non-empty verified_species
              matching model_prediction (when model_prediction is available).
            - 'corrected': Reviewer corrects the species. Requires non-empty verified_species
              differing from model_prediction (when model_prediction is available).
            - Historical records (model_prediction IS NULL): The caller must supply a species
              independently confirmed by the reviewer based on physical specimen identification
              or independent review. The historical 'species' column is NOT treated as proof of
              the model prediction and cannot be used for comparison. Both 'verified' and
              'corrected' require an independently confirmed non-empty verified_species.
            - Unidentified specimens remain unverified ('unverified'); NULL cannot be verified.
            - Original 'species' and 'model_prediction' are preserved permanently and never modified.

        Returns:
            (success: bool, reason: str)
        """
        if status not in ("verified", "corrected"):
            return False, f"Invalid status '{status}'. Permitted: 'verified', 'corrected'"

        if not verified_species or not isinstance(verified_species, str) or not verified_species.strip():
            return False, f"Status '{status}' requires a non-empty verified_species"

        cleaned_species = verified_species.strip()

        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cur = conn.execute(
                    "SELECT model_prediction FROM mosquito_registry WHERE mos_id = ?",
                    (mos_id,)
                )
                row = cur.fetchone()
                if not row:
                    return False, f"MOS-ID '{mos_id}' not found in registry"

                model_pred = row[0]
                if model_pred is not None:
                    if status == "verified" and cleaned_species != model_pred:
                        return False, (
                            f"Status 'verified' requires verified_species ('{cleaned_species}') "
                            f"to match model_prediction ('{model_pred}')"
                        )
                    if status == "corrected" and cleaned_species == model_pred:
                        return False, (
                            f"Status 'corrected' requires verified_species ('{cleaned_species}') "
                            f"to differ from model_prediction ('{model_pred}')"
                        )

                conn.execute("""
                    UPDATE mosquito_registry
                       SET verification_status = ?,
                           verified_species    = ?
                     WHERE mos_id = ?
                """, (status, cleaned_species, mos_id))
                conn.commit()

                if mos_id in self._cache:
                    self._cache[mos_id]["verification_status"] = status
                    self._cache[mos_id]["verified_species"]    = cleaned_species

                return True, f"Updated {mos_id} to status='{status}' ({cleaned_species})"
            except sqlite3.Error as e:
                if conn and getattr(conn, "in_transaction", False):
                    try:
                        conn.rollback()
                    except Exception as rb_err:
                        print(f"[MosquitoTracker] Rollback failed in verify(): {rb_err}")
                return False, f"Database error during verification: {e}"
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception as close_err:
                        print(f"[MosquitoTracker] Connection close failed in verify(): {close_err}")

    def get_count(self):
        return len(self._cache)

    def get_status(self) -> dict:
        """
        Thread-safe snapshot of currently active and pending mosquito tracks.
        Reads attributes verified from ActiveMosquitoTrack (confidence, species, hit_count, mos_id, box).
        """
        with self._lock:
            active_ids = [t.mos_id for t in self._active_tracks.values()]
            active_summary = [
                {
                    "mos_id": t.mos_id,
                    "species": t.species,
                    "confidence": round(t.confidence, 4),
                    "hit_count": t.hit_count,
                    "box": list(t.box) if t.box else None,
                }
                for t in self._active_tracks.values()
            ]
            return {
                "active_ids": active_ids,
                "active_tracks": active_summary,
                "active_count": len(active_ids),
                "pending_count": len(self._pending_tracks),
            }

    def flush(self):
        """Flush any pending active track statistics to SQLite."""
        with self._lock:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for track in self._active_tracks.values():
                if track.pending_db_hits > 0:
                    self._update_existing(track.mos_id, now_str, track.confidence, increment=track.pending_db_hits)
                    track.pending_db_hits = 0

    def close(self):
        self.flush()
        self._sync.stop()
