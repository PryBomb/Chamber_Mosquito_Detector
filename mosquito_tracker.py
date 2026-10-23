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
from datetime import datetime
from typing import Optional, Tuple

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
        if not SUPABASE_ENABLED:
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

            payload = {k: v for k, v in row.items() if k not in _SUPABASE_EXCLUDED_KEYS}
            if remote_url:
                payload["snapshot_url"] = remote_url
            try:
                supa.table(SUPABASE_TABLE).upsert(payload, on_conflict="mos_id").execute()
                conn = sqlite3.connect(self.db_path)
                conn.execute("UPDATE mosquito_registry SET synced=1 WHERE mos_id=?", (row["mos_id"],))
                conn.commit()
                conn.close()
                print(f"[Supabase] Synced {row['mos_id']}")
            except Exception as e:
                print(f"[Supabase] Upsert failed for {row['mos_id']}: {e}")


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
        self._lock   = threading.Lock()
        self._init_db()
        self._cache  = self._load_cache()
        self._sync   = _SupabaseSyncWorker(db_path)
        self._sync.start()
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM mosquito_registry").fetchall()
        conn.close()
        cache = {}
        for r in rows:
            d = dict(r)
            d["color_hist"] = np.array(json.loads(d["color_hist"]), dtype=np.float32)
            cache[d["mos_id"]] = d
        return cache

    def _next_mos_id(self, conn):
        row = conn.execute("SELECT MAX(id) FROM mosquito_registry").fetchone()
        next_idx = (row[0] or 0) + 1
        return f"MOS-{next_idx:03d}"

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
            conn   = sqlite3.connect(self.db_path)
            mos_id = self._next_mos_id(conn)
            x1, y1, x2, y2 = box
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

            if SUPABASE_ENABLED:
                self._sync.enqueue(row, snapshot_path)

            return mos_id, "new"

    def _update_existing(self, mos_id, now, confidence):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE mosquito_registry
               SET last_seen    = ?,
                   detect_count = detect_count + 1,
                   confidence   = MAX(confidence, ?),
                   synced       = 0
             WHERE mos_id = ?
        """, (now, round(confidence, 4), mos_id))
        conn.commit()
        conn.close()
        if mos_id in self._cache:
            self._cache[mos_id]["last_seen"]    = now
            self._cache[mos_id]["detect_count"] = self._cache[mos_id].get("detect_count", 1) + 1

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

    def close(self):
        self._sync.stop()
