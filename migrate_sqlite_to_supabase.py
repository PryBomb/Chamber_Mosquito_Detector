"""
migrate_sqlite_to_supabase.py
=============================
Migrates existing SQLite records and CSV detection logs into Supabase table 'mosquito_registry'.
"""

import os
import sys
import csv
import sqlite3
from datetime import datetime

# Windows console encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import config

try:
    from supabase import create_client
except ImportError:
    print("[ERROR] 'supabase' package is not installed. Run: pip install supabase")
    sys.exit(1)


def get_supabase_client():
    url = getattr(config, "SUPABASE_URL", "")
    key = getattr(config, "SUPABASE_KEY", "")
    if not url or not key:
        print("[ERROR] SUPABASE_URL or SUPABASE_KEY not configured in config.py")
        sys.exit(1)
    return create_client(url, key)


def migrate_sqlite_registry(supa):
    db_path = getattr(config, "TRACKER_DB_PATH", os.path.join(REPO_ROOT, "mosquito_registry.db"))
    if not os.path.isfile(db_path):
        print(f"[INFO] SQLite database {db_path} does not exist. Skipping.")
        return 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        rows = cur.execute("SELECT * FROM mosquito_registry").fetchall()
    except Exception as e:
        print(f"[WARNING] Could not read SQLite mosquito_registry: {e}")
        conn.close()
        return 0

    conn.close()
    print(f"[INFO] Found {len(rows)} record(s) in SQLite database.")
    if not rows:
        return 0

    count = 0
    for r in rows:
        payload = {
            "id": r["mos_id"],
            "species": r["species"],
            "confidence": float(r["confidence"]),
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "detect_count": int(r["detect_count"]),
            "bbox_x1": int(r["bbox_x1"]) if r["bbox_x1"] is not None else None,
            "bbox_y1": int(r["bbox_y1"]) if r["bbox_y1"] is not None else None,
            "bbox_x2": int(r["bbox_x2"]) if r["bbox_x2"] is not None else None,
            "bbox_y2": int(r["bbox_y2"]) if r["bbox_y2"] is not None else None,
            "snapshot_url": None
        }
        try:
            supa.table(config.SUPABASE_TABLE).upsert(payload, on_conflict="id").execute()
            count += 1
            print(f"  [SQLite -> Supabase] Migrated {payload['id']} ({payload['species']})")
        except Exception as e:
            print(f"  [ERROR] Failed to upsert {payload['id']}: {e}")

    return count


def migrate_detection_csv(supa):
    csv_path = os.path.join(REPO_ROOT, "detection_log.csv")
    if not os.path.isfile(csv_path):
        print(f"[INFO] CSV file {csv_path} does not exist. Skipping.")
        return 0

    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"[INFO] Found {len(rows)} record(s) in detection_log.csv.")
    if not rows:
        return 0

    count = 0
    for i, r in enumerate(rows, start=1):
        ts = r.get("Timestamp", "").strip()
        sp = r.get("Species", "Mosquito - species unconfirmed").strip()
        conf_str = r.get("Confidence", "0%").replace("%", "").strip()
        try:
            conf = float(conf_str) / 100.0 if float(conf_str) > 1.0 else float(conf_str)
        except Exception:
            conf = 0.85

        mos_id = f"MOS-LEGACY-{i:03d}"
        x1 = int(r.get("BBox_X1")) if r.get("BBox_X1") else None
        y1 = int(r.get("BBox_Y1")) if r.get("BBox_Y1") else None
        x2 = int(r.get("BBox_X2")) if r.get("BBox_X2") else None
        y2 = int(r.get("BBox_Y2")) if r.get("BBox_Y2") else None

        payload = {
            "id": mos_id,
            "species": sp,
            "confidence": conf,
            "first_seen": ts,
            "last_seen": ts,
            "detect_count": 1,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "snapshot_url": None
        }
        try:
            supa.table(config.SUPABASE_TABLE).upsert(payload, on_conflict="id").execute()
            count += 1
            print(f"  [CSV -> Supabase] Migrated {mos_id} ({sp}, {ts})")
        except Exception as e:
            print(f"  [ERROR] Failed to upsert {mos_id}: {e}")

    return count


def main():
    print("=" * 60)
    print("MIGRATING LOCAL SQLITE & CSV RECORDS TO SUPABASE")
    print("=" * 60)
    supa = get_supabase_client()
    sqlite_migrated = migrate_sqlite_registry(supa)
    csv_migrated = migrate_detection_csv(supa)
    print("=" * 60)
    print(f"Migration completed! Migrated {sqlite_migrated} SQLite and {csv_migrated} CSV records.")
    print("=" * 60)


if __name__ == "__main__":
    main()
