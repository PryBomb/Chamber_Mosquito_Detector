"""
view_db.py - Quick terminal viewer for mosquito_registry.db
Usage:
    python view_db.py
"""
import sqlite3
import os
import sys

def view(db_path="mosquito_registry.db"):
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "mosquito_registry" not in tables:
        print(f"[INFO] Database '{db_path}' exists, but 'mosquito_registry' table has not been created yet.")
        conn.close()
        return

    rows = conn.execute("""
        SELECT mos_id, species, confidence, detect_count, 
               verification_status, model_prediction, verified_species, 
               first_seen, last_seen 
        FROM mosquito_registry 
        ORDER BY id
    """).fetchall()

    if not rows:
        print(f"[INFO] 'mosquito_registry' in '{db_path}' is currently empty (0 records).")
        conn.close()
        return

    print("\n" + "=" * 115)
    print(f"{'MOS ID':<10} {'SPECIES':<20} {'CONF':<8} {'COUNT':<7} {'STATUS':<14} {'PREDICTION':<20} {'LAST SEEN'}")
    print("=" * 115)
    for r in rows:
        conf_str = f"{r['confidence']*100:.1f}%" if r['confidence'] is not None else "N/A"
        pred_str = str(r['model_prediction'] or '-')
        print(f"{r['mos_id']:<10} {r['species']:<20} {conf_str:<8} {r['detect_count']:<7} {r['verification_status']:<14} {pred_str:<20} {r['last_seen']}")
    print("=" * 115)
    print(f"Total: {len(rows)} registered mosquito(es)\n")
    conn.close()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "mosquito_registry.db"
    view(path)
