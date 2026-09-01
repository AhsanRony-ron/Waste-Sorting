"""
Analisis proporsi data REAL (webcam) vs TRASHNET dalam dataset training.

Cara deteksi:
- Data REAL  -> nama file mengandung pola timestamp: 8 digit tanggal + underscore + 6 digit jam
                contoh: 20250611_143022_123456.jpg, crop_20250611_143022.jpg, dsb.
- Data TRASHNET -> nama file TIDAK mengandung pola timestamp tsb (biasanya diawali nama kelas).

Output:
1. Tabel ringkasan jumlah & persentase real vs trashnet per kelas
2. File CSV berisi mapping tiap file -> kelas -> sumber (buat dipakai nanti
   untuk oversampling / split validation manual)
"""

import os
import re
import csv
from collections import defaultdict

DATASET_DIR = "dataset"          # sesuaikan kalau lokasi beda
OUTPUT_CSV = "dataset_source_mapping.csv"

# Pola timestamp: 8 digit tanggal + _ + 6 digit jam (opsional ada _digit tambahan / mikrodetik)
# Contoh yang match: 20250611_143022, 20250611_143022_123456, crop_20250611_143022_abc
TIMESTAMP_PATTERN = re.compile(r"\d{8}_\d{6}")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def detect_source(filename: str) -> str:
    """Return 'real' kalau nama file mengandung pola timestamp, else 'trashnet'."""
    if TIMESTAMP_PATTERN.search(filename):
        return "real"
    return "trashnet"


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Folder '{DATASET_DIR}' tidak ditemukan. "
              f"Sesuaikan variabel DATASET_DIR di script ini.")
        return

    summary = defaultdict(lambda: {"real": 0, "trashnet": 0})
    rows = []
    unmatched_examples = defaultdict(list)  # buat sanity-check manual kalau perlu

    class_folders = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    if not class_folders:
        print(f"[ERROR] Tidak ada subfolder kelas di dalam '{DATASET_DIR}'.")
        return

    for class_name in class_folders:
        class_path = os.path.join(DATASET_DIR, class_name)
        files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        for fname in files:
            source = detect_source(fname)
            summary[class_name][source] += 1
            rows.append({
                "filepath": os.path.join(class_path, fname),
                "class": class_name,
                "source": source,
            })
            # simpan sedikit contoh nama file yg diklasifikasi 'trashnet'
            # tapi ada digit-digit mencurigakan, buat manual double check
            if source == "trashnet" and any(ch.isdigit() for ch in fname):
                if len(unmatched_examples[class_name]) < 5:
                    unmatched_examples[class_name].append(fname)

    # ===== Cetak ringkasan =====
    print("\n" + "=" * 70)
    print(f"{'Kelas':<15}{'Real':>10}{'TrashNet':>12}{'Total':>10}{'% Real':>12}")
    print("=" * 70)

    total_real = 0
    total_trashnet = 0

    for class_name in class_folders:
        r = summary[class_name]["real"]
        t = summary[class_name]["trashnet"]
        total = r + t
        pct_real = (r / total * 100) if total > 0 else 0
        total_real += r
        total_trashnet += t
        print(f"{class_name:<15}{r:>10}{t:>12}{total:>10}{pct_real:>11.1f}%")

    grand_total = total_real + total_trashnet
    grand_pct = (total_real / grand_total * 100) if grand_total > 0 else 0
    print("-" * 70)
    print(f"{'TOTAL':<15}{total_real:>10}{total_trashnet:>12}{grand_total:>10}{grand_pct:>11.1f}%")
    print("=" * 70)

    # ===== Peringatan kalau proporsi timpang =====
    print("\n[CATATAN]")
    for class_name in class_folders:
        r = summary[class_name]["real"]
        t = summary[class_name]["trashnet"]
        total = r + t
        if total == 0:
            continue
        pct_real = r / total * 100
        if pct_real < 10:
            print(f"  - '{class_name}': data real cuma {pct_real:.1f}% dari total "
                  f"({r} dari {total}) -> risiko domain shift TINGGI, "
                  f"pertimbangkan oversampling.")
        elif pct_real < 25:
            print(f"  - '{class_name}': data real {pct_real:.1f}% -> masih minoritas, "
                  f"waspadai saat evaluasi.")

    # ===== Contoh file 'trashnet' yang punya digit tapi tidak match pola timestamp =====
    # (buat sanity check, siapa tahu ada format tanggal lain yang kelewat)
    any_unmatched = any(unmatched_examples[c] for c in class_folders)
    if any_unmatched:
        print("\n[SANITY CHECK] Beberapa file dianggap 'trashnet' tapi mengandung digit "
              "(cek manual kalau-kalau ada format timestamp lain yang tidak terdeteksi):")
        for class_name in class_folders:
            examples = unmatched_examples[class_name]
            if examples:
                print(f"  - {class_name}: {examples}")

    # ===== Simpan CSV mapping =====
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "class", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Mapping lengkap {len(rows)} file disimpan ke '{OUTPUT_CSV}'")
    print("File ini bisa dipakai buat langkah selanjutnya: oversampling data real, "
          "atau bikin validation split manual yang representatif.")


if __name__ == "__main__":
    main()