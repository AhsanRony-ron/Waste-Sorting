"""
Analisis proporsi data REAL (webcam) vs TRASHNET dalam dataset training,
plus fungsi untuk menambah data dari TrashNet secara terarah per kelas
yang cocok, sampai proporsi trashnet mencapai target (20-30%).

Cara deteksi sumber:
- Data REAL  -> nama file mengandung pola timestamp: 8 digit tanggal + underscore + 6 digit jam
                contoh: 20250611_143022_123456.jpg, crop_20250611_143022.jpg, dsb.
- Data TRASHNET -> nama file TIDAK mengandung pola timestamp tsb (biasanya diawali nama kelas).

Output:
1. Tabel ringkasan jumlah & persentase real vs trashnet per kelas
2. File CSV berisi mapping tiap file -> kelas -> sumber (buat dipakai nanti
   untuk oversampling / split validation manual)
3. (Opsional) Penambahan file dari folder TrashNet mentah ke dataset,
   dipetakan per kelas yang cocok, sampai target rasio trashnet tercapai.
"""

import os
import re
import csv
import random
import shutil
from collections import defaultdict

DATASET_DIR = "dataset"          # sesuaikan kalau lokasi beda
OUTPUT_CSV = "dataset_source_mapping.csv"

# Pola timestamp: 8 digit tanggal + _ + 6 digit jam (opsional ada _digit tambahan / mikrodetik)
# Contoh yang match: 20250611_143022, 20250611_143022_123456, crop_20250611_143022_abc
TIMESTAMP_PATTERN = re.compile(r"\d{8}_\d{6}")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ===================== Konfigurasi penambahan data TrashNet =====================

# Folder sumber TrashNet mentah (folder ini berisi subfolder: cardboard, glass,
# metal, paper, plastic, trash -- struktur asli dataset-resized TrashNet)
TRASHNET_SOURCE_DIR = r"C:\Users\ASUS\Documents\Waste Sorting\dataset-resized"

# Pemetaan kelas project -> subfolder TrashNet yang cocok.
# 'daun' dan 'background' sengaja tidak dipetakan -- TrashNet tidak punya
# kategori sampah organik atau piringan kosong.
CLASS_TO_TRASHNET = {
    "kaleng": "metal",
    "kertas": "cardboard",
    "plastik": "plastic",
}

# Subfolder TrashNet yang tidak dipakai sama sekali (tidak match kelas manapun)
UNUSED_TRASHNET_FOLDERS = ["paper", "glass", "trash"]

# Target proporsi akhir trashnet per kelas (titik tengah dari rentang 20-30%)
TARGET_TRASHNET_RATIO = 0.25

RANDOM_SEED = 42


def detect_source(filename: str) -> str:
    """Return 'real' kalau nama file mengandung pola timestamp, else 'trashnet'."""
    if TIMESTAMP_PATTERN.search(filename):
        return "real"
    return "trashnet"


def build_summary():
    """Scan DATASET_DIR, kembalikan (summary, rows, unmatched_examples, class_folders)."""
    summary = defaultdict(lambda: {"real": 0, "trashnet": 0})
    rows = []
    unmatched_examples = defaultdict(list)  # buat sanity-check manual kalau perlu

    class_folders = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

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
            if source == "trashnet" and any(ch.isdigit() for ch in fname):
                if len(unmatched_examples[class_name]) < 5:
                    unmatched_examples[class_name].append(fname)

    return summary, rows, unmatched_examples, class_folders


def print_summary(summary, class_folders):
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


def add_trashnet_samples(summary, dry_run=True):
    """
    Tambah gambar dari TRASHNET_SOURCE_DIR ke DATASET_DIR, per kelas yang
    dipetakan di CLASS_TO_TRASHNET, sampai proporsi trashnet di kelas itu
    mencapai TARGET_TRASHNET_RATIO (dihitung dari jumlah real yang sudah ada,
    bukan asal ambil sekian persen dari total TrashNet).

    dry_run=True (default): cuma hitung & preview, TIDAK menyalin file apapun.
    Set dry_run=False untuk benar-benar menyalin.
    """
    if not os.path.isdir(TRASHNET_SOURCE_DIR):
        print(f"[ERROR] Folder sumber TrashNet tidak ditemukan: {TRASHNET_SOURCE_DIR}")
        return

    print("\n" + "=" * 70)
    mode_label = "DRY RUN (preview saja, belum menyalin apapun)" if dry_run else "MENYALIN FILE"
    print(f"PENAMBAHAN DATA TRASHNET -- {mode_label}")
    print(f"Target proporsi trashnet per kelas: {TARGET_TRASHNET_RATIO * 100:.0f}%")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    for class_name, trashnet_subfolder in CLASS_TO_TRASHNET.items():
        real_count = summary[class_name]["real"]
        existing_trashnet = summary[class_name]["trashnet"]

        if real_count == 0:
            print(f"  - '{class_name}': tidak ada data real sama sekali, dilewati "
                  f"(butuh baseline real dulu sebelum nambah trashnet).")
            continue

        src_dir = os.path.join(TRASHNET_SOURCE_DIR, trashnet_subfolder)
        if not os.path.isdir(src_dir):
            print(f"  - '{class_name}': folder sumber '{trashnet_subfolder}' tidak ditemukan, dilewati.")
            continue

        # target_trashnet_total / (real_count + target_trashnet_total) = TARGET_TRASHNET_RATIO
        target_trashnet_total = int(round(
            real_count * TARGET_TRASHNET_RATIO / (1 - TARGET_TRASHNET_RATIO)
        ))
        need_to_add = target_trashnet_total - existing_trashnet

        if need_to_add <= 0:
            print(f"  - '{class_name}': sudah punya {existing_trashnet} trashnet "
                  f"(target {target_trashnet_total}), tidak perlu tambah.")
            continue

        available_files = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        if len(available_files) < need_to_add:
            print(f"  - '{class_name}': butuh {need_to_add} gambar tapi sumber cuma "
                  f"punya {len(available_files)}. Akan pakai semua yang tersedia.")
            need_to_add = len(available_files)

        if need_to_add == 0:
            print(f"  - '{class_name}': sumber '{trashnet_subfolder}' kosong, dilewati.")
            continue

        selected = random.sample(available_files, need_to_add)
        dest_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        copied = 0
        for fname in selected:
            src_path = os.path.join(src_dir, fname)
            # prefix biar gampang dikenali asalnya & gak tabrakan nama sama file lain
            dest_fname = f"trashnet_{trashnet_subfolder}_{fname}"
            dest_path = os.path.join(dest_dir, dest_fname)
            if os.path.exists(dest_path):
                continue  # sudah pernah disalin sebelumnya, skip
            if not dry_run:
                shutil.copy2(src_path, dest_path)
            copied += 1

        new_total = real_count + existing_trashnet + copied
        new_pct_trashnet = (existing_trashnet + copied) / new_total * 100 if new_total else 0
        print(f"  - '{class_name}': +{copied} gambar dari '{trashnet_subfolder}' "
              f"-> total jadi {new_total}, trashnet {new_pct_trashnet:.1f}%")

    unmapped_classes = [c for c in summary if c not in CLASS_TO_TRASHNET]
    if unmapped_classes:
        print(f"\n[INFO] Kelas tanpa padanan TrashNet (dilewati): {', '.join(unmapped_classes)}")
    print(f"[INFO] Folder TrashNet tidak dipakai (tidak match kelas project): "
          f"{', '.join(UNUSED_TRASHNET_FOLDERS)}")

    if dry_run:
        print("\n[INFO] Ini masih DRY RUN, belum ada file yang benar-benar disalin.")
        print("       Set dry_run=False di pemanggilan add_trashnet_samples() untuk eksekusi nyata.")


def save_csv(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "class", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Mapping lengkap {len(rows)} file disimpan ke '{OUTPUT_CSV}'")


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Folder '{DATASET_DIR}' tidak ditemukan. "
              f"Sesuaikan variabel DATASET_DIR di script ini.")
        return

    summary, rows, unmatched_examples, class_folders = build_summary()

    if not class_folders:
        print(f"[ERROR] Tidak ada subfolder kelas di dalam '{DATASET_DIR}'.")
        return

    print_summary(summary, class_folders)

    any_unmatched = any(unmatched_examples[c] for c in class_folders)
    if any_unmatched:
        print("\n[SANITY CHECK] Beberapa file dianggap 'trashnet' tapi mengandung digit "
              "(cek manual kalau-kalau ada format timestamp lain yang tidak terdeteksi):")
        for class_name in class_folders:
            examples = unmatched_examples[class_name]
            if examples:
                print(f"  - {class_name}: {examples}")

    save_csv(rows)

    # ===== Penambahan data TrashNet =====
    # Jalankan dulu dengan dry_run=True buat lihat preview jumlah yang akan
    # ditambah. Kalau sudah sesuai, ganti dry_run=False lalu jalankan ulang.
    add_trashnet_samples(summary, dry_run=True)


if __name__ == "__main__":
    main()