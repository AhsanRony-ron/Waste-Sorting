import cv2
import numpy as np
import time
import os
import yaml
from datetime import datetime
import tensorflow as tf

# =============================================================
# KONFIGURASI (dibaca dari config.yaml yang SAMA dengan program raspi)
# =============================================================

CONFIG_PATH = "config.yaml"
CONFIG = {}
_config_mtime = 0


def load_config():
    global CONFIG, _config_mtime
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
    _config_mtime = os.path.getmtime(CONFIG_PATH)


def reload_config_if_changed():
    """
    Cek mtime config.yaml tiap loop. Berguna buat debug: bisa tuning
    crop/threshold di config.yaml sambil live feed jalan, tanpa restart.
    """
    global _config_mtime
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return False
    if mtime != _config_mtime:
        load_config()
        print(">>> [CONFIG] config.yaml berubah, direload.\n")
        return True
    return False


load_config()

# --- Model (path & class_names dipakai sekali saat load) ---
MODEL_PATH = CONFIG["model"]["path"]
CLASS_NAMES = CONFIG["model"]["class_names"]

# --- Path (dipakai sekali saat startup) ---
CAPTURE_DIR = CONFIG["paths"]["capture_dir"]
DEBUG_DIR = CONFIG["paths"]["debug_dir"]


def crop_center(frame, width=None, height=None, offset_x=0, offset_y=0):
    """
    Crop area di tengah frame dengan resolusi custom (width x height).
    Tidak harus persegi. Jika width/height None, dimensi tsb tidak dipotong.

    offset_x / offset_y bisa dipakai untuk menggeser titik tengah crop
    jika kamera tidak terpasang persis center terhadap objek.
    """
    h, w = frame.shape[:2]

    crop_w = min(width, w) if width else w
    crop_h = min(height, h) if height else h

    cx = w // 2 + offset_x
    cy = h // 2 + offset_y

    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    return frame[y1:y2, x1:x2]


def gray_world_correction(frame, gain_min=0.6, gain_max=1.6):
    """
    Menormalkan warna frame dengan asumsi rata-rata warna keseluruhan
    frame seharusnya netral (abu-abu). Menstabilkan warna/saturasi
    saat cahaya ambient berubah, tanpa perlu kalibrasi manual berulang.

    gain dibatasi (gain_min..gain_max) supaya tidak overcorrect saat
    frame didominasi satu warna (misal objek besar berwarna solid).
    """
    b, g, r = cv2.split(frame.astype(np.float32))
    b_avg, g_avg, r_avg = b.mean(), g.mean(), r.mean()
    gray_avg = (b_avg + g_avg + r_avg) / 3.0

    gain_b = np.clip(gray_avg / max(b_avg, 1e-6), gain_min, gain_max)
    gain_g = np.clip(gray_avg / max(g_avg, 1e-6), gain_min, gain_max)
    gain_r = np.clip(gray_avg / max(r_avg, 1e-6), gain_min, gain_max)

    b = np.clip(b * gain_b, 0, 255)
    g = np.clip(g * gain_g, 0, 255)
    r = np.clip(r * gain_r, 0, 255)

    return cv2.merge([b, g, r]).astype(np.uint8)


def preprocess_frame(raw_frame):
    """Crop + koreksi warna, parameter dibaca live dari CONFIG (sama dgn raspi)."""
    pp = CONFIG["preprocessing"]
    f = crop_center(raw_frame, pp["crop_width"], pp["crop_height"],
                     pp["crop_offset_x"], pp["crop_offset_y"])
    if pp["enable_color_correction"]:
        f = gray_world_correction(f, pp["color_gain_min"], pp["color_gain_max"])
    return f


# =============================================================
# SETUP MODEL
# =============================================================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for cname in CLASS_NAMES:
    os.makedirs(os.path.join(CAPTURE_DIR, cname), exist_ok=True)
os.makedirs(os.path.join(CAPTURE_DIR, "unknown"), exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


def classify(cropped_bgr):
    img = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_idx = np.argmax(output)
    confidence = output[predicted_idx]
    return CLASS_NAMES[predicted_idx], confidence, output


# =============================================================
# INISIALISASI KAMERA & REFERENCE FRAME
# =============================================================
# Bagian ini SENGAJA tidak diambil dari config.yaml -- lihat catatan di atas.
# Kalau mau tuning nilai di bawah, edit langsung di sini (bukan config.yaml).

cap = cv2.VideoCapture(0)

# --- Kunci Auto Exposure & Auto White Balance ---
# Mencegah kamera "meloncat" mengubah exposure/warna sendiri.
# Sisa variasi cahaya ditangani software lewat gray_world_correction() di atas.
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = manual mode (0.25 di beberapa driver Windows/DirectShow)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)       # sesuaikan nilai sesuai kondisi lighting-mu

cap.set(cv2.CAP_PROP_AUTO_WB, 1)         # matikan auto white balance
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 500)  # kunci di suhu warna tertentu (Kelvin)

# Opsional: kunci saturasi/brightness/contrast juga
cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
cap.set(cv2.CAP_PROP_CONTRAST, 128)
cap.set(cv2.CAP_PROP_SATURATION, 128)
cap.set(cv2.CAP_PROP_GAIN, 0)

print("Ambil frame referensi dalam 3 detik, pastikan area kosong...")
time.sleep(3)

ret, reference = cap.read()

print("Resolusi aktual:", reference.shape)
print("FPS setting:", cap.get(cv2.CAP_PROP_FPS))
print("Exposure aktual:", cap.get(cv2.CAP_PROP_EXPOSURE))
print("Backend:", cap.getBackendName())

reference = preprocess_frame(reference)

blur_k = CONFIG["preprocessing"]["gaussian_blur_kernel"]
reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
reference_gray = cv2.GaussianBlur(reference_gray, (blur_k, blur_k), 0)

frame_area = reference_gray.shape[0] * reference_gray.shape[1]

prev_gray = reference_gray.copy()
normal_stable_count = 0
immediate_confirm_count = 0
object_present = False
last_activity_time = time.time()
last_refresh_time = time.time()
detection_start_time = None
prev_frame_time = time.time()

print("Sistem siap (mode debug, TANPA serial ESP). Tekan 'r' refresh manual, 'q' keluar.")
print("(Ubah config.yaml kapan saja -- otomatis di-reload, kecuali parameter kamera)\n")


# =============================================================
# LOOP UTAMA
# =============================================================

while True:
    reload_config_if_changed()

    det = CONFIG["detection"]
    blur_k = CONFIG["preprocessing"]["gaussian_blur_kernel"]
    confidence_threshold = CONFIG["model"]["confidence_threshold"]
    label_to_preset = CONFIG["model"]["label_to_preset"]
    refresh_cfg = CONFIG["refresh"]
    refresh_flag_file = CONFIG["paths"]["refresh_flag_file"]

    ret, frame = cap.read()
    if not ret:
        print("Gagal capture frame")
        break

    frame = preprocess_frame(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # --- Diff terhadap reference ---
    diff_ref = cv2.absdiff(reference_gray, gray)
    thresh_ref = cv2.threshold(diff_ref, det["diff_threshold"], 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh_ref = cv2.erode(thresh_ref, kernel, iterations=1)
    thresh_ref = cv2.dilate(thresh_ref, kernel, iterations=2)
    change_area = cv2.countNonZero(thresh_ref)

    # --- Diff terhadap frame sebelumnya (motion) ---
    diff_prev = cv2.absdiff(prev_gray, gray)
    thresh_prev = cv2.threshold(diff_prev, det["diff_threshold"], 255, cv2.THRESH_BINARY)[1]
    motion_area = cv2.countNonZero(thresh_prev)

    contour_area = 0
    aspect_ratio = 0
    shape_valid = False
    bbox = None
    confidence_mode = "normal"
    triggered = False

    display_frame = frame.copy()

    immediate_capture_area_threshold = frame_area * det["immediate_capture_area_ratio"]

    if change_area > det["change_area_threshold"]:
        if detection_start_time is None:
            detection_start_time = time.perf_counter()

        contours, _ = cv2.findContours(thresh_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            bbox = (x, y, w, h)

            if contour_area >= det["min_contour_area"] and det["min_aspect_ratio"] < aspect_ratio < det["max_aspect_ratio"]:
                shape_valid = True

            if shape_valid:
                is_immediate_candidate = contour_area >= immediate_capture_area_threshold

                if is_immediate_candidate:
                    immediate_confirm_count += 1
                else:
                    immediate_confirm_count = 0

                if immediate_confirm_count >= det["immediate_confirm_frames"]:
                    triggered = True
                    confidence_mode = "immediate"
                else:
                    if motion_area < det["motion_tolerance_normal"]:
                        normal_stable_count += 1
                    else:
                        normal_stable_count = 0

                    if normal_stable_count >= det["stable_frames_needed_normal"]:
                        triggered = True
                        confidence_mode = "normal"
            else:
                immediate_confirm_count = 0
                normal_stable_count = 0

            box_color = (0, 255, 0) if shape_valid else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)

            if triggered and not object_present:
                t_stable_reached = time.perf_counter()
                detection_duration = t_stable_reached - detection_start_time

                x, y, w, h = bbox
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)

                t_crop_start = time.perf_counter()
                cropped_object = frame[y:y + h, x:x + w]
                t_crop_end = time.perf_counter()
                crop_duration = t_crop_end - t_crop_start

                t_infer_start = time.perf_counter()
                label, confidence, all_scores = classify(cropped_object)
                t_infer_end = time.perf_counter()
                infer_duration = t_infer_end - t_infer_start

                total_duration = t_infer_end - detection_start_time

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if confidence >= confidence_threshold:
                    save_path = os.path.join(CAPTURE_DIR, label, f"{timestamp}.jpg")
                else:
                    save_path = os.path.join(CAPTURE_DIR, "unknown", f"{timestamp}_{label}_{confidence:.2f}.jpg")
                cv2.imwrite(save_path, cropped_object)
                cv2.imshow("Cropped Object", cropped_object)

                print(f"\n>>> STABIL & VALID [{confidence_mode}] -> disimpan {save_path}")
                print(f"    Prediksi: {label} ({confidence * 100:.2f}%)")
                for i, cname in enumerate(CLASS_NAMES):
                    print(f"      {cname}: {all_scores[i] * 100:.2f}%")
                print(f"    --- Timing ---")
                print(f"    Deteksi -> stabil : {detection_duration * 1000:.1f} ms")
                print(f"    Crop              : {crop_duration * 1000:.2f} ms")
                print(f"    Inference model   : {infer_duration * 1000:.1f} ms")
                print(f"    TOTAL (deteksi->hasil): {total_duration * 1000:.1f} ms")

                if confidence >= confidence_threshold and label in label_to_preset:
                    print(f"    -> (Simulasi) Akan kirim preset {label_to_preset[label]} ke ESP\n")
                elif label == 'background':
                    print(f"    -> Terdeteksi background, langsung update referensi\n")
                    reference_gray = gray.copy()
                    last_refresh_time = time.time()
                else:
                    print(f"    -> Confidence rendah / kelas tidak disortir\n")

                object_present = True
                last_activity_time = time.time()
        else:
            normal_stable_count = 0
            immediate_confirm_count = 0
            if object_present:
                print(">>> Objek hilang / hanya noise, reset baseline.\n")
            object_present = False
    else:
        if object_present:
            print(">>> Area kosong lagi, reset baseline.\n")
        object_present = False
        normal_stable_count = 0
        immediate_confirm_count = 0
        detection_start_time = None

    # --- Refresh reference (otomatis / manual) ---
    time_since_activity = time.time() - last_activity_time
    time_since_last_refresh = time.time() - last_refresh_time

    should_force_refresh = (
        not object_present and
        time_since_activity >= refresh_cfg["force_refresh_timeout"] and
        time_since_last_refresh >= refresh_cfg["refresh_cooldown"]
    )
    manual_refresh_requested = os.path.exists(refresh_flag_file)

    # --- Overlay debug: DIGAMBAR SEBELUM imshow, biar benar-benar kelihatan
    # di window (kalau digambar sesudah imshow, teksnya cuma nempel di frame
    # yang sudah dibuang, gak pernah kelihatan) ---
    now_frame_time = time.time()
    fps = 1.0 / max(now_frame_time - prev_frame_time, 1e-6)
    prev_frame_time = now_frame_time

    bbox_text = f"bbox=({bbox[0]},{bbox[1]},{bbox[2]}x{bbox[3]})" if bbox else "bbox=-"

    # Panel kiri: status runtime (angka yang lagi terjadi tiap frame)
    status_lines = [
        f"FPS={fps:.1f}  frame={frame.shape[1]}x{frame.shape[0]}",
        f"change_area={change_area}  motion_area={motion_area}",
        f"contour_area={contour_area:.0f}  aspect_ratio={aspect_ratio:.2f}  {bbox_text}",
        f"mode={confidence_mode}  normal_stable={normal_stable_count}  immediate={immediate_confirm_count}",
        f"object_present={object_present}",
        f"sejak aktivitas={time_since_activity:.1f}s  sejak refresh={time_since_last_refresh:.1f}s",
    ]

    # Panel kanan: nilai config yang lagi aktif (dari config.yaml)
    config_lines = [
        "-- config aktif --",
        f"diff_thr={det['diff_threshold']}  change_area_thr={det['change_area_threshold']}",
        f"min_contour={det['min_contour_area']}  aspect=({det['min_aspect_ratio']:.2f}-{det['max_aspect_ratio']:.2f})",
        f"stable_frames={det['stable_frames_needed_normal']}  motion_tol={det['motion_tolerance_normal']}",
        f"immediate_ratio={det['immediate_capture_area_ratio']:.2f}  immediate_frames={det['immediate_confirm_frames']}",
        f"immediate_area_thr={immediate_capture_area_threshold:.0f}px",
        f"blur_kernel={blur_k}  confidence_thr={confidence_threshold:.2f}",
        f"crop={CONFIG['preprocessing']['crop_width']}x{CONFIG['preprocessing']['crop_height']}",
    ]

    for i, line in enumerate(status_lines):
        cv2.putText(display_frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    panel_x = max(display_frame.shape[1] - 380, 10)
    for i, line in enumerate(config_lines):
        cv2.putText(display_frame, line, (panel_x, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    cv2.imshow("Live Feed", display_frame)
    cv2.imshow("Diff Mask", thresh_ref)
    key = cv2.waitKey(1) & 0xFF

    if should_force_refresh or manual_refresh_requested or key == ord('r'):
        if manual_refresh_requested:
            os.remove(refresh_flag_file)

        check_label, check_confidence, check_scores = classify(frame)

        if check_label == 'background' and check_confidence >= confidence_threshold:
            reference_gray = gray.copy()
            last_refresh_time = time.time()
            last_activity_time = time.time()

            if key == ord('r'):
                reason = "manual (tombol r)"
            elif manual_refresh_requested:
                reason = "manual (file trigger)"
            else:
                reason = f"otomatis (tidak ada aktivitas {time_since_activity:.1f}s)"
            print(f">>> [REFRESH] Terverifikasi background ({check_confidence * 100:.1f}%) — referensi diperbarui, {reason}\n")
        else:
            print(f">>> [REFRESH DITUNDA] Frame terdeteksi sebagai '{check_label}' ({check_confidence * 100:.1f}%), bukan background.")
            print(f"    (Simulasi tanpa ESP: tidak ada aksi mekanis, cuma dicatat)\n")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stuck_path = os.path.join(CAPTURE_DIR, "unknown", f"stuck_{timestamp}_{check_label}_{check_confidence:.2f}.jpg")
            cv2.imwrite(stuck_path, frame)

            last_refresh_time = time.time()
            last_activity_time = time.time()

    prev_gray = gray.copy()

    now_frame_time = time.time()
    fps = 1.0 / max(now_frame_time - prev_frame_time, 1e-6)
    prev_frame_time = now_frame_time

    bbox_text = f"bbox=({bbox[0]},{bbox[1]},{bbox[2]}x{bbox[3]})" if bbox else "bbox=-"

    # --- Panel kiri: status runtime (angka yang lagi terjadi tiap frame) ---
    status_lines = [
        f"FPS={fps:.1f}  frame={frame.shape[1]}x{frame.shape[0]}",
        f"change_area={change_area}  motion_area={motion_area}",
        f"contour_area={contour_area:.0f}  aspect_ratio={aspect_ratio:.2f}  {bbox_text}",
        f"mode={confidence_mode}  normal_stable={normal_stable_count}  immediate={immediate_confirm_count}",
        f"object_present={object_present}",
        f"sejak aktivitas={time_since_activity:.1f}s  sejak refresh={time_since_last_refresh:.1f}s",
    ]

    # --- Panel kanan: nilai config yang lagi aktif (dari config.yaml) ---
    config_lines = [
        "-- config aktif --",
        f"diff_thr={det['diff_threshold']}  change_area_thr={det['change_area_threshold']}",
        f"min_contour={det['min_contour_area']}  aspect=({det['min_aspect_ratio']:.2f}-{det['max_aspect_ratio']:.2f})",
        f"stable_frames={det['stable_frames_needed_normal']}  motion_tol={det['motion_tolerance_normal']}",
        f"immediate_ratio={det['immediate_capture_area_ratio']:.2f}  immediate_frames={det['immediate_confirm_frames']}",
        f"immediate_area_thr={immediate_capture_area_threshold:.0f}px",
        f"blur_kernel={blur_k}  confidence_thr={confidence_threshold:.2f}",
        f"crop={CONFIG['preprocessing']['crop_width']}x{CONFIG['preprocessing']['crop_height']}",
    ]

    for i, line in enumerate(status_lines):
        cv2.putText(display_frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    panel_x = display_frame.shape[1] - 380
    for i, line in enumerate(config_lines):
        cv2.putText(display_frame, line, (panel_x, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()