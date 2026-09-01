import cv2
import numpy as np
import time
import os
import serial
from datetime import datetime
from ai_edge_litert.interpreter import Interpreter

# ===== Setup serial ke ESP =====
SERIAL_PORT = '/dev/ttyUSB0'  # sesuaikan hasil ls /dev/ttyUSB*
ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
time.sleep(2)
print("Serial ke ESP:", ser.read(200).decode(errors='ignore'))

# ===== Setup model =====
interpreter = Interpreter(model_path="waste_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = ['background', 'daun', 'kaleng', 'kertas', 'plastik']

# Mapping label -> nomor preset ESP (SESUAIKAN dengan preset fisik piringan kamu)
label_to_preset = {
    'kertas': 1,
    'plastik': 2,
    'kaleng': 3,
    'daun': 4,
}
# 'background' sengaja TIDAK dimasukkan -> tidak akan pernah kirim ke ESP

CONFIDENCE_THRESHOLD = 0.6

# ===== Folder simpan hasil capture per kelas =====
CAPTURE_DIR = "captured_data"
for cname in class_names:
    os.makedirs(os.path.join(CAPTURE_DIR, cname), exist_ok=True)
os.makedirs(os.path.join(CAPTURE_DIR, "unknown"), exist_ok=True)


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
    return class_names[predicted_idx], confidence, output


def send_to_esp(preset_idx, max_read_lines=20, read_timeout=2.0):
    cmd = f"{preset_idx}\n"
    ser.write(cmd.encode())
    time.sleep(0.3)

    start = time.time()
    lines_read = 0
    while ser.in_waiting > 0:
        response = ser.readline()
        print("ESP:", response.decode(errors='ignore').strip())
        lines_read += 1
        # Jaring pengaman: kalau ESP kirim data terus-menerus, jangan
        # sampai loop ini menggantung selamanya.
        if lines_read >= max_read_lines or (time.time() - start) > read_timeout:
            print("    [WARNING] Berhenti baca respons ESP (batas lines/timeout tercapai).")
            break


# ===== Parameter dasar frame diff =====
DIFF_THRESHOLD = 50
CHANGE_AREA_THRESHOLD = 8000
MIN_CONTOUR_AREA = 10000
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0

# --- Mode normal (fallback) ---
STABLE_FRAMES_NEEDED_NORMAL = 10
MOTION_TOLERANCE_NORMAL = 500

# --- Mode immediate: kontur sudah dominan di frame -> ambil cepat -----
# TIDAK mengecek motion sama sekali, supaya goyang kecil sampah yang
# masih "kerak" di piringan tidak bikin nunggu lama. Cukup pastikan
# konturnya konsisten dominan selama beberapa frame berturut-turut.
IMMEDIATE_CAPTURE_AREA_RATIO = 0.20   # kontur >= 20% luas frame
IMMEDIATE_CONFIRM_FRAMES = 3          # 3 frame konsisten, tanpa cek motion

FORCE_REFRESH_TIMEOUT = 10.0
REFRESH_COOLDOWN = 30.0

# Delay setelah kirim preset & setelah kirim 0 (posisi netral)
POST_PRESET_DELAY = 2.0
POST_NEUTRAL_DELAY = 2.0

DEBUG_DIR = "calibration_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

REFRESH_FLAG_FILE = "refresh_now.flag"

cap = cv2.VideoCapture(0)

print("Ambil frame referensi dalam 3 detik, pastikan area kosong...")
time.sleep(3)
ret, reference = cap.read()
reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
reference_gray = cv2.GaussianBlur(reference_gray, (25, 25), 0)

frame_area = reference_gray.shape[0] * reference_gray.shape[1]
IMMEDIATE_CAPTURE_AREA_THRESHOLD = frame_area * IMMEDIATE_CAPTURE_AREA_RATIO

prev_gray = reference_gray.copy()
normal_stable_count = 0
immediate_confirm_count = 0
object_present = False
last_activity_time = time.time()
last_refresh_time = time.time()

detection_start_time = None

print("Sistem siap. Monitoring piringan...")
print(f"(Buat force-refresh manual dari SSH: touch {REFRESH_FLAG_FILE})\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal capture frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)

    diff_ref = cv2.absdiff(reference_gray, gray)
    thresh_ref = cv2.threshold(diff_ref, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh_ref = cv2.erode(thresh_ref, kernel, iterations=1)
    thresh_ref = cv2.dilate(thresh_ref, kernel, iterations=2)
    change_area = cv2.countNonZero(thresh_ref)

    diff_prev = cv2.absdiff(prev_gray, gray)
    thresh_prev = cv2.threshold(diff_prev, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    motion_area = cv2.countNonZero(thresh_prev)

    contour_area = 0
    aspect_ratio = 0
    shape_valid = False
    bbox = None
    confidence_mode = "normal"
    triggered = False

    if change_area > CHANGE_AREA_THRESHOLD:
        if detection_start_time is None:
            detection_start_time = time.perf_counter()

        contours, _ = cv2.findContours(thresh_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            bbox = (x, y, w, h)

            if contour_area >= MIN_CONTOUR_AREA and MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
                shape_valid = True

            if shape_valid:
                is_immediate_candidate = contour_area >= IMMEDIATE_CAPTURE_AREA_THRESHOLD

                if is_immediate_candidate:
                    immediate_confirm_count += 1
                else:
                    immediate_confirm_count = 0

                if immediate_confirm_count >= IMMEDIATE_CONFIRM_FRAMES:
                    triggered = True
                    confidence_mode = "immediate"
                else:
                    if motion_area < MOTION_TOLERANCE_NORMAL:
                        normal_stable_count += 1
                    else:
                        normal_stable_count = 0

                    if normal_stable_count >= STABLE_FRAMES_NEEDED_NORMAL:
                        triggered = True
                        confidence_mode = "normal"
            else:
                immediate_confirm_count = 0
                normal_stable_count = 0

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
                cropped_object = frame[y:y+h, x:x+w]
                t_crop_end = time.perf_counter()
                crop_duration = t_crop_end - t_crop_start

                t_infer_start = time.perf_counter()
                label, confidence, all_scores = classify(cropped_object)
                t_infer_end = time.perf_counter()
                infer_duration = t_infer_end - t_infer_start

                total_duration = t_infer_end - detection_start_time

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if confidence >= CONFIDENCE_THRESHOLD:
                    save_path = os.path.join(CAPTURE_DIR, label, f"{timestamp}.jpg")
                else:
                    save_path = os.path.join(CAPTURE_DIR, "unknown", f"{timestamp}_{label}_{confidence:.2f}.jpg")
                cv2.imwrite(save_path, cropped_object)

                print(f"\n>>> STABIL & VALID [{confidence_mode}] -> disimpan {save_path}")
                print(f"    Prediksi: {label} ({confidence*100:.2f}%)")
                for i, cname in enumerate(class_names):
                    print(f"      {cname}: {all_scores[i]*100:.2f}%")
                print(f"    --- Timing ---")
                print(f"    Deteksi -> stabil : {detection_duration*1000:.1f} ms")
                print(f"    Crop              : {crop_duration*1000:.2f} ms")
                print(f"    Inference model   : {infer_duration*1000:.1f} ms")
                print(f"    TOTAL (deteksi->hasil): {total_duration*1000:.1f} ms")

                # ===== Kirim ke ESP kalau confidence cukup & termasuk kategori yang disortir =====
                if confidence >= CONFIDENCE_THRESHOLD and label in label_to_preset:
                    preset = label_to_preset[label]
                    print(f"    -> Kirim preset {preset} ke ESP\n")
                    send_to_esp(preset)

                    time.sleep(POST_PRESET_DELAY)
                    send_to_esp(0)
                    time.sleep(POST_NEUTRAL_DELAY)

                elif label == 'background':
                    print(f"    -> Terdeteksi background, tidak ada aksi\n")
                else:
                    print(f"    -> Confidence rendah / kelas tidak disortir, TIDAK kirim ke ESP\n")

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

    # ===== Refresh: otomatis (timeout) atau manual (file trigger) =====
    time_since_activity = time.time() - last_activity_time
    time_since_last_refresh = time.time() - last_refresh_time

    should_force_refresh = (
        not object_present and
        time_since_activity >= FORCE_REFRESH_TIMEOUT and
        time_since_last_refresh >= REFRESH_COOLDOWN
    )

    manual_refresh_requested = os.path.exists(REFRESH_FLAG_FILE)

    if should_force_refresh or manual_refresh_requested:
        reference_gray = gray.copy()
        last_refresh_time = time.time()
        last_activity_time = time.time()

        if manual_refresh_requested:
            reason = "manual (file trigger)"
            os.remove(REFRESH_FLAG_FILE)
        else:
            reason = f"otomatis (tidak ada aktivitas {time_since_activity:.1f}s)"

        print(f">>> [REFRESH] Referensi diperbarui — {reason}\n")

    prev_gray = gray.copy()
    time.sleep(0.1)