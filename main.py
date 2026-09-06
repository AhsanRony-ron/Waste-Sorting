import cv2
import numpy as np
import time
import os
import serial
import csv
import json
import glob
import uuid
from datetime import datetime
from ai_edge_litert.interpreter import Interpreter

LOG_FILE = "hasil_pengujian.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "label_prediksi", "confidence",
            "detection_ms", "crop_ms", "inference_ms", "total_ms",
            "label_sebenarnya"  # kolom ini diisi MANUAL setelah pengujian, cocokkan dengan urutan sampah yang ditaruh
        ])

# ===== Setup serial ke ESP =====
SERIAL_PORT = '/dev/ttyUSB0'

_tmp = serial.Serial(SERIAL_PORT, 115200, timeout=1)
time.sleep(0.3)
_tmp.close()
time.sleep(0.5)

ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
time.sleep(2)
ser.reset_input_buffer()
ser.reset_output_buffer()

# ===== Setup model =====
interpreter = Interpreter(model_path="waste_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = ['background', 'daun', 'kaleng', 'kertas', 'plastik']

label_to_preset = {
    'kertas': 1,
    'plastik': 2,
    'kaleng': 3,
    'daun': 4,
}

CONFIDENCE_THRESHOLD = 0.6

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


# ===================== Sinkronisasi Telegram (file-based queue) =====================

EVENTS_DIR = "telegram_sync/events"
COMMANDS_DIR = "telegram_sync/commands"
COMMANDS_DONE_DIR = "telegram_sync/commands_done"
DEBUG_CAPTURE_DIR = "debug_captures"

os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(COMMANDS_DIR, exist_ok=True)
os.makedirs(COMMANDS_DONE_DIR, exist_ok=True)
os.makedirs(DEBUG_CAPTURE_DIR, exist_ok=True)


def write_event(event_type, data):
    fname = f"{time.time_ns()}.json"
    tmp_path = os.path.join(EVENTS_DIR, f".tmp_{fname}")
    final_path = os.path.join(EVENTS_DIR, fname)
    with open(tmp_path, 'w') as f:
        json.dump({"type": event_type, "data": data}, f)
    os.rename(tmp_path, final_path)


paused = False

PING_INTERVAL = 1.0
last_ping_sent = 0


def send_ping():
    global last_ping_sent
    if time.time() - last_ping_sent >= PING_INTERVAL:
        ser.write(b"PING\n")
        last_ping_sent = time.time()


def handle_camera_check(cmd, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(DEBUG_CAPTURE_DIR, f"{timestamp}.jpg")
    cv2.imwrite(path, frame)  # frame mentah, TANPA crop/resize
    write_event("command_result", {
        "command_id": cmd["id"], "command": "camera_check", "chat_id": cmd["chat_id"],
        "success": True, "message": "Posisi kamera saat ini", "image_path": path
    })


def handle_pause(cmd):
    global paused
    paused = True
    write_event("command_result", {
        "command_id": cmd["id"], "command": "pause", "chat_id": cmd["chat_id"],
        "success": True, "message": "Sistem dijeda"
    })


def handle_resume(cmd):
    global paused
    paused = False
    write_event("command_result", {
        "command_id": cmd["id"], "command": "resume", "chat_id": cmd["chat_id"],
        "success": True, "message": "Sistem dilanjutkan"
    })


def handle_manual_preset(cmd):
    preset = cmd["params"].get("preset")
    if preset is None or not (0 <= preset <= 5):
        write_event("command_result", {
            "command_id": cmd["id"], "command": "manual_preset", "chat_id": cmd["chat_id"],
            "success": False, "message": "Preset tidak valid (0-5)"
        })
        return
    send_to_esp(preset)
    write_event("command_result", {
        "command_id": cmd["id"], "command": "manual_preset", "chat_id": cmd["chat_id"],
        "success": True, "message": f"Preset {preset} terkirim ke ESP"
    })


def handle_refresh_reference(cmd):
    open(REFRESH_FLAG_FILE, 'w').close()
    write_event("command_result", {
        "command_id": cmd["id"], "command": "refresh_reference", "chat_id": cmd["chat_id"],
        "success": True, "message": "Refresh referensi dijadwalkan"
    })


def handle_status(cmd):
    write_event("command_result", {
        "command_id": cmd["id"], "command": "status", "chat_id": cmd["chat_id"], "success": True,
        "message": (
            f"Paused: {paused}\n"
            f"Object present: {object_present}\n"
            f"Last preset: {last_preset_sent}"
        )
    })


COMMAND_HANDLERS = {
    "camera_check": handle_camera_check,  # butuh frame, ditangani khusus di poll_commands
    "pause": handle_pause,
    "resume": handle_resume,
    "manual_preset": handle_manual_preset,
    "refresh_reference": handle_refresh_reference,
    "status": handle_status,
}


def poll_commands(frame):
    for path in sorted(glob.glob(os.path.join(COMMANDS_DIR, "*.json"))):
        try:
            with open(path) as f:
                cmd = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        handler = COMMAND_HANDLERS.get(cmd["command"])
        if handler is None:
            write_event("command_result", {
                "command_id": cmd["id"], "command": cmd["command"], "chat_id": cmd.get("chat_id"),
                "success": False, "message": "Command tidak dikenal"
            })
        elif cmd["command"] == "camera_check":
            handler(cmd, frame)
        else:
            handler(cmd)

        os.rename(path, os.path.join(COMMANDS_DONE_DIR, os.path.basename(path)))


# ===== Parameter dasar frame diff =====
DIFF_THRESHOLD = 20
CHANGE_AREA_THRESHOLD = 8000
MIN_CONTOUR_AREA = 10000
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0

STABLE_FRAMES_NEEDED_NORMAL = 10
MOTION_TOLERANCE_NORMAL = 100  # dinaikkan, biar goyangan wajar plastik tidak reset terus

IMMEDIATE_CAPTURE_AREA_RATIO = 0.10  # diturunkan, biar kontur sedang pun bisa immediate
IMMEDIATE_CONFIRM_FRAMES = 20

FORCE_REFRESH_TIMEOUT = 20.0
REFRESH_COOLDOWN = 30.0

POST_PRESET_DELAY = 2.0
POST_NEUTRAL_DELAY = 1.5

RECLASSIFY_INTERVAL = 2.0       # seberapa sering cek ulang objek yang lagi di piringan
STUCK_RESEND_COOLDOWN = 5.0     # jarak minimal antar kirim ulang preset YANG SAMA (biar ga spam ESP)

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

last_preset_sent = None
last_stuck_retry_time = 0.0

current_bbox = None
next_reclassify_time = None

print("Sistem siap. Monitoring piringan...")
print(f"(Buat force-refresh manual dari SSH: touch {REFRESH_FLAG_FILE})\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal capture frame")
        continue

    send_ping()
    poll_commands(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)

    if paused:
        prev_gray = gray.copy()  # tetap update biar gak ada lonjakan diff pas resume
        continue

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
            current_bbox = bbox

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

                write_event("sort_result", {
                    "timestamp": timestamp,
                    "label": label,
                    "confidence": float(confidence),
                    "image_path": save_path,
                    "detection_ms": round(detection_duration * 1000, 2),
                    "inference_ms": round(infer_duration * 1000, 2),
                    "total_ms": round(total_duration * 1000, 2),
                })

                print(f"\n>>> STABIL & VALID [{confidence_mode}] -> disimpan {save_path}")
                print(f"    Prediksi: {label} ({confidence*100:.2f}%)")
                for i, cname in enumerate(class_names):
                    print(f"      {cname}: {all_scores[i]*100:.2f}%")
                print(f"    --- Timing ---")
                print(f"    Deteksi -> stabil : {detection_duration*1000:.1f} ms")
                print(f"    Crop              : {crop_duration*1000:.2f} ms")
                print(f"    Inference model   : {infer_duration*1000:.1f} ms")
                print(f"    TOTAL (deteksi->hasil): {total_duration*1000:.1f} ms")

                with open(LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp, label, f"{confidence:.4f}",
                        f"{detection_duration*1000:.2f}", f"{crop_duration*1000:.2f}",
                        f"{infer_duration*1000:.2f}", f"{total_duration*1000:.2f}",
                        ""  # kosongkan dulu, isi manual belakangan pas review hasil
                    ])

                if confidence >= CONFIDENCE_THRESHOLD and label in label_to_preset:
                    preset = label_to_preset[label]
                    print(f"    -> Kirim preset {preset} ke ESP\n")
                    send_to_esp(preset)
                    last_preset_sent = preset
                    time.sleep(POST_PRESET_DELAY)
                    send_to_esp(0)
                    time.sleep(POST_NEUTRAL_DELAY)

                elif label == 'background':
                    # background terkonfirmasi saat siklus normal, ESP diam,
                    # langsung pakai frame ini sebagai referensi baru tanpa tunggu timeout
                    print(f"    -> Terdeteksi background, tidak ada aksi ke ESP")
                    print(f"    -> Langsung update referensi dari frame ini (background terkonfirmasi)\n")
                    reference_gray = gray.copy()
                    last_refresh_time = time.time()
                    last_preset_sent = None
                else:
                    print(f"    -> Confidence rendah / kelas tidak disortir, TIDAK kirim ke ESP\n")
                    last_preset_sent = None

                object_present = True
                last_activity_time = time.time()
                next_reclassify_time = time.time() + RECLASSIFY_INTERVAL
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
        last_preset_sent = None
        current_bbox = None
        next_reclassify_time = None

    # ===== Refresh: otomatis (timeout) atau manual (file trigger, termasuk dari Telegram) =====
    time_since_activity = time.time() - last_activity_time
    time_since_last_refresh = time.time() - last_refresh_time

    should_force_refresh = (
        not object_present and
        time_since_activity >= FORCE_REFRESH_TIMEOUT and
        time_since_last_refresh >= REFRESH_COOLDOWN
    )

    manual_refresh_requested = os.path.exists(REFRESH_FLAG_FILE)

    if should_force_refresh or manual_refresh_requested:
        if manual_refresh_requested:
            os.remove(REFRESH_FLAG_FILE)

        # sebelum commit sebagai referensi, verifikasi dulu ini benar background
        check_label, check_confidence, check_scores = classify(frame)

        if check_label == 'background' and check_confidence >= CONFIDENCE_THRESHOLD:
            reference_gray = gray.copy()
            last_refresh_time = time.time()
            last_activity_time = time.time()
            reason = "manual (file trigger)" if manual_refresh_requested else f"otomatis (tidak ada aktivitas {time_since_activity:.1f}s)"
            print(f">>> [REFRESH] Terverifikasi background ({check_confidence*100:.1f}%) — referensi diperbarui, {reason}\n")
        else:
            # Ada sesuatu yang nyangkut di piringan, bukan background beneran
            print(f">>> [REFRESH DITUNDA] Frame terdeteksi sebagai '{check_label}' ({check_confidence*100:.1f}%), bukan background.")
            print(f"    Kemungkinan ada barang nyangkut di piringan.\n")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stuck_path = os.path.join(CAPTURE_DIR, "unknown", f"stuck_{timestamp}_{check_label}_{check_confidence:.2f}.jpg")
            cv2.imwrite(stuck_path, frame)

            if check_confidence >= CONFIDENCE_THRESHOLD and check_label in label_to_preset:
                preset = label_to_preset[check_label]
                print(f"    -> Kirim preset {preset} ke ESP buat bersihkan barang nyangkut\n")
                send_to_esp(preset)
                time.sleep(POST_PRESET_DELAY)
                send_to_esp(0)
                time.sleep(POST_NEUTRAL_DELAY)

            # Referensi TIDAK di-refresh sekarang, coba lagi di siklus refresh berikutnya
            last_refresh_time = time.time()  # tetap update supaya tidak spam retry tiap frame
            last_activity_time = time.time()

    # ===== Reclassify berkala selama objek masih dianggap ada (bedakan stuck vs objek baru) =====
    if object_present and next_reclassify_time is not None and time.time() >= next_reclassify_time:
        if current_bbox is not None:
            rx, ry, rw, rh = current_bbox
            recheck_crop = frame[ry:ry+rh, rx:rx+rw]
        else:
            recheck_crop = frame  # fallback kalau bbox somehow kosong

        rc_label, rc_conf, _ = classify(recheck_crop)

        if rc_label == 'background' and rc_conf >= CONFIDENCE_THRESHOLD:
            print(">>> [RECLASSIFY] Piringan terkonfirmasi kosong, siap terima objek baru.\n")
            object_present = False
            last_preset_sent = None
            current_bbox = None
            next_reclassify_time = None
            normal_stable_count = 0
            immediate_confirm_count = 0
            detection_start_time = None

        elif rc_conf >= CONFIDENCE_THRESHOLD and rc_label in label_to_preset:
            rc_preset = label_to_preset[rc_label]

            if rc_preset == last_preset_sent:
                # objek sama, masih nyangkut -> retry, tapi dibatasi cooldown biar ga spam ESP
                if time.time() - last_stuck_retry_time >= STUCK_RESEND_COOLDOWN:
                    print(f">>> [RECLASSIFY] Objek sama ({rc_label}) masih nyangkut, retry preset {rc_preset}\n")
                    send_to_esp(rc_preset)
                    time.sleep(POST_PRESET_DELAY)
                    send_to_esp(0)
                    time.sleep(POST_NEUTRAL_DELAY)
                    last_stuck_retry_time = time.time()
                    last_activity_time = time.time()
            else:
                # label beda dari preset terakhir -> ini objek BARU, proses seperti deteksi baru
                print(f">>> [RECLASSIFY] Objek baru terdeteksi: {rc_label} ({rc_conf*100:.2f}%)\n")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(CAPTURE_DIR, rc_label, f"{timestamp}.jpg")
                cv2.imwrite(save_path, recheck_crop)
                with open(LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, rc_label, f"{rc_conf:.4f}", "", "", "", "", ""])

                send_to_esp(rc_preset)
                time.sleep(POST_PRESET_DELAY)
                send_to_esp(0)
                time.sleep(POST_NEUTRAL_DELAY)
                last_preset_sent = rc_preset
                last_activity_time = time.time()
        # else: confidence rendah/ambigu, biarkan, coba lagi di siklus reclassify berikutnya

        next_reclassify_time = time.time() + RECLASSIFY_INTERVAL

    prev_gray = gray.copy()