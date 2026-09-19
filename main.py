import cv2
import numpy as np
import time
import os
import serial
import csv
import json
import glob
import yaml
from collections import deque
from datetime import datetime
from ai_edge_litert.interpreter import Interpreter

# =============================================================
# KONFIGURASI (dibaca dari config.yaml, auto-reload tanpa restart)
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
    Cek mtime config.yaml tiap dipanggil. Kalau berubah, reload isinya dan
    terapkan ulang parameter kamera (yang butuh cap.set(), tidak otomatis
    kebawa cuma dari baca dict). Parameter software lain otomatis kepakai
    nilai baru karena selalu dibaca langsung dari CONFIG saat dipakai.
    """
    global _config_mtime
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return False

    if mtime != _config_mtime:
        load_config()
        apply_camera_config()
        print(">>> [CONFIG] config.yaml berubah, direload & parameter kamera diterapkan ulang.\n")
        return True
    return False


load_config()

LOG_FILE = CONFIG["paths"]["log_file"]

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "label_prediksi", "confidence",
            "detection_ms", "crop_ms", "inference_ms", "total_ms",
            "label_sebenarnya"  # kolom ini diisi MANUAL setelah pengujian, cocokkan dengan urutan sampah yang ditaruh
        ])


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
    # Pakai cv2.mean() -- dihitung native/optimized, jauh lebih cepat
    # daripada convert seluruh frame ke float32 cuma buat cari rata-rata.
    b_avg, g_avg, r_avg, _ = cv2.mean(frame)
    gray_avg = (b_avg + g_avg + r_avg) / 3.0

    gain_b = np.clip(gray_avg / max(b_avg, 1e-6), gain_min, gain_max)
    gain_g = np.clip(gray_avg / max(g_avg, 1e-6), gain_min, gain_max)
    gain_r = np.clip(gray_avg / max(r_avg, 1e-6), gain_min, gain_max)

    # cv2.convertScaleAbs beroperasi langsung di uint8 dengan saturasi
    # otomatis (setara clip 0-255), tanpa perlu convert ke float32 dulu.
    b, g, r = cv2.split(frame)
    b = cv2.convertScaleAbs(b, alpha=gain_b)
    g = cv2.convertScaleAbs(g, alpha=gain_g)
    r = cv2.convertScaleAbs(r, alpha=gain_r)

    return cv2.merge([b, g, r])

def preprocess_frame(raw_frame):
    """Crop + koreksi warna, parameter dibaca live dari CONFIG tiap kali dipanggil."""
    pp = CONFIG["preprocessing"]
    f = crop_center(raw_frame, pp["crop_width"], pp["crop_height"],
                     pp["crop_offset_x"], pp["crop_offset_y"])
    if pp["enable_color_correction"]:
        f = gray_world_correction(f, pp["color_gain_min"], pp["color_gain_max"])
    return f


# ===== Setup serial ke ESP (port & baudrate cuma dipakai sekali saat start) =====
SERIAL_PORT = CONFIG["esp"]["serial_port"]
BAUD_RATE = CONFIG["esp"]["baud_rate"]

_tmp = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(0.3)
_tmp.close()
time.sleep(0.5)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
ser.reset_input_buffer()
ser.reset_output_buffer()

# ===== Setup model (path & class_names cuma dipakai sekali saat load) =====
interpreter = Interpreter(model_path=CONFIG["model"]["path"])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = CONFIG["model"]["class_names"]

CAPTURE_DIR = CONFIG["paths"]["capture_dir"]
for cname in class_names:
    os.makedirs(os.path.join(CAPTURE_DIR, cname), exist_ok=True)
os.makedirs(os.path.join(CAPTURE_DIR, "unknown"), exist_ok=True)


def classify(cropped_bgr):
    img = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    mode = CONFIG["model"]["preprocessing"]
    if mode == "scale_255":
        img_array = np.array(img, dtype=np.float32) / 255.0
    elif mode == "raw":
        img_array = np.array(img, dtype=np.float32)  
    else:
        raise ValueError(f"preprocessing mode tidak dikenal: {mode}")

    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_idx = np.argmax(output)
    confidence = output[predicted_idx]
    return class_names[predicted_idx], confidence, output


def send_to_esp(preset_idx):
    cmd = f"{preset_idx}\n"
    ser.write(cmd.encode())
    time.sleep(0.3)

def send_bin_full_alert(label):
    # 1 jenis notif serial -- ESP yang urus buzzer & tampilan LCD-nya (nyusul)
    # TODO: format persis disepakati bareng main.cpp, sementara: "FULL:<label>\n"
    ser.write(f"FULL:{label}\n".encode())


# ===================== Sinkronisasi Telegram (file-based queue) =====================

EVENTS_DIR = CONFIG["paths"]["events_dir"]
COMMANDS_DIR = CONFIG["paths"]["commands_dir"]
COMMANDS_DONE_DIR = CONFIG["paths"]["commands_done_dir"]
DEBUG_CAPTURE_DIR = CONFIG["paths"]["debug_capture_dir"]

os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(COMMANDS_DIR, exist_ok=True)
os.makedirs(COMMANDS_DONE_DIR, exist_ok=True)
os.makedirs(DEBUG_CAPTURE_DIR, exist_ok=True)

def compute_bin_capacity_percent(label, distance_cm):
    bins_cfg = CONFIG["bins"]
    empty_d = bins_cfg["empty_distance_cm"][label]
    full_d = bins_cfg["full_distance_cm"][label]
    if empty_d == full_d:
        return 0.0
    pct = (empty_d - distance_cm) / (empty_d - full_d) * 100.0
    return float(np.clip(pct, 0, 100))


def _resolve_bin_label(key):
    # ESP sekarang kirim nama label langsung (Plastik/Kaleng/Kertas/Daun),
    # jadi cukup lowercase & cocokkan ke key kalibrasi di config.
    key_lower = key.strip().lower()
    if key_lower in CONFIG["bins"]["empty_distance_cm"]:
        return key_lower
    return None

def is_bin_full(label):
    pct = bin_capacity_percent.get(label)
    if pct is None:
        return False  # TODO: putuskan default kalau data kapasitas belum pernah masuk
    return pct >= CONFIG["bins"]["full_threshold_percent"]

def read_esp_sensor_data():
    """
    Baca semua baris serial yang sudah masuk dari ESP (non-blocking --
    cuma proses kalau ser.in_waiting > 0). Dipanggil tiap iterasi loop
    utama, jadi update kapasitas gak tergantung siapa yang memicu.
    """
    while ser.in_waiting > 0:
        try:
            raw = ser.readline()
        except Exception:
            break

        line = raw.decode(errors="ignore").strip()
        if not line or ":" not in line:
            continue  # bukan baris sensor, mungkin log lain dari ESP

        for part in line.replace(",", " ").split():
            key, sep, val = part.partition(":")
            if not sep:
                continue

            label = _resolve_bin_label(key)
            if label is None:
                continue
            try:
                distance = float(val.strip())
            except ValueError:
                continue

            bin_distance_cm[label] = distance
            bin_capacity_percent[label] = compute_bin_capacity_percent(label, distance)


def request_bin_capacity():
    """
    Kirim command "c" ke ESP -- dipanggil saat ada command pengecekan
    (misal /status), BUKAN polling periodik, karena ESP sudah kirim data
    otomatis tiap ULTRASONIC_READ_INTERVAL_MS. check_interval di config
    cuma jadi jeda minimum kalau beberapa command mepet-mepetan.
    Non-blocking -- balasannya diproses read_esp_sensor_data() belakangan.
    """
    global last_capacity_request_time
    interval = CONFIG["bins"]["check_interval"]
    if time.time() - last_capacity_request_time >= interval:
        ser.write(b"c\n")
        last_capacity_request_time = time.time()

def notify_bin_full(label, pct):
    # TODO: masih kirim tiap kejadian dulu -- logika "sekali aja" nyusul nanti
    write_event("bin_full_alert", {
        "label": label,
        "percent": pct,
    })
    
def write_event(event_type, data):
    fname = f"{time.time_ns()}.json"
    tmp_path = os.path.join(EVENTS_DIR, f".tmp_{fname}")
    final_path = os.path.join(EVENTS_DIR, fname)
    with open(tmp_path, 'w') as f:
        json.dump({"type": event_type, "data": data}, f)
    os.rename(tmp_path, final_path)


paused = False
last_ping_sent = 0

bin_capacity_percent = {}   # {label: persentase penuh (0-100)}
bin_distance_cm = {}        # {label: jarak mentah terakhir (cm), buat debug
last_capacity_request_time = 0.0

def send_ping():
    global last_ping_sent
    ping_interval = CONFIG["esp"]["ping_interval"]
    if time.time() - last_ping_sent >= ping_interval:
        ser.write(b"PING\n")
        last_ping_sent = time.time()


def handle_camera_check(cmd, frame):
    # frame yang sudah di-preprocess (crop + koreksi warna), sama persis
    # dengan yang dipakai untuk klasifikasi -- biar bisa cek framing &
    # hasil koreksi warna, bukan cuma posisi kamera mentah.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(DEBUG_CAPTURE_DIR, f"{timestamp}.jpg")
    cv2.imwrite(path, frame)
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
    open(CONFIG["paths"]["refresh_flag_file"], 'w').close()
    write_event("command_result", {
        "command_id": cmd["id"], "command": "refresh_reference", "chat_id": cmd["chat_id"],
        "success": True, "message": "Refresh referensi dijadwalkan"
    })

# ===== FPS kamera & suhu prosesor (buat /status) =====

_frame_times = deque(maxlen=30)  # rolling window 30 frame terakhir
current_fps = 0.0


def update_fps():
    global current_fps
    now = time.time()
    _frame_times.append(now)
    if len(_frame_times) >= 2:
        elapsed = _frame_times[-1] - _frame_times[0]
        if elapsed > 0:
            current_fps = (len(_frame_times) - 1) / elapsed


def get_cpu_temp():
    """
    Baca suhu CPU dari sysfs. File berisi suhu dalam milli-Celsius,
    dibagi 1000 buat dapet Celsius biasa. Return None kalau file
    gak ada (misal bukan Raspberry Pi / board ARM Linux).
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None

def handle_status(cmd):
    request_bin_capacity() 

    cpu_temp = get_cpu_temp()
    cpu_temp_str = f"{cpu_temp:.1f}°C" if cpu_temp is not None else "N/A (bukan Raspberry Pi?)"

    # <-- tambahan
    if bin_capacity_percent:
        capacity_str = "\n".join(
            f"  {label}: {pct:.0f}%" for label, pct in bin_capacity_percent.items()
        )
    else:
        capacity_str = "  (belum ada data sensor dari ESP)"

    write_event("command_result", {
        "command_id": cmd["id"], "command": "status", "chat_id": cmd["chat_id"], "success": True,
        "message": (
            f"Paused: {paused}\n"
            f"Object present: {object_present}\n"
            f"Last preset: {last_preset_sent}\n"
            f"FPS kamera: {current_fps:.1f}\n"
            f"Suhu prosesor: {cpu_temp_str}\n"
            f"Kapasitas tempat sampah:\n{capacity_str}"
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


DEBUG_DIR = CONFIG["paths"]["debug_dir"]
os.makedirs(DEBUG_DIR, exist_ok=True)


def set_and_verify(prop, value, name):
    cap.set(prop, value)
    print(f"{name}: minta {value}, aktual -> {cap.get(prop)}")


def apply_camera_config():
    """
    Terapkan semua parameter kamera fisik dari CONFIG['camera'] ke device.
    Dipanggil saat startup DAN tiap kali config.yaml ke-reload, karena
    kamera fisik nyimpen state-nya sendiri lewat cap.set() -- gak otomatis
    kebawa cuma dari baca dict CONFIG.
    """
    cam = CONFIG["camera"]
    # FOURCC harus MJPG (bukan parameter yang perlu diubah-ubah) -- banyak
    # webcam cuma dukung resolusi tinggi dalam format ini, YUYV kebanyakan
    # bandwidth USB dan diam-diam di-fallback ke resolusi rendah.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    set_and_verify(cv2.CAP_PROP_FRAME_WIDTH, cam["frame_width"], "frame_width")
    set_and_verify(cv2.CAP_PROP_FRAME_HEIGHT, cam["frame_height"], "frame_height")
    set_and_verify(cv2.CAP_PROP_FPS, cam["target_fps"], "target_fps")
    set_and_verify(cv2.CAP_PROP_AUTO_EXPOSURE, cam["auto_exposure"], "auto_exposure")
    set_and_verify(cv2.CAP_PROP_EXPOSURE, cam["exposure_time_absolute"], "exposure_time_absolute")
    set_and_verify(cv2.CAP_PROP_AUTO_WB, cam["auto_wb"], "white_balance_automatic")
    set_and_verify(cv2.CAP_PROP_WB_TEMPERATURE, cam["wb_temperature"], "white_balance_temperature")
    set_and_verify(cv2.CAP_PROP_BRIGHTNESS, cam["brightness"], "brightness")
    set_and_verify(cv2.CAP_PROP_CONTRAST, cam["contrast"], "contrast")
    set_and_verify(cv2.CAP_PROP_SATURATION, cam["saturation"], "saturation")
    set_and_verify(cv2.CAP_PROP_GAIN, cam["gain"], "gain")


cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # paksa backend V4L2 biar mapping property konsisten
apply_camera_config()

print("Ambil frame referensi dalam 3 detik, pastikan area kosong...")
time.sleep(3)
ret, reference = cap.read()
print("Resolusi asli dari kamera:", reference.shape)  # (height, width, channels)
reference = preprocess_frame(reference)
print("Resolusi setelah crop:", reference.shape)

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

last_preset_sent = None
last_stuck_retry_time = 0.0

current_bbox = None
next_reclassify_time = None

print("Sistem siap. Monitoring piringan...")
print(f"(Buat force-refresh manual dari SSH: touch {CONFIG['paths']['refresh_flag_file']})")
print(f"(Ubah config.yaml kapan saja -- otomatis di-reload, tidak perlu restart)\n")

while True:
    reload_config_if_changed()

    ret, raw_frame = cap.read()
    if not ret:
        print("Gagal capture frame")
        continue

    update_fps()

    frame = preprocess_frame(raw_frame)

    send_ping()
    poll_commands(frame)  # camera_check pakai frame yang sudah di-crop & dikoreksi warnanya
    read_esp_sensor_data()

    det = CONFIG["detection"]
    blur_k = CONFIG["preprocessing"]["gaussian_blur_kernel"]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    if paused:
        prev_gray = gray.copy()  # tetap update biar gak ada lonjakan diff pas resume
        continue

    diff_ref = cv2.absdiff(reference_gray, gray)
    thresh_ref = cv2.threshold(diff_ref, det["diff_threshold"], 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh_ref = cv2.erode(thresh_ref, kernel, iterations=1)
    thresh_ref = cv2.dilate(thresh_ref, kernel, iterations=2)
    change_area = cv2.countNonZero(thresh_ref)

    diff_prev = cv2.absdiff(prev_gray, gray)
    thresh_prev = cv2.threshold(diff_prev, det["diff_threshold"], 255, cv2.THRESH_BINARY)[1]
    motion_area = cv2.countNonZero(thresh_prev)

    contour_area = 0
    aspect_ratio = 0
    shape_valid = False
    bbox = None
    confidence_mode = "normal"
    triggered = False

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
            current_bbox = bbox

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

                confidence_threshold = CONFIG["model"]["confidence_threshold"]
                label_to_preset = CONFIG["model"]["label_to_preset"]

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if confidence >= confidence_threshold:
                    save_path = os.path.join(CAPTURE_DIR, label, f"{timestamp}.jpg")
                else:
                    save_path = os.path.join(CAPTURE_DIR, "unknown", f"{timestamp}_{label}_{confidence:.2f}.jpg")
                cv2.imwrite(save_path, cropped_object)

                if label != 'background':
                    write_event("sort_result", {
                        "timestamp": timestamp,
                        "label": label,
                        "confidence": float(confidence),
                        "image_path": save_path,
                        "detection_ms": round(detection_duration * 1000, 2),
                        "inference_ms": round(infer_duration * 1000, 2),
                        "total_ms": round(total_duration * 1000, 2),
                        "all_scores": {cname: float(all_scores[i]) for i, cname in enumerate(class_names)},
                        "bin_capacity_percent": bin_capacity_percent.get(label),
                        "bin_capacities": dict(bin_capacity_percent),
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
                        ""
                    ])

                if confidence >= confidence_threshold and label in label_to_preset:
                    preset = label_to_preset[label]

                    if is_bin_full(label):
                        pct = bin_capacity_percent.get(label)
                        print(f"    -> Bin '{label}' PENUH ({pct:.0f}%), servo TIDAK digerakkan\n")
                        send_bin_full_alert(label)
                        notify_bin_full(label, pct)
                        last_preset_sent = None
                        # TODO (iterasi berikutnya): state biar reclassify gak nganggep
                        # ini "objek baru" tiap siklus & gak nulis ulang CSV/foto terus
                    else:
                        print(f"    -> Kirim preset {preset} ke ESP\n")
                        send_to_esp(preset)
                        last_preset_sent = preset
                        time.sleep(CONFIG["esp"]["post_preset_delay"])
                        send_to_esp(0)
                        time.sleep(CONFIG["esp"]["post_neutral_delay"])

                elif label == 'background':
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
                next_reclassify_time = time.time() + CONFIG["reclassify"]["interval"]
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
    refresh_cfg = CONFIG["refresh"]
    refresh_flag_file = CONFIG["paths"]["refresh_flag_file"]

    time_since_activity = time.time() - last_activity_time
    time_since_last_refresh = time.time() - last_refresh_time

    should_force_refresh = (
        not object_present and
        time_since_activity >= refresh_cfg["force_refresh_timeout"] and
        time_since_last_refresh >= refresh_cfg["refresh_cooldown"]
    )

    manual_refresh_requested = os.path.exists(refresh_flag_file)

    if should_force_refresh or manual_refresh_requested:
        if manual_refresh_requested:
            os.remove(refresh_flag_file)

        confidence_threshold = CONFIG["model"]["confidence_threshold"]
        label_to_preset = CONFIG["model"]["label_to_preset"]

        check_label, check_confidence, check_scores = classify(frame)

        if check_label == 'background' and check_confidence >= confidence_threshold:
            reference_gray = gray.copy()
            last_refresh_time = time.time()
            last_activity_time = time.time()
            reason = "manual (file trigger)" if manual_refresh_requested else f"otomatis (tidak ada aktivitas {time_since_activity:.1f}s)"
            print(f">>> [REFRESH] Terverifikasi background ({check_confidence*100:.1f}%) — referensi diperbarui, {reason}\n")
        else:
            print(f">>> [REFRESH DITUNDA] Frame terdeteksi sebagai '{check_label}' ({check_confidence*100:.1f}%), bukan background.")
            print(f"    Kemungkinan ada barang nyangkut di piringan.\n")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stuck_path = os.path.join(CAPTURE_DIR, "unknown", f"stuck_{timestamp}_{check_label}_{check_confidence:.2f}.jpg")
            cv2.imwrite(stuck_path, frame)

            if check_confidence >= confidence_threshold and check_label in label_to_preset:
                preset = label_to_preset[check_label]
                print(f"    -> Kirim preset {preset} ke ESP buat bersihkan barang nyangkut\n")
                send_to_esp(preset)
                time.sleep(CONFIG["esp"]["post_preset_delay"])
                send_to_esp(0)
                time.sleep(CONFIG["esp"]["post_neutral_delay"])

            last_refresh_time = time.time()
            last_activity_time = time.time()

    # ===== Reclassify berkala selama objek masih dianggap ada (bedakan stuck vs objek baru) =====
    if object_present and next_reclassify_time is not None and time.time() >= next_reclassify_time:
        confidence_threshold = CONFIG["model"]["confidence_threshold"]
        label_to_preset = CONFIG["model"]["label_to_preset"]
        reclassify_cfg = CONFIG["reclassify"]

        if current_bbox is not None:
            rx, ry, rw, rh = current_bbox
            recheck_crop = frame[ry:ry+rh, rx:rx+rw]
        else:
            recheck_crop = frame

        rc_label, rc_conf, _ = classify(recheck_crop)

        if rc_label == 'background' and rc_conf >= confidence_threshold:
            print(">>> [RECLASSIFY] Piringan terkonfirmasi kosong, siap terima objek baru.\n")
            object_present = False
            last_preset_sent = None
            current_bbox = None
            next_reclassify_time = None
            normal_stable_count = 0
            immediate_confirm_count = 0
            detection_start_time = None

        elif rc_conf >= confidence_threshold and rc_label in label_to_preset:
            rc_preset = label_to_preset[rc_label]

            if rc_preset == last_preset_sent:
                if time.time() - last_stuck_retry_time >= reclassify_cfg["stuck_resend_cooldown"]:
                    print(f">>> [RECLASSIFY] Objek sama ({rc_label}) masih nyangkut, retry preset {rc_preset}\n")
                    send_to_esp(rc_preset)
                    time.sleep(CONFIG["esp"]["post_preset_delay"])
                    send_to_esp(0)
                    time.sleep(CONFIG["esp"]["post_neutral_delay"])
                    last_stuck_retry_time = time.time()
                    last_activity_time = time.time()
            else:
                print(f">>> [RECLASSIFY] Objek baru terdeteksi: {rc_label} ({rc_conf*100:.2f}%)\n")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(CAPTURE_DIR, rc_label, f"{timestamp}.jpg")
                cv2.imwrite(save_path, recheck_crop)
                with open(LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, rc_label, f"{rc_conf:.4f}", "", "", "", "", ""])

                send_to_esp(rc_preset)
                time.sleep(CONFIG["esp"]["post_preset_delay"])
                send_to_esp(0)
                time.sleep(CONFIG["esp"]["post_neutral_delay"])
                last_preset_sent = rc_preset
                last_activity_time = time.time()

        next_reclassify_time = time.time() + reclassify_cfg["interval"]

    prev_gray = gray.copy()