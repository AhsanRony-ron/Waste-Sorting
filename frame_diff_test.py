import cv2
import numpy as np
import time

cap = cv2.VideoCapture(1)

# Ambil frame referensi (kondisi "kosong")
print("Ambil frame referensi dalam 3 detik, pastikan area kosong...")
time.sleep(3)
ret, reference = cap.read()
reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
reference_gray = cv2.GaussianBlur(reference_gray, (25, 25), 0)

DIFF_THRESHOLD = 50
CHANGE_AREA_THRESHOLD = 5000
STABLE_FRAMES_NEEDED = 10

prev_gray = reference_gray.copy()
stable_count = 0
object_present = False

print("Mulai monitoring... taruh objek di depan kamera. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)

    diff_ref = cv2.absdiff(reference_gray, gray)
    thresh_ref = cv2.threshold(diff_ref, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh_ref = cv2.erode(thresh_ref, kernel, iterations=1)   # hilangin bintik kecil
    thresh_ref = cv2.dilate(thresh_ref, kernel, iterations=2)  # gabungin lagi area objek yang mungkin kepotong
    change_area = cv2.countNonZero(thresh_ref)

    diff_prev = cv2.absdiff(prev_gray, gray)
    thresh_prev = cv2.threshold(diff_prev, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    motion_area = cv2.countNonZero(thresh_prev)

    status_text = f"change_area={change_area} motion_area={motion_area} stable={stable_count}"

    if change_area > CHANGE_AREA_THRESHOLD:
        if motion_area < 500:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= STABLE_FRAMES_NEEDED and not object_present:
            print(f"OBJEK TERDETEKSI & STABIL. (change_area={change_area})")
            cv2.imwrite("captured_object.jpg", frame)
            object_present = True
    else:
        if object_present:
            print("Area kosong lagi.")
        object_present = False
        stable_count = 0

    prev_gray = gray.copy()

    # Visualisasi (cuma bisa di laptop, nanti dihapus pas pindah ke Pi Lite)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Diff Mask (vs reference)", thresh_ref)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()