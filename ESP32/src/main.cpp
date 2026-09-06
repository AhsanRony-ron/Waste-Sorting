#include <Arduino.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define SERVO_A_PIN 19
#define SERVO_B_PIN 18

#define TRIG_PIN 27
#define ECHO1_PIN 26
#define ECHO2_PIN 25
#define ECHO3_PIN 4
#define ECHO4_PIN 16

#define BUZZER_PIN 13

// ===== LCD I2C 2004 (20 kolom x 4 baris) =====
#define LCD_ADDR 0x27   // kalau layar blank/kotak-kotak, coba 0x3F (alamat umum kedua)
#define LCD_COLS 20
#define LCD_ROWS 4
#define I2C_SDA 22
#define I2C_SCL 21

#define NUM_PRESETS 6
#define SERVO_MOVE_DELAY_MS 500 // jeda antar gerak servo pertama & kedua, sesuaikan kebutuhan
#define IDLE_TIMEOUT_MS 4000    // LCD balik ke layar idle brp lama setelah hasil ditampilkan

#define NUM_ULTRASONIC 4
#define ULTRASONIC_TIMEOUT_US 25000UL   // ~4m, sesuaikan kalau jarak maksimal beda
#define ULTRASONIC_SETTLE_MS 50         // jeda antar trigger biar gema sensor sebelumnya reda
#define ULTRASONIC_READ_INTERVAL_MS 2000 // seberapa sering baca ke-4 sensor di loop()

#define PI_TIMEOUT_MS 3000
unsigned long lastPingFromPi = 0;
bool piOnline = false;
bool piOnlinePrev = false;

const uint8_t echoPins[NUM_ULTRASONIC] = {ECHO1_PIN, ECHO2_PIN, ECHO3_PIN, ECHO4_PIN};
float distanceCM[NUM_ULTRASONIC] = {-1, -1, -1, -1};
unsigned long lastUltrasonicRead = 0;

Servo servoA;
Servo servoB;
LiquidCrystal_I2C lcd(LCD_ADDR, LCD_COLS, LCD_ROWS);
String rxBuffer = "";

// preset[i] = {sudut servo A, sudut servo B}
// silakan sesuaikan nilai index 2-5 sesuai kebutuhan
int presets[NUM_PRESETS][2] = {
    {90, 95},   // 0 - netral
    {0, 25},    // 1 - kertas
    {0, 150},   // 2 - plastik - TODO sesuaikan
    {180, 25},  // 3 - kaleng  - TODO sesuaikan
    {180, 150}, // 4 - daun    - TODO sesuaikan
    {0, 0},     // 5 - cadangan - TODO sesuaikan
};

// nama tiap preset, dipakai di LCD kalau ESP cuma terima angka polos (tanpa label)
const char* presetNames[NUM_PRESETS] = {
    "Netral", "Kertas", "Plastik", "Kaleng", "Daun", "Preset5"
};

// ===== Counter debug: total tersortir sejak boot, per kategori =====
unsigned long countKertas = 0;
unsigned long countPlastik = 0;
unsigned long countKaleng = 0;
unsigned long countDaun = 0;
unsigned long countLain = 0;
unsigned long totalSortir = 0;

unsigned long lastActionTime = 0;
bool showingIdle = true;

// jarak sensor ke tumpukan sampah saat bin KOSONG (cm) — kalibrasi manual per bin
float binEmptyCM[NUM_ULTRASONIC] = {30.0, 30.0, 30.0, 30.0};

// jarak sensor ke tumpukan sampah saat bin PENUH (cm) — kalibrasi manual per bin
float binFullCM[NUM_ULTRASONIC]  = {10.0, 10.0, 10.0, 10.0};
int distanceToPercent(float distCM, float emptyCM, float fullCM) {
    if (distCM < 0) return -1;

    float percent = (emptyCM - distCM) / (emptyCM - fullCM) * 100.0f;
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;
    return (int)percent;
}

void lcdShowIdle() {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("WASTE SORTING SYSTEM");
    lcd.setCursor(0, 1);
    lcd.print("READY   PI:");
    lcd.print(piOnline ? "OK " : "OFF");;

    int pct0 = distanceToPercent(distanceCM[0], binEmptyCM[0], binFullCM[0]);
    int pct1 = distanceToPercent(distanceCM[1], binEmptyCM[1], binFullCM[1]);
    int pct2 = distanceToPercent(distanceCM[2], binEmptyCM[2], binFullCM[2]);
    int pct3 = distanceToPercent(distanceCM[3], binEmptyCM[3], binFullCM[3]);

    lcd.setCursor(0, 2);
    lcd.print("BIN1 ");
    lcd.print(pct0 >= 0 ? String(pct0) + "%" : "N/A");
    lcd.print(" BIN2 ");
    lcd.print(pct1 >= 0 ? String(pct1) + "%" : "N/A");

    lcd.setCursor(0, 3);
    lcd.print("BIN3 ");
    lcd.print(pct2 >= 0 ? String(pct2) + "%" : "N/A");
    lcd.print(" BIN4 ");
    lcd.print(pct3 >= 0 ? String(pct3) + "%" : "N/A");

    showingIdle = true;
}

// idx        : index preset yang dieksekusi
// label      : nama kelas hasil klasifikasi (kosong kalau ESP cuma terima angka)
// confidence : dalam persen (0-100)
// hasLabel   : true kalau data label & confidence memang dikirim dari Python
void lcdShowResult(int idx, String label, float confidence, bool hasLabel) {
    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print("Jenis : ");
    lcd.print(hasLabel ? label : String(presetNames[idx]));

    lcd.setCursor(0, 1);
    if (hasLabel) {
        lcd.print("Conf. : ");
        lcd.print(confidence, 1);
        lcd.print(" %");
    } else {
        lcd.print("(tanpa data conf.)");
    }

    lcd.setCursor(0, 2);
    lcd.print("Bin   : ");
    lcd.print(idx);
    lcd.print(" - ");
    lcd.print(presetNames[idx]);

    lcd.setCursor(0, 3);
    lcd.print("Total sortir: ");
    lcd.print(totalSortir);

    showingIdle = false;
    lastActionTime = millis();
}

// trigger 10us lalu ukur lebar pulsa HIGH di echoPin tertentu
// return -1 kalau timeout (di luar jangkauan / gak ada pantulan)
float readUltrasonicCM(uint8_t echoPin) {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    unsigned long duration = pulseIn(echoPin, HIGH, ULTRASONIC_TIMEOUT_US);
    if (duration == 0) return -1;

    return duration * 0.0343f / 2.0f; // cm
}

// baca ke-4 sensor bergantian karena TRIG_PIN dipakai bareng
void readAllUltrasonic() {
    for (int i = 0; i < NUM_ULTRASONIC; i++) {
        distanceCM[i] = readUltrasonicCM(echoPins[i]);
        delay(ULTRASONIC_SETTLE_MS);
    }
}

void buzzBeep(int times, int onMs = 100, int gapMs = 100) {
    for (int i = 0; i < times; i++) {
        digitalWrite(BUZZER_PIN, HIGH);
        delay(onMs);
        digitalWrite(BUZZER_PIN, LOW);
        if (i < times - 1) {
            delay(gapMs);
        }
    }
}

void setup() {
    Serial.begin(115200);

    Wire.begin(I2C_SDA, I2C_SCL);
    lcd.init();
    lcd.backlight();
    lcdShowIdle();

    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO1_PIN, INPUT); 
    pinMode(ECHO2_PIN, INPUT);
    pinMode(ECHO3_PIN, INPUT);
    pinMode(ECHO4_PIN, INPUT);
    pinMode(BUZZER_PIN, OUTPUT);

    // ESP32Servo perlu allocate timer PWM (1 timer per servo)
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);

    servoA.setPeriodHertz(50);
    servoA.attach(SERVO_A_PIN, 500, 2400);

    servoB.setPeriodHertz(50);
    servoB.attach(SERVO_B_PIN, 500, 2400);

    servoA.write(presets[0][0]);
    servoB.write(presets[0][1]);

    Serial.println("Servo & LCD siap.");
    Serial.println("Format serial: '<preset>' atau '<preset>,<label>,<confidence>'");
}

void loop() {

    if (millis() - lastUltrasonicRead > ULTRASONIC_READ_INTERVAL_MS) {
        readAllUltrasonic();
        lastUltrasonicRead = millis();

        // contoh debug, hapus/ganti sesuai kebutuhan logika deteksi objek
        Serial.print("US1:"); Serial.print(distanceCM[0]);
        Serial.print(" US2:"); Serial.print(distanceCM[1]);
        Serial.print(" US3:"); Serial.print(distanceCM[2]);
        Serial.print(" US4:"); Serial.println(distanceCM[3]);
            
        
        if (showingIdle) {
        lcdShowIdle();
        }

    }

    piOnline = (millis() - lastPingFromPi) < PI_TIMEOUT_MS;

    if (piOnline != piOnlinePrev) {
        if (piOnline) {
            buzzBeep(2);   // Pi baru konek/kembali online -> 2x bip
        } else {
            buzzBeep(1, 300);   // Pi disconnect -> 1x bip
        }
        piOnlinePrev = piOnline;
    }

    while (Serial.available() > 0) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (rxBuffer.length() > 0) {

                // di dalam blok parsing rxBuffer, sejajar sama `if (rxBuffer == "c")`:
                if (rxBuffer == "PING") {
                    lastPingFromPi = millis();
                    // gak perlu proses lain, cuma nandain Pi masih hidup
                }

                if (rxBuffer == "c") {
                    readAllUltrasonic();

                    Serial.print("US1:"); Serial.print(distanceCM[0]);
                    Serial.print(",US2:"); Serial.print(distanceCM[1]);
                    Serial.print(",US3:"); Serial.print(distanceCM[2]);
                    Serial.print(",US4:"); Serial.println(distanceCM[3]);
                    
                } else {
                    // parsing "idx" biasa, atau "idx,label,confidence" dari Python
                    int comma1 = rxBuffer.indexOf(',');
                    String idxStr = (comma1 == -1) ? rxBuffer : rxBuffer.substring(0, comma1);
                    int idx = idxStr.toInt();

                    String label = "";
                    float confidence = 0;
                    bool hasLabel = false;

                    if (comma1 != -1) {
                        int comma2 = rxBuffer.indexOf(',', comma1 + 1);
                        if (comma2 != -1) {
                            label = rxBuffer.substring(comma1 + 1, comma2);
                            confidence = rxBuffer.substring(comma2 + 1).toFloat();
                            hasLabel = true;
                        }
                    }

                    if (idx >= 0 && idx < NUM_PRESETS) {
                        if (idx == 0) {
                            // preset 0: servo A gerak dulu, baru servo B
                            servoA.write(presets[idx][0]);
                            delay(SERVO_MOVE_DELAY_MS);
                            servoB.write(presets[idx][1]);
                        } else {
                            // default: servo B gerak dulu, baru servo A
                            servoB.write(presets[idx][1]);
                            delay(SERVO_MOVE_DELAY_MS);
                            servoA.write(presets[idx][0]);
                        }

                        Serial.print("Preset ");
                        Serial.print(idx);
                        Serial.print(" -> A: ");
                        Serial.print(presets[idx][0]);
                        Serial.print(", B: ");
                        Serial.println(presets[idx][1]);

                        // LCD & counter cuma diupdate untuk aksi sortir sungguhan (idx != 0),
                        // supaya perintah "balik netral" yang dikirim otomatis setelah tiap
                        // sortir tidak langsung menimpa tampilan hasil di layar.
                        if (idx != 0) {
                            lcdShowResult(idx, label, confidence, hasLabel);
                        }
                    } else {
                        Serial.println("Preset tidak valid. Gunakan angka 0-5.");
                    }
                }
            }
            rxBuffer = "";
        } else {
            rxBuffer += c;
        }
    }

    // otomatis balik ke layar idle/statistik kalau sudah lewat IDLE_TIMEOUT_MS
    if (!showingIdle && millis() - lastActionTime > IDLE_TIMEOUT_MS) {
        lcdShowIdle();
    }
}