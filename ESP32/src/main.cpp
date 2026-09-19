#include <Arduino.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// =============================================================
// PIN
// =============================================================
constexpr uint8_t SERVO_A_PIN = 19;
constexpr uint8_t SERVO_B_PIN = 18;

constexpr uint8_t TRIG_PIN  = 27;   // dipakai bareng ke-4 sensor
constexpr uint8_t ECHO1_PIN = 26;   // Plastik
constexpr uint8_t ECHO2_PIN = 25;   // Kertas
constexpr uint8_t ECHO3_PIN = 4;    // Kaleng
constexpr uint8_t ECHO4_PIN = 16;   // Daun

constexpr uint8_t BUZZER_PIN = 13;

constexpr uint8_t I2C_SDA = 22;
constexpr uint8_t I2C_SCL = 21;

// =============================================================
// KONFIGURASI
// =============================================================

// ----- LCD I2C 2004 (20 kolom x 4 baris) -----
constexpr uint8_t LCD_ADDR = 0x27;   // kalau layar blank/kotak-kotak, coba 0x3F
constexpr uint8_t LCD_COLS = 20;
constexpr uint8_t LCD_ROWS = 4;
constexpr uint8_t LCD_RIGHT_COL = 11;   // kolom awal sisi kanan di layar idle

// ----- Servo & preset -----
constexpr int NUM_PRESETS = 6;
constexpr unsigned long SERVO_MOVE_DELAY_MS = 500;   // jeda antar gerak servo pertama & kedua
constexpr unsigned long IDLE_TIMEOUT_MS = 4000;      // LCD balik ke idle setelah hasil sortir tampil

// preset[i] = {sudut servo A, sudut servo B}
const int PRESETS[NUM_PRESETS][2] = {
    {90, 95},    // 0 - netral
    {0, 25},     // 1 - kertas
    {0, 150},    // 2 - plastik  - TODO sesuaikan
    {180, 25},   // 3 - kaleng   - TODO sesuaikan
    {180, 150},  // 4 - daun     - TODO sesuaikan
    {0, 0},      // 5 - cadangan - TODO sesuaikan
};

// dipakai di LCD kalau ESP cuma terima angka polos (tanpa label)
const char* const PRESET_NAMES[NUM_PRESETS] = {
    "Netral", "Kertas", "Plastik", "Kaleng", "Daun", "Preset5"
};

// ----- Ultrasonik -----
constexpr int NUM_ULTRASONIC = 4;
constexpr unsigned long ULTRASONIC_TIMEOUT_US = 25000UL;      // ~4 m
constexpr unsigned long ULTRASONIC_SETTLE_MS = 50;            // jeda antar sensor biar gema reda
constexpr unsigned long ULTRASONIC_READ_INTERVAL_MS = 2000;   // seberapa sering baca ke-4 sensor

// Urutan index sensor: 0 = Plastik, 1 = Kertas, 2 = Kaleng, 3 = Daun
// Nama ini juga jadi key di baris serial ke Pi ("Plastik:12.34 Kertas:...")
const uint8_t ECHO_PINS[NUM_ULTRASONIC] = {ECHO1_PIN, ECHO2_PIN, ECHO3_PIN, ECHO4_PIN};
const char* const BIN_NAMES[NUM_ULTRASONIC] = {"Plastik", "Kertas", "Kaleng", "Daun"};
const char* const BIN_SHORT[NUM_ULTRASONIC] = {"PLTK", "KRTS", "KLNG", "DAUN"};

// jarak sensor ke tumpukan sampah (cm) -- kalibrasi manual per bin
const float BIN_EMPTY_CM[NUM_ULTRASONIC] = {30.0f, 30.0f, 30.0f, 30.0f};   // bin kosong
const float BIN_FULL_CM[NUM_ULTRASONIC]  = {10.0f, 10.0f, 10.0f, 10.0f};   // bin penuh

// ----- Koneksi ke Pi -----
constexpr unsigned long PI_TIMEOUT_MS = 7000;

// ----- Alert (Pi kirim ping berulang, ESP mati sendiri kalau ping berhenti) -----
// Pastikan esp.alert_ping_interval di config.yaml jauh lebih kecil dari TIMEOUT_MS
constexpr unsigned long FULL_ALERT_TIMEOUT_MS  = 5000;
constexpr unsigned long FULL_ALERT_INTERVAL_MS = 5000;   // jeda ulang buzzer
constexpr int FULL_ALERT_BUZZ_COUNT  = 3;
constexpr int FULL_ALERT_BUZZ_ON_MS  = 150;
constexpr int FULL_ALERT_BUZZ_GAP_MS = 150;

constexpr unsigned long STUCK_ALERT_TIMEOUT_MS  = 5000;
constexpr unsigned long STUCK_ALERT_INTERVAL_MS = 5000;
constexpr int STUCK_ALERT_BUZZ_COUNT  = 5;   // beda dari FULL (3x) biar kebedain kupingnya
constexpr int STUCK_ALERT_BUZZ_ON_MS  = 100;
constexpr int STUCK_ALERT_BUZZ_GAP_MS = 100;

constexpr size_t RX_BUFFER_MAX = 64;

// =============================================================
// STATE
// =============================================================
Servo servoA;
Servo servoB;
LiquidCrystal_I2C lcd(LCD_ADDR, LCD_COLS, LCD_ROWS);

String rxBuffer = "";

float distanceCM[NUM_ULTRASONIC] = {-1, -1, -1, -1};
unsigned long lastUltrasonicRead = 0;

unsigned long lastPingFromPi = 0;
bool everPinged = false;       // belum ada ping = dianggap offline
bool piOnline = false;
bool piOnlinePrev = false;

unsigned long totalSortir = 0; // jumlah perintah sortir (preset != 0) sejak boot
unsigned long lastActionTime = 0;
bool showingIdle = true;

String binFullAlertLabel = "";

// =============================================================
// HELPER UMUM
// =============================================================
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

// return persen penuh (0-100), atau -1 kalau sensor gagal baca
int distanceToPercent(float distCM, float emptyCM, float fullCM) {
    if (distCM < 0 || emptyCM == fullCM) return -1;

    float percent = (emptyCM - distCM) / (emptyCM - fullCM) * 100.0f;
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;
    return (int)percent;
}

// =============================================================
// LCD
// =============================================================

// Tulis 1 baris penuh (dipotong/dipad spasi ke 20 kolom), jadi gak perlu
// lcd.clear() dan layar gak kedip tiap refresh.
void lcdLine(uint8_t row, const String& text) {
    String s = text;
    if (s.length() > LCD_COLS) s = s.substring(0, LCD_COLS);
    while (s.length() < LCD_COLS) s += ' ';
    lcd.setCursor(0, row);
    lcd.print(s);
}

// contoh: "PLTK 45%" atau "PLTK N/A"
String binCell(int bin) {
    int pct = distanceToPercent(distanceCM[bin], BIN_EMPTY_CM[bin], BIN_FULL_CM[bin]);
    String s = String(BIN_SHORT[bin]) + " ";
    s += (pct >= 0) ? String(pct) + "%" : String("N/A");
    return s;
}

String twoColumns(const String& left, const String& right) {
    String s = left;
    while (s.length() < LCD_RIGHT_COL) s += ' ';
    return s + right;
}

void lcdShowIdle() {
    lcdLine(0, "WASTE SORTING SYSTEM");
    lcdLine(1, String("READY   PI:") + (piOnline ? "OK" : "OFF"));
    lcdLine(2, twoColumns(binCell(0), binCell(2)));   // Plastik | Kaleng
    lcdLine(3, twoColumns(binCell(1), binCell(3)));   // Kertas  | Daun
    showingIdle = true;
}

// idx        : index preset yang dieksekusi
// label      : nama kelas dari klasifikasi (kosong kalau ESP cuma terima angka)
// confidence : dalam persen (0-100)
// hasLabel   : true kalau label & confidence memang dikirim dari Python
void lcdShowResult(int idx, const String& label, float confidence, bool hasLabel) {
    lcdLine(0, String("Jenis : ") + (hasLabel ? label : String(PRESET_NAMES[idx])));
    lcdLine(1, hasLabel ? String("Conf. : ") + String(confidence, 1) + " %"
                        : String("(tanpa data conf.)"));
    lcdLine(2, String("Bin   : ") + String(idx) + " - " + PRESET_NAMES[idx]);
    lcdLine(3, String("Total sortir: ") + String(totalSortir));

    showingIdle = false;
    lastActionTime = millis();
}

void lcdShowBinFull() {
    lcdLine(0, "TOLONG AMBIL LAGI");
    lcdLine(1, "SAMPAHNYA!");
    lcdLine(2, String("Bin ") + binFullAlertLabel);
    lcdLine(3, "PENUH");
    showingIdle = false;
}

void lcdShowStuck() {
    lcdLine(0, "OBJEK TERSANGKUT!");
    lcdLine(1, "TOLONG AMBIL LAGI!");
    lcdLine(2, "");
    lcdLine(3, "");
    showingIdle = false;
}

void lcdShowLowConfRetry(const String& label, float confidence, int attempt, int maxAttempts) {
    lcdLine(0, "Deteksi : " + label);
    lcdLine(1, "Conf.   : " + String(confidence, 1) + " %");
    lcdLine(2, "Conf terlalu rendah,");
    lcdLine(3, "coba lagi (" + String(attempt) + "/" + String(maxAttempts) + ")");
    showingIdle = false;
    lastActionTime = millis();
}

// =============================================================
// ALERT (FULL & STUCK pakai mekanisme yang sama)
// =============================================================
struct Alert {
    unsigned long timeoutMs;     // mati sendiri kalau gak ada ping selama ini
    unsigned long intervalMs;    // jeda ulang buzzer selama aktif
    int buzzCount, buzzOnMs, buzzGapMs;
    void (*showScreen)();

    bool active;
    unsigned long lastReceived;
    unsigned long lastBuzz;

    Alert(unsigned long timeout, unsigned long interval,
          int count, int onMs, int gapMs, void (*screen)())
        : timeoutMs(timeout), intervalMs(interval),
          buzzCount(count), buzzOnMs(onMs), buzzGapMs(gapMs),
          showScreen(screen),
          active(false), lastReceived(0), lastBuzz(0) {}
};

Alert fullAlert(FULL_ALERT_TIMEOUT_MS, FULL_ALERT_INTERVAL_MS,
                FULL_ALERT_BUZZ_COUNT, FULL_ALERT_BUZZ_ON_MS, FULL_ALERT_BUZZ_GAP_MS,
                lcdShowBinFull);

Alert stuckAlert(STUCK_ALERT_TIMEOUT_MS, STUCK_ALERT_INTERVAL_MS,
                 STUCK_ALERT_BUZZ_COUNT, STUCK_ALERT_BUZZ_ON_MS, STUCK_ALERT_BUZZ_GAP_MS,
                 lcdShowStuck);

// layar yang harus tampil sesuai alert yang aktif (STUCK diprioritaskan)
void showCurrentScreen() {
    if (stuckAlert.active)      lcdShowStuck();
    else if (fullAlert.active)  lcdShowBinFull();
    else                        lcdShowIdle();
}

void alertFire(Alert& a) {
    buzzBeep(a.buzzCount, a.buzzOnMs, a.buzzGapMs);
    a.showScreen();
    a.lastBuzz = millis();
}

// dipanggil tiap pesan alert dari Pi masuk
void alertReceived(Alert& a) {
    a.lastReceived = millis();
    if (!a.active) {
        // transisi off -> on: langsung buzz & tampilkan, jangan nunggu interval
        a.active = true;
        alertFire(a);
    }
}

void alertClear(Alert& a) {
    if (a.active) {
        a.active = false;
        showCurrentScreen();
    }
}

// dipanggil tiap loop, SETELAH baca serial (biar ping yang numpuk
// selama delay() servo gak bikin alert salah dianggap timeout)
void alertUpdate(Alert& a) {
    if (!a.active) return;

    unsigned long now = millis();
    if (now - a.lastReceived > a.timeoutMs) {
        alertClear(a);
    } else if (now - a.lastBuzz > a.intervalMs) {
        alertFire(a);   // ulang buzzer & refresh layar
    }
}

// =============================================================
// ULTRASONIK
// =============================================================

// trigger 10us lalu ukur lebar pulsa HIGH di echoPin
// return -1 kalau timeout (di luar jangkauan / gak ada pantulan)
float readUltrasonicCM(uint8_t echoPin) {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    unsigned long duration = pulseIn(echoPin, HIGH, ULTRASONIC_TIMEOUT_US);
    if (duration == 0) return -1;

    return duration * 0.0343f / 2.0f;   // cm
}

// baca ke-4 sensor bergantian karena TRIG_PIN dipakai bareng
void readAllUltrasonic() {
    for (int i = 0; i < NUM_ULTRASONIC; i++) {
        distanceCM[i] = readUltrasonicCM(ECHO_PINS[i]);
        delay(ULTRASONIC_SETTLE_MS);
    }
    lastUltrasonicRead = millis();
}

// format: "Plastik:12.34 Kertas:-1.00 Kaleng:8.50 Daun:20.10"
// (-1 = sensor gagal baca; Pi harus mengabaikannya)
void printSensorData() {
    for (int i = 0; i < NUM_ULTRASONIC; i++) {
        Serial.print(BIN_NAMES[i]);
        Serial.print(':');
        Serial.print(distanceCM[i]);
        if (i < NUM_ULTRASONIC - 1) Serial.print(' ');
    }
    Serial.println();
}

// =============================================================
// SERVO
// =============================================================
void movePreset(int idx) {
    if (idx == 0) {
        // netral: servo A dulu, baru servo B
        servoA.write(PRESETS[idx][0]);
        delay(SERVO_MOVE_DELAY_MS);
        servoB.write(PRESETS[idx][1]);
    } else {
        // sortir: servo B dulu, baru servo A
        servoB.write(PRESETS[idx][1]);
        delay(SERVO_MOVE_DELAY_MS);
        servoA.write(PRESETS[idx][0]);
    }
}

// =============================================================
// PARSING SERIAL
// =============================================================

// format: "<idx>" atau "<idx>,<label>,<confidence>"
void handlePresetCommand(const String& line) {
    if (!isDigit(line[0])) {
        // tanpa cek ini, teks apa pun akan di-toInt() jadi 0 -> servo ke netral
        Serial.print("Perintah tidak dikenal: ");
        Serial.println(line);
        return;
    }

    int comma1 = line.indexOf(',');
    String idxStr = (comma1 == -1) ? line : line.substring(0, comma1);
    int idx = idxStr.toInt();

    String label = "";
    float confidence = 0;
    bool hasLabel = false;

    if (comma1 != -1) {
        int comma2 = line.indexOf(',', comma1 + 1);
        if (comma2 != -1) {
            label = line.substring(comma1 + 1, comma2);
            confidence = line.substring(comma2 + 1).toFloat();
            hasLabel = true;
        }
    }

    if (idx < 0 || idx >= NUM_PRESETS) {
        Serial.println("Preset tidak valid. Gunakan angka 0-5.");
        return;
    }

    movePreset(idx);

    Serial.print("Preset ");
    Serial.print(idx);
    Serial.print(" -> A: ");
    Serial.print(PRESETS[idx][0]);
    Serial.print(", B: ");
    Serial.println(PRESETS[idx][1]);

    // LCD & counter cuma untuk sortir sungguhan (idx != 0), supaya perintah
    // "balik netral" otomatis setelah tiap sortir gak menimpa hasil di layar.
    if (idx != 0) {
        totalSortir++;
        lcdShowResult(idx, label, confidence, hasLabel);
    }
}

void handleLine(String line) {
    line.trim();
    if (line.length() == 0) return;

    if (line == "PING") {
        lastPingFromPi = millis();
        everPinged = true;

    } else if (line == "c") {
        readAllUltrasonic();
        printSensorData();

    } else if (line.startsWith("FULL:")) {
        String newLabel = line.substring(5);
        newLabel.trim();
        bool labelChanged = (newLabel != binFullAlertLabel);
        bool wasActive = fullAlert.active;

        binFullAlertLabel = newLabel;
        alertReceived(fullAlert);

        if (wasActive && labelChanged) showCurrentScreen();

    } else if (line == "STUCK") {
        alertReceived(stuckAlert);

    } else if (line.startsWith("RC:")) {
    String payload = line.substring(3);
    int c1 = payload.indexOf(',');
    int c2 = payload.indexOf(',', c1 + 1);
    int c3 = payload.indexOf(',', c2 + 1);

    String label = payload.substring(0, c1);
    float conf = payload.substring(c1 + 1, c2).toFloat();
    int attempt = payload.substring(c2 + 1, c3).toInt();
    int maxAttempts = payload.substring(c3 + 1).toInt();

    lcdShowLowConfRetry(label, conf, attempt, maxAttempts);
    } else {
        handlePresetCommand(line);
    }
}

void readSerialCommands() {
    while (Serial.available() > 0) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (rxBuffer.length() > 0) {
                handleLine(rxBuffer);
                rxBuffer = "";
            }
        } else if (rxBuffer.length() < RX_BUFFER_MAX) {
            rxBuffer += c;
        }
    }
}

// =============================================================
// SETUP & LOOP
// =============================================================
void setup() {
    Serial.begin(115200);

    pinMode(TRIG_PIN, OUTPUT);
    for (int i = 0; i < NUM_ULTRASONIC; i++) {
        pinMode(ECHO_PINS[i], INPUT);
    }
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    Wire.begin(I2C_SDA, I2C_SCL);
    lcd.init();
    lcd.backlight();
    lcd.clear();
    lcdShowIdle();

    // ESP32Servo perlu allocate timer PWM (1 timer per servo)
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);

    servoA.setPeriodHertz(50);
    servoA.attach(SERVO_A_PIN, 500, 2400);
    servoB.setPeriodHertz(50);
    servoB.attach(SERVO_B_PIN, 500, 2400);

    servoA.write(PRESETS[0][0]);
    servoB.write(PRESETS[0][1]);

    Serial.println("Servo & LCD siap.");
    Serial.println("Format serial: '<preset>' atau '<preset>,<label>,<confidence>'");
}

void loop() {
    // 1. Baca sensor berkala & kirim ke Pi
    if (millis() - lastUltrasonicRead > ULTRASONIC_READ_INTERVAL_MS) {
        readAllUltrasonic();
        printSensorData();

        if (showingIdle) lcdShowIdle();   // refresh persen kapasitas
    }

    // 2. Status koneksi Pi (offline sampai ping pertama masuk)
    piOnline = everPinged && (millis() - lastPingFromPi) < PI_TIMEOUT_MS;

    if (piOnline != piOnlinePrev) {
        if (piOnline) {
            buzzBeep(2);          // Pi baru konek / kembali online -> 2x bip
        } else {
            buzzBeep(1, 300);     // Pi putus -> 1x bip panjang
        }
        piOnlinePrev = piOnline;
    }

    // 3. Proses perintah dari Pi
    readSerialCommands();

    // 4. Alert: ulang buzzer atau matikan kalau ping berhenti
    alertUpdate(fullAlert);
    alertUpdate(stuckAlert);

    // 5. Balik ke layar idle setelah hasil sortir tampil beberapa detik
    if (!showingIdle && !fullAlert.active && !stuckAlert.active &&
        millis() - lastActionTime > IDLE_TIMEOUT_MS) {
        lcdShowIdle();
    }
}