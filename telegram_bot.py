import json
import os
import glob
import time
import uuid
import subprocess
from telegram.ext import Application, CommandHandler

BOT_TOKEN = "8786391966:AAEK7OxPSquJyaj86Pswk83Qso6IoJnkNV0"
CHAT_ID = "6738178793"

BASE_DIR = "/home/smart_bin/Waste-Sorting"  # sesuaikan kalau beda
 
EVENTS_DIR = os.path.join(BASE_DIR, "telegram_sync/events")
EVENTS_SENT_DIR = os.path.join(BASE_DIR, "telegram_sync/events_sent")
COMMANDS_DIR = os.path.join(BASE_DIR, "telegram_sync/commands")
 
POLL_INTERVAL_SEC = 2
MAIN_SERVICE_NAME = "wastesorting.service"  # sesuaikan nama service systemd yang sebenarnya
 
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(EVENTS_SENT_DIR, exist_ok=True)
os.makedirs(COMMANDS_DIR, exist_ok=True)
 
 
# ===================== Command -> main script (via file queue) =====================
 
def write_command(command_name, chat_id, params=None):
    cmd_id = str(uuid.uuid4())[:8]
    fname = f"{time.time_ns()}.json"
    tmp_path = os.path.join(COMMANDS_DIR, f".tmp_{fname}")
    final_path = os.path.join(COMMANDS_DIR, fname)
    with open(tmp_path, "w") as f:
        json.dump({
            "id": cmd_id,
            "command": command_name,
            "params": params or {},
            "chat_id": str(chat_id),
        }, f)
    os.rename(tmp_path, final_path)
    return cmd_id
 
 
async def cmd_pause(update, context):
    write_command("pause", update.effective_chat.id)
    await update.message.reply_text("Perintah pause dikirim, menunggu konfirmasi...")
 
 
async def cmd_resume(update, context):
    write_command("resume", update.effective_chat.id)
    await update.message.reply_text("Perintah resume dikirim, menunggu konfirmasi...")
 
 
async def cmd_status(update, context):
    write_command("status", update.effective_chat.id)
    await update.message.reply_text("Mengambil status...")
 
 
async def cmd_refresh(update, context):
    write_command("refresh_reference", update.effective_chat.id)
    await update.message.reply_text("Perintah refresh referensi dikirim...")
 
 
async def cmd_camera(update, context):
    write_command("camera_check", update.effective_chat.id)
    await update.message.reply_text("Mengambil snapshot kamera...")
 
 
async def cmd_preset(update, context):
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Format: /preset <0-5>")
        return
    preset = int(context.args[0])
    write_command("manual_preset", update.effective_chat.id, {"preset": preset})
    await update.message.reply_text(f"Mengirim preset manual {preset}...")
 
 
# ===================== Command yang dieksekusi langsung di sini =====================
 
async def cmd_restart(update, context):
    await update.message.reply_text("Merestart program utama...")
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", MAIN_SERVICE_NAME],
            timeout=15,
            check=True,
        )
        await update.message.reply_text("Program utama berhasil direstart.")
    except subprocess.TimeoutExpired:
        await update.message.reply_text("Restart timeout, cek manual via SSH.")
    except subprocess.CalledProcessError as e:
        await update.message.reply_text(f"Restart gagal: {e}")
 
 
HELP_TEXT = (
    "*Perintah yang tersedia:*\n\n"
    "/status - Cek status sistem saat ini\n"
    "/pause - Jeda proses sortir sementara\n"
    "/resume - Lanjutkan proses sortir\n"
    "/camera - Ambil snapshot posisi kamera saat ini\n"
    "/preset <0-5> - Gerakkan piringan ke preset tertentu (testing mekanik)\n"
    "/refresh - Paksa refresh referensi background\n"
    "/restart - Restart program utama\n"
    "/help - Tampilkan daftar perintah ini"
)
 
 
async def cmd_help(update, context):
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")
 
 
# ===================== Event <- main script (via file queue) =====================
 
def handle_sort_result(data):
    caption = (
        f"Jenis: {data['label']}\n"
        f"Confidence: {data['confidence']*100:.1f}%\n"
        f"Total waktu: {data.get('total_ms', '-')} ms"
    )
    return caption, data.get("image_path")
 
 
def handle_command_result(data):
    prefix = "OK" if data.get("success") else "GAGAL"
    caption = f"[{prefix}] /{data.get('command')}\n{data.get('message', '')}"
    return caption, data.get("image_path")
 
 
EVENT_HANDLERS = {
    "sort_result": handle_sort_result,
    "command_result": handle_command_result,
}
 
 
async def poll_events(context):
    for path in sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json"))):
        try:
            with open(path) as f:
                event = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # kemungkinan masih ke-race sama proses rename, skip dulu
 
        event_type = event.get("type")
        data = event.get("data", {})
 
        handler = EVENT_HANDLERS.get(event_type)
        if handler is None:
            os.rename(path, os.path.join(EVENTS_SENT_DIR, os.path.basename(path)))
            continue
 
        caption, img_path = handler(data)
        chat_id = data.get("chat_id", CHAT_ID)  # command_result punya chat_id sendiri, sort_result pakai default
 
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as img:
                await context.bot.send_photo(chat_id=chat_id, photo=img, caption=caption)
        else:
            suffix = "\n(gambar tidak ditemukan)" if img_path else ""
            await context.bot.send_message(chat_id=chat_id, text=caption + suffix)
 
        os.rename(path, os.path.join(EVENTS_SENT_DIR, os.path.basename(path)))
 
 
# ===================== Main =====================
 
def main():
    app = Application.builder().token(BOT_TOKEN).build()
 
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("camera", cmd_camera))
    app.add_handler(CommandHandler("preset", cmd_preset))
    app.add_handler(CommandHandler("refresh", cmd_refresh))
    app.add_handler(CommandHandler("restart", cmd_restart))
 
    app.job_queue.run_repeating(poll_events, interval=POLL_INTERVAL_SEC, first=0)
    app.run_polling()
 
 
if __name__ == "__main__":
    main()