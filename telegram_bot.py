rimport json
import os
import glob
from telegram.ext import Application

BOT_TOKEN = "8786391966:AAFupAO-71fzxyLBngIymkKOw6R20plcbCo"
CHAT_ID = "6738178793"

EVENTS_DIR = "telegram_sync/events"
EVENTS_SENT_DIR = "telegram_sync/events_sent"
POLL_INTERVAL_SEC = 2
os.makedirs(EVENTS_SENT_DIR, exist_ok=True)

async def poll_events(context):
    for path in sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json"))):
        try:
            with open(path) as f:
                event = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # kemungkinan masih ke-race sama proses rename, skip dulu

        data = event["data"]
        caption = (
            f"Jenis: {data['label']}\n"
            f"Confidence: {data['confidence']*100:.1f}%\n"
            f"Total waktu: {data['total_ms']} ms"
        )
        img_path = data["image_path"]
        if os.path.exists(img_path):
            with open(img_path, "rb") as img:
                await context.bot.send_photo(chat_id=CHAT_ID, photo=img, caption=caption)
        else:
            await context.bot.send_message(chat_id=CHAT_ID, text=caption + "\n(gambar tidak ditemukan)")

        os.rename(path, os.path.join(EVENTS_SENT_DIR, os.path.basename(path)))

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.job_queue.run_repeating(poll_events, interval=POLL_INTERVAL_SEC, first=0)
    app.run_polling()

if __name__ == "__main__":
    main()