# project/app/discord_log.py
import os, requests
from typing import List


WEBHOOK_URL = os.getenv("DISCORD_LOG_WEBHOOK_URL", "")
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
LOG_CHANNEL_ID = os.getenv("DISCORD_LOG_CHANNEL_ID", "")

def log_ask_to_discord(text: str, origin: str = "gradio") -> None:
    """Fire-and-forget: mirror the ask to the Discord log channel via webhook."""
    if not WEBHOOK_URL or not text:
        return
    payload = {
        "content": f"**ASK|{origin}**\n—\n{text[:1900]}",
        "allowed_mentions": {"parse": []},
    }
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=6)
    except Exception:
        print("EXCEPTIOOON")
        # Don't break the app if Discord is down
        pass

def get_last_asks(k: int = 20, min_len: int = 5) -> List[str]:
    """
    Return the last k asks (oldest→newest) from the log channel.
    Handles bold headers (**ASK|gradio**) and skips a divider line (e.g., '—', '---', '___', '***')
    immediately after the header. Paginates up to ~1000 messages.
    """
    if not BOT_TOKEN or not LOG_CHANNEL_ID:
        print(f"bottoken: {BOT_TOKEN}, logchannelid: {LOG_CHANNEL_ID}")
        return []

    def _is_header(line: str) -> bool:
        # remove common markdown wrappers and check for ASK|
        hdr = line.replace("*", "").replace("`", "").strip()
        return hdr.startswith("ASK|")

    def _is_divider(line: str) -> bool:
        s = line.strip()
        if not s:  # empty line after header => treat as divider
            return True
        # only divider characters?
        return all(c in "-—_•·*" for c in s) and len(s) <= 20

    url = f"https://discord.com/api/v10/channels/{LOG_CHANNEL_ID}/messages"
    headers = {"Authorization": f"Bot {BOT_TOKEN}"}
    asks: List[str] = []
    pages = 0
    before: str | None = None

    while len(asks) < k and pages < 10:  # up to ~1000 msgs
        params = {"limit": 100}
        if before:
            params["before"] = before

        try:
            r = requests.get(url, headers=headers, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
        except Exception:
            break

        if not isinstance(data, list) or not data:
            break

        for m in data:  # newest-first
            content = m.get("content", "")
            if not isinstance(content, str):
                continue

            lines = content.splitlines()
            if not lines:
                continue

            # header must be on the first line
            if not _is_header(lines[0]):
                continue

            # skip divider lines after the header
            i = 1
            while i < len(lines) and _is_divider(lines[i]):
                i += 1

            txt = "\n".join(lines[i:]).strip()
            if len(txt) >= min_len:
                asks.append(txt)
                if len(asks) >= k:
                    break

        before = data[-1].get("id")
        pages += 1
        if len(data) < 100:
            break

    asks = asks[-k:]
    asks.reverse()  # oldest→newest
    return asks
