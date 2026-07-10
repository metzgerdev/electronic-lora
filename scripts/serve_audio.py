"""Tiny local audio server for the review sheet's listen links.

    .venv/bin/python scripts/serve_audio.py     # serves training_music on :8765

Supports HTTP Range requests so the browser's audio player can seek
(python's stock http.server can't, which breaks scrubbing). Keep this
running during review sessions; links in review_sheet.xlsx point here.
Binds to localhost only — nothing is exposed to the network.
"""

import os
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

AUDIO_ROOT = Path.home() / "Documents/Code/training_music"
PORT = 8765

MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".aiff": "audio/aiff"}


class RangeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(AUDIO_ROOT), **kwargs)

    def guess_type(self, path):
        return MIME.get(Path(path).suffix.lower(), super().guess_type(path))

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not m:
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return open(path, "rb")

        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        f = open(path, "rb")
        f.seek(start)
        self._range_remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)
        self._range_remaining = None


if __name__ == "__main__":
    print(f"serving {AUDIO_ROOT} at http://localhost:{PORT} (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", PORT), RangeHandler).serve_forever()
