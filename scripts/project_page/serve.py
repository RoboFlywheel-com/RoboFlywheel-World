"""Serve only the review site and opaque display derivatives on loopback."""

import argparse
import re
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs/project-page"
PUBLIC = {
    "index.html": SITE / "index.html", "page.js": SITE / "page.js", "style.css": SITE / "style.css",
    "reproduction.md": ROOT / "docs/reproduction.md", "pipelines.md": ROOT / "docs/pipelines.md",
    "assets/project-demo.webp": SITE / "assets/project-demo.webp",
    "assets/rigid-construction.webp": SITE / "assets/rigid-construction.webp",
    "assets/task-scene-construction.webp": SITE / "assets/task-scene-construction.webp",
}
MEDIA_NAME = re.compile(r"(?:[a-f0-9]{32}\.(?:glb|webp|mp4)|catalog\.js|film\.mp4|viewer\.js|viewer-license\.txt)")


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".webp": "image/webp"}

    def send_head(self):
        path = urlsplit(self.path).path.lstrip("/")
        if path not in ("", *PUBLIC) and not (path.startswith("media/") and MEDIA_NAME.fullmatch(path[6:])):
            self.send_error(404)
            return None
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--media", type=Path, default=SITE / "preview-media.local")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        public = Path(tmp)
        for route, source in PUBLIC.items():
            dest = public / route
            dest.parent.mkdir(exist_ok=True)
            dest.symlink_to(source)
        (public / "media").symlink_to(args.media.absolute(), target_is_directory=True)
        with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=tmp)) as server:
            server.serve_forever()


if __name__ == "__main__":
    main()
