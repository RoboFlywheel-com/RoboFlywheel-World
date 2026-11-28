"""The same figure links must work at the server root and behind a port proxy."""

import unittest
from html.parser import HTMLParser
from urllib.parse import urljoin

from serve import Handler, PUBLIC, SITE


class Figures(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        url = attrs.get({"a": "href", "img": "src", "video": "poster"}.get(tag))
        if url and url.endswith(".webp"):
            self.urls.append(url)


class FigureRoutes(unittest.TestCase):
    def test_static_figures(self):
        links = Figures()
        links.feed((SITE / "index.html").read_text())
        self.assertEqual(len(set(links.urls)), 3)
        for base in ["http://preview.test/", "http://preview.test/forward/8081/"]:
            for link in links.urls:
                with self.subTest(base=base, link=link):
                    url = urljoin(base, link)
                    self.assertTrue(url.startswith(base))
                    route = url[len(base):]
                    self.assertTrue(PUBLIC[route].is_file())
                    self.assertEqual(Handler.extensions_map.get(".webp"), "image/webp")


if __name__ == "__main__":
    unittest.main()
