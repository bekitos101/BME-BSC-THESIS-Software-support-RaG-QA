# docs_chunker.py

import json
import re
from pathlib import Path
from PyPDF2 import PdfReader


class UserManualChunker:
    def __init__(self, pdf_path: str, metadata_path: str):
        self.pdf_path = pdf_path
        self.metadata_path = metadata_path
        self.full_text = ""
        self.nodes = []

    def _load_metadata(self):
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.nodes = json.load(f)['nodes']

    def _extract_clean_text(self):
        header_re = re.compile(r'^CPI Automation Portal User Guide')
        footer_re = re.compile(r'\s*\d+\s+\S.*\|\s*\d{4}-\d{2}-\d{2}\s*$')
        page_num_re = re.compile(r'^\s*\d+\s*$')

        reader = PdfReader(self.pdf_path)
        clean_pages = []

        for page in reader.pages:
            raw_lines = (page.extract_text() or '').splitlines()
            cleaned = [ln for ln in raw_lines
                       if not header_re.match(ln)
                       and not footer_re.match(ln)
                       and not page_num_re.match(ln)]
            clean_pages.append("\n".join(cleaned))

        self.full_text = "\n".join(clean_pages)

    def _locate_sections(self):
        for node in self.nodes:
            pattern = re.escape(node['id']) + r'\s+' + re.escape(node['title'])
            m = re.search(pattern, self.full_text)
            node['char_start'] = m.start() if m else None

        sorted_nodes = sorted(
            [n for n in self.nodes if n.get('char_start') is not None],
            key=lambda n: n['char_start']
        )

        for i, node in enumerate(sorted_nodes):
            node['char_end'] = sorted_nodes[i + 1]['char_start'] if i < len(sorted_nodes) - 1 else len(self.full_text)
            node['text'] = self.full_text[node['char_start']:node['char_end']].strip()

        self.nodes = sorted_nodes

    def run(self, output_path: str):
        self._load_metadata()
        self._extract_clean_text()
        self._locate_sections()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'nodes': self.nodes}, f, indent=2)

        print(f"\u2705 Chunking complete! Saved to {output_path} with {len(self.nodes)} sections")



