import re
import json
from collections import defaultdict
from typing import List, Dict



class TOCParser:
    def __init__(self, toc_txt_path: str):
        self.toc_txt_path = toc_txt_path
        self.entries: List[Dict] = []
        self.nodes: List[Dict] = []

    def _parse_lines(self) -> None:
        with open(self.toc_txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            m = re.match(r'^(\d+(?:\.\d+)*?)\s+(.+?)\s+(\d+)$', line.strip())
            if not m:
                continue
            num, title, page = m.groups()
            level = num.count('.') + 1
            self.entries.append({
                'number': num,
                'title': title,
                'page_start': int(page),
                'level': level,
            })

    def _assign_hierarchy(self) -> None:
        for i, entry in enumerate(self.entries):
            parent = None
            for j in range(i - 1, -1, -1):
                if self.entries[j]['level'] == entry['level'] - 1:
                    parent = self.entries[j]['number']
                    break
            entry['parent_id'] = parent

        children_map = defaultdict(list)
        for e in self.entries:
            if e['parent_id']:
                children_map[e['parent_id']].append(e['number'])

        for e in self.entries:
            node_type = {1: 'section', 2: 'subsection', 3: 'subsubsection'}.get(e['level'], 'paragraph')
            self.nodes.append({
                'id': e['number'],
                'title': e['title'],
                'type': node_type,
                'page_start': e['page_start'],
                'parent_id': e['parent_id'],
                'children_ids': children_map.get(e['number'], []),
            })

    def parse(self) -> List[Dict]:
        self._parse_lines()
        self._assign_hierarchy()
        return self.nodes

    def save_to_json(self, output_path: str) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'nodes': self.nodes}, f, indent=2)
        print(f"TOC saved to {output_path}")


