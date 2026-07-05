import json
from pathlib import Path
from typing import List, Dict, Optional


class LogViewer:
    def __init__(self, log_path: str = "backend/resources/chat_logs.jsonl"):
        self.log_path = Path(log_path)

    def read_entries(self) -> List[Dict]:
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def get_entry(self, index: int) -> Optional[Dict]:
        entries = self.read_entries()
        if 0 <= index < len(entries):
            return entries[index]
        return None

    def append_entry(self, entry: Dict):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
