"""P1.6 - Run artefacts: manifest, per-agent logs, message trace, deterministic trace, snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def dumps(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_path(p) -> str:
    """Stable content hash. XML comments are stripped: netconvert embeds a wall-clock timestamp there."""
    import re
    b = Path(p).read_bytes()
    if str(p).endswith(".xml"):
        b = re.sub(rb"<!--.*?-->", b"", b, flags=re.S)
    return hashlib.sha256(b).hexdigest()


class RunLogger:
    def __init__(self, run_dir: Path, trace_level: str = "full"):
        self.dir = Path(run_dir)
        for sub in ("agents", "evaluator"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.trace_level = trace_level
        self._open: dict = {}
        self._msg = open(self.dir / "messages.jsonl", "w", encoding="utf-8")
        self._trace = open(self.dir / "trace.jsonl", "w", encoding="utf-8")
        self._glob = open(self.dir / "global_snapshots.jsonl", "w", encoding="utf-8")
        self._truth = open(self.dir / "evaluator" / "truth.jsonl", "w", encoding="utf-8")
        self._events = open(self.dir / "evaluator" / "events.jsonl", "w", encoding="utf-8")
        self.n_msg_lines = 0

    # agent-local logs: what each agent saw/sent, nothing else
    def agent(self, agent_id: str, record: dict):
        f = self._open.get(agent_id)
        if f is None:
            f = self._open[agent_id] = open(self.dir / "agents" / f"{agent_id}.jsonl", "w", encoding="utf-8")
        f.write(dumps(record) + "\n")

    def close_agent(self, agent_id: str):
        f = self._open.pop(agent_id, None)
        if f:
            f.close()

    def msg(self, rec: dict, exception: bool):
        if self.trace_level == "full" or exception:
            self._msg.write(dumps(rec) + "\n")
            self.n_msg_lines += 1

    def trace(self, rec: dict):
        self._trace.write(dumps(rec) + "\n")

    def glob(self, rec: dict):
        self._glob.write(dumps(rec) + "\n")

    def truth(self, rec: dict):
        self._truth.write(dumps(rec) + "\n")

    def event(self, rec: dict):
        self._events.write(dumps(rec) + "\n")

    def close(self):
        for f in list(self._open.values()):
            f.close()
        self._open.clear()
        for f in (self._msg, self._trace, self._glob, self._truth, self._events):
            f.close()

    def write_json(self, name: str, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
