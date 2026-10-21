"""P1.1 - Environment diagnostic: can Python start and cleanly close a minimal SUMO process?"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_PY = (3, 9)


def _check(name, fn, results):
    try:
        ok, detail = fn()
    except Exception as e:  # diagnostics must never crash
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append({"check": name, "ok": bool(ok), "detail": detail})


def run_doctor(save: str | None = None, out=print) -> int:
    res: list = []
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "executable": sys.executable}

    _check("Python >= 3.9", lambda: (sys.version_info >= REQUIRED_PY, info["python"]), res)

    def mods():
        import importlib
        miss, ver = [], {}
        for m in ("yaml", "traci", "sumolib"):
            try:
                mod = importlib.import_module(m)
                ver[m] = getattr(mod, "__version__", "?")
            except Exception:
                miss.append(m)
        info["module_versions"] = ver
        return (not miss, f"missing: {miss}" if miss else str(ver))
    _check("Python packages (pyyaml, traci, sumolib)", mods, res)

    def opt():
        try:
            import matplotlib
            return True, f"matplotlib {matplotlib.__version__} (optional, for `report`)"
        except Exception:
            return True, "matplotlib not installed (optional: pip install matplotlib)"
    _check("Optional plotting", opt, res)

    def bins():
        import sumolib
        found = {}
        for b in ("sumo", "sumo-gui", "netconvert"):
            try:
                found[b] = sumolib.checkBinary(b)
            except Exception:
                found[b] = None
        info["binaries"] = found
        missing = [k for k, v in found.items() if not v]
        # sumo-gui is only needed for --gui
        hard = [k for k in missing if k != "sumo-gui"]
        return (not hard, f"missing {missing}" if missing else "sumo, sumo-gui, netconvert found")
    _check("SUMO binaries", bins, res)

    def ver():
        import sumolib
        out_ = subprocess.run([sumolib.checkBinary("sumo"), "--version"], capture_output=True, text=True).stdout
        v = out_.strip().splitlines()[0] if out_.strip() else "?"
        info["sumo_version"] = v
        return ("Eclipse SUMO" in v or "SUMO" in v), v
    _check("SUMO version", ver, res)

    def smoke():
        import sumolib, traci
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "n.nod.xml").write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="200" y="0"/></nodes>')
            (d / "e.edg.xml").write_text('<edges><edge id="ab" from="a" to="b" numLanes="1" speed="13.9"/></edges>')
            nc = sumolib.checkBinary("netconvert")
            r = subprocess.run([nc, "-n", str(d / "n.nod.xml"), "-e", str(d / "e.edg.xml"), "-o", str(d / "n.net.xml")],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return False, "netconvert failed: " + r.stderr[-200:]
            (d / "r.rou.xml").write_text('<routes><route id="r" edges="ab"/>'
                                         '<vehicle id="v" route="r" depart="0"/></routes>')
            traci.start([sumolib.checkBinary("sumo"), "-n", str(d / "n.net.xml"), "-r", str(d / "r.rou.xml"),
                         "--no-step-log", "true", "--end", "5"], label="doctor")
            c = traci.getConnection("doctor")
            for _ in range(3):
                c.simulationStep()
            n = c.vehicle.getIDCount()
            c.close()
            return n == 1, f"started, stepped 3 s, saw {n} vehicle, closed cleanly"
    _check("Start / step / close a minimal SUMO process via TraCI", smoke, res)

    def gui():
        if os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return True, "no display detected - fine for headless; --gui needs a desktop session"
        return True, "display available"
    _check("GUI availability (informational)", gui, res)

    out("TRAF-MA environment diagnostic")
    for r in res:
        out(f"  [{'OK' if r['ok'] else 'FAIL':4}] {r['check']}: {r['detail']}")
    ok = all(r["ok"] for r in res)
    if not ok:
        out("\nFix hints:\n  pip install eclipse-sumo traci sumolib pyyaml   (works on Windows, Linux, macOS)\n"
            "  then re-run: python -m trafma doctor")
    out(f"\nRESULT: {'READY' if ok else 'NOT READY'}")
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        Path(save).write_text(json.dumps({"info": info, "checks": res}, indent=1), encoding="utf-8")
        out(f"saved {save}")
    return 0 if ok else 1
