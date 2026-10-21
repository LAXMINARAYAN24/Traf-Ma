"""python -m trafma <command>   (doctor | build | simulate | collect | replay | gate | report)"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config, parse_set_overrides


def _cfg_from_args(a) -> "Config":
    ov = parse_set_overrides(getattr(a, "set", []))
    run = ov.setdefault("run", {})
    for flag, key in (("vehicles", "vehicles"), ("seed", "seed"), ("duration", "duration_s"),
                      ("run_id", "run_id"), ("output_dir", "output_dir")):
        v = getattr(a, flag, None)
        if v is not None:
            run[key] = v
    if getattr(a, "until_empty", False):
        run["stop_when_empty"] = True
    if getattr(a, "gui", False):
        run["gui"] = True
    if getattr(a, "profile", None):
        ov.setdefault("demand", {})["profile"] = a.profile
    return load_config(a.config, ov)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trafma", description="TRAF-MA SUMO simulation layer")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check Python/SUMO/TraCI and run a minimal SUMO process")
    d.add_argument("--save", help="write the environment report to this JSON path")

    def common(sp):
        sp.add_argument("--config", help="YAML config (default: built-in defaults)")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                        help="override any config value, e.g. agents.rsu_zone_m=120")

    b = sub.add_parser("build", help="generate network + routes + sumocfg only (no simulation)")
    common(b)
    b.add_argument("--vehicles", type=int); b.add_argument("--seed", type=int)
    b.add_argument("--profile", choices=["balanced", "asymmetric", "custom"])
    b.add_argument("--out", default="build_out")

    s = sub.add_parser("simulate", help="run one intersection simulation")
    common(s)
    s.add_argument("--vehicles", type=int, help="N vehicle agents")
    s.add_argument("--seed", type=int)
    s.add_argument("--duration", type=float)
    s.add_argument("--gui", action="store_true", help="open sumo-gui")
    s.add_argument("--until-empty", action="store_true", help="stop once every vehicle has arrived")
    s.add_argument("--profile", choices=["balanced", "asymmetric", "custom"])
    s.add_argument("--run-id"); s.add_argument("--output-dir")
    s.add_argument("--quiet", action="store_true")

    c = sub.add_parser("collect", help="P2.1: generate labelled multi-scenario data from SUMO")
    c.add_argument("--config", required=True)
    c.add_argument("--resume", action="store_true", help="skip runs that already have a manifest")
    c.add_argument("--jobs", type=int)

    r = sub.add_parser("replay", help="replay a saved run from its logs (no SUMO needed)")
    r.add_argument("--run", required=True)
    r.add_argument("--from", dest="t_from", type=float, default=0.0)
    r.add_argument("--to", dest="t_to", type=float)
    r.add_argument("--every", type=float, default=10.0)
    r.add_argument("--verify", action="store_true", help="re-run SUMO and compare the deterministic trace")

    g = sub.add_parser("gate", help="check the Phase 1 gate against a saved run")
    g.add_argument("--run", required=True)
    g.add_argument("--reproduce", action="store_true")
    g.add_argument("--pytest", action="store_true")

    rp = sub.add_parser("report", help="write layout.png and queues.png for a run (needs matplotlib)")
    rp.add_argument("--run", required=True)
    rp.add_argument("--at", type=float, help="time for the vehicle snapshot")

    a = p.parse_args(argv)
    try:
        if a.cmd == "doctor":
            from .doctor import run_doctor
            return run_doctor(a.save)
        if a.cmd == "build":
            from .demand import generate_demand, write_routes
            from .netgen import build_network, write_sumocfg
            cfg = _cfg_from_args(a)
            out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
            net, topo, add, gs = build_network(cfg, out)
            routes = write_routes(generate_demand(cfg), cfg, out / "routes.rou.xml")
            print(f"built {write_sumocfg(cfg, out, net, routes, add, gs)}")
            return 0
        if a.cmd == "simulate":
            from .runner import run_simulation
            res = run_simulation(_cfg_from_args(a), quiet=a.quiet)
            print(f"\nartifacts: {res.run_dir}")
            return 0 if res.status in ("completed", "sim_timeout") else 2
        if a.cmd == "collect":
            from .collect import collect
            collect(a.config, resume=a.resume, jobs=a.jobs)
            return 0
        if a.cmd == "replay":
            from .replay import replay_run
            r_ = replay_run(a.run, a.t_from, a.t_to, a.every, a.verify)
            return 0 if (r_ is None or r_["identical"]) else 3
        if a.cmd == "gate":
            from .gate import run_gate, format_gate
            res = run_gate(a.run, a.reproduce, a.pytest)
            print(format_gate(res))
            return 1 if any(x[1] == "FAIL" for x in res) else 0
        if a.cmd == "report":
            from .report import make_report
            for q in make_report(a.run, a.at):
                print("wrote", q)
            return 0
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 64
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
