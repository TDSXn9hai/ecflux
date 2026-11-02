"""脚本公共入口：把项目根目录加入 sys.path，解析 --config / --set。"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ecflux.utils import load_config, save_json, seed_everything, setup_logging  # noqa: E402


def base_parser(desc: str, default_config: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=default_config, help="YAML 配置文件")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="覆盖配置，如 apra.rounds=3")
    p.add_argument("--out", default=None, help="输出目录")
    return p


def init(args) -> dict:
    setup_logging()
    cfg = load_config(args.config, args.set) if args.config else {}
    seed_everything(int(cfg.get("seed", 42)))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        save_json({"config": cfg, "argv": sys.argv}, os.path.join(args.out, "run_config.json"))
    return cfg
