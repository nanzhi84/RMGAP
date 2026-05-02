from __future__ import annotations

import logging
from typing import Sequence

from rmgap.build.pipeline.core.config import build_arg_parser, load_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    cfg = load_config(args.config, args=args)
    from rmgap.build.pipeline.core.orchestrator import run

    run(cfg, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
