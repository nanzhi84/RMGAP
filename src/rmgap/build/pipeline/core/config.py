"""Pydantic-based configuration loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Minimal runtime configuration.

    Model- and stage-specific parameters are defined close to each stage
    implementation instead of being centralized here.
    """

    input_dir: str
    output_dir: str
    resume_dir: str | None = None
    embedding_model_path: str | None = None
    prompt_field: str = "prompt"
    limit: int = 0
    max_workers: int = 8


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt RM Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["res_gen", "res_eval", "pro_gen", "pro_eval", "rw_gen", "rw_eval", "write_test"],
        default="res_gen",
        help="Execution stage",
    )
    parser.add_argument("--input", type=str, help="Override input file path")
    parser.add_argument("--output", type=str, help="Override output directory")
    parser.add_argument("--resume", type=str, help="Resume from protocols.jsonl file path")
    parser.add_argument("--limit", type=int, help="Limit number of records to process")
    return parser


def load_config(path: str, *, args: argparse.Namespace | None = None) -> Config:
    """Load configuration from YAML file with CLI argument overrides."""
    cfg_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    config_base_dir = cfg_path.parent
    cli_base_dir = Path.cwd()

    def _resolve(p: str | None, *, base_dir: Path) -> str | None:
        """Resolve a path relative to its declaring context."""
        if p is None:
            return None
        pp = Path(p)
        if not pp.is_absolute():
            pp = (base_dir / pp).resolve()
        return str(pp)

    has_cli_input = bool(args and getattr(args, "input", None))
    has_cli_output = bool(args and getattr(args, "output", None))
    has_cli_resume = bool(args and getattr(args, "resume", None))

    input_dir_raw = args.input if has_cli_input else data.get("input_dir")
    output_dir_raw = args.output if has_cli_output else data.get("output_dir")
    resume_dir_raw = args.resume if has_cli_resume else data.get("resume_dir")
    embedding_model_path_raw = data.get("embedding_model_path")
    limit_raw = args.limit if args and getattr(args, "limit", None) is not None else data.get("limit")

    input_dir = _resolve(
        input_dir_raw,
        base_dir=cli_base_dir if has_cli_input else config_base_dir,
    )
    output_dir = _resolve(
        output_dir_raw,
        base_dir=cli_base_dir if has_cli_output else config_base_dir,
    )
    resume_dir = _resolve(
        resume_dir_raw,
        base_dir=cli_base_dir if has_cli_resume else config_base_dir,
    )
    embedding_model_path = _resolve(
        embedding_model_path_raw,
        base_dir=config_base_dir,
    )

    # Validate required fields
    if not input_dir:
        raise ValueError("input_dir is required (via config file or --input)")
    if not output_dir:
        raise ValueError("output_dir is required (via config file or --output)")

    return Config(
        input_dir=input_dir,
        output_dir=output_dir,
        resume_dir=resume_dir,
        embedding_model_path=embedding_model_path,
        prompt_field=data.get("prompt_field", "prompt"),
        limit=int(limit_raw) if limit_raw is not None else 0,
        max_workers=int(data.get("max_workers", 8)),
    )
