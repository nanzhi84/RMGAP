"""Merge prompts from multiple sources into a unified JSONL file."""

import argparse
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Iterator, List, Optional

DEFAULT_SOURCE_BASE = Path("source")
DEFAULT_OUTPUT_FILE = Path("filter") / "prompts.jsonl"
DOMAINS = ["Chat", "Reasoning", "Safety", "Writing"]

# Key name mapping: source_name -> key_name_in_json
KEY_MAPPING = {
    "math-500": "problem",
    "Writing-Bench": "query"
}

# Eval metadata extractors by source
EVAL_EXTRACTORS = {
    "math-500": lambda d: {
        "type": "math",
        "answer": d.get("answer"),
    },
    "openai_humaneval": lambda d: {
        "type": "code",
        "entry_point": d.get("entry_point"),
        "tests": d.get("test"),
    }
}


def get_prompt_key(source: str) -> str:
    """Get the prompt key name for a given source."""
    return KEY_MAPPING.get(source, "prompt")


def generate_record_id(source: str, prompt: str) -> str:
    """Generate SHA256 hash ID for a record."""
    return hashlib.sha256(f"{source}|{prompt}".encode("utf-8")).hexdigest()


def build_record(data: Dict, prompt_key: str, domain: str, source: str) -> Optional[Dict]:
    """Build a unified record from data dictionary."""
    prompt_value = data.get(prompt_key)
    if not prompt_value:
        return None
    
    record = {
        "id": generate_record_id(source, prompt_value),
        "prompt": prompt_value,
        "domain": domain,
        "source": source,
    }
    
    # Attach eval metadata for Reasoning domain
    if domain == "Reasoning" and source in EVAL_EXTRACTORS:
        eval_data = EVAL_EXTRACTORS[source](data)
        # Only add if has required fields for verification
        if source == "math-500" and eval_data.get("answer"):
            record["eval"] = eval_data
        elif source == "openai_humaneval" and eval_data.get("entry_point") and eval_data.get("tests"):
            record["eval"] = eval_data
    
    return record


def load_jsonl_records(file_path: Path) -> Iterator[Dict]:
    """Load records from a JSONL file (each line is a JSON object)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    yield data
            except json.JSONDecodeError:
                continue


def collect_all_prompts(source_base: Path) -> List[Dict]:
    """Collect all prompts from all domains and sources."""
    all_records = []

    for domain in DOMAINS:
        domain_path = source_base / domain
        if not domain_path.exists():
            continue
            
        print(f"Processing domain: {domain}")
        
        # Iterate through source directories (domain/source/)
        for source_name in os.listdir(domain_path):
            source_path = domain_path / source_name
            if not source_path.is_dir():
                continue
                
            print(f"  Processing source: {source_name}")
            prompt_key = get_prompt_key(source_name)
            
            # Process all .jsonl files in source directory
            records = []
            for file_path in source_path.glob("*.jsonl"):
                for data in load_jsonl_records(file_path):
                    record = build_record(data, prompt_key, domain, source_name)
                    if record is not None:
                        records.append(record)
            
            all_records.extend(records)
            print(f"    Added {len(records)} prompts from {source_name}")
    
    return all_records


def save_records(records: List[Dict], output_file: Path) -> None:
    """Save records to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def print_statistics(records: List[Dict], output_file: Path) -> None:
    """Print statistics about collected records."""
    print(f"\n✓ Successfully merged {len(records)} prompts")
    print(f"✓ Saved to: {output_file}")
    
    print(f"\nBreakdown by domain:")
    for domain in DOMAINS:
        count = sum(1 for r in records if r['domain'] == domain)
        print(f"  - {domain}: {count} prompts")
    
    print(f"\nBreakdown by source:")
    sources = sorted(set(r['source'] for r in records))
    for source in sources:
        count = sum(1 for r in records if r['source'] == source)
        print(f"  - {source}: {count} prompts")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description="Merge source prompts.")
    parser.add_argument(
        "--source-base",
        type=Path,
        default=DEFAULT_SOURCE_BASE,
        help="Directory containing domain/source JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL file.",
    )
    return parser


def main():
    """Main execution function."""
    args = build_arg_parser().parse_args()
    records = collect_all_prompts(args.source_base)
    save_records(records, args.output)
    print_statistics(records, args.output)


if __name__ == "__main__":
    main()
