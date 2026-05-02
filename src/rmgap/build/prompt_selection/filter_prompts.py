"""Filter prompts using regex, exact deduplication, and semantic deduplication."""

import argparse
import json
import os
import re
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
from typing import List, Callable, Dict

# ========== Configuration ==========
INPUT_FILE = "filter/prompts.jsonl"
OUTPUT_FILE_CHAT = "filter/prompts-filtered-chat.jsonl"
OUTPUT_FILE_REASONING = "filter/prompts-filtered-reasoning.jsonl"
OUTPUT_FILE_SAFETY = "filter/prompts-filtered-safety.jsonl"
OUTPUT_FILE_WRITING = "filter/prompts-filtered-writing.jsonl"

MODEL_PATH_ENV = "RMGAP_EMBEDDING_MODEL_PATH"
DEFAULT_MODEL_PATH = "model/embeddinggemma-300m"

ENCODE_BATCH_SIZE = 64
COMPARE_BATCH_SIZE = 1024
SIMILARITY_THRESHOLD = 0.7

# Regex patterns
PERSONA_REGEX = r"(?i).*?\b(you are|you're|imagine|take\s+\w+(?:\s+\w+)*\s+role)\b"
ENGLISH_REGEX = r"^[\x00-\x7F\u00B0\u0370-\u03FF\u2070-\u209F\u2100-\u214F\u2150-\u218F\u2200-\u22FF\u2A00-\u2AFF]+$"


# ========== Data Loading ==========
def load_prompts(file_path: str) -> pd.DataFrame:
    """Load prompts from JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(data)


# ========== Regex Filtering ==========
def create_regex_filter() -> Callable[[str], bool]:
    """Create a regex filter function for prompts."""
    def filter_prompt(prompt: str) -> bool:
        """Filter out style/tone constraint and non-English prompts."""
        if re.search(PERSONA_REGEX, prompt):
            return False
        if not re.fullmatch(ENGLISH_REGEX, prompt):
            return False
        return True
    return filter_prompt


def apply_regex_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply regex filtering to remove style/tone constraints and non-English prompts."""
    filter_func = create_regex_filter()
    return df[df['prompt'].apply(filter_func)].reset_index(drop=True)


# ========== Exact Deduplication ==========
def normalize_prompt(prompt: str) -> str:
    """Normalize prompts for exact deduplication."""
    return ''.join(re.findall(r'[\w]+', prompt, flags=re.UNICODE)).lower()


def apply_exact_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates by normalizing prompts."""
    df = df.copy()
    df['_normalized_prompt'] = df['prompt'].apply(normalize_prompt)
    df = df.drop_duplicates(subset=['_normalized_prompt']).drop(columns=['_normalized_prompt'])
    return df.reset_index(drop=True)


# ========== Semantic Deduplication ==========
def get_embedding_model(model_path: str) -> SentenceTransformer:
    """Load embedding model from local directory."""
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(
            f"Model directory not found at {model_path}. "
            f"Please ensure the model is downloaded to this location."
        )
    return SentenceTransformer(str(model_path_obj))


def resolve_embedding_model_path(model_path: str | None) -> str:
    """Resolve embedding model path from CLI, environment, or repo default."""
    if model_path:
        return model_path
    env_path = os.getenv(MODEL_PATH_ENV)
    if env_path:
        return env_path
    return DEFAULT_MODEL_PATH


def generate_embeddings(
    prompts: List[str],
    model: SentenceTransformer,
    device: torch.device,
    batch_size: int
) -> torch.Tensor:
    """Generate normalized embeddings for prompts with batch processing."""
    embeddings_list = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding batches"):
        batch_prompts = prompts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_prompts,
            batch_size=batch_size,
            show_progress_bar=False,
            device=device,
            convert_to_tensor=True
        )
        batch_embeddings_normalized = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        embeddings_list.append(batch_embeddings_normalized.cpu())
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return torch.cat(embeddings_list, dim=0)


def find_semantic_duplicates(
    embeddings: torch.Tensor,
    device: torch.device,
    threshold: float,
    compare_batch_size: int
) -> List[int]:
    """Find semantic duplicates using incremental comparison."""
    selected_indices = []
    selected_embeddings = []
    
    for i in tqdm(range(len(embeddings)), desc="Deduplicating prompts"):
        current_embedding_cpu = embeddings[i:i+1]
        
        if len(selected_embeddings) == 0:
            selected_indices.append(i)
            selected_embeddings.append(current_embedding_cpu)
            continue
        
        is_duplicate = False
        num_selected = len(selected_embeddings)
        
        for batch_start in range(0, num_selected, compare_batch_size):
            batch_end = min(batch_start + compare_batch_size, num_selected)
            
            current_embedding_gpu = current_embedding_cpu.to(device)
            selected_batch_cpu = torch.cat(selected_embeddings[batch_start:batch_end], dim=0)
            selected_batch_gpu = selected_batch_cpu.to(device)
            
            similarities = torch.mm(current_embedding_gpu, selected_batch_gpu.t())
            max_similarity = similarities.max().item()
            
            del current_embedding_gpu, selected_batch_gpu, similarities
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if max_similarity > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected_indices.append(i)
            selected_embeddings.append(current_embedding_cpu)
    
    return selected_indices


def apply_semantic_deduplication(
    df: pd.DataFrame,
    model_path: str | None = None,
    encode_batch_size: int = ENCODE_BATCH_SIZE,
    compare_batch_size: int = COMPARE_BATCH_SIZE,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> pd.DataFrame:
    """Remove semantically similar prompts using embedding-based comparison."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = get_embedding_model(resolve_embedding_model_path(model_path))
    model.to(device)
    
    prompts = df['prompt'].tolist()
    print("Generating embeddings (batch processing to save GPU memory)...")
    
    embeddings = generate_embeddings(prompts, model, device, encode_batch_size)
    print(f"All embeddings shape: {embeddings.shape}, stored in CPU memory")
    
    print("Deduplicating semantically (incremental comparison)...")
    selected_indices = find_semantic_duplicates(
        embeddings, device, similarity_threshold, compare_batch_size
    )
    
    result_df = df.iloc[selected_indices].reset_index(drop=True)
    
    # Free memory
    del embeddings
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return result_df


# ========== Output ==========
def save_prompts(df: pd.DataFrame, output_file: str) -> None:
    """Save filtered prompts to JSONL file."""
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    print(f"✓ Filtered prompts saved to: {output_file}")


def print_statistics(df: pd.DataFrame) -> None:
    """Print filtering statistics."""
    print("\nFiltering Summary:")
    print(f"  Total prompts after all filtering: {df.shape[0]}")
    print(f"\nBreakdown by domain:")
    
    for domain in sorted(df['domain'].unique()):
        domain_count = (df['domain'] == domain).sum()
        print(f"  {domain}: {domain_count} prompts")
        
        domain_df = df[df['domain'] == domain]
        for source in sorted(domain_df['source'].unique()):
            source_count = (domain_df['source'] == source).sum()
            print(f"    - {source}: {source_count} prompts")


def split_by_domain(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split DataFrame into subsets by domain."""
    domain_dfs = {}
    for domain in sorted(df['domain'].unique()):
        domain_df = df[df['domain'] == domain].copy().reset_index(drop=True)
        domain_dfs[domain] = domain_df
    return domain_dfs


def print_step_header(step_name: str) -> None:
    """Print step header."""
    print("\n" + "="*60)
    print(step_name)
    print("="*60)


# ========== Main Pipeline ==========
def filter_prompts(
    input_file: str = INPUT_FILE,
    output_file_chat: str = OUTPUT_FILE_CHAT,
    output_file_reasoning: str = OUTPUT_FILE_REASONING,
    output_file_safety: str = OUTPUT_FILE_SAFETY,
    output_file_writing: str = OUTPUT_FILE_WRITING,
    model_path: str | None = None,
) -> pd.DataFrame:
    """Main filtering pipeline: load -> regex filter -> exact dedup -> semantic dedup."""
    # Load data
    print_step_header("Step 0: Load Data")
    df = load_prompts(input_file)
    print(f"Original data: {df.shape[0]} prompts")
    
    # Regex filtering
    print_step_header("Step 1: Regex Filtering")
    df = apply_regex_filter(df)
    print(f"After regex filtering: {df.shape[0]} prompts")
    
    # Exact deduplication
    print_step_header("Step 2: Exact Deduplication")
    df = apply_exact_deduplication(df)
    print(f"After exact deduplication: {df.shape[0]} prompts")
    
    # Semantic deduplication
    print_step_header("Step 3: Semantic Deduplication")
    df = apply_semantic_deduplication(df, model_path=model_path)
    print(f"After semantic deduplication: {df.shape[0]} prompts")
    
    # Split by domain
    print_step_header("Step 4: Splitting by Domain")
    domain_dfs = split_by_domain(df)
    output_files = {
        'Chat': output_file_chat,
        'Reasoning': output_file_reasoning,
        'Safety': output_file_safety,
        'Writing': output_file_writing
    }
    
    for domain in sorted(domain_dfs.keys()):
        count = domain_dfs[domain].shape[0]
        print(f"  {domain}: {count} prompts")
    
    # Save results
    print_step_header("Saving Results")
    for domain, domain_df in domain_dfs.items():
        if domain in output_files:
            save_prompts(domain_df, output_files[domain])
        else:
            print(f"⚠ Warning: No output file configured for domain '{domain}', skipping...")
    
    print_statistics(df)
    
    print("\n" + "="*60)
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description="Filter merged RMGAP prompts.")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-chat", default=OUTPUT_FILE_CHAT)
    parser.add_argument("--output-reasoning", default=OUTPUT_FILE_REASONING)
    parser.add_argument("--output-safety", default=OUTPUT_FILE_SAFETY)
    parser.add_argument("--output-writing", default=OUTPUT_FILE_WRITING)
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Embedding model path. Falls back to {MODEL_PATH_ENV}.",
    )
    return parser


def main():
    """Main execution function."""
    args = build_arg_parser().parse_args()
    filter_prompts(
        input_file=args.input,
        output_file_chat=args.output_chat,
        output_file_reasoning=args.output_reasoning,
        output_file_safety=args.output_safety,
        output_file_writing=args.output_writing,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
