#!/usr/bin/env python3
"""
OpenAI Quality Filter

Filter prompts using OpenAI API to rate multiple quality dimensions.
Supports threshold-based filtering and cost control via sample limits.
"""

import json
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from tqdm import tqdm


RATING_PROMPT_TEMPLATE = """### Task
Carefully analyze the user's input prompt, rate it according to the detailed scoring criteria provided below for each dimension. You must return ONLY a single JSON object as the output.

### Dimensions to Rate (0-5)
- **clarity**: How clear, specific, and unambiguous the prompt is.
- **completeness**: How self-contained and whole the prompt is, without needing prior context.
- **naturalness**: How much the prompt resembles a natural, conversational human inquiry.
- **safety**: The prompt's alignment with general safety and content policies.
- **openness**: How much the prompt invites diverse, stylistically varied, or exploratory responses instead of a single factual answer.

### Detailed Scoring Criteria for Dimensions

**Clarity**
- **5 (Excellent)**: The prompt is perfectly clear, specific, and unambiguous. All constraints, context, and desired outputs are explicitly defined. There is no room for misinterpretation.
- **3 (Good)**: The prompt is generally clear but may have minor ambiguities or leave some details implicit. The model can likely infer the user's intent but might need to make reasonable assumptions.
- **1 (Poor)**: The prompt is vague, confusing, or contains contradictions. The main intent is difficult to discern, requiring significant guesswork.
- **0 (Unusable)**: The prompt is completely incomprehensible, nonsensical, or fails to state any coherent request.

**Completeness**
- **5 (Excellent)**: The prompt is a self-contained, complete question or instruction that can be fully understood without any prior context.
- **3 (Good)**: The prompt expresses a mostly complete thought but might be slightly abrupt or assume minor, easily inferable context.
- **1 (Poor)**: The prompt is clearly a fragment, a follow-up to a previous unstated conversation, or a sentence that cuts off mid-thought.
- **0 (Unusable)**: The prompt is so fragmented it conveys no discernible intent.

**Naturalness**
- **5 (Excellent)**: The prompt reads exactly like a human would speak or write to another person or an AI assistant. It's conversational, fluid, and uses everyday language.
- **3 (Acceptable)**: The prompt is somewhat formal or structured but is still clearly human-written. It might contain some "prompt engineering" elements but is mostly natural.
- **1 (Poor)**: The prompt is highly robotic, templated, or just a list of keywords. It feels like it was written *for* a machine, not *to* a machine.
- **0 (Not Applicable)**: The input is not in a linguistic format (e.g., just code, a URL, or random characters).

**Safety**
- **5 (Perfectly Safe)**: The prompt is completely harmless, helpful, and aligns with positive ethical principles.
- **3 (Generally Safe)**: The prompt is benign but might touch upon sensitive topics in a neutral, academic, or factual manner. It requires careful handling but is not seeking harmful content.
- **1 (Borderline)**: The prompt is in a gray area. It may hint at promoting unsafe acts, contain subtle hate speech, or request information that could be misused, but doesn't explicitly violate policy.
- **0 (Unsafe)**: The prompt clearly violates safety policies by requesting content related to severe harm, illegal acts, hate speech, explicit violence, self-harm, etc.

**Openness**
- **5 (Highly Open)**: The prompt sets up enough context or direction to inspire multiple legitimate response styles, perspectives, or creative approaches.
- **3 (Moderately Open)**: The prompt allows some variation in response style or angle but still hints at a relatively narrow range of answers.
- **1 (Low Openness)**: The prompt is a straightforward factual question or command that expects a single, brief answer.
- **0 (Not Applicable)**: The prompt contains no usable request or is devoid of semantic content.

### Scoring Summary
- Each dimension is an integer from 0 to 5.
- Provide an `overall` integer score (0-5). This should be a holistic assessment, often leaning towards the lowest score among the key dimensions (clarity, safety, completeness) to be conservative.

### Output JSON Schema
{{
  "clarity": int,
  "completeness": int,
  "naturalness": int,
  "safety": int,
  "openness": int,
  "overall": int,
  "reason": "A concise explanation for the scores, referencing the criteria above."
}}

### Input Prompt
{prompt}

### IMPORTANT
- Your output must be ONLY the JSON object, with no additional text or explanations before or after it."""


@dataclass
class FilterConfig:
    """Configuration for quality filtering."""
    input_file: str
    output_file: str
    prompt_field: str = "prompt"
    
    # Thresholds for each dimension (None means no threshold)
    min_clarity: Optional[int] = None
    min_completeness: Optional[int] = None
    min_naturalness: Optional[int] = None
    min_safety: Optional[int] = None
    min_openness: Optional[int] = None
    min_overall: Optional[int] = None
    
    # Cost control
    limit: int = 0  # 0 means no limit
    
    # OpenAI API settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 500
    max_workers: int = 1  # Number of parallel threads for API calls


class OpenAIQualityFilter:
    """Filter prompts using OpenAI API quality ratings."""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAIQualityFilter requires api_key or OPENAI_API_KEY."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            max_retries=0
        )
        self.passed_count = 0
        self.total_count = 0
        self.api_call_count = 0
        self.lock = Lock()  # Thread-safe counter updates
        
    def rate_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call OpenAI API to rate a prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "user", "content": RATING_PROMPT_TEMPLATE.format(prompt=prompt)}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            with self.lock:
                self.api_call_count += 1
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"\n⚠️  API error: {e}")
            return None
    
    def process_item(self, item: Dict[str, Any], index: int) -> Optional[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Process a single item (for parallel execution)."""
        prompt = item.get(self.config.prompt_field, "")
        if not prompt:
            return None
        
        rating = self.rate_prompt(prompt)
        if rating is None:
            return None
        
        if self.passes_filters(rating):
            output_item = {**item, "_rating": rating}
            return (index, output_item, rating)
        
        return None
    
    def passes_filters(self, rating: Dict[str, Any]) -> bool:
        """Check if a rating passes all configured filters."""
        # Check dimension thresholds
        if self.config.min_clarity is not None and rating.get("clarity", 0) < self.config.min_clarity:
            return False
        if self.config.min_completeness is not None and rating.get("completeness", 0) < self.config.min_completeness:
            return False
        if self.config.min_naturalness is not None and rating.get("naturalness", 0) < self.config.min_naturalness:
            return False
        if self.config.min_safety is not None and rating.get("safety", 0) < self.config.min_safety:
            return False
        if self.config.min_openness is not None and rating.get("openness", 0) < self.config.min_openness:
            return False
        if self.config.min_overall is not None and rating.get("overall", 0) < self.config.min_overall:
            return False
        
        return True
    
    def filter_dataset(self):
        """Filter the dataset and save results."""
        input_path = Path(self.config.input_file)
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read input data
        print(f"📖 Reading input from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        print(f"   Total samples: {len(data)}")
        print(f"   Prompt field: '{self.config.prompt_field}'")
        print()
        
        # Display filter configuration
        print("🔧 Filter Configuration:")
        if self.config.min_clarity is not None:
            print(f"   min_clarity: {self.config.min_clarity}")
        if self.config.min_completeness is not None:
            print(f"   min_completeness: {self.config.min_completeness}")
        if self.config.min_naturalness is not None:
            print(f"   min_naturalness: {self.config.min_naturalness}")
        if self.config.min_safety is not None:
            print(f"   min_safety: {self.config.min_safety}")
        if self.config.min_openness is not None:
            print(f"   min_openness: {self.config.min_openness}")
        if self.config.min_overall is not None:
            print(f"   min_overall: {self.config.min_overall}")
        
        print(f"   limit: {self.config.limit if self.config.limit > 0 else 'unlimited'}")
        print(f"   model: {self.config.model}")
        if self.config.base_url:
            print(f"   base_url: {self.config.base_url}")
        print(f"   max_workers: {self.config.max_workers}")
        print()
        
        # Process data with multi-threading
        has_limit = self.config.limit > 0
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit initial batch
                futures = {}
                data_index = 0
                
                # Submit first batch of tasks
                while data_index < len(data) and (not has_limit or len(futures) < self.config.limit * 2):
                    future = executor.submit(self.process_item, data[data_index], data_index)
                    futures[future] = data_index
                    data_index += 1
                
                progress_bar = tqdm(total=len(data), desc="Filtering", unit="sample")
                
                # Process results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    
                    with self.lock:
                        self.total_count += 1
                        
                        if result is not None:
                            index, output_item, rating = result
                            out_f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                            out_f.flush()
                            self.passed_count += 1
                    
                    # Update progress bar
                    pass_rate = (self.passed_count / self.total_count * 100) if self.total_count > 0 else 0
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "passed": self.passed_count,
                        "rate": f"{pass_rate:.1f}%",
                        "api_calls": self.api_call_count
                    })
                    
                    # Check if we've reached the limit
                    if has_limit and self.passed_count >= self.config.limit:
                        print(f"\n✅ Reached limit of {self.config.limit} passed samples. Stopping.")
                        # Cancel remaining futures
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                    
                    # Submit next item if available
                    if data_index < len(data):
                        future = executor.submit(self.process_item, data[data_index], data_index)
                        futures[future] = data_index
                        data_index += 1
                
                progress_bar.close()
        
        # Print summary
        print()
        print("=" * 70)
        print("📊 Filtering Summary")
        print("=" * 70)
        print(f"Total processed:  {self.total_count:,}")
        print(f"Passed filters:   {self.passed_count:,}")
        print(f"Filtered out:     {self.total_count - self.passed_count:,}")
        print(f"Pass rate:        {(self.passed_count / self.total_count * 100):.2f}%" if self.total_count > 0 else "N/A")
        print(f"API calls:        {self.api_call_count:,}")
        print(f"Output saved to:  {output_path}")
        print("=" * 70)


def main():
    """Example usage."""
    # Configuration example
    config = FilterConfig(
        input_file="./non-reasoning-filtered.jsonl",
        output_file="./non-reasoning-filtered-quality.jsonl",
        prompt_field="prompt",
        
        # Set your thresholds here (None means no filter)
        min_clarity=4,
        min_completeness=4,
        min_naturalness=4,
        min_safety=4,
        min_openness=4,
        min_overall=4,
        
        # Limit (0 = no limit)
        limit=0,  # Stop after 100 passed samples
        
        # OpenAI settings
        api_key=None,  # Uses OPENAI_API_KEY env var if not set
        base_url=os.getenv("OPENAI_BASE_URL"),  # Optional custom base URL
        model="deepseek-chat",
        temperature=0.0,
        max_workers=128,  # Number of parallel threads
    )
    
    filter_obj = OpenAIQualityFilter(config)
    filter_obj.filter_dataset()


if __name__ == "__main__":
    main()
