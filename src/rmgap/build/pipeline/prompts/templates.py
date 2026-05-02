# -*- coding: utf-8 -*-

# —— 1. Response Generation Prompt ——
# Goal: Generate a single stylistically controlled response to an original prompt.
nuanced_writer_prompt = """
You are an expert writer. Your task is to generate one response to a given prompt, strictly following a provided style profile.

**Original Prompt:**
`{{ORIGINAL_PROMPT}}`

**1. Style Profile**
You will be given exactly one style profile. You MUST follow it strictly.
`{{STYLE_PROFILE}}`

**Style Dimensions:**
- **Formality**: casual (colloquial, contractions); informal (lightly casual); neutral (plain professional); formal (polished, precise); highly_formal (ceremonial, meticulous).
- **Conciseness**: terse (very brief, minimal modifiers); concise (short, to the point); moderate (balanced detail); detailed (thorough explanations); verbose (extensive elaboration).
- **Technicality**: layman (everyday terms); accessible (light technical terms with explanations); semi_technical (moderate jargon); technical (domain jargon, assumes knowledge); highly_specialized (dense domain terminology).
- **Objectivity**: highly_subjective (opinions/feelings); subjective (some value-laden phrasing); balanced (mix of facts and qualified opinions); objective (primarily factual); strictly_objective (facts only, no hedging).
- **Structural_Coherence**: fragmented (disjoint snippets); loose (light organization); organized (clear sections); well_structured (explicit headings/flow); rigorous (precise structure, stepwise logic).

**2. Generation Rules**
- **Preserve Core Meaning**: Your response must convey the same essential information that the original prompt requires.
- **Prompt Adherence**: The response must fully and accurately answer the Original Prompt.
- **Style Obedience**: The response must reflect the given style profile along all dimensions (Formality, Conciseness, Technicality, Objectivity, Structural_Coherence).

**3. Output Format**
{{FORMAT_REQUIREMENTS}}
Write a single response in natural language that directly answers the Original Prompt.
Avoid adding meta-commentary about the instructions or any unrelated content.
"""

# —— 2. Pointwise Evaluation Prompt ——
# Goal: Independently score the four generated responses on quality, style diversity contribution, and semantic consistency.
pair_evaluator_prompt = """
You are an expert evaluator. Your task is to score four responses to a given prompt.

**Original Prompt:**
`{{PROMPT}}`

**Responses to Evaluate:**
- **Response 1:** `{{R1}}`
- **Response 2:** `{{R2}}`
- **Response 3:** `{{R3}}`
- **Response 4:** `{{R4}}`

**Evaluation Criteria & Instructions**
You must provide three types of scores:
- Per-response quality scores (one score for each response).
- A single global style diversity score for the batch of four responses.
- A single global semantic consistency score for the batch of four responses.

All scores must be integers from 1 to 10.

1. **Per-response Quality Score (1–10)**  
   For each response, assess its adherence to the prompt, including correctness, completeness, clarity, and factual accuracy.
   - **1–3 (Very Poor)**: Major misunderstandings of the prompt; severe factual errors; largely fails to address the task.
   - **4–6 (Fair)**: Partially answers the prompt but has important omissions, unclear reasoning, or notable factual issues.
   - **7–8 (Good)**: Largely correct and complete, with clear and coherent writing; only minor issues or small gaps.
   - **9–10 (Excellent)**: Fully correct, complete, insightful, and very clearly written, with no significant problems.

2. **Global Style Diversity Score (1–10)**  
   This is a single score for the entire batch of four responses. Measure how stylistically diverse the four responses are as a set (tone, structure, length, level of detail, technicality, etc.), while all still being appropriate answers to the prompt.
   - **1–3 (Very Low Diversity)**: The four responses are stylistically almost identical; they read like small edits of the same text.
   - **4–6 (Moderate Diversity)**: Some noticeable differences in tone, structure, or length, but overall the responses still feel quite similar.
   - **7–8 (High Diversity)**: Clear, substantial stylistic differences across multiple dimensions; each response feels meaningfully distinct.
   - **9–10 (Very High Diversity)**: The four responses exhibit very strong and coherent stylistic contrasts while all remaining good answers to the prompt.

3. **Global Semantic Consistency Score (1–10)**  
   This is a single score for the entire batch of four responses. Measure how closely the core meaning of the four responses aligns with each other.
   - **1–3 (Strong Inconsistency)**: The responses clearly disagree or diverge on core claims, conclusions, or key facts.
   - **4–6 (Partial Consistency)**: Some shared core ideas, but there are notable contradictions, omissions, or shifts in the main conclusions.
   - **7–8 (Good Consistency)**: The responses largely agree on the key points and conclusions; differences are mostly in emphasis, ordering, or additional details.
   - **9–10 (Near-Perfect Consistency)**: The responses express almost the same core meaning; differences are only in wording or minor nuances.

**Output Format**
Respond with a single, valid JSON object. Do not add any text before or after the JSON.
{
  "quality_scores": {"r1": <int>, "r2": <int>, "r3": <int>, "r4": <int>},
  "style_score": <int>,
  "semantic_score": <int>
}
"""


# —— 3. Reverse Prompt Generation ——
# Goal: Given four responses, craft one natural-language prompt that favors a specified target response.
reverse_prompt_generator = """
You are an expert prompt engineer. Given four stylistic responses to an original prompt, write one realistic user prompt that naturally favors a specified target response without explicitly naming style traits.

**Original Prompt:**
`{{ORIGINAL_PROMPT}}`

**Target Response (preferred):** R{{TARGET_ID}}

**Reference Responses (for context only):**
- R1: `{{R1}}`
- R2: `{{R2}}`
- R3: `{{R3}}`
- R4: `{{R4}}`

**Prompt Generation Rules:**
1. **Scenario-Based Bias (Critical)**  
   Design a specific user scenario, downstream task, or persona where the target response R{{TARGET_ID}} is the only logical fit. Do not use explicit style labels like "formal" or "concise".
2. **Hardness Injection**  
   The prompt must be a realistic query with enough detail to create a strong but implicit preference for the target response.
3. **Content Alignment**  
   Keep the request aligned with the original prompt’s intent while steering toward the target response’s treatment.

**Output**
Write exactly one natural-language user prompt. Do not include JSON, bullet lists, or annotations.
"""


# —— 4. Reverse Prompts Evaluation ——
# Goal: Pointwise evaluation of four prompts that each prefer one of the four responses.
reverse_prompts_evaluator = """
You are an expert evaluator. Your task is to evaluate four prompts that are each designed to prefer one of four responses.

**Input Responses:**
- **R1:** `{{R1}}`
- **R2:** `{{R2}}`
- **R3:** `{{R3}}`
- **R4:** `{{R4}}`

**Prompts to Evaluate (each prompt is designed to prefer the corresponding response):**
- **P1 (prefers R1):** `{{P1}}`
- **P2 (prefers R2):** `{{P2}}`
- **P3 (prefers R3):** `{{P3}}`
- **P4 (prefers R4):** `{{P4}}`

**1. Evaluation Criteria & Instructions**
For each prompt Pi, provide three scalar scores from 1 to 10 (integers only):

1. **Prompt Quality Score (1–10)**  
   Evaluate how clear, well-posed, and unambiguous the prompt is, and how well it specifies the task.
   - **1–3 (Very Poor)**: Vague, confusing, or badly phrased; the model would struggle to understand what to do.
   - **4–6 (Fair)**: Roughly understandable but with ambiguity, missing conditions, or awkward phrasing.
   - **7–8 (Good)**: Clear and well-structured; the model can reliably follow it, with only minor room for improvement.
   - **9–10 (Excellent)**: Very clear, precise, and natural; an ideal instruction for the intended task.

2. **Bias Effectiveness Score (1–10)**  
   Evaluate how effectively the prompt is biased toward its intended winner response (Pi prefers Ri).
   - **1–3 (Low Bias Effectiveness)**: Little or no preference toward the winner; could easily lead to answers similar to other responses.
   - **4–6 (Moderate Bias Effectiveness)**: Some preference is present but weak, ambiguous, or easily overshadowed.
   - **7–8 (High Bias Effectiveness)**: Clearly and logically biases the model toward the winner’s style and perspective.
   - **9–10 (Very High Bias Effectiveness)**: Strong, coherent bias toward the winner, while still natural and reasonable.

3. **Semantic Alignment Score (1–10)**  
   Evaluate how well the prompt is likely to elicit the core semantic content of its associated winner response Ri (not just the style).
   - **1–3 (Poor Alignment)**: Unlikely to elicit the core ideas or conclusions of the winner response.
   - **4–6 (Partial Alignment)**: Captures some core ideas but may miss or distort important aspects.
   - **7–8 (Good Alignment)**: Likely to elicit answers that share the main conclusions and key facts with the winner.
   - **9–10 (Excellent Alignment)**: Very likely to elicit responses with almost the same core meaning as the winner.

**2. Output Format**
Respond with a single, valid JSON object. Do not add any text before or after the JSON.
{
  "quality_scores": {"p1": <int>, "p2": <int>, "p3": <int>, "p4": <int>},
  "bias_scores": {"p1": <int>, "p2": <int>, "p3": <int>, "p4": <int>},
  "semantic_scores": {"p1": <int>, "p2": <int>, "p3": <int>, "p4": <int>}
}
"""


# —— 5. Rewrite Prompt Generation ——
# Goal: Generate one paraphrased variant per call while preserving meaning.
rewrite_prompt_generator = """
You are an expert paraphraser. Rewrite the following prompt into one paraphrased variant.

**Prompt:**
`{{BASE_PROMPT}}`

Rewrite Rules:
- Preserve the same meaning, intent, and constraints as the original prompt.
- Do not add new requirements or remove existing ones.
- Clearly differ in surface wording and structure (reorder clauses, change wording, vary connectors).
- Do NOT copy long spans verbatim from the original prompt.

**Output**
Write exactly one paraphrased prompt as plain text. Do not return JSON or bullet lists.
"""