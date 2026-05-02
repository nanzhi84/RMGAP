import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru._logger import Logger
from omegaconf import DictConfig, OmegaConf

from rmgap.rm.base import BaseRM

METRICS_FILENAME = "metrics.json"

CANONICAL_DOMAINS = ["Chat", "Writing", "Reasoning", "Safety"]
EXPECTED_RESPONSE_COUNT = 4
EXPECTED_PROMPT_GROUP_COUNT = 4
EXPECTED_PROMPTS_PER_GROUP = 3


class TaskRunner:
    def run(self, config: DictConfig, logger: Logger, rm: BaseRM):
        unified_items, unified_meta = self._load_dataset(config.data.path)

        unified_results = rm(
            model_path=config.model.path,
            sglang_cfg=config.rm.sglang_cfg,
            data=unified_items,
            **(
                OmegaConf.to_container(config.rm.params)
                if getattr(config.rm, "params", None)
                else {}
            ),
        )
        if len(unified_results) != len(unified_items):
            raise RuntimeError(
                f"RM returned {len(unified_results)} results for "
                f"{len(unified_items)} evaluation items."
            )

        pair_bon_stats = self._compute_pair_and_bon(
            unified_meta, unified_results, logger,
        )
        consistency_stats = self._compute_consistency(
            unified_meta, unified_results, logger,
        )
        metrics_by_domain = self._aggregate_metrics(pair_bon_stats, consistency_stats)

        self._report_and_save(config, logger, metrics_by_domain)

    def _load_dataset(
        self, dataset_path: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_records: List[Dict[str, Any]] = []
        with open(dataset_path, "r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                raw_records.append(json.loads(line))

        if not raw_records:
            raise ValueError("Empty dataset: no records loaded from JSONL file.")

        unified_items: List[Dict[str, Any]] = []
        unified_meta: List[Dict[str, Any]] = []

        for rec_idx, record in enumerate(raw_records):
            record_id = record.get("id", rec_idx)
            if "domain" not in record:
                raise ValueError(f"Missing 'domain' in record {record_id}.")
            domain: str = record["domain"]
            responses: List[Dict[str, Any]] = record["responses"]
            if len(responses) != EXPECTED_RESPONSE_COUNT:
                raise ValueError(
                    f"Record {record_id} must contain "
                    f"{EXPECTED_RESPONSE_COUNT} responses, got {len(responses)}."
                )
            response_key_to_text: Dict[str, str] = {}
            for resp in responses:
                if not str(resp.get("key", "")).strip():
                    raise ValueError(f"Empty response key in record {record_id}.")
                if not str(resp.get("text", "")).strip():
                    raise ValueError(
                        f"Empty response text for {resp.get('key')} "
                        f"in record {record_id}."
                    )
                response_key = str(resp["key"])
                if response_key in response_key_to_text:
                    raise ValueError(
                        f"Duplicate response key '{response_key}' "
                        f"in record {record_id}."
                    )
                response_key_to_text[response_key] = str(resp["text"])

            ordered_response_keys: List[str] = sorted(response_key_to_text.keys())
            ordered_response_texts: List[str] = [
                response_key_to_text[k] for k in ordered_response_keys
            ]

            prompt_groups: List[Dict[str, Any]] = record["prompt_groups"]
            if len(prompt_groups) != EXPECTED_PROMPT_GROUP_COUNT:
                raise ValueError(
                    f"Record {record_id} must contain "
                    f"{EXPECTED_PROMPT_GROUP_COUNT} prompt groups, "
                    f"got {len(prompt_groups)}."
                )
            for group_idx, group in enumerate(prompt_groups):
                winner_key: str = group["winner"]
                if winner_key not in response_key_to_text:
                    raise KeyError(
                        f"Winner key '{winner_key}' not found in responses "
                        f"for record {record_id}."
                    )

                prompts: List[Dict[str, Any]] = group["prompts"]
                if len(prompts) != EXPECTED_PROMPTS_PER_GROUP:
                    raise ValueError(
                        f"Prompt group {group_idx} in record {record_id} "
                        f"must contain {EXPECTED_PROMPTS_PER_GROUP} prompts, "
                        f"got {len(prompts)}."
                    )
                for prompt in prompts:
                    if not str(prompt.get("text", "")).strip():
                        raise ValueError(
                            f"Empty prompt text in group {group_idx} "
                            f"for record {record_id}."
                        )
                    unified_items.append(
                        {
                            "prompt": prompt["text"],
                            "responses": ordered_response_texts,
                        }
                    )
                    unified_meta.append(
                        {
                            "record_id": record_id,
                            "group_idx": group.get("group", group_idx),
                            "domain": domain,
                            "ordered_keys": ordered_response_keys,
                            "winner_key": winner_key,
                        }
                    )

        return unified_items, unified_meta

    def _compute_pair_and_bon(
        self,
        unified_meta: List[Dict[str, Any]],
        unified_results: List[Dict[str, Any]],
        logger: Logger,
    ) -> Dict[str, Any]:
        domain_pair_wins: Dict[str, float] = defaultdict(float)
        domain_pair_total: Dict[str, int] = defaultdict(int)
        domain_bon_flags: Dict[str, List[int]] = defaultdict(list)

        nan_count = 0
        for meta, result in zip(unified_meta, unified_results):
            scores = np.array(result["scores"], dtype=float)
            if scores.shape[0] != len(meta["ordered_keys"]):
                raise ValueError(
                    f"Score count mismatch for record {meta['record_id']}: "
                    f"expected {len(meta['ordered_keys'])}, got {scores.shape[0]}."
                )
            if np.isnan(scores).any():
                nan_count += 1
                continue
            ordered_keys = meta["ordered_keys"]
            winner_key = meta["winner_key"]
            winner_idx = ordered_keys.index(winner_key)

            winner_score = float(scores[winner_idx])
            loser_indices = [i for i, k in enumerate(ordered_keys) if k != winner_key]
            loser_scores = scores[loser_indices]

            domain = meta["domain"]
            domain_pair_wins[domain] += float(np.sum(winner_score > loser_scores))
            domain_pair_total[domain] += len(loser_indices)
            domain_bon_flags[domain].append(
                1
                if len(loser_indices) > 0 and np.all(winner_score > loser_scores)
                else 0
            )

        if nan_count > 0:
            logger.warning(
                f"Skipped {nan_count}/{len(unified_results)} samples with NaN scores "
                f"in Pair/BoN computation."
            )

        return {
            "domain_pair_wins": domain_pair_wins,
            "domain_pair_total": domain_pair_total,
            "domain_bon_flags": domain_bon_flags,
        }

    def _compute_consistency(
        self,
        unified_meta: List[Dict[str, Any]],
        unified_results: List[Dict[str, Any]],
        logger: Logger,
    ) -> Dict[str, Any]:
        domain_group_to_rankings: Dict[
            str, Dict[Tuple[Any, Any], List[List[str]]]
        ] = defaultdict(lambda: defaultdict(list))
        domain_group_expected_counts: Dict[
            str, Dict[Tuple[Any, Any], int]
        ] = defaultdict(lambda: defaultdict(int))

        nan_count = 0
        for meta, result in zip(unified_meta, unified_results):
            group_key = (meta["record_id"], meta["group_idx"])
            domain_group_expected_counts[meta["domain"]][group_key] += 1
            scores = np.array(result["scores"], dtype=float)
            if scores.shape[0] != len(meta["ordered_keys"]):
                raise ValueError(
                    f"Score count mismatch for record {meta['record_id']}: "
                    f"expected {len(meta['ordered_keys'])}, got {scores.shape[0]}."
                )
            if np.isnan(scores).any():
                nan_count += 1
                continue
            keys = meta["ordered_keys"]
            pairs = sorted(
                zip(keys, scores), key=lambda x: (-float(x[1]), str(x[0]))
            )
            ranking_keys = [k for k, _ in pairs]
            domain_group_to_rankings[meta["domain"]][group_key].append(ranking_keys)

        if nan_count > 0:
            logger.warning(
                f"Skipped {nan_count}/{len(unified_results)} samples with NaN scores "
                f"in Consistency computation."
            )

        return {
            "domain_group_to_rankings": domain_group_to_rankings,
            "domain_group_expected_counts": domain_group_expected_counts,
        }

    def _aggregate_metrics(
        self,
        pair_bon_stats: Dict[str, Any],
        consistency_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        domain_pair_wins = pair_bon_stats["domain_pair_wins"]
        domain_pair_total = pair_bon_stats["domain_pair_total"]
        domain_bon_flags = pair_bon_stats["domain_bon_flags"]
        domain_group_to_rankings = consistency_stats["domain_group_to_rankings"]
        domain_group_expected_counts = consistency_stats[
            "domain_group_expected_counts"
        ]

        domain_consistency_flags: Dict[str, List[int]] = {}
        for domain, expected_groups in domain_group_expected_counts.items():
            flags: List[int] = []
            rankings_by_group = domain_group_to_rankings.get(domain, {})
            for group_key, expected_count in expected_groups.items():
                rankings = rankings_by_group.get(group_key, [])
                if (
                    expected_count != EXPECTED_PROMPTS_PER_GROUP
                    or len(rankings) != expected_count
                ):
                    flags.append(0)
                    continue
                first = rankings[0]
                flags.append(int(all(r == first for r in rankings)))
            domain_consistency_flags[domain] = flags

        domains = set(domain_pair_total.keys()) | set(domain_consistency_flags.keys())
        metrics_by_domain: Dict[str, Any] = {}
        for dname in sorted(domains):
            pair_total = int(domain_pair_total.get(dname, 0))
            pair_correct = float(domain_pair_wins.get(dname, 0.0))
            pair_accuracy = float(pair_correct / max(1, pair_total))

            bon_list = domain_bon_flags.get(dname, [])
            bon_accuracy = float(np.mean(bon_list)) if bon_list else 0.0

            cons_flags = domain_consistency_flags.get(dname, [])
            consistency_accuracy = float(np.mean(cons_flags)) if cons_flags else 0.0

            metrics_by_domain[dname] = {
                "pair_accuracy": pair_accuracy,
                "bon_accuracy": bon_accuracy,
                "consistency_accuracy": consistency_accuracy,
            }

        return metrics_by_domain

    def _report_and_save(
        self,
        config: DictConfig,
        logger: Logger,
        metrics_by_domain: Dict[str, Any],
    ):
        def _format_category(category_key: str) -> Dict[str, float]:
            values: Dict[str, float] = {}
            present_values: List[float] = []
            for dname in CANONICAL_DOMAINS:
                val = float(
                    metrics_by_domain.get(dname, {}).get(category_key, 0.0)
                )
                values[dname] = val
                if dname in metrics_by_domain:
                    present_values.append(val)
            values["Avg"] = (
                float(np.mean(present_values)) if present_values else 0.0
            )
            return values

        pair_list = _format_category("pair_accuracy")
        bon_list = _format_category("bon_accuracy")
        consistency_list = _format_category("consistency_accuracy")

        metrics = {
            "Pair": pair_list,
            "BoN": bon_list,
            "Consistency": consistency_list,
        }

        def _list_str(v: Dict[str, float]) -> str:
            return (
                "Chat: {Chat:.4f}, Writing: {Writing:.4f}, "
                "Reasoning: {Reasoning:.4f}, Safety: {Safety:.4f}, Avg: {Avg:.4f}"
            ).format(**v)

        logger.info("Pair -> " + _list_str(pair_list))
        logger.info("BoN -> " + _list_str(bon_list))
        logger.info("Consistency -> " + _list_str(consistency_list))

        with open(
            os.path.join(config.output.exp_dir, METRICS_FILENAME),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
