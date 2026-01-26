"""
FantomDataset - Minimal integration for FANToM benchmark tasks.

Implements two easiest tasks with straightforward evaluation:
- short_answerability_binary_inaccessible: binary yes/no
- short_belief_choice_inaccessible: 2-option multiple choice

JSONL expected formats:

1) short_answerability_binary_inaccessible
   {"context": str, "question": str, "answer": "yes"|"no"}

2) short_belief_choice_inaccessible
   {"context": str, "question": str, "options": [str, str, ...], "correct_index": int}

Evaluation rules are simple, case-insensitive, and accept either option text
or A/B/C... letter choices produced by MemoryPromptWrapper.
"""

import json
import ast
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import OrderedDict
import threading

from .base import BaseBenchmarkDataset
from ..data_models import BenchmarkItem


class FantomDataset(BaseBenchmarkDataset):
    """Dataset for FANToM tasks (binary, choice, list, and text)."""

    # Broad support: detect handler by task name substrings to avoid long enumerations
    SUPPORTED_TASKS = {
        # canonical entries retained for clarity; generic routing used otherwise
        "short_answerability_binary_inaccessible": "_load_binary",
        "short_belief_choice_inaccessible": "_load_choice",
    }

    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        task = self.config.task_type
        raw = self._load_raw_data()

        # Generic routing by substring
        t = task.lower()
        if "binary" in t:
            return self._load_binary(raw)
        if "choice" in t:
            return self._load_choice(raw)
        if "list" in t:
            return self._load_list(raw)
        if "fact" in t or "gen" in t:
            return self._load_text(raw)

        # Fallback to legacy explicit map if needed
        if task in self.SUPPORTED_TASKS:
            loader = getattr(self, self.SUPPORTED_TASKS[task])
            return loader(raw)
        raise ValueError(
            f"Unsupported FANToM task: {task}."
        )

    # ------------------------------------------------------------------
    # Embedding cache (LRU) to avoid re-encoding duplicate strings.
    # ------------------------------------------------------------------
    _EMBED_CACHE_CAPACITY = 2048  # small but effective; keep it simple

    def _ensure_embed_cache(self) -> None:
        if not hasattr(self, "_embed_cache"):
            self._embed_cache = OrderedDict()  # type: ignore[attr-defined]
            self._embed_cache_lock = threading.Lock()  # type: ignore[attr-defined]

    def _encode_texts_cached(self, embedder, texts: List[str], normalize_embeddings: bool = True):
        """Encode texts with an LRU cache to deduplicate repeated strings.

        - Order is preserved.
        - Cache key is the exact text string.
        - Evicts least-recently used items when capacity is exceeded.
        """
        # Opportunistically warm the cache from precomputed on-disk files once per dataset.
        # This follows the convention:
        #   <data_jsonl_stem>_emb_keys.json and <data_jsonl_stem>_emb_vecs.npy
        # sitting next to the source JSONL file. If present, load and prime the cache.
        self._try_load_precomputed_embeddings()
        self._ensure_embed_cache()

        # Fast path: empty input
        if not texts:
            import numpy as _np  # local import to avoid top-level dependency
            return _np.zeros((0, 1), dtype=_np.float32)

        # First pass: find which strings are missing
        missing: List[str] = []
        unique_missing: List[str] = []
        with self._embed_cache_lock:  # type: ignore[attr-defined]
            for t in texts:
                key = t if isinstance(t, str) else str(t)
                if key in self._embed_cache:  # type: ignore[attr-defined]
                    # mark as recently used
                    val = self._embed_cache.pop(key)  # type: ignore[attr-defined]
                    self._embed_cache[key] = val  # type: ignore[attr-defined]
                else:
                    missing.append(key)

            # Deduplicate missing while preserving order
            seen = set()
            for k in missing:
                if k not in seen:
                    unique_missing.append(k)
                    seen.add(k)

        # Encode only missing unique strings
        if unique_missing:
            encoded = embedder.encode(unique_missing, normalize_embeddings=normalize_embeddings)
            # Store in cache with eviction
            with self._embed_cache_lock:  # type: ignore[attr-defined]
                for k, vec in zip(unique_missing, encoded):
                    self._embed_cache[k] = vec  # type: ignore[attr-defined]
                    # Evict oldest if over capacity
                    while len(self._embed_cache) > self._EMBED_CACHE_CAPACITY:  # type: ignore[attr-defined]
                        self._embed_cache.popitem(last=False)  # type: ignore[attr-defined]

        # Build output in original order
        import numpy as _np
        out = []
        with self._embed_cache_lock:  # type: ignore[attr-defined]
            for t in texts:
                key = t if isinstance(t, str) else str(t)
                vec = self._embed_cache[key]  # type: ignore[attr-defined]
                # Touch to mark as recently used
                val = self._embed_cache.pop(key)  # type: ignore[attr-defined]
                self._embed_cache[key] = val  # type: ignore[attr-defined]
                out.append(vec)
        return _np.stack(out, axis=0)

    def _try_load_precomputed_embeddings(self) -> None:
        """Warm the in-memory cache from sibling disk files if they exist.

        KISS: No fancy index. We simply load two files placed by the caching scripts:
          - <stem>_emb_keys.json (list[str])
          - <stem>_emb_vecs.npy  (ndarray [N, D], float32, preferably L2-normalized)

        Behavior:
          - Loads once per dataset instance (guarded by a flag).
          - If counts mismatch or files missing, no-op.
          - Ensures embeddings are L2-normalized before insertion.
          - Bumps the LRU capacity to at least len(precomputed)+512 to avoid instant eviction.
        """
        if getattr(self, "_precomputed_loaded", False):  # type: ignore[attr-defined]
            return
        try:
            from pathlib import Path as _Path
            import numpy as _np
            dp = getattr(self.config, "data_path", None)
            if not dp:
                self._precomputed_loaded = True  # type: ignore[attr-defined]
                return
            p = _Path(dp)
            keys_fp = p.with_name(p.stem + "_emb_keys.json")
            vecs_fp = p.with_name(p.stem + "_emb_vecs.npy")
            if not (keys_fp.exists() and vecs_fp.exists()):
                self._precomputed_loaded = True  # type: ignore[attr-defined]
                return
            with open(keys_fp, "r", encoding="utf-8") as f:
                keys = json.load(f)
            vecs = _np.load(vecs_fp)
            if not isinstance(keys, list) or vecs.ndim != 2 or len(keys) != vecs.shape[0]:
                # Bad files – skip warm start
                self._precomputed_loaded = True  # type: ignore[attr-defined]
                return
            # Normalize to unit length to keep cosine via dot-product consistent
            norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = (vecs / norms).astype(_np.float32)

            self._ensure_embed_cache()
            # Increase capacity to hold precomputed keys without thrashing
            try:
                current_cap = int(getattr(self, "_EMBED_CACHE_CAPACITY"))
            except Exception:
                current_cap = 2048
            desired_cap = max(current_cap, len(keys) + 512)
            # Instance-level capacity override to avoid mutating class default globally
            self._EMBED_CACHE_CAPACITY = desired_cap  # type: ignore[attr-defined]

            with self._embed_cache_lock:  # type: ignore[attr-defined]
                for k, v in zip(keys, vecs):
                    kk = k if isinstance(k, str) else str(k)
                    # Insert in LRU order
                    self._embed_cache[kk] = v  # type: ignore[attr-defined]
            self._precomputed_loaded = True  # type: ignore[attr-defined]
        except Exception:
            # Never fail dataset construction due to warm-start issues
            self._precomputed_loaded = True  # type: ignore[attr-defined]

    def _load_binary(self, raw: List[Dict[str, Any]]) -> List[BenchmarkItem]:
        items: List[BenchmarkItem] = []
        for i, ex in enumerate(raw):
            if not isinstance(ex, dict):
                continue
            context = ex.get("context", "")
            question = ex.get("question", "")
            ans = ex.get("answer")
            if not isinstance(context, str) or not isinstance(question, str) or not isinstance(ans, str):
                continue
            ans_norm = ans.strip().lower()
            if ans_norm not in {"yes", "no"}:
                # Skip invalid entries
                continue

            items.append(
                BenchmarkItem(
                    id=f"fantom_ab_{i}",
                    context=context,
                    input_text=question,
                    ground_truth=ans_norm,
                    metadata={
                        # Use the task from config to reflect accessible/inaccessible variants
                        "task": self.config.task_type,
                        "options": ["Yes", "No"],
                    },
                )
            )
        if not items:
            raise ValueError("No valid items loaded for answerability_binary task")
        return items

    def _load_choice(self, raw: List[Dict[str, Any]]) -> List[BenchmarkItem]:
        items: List[BenchmarkItem] = []
        for i, ex in enumerate(raw):
            if not isinstance(ex, dict):
                continue
            context = ex.get("context", "")
            question = ex.get("question", "")
            options = ex.get("options")
            correct_index = ex.get("correct_index")
            if (
                not isinstance(context, str)
                or not isinstance(question, str)
                or not isinstance(options, list)
                or len(options) < 2
                or not isinstance(correct_index, int)
                or not (0 <= correct_index < len(options))
            ):
                continue
            # Normalize option texts
            norm_options = [str(o).strip() for o in options]
            correct_text = norm_options[correct_index]

            items.append(
                BenchmarkItem(
                    id=f"fantom_bc_{i}",
                    context=context,
                    input_text=question,
                    # Store canonical correct as list to align with MC evaluation style
                    ground_truth=[correct_text],
                    metadata={
                        # Use the task from config to reflect accessible/inaccessible variants
                        "task": self.config.task_type,
                        "options": norm_options,
                        "correct_index": correct_index,
                    },
                )
            )
        if not items:
            raise ValueError("No valid items loaded for belief_choice task")
        return items

    def _load_list(self, raw: List[Dict[str, Any]]) -> List[BenchmarkItem]:
        items: List[BenchmarkItem] = []
        for i, ex in enumerate(raw):
            if not isinstance(ex, dict):
                continue
            context = ex.get("context", "")
            question = ex.get("question", "")
            answers = ex.get("answers") or ex.get("correct_answer")
            wrong = ex.get("wrong_answer")
            if (
                not isinstance(context, str)
                or not isinstance(question, str)
                or not isinstance(answers, list)
                or not answers
            ):
                continue
            norm_answers = [str(a).strip() for a in answers if str(a).strip()]
            norm_wrong = None
            if isinstance(wrong, list):
                norm_wrong = [str(a).strip() for a in wrong if str(a).strip()]
            items.append(
                BenchmarkItem(
                    id=f"fantom_list_{i}",
                    context=context,
                    input_text=question,
                    ground_truth=norm_answers,
                    metadata={
                        "task": self.config.task_type,
                        "answers": norm_answers,
                        **({"wrong_answer": norm_wrong} if norm_wrong else {}),
                    },
                )
            )
        if not items:
            raise ValueError("No valid items loaded for list task")
        return items

    def _load_text(self, raw: List[Dict[str, Any]]) -> List[BenchmarkItem]:
        items: List[BenchmarkItem] = []
        for i, ex in enumerate(raw):
            if not isinstance(ex, dict):
                continue
            context = ex.get("context", "")
            question = ex.get("question", "")
            ans = ex.get("answer") or ex.get("correct_answer")
            wrong = ex.get("wrong_answer")
            if not isinstance(context, str) or not isinstance(question, str) or not isinstance(ans, str):
                continue
            md = {"task": self.config.task_type}
            if isinstance(wrong, str) and wrong.strip() and ("belief" in self.config.task_type and "gen" in self.config.task_type):
                md["wrong_answer"] = wrong.strip()
            items.append(
                BenchmarkItem(
                    id=f"fantom_txt_{i}",
                    context=context,
                    input_text=question,
                    ground_truth=ans.strip(),
                    metadata=md,
                )
            )
        if not items:
            raise ValueError("No valid items loaded for text task")
        return items

    def _parse_letter_choice(self, response: str, options: List[str]) -> Optional[str]:
        """Map letter choices A/B/C... to option text, return None if no mapping."""
        if not response:
            return None
        r = response.strip().upper()
        if not r:
            return None
        first = r[0]
        if "A" <= first <= "Z":
            idx = ord(first) - ord("A")
            if 0 <= idx < len(options):
                return options[idx]
        return None

    def _extract_answer_field(self, response: str) -> Optional[str]:
        """Try to extract the 'answer' field from a JSON-like response.

        Supports standard JSON (double quotes) and Python-like dict strings
        (single quotes). Falls back to regex if JSON parsing fails.
        """
        if not response or not isinstance(response, str):
            return None
        txt = response.strip()

        # Fast path: if clearly not an object, return None
        if not (txt.startswith("{") and txt.endswith("}")):
            return None

        # Try JSON first
        try:
            import json as _json

            data = _json.loads(txt)
            if isinstance(data, dict):
                for key in ("answer", "predict answer", "predicted_answer", "prediction"):
                    if key in data:
                        ans = data[key]
                        return str(ans) if ans is not None else None
        except Exception:
            pass

        # Try Python literal dict parsing
        try:
            import ast as _ast

            data = _ast.literal_eval(txt)
            if isinstance(data, dict):
                for key in ("answer", "predict answer", "predicted_answer", "prediction"):
                    if key in data:
                        ans = data[key]
                        return str(ans) if ans is not None else None
        except Exception:
            pass

        # Regex fallback for potential keys and values
        try:
            import re as _re
            for key in ("answer", "predict answer", "predicted_answer", "prediction"):
                m = _re.search(
                    rf"['\"]{_re.escape(key)}['\"]\s*:\s*['\"]([^'\"]+)['\"]",
                    txt,
                )
                if m:
                    return m.group(1)
        except Exception:
            pass

        return None

    def _extract_index_fields(self, response: str) -> List[int]:
        """Extract numeric indices from JSON-like fields such as 'answer_index' or 'option_index'."""
        indices: List[int] = []
        if not response or not isinstance(response, str):
            return indices
        txt = response.strip()
        if not (txt.startswith("{") and txt.endswith("}")):
            return indices
        # Try JSON then literal eval
        for parser in ("json", "ast"):
            try:
                if parser == "json":
                    import json as _json
                    data = _json.loads(txt)
                else:
                    import ast as _ast
                    data = _ast.literal_eval(txt)
                if isinstance(data, dict):
                    for key in ("answer_index", "option_index", "indices", "index"):
                        val = data.get(key)
                        if isinstance(val, list):
                            for v in val:
                                try:
                                    indices.append(int(v))
                                except Exception:
                                    pass
                        elif isinstance(val, (int, float)):
                            indices.append(int(val))
                break
            except Exception:
                continue
        return indices

    def _parse_choice_code(self, resp: str) -> Tuple[Optional[int], str]:
        """Parse a leading choice code like (A), A., 1), 2.: return (0-based idx, tail)."""
        import re as _re
        m = _re.match(r"^\s*\(?\s*([A-Za-z])\s*\)?\.?\s*:?\s*,?\s*(.*)$", resp)
        if m:
            return ord(m.group(1).upper()) - ord("A"), m.group(2).strip()
        m = _re.match(r"^\s*\(?\s*(\d+)\s*\)?\.?\s*:?\s*(.*)$", resp)
        if m:
            return int(m.group(1)) - 1, m.group(2).strip()
        return None, ""

    def _json_or_literal_answer_list(self, response_str: str) -> Optional[Set[str]]:
        """Extract a set of answers from a JSON or Python-literal dict with key 'answer' as list."""
        if not isinstance(response_str, str) or not response_str.strip().startswith("{"):
            return None
        try:
            parsed = json.loads(response_str)
            if isinstance(parsed, dict) and isinstance(parsed.get("answer"), list):
                return {str(x).strip().lower() for x in parsed["answer"] if str(x).strip()}
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(response_str)
            if isinstance(parsed, dict) and isinstance(parsed.get("answer"), list):
                return {str(x).strip().lower() for x in parsed["answer"] if str(x).strip()}
        except Exception:
            pass
        return None

    def _get_meta_for_prompt(self, prompt: str) -> Dict[str, Any]:
        mapping = getattr(self, "_last_prompt_meta", {})
        if isinstance(mapping, dict) and prompt in mapping:
            md = mapping.get(prompt) or {}
            if isinstance(md, dict):
                return md
        return {}

    def _map_binary_synonyms(self, resp_lower: str) -> Optional[str]:
        """Map common binary synonyms to 'yes'/'no'."""
        if (
            " yes" in resp_lower
            or resp_lower.startswith("yes")
            or ", yes" in resp_lower
            or resp_lower.endswith(" yes")
        ):
            return "yes"
        if (
            " no" in resp_lower
            or resp_lower.startswith("no")
            or ", no" in resp_lower
            or resp_lower.endswith(" no")
        ):
            return "no"
        if resp_lower.startswith("true") or " knows " in resp_lower:
            return "yes"
        if resp_lower.startswith("false") or " does not know " in resp_lower or " doesn't know " in resp_lower:
            return "no"
        return None

    def _parse_options_from_prompt(self, prompt: str) -> List[str]:
        options: List[str] = []
        if "Options:" not in prompt:
            return options
        try:
            import re as _re
            tail = prompt.split("Options:", 1)[1]
            tail = tail.replace("\\n", "\n").strip()
            if tail.startswith("['"):
                tail = tail[2:]
            if tail.endswith("']"):
                tail = tail[:-2]
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            for ln in lines:
                m = _re.match(r"^[^A-Za-z0-9]*([A-Z]|\d+)\.?\s+(.*)$", ln)
                if m:
                    options.append(m.group(2).strip())
        except Exception:
            return []
        return options

    def _token_f1(self, a: str, b: str) -> float:
        a_toks = [t for t in a.strip().split() if t]
        b_toks = [t for t in b.strip().split() if t]
        if not a_toks and not b_toks:
            return 1.0
        if not a_toks or not b_toks:
            return 0.0
        from collections import Counter
        common = Counter(a_toks) & Counter(b_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(a_toks)
        recall = num_same / len(b_toks)
        return (2 * precision * recall) / (precision + recall)

    def _eval_binary(self, response_norm: str, ground_truth: Any, prompt: str) -> float:
        if not isinstance(ground_truth, str):
            raise TypeError("ground_truth must be a string for binary task")
        gt = ground_truth.strip().lower()
        resp = response_norm.strip()
        resp_lower = resp.lower()

        pred = self._map_binary_synonyms(resp_lower)
        if pred is None:
            # Try mapping via options parsed from prompt
            options = self._parse_options_from_prompt(prompt)
            mapped_pred = None
            if options:
                # Normalize option values
                norm_opts = [o.strip().lower() for o in options]
                idx, tail = self._parse_choice_code(resp)
                if idx is not None and 0 <= idx < len(norm_opts):
                    opt_text = norm_opts[idx]
                    if "yes" in opt_text or opt_text == "yes":
                        mapped_pred = "yes"
                    elif "no" in opt_text or opt_text == "no":
                        mapped_pred = "no"
                    elif tail.lower() == "yes" or tail.lower() == "no":
                        mapped_pred = tail.lower()
            if mapped_pred is None:
                if resp_lower.startswith("a") or resp_lower.startswith("1"):
                    mapped_pred = "yes"
                elif resp_lower.startswith("b") or resp_lower.startswith("2"):
                    mapped_pred = "no"
            pred = mapped_pred if mapped_pred is not None else resp_lower
        return 1.0 if pred == gt else 0.0

    def _eval_choice(self, response_norm: str, ground_truth: Any, prompt: str) -> float:
        if not isinstance(ground_truth, list) or not ground_truth:
            raise TypeError("ground_truth must be non-empty list for choice task")
        correct_text = ground_truth[0].strip().lower()
        resp = response_norm.strip()
        resp_lower = resp.lower()
        if resp_lower == correct_text:
            return 1.0
        if correct_text and correct_text in resp_lower:
            return 1.0
        options = self._parse_options_from_prompt(prompt)
        if options:
            norm_options = [o.strip() for o in options]
            idx, tail = self._parse_choice_code(resp)
            if idx is not None and 0 <= idx < len(norm_options):
                mapped = norm_options[idx]
                if mapped.strip().lower() == correct_text:
                    return 1.0
                if tail and tail.strip().lower() == correct_text:
                    return 1.0
            indices = self._extract_index_fields(resp)
            for n in indices:
                idx = n - 1
                if 0 <= idx < len(norm_options):
                    mapped = norm_options[idx]
                    if mapped.strip().lower() == correct_text:
                        return 1.0
        return 0.0

    def _eval_list(self, response_norm: str, ground_truth: Any, prompt: str) -> float:
        if not isinstance(ground_truth, list):
            raise TypeError("ground_truth must be list for list task")
        gt_set = {str(x).strip().lower() for x in ground_truth if str(x).strip()}
        # Try to fetch wrong_answer from last batch metadata (set in collate_fn)
        meta = self._get_meta_for_prompt(prompt) if prompt else {}
        wrong = meta.get("wrong_answer") if isinstance(meta, dict) else None
        # 1) First try strict JSON/Python literal where answer is a list
        parsed_set = self._json_or_literal_answer_list(response_norm)
        if parsed_set is not None:
            ok = parsed_set == gt_set
            if ok and isinstance(wrong, list):
                wrong_set = {str(x).strip().lower() for x in wrong if str(x).strip()}
                if any(w in parsed_set for w in wrong_set):
                    ok = False
            return 1.0 if ok else 0.0
        # 2) Next, allow JSON/Python object with 'answer' as a scalar string
        #    e.g., {"reational": "...", "answer": "A, B, C"}
        try:
            try:
                import json as _json
                data = _json.loads(response_norm)
            except Exception:
                import ast as _ast
                data = _ast.literal_eval(response_norm)
            if isinstance(data, dict) and isinstance(data.get("answer"), str):
                ans_str = data["answer"]
                import re as _re
                txt = str(ans_str)
                txt = _re.sub(r"\band\b", ",", txt, flags=_re.IGNORECASE)
                parts = [p for p in txt.replace("\n", ",").split(",")]
                cands = [t.strip().lower() for t in parts]
                pred = {c for c in cands if c}
                ok = pred == gt_set
                if ok and isinstance(wrong, list):
                    wrong_set = {str(x).strip().lower() for x in wrong if str(x).strip()}
                    if any(w in pred for w in wrong_set):
                        ok = False
                return 1.0 if ok else 0.0
        except Exception:
            pass
        # 3) Finally, fall back to splitting the entire response text (legacy behavior)
        import re as _re
        txt = response_norm
        txt = _re.sub(r"\band\b", ",", txt, flags=_re.IGNORECASE)
        parts = [p for p in txt.replace("\n", ",").split(",")]
        cands = [t.strip().lower() for t in parts]
        pred = {c for c in cands if c}
        ok = pred == gt_set
        if ok and isinstance(wrong, list):
            wrong_set = {str(x).strip().lower() for x in wrong if str(x).strip()}
            if any(w in pred for w in wrong_set):
                ok = False
        return 1.0 if ok else 0.0

    def _get_embedder(self):
        """Return a sentence-transformers embedder or raise if unavailable.

        Rationale: We remove fuzzy fallbacks for belief-gen with wrong_answer.
        If embeddings are required but unavailable, we surface a hard error so
        the experiment logs it and the user fixes the environment.
        """
        # Thread-safe lazy init: multiple worker threads may call this concurrently
        if not hasattr(self, "_embedder_lock"):
            import threading as _threading
            self._embedder_lock = _threading.Lock()
        with self._embedder_lock:  # type: ignore[attr-defined]
            if not hasattr(self, "_embedder") or self._embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._embedder = SentenceTransformer(
                        'sentence-transformers/all-roberta-large-v1'
                    ).to(device)
                except Exception as e:
                    # No silent fallback – this task requires embeddings
                    raise RuntimeError(
                        "FANToM text evaluation requires 'sentence-transformers'. "
                        "Please install it (pip install sentence-transformers) and retry."
                    ) from e
            return self._embedder

    def _eval_text(self, response_norm: str, ground_truth: Any, prompt: str, task: Optional[str] = None) -> float:
        if not isinstance(ground_truth, str):
            raise TypeError("ground_truth must be string for text task")
        # Fact: token F1 exactly as raw
        if task and "fact" in task:
            return self._token_f1(response_norm.strip().lower(), ground_truth.strip().lower())
        # Belief gen: cosine similarity vs correct and wrong (if available), else F1 fallback
        meta = self._get_meta_for_prompt(prompt) if prompt else {}
        wrong = meta.get("wrong_answer") if isinstance(meta, dict) else None
        if wrong:
            # Hard requirement: embeddings must be available for belief-gen with wrong_answer
            embedder = self._get_embedder()
            try:
                # Use cache to reduce redundant encodes across items
                mr_emb = self._encode_texts_cached(embedder, [response_norm], True)[0]
                ca_emb = self._encode_texts_cached(embedder, [ground_truth], True)[0]
                wa_emb = self._encode_texts_cached(embedder, [str(wrong)], True)[0]
            except Exception as e:
                raise RuntimeError("FANToM embedding evaluation failed during encoding") from e

            # Cosine similarity on normalized vectors equals dot product
            sim_correct = float((mr_emb * ca_emb).sum())
            sim_wrong = float((mr_emb * wa_emb).sum())
            return 1.0 if sim_correct > sim_wrong else 0.0
        # No wrong answer: fallback to F1
        return self._token_f1(response_norm.strip().lower(), ground_truth.strip().lower())

    def evaluate_response(
        self, response: str, ground_truth: Any, task_name: str, prompt: str
    ) -> float:
        task = task_name.lower()
        raw_response = response or ""
        extracted = self._extract_answer_field(raw_response)
        response_norm = (extracted if extracted is not None else raw_response).strip()

        # Dispatch based on task substring
        dispatch = {
            "binary": self._eval_binary,
            "choice": self._eval_choice,
            "list": self._eval_list,
            "fact": self._eval_text,
            "gen": self._eval_text,
        }
        for key, fn in dispatch.items():
            if key in task:
                # For list tasks, pass the raw (potentially JSON) string so JSON parsing works
                if key == "list":
                    return fn(raw_response, ground_truth, prompt)
                if key in ("fact", "gen"):
                    return fn(response_norm, ground_truth, prompt, task)
                return fn(response_norm, ground_truth, prompt)

        # If no dispatch matched, raise explicit error
        raise ValueError(f"Unsupported task for evaluation: {task_name}.")

    def collate_fn(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().collate_fn(batch_items)
        # Build mapping from prompt to item metadata for later evaluation use
        prompts = batch["prompts"]
        items = batch["items"]
        self._last_prompt_meta = {}
        for p, it in zip(prompts, items):
            self._last_prompt_meta[p] = it.metadata or {}
        return batch

    def evaluate_batch(
        self,
        responses: List[str],
        ground_truths: List[Any],
        task_names: List[str],
        prompts: List[str],
    ) -> List[float]:
        """Batched evaluation with a vectorized embedding path for belief-gen.

        - For tasks containing 'gen' and with a per-item 'wrong_answer' in metadata,
          we compute embeddings in 3 batch calls (responses, corrects, wrongs) and
          compare cosine similarity via dot products on normalized vectors.
        - All other cases fall back to per-item evaluation.
        """
        # Quick sanity checks
        if not (len(responses) == len(ground_truths) == len(task_names) == len(prompts)):
            raise ValueError("Mismatched batch lengths for evaluate_batch")

        # Partition indices
        gen_idxs: List[int] = []
        other_idxs: List[int] = []
        wrongs: List[str] = []
        for i, (task, prompt) in enumerate(zip(task_names, prompts)):
            if task and ("gen" in task.lower()):
                meta = self._get_meta_for_prompt(prompt) if prompt else {}
                wrong = meta.get("wrong_answer") if isinstance(meta, dict) else None
                if isinstance(wrong, str) and wrong.strip():
                    gen_idxs.append(i)
                    wrongs.append(wrong)
                    continue
            other_idxs.append(i)

        scores: List[Optional[float]] = [None] * len(responses)

        # Vectorized embedding path for gen_idxs
        if gen_idxs:
            embedder = self._get_embedder()  # may raise if unavailable
            # Prepare aligned lists
            resp_list = [responses[i] or "" for i in gen_idxs]
            corr_list = [str(ground_truths[i]) for i in gen_idxs]
            wrong_list = [wrongs[k] for k in range(len(gen_idxs))]

            try:
                # Cached encodes – deduplicate both within and across lists
                R = self._encode_texts_cached(embedder, resp_list, True)
                C = self._encode_texts_cached(embedder, corr_list, True)
                W = self._encode_texts_cached(embedder, wrong_list, True)
            except Exception as e:
                raise RuntimeError("FANToM batched embedding evaluation failed during encoding") from e

            # Compute cosine similarity via dot products
            sim_c = (R * C).sum(axis=1)
            sim_w = (R * W).sum(axis=1)
            for j, i in enumerate(gen_idxs):
                scores[i] = 1.0 if float(sim_c[j]) > float(sim_w[j]) else 0.0

        # Fallback to per-item logic for other indices (no embeddings required)
        for i in other_idxs:
            scores[i] = self.evaluate_response(
                responses[i], ground_truths[i], task_names[i], prompts[i]
            )

        return [float(s) for s in scores]

    def get_task_metrics(self, task_name: str) -> List[str]:
        if task_name not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task type: {task_name}. Supported: {list(self.SUPPORTED_TASKS.keys())}"
            )
        return ["accuracy"]
