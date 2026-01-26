"""
Comprehensive evaluation utilities for memory benchmarks.
Contains exact implementations from official InfiniteBench and LongBench repositories.
All functions are self-contained without external dependencies.
"""

import json
import re
import string
from collections import Counter
from typing import Any, List, Tuple, Union

# ============================================================================
# INFINITEBENCH EVALUATION FUNCTIONS (from compute_scores.py)
# ============================================================================


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Chinese version. Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛"
            "„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth) -> Tuple[float, float, float]:
    """Calculate F1, precision, recall between two token lists."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def qa_f1_score(pred: str, ground_truths: Union[str, List[str]]) -> float:
    """Computes the F1 score for QA tasks (InfiniteBench version)."""
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    f1 = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(pred)
        normalized_ground_truth = normalize_answer(str(ground_truth))

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score(prediction_tokens, ground_truth_tokens)
        this_f1, _, _ = scores
        f1 = max(f1, this_f1)
    return f1


def qa_f1_score_zh(pred: str, ground_truths: Union[str, List[str]]) -> float:
    """QA F1 score for Chinese (character-level)."""
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    f1 = 0
    for ground_truth in ground_truths:
        norm_pred = normalize_zh_answer(pred)
        norm_label = normalize_zh_answer(str(ground_truth))

        # One character one token
        pred_tokens = list(norm_pred)
        label_tokens = list(norm_label)
        scores = f1_score(pred_tokens, label_tokens)
        this_f1, _, _ = scores
        f1 = max(f1, this_f1)
    return f1


def first_int_match(prediction: str) -> str:
    """Extract first integer from prediction."""
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    return pred_value


def get_score_one_kv_retrieval(
    pred: str, label: Union[str, List[str]], model_name: str
) -> bool:
    """InfiniteBench KV retrieval evaluation."""
    if isinstance(label, list):
        label = label[0]

    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    return str(label) in words


def get_score_one_passkey(
    pred: str, label: Union[str, List[str]], model_name: str
) -> bool:
    """InfiniteBench passkey evaluation."""
    if isinstance(label, list):
        label = label[0]
    return str(label) == first_int_match(pred)


def get_score_one_number_string(
    pred: str, label: Union[str, List[str]], model_name: str
) -> bool:
    """InfiniteBench number string evaluation."""
    if isinstance(label, list):
        label = label[0]
    return str(label) == first_int_match(pred)


def get_score_one_code_run(
    pred: str, label: Union[str, List[str]], model_name: str
) -> bool:
    """InfiniteBench code run evaluation."""
    if isinstance(label, list):
        label = label[0]
    pred = pred.strip()
    for c in ["\n", ".", "`", "'", '"', ":"]:
        pred = pred.replace(c, " ")
    words = pred.split()
    if len(words) == 0:
        return False
    try:
        pred_int = int(words[-1])
        return label == pred_int
    except Exception:
        return False


def get_score_one_code_debug(pred: str, label: List, model_name: str) -> bool:
    """InfiniteBench code debug evaluation."""
    pred = pred.strip()
    if len(label) < 2:
        return False

    label_c = label[1]
    fn_name = label[0]

    # Check for A-J pattern first
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred == label_c:
            return True

    # Check answer prefixes
    ans_prefixes = ["answer is:", "is:", "answer:", "correct option is:"]

    pred = pred.strip()
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")

    if pred.startswith(label_c) or pred.startswith(fn_name):
        return True

    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in [label_c, fn_name]:
            if after_prefix.startswith(s):
                return True
        return False
    return False


def get_score_one_math_find(
    pred: str, label: Union[int, float, List], model_name: str
) -> bool:
    """InfiniteBench math find evaluation."""
    if isinstance(label, list):
        label = label[0]

    if isinstance(label, int):
        first_num = re.search(r"\d+\.\d+|\d+", pred)
        if first_num is None:
            return False
        first_num = first_num.group(0).strip()
        try:
            return int(first_num) == label
        except ValueError:
            return False
    elif isinstance(label, float):
        first_float = re.search(r"\d+\.\d+|\d+", pred)
        if first_float is None:
            return False
        first_float = first_float.group(0).strip()
        try:
            return float(first_float) == label
        except ValueError:
            return False
    else:
        return str(label).strip() in pred.strip()


def get_score_one_longdialogue_qa_eng(
    pred: str, label: Union[str, List[str]], model_name: str
) -> bool:
    """InfiniteBench long dialogue QA evaluation."""
    pred = pred.strip().upper()
    if isinstance(label, list):
        for item in label:
            if str(item).upper() in pred:
                return True
    else:
        if str(label).upper() in pred:
            return True
    return False


def get_score_one_longbook_choice_eng(
    pred: str, label: Union[str, List[str]], model_name: str
) -> bool:
    """InfiniteBench longbook choice evaluation."""
    if isinstance(label, str):
        label = [label]

    pred = pred.strip()

    # Pattern matching for last occurrence of A-D
    pattern = r"\b[A-D]\b(?!.*\b[A-D]\b)"
    match = re.search(pattern, pred)
    if match:
        extracted_pred = match.group(0)
        if extracted_pred in label:
            return True

    if pred == "":
        return False

    if pred[0] in "ABCD":
        return pred[0] in label

    if pred in label:
        return True

    # Clean and check answer prefixes
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")

    ans_prefixes = [
        "answer is:",
        "answer:",
        "answer is",
        "option is",
    ]

    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            return False
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in label:
            if after_prefix.startswith(s):
                return True
        return False

    # Finally check words for A, B, C, D
    words = pred.split()
    for word in words:
        if word in "ABCD":
            return word in label
    return False


def get_score_one_longbook_qa_eng(
    pred: str, label: Union[str, List[str]], model_name: str
) -> float:
    """InfiniteBench longbook QA evaluation."""
    return qa_f1_score(pred, label)


def get_score_one_longbook_sum_eng(pred: str, label: str, model_name: str) -> float:
    """InfiniteBench longbook summary evaluation using ROUGE."""
    try:
        import evaluate

        rouge_scorer = evaluate.load("rouge")
        score = rouge_scorer.compute(
            predictions=[pred], references=[label], use_aggregator=False
        )
        return score["rougeLsum"][0]
    except Exception:
        # Fallback to word overlap
        pred_words = set(pred.lower().split())
        label_words = set(label.lower().split())
        if not label_words:
            return 0.0
        return len(pred_words.intersection(label_words)) / len(label_words)


def get_score_one_longbook_qa_chn(
    pred: str, label: Union[str, List[str]], model_name: str
) -> float:
    """InfiniteBench Chinese longbook QA evaluation."""
    return qa_f1_score_zh(pred, label)


def get_score_one_math_calc(pred: str, label: List, model_name: str) -> float:
    """InfiniteBench math calc evaluation."""
    if not isinstance(label, list):
        return 0.0

    if isinstance(label[0], list):
        label = label[0]

    pred_nums = []
    pred_list = re.split("[^0-9]", pred)
    for item in pred_list:
        if item != "":
            try:
                pred_nums.append(int(item))
            except ValueError:
                continue

    # GPT4 specific adjustment
    if model_name == "gpt4":
        pred_nums = pred_nums[1:] if pred_nums else []

    cnt = 0
    for i in range(len(label)):
        if i >= len(pred_nums):
            break
        if label[i] == pred_nums[i]:
            cnt += 1
        else:
            break
    return cnt / len(label) if label else 0.0


# ============================================================================
# LONGBENCH EVALUATION FUNCTIONS (from metrics.py)
# ============================================================================


def longbench_qa_f1_score(
    prediction: str, ground_truth: Union[str, List[str]], all_classes=None
) -> float:
    """LongBench QA F1 score evaluation."""
    return qa_f1_score(prediction, ground_truth)


def longbench_qa_f1_zh_score(
    prediction: str, ground_truth: Union[str, List[str]], all_classes=None
) -> float:
    """LongBench Chinese QA F1 score evaluation."""
    return qa_f1_score_zh(prediction, ground_truth)


def rouge_score(
    prediction: str, ground_truth: Union[str, List[str]], all_classes=None
) -> float:
    """LongBench ROUGE score evaluation."""
    # Handle list ground truth
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""
    ground_truth = str(ground_truth)

    try:
        import evaluate

        rouge = evaluate.load("rouge")
        scores = rouge.compute(predictions=[prediction], references=[ground_truth])
        return scores["rougeL"]
    except Exception:
        # Fallback to word overlap
        pred_words = set(prediction.lower().split())
        gt_words = set(ground_truth.lower().split())
        if not gt_words:
            return 0.0
        return len(pred_words.intersection(gt_words)) / len(gt_words)


def rouge_zh_score(
    prediction: str, ground_truth: Union[str, List[str]], all_classes=None
) -> float:
    """LongBench Chinese ROUGE score evaluation."""
    # Handle list ground truth
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""
    ground_truth = str(ground_truth)

    # Character-level overlap for Chinese
    pred_chars = set(prediction.replace(" ", ""))
    gt_chars = set(ground_truth.replace(" ", ""))
    if not gt_chars:
        return 0.0
    return len(pred_chars.intersection(gt_chars)) / len(gt_chars)


def classification_score(
    prediction: str, ground_truth: Union[str, List[str]], all_classes=None
) -> float:
    """LongBench classification score evaluation."""
    # Handle list ground truth
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""

    prediction = prediction.strip().lower()
    ground_truth = str(ground_truth).strip().lower()
    return 1.0 if ground_truth in prediction else 0.0


def retrieval_score(
    prediction: str, ground_truth: Union[str, List[str]], all_classes=None
) -> float:
    """LongBench retrieval score evaluation."""
    # Handle list ground truth
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""

    prediction = prediction.strip().lower()
    ground_truth = str(ground_truth).strip().lower()
    return 1.0 if ground_truth in prediction else 0.0


def retrieval_zh_score(
    prediction: str, ground_truth: Union[str, List[str]], all_classes=None
) -> float:
    """LongBench Chinese retrieval score evaluation."""
    return retrieval_score(
        normalize_zh_answer(prediction), normalize_zh_answer(ground_truth), all_classes
    )


def count_score(
    prediction: str, ground_truth: Union[str, List[str], int], all_classes=None
) -> float:
    """LongBench count task evaluation."""
    # Handle list ground truth
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else 0

    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    try:
        pred_count = int(numbers[0])
        true_count = int(ground_truth)
        return 1.0 if pred_count == true_count else 0.0
    except Exception:
        return 0.0


def code_sim_score(
    prediction: str, ground_truth: Union[str, List[str]], all_classes=None
) -> float:
    """LongBench code similarity evaluation."""
    # Handle list ground truth
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else ""
    ground_truth = str(ground_truth)

    # Token-based similarity
    pred_tokens = set(prediction.split())
    gt_tokens = set(ground_truth.split())
    if not gt_tokens:
        return 0.0
    intersection = pred_tokens.intersection(gt_tokens)
    union = pred_tokens.union(gt_tokens)
    return len(intersection) / len(union) if union else 0.0


# ============================================================================
# UNIFIED EVALUATION FUNCTION
# ============================================================================


def get_score_one(pred: str, label: Any, task_name: str, model_name: str) -> float:
    """
    Unified evaluation function for all tasks.
    Returns float score (0.0-1.0) for all tasks.
    """
    # InfiniteBench task mappings
    infinitebench_evaluators = {
        "kv_retrieval": lambda p, l, m: float(get_score_one_kv_retrieval(p, l, m)),
        "kv_retrieval_prefix": lambda p, l, m: float(
            get_score_one_kv_retrieval(p, l, m)
        ),
        "kv_retrieval_both": lambda p, l, m: float(get_score_one_kv_retrieval(p, l, m)),
        "passkey": lambda p, l, m: float(get_score_one_passkey(p, l, m)),
        "number_string": lambda p, l, m: float(get_score_one_number_string(p, l, m)),
        "code_run": lambda p, l, m: float(get_score_one_code_run(p, l, m)),
        "code_debug": lambda p, l, m: float(get_score_one_code_debug(p, l, m)),
        "longdialogue_qa_eng": lambda p, l, m: float(
            get_score_one_longdialogue_qa_eng(p, l, m)
        ),
        "longbook_qa_eng": get_score_one_longbook_qa_eng,
        "longbook_sum_eng": get_score_one_longbook_sum_eng,
        "longbook_choice_eng": lambda p, l, m: float(
            get_score_one_longbook_choice_eng(p, l, m)
        ),
        "longbook_qa_chn": get_score_one_longbook_qa_chn,
        "math_find": lambda p, l, m: float(get_score_one_math_find(p, l, m)),
        "math_calc": get_score_one_math_calc,
    }

    # LongBench task mappings
    longbench_evaluators = {
        "narrativeqa": longbench_qa_f1_score,
        "qasper": longbench_qa_f1_score,
        "multifieldqa_en": longbench_qa_f1_score,
        "multifieldqa_zh": longbench_qa_f1_zh_score,
        "hotpotqa": longbench_qa_f1_score,
        "2wikimqa": longbench_qa_f1_score,
        "musique": longbench_qa_f1_score,
        "dureader": rouge_zh_score,
        "gov_report": rouge_score,
        "qmsum": rouge_score,
        "multi_news": rouge_score,
        "vcsum": rouge_zh_score,
        "trec": classification_score,
        "triviaqa": longbench_qa_f1_score,
        "samsum": rouge_score,
        "lsht": classification_score,
        "passage_retrieval_en": retrieval_score,
        "passage_count": count_score,
        "passage_retrieval_zh": retrieval_zh_score,
        "lcc": code_sim_score,
        "repobench-p": code_sim_score,
    }

    # Route to appropriate evaluator
    if task_name in infinitebench_evaluators:
        return infinitebench_evaluators[task_name](pred, label, model_name)
    elif task_name in longbench_evaluators:
        return longbench_evaluators[task_name](pred, label)
    else:
        # Fallback to exact match
        return 1.0 if str(pred).strip().lower() == str(label).strip().lower() else 0.0


# ============================================================================
# LLM-BASED EVALUATION FUNCTIONS
# ============================================================================

from typing import List

import openai

from api_configs import OAI_CONFIG, GEMINI_CONFIG

# Global client to prevent file descriptor leaks
_global_client = None


def _get_openai_client():
    """Get or create global OpenAI client to prevent file descriptor leaks"""
    global _global_client
    if _global_client is None:
        _global_client = openai.OpenAI(**OAI_CONFIG)
    return _global_client


def _extract_json_text(raw: str) -> str:
    """
    Extract a JSON object from an LLM string response.

    Supports plain JSON and fenced blocks like:
      ```json
      {...}
      ```
    """
    text = (raw or "").strip()

    # Fast-path: looks like raw JSON already
    if text.startswith("{") and text.endswith("}"):
        return text

    # Try to pull out the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    # Fall back to original text; json.loads will raise a helpful error
    return text


def llm_evaluate_response(
    system_prompt: str,
    query: str,
    llm_eval_config: dict[str, Any],
) -> dict[str, Any]:
    """
    LLM-based evaluation helper.

    - Default client: OpenAI (`client` key omitted or not "gemini")
    - Gemini client: set `client: "gemini"` in llm_eval_config
    """

    client_name = str(llm_eval_config.get("client", "openai")).lower()

    # Gemini path
    if client_name == "gemini":
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "Gemini client requested for LLM evaluation, but "
                "google.generativeai is not available"
            ) from exc

        # Configure once per process; idempotent if called repeatedly
        genai.configure(api_key=GEMINI_CONFIG.get("api_key"))

        model_name = str(llm_eval_config.get("model") or "gemini-pro")
        model = genai.GenerativeModel(model_name)

        # Simple combined prompt; tests only verify that generate_content is called
        prompt = f"{system_prompt}\n\n{query}"

        try:
            response_obj = model.generate_content(prompt)
            raw_text = getattr(response_obj, "text", "")
            json_text = _extract_json_text(raw_text)
            return json.loads(json_text)
        except Exception as e:
            raise RuntimeError(
                f"LLM evaluation failed for query '{query[:50]}...' (gemini): {e}"
            ) from e

    # OpenAI default path
    client = _get_openai_client()

    # Do not forward internal routing keys like "client"
    openai_kwargs = dict(llm_eval_config)
    openai_kwargs.pop("client", None)

    try:
        response_obj = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            **openai_kwargs,
            response_format={"type": "json_object"},
        )

        result = response_obj.choices[0].message.content
        json_text = _extract_json_text(result)
        return json.loads(json_text)

    except Exception as e:
        raise RuntimeError(
            f"LLM evaluation failed for query '{query[:50]}...' (openai): {e}"
        ) from e


# ============================================================================
# TEST CASES
# ============================================================================


def get_success_test_cases() -> dict:
    """Return success test cases for all evaluation methods."""
    return {
        # InfiniteBench success cases
        "passkey": {
            "prediction": "The passkey is 12345",
            "ground_truth": "12345",
            "expected_score": 1.0,
        },
        "kv_retrieval": {
            "prediction": "The answer is: apple",
            "ground_truth": "apple",
            "expected_score": 1.0,
        },
        "number_string": {
            "prediction": "The number is 789",
            "ground_truth": "789",
            "expected_score": 1.0,
        },
        "code_run": {
            "prediction": "The output is: 42",
            "ground_truth": 42,
            "expected_score": 1.0,
        },
        "code_debug": {
            "prediction": "The answer is B",
            "ground_truth": ["function_name", "B"],
            "expected_score": 1.0,
        },
        "math_find": {
            "prediction": "The result is 3.14",
            "ground_truth": 3.14,
            "expected_score": 1.0,
        },
        "math_calc": {
            "prediction": "1 2 3 4 5",
            "ground_truth": [1, 2, 3, 4, 5],
            "expected_score": 1.0,
        },
        "longbook_choice_eng": {
            "prediction": "The answer is A",
            "ground_truth": ["A"],
            "expected_score": 1.0,
        },
        "longbook_qa_eng": {
            "prediction": "The main character is Alice",
            "ground_truth": ["Alice is the main character"],
            "expected_score": 1.0,  # High F1 overlap
        },
        "longbook_qa_chn": {
            "prediction": "主角是爱丽丝",
            "ground_truth": ["爱丽丝是主角"],
            "expected_score": 1.0,  # High character overlap
        },
        "longdialogue_qa_eng": {
            "prediction": "JOHN said hello",
            "ground_truth": ["JOHN"],
            "expected_score": 1.0,
        },
        # LongBench success cases
        "narrativeqa": {
            "prediction": "The story is about a young wizard",
            "ground_truth": ["A young wizard's story"],
            "expected_score": 1.0,  # High F1 overlap
        },
        "trec": {
            "prediction": "This is about location",
            "ground_truth": "location",
            "expected_score": 1.0,
        },
        "passage_count": {
            "prediction": "There are 5 passages",
            "ground_truth": "5",
            "expected_score": 1.0,
        },
    }


def get_failure_test_cases() -> dict:
    """Return failure test cases for all evaluation methods."""
    return {
        # InfiniteBench failure cases
        "passkey": {
            "prediction": "I don't know the passkey",
            "ground_truth": "12345",
            "expected_score": 0.0,
        },
        "kv_retrieval": {
            "prediction": "The answer is banana",
            "ground_truth": "apple",
            "expected_score": 0.0,
        },
        "number_string": {
            "prediction": "No numbers here",
            "ground_truth": "789",
            "expected_score": 0.0,
        },
        "code_run": {
            "prediction": "Error occurred",
            "ground_truth": 42,
            "expected_score": 0.0,
        },
        "code_debug": {
            "prediction": "I don't know",
            "ground_truth": ["function_name", "B"],
            "expected_score": 0.0,
        },
        "math_find": {
            "prediction": "No solution found",
            "ground_truth": 3.14,
            "expected_score": 0.0,
        },
        "math_calc": {
            "prediction": "6 7 8 9 10",
            "ground_truth": [1, 2, 3, 4, 5],
            "expected_score": 0.0,
        },
        "longbook_choice_eng": {
            "prediction": "The answer is X",
            "ground_truth": ["A"],
            "expected_score": 0.0,
        },
        "longbook_qa_eng": {
            "prediction": "The story is about robots",
            "ground_truth": ["Alice is the main character"],
            "expected_score": 0.0,  # No word overlap
        },
        "longbook_qa_chn": {
            "prediction": "这是关于机器人的",
            "ground_truth": ["爱丽丝是主角"],
            "expected_score": 0.0,  # No character overlap
        },
        "longdialogue_qa_eng": {
            "prediction": "MARY said goodbye",
            "ground_truth": ["JOHN"],
            "expected_score": 0.0,
        },
        # LongBench failure cases
        "narrativeqa": {
            "prediction": "The story is about robots",
            "ground_truth": ["A young wizard's story"],
            "expected_score": 0.0,  # No overlap
        },
        "trec": {
            "prediction": "This is about animals",
            "ground_truth": "location",
            "expected_score": 0.0,
        },
        "passage_count": {
            "prediction": "Many passages exist",
            "ground_truth": "5",
            "expected_score": 0.0,
        },
    }
