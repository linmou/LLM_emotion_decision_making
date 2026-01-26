"""
LoCoMoDataset - Specialized dataset for LoCoMo evaluation.
Handles LoCoMo conversational data format and custom F1 scoring with stemming.
"""

import re
import string
from collections import Counter
from typing import Any, Dict, List
from ..data_models import BenchmarkConfig, BenchmarkItem
from .base import BaseBenchmarkDataset


class LoCoMoDataset(BaseBenchmarkDataset):
    """
    Specialized dataset for LoCoMo (Long Context Memory) conversational QA tasks.
    
    Key features:
    - Handles multi-session conversational data format
    - Formats conversations with session headers and timestamps
    - Uses custom F1 scoring with stemming for evaluation
    - Expands each sample into multiple QA pairs
    """
    
    def _load_and_parse_data(self) -> List[BenchmarkItem]:
        """Load LoCoMo-specific data format"""
        raw_data = self._load_raw_data()
        return self._parse_locomo_data(raw_data)
    
    def _parse_locomo_data(self, raw_data: List[Dict[str, Any]]) -> List[BenchmarkItem]:
        """Parse LoCoMo format to BenchmarkItem objects"""
        items = []
        for sample in raw_data:
            sample_id = sample.get("sample_id", len(items))
            conversation = sample.get("conversation", {})
            qa_pairs = sample.get("qa", [])
            
            # Format conversation context once per sample
            formatted_conversation = self._format_locomo_conversation(conversation)
            
            # Create one BenchmarkItem per QA pair
            for qa in qa_pairs:
                items.append(BenchmarkItem(
                    id=f"{sample_id}_{len(items)}",
                    input_text=qa["question"],
                    context=formatted_conversation,
                    ground_truth=qa["answer"],
                    metadata={
                        "sample_id": sample_id,
                        "category": qa.get("category", "unknown"),
                        "evidence": qa.get("evidence", []),
                    }
                ))
        
        return items
    
    def _format_locomo_conversation(self, conversation: Dict[str, Any]) -> str:
        """Format LoCoMo conversation data with session headers and timestamps"""
        formatted_sessions = []
        
        # Get all session keys and sort them
        session_keys = [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")]
        session_keys.sort()
        
        for session_key in session_keys:
            session_data = conversation[session_key]
            date_key = f"{session_key}_date_time"
            session_date = conversation.get(date_key, "Unknown date")
            
            # Add session header
            formatted_sessions.append(f"=== {session_key.upper()} ({session_date}) ===")
            
            # Add conversation turns
            for turn in session_data:
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")
                formatted_sessions.append(f"{speaker}: {text}")
            
            # Add blank line between sessions
            formatted_sessions.append("")
        
        return "\n".join(formatted_sessions)
    
    def evaluate_response(self, response: str, ground_truth: Any, task_name: str) -> float:
        """Evaluate using LoCoMo F1 scoring with stemming"""
        return self._locomo_f1_score(response, ground_truth)
    
    def _locomo_f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        LoCoMo F1 score with stemming and normalization.
        Based on the original SmartDataset implementation but extracted here.
        """
        def normalize_answer(s):
            """Normalize text for comparison"""
            s = str(s).replace(",", "")
            
            def remove_articles(text):
                return re.sub(r"\b(a|an|the|and)\b", " ", text)
            
            def white_space_fix(text):
                return " ".join(text.split())
            
            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
            
            def lower(text):
                return text.lower()
            
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        
        def simple_stem(word):
            """Simple stemming for common suffixes"""
            if word.endswith("ing"):
                return word[:-3]
            elif word.endswith("ed"):
                return word[:-2]
            elif word.endswith("s") and len(word) > 2:
                return word[:-1]
            return word
        
        # Normalize and stem both texts
        prediction_tokens = [simple_stem(w) for w in normalize_answer(prediction).split()]
        ground_truth_tokens = [simple_stem(w) for w in normalize_answer(ground_truth).split()]
        
        # Calculate token overlap
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = 1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0.0
        recall = 1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
        
        # Calculate F1
        if precision + recall == 0:
            return 0.0
        
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def get_task_metrics(self, task_name: str) -> List[str]:
        """Return metrics available for LoCoMo tasks (always F1)"""
        # LoCoMo uses F1 scoring for all tasks
        return ["f1_score", "precision", "recall"]
    
    def evaluate_with_detailed_metrics(self, response: str, ground_truth: Any, task_name: str) -> Dict[str, float]:
        """Return detailed F1 metrics breakdown"""
        # We could extend _locomo_f1_score to return precision/recall separately
        # For now, return the F1 score for all metrics
        f1_score = self._locomo_f1_score(response, ground_truth)
        
        return {
            "overall_score": f1_score,
            "f1_score": f1_score,
            "precision": f1_score,  # Simplified - could calculate separately
            "recall": f1_score      # Simplified - could calculate separately
        }