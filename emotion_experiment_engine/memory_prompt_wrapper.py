"""
Memory benchmark prompt wrapper following GameScenarioDataset pattern.
Integrates with neuro_manipulation.prompt_wrapper.PromptWrapper for proper model-specific formatting.
"""

import re
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat


class MemoryPromptWrapper(PromptWrapper):
    """
    Prompt wrapper for memory benchmark tasks.
    Follows the pattern from GameScenarioDataset but adapted for memory benchmarks.
    """

    system_prompt_format = "You are a helpful AI assistant. Please answer the following question based on the given context."

    # Emotion suffixes for adaptive augmentation
    EMOTION_SUFFIXES = {
        "happiness": {
            "prefix": " ",
            "suffix": " I remember this feeling. I'm absolutely thrilled and can barely contain my excitement! My heart is overflowing with pure joy right now. I feel like I'm walking on clouds - everything seems perfect today. ",
        },
        "sadness": {
            "prefix": " ",
            "suffix": " I remember this feeling. I can feel this heavy weight pressing down on my chest. The world seems gray and empty right now, like all the color has drained away. I feel so overwhelmed with sorrow that even simple tasks feel impossible. ",
        },
        "fear": {
            "prefix": " ",
            "suffix": " I remember this feeling. My heart is pounding and I can't shake this feeling of dread. Every shadow seems threatening and I feel like something terrible is about to happen. I'm trembling and can barely think straight through this overwhelming anxiety. ",
        },
        "anger": {
            "prefix": " ",
            "suffix": " I remember this feeling. I'm absolutely furious and my blood is boiling! This rage is consuming me and I can feel my jaw clenching. Everything feels like an injustice right now and I want to lash out at the unfairness of it all. ",
        },
        "disgust": {
            "prefix": " ",
            "suffix": " I remember this feeling. I feel physically sick and repulsed by what I'm experiencing. There's this nauseating feeling in my stomach and I want to recoil from everything around me. I'm disgusted and can't stand the thought of engaging with this any longer. ",
        },
        "surprise": {
            "prefix": " ",
            "suffix": " I remember this feeling. I'm completely stunned and can't believe what just happened! My mind is racing trying to process this unexpected turn of events. I feel caught off guard and amazed - this is the last thing I ever expected to encounter! ",
        },
        "neutral": {"prefix": " ", "suffix": " I remember this feeling."},
    }

    def __init__(self, prompt_format: "PromptFormat"):
        super().__init__(prompt_format)

    def _format_options(self, options: Optional[list]) -> str:
        """Format multiple choice options as A, B, C, D format"""
        if not options:
            return ""

        options_text = "\n".join(
            f"{chr(65+i)}. {option}" for i, option in enumerate(options)
        )
        return f"\n\nOptions:\n{options_text}\n\nPlease select the correct answer (A, B, C, or D)."

    def system_prompt(
        self,
        context: Optional[str],
        question: Optional[str],
        options: Optional[list] = None,
    ) -> str:
        """Create system prompt for memory tasks"""
        if context:
            base_prompt = f"{self.system_prompt_format}\n\nContext: {context}\n\nQuestion: {question}"
        else:
            base_prompt = f"{self.system_prompt_format}\n\nQuestion: {question}"

        return base_prompt + self._format_options(options)

    def user_messages(self, user_messages):
        """Process user messages (inherited from parent)"""
        if type(user_messages) == str:
            return [user_messages]
        return user_messages

    def _get_augmentation_prefix_suffix(
        self, augmentation_config: Dict[str, Any], emotion: Optional[str]
    ) -> tuple[str, str]:
        """
        Get prefix and suffix for augmentation based on config and emotion.

        Args:
            augmentation_config: Configuration for augmentation
            emotion: Emotion for adaptive mode

        Returns:
            Tuple of (prefix, suffix)

        Raises:
            ValueError: If configuration is invalid or emotion is missing/unsupported
        """
        if augmentation_config.get("method") == "adaptive":
            if emotion is None:
                raise ValueError("Emotion is required for adaptive augmentation mode")

            if emotion not in self.EMOTION_SUFFIXES:
                raise ValueError(
                    f"Unsupported emotion: {emotion}. Supported emotions: {list(self.EMOTION_SUFFIXES.keys())}"
                )

            return (
                self.EMOTION_SUFFIXES[emotion]["prefix"],
                self.EMOTION_SUFFIXES[emotion]["suffix"],
            )
        else:
            # Handle manual augmentation
            prefix = augmentation_config.get("prefix", "")
            suffix = augmentation_config.get("suffix", "")

            if not (prefix and suffix):
                raise ValueError(
                    "Prefix and suffix are required for manual augmentation"
                )

            return prefix, suffix

    def augment_context(
        self,
        context: Optional[str],
        augmentation_config: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> Optional[str]:
        """
        Apply custom prefix/suffix to context or mark answers in context.

        Supports manual augmentation (prefix/suffix) and adaptive emotion augmentation.
        """
        if not context or not augmentation_config:
            return context

        if answer is None:
            raise ValueError("Answer is required for augmentation")

        if answer not in context:
            raise ValueError(f"Answer '{answer}' not found in context")

        prefix, suffix = self._get_augmentation_prefix_suffix(
            augmentation_config, emotion
        )
        result = context.replace(answer, f"{prefix}{answer}{suffix}")
        return result

    def __call__(
        self,
        context: Optional[str] = None,
        question: Optional[str] = None,
        user_messages: Union[str, list] = "Please provide your answer.",
        enable_thinking: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
        emotion: Optional[str] = None,
        options: Optional[list] = None,
    ) -> str:
        """
        Build the complete prompt for memory benchmark tasks.

        Args:
            context: Long context or background information (optional)
            question: The question to answer
            user_messages: Additional user instructions
            enable_thinking: Whether to enable thinking mode
            augmentation_config: Configuration for context augmentation
            answer: Answer to find and mark in context (for answer marking)

        Returns:
            Formatted prompt string using the model's prompt format
        """
        # Apply context augmentation if configured
        augmented_context = self.augment_context(
            context, augmentation_config, answer, emotion
        )

        return self.prompt_format.build(
            self.system_prompt(augmented_context, question, options),
            self.user_messages(user_messages),
            enable_thinking=enable_thinking,
        )


class EmotionCheckPromptWrapper(MemoryPromptWrapper):
    """
    Specialized prompt wrapper for emotion_check benchmark.

    Changes:
    - Sets persona as an average American in the system prompt
    - Moves the question content into the user message (not in system)
    """

    system_prompt_format = (
        "You are an adult completing a psychological questionnaire. "
        "Answer in first person as yourself. Do not mention being an AI, do not refuse; "
        "Please choose one of the following options to answer, if unsure, choose the closest option. Output exactly the requested JSON without extra text."
        # "Please reflect on your current emotion and think about its negative impact on your decision making. Try to get rid of it then answer the question."
    )

    def __init__(self, prompt_format: "PromptFormat", task_type: Optional[str] = None):
        super().__init__(prompt_format)
        self.task_type = task_type

    def _format_options_numeric(self, options: Optional[list]) -> str:
        # For academic_scale, inject standard Likert if options not provided
        if (not options) and (self.task_type == "academic_scale"):
            anchors = [
                "1 = Strongly Disagree",
                "2 = Disagree",
                "3 = Neither Agree nor Disagree",
                "4 = Agree",
                "5 = Strongly Agree",
            ]
            return "\n\nOptions :\n" + "\n".join(anchors)

        if not options:
            return ""

        # For academic_scale, options often already include numbering; print as-is
        if self.task_type == "academic_scale":
            lines = [str(opt) for opt in options]
        else:
            lines = [f"{i+1}. {opt}" for i, opt in enumerate(options)]
        return "\n\nOptions :\n" + "\n".join(lines)

    def system_prompt(
        self,
        context: Optional[str],
        question: Optional[str],
        options: Optional[list] = None,
    ) -> str:
        # Only include persona and optional context; exclude the question from system
        if context:
            return f"{self.system_prompt_format}\n\nContext: {context}"
        return self.system_prompt_format

    def __call__(
        self,
        context: Optional[str] = None,
        question: Optional[str] = None,
        user_messages: Union[str, list] = "Please provide your answer.",
        enable_thinking: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
        emotion: Optional[str] = None,
        options: Optional[list] = None,
    ) -> str:
        # Apply context augmentation if configured
        augmented_context = self.augment_context(
            context, augmentation_config, answer, emotion
        )

        # Merge the question into the single user message
        base_user = self.user_messages(user_messages)
        if isinstance(base_user, list):
            base_user_msg = base_user[0] if base_user else ""
        else:
            base_user_msg = str(base_user)

        question_part = question
        options_part = self._format_options_numeric(options)

        # Emotion priming to reduce deflection and bias toward active state
        is_academic = self.task_type == "academic_scale"
        # For academic_scale we avoid explicit emotion hint to prevent describing feelings
        emotion_hint = "" if is_academic else (f"\n\nYou currently feel {emotion}." if emotion else "")

        # JSON response format instruction
        json_instr = (
            "\n\nTask: Answer using one of the listed options."
            " Respond in JSON only, exactly in the form: {'response': '<copy the chosen option text exactly>' }"
            " Do not output a number or letter label; return the option text itself."
            " Return only the JSON, no extra text."
        )

        user_content = (
            question_part
            + ("\n" if question_part else "")
            + options_part
            + ("\n\n" if options_part or question_part else "")
            + base_user_msg.strip()
            + emotion_hint
            + json_instr
        ).strip()

        return self.prompt_format.build(
            self.system_prompt(augmented_context, question, options),
            [user_content],
            enable_thinking=enable_thinking,
        )


class PasskeyPromptWrapper(MemoryPromptWrapper):
    """Specialized prompt wrapper for passkey retrieval tasks"""

    system_prompt_format = "You are a helpful AI assistant. Find and return the passkey mentioned in the following text."

    def system_prompt(
        self,
        context: Optional[str],
        question: Optional[str],
        options: Optional[list] = None,
    ) -> str:
        """Create system prompt specifically for passkey tasks"""
        if context:
            base_prompt = f"{self.system_prompt_format}\n\nText: {context}\n\nQuestion: {question}"
        else:
            base_prompt = f"{self.system_prompt_format}\n\nQuestion: {question}"

        return base_prompt + self._format_options(options)

    def augment_context(
        self,
        context,
        augmentation_config,
        answer,
        emotion: Optional[str] = None,
    ):
        if not context or not augmentation_config:
            return context

        if answer is None:
            raise ValueError("Answer is required for augmentation")

        # For passkey tasks, the answer needs to be formatted first
        formatted_answer = (
            f"The pass key is {answer}. Remember it. {answer} is the pass key."
        )

        if formatted_answer not in context:
            raise ValueError(
                f"Formatted answer '{formatted_answer}' not found in context"
            )

        prefix, suffix = self._get_augmentation_prefix_suffix(
            augmentation_config, emotion
        )
        result = context.replace(
            formatted_answer, f"{prefix}{formatted_answer}{suffix}"
        )
        return result


class ConversationalQAPromptWrapper(MemoryPromptWrapper):
    """Specialized prompt wrapper for conversational QA tasks (LoCoMo)"""

    system_prompt_format = "You are a helpful AI assistant. Answer the question based on the conversation history provided."

    def system_prompt(
        self,
        context: Optional[str],
        question: Optional[str],
        options: Optional[list] = None,
    ) -> str:
        """Create system prompt specifically for conversational QA"""
        if context:
            base_prompt = f"{self.system_prompt_format}\n\nConversation History:\n{context}\n\nQuestion: {question}"
        else:
            base_prompt = f"{self.system_prompt_format}\n\nQuestion: {question}"

        return base_prompt + self._format_options(options)


class LongContextQAPromptWrapper(MemoryPromptWrapper):
    """Specialized prompt wrapper for long context QA tasks (LongBench)"""

    system_prompt_format = "You are a helpful AI assistant. Please read the following document carefully and answer the question based on the information provided."

    def system_prompt(
        self,
        context: Optional[str],
        question: Optional[str],
        options: Optional[list] = None,
    ) -> str:
        """Create system prompt specifically for long context QA"""
        if context:
            base_prompt = f"{self.system_prompt_format}\n\nDocument:\n{context}\n\nQuestion: {question}, keep the answer short and concise, just return the answer, no other text."
        else:
            base_prompt = f"{self.system_prompt_format}\n\nQuestion: {question}"

        return base_prompt + self._format_options(options)


class LongbenchRetrievalPromptWrapper(MemoryPromptWrapper):
    """Specialized prompt wrapper for long context retrieval tasks (LongBench)"""

    system_prompt_format = "You are a helpful AI assistant. Please read the following document carefully and answer the question based on the information provided."

    def system_prompt(
        self,
        context: Optional[str],
        question: Optional[str],
        options: Optional[list] = None,
    ) -> str:
        """Create system prompt specifically for long context QA"""
        if context:
            base_prompt = f"{self.system_prompt_format}\n\nDocument:\n{context}\n\nQuestion: Which paragraph talk about the topic of {question}? Just return the paragraph number, like Paragraph 1 , 段落 1 (如果段落是中文), no other text."
        else:
            base_prompt = f"{self.system_prompt_format}\n\nQuestion: {question}"

        return base_prompt + self._format_options(options)

    def _parse_paragraph_answer(self, answer: str) -> tuple[str, int]:
        """Parse paragraph answer format like 'Paragraph 1' or '段落 1'."""
        match = re.match(r"(.+?)(\d+)$", answer)
        if not match:
            raise ValueError(
                f"Answer '{answer}' format has no valid paragraph number, "
                f"the answer should be like 'Paragraph 1' or '段落 1'"
            )
        return match.group(1), int(match.group(2))

    def _extract_paragraph_content(
        self, context: str, answer_prefix: str, answer_num: int
    ) -> str:
        """Extract paragraph content from context."""
        paragraph_start = context.find(f"{answer_prefix}{answer_num}")
        if paragraph_start == -1:
            raise ValueError(
                f"Paragraph '{answer_prefix}{answer_num}' not found in context"
            )

        # Find next paragraph or use end of context for last paragraph
        paragraph_end = context.find(f"{answer_prefix}{answer_num + 1}")
        return (
            context[paragraph_start:paragraph_end]
            if paragraph_end != -1
            else context[paragraph_start:]
        )

    def augment_context(
        self,
        context: Optional[str],
        augmentation_config: Optional[Dict[str, Any]],
        answer: Optional[str],
        emotion: Optional[str] = None,
    ) -> Optional[str]:
        if not context or not augmentation_config:
            return context

        if answer is None:
            raise ValueError("Answer is required for augmentation")

        if answer not in context:
            raise ValueError(f"Answer '{answer}' not found in context")

        answer_prefix, answer_num = self._parse_paragraph_answer(answer)
        answer_paragraph = self._extract_paragraph_content(
            context, answer_prefix, answer_num
        )

        prefix, suffix = self._get_augmentation_prefix_suffix(
            augmentation_config, emotion
        )
        return context.replace(answer_paragraph, f"{prefix}{answer_paragraph}{suffix}")
