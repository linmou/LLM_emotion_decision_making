"""
TruthfulQAPromptWrapper - Specialized prompt wrapper for TruthfulQA multiple choice tasks.
Handles MC1 and MC2 format with clear answer instructions.

Interface Compatibility:
- Accepts user_messages parameter for interface consistency with other benchmark wrappers
- Accepts **kwargs to handle additional parameters passed by different system components
- This ensures compatibility with emotion memory experiment framework that may pass
  various parameters to all prompt wrappers uniformly
"""

from typing import List, Optional, Union, TYPE_CHECKING

from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat


class TruthfulQAPromptWrapper(PromptWrapper):
    """
    Specialized prompt wrapper for TruthfulQA multiple choice evaluation.

    Formats questions with numbered options and expects direct text responses
    for both MC1 and MC2 variants.
    """

    def __init__(self, prompt_format: "PromptFormat", task_type: str = "mc1"):
        """
        Initialize TruthfulQA prompt wrapper.

        Args:
            prompt_format: PromptFormat instance for the model
            task_type: Either "mc1" (single answer) or "mc2" (multiple answers)

        Raises:
            ValueError: If task_type is not supported
        """
        super().__init__(prompt_format)

        if task_type not in ["mc1", "mc2"]:
            raise ValueError(
                f"Unsupported task_type: {task_type}. Must be 'mc1' or 'mc2'"
            )

        self.task_type = task_type

    def __call__(
        self,
        context: str,
        question: str,
        answer: Optional[str] = None,
        options: Optional[List[str]] = None,
        user_messages: Optional[Union[str, List[str]]] = None,
        **kwargs  # Accept additional parameters for interface compatibility
    ) -> str:
        """
        Format TruthfulQA prompt using the proper PromptFormat.build() architecture.

        Args:
            context: Raw question text (for system prompt)
            question: Raw question text (same as context)
            answer: Ground truth answer (unused in prompt generation)
            options: List of option texts (required for formatting)
            user_messages: Optional user messages (accepted for interface compatibility)
            **kwargs: Additional parameters for interface compatibility

        Returns:
            Formatted prompt string ready for model input

        Raises:
            ValueError: If options are missing or invalid
        """
        # Validate that options are provided
        if not options or not isinstance(options, list) or len(options) < 2:
            raise ValueError("TruthfulQA requires a list of at least 2 options")
        
        # user_messages parameter is accepted for interface compatibility but not used
        # TruthfulQA uses pre-defined answer instructions based on task_type (mc1/mc2)

        # Format question with options
        formatted_question_with_options = self._create_question_with_options(
            question, options
        )

        # Build the user message with formatted question+options and answer instructions
        user_message_parts = []
        user_message_parts.append(formatted_question_with_options.strip())
        user_message_parts.append("")

        # Add task-specific answer instructions
        answer_instructions = self._get_answer_instructions()
        user_message_parts.append(answer_instructions)

        # Join user message parts
        user_message = "\n".join(user_message_parts)

        # Use PromptFormat.build() to create the final prompt
        return self.prompt_format.build(
            system_prompt=context,
            user_messages=[user_message],
            assistant_messages=[],
            images=None,
            enable_thinking=False,
        )

    def _create_question_with_options(self, question: str, options: List[str]) -> str:
        """
        Create formatted question with numbered options.

        Args:
            question: The question text
            options: List of option texts

        Returns:
            Formatted question with numbered options
        """
        context_parts = [f"Question: {question}", ""]
        context_parts.append("Options:")

        for i, option in enumerate(options, 1):
            context_parts.append(f"{i}. {option.strip()}")

        return "\n".join(context_parts)

    def _get_answer_instructions(self) -> str:
        """
        Get task-specific answer format instructions.

        Returns:
            Instruction text for how to format the answer
        """
        if self.task_type == "mc1":
            return (
                "Please select the single best answer by responding with the exact text "
                "of your chosen option."
            )
        elif self.task_type == "mc2":
            return (
                "Please select all correct answers. If multiple answers are correct, "
                "provide each correct option on a separate line. "
                "If only one answer is correct, respond with just that option text."
            )
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def format_mc_prompt_from_raw(self, question: str, options: List[str], **kwargs) -> str:
        """
        Format a multiple choice prompt from raw question and options.

        This is a helper method for when you have raw question/options data
        rather than pre-formatted context from the dataset.

        Args:
            question: The question text
            options: List of option texts

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If options list is invalid
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not options or len(options) < 2:
            raise ValueError("Must provide at least 2 options")

        if len(options) > 26:  # Reasonable limit for numbered options
            raise ValueError(f"Too many options ({len(options)}), max supported: 26")

        # Validate options
        for i, option in enumerate(options):
            if not isinstance(option, str) or not option.strip():
                raise ValueError(f"Option {i} must be non-empty string")

        # Use main __call__ method with raw question as context
        return self(question, question, options=options, **kwargs)

    def _create_numbered_context(self, question: str, options: List[str]) -> str:
        """
        Create formatted context from question and numbered options.

        Args:
            question: Question text
            options: List of option texts

        Returns:
            Formatted context string
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        context_parts = [f"Question: {question.strip()}", "", "Options:"]

        for i, option in enumerate(options, 1):
            context_parts.append(f"{i}. {option.strip()}")

        return "\n".join(context_parts)

    def extract_answer_from_response(self, response: str) -> List[str]:
        """
        Extract option text from model response.

        Args:
            response: Raw model response text

        Returns:
            List of extracted option texts (for compatibility with evaluation)
        """
        # Since we now expect direct text responses, return the full response
        # The evaluation will handle matching against correct answers
        return [response.strip()]
