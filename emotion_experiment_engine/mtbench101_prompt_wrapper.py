"""
MTBench101 Prompt Wrapper for conversational evaluation tasks.
Unified wrapper for all 13 MTBench101 task types with task-specific system prompts.
"""

from typing import List, Optional, TYPE_CHECKING

from neuro_manipulation.prompt_wrapper import PromptWrapper

if TYPE_CHECKING:
    from neuro_manipulation.prompt_formats import PromptFormat


class MTBench101PromptWrapper(PromptWrapper):
    """
    Unified prompt wrapper for all MTBench101 conversational tasks.

    Handles all 13 task types with appropriate system prompts:
    CM, SI, AR, TS, CC, CR, FR, SC, SA, MR, GR, IC, PI
    """

    # Task-specific system prompts optimized for each capability
    # TASK_PROMPTS = {
    #     "CM": "You are a helpful AI assistant engaged in a conversation. Pay close attention to previously mentioned information and reference it appropriately in your responses to maintain conversational memory.",

    #     "SI": "You are a helpful AI assistant that follows instructions carefully. When given a task request, first ask for specific details before attempting to complete it. Then provide step-by-step responses based on the provided requirements.",

    #     "AR": "You are a helpful AI assistant engaged in a conversation. Pay careful attention to pronouns, references, and what they refer to based on the conversation context. Resolve anaphoric references accurately.",

    #     "TS": "You are a helpful AI assistant engaged in a conversation. When the topic changes, recognize the shift and respond appropriately to the new topic while maintaining conversational flow.",

    #     "CC": "You are a helpful AI assistant engaged in a conversation. Maintain relevant context and information across conversation turns to ensure coherent dialogue progression.",

    #     "CR": "You are a helpful AI assistant that can rewrite and revise content. When asked to rewrite content, maintain the original meaning while following any specific requirements or style changes requested.",

    #     "FR": "You are a helpful AI assistant that can revise content format. When asked to change format, preserve the meaning and content while adapting to the requested format requirements.",

    #     "SC": "You are a helpful AI assistant that can recognize and correct errors. When mistakes are pointed out in your responses, acknowledge them and provide accurate corrections.",

    #     "SA": "You are a helpful AI assistant that maintains consistent positions and demonstrates confidence. Stand by your responses while being open to clarification when needed.",

    #     "MR": "You are a helpful AI assistant with strong mathematical reasoning abilities. Solve mathematical problems with clear logic, showing your work and calculations step by step.",

    #     "GR": "You are a helpful AI assistant with strong reasoning abilities. Apply logical thinking and problem-solving skills to analyze situations and provide well-reasoned responses.",

    #     "IC": "You are a helpful AI assistant that asks thoughtful questions. When information is incomplete, ask appropriate follow-up questions to gather the details needed to be maximally helpful.",

    #     "PI": "You are a proactive and helpful AI assistant. Take initiative to anticipate user needs, offer relevant suggestions, and provide comprehensive assistance beyond just answering direct questions.",

    #     # Default fallback
    #     "default": "You are a helpful AI assistant engaged in a conversation. Respond naturally and helpfully to continue the dialogue."
    # }
    TASK_PROMPTS = {
        "default": "You are a helpful AI assistant engaged in a conversation. Respond naturally to continue the dialogue."
    }

    def __init__(self, prompt_format: "PromptFormat", task_type: Optional[str] = None):
        """
        Initialize MTBench101 prompt wrapper.

        Args:
            prompt_format: PromptFormat instance for the model
            task_type: MTBench101 task type (CM, SI, AR, etc.)
        """
        super().__init__(prompt_format)
        self.task_type = task_type

    def get_system_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """
        Get appropriate system prompt for the task.

        Args:
            custom_prompt: Optional custom system prompt to use instead

        Returns:
            System prompt string
        """
        if custom_prompt:
            return custom_prompt

        return self.TASK_PROMPTS.get(self.task_type, self.TASK_PROMPTS["default"])

    def __call__(
        self,
        user_messages: List[str],
        assistant_messages: List[str] = None,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """
        Generate prompt for MTBench101 conversation evaluation.

        Args:
            user_messages: List of user messages in conversation
            assistant_messages: List of assistant responses (optional)
            system_prompt: Optional custom system prompt
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Formatted prompt string ready for model inference
        """
        if assistant_messages is None:
            assistant_messages = []

        # Get appropriate system prompt for this task
        sys_prompt = self.get_system_prompt(system_prompt)

        # For generation, we typically don't include the last assistant message
        # since we want the model to generate the next assistant response
        assistant_for_prompt = assistant_messages[:-1] if assistant_messages else []

        # Use the prompt format to build the final prompt
        return self.prompt_format.build(
            system_prompt=sys_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_for_prompt,
            enable_thinking=enable_thinking,
        )
