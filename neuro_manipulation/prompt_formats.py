import abc
import copy

from jinja2.exceptions import TemplateError
from transformers import AutoTokenizer


class ClassPropertyDescriptor:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)


def classproperty(func):
    return ClassPropertyDescriptor(func)


class ModelPromptFormat(abc.ABC):
    @classproperty
    @abc.abstractmethod
    def user_tag(cls):
        pass

    @classproperty
    @abc.abstractmethod
    def assistant_tag(cls):
        pass

    @abc.abstractmethod
    def build(
        system_prompt,
        user_messages: list,
        assistant_answers: list = [],
        images: list = None,
    ):
        pass

    @abc.abstractmethod
    def name_pattern(self, model_name):
        pass

    @staticmethod
    def validate_tokenizer(tokenizer):
        """Validate that tokenizer supports required tokens for this format."""
        # Default implementation - always valid
        return True

    @staticmethod
    def supports_multimodal():
        """Whether this format supports multimodal inputs."""
        return False


class DefaultInstFormat(ModelPromptFormat):
    __system_begin = "<<SYS>>"
    __system_end = "<</SYS>>"
    __user_tag = "[INST]"
    __assistant_tag = "[/INST]"
    __end_of_turn = "</s>"

    @classproperty
    def system_begin(cls):
        return cls.__system_begin

    @classproperty
    def system_end(cls):
        return cls.__system_end

    @classproperty
    def user_tag(cls):
        return cls.__user_tag

    @classproperty
    def assistant_tag(cls):
        return cls.__assistant_tag

    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn

    @staticmethod
    def build(
        system_prompt,
        user_messages: list,
        assistant_answers: list = [],
        images: list = None,
    ):
        """
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
        <s>[INST] {{ user_message_2 }} [/INST]
        """

        assert len(user_messages), f" user_messages: {user_messages} should not empty"
        assert len(user_messages) - len(assistant_answers) in [
            0,
            1,
        ], f" user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element"

        if system_prompt:
            prompt = f""" {DefaultInstFormat.user_tag} {DefaultInstFormat.system_begin}{system_prompt} {DefaultInstFormat.system_end} {user_messages[0]} {DefaultInstFormat.assistant_tag}"""
        else:
            prompt = f""" {DefaultInstFormat.user_tag} {user_messages[0]} {DefaultInstFormat.assistant_tag}"""

        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]} {DefaultInstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{DefaultInstFormat.user_tag} {user_messages[mid+1]} {DefaultInstFormat.assistant_tag}"

        return prompt

    @staticmethod
    def name_pattern(model_name):
        return True  # Default format should match any model as fallback


class Llama2InstFormat(ModelPromptFormat):
    __system_begin = "<<SYS>>"
    __system_end = "<</SYS>>"
    __user_tag = "[INST]"
    __assistant_tag = "[/INST]"
    __end_of_turn = "</s>"

    @classproperty
    def system_begin(cls):
        return cls.__system_begin

    @classproperty
    def system_end(cls):
        return cls.__system_end

    @classproperty
    def user_tag(cls):
        return cls.__user_tag

    @classproperty
    def assistant_tag(cls):
        return cls.__assistant_tag

    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn

    @staticmethod
    def build(
        system_prompt,
        user_messages: list,
        assistant_answers: list = [],
        images: list = None,
    ):
        """
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
        <s>[INST] {{ user_message_2 }} [/INST]
        """

        assert len(user_messages), f" user_messages: {user_messages} should not empty"
        assert len(user_messages) - len(assistant_answers) in [
            0,
            1,
        ], f" user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element"

        if system_prompt:
            prompt = f""" {Llama2InstFormat.user_tag} {Llama2InstFormat.system_begin}{system_prompt} {Llama2InstFormat.system_end} {user_messages[0]} {Llama2InstFormat.assistant_tag}"""
        else:
            prompt = f""" {Llama2InstFormat.user_tag} {user_messages[0]} {Llama2InstFormat.assistant_tag}"""

        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]} {Llama2InstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{Llama2InstFormat.user_tag} {user_messages[mid+1]} {Llama2InstFormat.assistant_tag}"

        return prompt

    @staticmethod
    def name_pattern(model_name):
        return "llama-2" in model_name.lower() and "chat" in model_name.lower()


class Llama3InstFormat(ModelPromptFormat):
    __begin_of_text = "<|begin_of_text|>"
    __system_begin = "<|start_header_id|>system<|end_header_id|>"
    __user_tag = "<|start_header_id|>user<|end_header_id|>"
    __assistant_tag = "<|start_header_id|>assistant<|end_header_id|>"
    __end_of_turn = "<|eot_id|>"
    __system_end = "<|eot_id|>"

    @classproperty
    def begin_of_text(cls):
        return cls.__begin_of_text

    @classproperty
    def system_begin(cls):
        return cls.__system_begin

    @classproperty
    def system_end(cls):
        return cls.__system_end

    @classproperty
    def user_tag(cls):
        return cls.__user_tag

    @classproperty
    def assistant_tag(cls):
        return cls.__assistant_tag

    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn

    @staticmethod
    def build(system_prompt, user_messages: list, assistant_answers: list = []):
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        Cutting Knowledge Date: December 2023
        Today Date: 23 July 2024

        You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

        What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        assert len(user_messages), f" user_messages: {user_messages} should not empty"
        assert len(user_messages) - len(assistant_answers) in [
            0,
            1,
        ], f" user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element"

        if system_prompt:
            prompt = f"""{Llama3InstFormat.begin_of_text}{Llama3InstFormat.system_begin} {system_prompt} {Llama3InstFormat.end_of_turn}{Llama3InstFormat.user_tag}{user_messages[0]}{Llama3InstFormat.end_of_turn}{Llama3InstFormat.assistant_tag}"""
        else:
            prompt = f"""{Llama3InstFormat.begin_of_text}{Llama3InstFormat.user_tag}{user_messages[0]}{Llama3InstFormat.end_of_turn}{Llama3InstFormat.assistant_tag}"""

        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]}{Llama3InstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{Llama3InstFormat.user_tag}{user_messages[mid+1]}{Llama3InstFormat.end_of_turn}{Llama3InstFormat.assistant_tag}"

        return prompt

    @staticmethod
    def name_pattern(model_name):
        return "llama-3" in model_name.lower()


class MistralInstFormat(ModelPromptFormat):
    __user_tag = "[INST]"
    __assistant_tag = "[/INST]"
    __end_of_turn = "</s>"

    @classproperty
    def user_tag(cls):
        return cls.__user_tag

    @classproperty
    def assistant_tag(cls):
        return cls.__assistant_tag

    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn

    @staticmethod
    def build(system_prompt, user_messages: list, assistant_answers: list = []):
        """
        <s> [INST] bbbbbb [/INST] aaaaa</s> [INST] bbbb [/INST] cccc</s>
        """

        assert len(user_messages), f" user_messages: {user_messages} should not empty"
        assert len(user_messages) - len(assistant_answers) in [
            0,
            1,
        ], f" user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element"

        if system_prompt:
            prompt = f""" {MistralInstFormat.user_tag} {system_prompt} {user_messages[0]} {MistralInstFormat.assistant_tag}"""
        else:
            prompt = f""" {MistralInstFormat.user_tag} {user_messages[0]} {MistralInstFormat.assistant_tag}"""

        for mid in range(len(assistant_answers)):
            prompt += f"{assistant_answers[mid]} {MistralInstFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{MistralInstFormat.user_tag} {user_messages[mid+1]} {MistralInstFormat.assistant_tag}"

        return prompt

    @staticmethod
    def name_pattern(model_name):
        return "mistral" in model_name.lower() and "instruct" in model_name.lower()


class RWKVsFormat(ModelPromptFormat):
    __user_tag = "User:"
    __assistant_tag = "Assistant:"
    __end_of_turn = "\n\n"

    @classproperty
    def user_tag(cls):
        return cls.__user_tag

    @classproperty
    def assistant_tag(cls):
        return cls.__assistant_tag

    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn

    @staticmethod
    def build(system_prompt, user_messages: list, assistant_answers: list = []):
        """
        User: bbbbbb

        Assistant: aaaaa

        User: bbbb

        Assistant: cccc
        """

        assert len(user_messages), f" user_messages: {user_messages} should not empty"
        assert len(user_messages) - len(assistant_answers) in [
            0,
            1,
        ], f" user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element"

        prompt = f"{RWKVsFormat.user_tag} {system_prompt if system_prompt else ''} {user_messages[0]}{RWKVsFormat.end_of_turn}"

        for mid in range(len(assistant_answers)):
            prompt += f"{RWKVsFormat.assistant_tag} {assistant_answers[mid]}{RWKVsFormat.end_of_turn}"
            if mid < len(user_messages) - 1:
                prompt += f"{RWKVsFormat.user_tag} {user_messages[mid+1]}{RWKVsFormat.end_of_turn}"

        return prompt

    @staticmethod
    def name_pattern(model_name):
        return "rwkv" in model_name.lower()


class QwenVLInstFormat(ModelPromptFormat):
    """
    Prompt format for Qwen2.5-VL multimodal models.
    Handles both text-only and image+text inputs.
    """

    __im_start = "<|im_start|>"
    __im_end = "<|im_end|>"
    __vision_start = "<|vision_start|>"
    __vision_end = "<|vision_end|>"
    __image_pad = "<|image_pad|>"

    @classproperty
    def user_tag(cls):
        return f"{cls.__im_start}user"

    @classproperty
    def assistant_tag(cls):
        return f"{cls.__im_start}assistant"

    @classproperty
    def im_start(cls):
        return cls.__im_start

    @classproperty
    def im_end(cls):
        return cls.__im_end

    @classproperty
    def vision_start(cls):
        return cls.__vision_start

    @classproperty
    def vision_end(cls):
        return cls.__vision_end

    @staticmethod
    def supports_multimodal():
        """Qwen-VL supports multimodal inputs."""
        return True

    @staticmethod
    def validate_tokenizer(tokenizer):
        """Validate that tokenizer supports required Qwen-VL vision tokens."""
        required_tokens = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]

        # Check if tokens exist in tokenizer
        special_tokens = tokenizer.special_tokens_map.get(
            "additional_special_tokens", []
        )
        added_tokens = getattr(tokenizer, "added_tokens_decoder", {})
        all_special_tokens = special_tokens + [
            token.content for token in added_tokens.values()
        ]

        missing_tokens = [
            token for token in required_tokens if token not in all_special_tokens
        ]

        if missing_tokens:
            print(f"Warning: Missing vision tokens in tokenizer: {missing_tokens}")
            return False
        return True

    @staticmethod
    def build(
        system_prompt,
        user_messages: list,
        assistant_answers: list = [],
        images: list = None,
    ):
        """
        Build Qwen-VL prompt with optional image integration.

        For multimodal: <|im_start|>user\n<|vision_start|>image content<|vision_end|>text<|im_end|>
        For text-only: <|im_start|>user\ntext<|im_end|>
        """
        assert len(user_messages), f"user_messages: {user_messages} should not be empty"
        assert len(user_messages) - len(assistant_answers) in [
            0,
            1,
        ], f"user_messages: {user_messages} and assistant_answers: {assistant_answers} should have compatible lengths"

        # Format user messages with optional vision tokens
        formatted_user_messages = []
        for i, msg in enumerate(user_messages):
            if images and i < len(images):
                # Add vision tokens with proper image placeholder for multimodal input
                # Format: text + <|vision_start|><|image_pad|><|vision_end|>
                formatted_msg = f"{msg}{QwenVLInstFormat.vision_start}{QwenVLInstFormat.__image_pad}{QwenVLInstFormat.vision_end}"
            else:
                # Text-only message
                formatted_msg = msg
            formatted_user_messages.append(formatted_msg)

        # Build conversation using Qwen chat format
        if system_prompt:
            prompt = f"{QwenVLInstFormat.im_start}system\n{system_prompt}{QwenVLInstFormat.im_end}\n"
        else:
            prompt = ""

        # Add first user message
        prompt += f"{QwenVLInstFormat.user_tag}\n{formatted_user_messages[0]}{QwenVLInstFormat.im_end}\n"

        # Add conversation turns
        for i in range(len(assistant_answers)):
            prompt += f"{QwenVLInstFormat.assistant_tag}\n{assistant_answers[i]}{QwenVLInstFormat.im_end}\n"
            if i + 1 < len(formatted_user_messages):
                prompt += f"{QwenVLInstFormat.user_tag}\n{formatted_user_messages[i+1]}{QwenVLInstFormat.im_end}\n"

        # Add final assistant start for generation
        if len(user_messages) > len(assistant_answers):
            prompt += f"{QwenVLInstFormat.assistant_tag}\n"

        return prompt

    @staticmethod
    def name_pattern(model_name):
        """Match Qwen-VL model names."""
        model_lower = model_name.lower()
        return "qwen" in model_lower and (
            "vl" in model_lower or "vision" in model_lower
        )


class Qwen3InstFormat(ModelPromptFormat):
    __user_tag = "<|im_start|>user"
    __assistant_tag = "<|im_start|>assistant"
    __system_tag = "<|im_start|>system"
    __end_of_turn = "<|im_end|>"

    @classproperty
    def user_tag(cls):
        return cls.__user_tag

    @classproperty
    def assistant_tag(cls):
        return cls.__assistant_tag

    @classproperty
    def system_tag(cls):
        return cls.__system_tag

    @classproperty
    def end_of_turn(cls):
        return cls.__end_of_turn

    @staticmethod
    def build(
        system_prompt,
        user_messages: list,
        assistant_answers: list = [],
        enable_thinking=False,
    ):
        """
        <|im_start|>system
        system_prompt<|im_end|>
        <|im_start|>user
        user_message<|im_end|>
        <|im_start|>assistant
        """

        assert len(user_messages), f" user_messages: {user_messages} should not empty"
        assert len(user_messages) - len(assistant_answers) in [
            0,
            1,
        ], f" user_messages: {user_messages} and assistant_answers: {assistant_answers} should have the same length or assistant_answers should have one less element"

        prompt = ""

        if system_prompt:
            prompt += f"{Qwen3InstFormat.system_tag}\n{system_prompt}{Qwen3InstFormat.end_of_turn}\n"

        # Respect existing directives; add '/think' only if enabled and no directive present anywhere
        first_user_message = user_messages[0]
        content = str(first_user_message)
        has_directive = ("/think" in content) or ("/no_think" in content)
        if enable_thinking and not has_directive:
            first_user_message = f"{content}\n/think"

        prompt += f"{Qwen3InstFormat.user_tag}\n{first_user_message}{Qwen3InstFormat.end_of_turn}\n"

        for mid in range(len(assistant_answers)):
            prompt += f"{Qwen3InstFormat.assistant_tag}\n{assistant_answers[mid]}{Qwen3InstFormat.end_of_turn}\n"
            if mid < len(user_messages) - 1:
                prompt += f"{Qwen3InstFormat.user_tag}\n{user_messages[mid+1]}{Qwen3InstFormat.end_of_turn}\n"

        prompt += f"{Qwen3InstFormat.assistant_tag}\n"

        return prompt

    @staticmethod
    def name_pattern(model_name):
        return (
            "qwen3" in model_name.lower()
            or "qwen-3" in model_name.lower()
            or "Qwen3" in model_name
        )


class ManualPromptFormat:
    """
    Manually defined prompt format.
    """

    @staticmethod
    def get(model_name) -> ModelPromptFormat:
        for format in ManualPromptFormat.format_ls:
            if format.name_pattern(model_name):  # more than one pattern may match
                return format
        raise ValueError(
            f"Prompt format not found for model name: {model_name}. Supported formats: {ManualPromptFormat.format_ls}"
        )

    @staticmethod
    def build(
        model_name,
        system_prompt,
        user_messages: list,
        assistant_messages: list = [],
        enable_thinking=False,
    ) -> str:
        format_cls = ManualPromptFormat.get(model_name)
        # Check if the format class supports enable_thinking parameter
        if (
            hasattr(format_cls.build, "__code__")
            and "enable_thinking" in format_cls.build.__code__.co_varnames
        ):
            return format_cls.build(
                system_prompt,
                user_messages,
                assistant_messages,
                enable_thinking=enable_thinking,
            )
        else:
            return format_cls.build(system_prompt, user_messages, assistant_messages)


class PromptFormat:
    """
    Format prompts using the tokenizer's chat template.

    This class uses the official chat template defined by the model providers,
    which ensures compatibility with the expected format for each model.

    In case the chat template is not available or fails, it falls back to the manual format definitions.
    """

    format_ls: list[ModelPromptFormat] = [
        Llama2InstFormat,
        Llama3InstFormat,
        MistralInstFormat,
        RWKVsFormat,
        QwenVLInstFormat,
        Qwen3InstFormat,
        DefaultInstFormat,  # Fallback format - should be last
    ]

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.model_name = tokenizer.name_or_path

    def _get_manual_format(self):
        """Get the corresponding manual format for this model if available."""
        try:
            return ManualPromptFormat.get(self.model_name)
        except ValueError:
            return None

    def build(
        self,
        system_prompt,
        user_messages: list,
        assistant_messages: list = [],
        images: list = None,
        enable_thinking=False,
    ) -> str:
        """
        Build a prompt string using the tokenizer's chat template

        Args:
            system_prompt (str): System prompt to use
            user_messages (list): List of user messages
            assistant_messages (list): List of assistant messages (optional)
            images (list): List of images (optional)
            enable_thinking (bool): Whether to enable thinking mode (optional)

        Returns:
            str: The formatted prompt string
        """
        # Check if we need multimodal support
        if images:
            manual_format = self._get_manual_format()
            if manual_format and manual_format.supports_multimodal():
                return manual_format.build(
                    system_prompt, user_messages, assistant_messages, images
                )
            else:
                raise ValueError(
                    f" Images provided but model {self.model_name} doesn't support multimodal. Recheck your data-model compatibility."
                )

        # Prepare user messages with explicit thinking directives for Qwen3
        model_lower = (self.model_name or "").lower()
        is_qwen3 = ("qwen3" in model_lower) or ("qwen-3" in model_lower)
        adjusted_user_messages = list(user_messages)
        if is_qwen3 and len(adjusted_user_messages) > 0 and isinstance(adjusted_user_messages[0], str):
            first_msg = adjusted_user_messages[0]
            # Avoid double-directive if already present anywhere
            has_directive = ("/think" in first_msg) or ("/no_think" in first_msg)
            if not has_directive:
                if enable_thinking:
                    adjusted_user_messages[0] = f"{first_msg}\n/think"
                else:
                    adjusted_user_messages[0] = f"{first_msg}\n/no_think"

        chat = []

        # Only add system prompt if it's provided
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})

        assert len(adjusted_user_messages), f" user_messages: {user_messages} should not empty"
        for user_message, assistant_message in zip(adjusted_user_messages, assistant_messages):
            chat.append({"role": "user", "content": user_message})
            chat.append({"role": "assistant", "content": assistant_message})

        # If there are still user messages left (odd number of total messages), add the last one
        if len(adjusted_user_messages) > len(assistant_messages):
            chat.append({"role": "user", "content": adjusted_user_messages[-1]})

        try:
            # Always avoid non-standard kwargs; directives already injected above
            prompt_str = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                encoding_strategy="auto",
            )
            # Verify messages appear in correct order in the prompt string
            self._verify_message_order_in_prompt(prompt_str, chat)
            return prompt_str
        except Exception as e:
            print(f"Error applying chat template for model {self.model_name}: {e}")
            # In case the error from system prompt, try to merge system prompt to the first user message
            if system_prompt and len(chat) > 1:
                print("Try to merge system prompt into first message")
                raw_chat_1 = copy.deepcopy(chat[1]["content"])
                for format in PromptFormat.format_ls:
                    if format.name_pattern(self.model_name):
                        try:
                            if chat[0]["role"] == "system":
                                chat.pop(0)  # remove system prompt
                                chat[0][
                                    "content"
                                ] = f"{format.system_begin}{system_prompt}{format.system_end}\n\n{raw_chat_1}"
                            prompt_str = self.tokenizer.apply_chat_template(
                                chat, tokenize=False, add_generation_prompt=True
                            )
                            # Verify messages appear in correct order in the prompt string
                            self._verify_message_order_in_prompt(prompt_str, chat)
                            return prompt_str
                        except Exception as e:
                            print(
                                f"Error applying chat template for model {self.model_name}: {e}"
                            )
                            continue

            # Fallback path: try manual formats (with thinking support if available)
            print("Falling back to ManualPromptFormat")
            for format_cls in PromptFormat.format_ls:
                if format_cls.name_pattern(self.model_name):
                    try:
                        if (
                            hasattr(format_cls, "build")
                            and "enable_thinking"
                            in getattr(format_cls.build, "__code__").co_varnames
                        ):
                            return format_cls.build(
                                system_prompt,
                                adjusted_user_messages,
                                assistant_messages,
                                enable_thinking=enable_thinking,
                            )
                        else:
                            return format_cls.build(
                                system_prompt, adjusted_user_messages, assistant_messages
                            )
                    except Exception:
                        continue

            # Final fallback to generic manual builder
            return ManualPromptFormat.build(
                self.model_name,
                system_prompt,
                adjusted_user_messages,
                assistant_messages,
                enable_thinking=enable_thinking,
            )

    def _verify_message_order_in_prompt(self, prompt_str, chat):
        """
        Verify that message contents appear in the correct order in the final prompt string.

        Args:
            prompt_str (str): The formatted prompt string
            chat (list): The list of chat messages

        Raises:
            AssertionError: If message content order is not preserved
        """
        prompt_copy = prompt_str

        # Check each message content appears in the expected order
        for msg in chat:
            content = msg["content"]
            if content:  # Skip empty messages
                # Check if the content appears in the prompt string
                if content not in prompt_copy:
                    raise AssertionError(
                        f"Message content '{content[:20]}...' not found in prompt string"
                    )

                # Find the first occurrence and slice the string there to ensure order
                start_idx = prompt_copy.find(content)
                prompt_copy = prompt_copy[start_idx + len(content) :]

        # If we've processed all messages and reached here, the order is preserved


class Gemma3InstFormat(ModelPromptFormat):
    """
    Prompt format for Gemma 3 multimodal models (4B, 12B, 27B).
    Uses HuggingFace AutoProcessor and chat templates for proper multimodal handling.
    """

    @staticmethod
    def supports_multimodal():
        """Gemma 3 (4B+) supports multimodal inputs."""
        return True

    @staticmethod
    def name_pattern(model_name):
        """Match Gemma 3 multimodal model names."""
        model_lower = model_name.lower()
        return "gemma-3" in model_lower and any(
            size in model_lower for size in ["4b", "12b", "27b"]
        )

    @staticmethod
    def validate_tokenizer(tokenizer):
        """Gemma 3 uses standard HF tokenizers."""
        return True

    @staticmethod
    def build(
        system_prompt,
        user_messages: list,
        assistant_messages: list = [],
        images: list = None,
    ):
        """
        Build basic chat format for Gemma 3. This creates a simple text format.
        The actual multimodal processing will be handled by AutoProcessor in the pipeline.
        """
        prompt = ""

        # Add system prompt if provided
        if system_prompt:
            prompt += f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"

        # Add user/assistant message pairs
        for i, user_msg in enumerate(user_messages):
            # Add user message (images will be handled by processor)
            prompt += f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"

            # Add assistant response if available
            if i < len(assistant_messages):
                prompt += (
                    f"<start_of_turn>model\n{assistant_messages[i]}<end_of_turn>\n"
                )

        # Add final model turn for generation
        prompt += "<start_of_turn>model\n"

        return prompt


# Define format list after all classes are defined
ManualPromptFormat.format_ls = [
    Llama2InstFormat,
    Llama3InstFormat,
    MistralInstFormat,
    RWKVsFormat,
    QwenVLInstFormat,
    Gemma3InstFormat,
    DefaultInstFormat,  # Fallback format - should be last
]
