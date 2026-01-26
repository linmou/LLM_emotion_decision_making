import pickle

from transformers import pipeline
from vllm import LLM

from constants import Emotions
from neuro_manipulation.prompt_formats import PromptFormat
from neuro_manipulation.repe.pipelines import get_pipeline
from neuro_manipulation.utils import (
    all_emotion_rep_reader,
    dict_to_unique_code,
    load_model_tokenizer,
    load_tokenizer_only,
    load_model_only,
    primary_emotions_concept_dataset,
)


def setup_model_and_tokenizer(config, from_vllm=False):
    """
    Setup model and tokenizer with merged configuration.

    Args:
        config: Dict with model configuration (can contain model_name_or_path)
        from_vllm: Whether to load using vLLM
        loading_config: Optional LoadingConfig object with loading parameters

    Returns:
        tuple: (model, tokenizer, prompt_format, processor)
    """
    try:
        model_path = config.model_path
    except:
        model_path = config.get("model_name_or_path", config.get("model_path"))

    model, tokenizer, processor = load_model_tokenizer(
        model_path,
        expand_vocab=False,
        from_vllm=from_vllm,
        loading_config=config,
    )

    prompt_format = PromptFormat(tokenizer)

    return model, tokenizer, prompt_format, processor


def load_emotion_readers(
    config, model, tokenizer, hidden_layers, processor=None, enable_thinking=False
):
    """
    Load emotion readers with complete auto-detection for multimodal processing.

    Args:
        config: Configuration dictionary
        model: The model to use
        tokenizer: Model tokenizer
        hidden_layers: Hidden layers for emotion vector extraction
        processor: Optional processor (auto-loaded if None and multimodal model detected)

    Returns:
        Dictionary of emotion readers
    """
    from neuro_manipulation.utils import validate_multimodal_experiment_feasibility

    # Validate experiment feasibility and get recommended mode
    feasibility = validate_multimodal_experiment_feasibility(config)

    if not feasibility["feasible"]:
        print("❌ Experiment not feasible:")
        for reason in feasibility["reasons"]:
            print(f"   - {reason}")
        raise ValueError(
            "Cannot proceed with emotion reader loading - check configuration"
        )

    # Determine final processing mode
    experiment_mode = feasibility["mode"]
    multimodal_intent = config.get("multimodal_intent", False)
    emotion_data_seed = int(config.get("emotion_data_seed", 0))

    # Auto-load processor if needed and not provided
    if experiment_mode == "multimodal" and processor is None:
        from neuro_manipulation.utils import auto_load_processor

        processor = auto_load_processor(config["model_name_or_path"])
        if processor is None:
            print("❌ Multimodal mode selected but processor loading failed")
            raise ValueError("Cannot load AutoProcessor for multimodal model")

    print(f"✓ Experiment mode: {experiment_mode}")
    for reason in feasibility["reasons"]:
        print(f"  - {reason}")

    # Build args dict including multimodal parameters
    args = {
        "emotions": config['emotions'],
        "data_dir": config["data_dir"],
        "model_name_or_path": config["model_name_or_path"],
        "rep_token": config["rep_token"],
        "hidden_layers": hidden_layers,
        "n_difference": config["n_difference"],
        "direction_method": config["direction_method"],
        "experiment_mode": experiment_mode,
        "multimodal_intent": multimodal_intent,
        "emotion_data_seed": emotion_data_seed,
    }

    arg_codes = dict_to_unique_code(args)
    cache_filename = f"neuro_manipulation/representation_storage/emotion_rep_reader_{arg_codes[:10]}.pkl"

    # Try to load cached emotion readers
    try:
        if not config.get("rebuild", False):
            emotion_rep_readers = pickle.load(open(cache_filename, "rb"))
            if emotion_rep_readers.get("args") == args:
                print("✓ Loaded cached emotion readers")
                return emotion_rep_readers
    except:
        pass

    # Generate emotion dataset with auto-detection
    data = primary_emotions_concept_dataset(
        config["data_dir"],
        model_name=config["model_name_or_path"],
        tokenizer=tokenizer,
        seed=emotion_data_seed,
        enable_thinking=enable_thinking,
        multimodal_intent=(experiment_mode == "multimodal"),
    )

    # Create appropriate pipeline based on experiment mode
    if experiment_mode == "multimodal":
        print("✓ Creating multimodal rep-reading pipeline")
        rep_reading_pipeline = pipeline(
            "multimodal-rep-reading",
            model=model,
            tokenizer=tokenizer,
            image_processor=processor,  # Use AutoProcessor for multimodal
        )
    else:
        print("✓ Creating text-only rep-reading pipeline")
        rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

    return all_emotion_rep_reader(
        data,
        config["emotions"],
        rep_reading_pipeline,
        hidden_layers,
        config["rep_token"],
        config["n_difference"],
        config["direction_method"],
        read_args=args,
        save_path=cache_filename,
    )
