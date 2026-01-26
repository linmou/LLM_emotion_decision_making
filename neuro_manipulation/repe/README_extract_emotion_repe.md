# Extracting Emotion Activation Directions using RepReadingPipeline

This document explains how to use the `RepReadingPipeline` and the utility function `all_emotion_rep_reader` to identify the activation directions associated with different emotions within the hidden states of a language model.

## Overview

The core idea is to train a "representation reader" (`RepReader`) for each emotion. This reader learns a direction in the model's activation space (specifically, within the hidden states of specified layers) that corresponds to the presence or intensity of that emotion.

The `RepReadingPipeline` provides the mechanism to access the model's hidden states for given inputs. The `all_emotion_rep_reader` function orchestrates this process across multiple emotions, handling data preparation, training the `RepReader` for each emotion, and saving the results.

## Steps

1.  **Prepare Data**:
    *   Organize your data into a dictionary structure where keys are emotion labels (e.g., 'anger', 'joy').
    *   Each emotion key should map to another dictionary containing 'train' and 'test' keys.
    *   The 'train' and 'test' values should be lists of text inputs representing examples of that emotion. For directional methods like 'pca' or 'rotation', these lists should contain pairs of texts (e.g., \[negative\_example, positive\_example, negative\_example, positive\_example,...]). For 'clustermean', single examples are sufficient.

2.  **Initialize `RepReadingPipeline`**:
    *   Instantiate the `RepReadingPipeline` with the desired transformer model and tokenizer.
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from neuro_manipulation.repe import RepReadingPipeline # Assuming RepReadingPipeline is in this path

    model_name = "mistralai/Mistral-7B-Instruct-v0.1" # Example model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id # Or another appropriate pad token ID

    rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer, pipeline_class=RepReadingPipeline)
    ```

3.  **Define Parameters**:
    *   `emotions`: A list of emotion strings you want to find directions for.
    *   `hidden_layers`: A list of layer indices (e.g., `[-1]`, `[0, 6, 12]`) from which to extract hidden states.
    *   `rep_token`: The index of the token whose representation you want to analyze (e.g., -1 for the last token).
    *   `n_difference`: The number of times to take differences between consecutive pairs in the input. Typically 1 for methods like 'pca'.
    *   `direction_method`: The strategy for finding directions ('pca', 'rotation', 'clustermean').
    *   `save_path`: Optional path to save the resulting `RepReader` objects.

4.  **Call `all_emotion_rep_reader`**:
    *   Pass the prepared data, initialized pipeline, and parameters to the function.
    ```python
    from neuro_manipulation.utils import all_emotion_rep_reader # Assuming the function is here

    # Example parameters
    emotions = ['anger', 'joy', 'sadness']
    hidden_layers = [-1] # Last layer
    rep_token = -1
    n_difference = 1
    direction_method = 'pca'
    save_path = 'exp_records/emotion_rep_readers.pkl'
    read_args = { # Store parameters used
        'rep_token': rep_token,
        'hidden_layers': hidden_layers,
        'n_difference': n_difference,
        'direction_method': direction_method,
    }


    # Assuming 'data' is the prepared data dictionary from step 1
    emotion_rep_readers = all_emotion_rep_reader(
        data=data,
        emotions=emotions,
        rep_reading_pipeline=rep_reading_pipeline,
        hidden_layers=hidden_layers,
        rep_token=rep_token,
        n_difference=n_difference,
        direction_method=direction_method,
        save_path=save_path,
        read_args=read_args
    )
    ```

5.  **Access Results**:
    *   The `all_emotion_rep_reader` function returns a dictionary (`emotion_rep_readers`).
    *   This dictionary contains the trained `RepReader` object for each emotion under its respective key (e.g., `emotion_rep_readers['anger']`).
    *   Each `RepReader` object holds the calculated `directions` (a dictionary mapping layer index to the direction vector) and potentially `direction_signs`.
    *   The dictionary also stores the accuracy per layer for each emotion under `emotion_rep_readers['layer_acc']` and the parameters used under `emotion_rep_readers['args']`.

## How it Works Internally

*   **`RepReadingPipeline._forward`**: Extracts hidden states from the specified `hidden_layers` for the `rep_token` of the input sequences.
*   **`RepReadingPipeline._get_hidden_states`**: Processes the model outputs to isolate the desired hidden states.
*   **`RepReadingPipeline.get_directions`**:
    *   Uses `_batched_string_to_hiddens` to get hidden states for all training inputs.
    *   Calculates differences between pairs of hidden states if `n_difference > 0`.
    *   Initializes a `DIRECTION_FINDER` based on `direction_method`.
    *   Calls the finder's `get_rep_directions` method, passing the (potentially differenced) hidden states to compute the direction vectors for each layer.
    *   Optionally computes `direction_signs` to orient the directions consistently.
*   **`all_emotion_rep_reader`**:
    *   Iterates through each specified `emotion`.
    *   Calls `get_rep_reader` (which likely wraps `RepReadingPipeline.get_directions` and evaluates accuracy) for each emotion's train/test data.
        *   **Validation Step (within `get_rep_reader`/`test_direction`):**
            *   Assumes the `test_data` follows the same structure as `train_data` (e.g., contrastive pairs if used for training).
            *   Uses the trained `rep_reader` (containing directions and signs) to transform the hidden states of the `test_data` via `RepReadingPipeline`. This projects the test hidden states onto the learned direction.
            *   Compares the projection scores of adjacent examples (the contrastive pairs) in the test set.
            *   Checks if the example with the higher/lower score within the pair matches the expected sign (`rep_reader.direction_signs`) determined during training. For instance, if the sign is +1, it checks if the second element of the pair (presumed positive) has a higher projection score.
            *   Calculates the accuracy based on how often this expected relationship holds true across all test pairs. This accuracy is stored in `emotion_rep_readers['layer_acc']`.
    *   Stores the resulting `RepReader` and accuracy.
    *   Saves the collected readers and metadata to a file if `save_path` is provided.

This process yields a set of vectors, one per layer per emotion, representing the direction in activation space associated with that emotion. These directions can then be used for analysis or manipulation of model behavior. 