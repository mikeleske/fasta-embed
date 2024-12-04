from transformers import AutoTokenizer, AutoModel

def load_model_from_hf(model_name: str):
    """
    Loads a model and tokenizer from Hugging Face.

    Args:
        model_name (str): The model name or path in Hugging Face Hub.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.

    Raises:
        ValueError: If the model name is invalid or loading fails.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map={"": 0}  # Loads the model onto the first available GPU
        )
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load model/tokenizer: {e}")