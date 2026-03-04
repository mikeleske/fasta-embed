import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

def load_model_from_hf(model_name: str, maskedlm: bool = False):
    """
    Loads a model and tokenizer from Hugging Face.

    Args:
        model_name (str): The model name or path in Hugging Face Hub.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.

    Raises:
        ValueError: If the model name is invalid or loading fails.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model_cls = AutoModelForMaskedLM if maskedlm else AutoModel
        model = model_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map={"": device}
        )
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load model/tokenizer: {e}")