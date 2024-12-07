import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import config as cfg
from utils_model import load_model_from_hf
from utils import parse_fasta
from utils_bio import get_region


def get_emb(model, tokenizer, seq: str, device: str = "cuda") -> torch.Tensor:
    """
    Generates an embedding for a DNA sequence using a model with mean pooling.

    Args:
        model (torch.nn.Module): The model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        seq (str): The DNA sequence to embed.
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        torch.Tensor: Mean-pooled embedding of the input DNA sequence.

    Raises:
        ValueError: If the input sequence is invalid.
        RuntimeError: If CUDA is selected but unavailable.
    """
    if not isinstance(seq, str):
        raise ValueError("Input sequence must be a string.")
    
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use device='cpu' instead.")

    device = torch.device(device)
    
    try:
        inputs = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            hidden_states = model(inputs)[0]
        embedding_mean = torch.mean(hidden_states[0], dim=0).detach().cpu()
        return embedding_mean
    except Exception as e:
        raise RuntimeError(f"Error during embedding generation: {e}")


def drop_duplicate_sequences(df: pd.DataFrame, column: str = "Seq") -> pd.DataFrame:
    """
    Removes duplicate sequences from a DataFrame based on 'Taxa' and a specified column.

    Args:
        df (pd.DataFrame): The DataFrame containing sequence data.
        column (str): Column name to consider for duplicates. Defaults to 'Seq'.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.

    Raises:
        ValueError: If required columns are missing in the DataFrame.
    """
    if column not in df.columns or "Taxa" not in df.columns:
        raise ValueError(f"DataFrame must contain columns 'Taxa' and '{column}'.")

    return df.drop_duplicates(subset=["Taxa", column], keep="first").reset_index(drop=True)



def vectorize(
    model,
    tokenizer,
    df: pd.DataFrame,
    column: str = "Seq",
    embeddings_numpy_file: str = None,
    batch_size: int = 10000
) -> None:
    """
    Generate embeddings for sequences in a DataFrame column and save to a NumPy file.

    Args:
        model (torch.nn.Module): The model used for generating embeddings.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        df (pd.DataFrame): The DataFrame containing sequences.
        column (str): The column containing sequences. Defaults to 'Seq'.
        embeddings_numpy_file (str): The file path for saving embeddings.
        batch_size (int): The number of vectors to process in each batch. Defaults to 10,000.

    Raises:
        ValueError: If required parameters are missing or column doesn't exist in DataFrame.
        RuntimeError: If embeddings saving/loading fails.
    """
    if embeddings_numpy_file is None:
        raise ValueError("embeddings_numpy_file cannot be None.")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    embeddings_numpy_file = embeddings_numpy_file
    vectors = []
    embeddings = None

    try:
        # Process rows in batches
        for i in tqdm(range(df.shape[0]), desc="Processing sequences"):
            emb = get_emb(model, tokenizer, df.loc[i, column])
            vectors.append(emb.reshape(1, -1))

            # Save batch to file when reaching batch_size
            if len(vectors) >= batch_size:
                vectors = np.vstack(vectors)
                embeddings = _save_embeddings_batch(vectors, embeddings_numpy_file)
                vectors = []

        # Save any remaining vectors after processing all rows
        if vectors:
            vectors = np.vstack(vectors)
            embeddings = _save_embeddings_batch(vectors, embeddings_numpy_file)

        print(f"Final embeddings shape: {embeddings.shape if embeddings is not None else (0,)}")
    except Exception as e:
        raise RuntimeError(f"Error during vectorization: {e}")


def _save_embeddings_batch(vectors: np.ndarray, file_path: str) -> np.ndarray:
    """
    Save a batch of embeddings to a NumPy file, appending if the file already exists.

    Args:
        vectors (np.ndarray): Batch of vectors to save.
        file_path (str): Path to the NumPy file.

    Returns:
        np.ndarray: The combined embeddings.
    """
    try:
        if _file_exists(file_path):
            existing_embeddings = np.load(file_path)
            embeddings = np.concatenate((existing_embeddings, vectors), axis=0)
        else:
            embeddings = vectors
        np.save(file_path, embeddings)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to save embeddings to {file_path}: {e}")


def _file_exists(file_path: str) -> bool:
    """
    Check if a file exists.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    try:
        with open(file_path, "rb"):
            return True
    except FileNotFoundError:
        return False


def main() -> None:
    model, tokenizer = load_model_from_hf(cfg.MODEL_ID)

    df = parse_fasta(
        file=Path(cfg.DATA_FILE),
        gzipped=False
    )

    embed_column = 'Seq'

    if cfg.REGION:
        df[cfg.REGION] = df['Seq'].apply(lambda x: get_region(region=cfg.REGION, seq=x))
        embed_column = cfg.REGION

    vectorize(
        model=model, 
        tokenizer=tokenizer, 
        df=df, 
        column=embed_column, 
        embeddings_numpy_file=cfg.EMB_FILE
    )

# --------------------------------------------------
if __name__ == '__main__':
    main()
