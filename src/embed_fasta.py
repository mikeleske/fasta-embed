import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import config as cfg
from utils_model import load_model_from_hf
from utils import parse_fasta
from utils_bio import get_region


def get_emb(model, tokenizer, seq: str, device: str = None) -> torch.Tensor:
    """
    Generates an embedding for a DNA sequence using a model with mean pooling.

    Args:
        model (torch.nn.Module): The model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        seq (str): The DNA sequence to embed.
        device (str): Device to use ('cuda' or 'cpu'). Auto-detected if None.

    Returns:
        torch.Tensor: Mean-pooled embedding of the input DNA sequence.

    Raises:
        ValueError: If the input sequence is invalid.
    """
    if not isinstance(seq, str):
        raise ValueError("Input sequence must be a string.")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    try:
        inputs = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            hidden_states = model(inputs)[0]
        embedding_mean = torch.mean(hidden_states[0], dim=0).detach().cpu()
        return embedding_mean
    except Exception as e:
        raise RuntimeError(f"Error during embedding generation: {e}")

def get_emb_nt(model, tokenizer, seq: str, device: str = None) -> torch.Tensor:
    """
    Generates an embedding for a DNA sequence using a model with mean pooling.

    Args:
        model (torch.nn.Module): The model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        seq (str): The DNA sequence to embed.
        device (str): Device to use ('cuda' or 'cpu'). Auto-detected if None.

    Returns:
        torch.Tensor: Mean-pooled embedding of the input DNA sequence.

    Raises:
        ValueError: If the input sequence is invalid.
    """
    if not isinstance(seq, str):
        raise ValueError("Input sequence must be a string.")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    try:
        inputs = tokenizer(seq, return_tensors="pt")["input_ids"].to(device)
        attention_mask = inputs != tokenizer.pad_token_id
        torch_outs = model(
            inputs,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        embeddings = torch_outs['hidden_states'][-1].detach()#.numpy()
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        mean_sequence_embeddings = (torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)).cpu().numpy()
        return mean_sequence_embeddings
    except Exception as e:
        raise RuntimeError(f"Error during embedding generation: {e}")


def get_emb_ntv3(model, tokenizer, seqs: list[str], device: str = None) -> np.ndarray:
    """
    Generates mean-pooled embeddings for a batch of DNA sequences.

    Args:
        model (torch.nn.Module): The model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        seqs (list[str]): Batch of DNA sequences to embed.
        device (str): Device to use ('cuda' or 'cpu'). Auto-detected if None.

    Returns:
        np.ndarray: Array of shape (batch_size, embed_dim).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        inputs = tokenizer(seqs, add_special_tokens=False, padding=True, pad_to_multiple_of=128, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        with torch.no_grad():
            out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = out.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1)
        mean_emb = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        return mean_emb.detach().cpu().numpy()
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
    infer_batch_size: int = 32,
    save_batch_size: int = 10000
) -> None:
    """
    Generate embeddings for sequences in a DataFrame column and save to a NumPy file.

    Args:
        model (torch.nn.Module): The model used for generating embeddings.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        df (pd.DataFrame): The DataFrame containing sequences.
        column (str): The column containing sequences. Defaults to 'Seq'.
        embeddings_numpy_file (str): The file path for saving embeddings.
        infer_batch_size (int): Sequences per model forward pass. Defaults to 16.
        save_batch_size (int): Vectors to accumulate before saving to disk. Defaults to 10,000.

    Raises:
        ValueError: If required parameters are missing or column doesn't exist in DataFrame.
        RuntimeError: If embeddings saving/loading fails.
    """
    if embeddings_numpy_file is None:
        raise ValueError("embeddings_numpy_file cannot be None.")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    vectors = []
    embeddings = None
    n = df.shape[0]

    try:
        for start in tqdm(range(0, n, infer_batch_size), desc="Processing sequences"):
            batch_seqs = df[column].iloc[start:start + infer_batch_size].tolist()
            batch_embs = get_emb_ntv3(model, tokenizer, batch_seqs)
            vectors.append(batch_embs)

            if sum(v.shape[0] for v in vectors) >= save_batch_size:
                vectors = np.vstack(vectors)
                embeddings = _save_embeddings_batch(vectors, embeddings_numpy_file)
                vectors = []

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
    model, tokenizer = load_model_from_hf(cfg.MODEL_ID, maskedlm=True)

    # df = parse_fasta(
    #     file=Path(cfg.DATA_FILE),
    #     gzipped=False
    # )
    df = pd.read_csv("C:\\Users\\mikel\\Documents\\PhD\\Data\\Genomes\\Greengenes2\\2024.09\\df.2024.09.backbone.full-length.V3V4.csv.gz", sep="\t")
    embed_column = 'V3V4'

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
