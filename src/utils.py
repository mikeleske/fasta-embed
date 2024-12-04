import pandas as pd
from Bio import SeqIO
import gzip

def parse_fasta(file: str, gzipped: bool = True) -> pd.DataFrame:
    """
    Parse a FASTA file into a pandas DataFrame.

    Args:
        file (str): Path to the input FASTA file.
        gzipped (bool): Indicate of FASTA file is gzipped.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed FASTA data with columns: ['ID', 'SeqLen', 'Seq'].
    """
    # Define the columns for the resulting DataFrame
    columns = ['ID', 'SeqLen', 'Seq']
    rows_list = []

    # Parse the FASTA file
    if gzipped:
        with gzip.open(file, "rt") as handle:
            for rec in SeqIO.parse(handle, "fasta"):
                rows_list.append({
                    'ID': str(rec.description).split()[0],
                    'SeqLen': len(rec.seq),
                    'Seq': str(rec.seq)
                })
    else:
        with open(file, "rt") as handle:
            for rec in SeqIO.parse(handle, "fasta"):
                rows_list.append({
                    'ID': str(rec.description).split()[0],
                    'SeqLen': len(rec.seq),
                    'Seq': str(rec.seq)
                })

    # Create a DataFrame
    df = pd.DataFrame(rows_list, columns=columns)

    return df
