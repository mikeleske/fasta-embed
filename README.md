# fasta-embed

**Embed your DNA sequences into vector space**

This small repository is meant to assist you in quickly creating semantic embeddings for your DNA sequences.
The code is optimized for generating DNABERT-S embeddings. Using a GPU is highly recommended.

# Installation

1. Install necessary Python packages `pip install -r requirements.txt`.
2. Uninstall triton as it conflicts with the FlashAttention code of DNABERT-S

# Execution

1. Download your FASTA file of interest.
2. Update `config.py` to specify your FASTA file.
3. Update `config.py` REGION parameter, if you are interested in embeddings for specific 16S regions, e.g. "V3V4".
4. Run `python embed_fasta.py`.
