"""16S rRNA primer definitions and region extraction."""

from __future__ import annotations

import re

from Bio.Seq import Seq

PRIMERS: dict[str, str] = {
    "27F": "AGAGTTTGATYMTGGCTCAG",
    "68F": "TNANACATGCAAGTCGRRCG",
    "341F": "CCTACGGGNGGCWGCAG",
    "515F": "GTYCAGCMGCCGCGGTAA",
    "799F": "AACMGGATTAGATACCCKG",
    "939F": "GAATTGACGGGGGCCCGCACAAG",
    "967F": "CAACGCGAAGAACCTTACC",
    "1115F": "CAACGAGCGCAACCCT",
    "338R": "GCTGCCTCCCGTAGGAGT",
    "518R": "WTTACCGCGGCTGCTG",
    "534R": "ATTACCGCGGCTGCTGG",
    "785R": "GACTACHVGGGTATCTAATCC",
    "806R": "GGACTACHVGGGTWTCTAAT",
    "926R": "CCGYCAATTYMTTTRAGTTT",
    "944R": "GAATTAAACCACATGCTC",
    "1193R": "ACGTCATCCCCACCTTCC",
    "1378R": "CGGTGTGTACAAGGCCCGGGAACG",
    "1391R": "GACGGGCGGTGWGTRCA",
    "1492R": "TACGGYTACCTTGTTACGACTT",
}

PRIMERS_REGEX: dict[str, str] = {
    "27F": "AGAGTTTGAT[CT][AC]TGGCTCAG",
    "68F": "T[ACGT]A[ACGT]ACATGCAAGTCG[AG][AG]CG",
    "341F": "CCTACGGG[ACGT]GGC[AT]GCAG",
    "515F": "GTG[CT]CAGC[AC]GCCGCGGTAA",
    "799F": "AAC[AC]GGATTAGATACCC[GT]G",
    "939F": "GAATTGACGGGGGCCCGCACAAG",
    "967F": "CAACGCGAAGAACCTTACC",
    "1115F": "CAACGAGCGCAACCCT",
    "338R": "GCTGCCTCCCGTAGGAGT",
    "518R": "[AT]TTACCGCGGCTGCTG",
    "534R": "ATTACCGCGGCTGCTGG",
    "785R": "GACTAC[ACT][GCA]GGGTATCTAATCC",
    "806R": "GGACTAC[ACT][GCA]GGGT[AT]TCTAAT",
    "926R": "CCG[CT]CAATT[CT][AC]TTT[AG]AGTTT",
    "944R": "GAATTAAACCACATGCTC",
    "1193R": "ACGTCATCCCCACCTTCC",
    "1378R": "CGGTGTGTACAAGGCCCGGGAACG",
    "1391R": "GACGGGCGGTG[AT]GT[AG]CA",
    "1492R": "TACGG[CT]TACCTTGTTACGACTT",
}

# Pre-compiled patterns -- avoids re-compiling on every call to _find_primers.
_COMPILED_REGEX: dict[str, re.Pattern] = {
    name: re.compile(pattern) for name, pattern in PRIMERS_REGEX.items()
}

_REGION_PRIMERS: dict[str, tuple[str, str]] = {
    "V1V2": ("27F", "338R"),
    "V1V3": ("27F", "534R"),
    "V3V4": ("341F", "785R"),
    "V4": ("515F", "806R"),
    "V4V5": ("515F", "944R"),
    "V6V8": ("939F", "1378R"),
    "V7V9": ("1115F", "1492R"),
    "V1V8": ("27F", "1378R"),
    "V1V9": ("27F", "1492R"),
}

_FALLBACK_SEQUENCE = "ACGT"


def _find_primers(
    forward_name: str, reverse_name: str, seq: str
) -> tuple[str | None, str | None]:
    """Locate forward and reverse primer sequences within *seq*."""
    rev_seq = str(Seq(seq).reverse_complement())
    f_match = _COMPILED_REGEX[forward_name].search(seq)
    r_match = _COMPILED_REGEX[reverse_name].search(rev_seq)
    return (
        f_match.group() if f_match else None,
        r_match.group() if r_match else None,
    )


def get_region(region: str, seq: str) -> str:
    """Extract a 16S rRNA variable *region* from *seq*.

    Returns a fallback ``"ACGT"`` if primers cannot be found.
    """
    if region not in _REGION_PRIMERS:
        raise ValueError(
            f"Unknown region '{region}'. Available: {sorted(_REGION_PRIMERS)}"
        )

    fwd_name, rev_name = _REGION_PRIMERS[region]
    f_primer, r_primer = _find_primers(fwd_name, rev_name, seq)

    try:
        after_fwd = seq.split(f_primer)[1]
        rev_comp = str(Seq(after_fwd).reverse_complement())
        after_rev = rev_comp.split(r_primer)[1]
        return str(Seq(after_rev).reverse_complement())
    except (IndexError, TypeError):
        return _FALLBACK_SEQUENCE
