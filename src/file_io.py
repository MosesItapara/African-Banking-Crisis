"""
Shared read/write helpers, keyed by file extension
Defined once here so pipeline classes don't each redefine their own readers dict.
"""

import pandas as pd
from pathlib import Path


READERS = {
    ".csv": pd.read_csv,
}

WRITERS = {
    ".csv": lambda df, path, **kwargs: df.to_csv(path, index=False, **kwargs),
}

def read_any(path: str) -> pd.DataFrame:
    """Read a single file into a DataFrame based on its file extension."""
    ext = Path(path).suffix
    if ext not in READERS:
        raise ValueError(f"Unsupported file extension: {ext}")
    return READERS[ext](path)

def write_any(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Write a DataFrame to disk based on the file extension."""
    ext = Path(Path).suffix
    if ext not in WRITERS:
        raise ValueError(f"Unsupported file extension: {ext}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    WRITERS[ext](df, path, **kwargs)