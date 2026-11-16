"""Portable package paths, including long Windows research-workspace paths."""
from pathlib import Path
import os
def long_path(path):
    p=Path(path).resolve()
    if os.name=='nt' and not str(p).startswith('\\\\?\\'):
        raw=str(p)
        return Path('\\\\?\\UNC\\'+raw[2:] if raw.startswith('\\\\') else '\\\\?\\'+raw)
    return p
