"""Atomic provenance and invariant checks."""
import hashlib
import json
from pathlib import Path
from ..post_perbutation.config import replace_output


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str((array.shape, array.dtype.str)).encode())
    h.update(memoryview(array).cast('B'))
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False), encoding='utf-8')
    replace_output(temporary, path)


def input_record(inputs):
    return {key: {'path': str(Path(path).resolve()), 'sha256': file_hash(path)} for key, path in inputs.items()}
