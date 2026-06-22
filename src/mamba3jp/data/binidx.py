"""Binary index format for tokenized training corpora.

Layout
------
``.bin`` file: raw little-endian ``uint32`` tokens, concatenated across all
    documents.

``.idx`` file::

    bytes 0..15   : magic     b"MAMBA3IDXv1\0\0\0\0"
    bytes 16..19  : dtype     uint32 (0 = uint32 tokens; only value supported)
    bytes 20..27  : n_docs    uint64
    bytes 28..    : doc_lens  n_docs * uint64

We deliberately keep this simpler than Megatron's `.bin`/`.idx`: there is one
data type, no per-document offsets table (offsets are a cumsum of ``doc_lens``).
The point is to be readable with a single ``np.memmap`` and trivially writable
in an append-friendly stream.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

MAGIC = b"MAMBA3IDXv1\0\0\0\0\0"
assert len(MAGIC) == 16
TOKEN_DTYPE = np.uint32
DTYPE_CODE = 0  # only one supported for now
_HEADER_FIXED = 16 + 4 + 8  # magic + dtype + n_docs


class BinIdxWriter:
    """Streaming writer; one ``add_document`` per source document."""

    def __init__(self, bin_path: str | Path, idx_path: str | Path) -> None:
        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)
        self._bin_file: BinaryIO | None = None
        self._doc_lengths: list[int] = []
        self._closed = False

    def __enter__(self) -> BinIdxWriter:
        self.bin_path.parent.mkdir(parents=True, exist_ok=True)
        self.idx_path.parent.mkdir(parents=True, exist_ok=True)
        self._bin_file = self.bin_path.open("wb")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def add_document(self, tokens: npt.NDArray[np.unsignedinteger]) -> None:
        if self._bin_file is None:
            raise RuntimeError("writer is not open; use as a context manager")
        if tokens.dtype != TOKEN_DTYPE:
            raise ValueError(
                f"BinIdxWriter expects dtype={TOKEN_DTYPE}, got {tokens.dtype}"
            )
        if tokens.ndim != 1:
            raise ValueError(f"BinIdxWriter expects 1-D array, got {tokens.ndim}-D")
        self._bin_file.write(tokens.tobytes())
        self._doc_lengths.append(int(tokens.size))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._bin_file is not None:
            self._bin_file.close()
            self._bin_file = None
        # Write the idx file.
        n_docs = len(self._doc_lengths)
        doc_arr = np.asarray(self._doc_lengths, dtype=np.uint64)
        with self.idx_path.open("wb") as f:
            f.write(MAGIC)
            f.write(np.uint32(DTYPE_CODE).tobytes())
            f.write(np.uint64(n_docs).tobytes())
            f.write(doc_arr.tobytes())


class BinIdxReader:
    """Random-access reader backed by a ``np.memmap`` over the ``.bin`` file."""

    def __init__(self, bin_path: str | Path, idx_path: str | Path) -> None:
        self.bin_path = Path(bin_path)
        self.idx_path = Path(idx_path)

        with self.idx_path.open("rb") as f:
            magic = f.read(16)
            if magic != MAGIC:
                raise ValueError(
                    f"unrecognised idx magic: expected {MAGIC!r}, got {magic!r}"
                )
            dtype_code = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
            if dtype_code != DTYPE_CODE:
                raise ValueError(f"unsupported dtype code {dtype_code}")
            n_docs = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
            self.doc_lengths = np.frombuffer(f.read(8 * n_docs), dtype=np.uint64).copy()

        self.n_documents = n_docs
        self.total_tokens = int(self.doc_lengths.sum()) if n_docs else 0
        self._mmap: npt.NDArray[np.uint32] | None = None
        if self.total_tokens > 0:
            self._mmap = np.memmap(self.bin_path, dtype=TOKEN_DTYPE, mode="r")
            if self._mmap.size < self.total_tokens:
                raise ValueError(
                    f"bin file shorter than idx claims: {self._mmap.size} < {self.total_tokens}"
                )

    def __len__(self) -> int:
        return self.total_tokens

    def __getitem__(
        self, key: int | slice
    ) -> npt.NDArray[np.uint32] | np.uint32:
        if self._mmap is None:
            raise IndexError("reader is empty")
        return self._mmap[key]
