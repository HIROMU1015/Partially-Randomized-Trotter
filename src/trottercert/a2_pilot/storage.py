"""Single-contract atomic claim and append-only local records. No grant issuer."""

import os
from pathlib import Path

from ..a2_spec01_03.common import canonical_bytes, require


class LocalStore:
    def __init__(self, directory, key, *, output_bytes, log_bytes):
        require(len(key) == 64 and all(c in "0123456789abcdef" for c in key), "STORE", "key", "Contract digest required")
        self.root = Path(directory) / key
        self.output_limit, self.log_limit = output_bytes, log_bytes
        self.written = 0

    def claim(self, record):
        # mkdir itself is the exclusive, persistent consumed marker. Never retry,
        # unlink, reuse or remove it, even if the following write fails.
        self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.write("claim.json", canonical_bytes(record))

    def write(self, name, data):
        require(type(name) is str and "/" not in name and name not in ("", ".", ".."), "STORE", "path", "Flat owned output names only")
        limit = self.log_limit if name in ("stdout.txt", "stderr.txt") else self.output_limit
        require(type(data) is bytes and len(data) <= limit, "OUTPUT_LIMIT", name, "Record exceeds adopted byte limit")
        # Per-record output/log limit; aggregate size is measured separately.
        fd = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            self.written += len(data)
