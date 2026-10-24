"""Stable, non-sensitive Stage-1 failure codes for retained STOP evidence."""


class Stage1Failure(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)
