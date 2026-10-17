"""Only scope/provenance bindings differ from frozen SPEC-01--03 common."""

from ..a2_spec01_03.common import *
from ..a2_spec01_03 import common as frozen
from .identity import source_identity

VERSION = "A2_PILOT_SPEC01_03_V1"
SCOPE = "FIXED_DF_PILOT_CORE_ONLY_NOT_EXECUTION_AUTHORITY"


def rejection(exc, stage, partial, *, verdict):
    record = frozen.rejection(exc, stage, partial, verdict=verdict)
    record.update(schema=VERSION, scope=SCOPE)
    return record
