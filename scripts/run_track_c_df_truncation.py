"""Run a named stage in a fresh study output; never resume historical runners."""
import sys
from trottertracks.track_c_df_truncation import run
run(sys.argv[1])
