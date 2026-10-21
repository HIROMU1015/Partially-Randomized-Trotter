"""Fresh supplemental physical validation; never rerun input or cost units."""
import argparse
import importlib.util
from pathlib import Path
import sys

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--system',choices=['H2','H4','H6','H8'],required=True)
    system=parser.parse_args().system
    p=Path(__file__).resolve().parents[1]/'src/trotterlib/recomputed_s2_physics_validation.py'
    spec=importlib.util.spec_from_file_location('recomputed_s2_physics',p)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    print(module.run(system),flush=True)
