"""Copy fixed bytes to a fresh owned artificial-only runtime; execute nothing."""
from pathlib import Path
import argparse,shutil
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--new-root',required=True,type=Path);a=p.parse_args()
 source=Path(__file__).resolve().parent/'fixed_reference/tree';target=a.new_root.resolve()
 if target.exists() or target.is_relative_to(source) or source.is_relative_to(target):raise SystemExit('Require a fresh independent runtime path')
 shutil.copytree(source,target);print(target)
