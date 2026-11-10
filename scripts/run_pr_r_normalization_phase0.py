"""Fixed24 Phase0 entry. Default is preflight only; --execute needs permission."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_r_normalization_phase0 as phase


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--execute',action='store_true',help='Separately permitted fixed24 normalization science only')
    args=parser.parse_args();out=Path(args.output_dir)
    if out.exists():raise FileExistsError('Fresh output required; no resume/retry/overwrite')
    packet=phase.preflight(ROOT);plan=phase.public_plan(packet)
    if not args.execute:
        out.mkdir(parents=True);phase.save(out/'PLAN.json',plan)
        print(json.dumps(dict(status=plan['status'],candidates=24,normalization_evaluations=0,mandatory_stop=True)));return 0
    if json.loads((ROOT/phase.PLAN_PATH).read_bytes())!=plan:raise ValueError('Published Phase0 plan differs')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in phase.REUSED_FILES+phase.NEW_FILES:
        if (ROOT/path).read_bytes()!=phase.blob(ROOT,source,path):raise ValueError('Execution SOURCE differs from commit')
    result=phase.execute(packet,ROOT,source,out)
    result.update(source_files=packet['sources'],input_hashes=packet['input_hashes'],saved_sources=packet['saved_sources'],saved_tails=packet['tails'])
    phase.save(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],normalization_rows=len(result['records']),N_C_G_evaluations=0,mandatory_stop=True)))
    return 0 if result['status']=='PHASE0_NORMALIZATION_RECORDED' else 1


if __name__=='__main__':sys.exit(main())
