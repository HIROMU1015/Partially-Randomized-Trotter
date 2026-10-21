"""Run each authorized scientific unit once; never resume or retry units."""
import argparse
import json
import recomputed_s2_runtime_v2 as r

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--system',choices=['H2','H4','H6','H8'],required=True)
    args=parser.parse_args()
    result=r.run_system(args.system)
    r.save(r.BASE/(args.system+'_terminal.json'),result)
    print(json.dumps(result),flush=True)
