"""Join supplied theory rows to saved shot/compiled accounting, without selection.

Exploration and confirmation scopes remain distinct. Missing measured costs stay
missing. No candidate ranking, science import, compile, or shot derivation.
"""
from fractions import Fraction as F
import argparse
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def rational(x):return F(int(x['numerator']),int(x['denominator']))
def encode(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator),decimal=float(x))


def connect(manifest,theory):
    index={(x['system'],x['candidate_id'],x['model'],x['axis']):x for x in theory}
    result=[]
    for system in ('H6','H8'):
        saved=read(manifest['inputs'][system+'_d0']['path'])
        family=read(manifest['inputs'][system+'_B0I_results']['path'])
        scopes={k:v for k,v in saved.items() if k.startswith(system+'_')}
        extra=[]
        for case in family['cases']:
            for condition in case['accounting']:
                extra.append(dict(cell_id=f"B0I_p{case['prefix']}_q{case['q']}_2nd",epsilon=condition['epsilon'],
                    factor=condition['factor'],accounting=condition['accounting'],metrics={},
                    saved_axes=case.get('axes'),compiled_origin_SOURCE=case['origin_SOURCE'],sample_count=1))
        scopes[system+'_B0I_FAMILY12']=extra
        for scope,records in scopes.items():
            for row in records:
                shots={axis:row['accounting']['axes'][key].get('shots') for axis,key in [('cosine','real'),('sine','imag')]}
                for model in ('A','B'):
                    for metric,field in [('RZ','rz'),('CX','cx')]:
                        selected={axis:index[system,row['cell_id'],model,axis] for axis in shots}
                        valid=all(type(x) is int and x>0 for x in shots.values())
                        lo=sum((shots[a]*rational(t['lower'][field]) for a,t in selected.items()),F(0)) if valid else None
                        hi=sum((shots[a]*rational(t['upper'][field]) for a,t in selected.items()),F(0)) if valid else None
                        original=row.get('metrics',{}).get(metric)
                        axes=row.get('saved_axes')
                        means=(original.get('axis_means') if original else
                               {a:axes[a]['metrics'][metric] for a in shots} if axes else None)
                        compiled=sum((F(means[a])*shots[a] for a in shots),F(0)) if valid and means else None
                        if compiled is not None and original and compiled!=rational(original['G_exact']):
                            raise ValueError('SAVED_COMPILED_ACCOUNTING_MISMATCH')
                        result.append(dict(system=system,scope=scope,candidate_id=row['cell_id'],model=model,metric=metric,
                            epsilon=row['epsilon'],factor=str(row['factor']),shots=shots,
                            theory_lower=None if lo is None else encode(lo),theory_upper=None if hi is None else encode(hi),
                            saved_compiled_G=None if compiled is None else encode(compiled),saved_axis_means=means,
                            theory_to_saved_compiled_lower=None if compiled in (None,0) else encode(lo/compiled),
                            theory_to_saved_compiled_upper=None if compiled in (None,0) else encode(hi/compiled),
                            sample_phase=row.get('sample_phase'),sample_count=row.get('sample_count'),
                            compiled_origin_SOURCE=row.get('compiled_origin_SOURCE'),
                            identity_status='CONDITIONALLY_ALIGNED_METADATA_NOT_FULL_NUMERIC_ENVELOPE',
                            status='INELIGIBLE_SAVED_SHOTS' if not valid else 'COMPILED_NOT_ACQUIRED' if compiled is None else 'JOINED_WITH_DECLARED_LIMITS',
                            statistical_SE_recomputed=False,new_shots_derived=False,candidate_selection_performed=False))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--theory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=read(a.manifest)
    if m.get('approved') is not True or m.get('runtime_authorization') is not True:raise ValueError('NOT_AUTHORIZED')
    if not a.output.resolve().is_relative_to('/home/AbeHiromu'):raise ValueError('HOME_OUTPUT_REQUIRED')
    for name in ('H6_d0','H8_d0','H6_B0I_results','H8_B0I_results'):
        item=m['inputs'][name];data=Path(item['path']).read_bytes()
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('INPUT_BYTES_MISMATCH')
    rows=connect(m,read(a.theory));data=json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    if len(data)>m['limits']['output_bytes']:raise ValueError('OUTPUT_CAP')
    with a.output.open('xb') as f:f.write(data)
    print(json.dumps(dict(rows=len(rows),joined=sum(x['status']=='JOINED_WITH_DECLARED_LIMITS' for x in rows),ranking=False)))


if __name__=='__main__':main()
