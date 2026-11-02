"""Persistent evaluation context; existing CAR/DF/PR implementations."""
from pathlib import Path
import json
from . import c1_stage1_execution as io


def arrays(path,**values):
    import numpy as np
    with Path(path).open('xb') as f:np.savez(f,**values)


class Evaluation:
    def __init__(self,root,c,out):
        from . import c1_molecule_extension as molecule
        from .c_pr_aware_df_stage_b_runner import load
        self.root=Path(root);self.c=c;self.out=Path(out);self.molecule=molecule;self.cache={}
        self.packet=load(self.root/c['input_paths']['input'],('h','g','constant'))
        self.ctx=molecule.SectorCAR(4)
        self.physical=self.ctx.physical(**self.packet)
        arrays(self.out/'physical_reference.npz',H=self.physical,sector_basis_indices=self.ctx.sector.basis_indices)

    def assess(self,label,path):
        import numpy as np
        from .c_separated_j_u import linear
        from .c_pr_aware_df import coefficients
        from .c_pr_aware_df_stage_b_runner import load
        d=load(path,('J','U'));J,U=d['J'],d['U']
        if J.shape!=(8,4,4) or U.shape!=J.shape or any(x.dtype!=np.float64 or not np.isfinite(x).all() for x in (J,U)):
            raise ValueError('PHYSICAL_CANDIDATE_SCHEMA')
        if not np.array_equal(J,J.transpose(0,2,1)) or np.max(abs(U.transpose(0,2,1)@U-np.eye(4)))>1e-10:
            raise ValueError('PHYSICAL_J_U_INVARIANTS')
        row,G=self.molecule.rank_assessment(self.ctx,self.packet,J,U,self.physical,8)
        representation=self.molecule.representation_gate(self.ctx,self.packet,J,U,G)
        residual=linear.reconstruct(J,U)-self.packet['g']
        row.update(status='NUMERICAL_DF_ASSESSED',representation=representation,
            tensor_frobenius=float(np.linalg.norm(residual)),tensor_max_entry=float(np.max(abs(residual))),
            exact_tail_lambda_Ha=float(np.sum(abs(coefficients(J[2:],np)))),
            budget_Ha=self.c['DF_budget_Ha'],budget_PASS=bool(row['sector_error']+row['numerical_guard']<=self.c['DF_budget_Ha']),
            candidate_sha256=io.sha(path),kappa='original_g_corrected',scalar='nuclear_once',optimizer_success_inferred=False)
        directory=self.out/'physical';directory.mkdir(exist_ok=True)
        arrays(directory/(label+'.npz'),**d,generators=np.asarray(G),H=sum(G),tensor_residual=residual)
        io.save(directory/(label+'.json'),row)
        self.cache[label]=dict(d=d,G=G,H=sum(G),row=row)
        return row

    def performance(self,label):
        from .c_pr_integrated_performance import run
        return run(self.root,self.c,self.out/'C',self.packet,self.ctx,self.physical,self.cache[label])


def worker(root,c,out):
    import sys
    context=None
    for line in sys.stdin:
        request=json.loads(line)
        if request['operation']=='close':break
        try:
            if request['operation']=='initialize':context=Evaluation(root,c,out);value=dict(status='READY')
            elif request['operation']=='assess':value=context.assess(request['label'],Path(request['path']))
            elif request['operation']=='performance':value=context.performance(request['label'])
            else:raise ValueError('EVALUATION_REQUEST')
            print(json.dumps(dict(ok=True,value=value),allow_nan=False),flush=True)
        except BaseException as e:
            print(json.dumps(dict(ok=False,error=repr(e))),flush=True);raise
