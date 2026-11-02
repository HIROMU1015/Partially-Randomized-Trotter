#!/usr/bin/env python3
"""Limited saved-raw consistency diagnosis. Default byte/environment preflight."""
import argparse,json,os,subprocess,sys,time,importlib.metadata
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_separated_j_u_curvature_pilot as old
from trottertracks import c_j_u_consistency_repair as fix
from trottertracks.c_common_convergence_export_repair import save_json as save
SCOPE='C_J_U_SAVED_CONSISTENCY_DIAGNOSIS_V1'
FRESH='artifacts/research_c_j_u_consistency_repair/2026-11-01/diagnostic01'
KINDS=('original_load','PR_raw_load','PR_chart_reconstruct','PR_chart_evaluate','PR_physical_evaluate','PR_compare','RC2_state_load','RC2_codec','RC2_native_evaluate','RC2_original_evaluate','RC2_numpy_evaluate','RC2_jax_decode','RC2_jax_evaluate','RC2_compare','finalize')
LIMITS={k:1 for k in KINDS}

def preflight(c):
    if c['scope']!=SCOPE or c['limits']!=LIMITS or c['maximum_units']!=15 or c['output']!=FRESH:raise ValueError('DIAGNOSTIC_SCOPE')
    if old.io.sha(ROOT/c['SOURCE'])!=c['source_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in [old.io.read(ROOT/c['SOURCE'])['files'],c['inputs']]:
        for p,h in group.items():
            t=(ROOT/p).resolve()
            if not t.is_relative_to(ROOT) or old.io.sha(t)!=h:raise ValueError('SOURCE_INPUT:'+p)
    for name,v in c['packages'].items():
        if importlib.metadata.version(name)!=v:raise ValueError('ENVIRONMENT:'+name)
    return c

def run(c,out):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import ffsim.linalg.util as util
    from trottertracks import c_separated_j_u_curvature as core
    out.mkdir(parents=True)
    calls=old.common.old.Calls(out,LIMITS,15);start=time.monotonic()
    def bundle(path):
        with np.load(ROOT/path,allow_pickle=False) as a:return {k:np.array(a[k]) for k in a.files}
    try:
        save(out/'contract.json',c);save(out/'SOURCE.json',dict(old.io.read(ROOT/c['SOURCE']),execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        def original():
            a=bundle(c['input']);b=bundle(c['initial']);return a['g'],b
        g,initial=calls('original_load','g and same initialJ/U',original)
        info=old.io.read(ROOT/c['scales'])
        _,pr=old.common.setup('pr',info);module,rc2=old.common.setup('rc2',info)
        def raw():
            a=bundle(c['PR_raw'])
            lines=[json.loads(x) for x in (ROOT/c['PR_trace']).read_text().splitlines()]
            m=next(x for x in lines if x['kind']=='J_raw' and x['label']=='0');return a,m
        a,meta=calls('PR_raw_load','failedJ0',raw)
        def chart():
            p=core.Coefficients(g,initial['U'],pr);q=core.CoefficientChart(p,initial['J'])
            if not np.array_equal(a['x0'],q.x0) or not np.allclose(a['coordinate_transform'],q.W,atol=1e-12,rtol=0):raise ValueError('SAVED_CHART_IDENTITY')
            return q
        q=calls('PR_chart_reconstruct','same saved affinechart',chart)
        cf=calls('PR_chart_evaluate','savedz',lambda:q.vg(a['z']))
        pf=calls('PR_physical_evaluate','savedphysicalx',lambda:q.p.vg(a['x'],independent=True))
        report_pr=calls('PR_compare','all old subconditions',lambda:fix.pr_report(a,meta,cf,pf,q));save(out/'PR_comparison.json',report_pr)
        state=calls('RC2_state_load','last unexported state64',lambda:bundle(c['RC2_state']))
        J,U=state['J'],state['U']
        Jn,Un,x,codec,_=calls('RC2_codec','same frozen codec',lambda:core.pr.corrected.codec.roundtrip(J,U))
        def native():
            v,grad=module.make_value_and_grad(rc2)(x,g);return float(v),np.asarray(grad)[48:]/core.old.weights(8,4)
        native_value=calls('RC2_native_evaluate','ffsim atencodedx',native)
        def vg(j,u):
            p=core.Coefficients(g,u,rc2);return p.vg(p.pack(j),independent=True)
        original_value=calls('RC2_original_evaluate','originalJ/U',lambda:vg(J,U))
        numpy_value=calls('RC2_numpy_evaluate','same numpy decoded point',lambda:vg(Jn,Un))
        Jj,Uj=calls('RC2_jax_decode','actual AD decodepoint',lambda:tuple(np.asarray(v) for v in util.df_tensors_from_params_jax(jnp.asarray(x),8,4,None,real=True)))
        jax_value=calls('RC2_jax_evaluate','same AD decodepoint',lambda:vg(Jj,Uj))
        report_rc2=calls('RC2_compare','same versus different points',lambda:fix.rc2_report(original_value,numpy_value,jax_value,native_value,codec,dict(U=U,Un=Un,Uj=Uj)));save(out/'RC2_comparison.json',report_rc2)
        def final():
            old.existing.arrays(out/'gradient_evidence.npz',pr_raw_jac=a['solver_jac'],pr_endpoint_chart_jac=cf[1],pr_physical_J_jac=pf[1],rc2_native_J_jac=native_value[1],rc2_original_J_jac=original_value[1],rc2_numpy_J_jac=numpy_value[1],rc2_jax_J_jac=jax_value[1])
            return dict(status='DIAGNOSTIC_MEASUREMENTS_COMPLETE',PR=report_pr,RC2=report_rc2,new_optimization=0,operator_sector_PF_native_shots=0,mandatory_stop=True)
        save(out/'result.json',calls('finalize','evidence only;no optimization',final))
    except BaseException as e:
        calls.stop.set();save(out/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:old.existing.finish(out,calls,start)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--contract',type=Path,required=True);a.add_argument('--execute',action='store_true');args=a.parse_args();c=preflight(old.io.read(args.contract))
    if not args.execute:print(json.dumps(dict(status='PASSIVE_PASS',science=0,maximum=15)));return
    out=ROOT/FRESH
    if out.exists():raise ValueError('FRESH_DIAGNOSTIC_NO_RESUME')
    os.environ['JAX_ENABLE_X64']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    for name,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('TMPDIR','tmp')]:
        p=ROOT/'.cache/j-u-consistency-repair'/leaf;p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
    run(c,out)
if __name__=='__main__':main()
