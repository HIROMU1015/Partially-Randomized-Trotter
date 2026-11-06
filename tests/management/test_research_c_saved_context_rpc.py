"""Real RPC processes and real legacy classes with entirely invented arrays."""
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_integrated_execution_rpc as rpc
from trottertracks import c1_stage1 as stage
from trottertracks import c1_stage1_execution as io


def test_real_subprocess_unpickles_saved_legacy_context_without_rebuilding(tmp_path):
    from trottertracks.c1_molecule_extension import SectorCAR
    packet=dict(h=np.zeros((4,4)),g=np.zeros((4,)*4),constant=np.array(0.))
    source=tmp_path/'invented_input.npz';np.savez(source,**packet)
    context=SectorCAR(4)
    common=dict(p=packet,ctx=context,physical_reference=np.eye(context.d,dtype=complex),
        input_identity=io.sha(source),basis=stage.bit_reverse_indices(8)[context.sector.basis_indices])
    saved=tmp_path/'invented_context.pkl'
    with saved.open('wb') as f:pickle.dump(common,f,protocol=5)
    script=tmp_path/'artificial_rpc.py'
    script.write_text('import sys\nsys.path.insert(0,'+repr(str(ROOT/'src'))+')\n'+'''
import argparse
assert 'trottertracks._c1_stage1_legacy' not in sys.modules
from trottertracks import c1_molecule_extension as molecule
from trottertracks import c_integrated_execution_rpc as rpc
def forbidden(*args,**kwargs):
    raise AssertionError('SAVED_CONTEXT_MUST_NOT_REBUILD_CAR_OR_PHYSICAL_H')
molecule.SectorCAR.__init__=forbidden
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--worker');p.add_argument('--worker-config')
a=p.parse_args();rpc.worker_main(a.root,a.worker,a.worker_config)
''')
    with rpc.RpcProcess(tmp_path,sys.executable,'physical','worker',script_path=script) as worker:
        ready=worker.request('initialize',system=dict(norb=4,rank=8,ld=2,input_sha256=io.sha(source)),
            input_rel=source.name,reuse_common_pickle=saved.name)
        assert ready['status']=='READY'
    with (tmp_path/ready['common_pickle']).open('rb') as f:observed=pickle.load(f)
    assert observed['ctx'].d==36 and observed['input_identity']==common['input_identity']
    np.testing.assert_array_equal(observed['physical_reference'],common['physical_reference'])
    np.testing.assert_array_equal(observed['basis'],common['basis'])
