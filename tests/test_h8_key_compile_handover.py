"""Artificial byte-only handover; no real circuit construction or compile."""
from copy import deepcopy
from fractions import Fraction
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h8_key_compile_contract as c
import h8_key_compile_handover as h
import h8_key_compile_preparation as prep
import h8_key_compile_adapter as adapter
from test_h8_key_compile import temp
import test_h8_key_compile_preparation as prep_tests
artificial_permit=prep_tests.artificial_permit


def fixture(directory,*,measured=False):
    root=Path(directory)/'old';(root/'metadata').mkdir(parents=True);(root/'output').mkdir()
    inp=root/'source'/c.INPUTS;inp.parent.mkdir(parents=True);inp.write_bytes((c.ROOT/c.INPUTS).read_bytes())
    case=dict(id='artificial',method='B0',prefix=1,q=1,formula='2nd')
    identity=dict(constant_hex=(1.).hex());target=dict(case_id='artificial',corrected=dict(real=1.,imag=0.),independent_forward=dict(real=1.,imag=0.))
    binding=dict(cases=[case],prefix_preparations={'1':identity},targets=[target],axes=[dict(case_id='artificial',axis=a,exact_prediction='100') for a in ('cosine','sine')])
    cfg=dict(SOURCE=h.OLD_SOURCE,expected_profile_digest=c.PROFILE,provider_hashes={},runtime_file_hashes={},files=[],input_mapping={},
             plan_sha256='old-plan',source_hashes={c.INPUTS:c.sha(inp)})
    grant=dict(SOURCE=h.OLD_SOURCE,approved=True,runtime_authorization=True,config_sha256=c.digest(cfg),plan_sha256='old-plan')
    c.save(root/'metadata/config_SEALED_PRIVATE.json',cfg);c.save(root/'metadata/grant_APPROVED_PRIVATE.json',grant)
    new=dict(cfg,SOURCE='f'*40,output=str(Path(directory)/'new_output'),control=str(Path(directory)/'new_control'))
    new['handover']=dict(old_root=str(root),old_SOURCE=h.OLD_SOURCE,
        old_config_file_sha256=c.sha(root/'metadata/config_SEALED_PRIVATE.json'),old_grant_file_sha256=c.sha(root/'metadata/grant_APPROVED_PRIVATE.json'),
        release_path=str(Path(new['control'])/'HANDOVER_RELEASE.json'))
    def group(name,values):
        directory=root/'output'/name;directory.mkdir();workers=[]
        for axis,value in values.items():
            work=directory/axis;work.mkdir();path=work/'task_result.json';c.save(path,value)
            workers.append(dict(task_id=axis,directory=str(work),exit_code=0,reaped=True,result_sha256=c.sha(path),identity=[99999999,1,1,os.getuid()]))
        c.save(directory/'parallel_receipt.json',dict(status='PASS',all_reaped=True,cleanup_errors=[],workers=workers))
    group('artificial_directional_controls',dict(left=dict(errors=[0.]),right=dict(errors=[0.,0.,0.],overlap=dict(real=1.,imag=0.))))
    spools={};fingerprints={a:'artificial-'+a for a in ('cosine','sine')}
    for axis in ('cosine','sine'):
        path=root/'output'/(axis+'.qpy');path.write_bytes(('FAKE_QPY_'+axis).encode())
        spools[axis]=dict(path=str(path),bytes=path.stat().st_size,sha256=c.sha(path))
    group('artificial_axis_controls_and_fingerprints',{a:dict(axis=a,fingerprint=fingerprints[a],qpy_spool=spools[a],max_control_error=0.) for a in spools})
    proof=dict(case=case,preparation_binding=identity,scalar_hex=(1.).hex(),max_control_error=0.,fingerprints=fingerprints,spools=spools,SOURCE=h.OLD_SOURCE,profile_sha256=c.PROFILE)
    c.save(root/'output/artificial_control_preparation.json',proof)
    if measured:
        raw=dict(case_id='artificial',axis='cosine',SOURCE=h.OLD_SOURCE,model_sha256=c.MODEL,compiler_hash=c.COMPILER,
            profile_sha256=c.PROFILE,measured_RZ=100,preparation_binding=identity,scalar_hex=(1.).hex(),max_control_error=0.,
            fingerprint_before=fingerprints['cosine'],fingerprint_after=fingerprints['cosine'])
        group('compile_01',{'artificial_cosine':raw});path=root/'output/compile_01/artificial_cosine/task_result.json'
        c.save(root/'output/axis_01_measured.json',dict(raw,native_exit_code=0,reaped=True,result_sha256=c.sha(path),model_check=c.axis_check('100',100)))
    native=dict(SOURCE=h.OLD_SOURCE,status='NATIVE_LAUNCHER_REAPED',reaped=True,native_exit_code=1)
    c.save(root/'metadata/launcher_native_exit.json',native)
    (root/'control/driver').mkdir(parents=True)
    c.save(root/'control/driver/parallel_receipt.json',dict(status='STOP',all_reaped=True,cleanup_errors=[],workers=[]))
    c.save(root/'control/STOP.json',dict(first_stop='OWN_LAUNCHER_SIGNAL:15'))
    c.save(root/'output/terminal.json',dict(status='STOP',first_stop='OWN_WORKER_SIGNAL:15'))
    c.save(root/'output/reservation_compile_1.json',dict(call='compile',ordinal=1))
    return new,grant,binding,root


class Handover(unittest.TestCase):
    def test_byte_and_native_proofs_imported_without_science(self):
        with temp() as d:
            cfg,g,b,root=fixture(d,measured=True);out=Path(cfg['output']);out.mkdir();Path(cfg['control']).mkdir()
            with patch.object(c,'binding',return_value=b):
                audit=h.audit_old(cfg);self.assertEqual(len(audit['rows']),1)
                c.save(cfg['handover']['release_path'],dict(status='PASS_OLD_NATIVE_EXIT_AND_REUSABLE_PROGRESS',**h.header(cfg,g),**audit))
                prepared,rows,prior=h.import_prepared(cfg,g,out)
            self.assertEqual(prior['compile'],1);self.assertEqual(rows[0]['SOURCE'],h.OLD_SOURCE)
            self.assertEqual(prepared['artificial']['preparation_SOURCE'],h.OLD_SOURCE)
            for axis,spool in prepared['artificial']['spools'].items():
                self.assertTrue(Path(spool['path']).is_relative_to(out))
                self.assertEqual(Path(spool['path']).read_bytes(),('FAKE_QPY_'+axis).encode())

    def test_missing_native_exit_wrong_stop_or_live_identity_rejected(self):
        for mode in ('native','stop','identity'):
            with self.subTest(mode=mode),temp() as d:
                cfg,g,b,root=fixture(d)
                if mode=='native':(root/'metadata/launcher_native_exit.json').unlink()
                if mode=='stop':(root/'control/STOP.json').write_text('{"first_stop":"SCIENTIFIC_MISMATCH"}')
                with patch.object(c,'binding',return_value=b),patch.object(h.pool,'identity',return_value=(99999999,1,1,os.getuid()) if mode=='identity' else None):
                    with self.assertRaises((ValueError,FileNotFoundError)):h.audit_old(cfg)

    def test_changed_qpy_auth_or_failed_measurement_rejected(self):
        for mode in ('qpy','auth','model'):
            with self.subTest(mode=mode),temp() as d:
                cfg,g,b,root=fixture(d,measured=True)
                if mode=='qpy':(root/'output/sine.qpy').write_bytes(b'CHANGED')
                elif mode=='auth':(root/'metadata/grant_APPROVED_PRIVATE.json').write_text('{}')
                else:
                    path=root/'output/axis_01_measured.json';row=c.read(path);row['measured_RZ']=1;path.write_text(__import__('json').dumps(row))
                with patch.object(c,'binding',return_value=b),self.assertRaises(ValueError):h.audit_old(cfg)

    def test_release_binding_mismatch_rejected(self):
        cfg={'SOURCE':'a'*40};g={'approved':True}
        r=h.header(cfg,g);r['config_sha256']='bad'
        with self.assertRaises(ValueError):h.check_header(r,cfg,g)

    def test_existing_preparation_skips_build_and_credits_reuse(self):
        helper=prep_tests.PreparationContract();cases=c.binding()['cases']
        with temp() as d:
            permit=artificial_permit(d,range(6));seen=[];original=helper.fake_map()
            def execute(*args,**kwargs):
                helper.assertEqual(len(kwargs['cpus']),2)
                # Existing fake helper expects three CPU entries; only its observation is normalized.
                kwargs['cpus']=[0,2,4]
                return original(*args,**kwargs)
            def build(case,child,cpus):seen.append(case['id']);return helper.consume(case,child,cpus)
            result=prep.prepare_cases(permit,build,existing={cases[0]['id']:{'imported':True}},map_owned=execute)
            self.assertEqual(seen,[v['id'] for v in cases[1:]])
            self.assertEqual(result[cases[0]['id']],{'imported':True})
            for key,count in c.CASE_PREPARATION_CAPS.items():
                self.assertEqual(permit.calls[key],3*count);self.assertEqual(permit.new_calls[key],2*count)
                self.assertEqual(permit.reused_calls[key],count)

    def test_reused_axis_never_dispatched_and_next_failure_stops(self):
        with temp() as d:
            permit=artificial_permit(d,range(6));axis=c.binding()['axes'][0]
            reused=[dict(case_id=axis['case_id'],axis=axis['axis'],measured_RZ=round(Fraction(axis['exact_prediction'])))]
            seen=[]
            def fail(tasks,*args,**kwargs):seen.append(tasks[0]['axis']);raise RuntimeError('NEXT_AXIS_TEST_STOP')
            with self.assertRaisesRegex(RuntimeError,'NEXT_AXIS_TEST_STOP'):
                adapter.serial_axes(permit,lambda case:{},None,map_owned=fail,reused=reused)
            self.assertEqual(seen,['sine']);self.assertEqual(permit.reused_calls['compile'],1)
            self.assertEqual(permit.new_calls['compile'],1)

    def test_atomic_release_never_replaces_existing_bytes(self):
        with temp() as d:
            path=Path(d)/'release.json';h.publish(path,{'PASS':True})
            with self.assertRaises(FileExistsError):h.publish(path,{'PASS':False})
            self.assertEqual(c.read(path),{'PASS':True})


if __name__=='__main__':unittest.main()
