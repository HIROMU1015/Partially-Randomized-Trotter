"""Reuse byte receipts and pin a draft only. This command cannot issue a grant."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import h8_key_compile_contract as c


def prepare(previous_config, directory, *, source=None):
    old=c.read(previous_config);directory=c.home(directory)
    c.require(not directory.exists(),'EXCLUSIVE_NEW_DRAFT_DIRECTORY')
    specs=c.binding()['inputs'];by={r['logical_name']:r for r in specs}
    files=[dict(by[name],path=old['input_mapping'][name]) for name in by]
    cfg=dict(schema='H8_KEY_COMPILE_CONFIG_V2',SOURCE=source,execution_sealed=False,
        plan_sha256=c.digest(c.plan()),source_hashes={name:c.sha(c.ROOT/name) for name in c.source_paths()},
        provider_source=c.PROVIDER,provider_root=old['provider_root'],provider_hashes=old['provider_hashes'],
        runtime_file_hashes=old['runtime_file_hashes'],python=old['python'],python_sha256=c.sha(old['python']),
        expected_profile_digest=c.PROFILE,files=files,input_mapping={r['logical_name']:r['path'] for r in files},
        font_seed=old['font_seed'],output=str(directory/'output01'),control=str(directory/'control01'),
        lock=str(directory/'KEY_COMPILE_ONE_SHOT'))
    check=c.verify(cfg,runtime=False,git=source is not None)
    grant=dict(schema='H8_KEY_COMPILE_GRANT_V2',scope=c.SCOPE,SOURCE=source,approved=False,
        runtime_authorization=False,execution_sealed=False,approval_text=None,
        plan_sha256=cfg['plan_sha256'],config_sha256=c.digest(cfg),call_caps=c.CAPS,
        one_shot=True,automatic_retry=False,cost_reaccount_authorized=False,next_stage_authorized=False,
        driver_cpu=None,worker_cpus=[],allowed_cpus=[],absolute_launch_command=None,valid=False)
    directory.mkdir(mode=0o700)
    c.save(directory/'config_UNAPPROVED_PRIVATE.json',cfg);c.save(directory/'grant_INVALID_DRAFT.json',grant)
    receipt=dict(status='PREPARED_UNAPPROVED_UNSEALED',SOURCE=source,plan_sha256=cfg['plan_sha256'],
        input_receipts=specs,source_hashes=cfg['source_hashes'],compiler_hash=c.COMPILER,profile_sha256=c.PROFILE,
        provider_map_sha256=c.digest(cfg['provider_hashes']),runtime_map_sha256=c.digest(cfg['runtime_file_hashes']),
        full_runtime_rehashed_during_preparation=False,fresh_gate_issued=False,
        source_and_inputs=check,NPZ_arrays_loaded=0,native_build=0,control_simulation=0,compile=0,
        fit=0,cost_reaccount=0,new_valid_grants=0,approved=False,runtime_authorization=False,
        absolute_launch_command=None,mandatory_STOP=True)
    c.save(directory/'preparation_receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-config',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--source');a=p.parse_args();prepare(a.previous_config,a.directory,source=a.source)


if __name__=='__main__':main()
