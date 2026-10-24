"""Prepare manifests from existing R0 and selected A_GATE records only."""
import importlib.util
from pathlib import Path
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
OLD = ROOT / 'artifacts/non_df_h4_central_r0_empirical/2026-10-24'
spec = importlib.util.spec_from_file_location('r0_metadata', OLD / 'prepare_metadata.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
read, sha, digest, save = old.read, old.sha, old.digest, old.save


if __name__ == '__main__':
    files = dict(read(OLD / 'inputs.json')['files'])
    for p, h in files.items():
        assert sha(ROOT / p) == h, p
    for name in ['run01/results.json', 'run01/state_identity.json', 'run01/STOP.json', 'run01/grant.json',
                 'run01/CONSUMED.json', 'seal.json', 'runtime.json', 'verification.json']:
        p = OLD / name
        files[str(p.relative_to(ROOT))] = sha(p)
    target = next(t for t in read(ROOT / 'artifacts/non_df_h4_central_rho_preparation/2026-10-24/targets.json')['targets'] if t['key'] == 'PR1_cosine')
    save(BASE / 'inputs.json', dict(kind='R1_ORDER2_FIXED_INPUTS_V1', files=files, target=target, state_actions_performed=0))
    preservation = dict(read(OLD / 'preservation_before.json'))
    paths = subprocess.check_output(['git', 'ls-files', '--', str(OLD.relative_to(ROOT))], cwd=ROOT, text=True).splitlines()
    paths += ['docs/non_df_h4_central_r0_empirical_20261024.md', 'docs/research/研究ノート/2026-10-24_H4_R0_empirical.md']
    for p, h in preservation.items():
        assert sha(ROOT / p) == h, p
    preservation.update({p: sha(ROOT / p) for p in paths})
    save(BASE / 'preservation_before.json', preservation)
    manifest = read(ROOT / 'artifacts/validation_manifest.json')
    save(BASE / 'prior_manifest_rows.json', dict(row_count=len(manifest), row_sha256={k: digest(v) for k, v in manifest.items()}, base_commit='f01c4018201de9335f20a8901ec5f40f9cf9e89b'))
    shutil.copyfile('/home/AbeHiromu/.codex/attachments/d0b18df2-34f6-459a-835e-f17051ba7526/貼り付けたテキスト.txt', BASE / 'approval.txt')
    print(old.json.dumps(dict(inputs=len(files), preserved=len(preservation), prior_manifest_rows=len(manifest), state_actions=0)))
