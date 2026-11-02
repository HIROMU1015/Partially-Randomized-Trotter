"""Known saved paths and hashes only, no numerical backend or QPY decode."""
import importlib.util
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
OLD0 = ROOT / 'artifacts/non_df_h4_central_r0_empirical/2026-10-24'
OLD1 = ROOT / 'artifacts/non_df_h4_central_r1_order2/2026-10-24'
spec = importlib.util.spec_from_file_location('reused_metadata', OLD0 / 'prepare_metadata.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
read, sha, digest, save = M.read, M.sha, M.digest, M.save


if __name__ == '__main__':
    files = dict(read(OLD1 / 'inputs.json')['files'])
    for p, h in files.items(): assert sha(ROOT / p) == h, p
    for old, names in [(OLD0, ['run01/PR0_C_target.json']), (OLD1, ['run01/PR1_C_target.json', 'run01/state_identity.json',
        'run01/STOP.json', 'run01/results.json', 'run01/grant.json', 'run01/CONSUMED.json', 'seal.json', 'runtime.json', 'verification.json'])]:
        for name in names:
            p = old / name
            files[str(p.relative_to(ROOT))] = sha(p)
    for path in ['artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PF/source_binding.json',
                 'artifacts/non_df_h4_a_axes/2026-10-24/run01/trajectories/PR0000/source_binding.json']:
        files[path] = sha(ROOT / path)
    rows = read(ROOT / 'artifacts/non_df_h4_central_rho_preparation/2026-10-24/targets.json')['targets']
    sine = [t for t in rows if t['axis'] == 'sine']
    cosine = [next(t for t in rows if t['key'] == s['key'].replace('sine', 'cosine')) for s in sine]
    save(BASE / 'inputs.json', dict(kind='R1_SINE_FIXED_INPUTS_V1', files=files, targets=sine, cosine_targets=cosine,
        new_state_actions=0, new_C_target_state_actions=0))
    preserved = dict(read(OLD1 / 'preservation_before.json'))
    for p, h in preserved.items(): assert sha(ROOT / p) == h, p
    paths = subprocess.check_output(['git', 'ls-files', '--', str(OLD1.relative_to(ROOT))], cwd=ROOT, text=True).splitlines()
    paths += ['docs/non_df_h4_central_r1_order2_20261024.md', 'docs/research/研究ノート/2026-10-24_H4_R1_order2.md']
    preserved.update({p: sha(ROOT / p) for p in paths})
    save(BASE / 'preservation_before.json', preserved)
    manifest = read(ROOT / 'artifacts/validation_manifest.json')
    save(BASE / 'prior_manifest_rows.json', dict(row_count=len(manifest), row_sha256={k: digest(v) for k, v in manifest.items()},
        base_commit='c7585f2d6e4a0fa3f0cc873fb6e468be1d42bce9'))
    save(BASE / 'cache_binding.json', dict(PR0='artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/PR0_C_target.json',
        PR1='artifacts/non_df_h4_central_r1_order2/2026-10-24/run01/PR1_C_target.json',
        PF='saved C PF corrected_encoded_mean.imag interval / same exact raw nu',
        new_C_target_state_actions=0, new_cosine_state_actions=0))
    print(M.json.dumps(dict(inputs=len(files), preserved=len(preserved), prior_manifest_rows=len(manifest), new_state_actions=0)))
