"""Complete the fixed holdout with saved completion-order metadata canonicalized.

The 43f748655 model, counter, and original runner stay byte-identical. Only
the in-memory order of the original confirmation trial-file list is changed.
Record this launcher's own execution SOURCE separately from the model SOURCE.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = '43f748655c64756c2c2a21d754870c2f7b321a77'


def canonical_trial_files(files):
    identities = [int(Path(p).parent.name) for p in files]
    if len(files)!=33 or set(identities)!=set(range(33)):
        raise ValueError('expected the original 33 unique confirmation paths')
    return sorted(files,key=lambda p:int(Path(p).parent.name))


def main():
    spec = importlib.util.spec_from_file_location('frozen_v1_runner',ROOT/'scripts/run_h4_df10_price_v1.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    execution_source = runner.git('rev-parse','HEAD').decode().strip()
    adapter_path = 'scripts/complete_h4_df10_price_v1.py'
    if Path(__file__).read_bytes()!=runner.blob(execution_source,adapter_path):
        raise ValueError('commit completion adapter before execution')
    def frozen_binding():
        hashes = {}
        for path in runner.MODEL_FILES:
            current = (ROOT/path).read_bytes()
            if current != runner.blob(FROZEN,path):
                raise ValueError('frozen scientific engine changed')
            hashes[path] = runner.digest(current)
        return FROZEN,hashes
    original_saved = runner.saved
    def ordered_saved(name):
        values = original_saved(name)
        if name=='confirmation_costs.json.gz':
            row = values[runner.PR]
            row['trial_files'] = canonical_trial_files(row['trial_files'])
        return values
    runner.source_binding = frozen_binding
    runner.saved = ordered_saved
    destination = runner.PUBLIC/'HOLDOUT.json'
    if destination.exists():
        raise ValueError('preserve completed holdout')
    result = runner.run('holdout')
    result.update(execution_source=execution_source,
                  completion_adapter_sha256=runner.digest(Path(__file__).read_bytes()),
                  metadata_correction='order original trial-file references by saved path index; original set unchanged',
                  frozen_model_and_runner_bytes_unchanged=True)
    with destination.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','model_source','execution_source','summaries')},indent=2))


if __name__=='__main__':main()
