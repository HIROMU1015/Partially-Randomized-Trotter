"""Reuse sealed renderer with smaller tick labels; preserve first exports."""
import json
from pathlib import Path
import tempfile
import os

import plot_pf_q_model_diagnostics as renderer
from run_pf_q_model_diagnostics import OUT, sha


def main():
    destination=OUT/'display'
    destination.mkdir(exist_ok=False)
    completion=json.loads((OUT/'numeric_completion.json').read_text())
    links=[]
    try:
        for name in ['numeric_completion.json',*completion['result_hashes']]:
            path=destination/name;path.symlink_to(OUT/name);links.append(path)
        def write(name,value):
            (destination/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        with tempfile.TemporaryDirectory(prefix='pf-q-display-') as cache:
            os.environ['MPLCONFIGDIR']=cache
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams.update({'xtick.labelsize':7,'ytick.labelsize':8})
            renderer.OUT=destination;renderer.write=write;renderer.main()
        if sha(destination/'figure_data.json')!=sha(OUT/'figure_data.json'):
            raise ValueError('DISPLAY_REVISION_MUST_KEEP_IDENTICAL_NUMERIC_FIGURE_DATA')
        write('display_provenance.json',dict(status='DISPLAY_ONLY_SAME_FIGURE_DATA',
            SOURCE=completion['SOURCE'],wrapper_sha256=sha(Path(__file__)),
            original_renderer_sha256=sha(Path(renderer.__file__)),
            original_rendering_completion_sha256=sha(OUT/'rendering_completion.json'),
            figure_data_sha256=sha(destination/'figure_data.json'),
            changes={'xtick.labelsize':7,'ytick.labelsize':8},new_numeric_analysis=False,
            first_exports_preserved=True))
    finally:
        for path in links:path.unlink()


if __name__=='__main__':main()
