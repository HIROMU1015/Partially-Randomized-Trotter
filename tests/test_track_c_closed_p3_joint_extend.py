"""The retired coarse holdout must never enter final independent evaluation."""
import gzip
import importlib
import json
from trottertracks import track_c_closed_p3_joint_opt as j


def test_extension_training_and_holdout_are_disjoint(monkeypatch,tmp_path):
    coarse=tmp_path/'coarse';out=tmp_path/'extended'
    monkeypatch.setattr(j,'OUT',out)
    old=j.cr.load_prices
    monkeypatch.setattr(j.cr,'load_prices',old)
    ex=importlib.import_module('trottertracks.track_c_closed_p3_joint_extend')
    monkeypatch.setattr(ex,'COARSE',coarse)
    for directory,value in [(coarse/'training',1),(coarse/'holdout',2),(out/'extension_training',3),(out/'extension_holdout',99)]:
        directory.mkdir(parents=True)
        with gzip.open(directory/'one.json.gz','wt') as f:json.dump(dict(t=.02,n=1,rare=False,j=0,marker=value),f)
    training=ex.load_prices('training');held=ex.load_prices('holdout');key=j.cp.cellkey(.02,1)
    assert [r['marker'] for r in training[key]['False']]==[1,2,3]
    assert [r['marker'] for r in held[key]['False']]==[99]
