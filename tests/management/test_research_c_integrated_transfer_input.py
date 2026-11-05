"""Invented chemistry interfaces only; no real molecule or saved array reads."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_integrated_transfer_input as m
from trottertracks import c1_molecule_extension as existing


def test_fixed_target_metadata_and_passive_import():
    s = m.specification()
    assert (s['norb'], s['rank'], s['LD']) == (4, 8, 2)
    assert (s['Nalpha'], s['Nbeta'], s['sector_dimension']) == (2, 2, 36)
    assert s['distance_angstrom'] == 1.15 and s['orbital_frame'] == 'canonical_RHF_MO'
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_integrated_transfer_input as m;m.specification();assert not any(x in sys.modules for x in ['numpy','scipy','pyscf','ffsim','qiskit'])"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


def test_H4_115_uses_existing_RHF_ao2mo_without_new_integral_engine():
    n = 4
    h = np.diag(np.arange(n, dtype=float)/10)
    # Deliberately invented non-physical array: mock export/dispatch only.
    g = np.arange(n**4, dtype=float).reshape((n,)*4)
    mf = SimpleNamespace(converged=True, conv_tol=None, mo_coeff=np.eye(n),
                         kernel=Mock(), get_hcore=lambda: h,
                         get_ovlp=lambda: np.eye(n))
    mol = SimpleNamespace(energy_nuc=lambda: .125)
    mol_factory = Mock(return_value=mol)
    pyscf = SimpleNamespace(gto=SimpleNamespace(M=mol_factory),
                            scf=SimpleNamespace(RHF=Mock(return_value=mf)),
                            ao2mo=SimpleNamespace(kernel=Mock(return_value='MOCK_ERI'),
                                                  restore=Mock(return_value=g)))
    calls = []
    def call(kind, label, f):
        calls.append(kind)
        return f()
    with patch.dict(sys.modules, {'pyscf': pyscf}), \
         patch('numpy.load', side_effect=AssertionError('NO_SAVED_ARRAY_READ')):
        packet = m.generate(call)
    assert calls == ['scf', 'spatial_integrals'] and mf.conv_tol == 1e-12
    k = mol_factory.call_args.kwargs
    assert k['atom'] == [('H', (0., 0., i*1.15)) for i in range(4)]
    assert k['unit'] == 'Angstrom' and k['basis'] == 'sto-3g'
    assert k['charge'] == k['spin'] == 0
    np.testing.assert_array_equal(packet['h'], h)
    np.testing.assert_array_equal(packet['g'], g)
    assert packet['constant'].shape == () and float(packet['constant']) == .125
    pyscf.ao2mo.restore.assert_called_once_with(1, 'MOCK_ERI', 4)


def test_factory_delegation_and_old_entry_conditions_are_preserved():
    seen = []
    def factory(spec, call):
        seen.append(spec)
        return 'ARTIFICIAL_PACKET'
    assert m.generate(lambda *a: None, factory=factory) == 'ARTIFICIAL_PACKET'
    assert seen[0] == m.specification()
    for n in (2, 6):
        existing.integrals(n, lambda *a: None, factory=factory)
        assert seen[-1] == existing.specification(n)
    with pytest.raises(ValueError, match='Only registered H2/H6'):
        existing.integrals(4, lambda *a: None, factory=factory)


@pytest.mark.parametrize('change', [dict(distance_angstrom=-1),
                                   dict(distance_angstrom=float('nan')),
                                   dict(distance_angstrom=True), dict(norb=3),
                                   dict(basis='other'), dict(spin=2)])
def test_invalid_recipe_rejected_before_chemistry(change):
    spec = dict(m.specification(), **change)
    factory = Mock(side_effect=AssertionError('NO_CHEMISTRY'))
    with pytest.raises(ValueError, match='INPUT_RECIPE'):
        existing.integrals_from_specification(spec, lambda *a: None, factory=factory)
    factory.assert_not_called()
