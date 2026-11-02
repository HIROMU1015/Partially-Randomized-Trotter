"""Controlled future chemistry path; no molecular imports until generation permit."""
import io
import json
import math
from .identity import Stop, require, sha, fingerprint
from .gates import DISTANCES, Permit

SCF_CONTROLS = dict(init_guess='minao', conv_tol=1e-9, conv_tol_grad=math.sqrt(1e-9),
    max_cycle=50, diis_space=8, diis_start_cycle=1, diis_damp=0, diis_space_rollback=0,
    diis_file=None, damp=0, level_shift=0, direct_scf=True, direct_scf_tol=1e-13,
    max_memory=4000, chkfile=None, check_convergence=None, conv_check=True,
    disp=None, disp_with_3body=False)


def coordinates(distance):
    require(distance in DISTANCES, 'distance outside adopted scope')
    d = float(distance)
    return [('H', (0.0, 0.0, (i-1.5)*d)) for i in range(4)]


def apply_scf_controls(mf, cdiis_class):
    for k, v in SCF_CONTROLS.items():
        setattr(mf, k, v)
    # Preserve PySCF's ordinary DIIS creation and Corth initialization.
    class MemoryCDIIS(cdiis_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.incore = True
    mf.DIIS, mf.diis = MemoryCDIIS, True
    return mf


def initialize_no_save(mf,mol):
    """Pinned SCF constructor state without tempfile creation or global config."""
    require(mol._built,'molecule must be built')
    mf.mol,mf.verbose,mf.max_memory,mf.stdout=mol,mol.verbose,mol.max_memory,mol.stdout
    mf.chkfile,mf._chkfile=None,None
    mf.mo_energy=mf.mo_coeff=mf.mo_occ=None
    mf.e_tot,mf.converged,mf.cycles=0,False,0
    mf.scf_summary,mf._opt,mf._eri={},{None:None},None
    return mf


def scf_accept(converged, main_cycles, final_gradient, arrays):
    import numpy as np
    require(bool(converged), 'SCF not converged; no rescue')
    require(any(abs(de)<1e-9 and grad < math.sqrt(1e-9) for de, grad in main_cycles), 'strict main cycle')
    require(math.isfinite(final_gradient) and final_gradient < math.sqrt(1e-9), 'independent final raw gradient')
    require(all(np.all(np.isfinite(a)) for a in arrays), 'nonfinite SCF/integrals')


def canonical_mos(energies, coeff):
    import numpy as np
    e, c = np.asarray(energies), np.asarray(coeff).copy()
    require(e.shape == (4,) and c.shape == (4, 4) and not np.iscomplexobj(c), 'real four spatial MOs')
    require(np.all(np.isfinite(e)) and np.all(np.isfinite(c)) and np.all(np.diff(e) > 1e-12*max(1, np.max(np.abs(e)))), 'MO order/degeneracy')
    for i in range(4):
        pivot = int(np.argmax(np.abs(c[:, i])))
        if c[pivot, i] < 0:
            c[:, i] *= -1
    return c


def factorize(two_body, *, chemist_transform):
    """Port of pinned OpenFermion low_rank.py; one full eigensolver invocation.

    Additional D2 gates and retained raw eigenpairs; no reordering by Frobenius.
    The supplied transform is the pinned spin_basis=True OpenFermion helper.
    """
    import numpy as np
    correction, chemist = chemist_transform(two_body, spin_basis=True)
    require(chemist.shape == (4,)*4, 'four spatial orbitals required')
    interaction = chemist.reshape(16,16)
    require(np.all(np.isfinite(interaction)) and np.sum(np.abs(interaction-interaction.T))<=1e-8 and
            np.sum(np.abs(interaction.imag))<=1e-8, 'DF tensor real/symmetry gate')
    values, vectors = np.linalg.eigh(interaction)
    squares = np.asarray([np.kron(vectors[:,i].reshape(4,4), np.eye(2)) for i in range(16)])
    weights = np.abs(values)*np.sum(np.abs(squares), axis=(1,2))**2
    order = np.argsort(weights)[::-1]  # exact installed generated order; ties are not stable
    scale = max(1.0, float(np.max(np.abs(values))))
    retained = set(int(i) for i in order[:12])
    for i in range(16):
        for j in range(i+1,16):
            if i in retained or j in retained:
                require(not (abs(values[i]-values[j])/scale<=1e-12 and
                             max(abs(values[i]),abs(values[j]))/scale>1e-12), 'significant DF degeneracy')
    for prefix in (3,4,5,6,9,12):
        for i in order[:prefix]:
            for j in order[prefix:]:
                require(not (weights[i] == weights[j] and max(abs(values[i]),abs(values[j]))/scale>1e-12), 'significant prefix weight tie')
    require(np.sum(np.abs(vectors.imag))<=1e-8, 'complex spatial DF eigenvectors')
    signs = np.ones(16)
    for i in range(16):
        pivot = int(np.argmax(np.abs(vectors[:,i].real)))
        if vectors[pivot,i].real < 0:
            signs[i] = -1
    canonical_vectors = vectors.real*signs
    canonical_squares = squares.real*signs[:,None,None]
    require(len(order[:12]) == 12 and np.all(np.isfinite(canonical_squares)), 'returned fragment count')
    error = np.cumsum(weights[order])[-1]-np.cumsum(weights[order])[11]
    return dict(lambdas=values[order[:12]], G=canonical_squares[order[:12]], correction=correction,
                raw_eigenvalues=values, raw_eigenvectors=vectors, canonical_eigenvectors=canonical_vectors,
                permutation=order, raw_indices=order[:12], signs=signs, weights=weights, truncation_value=error)


def sector_indices():
    return [i for i in range(256) if sum((i>>(7-p))&1 for p in (0,2,4,6))==2
            and sum((i>>(7-p))&1 for p in (1,3,5,7))==2]


def reverse_bits(n, width=8):
    return int(format(n, '0%db' % width)[::-1], 2)


def ground_state(H):
    import numpy as np
    H = np.asarray(H, dtype=np.complex128)
    require(H.shape == (256,256) and np.all(np.isfinite(H)), 'full eight-qubit DF H')
    indices = sector_indices()
    require(len(indices)==36, 'sector dimension')
    relative = float(np.linalg.norm(H-H.conj().T,2))/max(1.,float(np.linalg.norm(H,2)))
    require(relative<=1e-12, 'original H Hermiticity')
    sector = H[np.ix_(indices,indices)]
    energy, vectors = np.linalg.eigh((sector+sector.conj().T)/2, UPLO='L')
    require(energy[1]-energy[0]>1e-10, 'sector ground gap STOP')
    local = vectors[:,0].astype(np.complex128)
    pivot = int(np.argmax(np.abs(local)))
    local *= np.exp(-1j*np.angle(local[pivot]))
    full = np.zeros(256, dtype=np.complex128)
    full[indices] = local
    rayleigh = np.vdot(full,H@full)
    require(abs(np.vdot(full,full)-1)<=1e-12, 'state norm')
    require(abs(rayleigh.imag)<=1e-11, 'imaginary energy')
    require(np.linalg.norm(H@full-rayleigh.real*full)<=1e-9, 'original H residual')
    require(np.max(np.abs(full[indices]-local))<=1e-12 and
            np.max(np.abs(full[[i for i in range(256) if i not in indices]]))<=1e-12, 'sector consistency')
    qiskit = full[[reverse_bits(i) for i in range(256)]]
    return dict(sector=np.asarray(indices,dtype=np.int64), sector_H=sector, sector_state=local,
                state=full, qiskit_state=qiskit, energy=np.asarray(rayleigh.real), gap=np.asarray(energy[1]-energy[0]))


def one_body_operator(matrix):
    # Imported only by the authorized science paths.
    from openfermion import FermionOperator, get_sparse_operator
    op = FermionOperator()
    for p in range(len(matrix)):
        for q in range(len(matrix)):
            if matrix[p,q] != 0:
                op += FermionOperator(((p,1),(q,0)), matrix[p,q])
    return get_sparse_operator(op, n_qubits=len(matrix)).toarray()


def generate_input(permit, distance, pulse):
    require(isinstance(permit, Permit) and permit.stage=='input_generation', 'generation-only permit')
    require(distance in DISTANCES, 'scope')
    # All molecular dependencies and molecules stay behind the production gates.
    import numpy as np
    from pyscf import gto, scf, ao2mo
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from openfermion.circuits.low_rank import get_chemist_two_body_coefficients

    class NoSaveRHF(scf.hf.RHF):
        def __init__(self, mol):
            # Pinned hf.SCF.__init__ port without NamedTemporaryFile. No global patch.
            initialize_no_save(self,mol)

    mol = gto.Mole()
    mol.atom, mol.basis, mol.unit = coordinates(distance), 'STO-3G', 'Angstrom'
    mol.charge, mol.spin, mol.symmetry, mol.cart, mol.incore_anyway = 0, 0, False, False, False
    mol.verbose, mol.max_memory = 0, 4000
    mol.build()
    require(mol.nao_nr()==4 and all(z==1 for z in mol.atom_charges()), 'H4/minao no atom fallback')
    mf = apply_scf_controls(NoSaveRHF(mol), scf.diis.CDIIS)
    cycles = []
    def callback(values):
        pulse()
        grad = np.linalg.norm(mf.get_grad(values['mo_coeff'], values['mo_occ'], values['fock']))
        cycles.append((float(values['e_tot']-values['last_hf_e']),float(grad)))
    mf.callback = callback
    mf.kernel(dm0=None)
    pulse()
    coeff = canonical_mos(mf.mo_energy, mf.mo_coeff)
    hAO, overlap, eriAO = mf.get_hcore(), mf.get_ovlp(), mol.intor('int2e')
    hMO = coeff.T@hAO@coeff
    # AO array forces in-core transform; passing mol invokes outcore temp HDF5.
    eriMO = ao2mo.restore(1, ao2mo.kernel(eriAO,coeff),4).transpose(0,2,3,1)
    fock = mf.get_fock(dm=mf.make_rdm1())
    final = float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ,fock)))
    scf_accept(mf.converged,cycles,final,[mf.e_tot,mf.mo_energy,coeff,hAO,overlap,eriAO,hMO,eriMO])
    hspin, twospin = spinorb_from_spatial(hMO,eriMO)
    df = factorize(twospin*0.5,chemist_transform=get_chemist_two_body_coefficients)
    one = hspin+df['correction']
    base = one_body_operator(one)
    blocks = np.asarray([lam*(one_body_operator(g)@one_body_operator(g))
                         for lam,g in zip(df['lambdas'],df['G'],strict=True)])
    nuclear = float(mol.energy_nuc())
    H = nuclear*np.eye(256)+base+np.sum(blocks,axis=0)
    state = ground_state(H)
    arrays = dict(coordinates=np.asarray([x[1] for x in coordinates(distance)]), hAO=hAO, overlap=overlap,
                  eriAO=eriAO, MOs=coeff, MO_energies=mf.mo_energy, hMO=hMO, eriMO=eriMO,
                  hspin=hspin, twospin=twospin, one=one, nuclear=np.asarray(nuclear), H=H, blocks=blocks,
                  **df, **state)
    return arrays


def array_identity(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    require(a.dtype.kind in 'fciub' and np.all(np.isfinite(a)), 'freeze finite numeric arrays')
    return {'dtype':a.dtype.str, 'shape':list(a.shape), 'bytes_sha256':sha(a.tobytes())}


def validate_frozen_state(arrays):
    """Verify saved state without a new solve, state choice, or substitution."""
    import numpy as np
    H=arrays['H'];full=arrays['state'];local=arrays['sector_state'];indices=sector_indices()
    require(H.shape==(256,256) and full.shape==(256,) and local.shape==(36,), 'frozen state dimensions')
    require(all(np.all(np.isfinite(a)) for a in (H,full,local,arrays['qiskit_state'],arrays['energy'],arrays['gap'])), 'finite frozen state')
    require(np.array_equal(arrays['sector'],indices),'fixed sector order')
    require(abs(np.vdot(full,full)-1)<=1e-12,'frozen norm')
    require(np.linalg.norm(H-H.conj().T,2)/max(1.,float(np.linalg.norm(H,2)))<=1e-12,'frozen Hermiticity')
    rayleigh=np.vdot(full,H@full)
    require(abs(rayleigh.imag)<=1e-11 and abs(rayleigh.real-float(arrays['energy']))<=1e-9,'frozen energy')
    require(np.linalg.norm(H@full-rayleigh.real*full)<=1e-9 and float(arrays['gap'])>1e-10,'frozen residual/gap')
    require(np.max(np.abs(full[indices]-local))<=1e-12 and
            np.max(np.abs(full[[i for i in range(256) if i not in indices]]))<=1e-12,'frozen sector consistency')
    pivot=int(np.argmax(np.abs(local)))
    require(local[pivot].real>0 and abs(local[pivot].imag)<=1e-12,'frozen phase convention')
    require(np.max(np.abs(arrays['qiskit_state']-full[[reverse_bits(i) for i in range(256)]]))<=1e-12,'frozen bit conversion')


def freeze_input(permit, distance, arrays):
    require(permit.stage == 'input_generation', 'freeze generation stage')
    import numpy as np
    ids = {k:array_identity(v) for k,v in sorted(arrays.items())}
    identity = dict(geometry=distance, source_commit=permit.plan['source_commit'], arrays=ids)
    buffer = io.BytesIO()
    # No external snapshot path; fully owned new bytes after authorization only.
    np.savez(buffer, **arrays)
    data = buffer.getvalue()
    record = dict(file='input-'+distance+'.npz', bytes_sha256=sha(data),
        input=fingerprint('h4-input-freeze-v1',identity), H=fingerprint('h4-H-v1',ids['H']),
        DF=fingerprint('h4-DF-v1',{k:ids[k] for k in ('lambdas','G','correction','permutation','raw_indices','raw_eigenvalues','raw_eigenvectors','signs')}),
        state=fingerprint('h4-state-v1',{k:ids[k] for k in ('sector','sector_state','state','qiskit_state')}))
    return data, record, identity
