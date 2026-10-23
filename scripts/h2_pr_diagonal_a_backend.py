"""Dormant scientific adapter; no authorization or active real execution entry.

Reuses fixed scientific implementations without copying formulas. The current
public driver refuses real evaluation. These methods also require an activation
object that is never issued by this SOURCE. Future authorization work must bind
that object to new SOURCE/approval/grant/import-closure before any real call.
"""
from pathlib import Path
import h2_pr_diagonal_a as protocol
import h2_pr_identity_recovery as r


def _require_activation(activation):
    # Deliberately no activation implementation or grant issuance in preparation.
    protocol.require(False,'SEPARATE_APPROVED_SCIENCE_SOURCE_ACTIVATION_REQUIRED')


class PlannedExistingBackend:
    domain='SAVED_H2'
    def __init__(self,activation,private_binding):
        _require_activation(activation)
        self.cfg=private_binding

    def bootstrap(self):
        # Deferred imports are never reached by current CLI/tests.
        import importlib
        self.np=importlib.import_module('numpy')
        self.gaussian=importlib.import_module('h46_new2_native')
        self.ops=importlib.import_module('trotterlib.df_trotter.ops')
        self.df=importlib.import_module('trotterlib.df_hamiltonian')
        self.partial=importlib.import_module('trotterlib.df_partial_s2')

    def setup(self):
        # Minimal immutable input loader; no OccupationOracle/setup spectra,
        # SCF, DF recomputation, state generation or saved-QPY reads.
        np=self.np;receipt=r.loads(Path(self.cfg['snapshot_receipt']).read_bytes())
        protocol.require(r.file_digest(self.cfg['snapshot'])==receipt['sha256']==self.cfg['binding']['snapshot'],'SNAPSHOT_SHA')
        with np.load(self.cfg['snapshot'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        import hashlib
        for key,value in arrays.items():
            observed=np.ascontiguousarray(value);saved=receipt['arrays'][key]
            protocol.require(list(observed.shape)==saved['shape'] and observed.dtype.str==saved['dtype'] and
                hashlib.sha256(observed.tobytes()).hexdigest()==saved['sha256'],'SNAPSHOT_ARRAY_SHA')
        ham=self.df.DFHamiltonian(float(arrays['constant']),arrays['one_body'],arrays['lambdas'],
                                 tuple(arrays['g_matrices']),receipt['hamiltonian_metadata'])
        protocol.require(ham.n_qubits==4 and ham.n_blocks==3 and
            self.partial.df_hamiltonian_hash(ham)==self.cfg['binding']['hamiltonian'],'ORIGINAL_HAMILTONIAN')
        return ham

    def prepare(self,ham):
        # Same function and QR/preparation algorithm as old common production.
        groups=[];previous=self.gaussian.gaussian_operations
        def observe(rotation):
            ops=previous(rotation)
            import hashlib
            groups.append(dict(rotation_sha256=hashlib.sha256(self.np.ascontiguousarray(rotation).tobytes()).hexdigest(),
                operations=[dict(gate=g.name,params=[float(x).hex() for x in g.params],wires=list(w)) for g,w in ops]))
            return ops
        self.gaussian.gaussian_operations=observe
        try:prep=self.gaussian.prepare(ham,1)
        finally:self.gaussian.gaussian_operations=previous
        protocol.require(prep.preparation_hash==self.cfg['binding']['preparation'] and
            r.digest(groups)==self.cfg['binding']['common_operations'],'ORIGINAL_PREPARATION_AND_QR_FIELDS')
        protocol.require([b.block_kind for b in prep.deterministic_blocks]==list(protocol.BLOCKS),'ORIGINAL_TWO_BLOCKS')
        return prep

    def primitive(self,prep,block,argument_hex):
        # No division here: exact primitive argument has already been fixed.
        argument=protocol.canonical_hex(argument_hex)
        item=prep.deterministic_blocks[protocol.BLOCKS.index(block)]
        if block=='one_body':
            return self.ops.one_body_diagonal_primitives(self.np.asarray(item.diagonal_eigenvalues),argument)
        return self.ops.df_squared_diagonal_primitives(self.np.asarray(item.diagonal_eta),item.lam,argument)

    def input_dependencies(self):
        return [dict(role='snapshot',sha256=self.cfg['binding']['snapshot']),
                dict(role='snapshot_receipt',sha256=self.cfg['binding']['snapshot_receipt']),
                *[dict(role='scientific_SOURCE',sha256=x) for x in self.cfg['producer_sources'].values()]]

    def compare(self,job,raw,sha256):
        # This stage alone may see original QPY/event/result paths. The
        # independent setup/prepare/primitive interface never receives a job.
        import hashlib
        import json
        import audit_h2_pr_original_identity as old
        protocol.require(hashlib.sha256(raw).hexdigest()==sha256,'SEALED_INDEPENDENT_RECEIPT_BYTES')
        circuit=old.original_qpy(Path(job['wrapper_path']),job['codec_receipt'])
        packet=json.loads(raw)
        return protocol.compare_saved_axis(packet,self.cfg['draft'],circuit,job['events'],
            self.cfg['common_preparation'],job['axis'],job['scalar'],job['piece_times'],job['identity'])
