"""Static SOURCE/import/ELF evidence only; no scientific library is loaded."""
import ast
from pathlib import Path
import re
import subprocess
import h2_pr_identity_recovery as r


def source_record(path, role, needles, counterpart=None):
    path=Path(path);raw=path.read_bytes();text=raw.decode('utf8')
    tree=ast.parse(text);imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports += [a.name for a in node.names]
        elif isinstance(node,ast.ImportFrom):imports.append('.'*node.level+(node.module or ''))
    hits=[dict(line=i,text=line.strip()) for i,line in enumerate(text.splitlines(),1) if any(s in line for s in needles)]
    other=Path(counterpart) if counterpart else None
    return dict(role=role,filename=path.name,sha256=r.file_digest(path),
        counterpart_exists=other.is_file() if other else None,
        counterpart_byte_match=other.is_file() and other.read_bytes()==raw if other else None,
        imports=sorted(set(imports)),source_anchors=hits,scientific_module_imported=False)


def needed_libraries(text):
    return re.findall(r'\(NEEDED\).*?\[(.*?)\]', text)


def elf_record(path, role):
    path=Path(path)
    result=subprocess.run(['readelf','-d',str(path)],capture_output=True,text=True)
    r.require(result.returncode==0,'ELF_DYNAMIC_METADATA_READ_FAILED')
    needed=needed_libraries(result.stdout)
    return dict(role=role,filename=path.name,sha256=r.file_digest(path),DT_NEEDED=needed,
        direct_lib_lzma_declared=any('lzma' in s for s in needed),library_loaded=False,
        scope='ELF direct dynamic dependencies only. Not a transitive closure or proof of worker load/non-load.')


def investigate(old, compare):
    sealed=old['sealed_files']
    old_init=next(Path(x['path']) for x in sealed if x['path'].endswith('/numpy/__init__.py'))
    old_site=old_init.parents[1]
    new_site=next(Path(p) for p in compare['runtime']['sys_path'] if 'site-packages' in p)
    acquisition=Path(compare['acquisition_root'])/'scripts'
    provider=Path(compare['provider_root'])/'src'
    sources=[]
    for rel,role,needles in (
        ('qiskit/circuit/parameterexpression.py','SYMBOLIC_PARAMETER_ENGINE',('import symengine','numpy.isfinite','symbol_type =')),
        ('qiskit/circuit/parameter.py','SCALAR_GUARD_PARAMETER',('symengine','ParameterExpression','Symbol(')),
        ('qiskit/synthesis/one_qubit/one_qubit_decompose.py','NUMERIC_ONE_QUBIT_SYNTHESIS',('import numpy','qiskit._accelerate','unitary_to_gate_sequence','asarray(')),
        ('qiskit/transpiler/passes/basis/basis_translator.py','BASIS_TRANSLATION',('_accelerate','Parameter','global_phase','run(')),
        ('qiskit/circuit/library/standard_gates/xx_plus_yy.py','GAUSSIAN_INVERSE_RULE',('def inverse','return XXPlusYYGate(-self.params[0]')),
        ('qiskit/circuit/library/standard_gates/p.py','PHASE_INVERSE_RULE',('def inverse','return PhaseGate(-self.params[0]'))):
        sources.append(source_record(old_site/rel,role,needles,new_site/rel))
    for rel,role,needles in (
        ('h2_m5_compiler_phase_guard.py','RELEVANT_SYMENGINE_COMPILER_PATH',('Parameter','scalar=','before==after','state_fit','transpile')),
        ('h46_new2_native.py','COMMON_PREPARATION_NUMERIC_DEPENDENCY',('np.linalg','eigh','gaussian_operations','np.asarray','diagonal_eigenvalues'))):
        sources.append(source_record(acquisition/rel,role,needles))
    for rel,role,needles in (
        ('trotterlib/df_trotter/ops.py','ABSOLUTE_DIAGONAL_PARAMETERS',('def one_body_diagonal_primitives','def df_squared_diagonal_primitives','np.asarray','phase +=','tau * lam','angle =')),
        ('trottertracks/resource_applicability/ax2b_native_df_v5.py','DIRECTIONAL_AND_WRAPPER_SOURCE',('def _append_block','append_directional','global_phase','def build_native_hadamard_wrapper','inverse=')),
        ('trotterlib/df_rte_qiskit.py','RTE_REUSE_AND_RELATIVE_PHASE_SOURCE',('accumulated_phase','active_key','can_reuse','inverse=','def _identity_application_phase'))):
        sources.append(source_record(provider/rel,role,needles))
    binaries=[]
    for token,role in (('/qiskit/_accelerate.abi3.so','QISKIT_RUST_ACCELERATE'),
                       ('/numpy/core/_multiarray_umath.','NUMPY_NUMERIC_ABI'),
                       ('/symengine/lib/symengine_wrapper.','SYMENGINE_SYMBOLIC_ABI'),
                       ('/_lzma.cpython-','STDLIB_LZMA')):
        p=next(x['path'] for x in sealed if token in x['path'] and x['path'].endswith('.so'))
        binaries.append(elf_record(p,role))
    return dict(status='STATIC_RELEVANCE_MAP_WITH_UNRESOLVED_OUTPUT_SENSITIVITY',source_records=sources,
      ELF_records=binaries,components=[
       dict(component='Python3.12_vs_3.11',native_generation='Interpreter/numeric orchestration and API execution; saved QPY is fixed',
            compiled_RZ='Potential effect through extension ABI, symbolic/numeric compiler behavior; seed17 alone is not output proof',
            existing_evidence='Both historical and adopted identities, original completed receipts and archived output',
            unresolved='No complete historical worker observation; no cross-runtime output comparison',required_condition='Compiler-relevant output equivalence across exact axes or explicitly adopted archived-cost-object policy'),
       dict(component='NumPy_ABI_and_build_config',native_generation='Preparation numeric arrays/diagonal and Gaussian calculations; saved Gaussian fields already match common receipt',
            compiled_RZ='Source uses numeric synthesis/array conversion; possible output-sensitive threshold or decomposition effects',
            existing_evidence='408 corresponding source files equal, ABI binaries differ; all original raw receipts recovered',
            unresolved='Independent common deterministic diagonal parameter receipt absent; counterfactual compiler output unmeasured',required_condition='Independent absolute-diagonal binding and compiler-relevant sensitivity proof'),
       dict(component='SymEngine_ABI',native_generation='No new gate generation performed here; do not infer ABI irrelevance from numeric final QPY',
            compiled_RZ='Direct symbolic path in scalar_guard creates Parameter/exact_zero_scalar and ParameterExpression imports SymEngine',
            existing_evidence='Qiskit sources/guard and original unchanged-gate-stream receipts match',
            unresolved='Original same-runtime guard guarantee does not compare ABI3.12 to ABI3.11',required_condition='Sensitivity test/proof with guard active for affected symbolic translation paths'),
       dict(component='Qiskit1.3.0_and_Rust_acceleration',native_generation='830 corresponding files equal including serialized gate semantics',
            compiled_RZ='Translator/decomposer compiled extension participates; identical package bytes strengthen source identity',
            existing_evidence='Original versus adopted 830/830 file match; ELF direct dependencies recorded',
            unresolved='Byte equality of package does not prove all transitive runtime libraries or numeric inputs equal',required_condition='Keep fixed settings/source plus relevant closure/output sensitivity'),
       dict(component='liblzma_drift',native_generation='No direct dependency established in inspected circuit/compiler paths',
            compiled_RZ='No direct cost effect established; absence from selected DT_NEEDED is not non-load proof',
            existing_evidence='Historical seal vs current library differs; 402 zlib archive/raw hashes verified',
            unresolved='Full historical actual-load membership unknown',required_condition='Preserve unknown membership; archived-byte identity protects stored evidence, not counterfactual compilation')],
      full_runtime_byte_identity=False,historical_actual_worker_manifest_available=False,
      historical_load_reconstructed=False,A_does_not_imply_B=True,new_scientific_calls=0,new_scientific_imports=0)
