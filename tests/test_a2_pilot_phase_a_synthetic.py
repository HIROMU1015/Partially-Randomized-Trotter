"""Phase-A artificial NPZs, memory-only authorization vectors and execution store.

No repository scientific arrays are read. No scientific authorization or physical
execution marker is issued by these tests. Run directly with unittest, never pytest.
"""

import copy
from fractions import Fraction as F
from hashlib import sha256
import importlib.util
from io import BytesIO
import itertools
import json
from pathlib import Path
import random
import struct
import unittest
from unittest.mock import patch
import zipfile

from trottercert.a2_spec01_03.common import ContractError, canonical_bytes, content_id, rat, read_rat
from trottercert.a2_spec01_03.raw import validate_raw as old_validate_raw
from trottercert.a2_spec01_03.generator import generate_spec01_03 as frozen_generate01
from trottercert.a2_spec04_07 import producer_algebra as old_pa, verifier_algebra as old_va
from trottercert.a2_pilot import adapter, contract, runner, producer_algebra as pa, verifier_algebra as va
from trottercert.a2_pilot.core01_generator import generate_spec01_03
from trottercert.a2_pilot.core01_checker import check_spec01_03
from trottercert.a2_pilot.core04_generator import generate_spec04_07
from trottercert.a2_pilot.core04_checker import check_spec04_07
from trottercert.a2_pilot.identity import source_identity
from trottercert.a2_pilot.measure import BoundedLog
from trottercert.a2_pilot.storage import LocalStore

_spec = importlib.util.spec_from_file_location("a2_artificial_helpers", Path(__file__).with_name("test_a2_spec01_03_synthetic.py"))
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)


def npy(descr, shape, data, *, fortran=False, version=1):
    head = repr({"descr": descr, "fortran_order": fortran, "shape": tuple(shape)}).encode("latin1")
    length_size = 2 if version == 1 else 4
    head += b" " * (-(8 + length_size + len(head) + 1) % 64) + b"\n"
    return b"\x93NUMPY" + bytes((version, 0)) + len(head).to_bytes(length_size, "little") + head + data


def unicode_npy(text):
    return npy("<U" + str(len(text)), (), text.encode("utf-32-le"))


def archive_bytes(members):
    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, value in members.items():
            z.writestr(name + ".npy", value)
    return out.getvalue()


def invented_snapshot(*, complex_values=False, zero=False):
    raw = old.fixture(complex_values=complex_values, zero=zero)
    descs = raw["arrays"]

    def member(desc):
        descr = ("<" if desc["endian"] == "little" else ">") + ("f8" if desc["dtype"] == "float64" else "c16")
        return npy(descr, [int(x) for x in desc["shape"]], bytes.fromhex(desc["data_hex"]))

    # Artificial writer only: promote these literal blocks to one complex array
    # by appending exact +0 imaginary words. Production adapter never casts.
    parts = []
    for item in descs["g_blocks"]:
        desc = item["array"]
        b = bytes.fromhex(desc["data_hex"])
        parts.append(b if desc["dtype"] == "complex128" else b"".join(b[k:k+8] + bytes(8) for k in range(0, len(b), 8)))
    members = {"schema_version": unicode_npy(adapter.SCHEMA), "hamiltonian_hash": unicode_npy("f" * 64),
               "constant": member(descs["constant"]), "one_body": member(descs["one_body"]),
               "lambdas": member(descs["lambdas"]),
               "g_matrices": npy("<c16", (4, 2, 2), b"".join(parts)),
               "metadata_json": unicode_npy(json.dumps({"fixture": "INVENTED_NOT_MOLECULAR", "df_truncation_value": 9999}))}
    plan = {"n_modes": "2", "mode_order": raw["mode_order"], "block_inventory": raw["block_inventory"],
            "sector_N": "1", "mapping": "STORED_AXES_AS_DECLARED_NO_PERMUTATION",
            "representation": "DFHAMILTONIAN_FULL", "constant_role": "FULL_CONSTANT_ONCE",
            "one_body_role": "FULL_ONE_BODY_ONCE", "df_truncation_value_role": "METADATA_ONLY_NOT_SCALAR",
            "full_block_inventory": True}
    return raw, members, plan


def task_policy():
    return {"d": rat(F(1, 2)), "alpha_spec": rat(F(1, 10)), "root_width": rat(F(1, 2 ** 12)), "max_bisections": "128"}


def test_documents(members=None, plan=None):
    _, default_members, default_plan = invented_snapshot()
    members, plan = members or default_members, plan or default_plan
    blob = archive_bytes(members)
    manifest = adapter.inspect_snapshot(blob)
    record = {"schema": contract.CONTRACT_SCHEMA, "adopted": True, "contract_grants_execution": False,
              "scope": contract.TASK, "source": {"commit": "1" * 40, "identity": source_identity()},
              "input": {"commit": "2" * 40, "repository_path": "artifacts/INVENTED.npz", "manifest": manifest,
                        "mapping": plan, "expected_model_id": "DERIVE_EXACT_FROM_PINNED_RAW_AND_MAPPING",
                        "provenance_basis": "ARTIFICIAL_MEMORY_TEST_NOT_REAL_INPUT"},
              "task": {"occupied_modes": ["0"], "fragment_order": ["one_body"] + ["df:" + s for s in plan["block_inventory"]],
                       "sqrt_bits": "16", "comparison_policy": task_policy(), "cap_policy": "ENTRYWISE_RE_IM_ABS_SUM_V1",
                       "guarantees": ["RF0", "RF1", "EW2", "CENTERED_CBAR2_WITH_GROUPED_STRUCTURE"],
                       "information_policy": "SAME_ALL_VERIFIED_CERTIFICATES_NO_FREE_ACQUISITION"},
              "limits": {"calls": dict(contract.CALLS), "output_bytes": "2097152", "log_bytes": "4096",
                         "cpu_cap": None, "wall_cap": None, "memory_cap": None, "worker_policy": "AVAILABLE_RESOURCES_NO_SHARED_CHANGES"},
              "failure_policy": {"one_shot": True, "retry": "NO_AUTOMATIC_RETRY", "stop": "MANDATORY_STOP", "claim_before_input_IO": True, "preserve_partial": True},
              "state": {"A2": "NOT_ADOPTED", "A1": "CLOSED", "K1_F": "UNAPPROVED"},
              "excluded": ["SPEC08", "QPE", "NATIVE_COMPILE", "STATE_PREPARATION", "ORIGINAL_MOLECULAR_CERTIFICATION", "SCIENTIFIC_GO_NO_GO"],
              "execution_store": {"canonical_repository": "/ARTIFICIAL_MEMORY_ONLY", "ledger_relative": "artifacts/qc_a2_pilot_execution"}}
    return blob, record


def pack(record, *, run_id="ARTIFICIAL_VECTOR"):
    cb = canonical_bytes(record)
    approval = {"schema": contract.TEST_AUTH_SCHEMA, "execution_authorized": True,
                "user_statement": "ARTIFICIAL TEST VECTOR ONLY; NO SCIENTIFIC EXECUTION AUTHORIZATION",
                "contract_sha256": sha256(cb).hexdigest(), "source_commit": record["source"]["commit"],
                "input_manifest_id": content_id("a2_input_manifest_v1", record["input"]["manifest"]),
                "run_id": run_id, "one_shot": True, "retry": "NO_AUTOMATIC_RETRY"}
    approval["user_statement_sha256"] = sha256(approval["user_statement"].encode()).hexdigest()
    ab = canonical_bytes(approval)
    pins = {k: approval[k] for k in ("contract_sha256", "source_commit", "input_manifest_id")}
    pins["authorization_sha256"] = sha256(ab).hexdigest()
    return cb, ab, pins


class MemoryStore:
    def __init__(self, fail_on=None):
        self.records, self.consumed, self.written, self.fail_on = {}, False, 0, fail_on

    def claim(self, record):
        if self.consumed:
            raise FileExistsError("already consumed artificial memory claim")
        self.consumed = True
        self.write("claim.json", canonical_bytes(record))

    def write(self, name, data):
        if name == self.fail_on:
            raise OSError("artificial persistence failure")
        if name in self.records:
            raise FileExistsError(name)
        self.records[name] = data
        self.written += len(data)


def memory_run(blob, record, *, store=None, load=None, source_check=None, vectors=None, artificial=True):
    store = MemoryStore() if store is None else store
    vectors = pack(record) if vectors is None else vectors
    result = runner._run(*vectors, load_input=(lambda c: blob) if load is None else load,
                         source_check=(lambda c: None) if source_check is None else source_check,
                         store_factory=lambda key, limits: store, artificial_test=artificial)
    return result, store


class AdapterTests(unittest.TestCase):
    def test_identity_only_inspection_does_not_decode_coefficients(self):
        _, members, _ = invented_snapshot()
        with patch("trottercert.a2_pilot.core01_generator.decode_binary64", side_effect=AssertionError("No decoding")):
            manifest = adapter.inspect_snapshot(archive_bytes(members))
        self.assertFalse(manifest["coefficient_values_decoded"])
        self.assertEqual(len(manifest["members"]), 7)

    def test_exact_bit_mapping_real_and_complex(self):
        for complex_values in (False, True):
            with self.subTest(complex=complex_values):
                original, members, plan = invented_snapshot(complex_values=complex_values)
                blob = archive_bytes(members)
                raw = adapter.adapt_snapshot(blob, adapter.inspect_snapshot(blob), plan)
                for key in ("constant", "one_body", "lambdas"):
                    self.assertEqual(raw["arrays"][key], original["arrays"][key])
                self.assertEqual(raw["provenance"]["input_class"], "FIXED_DF_PILOT")
                with self.assertRaises(ContractError): old_validate_raw(raw)
                base = generate_spec01_03(raw, particle_number="1")
                report = check_spec01_03(raw, base["evidence"], expected_particle_number="1")
                self.assertEqual(report["verdict"], "SPEC01_03_VERIFIED_EXACT_FIXED_DF")

    def test_changed_container_member_and_array_rejected(self):
        _, members, plan = invented_snapshot()
        blob = archive_bytes(members);pinned = adapter.inspect_snapshot(blob)
        members["constant"] = npy("<f8", (), struct.pack("<d", 9.))
        with self.assertRaises(ContractError): adapter.adapt_snapshot(archive_bytes(members), pinned, plan)
        for field in ("npy_sha256", "array_sha256", "npy_bytes", "array_bytes"):
            with self.subTest(field=field):
                bad = copy.deepcopy(pinned);bad["members"][0][field] = "0"
                with self.assertRaises(ContractError): adapter.adapt_snapshot(blob, bad, plan)

    def test_missing_extra_and_duplicate_members(self):
        _, members, _ = invented_snapshot()
        for change in ("missing", "extra", "unsafe"):
            with self.subTest(change=change):
                bad = dict(members)
                if change == "missing": del bad["lambdas"]
                else: bad["../unsafe" if change == "unsafe" else "extra"] = unicode_npy("x")
                with self.assertRaises(ContractError): adapter.inspect_snapshot(archive_bytes(bad))
        out = BytesIO()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(out, "w") as z:
                for k, value in members.items(): z.writestr(k + ".npy", value)
                z.writestr("constant.npy", members["constant"])
        with self.assertRaises(ContractError): adapter.inspect_snapshot(out.getvalue())

    def test_no_pickle_structured_native_or_fortran_cast(self):
        for dtype in ("|O", "<f4", "=f8", [("x", "<f8")]):
            with self.subTest(dtype=str(dtype)), self.assertRaises(ContractError): adapter._npy(npy(dtype, (), bytes(8)))
        with self.assertRaises(ContractError): adapter._npy(npy("<f8", (1,), bytes(8), fortran=True))

    def test_header_versions_and_truncation(self):
        for v in (1, 2):
            with self.subTest(v=v):
                header, payload = adapter._npy(npy(">f8", (1,), struct.pack(">d", .5), version=v))
                self.assertEqual(header["endian"], "big")
                self.assertEqual(payload, struct.pack(">d", .5))
        for bad in (b"", b"\x93NUMPY\x09\x00", npy("<f8", (), bytes(8))[:-1]):
            with self.subTest(bad=bad[:10]), self.assertRaises((ContractError, ValueError)): adapter._npy(bad)

    def test_no_missing_blocks_or_scalar_correction(self):
        _, members, plan = invented_snapshot();blob = archive_bytes(members);manifest = adapter.inspect_snapshot(blob)
        for key, val in [("constant_role", "CORRECTION_ONLY"), ("one_body_role", "CORRECTION_ONLY"),
                         ("df_truncation_value_role", "ADD_TO_CONSTANT"), ("full_block_inventory", False),
                         ("block_inventory", plan["block_inventory"][:-1]), ("n_modes", "3")]:
            with self.subTest(key=key):
                bad = copy.deepcopy(plan);bad[key] = val
                with self.assertRaises(ContractError): adapter.adapt_snapshot(blob, manifest, bad)

    def test_exact_Hermiticity_failure_is_not_repaired(self):
        _, members, plan = invented_snapshot()
        members["one_body"] = npy("<f8", (2, 2), struct.pack("<dddd", 1., .75, .5, -1.))
        blob = archive_bytes(members);raw = adapter.adapt_snapshot(blob, adapter.inspect_snapshot(blob), plan)
        result = generate_spec01_03(raw, particle_number="1")
        self.assertEqual(result["verdict"], "GENERATION_REJECTED")
        self.assertIn("NON_HERMITIAN", str(result))

    def test_model_identity_matches_frozen_exact_target(self):
        raw, members, plan = invented_snapshot()
        blob = archive_bytes(members);converted = adapter.adapt_snapshot(blob, adapter.inspect_snapshot(blob), plan)
        old_base, new_base = frozen_generate01(raw, particle_number="1"), generate_spec01_03(converted, particle_number="1")
        self.assertEqual(old_base["evidence"]["payload"]["context"]["model_id"], new_base["evidence"]["payload"]["context"]["model_id"])
        self.assertNotEqual(old_base["evidence"]["payload"]["intake"]["raw_input_id"], new_base["evidence"]["payload"]["intake"]["raw_input_id"])


class FastAlgebraTests(unittest.TestCase):
    def test_exact_trace_batch_against_frozen_literal_trace(self):
        rng = random.Random(1801)
        for n in range(5):
            dim = 2 ** n
            matrix = {(i, j): (F(rng.randrange(-2, 3), 2), F(rng.randrange(-2, 3), 4)) for i in range(dim) for j in range(dim)}
            with self.subTest(n=n): self.assertEqual(va.trace_pauli(matrix, n), old_va.trace_pauli(matrix, n))

    def test_target_size_eight_mode_exact_diagonal_and_flip(self):
        n, word = 8, "XYZIXYZI"
        # Independent test-only Pauli matrix action; no Hamiltonian or physical data.
        matrix = {}
        for ket in range(2 ** n):
            target, phase = ket, 0
            for k, c in enumerate(word):
                bit = (ket >> k) & 1
                if c in "XY": target ^= 1 << k
                if c == "Y": phase += 1 + 2 * bit
                if c == "Z": phase += 2 * bit
            matrix[target, ket] = ((F(1), F(0)), (F(0), F(1)), (F(-1), F(0)), (F(0), F(-1)))[phase % 4]
        self.assertEqual(va.trace_pauli(matrix, n), [{"pauli_mode0_left": word, "coefficient": {"re": rat(1), "im": rat(0)}}])

    def test_bit_commutator_all_three_mode_word_pairs(self):
        words = ["".join(p) for p in itertools.product("IXYZ", repeat=3)]
        for p in words:
            for q in words:
                with self.subTest(p=p, q=q):
                    left, right = {p: (F(1, 2), F(1, 4))}, {q: (F(-1, 3), F(2, 3))}
                    self.assertEqual(pa.commutator(left, right), old_pa.commutator(left, right))

    def test_grouped_cancellation_and_multiple_coefficients(self):
        left = {"XI": (F(1), F(0)), "IZ": (F(1, 2), F(0)), "YY": (F(-1), F(0))}
        right = {"ZI": (F(1), F(0)), "YZ": (F(1), F(0)), "IX": (F(2), F(0))}
        self.assertEqual(pa.commutator(left, right), old_pa.commutator(left, right))
        self.assertEqual(pa.commutator(left, left), {})


class RunnerTests(unittest.TestCase):
    def test_complete_artificial_memory_pipeline(self):
        blob, record = test_documents();result, store = memory_run(blob, record)
        self.assertEqual(result["status"], "COMPLETED_FIXED_DF_CERTIFICATE_COMPARISON", result)
        self.assertEqual(read_rat(result["U"]), F(13, 4))
        self.assertEqual(result["calls"], contract.CALLS)
        self.assertTrue(result["mandatory_STOP"] and store.consumed)
        self.assertEqual(json.loads(store.records["result.json"]), result)
        for stage in ("SPEC05_KSOS", "SPEC06_COMMUTATOR"):
            self.assertTrue(any(stage in row["stage"] for row in result["costs"]))
        self.assertEqual(result["state"]["A2"], "NOT_ADOPTED")

    def test_candidate_contract_is_closed_before_all_input_IO(self):
        blob, record = test_documents();record["adopted"] = False
        load = unittest.mock.Mock(side_effect=AssertionError("No scientific input IO"))
        result, store = memory_run(blob, record, load=load)
        self.assertEqual(result["status"], "PREEXECUTION_REJECTED")
        load.assert_not_called();self.assertFalse(store.consumed)

    def test_UNSET_conditions_and_invalid_tasks_closed(self):
        for path, value in [(('task','sqrt_bits'),'UNSET'), (('task','occupied_modes'),[]),
                            (('task','comparison_policy'),'UNSET'), (('input','expected_model_id'),'UNSET'),
                            (('limits','output_bytes'),'UNSET'), (('execution_store','canonical_repository'),'UNSET')]:
            with self.subTest(path=path):
                blob, record = test_documents();record[path[0]][path[1]] = value
                result, store = memory_run(blob, record, load=lambda c: self.fail("input opened"))
                self.assertEqual(result["status"], "PREEXECUTION_REJECTED");self.assertFalse(store.consumed)

    def test_artificial_authorization_not_accepted_by_production(self):
        blob, record = test_documents();result, store = memory_run(blob, record, artificial=False, load=lambda c: self.fail("input"))
        self.assertEqual(result["status"], "PREEXECUTION_REJECTED");self.assertFalse(store.consumed)

    def test_missing_explicit_execute_does_not_even_read_documents(self):
        with patch.object(Path, 'read_bytes', side_effect=AssertionError("No file IO")):
            result = runner.run_files('/MISSING', '/MISSING_C', '/MISSING_A', {}, execute=False)
        self.assertEqual(result["status"], "PREEXECUTION_REJECTED")

    def test_authorization_and_external_pins_tamper(self):
        blob, record = test_documents();cb, ab, pins = pack(record)
        for change in ("contract", "authorization", "source", "input", "statement", "runid"):
            with self.subTest(change=change):
                p = dict(pins);a = json.loads(ab)
                if change == "contract": p['contract_sha256'] = "0"*64
                elif change == "authorization": p['authorization_sha256'] = "0"*64
                elif change == "source": p['source_commit'] = "0"*40
                elif change == "input": p['input_manifest_id'] = "0"*64
                elif change == "statement": a['user_statement'] = 'changed'
                else: a['run_id'] = '../escape'
                new_ab = canonical_bytes(a)
                if change in ('statement','runid'): p['authorization_sha256'] = sha256(new_ab).hexdigest()
                result, store = memory_run(blob, record, vectors=(cb,new_ab,p), load=lambda c:self.fail('input'))
                self.assertEqual(result["status"], "PREEXECUTION_REJECTED");self.assertFalse(store.consumed)

    def test_source_identity_mismatch_before_claim(self):
        blob, record = test_documents();record['source']['identity']['version'] = 'forged'
        result, store = memory_run(blob, record, load=lambda c:self.fail('input'))
        self.assertEqual(result['status'],'PREEXECUTION_REJECTED');self.assertFalse(store.consumed)

    def test_source_commit_checker_failure_before_claim(self):
        blob, record = test_documents()
        result, store = memory_run(blob, record, source_check=lambda c: (_ for _ in ()).throw(ContractError('SOURCE','commit','missing')))
        self.assertEqual(result['status'],'PREEXECUTION_REJECTED');self.assertFalse(store.consumed)

    def test_double_execution_and_changed_runid_rejected(self):
        blob, record = test_documents();result, store = memory_run(blob, record)
        self.assertEqual(result['status'],'COMPLETED_FIXED_DF_CERTIFICATE_COMPARISON')
        before=copy.deepcopy(store.records)
        repeat,_=memory_run(blob,record,store=store,vectors=pack(record,run_id='DIFFERENT_ARTIFICIAL_ID'),load=lambda c:self.fail('input'))
        self.assertEqual(repeat['status'],'CLAIM_REJECTED_OR_INCOMPLETE');self.assertEqual(store.records,before)

    def test_failure_consumes_claim_preserves_partial_no_retry(self):
        _, members, plan = invented_snapshot()
        members['one_body']=npy('<f8',(2,2),struct.pack('<dddd',1.,.75,.5,-1.))
        blob,record=test_documents(members,plan);result,store=memory_run(blob,record)
        self.assertEqual(result['status'],'TECHNICAL_STOP');self.assertTrue(store.consumed)
        self.assertIn('SPEC01_03_proposal.json',store.records)
        self.assertIn('NON_HERMITIAN',store.records['SPEC01_03_proposal.json'].decode())
        self.assertNotIn('SPEC04_07_generator',result['calls'])
        self.assertEqual(memory_run(blob,record,store=store)[0]['status'],'CLAIM_REJECTED_OR_INCOMPLETE')

    def test_nonfinite_is_not_repaired(self):
        _,members,plan=invented_snapshot()
        members['constant']=npy('<f8',(),bytes.fromhex('000000000000f87f'))
        blob,record=test_documents(members,plan);result,store=memory_run(blob,record)
        self.assertEqual(result['status'],'TECHNICAL_STOP');self.assertIn('NONFINITE',store.records['SPEC01_03_proposal.json'].decode())

    def test_input_byte_mismatch_consumes_once_and_STOPS(self):
        blob,record=test_documents();result,store=memory_run(blob+b'extra',record)
        self.assertEqual(result['status'],'TECHNICAL_STOP');self.assertTrue(store.consumed)
        self.assertEqual(result['calls'],{'adapter':'1'})

    def test_keyboard_interrupt_preserves_consumed_attempt(self):
        blob,record=test_documents()
        with patch.object(runner,'generate_spec04_07',side_effect=KeyboardInterrupt('ARTIFICIAL_INTERRUPT')):
            result,store=memory_run(blob,record)
        self.assertEqual(result['status'],'TECHNICAL_STOP');self.assertTrue(store.consumed)
        self.assertEqual(result['failure']['type'],'KeyboardInterrupt')
        self.assertIn('SPEC01_03_checked.json',store.records);self.assertIn('result.json',store.records)

    def test_checker_rejection_saved_not_success(self):
        blob,record=test_documents()
        with patch.object(runner,'check_spec04_07',return_value={'verdict':'INVALID_CERTIFICATE','reason':'ARTIFICIAL_FAULT'}):
            result,store=memory_run(blob,record)
        self.assertEqual(result['status'],'TECHNICAL_STOP')
        self.assertIn('SPEC04_07_proposal.json',store.records)
        self.assertIn('SPEC04_07_checked.json',store.records)

    def test_generator_failure_partial_saved(self):
        blob,record=test_documents()
        with patch.object(runner,'generate_spec04_07',return_value={'verdict':'TECHNICAL_ERROR_OR_INTERRUPTED','partial':{'artificial':'preserved'}}):
            result,store=memory_run(blob,record)
        self.assertEqual(result['status'],'TECHNICAL_STOP')
        self.assertIn('partial',json.loads(store.records['SPEC04_07_proposal.json']))
        self.assertNotIn('SPEC04_07_checker',result['calls'])

    def test_persistence_failure_does_not_release_claim(self):
        blob,record=test_documents();store=MemoryStore(fail_on='result.json')
        result,_=memory_run(blob,record,store=store)
        self.assertEqual(result['status'],'PERSISTENCE_FAILURE_STOP');self.assertTrue(store.consumed)
        self.assertIn('SPEC04_07_checked.json',store.records)

    def test_early_write_failure_keeps_partial_and_STOP(self):
        blob,record=test_documents();store=MemoryStore(fail_on='SPEC04_07_proposal.json')
        result,_=memory_run(blob,record,store=store)
        self.assertEqual(result['status'],'TECHNICAL_STOP');self.assertTrue(store.consumed)
        self.assertIn('SPEC01_03_checked.json',store.records);self.assertNotIn('SPEC04_07_checker',result['calls'])

    def test_wrong_guarantee_call_scope_and_scientific_state(self):
        for key in ('guarantees','calls','state','retry'):
            with self.subTest(key=key):
                blob,record=test_documents()
                if key=='guarantees':record['task'][key]=['RF1']
                elif key=='calls':record['limits'][key]['SPEC04_07_generator']='2'
                elif key=='state':record[key]['A2']='ADOPTED'
                else:record['failure_policy'][key]='AUTOMATIC_RETRY'
                result,store=memory_run(blob,record,load=lambda c:self.fail('input'))
                self.assertEqual(result['status'],'PREEXECUTION_REJECTED');self.assertFalse(store.consumed)

    def test_bad_policy_and_path_guard(self):
        for target in ('d','alpha','path','trial'):
            with self.subTest(target=target):
                blob,record=test_documents()
                if target=='d':record['task']['comparison_policy']['d']=rat(0)
                elif target=='alpha':record['task']['comparison_policy']['alpha_spec']=rat(1)
                elif target=='trial':record['task']['occupied_modes']=['9']
                else:record['input']['repository_path']='../private'
                result,store=memory_run(blob,record,load=lambda c:self.fail('input'))
                self.assertEqual(result['status'],'PREEXECUTION_REJECTED');self.assertFalse(store.consumed)

    def test_bounded_logs_and_no_silent_truncation(self):
        sink=BoundedLog(4);sink.write('abcdefgh');self.assertEqual(bytes(sink.data),b'abcd');self.assertEqual(sink.dropped_bytes,4)

    def test_local_store_exclusive_claim_and_append_only_flags(self):
        # Test production filesystem calls through mocks; create no actual marker.
        store=LocalStore('/ARTIFICIAL_NO_DISK','a'*64,output_bytes=1024,log_bytes=64)
        with patch.object(Path,'mkdir',side_effect=FileExistsError('old claim')) as mkdir, patch('os.open') as opening:
            with self.assertRaises(FileExistsError):store.claim({'artificial':True})
        mkdir.assert_called_once_with(mode=0o700,parents=False,exist_ok=False);opening.assert_not_called()
        with self.assertRaises(ContractError):store.write('../escape',b'x')

    def test_local_store_output_and_log_limits_precede_open(self):
        store=LocalStore('/ARTIFICIAL_NO_DISK','a'*64,output_bytes=1024,log_bytes=8)
        with patch('os.open') as opening:
            for name,data in [('certificate.json',bytes(1025)),('stdout.txt',bytes(9)),('stderr.txt',bytes(9))]:
                with self.subTest(name=name),self.assertRaises(ContractError):store.write(name,data)
            opening.assert_not_called()

    def test_local_record_uses_exclusive_create_and_fsync(self):
        import os
        handle=unittest.mock.MagicMock();handle.fileno.return_value=123
        context=unittest.mock.MagicMock();context.__enter__.return_value=handle
        store=LocalStore('/ARTIFICIAL_NO_DISK','a'*64,output_bytes=1024,log_bytes=8)
        with patch('os.open',return_value=123) as opening, patch('os.fdopen',return_value=context), patch('os.fsync') as syncing:
            store.write('record.json',b'{}')
        self.assertEqual(opening.call_args.args[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL)
        self.assertEqual(opening.call_args.args[2],0o600)
        handle.write.assert_called_once_with(b'{}');handle.flush.assert_called_once();syncing.assert_called_once_with(123)


if __name__ == '__main__':
    unittest.main()
