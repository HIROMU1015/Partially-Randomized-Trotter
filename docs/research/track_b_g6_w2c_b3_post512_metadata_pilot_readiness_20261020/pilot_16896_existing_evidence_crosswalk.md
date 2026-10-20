# 既存16,896件Pilotと実科学512件SOURCEの照合

旧固定版は公開HEAD22ef92cef31f98e3e31f27a209405f81f5c4a0e4内の限定Pilot packet。旧測定SOURCE b9cabcfc6cf304ab65fc8a540c33ce2173f6cbaa、producer基点7ae4bd0b951dae05380538cd5c22a3ead5328ab2、consumer原本bf601f8a9723a61b7a05b534019f8e036a362735。inventory対象files/protectedの完全SHAを照合し、旧科学Core・Adapter・Global Bundle・F01・Storage V2を変更していない。

旧fake v3 projection768件と、後続closed-port人工16896件、実科学512件は別の来歴。実科学512件でresolver、factory、Evidence、S/P1一束、簡易one-shot、保存入口が観測済みとなり、「実科学bridgeが一切ない」は現在の説明には使わない。profile2passのproduction releaseは未実装のままで、512をnだけ変えて使えない。

| 対象 | 分類 | 根拠 | 本当に残るもの |
|---|---|---|---|
| fixed_science_input_precision | ALREADY_VERIFIED | docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020/source_inventory.json | same fixed P5/m5/precision/A6; pilot adoption still separate |
| resolver_real_core_signature | ALREADY_VERIFIED | docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020/result_summary.json | finite adopted512 observed; untested inputs stay unverified |
| Canonical_Full_factories | REUSABLE | src/trottertracks/algorithm_codesign/g6_w2c_b3_final_live_v1/wrapper.py | reuse science constructors; version only trial/phase/task/boundary limits after design adoption |
| ScientificBits_counter | REUSABLE | src/trottertracks/algorithm_codesign/g6_w2c_b3_final_live_v1/bits.py | reuse recorder and SHA256 algorithm; proposed pilot namespace currently rejected by fixed512 guard |
| Exact_Codec_Result | REUSABLE | src/trottertracks/algorithm_codesign/g6_w2c_b3_real_adapter_v1/codec.py | no redevelopment; phase-aware caller/metadata extension only |
| AB_SP1_prefix | ALREADY_VERIFIED | docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020/saved_evidence_check.json | fixed512 prefix confirmed; profile pass/correctness prefix joins need new adapter |
| Full_ZERO_coin_accepted | ALREADY_VERIFIED | docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020/result_summary.json | observed fixed paths; no sampling to manufacture further coverage |
| F01_source_protocol | REUSABLE | src/trottertracks/algorithm_codesign/g6_w2c_b3_final_live_v1/execution.py:run_pair; unchanged AST | reuse with artificial failure tests; do not redevelop |
| real_F01_failure | UNKNOWN | docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020/result_summary.json:F01_real_failure_triggered=False | not observed; does not by itself require a new scientific failure run |
| one_scope_SP1_Global_Bundle | REUSABLE | docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020/publication_receipt.json | single claim/call/byte/STOP was real512 verified; bind extended phases after adoption |
| Storage_V2 | REUSABLE | src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_storage_v2.py | reuse reservation/trace/no refund/drain; artificial16896 evidence already exists |
| SimpleStore | REUSABLE | src/trottertracks/algorithm_codesign/g6_w2c_b3_final_live_v1/run_control.py | reuse exclusive intents/SQLite UNIQUE/spent; parameterize mark_completed planned512 checks for new release |
| metadata_source_fix | ALREADY_VERIFIED | artificial/results.json | 80 tests +4 integration+20mutation; pure saved-USER metadata fixtures issue no permits |
| old_private_47_history | NOT_REQUIRED | docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/pending_evidence_resolution.md | keep historical missing evidence missing; not a permanent block for new self-contained run |
| external_authority_service | NOT_REQUIRED | src/trottertracks/algorithm_codesign/g6_w2c_b3_final_live_v1/run_control.py | owner simple one-shot suffices if adopted; no keys/server redesign proposed |
| pilot_n256_16896_design | CONTRACT_ADOPTION_REQUIRED | pilot_execution_contract_candidate.json | n256, S/P1, 1 repetition and phase design remain unadopted |
| seed_and_scope | CONTRACT_ADOPTION_REQUIRED | docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/pilot_scientific_scope_candidate.md | new namespace and fresh independent scope; no reuse/deduction of spent512 |
| coverage_transition | CONTRACT_ADOPTION_REQUIRED | pilot_contract_and_source_gaps.md | old pilot STOP vs current512 INCONCLUSIVE coverage semantics; explicitly choose before release |
| retention_and_bytes | CONTRACT_ADOPTION_REQUIRED | docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/pilot_measurement_design.md | 64GiB/256MiB old proposal, full vs first16+hash profile retention; no automatic adoption or sufficiency claim |
| measurement_estimand | CONTRACT_ADOPTION_REQUIRED | pilot_execution_contract_candidate.md | pre-admission outer window, baseline observer-off/instrumented-on, block interpretation, repetitions/order must be adopted |
| phase_wrapper_seed_budget_adapter | TECHNICAL_FIX_REQUIRED | src/trottertracks/algorithm_codesign/g6_w2c_b3_final_live_v1/wrapper.py;bits.py;contract.py;offline.py;integration.py | extend bounded512 assumptions in new pilot version; exact worklist below; not authorized as formal executable release now |
| baseline_instrumented_observer | TECHNICAL_FIX_REQUIRED | src/trottertracks/algorithm_codesign/g6_w2c_b3_final_live_v1/execution.py:worker | current worker always CachedObserver for correctness diagnostics; reuse old scheduler conditional only in new adopted pilot worker |
| CPU_window_worker_helper_split | TECHNICAL_FIX_REQUIRED | docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020/resource_summary.json:CPU_accounting_definition | new wrapper records before-admission to full-finalization window; label child hierarchy and worker END observations separately |
| new_fixed_release_and_binding | CONTRACT_ADOPTION_REQUIRED | pilot_preflight_readiness.md | actual new fullSHA/source pins, executable/schema identity and exact global manifest after design adoption |
| science_execution_approval | SCIENCE_EXECUTION_APPROVAL_REQUIRED | pilot_execution_contract_candidate.json | separate explicit one-shot16896 approval after fixed source/contract; no grant/claim now |
| Full_N_native_new_theory_review | NOT_REQUIRED | docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/pilot_vs_full_N_decision_matrix.md | no Full-N/native/scientific review prerequisite inferred; none started |
