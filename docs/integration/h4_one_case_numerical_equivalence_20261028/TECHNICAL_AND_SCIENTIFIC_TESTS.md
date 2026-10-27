# Technical and scientific tests

2026-10-28 JST。元SOURCE・kernelを変更しない独立harness。固定contractを実行前に保存し、実行記録にそのSHAを結合した。新grant/receiptや旧実験の再開機構を使わない。

## Actual executions

- New artificial harness regression: **11 PASS / 0 FAIL / 0 SKIP**。既存の105/147技術PASSは再利用し、今回再実行・加算しない。
- Real saved one-case numeric execution: **one process / one attempt**, four MATCH_INDEPENDENT, two MATCH_METADATA_ONLY, one MATCH_WIRING_ONLY, three INCONCLUSIVE, zero MISMATCH/TECHNICAL_FAILURE。
- Wall 14.415s, peak RSS 591528KiB。NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Python3.12.3。読み取り専用の本人所有dependency環境を使用し、固定P1 source snapshot・全cache/outputをtask専用に分離した。LOCALのPython3.11.1環境とbitwise再現を主張しない。

| Actual API / batch | Calls |
| --- | ---: |
| H4GeometryAdapter.bind_core | 1 |
| DFFragmentPartition_record | 1 |
| H4GeometryAdapter.input | 1 |
| DFHamiltonian | 1 |
| df_linear_operator_python | 1 |
| LinearOperator.matmat_identity256 | 1 |
| internal_python_matvec | 256 |
| state_scalar_checks_batch | 1 |
| uncontrolled_identity_evolution_operator | 1 |
| controlled_identity_evolution_operator | 1 |

internal matvec256回は1回のidentity256 matmat内部で、256候補やsamplingではない。元生成freezeの独立保存full Hに対し、既存Python fermionic-action coreを比較した。新Hamiltonian kernelを実装していない。state checksは同じ再構成Hを再利用する。核scalarphaseは数学的cos/sinとdiag(I,U)参照に限定する。

## Guards and test purpose

Public P1 scientific/real/resume拒否を3modeで確認。frozen_scientificをsynthetic composerへ通さず、cohort/input binding不一致はconstructor前に拒否。新規出力directoryを必須にし、既存出力に対する再実行を拒否。shape/nonfinite/許容差超過を拒否し、wiring/metadataをindependentへ昇格させない。科学NPZを回帰testで再読込・再評価しない。

数値processではQiskit circuit/Bogoliubov生成、transpile、sampling/RNG、ground-state/SCF/eigensolver entry pointをtripwireで拒否し、NUMBA_DISABLE_JIT=1とbackend=pythonを明示。既存kernelの計算結果をmockに差し替えず、禁止呼出だけ拒否した。禁止entry point実行0。旧initializer/CLI/source fileは不変。

既存moduleのinvalid escape SyntaxWarningが5件あった。これは旧docstringのwarningで、test failureや科学kernel修正の理由にはしていない。

## Reproduction boundary

固定P1 src snapshotをPYTHONPATHへ明示し、公開harnessを別の独立作業領域から実行する。本人が許可したprivate locations/qualificationと、実行前fixed contract/CONTRACT_SHA256を用意する。これらの正確なpath・全array hash・stateは公開しない。公開CLIをscience modeへ変更する手順ではない。

```bash
python -B scripts/integration/validate_h4_one_case_equivalence.py \
  --locations PRIVATE_LOCATIONS.json \
  --contract FIXED_COMPARISON_CONTRACT_PRIVATE.json \
  --qualification FINAL_QUALIFICATION_PRIVATE.json \
  --output NEW_PRIVATE_NUMERIC_DIRECTORY
python -B -m pytest -q tests/test_h4_one_case_equivalence_harness.py
```

この文書は新しい再実行許可ではない。再現する場合も対象・入力・固定source・resource/STOPの範囲を維持する。old runner/worker/controller、SCF/state再生成、sampling、回路build/transpile/compile、RZ/N/G/SE、full RPE/QPE・新候補・別分子の実行は含めない。
