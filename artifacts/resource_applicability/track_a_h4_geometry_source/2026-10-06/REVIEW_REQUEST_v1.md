# H4 geometry server-native source固定後レビュー依頼

状態は `H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`。科学未実行、公開後STOP。
今回の判断対象は、**入力生成専用authorizationの作成へ進めるか**である。
この資料はexecution plan・authorizationを発行せず、入力生成・signal/compileをlaunchしない。

## 固定対象と資料入口

- Repository: `HIROMU1015/Partially-Randomized-Trotter`
- Branch: `track-a-h4-geometry-source-20261006`
- BASE: `b662dbd72e49fa713a25c716f323843e547e973b`
- SOURCE_COMMIT: `d6c7afd02dc0603216982a3cd4ea71b3d047dd8d`
- REVIEW_BUNDLE_COMMIT: この資料と監査4ファイルだけを追加する次commit。実SHAは公開時の報告およびGit履歴から確認する。
- Actual source checkout: `/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-source-20261006`
- Contract artifact anchor: `/home/AbeHiromu/projects/partially-randomized-trotter`
- Future fixed run: `track-a-h4-geometry-v2-20261006-run01`。本番output/registryはアクセス・作成していない。

[実装資料](../../../../docs/research/track_a_h4_geometry_server_native_source_implementation.md)と
[D1〜D4採用記録](source_adoption_v1.json)を先に参照する。
過去v2 artifactの「当時レビュー未承認」という履歴は変更していない。

[source_freeze_v1.json](source_freeze_v1.json)は、actual SOURCE_COMMITに対する16件のsource/runner/testの
blob SHA-256、既存namespace parent2件、AST import一覧、依存45件のversion/installed RECORD hash、
installed critical source11件、compiler options/defaults/plugins、および旧source・保存JSON・bundleの不変性を記録する。
移植元path・BASE commit・hash・意味論差分は同JSONの `migrations` を参照する。
このclosureは自作Pythonと監査対象installed source/metadataの範囲であり、全wheelのbinary内容や独立環境での再現性まで検証したものではない。

[source_stage_manifest_v1.json](source_stage_manifest_v1.json)の49 entryとmanifest自身の計50ファイルをSOURCE_COMMITへ収録した。
[publication_scope_audit_v1.json](publication_scope_audit_v1.json)で全50 blobを照合する。
[review_bundle_manifest_v1.json](review_bundle_manifest_v1.json)はreview資料3 entryと自身の計4ファイルを対象とする。
review commitではSOURCE_COMMIT収録のsource・tests・文書・ログを一切変更しない。
この資料作成時点のremote publicationは未試行であり、push成功の証拠にはしない。

## 科学scopeとstage境界

将来scopeはH4 linear/STO-3G、DF requested/returned fragments12、8 system＋ancilla index8の9 qubits、
T=0.8、二次DF-prefix PF/canonical finite-RTE、delta=T/q。
距離は0.70/0.80/0.90/1.10/1.40/1.60 Å。固定218 templatesはB0=20/B1=4/B2=145/B3=49。
random194 cells各32 paired trajectoriesとbaseline24、1点12,464/6点74,784 logical wrappers、signal1,308、
actual science transpile invocation cap74,784を維持する。302 precision pointsは保存値の表示のみ。
候補・距離・trajectoryを追加せず、実geometryのbias/cost/identityを旧geometryからコピーしない。

順序は source固定 → 入力生成専用source-bound plan・別authorization・review・利用者の明示launch
→ 6入力生成/freeze/STOP → input-bound signal/compile plan・別authorization・最終review・明示launch
→ map/STOP。入力生成認可の流用、input hash placeholder、暗黙retry/resume、winner・次認可の自動出力を拒否する。
今回は二種類のactual authorizationも、execution planも発行していない。

## 実行済みsynthetic検査

最終結果は[attempt09 JSON](test-attempt-09.json)と[ログ](test-attempt-09.log)：
**94 tests PASS、fail0/error0/skip0**。operator checks9、最終attemptのsynthetic transpile3。
[全attempt予約監査](synthetic_transpile_reservations.jsonl)の累積は**25/64**、初回1＋後続8回各3。
失敗attempt01/02/04と修正後の記録を保持した。追加のtranspile、全repository tests、性能benchmarkは実行しない。
旧benchmark128の内訳は比較120（30 tasks×workers1/6/12/16）＋axis/phase4＋full-operator4。今回再実行0。

以下は専用worktreeで既に実行した最終commandの記録である。同じ監査先への追加実行は今回の依頼に含めない。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  scripts/resource_applicability/run_h4_geometry_source_tests.py \
  --audit-dir artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06 \
  > artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/test-attempt-09.log 2>&1
```

source固定後のread-only監査は、同じprocess環境で次を実行した。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  scripts/resource_applicability/audit_h4_geometry_source.py \
  --source-commit d6c7afd02dc0603216982a3cd4ea71b3d047dd8d \
  --output artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/source_freeze_v1.json
```

| 確認範囲 | synthetic/mock検査の内容 |
|---|---|
| serializer/operator | ordered bits/registers、exact parameter/global phase、condition/control state、custom definition閉包、open control、axis/measurement、非有限/symbolic/未対応拒否。round-tripは全matrix一致。transpileもcontrol込み全wrapper operatorでrelative branch phaseを確認し、全wrapper共通overall phaseだけ許容する。数値fingerprintではそのphaseも保持する。 |
| 科学入力境界 | NoSaveRHF constructor/controls、gradient・不収束拒否、raw DF order/permutation/sign/null mode、縮退/tie STOP、sector/bit/phase/numerical gateはfake objectと架空matrixのみ。 |
| seed/signal | 独立D3 seed参照、source/input固定前拒否、step/occurrence独立stream、cosine/sine共有evolution、finite-RTE期待値・PF積順序・normalization・shot、candidate/input共通identity、32論理weight、covariance/2SE、null/exact ties、302保存値表示、共通P affine感度。 |
| stage gate | structureとsemanticsを分離。架空plan/authorization/reviewでstage流用・placeholder・距離置換を拒否。private input/output boundaryをmock禁止し、production runnerを試しlaunchしない。 |
| atomic/cache | private temporary ledgerの予約→COMPLETE→owner再利用、独立registry/外部digest、改変・欠測・cross scope・cache chain拒否、orphan/compile例外STOP、cap前予約、fsync/exclusive/symlink/上書き拒否。 |
| resource/process | fresh memory/CPU許可、64GiBでw<=5・120GiBでw<=12、上位cgroup/hidden ancestor、AS/RSS区別、pressure/OOM/72h、10GiBのtemp/log込み事前課金とstage引継ぎ、foreign PID拒否、owned pipe framing/stage/log制限をmock検査。実worker subprocessは起動しない。 |

一時matrix/circuitはメモリ内、ledgerはprivate `/tmp/h4-synthetic-*`のみ。
最終test guardではmolecular import/access、actual science transpile、protected attemptsは0。
依存metadata・installed Python sourceの静的読み取りは分子入力アクセスではない。

## 残課題と次レビューの判断点

1. **実科学入力は未検証。** NoSaveRHFの実minao/RHF収束・厳密gradient、実DF returned-rank/ties/縮退、
   integral/MO/元H/DF/stateの物理的一致、承認6距離での数値gate成立は、別認可後の入力生成段で初めて確認する。
   今回のsynthetic PASSから科学成立や完成したtotal-cost結果を主張しない。
2. **live production条件は未検証。** 専用匿名pipe workersの実spawn/handshake、own process tree停止、
   8GiB AS/RSSと実cgroup hierarchy/pressure/OOM、production filesystemのpower loss/fsync耐性、
   全74,784 wrapperの72h/10GiB内完了は未検証。性能benchmarkも追加しない。
3. **source/environment/root bindingをレビューする。** 実行checkoutとcontract artifact/output anchorを区別したgate、
   SOURCE_COMMIT/audit hashを結合する新production wire、依存45件とinstalled critical source11件の監査範囲を確認する。
   許可CPUがprocess affinityより狭い場合は設定を変更せずSTOPするため、将来launch contextも別途レビューが必要。
4. **入力生成専用plan/authの設計をレビューする。** actual SOURCE_COMMITと外部source auditを結合し、
   input hashが存在しないsource-bound段でplaceholderを用いないことを確認する。
   このbundleの採用が入力生成の認可・明示launchを兼ねることはない。mapはさらに別のinput-bound認可が必要。

## 旧証拠・安全な停止

旧247 Python source、保存6 JSON、公開準備25・v1 26・v2 30 bundleはbyte/hash不変。
BASEの契約manifest37 entryを照合した。旧結果/status/validation manifest・原稿・図・Track BはGit変更対象外。
既存8件のindex/研究概要/当日noteへの今回追加だけを既存path変更として許可し、旧記述を上書きしない。
分子NPZ・snapshot・runtime/checkpoint/cache/registryのresolve/stat/hash/load/materializeは行っていない。
mainの未commit変更を保持し、reset/clean/stash/main merge/force-pushは行わない。

分子アクセス/生成、実SCF/DF/state、実signal/sampling/build/compile、GPU操作、
環境・他job変更、actual authorization発行、production runner launchはすべて0。
今回はlocal synthetic evidenceであり、immutable CI・独立外部再現ではない。
公開後STOPし、入力生成、本計算、追加距離/trajectory、strong synthesis、高次PF、energy/RPE、Track Bへ進まない。
