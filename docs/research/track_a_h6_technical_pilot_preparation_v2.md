# Track A H6技術pilot v2：coverage接続修正・実行前固定

2026-10-10 JST。今回のユーザー「次の作業に進んで」を、直前に説明した**v2接続修正・synthetic検証・source/実行条件の固定と公開**へ結合する。
実H6の再実行は含めない。準備・seal・local synthetic PASSは科学実行の認可ではない。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP、next_stage_authorized=false。

## 旧STOPと修正の範囲

[原v1 STOP](track_a_h6_technical_pilot_stop_v1.md)：source `a643220cde1e24b7e3d637f4bc5d1b0342ce2e86`、
原結果commit `01ecd3db299d93301d64e351c39e9587f16f9f09`、公開確認までのcommit `19fa5f6755941f57f9cc9c9f0cf646a908e281c1`。
worker一回、ACTUAL_PRIMITIVE_COVERAGE、correctness0/7・wrapper0/36、欠測59件。旧source/manifest/一回grant/STOP/rawを編集しない。

旧runtime `actual_bounds`はscheduleへ`registered_validation_times_v2`を追加するが、旧固定coverageはそのfieldを含まず、dict全体のdigestが一致しなかった。
原runのactual boundsは未保存であり、今回のtoy結果を過去の実測値として代用しない。他のactual差異がないこともまだ未確認。

v2ではH6各cellの全`unique_primitive_times`を登録validation timesとして固定coverageにも明記する。
凍結runtimeの`validation_times`にはH4-E専用の追加時間があるが、H6の7 IDには適用されない。
245 physical keys・3 probes・735 actions、数値schedule/負時間/undo signs/順序/seedを維持する。
fieldを除去せず、両側のscheduleの全キー・JSON scalar型・binary64値・順序・重複数・cell/probe件数を比較する。
許容する表記差はtuple/list serializationだけ。unknown fieldや登録時間の変更も拒否する。

実行時の保存順は、native準備 → prepared representation全件 → actual bounds → actual coverage/差分 → coverage判定 → 既存instruction cap → 既存order/cutoff判定 → reference。
準備receiptは`gates_pending=true`で保存し、保存されたことだけで受理や費用PASSを主張しない。
coverage以外のinstruction bound値は動的な構造上限であり、schema/非負整数型を検査し、凍結capで判定する。compiled costとは区別する。
準備中に軌道basis変換回路が作られる可能性を明記し、receiptはreference/probe/sampling/full evolution・Hadamard wrapper build前のものとする。
serializableなboundsの比較/上限拒否では全boundsと差分が残る。bounds計算自体の例外やoutput cap等で未保存となる可能性は残り、欠測として扱う。

[科学条件の不変監査](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/scientific_scope_preservation_v2.json)：plan変更はversionと登録時間fieldの追加のみ。
信号/独立oracle/primitive/control/sampling/費用の各methodはv1と同じfunction objectを継承する。
研究意味論・primary target・coefficient policy・比較の独立性を変更しない。[独立レビュー](track_a_h6_df_hermitization_independent_review_2026-10-10.md) §15に従う通常実装修正である。

## Source・入力・保存schema

- 新source commit `0b04886869efb9d08b07d6517300da2bc0123f4a`、[freeze](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/source_freeze_v2.json)：science206 / validation3。
- [contract](../../src/trottertracks/resource_applicability/ax2b_h6_pilot_contract_v2.py)、[setup port](../../src/trottertracks/resource_applicability/ax2b_h6_pilot_port_v2.py)、[coverage/boundsと保存監査](../../src/trottertracks/resource_applicability/ax2b_h6_pilot_coverage_v2.py)、[saved auditor](../../src/trottertracks/resource_applicability/ax2b_h6_pilot_audit_v2.py)。
- [science runner](../../scripts/resource_applicability/run_track_a_h6_pilot_v2.py)、[byte auditor入口](../../scripts/resource_applicability/audit_track_a_h6_pilot_v2.py)、[tests](../../tests/tracks/resource_applicability/test_ax2b_h6_pilot_v2.py)。旧v1 watchdog/strict coverage comparatorを再利用する。
- [入力identity](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/input_identity_v2.json)：親source `f37005f01b2be38c5993d6e82df91abe9c643d21`、raw `554fc52add39c2c1b45b765a3135df76fda6f15a`、evidence `f52787a22542b31bd39fd004a8d3d71325bc56b0`の33ファイルをlocal/Git bytesで照合。
- [保存snapshot](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/h6_input_snapshot.npz) SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、Hamiltonian hash `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、saved state hash `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3`。
- [政策exact copy](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/fixed_policy_v1_unchanged.json)：weighted projection追加予算1e-10 Ha / decision9.9e-11 Haを維持。

新raw schemaは旧required recordsに`actual_bounds_v2.json`、`actual_coverage.json`、`coverage_comparison_v3.json`を追加する。
v2 saved auditorは成功時に、manifest/actual bounds/prepared bounds/coverage比較receiptのbytesと一致を照合する。
STOP時は原STOPと欠測を保存し、欠測を成功扱いしない。旧結果への適用・再fit・再計算は行わない。

## 固定科学条件と計算上限

linear H6 / 1.00 Å / STO-3G、12 modes、α3/β3、sector400、actual rank19。
全19 signed fragmentsと生成順、tol-only1e-8/cutoff0、final_rankなし・fallbackなし・削除なし。
T0.8、q1/q2（outer δ0.8/0.4）；epsilon_signal0.001は診断ラベルで、同精度達成や化学的energy精度の認定ではない。

| cell | formula | prefix | q | R/r/K | replica |
|---|---|---:|---:|---|---:|
| B0 | S2 | 10 | 2 | — | 1 |
| B1 | S2 | 19 | 1 | — | 1 |
| B1 | S2 | 19 | 2 | — | 1 |
| B1 | S4 | 19 | 1 | — | 1 |
| B1 | S4 | 19 | 2 | — | 1 |
| B2 | S2 | 10 | 2 | 4/2/2 | 2 |
| B3 | S2 | 0 | 2 | 4/2/6 | 2 |

prefix0でもone-bodyは決定論的。7 cell・36 measured wrappers、ordinary/symmetric_directional×cosine/sine、primary symmetric_directional。
trajectory4/occurrence8はcost用で測定shotではない。同じeventsをgroup内4 wrapperで共有。state preparation費用は含まない。
random n=2は精密な期待cost・勝者選択の根拠にしない。u/ground-state未認定、N/Gnull、UNDETERMINEDを維持する。

[環境・CPU観測](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/environment_resource_observation_v2.json)：旧CPU IDs [0,2,5,6]が利用可能であることを確認し、同じ割当を固定。
worker1・Numba4/OMP4/BLAS1・chunk1/GPUなし。CPU topology/並列速度・ホスト専有は今回再測定していない。
phase input-reference1800 / correctness1800 / wrapper-cost3600秒、total7200秒、AS8GiB/output512MiB/log64KiB。
primitive2000/control256、reference matvec20000、deterministic100000/cell、tail B2=24/B3=56、pre1000000/post5000000 instructionsを維持。
入力生成・DF再分解・state solverは0。7200秒は安全上限でありETAではない。

## Synthetic検証と限界

[監査](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/synthetic_test_audit_v2.json)、[JUnit](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/synthetic_tests_v2.xml)：**156 passed**（v2 70 / v1 regression37 / coverage regression49）。local evidenceでありimmutable CI/外部再現ではない。

- 実際の凍結`actual_bounds`/`validation_times`をsynthetic prepared metadataで呼び、全7 cell・JSON roundtripでv2契約に一致。旧v1のschema不一致もmetadataだけで再現。
- 時間/index/順序/登録時間の不足・追加/cell/件数/scalar型/unknown field/control-bound schemaの21変更例を拒否し、boundsと差分の保存を検査。
- setupへtoy load/prepareを注入し、validの場合だけreference境界へ到達。coverage/instruction cap/order/cutoff拒否では全recordが残りreferenceへ進まない。
- saved bounds/prepared/coverage/receipt改ざんを拒否。旧v1 grantをv2で拒否。同一科学plan/継承methodsを照合。
- 旧toy numeric/injected wrapper scheduler/dummy child/watchdog/byte auditを回帰検証。

fixturesは実分子NPZ読込・SCF/DF/solver・real native prepare・trajectory draw・QuantumCircuit構築/transpileを禁止する。
**今回は実H6 snapshotの数値decode・native準備・signal/probe/sampling/build/compileを実行していない。**
actual H6 bounds、reference/correctness/cost、数値allowance、時間/メモリ実用性は今後の一回runで検査する。
129旧testsだけではinterface接続を検査できなかった問題に対し、今回は実runtime functionを使う接続検証を追加した。

## 実行認可待ちSTOPと次の具体的対象

[sealed preparation](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/sealed_preparation_v2.json) digest `2b8bc391d4e7f41d323ed3fe70812da3111a06223a1bc21fa4119532cc153be8`。
正本digestは[実行対象](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/launch_target_NOT_AUTHORIZED_v2.json)の`manifest_digest`。source/input/environment/resources/output/capsを固定済み。
[authorization template](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/authorization_template_NOT_AUTHORIZED_v2.json)はapproved_by_user=false。新schema `track_a_h6_technical_pilot_authorization_v2`の別grantを必要とする。
新output `artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1`は未作成。旧v1 output/消費済みgrantを流用せず、retry/resumeしない。
次の具体的認可対象は**このv2 source/manifest/snapshot/CPU/上限でH6技術pilotを一回実行すること**。
実行前にremote/source/input/環境・affinityを再確認し、正確な新grant bytesとSHAを別execution identityに保存する。
結果・STOP・欠測と保存監査を公開してremote再取得照合後mandatory STOP。H4補完＋H6 pilotの科学的次判断はGPTへ戻す。
H6本検証・H8・政策変更・scientific GOを自動認可しない。今回の準備は実行許可ではない。

[準備inventory](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/preparation_inventory_v2.json)、[保全監査](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/preservation_audit_v2.json)。
旧4244ファイル/rootレビュー42原本・既存dirty/untracked・Track B/別representation探索を保全し、新しい証拠だけをstage/publicationする。

## GitHub公開確認

[remote bytes照合記録](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/remote_verification_v2.json)。
公開commit `5eb66998f663e7cf8242b4839492532a65d4a2a9`を独立bare repositoryへGitHubから再取得し、science206/validation3・親入力33・旧v1 science202/raw14を照合した。
文書相対リンク23件、旧4244ファイル/rootレビュー42原本・dirty/untrackedの保全を確認。未認可template・未作成v2 outputも確認した。
この追記とremote記録を含む最終commitも再取得して照合する。科学再実行・認可は追加しない。
