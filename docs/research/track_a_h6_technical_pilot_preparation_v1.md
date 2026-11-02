# Track A H6技術pilot：保存snapshot接続・実行前固定 v1

2026-10-10 JST。[独立Hermitizationレビュー](track_a_h6_df_hermitization_independent_review_2026-10-10.md) §14–16と
[保存DF入力完成結果](track_a_h6_saved_df_completion_parallel_result_v2.md)に基づく**準備だけ**を完了した。
この文書、source、seal、synthetic PASSは実行認可・科学GOではない。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION。
次の明示的ユーザー認可があるまでpilot runnerを起動しない。Track Bと別branchのrepresentation探索は対象外。

## 入力とsource来歴

linear H6 / 1.00 Å / STO-3G、12 modes、Nalpha=Nbeta=3、sector400、actual rank19。
生成順の全19 signed fragmentsを保持、tol-only1e-8、cutoff0、final_rank/fallback/fragment削除なし。
weighted projection追加予算1e-10 Ha、decision9.9e-11 Haを変更しない。
[政策exact copy](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/fixed_policy_v1_unchanged.json)。工学PASSでありrepresentation-error certificateではない。

- pilot source commit `a643220cde1e24b7e3d637f4bc5d1b0342ce2e86`：[source freeze](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/source_freeze_v1.json) science 202 / validation3。
- 親入力source `f37005f01b2be38c5993d6e82df91abe9c643d21`、raw結果 `554fc52add39c2c1b45b765a3135df76fda6f15a`、監査結果 `f52787a22542b31bd39fd004a8d3d71325bc56b0`。
- [入力snapshot](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/h6_input_snapshot.npz) SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、[snapshot receipt](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/snapshot_receipt.json)、[DF receipt](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/df_receipt.json)。
- Hamiltonian hash `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、designated saved state hash `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3`。
- [入力identity](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/input_identity_v1.json)は親33ファイルのlocal/Git bytesとNPZ headerを照合済み。今回は数値配列としてdecodeしない。
  親入力完成grantは消費済みであり新pilotへ流用できない。

旧loader、旧bound-launch gate、H4 source/freezes、過去のSTOPは変更せず、新versionを追加した。
[contract](../../src/trottertracks/resource_applicability/ax2b_h6_pilot_contract_v1.py)、[port](../../src/trottertracks/resource_applicability/ax2b_h6_pilot_port_v1.py)、
[watchdog](../../src/trottertracks/resource_applicability/ax2b_h6_pilot_watchdog_v1.py)、[runner](../../scripts/resource_applicability/run_track_a_h6_pilot_v1.py)、
[byte auditor](../../scripts/resource_applicability/audit_track_a_h6_pilot_v1.py)、[tests](../../tests/tracks/resource_applicability/test_ax2b_h6_pilot_v1.py)。
portは既存MolecularPortのnative作用、cost-only sampling、ordinary/directional lowering、measured Hadamard wrapper/実compileを再利用する。
入力読込には新policy-bound saved-completion loaderを使い、保存stateを再正規化しない。

## 固定cellと検証

T=0.8、epsilon_signal=0.001は診断ラベルのみ。同精度を達成したとの主張・化学的energy精度への換算をしない。

| cell | formula | prefix | q | R | r | K | cost replica |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | S2 | 10 | 2 | — | — | — | 1 |
| B1 | S2 | 19 | 1 | — | — | — | 1 |
| B1 | S2 | 19 | 2 | — | — | — | 1 |
| B1 | S4 | 19 | 1 | — | — | — | 1 |
| B1 | S4 | 19 | 2 | — | — | — | 1 |
| B2 | S2 | 10 | 2 | 4 | 2 | 2 | 2 |
| B3 | S2 | 0 | 2 | 4 | 2 | 6 | 2 |

prefix0でもone-bodyは決定論的。7 cell、9 replica group、ordinary/symmetric_directional×cosine/sineで36 wrapper。
random group4回のwhole-trajectory sampling、outer occurrence8回。group内4 wrapperは同じevents/seedを共有しdigestを検査する。
seedは旧draftのSHA256規則を維持、collisionでSTOP。これらはcost用の4 trajectoryで、測定shot/信号MCではない。
primary symmetric_directional、ordinaryはpaired sensitivity。費用を個別に保存し、n=1/2のmin/max/mean/sample SDを記録する。
random n=2を精密な平均・勝者選択の根拠にしない。N/Gはnull、u/ground-state未認定、UNDETERMINED。

予定している検査は以下の順序で実行する。ここでは未実行。

1. snapshot/receipt/source/environmentを再照合し、primitive sector構造証明・basis bridgeと新prepared representationを作る。
   rank/order/cutoffを再検査しpreparation/partition/basis hash、lambda_r、identity抽出を保存する。
   静的なactual-time scheduleは245 physical keys×saved state/first/last sector column=735 probesで固定済み。
   **実際のprepared representationに依存するinstruction boundは今回未計算**。将来の実行時に既存actual_boundsを算出し、schedule一致・735回・上限を確認して保存する。
   このgateに失敗したらnative作用・sampling・circuit buildへ進まない。入力の小さなprojectionから回路費用不変を推測しない。
2. 400次元sector referenceをbounded NumBa4 matrix-freeの400 columnsから一回作る。独立occupation全columnsを照合し、expm/eighのstate差<=1e-10を要求。
   full-space4096²行列やfragment行列の一括cacheは作らない。primitive oracleは一個を置換するsector cache。
3. primitive完全作用後のsector leakageと独立sector expmを検査。S4負時間、undo signs、全登録時間を含める。
   corrected/raw有限平均はnative Hornerと独立occupation行列＋Taylor前進漸化式で照合し、中間norm・trace・b/logB・B×raw=correctedを保存する。
   B0はexact truncated referenceとの差からsigned discard/PFを分離。B2/B3はexact-tailとのsigned差からfinite-RTE/outer-PFを分離。
   複素数での加法分解を記録し、absolute errorの加法性を主張しない。
4. 全7 correctness通過後だけsampling/build/compile。3 probes、both ancilla branches、ordinary/directional、Hadamard X/Yを照合し、216 control probe actionsを見込む。
   同一eventsの36 measured full wrapperについてRZ/CX count/depth、total depth/size、fingerprint、compiler hash、classical時間/RSSを個別保存する。state preparation費用は含まない。
5. 入力/source/environmentを再照合、raw監査・保存bytes監査・GitHub公開・remote hash照合を行いmandatory STOP。
   科学GO/STOPはGPTへ戻す。failure/欠測も保存し、再試行・rank/R/tolerance/seed救済変更を行わない。

norm/leakage1e-12、mixed agreement 1e-9 + 1e-10×max(1,intermediate scale)を維持。
b/logBのoverflow・raw subnormal/underflowはaction開始前にSTOP。非unitary有限平均を再正規化しない。
forward/expm/eigh照合はbinary64の工学診断であり共通roundoffを除外する厳密u証明ではない。
H4新policy bridge、精度を一致させる探索、H6本検証・H8 held-out・energy estimationはこのpilot外。

## 資源と認可対象

[sealed preparation](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/sealed_preparation_v1.json) digest `f05a6c26ae2d1a19a15808a0dd45b416848cadf872f84b9396c526b77bb8aba5`。
[環境/CPU観測](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/environment_resource_observation_v1.json)、[実行対象（未認可）](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/launch_target_NOT_AUTHORIZED_v1.json)。
CPU IDs `[0, 2, 5, 6]`は4論理CPU（観測時4 distinct cores）、worker1/Numba4/OMP4/BLAS1/chunk1、GPUなし、host専有保証なし。
並列化は既存sector matvecに限定。sector expm/eigh/native simulator/Qiskit compileの速度改善は未測定。
phase input-reference1800 / correctness1800 / wrapper-cost3600秒、total7200秒（2時間の安全上限、ETAではない）。
AS8GiB、output512MiB、log64KiB、diagnostics/progress1024。primitive2000、control256、reference matvec20000、deterministic100000/cell、
corrected+raw tail B2=24/B3=56、oracleには別の同上budget。pre-transpile1000000/post5000000 instructions、compile36/trajectory4/occurrence8。
入力生成・DF再分解・state solver・solver matvecは0。量子資源（gate/circuit）と古典資源（CPU/wall/RSS/出力量）を別記録する。

[authorization template](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/authorization_template_NOT_AUTHORIZED_v1.json)はapproved_by_user=falseであり、新schemaの明示的pilot一回grantだけを受け付ける。
今回のユーザー指示は準備への認可として扱った。次に認可する具体的対象は**このmanifest/source/input/CPU/output/capsに結合したH6技術pilot一回**である。
将来の認可文・新grantのexact bytes/SHA・source commit・manifest digestを別execution namespaceへ保存し、source/remote preflight後に起動する。
output `artifacts/resource_applicability/track_a_h6_technical_pilot_v1/2026-10-10/launch_v1` は未作成。retry/resumeなし、旧run/旧認可の再利用なし。runnerをdefault modeで呼ぶとmetadataだけを出し科学importしない。

## 準備で得た証拠と限界

[synthetic監査](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/synthetic_test_audit_v1.json)、[JUnit](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/synthetic_tests_v1.xml)：129 passed（新pilot37、旧並列51、旧入力完成41）。
2×2 toy forward/Horner照合、schema/計画/CPU/one-shot gate、injected cost schedule36/4/8、dummy watchdog、byte audit/tamperingを検査。
実分子NPZ decode、SCF/DF/solver/H6 signal、実trajectory draw、量子回路build/transpile/compileを禁止するfixtureで実施した。
ローカルsynthetic検証であり、CI/外部再現・real H6 correctness・run時間/メモリ実測・回路費用の科学証拠ではない。
[準備inventory](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/preparation_inventory_v1.json)。古い4188ファイル、root review originals、dirty/untracked、他branchを保全する。

H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP、next_stage_authorized=falseを維持する。
次の科学的GPT checkpointはH4補完＋H6 pilot結果。準備のみで研究意味論/target/coefficient/独立性を変更していないため、追加科学承認を代行しない。

## GitHub公開確認

[remote bytes照合記録](../../artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v1/2026-10-10/remote_verification_v1.json)。
公開commit `dbaed646d5369f19c2431668cd5c7a49c6340daa`をGitHubから独立bare repositoryへ再取得し、
source202/validation3、親入力33、親frozen source/tests202、相対リンク22件を確認した。
旧4188ファイル/root review原本42件・dirty/untracked保全、未認可template・未作成pilot outputを確認。
この追記とremote記録を公開した最終commitも再取得して照合する。新しい科学結果・実行認可は追加しない。
