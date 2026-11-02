# H4 Pauli-JW PR central-control A_GATE：保存order2 index1の両軸費用

2026-10-24 JST。**Stage A/B/C完了、cosine/sine各1回の費用compile成功、mandatory STOP。** 保存order2 eventを含むindex1を、元SOURCEに対応するbody-only接続と変更していないD2 central-controlで構築した。両軸ともRZ **1,129**、CX **1,793**、total depth **1,951**。同じindex1の旧generic-controlのRZ **9,082** より小さい。この1本の構造coverage・固定compiler費用の結果であり、全256本平均・母平均・精度適格性を確定しない。

[結果JSON](../artifacts/non_df_h4_a_gate_order2/2026-10-24/result_summary.json)、[2軸CSV](../artifacts/non_df_h4_a_gate_order2/2026-10-24/axis_costs.csv)、[保存証拠検証](../artifacts/non_df_h4_a_gate_order2/2026-10-24/verification.json)、[証拠入口](../artifacts/non_df_h4_a_gate_order2/2026-10-24/README.md)。

## 固定条件と来歴

Linear H4、隣接1.00 Å、STO-3G、system8＋ancilla8、Pauli-JW Hamiltonian184非identity項、PR prefix27。S2、T=.8、q8、r4、K4、元PR law・係数・分割を維持した。新しいHamiltonian・状態・eventは生成していない。

Qiskit1.3.0、native basis rz/sx/x/cx、optimization level1、seed17、backend/coupling/layout/routing指定なし。元Hadamard wrapper全体を費用compileし、状態準備費用を除外した。

| 来歴 | 固定値 |
|---|---|
| 専用branch | `codex/non-df-h4-a-gate-order2-execution-20261024` |
| GitHub取得済み基点 | `966c2cb323e3fe34032dc3be9f6b962936415e77` |
| 元H4 SOURCE | `0c329b30a81accb35c212be8defa66fee0dd0f12` |
| D1 SOURCE | `9843b803878db6c3aee7ab5e7c6bb4818572c944` |
| D2 SOURCE | `74515d8eb4f073c40fc783681a3775dbca89e9b1` |
| 今回の科学SOURCE | `ee7d281b5daead1d0d5e81e00c9a0ed4727792b0` |
| 結果commit | `f4ebb25292dd246b4d77e74f004b3bc50b377f37` |
| remote SHA | push認証失敗・branch検索0件のため未取得 |
| trajectory index | **1** |
| trajectory ID | `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa` |
| trajectory hash | `0d16d201c87fdbb5f3e219dffb7dca7d48a69799290453e9ebb110a267db0e4b` |
| event hash | `9cea6ce6d3659094b6b0d0a6da8b935cf32860c97a2e01a75a0761eded9f1779` |

[GitHub確認](../artifacts/non_df_h4_a_gate_order2/2026-10-24/github_verification.json)には基点commitとDRAFT契約の実取得・Git blob照合を保存。[承認](../artifacts/non_df_h4_a_gate_order2/2026-10-24/approval.json)は添付35fb6e4cの原文SHAに結び付く。事前選択された「order2を含む最小index」を使用し、代表random sampleとは扱わない。

初期実装commit `c93261c506baeacf3d6b682c1976d051bf8168b8` の後、両軸scopeと旧D2比較入力を追加固定した `ee7d281…` を科学SOURCEとした。このcommit後に初めて新seal・新grantを発行し、実H4 replayを行った。結果commit・push・remote照合は[公開状態](../artifacts/non_df_h4_a_gate_order2/2026-10-24/publication_status.json)を参照。

## 再利用と人工検証

[接続source](../src/trotterlib/h4_a_gate.py)、[runner](../scripts/run_h4_a_gate.py)、[人工テスト](../tests/test_h4_a_gate.py)。D1 append関数・保存event復元・forward/event/reverse schedule、D2 central変換・wrapper/compiler・seal検証・並列batch・native witnessを再利用した。旧sourceは変更せず、再利用関数のnamespace/axis/script bindingだけを接続した。D1 `build` はgeneric whole-body controlを含むため、実H4では呼ばずbody-only接続を追加した。

新人工**12件PASS、failure/error0**。元D1 bodyとのtoken/phase対応、非可換order2のproduct順序・qubit配置、negative product sign・order/constant phase、両control branch・両軸wrapperを小規模oracleで検証。権限前の量子import禁止、generic wrapper/sampling/compileの不呼出し、grant・非返還予算・STOP、SOURCE gate失敗時compile0、2軸予約/回収、重複axis拒否も対象とした。

既存D2候補12件＋比較再設計8件の**保存済み20件の意味対応証拠**をSHA保全して再利用し、今回再実行したとは扱わない。初期の人工負fixtureのtuple操作誤りはSOURCE前に修正し、失敗ログも保存した。科学callのretryではない。主agent自身の検査であり、独立別担当reviewやcompiled全H4誤差certificateではない。

## Stage A/B：保存order2と位相のSOURCE対応

[Stage A](../artifacts/non_df_h4_a_gate_order2/2026-10-24/run01/stage_A.json)は `PASS_FIXED_ORDER2_SOURCE`。ID/index/hash/q/r/元H4 SOURCE、component label・係数絶対値・sign、event順序、rotation angle、phase規則を元append処理に照合した。32保存eventのうちposition30（step7、inner2）がorder2、他31件はorder0。sampling・RNG replayなし。

order2はproduct ID43（sign+1）、ID4（sign−1）、rotation ID72（sign−1）の順。保存rotation angle `0.029672025283184623`、order phase−1。元処理どおり負productのπとorder phaseをそれぞれ適用し、constant phaseは各stepのreverse保持部の後に適用した。

[Body照合](../artifacts/non_df_h4_a_gate_order2/2026-10-24/run01/body_source_check.json)は469 primitiveの数値tokenが固定SOURCE recipeと一致。[変換mapping](../artifacts/non_df_h4_a_gate_order2/2026-10-24/run01/controlled_mapping.json)には保持27項forward/reverse・order2 product・rotation・qubit/ancilla・phase所有者を保存。[Stage B](../artifacts/non_df_h4_a_gate_order2/2026-10-24/run01/stage_B.json)は `PASS_BODY_ONLY_AND_CENTRAL_ONCE`。

body replayと変更していないD2変換を各1回実施。集約global phase `0.26518225073344787` をancilla8のP gateへ1回転送し、event phaseを別途重複加算していない。両wrapperは同じcontrolled body QPY SHA `ac1630c27caae5b5eea90f76b9e2d4de4b51d8a57b78d2d1bdcd139781e0bd21` を参照する。

actual/expected tokenのraw JSON SHAは異なるが、固定codeの条件は数値token一致でありPASS。int/float表現差を含み得るraw JSONのbyte同一性を要求する契約ではない。両digestを保存し、旧historical hash問題の原因解明とは扱わない。元Pauli-PR historical hash **FAIL／原因UNKNOWN** と `historical_program_identity=NOT_ESTABLISHED` を維持した。

## Stage C：両軸費用と限定比較

[固定SOURCE生成の比較JSON](../artifacts/non_df_h4_a_gate_order2/2026-10-24/run01/comparisons.json)。共通bodyから各軸1回wrapper化し、別workerで各1回compileした。価格一致を事前仮定していない。

| 対象 | RZ | CX | total depth | RZ depth | CX depth |
|---|---:|---:|---:|---:|---:|
| 今回 index1 cosine | **1,129** | **1,793** | **1,951** | 651 | 1,278 |
| 今回 index1 sine | **1,129** | **1,793** | **1,951** | 651 | 1,278 |
| 保存D2 index0 cosine（再compileなし） | 1,145 | 1,856 | 2,035 | 664 | 1,357 |

同じindex1の旧genericは両軸9,082 RZ、今回centralは各軸1,129 RZ、差（central−generic）は各−7,953 RZ。D2 index0とのcosine RZ差は−16。index0/1ではevent列も異なり、order2単独の因果効果や代表平均とは扱わない。

今回sine−cosine差は上記5費用指標すべて0。一方pre-transpile sdgはcosine40、sine41で、input/output fingerprintおよびQPY SHAも異なる。同価格はこのtrajectory・compilerでの実測であり、全trajectoryの一致を保証しない。

| Gate histogram | cosine | sine |
|---|---|---|
| pre-transpile | crz464, cx872, cy2, cz1, h170, measure1, p1, s40, sdg40 | 左と同じ、sdgのみ41 |
| post-transpile | cx1793, measure1, rz1129, sx120 | 同左 |

各軸の `run01/variants/{cosine,sine}/` にinput/output QPY、fingerprint/SHA、histogram、compiler profile、SOURCE/trajectory ID、予約・開始・完了markerを保存した。固定wrapperにcallbackがないためpass traceは `null / NO_CALLBACK_IN_FIXED_WRAPPER`。新しいpass traceやcompiled誤差保証は生成していない。

## 実行契約・終了監査・保全

[seal](../artifacts/non_df_h4_a_gate_order2/2026-10-24/seal.json)、[active contract](../artifacts/non_df_h4_a_gate_order2/2026-10-24/active_contract.json)、[新grant](../artifacts/non_df_h4_a_gate_order2/2026-10-24/run01/grant.json)、[ledger](../artifacts/non_df_h4_a_gate_order2/2026-10-24/run01/call_ledger.json)、[STOP](../artifacts/non_df_h4_a_gate_order2/2026-10-24/run01/STOP.json)。grant ID `a6bca3a5e33b9ec0f285a173a170505945553e1ef287912767c24ff31229c744` は一度消費済み・非返還・retryなし。

| call | 消費／固定最大 |
|---|---:|
| body replay | 1/1 |
| central-control build | 1/1 |
| axis wrapper build | 2/2 |
| cost transpile | 2/2 |
| generic control／sampling／state-signal／N_G | 各0/0 |

07:21:42 UTCに費用call2件を先行予約。engine PID559358、cosine PID559490、sine PID559491は全てnative exit0・reaped。engine終了07:21:47 UTC、保全再照合後07:21:49 UTC（16:21:49 JST）にmandatory STOP。記録PID/start_ticksに一致するprocessは残っていない。他processを走査・停止していない。

[読取専用verifier](../artifacts/non_df_h4_a_gate_order2/2026-10-24/verify_saved_outputs.py)で旧D2 seal verifierを再利用し、code52、入力100、runtime6,999、旧保全対象1,822、run inventory48をSHA照合した。既存manifest177項目も不変。構造・保存関係31check PASS。QPY再load・compile・科学数値診断はなく、fingerprintは保存metadata間の対応とQPY byte SHAを検査した。

最新版運用指示§8.4/§10.1/§10.2を適用。参照された§10.3は同ファイルに存在せず、未確認の規則を仮定していない。CPU affinity64・メモリ・PSI・空き容量を観測し、承認済み2軸compileを並列化した。共有設定・他ユーザー・他worktreeは変更していない。

旧SOURCE/raw/grant/ledger/marker/STOPを保存し、**Pauli-PR historical hash FAIL／原因UNKNOWN、元H4 native UNKNOWN、旧N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCE、正式certificate未受理**を維持した。今回のnative PASSは新A_GATE runだけの判定である。

## Q1〜Q6と次の承認対象

| 問い | 今回の回答 |
|---|---|
| Q1 保存order2にD2を接続できるか | このindex1でSOURCE token対応・位相転送条件を満たし両軸compile完了。 |
| Q2 product/sign/phase条件を満たすか | 保存列・負sign・order/constant phaseを元append処理で維持し、人工oracleと固定SOURCE接続条件にPASS。全H4 compiled数値誤差の厳密保証ではない。 |
| Q3 両軸価格 | 各RZ1129/CX1793/depth1951。 |
| Q4 同じindex1の旧genericより低いか | 両軸RZ9082→1129。同一trajectoryでの実装・compiler比較。 |
| Q5 軸価格差 | 測定5指標の差0。回路・fingerprintは異なり他trajectoryへ一般化しない。 |
| Q6 全cohortへの拡張が必要か | 保存256本のcentral-control標本費用を議論するなら必要。このgateだけでは平均・tail・母平均を確定できない。 |

**実行済み作業はここでmandatory STOP。** 以下は提案のみで、新grantやDRAFT契約の有効化は行わない。

| 優先候補 | 担当・次の作業／必要承認 |
|---|---|
| A_COHORT_COS | ユーザー：保存256本の費用評価を行うか判断。Codex：D2 index0と今回index1の再利用条件を固定し、成立すれば残254本のbody/central/wrapper/compileを新契約・新grantで評価。新samplingなし。 |
| A_AXES | ユーザー：complex signalの比較へ進む際、sine費用を揃える範囲を承認。Codex：今回index1 sineの再利用と残対象を新契約に明記。価格一致を仮定しない。 |
| B_CONDITIONAL_CNG | ユーザー：paired費用と条件付き誤差会計を接続する再解析を別承認。Codex：tail・rho・certificate未確定を保持して準備。今回はN/G0。 |
| C_SELECTED_DIAGNOSTIC | 必要な場合のみユーザーが残差診断対象・上限を承認し、Codexが別SOURCE/grantで実施。人工PASSで代替しない。 |
| D3 | DF basis reuseの対照実験は後順位。今回実装・実行せず別承認。 |

未観測order4・母平均tail・compiled実装残差・正式certificateは未確定。全256本の費用評価を行っても、それだけで母平均保証やN/G適格性が成立するとは扱わない。
