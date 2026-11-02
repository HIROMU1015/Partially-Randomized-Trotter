# H4 Pauli central B0/B0I：matched比較の準備

2026-10-25。**既存実装を使って、prefix27の決定論的B0/B0Iをcentral PRの比較対象へ追加できる。**
専用接続と人工検証を用意した。今回、実H4のbody生成・compile・signal・bias・N/G計算は行っていない。
新しいB0の科学結果は未取得であり、B0対PRの優劣は未判定である。

基点は `codex/non-df-h4-data-catalog-20261025` / `7a83dc186bddcc35e528e38032814423ef1cba88`。
作業branchは `codex/non-df-h4-pauli-b0-preparation-20261025`。
並行する `codex/non-df-h4-pr-mean-cost-design-20261025` のworktree・SOURCE・結果・実行状態には変更を加えていない。
母平均費用の検証完了を前提にもしていない。

## 現在の利用可能範囲

| 項目 | 確認結果 | 実H4について未取得のもの |
|---|---|---|
| B0/B0Iのbody | `PauliSignalAdapter`が既に対応。両methodは同じ保持half列とconstantを使用 | 新候補としての入力・結果の固定 |
| 同等性 | Pauliのtailは非identityのみ。identity係数はconstantとして一度保持。B0とB0Iは同じ1候補 | 別々の科学評価は不要 |
| 保持prefix | receiptとC modelのPR保持27項が対応し、I/Zだけのsupport1または2 | prefix変更・探索は対象外 |
| 制御 | D2の `central_control_body` と既存Hadamard wrapperを接続済み | central B0の両軸compile価格 |
| D1 retained-only | bodyの保持27、S2順序、constant、q/Tが対応する再利用候補 | generic価格6,524をcentral価格とみなすことは不可 |
| 信号・会計 | Cの演算・保存full-H参照区間、既存shot会計を再利用可能 | B0固有のbias・数値allowance・headroom・条件付きN/G |

[接続コード](../src/trotterlib/h4_pauli_b0.py)は入力ファイル読込、signal、compile、samplingを持たない。
`matched_config(saved_PR_config)` が `L_D` を必須にしてB0へ変更し、`cost_samples=1` とする。
B0Iという別名も同じB0設定へ正規化する。H4ではreceipt由来の **`L_D=27`** を明示する。
`r=4/K=4/seed` は比較metadataとして保持されるが、B0ではevent生成に使われない。

旧 `h4_pilot_science.case_config` はPR以外で `L_D=None` とし、adapterはB0のNoneを0にする。
さらに旧pilotのcase/seed経路は4方式向けである。旧runnerへB0を単純追加しない。
今回のhelperはprefix欠落・空保持・別表現・別methodを拒否する。既存SOURCEは修正していない。

## 科学条件の対応

[機械可読の対応表](../artifacts/non_df_h4_pauli_b0_preparation/2026-10-25/condition_correspondence.csv)。

| 条件 | Pauli B0/B0I候補 | 保存Pauli central PR |
|---|---|---|
| full target | 同じ保存H4、隣接1.00 Å/STO-3G、184非identity項＋constant | 同左 |
| 保持集合 | 同じabs係数順prefix27、Z/ZZ。tail157を切り捨てる | 同じprefix27、tail157をRTEで補う |
| evolution | S2/T=.8/q8。保持forward half→reverse half | 保持half→r4のevent→reverse half |
| law | なし、決定論的 | 保存finite paired RTE、K4/order0・2・4 |
| constant/control | constantを各stepで一度body phaseへ。D2がaggregateをancillaへ一度移す | 同じ位相所有・central制御 |
| state/wires | 保存raw ray、Qiskit little endian、system0–7、ancilla8 | 同左 |
| measurement | full Hadamard cosine/sine、sineはSdg-before-final-Hの+Im | 同左 |
| compiler | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし | 同左 |
| price scope | 状態準備を除くfull wrapper、RZ/shot。CX/depth等も保存予定 | 保存256本の両軸価格、PF価格とは区別 |
| tolerances | .05/.01/.005/.001/.0005 | 同左 |

保存Hamiltonian hashは `e0dbbe9a2b6e86f81004307cd337ba83ee1e25642f04a30b02365be4d2dae795`、
共通raw state hashは `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。
input receipt、C model、D1 sidecar・価格CSVと実際に照合した。
この対照はPauli内部のmatched B0対PRで、異なるDF rank/prefixや別研究B0+Iの費用値を持ち込まない。

保持27項は全て対角Pauliなので相互に可換である。理想的な保持S2は保持Hamiltonianの厳密evolutionと一致する。
これはSOURCEからの構造的事実であり、full-Hに対する切り捨てbiasが小さいという数値的結論ではない。
回路はq8の元順序を保ち、commutingだからという理由で別の合成familyへ変更しない。

## Central-controlとD1の再利用境界

`build_body` は既存adapterのretained列とconstant更新をそのまま使う。
`central_wrappers` はD2を一度呼び、非制御のbasis/parity、central controlled rotation、ancilla phaseを保持して両軸を作る。
新しいcontrol合成やcompilerは実装していない。実H4ではこの関数を呼んでいない。

D1 SOURCE基準Bの
`artifacts/non_df_h4_d1_source_execution/2026-10-24/run01/variants/pauli_jw_PR_retained/body_before_control.qpy`
は、sidecar上432個のPauliEvolution（8 step×forward/reverse各27）を持つ。
`h4_d1_adapter.build(..., part='retained')` はeventを挿入せず、各stepのconstantを保持する。
`PauliSignalAdapter` のB0 trajectoryも同じloop順序・half time・constant更新である。
人工fixtureではD1の保持bodyと専用B0 bodyのoperatorが一致した。
実H4 QPYの新decode・数値operator照合は今回は実施せず、次回は保存QPYのoperand/phaseをSOURCEと照合してから流用する。

一方、D1の `input.qpy/output.qpy` はwhole-body generic制御のcosine wrapperである。
保存RZ **6,524**（CX4,896/depth8,803）はこの診断の価格であり、新central B0価格ではない。
D2のPF15,028、PR0 1,145 RZから保持部のcentral価格を引き算・推定もしない。
初回D1の旧fingerprint FAIL、後続SOURCE基準Bの9/9成功、旧原因UNKNOWNは別々に保全する。

## Signal・normalization・N/G会計の設計

ここは**次回実行案**で、値の計算は未実施。

1. C modelのexact binary64 coefficient、step time、保存raw stateを使い、保持列とconstantのみのB0 targetを評価する。
   `h4_ball_evidence` のPauli/phase/evolve/signalを再利用する。tail演算はNoneで、eventやPR meanは使わない。
   Cの `EnclosureEngine.evaluate` 全体はfull-H referenceとPF/PRを再計算するので呼ばない。
2. 保存 `pauli_jw_result.json#/reference` のreal/imag **全区間**を再利用する。
   `d = (s_B0 − s_full_saved)/ν`、`ν=<raw|raw>` を共通に扱う。
   raw stateを再正規化せず、exact norm又はその保証区間で最終内積を換算する。
   これで同じfull-H/同じrayに対するB0固有のbias区間を得る。新full-H状態作用は不要。
3. 各bias区間からexact有理数の中心 `b0` と、両端を覆う `u_enc` を保存する案とする。
   `u_enc` は今回未取得であり0とは置かない。保存参照区間の幅・norm換算・Arb誤差も全て含める。
   ideal C-targetとcompiled信号の差は独立のρに置き、位相/angle変換誤差を二重計上しない。
   この会計表現とproduction接続は次回実装・人工検証してから使用する。
4. 理想B0は実係数Pauli evolutionとreal constantのunitary積で、数学的B=1/log_B=0、測定範囲±1。
   PRのfinite-RTE B=1.2873709223899619やPFの旧log marginを転用しない。
   会計は既存 `scale_bounds(1,0)` の外向き丸め規則で **別のB_up** を記録する案とする。
   その数値は今回未計算。B_upを手動で1に固定しない。compiledとの偏差はρで別に扱う。
5. `h4_conditional_cng.base_axis/shot` のexact会計を使い、
   `ε_axis=float(ε_sig)/sqrt(2)`、`α_axis=.025`、`u=u_enc+ρ`、
   `h=ε_axis−|b0|−u`、`u/h≤.01`、`N=ceil(2 B_up² ln(2/α_axis)/h²)` を維持する。
   headroom/guard不足は理由付きでN/G=null。未知ρもactual N/G=nullとする。
   不適格、証拠不足、まだ未計算を分け、nullを0で補わない。
6. 新B0の両軸compiled価格で `G=N_cos C_cos+N_sin C_sin` を会計する。
   まず双方ρ=0という**仮想条件**で保存PRと比較する。actual精度保証にはしない。
   次にB0自身のguard端点と、既存PRの `G_guard_range_exact` を使い、独立guard内の区間順序を確認する案。
   端点域が重なる場合は順位未確定とし、再ランキングのために仮定を変えない。

PRは保存256本の価格・bias・normalization・条件付きN/Gをそのまま引用する。
新B0 biasをPRへも転記しない。PRの母平均tail、全lawρ、正式certificateは未保証のまま。
既存PF/PR0/PR1の6局所Arb保証は新B0のcompiled保証ではない。
この限定G比較のためにR2-C全compiler証明やPR母平均検証の完了を必須にはしない。

## 次回の最小実験案：未承認・実行無効

[非実行計画JSON](../artifacts/non_df_h4_pauli_b0_preparation/2026-10-25/experiment_plan_DRAFT_DISABLED.json)に有限範囲を記載した。
実行runner/科学grantは作っていない。下記はユーザーの別承認後だけ行う提案である。

| 新規作業 | 上限案 | 何が分かるか |
|---|---:|---|
| 保存D1保持bodyのSOURCE/operand/phase照合 | 1 body | prefix27・constant・q/Tとの対応。失敗ならSTOPし勝手に再生成しない |
| D2 central body＋両軸wrapper構築 | 1＋2回 | 比較対象の固定入力。B0/B0Iを二重構築しない |
| full-wrapper compile | **2回**（各軸1） | 新B0のRZ/CX/depth/RZ-depth/CX-depth |
| B0 retained-targetの複素Arb状態作用 | **64bitで1回、必要時96→128、最大3回** | 1回で両軸bias区間と自分の数値allowance。新full-H作用0 |
| scalar会計 | 5精度×2軸、ρ=0/自身のguard端の最大20 shot評価、5条件のG区間とPR比較 | 十分条件付きG、guard内の比較範囲。PR shot再計算0 |

Arb案は許容誤差に対する数値的不確かさを明示してGを比較するための一つの最小接続案。
各bitの状態作用は1回だけで両軸を同時回収する。64bitで各条件のguard成立/不成立が決まれば終了する。
境界を跨ぐ有限幅が判定を妨げる場合だけ次へ進む。切り捨てbias自体の大きさを精度昇格で改善しようとしない。
128bit、非有限、入力不整合、保存参照の幅が律速して追加精度が役立たない場合は未確定を保存してSTOP。
1e-8等の幅を新しい受理閾値にせず、既存guardを判断基準にする。

今回の範囲で上記Arb production接続は実行しておらず、これは実行承認を兼ねる契約ではない。
実行時には新B0 SOURCE/runtime/入力・数値方針を固定し、通常の結果とnative終了を保存する。
承認の証明書・新grant発行システムは不要。旧grant・旧契約・旧STOPへ変更しない。

実際のcompiled精度まで求める場合は、別途B0の両軸compiled Arb診断が候補となる。
これは上記の条件付き比較の最小案には含めず、自動追加しない。
prefix/q探索、追加PR trajectory/compile/sampling、母平均tail、DF研究、R2-C継続も対象外である。

## 人工検証・保全・公開

[人工結果JSON](../artifacts/non_df_h4_pauli_b0_preparation/2026-10-25/technical_checks.json)。
新接続6、既存D2 12、純scalar会計16、既存B0同値性/証拠不足/null処理5、**計39件PASS**。
新テストは2〜3 system-qubitのみ。D1のtoy保持body再利用、負係数、非可換順序、Y/bit順序、
constant relative phase、両測定軸、共通raw norm、sampling/generic-control/top-level compile不使用を検査した。
既存テストには人工generic control/人工compileが含まれる。これを実H4科学実行と混同しない。
主agent自身による技術確認で、独立した科学レビューや性能証拠ではない。

Qiskit1.3.0の既存専用runtimeを読み取り利用した。パッケージ変更・共有設定変更・他ユーザー操作なし。
CPU/RAM/workerに新しい上限は設けていない。軽量の技術検査であり科学workerは起動していない。
初めに確認したC専用PythonにはQiskitがなく、既存D1/D2 runtimeへ切り替えた。科学callの失敗・retryはない。

[証拠参照](../artifacts/non_df_h4_pauli_b0_preparation/2026-10-25/evidence_references.json)は既存SOURCE/保存項目とSHAを記載し、rawを複製しない。
旧SOURCE・raw・入力・結果・grant・ledger・marker・STOP・データカタログは変更していない。
[公開状態](../artifacts/non_df_h4_pauli_b0_preparation/2026-10-25/publication_status.json)でcommit/remote確認を記録する。

**mandatory STOP。** 次の担当は**ユーザー**：上のmatched B0/B0I一候補・両軸2compile・最大3 target作用・限定会計を実行するか判断する。
**Codex**：明示承認後のみ、B0固有bias/allowanceのproduction接続を実装・人工検証し、その範囲だけを実行する。
追加科学計算、研究方針採択、新規レビューへ自動移行しない。
