# H4 Pauli central PF cosine：Arb包含評価

利用者は2026-10-25にPF cosine 1件の実H4 Arb包含評価を明示承認した。
基点 `5c69e1339955c77273b356df30e94ec3c57458b1`、専用branch
`codex/non-df-h4-central-arb-pf-cosine-20261025`。

**局所実装残差ρの包含上界を取得し、保存ρ予算を満たした。64bit・状態作用成功1/総試行1で終了。**
ρ上端は `180675711/18889465931478580854784`（約9.565e-15）。
保存予算約2.6490038338323596e-6との関係が確定したため、96/128bitは未実行。
新grant発行/消費1、worker native exit0/reap・controller exit0・mandatory STOPを保存した。
**局所数値包含と正式certificate受理は別であり、受理は行っていない。**

科学SOURCE：**`92a5f104a2e9962353870e87f1d910d236adba41`**。
[seal](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/seal.json)と
[active契約](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/active_contract.json)を実行前に固定した。
前段の非実行契約は変更せず、今回の[明示承認](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/approval.md)へ結合した新契約を使用した。

## Compiled信号・直接差による局所ρ

以下の十進端点は外向き表示。判定は保存exact rational端点で行う。

| 量 | 包含区間・値 |
|---|---|
| compiled cosine signal | `[-0.1615158549129582801844, -0.1615158549129463867033]` |
| signal区間幅 | `54849/4611686018427387904` ≈1.18935e-14 |
| ρ raw Arb ball | `[-8.802291796027771541775e-15, 9.564892499099763223045e-15]` |
| **ρ≥0との共通部分** | **`[0, 9.564892499099763223045e-15]`** |
| ρ上端 exact | `180675711/18889465931478580854784` |
| raw ρ区間幅 exact | `173473151/9444732965739290427392` |
| 非負ρ区間幅 exact | `180675711/18889465931478580854784` |
| 保存ρ予算 exact | `10107730069649565297/3815672118158673332666368` |
| 保証付き比較 | **UPPER_WITHIN_BUDGET** |
| 精度・call | **64bit・状態作用成功1/総試行1、局所差scalar1** |
| wall時間 | 約47.2747秒 |

[exact端点・外向き表示](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/reported_intervals.json)、
[信号](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/signal_64.json)、
[直接差のρ](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/rho_64.json)、
[集計](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/result_summary.json)。

Arbの対称ball表現は微小な負下端を持つため、報告区間には定義ρ≥0との共通部分を併記した。
**raw ballは無変更で、ρ=0の確定を意味せず、旧nullも0へ変更しない。**
信号/probabilityのclippingやstate再正規化ではない。
監査では保存signal/C-target区間の全端点差が保存ρ区間に含まれることと、ρ上端≤予算をFractionで検算した。
Arb/state作用の再実行は0。64bitで確定したため、幅目標1e-8を理由に昇精度していない。

## 固定科学範囲

linear H4、隣接1.00 Å、STO-3G、Pauli-JW非identity184項/prefix27、S2、T=.8/q8/r4/K4。
system8＋ancilla wire8、保存D2 PF cosine compiled QPY、既存raw ray・PF C-target区間を使用した。
新compile/sampling/Hamiltonian/state生成なし。C SOURCE e80d7510880f8db993b777c627bb8e95b05a070d、
D2 SOURCE 74515d8eb4f073c40fc783681a3775dbca89e9b1。

QPY SHA=`8537e01152d7b9a169bcea2dc20f4ce107d560b3af39b47b326e37c3c309a77d`、
fingerprint=`ac65c4de7d57b254b5fced776a039dd36a76b35fed0a33c79eb5478cdc44a173`。
RZ15,028/SX7,170/CX26,768、計48,966gate＋最終measure1。
state hash=`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。
実QPYをdecodeし、global phase・全binary64パラメータ・wireが保存descriptorと一致した。
[構造検証](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/structural_preflight.json)。
固定encoded program/raw ray/C-selected PF targetの局所量だけを保証対象とし、物理デバイス誤差・FCI stateとの差・一般compiler・PR全lawには外挿しない。

## 再利用とproduction差分

`h4_arb_bridge.py`のnative作用・exact center＋global L2半径・最終Z/ν包含を共通のprivate kernelへ分離した。
演算式は準備SOURCEと同じ。人工入口の小規模fixture guardは保持し、実H4では新adapter/controllerが固定入力・新grant・非返還call予約を確認する。
global/relative phase、標準rz/sx/x/cx、CXの向き、little endian、末尾measure、binary64値、rawνを変更しない。
入力state/出力stateを再正規化せず、出力norm除算やclippingを使わない。
PF C-targetは元C resultとR0保存区間をexact照合し、区間全体を使う。midpointをcertificateへ格上げしない。
初期vectorは`[raw, zeros]`。raw成分をexact dyadicとして入力し、同じraw rayのexact νを分母にした。
保存表示ν=1.0000000000000002を1へ置換せず、出力normによる除算や二重ν除算をしない。
PF targetはCのraw corrected_encoded_mean.realを同一exact νで除したR0保存区間。B_saved=1を一度だけ適用した。

既存Cの`real/complex_ball/phase/dyadic/endpoints/array_hash`、`CallBudget`、durable `save`、native witnessを再利用。
native witnessは待機・subreaper・fsyncの本体を変えずscope labelだけ今回のworkerへ変更する。
run01内の新しいgrant/CONSUMED/claim/ledger/native記録/STOPだけを作成する。
workerが照合するsealは実際に使うcode/runtime/入力に限定し、他worktreeのHEAD/statusや一時PSIを停止条件にしない。

## 人工検証

最終構成は新production11＋既存Arb10＋QPY6＋sine9＝36 PASS（修正前35件の記録も保全）。
早期停止（幅目標未達でも判定後は終了）、straddle/nonfiniteの有限昇精度、128bit後INCONCLUSIVE、
contract改変拒否、direct interval差とbudget3分岐、共通kernel一致、非返還call、grant再消費/STOP拒否、
人工processのnative exit/reapとscopeを確認した。
新たな実H4のStatevector作用やPR/C-target生成は行っていない。

## 精度・終了条件

64→96→128bit、各精度のcompiled状態作用最大1回、初回最大3状態作用、各精度の局所残差scalar縮約最大1回。
保存ρ予算は `10107730069649565297/3815672118158673332666368`（約2.6490038338323596e-6）。
`ρ_local = B_saved abs(m_compiled − m_C)`、B_saved=1。
上端≤予算または下端>予算ならその時点で終了する。straddle/nonfiniteの場合だけ次精度へ進む。
幅1e-8は参考目標であり、判定後の高精度化の理由にしない。128bit後はINCONCLUSIVEでSTOP。
失敗callも保持し、技術回復が必要なら新SOURCE/未使用grantで成功済み結果を再利用する。
正式certificate受理/LOCAL_BUDGET_MET更新、PR/R2/N_Gへ自動移行しない。

## 運用と保全

最新2026-10-23 policyの§8.4/§10/§10.1/§10.2を適用する。提供ファイルに§10.3はないため仮定しない。
CPU/RAM/worker数の任意上限なし。1科学対象の順序依存walkは逐次とし、phase cacheとexact permutationで重複を削減する。
共有設定・他ユーザーprocessの操作、AI subagent利用は0。旧証拠412 SHAと旧manifest186行を保全する。
局所数値包含と正式certificate受理、今回native終了と元H4のnative UNKNOWNは別に扱う。

[承認・入力・人工・SOURCE/grant・結果・監査](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/README.md)。
結果を保存し、以下の終了・保全照合を完了した。公開処理後mandatory STOPを維持する。

## grant発行前の技術回復

最初のproduction SOURCE bbb29c868e7464e729253fe6881d341c55be96b7では、旧C runtimeの
システムlibz.so.1.3 SHAと現在値の相違を検出して、grant発行前にprepareを停止した。
実H4状態作用0・grant発行/消費0、run01未作成。変更主体や時点はUNKNOWNであり、共有ファイルは変更しない。
pre_grant_runtime_diagnosis.jsonとpre_grant_STOP.jsonに旧SOURCEと失敗を保存する。
修正ではFLINT数値packageの旧版一致を維持し、現在のsystem runtime差を明示記録して新SOURCE/新sealへ固定する。
runtimeに関する回帰を追加し11件PASS。既存の旧SOURCE/runtime/STOPは書き換えず、新しい未使用grantだけを発行する。
科学的条件・許可call数・精度scheduleは変更しない。

旧SOURCE/setup STOPを保全し、新SOURCE92a5f104aでruntime197ファイルを固定した。
Python3.12.3/python-flint0.9.0/FLINT3.6.0、数値package103ファイルの旧hash一致。
[runtime](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/runtime.json)、
[grant前診断](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/pre_grant_runtime_diagnosis.json)、
[旧setup STOP](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/pre_grant_STOP.json)。
科学失敗0、科学retry0、成功済み状態作用の再実行0。

## Call・native終了・保全の照合

新grant ID：`13588ee95bcf7431f83df4f3880b7845b36f67181474a96d58ea5726b7646ab0`。
[grant](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/grant.json)を1回消費し、
[ledger](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/call_ledger.json)に作用前の非返還予約を保存した。
compiled64=1/residual64=1、96/128の全call=0、新C-target/input作用=0。
PR/sine/DF/compile/sampling/Hamiltonian/state生成/N_G/R2/正式受理=0。

worker652280は2026-10-25 00:24:17 JSTにnative exit0/reap、controllerもtoolでexit0を確認。
[worker receipt](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/native/worker_native_exit.json)、
[native STOP](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/native/STOP.json)、
[controller終了](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/controller_native_exit.json)、
[今回STOP](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/STOP.json)。
今回の証拠を失われた元H4 native PASSの代用にはしない。

SOURCE19/runtime197/input9・旧証拠412 SHA・旧manifest186行を照合、
[保存値監査65項目](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/verification.json)PASS。
[監査器](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/verify_saved_results.py)はstdlib/Fraction/SHAだけを使い、Arb/Qiskit/科学runnerをimportしない。
全検証は主agent本人が実施し、別reviewerの独立reviewとは呼ばない。

## 結論の範囲・次に必要な承認

固定PF cosineの局所数値上界は取得した。numerical_local_budget_satisfied=trueだが、
formal_certificate_accepted=false/LOCAL_BUDGET_MET=false。後者は正式gate受理を実施していない状態。
旧N/G=null、元native UNKNOWN、旧compiler証拠不足、C/L COMBINATION_BLOCKED、PR全lawρ/費用母平均tailは未解決のまま。
PF sineや他方式・trajectory・軸へ外挿しない。

PR検証には、経験的pointしかない**個別PR C-targetの包含**と、その保存compiled回路の包含が別途必要。
次の担当：**ユーザー**がPR0/PR1などの対象・cosine/sine軸・C-target/compiled作用・精度/call上限を明示承認する。
**Codex**は承認後に別SOURCE/契約/未使用grantで実装・人工検証・限定評価・終了監査を行う。
PR全law/R2/tail/N_G/正式受理まで拡張するなら各範囲の別承認が必要。現在はいずれも未実行。

## Commit・公開状態

科学SOURCEと結果commitを区別し、結果commit・push試行・remote SHAの時点状態を
[publication_status.json](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/publication_status.json)に保存する。
認証失敗ならlocal commitへ全結果を保全し、最終回答で再pushコマンドを示す。mandatory STOP。
