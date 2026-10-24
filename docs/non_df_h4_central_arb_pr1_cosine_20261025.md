# H4 Pauli PR1 cosine Arb包含評価（2026-10-25）

## 固定対象と承認

基点 `e90d6cfed1721b59c33c22bb146aebcc1fea3973`、branch `codex/non-df-h4-central-arb-pr1-cosine-20261025`。
ユーザーはPR index1 cosineのArb包含評価を明示承認した。
[承認](../artifacts/non_df_h4_central_arb_pr1_cosine/2026-10-25/approval.md)、[GitHub基点確認](../artifacts/non_df_h4_central_arb_pr1_cosine/2026-10-25/remote_base.json)。

固定linear H4／隣接1.00 Å／STO-3G、Pauli-JW非identity184項／retained prefix27、S2／T=.8／q8／r4／K4、system8＋ancilla8。
trajectory `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa` だけ。
C SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`、D2 `74515d8eb4f073c40fc783681a3775dbca89e9b1`、PR1保存QPYのA_GATE SOURCE `ee7d281b5daead1d0d5e81e00c9a0ed4727792b0` を参照する。
PR0/PF/sine/追加trajectoryの科学計算は行わない。

## Order2の接続と再利用

32eventのposition30（outer7/inner2）だけがorder2。他31eventはorder0。

| 順番 | role/ID | 保存Pauli | sign |
|---|---|---|---:|
| 0 | product43 | IYXXYIII | +1 |
| 1 | product4 | IIIZIIII | −1 |
| 2 | rotation72 | YYIIIIXX | −1 |

θは保存binary64値 `0.029672025283184623`、order phase−1。
unsigned P43/P4/P72で `U30 = −[cosθ I − i sinθ (−P72)] (−P4) (+P43)`。
vector作用は43→signed4→signed rotation72の順で、order phaseを一度だけ適用する。
constant scalar phaseは各outer stepのreverse retained後。retained角はexactなbinary64係数×binary64 delta/2であり、一度floatへ丸めた積にしない。

保存sourceのaggregate phase **0.26518225073344787**、compiled global phase **3.2741837789565587** を保持する。
前者をPR0の近い値へ置換せず、C-targetはCに定義されたexact scalar operandを毎step用いる。
signed productとorder phaseを相殺した別の所有者へ移さず、保存appendのπ/constant更新順・control相対phaseとの対応を既存R1 preflightで検査する。

旧PR0 bridge・production・controller・R1 sourceを一切編集せず、新adapterがASTで対象label/index/order-profileだけを変更して同じ関数を接続する。
selected数値カーネルの算術式は無変更（domainの監査labelだけPR1へ変更）。
Pauli permutation、exact C/Arb変換、unitary L2 recenter、exact-bra内積の誤差転送、compiled native gate作用、精度schedule、nonrefundable予約、native wait/reapを再利用した。
新しいcodeはPR1の固定ID/hash/labels/signs/profile検証と入出力・実行承認の接続である。
[再利用記録](../artifacts/non_df_h4_central_arb_pr1_cosine/2026-10-25/reuse.json)。

raw rayは保存binary64複素ベクトルをexact入力とし、共通exact νでC-targetとcompiledを正規化する。
selected C-targetは`Re〈raw|U_selected|raw〉/ν`、compiledは保存Hadamard回路のancilla-Z/ν。
`rz/sx/x/cx`・原global/relative phase・little endian・ancilla8→clbit0を保持する。
C-targetはsqrt(ν)×state-L2-error、compiledは2||center||e+e²の転送を使う。
state再正規化、出力normによる除算、clipping、PR law mean／経験点値のtarget代用は0。

`ρ_local = B_saved × |m_compiled − m_C_selected|`、
`B_saved = 1449450801590761/1125899906842624` をwhole区間差へ**一度だけ**適用する。
各precisionの独立2作用はprocessごとのArb contextで並行。precision間は早期判定への依存がある。
trig cache/Pauli permutation/raw入力共有を再利用し、恣意的なCPU/RAM/worker制限や共有環境変更はしない。

## 人工検証と科学契約

**56人工テストPASS**：PR1新7＋再利用production13＋selected13＋compiled10＋QPY6＋R1 order2 7。
2〜3qubit系で、保存ID43/4/72と同じ非可換署名・符号・angle・32event profileを持つ人工scheduleを評価し、元C primitiveによる256bit dense参照を64/96/128bitで包含した。
逆product、負符号欠落、rotation sign、order phaseの誤りは解析的counterexampleまたはmetadata拒否で検出できる。
旧R1テストは元append/event_unitary・control両branchを検査し、新productionテストは共通ν/B一回・grant排他・並行2childのnative回収・早期STOP/非有限を確認する。
保存実QPY4個は静的decodeだけでSOURCE tokens/control/phase履歴/fingerprintと照合した。ここまで実H4状態作用0。
テストと照合は主agent本人によるもので、別reviewerの独立reviewとはしない。

新SOURCE/runtime/入力/active contractを封印して未使用one-shot grantを発行する。
64→96→128bit、各precision C-target1＋compiled1、最大6状態作用・区間縮約最大3。失敗callも非返還。
保存予算 `19792741170690161541/7631344236317346665332736`（約2.593611368819805e-6）に対し、上端≤予算または下端>予算で即STOP。
判定不能時だけ昇精度し、128でも未決ならINCONCLUSIVE。幅1e-8は参考で、決定後の高精度化はしない。
意図しない技術中断時は失敗記録を保全し、新SOURCE/未使用grantでの限定回復だけを許可する。成功済み計算の不要な再実行はしない。

## 実行結果・監査

この節へ実行後のSOURCE/grant/信号区間/ρ/予算比較/precision/call/native終了を追記する。
実行前の時点では実H4区間は未取得である。

## 範囲・保全・次段階

局所数値ρと正式certificate受理を区別し、`LOCAL_BUDGET_MET=false`、PR全lawρ/母平均tail/旧N_G=null/C-L/旧compiler証拠不足/元native UNKNOWNを維持する。
旧SOURCE/results/grants/ledger/markers/STOPを保全し、worker取消gateは実際に使用するcode/runtime/入力/契約だけへ限定する。
最新policyの§8.4/10/10.1/10.2を適用し、提供版に§10.3がない版差を記録した。直接ユーザー指示を優先する。

次は**ユーザー**がsine包含評価の対象/精度/callを別承認する。
**Codex**は承認後に測定Sdg/+Im・個別C-target imag転送の適合を準備・人工検証し、指定範囲の数値評価を行う。
sineは実部だけでは確認できない位相・測定軸のcoverageを増やす価値があるが、PR全law保証や正式受理を自動的に満たさない。
結果保存・commit/push/remote確認後mandatory STOP。
