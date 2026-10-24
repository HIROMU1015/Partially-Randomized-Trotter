# H4 Pauli central PF・PR0 sine Arb包含評価（2026-10-25）

## 対象・承認・固定条件

基点 `06ca7c838808cbde9e1c1b5030c46f260d929c32`、branch `codex/non-df-h4-central-arb-pf-pr0-sine-20261025`。
PF sineとPR index0 sineの実装・人工検証・限定数値評価・終了監査をユーザーが一括承認した。
[承認](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/approval.md)、[GitHub基点](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/remote_base.json)。

固定linear H4／隣接1.00 Å／STO-3G、Pauli-JW184非identity項／PR retained prefix27、S2／T=.8／q8／r4／K4、system8＋ancilla8。
PF trajectory `d761627d56a4454e644358fef8907b8a7db28c4feeb44130224c20e4e50aae4d`、
PR0 trajectory `93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97`。
PF・PR0 sineだけを新評価し、既存cosine/PR1の4局所値は引用だけで一覧へ接続する。
R2/N_G再解析・正式certificate受理は承認範囲外。

## 再利用と包含規約

旧PF/PR0 cosine、PR1 sine、CのArb primitive/compiled native kernel・有限precision scheduleを再利用し、旧codeは変更しない。
新scope adapterはsineのlabel/予算/QPY/委譲markerと最終`dot.imag`だけを接続する。
PFは保存`implemented_law.PF.corrected_encoded_mean.imag`の**包含区間全体**を共通exact νで割り、target状態作用0。
PR0は保存32event（全order0）・retainedのexact係数×delta/2・binary64 event角・scalar phaseを順番通り作用し、`Im〈raw|U_selected|raw〉/ν`を包含する。
経験的点値やPR law meanはPR0のtargetに使わない。

raw ray hash `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`、同じexact raw νを使用する。
再正規化、出力normによる除算、clippingは0。Cのexact変換・unitary L2 recenterと、exact-bra Cauchy転送/ancilla-Z誤差転送を保持。
`ρ = B_saved × |m_compiled_sine − m_C_sine|`、BはPF1、PR0 `1449450801590761/1125899906842624` を一度だけ適用する。
raw Arb ρ ballと、数学的非負性との共通部分は区別して保存/表示する。

実保存sine QPYはA_AXES SOURCE `a37232f85812688c3b9ab957c5073c5ce143b17e`、source/controlはD2、C SOURCEは `e80d7510880f8db993b777c627bb8e95b05a070d`。
PF compiled SHA `a478034cc3bcafa4e3d85e69739ad92a997d5073b5db3e452d2d0030542549b1`、
PR0 compiled SHA `5228f426d532ffd04f973b19fb65162c10eceb064f555b46c06af337f0106c0f`。
既存sine静的preflightの対象をPF/PR0の2件へ限定し、empirical cache段階を省略して再利用した。
source/control/cosine input/sine input/compiledの各5 QPY、計10を読取り、SOURCE recipe・hash/fingerprint・Sdg-before-final-H/+Im・global/relative phase・little endian・wire8→clbit0を確認した。状態作用0。
新回路生成/compile/sampling/Hamiltonian/state生成は0。

## 人工検証・one-shot制御

**37人工テストPASS**（新12＋native Arb10＋既存sine9＋QPY6）。
PFでwhole imag区間をnorm変換しmidpoint/実部へ置換しないこと、PR0の2〜3qubit imagを256bit dense参照が包含すること、Sdg逆符号の検出、B一回/共通normを確認した。
各対象の独立精度schedule・上限3/6/計9・未決時だけ昇精度・非返還call・一回だけの親grant消費・子の委譲・STOP拒否・native join/reapを検証。
検査は主agent自身によるもので、別reviewerの独立reviewではない。
[人工結果](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/artificial_verification.json)、[再利用](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/reuse.json)。

新SOURCE/runtime/入力/active contractを固定し、2対象を含む新root one-shot grantを一回発行・消費する。
子targetは`PARENT_AUTHORIZATION`で消費済みroot grantを参照するだけで、子grantや新消費receiptを発行しない。
PFのcompiled作用と、PR0のtarget/compiled作用は独立processで並行する。precision間は各対象のSTOP判定に依存する。
各caseのcall ledgerと排他的claimで非返還上限を守り、合計は最大9。旧grant/ledger/STOPを再利用しない。
恣意的なCPU/RAM/worker/affinity上限、PSI競合だけによる強制中断、共有設定や他process変更は0。

| 対象 | 最大状態作用 | 最大縮約 | 保存sine予算exact |
|---|---:|---:|---|
| PF | compiled各precision1・最大3、target0 | 3 | 25651956654414547227/7631344236317346665332736 |
| PR0 | target1＋compiled1/precision・最大6 | 3 | 102330934580088554157/30525376945269386661330944 |

64→96→128bit、上端≤予算または下端>予算でその対象は直ちに終了。未決時だけ昇精度し、128でも未決ならINCONCLUSIVE。
幅1e-8は参考で、判定後の高精度化は禁止。意図しない技術中断は失敗記録保全・新SOURCE/未使用grantによる限定回復だけ許可し、成功作用を不要に再実行しない。

## 新2件の結果・全6局所一覧・終了監査

実行後にSOURCE/grant/区間/上界/予算比較/precision/call/native終了を追記する。
この準備時点では実H4科学作用・科学grant発行/消費0。

## 保全・残る課題・次の承認

旧SOURCE/results/grants/ledger/markers/STOPと旧manifest行を保持し、実行gateは実際に使用するcode/runtime/入力/契約に限定する。
提供policy§8.4/10/10.1/10.2を適用し、§10.3なしの版差を記録。直接ユーザー指示を優先する。
6局所結果が揃ってもPR全lawρ・正式certificate受理・母平均tail・旧N_G null・C/L・元H4 native UNKNOWNを解消済みにしない。

PR全lawの保証へ進むには、選択PR0/PR1の局所値を未観測order4を含む全eventへ転送するR2が必要。
次は**ユーザー**がR2の静的設計（全lawへのuniform/正当なlaw-weighted上界、compiler/literal転送のscope、未観測event、最小診断契約案。科学call0）を別承認し、**Codex**が承認後に設計する。
新科学診断・compile/sampling・正式受理・N_Gは、その設計と固定契約を確認した別承認後だけ実行する。今回R2へ自動移行しない。
結果保存・commit/push/remote確認後mandatory STOP。
