# H4 Pauli PR1 sine Arb包含評価（2026-10-25）

## 固定対象・承認

基点 `5c549779536bc272eb500d126a94845ae2bb1d6b`、専用branch `codex/non-df-h4-central-arb-pr1-sine-20261025`。
ユーザーはPR1 sineの実装・人工検証・限定Arb計算・保存監査を一括で明示承認した。
[承認記録](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/approval.md)、[GitHub基点確認](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/remote_base.json)。

固定linear H4／隣接1.00 Å／STO-3G、Pauli-JW184非identity項／retained prefix27、S2／T=.8／q8／r4／K4、system8＋ancilla8。
trajectory `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa` の**sineだけ**。
C SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`、D2 `74515d8eb4f073c40fc783681a3775dbca89e9b1`、A_GATE `ee7d281b5daead1d0d5e81e00c9a0ed4727792b0`、PR1 cosine Arb SOURCE `4a14964310f6cca4f824d22c4dbae41f2ae73b34` を参照。
旧SOURCE/code・cosine結果・grant/ledger/STOPは編集しない。

## 虚部・測定軸の包含接続

既存selected C-targetのunitary伝播・exact operand・Pauli permutation・L2 recenterを再利用し、最後の`dot.real`だけを`dot.imag`へ変更するscope adapterを追加した。
`dot = 〈raw|U_selected|raw〉`、`m_C_selected_sine = Im(dot)/ν`。
exact raw braに対するCauchy bound `|〈raw|x−center〉| ≤ sqrt(ν)e` は虚部にも適用できるため、既存の最終誤差転送をそのまま使う。
stateの再正規化、出力norm除算、clippingは0。νは保存binary64複素値からのexact有理normで共通化する。

32eventのposition30／outer7／inner2だけがorder2。
product43 `IYXXYIII` (+1) → product4 `IIIZIIII` (−1) → rotation72 `YYIIIIXX` (−1)、θは保存binary64値 `.029672025283184623`、order phase−1を一度だけ適用。
retained角はexact係数×delta/2、constant phaseは各outer stepのreverse retained後。元C `phase(φ)=exp(+iφ)` とexact scalar operandを保持する。
source aggregate phase `.26518225073344787`、controlledの相対phase転送、保存compiledのglobal phaseを保持し、PR0の近似値や新しいphase規約へ置換しない。

実保存sine compiled QPYは `artifacts/non_df_h4_a_gate_order2/2026-10-24/run01/variants/sine/output.qpy`。
SHA256 `38a1a74afe2452685c3e4dfdeb6182085b6d6a28d6c9d90ed1ab2f4026f7abdd`、fingerprint `88287619a8c27fa8370115ae33fd7f945984884391c6824a56b18490f81d564d`。
既存Arb native bridgeで`rz/sx/x/cx`・binary64 params・global phase・little endian・ancilla8→clbit0をそのまま評価する。
既存R1 `validate_sine`を使い、保存sine inputが同じcosine inputの**最終H直前へSdgを一つ追加した列**であることを検査し、+Im規約を結ぶ。
元PR1 source/control/cosineの静的照合4 QPYとsine input/output2 QPYの計6decodeを行ったが、cosine状態作用は0。
PF/PR0のQPY・科学結果は評価しない。

`ρ_local = B_saved × |m_compiled_sine − m_C_selected_sine|`。
`B_saved = 1449450801590761/1125899906842624`をwhole区間差へ**一度だけ**適用し、C-targetは非scale。
law meanや過去のbinary64点値はtargetに使わない。負下端のraw Arb ballが出た場合はraw ballを保全し、ρ≥0との共通部分を別fieldで示す。

## 再利用・人工検証・実行契約

旧PR0/PR1 codeは変更せず、metadata/authority/native関数をASTのscope label変更で再利用する。
数値差分はselected内積の虚部選択だけ。compiled kernel、whole interval union、B尺度、有限precision schedule、非返還call予約、native wait/reapは共通である。
[再利用記録](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/reuse.json)。

**43人工テストPASS**（新sine7＋既存authority11＋compiled10＋sine9＋QPY6）。
2〜3qubitのorder2付き32eventで、元C primitiveの256bit dense imag参照を64/96/128bit区間が包含。
純位相で+sinφが入り、−Im/実部が入らないこと、Sdgを逆符号にすると局所差が予算超過になることを人工native pairで確認した。
同じraw ν・B一回・input非改変・zero norm拒否、誤ったproduct/sign/order phase/angle、cosine予算・軸・trajectoryの拒否、grant排他・早期STOP・native回収を検査した。
既存sine testsはwrong S/missing Sdg/wire/phase拒否とcontrolを検査する。新compileは人工を含め0。
確認は主agent自身によるもので、別reviewerによる独立reviewではない。

新SOURCE/runtime/入力/active contractを固定して新one-shot grantを発行する。
64→96→128bit、各precision C-target1＋compiled1、初回最大6状態作用、区間縮約最大3。失敗callも非返還。
保存sine予算 **`102330934580088554157/30525376945269386661330944`**（約3.3523233722408497e-6）を使用し、cosine予算を流用しない。
上端≤予算／下端>予算で即STOP、未決の場合だけ昇精度、128でも未決ならINCONCLUSIVE。
幅1e-8は参考であり、決定後の高精度化は禁止。
独立2作用はprocess-local Arb contextで並行し、precision間は判定への依存で逐次実行する。
trig cache/Pauli permutation/raw入力共有を利用し、CPU/RAM/workerの恣意的上限・共有設定変更・他processへの操作は0。

## 実行結果・終了監査

科学SOURCE **`473cc930c9d1a13d3a7662b6ad0b996f1e60517a`**、新one-shot grant **`a714406d159973472eddc5c2f62de2c1ed6839efd2bae93526e9c4755deb3014`** を一回発行・消費した。
[封印](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/seal.json)、[active contract](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/active_contract.json)、
[保存結果](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/run01/result_summary.json)。

**64bitで上端≤保存sine予算が確定。状態作用試行/成功2（C-target虚部1＋compiled sine1）、残差区間縮約1。96/128bitは未使用。**
科学失敗・技術retry・追加grantは0。不要な高精度化はしなかった。

| 保存量 | 包含区間（十進は外向き24桁表示） | exact区間幅 |
|---|---|---|
| PR1 selected C-target sine | [-0.0237293346657786360672622, -0.0237293346657784497064612] | 13751/73786976294838206464 ≈1.86361e-16 |
| 実保存compiled sine | [-0.0237293346657789416835258, -0.0237293346657782549295409] | 101347/147573952589676412928 ≈6.86754e-16 |
| raw Arb ρ ball | [-4.90665622449783279945845e-16, 6.33356970130103477878240e-16] | 23345035905498873857/20769187434139310514121985316880384 ≈1.12402e-15 |

ρ≥0との共通部分は **`[0,13154309625350691783/20769187434139310514121985316880384]`**。
よって **ρ_local ≤6.33356970130103477878240e-16 < 保存sine予算約3.3523233722408497e-6**。
raw ballと非負共通部分を別々に保存した。0は新ballとの数学的交差で、ρ=0の証明や旧nullの読み替えではない。
[保存ρ](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/run01/rho_64.json)。
幅1e-8も満たすが、停止理由は保存予算との関係が包含区間から確定したこと。

C-targetはretained432 rotation・32event（Pauli product2）・scalar8回で約0.63821秒。
compiledは3042 native gate（保存3043命令から最後のmeasurement1だけを除外）で約2.67176秒。
2独立processは重なって実行。raw ray hash `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80` とexact νを共用し、B_savedは一度だけ。
符号付き信号が負であることは+Im規約と整合しており、−Imへ読み替えていない。

保存source aggregate phase `.26518225073344787`、sine compiled global phase **2.4887856155591095**、ancilla8の相対phaseを保持。
**今回の固定scopeで、order2の位相・符号・Sdg/+Im測定軸に不一致を検出しなかった。**
人工包含検証とSOURCE/QPY照合・局所ρ比較の結果であり、全compiler/全trajectoryのcertificateではない。
[構造/phase照合](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/structural_inspection.json)、
[人工43件](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/artificial_verification.json)。

親worker PID676190と2childはnative exit0・回収。残留子孫の強制終了0、all_owned_reaped=true。
[非返還ledger](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/run01/call_ledger.json)、
[target native](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/run01/target_64_native_exit.json)、
[compiled native](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/run01/compiled_64_native_exit.json)、
[親native](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/run01/native/worker_native_exit.json)、
[最終STOP](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/run01/STOP.json)。
UTC2026-10-24T17:04台はJST2026-10-25T02:04台。

code36/runtime211（FLINT数値package103）/実入力23を新SOURCE・封印へ固定した。
既存private Python-FLINT0.9.0/FLINT3.6.0を再利用。旧C runtimeとの既知libz.so.1.3 SHA差は原因UNKNOWNとして観測し、今回の実runtimeを固定した。FLINT packageは同一、共有library/package変更0。
旧証拠654file・旧manifest190行と旧SOURCE/results/grants/ledger/markers/STOPを保全した。
[stdlibだけの保存検証](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/verify_saved_results.py)、
[検証結果](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/verification.json)。
保存検証129項目PASS。後検証の状態作用・QPY再decode・保存vectorからのsignal再縮約は0。

## 残る課題・次の承認

局所数値ρを正式certificate受理やPR全lawのρへ昇格せず、`LOCAL_BUDGET_MET=false`を保持する。
旧N_G=null、C/L重複、旧compiler証拠不足、母平均tail、元H4 native UNKNOWNは解消済みにしない。
今回の禁止範囲はcosine/PF/PR0再計算、追加trajectory、新compile/sampling/Hamiltonian/state生成、R2/N_G再解析、正式受理。
最新版policyの実在する§8.4/10/10.1/10.2を適用し、§10.3が提供版にない版差を記録した。直接ユーザー指示を優先する。

選択3対象の両軸局所保証を揃えるには、PF/PR0 sineにも進む価値がある。PR1の結果からそれらの上界を推定しない。
次は**ユーザー**がPF sine／PR0 sineの対象・precision・callを別承認し、**Codex**が保存PF imag区間またはPR0個別imag包含との接続・人工検証を行って指定範囲だけ評価する。
PFは既存C imag区間との適合が確認できれば新target作用0＋compiled最大3、PR0はtarget/compiled各precision1で最大6が最小案。いずれも新SOURCE/runtime/入力/契約/未使用grantが必要。
今回は準備・実行しない。結果保存・commit/push/remote確認後mandatory STOP。

## Commit・push・remote状態

科学SOURCE `473cc930c9d1a13d3a7662b6ad0b996f1e60517a`、結果commit `3745e63e9c674d69bfc9b6e217766522f3ca78e2`。
origin専用branchへのpush1回はHTTPS Username取得不能でexit128。
GitHub connectorの当該branch検索は0件、remote SHA=null。基点5c5497795…の実取得とは区別する。
[公開状態](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/publication_status.json)、
[push失敗log](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/push_attempt01.log)、
[remote再確認](../artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/remote_after_push.json)。
後続local metadata commitは公開状態だけを保存し、科学SOURCE/runtime/入力/区間/結果/grant/ledger/STOPを変更しない。

ユーザーが認証可能な環境で保存commandを再pushする。次の科学段階はPF/PR0 sineの具体的な別承認待ち。
Codexはmandatory STOPし、追加対象・正式受理・N/Gへ自動移行しない。
