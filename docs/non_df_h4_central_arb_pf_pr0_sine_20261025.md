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

科学SOURCE **`1e6a702f51c2ba7bb1577610919b99cb9990f8d6`**、root grant **`b79b1eab4fcbcc799c544d8774e1f244ff1e812b2d4fd93fcf4d7563adc4a7bd`**。発行1・消費1、技術的な科学失敗/回復0。
SOURCE/code40、runtime211、入力38をsealし、親worker・case PF/PR0・PR0内target/compiledのnative exit0/join/reapとSTOPを保存した。
PF compiled約48.320秒、PR0 target約0.626秒/compiled約2.679秒（独立並行）。
**両対象64bitでUPPER_WITHIN_BUDGET。総試行/成功3、縮約2、96/128bit作用0。PF target作用0。**

| 対象 | C-target sine包含区間（外向き表示） | compiled sine包含区間（外向き表示） |
|---|---|---|
| PF | [0.986870090295708622377272, 0.986870090295715194919262] | [0.986870090295702870710533, 0.986870090295714798127054] |
| PR0 | [0.900307652047570361366745, 0.900307652047570548391621] | [0.900307652047569947526776, 0.900307652047570692807350] |

- PF局所ρ非負区間：`[0,465594429/37778931862957161709568]`、上端 **≤1.23241819194079088800945e-14**。
- PR0局所ρ非負区間：`[0,16065712728141281981/20769187434139310514121985316880384]`、上端 **≤7.73535930526261160074741e-16**。

両方とも上表の保存exact予算との有理数比較で予算内。非負区間の下端0は新しいArb ballと数学的非負性の共通部分であり、旧nullや実装誤差そのものを0と判定したものではない。
raw Arb ballの負下端も変更せず保存した。PF rawρ区間幅約1.8500e-14、compiled幅約1.1928e-14。
PR0 rawρ幅約1.2003e-15、compiled幅約7.4529e-16。参考幅1e-8のための追加callなし。

PFは旧実装通り、保存C全区間と**内部Arb signal ball**を直接減算する。
JSON signal端点は`h4_ball_evidence.endpoints`の`lower()/upper()`でさらに外向きに丸めるため、JSON区間全体同士の直積差は内部ballによるρと同一ではない。
保存端点だけの有理数照合では保守的上端約1.2324208729e-14となり、これも同じ保存予算内。
初回の監査条件「ρがJSON直積差の全点を包含する」はPFの実経路より強すぎたため、内部ball経路のSOURCE照合・保存端点の整合性・独立した保存端点上端の予算確認に訂正した。
科学SOURCE/値/allowance/閾値の変更、科学再実行は0。PR0は保存端点を再度外向きunionして縮約するため、保存直積差全体の包含も直接確認できた。

### 全6件の局所包含結果

| 対象 | 軸 | 局所ρ上端（外向き表示） | 保存予算（概数） | bit | 状態作用 | 来歴 |
|---|---|---:|---:|---:|---:|---|
| PF | cosine | 9.56489249909976322304440E-15 | 0.00000264900383383235944762061 | 64 | 1 | 保存値引用 |
| PF | sine | 1.23241819194079088800945E-14 | 0.00000336139425244868746478033 | 64 | 1 | 今回 |
| PR0 | cosine | 6.79574416339626407633468E-16 | 0.00000259361136881980481818957 | 64 | 2 | 保存値引用 |
| PR0 | sine | 7.73535930526261160074741E-16 | 0.00000335232337224084958653573 | 64 | 2 | 今回 |
| PR1 | cosine | 6.27209038472283370091352E-16 | 0.00000259361136881980481818957 | 64 | 2 | 保存値引用 |
| PR1 | sine | 6.33356970130103477878240E-16 | 0.00000335232337224084958653573 | 64 | 2 | 保存値引用 |

全6件が保存予算内の局所上界を持つ。今回の新評価はPF sineとPR0 sineだけで、他4件の区間・SOURCE・成功call・STOPは保存値の引用。
既存4件を合算した新科学評価やPR law平均/資源順位再解析は行っていない。
[exact全6件一覧](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/six_local_results.json)、[CSV](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/six_local_results.csv)。

### 保存監査

126照合PASS。新SOURCE/runtime/入力、exact共通ν、PFのC虚部全区間/target作用0、PR0個別32event・B一回、sine QPY/+Im、precision/call/早期停止、native PID/exit/reap/STOPを確認。
旧証拠723ファイル・旧manifest191行・既存4件の重要24ファイルをSHA照合し保全。新manifest行だけ追加。
37人工テストlogのSHAも照合し、共有設定/他process変更0。科学nativeの失敗なし。
[監査結果](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/verification.json)、[raw結果](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/run01/result_summary.json)、[native STOP](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/run01/STOP.json)、[表示区間](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/reported_intervals.json)。

`numerical_local_budget_satisfied=true`は今回の包含区間による比較。正式受理/`LOCAL_BUDGET_MET=false`、`PR_law_rho=null`、旧N/G=null、元H4 native UNKNOWNを保持。
今回のnative終了を元H4の失われた終了証拠へ代用しない。全law/費用tail/C-L/正式certificateの問題は独立に残る。


## 保全・残る課題・次の承認

旧SOURCE/results/grants/ledger/markers/STOPと旧manifest行を保持し、実行gateは実際に使用するcode/runtime/入力/契約に限定する。
提供policy§8.4/10/10.1/10.2を適用し、§10.3なしの版差を記録。直接ユーザー指示を優先する。
6局所結果が揃ってもPR全lawρ・正式certificate受理・母平均tail・旧N_G null・C/L・元H4 native UNKNOWNを解消済みにしない。

PR全lawの保証へ進むには、選択PR0/PR1の局所値を未観測order4を含む全eventへ転送するR2が必要。
次は**ユーザー**がR2の静的設計（全lawへのuniform/正当なlaw-weighted上界、compiler/literal転送のscope、未観測event、最小診断契約案。科学call0）を別承認し、**Codex**が承認後に設計する。
新科学診断・compile/sampling・正式受理・N_Gは、その設計と固定契約を確認した別承認後だけ実行する。今回R2へ自動移行しない。
結果保存・commit/push/remote確認後mandatory STOP。

## Git保存・公開状態

科学SOURCE `1e6a702f51c2ba7bb1577610919b99cb9990f8d6`、専用branch `codex/non-df-h4-central-arb-pf-pr0-sine-20261025`。
結果commit後にpushし、GitHubで実際のbranch SHAを照合する。公開成否は`publication_status.json`へ保存する。
実行namespaceは既にmandatory STOP・再launch禁止。
