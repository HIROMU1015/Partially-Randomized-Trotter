# Mechanism Analysis Contract

状態：**DRAFT_UNAUTHORIZED**。全系のFigure 2は**PENDING_NUMERICAL_AUTHORIZATION**。
今回の作業は契約の準備と人工テストまで。以下の算術を実科学値へ適用していない。実行可能な科学runner、有効grant、起動commandは作成していない。

## 1. 固定対象とSOURCE

[mechanism_targets_DRAFT.csv](mechanism_targets_DRAFT.csv)に保存comparisonの候補IDをそのまま固定した。主系列120比較×新／旧PRの240 contrastと、H8 n48の主副4 contrast、計244枠。これは保存済み選択の列挙であり新しい最小化ではない。新A倍率10欠測等の枠も残し、未確定には数値を出さない。

主図対象はH4/H6/H8×主要5ε×新A探索n8／旧PR n32の30 contrast。H8固定n48主2条件は確認panelへ別置。その他の会計・倍率・旧比較集合は補足。H6高次PF独立確認6e197973…はこの契約外。

共通：linear H-chain、1Å、STO-3G、T=.8、P=0、保存Hamiltonian/DF/reference、full Hadamard wrapper、compiled RZ、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/couplingなし。H4 legacy DF12＋測定subset16、H6 tol-only DF19＋PF136、H8 tol-only DF28＋PF184の差を保持。H8の原profile bytes・旧compiler output完全同一性は未確認。

固定公開入力：

| alias | commit | repository file |
|---|---|---|
| NEW_COMPARISONS / NEW_POINTS |64c85f7a833b5950ef2ccae623708b73081edcd6|track_a_pr_new_a_total_cost_results/2026-10-18/comparisons.csv / new_pr_cost_points.csv|
| NEW_AXES / REGISTRATION / OLD_D_SUMMARY |fc6bd039dc7171d5674ac04f0b3ad427f1b6495b|track_a_pr_cost_batch_results/2026-10-18/a_cost_measured_axes.csv / a_candidates.json / d_saved_comparison_summary.json|
| BOUNDARY / H6_LINEAGE |bf8c19057fd82c900c6234152e152e79cb52e707|track_a_final_technical_audit/2026-10-19/B_domain/* / D_h6_lineage/*|
| H8_CONFIRM_COMPARISONS / H8_CONFIRM_AXES |e6eea5387a6e84d1e39a9adb3b61975acc74e200|track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv / paired_metrics.csv|

表のfile prefixはartifacts/resource_applicability/。公開file SHA256はmanifestの各出典欄、private byte鎖と全aliasは内部`private_input_lock.json`／`mechanism_input_binding_internal.json`に固定する。internal lock自体のSHAとdoc／targets／人工テストSOURCEのSHAは内部最終freeze記録に残す。

新A保存会計SOURCE8601b921342fc1cc83780a830e1f65bf64fe9e93、取得31ef62a1…／H8継続87f1f0d5…。旧Dは31ef62a1…／H8修正8ff002dd…。H8確認runtime SOURCE745a674546f2cb6258526fbfde1d9a00601689f8、dataset/seed SOURCE1549b6f8…を区別する。

## 2. 値の取得元

| 値 | 新A n8 | 旧n32 / PF | H8 n48 |
|---|---|---|---|
| C_real/C_imag | NEW_AXESのcosine/sine RZ、内部A_result.paired_costs[].axes | 内部D_payload.paired_PR[candidate]のpaired軸／D_result.rows[].axis_costs。軸実測欠落は補完禁止 | H8_CONFIRM_AXESの同scope/candidate/replica両軸。PFは固定D_result軸証拠と元設計の直接単価 |
| N_real/N_imag | 内部NEW_A_RESULT.new_rows[].accounting.axes.real/imag.shots | 内部D_result.rows[].shots.cosine/sine、元accountingとの対応 | PRは公開比較表shots_cosine/shots_sine、PFは固定endpointのD_result.shots |
| normalization B/log_B | 内部A_signal.signals[]のb/log_B | 元の保存signal/元会計record。D_resultに無ければ欠測のまま | 元p7/p8 A_signal。通常と副評価の同一性は保存recordに従う |
| bias / allowance / headroom | A_signalのerror_decomposition等と保存precision_rowsまたはnoncancellation/empirical_noncancellation_precision_rows | 元保存accounting・signal。headroomからbiasを逆算しない | 同じ元signal/保存会計。副評価で固定PFが不適格なら比較null |
| G / SE照合対象 | NEW_COMPARISONS/NEW_POINTS | NEW_COMPARISONSの旧PR/PFと固定旧D結果 | H8_CONFIRM_COMPARISONSのmean_exact/paired_SE/fixed_PF_G_exact |

軸名はreal=cosine、imag=sine。標本対をseed/event/replica/cohortで同定する。H4のrawラベルCONFIRMATION_N32を科学上の独立確認へ格上げしない。新旧SOURCEの意味を保つ。

内部bindingでC/Nのfield存在・candidate/endpoint対応を記録する。存在確認だけでは数値の正しさ・G一致PASSにならない。不足している旧signal/headroomや軸来歴は`EXPLANATORY_INPUT_MISSING`または`CORE_INPUT_MISSING`とし、無断の新signal生成・G/Nからの軸単価推定は行わない。全244枠の数値完了を事前に約束しない。

## 3. 正確な費用分解

各方法mと軸aについて、保存整数shot N_m,aと直接単価／同じcohortの軸単価平均 C_m,aを使う。

G_m = N_m,real C_m,real + N_m,imag C_m,imag
R_G = G_PR / G_PF
R_N = (N_PR,real+N_PR,imag)/(N_PF,real+N_PF,imag)
C_eff,m = G_m/(N_m,real+N_m,imag)
R_Ceff = C_eff,PR/C_eff,PF
R_G = R_N × R_Ceff

C_effはそれぞれのshot構成で重み付けした単価であり、純粋なcompiler効果の因果推定ではない。軸ごとの回路比R_C,a=C_PR,a/C_PF,a、shot比R_N,a=N_PR,a/N_PF,aも保存する。

PF費用重みw_a=N_PF,a C_PF,a/G_PFを用いるもう一つの正確な表示：

R_G = Σ_a w_a R_N,a R_C,a,  Σ_a w_a=1

両軸の単純平均回路比×一つのheadroom比は採用しない。PR単価比が1未満とも仮定しない。別の重みを使う場合は別の表示であると明示し、上式を置き換えない。

## 4. normalization/headroomと丸め

固定参照は8601b921…:scripts/d0_frozen_shot_rule_v1.py（origin c31ef424…）。保存した非負axis bias b_m,a、数値allowance u_m,aに対しh_m,a=ε/√2−b_m,a−u_m,a。signed会計のbは保存signed成分の結合後の絶対biasであり、保存signed成分の符号を消して別会計にしない。

対応する参照式はN*_m,a = 2 B_m² log(2/α_m,a) / h_m,a²、整数化N=max(1,ceil(N*))。参照実装はbinary64の演算順序・normalization_upperの保存bitを保持し、log_n>=log(2^52)ではLOG_DOMAIN_ONLYを使う。**本契約の費用は保存整数Nを使い、shotの再最適化・再丸めはしない。**

保存入力が揃う軸だけで、説明用の連続比R_N*,a=(B_PR/B_PF)²×[log(2/α_PR,a)/log(2/α_PF,a)]×(h_PF,a/h_PR,a)²を別欄に算出する。κ_m,a=N_m,a/N*_m,aとするとR_N,a=R_N*,a×κ_PR,a/κ_PF,a。B共通軸という元契約の条件を明示し、軸別Bの記録がある別schemaを暗黙に共通化しない。

N*は保存log_shot_bound／保存boundを最優先。exp(log_N)で元binary64 boundを完全再現したと主張しない。元boundがない場合、binary64の固定順序を再現した説明値に限定し、整数ceilの新結果で保存Nを置き換えない。必要なB・headroom・αがない場合、R_G/R_N/R_Cのcoreと説明欠測を分ける。

## 5. exact算術と一致検査

整数RZ・shotは任意精度整数、modelの有理数単価・保存exact費用はFractionで扱う。binary64の保存値は元十進表記とfloat.hexを併記し、decimal文字列からの有理数と元binary64の有理数を混同しない。SEは元保存値を転記し、必要な標本計算を行う場合もpaired費用を先に作る。

受入条件：

1. 実際に使うSOURCE・input・compiler/runtime来歴・結果・新grant/ledger/markerのbyte一致。無関係worktreeのHEAD/statusは観測のみ。
2. 各採用候補の両軸・全保存replicaの一意性、同cohort・law・binding・seed/event対応。新旧n8/n32/n48をpoolしない。再利用された確認replica0も1回だけ数える。
3. 保存N_totalがある場合、両軸整数和と完全一致。新shotの計算はしない。
4. Σ_a N_a C_aによるG、PRのpaired整数費用平均は保存G_exactと有理数で**完全一致**。R_G=R_N R_Ceff=ΣwR_NR_Cも有理数で完全一致。保存削減率がある場合100(1−R_G)との完全一致を要求。
5. 保存がexactでなく丸め表示だけならexact一致PASSを出さずDISPLAY_ONLY_SOURCEとする。都合のよい誤差許容幅へ緩めない。
6. SEを再構成する承認が明記された場合のみ、同じpaired順序・n−1分散・SE=s/sqrt(n)を用い、保存二進浮動値との差をmax(1e−12,1e−12×abs(saved_SE))以下で検査。これは再現許容幅であり科学的CIではない。新たな差のSE・p値・CI・bootstrapは契約外。
7. normalization/headroomの参照式は保存binary64の順序とガードを維持。log bound比較はabs差<=1e−12、ceil境界をまたぐ不一致・欠測・非finiteはEXPLANATION_UNRESOLVED。coreのexact費用不一致を説明誤差で救済しない。

## 6. 欠測と解釈

eligible:null→UNRESOLVED、eligible:false→INELIGIBLE、入力不在→INPUT_MISSING。どれも0費用／0shot／Falseにまとめない。H6非相殺の旧12 B0I分解欠測はfull-domain比較null。旧D有限subsetは別の保存系列として引用し、そのsubsetから新winnerを選ばない。新A H6/H8 factor10の保存shot欠測は別の理由を維持する。

PF modelはVALIDATED_MODEL_ESTIMATEと原model SHA／適用範囲を保持。選択PFが直接実測でも比較集合全体の直接実測最良とは呼ばない。モデル残差を確率分布・SEへ混ぜない。

SEは有限trajectory費用標本のMCSE。探索選択効果や未観測tailへの保証ではない。H4 fixed anchor、H6/H8確認union内の事後選択、新A探索、H8事前固定n48を分離する。H8 q16は登録q/r scheduleの変更を含み、qだけの効果を主張しない。

## 7. 将来出力schema

mechanism_axes.csv：contrast_id、system/DF/cohort/ε/bias/factor、PR/PF_ID、axis、N_PR/N_PF、C_PR/C_PF、cost_class、R_N_axis/R_C_axis/w_PF、B/log_B/bias/allowance/headroom/α、optional_Nstar/κ、入力alias/pointer、status。

mechanism_totals.csv：contrast_id、G_PR/G_PF、Ntot_PR/Ntot_PF、Ceff_PR/Ceff_PF、R_G/R_N/R_Ceff、保存削減率／再構成値、G一致・恒等式検査、保存SE、modelフラグ、core_status、explanation_status、population_conclusion=UNRESOLVED。

verification.json：byte gate・exact一致・欠測伝播・pairing・許可境界・終了状態。ここに列挙した新しいderived numeric columnは**将来の出力仕様のみ**であり、今回その表や数値は生成しない。

## 8. GO / STOPと承認境界

現在approved=false、runtime_authorization=false、production_SOURCE=null、launch_command=null。人工テスト17件はPASSだが実データのG一致・数値解析は未実施。

GOにはユーザーの対象scope・内部input lock・この契約／targetsへの明示承認、実装SOURCEとruntimeの別seal、未使用output/grant/ledger/marker、freshなI/O/memory整合性の確認を要する。準備文書への承認を科学実行承認と読み替えない。実装SOURCE未固定の現時点は実行不可。

将来の許可候補はH4_MAIN、H6_MAIN、H8_MAIN、H8_CONFIRM_N48の各保存値解析1pass。call予算は新sampling/native/quantum回路生成/compile/signal/SCF/DF/state/fit/quantum shots/rare Tier R/非Clifford角分類/再ランキング=各0。保存値の新たな算術・明記したG一致検査だけを新しい承認対象とする。

C/N欠測のrowはCORE_INPUT_MISSINGとして数値を出さず、B/headroomのみ欠測ならcoreと説明を分ける。科学的不一致、実入力破損、native/IO異常、実OOMや継続不能は当該新解析scopeでSTOP。事前に記載した欠測を失敗や0値へ変えない。無関係worktree更新・一時的PSI/CPU競合を停止理由にしない。既存科学jobは操作しない。

小さいscalar処理は系別の独立process等で重複読込を避ける。CPU/RAM/workerの恣意的な恒常上限は設けず、利用可能資源に応じる。OS/cgroup/共有環境/他ユーザー/既存venvを変更しない。新one-shot記録は成功・失敗とも保持し、retry・候補追加・次stage自動進行なし。完了後mandatory STOP。
