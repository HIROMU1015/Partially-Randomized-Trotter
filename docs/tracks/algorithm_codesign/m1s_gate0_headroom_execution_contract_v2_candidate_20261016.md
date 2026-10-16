# M1-S Gate 0：科学scope採択後の最終実験契約v2候補

2026-10-16 JST。**LIMITED SCIENTIFIC SCOPE ADOPTED BY USER / FULL EXECUTION CONTRACT CANDIDATE NOT ADOPTED / SCIENTIFIC EXECUTION NOT AUTHORIZED**。

ユーザーが今回、対象・law・oracle診断・比較規約を正式採択した。数値精度とruntime/resource capの最終値、Gate 0契約全体、科学実施承認は採択していない。旧[契約候補v1][V1]の本文・過去のSTOPを変更せず、本書を新設した。SOURCEは**IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED**、科学的correctness未検証。科学判断の基準は固定[Checkpoint 2 GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md)のままである。

## 1. 採択した範囲と正本

[scope採択記録][SCOPE]は採択主体=user、根拠は[今回の指示原本][USER]。**SCOPE_ADOPTION_NOT_EXECUTION_AUTHORIZATION**であり、科学実行authorizationやrunner markerではない。

H4 linear 1.00 Å／STO-3G／8 spin orbitals／DF rank12、charge0/multiplicity1、保存N=4、Nα=Nβ=2、Sz=0。入力は保存development NPZ、ref `8b7666d0270f7fac01e430e195bef7b58ac0116c`、8,652 bytes、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。

| Split | HD block | HR block |
|---|---|---|
| G3 | 0–2 | 3–11 |
| G6 | 0–5 | 6–11 |
| G9 | 0–8 | 9–11 |

保存生成順を使い、constantとone-bodyはHDへ置く。全3分割を成功・失敗・未評価も含め報告し、後から有利な分割を追加・選別しない。H4 1.30 Å、H6、別rank、SCF/積分/DF因子の再生成は追加しない。

## 2. Law・Identity・state

JWは保存orbital順、mode0 MSB、alpha偶数/beta奇数。8文字I/X/Y/Z wordを辞書順にし、同じwordの係数を**符号付きで集約してから**`λ_R=Σ|c_P|`、`p_P=|c_P|/λ_R`、`P_signed=sgn(c_P)P`を定義する。自動の小係数cutoffは0、浮動小数点のexact zeroのみ除く。

Identity scalar sを`HD'=HD+sI`、`HR'=HR−sI`へ移し、保存constant・両側scalar・正符号relative phase/T_relの+s/−s台帳を残す。Identity移動でsampling law/λが変わる。DF-nativeの旧event・normalizer・価格との同値性は認定しない。

保存full長256 stateをoracleに使い、sector36への埋込み、norm、全DF-H残差を別gateで検査する。stateを再正規化・再生成・上書きせず、期待値は明示的norm²で割る。Labelは`NUMERICAL_SECTOR_GROUND_ORACLE_NOT_THEOREM_CERTIFIED`。exact固有state距離、全H非縮退、全Fock ground/端点性、pair-sum条件、Jensen floor適用を合格扱いにしない。

## 3. 比較と診断の意味

W=1−<h²>、v=Var_P<P_signed>、h=HR/λ_R、ρ=(W+v)/(2W)。同H・分割・state・理想相対時間・Pauli law、同概念的総event位置Rでshared r=R、ordinary r=R、independent各枝R/2（R偶数）とする。native価格・depth・shotsの等価性は仮定しない。

小step先導係数のoracle headroom診断に限定する。δ/Rの具体値、有限step信号、σ/gap/coupling、native総資源、T1c/Q4/Q5は今回の実行範囲へ含めない。Checkpoint 2 GPTの限定受理、ρ単独の総資源STOP不採用を維持する。

## 4. SOURCE固定と今回の修正

v2 SOURCE：`94e6089eba37b022a79eb262a2aa5025f5e8ddcd`。[source identity][SOURCE]に全fileのbytes/SHA256を記録。新namespace `src/trottertracks/algorithm_codesign/m1s_gate0_v2/`、runner `scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v2.py`、追加static checker `check_m1s_gate0_precision_static.py`。

旧SOURCEは保存したままcopyを基に限定修正した。scope記録binding、単位別ゼロ許容の完全一致gate、NumPy主要build hash、log cap、raw区間保存、raw ρの精度分類、科学処理前contextとnative log captureを追加した。これらは静的確認済みで、数値・runtime correctnessは未検証。

## 5. 精度17項目と誤差伝播

[precision設計書][PREC]と[候補表JSON][TABLE]が各項目の定義・norm・単位・根拠の正本。SOURCE keyは全17個を維持する。

- 構造に基づく候補：eps_hermiticity_factors=0、eps_basis=0、eps_state_norm=2^-40、eps_prob=2^-46、eps_imag_expectation=2^-40。
- 科学的分解能の提案：eps_W=eps_v=2^-24、eps_rho=2^-12。科学的十分性は未採択。
- 絶対値null：eps_H、eps_hermiticity_H、eps_phase、eps_state_residual、eps_imag_coeff、eps_imag_energy、W_error_budget、v_error_budget、W_floor。尺度・予算coverage等の未決条件を候補表に具体化した。

W/v予算からのρ区間は、元の予算がcoverするという条件付きengineering感度である。raw点・交差前区間を残し、理論範囲との交差を検査する。正のW下端が解決した場合のみ矩形端点からρ区間を作る。小分母、数値W=0、λ=0、ρ精度不足は別分類とし、不一致をclampで隠さない。SOURCEのeps_rhoはraw点から両端までの最大距離で、区間全幅ではない。

exact-state誤差を含む厳密certificate、外向き丸めのinterval arithmetic、true eigenstateへの信頼区間は提供していない。9個のnullが残り、精度契約は閉じていない。現在のcandidateは**正式実験契約ではない**。

## 6. Runtime/resource候補

[技術契約候補][RUNTIME]に根拠を記録する。Python3.11.0rc1、NumPy1.26.4、主要build6件hash、CPU1 worker、BLAS/OpenMP等各1 thread、GPUなし。4 GiB address space、result/log各256 MiB、context4 MiB、Pauli表65536、全_combine寄与1,000,000（静的包絡706,837）。科学wall-time capなし。

Output候補は`artifacts/track_b_m1s_gate0_result/2026-10-16/v2_once`。今回作成しない。各capは未採択・benchmark未実施で、正常実行の実測保証ではない。launch環境と実効limitを別途確認する。

## 7. Technical PASS/STOPと結果形式

入力identity、SOURCE/契約/別実施承認、basis、full-HとHD/HR再構成、Hermiticity、identity/phase、p正規化、state norm/残差、W/v整合、ρ定義域/精度、domain labelを独立に扱う。technical PASSは当該検査の合格で、研究全体の科学GOではない。

結果は全3分割のinput/SOURCE/契約/承認hash、Pauli canonical表・hex係数・p・表hash、phase台帳、λ/W/v/ρ、raw区間・engineering区間・分類、全gate、未評価事項、runtime/実効上限を保存する。global失敗でも全3行を未評価として残す設計。failureで結果JSONを保存できない場合はcontext/logと終了状態を保全し、自動再試行しない。

## 8. 拒否経路・未承認状態

既定はstatusのみ。help/schema/validate-configに科学数値処理はない。最終契約adopted=falseと9精度nullによりexecution_ready=false。

将来、SOURCE→最終採択契約→別明示実施承認の3段階とscope hashを照合する。scope採択記録単独・過去authorizationでは科学moduleへ進めない。今回、新しいscientific execution authorizationやrunner markerを発行していない。

## 9. 未採択の科学判断とmandatory STOP

ρの0.7/0.85、dressingの0.1、総資源目標は採用しない。最終的な17許容値の採択・誤差coverageの十分性・runtime/caps受理・Gate 0全体契約採択はユーザーへ戻す。oracle診断と定理認証の区別は採択済みであり、追加spectrumが必要なら別scopeを決める。

次の実施承認までに必要なのは、(1) null9項目を根拠付きで閉じる手順と値、(2)全17値・runtime/capsを含む最終契約の正式採択、(3)その後の**別の明示科学実施承認**。静的PASSは数値correctnessを認定しない。最初の数値gate・ログ/メモリ動作確認も別実施承認に含める。

**新規科学計算・Pauli係数生成・実/模擬数値correctness・simulation/回路/compile/benchmark・新定理・新authorization/markerは全て0。公開照合後mandatory STOP。**

[SCOPE]: ../../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/gate0_scientific_scope_adoption_record.json
[USER]: ../../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/user_instruction_original.txt
[SOURCE]: ../../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/source_identity.json
[TABLE]: ../../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/precision_candidate_table.json
[V1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83fbac28bca8e308b3039d8532492e6510492f89/docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_candidate_20261016.md
[PREC]: m1s_gate0_precision_error_budget_design_20261016.md
[RUNTIME]: m1s_gate0_runtime_and_resource_contract_candidate_20261016.md
