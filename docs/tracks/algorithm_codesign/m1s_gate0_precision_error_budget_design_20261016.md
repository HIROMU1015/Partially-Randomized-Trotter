# M1-S Gate 0：数値精度・誤差予算の技術検討

2026-10-16 JST。**科学的評価範囲はユーザー採択済み。精度はCANDIDATE / NOT ADOPTED。SCIENTIFIC EXECUTION NOT AUTHORIZED。** SOURCEはIMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED。

今回の承認は数式・演算構造・静的検査による精度設計までである。実入力・模擬入力の数値評価、Pauli表生成、correctness test、新規定理の証明を実施していない。本書の式は通常の摂動・丸め誤差モデルを使った技術会計であり、M1-Sの新しい定理や厳密certificateではない。[scope採択記録][SCOPE]と[最終契約v2候補][V2]を参照する。

## 1. 誤差の対象と根拠

対象はH4 linear 1.00 Å／STO-3G／8 spin orbitals／保存DF rank12、生成順G3/G6/G9の全3分割。δ窓や有限時間の計算は設定しない。凍結NPZのbinary64値が定義するDF-Hを正本とし、原integral HamiltonianへのDF打切り誤差やSCFの物理誤差を今回の変換誤差へ混ぜない。hash一致は入力同一性の証拠で、演算誤差やstate精度の証拠ではない。

NumPyのbinary64はeps=2^-52、通常のround-to-nearestモデルのunit roundoffをu=2^-53とする。[NumPy finfo](https://numpy.org/doc/1.26/reference/generated/numpy.finfo.html)による表現仕様である。以下では既知の`gamma_m=m u/(1−m u)`を使う。[Higham–Mary 2018, Eq.(1.1),(1.2)](https://eprints.maths.manchester.ac.uk/2673/1/paper.pdf)の通常の丸めモデルだけを参照し、確率的な独立丸め・sqrt(m)への置換は採用しない。

`math.fsum`は中間部分和を保持するが、公式文書はrounding modeへの依存やdouble roundingの可能性も述べている。正しく丸められた厳密和と無条件に同一視しない。[Python fsum](https://docs.python.org/3.11/library/math.html#math.fsum) また、NumPyの内積はBLASに依存し得るため、versionだけで算術経路全体のcertificateにはならない。[NumPy dot](https://numpy.org/doc/1.26/reference/generated/numpy.dot.html)

設計上の値には、A：SOURCE/演算構造から動機づけられる数値品質候補、B：科学的な報告分解能の提案、C：尺度・誤差coverageの情報不足、を明記する。Aも正式採択済み・実データPASSではない。数値実施後に確認が必要という状態は全17項目に付く。

## 2. 17項目の候補一覧

数値表現はreal budgetがbinary64、operator/stateがcomplex128。ゼロは完全一致を要求する候補であり、SOURCE v2ではnormのunderflowを避けるelementwise equalityも検査する。

| Key | 定義・norm・単位 | 数値候補／未決式 | 分類 |
|---|---|---|---|
| eps_H | full-Hと移動後HD/HRの再構成差、絶対F norm、Hartree | null。`16 gamma_(8C+16d+64) S_H`を尺度依存の検討式とする | C |
| eps_hermiticity_H | 入力h/full-H/HD/HRの`||M−M†||_F`、Hartree | null。`2 e_ref+tau_skew` | C |
| eps_hermiticity_factors | 各保存gのskew F norm、無次元（元の正規化固有vector規約） | **0**＋完全一致 | A |
| eps_phase | scalar移動後のHD差+HR差のF norm、Hartree | null。`2 eps_H+16u S_scalar,F` | C |
| eps_state_norm | `abs(<ψ|ψ>−1)`、無次元 | **2^-40 ≈ 9.09e−13** | A |
| eps_state_residual | `||Hψ−E_Rψ||_2/||ψ||_2`、Hartree | null。相対残差目標とH尺度を先に決める | C／科学判断 |
| eps_basis | JW ladder差のF norm／sector埋込み差の2 norm、無次元 | **0**＋完全一致 | A |
| eps_prob | p非負・finite・`abs(Σp−1)`、無次元 | **2^-46 ≈ 1.42e−14** | A |
| eps_imag_coeff | 各HD/HRの`Σ|Im c_P|`、Hartree | null。`16 gamma_(8C) S_coefficient` | C |
| eps_imag_energy | Rayleigh値の虚部絶対値、Hartree | null。`gamma_2048 S_H,2+e_ref+tau_div` | C |
| eps_imag_expectation | norm²で割ったPauli期待値虚部、無次元 | **2^-40 ≈ 9.09e−13** | A |
| eps_W | W絶対精度の目標、無次元。SOURCEの別表現mean照合にも使う | **2^-24 ≈ 5.96e−8** | B |
| eps_v | v絶対精度の目標、無次元 | **2^-24 ≈ 5.96e−8** | B |
| eps_rho | raw点推定からengineering区間両端までの最大距離、無次元 | **2^-12 = 0.000244140625** | B |
| W_error_budget | 保存数値oracleのWの絶対誤差予算、無次元 | null。`2 eta_h+eta_h²+tau_W` | C |
| v_error_budget | 保存数値oracleのvの絶対誤差予算、無次元 | null。§4の係数・law・期待値会計 | C |
| W_floor | W区間下端に対する小分母の事前判定、無次元 | null。`8 W_error_budget`案 | C／受理待ち |

8件に数値候補を提示し、9件の絶対値materializationはnullを維持した。全項目の定義、norm、単位、誤差源、伝播、候補値・根拠、SOURCE対応、不適格・technical STOP・科学判断への返却条件、未採択理由は[precision candidate table][TABLE]に1項目ずつ保存する。

### Aの値の根拠と限界

元のOpenFermion1.6.1のlow_rank.pyは正規化された固有vectorをreshapeしてspin identityとKronecker積を取り、energy尺度をλへ置くため、この保存規約のgは無次元である。これはSOURCEの静的確認で、生成関数や対角化は実行していない。[OpenFermion v1.6.1 SOURCE](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/circuits/low_rank.py)

保存gは固定生成SOURCEの`_as_hermitian`で`0.5*(g+g†)`とされる。finite値・同じbinary64演算なら共役対の一致を期待できるため、gのみの完全一致0を候補とする。行列積を含むfull-Hへこの0を拡張しない。現在の保存gが通過したという確認はしていない。

JW ladderは1/2と±iのdyadic算術、basis埋込みは既存`expand_sector_state`の代入とzero paddingである。従ってbasisの0は近似誤差を許す対象ではなく、表現・ordering整合性を確認する候補である。v2の直接array equalityはnorm²がunderflowして偽の0になる場合を避ける。保存stateに別の変換が必要なら、thresholdを緩めずその由来を調べる。

256成分の複素内積を保守的に2048 scalar rounding段で見ると、norm1付近でgamma_2048が丸めの尺度になる。2^-40=8192uはその約4倍のengineering余裕で、normとPauli期待値虚部の品質候補にする。Pauliはphase/permutationであり、内積の絶対値和はCauchy–Schwarzでnorm²の尺度を持つ。ただし実装backend・例外域・保存stateの既存誤差は未検証で、この倍率を厳密保証とは呼ばない。

正の係数絶対値のfsumと各pの除算には大きな相殺がない。normal-rangeモデルでpの総和誤差はO(u)となるため、128u=2^-46を余裕付き候補とする。fsumの仕様境界、underflow、λの誤差は別台帳に必要である。eps_probだけで正しいlawやW/vの精度を証明しない。

### BとCを採択前に残す理由

eps_W/eps_vの約6e−8、eps_rhoの1/4096は報告する小数の分解能に関する具体案で、研究継続の境界ではない。その分解能が科学目的に十分かはユーザーへ戻す。0.7/0.85等のρ閾値を採択しない。

Cの式ではd=256、C≤706,837（[runtime検討][RUNTIME]の静的展開回数）、`S_H=16(|c|+Σ|h_pq|+Σ_l|λ_l|(Σ|g_l,pq|)²)`等の相殺前尺度を使う。S_Hはabsolute F normの尺度を上から見積もるための演算構造上の候補であり、数値は計算していない。16という余裕係数やscalar-op段数の包絡はengineering設計で、BLAS・fsum・全DAGに対する証明済みerror boundではない。Cの値はこれらの仮定・尺度・coverageを閉じてから絶対値へ具体化する。

SOURCEは現在、eps_H等に**数値scalar**を要求する。上表の式文字列をそのまま実行configへ入れたり、結果を見た後に許容値を増やすことはできない。式による予算を結果前に固定する別SOURCEが必要か、別途承認した初回精度診断が必要かを最終採択前に決める。

## 3. 誤差台帳の境界

1. **入力**：raw hash一致で保存binary64入力を固定。元の量子化学近似の誤差を0と認定する意味ではない。
2. **変換**：fermionic ordering・contractionを保存したJW積と符号付き集約。相殺前係数の尺度が丸め予算に必要。
3. **虚部**：捨てた虚部の係数L1はoperator差の会計に使える。再構成reference自身の丸めを加えずに実差のcertificateにはしない。
4. **Identity**：scalar移動のHD/HR差を両方残す。全Hのscalar整合と、移動後sampling lawの変化を区別する。
5. **State**：保存ψを再生成・上書きせず、期待値は明示的norm²で割る。規格化誤差と割算の丸め、Rayleigh残差は別量。
6. **W/v**：変換・λ・p・期待値・演算の誤差を合算した予算が必要。設定された予算とaccuracy targetの大小関係を検査しても、予算coverage自体を実証したことにはならない。

## 4. W/vへの技術的伝播

以下は既知のnorm・分数の摂動会計を、この演算へ適用するための条件整理である。新規数学定理を提出しない。

R̂、λ̂の誤差会計をe_R、e_λとし、λ̂>e_λ、canonical表から`||R̂||≤λ̂`が使える場合、正規化hの誤差尺度を

`eta_h=(e_R+e_lambda)/(lambda_hat−e_lambda)+tau_h`

で設計する。τ_hは除算等の追加丸め。λ̂の下限が解決しなければこの式を使えない。固定保存oracleに対するWの予算候補は`e_W=2 eta_h+eta_h²+tau_W`。τ_Wにはmatvec、norm²の商、`1−<h²>`の減算を含める。Wが小さいと絶対誤差が小さくても相対誤差は大きくなる。

vについて、pのL1誤差をe_p、Pauli期待値の誤差をe_a、meanの誤差をe_muとする。unsigned Pauliの二次momentではsignが二乗で消えるため、微小係数の符号反転を無視してeventを固定する必要はない。係数差と確率差を会計した上で、

`e_v=e_p+2e_a+e_a²+2e_mu+e_mu²+tau_v+|s_p−1|(1+e_a)²`

を保守的な候補とする。`s_p=Σp̂`。最後の項は、SOURCEのcentered和がpの総和が厳密1でない場合に持つ補正である。二次moment−mean²の関係だけを使ってこれを落とさない。τ_vにはcentered和の丸めも含む。

いずれも各入力誤差のcoverage、正の分母、算術モデルが条件であり、現在その数値を取得・認証していない。exact固有stateとの距離は会計に含めない。独立referenceとの小さな差だけでは、共通の誤りや相関した丸めを除外できない。

## 5. ρ区間と理論範囲の整合

SOURCEはraw点Ŵ/v̂を残し、raw区間`[Ŵ−e_W,Ŵ+e_W]`、`[v̂−e_v,v̂+e_v]`もv2で保存する。理論範囲`0≤v≤W≤1`と交差できることを先に検査する。交差後の座標区間を[W_lo,W_hi]、[v_lo,v_hi]とした場合、W_lo>0でのみ

`rho_lo=(1+v_lo/W_hi)/2`, `rho_hi=(1+v_hi/W_lo)/2`

を使う。f(W,v)はv≥0でvに増加、Wに減少するため、**元の予算が正しくcoverするという条件の下で**矩形区間として整合する。v≤Wという結合制約を外した矩形の上端は1を超え得るが、それは過大な包絡であり矛盾ではない。SOURCEはこれを1に丸めず残す。区間端点の算術は外向き丸めの厳密interval arithmeticではない。

raw点が理論範囲の外なのにbudget内で交差可能な場合、点をclampして整合を偽装しない。v2ではraw ρが[1/2,1]外ならPRECISION_UNRESOLVEDとする。予算で説明できない不一致はtechnical STOP。これらは研究のGO/STOPではない。

eps_rhoのSOURCE上の意味は**raw点から両端までの最大距離**で、区間全幅ではない。相関を無視した矩形は保守的であり、初回の報告ではこのradiusを使う。小分母prescreenの`W_floor=8e_W`案は分母の相対変動を小さく保つためで、eps_rhoを保証する条件ではない。最終的には実際のengineering区間radiusで分類する。

## 6. 分母と精度の分類

| 状況 | 分類・保存 |
|---|---|
| λ_R=0 | EMPTY_TAIL_NOT_APPLICABLE。W/v/ρをnull |
| 浮動小数点Ŵ=0 | W_ZERO_NUMERICAL_NOT_APPLICABLE。数学的exact-zeroや改善余地なしの証明ではない |
| W区間が0を含む | DENOMINATOR_INTERVAL_CONTAINS_ZERO_UNDETERMINED |
| W_lo>0だがW_floor以下 | SMALL_DENOMINATOR_UNDETERMINED |
| ρ区間radiusがeps_rho超過／raw ρが理論範囲外 | RHO_PRECISION_UNRESOLVED。raw値と区間を保存 |
| 理論範囲と予算区間が交差しない／検証gate失敗 | 当該technical STOP。残り対象も成功・失敗・未評価として全行を保存 |

## 7. Stateの証拠区分と次の判断

norm、全H Rayleigh残差、sector埋込みは独立に検査する。これらの合格と、exact eigenstateへの距離、全H非縮退性、全Fock端点性、pair-sum条件、Jensen floor適用条件は別である。gapなしに残差から距離を保証しない。保存数値oracleの代数的診断自体に追加スペクトル計算は必須ではないが、定理domainの認証を求めるなら別の科学scope/実施承認が必要になる。

9個のnullを閉じる手段は、(a)ユーザーが根拠付きの絶対上界・精度値を提供する、(b)結果前に誤差尺度と予算の算定式を採択し、それを扱うSOURCEを追加準備する、(c)精度尺度だけを取得する別の限定契約・SOURCE・明示実施承認を用意する、のいずれかを先に決めることである。現在のv2 runnerはfixed scalar contractを要求するため、未採択の式を自動評価したり、精度診断の名目でGate 0を先行実行したりできない。今回(a)–(c)の科学的選択や新しい数値取得は行っていない。

未決は17値の最終採択、9値のmaterialization・coverage、Bの分解能の科学的十分性、runtime/capsの受理、全Gate 0契約の採択。その後も別の明示科学実施承認を要する。**新科学/数値correctness/Pauli表0、mandatory STOP。**

[SCOPE]: ../../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/gate0_scientific_scope_adoption_record.json
[TABLE]: ../../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/precision_candidate_table.json
[V2]: m1s_gate0_headroom_execution_contract_v2_candidate_20261016.md
[RUNTIME]: m1s_gate0_runtime_and_resource_contract_candidate_20261016.md
