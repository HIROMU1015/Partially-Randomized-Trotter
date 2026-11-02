# Track A AX-1a：モデル比較・fit仕様

2026-10-09、version 1。**fit未実施・AX-1b未認可。** [結果前契約](track_a_ax1a_preanalysis_contract.md)の追加仕様。機械設定は `artifacts/resource_applicability/track_a_ax1a/2026-10-09/execution_plan_draft_v1.json`。

## 1. 共通target、単位、許可情報

algorithm B0/B1/B2/B3はdiscard/full deterministic/partial random/random-dominant。予測モデルは必ず `PRED_BASE_*` と呼び、方式名と区別する。

RQ-P1のtargetはaxis `cosine`/`sine` ごとの **one-shot measured full-wrapper native RZ count**。state preparationを除外、ancilla/control/phase/measurementを含む。Qiskit 1.3.0、basis `rz,sx,x,cx`、opt1、seed17、coupling-mapなしという保存compile scopeを固定する。予測とdirect参照が異なるcompiler/scope/unitなら比較不能。RZ depth/CX count/depth/total depth/circuit sizeは補助診断とし、主モデルの係数を別gate単位へ無断転用しない。

情報区分は I1=予測時利用可、I2=H4校正のみ、I3=参照後診断のみ、I4=oracle-assistedのみ、I5=欠測/未確認/N/A。candidate q/r/R/K/prefix、sourceで定義したaction数、exact λ_R/finite probabilities/BはI1（新サイズでの取得費用は別途測定が必要）。H4 compiled Cはtraining fold内でI2、test foldではI3。test bias/shot/eligibleはI4でありcost model inputには入れない。eventをseedから再生成してI1へ昇格させない。

## 2. AX-M0/1/2の分類と比較可否

| モデル階層・対象 | 分類 | 保存値によるAX-1b評価 |
|---|---|---|
| AX-M0 原論文App. A23〜A29 | LCU会計・解析上界 | finite weight/probability/normalizationのsource対応。b/B再現と上界slackを別指標で評価 |
| AX-M0 原論文E21、QPE schedule/FT gate totals | 原論文条件の会計 | 現行single-T signal/native RZとは不一致、総量の直接誤差はN/A |
| AX-M0 C_gs・旧paper_d6等 | 経験的energy推定・出典未解決 | finite-time axis-bias predictorへ転用しない。U01継続 |
| AX-M1 現行source action/finite-B会計 | 未校正の定義式 | 再現照合可能。action indexはRZ単位ではない |
| AX-M1 現行source native構造会計 | 未校正構造予測 | actual basis/event/category入力が不足、REGISTERED_INPUTS_UNAVAILABLE、N/A |
| AX-M2 H4校正 | 経験予測 | 下記SINGLE_COEFF/FEW_PARAMの同単位RZ診断。保存参照shotを掛ければCONDITIONAL_ORACLE |

`df_deterministic_step_rz_cost`などの既存helperはlocal compiled U_opsとdiagonal会計を混ぜる。これを未校正解析modelの係数として使用しない。boundのslack、heuristicのprediction error、accountingのidentity/scope mismatchは別tableにする。

## 3. 特徴量のsource定義

| feature | 単位・定義 | 保存field/静的source |
|---|---|---|
| q | 外側二次PFの反復数、dimensionless | `candidate.q` |
| R=qr | tailの全substep数、dimensionless | q/r identityから算術、deterministicではrandom modelには使用しない |
| n_det | DF fragment action数 | M1 `_deterministic_fragment_count`：二次対称列の `2q L_D`（ALL selected fragmentsを保持） |
| E_rand | 期待component application数、未丸め | `expected_random_applications_exact`。randomは `qr Σ_n p_n(n+1)`、deterministic/discardは0 |
| n_fixed | one-body/scalar/phase等のaction数 | `_fixed_action_count`：`2 + 2q n_one_body_blocks + q(1_{constant≠0}+1_{extracted_identity≠0})` |
| A_exact | action数 | `n_det + E_rand + n_fixed` |
| A_ceil | action index | `n_det + ceil(E_rand−1e−15) + n_fixed`、既存policy `ceil_expected_applications_v1` |

E_randが未保存でもallowlist内のfinite p_n/q/rが全てあり、source定義と一致する場合に限り再計算できる。samplingやstate作用ではない。field欠測を0としない。randomのE_randが不明ならその予測はINPUT_MISSING。deterministicのE_rand=0はsource意味論による確定値である。

PM-1はn_det/n_fixed未保存。n_detは保存prefix/qとsourceの全fragment保持規則から導出可能。n_fixedは `_prepare_discard` がfull one-body/constantを保持しempty tailとする不変条件に基づき、同Hamiltonian/state/identity/compiler/qのM1-A B0 anchor全件が同じ値を持つ場合だけ共有できる。anchorの一致・source invariant確認に失敗したらPM-1 featuresをN/Aとする。compile Cやbiasから推定・逆算しない。全derived fieldにanchor fingerprints、source hash、導出式を記録する。

## 4. 登録モデル

| ID（version 1） | 入力 → 出力 | 校正・評価 |
|---|---|---|
| PRED_BASE_ACTION_INDEX | A_ceil → actions/shot。保存W_actionは `N_ref,total A_ceil` | 未校正index。rank correlation/順位の診断のみ。RZ相対誤差・RZ-regretへの無校正代入は禁止。W_actionはoracle-assisted index |
| PRED_BASE_SINGLE_COEFF | `C_hat,a = β_a A_exact` | axis別β≥0、interceptなし、candidate均等weight。最小の校正baseline |
| PRED_BASE_STRUCT_ACCOUNT | `C_hat,a = C_wrapper,a + C_basis/transition + C_det_diagonal + E[C_random_events] + C_control + C_scalar_phase − C_boundary_cancellation` | source-defined gate単位の未校正会計。重複計上を避けるcomponent partitionが必要。全termのactual basis/event/transition/control入力が未保存なのでAX-1bではN/A。generic coefficientやlocal Cで埋めない |
| PRED_BASE_FEW_PARAM | `C_hat,a = θ0,a + θD,a n_det + θR,a E_rand + θF,a n_fixed + θq,a q` | axis別非負NNLS、最大5係数、下記rank規則、全method pooled。method別fit/one-hotなし |
| AXM1_FINITE_NORMALIZATION | `(λ_R,T,q,r,K,p_n)` → b/B | 未校正source会計。RQ-P2のbias/Nを供給しない |

finite orderは `n=0,2,…,K`（K even）、`τ=λ_R T/R`、`w_n=|τ|^n/n! sqrt(1+(|τ|/(n+1))²)`、`b_K=Σw_n`、`p_n=w_n/b_K`、`log B=R log b_K`。τ=0はw_0=1、他0。sourceのlgamma/log-domain実装・overflow policyを維持し、K+1次平均作用のsignalを計算しない。原論文対応upper bound `b≤exp(τ²)` のslackは別評価。Bが不明なcellを1に補完しない。

## 5. Training、fold、数値規則

主training集合はM1-A/B1のfingerprintで1対1joinできる **210候補**（H4 1.00 Å rank12）。accuracy-ineligibleも含める。PM-1の8候補、M2の5候補、PM-0/2の派生ledgerはtrainingに入れない。302精度点を302倍の独立cost training rowsと数えない。候補membershipとjoinはallowlistに固定する。

| 診断 | train/test分割（ref Cを見る前に固定） | 表示 |
|---|---|---|
| full H4 fit | 210でfit、同210で評価 | IN_SAMPLE_DEVELOPMENT |
| 主leave-one-q-out | q={1,2,4,8}ごと、そのqの全method/prefix/Kをtest、残りtrain | INTERNAL_GROUP_DIAGNOSTIC。4foldのpooled OOFを作る |
| leave-one-prefix-out | L_D={0,3,6,9,12}ごと全件test | INTERNAL_GROUP_DIAGNOSTIC |
| leave-one-method-out | B0/B1/B2/B3ごと全件test | INTERNAL_GROUP_DIAGNOSTIC、外挿flagを保持 |
| leave-one-K-out | random K=2/4を片方test、他のrandomと全deterministicをtrain | INTERNAL_GROUP_DIAGNOSTIC。方式B0/B1のdeterministicをtestへ混ぜない（保存Kのlabelだけでrandom判定しない） |
| PM-1 nearby prefix | full210 fitの凍結model→8件 | OBSERVED_PREFIX_DIAGNOSTIC、独立評価ではない |
| M2 fixed-five | full210 fitの凍結model→実行fingerprint5件 | OBSERVED_GEOMETRY_DIAGNOSTIC、候補最適性ではない |

各fitは参照compile cost **Cのみ**をtargetにする。bias/B/eligible/N/G/PM-0 selectorやPM-2 frontierで特徴・weight・modelを選ばない。R/K別は保存された値単位でgroup化し、結果後binを選ばない。

**fit手法**：非負least squares `min ||Xθ−y||², θ≥0`、candidate均等weight（axis別1row/candidate）。SEの逆数weightなし。SINGLEは同じ目的の閉形式 `β=max(0,Σ A_exact C/Σ A_exact²)`。FEWは固定featureのみ、ridge/robust loss/grid探索/追加interactionなし。

NNLSのreference APIは [SciPy 1.14.1 `optimize.nnls`](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.nnls.html)、`maxiter=10000, atol=1e−12`。backend/versionは実行前に実際の環境と固定し、異なるversion/adaptorを使うならreview済みconformanceが必要。今回SciPyはimport/インストールせず、解析実装も未作成である。

**conditioning/collinearity**：binary64。trainのみのRMSで各columnをscale（intercept=1）、全zero columnはdrop。scaled train XのSVD relative rank tolerance `1e−12`。列保持優先は `[intercept,E_rand,n_det,q,n_fixed]`、削除優先は逆順 `[n_fixed,q,n_det,E_rand,intercept]`。この順に追加してrankが増える列だけ保持する（Cは参照しない）。従属性、singular values、drop理由をfoldごとに保存する。targetもtrain cost RMSでscaleし、係数を元の単位へ戻す。全target0なら係数0を明示、scale1で扱う。

drop済みfeatureのtest値がtrainで成立した線形関係を破る場合は `OUTSIDE_TRAIN_FEATURE_RELATION` とする。予測は固定zero/drop係数のまま出し、誤差とcoverageに残す。test Cを使った復活/refitはしない。係数の物理的因果解釈は不可。

min fit rowsは `2×retained_parameter_count`、rank>0、全active feature/target finiteを要求。不足はFIT_UNIDENTIFIABLE（fold N/A）でありtestを使って救済しない。負cost・負action・identity不一致はSTOP。zero costはfitには残し、比率指標だけN/A。ineligible候補もone-shot cost評価に残す。欠測で候補を無告知除外せず、モデル別coverageと理由を出す。

fit後は係数がfiniteかつθ≥0であることを確認する。scaled spaceで `g=Xᵀ(Xθ−y)`、active cutoff `θ>1e−12` とし、KKT residualは `max(max_active |g|, max_inactive max(−g,0), max |θ*g|)`（空集合のmax=0）、許容 `1e−8`。負係数をclipして救済しない。違反なら `STOP_NUMERICAL_FIT`、solver fallbackなし。saved-vs-derived cost/aggregate照合はabsolute `1e−9`、relative `1e−10`を `abs(diff)≤abs_tol+rel_tol*abs(reference)` と適用する。finite-B一致はabsolute `1e−12`、relative `1e−10`、log B absolute `1e−10`。candidate identity/hash/seedの一致はexact。toleranceでcandidateを同一化しない。

## 6. 単純モデルを維持するcomplexity gate

未来サイズに持ち越すcost modelのdefaultはSINGLE_COEFF。FEW_PARAM採用の候補条件は主leave-one-q-outの同coverageで、以下を全て満たすこと。

1. 各q-foldのpositive-ref axis absolute relative errorの平均を取り、q-foldを均等weightで平均する（zero-ref/missing countsを別報告）。SINGLEからFEWへの低下が相対10%以上 **かつ絶対.01以上**。
2. 個々のq-foldでFEWの同metricがSINGLEより.02を超えて悪化しない。
3. 両modelとも全4foldが識別可能で同coverage。欠測がある場合はgateを判定不能としdefaultを維持する。

SINGLEのscore=0ならSINGLEを維持する。これらは設計上の複雑性採用基準でありformal statistical testではない。conditional-regretは副診断として全値を出し、regret改善だけでcost gate失敗を覆さない。secondary folds/PM-1/M2の都合のよい結果だけで採用しない。

改善なし・不安定・rank不足ならFEWを拡張しない。新feature、method別fit、非線形model、候補選び直しは別versionのexploratory提案・別レビューが必要。PR資源優位の有無もmodel複雑化の理由にしない。

## 7. 予測recordと未使用サイズへの凍結

各recordにmodel ID/version、model source/hash、training membership/hash/fold、feature source/derivation、candidate execution fingerprint、prediction time、unit/scope、C_pred、B_pred、bias_pred、N_pred、G_operational_pred、G_conditional_oracle、情報区分、oracle flags、欠測・extrapolation・fit statusを保持する。現在bias_pred/N_pred/G_operational_predはnull、availabilityは `N_A_NO_OPERATIONAL_BIAS_PREDICTOR`。

H4 internal foldsと既観測geometryは独立予測性能でない。将来H6/H8は参照C/biasを開く前にmodel・候補・予測record hashesを保存する。新サイズのE_rand/λ_R/basis情報取得費用をcompile実測費用と分けて記録する。H6を使った変更はH6 developmentとして明示、H8を見た後の変更はexploratory。AX-1a/AX-1bの完了はこれらのscience実行認可にならない。
