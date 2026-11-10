# R依存compiled価格モデル Phase 1結果（2026-11-11）

指定したH4/H6の校正・独立検証・限定rare診断を一回ずつ完了した。新規full-wrapper compileは**通常64＋診断16＝80件**、保存C/Sは59 pathを再利用した。独立検証のpath別相対RMSEはH4 **1.039841%**、H6 **0.229959%**だった。これは固定条件・各n8の有限cohortでの観測結果であり、全law期待価格、長q/R、全RPEへの採用保証ではない。事前の採用閾値は未設定のため、モデルの`qualified_for_candidate=false`を維持する。

[固定PLAN](../r_price_phase1_preparation_20261111/PLAN.json)／[凍結モデル](run01/FROZEN_MODELS.json)／[独立検証](run01/VALIDATION.json)／[path別残差](run01/PATH_RESIDUALS.csv)／[rare診断](run01/RARE_DIAGNOSTICS.json)／[作者による保存・scope照合](CHECKS.json)／[HANDOFF](HANDOFF.md)。

## SOURCE・条件・実行順序

科学コードSOURCEは`1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`。準備公開`c596e88d2a88fb219c2e373fb0d72c6e39cb723a`を基点に、既存C branch `codex/r-variable-compiled-price-design-20261111`を継続した。ユーザーの「実行してよい」を、この固定Phase 1の実行許可として適用した。追加の承認書・grantは発行していない。

Linear H4/H6、1.00 Å、STO-3G、保存tol-only DF10/DF19・weighted Hermitian completion・元fragment順・保存Hamiltonian/state/snapshotを使用した。分割はH4 PR prefix1、H6 PR prefix7、K2。H4はS2/δ=.1、H6は4次m5_best/δ1.6。保存DF-Hの同sector基底エネルギーをtargetとする従来の数値state条件付き設定を保持し、厳密ground資格へ昇格していない。価格は状態準備P=0の一回Hadamard interrogationのcompiled RZ count。

元のsymmetric-directional制御、測定付きfull Hadamard wrapper、Qiskit1.3.0/NumPy1.26.4、basis `rz/sx/x/cx`、opt1、transpiler seed17を使用した。compiler hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。SCF/DF/Hamiltonian/state再生成、signal/N/prior/horizon/Gの新計算は0。h=.1、13round、εE=.00159360010199040 Ha、αtotal=.05各方式、βtotal/stat=.4/.2は固定metadataの転記のみ。

校正完了後、係数・共分散・校正結果・PLANを**独立検証価格の取得前**に[7eeabf0ae482ee610d667e28470692d209f2bb21](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/7eeabf0ae482ee610d667e28470692d209f2bb21)へcommit/pushし、GitHub commit/treeを照合した。そのSHAを指定して独立検証、続いて別unitのrare診断を実施した。検証後も凍結ファイル・校正結果・PLANのbytesは同commitと一致し、再fitしていない。最終runnerは[STOP](run01/STOP.json)に`PHASE1_COMPLETE_MANDATORY_STOP`を記録した。科学retryは行っていない。

## 校正・検証coverage

qはqPF、nは古典価格path数。X/Yは同じpathに対応するpaired応答であり、2n独立標本ではない。

| 系 | qPF | R_outer | T | candidate ID | 役割 | n |
|---|---:|---:|---:|---|---|---:|
| H4 | 8 | 4 | .8 | `14bd81a375400c9b0305` | 保存探索・校正 | 9 |
| H4 | 16 | 4 | 1.6 | `64e4f7e9ec7fba28815e` | 保存旧独立q点を今回の校正へ再利用 | 33 |
| H4 | 8 | 8 | .8 | `6e1efff1ba1f7ac95a3b` | 新校正 | 8 |
| H4 | 16 | 8 | 1.6 | `201d498a3fa5eaa3b5c5` | 新独立検証 | 8 |
| H6 | 1 | 16 | 1.6 | `2d2af6ca830dff14a695` | 保存探索・校正 | 9 |
| H6 | 2 | 16 | 3.2 | `72255c43facdfb8b4bee` | 保存校正 | 8 |
| H6 | 1 | 32 | 1.6 | `4269eafc8c0ef8922c23` | 新校正 | 8 |
| H6 | 2 | 32 | 3.2 | `24fec6bdd44d6a728f1e` | 新独立検証 | 8 |

H6は元順11 signed pieceを保持した。R16の整数allocationは`[1,2,1,2,1,2,1,2,1,2,1]`、R32は`[3,3,3,3,2,5,2,3,2,3,3]`。各和はR、全event数はqPF×R。元`ABS_TIME_PER_OUTER`・min-one/largest-remainder規則を用いた。

保存59 path＋新校正16 path＝75 pathが校正入力、新独立検証16 path、別forced診断8 path。通常の新32 pathで64件、診断8 pathで16件をcompileし、cache hitは0。通常はwhole-pathの16 workerで並列化し、compiler内部poolとの重複を避けた。診断は固定runnerの逐次経路を使用した。共有設定・他processを変更していない。

## 凍結したモデル

各系を独立に、3校正セルのpaired C−S平均へ等セルweightでfitした。

\[
\widehat C(q,R)=S(q,R)+\theta_0+\theta_q q+\theta_L qR.
\]

Sは元構造計数器で各pathから取得する構造価格で、full-wrapper compiled価格そのものではない。新RのpathではSをC取得前に保存し、保存RのSは再計数せず再利用した。旧Mq/Affine係数を別Rへ移植していない。

| 系 | θ0 | θq | θL |
|---|---:|---:|---:|
| H4 | 70.67676767676816 | 94.89630681818181 | 18.289496527777757 |
| H6 | −15.486111111007617 | 28438.75000000001 | 23.26128472221899 |

X/Yを別responseとしてfitした結果、今回の各係数は両軸で同値だった。各design rank=3、3セル3係数なのでcell-level lack-of-fit DOF=0。校正cohortのpaired covarianceから得た6×6係数共分散は保存しているが、独立cohort仮定による有限標本のnominal量であり、rare/law/長q・R systematicを含まない。検証セルを校正へ混入させていない。

## 独立検証結果

以下は[VALIDATION.json](run01/VALIDATION.json)の保存指標の転記。e=C−Ĉ、相対RMSEは`√mean((e_i/C_i)²)`、最大相対誤差は`max|e_i/C_i|`。平均価格の比の誤差とは区別する。両軸の保存指標は同値だった。

| 系・独立点（各n8） | compiled平均 C | 予測平均 Ĉ | bias e | MAE | RMSE | 相対RMSE | 最大絶対相対誤差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H4 q16/R8 | 49638.125 | 49808.823232 | −170.698232 | 434.625 | 516.149320 | 1.039841% | 1.664619% |
| H6 q2/R32 | 377494.75 | 377604.736111 | −109.986111 | 708.5 | 867.371392 | 0.229959% | 0.405285% |

価格・残差のpaired X/Y covarianceとnominal平均共分散も保存した。有限cohort分散とモデル系統誤差、係数不確かさ、共有joint uncertaintyを同一視しない。`total_prediction_CI`、全law期待価格、full RPE resourcesはnull。事前のacceptance thresholdがないため、科学的採用PASS/FAILは付けず`FINITE_INDEPENDENT_POINT_EVALUATED_NOT_ADOPTED`とする。

## 希少order2の別診断

通常新32 pathの保存event metadataでは、観測2304 eventは全てorder0だった。これはorder2の全law確率・費用寄与が0であることを意味しない。事前PLANに従い、同じrotationを持つlegal order0基準pathと、指定位置だけlegal order2へ変更した対応pathを別datasetとして取得した。

| 系・事前指定pair | 基準C（X/Y各） | order2変更C（X/Y各） | 保存ΔC（X/Y各） |
|---|---:|---:|---:|
| H4 head/tail先頭 | 23822 | 24090 | +268 |
| H4 tail内・異basis | 24507 | 24699 | +192 |
| H6 正piece先頭 | 188344 | 188485 | +141 |
| H6 負piece先頭 | 188707 | 189803 | +1096 |

単位は`FORCED_LEGAL_PAIR_NOT_UNCONDITIONAL_COHORT`。通常cohort・fit・平均価格へpoolしていない。全位置・複数order2・全law重み付き寄与は評価しておらず、`all_law_weighted_contribution=null`を維持する。

## 接続上の注意・未資格

公開A `5ca2dce3234f3e4a971f4a8c852da676f050e315`、B `7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae`の固定metadata/APIを使用した。Bの共通API・N・会計・旧H4/H6 providerは変更していない。A/Bの作業中branch/worktree、private raw、process、環境/cacheへ介入していない。

**時刻metadataの注意**：固定B由来の`identity.binding.T`（model.binding.Tも同様）は旧taskの参照値.8を保持する。各セルの実評価時刻は上表と`identity.candidate.T`であり、`qPF×δ`と一致する。科学backendはcandidate.Tとsigned piece時刻・allocationを照合して回路を構成した。元PLAN/結果は変更しない。Bが取込む際もcandidate全体・qPF・R・lawをkeyとし、参照binding.Tだけで価格を同一視しない。共通APIの時刻fieldを正規化する場合は、B側menu/N/Cを一緒に扱う技術変更として実施し、Cの固定結果を黙って修正しない。

価格recordはdirect有限cohortとmodel予測を別scenarioで保存し、全32 recordの`expectation=FINITE_COHORT_ONLY`、`normalization_is_price_multiplier=false`を保持する。nと量子shot N、normalization Bとcompiled価格を分離する。保存compiledの個別path価格は、そのfingerprintでの決定論的compile結果であって、有限path平均の全law保証ではない。

未観測q/R、長q/R systematic、希少eventの全law寄与、finite CDF samplerとideal lawの距離、shared/cross-candidate joint covariance、総CI、全RPE成功・総費用・R最適性は未資格。H6 m0..3の別δ/prefix5/R32、他分割・compiler・Hamiltonianへ移送しない。この式だけで未取得path構造Sや全law平均Sを得たとは扱わない。

## 確認・保全・次の担当

[VERIFY_RESULTS.py](VERIFY_RESULTS.py)による作者の技術照合はPASS。固定4科学SOURCE、PLAN、freeze前校正bytes、candidate/cohort、private結果と公開C/S・event digest・axis fingerprint、原入力hash、旧pipeline/provider/counterの不変、64＋16上限、pool/re-fitなしを確認した。[CHECKS.json](CHECKS.json)に記録した。既存の[人工/mock14件](../r_price_phase1_preparation_20261111/TESTS.txt)＋[R adapter回帰9件](../r_price_phase1_preparation_20261111/R_ADAPTER_REGRESSION.txt)の23 PASSを再利用し、今回固定コードを変更していない。新しい独立科学レビューやfull-wrapper statevector作用検証は実施していない。

原本SOURCE/input/raw/result/grant/ledger/marker/KEEP・過去STOPを保持した。新private events/構造checkpoint/axis価格はrepo外`/home/AbeHiromu/projects/r-price-phase1-20261111-private/run01`に保全し、Gitには公開可能な価格・残差・統計・hash/fingerprint索引だけを追加する。stdout/stderrとstage markerも保存する。

**次の担当**：ユーザーが採用に必要な価格精度・全law/rare/追加q/R検証の範囲を判断する。Codex Bは[HANDOFF](HANDOFF.md)に従い公開保存recordを技術的に取り込めるが、model資格を自動昇格せず、N×C/G・R最適化は別の実行scopeとする。Codex Cはcommit/push・remote SHA/tree/主要成果物照合後mandatory STOP。追加科学計算へ進まない。
