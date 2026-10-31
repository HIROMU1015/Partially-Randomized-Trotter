# 研究C：RC2／PR-tailの共通収束政策・公平な再fitの技術準備

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。
起点：5fdcd81addf83e893d76c89e64f733433a4e76b3。

**同じ元qualified x0から、固定α=γ=.1の両armを独立に新fitし、共通の事後一次条件とDF物理予算を別々に判定する経路を準備した。** 新しいloss・fitter・optimizerを作らず、旧SOURCEを変更せずにSciPyへ渡すoptionsだけを独立adapterで指定する。

人工新25＋回帰123＝**148 PASS**。Fit／evaluation両環境のSOURCE44・入力14・package byte preflightはPASS。今回の実H4 decode・新fit・保存終点の新規再解析・PF/RTE・ground・実回路build/compile・shots／総費用はすべて0。Future run01は未作成。共通運用指示2026-10-25版§8.1/8.2/8.4/10/10.1/10.3をworkspace原本で確認し、承認証明・保存承認原文・新grantは作成していない。

## 1. 引継ぎと検証目的

[14-call結果](research_c_regularization_first_order_execution_20261101.md)（科学SOURCE0996a5abb4876c2e450439044c26e7df61dfd04c）では、初期寄与=.1一致でもpenalty gradient L2はRC2 .1238844144、PR1.2559620852と異なり、自己終点gradient∞ .4222446601／1.4825504139は既存engineering基準1e-5の外だった。RC2にはtensor-nullでL2を下げる方向、PRにも自身のpenalty下降成分が残った。これらは**保存報告からの引用**で、今回は旧配列のgradientや投影normを新たに計算していない。

[強度・停止診断](research_c_regularization_strength_stopping_diagnostic_20261101.md)、[RC2実行](research_c_regularization_comparator_execution_20261101.md)、[RC2設計](research_c_regularization_comparator_design_20261101.md)、[PR元Stage A](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)、[Evidence Map](research_c_evidence_synthesis_and_decision_readiness_20261031.md)を引き継ぐ。元fitの科学SOURCEはRC2=150f904f6975868abe09966dda8ab41722e2a7cc、PR=e1085c95f5c01b68566e17d543ace5832cb94061。旧successや固定pilotの結果は変更しない。既存の仮定付きPR内部改善と、強決定論への主RZ優位が未支持という判断も不変。

次の問いは「**同じ事前登録された一次収束政策を満たした点同士でもtail λ差が残るか**」。最適強度の探索、同じiteration数、同一最終DF誤差、大域最適の比較ではない。条件を満たさず終わる可能性を残す。

## 2. 維持する科学条件

保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2・sector36、通常number CDF rank8、一般実対称J・実直交U、元順LD2、元g補正κ固定、scalar一回を維持する。PFの時間窓／δはこのfit比較に非適用。

\[
f=\tfrac12\|g-g_{DF}(J,U)\|_F^2,\quad
F_C=f/f_*+.1P/P_0,\quad P=\sum_{tij}J_{tij}^2,
\qquad F_P=f/f_*+.1(M_\epsilon/M_*)^2.
\]

f*=2.1839632053132894e-9 Ha²、P0=4.410884790420646 Ha²、M*=2.2093719370722074 Ha、ε=1e-8 Ha、旧RC2のrawρ=4.951304123962429e-11をそのまま使う。Scaleを新しい終点から作り直さない。Pは全frame ordered空間Jを計数し、spin K=J/2と混同しない。PRは元frame順tailの非identity Z/ZZ smooth mass二乗。

**同じ元qualified J/U/x0から各arm一回、最大2 fit**を第一方式に選んだ。旧終点からのwarm-startは採用しない。既存ffsim pack/unpack形式、128変数＝回転48＋独立上三角J80を保ち、初期値の再logm・新Cholesky・gauge移動・parameter rescaling・bounds追加・preconditionerは行わない。

## 3. 共通solver設定と終了状態

| SciPy L-BFGS-B設定 | 両arm共通の提案 |
|---|---|
| ftol | **1e-15**、正の値で減少率停止を保持 |
| gtol | **1e-6**、事後1e-5に対する余裕を持たせる |
| maxcor / maxls | 10 / 20、旧既定値を維持 |
| maxiter / maxfun | 15000 / 15000、旧既定値を明示。終了不能を避けるalgorithm上の出口で、wall／CPU上限ではない |
| method / jac / callback / bounds | L-BFGS-B / 既存解析AD / None / なし |
| dtype | 既存ffsim0.0.83／JAX x64・binary64 |

SciPy1.14.1では減少率(F_k−F_{k+1})/max(|F_k|,|F_{k+1}|,1)とprojected gradientの出口が別に存在する。旧options=Noneはftol約2.22e-9／gtol1e-5。今回ftolをbinary64 epsに近い正の水準へ下げるが、line search／丸め／flat directionによる早期停止は残る。ftol=0による「gradient収束保証」は採らない。[SciPy仕様](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-lbfgsb.html)、[同版SOURCE](https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_lbfgsb_py.py)

**小さいftolを採っただけで収束PASSとはしない。** 独立な終点value/gradientを既存objectiveで再評価し、生OptimizeResultと整合を確認してから下記二基準を判定する。Objective／gradientの新実装や別solverへのfallbackはない。

Solverのsuccess/status/message/fun/nit/nfev/njev、生x/jac/funを先に保存する。Finiteなline-search failure、iteration/evaluation-limit、相対減少停止も終了点の診断とDF評価を行う。再起動・追加optimizer・強度変更は0。NaN／schema／独立評価不一致等はcorrectness上のTECHNICAL_FAILUREとして停止する。

## 4. 共通の事後一次政策

### 4.1 元128座標のgradient∞

同じnative θで\|∇θF\|∞≤1e-5を要求する。Boundsがないため、この問題のprojected parameter gradientはraw gradientと同じであり、両方の意味を出力する。

J対称性のため、独立上三角の微分はdiag2αJ/P0、off4αJ/P0となる。Uには実反対称生成子の指数写像を使う。Uの直交制約は既存parameterizationが保ち、penaltyの直接U微分は0。θのUは角度、JはHaで測る従来数値座標で、勾配thresholdはその**固定座標でのengineering規則**である。

### 4.2 既知のtensor-null部分空間での正規化勾配

大きさ任意の旧方向微分を閾値へ直接使わず、対称JのFrobenius metricで正規化する。各frameのQ_ti=u_ti u_tiᵀはΣ_iQ_ti=Iなので、

\[
dJ_t=a_t\mathbf1\mathbf1^T/n,\quad dU=0,\quad
\sum_ta_t=0,\quad\sum_ta_t^2=1
\]

は全tensorを保ち、Σ_t\|dJ_t\|F²=1。これは全null空間ではなく、basisによらず分かるidentity移送部分空間である。

独立上三角の目的勾配をb_tijとすると、Frobenius勾配G_tはdiag b、off b/2。従って

\[
c_t=\langle G_t,\mathbf1\mathbf1^T/n\rangle
 =\frac1n\sum_{i\le j}b_{tij},\qquad
z_t=c_t-\overline c,\qquad \|z\|_2\le10^{-5}
\]

を第二基準とする。\|z\|2はこの単位方向群での最大絶対方向微分であり、対称成分の二重計数と方向normを明示した一次条件となる。表現を移動／固定／削減せず、既存combined gradientから計算するだけ。人工例ではnative gradient∞が閾値内でも、この基準が未達になる場合も確認した。

これは対称Jの元metricでの必要な局所条件の検査であり、Uの全接空間や全gauge quotientでのcoordinate-independent stationarityではない。U指数写像の周期性・chartの特異性、他のtensor-null方向、非凸性を解消しない。目的のoptimality gap・局所最小・大域最小を保証しない。旧H4終点にこの新しい投影診断を適用したわけではない。

### 4.3 数値marginと物理精度

両normについて事前固定m=1e-8+1e-7×max(norm,threshold)とし、norm+m≤thresholdならMET、norm−m>thresholdならNOT_MET、間はUNRESOLVED_MARGIN。独立value整合はatol1e-10／rtol1e-8、gradient整合はatol1e-8／rtol1e-7。これらはengineering allowanceで、AD／浮動小数点全誤差を囲った証明ではない。高いcondition numberや丸めfloorにより到達できない可能性もある。

**二つの一次基準・solver終了・物理DF予算を分離する。** convergence_criteria_metは二基準のMET、qualifiedはそれにsolver success=trueを加えた保守的資格。Finiteなsuccess=falseでも基準自体を満たす場合はその事実を残すが、正常終了の資格を付けない。success=trueの相対減少停止でも二基準を満たせば登録一次資格は得られ、満たさなければ未資格のまま。

科学的比較のadmissionには、さらにsector operator norm＋既存sector_operation_scale_guard8_proposal_v1≤CA/8=.00019875 Ha、一般K対raw CAR表現gate（既存component1e-10）を求める。Exact tail λ、tensor Frobenius/max-entry、初期／終了objective、U/J block gradient、optimizer生情報を併記する。**gradient∞だけで科学的精度・性能を決めない。** 最終DF誤差の完全一致、厳密CA、厳密enclosureは要求も主張もしない。

## 5. 最終分類と公平性

| 結果 | 登録上の扱い |
|---|---|
| BOTH_CONVERGENCE_CRITERIA_MET | 両armが二基準＋正常solver終了を満たし、両DF予算もPASS。この固定政策での数値tail比較を行う |
| ONE_OR_BOTH_UNQUALIFIED | Finite終了だが片方／両方が一次基準または正常終了資格を欠く。記述的値は保存し、資格付きtail順位はnull |
| DF_BUDGET_NOT_MET | 片方／両方が物理DF予算外。分類を優先し、共通精度の比較を行わない |
| TECHNICAL_FAILURE | SOURCE/input/schema/表現/gradient整合/nonfinite/worker例外。依存後続処理STOP、部分証拠保全 |

両armは同x0・同表現・同固定strength/scales・同solver options・同事後二基準・同DF予算で比較する。Iteration数／nfevを揃えない。片armだけの追加最適化やwarm-startはない。双方が資格を得ても「この政策を通った点同士」であり、同じ真の最適化誤差／最適ρ／原論文RC-DF solver全体／独立新規性／資源優位を意味しない。

## 6. 再利用と最小差分

| SOURCE | 再利用／差分 |
|---|---|
| c_pr_aware_df.py／c_rcdf_comparator.py | 既存目的、AD、components、fitを無変更で使用 |
| c1_ffsim_corrected_codec.py／c1_real_orthogonal_codec.py | 保存x0をそのまま渡す既存実ffsim経路、pack/unpack、initializer置換、hookを無変更で使用 |
| [c_common_convergence.py](../../src/trottertracks/c_common_convergence.py) | 新規部分はSciPy最終hopのoptionsと純粋な事後norm／整合診断だけ |
| [c_common_convergence_pilot.py](../../src/trottertracks/c_common_convergence_pilot.py) | 既存snapshot load、Calls/finish、SectorCAR、rank_assessment、representation_gateを呼ぶ独立paired runner |
| [新entry](../../scripts/run_research_c_common_convergence.py) | 既定はpassive。別fit/evaluation環境とfresh run01を明示し、fit-workerは専用process |

旧corrected bridgeはoptions=Noneを検査するため、そのファイル／SETTINGSを変更しない。外側のtask-local wrapperを旧bridgeがoriginal_minimizeとして捕捉し、旧x0検査／hookを通過した後、実SciPy呼出しへだけ新optionsを渡す。Loss function／x0／jac=True／callback=Noneは同一のまま。例外時もbindingを復元し、他fitとprocessを共有しない。None指定で旧dispatchへ退化することは人工回帰で確認したが、今回の科学契約では新共通optionsを固定する。

親evaluation環境ではffsimを使わず、保存初期sector資格を再利用してphysical sectorを一回構築。Worker内で旧codecの初期fidelityと終了pack/unpackを検査し、終了tensorを評価する。親では終了J/Uを既存一般K／CARで比較し、両DF誤差を別測定する。新しいCARやDF reconstructionは作らない。

## 7. 人工・回帰検証

[新policy試験](../../tests/management/test_research_c_common_convergence.py)／[新pipeline試験](../../tests/management/test_research_c_common_convergence_pilot.py)と[検証原本](../../artifacts/research_c_common_convergence/2026-11-01/technical/)を公開する。

- 新25件：人工2/4軌道、det±1/π、実ffsim経路・mock optimizerで両目的／x0／共通options一致、旧options=Noneへの退化、独立gradient整合、対称J Frobenius投影・tensor-null、二基準の区別、success／relative／line-search／limit／境界／nonfinite、paired25-call E2E、DF未達／correctness STOP／fresh／worker回収／passive preflight。
- 回帰123件：既存RC2 22、一次診断13、PR目的／x0／workers／修正codec／実直交codec88。**合計148 PASS、6.72秒**。旧一般Kの別process専用試験は今回再実行していない。
- 初回はMockの動的属性を真と扱う新shimの判定で6件失敗した。明示的なTrue検査に修正後、148件PASS。科学的FAILや科学retryではない。

人工rank8/norb4のworker試験も乱数から作ったJ/U/gのみを使い、実optimizer iterationはmockで遮断した。既存人工二次関数回帰のSciPy反復と、未実施の実H4 fitを区別する。今回新しい実H4のgradient・Hessian・SVD・方向微分は0。ローカル人工PASSを科学的GO／CI／外部独立再現と称さない。

## 8. 固定SOURCEと未承認25-call契約

| 固定物 | Identity |
|---|---|
| 実装commit | 38c2bc02b9578ab6ab7734bb61899b6cbc1733e2 |
| [SOURCE44](../../artifacts/research_c_common_convergence/2026-11-01/technical/SOURCE.json) SHA256 | 358df00f1390e23866abc7f78355e95ec061a6feb5173278cfcd5aec7e3e1927 |
| [未承認契約](research_c_common_convergence_contract_unapproved_20261101.json) SHA256 | 3650d105f420e4e5e93a0756ceb6df90930a612b2828e74c066616f7ea6c1a3d |
| 共通input snapshot SHA256 | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |
| 共通qualified initial SHA256 | 78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856 |
| 元P-H4 provenance | 8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50 |
| Future output | artifacts/research_c_common_convergence/2026-11-01/run01、未作成 |

入力14件は元g／同x0、scale／P0 metadata、保存初期sector、旧γ0／RC2／PRのdiagnostic・optimizer・sector scalar。旧終点NPZは対象に含めず、warm-start／旧gradient再解析をしない。初期の独立tensor/sector normは不要に再計算せず、同じhashのqualified初期点を使う。終了点は新候補なので独立に評価する。

| Logical処理単位 | 最大units |
|---|---:|
| Shared snapshot load／保存初期資格gate | 1＋1 |
| Sector context／physical Hamiltonian | 1＋1 |
| 各worker：input load、objective setup | 2×2 |
| 各worker：既存fitter呼出し（optimizer各最大1回） | 2 |
| 各worker：初期診断、raw terminal export、独立終点診断 | 3×2 |
| 各worker：終了fidelity／独立tensor評価 | 2 |
| 終了load、sector評価、一般K／CAR表現 | 3×2 |
| 両arm資格／λ／旧scalar比較 | 1 |
| 合計 | **25 units、最大2 fit** |

内部L-BFGS反復／nfevは各fitter unitに含めて生数を保存する。旧17／14-callを再利用した予算ではない。新規namespace、各arm一回、retry/resume0。Finite未収束は登録候補として診断を完了し、TECHNICAL_FAILUREでは新規依存launch／親sector／比較を停止する。既に開始した独立workerのprivate診断／STOPは回収し、別条件で再実行しない。両終了点が予算内でも収束未達なら資格付き比較はnull。

Fit venv（ffsim0.0.83／JAX0.4.38／NumPy1.26.4／SciPy1.14.1等）とevaluation venv（Qiskit1.3.0等）は旧環境を読み取り専用で使用し、契約内package pinsとffsim2 SOURCEを確認。以下の二つのpassive preflightだけを実行した（科学call0）。

~~~bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B scripts/run_research_c_common_convergence.py --contract docs/research/research_c_common_convergence_contract_unapproved_20261101.json
/tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B scripts/run_research_c_common_convergence.py --contract docs/research/research_c_common_convergence_contract_unapproved_20261101.json --check-fit-environment
~~~

科学実行の別明示許可後だけ、親entryへ`--execute --output artifacts/research_c_common_convergence/2026-11-01/run01 --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python`を追加して一度実行する。今回は実行しない。

## 9. 資源・保全・Stage Bへの引継ぎ

準備時はaffinity32、MemAvailable約56GB・load約1.1。Future executionでは再確認し、既存resource_planから最大2個の独立fit processを選ぶ。既存診断のpeak387852 KiBを基に余裕を含むworkerメモリ推定を併用するが、RSS上限ではない。Tiny 4×4/sector36 BLASのみ自process内1 threadでJAX CPU poolと協調し、各roleのcache/tmpを分離する。任意のCPU／GPU／RAM／worker／wall上限、共有設定／venv／他job変更はない。Python起動は旧venv問題を避ける既存方式のabsolute pathを使い、実体へresolveしてvenvを外さない。

旧SOURCE39・入力・旧fit/raw・科学契約・STOP・必須報告を含む160ファイルについて、起点commitのGit blobとの一致を確認した。QPY・grant／ledger／markerを含むその他の旧科学資産も変更せず、既存ファイルの変更は四つの索引追記だけである。新技術SOURCEだけを追加し、旧科学結果の資格を書き換えない。[検証記録](../../artifacts/research_c_common_convergence/2026-11-01/technical/verification.json)とremote SHA/tree・公開本文を照合後mandatory STOP。

Stage Bは[既存prepare_arm／candidate_factor／atom_operators／tail_gate／configure／kernels](../../src/trottertracks/c_pr_aware_df_stage_b.py)、[ground／aliasとB1順序](../../src/trottertracks/c_pr_aware_df_stage_b_runner.py)、[reference_kernels／reference_round](../../src/trottertracks/c1_stage3_reference.py)、[M4の補正RTE／13round／RPE接続](../../src/trottertracks/c_pr_aware_df_m4.py)を再利用できる。

既存prepare_armのroleはbaseline/proposedに限定されるため、後続adapterではrc2/prの科学identityを保持した上でAPI labelの明示対応が必要。旧Baselineγ0と新RC2を同一候補としない。元g補正κ・scalarを含む表現一致→各新armのtail law→own-DF ground/alias→PF／補正RTE／attenuation／13round phase→必要時RPE、の順となる。新J/U固有のlaw・ground・誤差・価格を旧結果から移送しない。今回は新fit結果がなく、Stage Bのrunner／契約を先回りして作らず、数値も仮定しない。

## 10. Q1〜Q10と次の担当

| 問い | 回答 |
|---|---|
| Q1 共通精度 | 同native gradient∞と正規化identity-null投影normの二基準、独立value/gradient整合。物理DF予算は別admission。 |
| Q2 表現と限界 | 同じ128座標・対称off二重計数を保持し、既知冗長方向を検査。全U接空間／全gaugeのstationarityや大域性は保証しない。 |
| Q3 旧relative停止 | 旧結果を保全。新policyでも出口として残すが、独立事後資格を別判定する。 |
| Q4 successとの区別 | successだけでPASSにせず、基準MET・正常solver資格・DF予算を個別保存する。 |
| Q5 solver選択 | 既存L-BFGS-B/ADを維持しftol1e-15／gtol1e-6を両arm共通に。再試行や保証の主張はしない。 |
| Q6 公平性 | 同じ元x0・strength/scales・class・solver・事後規則。異なるiterationや終了DF誤差を許す。 |
| Q7 再利用 | 旧39 SOURCEを変更せず、最終hop options／純粋診断／paired接続だけ追加。 |
| Q8 人工検証 | 新25＋回帰123＝148 PASS、両実ffsim経路をmock optimizerで確認、両環境passive preflight PASS。 |
| Q9 最小新科学実験 | 各arm新1 fit・計最大2、最大25 logical units。まだ未実行。 |
| Q10 未解決・承認 | 実H4で基準到達可能か、λ差が残るか、最適ρ・下流性能は不明。ユーザー／研究Cが本policyと25-callの実行可否を判断。Codexは明示許可後だけ実行・公開・STOP。 |

**今回の公開照合後はmandatory STOP。** 新実H4 fit、保存終点の新規再解析、Stage B/C、追加重要科学レビューへ自動進行しない。
