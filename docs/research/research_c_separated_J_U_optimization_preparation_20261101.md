# 研究C：係数J・軌道U分離型PR-aware DFの技術準備

## 1. 完了範囲・引継ぎ

起点`77b021802e0edbc330fae0387aa525637557f224`から独立branch `codex/research-c-separated-j-u-preparation-20261101`で実装した。実装commitは`e75c6b66bfd7711ca93d3e3942b0dae6ece08be7`。**新規28＋既存199＝重複を除く227 test methods PASS**。PR／RC2の独立した分離型最適化、各ブロックの一次診断、物理DF評価、比較・STOPまで接続した実行可能なpipelineを準備した。

**実H4の数値配列読込、新規最適化、保存終点の数値再解析、PF/RTE、shots、native build/compile、総費用計算は0。** 実入力についてはhash・metadata・環境のpassive preflightだけを行った。新しい417-call案は未承認であり、科学run01は未作成である。

必須の[固定U LP結果](research_c_fixed_u_tail_lp_numerics_execution_20261101.md)、[共通joint終点](research_c_common_convergence_execution_20261101.md)、[RC2比較設計](research_c_regularization_comparator_design_20261101.md)、[Stage B結果](research_c_rc2_pr_stage_b_execution_20261101.md)を確認し、既存の目的・正規化・表現・来歴と未資格事項を引き継いだ。共通運用指示2026-10-25版のworkspace原本、特に§8・§10を実際に確認した。

指定レビュー`research_c_df_algorithm_design_scientific_review_20261101.md`はworkspace・添付で見つからず、起点commitのGitHub取得も404だった。**原本や§4/6/7を読了したとは扱わず**、依頼の自己完結した数理仕様に従った。重要科学レビューを新たに実施した記録ではない。

前回の保存直鎖H4/1Å/STO-3G、number CDF8/sector36、LD2、h=.1/s1/r128/K0の固定U・固定head・tail tensor保存LPでは、λ .9545432846581517→.9544320281272091Ha、約.0116555%減だった。これはその制限問題の数値結果であり、別Uでの改善余地を否定しない。**今回はtail tensor保存等式を外し、headを含む全JとUを動かす。** 以前のLPを再実行する経路ではない。

## 2. 数学的定義と対称係数座標

既存のnumber DF再構成を`g_DF=A_U j`とする。各frameの独立上三角Jを並べたjに対し、対角要素の重みを1、非対角を√2とする対角行列Dで`x=Dj`を定義する。これにより`||x||₂²=Σ_tij J_tij²`となる。非対角Jの二重計数を保ち、全frameのsigned実対称Jを自由変数にする。H4では80係数である。

以下では`A=A_U D⁻¹`、`T=T_R D⁻¹`と書く。Tは既存のspin K=J/2→非identity Z/ZZ係数mapをtailへ制限したもの。HeadはPR penaltyへ直接入らないが、tensor lossの最適化変数である。

\[
 F_{PR}(x,U)=\frac{\|Ax-g\|_F^2}{2f_*}
       +\frac{\gamma}{M_*^2}M_\epsilon(x)^2,
 \qquad M_\epsilon=\sum_\ell\sqrt{(Tx)_\ell^2+\epsilon^2}.
\]

固定値はγ=.1、ε=1e-8Ha、f*=2.1839632053132894e-9Ha²、M*=2.2093719370722074Ha。Smoothingの零点floorを引かず、identity phaseをmassへ含めず、entrywise |J|に置き換えない。

`c=Tx`、`h_l=sqrt(c_l²+ε²)`、`v=Tᵀ(c/h)`とすると、

\[
 \nabla_xF_{PR}=A^T(Ax-g)/f_*+2\gamma M_\epsilon v/M_*^2,
\]
\[
 \nabla_x^2F_{PR}=A^TA/f_*+
 \frac{2\gamma}{M_*^2}\left[vv^T+
 M_\epsilon T^T\operatorname{diag}(\epsilon^2/h^3)T\right].
\]

各項は正半定値で、固定U問題は凸。ただしhead等にflat directionがあり得て、強凸性・解の一意性を一般には仮定しない。小さい数値勾配を厳密なsuboptimality enclosureや最適性certificateへ昇格させない。

## 3. PR固定U solverと前処理

[c_separated_j_u.py](../../src/trottertracks/c_separated_j_u.py)の`Coefficients`が、既存`FixedU`の全frame線形map・独立contraction/adjoint・Pauli mapを再利用する。内側iteration中は固定Uのdense Aを再利用し、Uのexpm/ADを毎回計算しない。独立contraction/adjointはstate診断で再確認する。旧LPの`problem`／`solve_once`や保存等式は呼ばない。

PRの内側にはSciPy1.14.1のBFGSとcallable gradientを採用した。80係数という小規模問題で、full inverse-Hessian近似を保持できる。初期計量は

\[
 B=\begin{bmatrix} A/\sqrt{f_*}\\ \sqrt{2\gamma}\,T/M_*\end{bmatrix}
   =L\operatorname{diag}(s)V^T,
 \quad H_0^{-1}=V\operatorname{diag}\!\left(1/\max(s,1)^2\right)V^T.
\]

**全方向を残すinvertibleな初期計量**であり、Bの第二blockやIを目的に追加していない。横長の人工mapでも完全な右null基底を補い、SVDの小さい方向を変数集合から落とさず、特異値floor=1は初期計量だけに使う。J勾配の採用判定は元のisometric x座標で行う。ライブラリの[`hess_inv0`](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-bfgs.html)と[既存SVD](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.svd.html)を使用し、独自optimizer・追加正則化・後付けのsolver切替はない。

[人工solver選択記録](../../artifacts/research_c_separated_J_U/2026-11-01/technical/solver_selection.json)では、ε=1e-8を保った小問題でtrust-exactが400反復上限に達し、identity metricのBFGSは一次条件を満たした。一方、保存スカラーの正規化だけを合わせた**人工**rank8・norb4例では、対角計量BFGSがgradient∞≈1465.76でprecision-loss停止し、identity計量でも1000反復後≈3.53e-4だった。選択したfull-direction SVD計量では同じ目的で≈5.04e-9となった。元の実H4配列を使った試し解きではなく、万能な収束保証とも扱わない。

## 4. RC2固定Uの強い係数対照

全ordered空間係数Z=Jについて、既存α=.1、P0=4.410884790420646Ha²を保持する。

\[
 F_{RC2}(x,U)=\frac{\|Ax-g\|_F^2}{2f_*}+\frac{\alpha}{P_0}\|x\|_2^2
 =\frac12\left\|
 \begin{bmatrix}A/\sqrt{f_*}\\\sqrt{2\alpha/P_0}I\end{bmatrix}x
 -\begin{bmatrix}g/\sqrt{f_*}\\0\end{bmatrix}\right\|_2^2.
\]

[`scipy.linalg.lstsq`](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.lstsq.html)のgelsdでaugmented least squaresを一度解く。Normal equationを直接解かず、rank・全特異値・cutoff・独立元勾配を保存する。Cutoffはeps64×max(augmented shape)。α>0の数学的問題は強凸だが、数値solverの返却だけで一次資格を認定しない。QR/SVDの数値rankと物理rank8も区別する。

PRとRC2に同じ内側solverを強制しない。物理入力・表現集合・初期点・U更新・一次判定・DF予算を揃え、内側はそれぞれの目的に適した方法を使う。この対照も非凸Uの大域最適化や原論文の全RC-DF実装を再現したものではない。

## 5. Uブロックと交互更新

現在のUに対する右側局所更新`U_new=U exp(A(η))`を使う。Aの独立上三角はη/√2、下三角はその負で、ηはFrobenius-isometricな実反対称座標。既存ffsimのreal rotation pack/expmとJAX x64を再利用する。元frame順・J対称性・直交性を保持し、polar補正・frame並替え・新gauge・U再初期化を行わない。

J固定時のpenaltyはUに直接依存せず、同じtensor lossのU方向ADを計算する。局所積chartには旧global exp parameter目的を直接差し込めないため、ffsimと同じcontraction式に局所積を接続する最小JAX adapterを追加した。既存full objective・有限差分と人工照合した。`J*(U)`やlogmを微分せず、implicit differentiation／厳密variable projectionではない。

更新順序は**初期state→J0→state0→(U_k→J_k→state_k)**。U更新後には必ずJを再最適化する。U solverのendpoint chart gradientだけでなく、実際の新Uにη=0を再設定し、post-refit Jに対する全接空間勾配を測る。π付近でglobal exp chartが退化する場合も、そのchart勾配を物理stationarityの代用にしない。

## 6. 収束・停滞・数値条件

| 項目 | 固定したpilot政策 |
| --- | --- |
| PR J solver | BFGS、gtol1e-7、maxiter1000、上記SVD計量、boundsなし |
| RC2 J solver | Augmented gelsdを1回／block |
| U solver | 局所L-BFGS-B、gtol1e-7、ftol1e-14、maxiter25、maxfun500、maxls40、maxcor10 |
| Outer | 最大64 complete U→J cycles、各blockのsigned objective変化を保存 |
| J一次資格 | isometric xの独立gradient∞ ≤1e-5、下記margin付き |
| U一次資格 | η=0の物理接空間gradient₂ ≤1e-5、同margin付き |
| Gradient margin | 1e-8＋1e-7×max(測定値,閾値)。norm＋margin≤閾値でMET、境界はUNRESOLVED_MARGIN |
| Objective一致／非増加 | absolute1e-10＋relative1e-8。許容内の微小増加もsignedで記録し、0へ丸めない |
| Native J-gradient照合 | atol1e-6／rtol1e-7。小さい固定f*による丸め増幅を含む整合性allowanceで、一次閾値とは別 |
| Stall | |cycle変化|≤1e-12＋1e-10×|前objective|が3 cycles続き、勾配未達ならSTALLED_NOT_STATIONARY |
| Orthogonality／representation | 1e-10、後者rtol0。既存DF guard・予算を保持 |

1e-5という値が旧native parameter∞と同じでも、**係数のisometric勾配と全物理U接空間という別の定義**である。人工のfinite-difference・flat/poor-conditioning・false-success試験を踏まえたengineering pilot閾値として提案した。旧maxfun15000や旧native gradient判定を無条件に移植したものではなく、energy精度への換算保証もない。

人工の交互例はPR16 cycles／RC2 39 cyclesで一次条件を満たし、64上限・stall・U stopを別のmockで検証した。この観測から有限pilotの64 cyclesを固定したのであり、H4でも収束するとは予測しない。Maxiter/maxfunはSciPyの停止optionsとして使用し、実際のnfev/njev・全VG回数を記録する。CPU/RAM/wallの上限ではない。

各solverのsuccess/status/messageを一次残差と分離する。Precision-loss flagでも独立一次条件を満たすケースはflagを残し、stationarityを別に記録する。J一次未達ならそのarmを停止。Uのstatus1は登録した有限block更新として扱い、その他の有限U停止はJ refit後に未資格終了する。ただし独立joint一次条件がすでにMETなら、その事実とsolver flagを分離して保存する。非finite・schema・objective/gradient不整合・直交性・非増加違反はcorrectness STOP。再solveや別solverへのfallbackはしない。

## 7. 再利用と最小新規実装

| 既存資産 | 利用範囲 |
| --- | --- |
| `c_pr_aware_df.py`／`c_rcdf_comparator.py` | 同じtensor/penalty定義、coefficients、smooth mass、full L2、独立ffsim value/gradient |
| `c_fixed_u_tail_lp.py` | 全frame線形map・独立contraction/adjoint・pack。旧保存等式／LP solveは不使用 |
| `c1_ffsim_corrected_codec.py`／実Schur codec／ffsim util | 元qualified x0のdecode fidelity、real局所expm、終了点の診断用native pack |
| `c1_h6_initialization.py` | 同じDF再構成とtensor metrics |
| `c_common_convergence_pilot.py`等 | 元snapshot/scales、独立arm launcher、Calls、STOP・資源、両環境の構成 |
| `SectorCAR`／`rank_assessment`／一般K | 元g補正κ・scalar一回のsector36 physical DF norm、guard、独立表現gate |
| JSON export repair | NumPy scalar対応とatomic/exclusive出力。旧serialization不具合を再導入しない |

新規は[block algorithm](../../src/trottertracks/c_separated_j_u.py)、[研究固有pilot接続](../../src/trottertracks/c_separated_j_u_pilot.py)、[CLI](../../scripts/run_research_c_separated_J_U.py)と人工試験だけ。旧joint fitterは一回のglobal optimizer接続を前提にするため再起動せず、目的・AD・codecを再利用した。新しい数値計算library、Gaussian合成器、approval-proof機構は作っていない。

新しいnative x/jacは終了J/Uの**診断用parameterizationとAD**で、単一global optimizerの生出力と偽らない。実solverの生x/jacはJ block／U blockごとに保存する。旧`rank_assessment`のstatus文字列はoptimizer収束資格の代用にせず、新しい一次判定とDF判定を使う。

## 8. 人工・回帰結果

[Core tests](../../tests/management/test_research_c_separated_J_U.py)／[pipeline tests](../../tests/management/test_research_c_separated_J_U_pilot.py)／[227件の実行記録](../../artifacts/research_c_separated_J_U/2026-11-01/technical/test_results.json)／[人工study](../../artifacts/research_c_separated_J_U/2026-11-01/technical/synthetic_study.json)。全データは人工で、実H4配列を使っていない。

確認項目は、全Jの線形作用/随伴・off-diagonal二重計数、既存joint objective/gradient一致、PR Hessianと凸性、RC2解析解/独立解、flat/nonunique/near-singular、det±1/π branch、物理U有限差分、block非増加、post-U J refit、両一次資格、finiteな非収束、false solver success、nonfinite・表現不一致STOP、DF未達の独立分類、input/SOURCE/fresh/STOP・原本保全である。

人工E2Eはfit child processの実block optimizerと、評価parentの実人工sector/general-Kを接続し、33-unit成功と、意図的な表現不一致で29-unit partial STOPを確認した。Mockの64-cycle経路は417上限と整合した。人工joint対照ではRC2 jointのobjectiveが分離型よりわずかに低い例も保存し、分離型が常に優れると仮定していない。

既存PR pipelineの試験を初めfit環境へ混在投入した8 failuresはthreadpoolctl/OpenFermion不在によるものだった。該当10件を既存評価環境へ分離して全PASSとし、旧SOURCEや共有venvを変更していない。CUDA-enabled jaxlibは既存fit環境に無いため、既存CPU JAX経路を使用し、追加installしていない。人工の反復数や時間をH4の収束改善率・高速化率とは報告しない。

## 9. 未承認H4最小科学実験案

[固定契約](research_c_separated_J_U_contract_unapproved_20261101.json)／[新SOURCE114](../../artifacts/research_c_separated_J_U/2026-11-01/technical/SOURCE.json)。

| 固定物 | 条件／identity |
| --- | --- |
| Source manifest SHA256 | `f2c3618c9e75bd7949a3824f9fed9213f537d3f58cea1f1616dd59198910c4f8` |
| 契約SHA256 | `fb0e137564e4013225bd12c4812017af882cee4fd5af69c382149a5eb75735df` |
| 対象 | 保存直鎖H4/1.00Å/STO-3G、Nα=Nβ2/sector36、number CDF rank8、元順LD2 |
| 元h/g/scalar入力SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 共通qualified J/U/x0 SHA256 | `78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856` |
| 初期化来歴 | prepare snapshotのSOURCE e1085c95f5c01b68566e17d543ace5832cb94061。旧joint終点・LP候補warm-startなし |
| 新候補 | PR／RC2各1独立trajectory。ffsim joint fitter呼出し0、同じ初期J/U、強度/scale固定 |
| 旧joint対照 | e2f7dfd44357510d54ab9c6c9c5032e0072ab376のRC2／PR保存終点。原本を変更せず、同じ新metricの診断だけ各1回 |
| DF条件 | 元g補正κ、nuclear scalar一回、norm＋既存guard≤CA/8=.00019875Ha。Tensor lossとの同一視なし |
| Fresh output | `artifacts/research_c_separated_J_U/2026-11-01/run01`、未作成 |
| Retry／resume | 0／0 |

旧終点は、global exp-chart勾配と新しい全物理接空間勾配を直接比較できないため、**J/U読込＋same-metric診断を各arm2 units、計4 units**だけ契約へ明記した。これも将来の保存値再解析であり、今回は実行していない。旧終点を再最適化する許可ではない。

| 科学call（logical unit） | 最大 |
| --- | ---: |
| Parent：元snapshot/input読込、保存初期DF資格、sector context、元physical sector | 4 |
| Parent：両final読込、両sector DF norm/guard、両独立representation、比較 | 7 |
| Worker：load/setup/初期identity、旧終点load/metric、終了native/tensor/export | 8×2=16 |
| Worker：初期J＋各cycle J solve | 65×2=130 |
| Worker：U solve | 64×2=128 |
| Worker：initial＋post-J0＋各post-J state | 66×2=132 |
| **最大合計** | **417** |

一つのmulti-block trajectoryを一solveとして隠さず、全blockを計上する。J nonlinear BFGSはPR最大65、U L-BFGS-Bは両arm合計最大128、RC2 augmented solve最大65。内部のobjective/AD評価はblock内として実回数と時間を別記録する。終了点が早く資格を得たり停滞した場合、未使用枠を他開始点・rank・強度・追加試行へ転用しない。

新候補の一次資格・DF予算を別々に判定し、両者が両armでPASSした場合だけqualifiedなtail λ比較を出す。同じ目的のold/new Fと勾配は比較するが、PRとRC2のraw Fを直接順位付けしない。同x0各1 trajectoryの初回pilotであり、多初期値や分子間の統計的安定性は評価しない。旧jointと新手法は古典計算予算が完全一致しておらず、観測差だけでequal-effort速度優位や分離そのものの因果効果を認定しない。

Q1のsame-metric一次状態、Q2のPR目的/exact λ、Q3の物理DF予算、Q4のRC2対照下のλ関係、Q5のJ/U別呼出し数・VG実績・inclusive wall/CPU・停滞原因を保存できる。PF/RTE、attenuation、ground/alias、shots、native価格は含めず、新candidateへ旧下流結果を転記しない。

## 10. 実行順序・資源・STOP

Byte/環境/fresh/resource確認→保存初期資格→physical sector一度→独立PR/RC2 worker→各blockと診断→有限終了点のtensor/native export→各sector/表現→一次/DF/λの比較→公開・STOPという順序である。

J未資格・U stop・outer上限・stallは有限な未資格終点として保存し、追加最適化なしで登録DF診断まで行う。DF NOT_METをsolver不具合や目的の失敗と混同しない。Correctness FAILでは依存する後続を止め、既存launcherの範囲で開始済み独立workerを安全に回収し、partial evidenceを保存する。閾値変更・初期値fallback・solver切替・rank探索はない。

支配的な反復では固定UのA/Tをキャッシュし、Uはshape-cached JAX graphを再利用する。J→U→Jの逐次依存は守り、独立したPR/RC2の2 workersを実際のCPU/quota/RAMから選ぶ。既存AD peakに基づくmemory estimateはquotaではない。Tiny BLASのprocess-local調整でJAX poolとの過剰競合を避け、task-local cache/tmpを使う。CPU/RAM/worker/wallの恣意的な上限、共有環境・他ユーザーの変更は行わない。

## 11. 保全・残る課題・次担当

[技術postflight](../../artifacts/research_c_separated_J_U/2026-11-01/technical/preparation_postflight.json)で新SOURCE114・入力13と両環境のpassive PASS、旧LP成果31を含む旧raw976件、旧SOURCE/契約/STOP、旧科学manifestの不変を照合する。今回の技術資料は科学的性能結果として新result entryを登録していない。研究方針Bの技術準備状況を研究概要とdated noteへ追記し、過去の結果を書き換えない。

既知の実装blockerは残っていない。ただし、PR内側が実H4でも十分に一次条件へ到達するか、外側が停滞しないか、DF予算内でλが改善するか、RC2も同じ資格を得るかは未検証である。SVD計量・有限budget・一次閾値はpilot政策で、収束定理や物理予算適合の保証ではない。旧未収束終点からの一般的優位・独立新規性・RC-DF全体に対する優位も未認定である。

次担当はユーザー／研究C：固定417-call案の初期化・停止・比較政策を確認し、限定科学実行を別途明示許可するか判断する。Codexは今回、技術準備・人工検証・契約公開・remote SHA/tree/本文/SOURCE照合までを行う。**照合後mandatory STOPし、実H4の最適化・保存終点再解析や下流科学計算へ自動進行しない。**
