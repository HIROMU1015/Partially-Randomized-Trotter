# 研究C：J/U結合・reduced objectiveの実装と限定pilot準備

2026-11-02 JST。Repository: HIROMU1015/Partially-Randomized-Trotter。起点 `e5a58753fa64124233522b92182e68f4a00a0461`、独立branch `codex/research-c-j-u-coupled-preparation-20261102`。

## 1. 完了範囲

RC2の一意な内側解を使うreduced objective、およびPRの非一意性を保守的に判定するguard付き経路を実装した。外側は既存SciPy BFGS、各試行UでJをsolve・再精密化し、値とenvelope偏微分を返す。Fixed-J TRFをreduced solverと呼び替えたものではない。PRに追加の非一意性がある場合の、局所J/η joint更新も人工問題で比較した。こちらは代替prototypeであり、今回のH4科学契約の自動fallbackには含めない。

**人工・回帰186 pytest items PASS（新28、既存158、parameterized casesを含む）。** 同一点jac/codec修正、独立終点一次診断、親子process・sector/CAR、partial STOPも確認した。Source133、入力25、両環境のpassive preflight PASS。

**今回の実H4 decode・最適化・保存配列/traceの新規数値再解析・実H4 Hessian・科学callはすべて0。** PF/RTE、ground、shots、native、総資源、重要科学レビューも実施していない。将来run01は未作成である。共通運用指示2026-10-25版のworkspace原本、特に§8/8.4/10/10.1/10.3を確認した。承認原文・証明書・新grantを作らない。

- [新core](../../src/trottertracks/c_j_u_reduced.py)／[研究固有pilot](../../src/trottertracks/c_j_u_reduced_pilot.py)／[CLI](../../scripts/run_research_c_j_u_reduced.py)
- [core人工test](../../tests/management/test_research_c_j_u_reduced.py)／[pipeline人工test](../../tests/management/test_research_c_j_u_reduced_pilot.py)
- [人工比較](../../artifacts/research_c_j_u_reduced/2026-11-02/technical/synthetic_study.json)／[test記録](../../artifacts/research_c_j_u_reduced/2026-11-02/technical/test_results.json)
- [未承認1051-call契約](research_c_j_u_reduced_contract_unapproved_20261102.json)／[SOURCE](../../artifacts/research_c_j_u_reduced/2026-11-02/technical/SOURCE.json)／[保全・準備STOP](../../artifacts/research_c_j_u_reduced/2026-11-02/technical/preparation_postflight.json)

## 2. 保存結果から引き継ぐ課題

[修正版実行報告](research_c_j_u_consistency_repair_execution_20261101.md)およびそのexecution_summary/result、保存traceの記録形式・optimizer診断を確認した。以下は公開済み報告の引用であり、新しいtrace集計やHessian計算ではない。

保存直鎖H4・1Å・STO-3G、number CDF8、Nα=Nβ2/sector36、元順LD2、同じ元qualified x0、α=γ=.1について、修正版SOURCE `24a7268359506abd171686132f12b4876f9e62bf`の結果は次の通り。

| 保存測定 | PR-tail | RC2 |
| --- | ---: | ---: |
| Cycles | 35 | 64 |
| F | .012265732322593252 | .029193402354005357 |
| J gradient∞ | 1.80769e-5、未達 | 6.03710e-7、MET |
| 最終fixed-J U更新直後の物理gradient₂ | 1.37612e-6 | 6.48021e-8 |
| J refit後の同じUでのgradient₂ | 2.954838 | .866717 |
| DF予算 | PASS | PASS |

J再solveが目的を減らしても、直前に小さかったU偏微分を保つ保証はない。小さいΔJについて

\[
 F_U(J+\Delta J,U)-F_U(J,U)
       =H_{UJ}(J,U)\Delta J+O(\|\Delta J\|^2)
\]

であり、交差曲率が大きいと小さい係数変更でも勾配が大きく変わる。一方、これだけで**実H4の原因・支配的固有方向を特定したとは言えない**。今回その交差Hessianは測定していない。

[旧分離型結果](research_c_separated_J_U_execution_20261101.md)、[旧設計](research_c_separated_J_U_optimization_preparation_20261101.md)、[旧joint](research_c_common_convergence_execution_20261101.md)、[固定U LP](research_c_fixed_u_tail_lp_numerics_execution_20261101.md)も参照した。LPのtail tensor保存等式を今回へ転用せず、旧終点を新しい初期値に使わない。解消済みのNewton pre-step jac／codec異点比較を再び主原因とは扱わない。

## 3. RC2：数学的に成立するreduced objective

既存のisometric係数座標をx=D upper(J)、A=A(U)D⁻¹とする。全ordered JのL2と一致するため、元の目的は

\[
 F_{RC2}(x,U)=\frac{\|Ax-g\|^2}{2f_*}
                 +\frac{\alpha}{P_0}\|x\|^2.
\]

α>0、P0>0なので、Hxx=AᵀA/f*+2αI/P0は一様に正定値、最小固有値は少なくともμ=2α/P0。各Uの最適x*は一意で、局所実直交chartにおいて滑らか。**正確な内側解**では

\[
 \Phi(U)=F(x^*(U),U),\qquad \nabla\Phi=F_U(x^*,U),
\]
\[
 D_Ux^*=-H_{xx}^{-1}H_{xU},\qquad
 H_{red}=H_{UU}-H_{Ux}H_{xx}^{-1}H_{xU}.
\]

これがfixed-J U曲率と異なる点である。Artificial testでは既存ADの有限差分から各blockを作り、Schur式・内側解の感度・各±UでJを解き直したreduced勾配差を照合した。本番solverではnormal equations、Hessian逆行列、擬似逆行列を形成しない。

線形係数を各非線形試行で消去する考え方は既存variable projectionに従う。[RustのNIST解説、§The Variable Projection Method](https://math.nist.gov/~BRust/pubs/NaturesBF/Ms4Final.pdf)。ここでの正L2とPRへの条件付き拡張、数値guardの採用は、この研究の目的に合わせた下記の実装判断であり、文献だけを根拠に新実装の収束を認定していない。

### 近似解では何が残るか

内側residual rJ=Fx(xhat,U)なら、数学的RC2では ||xhat-x*||≤||rJ||/μ。しかし外側gradient誤差を評価するには、この差に掛かる混合微分の制御も必要である。Fxが小さいという理由だけでは、F_U(xhat,U)を正確なreduced gradientとは言えない。PRではμ>0を全空間に仮定すること自体ができない。

今回の実装は**数値pilotとしてのadmission**に留める。以下を記録するが、内側誤差の厳密enclosureや勾配誤差保証はnullである。

1. Refined Jの独立元座標gradient∞＋元marginが1e-5以下。
2. RC2のaugmented数値rankが2 solveともfull。
3. 同一点のreverse AD gradientとforward residual-Jacobianのchain ruleが元allowance内。
4. 再精密化前後のchart／physical U gradientの差normの大きい方が1e-6以下。

最後の1e-6は物理U一次閾値1e-5の10%を使う**新pilotの事前固定stability allowance**である。元の一次閾値・整合allowanceは緩めない。二つの解が似ていても共通の誤差は残り得るため、これを真のgradient誤差上界にはしない。実H4での安定性は未検証である。

## 4. PR：非一意性を無条件に無視しない

元のPR目的、γ=.1、ε=1e-8、全supportのsmooth floorを保持する。

\[
 F_{PR}(x,U)=\frac{\|Ax-g\|^2}{2f_*}
        +\frac{\gamma}{M_*^2}\left(\sum_l\sqrt{(Tx)_l^2+\epsilon^2}\right)^2.
\]

既存解析Hessianから、γ,ε>0ではker Hxx=ker A∩ker Tである。Tのtail部分は一般対称Jに対しinjective：同一空間orbitalの異spin対を含むZZ係数が各J成分を保持する。したがって非一意性はhead側のAの核に限られる。

各head frameのJ_t=c_t 11ᵀは、直交Uで再構成するとc_t δpqδrsとなりUに依存しない。Σ_head c_t=0というLD−1次元のredistributionは、**全Uに共通する厳密なgauge**である。Headの係数数をLD*d、d=n(n+1)/2とし、rank A_head=LD*d−(LD−1)であれば、核はこの共通gaugeだけとなる。局所的にはquotient上の内側Hessianが正定値で、minimizer間のU偏微分も同じになる。追加の正則化やgauge固定制約なしに、正確な解のenvelopeを正当化できる十分条件である。

実装は各試行のSVDを保存し、上のrank/gap・tail injectivityを数値確認する。Rank guardは256 eps64 max(shape) max(1,smax)、retained singular gapはその1024倍を超えること、null側はguard以下を要求する。**有限精度SVDによる判定は厳密なrank証明ではない。** より一般の非一意性でもvalueが滑らかな場合はあり、このguardは必要条件ではない。

人工n2/rank3/LD2では共通gaugeに沿って異なるJ解を得ても、value/U derivativeが一致した。一方、head二つを同じbasisにした追加の非一意例は`REDUCED_GRADIENT_UNQUALIFIED`として停止した。これは不適切な微分を返さないことを示し、その例のvalueが必ず非滑らかという判定ではない。一般の境界例として、min_{j∈[-1,1]} uj=−|u|はu=0で非一意かつ非滑らかであり、独立のtoy testで区別した。DFがこの有界toyと同じ問題とは扱わない。Value関数の微分は最適解・正則性の条件に依存する。[Ito, Value Function Calculus, §2](https://kito.wordpress.ncsu.edu/files/2021/07/vfc.pdf)。

PRのJ一次資格が未達、追加null/gapが不明、再精密化後のgradientが不安定な場合は、滑らかなreduced gradientを返さない。実H4のA_headを今回は数値評価していないため、**PRでこの条件が通るかは未解決**。

### 条件が通らない場合の代替

既存JAX loss/penaltyを、局所変数(η,x)へ同時に接続した`coupled_value_gradient`を人工検証した。SciPy BFGSでjoint blockを更新し、終点Uで既存J solveを行って物理一次条件を測る。argminの微分公式を使わないため、PRの追加非一意性がある人工例でも目的を減らせた。新しいridge、PSD、sign、rank制約は導入していない。

これは合理的な次候補だが、inexact joint blockにもconditioning・chart・内側再solve後のreboundは残り得る。今回のH4契約はguard付きreduced pilotだけとし、**不通過を見てjointへ自動変更しない**。その場合の次契約の判断材料として人工結果を残す。

## 5. 実装の差分と再利用

| 既存資産 | 再利用と今回の差分 |
| --- | --- |
| `c_separated_j_u.py`: Coefficients, state, retract, _u_factory | 全J map/adjoint・penalty・isometry・局所右expm・JAX64・独立物理一次診断を無変更で再利用 |
| `c_separated_j_u_consistent.py`: solve_j/CoefficientChart | PR Newton-CG、affine residual、解析Hessian、全方向metric、raw pre-step jacの正しい検証を再利用 |
| 既存RC2 augmented gelsd | 1 solveの後、同じaugmented系のresidual correctionをもう1 solve。残差のみlongdoubleで累積し、LAPACK入出力はbinary64。追加L2なし |
| 既存residual_jacobian | 同じchart/同じJ点でreverse/forward AD chain ruleを照合。別U点の微分との同一視をしない |
| SciPy1.14.1 BFGS | 外側用に使用。各trialで内側を解き直すoracleを新設、既存fixed-J TRFは本番経路に使わない |
| 修正版pilot.native_diagnostic | 実AD decode点の同一点J-gradient、元candidate/decodedのcodec忠実性を別々に検証。原candidateを置換しない |
| 既存snapshot/Calls/launcher/STOP/JSON、SectorCAR/rank_assessment/representation_gate | 分離した両環境・元入力・保存初期資格・独立sector DF予算へ再接続。新しいapproval機構なし |

旧SOURCEを編集せず、新pilotは旧worker/parentの適合部分を継承する短いscope固有接続である。Budgetと記録形式が異なるため、新entry/新preflightを用意した。DF reconstruction、Gaussian合成器、optimizer、ADや数値計算libraryを新規に作っていない。

各新ηは、元のJから内側solve→同じUでの再精密化→独立state→envelope admissionという4 logical units。PR内側は同じNewton-CGを二度、RC2はgelsdとresidual correction各一度である。二度目は事前登録した精密化であり、結果を見て追加するretryではない。前試行の不正確解に依存するwarm-startは使わない。

Callbackはcache済み点を使用し、新しい内側solveを隠さない。SciPyにはcallable gradientを渡し、BFGS line searchのvalueもreduced valueである。Accepted点ごとにsigned非増加を確認し、trialとaccepted labelを区別する。Chart gradientが小さくても、現在Uのη=0で再評価した全物理接空間gradientが未達なら一次成功にはしない。[SciPy1.14.1 BFGS options](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-bfgs.html)。

各内側の生x/z/jac、metric、対応するjac評価点、solver flags、VG/Hessian回数、線形solve数、値と時間を保存する。Outerがcallbackから停止した場合はSciPyの最終raw結果が得られないことをnullと理由で明記し、取得できたaccepted/trialと独立gradientを保存する。架空のsolver successを作らない。

## 6. 人工比較と限界

全て人工tensor、seed41、norb2/rank3/LD1、f*=M*=P0=1、α=γ=.1、ε=1e-8。分子の保存配列は使わない。交互対照は旧修正版の4 cycles、reducedとjoint対照は各最大64 outer iterations。**古典評価budgetは同一ではない**。

| Arm / 方法 | F | 終端J∞ | 物理U₂ | 状態 |
| --- | ---: | ---: | ---: | --- |
| RC2 交互4 cycles | .01530143917 | 2.98e-16 | .006292575 | 上限、未資格 |
| RC2 reduced | .01510539587 | 4.16e-17 | 6.57e-6 | 両一次MET、22 points/44 inner solves |
| RC2 local joint＋J refit | .01501432553 | 人工記録参照 | 1.29e-8 | reducedより低いFの対照 |
| PR 交互4 cycles | .005044457234 | 5.69e-10 | .004265850 | 上限、未資格 |
| PR guarded reduced | .004902096349 | 9.90e-11 | 6.73e-6 | 両一次MET、12 points/24 inner solves |
| PR local joint＋J refit | .004902096276 | 人工記録参照 | 1.83e-10 | 別対照 |

これらは**同じ目的内での技術比較**。Reducedが交互4-cycle対照より改善する例はあるが、RC2ではjoint対照がさらに低いFに到達した。Reducedの常時優位・大域最適性・速度改善・実H4への一般化は主張しない。両armのraw F同士も順位付けしない。

加えて、強い交差曲率を持つ解析toy F(j,u)=a(j−u)²/2+b(j+u−2)²/2、a=1e8、b=1e-3を使った。固定JのUをSciPy TRFで解き、Jを正確にrefitすると、3回連続のF変化が元stall絶対閾値1e-12未満でもU gradientは元一次閾値1e-5より大きい。一方、reduced BFGSはu≈1へ進めた。これは**登録閾値で交互更新が停滞する機構の人工反例**であり、H4そのもののmodelではない。

失敗例も検証した：PR追加null、inner maxiter1でJ未達、極端に厳しい人工refinement基準、max_points1／outer iteration1による未収束、false outer gradient、forward/reverse不一致、非直交・nonfinite、input不一致。これらは未資格またはcorrectness STOPになり、無断の再solve・条件変更でPASSへ変えない。再精密化試行の誤差差や最大値を厳密boundとしていない。

## 7. テスト・資源・技術的な限界

Fit環境159 items＋fit専用worker回帰2 items＋評価環境25 items＝重複のない186 items PASS。新core21＋新pipeline7＝28 items。後続の同じ21件の再確認は合計へ重複計上しない。

| 検証 | 記録 |
| --- | --- |
| RC2内側独立解／PR凸性／isometry／objective保持 | [fit log](../../artifacts/research_c_j_u_reduced/2026-11-02/technical/fit_tests.txt) |
| Envelope有限差分、各±UのJ re-solve、感度/Schur、非一意性、交互停滞と失敗 | 同fit logと[new tests](../../tests/management/test_research_c_j_u_reduced.py) |
| Pre-step jac・codec実AD同一点・元J/U不変 | 既存point semantics/consistency repair回帰、同fit log |
| 新親子E2E、実人工sector/CAR、PR guard停止、partial correctness STOP | [evaluation log](../../artifacts/research_c_j_u_reduced/2026-11-02/technical/evaluation_tests.txt)：33 COMPLETEの未資格正常終了、表現FAIL時29 unitsでpartial STOP |
| 保存schema・SOURCE/input/entry・fresh・未知下流null | 新pipeline test、両環境passive preflight |

評価環境へffsimが必要な旧worker試験2件を最初に混在させ、ModuleNotFoundErrorとなった。環境選択の誤りであり、該当2件を既存fit環境へ分離してPASSした。[初回log](../../artifacts/research_c_j_u_reduced/2026-11-02/technical/evaluation_initial_environment_mismatch.txt)／[分離後](../../artifacts/research_c_j_u_reduced/2026-11-02/technical/fit_worker_regressions.txt)。共有venvの追加install・変更はしていない。

利用可能affinity32 CPUs、空きmemory約56GBを確認した。内側はsmall dense augmented SVD／既存A,T cache、UはJAX shape cache、cache済みηはsolve重複なし。将来は逐次依存の内側/line searchを守り、独立2 armを既存resource plannerで並列化する。BLAS1は二つの小行列workerの過剰並列を避けるprocess内設定で、CPU/RAM/worker/wallの人工quotaではない。JAX CPUを含む既存環境を使い、共有設定・他ユーザーjobを操作しない。

## 8. 未承認H4最小科学契約

[契約JSON](research_c_j_u_reduced_contract_unapproved_20261102.json)を実行条件の正本とする。今回は技術準備のみ。

SOURCE manifest SHA256: `631824b62bb0ef0112670a3fe1a9ffdc8cf451cce738e8e686f91cd15bc86949`。契約SHA256: `377db724f81353d47f7d38c7ece5cde3219cd8f53be24eabdfc5103503f06343`。原SOURCE133/input25・旧raw1825と旧SOURCE11 manifest群のbyte照合は[postflight](../../artifacts/research_c_j_u_reduced/2026-11-02/technical/preparation_postflight.json)を参照。旧grant/ledger/marker/STOPや既存science条件を変更していない。

| 固定物 | 条件 |
| --- | --- |
| 対象 | 保存直鎖H4、1Å、STO-3G、Nα=Nβ2、sector36、number CDF8、元frame順LD2 |
| 変数 | 全signed実対称J、実直交U。Head固定・tail tensor保存等式・PSD/sign制約なし |
| 同じ元input SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 同じ元qualified J/U/x0 SHA256 | `78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856` |
| 元SOURCE | prepare snapshot SOURCE e1085c95f5c01b68566e17d543ace5832cb94061 |
| PR / RC2 | 元smooth mass二乗γ=.1 / 全ordered J L2 α=.1、ε=1e-8 |
| f* / M* / P0 | 2.1839632053132894e-9 / 2.2093719370722074 / 4.410884790420646 |
| 物理条件 | 元g補正κ・scalar一回、norm＋既存engineering guard≤CA/8=.00019875Ha。CA=.00159Haの厳密保証ではない |
| アルゴリズム | 上記guard付きreduced、各arm独立1 trajectory、ffsim joint fitter呼出し0 |
| 内側 | 各新ηで2 solves、PR Newton-CG maxiter1000/xtol1e-12、RC2同じaugmented gelsd＋residual correction |
| 外側 | BFGS最大64 iterations、最大128 distinct points/arm、最大256 J solves/arm |
| 新しいfresh output | `artifacts/research_c_j_u_reduced/2026-11-02/run01`、未作成 |
| Retry / resume / 自動fallback | 0 / 0 / なし |

| 科学call群 | 各arm / parent | 両arm合計 |
| --- | ---: | ---: |
| Worker input/setup/元x0 identity/保存scalar history | 4 / arm | 8 |
| Inner J solve（geometry＋初solve128、refinement128） | 256 / arm | 512 |
| Independent physical J/U state | 128 / arm | 256 |
| Envelope/構造/gradient stability admission | 128 / arm | 256 |
| Outer BFGS driver（nested callとは別） | 1 / arm | 2 |
| 終端same-point/codec・tensor・export | 3 / arm | 6 |
| Parent input/初期資格/context/physical、最終load/norm/representation各2、比較 | 11 | 11 |
| **最大** | **520 / arm＋11 parent** | **1051** |

各inner solveをjournalへ別々に記録する。実際のVG/Hessian/linear solve回数、trial/accepted、wall/CPU/RSSも保存し、outer driver1 callを計算量の代用にしない。旧415/430/417-callの残り枠とは別の新scope。

### STOP・比較の解釈

SOURCE/input、不正値、objective/gradient、直交性、codec、CAR/general-K不一致はcorrectness FAILとしてpartial evidenceを保存し依存後続を停止する。J未達、PR構造未資格、gradient不安定はそのarmの登録未資格STOP。他armは独立範囲内で完了できる。未資格trialを採用せず、最後のaccepted pair（初回なら計算済み有限pair）は**診断用終点**としてのみ保存する。初期値への自動fallbackや代替solverへの変更はない。

`BOTH_FIRST_ORDER_MET`、`J_UNQUALIFIED`、`U_UNQUALIFIED`、`REDUCED_GRADIENT_UNQUALIFIED`、`OUTER_STALLED`、`OUTER_LIMIT_UNQUALIFIED`と、独立の`DF_BUDGET_NOT_MET`／`TECHNICAL_FAILURE`を区別する。Physical J/U一次の測定値、numerical envelope admission、solver raw flagは別fieldで保持する。小さいnormだけで大域最適性やPR性能を認定しない。

同じ目的内で、旧joint・旧分離v1・修正版分離の**保存JSON scalarだけ**と比較する。旧method再実行、旧配列の再decode、実H4 cross-Hessian sweepはこの最小契約に含めない。新両armの一次資格とDF予算が通らなければqualified tail comparisonはnull。通っても、同じ1-start条件での数値比較であり、強いRC-DF全体への一般的優位やnative資源性能へ拡張しない。

## 9. A〜Fへの回答と次担当

| 問い | 今回の回答 |
| --- | --- |
| A. 勾配再増大をどこまで説明したか | H_UJ ΔJとSchur曲率差の機構を人工AD/有限差分・解析toyで確認。実H4の支配要因/割合は未測定で、原因を一意に確定しない |
| B. RC2へ正しく実装できたか | 正L2の一意解、augmented解消去、各trialでのJ solve、独立FD・感度・両一次/失敗/STOPまで実装・人工PASS。実H4の数値条件では未検証 |
| C. PRへの条件は確認できたか | 共通head gaugeだけなら滑らかなquotientという十分条件を導出し人工確認。数値rank/inner/stability guardを実装。一般の非一意性と実H4適用性は未解決 |
| D. 条件不成立時の代替 | 元Fの局所J/η joint block＋J refit。人工の追加非一意例で減少を確認。新regularizerなし。将来に別契約で選ぶ候補であり自動fallbackしない |
| E. 人工問題で改善したか | 4-cycle交互DF対照よりFとU残差が改善し、解析toyでは登録stallから脱した。別budget・別局所解の比較で、joint対照より悪い例/未収束例も残る |
| F. 次に最小限何を測るか | 同じ元x0のPR/RC2各1 trajectoryで、毎trialのinner一次/再精密化gradient shift/PR rank gap、accepted F/物理U残差、actual solve数/負荷、最終独立DF norm/guardとexact λを測る。これがguard不成立か外側改善かを切り分ける |

次担当は**ユーザー／研究C**：公開SOURCE・1051-call未承認契約と、numerical admissionが厳密gradient保証ではない点を確認し、実H4限定pilotを別途明示的に許可するか判断する。**Codex**はその許可後のみfresh run01を一度実行し、規則通り公開・mandatory STOPする。今回の技術準備では新しい実分子候補は生成していない。

Commit/push後にremote SHA/tree・公開本文・SOURCE/input・旧原本の整合を照合し、**mandatory STOP**。追加最適化、実H4再解析、PF/RTE、ground、shots、native、総資源、重要科学レビューへ自動進行しない。
