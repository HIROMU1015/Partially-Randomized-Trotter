# 研究C：J/U分離型の停滞調査・曲率利用版の技術準備（2026-11-01）

## 1. 到達点と実行境界

旧結果の静的確認と人工問題による切り分けを行い、**PR Jを明示的なwhitened座標・解析Hessian付きNewton-CG、固定JのUを残差Jacobian付きTRFへ変更した実行可能なpipeline**を準備した。RC2のaugmented gelsd、目的、表現、初期化、独立の物理J/U一次基準、DF予算は維持した。

257 distinct人工・回帰methods PASS（新30、既存227、別途29 subtests）。ただし、**強いconditioningの人工例では、固定JのU勾配が改善してもJ再solve後に大きく戻り、64 cycles後もU未資格**である。結合問題の解消や実H4での性能改善を認定したものではない。

今回の実H4配列decode、新しい目的/勾配/Hessian評価、保存traceの新しい数値再解析、実最適化、PF/RTE、ground、shots、nativeはすべて0。保存診断の値は引用だけとし、実H4での新しい方向・step・曲率の数値は計算していない。新しい最大415-call案は未承認で、科学run01は未作成。科学許可・重要科学レビューを代行せず、公開照合後mandatory STOPする。

## 2. 必須引継ぎと確定している事実

[旧科学結果](research_c_separated_J_U_execution_20261101.md)、[旧技術準備](research_c_separated_J_U_optimization_preparation_20261101.md)、[旧契約](research_c_separated_J_U_contract_unapproved_20261101.json)、[既存集計](../../artifacts/research_c_separated_J_U/2026-11-01/execution_summary.json)、[旧result](../../artifacts/research_c_separated_J_U/2026-11-01/run01/result.json)、solver記録を静的に確認した。GitHubの起点4b86958d3の公開SOURCE本文も取得した。共通運用指示2026-10-25のworkspace原本、特に§8/8.4/10/10.1/10.3を実際に確認した。

旧科学SOURCEは`cbbb11e7e4f2a133bd53c1efdf59b1e02c4d5287`、結果commitは`4b86958d3109f1904b54d320ef8ba5e276587ad4`。保存直鎖H4/1Å/STO-3G、number CDF8、Nα=Nβ2/sector36、元順LD2、同じ元qualified x0・γ=α=.1の2 trajectoryで258/417 COMPLETE、correctness FAIL/retry/resume0。残り159 unitsは消費・転用しない。

| 既存保存値の引用 | PR-tail | RC2 |
| --- | ---: | ---: |
| 終了F（それぞれの自己目的） | .01226717465 | .02919345066 |
| J gradient∞ | 1.10244e-5 | 3.36198e-7 |
| post-J物理U gradient₂ | 2.954864 | .866764 |
| complete cycles | 11 | 64 |
| DF予算 | PASS | PASS |
| 同時一次資格 | 未達 | 未達 |

旧PR全12 J solveはraw status2。初回J0は独立一次MET、最終はNOT_MET。U全75 blockは各25 iterationのstatus1であり、収束を表さない。旧PRの最終U blockに保存された固定Jでの物理U勾配は.10743902517118696、RC2は.03818403260585299で、post-J値とは異なる。これは当該保存欄の引用であり、新しい差分・比・曲率の計算ではない。

## 3. U停滞の原因候補：どこまで分かったか（A）

右更新`U_new=U exp Ω(η)`、独立上三角Ω=η/√2というFrobenius-isometric座標は、人工n=2/6・det±1・π branchを含め、SciPy expmによる独立再構成、中心有限差分、JAX ADで整合した。η≠0でのchart勾配と、新Uのη=0で再測定する物理勾配も接空間へのchain ruleと整合した。勾配の大きさが変わることだけで座標誤りとは言えない。

考慮する要因は次の三つ。

1. 小さいf*でtensor loss曲率が強くなる。大きい勾配があっても、許される下降stepや目的減少が小さいことは数理的に両立する。人工の小loss-scale問題で再現した。
2. 旧L-BFGS-Bはidentity初期scaleと有限rankの曲率補正を使い、固定25 iterationで停止していた。status1はiteration上限で、gradient PASSでもline-search故障の証明でもない。既存traceには全探索方向/accepted step長/内部L-BFGS historyが保存されておらず、実H4のstepを一意に再構成できない。
3. 固定JでUを解いた後、Jを最適化するとU方向勾配が変わる。**J refitは同じFの非増加を保つが、U stationarityを保存する必要はない。** 「目的改善を相殺」と「U勾配が戻る」を区別する。人工例では明瞭な勾配reboundを確認した。

実H4のJacobian spectrum、下降角、line-search各trial、J/U結合Hessianの新しい診断は未実行。実H4の唯一の原因・実装欠陥・数学的停留点を確定したとは扱わない。

## 4. PR J precision-loss：確定と未特定（B）

既存解析gradient/Hessianは人工の独立contraction、有限差分、既存ffsim objectiveと整合した。固定U問題は凸だが、head等にflat方向があり、強凸/一意解を仮定しない。tensor curvature `AᵀA/f*`とsmooth massの`ε²/(c²+ε²)^(3/2)`曲率が併存する。

固定SciPy1.14.1のBFGS SOURCEでは、finite目的でのstatus2はWolfe line-search failureへ至る経路と整合する。これは近接したlibrary停止原因の説明であり、実H4でそのline searchが失敗した数値原因の特定ではない。公式BFGS文書の「有限差分勾配が精度不足」という典型例を、今回のanalytic gradientへそのまま当てはめない。初回Jのstatus2+独立METと、最終Jのstatus2+NOT_METを両方保持する。

小さいf*によるconditioning、Ax-gの相殺、変動するsmooth曲率、準Newton更新の精度が原因候補。人工でこれらへの感度を確認したが、実H4に新Hessian解析を行っていないため、根本原因の配分は未特定。閾値やεを緩めて警告を消していない。

## 5. 採択した最小変更（C）

### J：同じ凸目的をaffine座標でNewton-CG

既存`Coefficients`のisometric x=D upper(J)、A/T、作用/随伴、元のsmooth目的を使用する。旧SVD構造をそのまま利用し、

\[
B=[A/\sqrt{f_*};\sqrt{2\gamma}T/M_*]=L\,diag(s)V^T,
\quad W=V\,diag(1/\max(s,1)),\quad x=x_0+Wz.
\]

全方向を保持し、null方向を切り捨てない。Wは座標計量だけで、目的・Hessianにridgeを加えない。

`r=r0+(AW)z`、`c=c0+(TW)z`としてaffine residualを使い、反復ごとに大きいAxとgを引き直すことを避ける。既存のexact smooth HessianをAW/TWから直接組み立てる。`WᵀQW`の大行列相殺を経由しない。これはbinary64の再構成方法の変更で、高精度化の保証ではない。終了時には元座標の独立contraction/adjointへ戻して同値性・一次残差を判定する。

SciPy [Newton-CG](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-newtoncg.html)を1 block1回、callable gradient/Hessian、xtol1e-12、maxiter1000。旧BFGSのmaxiterを増やしたものではない。Newton-CGにもWolfe line searchがあり、precision-lossを必ず解消する方式ではない。xtol成功と物理J gradient基準を分離する。RC2のaugmented gelsdは無変更。

### U：同じ固定J tensor lossをTRFで解く

同じ右expm、J固定、元順・対称性・直交性を維持する。

\[
r(\eta)=vec(g_{DF}(J,U\exp\Omega(\eta))-g)/\sqrt{f_*},\quad
D=\partial r/\partial\eta.
\]

[JAX jacfwd](https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html)でexact residual Jacobianを生成し、D0の全方向SVDから同じfloor1のWを作りη=Wzとする。[SciPy least_squares](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.least_squares.html)のTRF/exactを使用する。これはGauss-Newton型の残差構造利用で、Uのexact Hessianではない。J固定時のpenaltyは定数なのでresidualから除外し、報告Fには定数を加える。新しい目的やrobust lossではなく、linear lossである。ここでexactは解析/ADの微分とSciPyのsubproblem方式名を指し、validated numerical enclosureではない。

max_nfev25、gtol1e-12、xtol1e-14、ftol=None。相対目的減少の早期停止を使わず、物理勾配はretract先で独立評価する。旧U maxfun500を増やさず、むしろcallback予算を小さくした。25 function evaluationsと25 iterationsは同じ指標ではなく、equal-budget速度比較とは扱わない。TRF status0は新しい登録有限上限停止で、旧L-BFGS-B status1とは意味が異なる。

J→U→J、mandatory J refit、最大64 cycles、stall3、J∞/物理U₂各1e-5と既存margin、DF予算は維持。Implicit differentiation、argmin微分、joint update、新gauge、polar補正、PSD/符号制約、追加penalty、rank探索はない。

## 6. 選定の根拠・再利用・実装差分

[選定記録](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/technical/solver_selection.json)。実データoption探索をせず、固定人工seed/scaleで数理的な候補を限定比較した。Whitened trust-exactは単blockで一次資格を得る場合もあったが、外側人工試験1件を進捗出力が長く得られず技術的に中断した（無限loopの証明ではない）。Trust-ncg、明示座標BFGSにも未資格例があり、解析Hessianを利用しつつ人工で良い元座標残差を得たNewton-CGを選んだ。万能なsolver順位ではない。

新規は[core](../../src/trottertracks/c_separated_j_u_curvature.py)、[pilot](../../src/trottertracks/c_separated_j_u_curvature_pilot.py)、[CLI](../../scripts/run_research_c_separated_J_U_curvature.py)と人工test。`Coefficients`、`state`、`retract`、元PR/RC2、ffsim real codec、JAX、sector/rawCAR/general-K、process launcher、JSON保存・journal/STOPを再利用する。旧core/pilot/契約は変更していない。

研究固有dispatchは旧pilotを基に別ファイルとした。新scope/policy/entryと、「旧配列を再評価せず既存同metricのJSONを再利用する」変更が必要なためで、旧moduleのglobalを書き換えたり旧SOURCEの実行を装ったりしない。新しい数値libraryやRiemannian optimizerを自作していない。

## 7. 人工・回帰検証（D）

[257 methods記録](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/technical/test_results.json)、[再現可能な人工比較](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/technical/synthetic_comparison.json)、[core tests](../../tests/management/test_research_c_separated_J_U_curvature.py)、[pipeline tests](../../tests/management/test_research_c_separated_J_U_curvature_pilot.py)。人工n=2/6のrotation、全signed J、off-diagonal metric、head自由、flat/wide/null、exact Hessian/chain rule/有限差分、J/U非増加、refit順、false success、警告+MET、nonfinite/不一致STOP、stall/64上限、DF資格の独立性を検証した。

人工parent＋実fit子processのE2Eは成功31 units、意図的表現不一致で27-unit partial STOPを確認した。旧original bytes/fresh/no resumeも検証。新pipeline試験の初回2 failuresは、広すぎるgit用mockが遅延importのmatplotlib fc-listも捕捉したためで、mockをgit rev-parseだけに限定して解消した。共有パッケージ・旧SOURCEを変更していない。

以下はすべて**人工seed41、norb2/rank3/LD1**。小loss-scaleは固定スカラーを借りただけでH4配列ではない。

| 人工control | 旧 | 新 | 含意 |
| --- | ---: | ---: | --- |
| PR 小scale・同じ初期Jのgradient∞ | 5.42777e-6 | 2.90072e-7 | 内側残差改善例 |
| PR 小scale・同じ固定JのU gradient₂ | 1.40709e-4（17 eval） | 6.40495e-8（3 eval） | U単block改善例 |
| RC2 小scale・同じ固定JのU gradient₂ | 1.76627e-4（39 eval） | 1.43983e-7（16 eval） | U単block改善例 |
| PR scale1・交互cycle | 16、両一次MET | 16、両一次MET | cycle短縮なし |
| RC2 scale1・交互cycle | 39、両一次MET | 39、両一次MET | cycle短縮なし |
| PR 小scale・終了 | J未資格、30 cycle | U未資格、64 cycle | J停止改善も全体収束せず |
| RC2 小scale・64 cycle U gradient₂ | 約9.94985、未達 | 約9.94985、未達 | 交互結合の制約が残る |

Soft controlでは新Uが25 evaluationsを使い、旧9/15 evaluationsより多い例もある。新Jがraw status2でも独立METの例も残った。人工例から実H4の速度・収束率・λ改善を推論しない。今回の主な技術的改善は、勾配定義/残差構造/曲率と停止原因を識別できること、および限定controlでのblock残差の改善である。

## 8. 新しい未承認H4比較案（E）

[固定415-call契約](research_c_separated_J_U_curvature_contract_unapproved_20261101.json)／[SOURCE119](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/technical/SOURCE.json)／[passive preflight](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/technical/preflight.json)。

- 実装commit `d1b37cd8e1cc9916a5d2948133d769be1d6119da`。
- SOURCE manifest SHA256 `6170207a0b86badd26bda1a0c46a5748d32045ddc8cfdd2edbbc6af5fb6d7f8e`。
- 契約SHA256 `4c631022c08b2d2a88f67ffbcc8ce960119bc228035c861ee79ce0933d7a9796`。
- 保存直鎖H4/1Å/STO-3G、number CDF8/sector36/元順LD2、元g補正κ/scalar、CA/8=.00019875Ha。元のγ/α=.1、ε/f*/M*/P0を維持。
- 同じ元qualified J/U/x0から独立PR/RC2各1 trajectory。旧joint/分離型終点をwarm startしない。
- 同じ環境を分離使用し、保存入力17ファイルを固定。旧joint同metricと旧分離型のJSON scalarを再利用し、旧J/U decode・新しい旧点勾配再評価は0。
- Fresh outputは`artifacts/research_c_separated_J_U_curvature/2026-11-01/run01`。未作成。Retry/resume0、結果を見たsolver/閾値/初期値/rank変更なし。

| 科学call | 最大 |
| --- | ---: |
| Parent初期読込/資格/sector/physical＋両final/DF/表現/比較 | 11 |
| 各arm：load/setup/initial identity/history/terminal/tensor/export | 7×2 |
| J solve（初回＋64 refits） | 65×2 |
| U solve | 64×2 |
| State診断（initial/J0/64 post-refits） | 66×2 |
| **計** | **415** |

Block内callback/Jacobian/Hessian回数は別記録する。新Jの生solver変数・勾配はz座標で、x=x0+Wzという物理係数も保存する。UのTRF trial一覧をaccepted iteration履歴と偽らず、iteration数は取得不能ならnull。SVD全特異値、eta norm、trialの方向微分、U固定J後とJ refit後の物理勾配、実wall/CPUを保存する。これらは必要な新solver内部の計算で、別のH4 Hessian scanや有限差分探索を追加しない。

次の実験ではQ1/J精度、Q2/U停滞、Q3/両一次資格、Q4/PR自己F・exactλ、Q5/RC2自己F・U残差、Q6/DF予算、Q7/旧joint/分離型との費用・限界を判定できる。J/U分類とDF予算を分離し、両候補の両資格がPASSする場合だけqualified tail比較を出す。一次PASSは大域最適性ではなく、PRとRC2のraw Fを順位付けしない。

## 9. 残る課題・資源・保全・次担当

既知の実装blockerはないが、**交互ブロック間の強い結合が残り、今回の改良だけで実H4の同時収束が得られる根拠はまだない**。新H4比較はblock精度の改善が全体へ波及するか確認する限定pilotである。Newton-CGの警告や元座標gradientの丸め誤差も残り得る。初期点/強度/rank探索やimplicit differentiationへ自動拡張しない。

JのAW/TWをblock内再利用し、U residual/Jacobianはshape-cached JAX graphを使う。将来のH4 Jacobianは256×48、J Hessianは80×80で、巨大なFock/controlled行列や回路compileを必要としない。独立2 armは空きCPU/quota/RAMに応じた既存plannerで並列化し、J→U→Jの依存は逐次。CPU/RAM/wallの恣意的上限、共有venv/cache設定や他者jobへの変更はない。

[技術postflight](../../artifacts/research_c_separated_J_U_curvature/2026-11-01/technical/preparation_postflight.json)で旧分離run269＋従来976=1245 raw、旧SOURCE/契約/grant/ledger/marker/STOP、新SOURCE119/input17を照合する。科学manifestへ新しい実分子性能結果は追加しない。今回の技術準備の記録だけを更新し、旧未資格結果を成功へ書き換えない。

次担当はユーザー／研究C：上の限界を含む固定415-call案を確認し、実H4限定科学実行を別途明示許可するか判断する。CodexはSOURCE・人工検証・契約の公開とremote SHA/tree/本文照合までを完了する。**照合後mandatory STOPし、今回の技術依頼から実H4再解析・最適化・PF/RTE/native/shots・重要科学レビューへ自動進行しない。**
