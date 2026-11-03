# M1-S γ依存V1完了結果：限定独立科学レビュー

- **レビュー日**：2026-10-17（JST）
- **対象**：Track B／M1-S（shared ± gate labels）のγ依存・保存有限step信号に対するTask A解析
- **開始承認**：ユーザーの「レビューを開始して」。対象は直前に完了したγ依存V1の結果解釈および次段階の判断。
- **重要度**：Level 2（GPT-6 xhighで十分）。既存研究方針・採択済み解析モデルの下での結果判断であり、研究全体の新しい正式方針の採択ではない。
- **対象repository**：[`HIROMU1015/Partially-Randomized-Trotter`](https://github.com/HIROMU1015/Partially-Randomized-Trotter)
- **公開固定HEAD／結果commit**：`b34572770e7c2105e16da003af050d789f490662`。branch `track-b-m1s-v1-gamma-completion-20261017` のremote HEADとの一致をGitHub connectorで確認。
- **評価**：`V1_CONDITIONAL_GAMMA_CROSSOVER_SUPPORTED / G3_G6_TRADEOFF_SCOPE_ONLY / G9_ABSOLUTE_GAIN_UNRESOLVED / INTEGER_SHOT_AND_NATIVE_RESOURCE_GO_NOT_ESTABLISHED / NARROW_TASK_B_RELEVANCE_GATE_RECOMMENDED_NOT_ADOPTED / MANDATORY_STOP`。
- **実施していないこと**：新しい科学計算、保存raw値の再集計・再ランキング・閾値への数値代入、独立H4行列／signal再生成、量子shot simulation、RPE decoder、native回路合成・compile、SOURCE修正、既存repoのcommit/push、契約採択、execution authorization/grant/marker発行。

> **要旨**：G3/G6の48対照中32対照について、保存半径を採用した**連続・十分shot設計の優劣が位相許容幅γによって反転する**という条件付き境界を確認した。G9の16対照は形式上shared優位でも、元の絶対visibility差が未解像のため科学的改善として認定できない。V1の成果は「事後的な位相–可視度トレードオフの条件図」であり、整数shot削減、実装可能なRPE、native回路費用・総資源利益の実証ではない。大規模V2やTask Bへ直行せず、**RPEで有効なγ領域との交差と、回路費用のbreak-evenを判別する最小のdecision gate**を先に契約設計することを推奨する。推奨は未採択である。

---

## 1. 研究目的と今回の判断問題

M1-Sは、Hadamard-testの正負branchで**同じ乱択Pauli labelを各位置で共有**し、負branchで符号を反転するcoherent randomizationである。負branchは同順sign reversalであり、全wordのadjointではない。shared ±によってcoherent平均信号のvisibilityを維持し得る一方、finite-stepの位相biasやcontrolled/native実装の追加費用が生じ得る。

直前までの成果は、(i) T1a/T1bの適用域を明示した解析、(ii) H4保存状態における有限step複素信号の比較、(iii) raw-H位相biasとvisibilityの交換関係、(iv) γ依存の十分shot設計の数理的条件である。今回のV1によって問いは、単なる「visibilityが上がるか」から次へ進んだ。

1. 採択済み**Task A**：保存raw-H理想平均信号の、各単一時刻における円周位相を所定のγ以内へ推定する条件付き、事後的なshot設計。
2. 将来の**Task B**：特定のエネルギー（最終的には分子基底状態エネルギー）を多round robust phase estimationで推定し、状態準備と回路費用を含めて既知の強い対照を上回るか。

**Task Aでの境界の存在をTask Bの実用資源GOへ読み替えてよいか**、また**次に投資するべき最小の証拠**が今回のレビュー対象である。

## 2. 固定資料と取得・検証の範囲

GitHub connectorで次の固定資料を実際に取得して照合した。リンクのcommitは、特記しない限り今回の結果commit `b3457277...` である。

| 役割 | 取得・閲覧した一次資料 |
|---|---|
| 完了結果 | [γ依存V1完了報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/docs/tracks/algorithm_codesign/m1s_v1_gamma_completed_results_and_stop_20261017.md) |
| 科学raw | [試行02 result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02/result.json) |
| 48比較 | [gamma_comparisons.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/gamma_comparisons.csv) |
| 元信号に基づくbounds | [method_bounds.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/method_bounds.csv)（存在と参照を確認。全数値再解析はしない） |
| 契約C2 | [採択契約 JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/execution_contract_adopted.json) |
| 実行・修正来歴 | [runtime修正記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/docs/tracks/algorithm_codesign/m1s_v1_gamma_runtime_repair_and_completion_contract_20261017.md) |
| 技術監査 | [saved_completion_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/saved_completion_audit.json) |
| 公開監査 | [publication_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/artifacts/track_b_m1s_v1_gamma_completion/2026-10-17/attempt_02/publication_receipt.json) |
| 実際の数学SOURCE | [`gamma_v2/boundary.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/677dd6957e0e1913b70ca5f999458835ae8a2e23/src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v2/boundary.py)、[`analysis.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/677dd6957e0e1913b70ca5f999458835ae8a2e23/src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v2/analysis.py)、[`bounds.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/677dd6957e0e1913b70ca5f999458835ae8a2e23/src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v2/bounds.py) |
| 前回の科学判断 | [V1科学条件レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/docs/research/m1s_v1_scientific_conditions_independent_review_20261017.md)、[RPE科学設計レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b34572770e7c2105e16da003af050d789f490662/docs/research/m1s_rpe_applicability_scientific_design_review_20261017.md)、[有限step結果レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b/docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md) |
| 共通運用 | プロジェクト添付 [`research_project_instructions_unified_2026-10-16.md`]（本レビューで本文を読了） |

今回確認した証拠固定：

- SOURCE S2：`677dd6957e0e1913b70ca5f999458835ae8a2e23`。
- 契約 C2：`2cb5160d9051a6568b8acc0cfb29fc169bdfca09`。
- 専用実行承認 A2：`161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c`。
- raw SHA256：`bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d`（保存報告・evidence indexによる）。
- 元の有限step入力 SHA256：`ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`。
- rawの `status=GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT`、`failure=null`、`scientific_GO=null`、`mandatory_STOP=true` を閲覧確認。
- GitHub blobと固定commitで実資料を直接取得したが、**本レビューで公開全45ファイルを再hashした、CIを再実行した、元H4を再計算したとは主張しない**。公開receiptは44件PASS/FAIL0を記録し、最終commitのremote HEADと一致した。ユーザーの「公開45ファイル」は最終receiptを含む数え方と整合する。

### 2.1 別試行と保全

初回試行01はPython 3.11.0rc1に`sys.set_int_max_str_digits`がなく、**科学入力decode前**に停止。ユーザーの技術修正・完了までの別試行承認の下、互換呼出しに変え、**数学kernel／比較アルゴリズムのbytesと科学条件を保った**SOURCE S2、再固定契約C2、専用authority A2で一度成功した。元の失敗・markerは消費済みのまま保存し、新旧試行の成果を混ぜない。この技術修正の経緯そのものはV1の科学的有効性の独立証明ではない。

## 3. 保存結果の正確なscopeと観測

条件はlinear H4 1.00 Å、STO-3G、8 modes、保存DF rank12、split G3/G6/G9。保存したδ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1を使用。**24非零cell×3方式=72方式行**、36零時刻controlは保存のみ、48対照を分類した。同じ保存入力を繰り返した診断であり、独立72物理分子・有限量子shotの観測ではない。

| split | 保存監査による対照分類 | 今回認定できる範囲 |
|---|---|---|
| **G3** | `UNIQUE_TRADEOFF_BOUNDARY` ×16 | 採択した保守的margin設計で一意の内点境界が存在。境界前はshared不利、後は有利。境界区間内の順序は未解像。 |
| **G6** | `UNIQUE_TRADEOFF_BOUNDARY` ×16 | 同じ条件付き数学的境界。有限時間・分割・保存stateに限定。 |
| **G9** | `ALL_COMMON_GAMMA_SHARED_HIGHER` ×16 | **形式的な保守的marginの順序のみ**。元の絶対visibility contrast `UNRESOLVED_ABSOLUTE_CONTRAST`を16件とも保持。物理的に意味ある改善とは言えない。 |

監査は`row_status_counts={READY_FOR_GAMMA_DOMAIN:72,SAVED_ZERO_CONTROL_NOT_REANALYZED:36}`、515 PASS／FAIL0、`independent_H4_boundary_numerical_validation=false`、監査内での新engine評価0と記録する。テスト38件、保全11件などは技術的に有用だが、独立した科学的再現試験ではない。

**観測から言えないこと**：何radの位相精度が現実に必要か、各γでの厳密な整数shot節約、回路一試行のnative費用、量子shotを実際に発生させた時の分布、基底状態energyの精度・confidence、強いPR/RTEや決定論endpointよりも優れること。これらは本V1では評価していない。

## 4. ショット十分条件の独立検討

同じ保存raw-H参照位相を目標とする各方式mについて、保存中心値（ハット記号）と条件付き数値半径から

\[
A_m^- =\max(0,\widehat A_m-r_{A,m}),\qquad
b_m^+ =\min(\pi,|\widehat b_m|+r_{b,m}).\tag{1}
\]

ここで`phase_bias_rad`は`TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H`であり、\(H_{\rm tilde}-H_{\rm full}\)のモデル誤差をすでに含む。モデル差を\(b_m^+\)にもう一度加算してはならない。\(r_{b,m}\)は計算上の角度不確かさであって許容位相biasではない。

真の物理的測定平均が保存理想signalと一致し、各軸の±1結果がshot間で独立な同一分布となり、\(A_m^->0\)、\(b_m^+<\gamma<\pi/2\)と仮定する。この**仮定された測定モデル**に限ればHoeffdingの二側集中不等式、X/Yの二座標円板と円周距離から、

\[
M_m(\gamma)=A_m^-\sin(\gamma-b_m^+),\quad
N_{m,a}^{\rm suff}(\gamma)
=\left\lceil\frac{K}{M_m(\gamma)^2}\right\rceil,
\quad K=4\log\frac{2}{\alpha_{\rm axis}}>0.\tag{2}
\]

`PER_METHOD_ROW`では\(\alpha_{\rm axis}=\alpha_{\rm row}/2\)、\(\alpha_{\rm row}=0.05\)。X/Yそれぞれに同一の十分shot設計を適用する。各方式行での95%保証であり、**72行同時の95%保証ではない**。保存数値のengineering coverage自体が統計的に95%となったわけでもない。

この式の正しさは既知の±1 Hoeffding boundと複素平面での円板幾何から条件付きで導ける。SOURCEの`analysis.py`は`A_lower`、unclipped bias、共通領域を構成し、`boundary.py`はその正の領域でmargin差だけを分類する。旧半径がplatform-certifiedでないことは残り、新算術が有理数区間であることだけでは解消しない。

**決定的な意味論**：`N_suff`は**同じ一つの選択された保守的shot設計**の十分予算であって、量子統計の最小必要shot数や情報論的下界ではない。片方の上界が小さいだけで、片方の物理手法が一般に優れていると証明したことにはならない。

## 5. γ依存境界の数学的意味と独立した紙上検討

同じcellのsharedと対照を比較し、

\[
a=A_{\rm sh}^->0,\quad c=A_{\rm cmp}^->0,\quad
u=b_{\rm sh}^+,\quad v=b_{\rm cmp}^+.
\]

既存レビューで定式化され、SOURCEが検査する直接のtrade-off仮定
\(a>c>0\)、\(0\le v<u<\pi/2\) の下で

\[
\begin{aligned}
\Delta M(\gamma)
 &=a\sin(\gamma-u)-c\sin(\gamma-v)\\
 &=X\sin\gamma-Y\cos\gamma,\\
X&=a\cos u-c\cos v,\qquad
Y=a\sin u-c\sin v>0.
\end{aligned}\tag{3}
\]

- \(\gamma\downarrow u\) では\(\Delta M=-c\sin(u-v)<0\)。sharedの大きいbiasの支払いが支配的。
- \(X\le0\) なら共通領域全体で\(\Delta M<0\)。このrouteではsharedの連続十分shot設計は改善しない。
- \(X>0\) なら\(\gamma\uparrow\pi/2\) で\(\Delta M\to X>0\)。また\(\Delta M'=X\cos\gamma+Y\sin\gamma>0\)なので、\(u<\gamma_*<\pi/2\) に**一意の境界**
\[
\boxed{\gamma_* =\arctan(Y/X)}\tag{4}
\]
が存在し、\(\gamma<\gamma_*\) 側はsharedの連続marginが小さく、\(\gamma>\gamma_*\) 側は大きい。逆向きのtrade-offはラベルを入れ替えて同じ議論を行う。

これは数理的に整合している。SOURCEでは\(X,Y\)を有理数区間で包み、区間による符号の判別、rootの区間 enclosure、共通領域外・同一bias・dominance等の分類を行う。G3/G6の`UNIQUE_TRADEOFF_BOUNDARY`は、保存入力に対するこの定理の仮定が確認されたという分類であって、新しい独立実験で効果が再観測されたという意味ではない。

### 5.1 なぜγ境界の存在だけで「良い位相推定」ではないか

(4)は、目標精度の選択がまだ与えられていない\(0<\gamma<\pi/2\)という**幅の広いTask A領域**での順序変化を述べる。\(\gamma\)を大きく取るほど緩い位相精度を許す。境界より上に領域が存在するだけでは、研究目的に必要な精度でsharedが有利とは言えない。またこの境界は、保存した\(A^-\)と\(b^+\)を用いた保守的shot設計の境界であり、真の最適sample complexityやnative実装の境界ではない。

### 5.2 天井関数・confidenceの扱い

\(M_{\rm sh}>M_{\rm cmp}>0\)なら、共通\(K>0\)を用いた**連続式**\(K/M^2\)はsharedで小さい。しかし採択済み実装の**整数**予算は

\[
\left\lceil K/M_{\rm sh}(\gamma)^2\right\rceil
\quad\text{と}\quad
\left\lceil K/M_{\rm cmp}(\gamma)^2\right\rceil.\tag{5}
\]

従って弱い不等式（\(\le\)）までは導けても、厳密な整数shot節約（\(<\)）は別条件であり、天井関数による同数があり得る。共通の\(\alpha\)は連続境界(4)の位置へは現れないが、\(K\)を介して厳密な整数差には関与する。本V1は固定γの数値N・N比を計算していない。

## 6. Task B／RPEへの橋：今回の結果が欠いているもの

### 6.1 phase budgetとの交差

具体的なRPE decoderを採択するなら、phase branchを正しく接続するための一round総角度予算が必要である。例えばKimmel–Low–Yoderの訂正を踏まえた、前回Decision Packetの保守的`nearest-consistent-branch` routeは\(\gamma_k<\pi/3\)を候補にする。ただしこれは**そのdecoderの十分条件**であり、Task Aの一時刻精度の普遍的必要条件ではない。

このrouteの科学的な第一関門は

\[
\boxed{(\gamma_*,\pi/2)\cap(0,\pi/3)\ne\varnothing}
\quad\Longleftrightarrow\quad \gamma_*<\pi/3.\tag{6}
\]

直接のtrade-offで\(X,Y>0\)なら、これは解析的に\(Y<\sqrt3 X\)と同値である。**これは紙上の条件変形であり、保存されたG3/G6の境界を\(\pi/3\)と数値照合した結果ではない。** さらに各roundの利用可能位相誤差はstate contamination・HD近似・synthesis・SPAM等を差し引いて狭まるため、(6)を通過しても有用性は保証されない。

### 6.2 Task Aのraw位相は単一Eの位相とは限らない

保存対象は正規化された保存stateに対する\(z_{\rm raw}(T)=\langle\psi|e^{+iH_{\rm full}T}|\psi\rangle\)の位相である。一般のguiding stateは複数固有値の重ね合わせであり、\(\arg z_{\rm raw}(T)\)が全roundで一つの\(E_*T\)に従うとは限らない。保存oracleの正規化、有限残差チェック、従来のT1b rateは、対象固有stateのcertified overlapや適切なpair-sum gapを自動保証しない。

Task Bの成立には、対象energyとprior・alias初期化、\(T_k=2^k t_0\)の全必要round、適切なstate weight／参照位相の橋、\(b_{mjk}^+ + q_{mjk}\le\gamma_k\)、最終\(\gamma_K/T_K\le\epsilon_E\)、union-bound confidence、正の物理signal下界などが別途必要である。保存T=1,4の間に必要なT=2がなく、このデータだけで一連のdyadic RPEを検証したとは言えない。

**従って、RPE routeの物理的な有用性の判定には、単なるγ境界の存在よりも先に「どのaccuracy taskで、そのγが必要か」を定める必要がある。**

## 7. G3/G6/G9の相違と科学的claim

### G3：visibilityの明確な増加と位相代償

前回のfinite-step保存レビューに引用された例（G3、R=2、δ=1/8 Ha^-1、T=4 Ha^-1）では、sharedのvisibilityは約0.955637、ordinaryは約0.923390。一方、raw-Hに対するsharedのprincipal phase biasは約−0.007856 rad、ordinaryは約+0.000355 radであった。これらは**以前の保存値の引用であり、本レビューで差を再計算したものではない**。visibilityだけを用いる誤った優劣判定に対して、γ依存境界は学術的に有用な診断である。

G3で16/16に境界があるのは、**採択した設計上は精度を緩くした側に改善余地がある**ことを意味する。研究上重要なのは、その改善余地が、実際に要求する誤差予算と接続するかである。現段階では具体的なRPE角予算との照合をしていない。

### G6：構造は同じ、物理的な改善量は別問題

G6も16/16境界があり、定性的にはG3と同じ。ただし前回レビューはG6の絶対visibility差が小さいことを記録している。今回の境界分類が一意でも、費用増加・state誤差・実装誤差に対して十分な**絶対的headroom**があるかは未判定。G3とG6を「同等の実用的改善」と扱わない。

### G9：連続指標の形式的優位を科学的改善へ格上げしない

G9の16件が`ALL_COMMON_GAMMA_SHARED_HIGHER`でも、以前の絶対contrastが数値半径内で未解像であることがそのまま保存されている。今回の数式は**保守的入力を代入した記述的順序**を判定したに過ぎず、G9で物理的に意味あるvisibility・位相・費用の改善を立証しない。相対比が大きく見える場合も、絶対差の不確かさを無視しない。

## 8. Native・状態準備費用を含む次の必要条件

共通精度と統計的成功基準を満たす実装済みscheduleに対して、研究対象のコストmetric \(g\) ごとに

\[
G_m^{(g)}=F_m^{(g)}+
\sum_{k,a}N_{m,k,a}\left(C_{m,k,a}^{(g)}+P_{m,k,a}^{(g)}\right).\tag{7}
\]

\(C\)はcontrolled/directional/random tail/HD実装/ancilla/readoutを含む全wrapper費用、\(P\)はshotごとの状態準備費用、\(F\)は同一単位で比較可能な前処理等の固定費用である。量子T-count、CX、RZ、depth等を単一数に無根拠に混ぜない。

単一時刻・X/Y価格同一・固定費なし・整数丸めなし等の限定proxyでは

\[
\frac{C_{\rm sh}+P_{\rm sh}}{C_{\rm cmp}+P_{\rm cmp}}
<\left[\frac{M_{\rm sh}(\gamma)}{M_{\rm cmp}(\gamma)}\right]^2\tag{8}
\]

が十分設計による費用break-even条件を与える。実際に整数shotを採用するなら右辺を\(N_{\rm cmp}^{\rm suff}/N_{\rm sh}^{\rm suff}\)とした限定比較を別途必要とする。**一方のshot十分上界が小さくても、真の最小shot数に関する一般的な優位性ではない**。

shared ±では形式的なbranch指数演算数がordinaryと異なるが、\(\operatorname{diag}(e^{-i\theta P},e^{i\theta P})=e^{-i\theta Z_a\otimes P}\)などの融合構造もあるため、枝数だけでnative費用を2倍と断定できない。反対に「共有だから追加費用なし」とすることもできない。実際のHD approximation、phase/synthesis誤差、state準備、強い部分乱択RTEおよびdeterministic endpointとの公平な比較が必要となる。

このため**現時点で広範なnative V2へ進む決定は科学的に早い**。少なくとも実際の位相目標に関連するγ領域と、そこで必要な費用比の余裕が存在するかを判別してから最小限のwrapper費用評価を設計すべきである。

## 9. 解析の信頼性・独立性・制約

1. **保存finite-step信号のcoverage**：従来のIEEE/complex BLAS/trig/log/underflow等のengineering条件に依存し、platform certificateではない。V1の正確な有理区間は入力の不確かさを消さない。
2. **数値独立検証**：38のtoy/metadata技術テスト、515の保存結果監査、11の保全監査は有用。だが監査JSONに`independent_H4_boundary_numerical_validation=false`と明記され、異なる科学SOURCEで元H4/γ境界を独立再生成してはいない。
3. **開発データと検証データ**：G3/G6/G9、δ/R/Tは開発・診断で用いた既知の同一H4データ。36 zero controlは新しいheld-out検証ではない。今後都合の良いγやcellを選んでも独立な一般化証拠にはならない。
4. **仮想shot仮定**：fresh-IIDで同じcoherent平均を持つ±1観測の存在・安定性は物理的なnative wrapperで要確認。保存pilot自体はquantum shotsを出していない。
5. **統計の適用範囲**：`PER_METHOD_ROW`のα=0.05は各行ごとの条件付き成功保証であり、複数方式の同時成功・最良rowの事後選択を95%で覆わない。72行family-wide inferential claimへ変更しない。
6. **精度・予算**：128-bit有理算術の境界区間が細くても、元signalのengineering不確かさより強い物理証拠にはならない。CPU160秒/wall320秒以内で今回完了した事実は、全入力に対する終了時間保証ではない。
7. **試行差**：初回のPython API互換障害は科学的negative dataではない。修正版で既存数学SOURCEを保ったことは、結果依存閾値緩和とは性質が異なる。ただし「独立一発成功」ではないことを保全する。

**科学的な誤りを直ちに示すSOURCE上の反例はこの静的読解からは確認していない**。これは全コードの形式検証やraw結果の独立数値再計算を実施したという意味ではない。V1の主要claimは上記の条件付き範囲なら整合する。

## 10. 先行研究・新規性・T1bとの関係

- **Hoeffding**：今回の有界±1 shot設計式は既知の集中不等式の特殊化であり、独立した新統計手法としての新規性はない。W. Hoeffding, *Probability Inequalities for Sums of Bounded Random Variables*, JASA 58 (1963), DOI: [10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830)。
- **RPE／訂正**：Kimmel, Low, Yoder, *Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation*, Phys. Rev. A 92, 062315 (2015), [DOI](https://doi.org/10.1103/PhysRevA.92.062315)。[2021 Erratum](https://doi.org/10.1103/PhysRevA.104.069901)は旧\(\pi/(2k_j)\)条件をエラー保証とする誤りを修正している。今回のTask Aの\(\pi/2\)領域はこのRPE branch保証の\(\pi/3\)と同一視しない。
- **既知PR phase estimation**：Günther, Witteveen, Schmidhuber, Miller, Christandl, Harrow, *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332 (2026), [DOI](https://doi.org/10.1103/ynxb-p2xq)、[arXiv](https://arxiv.org/abs/2503.05647)。部分乱択による位相推定・resource推定は先行している。M1-Sの新規性があるとすれば、**gate-labelを正負branch間で共有したcoherent平均の特定の構造と有限stepの位相–可視度trade-off**の範囲であり、そのpriorityや論文化十分性は今回未認定。
- **T1b**：T1bのsmall-step／長時間の減衰rate理論と、V1の固定finite-step phase crossoverは対象極限が異なる。V1のgamma境界はT1bのscalar rate優位の検証でも反証でもない。理論的な見通しの根拠を誇張しない。

先行研究は今回関連する一次文献と訂正の存在・主要scopeを確認した。関連する全先行研究の網羅的priority auditや、RPEの全定理の再証明を行ったと主張しない。

## 11. 次の研究方針：条件付きの限定ゲートを提案

今回のV1は、限定ノートに値する一つの科学的整理である。しかしこれ以上の資源改善路線への投資を直ちに正当化するものではない。以下の**一度限りの小規模decision gate**を、ユーザーの採否判断に供することを推奨する。

### Gate A：目標位相精度の意味を固定する

Task Bへ接続する具体的なdecoder、RPEで使える\(\gamma_k\)領域、将来のenergy許容精度・confidence、state contamination／実装誤差の予算を、**数値が得られる範囲と未決を分けて**仕様化する。Task Aのγ領域を、何の制限もなく実用RPEに持ち込まない。未取得のoverlap・gap・prior・中間roundを0円・既知としない。

### Gate B：保存境界と必要γ領域の交差

事前に定義した有意なγ領域に、保存V1境界の**shared有利側**が交差するかを確認する。例えば選択したnearest-consistent decoderの\(\gamma_k<\pi/3\)との関係なら、式(6)が一つの候補。ただし保存分数への新しい閾値代入・γ境界の数値再集計・ランキングは**別途の科学数値再解析承認が必要**であり、本レビューでは行っていない。

### Gate C：strict整数shotと実装費用のheadroom

そのγ領域で同一policyの整数設計が本当に減少するか、および(8)のnative価格比を満たす余裕があるかを限定評価する。将来選ぶ\(C/P/F\)・native metric・compiler条件は同等のbaselineを許し、先に保存結果へ合わせて恣意的に設定しない。技術的なwrapper costだけで分かる場合はCodexに限定実装させるが、**数値shot設計やnative compileはそれぞれ新たな実行承認が必要**。

### Gate D：Task Bおよび一般化に必要な独立性

前のgateで改善余地が残るなら、最小限のG3/G6＋適切な対照に限定してTask Bやnative V2の契約を設計する。新しい物理instance、状態保証、energy prior、complete round設計、held-out比較が必要か判定する。未使用の分子・分割を追加するのはこの時点であり、自動的には行わない。

### 意思決定規則の提案（未採択）

| 判定条件 | 科学的に妥当な次の対応 |
|---|---|
| 実用的に必要なγ領域とshared有利側が交わらない | **資源改善の追求を縮小／停止**。有限信号・trade-off理論ノートとして保全。ただし一般的NO-GOではない。 |
| γ領域は交わるが整数shot・費用の余裕がない | native大型評価へ進まず、理論・限定数値ノートに整理。手法の一般NO-GOとは呼ばない。 |
| γ領域が交わり、strict shot・native価格の合理的headroomが残る | 条件を固定した**限定V2**の検証を提案し、別途契約・実行承認後に実施。 |
| Task Bに必要なstate／prior／全roundの橋が依然未取得 | 仮にTask A利点があっても**energy RPEの資源GOは保留**し、必要な証拠を個別に回収。 |

以上のgateに新しい科学的閾値を数値で固定したわけではなく、一般的成功率や「何%以上の改善なら採択」といったmateriality thresholdを採択したわけでもない。特に、V1の48対照の中から結果を見て一番良いcellだけを選び、独立検証済みの最良M1-Sと呼ばない。

## 12. 他の選択肢と研究成果の着地点

1. **直ちに大規模V2へ**：不推奨。γの研究上の意味、RPEでの有効領域、1 shot費用が未固定で、回路費用を測ってもGO/NO-GO判断を返せない可能性が高い。
2. **直ちに完全Task B RPEへ**：不推奨。stateのspectral保証、prior、全dyadic round、decoder、実装biasが欠ける。保存T=1,4を完結したRPEと誤認できない。
3. **今ここで研究を区切る**：合理的な代案。T1a/T1b、固定有限step visibility positive、phase-bias trade-off、γ依存の十分shot境界を、適用域の狭い理論・数値ノートとして保存する。新規性と投稿適格性は別途審査。
4. **推奨：限定decision gateだけ継続**：科学的な前進を評価しつつ、次に必要な証拠が何かを先に確定できる。Gate B/Cが成立しなければ実用資源路線を深追いしない。

今回のレビューが更新したことは、以前は予測だった**γ境界の存在が保存H4・G3/G6の固定cellで観測された**点である。ただし、その存在は採択した数式と元のvisibility/bias trade-offから期待された現象であり、独立H4の新規観測や実用優位の成立を意味しない。これまで保留していた`END_TO_END_RESOURCE_GO`は引き続き未確立である。

## 13. 結論・承認とmandatory STOP

**科学的に支持する最終結論**：G3/G6で、既存の保存finite-step信号に対する保守的・条件付きの連続十分shot設計が、位相許容幅によって共有label方式と対照の順序を反転させる。G9は元の絶対visibility差未解像を保持する。これは事後的oracle診断に限定され、実機ショット・多round energy RPE・native費用・総資源改善へのGOは与えない。

**提案する次の作業**：ユーザーが限定decision gate（特にRPE-compatibleなγ範囲への橋）を採択するか判断し、採択するならCodexがまず非数値のtask/契約設計と証拠整備を行う。新しい数値閾値照合、保存結果からの科学的再解析、shot評価、native compileにはそれぞれ固定scopeと**別途明示実行承認**を必要とする。いずれも今回のレビュー承認に含まれない。

**保全状態**：SOURCE S2、契約C2、authority A2、元試行01のSTOP・消費済marker、試行02 raw結果、既存ledger/grant、親および他worktreeには変更を加えていない。レビュー記録はChatGPT回答用に別Markdownとして作成したもので、GitHub commit/pushや公式契約採択ではない。**Mandatory STOPを維持する。**
