# PR候補探索のための平均compiled価格評価と区間選別

2026-11-13 JST。**推奨は、外部で生成・評価されたwhole-schedule候補を、同時compiled価格区間によって選別し、順位を左右する未確定価格だけを精密化する方式である。** 固定L_D1の残差を全roundで高精度化することは先行要件にしない。構造Sと未検証残差は測定順序の参考に使い、最終除外・優劣の証拠には使わない。

[候補・N・価格のHANDOFF](HANDOFF.md)／[段階的探索と最小科学案](MINIMAL_SCIENCE_PLAN.md)／[SOURCE・テスト](SOURCES.md)。今回は設計と人工・回帰のみ。新実H4価格、回路build/compile、保存値の数値再解析/fit、新E[C]/Gは0。

## 1. 判断対象と再利用

候補xは分割L_D、PF recipe、finite-K、ランダム化、物理scheduleに加え、目標energy精度・全roundのN契約を持つ**RPE全体**である。状態準備は除外し、価格単位は測定を含むcontrolled full Hadamard X/Yのcompiled RZ/shotとする。

\[
G_x=\sum_{m,a}N_{x,m,a}\mu_{x,m,a},\qquad \mu_{x,m,a}=E_{P_x}[C_{x,m,a}].
\]

対象H/state/decomposition、energy target/prior/decoder・成功又はRMSE条件、price profileを共通の比較契約にする。異なるM/κ/L_Dの比較はwhole scheduleとして行い、roundごとに別候補の最小値を集めたhybridを新たな適格候補としない。Nの生成・資格はBとそのN供給担当、候補生成は探索担当が担う。Dはどちらも再実装しない。

| 既存成果・実装 | 今回再利用する部分と制約 |
|---|---|
| C `9d95d3ea`、全26 E[S] | 元H4 DF10/S2/L_D1/K2/元schedule・lawの構造平均と数値区間。別L_D・angle・law・境界へ数値を移植しない |
| C `37cbc331` structural_feature_means | supplied K2/nonidentity lawのcount閉形式とbasis小kernel。実H4 feature値は未取得。feature≠compiled price |
| D `39ddd59ed` P2 | L_D1のm0/m6有限pilot、技術同値性・性能の既存記述。新しい分散比較/fit/価格推定は今回しない |
| D `952ce011` outer-window | connected恒等式、target R/caps固定、未証明tailの明示。m12 E[C]はUNKNOWN。候補選別の必須経路にはしない |
| B `cdf6a806` | supplied区間のΣN b伝播、精度/law/rare・最大domain検証条件。今回の追加は候補依存の部分下界・差分・精密化方針 |
| 既存V2 assess/evidence.plan/cost/select | identity・N/RMSE・quantity・law・cohortとwhole-configurationの照合、exact会計。新候補探索器やN solverを追加しない |

## 2. パラメータを変えた場合の再利用境界

「コードを再利用できる」「保存数値を再利用できる」「科学資格が移送できる」を分ける。

| 変更 | 再利用できる技術 | 新しく必要なもの |
|---|---|---|
| N配分・energy精度・prior/decoderだけ | 物理回路/law/profileが同一なら価格そのものは同じ。Bのidentityはconfigurationを含むため自動matchはしない | whole-schedule N資格と、物理価格同一性を明示した読取専用reference bridge。旧recordを書換えない |
| L_D・fragment集合/順・one-body方針 | providerのprepare、PR tail/PF builder、Accountingの展開ロジック、fast fingerprint | 候補ごとのpreparation、head/tail、scalar、component/basis/law、構造state/境界の再確認。全26 E[S]とD係数は流用不可 |
| ΔPF、q、T、κ/R・piece allocation | 既存signed schedule・compose・測定wrapper・compiler/cache API | 元piece別時間/角度・order law・scalar Tに一致するrequest。q-only価格外挿やangleを外したcacheは禁止 |
| PF formula・signed piece列 | 実装済みrecipeのweights/allocation/順序付きcompose | 新recipeのword/境界とcontrol定義。現V2のPRはS2又は既存4th(m5_best)。標準Yoshida4の現拡張はdeterministic PF限定。未対応recipeはOUTSIDE_DOMAIN |
| finite-K・sampling法 | 一般finite-RTE libraryとfull-wrapper取得部品 | k2 feature/phase縮約はK2限定。別K/identity/相関lawには再導出、対象lawとnormal/forced strata、必要なjoint状態 |
| lowering/compiler/control/wrapper | プロファイル検査・canonical hashの仕組み | 別価格quantity/context。新旧loweringを一つのresidual/modelにpoolしない |
| H/DF rank/basis/qubit構成 | generic builder/compilerの対応範囲 | 新input/preparationと寸法別Accounting。H4固定wireのfast counter/kernelを一般系へ無条件適用しない |

**C1 Backendはgeneric候補runnerではない。** pinned `context(system)`はH4でprepare(ham,1)、S2/δ=.1のheadを作り、systemだけでcacheする。`compiled_price_fast`はこのcontextを継承する。他L_Dをcandidate欄に書くだけでは誤ったS/Cを取得する。将来必要なら、元prepare/compose/compilerを呼ぶ薄いcandidate-context adapterを新SOURCEで作り、preparation・recipe・angleをkeyにする。既存P2 runner/契約は変更しない。

C kernelのModelは324 component/9 basis、保存preparation、固定outer出口等を照合する。MeanKernelの合成/二乗法は再利用できるが、旧transition/reward/出口証明は別L_Dのものではない。identityが増えるとbasis reset分岐、空tailなら別境界、signed pieceなら順序付きstate伝播が必要になる。

## 3. 全roundを評価する方法の使い分け

E=全event数、L=全gate語長、s=十分性を証明した状態数、n=path数、J=特徴量数とする。

| 方法 | 費用・成立条件 | 探索と最終比較での扱い |
|---|---|---|
| metadata proxy・event/head/basis count | O(component数)、既存K2閉形式はqR列不要。新実期待値の評価は科学scope | 優先順位・予算の参考。proxyによる候補打切りは探索heuristicであり最適性証明にならない |
| E[S]の有限state reward | 概ねO(s³ log R + s³ log q)、pieceは順序合成。状態構築・parity・enclosureが必要 | 新候補の安価なcontrol variate。SをCの上下界にしない |
| normal pathの直接C平均 | n回の語処理/compile、O(nE)以上。全law平均CI・law差・rare保証が別途必要 | 実行可能な価格セルの直接参照。全候補/全roundで一律高精度にしない |
| paired D=C−S、E[S]+平均D | compile費は直接Cとほぼ同じ。Sの平均と標本が同じ定義/lawで、数値幅を加算 | Sとの相関で分散が減る場合に有利。Dの分散がCより小さい保証はなく、分散×path費用を比較する |
| analytic/少数feature残差 | E[feature]と係数・bias・domain・joint uncertaintyを扱う。単純countはO(J)、kernelは状態数依存 | 全使用domainの誤差根拠がある範囲だけ価格区間に昇格。旧P1/P2へ合うことは移送根拠でない |
| 短outer-window | window語O(hR)、差分重みとtail誤差。短windowの分散はqで増幅し得る | 長q tail根拠が未取得。現m12はUNKNOWNで、候補を落とす根拠にも使わない |
| compiled価格の直接上下界 | 0は普遍的下界。有限上界には対象compiler全体のpathwise証明又は平均の有効区間が必要 | 差が大きければ粗い区間で十分。Sやraw RZ、native instruction上限を、そのままC上限と呼ばない |

既存S区間[L_S,U_S]と同law/domainの根拠付き残差区間[d^−,d^+]があれば、
compiled meanは[max(0,L_S+d^−),U_S+d^+]に入る（空区間は証拠不整合としてSTOP）。
現在、この残差区間は新候補に未取得。
Sの小さい数値幅だけではcompiled価格区間は狭まらない。

raw size上界からのcompiler boundは将来候補になり得るが、basis translation/UnitarySynthesis/1q再合成を含む全passのsize増加を囲む証拠が要る。QiskitがRZ数を最小化するわけではない。現時点で新しい非自明なcompiler上界は証明・実装していない。

異なるlawの結合にも注意する。target ideal P、sampler Qなら
E_P[C]=E_P[S]+E_Q[D]+(E_P[D]−E_Q[D])。
actual Qをtargetにする場合はE_Q[C]=E_P[S]+E_Q[D]+(E_Q[S]−E_P[S])。
最後のlaw項をfinite-cohort SEへ混ぜたり0にしない。未知なら区間はその未知を残す。
対象lawのpathにcompiler失敗があれば、そのpathを価格0として除外・再seedせず、元の価格定義と実行可能性をUNKNOWN/STOPとして扱う。

## 4. 差が大きい候補を価格区間で除外する条件

以下は数式設計で、今回実N/価格へ代入していない。
比較契約とNが適格に固定され、使用する全price meanに**同時に**0≤L_j≤μ_j≤U_jが成立するなら、
G_xは[ΣN_xj L_j, ΣN_xj U_j]に入る。cell独立性は不要。
未測定セルは[0,+∞]として数学的に扱う（データ上のnull upperは「未取得」で、有限上界の証拠ではない）。

- 実行可能なincumbent bの有限上界U_bを得る。
- 候補xの下界L_x>U_bなら、xは厳密に劣る。**xの未測定roundの上界は不要**である。
- 大きな差γを要求する場合はU_b+γ<L_x。区間が重なる候補はUNRESOLVEDとして残す。
- 全候補を覆う下界とfeasible bがあれば、U_b−min_x L_x≤τはfinite menu内のτ-regret証拠になる。厳密最適性とは区別する。

価格が非負であるため、xの既知roundだけの和も下界になる。欠測を0価格として埋めた点推定とは違う。incumbentにも未被覆の正重みセルがあり有限U_bが作れなければ、この除外はまだできない。
ここで「部分」はwhole-round価格の一部セルを指す。回路prefixのcompile価格は、そのround全体のcompiled価格の下界ではない。
Nが未確定なら、B供給の根拠付き集合𝒩_xに対してinf/sup_(N∈𝒩_x)ΣN_j μ_jを囲む。
boxの場合は[ΣN_j^- L_j,ΣN_j^+ U_j]が保守的。未資格nominalをN^-又はN^+とみなさない。
Nを後で最適化する候補の下界は、その許される全Nを覆わなければ候補そのものを除外できない。
固定された実行可能Nの上界はincumbentの実行方策の比較に使えるが、それだけでその候補内のN最適性は証明しない。
τ-regretの最適値は同じ保証条件を満たすfeasible候補集合に対して定義する。未資格候補をinfeasibleと見なして集合から外さず、未解決として残す。

### 同じ価格quantityを共有する場合の差分

物理回路/law/profileの同一性を証明したprice quantityだけを同一jにまとめ、v_j=N_xj−N_bjとする。

\[
G_x-G_b\in\left[\sum_j v_j(L_j\;\text{if }v_j\ge0\text{ else }U_j),\;
\sum_j v_j(U_j\;\text{if }v_j\ge0\text{ else }L_j)\right].
\]

v_j=0の項は区間演算前に除く。未知価格でも同じmean・同じ重みなら差分では消える。
同じ回路unitary、同じN_ZZ、同じ観測価格、同じseedだけではprice量の同一性を証明しない。
Nが異なる共通セルは残る。差分下端>0ならxを除外できるが、共通未知項が残る各Gの絶対値を推定したことにはならない。全G取得が最終成果要件なら共通項も後で評価する。

## 5. 必要な精度と次に測るround

現在の競合組x,bについて、同時boxの価格区間幅の寄与は|v_j|(U_j−L_j)。
絶対GならN_xj(U_j−L_j)。これを「縮められる幅/取得費用」で優先し、長round順やSの大きい順にしない。Nが未資格なら記号式又は適格𝒩でのworst caseに留める。law/window/modelの縮まらない誤差は、MCを増やしても解決しない。

直接paired X/YにはZ_i=N_X D_Xi+N_Y D_Yiを用い、Var(Z)は2N_XN_Y Cov(D_X,D_Y)を含む。同じpathを2独立標本としない。候補間paired差を測るなら各候補の周辺lawを保つcouplingが別途必要で、元trial_seedはcandidate IDもkeyにするため、seed再利用だけで共通乱数法になったとはしない。

独立cell、根拠付きvariance σ_j²、1path費用k_j、固定MC分散予算Vなら連続配分の参考形は
n_j ∝ |v_j|σ_j/√k_j。これはCIでも固定nでもない。bias・同時被覆・正整数丸めを別に扱う。
共有modelやpathによる共分散があればvᵀΣvを使い、未知共分散を0にしない。
S/D/追加featureのcontrol variateは、同lawの期待値・同path計数・固定係数又は別校正が揃う場合だけ使う。

最大使用q/R・新L_D・signed/小角等のchallengeはmodelのdomainを問う作業で、N重みの大小とは別である。未証明の外挿で大きなgapが見えても除外しない。

## 6. 適応探索と実装上の到達点

推奨順序は、外部menu/資格→metadata不足照合→安い探索用proxy→同時価格区間→競合点だけ精密化→whole-schedule比較。τ、γ、confidenceは研究者が選ぶ。候補探索全体の最適性を言う場合は、heuristicで省いた候補も含む有限menuの被覆又は別の探索証明が必要である。

dataを見て追加cell/追加batch/停止時点を選ぶ場合、固定n CIの再観測を無制限に繰返さない。有限取得familyと同時α、又は有効な逐次区間を定める。独立固定batchごとにfailure budgetを配るunion法も可能だが、各区間自体の適用条件が必要。価格/Nのfailure budgetはRPE測定成功/RMSE契約と別に記録する。今回新統計libraryは実装しない。

既存Bのrpe_r_selectionはR-only・point record付きの完全menu、V2 cost/selectは点会計である。boundだけ・部分未知・signed差分による上記選別は**今回未実装の接続仕様**。既存APIを使った人工検証と数式確認に留め、Dで第二の候補/N/price controllerを作らない。B側で必要になったときにexact有理数の区間比較だけを最小追加する。binary64のjoint_uncertainty出力を巨大Nの最終区間証明へ流用しない。

今回の新12＋既存controller17/R-menu12/evidence13＝54人工・回帰test PASS。既存production codeは変更していない。C/B追加SOURCEはgit showで読解し、C branchを丸ごとmergeしない。新しい実H4計算・保存値再解析・E[C]/Gを行わず、既存SOURCE/input/raw/KEEP/grant/ledger/marker/STOPを保全した。

**次：探索担当**が有限physical menu、**B/N担当**がN資格又は許容集合、**ユーザー**が比較差・許容regret・confidenceと有限科学scopeを指定する。**D/C**はその後に必要な価格セル/特徴量だけを取得する。現在はcommit/push・remote照合後mandatory STOP。
