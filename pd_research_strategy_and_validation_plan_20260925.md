# P-D研究方針・着地点・本検証計画

**作成日：2026-09-25**  
**対象：HIROMU1015/Partially-Randomized-Trotter**  
**証拠基準：`9a494bd6bdda331f67609b5046514a505978904b`（`Complete P-D realization gate`）**  
**位置付け：研究方針の提案。リポジトリの正式仕様を変更した文書ではない。**

本書は前の再設計文書の「案D」を具体化・補正する提案である。正式採用する場合、今後の案Dの作業順は本書S0–S5を基準とし、完了済みD1–D3の条件・結果は変更しない。

本書では、既存P-DのD1–D3完了結果を出発点に、先行研究と重複しない主張の候補、研究として閉じる条件、そのための実行順と検証を定義する。新しい量子シミュレーションや回路compileは行っていない。既存artifactの選択規則・数値の再読解と、明示した条件付きの算術再計算を行った。

## 目次

1. [提案の要約](#s1)
2. [今回の再検討で修正すべき点](#s2)
3. [先行研究との境界](#s3)
4. [研究テーマとResearch Questions](#s4)
5. [対象タスク・範囲・比較契約](#s5)
6. [数理モデルと設計する選択法](#s6)
7. [比較すべきbaseline](#s7)
8. [着地点・完了条件・想定論文構成](#s8)
9. [実行順と途中の停止点](#s9)
10. [必要な検証の具体仕様](#s10)
11. [小さく始める実験設計](#s11)
12. [不確かさと判定規則](#s12)
13. [実装・成果物・Codexへの引継ぎ](#s13)
14. [今は行わないこと／方向転換の条件](#s14)
15. [出典と確認範囲](#s15)

<a id="s1"></a>
## 1. 提案の要約

### 1.1 推奨する中心テーマ

> **部分ランダム化によるエネルギー推定について、既知の絶対時間ベースの資源モデルがどこまでPF選択を正しく説明できるかを明らかにし、有限RTE・内部近似・実装コストによる補正が必要な領域に対して、精度と必要shot数を含む選択方法を構成する。**

仮題は、**「部分ランダム化位相推定における積公式と近似精度の共同選択――絶対時間モデルの成立範囲と有限資源補正」**とする。

新しいPF係数そのものの発見は、発展課題とする。最初の研究をそれに依存させない。逆に、単に既存5公式を順位付けするだけでも完了としない。既知モデルとの違いを説明する方法・適用条件・予測能力のいずれかを必要とする。

### 1.2 重要な修正

**負時間を含むtail絶対時間、RTEステップの比例配分、絶対時間の二乗に依存するランダム側負担は、元のPR論文Appendix A.3、Eqs. (A34)–(A40)ですでに扱われている。** したがって、これらをP-Dの新規発見にはできない。[W1]

さらに、今回のartifactの固定条件では、tail-awareが選ぶ公式は、精度を満たす中でstage proxyが最小の公式とも一致する。従って「誤差最小の8次より、必要精度を満たす低次が軽い」という現象だけでは、RTE特有の選択改善を切り分けられていない。[R2][R3]

D1–D3の通過は有効な実装・現象検証である。しかし、新規性やタスク資源の改善が確定したこととは別である。

### 1.3 何を確定し、何を保留するか

| 項目 | 本書の提案 |
|---|---|
| 中心の問い | 誤差・コストを考えた既知の選択を、有限・入れ子近似でどこまで改善できるか |
| 主な方法 | PFごとの再最適化、既知モデルとの段階的比較、必要部分だけの有限資源補正 |
| 必須の成果 | 既知baselineに対する有用な差分と、その機構または予測・判定方法 |
| 新PF係数の生成 | 必須にしない |
| H12・実機・noise | 本研究の共通要件にしない |
| 全RPE回路の直接compile | 必須にしない。ただしエネルギー推定の資源を主張するなら必要roundの集計は省かない |
| 主題としての最終採否 | 下記S0–S2の比較で判断する。P-Dの既存gate通過だけでは確定しない |

<a id="s2"></a>
## 2. 今回の再検討で修正すべき点

### 2.1 D1–D3で確認済みのもの

証拠はH4 chain、1.0 Å、STO-3G、8 qubit、4-electron sector、DF rank 12。PF候補5種、主判断delta 0.4、PF energy tolerance 1e-6 Ha、K=2、outer step当たりtail short-step総数64、各outer H_D occurrence当たり内部二次32 substepである。[R1][R2]

| 検証 | 確認したこと | 確認していないこと |
|---|---|---|
| D1 | 小行列における負時間finite-RTEの平均演算子、adjoint、controlled block、identity相対位相、sampling整合性 | H4のproduction Qiskit controlled回路での同値性・compile性能 |
| D2 | H_Dのfragment内部誤差を戻したL_D=3,4で、二つの選択規則が異なる | H_Rをfinite-RTEへ置き換えた同一H4演算子の精度 |
| D3 | 事前固定したfresh L_D=5でも選択差が残る | 別分子・geometry、PF別delta・内部精度の最適化、最終shot×cost |

初回v1は結果生成前に技術エラーで停止し、条件・閾値を変えずv2として再凍結した。この履歴を理由に結果を無効化する必要はない。ただし、実装テスト数は研究上の新規性や一般性の証拠とは分ける。[R1]

### 2.2 「energy-only」の対照を強くする必要がある

現行の`_realized_decision`は、固定deltaの実行可能候補からabsolute energy biasを最小にするものをenergy-onlyとする。tail-awareはlog-normalizationを優先する辞書式選択であり、総資源を直接最小化していない。[R3]

これはpilotで現象を見つけるためには使えるが、本研究では以下を区別する。

- 誤差をひたすら小さくする選択。
- 必要精度を満たした中で、stageまたは回路コストを最小にする選択。
- 既知のGamma_R依存まで含む資源モデルで選ぶ方法。
- 有限RTE・内部誤差・実装コストを含めた方法。

今回のartifactをそのまま読むと、次のようになる。[R2]

| L_D | energy-only | 精度内でstage proxy最小 | 現行tail-aware |
|---:|---|---|---|
| 3 | Morales 8次 | 新4次：1408 | 新4次 |
| 4 | Morales 8次 | 新4次：1792 | 新4次 |
| 5 | Morales 8次 | 二次：768 | 二次 |

**現3条件では、stage-awareな対照に対する追加の選択差はまだ示されていない。** これは新実験ではなく既存表からの判定である。今後、同じ表のenergy-only対tail-aware差を何度再現しても、この区別は埋まらない。

### 2.3 log Bの削減率をshot削減率と混同しない

L_D=5、delta=0.4のartifactでは、1 outer stepの値が

- Morales 8次：ell = log B = 0.00021636100180821876
- 二次：ell = 0.0000073996828276954015

である。[R2]

log Bの減少率は96.58%だが、他の条件を完全に固定し、減衰だけによる連続shot係数をB^2とした比は

\[
\frac{B_{\mathrm{2nd}}^2}{B_{\mathrm{Morales}}^2}
=\exp\{2(\ell_{\mathrm{2nd}}-\ell_{\mathrm{Morales}})\}
\simeq0.99958216.
\]

すなわち1-stepでは約0.0418%の減少である。stage proxyの6400→768、88%減とは桁が異なる。

L_D=3では、同様の比は約0.960221、約3.98%減となる。これはnormalizationのみの条件付き算術であり、総shot数、長round、RPE総資源の削減率ではない。長時間では反復数、r、K、許容統計誤差を再評価する。

### 2.4 固定delta・固定内部32 substepは、公平な最適化の終点ではない

高次PFは大きいdeltaを許す可能性がある。内部近似は各outer coefficientの時間幅に依存するため、一律32 substepが全公式に適切とは限らない。内部誤差がouter高次効果を隠している場合もある。

例として、L_D=5のMorales 8次は、H_Dをexactとしたときのbias約1.15e-14 Haに対し、固定32内部substepでは約2.78e-7 Haとなっている。ただし前者は浮動小数点floorに近く、真の次数や精度をその値から外挿してはならない。[R2]

本研究の比較では、PFごとにdeltaと内部精度を選ばせる。また、入れ子構成だけでなく、DF fragmentとH_Rへ直接高次PFを作用させるnative構成を適切な対照とする。新しい入れ子構成が、従来の直接構成より不必要に重いだけ、という可能性を先に除く。

### 2.5 「既知に還元されるか」を早く調べる

P-Aでは、DPという形式があっても、強い一区間baselineを入れると追加効果が消えた。P-Dでも、見かけ上異なる規則が既存の安価な選択に還元されないかを最初に調べる。同じ失敗を防ぐため、以下S0–S2を大型化前の明示的な停止点にする。[R0]

<a id="s3"></a>
## 3. 先行研究との境界

### 3.1 最も直接的な先行研究は元のPR論文

PR論文v2 Appendix A.3は、任意のp次PFでH_Rを一termとして扱い、tail occurrenceのsigned timeをdelta_j、絶対時間の総和をtilde deltaとする。Eqs. (A36)–(A40)はnormalizationの合成、絶対時間に比例するステップ配分、その二乗に依存する回数を示す。二次公式の前向き時間という利点にも言及する。[W1]

従って、以下は**既知の基礎として引用する**。

- 高次PFのtailに負時間が現れること。
- Gamma_R = sum_j |b_j|という絶対時間増幅の考え方。
- small-stepまたは上界モデルでr_jを|b_j|に比例配分すること。
- ランダム側負担がGamma_Rの二乗へ関係すること。

前回回答でこれを新規性の中心候補として強調したのは不十分だった。本書では修正する。

### 3.2 近い研究との比較表

| 文献 | すでに扱うこと | 今回の計画で残り得る差分／必要な対照 |
|---|---|---|
| Günther et al., PR論文 [W1] | 部分ランダム化、RTE、位相推定、高次PFのsigned time・絶対時間負担、資源モデル | 有限K、整数制約、実際の内部PF誤差、実装コストを含めると、既知モデルによる選択がどこで変わるか |
| Morales et al. [W2] | 長さ・次数の違うPFの公平比較、最適化係数、processing | 誤差最小ではなく、少なくとも時間刻みとstage costを考えた公式選択を対照にする。v1係数とv3の同名公式を混同しない |
| Hejazi et al. [W3] | 固有値推定に特化したPF誤差と公式 | operator次数だけで候補を評価しない。主張に直結するenergy向け公式を必要な範囲で対照にする |
| Casares et al., SPRINT [W4] | near-integrableなgroup別設計、内部subcycle、誤差寄与に応じた配分、random compilationの利用 | 「dominant部分を高精度にする」だけを新規としない。有限RTEの推定資源への影響を切り分ける |
| Cugini et al. [W5] | 回路費用と推定量varianceの共同最適化 | cost×sampling burdenという抽象原理を新規としない。今回はimportance sampling自体の開発を必須にしない |
| Hagan–Wiebe [W6]、Wan et al. [W7] | 複合simulation、ランダム位相推定 | 高次決定論と乱択の組合せや、RTEを位相推定に使うことだけを成果にしない |
| Bosse et al., THRIFT [W8] | エネルギースケール分離を使うsimulation | Hamiltonian分割を使うという一般的動機は既知。exp(H_D+H_R部分)などの実装可能性は別問題 |

### 3.3 今回、何が未確認か

本書は、上記一次文献の該当部分との初期照合を行った。全世界の文献を網羅し、「有限RTEと内部近似の全く同じ最適化が存在しない」と証明したものではない。

未確認項目を無視して大型計算へ進まない一方、文献探索を無期限にも続けない。S0の差分表に、各主張の最も近い文献・式・対象タスクと、今回追加する具体的な命題または方法を記録する。差分が既知の単純な代入だけなら、方法論上の新規性とは主張しない。

<a id="s4"></a>
## 4. 研究テーマとResearch Questions

### 4.1 主RQ

> **部分ランダム化による基底状態エネルギー推定で、PF次数・時間刻み・内部H_D近似・finite-RTE配分を公平に選んだとき、既知の誤差・stage・絶対時間モデルで十分な領域と、有限資源補正が候補選択を変える領域を説明・予測できるか。**

「最低biasの公式が最安か」は主RQにしない。異なる目的関数で異なる答えが出ること自体は、強い研究上の差分ではない。

### 4.2 副RQ

**RQ-A：内外の精度配分。** outer PFの高次性を活かすには、内部H_Dの精度をどの程度にする必要があるか。固定32 substepではなく、必要精度に合わせた配分がPF選択を変えるか。

**RQ-B：既知descriptorの適用範囲。** Gamma_Rに加えた通常の決定論costだけで選択できる範囲はどこか。有限K、整数r、tail occurrence数、回路境界のどれを追加すれば、必要な候補順位を予測できるか。

**RQ-C：実際の推定タスク。** 予測された候補が同じ精度・成功確率を満たし、選択に使わなかった小系条件でも低costまたは所定の選択損失内に収まるか。

### 4.3 期待する貢献の組

中心貢献は次のうち少なくとも一つを満たすことを目標とする。全てを最初から要求しない。

1. 既知の絶対時間モデルの十分性と破綻を説明する、計算可能な採否条件。
2. 全候補の高統計compileより軽く、同じタスクの良い候補を選べる有限資源選択法。
3. 内部誤差と外側PFの競合を利用した、実行可能な精度配分法。

これに、正しい同一条件評価と未使用条件での検証を組み合わせる。単なるgrid探索を「新しい最適化アルゴリズム」と言い換えない。

<a id="s5"></a>
## 5. 対象タスク・範囲・比較契約

### 5.1 最終的な目的量

基底状態エネルギーE_0を、指定epsilon_Eと失敗確率alpha_totのもとで推定する、という現行タスクを中心に残す。これを別のタスクへ変更する場合は、結果を分けて報告する。

ただし、既存pilotの`1e-6 Ha`はPF biasの受理条件であって、有限RTE・統計まで含む最終epsilon_Eを達成した結果ではない。これを本研究の最終精度へ無条件に流用しない。

最初の理論・小行列診断では、共通のinterrogation時間T、許容位相誤差beta、失敗確率alphaを固定した単一roundタスクを使える。しかし、この結果の着地点は「その信号推定タスクの資源」であり、それだけで基底エネルギー推定全体の優位性は主張しない。

### 5.2 最初に固定する範囲

以下は推奨する初期実行契約であり、ユーザーが既に決定した条件を追加するものではない。

- 入力DF Hamiltonian、電子数sector、fragment順序、threshold、identity policyをcandidate内で固定する。
- `L_D`は最初は説明変数。全prefix最適化を同時に始めない。
- outer formula、内部H_D formula・配分、base deltaを候補ごとに定める。
- 一つの候補では、全RPE roundに共通のouter PFと内部近似を最初は使用する。r,Kはround別に調整できる。
- 1-shot回路scopeは、状態準備を除くcontrolled interrogationとする。
- 主指標は現行のcompiled RZ countを維持する案とし、CX・depthを併記する。FT Toffoli/Tコストへの換算は別taskとする。
- topology-freeとcoupling付き、compiler versionやoptimization levelが違う結果は混ぜない。

### 5.3 入力状態・エネルギー窓

入力状態の利用可能性と、alias-freeなenergy windowを明示する。小系でexact基底状態を使うことと、実機でその準備が無料であることは別である。

また、exact Hの固有状態が、近似PFの厳密な固有状態であるとは限らない。対象分枝weightやsignal半径を候補ごとに確認する。過去P-Bの二次PFの良いweightを、新しい公式へ転用しない。

初期比較では状態準備costを除外できる。最終的な主張はそのscopeに限定し、必要なら共通準備cost Pを加えた感度だけを添える。準備法開発を必須課題へ増やさない。

### 5.4 比較する時間は共通にする

固定deltaはstage負担の診断には使えるが、最適PF比較の最終条件にしない。単一roundなら同じT、エネルギー推定なら同じepsilon_E・alpha_totに対し、公式ごとに反復数とdeltaを選ばせる。

round数が切り替わる点、alias制約、実測誤差の有効窓、探索境界を記録する。差が小さい結果で、片方のPFだけ大きいdeltaや精密な内側配分を許してはいけない。

<a id="s6"></a>
## 6. 数理モデルと設計する選択法

### 6.1 三層の近似を混ぜない

概念的にPF fを

\[
S_f(\delta)=\prod_j e^{-ia_{fj}\delta H_D}e^{-ib_{fj}\delta H_R}
\]

と書く。実際にはendpointのA/Bの有無やstage順序をregistryにそのまま保持する。

内部H_D近似をm_j substepで実装したunitaryをV_f(delta,m)とする。この時点ではH_Rをexactにする。

次に各tailをfinite-RTEへ置き換える。平均演算子をM_f、既知normalization補正後の演算子をF_fとする。一般にF_fはunitaryではないため、その固有位相を無条件に実効Hamiltonianのenergyと呼ばない。

分けて記録する量は、exact outer PFのbias、内部H_Dを戻したbias、finite-RTEを含む信号の系統位相差、信号半径、sampling uncertaintyである。符号付きbiasの偶然相殺も記録し、上界と実測を別欄にする。

### 6.2 有限RTEの量

偶数cutoff Kについて、paired Taylor表現のnormalizationは

\[
B_K(\tau)=\sum_{\substack{0\le n\le K\\n\ \mathrm{even}}}
\frac{|\tau|^n}{n!}\sqrt{1+\frac{\tau^2}{(n+1)^2}}.
\]

round mでq_m回のouter stepを実行する場合、同じstage配分r_{mj},K_{mj}を反復するモデルでは

\[
\ell_{f,m}
=q_m\sum_j r_{mj}\log B_{K_{mj}}
\left(\frac{\lambda_R|b_{fj}|\delta}{r_{mj}}\right),
\qquad A_{f,m}=e^{-\ell_{f,m}}.
\]

timeの符号はevent演算子と相対位相へ保持する。normalizationの引数が絶対値であることを理由に、負時間unitaryを正時間へ置き換えてはいけない。

K=0とK>=2はsmall-time係数や打切りbiasが異なる。Kを変えてBが小さくなったことを、誤差も減ったと解釈しない。打切り残差とnormalizationは別量である。[R3][W1][W7]

### 6.3 Gamma_Rモデルは既知baseline

同じ有限K>=2のsmall-step領域ではlog B_K(tau)=tau^2+O(tau^4)。固定T、total short-step数mathcal R、連続配分を考えると、Gamma_R=sum|b_j|により

\[
\ell_f\approx\frac{\lambda_R^2T^2\Gamma_{R,f}^2}{\mathcal R}
\]

という基準モデルになる。これと絶対時間比例配分はPR論文を出発点とする既知モデルであり、新しい定理として出さない。[W1]

重要なのは、このleading modelでは、Tとmathcal Rを固定するとdeltaを細かくするだけではtail絶対時間T Gamma_Rは減らないこと。一方、決定論側の反復回数はdeltaに依存する。この競合は説明用の基準であり、実際の整数配分、有限K、costの変化を自動的に含むものではない。

### 6.4 「normalization最小」ではなく「shot×cost」

正当化された基準半径下界rho_starと、normalization補正後の信号誤差etaがあれば、観測半径の保守値を

\[
\rho_{\mathrm{obs,lb}}=e^{-\ell}(\rho_{\star,lb}-\eta)>0
\]

と置ける。実際のshot式には採用した位相推定法の定数を使う。概念的には

\[
N_{m,b}\propto
\frac{\log(1/\alpha_{m,b})}{\rho_{\mathrm{obs,lb}}^2\sin^2\beta_{\mathrm{stat},m}}.
\]

これを用い、目的関数を

\[
G_f=\sum_{m,b}N_{m,b}\,\mathbb E[C_{f,m,b}^{\mathrm{interrogation,no\text{-}prep}}]
\]

とする。定数やfailure配分は両候補で同じprotocolへそろえる。

Rを増やせばnormalizationは通常軽くなる一方、回路が長くなる。従ってlog B最小だけを目的にしない。簡略モデルG(R)=(D+cR)exp(2A/R)ですら最適RはDにも依存する。これはcostと標本数の交換関係を示す初等的なモデルで、独立した新規性ではない。[W1][W5]

### 6.5 内部近似の誤差とouter次数

内部s次PFの局所誤差を粗く評価すると、outer occurrence jに対し

\[
\varepsilon_{D,j}^{\mathrm{ub}}
\lesssim d_j\frac{|a_{fj}\delta|^{s+1}}{m_j^s}
\]

という形を使える領域がある。係数と有効窓は検証が必要で、数値的な固有値biasと同一ではない。

符号付き和では相殺があり得るため、粗いnorm上界と実測energy biasを分けて保存する。固定m_jでouter p次がそのまま実現されるとは仮定しない。nested構成とnative高次PFも分ける。

まずはm_jの共通倍率を変えるだけでよい。stage別自由度は、その単純対照では足りない場合だけ追加する。SPRINTにはgroup別の次数・subcycle・誤差配分の先行例があるため、内部配分そのものを新規性とせず、今回の選択改善が何に由来するかを示す。[W4]

### 6.6 提案する選択法の形

候補fごとに、以下の三段階を使う。

1. **安価な第一段階**：誤差対delta/内部精度、stageコスト、Gamma_Rを使い、明らかに不適格・高コストな候補を整理する。
2. **必要な有限補正**：差が小さい、integer floorやfinite Kが効く、内部近似が支配する候補についてだけ、正確なB_K、誤差合成、shot数を評価する。
3. **局所compile**：候補順位を変え得る回路部分だけを直接compileし、選択を確定または未判定とする。

この手順自体も一般的な多段評価の考え方である。研究成果にするには、適用範囲を示す条件、選択損失の制御、または従来対照より少ない評価で同等以上の設計が得られることを示す。

finite candidate set上のbestを参照値にしてよい。無限の全PF familyでのglobal optimumを求める必要はない。代わりに、候補集合・探索範囲・境界を明示する。

<a id="s7"></a>
## 7. 比較すべきbaseline

| ID | 対照 | 目的 |
|---|---|---|
| B0 | 現行の誤差最小選択 | 旧pilotの再現用。主たる改善率の分母にはしない |
| B1 | 必要精度を満たす中でstage／1-shot costが最小 | 過剰精度を避けるだけの効果を除く |
| B2 | PRの絶対時間依存と比例配分を保持した係数-awareモデル | Gamma_Rを考慮したこと自体による見かけの新規性を除く |
| B3 | nativeなDF fragment＋H_R全体へ直接PFを適用する構成 | exact H_Dを後から32分割する入れ子化に固有の過剰負担を除く |
| B4 | 本研究の有限RTE・内部精度・実装costモデル | 提案する補正／選択方法の評価対象 |
| B5 | 同じタスクで再最適化した決定論PF | PR全体が決定論より有利と主張する場合に必要 |

### 7.1 B2を公平に作る

PR論文のRMSE定数を、今回のfailure-probability規約へ説明なく移さない。二つの評価を分ける。

- 先行論文の計算条件にそろえた再現。
- 同一の今回タスクに対して、既知の絶対時間・比例配分モデルを適用した係数-aware対照。

後者は先行論文そのものの忠実再現と呼ばず、「同一タスクに移した既知モデル」と記す。負時間の絶対値や整数ceilをわざと落とす弱い対照にはしない。

### 7.2 反実仮想と再最適化を分ける

**固定設計の寄与分解**：同じPF、delta、m,r,K,shotに対して、有限Bをleading Bへ置き換える等、一因子ずつ変える。

**設計選択の比較**：各モデルにそれぞれ設計を選ばせ、その選んだものを共通の詳細参照で再評価する。

前者はどの因子がcostへ効くか、後者は選択を誤ると何を失うかを見る。逐次改善率を単純加算しない。

### 7.3 選択損失

選択法Mが選んだ設計x_Mを共通参照で評価し、同じ有限候補集合の最良実行可能設計x_refに対して

\[
\mathrm{regret}(M)=G_{\mathrm{ref}}(x_M)/G_{\mathrm{ref}}(x_{\mathrm{ref}})-1
\]

を報告する。不適格な設計を選んだ場合はcostが安くても成功とせず、false acceptanceとして別記する。

候補集合にM自身の候補が含まれるからoracle regretが小さい、という循環的な説明を避ける。小さい問題では独立な全列挙を用意する。

<a id="s8"></a>
## 8. 着地点・完了条件・想定論文構成

### 8.1 最小の研究として閉じる条件

次の四つがそろった時点を最小着地点とする。

**(i) 主張が既知から分かれる。** Gamma_Rやnegative timeの再説明ではなく、既知モデルの適用条件・有限補正の必要条件・新しい選択法のどれを与えたかが明確。

**(ii) 一つの方法がある。** 単なる5公式の表でなく、別の入力にも適用できる判定式、配分アルゴリズム、または安価な予測法がある。

**(iii) 適切な対照と未使用条件で意味がある。** B1/B2/B3と比較し、候補選択、精度・成功条件、または評価に必要な古典計算量を改善する。現在見たL_D=3,4,5は今後の新規holdoutにはしない。

**(iv) scopeが閉じている。** 固定DF近似に対する結果なのか化学Hamiltonian全体なのか、RZかFT costか、単一roundかエネルギー推定全体かが区別されている。

この4条件を満たすことと、特定journalでの採録や学位認定は別である。論文・修士研究としての価値を結果前に保証しない。

### 8.2 目標の着地点

> **与えられたDF表現・PF候補集合・精度要求について、既知の絶対時間モデルで十分かを判定し、必要な場合だけ内部近似・有限RTE・実装costを補正することで、共通タスクの低資源な設計を選べる。適用できる領域と失敗領域を、未使用条件で説明・検証した。**

「常に8次が最良」「常に二次が最良」「PRは常に決定論より低cost」を着地点にしない。

### 8.3 否定的結果で閉じるための条件

B2の既知モデルで全評価が説明できた場合、詳細モデルの導入に新規性があるとは主張しない。ただし、次が得られれば別の着地点になり得る。

- 定義した領域で、安価なB2で候補選択が十分なことを支える新しい条件。
- 高次outerの利点を内部近似が潰す条件を、入力から予測する方法。
- 特定の新規提案が既知構成へ還元されることの一般的な説明。

「数点で追加改善がなかった」だけでは、これらを得たことにはならない。その場合はP-Dを独立主題にせず、既存研究の検証結果として保存する。

### 8.4 発展的着地点

新しいPF係数、stage別内部精度のより一般的な最適化、DF表現との共同設計、FT回転合成、実用分子へ進むことはできる。しかし最小着地点に後付けしない。

### 8.5 想定する論文の骨格

1. 問題：誤差・stage・絶対時間を考えても残る有限実装上の設計問題。
2. 既知理論：PRの絶対時間負担、energy-PF、nested/near-integrable設計。
3. 方法：採用した有限補正／精度配分／候補選択法。
4. 小系参照：平均演算子・誤差・signal・samplingを同時に検証。
5. 本結果：B1/B2/B3との差、原因、未使用条件、選択損失。
6. 限界：経験的PF入力、有限候補集合、未検証の大系・hardware。

中心図は最大5種類を目安にする。

| 図 | 内容 | 何を主張するか |
|---|---|---|
| 1 | 内外PF・finite RTEの誤差／資源関係 | 問題設定と既知モデルからの差分 |
| 2 | 各PFのcost–accuracy frontier（各PF内で再最適化） | 固定delta比較では見えない競合 |
| 3 | B2と詳細参照の一致・不一致領域 | finite補正の必要条件 |
| 4 | 内部誤差・有限K・整数配分・compileのablation | 選択差の原因 |
| 5 | 未使用条件のregret、feasibility、信号／成功率 | 方法の予測能力と範囲 |

lambda_Rだけで普遍的な最適PFを予測できるとは仮定しない。error structure、D側cost、fragment基底も変わるため、lambda軸の図には他の固定条件を明記する。

<a id="s9"></a>
## 9. 実行順と途中の停止点

以下S0–S5は本書の新しい実行段階名であり、完了済みD1–D3とは別である。

| 段階 | 作業 | 計算の性質 | 終了時に判断すること |
|---|---|---|---|
| S0 | 新規性・baseline・task契約の固定 | 文献・コード・artifactの整理 | 既知との差分を一文で言えるか |
| S1 | 公平な比較を最小範囲で作る | 既存再集計＋少数小行列・解析 | stage-aware／Gamma-aware／native対照後にも説明すべき差が残るか |
| S2 | 内部H_D＋finite-RTEの同時意味論を確認 | 小行列、production sampler、短回路 | 比較している候補が実際の同じタスクを満たすか |
| **Gate G** | **ここで一度停止** | **S0–S2の統合** | **P-Dの本研究へ進むか、scope縮小／停止するか** |
| S3 | 選択法・適用条件を構成 | 導出・有限候補参照・必要な較正 | 単なる全探索を超える方法・説明ができたか |
| S4 | 未使用条件と必要なtask統計を検証 | 小規模blind試験・限定compile | 一般化とタスクの妥当性を主張できるか |
| S5 | 主張を固定し論文・報告へ | 再現パッケージ・図表 | 完了条件を満たすか。追加scopeは別研究に分ける |

### 今からの最初の区切り

**S0とS1の結果まで**をまず共有する。S1で有望な候補が見えれば、その候補だけS2へ進む。S0–S2を通過する前に、別分子群や多数のPF係数へ広げない。

S1を通るために、無理にselection reversalを見つける必要はない。既知モデルで十分な領域がはっきりし、補正を導入すべき境界を予測できるなら、その内容を評価する。何も新しい方法・条件が残らない場合は、探索gridを広げ続けない。

<a id="s10"></a>
## 10. 必要な検証の具体仕様

### V0：主張・出典・baseline監査【S0】

**目的**：新規性のある主張と、既知理論の実装確認を分ける。

**手順**：PR Appendix A.3/Eqs. (A34)–(A40)、Moralesの公平比較、energy-PF、SPRINTの内部subcycle、資源最適importance samplingの各項目を一枚の差分表へ記す。baseline B0–B5のうち、主張に必要なものを固定する。

**成果物**：RQ一文、対象task、closest prior art、提案する差分、最小着地点、baseline registry、未確認の文献項目。

**停止**：独立した差分が「log Bを考えた」「高次PFを乱択と組み合わせた」だけなら、本格計算へ進めない。

### V1：既存P-D表の非循環的再比較【S0–S1】

**目的**：現在のselection reversalがどの対照に対して生じているか確認する。

**手順**：artifactを変更せず、B0、精度内stage最小B1、Gamma-aware B2、現行lexicographic tail選択を並べる。normalizationはlog値だけでなくB、B^2、time、反復数を併記する。

**現時点の結果**：B1と現tail-awareはL_D=3,4,5で一致する。log Bとshot係数の違いも2.3節で再計算済み。これを追加の大規模検証課題にしない。

**注意**：B2の正しいtask評価には次V2が必要。B1一致だけでP-D全体を一般的に棄却しない。

### V2：共通時間・PF別再最適化【S1】

**目的**：固定delta、内部32、R=64が選択差の原因でないかを調べる。

**最小手順**：最初は現在のH4 snapshotと既知splitから始める。各PFに同じTと許容phase/energy条件を与え、delta=T/qを各自選ばせる。内部substepは32の上下の少数候補から始める。r,Kは既知比例配分を初期値にし、有限式でfeasibilityを確認する。

**対照**：B1、同一taskへ移したB2、native PFのB3。PF候補は最初に既存registryを監査し、正式に同定できない係数を最良baselineと断定しない。

**観測**：各PFの最良実行可能delta・m・r・K、bias、半径、B、必要shot、stage／cost proxy、search-boundary flag、適用したmodel。

**判定**：固定パラメータでだけ生じる差と、再最適化後も残る差を分離する。高次PFが再び選ばれても失敗ではない。比較する最適化問題が正しくなったという結果である。

### V3：nested構成とnative構成、内部誤差の機構【S1】

**目的**：outer高次を使う意味と、内部近似の必要精度を切り分ける。

**比較**：exact H_Dのouter PF、固定内部精度のnested PF、調整可能内部精度のnested PF、DF fragmentとH_Rへ直接適用するnative PF。

**観測**：signed biasのdelta依存、内部substep依存、target weight、unitary defect、stage数。error floor付近のfitを採用しない。

**重要な実装事項**：同じH_Dの連続exact exponentialを融合してから近似する場合と、近似subcircuitを先に作って融合する場合は、一般に同じではない。構成順を仕様として保存する。対称性、負時間のadjoint、scalar phaseも維持する。

**進む根拠**：高次効果が保たれる必要精度、過剰内部精度の損失、またはnative/nestedの適切な使い分けを予測できる。

### V4：H4でfinite-RTEと内部誤差を同時に戻す【S2】

**目的**：D1 toyとD2/D3 exact-tailの間に残る意味論上の空白を埋める。

**手順**：V2で有望な少数候補について、内部H_D近似はそのままにH_R occurrenceをfinite-RTE平均へ置換する。小系ではfinite Taylor平均演算子を計算し、各stageの積を参照とする。production samplerの独立trajectory平均を同じ参照と比較する。

**注意**：有限平均演算子はunitaryとは限らない。誤差評価は必要なqのsignal・phase・radiusへ接続する。scalar normalizationを掛けたcontrolled meanは、control=0 blockまで同じ倍率が掛かる演算子ではない。測定信号としての補正と、diag(I,U)の意味論を分ける。

**観測**：負・正time、identity相対位相、normalization補正前後のsignal、operator平均残差、truncate bound、combined phase error、基準半径、sample SE。

**停止**：不一致ならまず技術的原因を修正し、旧結果を保持する。実装不具合だけで研究仮説を棄却しない。正しい実装で要求精度が破れるならcandidateを再評価する。

### V5：有限RTE補正の必要性を同じ設計で切り分ける【S1–S3】

**目的**：既知Gammaモデルで説明できない部分を特定する。

**比較因子**：無限／finite K、連続／整数r、共通／stage別内部精度、加法cost／境界を含むcost。最初から全factorialを行わず、差が出た因子の組合せだけ追加する。

**観測**：固定設計のcost差、各モデルが選んだ設計の共通詳細参照でのregret、feasibilityの誤判定。

**分岐**：Gamma-aware baselineで十分ならfinite-K新規性を主張しない。内部配分が主因なら研究の核をそこへ狭める。compiler境界だけが効くなら既知合成との重複を確認し、汎用的PF原理として広げない。

### V6：限定したproduction回路cost【S2–S4】

**目的**：stage proxyの順位が、宣言した回路scopeの実装指標へ残るか。

**手順**：選択判断を変え得る少数のf,delta,m,r,Kに限って、同じtrajectoryを用いたpaired compileを行う。符号・relative phaseを保つcontrolled意味論を先に確認する。

**観測**：RZ/CX count、RZ/CX/total depth、circuit size、生成・compile時間、MC SE、paired差、holdout残差。stageをgate数と呼ばない。

**停止**：候補差を決めない部分の高統計化は行わない。長回路proxyが必要なら適用domainと移送scenarioを明記する。compilerが違う値を一つのintervalに混ぜない。

### V7：同じ精度・成功確率のtask評価【S4】

**目的**：単一step負担ではなく、必要な推定taskを比較する。

**手順**：採用RPEの全roundについて解析信号と必要shotを集計する。小系ではmatrix power等を使い、巨大なqの一体回路を生成せずにsignalを確認できる。仮想Hadamardの測定分布から分枝復元と最終errorを評価する。

**重要**：各shotでfresh IID trajectoryを使う規約を維持する。乱択をreuseした別protocolへ変えるなら別解析が必要。roundごとにPFや内部近似を変える場合は、共通の基準phaseへの誤差評価を作り直し、同じUのpowersとして扱わない。

**観測**：最終energy error、失敗率とその上限、各roundのphase/radius、shot、cost、主な支配round、モデルと実測の差。

**区別**：信号の長q行列検証は長q回路costの検証ではない。必要roundの計算上の集計は可能でも、そのcostがproxy依存ならscopeに明記する。

### V8：独立条件への移送【S4】

**目的**：development条件を覚えた選択でないことを確認する。

**候補**：未使用のgeometryまたは小系を一つずつから始める。何を独立と呼ぶかを事前に定義する。既にP-C等で詳細に見たsnapshotを新規blindと呼ばない。

**凍結**：候補family、選択規則、較正方法、誤差予算ルール、主要評価指標、意味のある選択損失を結果前に固定する。移送時に必要なsystem-specific量を測り直すことと、結果を見て規則を作り直すことを分ける。

**判定**：改善、既知モデルで十分、失敗という3分類を保持する。一件の失敗で全仮説を棄却しないが、事後的に対象を選び続けて成功率を上げない。

### V9：係数registryと再現性【横断】

候補の係数、出典版、精度、stage順、fusion規則、operator orderとenergy orderを区別する。Morales v1/v3、8桁表示係数のnew4などを同一と仮定しない。既存D1–D3の定義は凍結したまま保持する。

結果はsource/input hash、snapshot、state sector、compiler、seed、task、選択／評価の区分を保存する。rawがGit管理外なら、第三者が何を再集計できるかを明記する。test成功は保存値整合・数理意味論・科学的主張のどれを検証したか分類する。

<a id="s11"></a>
## 11. 小さく始める実験設計

### 11.1 S1の設計案

最初は現在のH4をdevelopmentとして使い、splitを増やさず、PF×内部精度×deltaの一部を計算する。数値gridはこの計画だけで確定しない。

例として、代表PFを二次・新4次・Morales 8次とし、他の登録公式は安価な解析で除外できるか確認する。内部精度は既存32の上下を含む少数値、deltaは共通Tの整数反復条件から決める。RTEは既知比例配分を初期値とし、K=2だけを普遍的最適値として固定しない。

この段階で、全PF×全delta×全m×全r×全Kの直積を直接compileしない。

### 11.2 物理条件と解析用反実仮想を分ける

lambda_Rだけを人工的に変える試験は、係数依存の機構を切り分けるには使える。しかしH_D,H_Rの物理系が変わるため、状態・誤差係数を固定したまま分子の結果と呼ばない。解析的な感度図は「診断モデル」、実Hamiltonianの図は「物理instance」として別出力にする。

### 11.3 baselineへの探索予算

比較する方法に同じ最適化の機会を与える。全方法の同じパラメータ値を強制するという意味ではない。

例えば8次が内部mを変える自由度を持つなら、二次にも同じ種類の自由度を与える。ある方法だけ探索が粗い場合は、grid refinementの効果を別に示す。精度不足でbaselineが不適格になった点を、新手法の無限大改善として扱わない。

### 11.4 予算・時間

実行時間や必要sample数は、入力・実装・CPU環境を見ないで断定しない。最小pilotでmemory/timeを計測し、必要なprecisionからsample数を決める。ユーザーの締切や計算資源を本書で勝手に固定しない。

<a id="s12"></a>
## 12. 不確かさと判定規則

### 12.1 三つの不確かさ

1. 数値誤差：固有状態残差、matrix exponential、unitarity、floating-point floor。
2. sampling誤差：古典的なcost MCと、量子Hadamard shot。両者を混ぜない。
3. モデル誤差：PF係数外挿、finite-cost proxy、compiler/domain移送。

元の5% gate、20%burden gateを本研究の普遍的な成功条件へ流用しない。

### 12.2 選択に必要な精度で止める

候補間のtask cost差が不確かさより十分大きければ、全metricを同じ高精度にしない。候補差が不確かさ以下なら「同点」とせず「未判定」とする。順位逆転だけでなく、誤った候補を選んだときのregretを報告する。

同一snapshot・同一較正を全roundで共有する場合、その誤差を独立として平均しない。paired compileの差については共分散を保持し、元のcostのSEを単純に足した値だけで判断しない。

### 12.3 gateの凍結

各段階の期待task、技術的tolerance、最小の意味ある差／許容regret、未使用条件、再実行条件を結果前に固定する。数値はpilot負荷と要求精度から設定する提案であり、本書は未相談の一律閾値を追加しない。

技術失敗はfailed_technical、科学的に不利な結果はcompleted_negative、判別不能はundeterminedとして分ける。v1→v2のような技術修正では、変更前のsource/input/expectedを残し、結果後に合格基準を変更しない。

<a id="s13"></a>
## 13. 実装・成果物・Codexへの引継ぎ

### 13.1 再利用する部分

- `research_direction_energy_tail_pareto.py`：係数registryとtwo-block参照。
- `research_direction_pd_realization.py`：内部H_D構成、小行列負時間検証、既存decisionの再現。
- `rte.py`：finite distribution、normalization、打切り、sampling。
- DF/partial-S2/controlled/Hadamardの既存builderとcost provider：実際の支援範囲を確認して使う。
- 既存manifest・snapshot・state/sector hash・再開可能runner。

新しい比較は既存結果を上書きせず、別の計画・artifactとして追加する。汎用的な再設計を先に大量実装する必要はない。

### 13.2 まず作る文書

次のファイル名は**新規作成候補**であり、現在存在するという意味ではない。

- `docs/research/pd_primary_research_contract.md`：本書4・5・8節を1–2ページへ要約。
- `docs/research/pd_prior_art_and_baselines.md`：W1等の既知部分と差分、baselineの実装定義。
- `docs/research/pd_s1_fair_comparison_preregistration.md`：S1のtask、範囲、出力、停止条件。

正本の旧「全RPE総costが主目的」という説明は、P-Dの採否とscopeが決まった時点で更新する。技術backlogは残してよいが、全てが新主題の必須条件でないことを明記する。

### 13.3 結果の最小記録項目

```json
{
  "schema_version": "proposal_pd_fair_comparison_v1",
  "source_commit": "<actual full SHA>",
  "task": {
    "kind": "fixed_time_signal_or_energy_estimation",
    "target_energy_reference": "fixed_DF_Hamiltonian_or_original_Hamiltonian",
    "time_or_round_schedule": "<explicit>",
    "accuracy_and_failure_contract": "<explicit>",
    "state_preparation_included": false
  },
  "instance": {
    "snapshot_sha256": "<actual hash>",
    "sector": "<explicit>",
    "ld": "<integer>",
    "identity_policy": "<explicit>"
  },
  "candidate": {
    "formula_and_coefficient_source": "<explicit>",
    "construction": "native_or_nested",
    "delta": "<number>",
    "inner_substeps": "<array>",
    "rte_substeps_and_cutoffs": "<round-by-stage arrays>"
  },
  "evidence": {
    "outer_bias": "<number and status>",
    "combined_signal_bias_and_radius": "<numbers and scope>",
    "log_B_and_B_squared": "<per-round values>",
    "shots": "<per-round/axis integers>",
    "cost": "<value and metric/scope>",
    "numerical_sampling_model_uncertainty": "<separated>",
    "baseline_selected_by": "B0_B1_B2_B3_B4",
    "evaluation_reference": "<shared reference>",
    "holdout_role": "development_calibration_or_unseen",
    "boundary_flags": "<explicit>"
  },
  "status": "completed_positive_negative_or_undetermined"
}
```

これはschemaの説明例であり、文字列placeholderを数値として扱う実装ではない。

### 13.4 最初にCodexへ渡す実行方針

> 既存P-D結果は上書きしない。D1–D3を再実行することではなく、P-Dを研究主題として採用できるかを強い対照で判断するS0/S1を行う。まずPR v2 Appendix A.3で絶対時間・比例配分・二乗負担が既知であることを差分表へ反映する。現artifactからB0／精度内stage最小B1／現tail-awareを再集計し、既存3splitのB1とtail-awareの一致を記録する。次に、同一taskでPFごとにdelta・内部精度を選べる比較と、native対nested構成の対照を最小範囲で設計する。Gamma-aware B2を弱く作らず、finite correctionに固有の差分を切り分ける。task・最小grid・出力・停止条件を事前に固定し、合意した小行列・解析計算以外は開始しない。H12、全PF探索、長回路一体compileは行わない。S1の結果で一度停止して報告する。

<a id="s14"></a>
## 14. 今は行わないこと／方向転換の条件

### 14.1 行わないこと

全prefix、H12、広い分子群、全RPE回路の直接compile、noise、状態準備法開発、FT hardware設計、新しい全次数PF探索を同時に始めない。

新規性監査を新しいpilotのたびに無期限でやり直さない。主張とclosest prior artが変わる箇所だけ更新する。

「2次→4次→8次が精度順に選ばれるはず」と結果を先に決めない。内部誤差、偶然相殺、alias制約、離散round、tail弱さによって非単調な選択もあり得る。

### 14.2 Gate Gの分岐

| S0–S2で分かったこと | 判断 |
|---|---|
| 既知Gammaモデルを超える有限補正が選択・feasibilityを変え、その原因が説明できる | P-Dの本研究へ。必要な補正を中心にS3以降を実施 |
| 主因が内部H_D精度とouter PFの競合だった | 主題を「入れ子PFの精度配分と資源選択」へ狭め、RTEは応用または一因子とする。SPRINT等との差を再確認 |
| 詳細モデルと既知B2が一致し、安価な十分条件を作れる | 「既知モデルが十分な領域の判定」へ着地点を変更する候補 |
| B1またはB2で全て説明でき、新しい方法・適用条件が残らない | P-Dを独立主題としては止め、検証結果として保存 |
| 意味論の不一致・precision不足 | 技術修正／未判定。研究仮説の棄却とは分ける |

### 14.3 終点を後付けで拡大しない

方法、対照、未使用条件、scopeの4点がそろえば、H12や全実機stackを達成していなくても研究は閉じられる。逆に、これらがないまま系数やbenchmark数を増やしても、新規性は自動的には増えない。

**次の判断に必要なのは、「P-Dのselection reversalの再現件数」ではなく、「適切な対照を置いても残る説明・方法・予測は何か」である。**

<a id="s15"></a>
## 15. 出典と確認範囲

### リポジトリ（全て基準commit固定）

- [R0] [PROJECT_MAP.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a494bd6bdda331f67609b5046514a505978904b/PROJECT_MAP.md)：A/B/C停止とP-Dへの移行。
- [R1] [P-D現実化結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a494bd6bdda331f67609b5046514a505978904b/docs/research_direction_pd_realization.md)：D1–D3条件、結果、未検証scope。
- [R2] [P-D v2結果artifact](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a494bd6bdda331f67609b5046514a505978904b/artifacts/research_direction_pd_realization/2026-09-25/pd_realization_go_no_go_v2.json)：fingerprint `805a17f95497a4d61286748a126c01b1235fbe0d987528be86ea3938700b9ede`。configuration、d1_signed_time、d2_d3_internal_hd_splitsのdecision/rowsを照合。
- [R3] [P-D現実化実装](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a494bd6bdda331f67609b5046514a505978904b/src/trotterlib/research_direction_pd_realization.py)：`_realized_decision`、`_realized_outer_unitary`、`evaluate_internal_hd_split`。固定内部分割と辞書式選択を確認。
- [R4] [P-D初期Pareto結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a494bd6bdda331f67609b5046514a505978904b/docs/research_direction_energy_tail_pareto.md)：初期two-block研究の記録。本書の主数値はR1/R2を優先。

### 一次文献（2026-09-25確認）

- [W1] Günther et al., *Phase estimation with partially randomized time evolution*, [arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2), PRX Quantum 7, 020332 (2026)。特に[PDF](https://arxiv.org/pdf/2503.05647v2) pp.22–23、Appendix A.3、Eqs. (A34)–(A40)。本文画像でも絶対時間、比例配分、二乗負担を確認。Appendix D pp.34–35の二次／四次比較も参照。
- [W2] Morales et al., *Selection and improvement of product formulae for best performance of quantum simulation*, [arXiv:2210.15817v3](https://arxiv.org/abs/2210.15817v3) (2025 revision)。今回リポジトリの凍結係数版とは区別する。
- [W3] Hejazi et al., *Better product formulas for quantum phase estimation*, [arXiv:2412.16811v1](https://arxiv.org/html/2412.16811v1)。energy estimation向け誤差・公式設計。
- [W4] Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, [arXiv:2606.30741v1](https://arxiv.org/html/2606.30741v1)。SPRINT、特にSec. III.2、Eqs. (25)–(31)のnear-integrable・内部subcycleと誤差寄与配分、random compilationの位置付け。
- [W5] Cugini, Atif, Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, [arXiv:2603.13495v1](https://arxiv.org/abs/2603.13495v1)。回路実行costとestimator varianceの共同目的。
- [W6] Hagan and Wiebe, *Composite Quantum Simulations*, [arXiv:2206.06409](https://arxiv.org/abs/2206.06409)。高次Trotterと乱択simulationの複合化。
- [W7] Wan, Berta, Campbell, *A randomized quantum algorithm for statistical phase estimation*, [arXiv:2110.12071](https://arxiv.org/abs/2110.12071)。ランダム位相推定とRTEの背景。
- [W8] Bosse et al., *Efficient and practical Hamiltonian simulation from time-dependent product formulas*, [arXiv:2403.08729](https://arxiv.org/abs/2403.08729)。THRIFT、エネルギースケール分離。

本書の研究提案、比較設計、条件付き算術は、これらの文献がそのまま結論した内容ではない。既知事実、現artifactの読解、本書で提案する作業を区別して記述した。新規性の不在証明、最終的な科学的優位性、採録可能性の保証は行っていない。
