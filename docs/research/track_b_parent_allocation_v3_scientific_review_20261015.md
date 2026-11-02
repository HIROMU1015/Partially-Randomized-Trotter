# Track B：Parent allocation v3の科学的レビュー
## 頂点配分の条件付きT利益と、内部KKT候補の目的関数不一致

- 作成日：2026-10-15（JST）
- 開始承認：利用者の「レビューを開始して　丁寧に考えて」
- 対象：Parent allocation v3の実行結果、比較Evidence Packet、採択済み科学設計、対応する実行source。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- Evidence Packet commit：`17bfcaac5a53d648ee94de0067de24d719ed3911`
- 原結果commit：`cd055232f9612982a60953341b9cbf6168dd0bd8`
- Science source S3：`53a36adf9bdd84690e9cdf3e1f97f38a178441dc`
- 実行authorization-only A3：`b1dfe53ca98a80d60831dbf9d28c5c9fc6a08f31`
- 適用指示：会話に添付された2026-10-12版の研究プロジェクト共通指示。
- レビュー状態：**科学レビュー完了。頂点配分＋ISの限定的なT利益を支持する。一方、内部KKT候補に科学設計と実装の不一致を確認したため、意図した内部候補の評価と比較計画全体の科学的完遂は保留。次はCodexによる限定修正・影響範囲確認・補完実行準備。**
- 実行承認：新しい登録計算・合成・source再実行は未承認。既存mandatory STOP、retry=0、消費済みmarkerを維持する。

本書は今回のGPT研究判断であり、v3のterminal、元classification、保存監査、既存source/resultを改稿・置換するものではない。

---

## 0. 最終判断

今回の判断は次の三点である。

**第一に、Parent allocationは、単に構成できるだけの仮説ではなくなった。** 保存された共通費用会計では、ordinaryの頂点配分＋正則化IS、closed P5の頂点配分＋解析ISが、それぞれ共通角配分に同じproposal機会を与えた対照より低い期待T費用を持つ。今回見つかったKKT生成式の不一致は、この二つの頂点を生成する式そのものには入っていない。この限定的なpositive evidenceは保持する。[R1–R4, R7–R8]

**第二に、約14%・19%の低下を、配分単独の利益、全資源優越、一般PR/QPE改善とはしない。** P5の大きな利益は、頂点配分と既知ISを組み合わせたときに現れ、同じ頂点のcanonical評価は共通角より高い。試行数・CXが増え、共通準備費用を加えると指定law間の順位も変わる。[R1–R4]

**第三に、内部KKT候補の比較に新しいblocking issueがある。** 採択設計の目的関数は `sum sqrt(C_i) sqrt(z_i^2+b_i^2)` だが、実行sourceはraw T単価を受け取りながら `sqrt(C_i^2-lambda^2)` を使っている。これは `sum C_i sqrt(z_i^2+b_i^2)` のKKTに対応する。両者は不等単価では一般に異なる。したがって、保存名`kkt`と`halfway`の結果を、採択したKKT候補がordinaryで成功・P5で失敗した証拠には使わない。[R5–R8]

次は、他provider、r、precision、seedを広げるのではなく、この**目的関数と入力単位の不一致を修正し、影響を受ける部分だけを補完できる契約・sourceを準備する**。新しい値や価格の取得は、別途の実行承認後とする。これは同じ資料整理を繰り返す判断ではなく、本レビューで新たに見つかった科学的意味論の不整合への対処である。

---

## 1. 研究全体での位置付け

### 1.1 研究の問い

利用者は論文化をいったん切り離し、「Return aggregationを発展させるか、それも含めた過去の検証から別のアルゴリズム改善へ進むか」を問いとしていた。前回の方針レビューは、Return aggregationを最終的なwinnerに固定せず、係数配分・回転角・proposalを回路費用に合わせて変える方向を第一候補とした。[R11]

その後の科学設計は、ordinaryとclosed P5に同じroot配分自由度を与え、partial-return、P3+tail、matched CTSを参照対照として、同じfinite P5と信号精度へ戻す限定比較を採択した。[R5, R6]

今回のv3は、この採択比較を実行した結果である。従ってレビューの問いは、単なるsourceの動作確認でも、論文採択可能性の採点でもない。

> どの配分変更が、どのsampling policyと資源モデルの下で実装上の利益を生み、その証拠から次の研究方向をどう決めるべきか。

### 1.2 過去の結果とのつながり

前回方針レビューの整理を継承すると、B-F/B-Mは評価式・表現の変更だけでは新しい実行法の利益にならないこと、R1/SPは係数質量・bias・角度価格・sampling負担が別々に効くこと、G5は固定classの強い対照を戻す重要性を示していた。G6–G10には生成可能なreturn集約構成が残り、D1–D3は集約量の増加とnative総費用の減少が同義でないことを具体化した。[R11]

v3の頂点＋ISの結果は、「normalizationを増やしてでも回転の費用を下げる」という、その後の設計を限定条件で支持する。しかし、SPでの累積時の反転や、D系列での価格依存性を無視して全PRへ拡張する証拠ではない。旧G5、generic whole、Gaussian、時間相関等のclosureは変更しない。

過去系列の全source・全rawを今回再監査したわけではない。これらは固定された前回の横断レビューにおける位置付けを継承したもので、異なるfixtureの改善率を横並びの性能順位に使っていない。

---

## 2. 使用資料と確認範囲

### 2.1 今回直接読んだ内部資料

| ID | 固定資料 | 今回確認した範囲・用途 |
|---|---|---|
| R1 | 比較Evidence Packet、`17bfcaac...` | 本文全体を分割取得。B1/B2/B3、bias、h交点、未取得情報 |
| R2 | v3結果・監査handoff、`cd055232...` | 実行identity、coverage、保存値、監査の範囲、停止状態 |
| R3 | law索引CSV、`17bfcaac...` | header、ordinary共通角、P5共通角・vertex_0の関連行。全75行の独立再監査ではない |
| R4 | `rows/row_34.json`、`cd055232...` | P5 vertex_0 analytic ISの原row、allocation、実N、費用・moment・range・bias field |
| R5 | 採択済み科学設計、`cd055232...` | 特に§4–7のKKT目的、候補・対照・freeze規則。会話添付原本も参照 |
| R6 | `execution_contract_v3.json`、`cd055232...` | task、候補、proxy、proposal、精度・alpha、source/caps/禁止事項 |
| R7 | `parent_experiment/design.py`、S3 | 候補生成・KKT、proposal、ledger、comparisonsを静的に読取り |
| R8 | `parent_experiment/pipeline.py`、S3 | raw Tの受渡し、C*、candidate freeze、非root再利用、law/Nへの接続 |
| R9 | `candidate_freeze.json`、原結果commit | raw proxy_T=[136,138,140]、common/vertex/kkt/halfwayの保存配分と候補ラベル |
| R10 | v3 source manifest、原結果commit | 対応sourceのcritical hash記載と93 testsの位置付け。全hashの独立再照合ではない |
| R11 | 2026-10-14研究方針レビュー、`4d80b456...` | 研究全体の目的、過去系列の判断、Parent allocationを選んだ理由 |
| R12 | Parent allocation先行研究対応、`bf1366ac...` | Euler/IS/SPS/SCUとの包含と相違、新規性未確立の範囲 |

GitHub connectorから固定refの資料を取得した。主要な内部証拠が未取得であることを理由とする保留ではない。今回の保留は、取得した設計と実行sourceが一致しない箇所を確認したためである。

### 2.2 外部一次資料

- Resource-Optimal Importance Sampling、arXiv:2603.13495v1：HTMLの§II、Theorem 1、§III、Theorem 2を直接確認した。[P1]
- Sparse Probabilistic Synthesis、arXiv:2402.15550v2：HTMLのprocess辞書・係数最適化の枠組みを参照した。[P2]
- SCU/CTS：npj Quantum Informationの出版社HTMLを補助的に取得した。[P3]

これらの全証明、全後続論文、全実装を再監査したわけではなく、新規性の網羅的な不存在証明も行っていない。今回PDFは解析せず、図表の読取りから新しい性能値を取得していない。

### 2.3 今回していないこと

科学sourceのimport、candidate solver、登録入力の新配分計算、N再計算、資源表の再生成、全h包絡線、価格感度の数値計算、matrix/guard/synthesis/compile、quantum shots、repository tests、GitHub書込みは行っていない。

補助表示のN比・CX比等について、保存された整数と表示値のDecimal比算術だけを実施した。これは新しい最適化、元exact資源表の置換、訂正後KKTの評価ではない。

918保存監査と450資料整形チェックの完了はCodexの固定記録として参照し、GPTがそれらを再実行したとはしない。

---

## 3. 評価taskと保証を維持する

対象は既知developmentの3-system-qubit synthetic involutionモデルである。[R1,R2,R6]

\[
p=(1/5,3/10,1/2),\quad x=\tau=5/7,\quad \sigma=+1,\quad r=1,
\]
\[
R=\sum_i p_iQ_i,\quad M=P_5(-ixR),\quad U=e^{-ixR}.
\]

入力stateは000。分子geometry、DF rank、D/R split、QPE全体を含む実験ではない。既知入力であって、新held-outではない。

比較はcontrolled first operator momentのRe/Imについて、各axisの絶対誤差1/200、alpha_axis=1/40、選択した一complex taskのfailure 1/20という採択済みの意味を保つ。70 lawへのalpha再分配は行わない。law選択に量子outcomeを使わないことが、この保証の前提である。[R5,R6]

実dyadic proposalをq、digital係数を\(\widetilde c\)、weightを\(w=\widetilde c/q\)とすると、単一Hadamard outcomeが±1であることから

\[
\nu=\sum_e\widetilde c_e^2/q_e,\qquad W=\max_e\widetilde c_e/q_e
\]

を用いる。sourceはTaylor、係数、strict native誤差を戻し、統計残余幅Δと既定のBernstein十分Nを計算する。[R6,R7]

\[
N=\left\lceil\log(2/\alpha_{\rm ax})
\left(2\nu/\Delta^2+4W/(3\Delta)\right)\right\rceil.
\]

主費用は\(G_T(0)=2N\mathbb E[C_T]\)。必要最小shotsや実測shotsではなく、固定された十分予算policyの割当費用である。既知ISのcost×second-moment最適化を、この有限confidence問題全体の最適性と同一視しない。[P1]

状態のスペクトル重み・exact ground情報は配分やNの選択に用いていない。「厳密基底状態ではないことを利用して改善した」という解釈にも変更しない。状態非依存の演算子構成と、未知の状態準備費用を含む総実行費用の改善は別問題である。

---

## 4. 新たに確認した重要事項：KKTの目的関数不一致

### 4.1 採択した式

採択科学設計の§4.2は、旧common rootの完全なevent T単価\(C_i>0\)に対して

\[
S(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2},\qquad \sum_i z_i=a
\tag{1}
\]

を目的とし、その内部KKT候補を

\[
z_i^K=\frac{b_i\lambda}{\sqrt{C_i-\lambda^2}},\qquad
0<\lambda<\sqrt{\min_i C_i}
\tag{2}
\]

と定めている。§7も、入力proxyが「完全なevent単価\(C_i^0\)」であることを明示している。[R5]

この目的は、固定dictionaryへのglobal ISを先に最適化したcost×moment proxyの平方根部分に対応する。非rootの寄与は固定定数なので、rootの(1)を最小化すればよい。[R12,P1]

### 4.2 実行sourceの式

S3の`design.py`の`kkt_candidate()`には次の処理がある。[R7]

```python
lo, hi = F(0), min(costs)
...
rt = root_interval(ci*ci - lam*lam, root_bits)
...
out.append((lam*bi/rt.hi, lam*bi/rt.lo))
```

`pipeline.py`の呼出し側は次のとおりで、平方根への変換を挟んでいない。[R8]

```python
proxy = [native_accounting(item['native_ir'], cache)['T']
         for item in sources[arm] if not item['event']['raw_word']]
cand = candidates(base.root.a, base.root.b, proxy, c['candidates']['KKT'])
```

`candidates()`もproxyをそのまま`kkt_candidate()`へ渡している。保存された`candidate_freeze.json`にはordinary、closed P5の両方について

```text
proxy_T = [136, 138, 140]
```

が記録されている。[R9]

したがって、入力が\(\sqrt{C_i}\)だったので`ci*ci`が正しい、という説明にはなっていない。実際にはraw T単価の二乗が分母へ入っている。

### 4.3 別の最適化問題になっていることの解析

実装の理想的な式は

\[
z_i^{\rm code}=\frac{b_i\ell}{\sqrt{C_i^2-\ell^2}}
\tag{3}
\]

である。これから

\[
\frac{z_i^{\rm code}}{\sqrt{(z_i^{\rm code})^2+b_i^2}}
=\frac{\ell}{C_i}
\]

が従い、停留条件は

\[
C_i\frac{z_i}{\sqrt{z_i^2+b_i^2}}=\ell.
\tag{4}
\]

これは

\[
\widetilde S(z)=\sum_i C_i\sqrt{z_i^2+b_i^2}
\tag{5}
\]

のKKT条件である。採択した(1)の条件は

\[
\sqrt{C_i}\frac{z_i}{\sqrt{z_i^2+b_i^2}}=\lambda.
\tag{6}
\]

正の内部配分で(4)と(6)が同時に成立するには\(\ell/\sqrt{C_i}\)が全iで同じ必要がある。非退化の不等単価では一般に成立しない。raw proxyは136,138,140と不等なので、単なるKKT乗数の名前変更・共通scale変更では解消できない。

これは丸め誤差ではなく、**最小化する費用重みの不一致**である。ただし、その修正がnative費用を良くするか悪くするかは未取得であり、ここでは予測しない。

### 4.4 登録外の独立検証に使える解析例

次は登録分子・v3入力を使わない解析例であり、今回新たなテストプログラムは実行していない。

\[
a=7,\quad b=(4,3),\quad C=(1,9/16),\quad z=(3,4),\quad c=(5,5).
\]

正しい目的では

\[
\sqrt{C_1}z_1/c_1=3/5,
\qquad \sqrt{C_2}z_2/c_2=(3/4)(4/5)=3/5.
\]

一方、raw C重みでは\(C_1z_1/c_1=3/5\)、\(C_2z_2/c_2=9/20\)で一致しない。このような不等単価の残差確認は、同じ誤った式からexpected値を作る検査や、両式が一致する等単価controlだけより直接的である。

修正では、分母だけでなく探索区間も整合させる必要がある。raw Cを引数とするなら、(2)のλ区間、または\(t=\lambda^2\in(0,\min C)\)という等価な変数で、安全な固定精度の区間法を設計する。具体的な数値実装はCodexへ委ねる。

### 4.5 影響範囲

直接影響する構成は次の4つである。

| config | 保存ラベル | 問題となる解釈 |
|---|---|---|
| 04 | ordinary / kkt | 採択したsqrt(C)目的のKKT候補としての解釈 |
| 05 | ordinary / halfway | 採択KKTとの半移動点としての解釈 |
| 10 | closed P5 / kkt | 同上 |
| 11 | closed P5 / halfway | 同上 |

各5 proposal、合計20保存lawがこのラベルに対応する。[R1] これは20行の数値が架空という意味ではない。保存された実際の有理配分に対する資源会計としての記録は保持する。一方、「意図したKKTを評価した結果」という役割が満たされない。

`candidates()`のcommonと頂点はKKT解とは別に直接構成される。`pipeline.py`のC*は候補集合からではなく、旧五方式のbaseline単価から先に作られる。lawは各構成ごとに生成され、alphaもlaw数では割らない。[R7,R8]

このsource構造に基づき、common・頂点・参照方式の結果を、今回のKKT不一致だけを理由に一括無効とはしない。ただし、修正の実務ではCodexが候補・cache・index・bindingの依存関係を照合して、この切分けを確認する必要がある。

### 4.6 科学的な完了判定への影響

- 元の実行が15構成・70 lawを保存し、正常終了した技術記録は維持する。
- 元のterminalや`PARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW`を書き換えない。
- 新しいレビュー注記として、内部候補の数学仕様との不一致と影響範囲を追加する。
- 「採択したKKT候補はordinaryでは有効、P5では無効」とする科学的結論は保留する。
- 「採択比較の全要素を意図どおり検証し終えた」とする認定も保留する。

保存値・source hashの一致は重要だが、実装した目的関数が研究設計と一致することとは別の検証である。今回の発見は、918監査や既存testsの来歴確認を否定するものではなく、それだけでは閉じていなかった設計と実装の対応を特定したものである。

---

## 5. それでも保持できる頂点配分の結果

### 5.1 Best-to-bestの保存事実

以下のT/CXは百万gate、Nはaxis当たりの保存整数。最良は原sourceが既に選択したlawであり、今回新たな最適化を行っていない。[R1–R4]

| 表現・配分・proposal | N | T費用 | CX費用 |
|---|---:|---:|---:|
| ordinary common / regularized_16 | 924,186 | 258.666316 | 15.263828 |
| ordinary vertex_2 / regularized_1/16 | 1,587,522 | 222.430657 | 21.883969 |
| P5 common / regularized_1 | 680,318 | 189.771852 | 11.574442 |
| P5 vertex_0 / analytic IS | 2,270,017 | 153.775923 | 29.657568 |

保存報告のcommon比はordinary約14.0087%低下、P5約18.9680%低下。どちらの比較にも問題のKKT/halfwayを使っていない。

**本レビューは、同じ十分予算policyと固定native価格の下で、非common頂点がcommon＋同じproposal機会より低Tになるという限定命題を支持する。** 一般的な全資源改善・全入力改善・連続family最適性は支持しない。

### 5.2 単なるIS追加だけではないが、配分単独でもない

P5について、保存された四条件を並べると次のとおりである。[R1,R3]

| 配分 | canonical | analytic IS |
|---|---:|---:|
| common | 189.773346 | 189.782118 |
| vertex_0 | 202.319741 | 153.775923 |

単位は百万T。

commonからvertex_0へ変更するだけでは、canonicalで費用が増えている。commonにISを追加しただけでも、この有限confidence会計では改善していない。一方、vertex_0へ変更した後でISを使うと大きく下がる。

従って、今回は「既知ISを新側だけへ与えた見かけの利益」ではない。commonにも同じIS機会があるためである。しかし、「配分変更単独で19%改善」とも言えない。**表現・回転費用を変える操作と、その後の抽出確率変更の組合せに追加価値がある**、というのが適切な解釈である。

ここで「同じproposal」は、同じ生成規則を与えたという意味で、qの各成分が同一という意味ではない。係数とevent価格が変わるので、同じ解析式から得るqも変わる。[R1]

また、P5 vertex_2はcanonicalでも共通角より低い保存点がある。したがって「頂点配分は必ずISがなければ損」と一般化もしない。[R1]

### 5.3 参照集合との関係

partial-return共通角の保存T切片は最も低い掲載proposalで約191.973637百万、P3+tailは約198.483503百万、matched CTSは約237.608885百万である。P5 vertex_0 analytic ISは約153.775923百万で、保存70 lawのh=0最良として記録されている。[R1,R2]

従って、今回の低T点はordinaryという遠い対照だけに勝った例ではない。ただし、partial/P3+tail/CTSへ任意parent配分や新辞書最適化まで同じように追加した全面比較ではない。**全表現を同じ程度に最適化した後もP5が不可欠、という結論は出せない。**

原結果の「全70 lawの最良」は実際に評価した集合内の記録として保持する。訂正後KKTが入る意図した候補集合での最良も同じだとは、まだ言わない。

---

## 6. T利益の機構と限界

### 6.1 係数質量・sampling負担を増やして回路を安くする

commonは配分family内の係数質量最小点である。新しい頂点がnormalizationを改善しているわけではない。[R5]

P5の保存best-to-bestでは、単価約139.4729 Tから約33.8711 Tへ低下する一方、Nは680,318から2,270,017へ増える。[R1,R4]

保存値の補助比算術では、単価比は約0.24285、N比は約3.33670であり、その積が総T比約0.81032になる。これは既存数値の因数分解であり、特定要因へ一意な因果百分率を割り当てたものではない。

M2は約1.660139から5.538145へ、rangeは約1.334415から5.114787へ増えている。bias上界と統計残余幅も表現間で微小に変わるが、これらの保存値を混ぜて「誤差が減ったから低T」とはしない。[R1]

### 6.2 端点の説明は支持できるが、全価格寄与は未分解

頂点では一つのchildだけが回転を持ち、他はphase付きpure-Qとなる。そのため、回転の取得・実行箇所を減らす機構がある。ただし、providerの基底変換、controlled-Q、位相ゲートは残る。pure-Qであることはzero-costの意味ではない。[R5,R12]

現記録は、固定されたsequence価格と費用低下を追跡できる。一方、どのkeyが利益の何割を担ったか、別の合成列やprice perturbationで順位が維持されるかという解析は未取得である。[R1]

従って「安い端点を増やした構造とISの組合せ」という機構説明は支持するが、「角度の偶然に依存しない頑健な改善」とはまだ認定しない。逆に、一つの価格modelしかないことを理由に「偶然にすぎない」と決め付けもしない。

### 6.3 試行数とCXの代償

保存値からの補助比は以下である。

| 同じbest-to-bestの比較 | N比 | CX比 | T比 |
|---|---:|---:|---:|
| ordinary noncommon / common | 約1.718 | 約1.434 | 約0.860 |
| P5 noncommon / common | 約3.337 | 約2.562 | 約0.810 |

これはTをprimaryにしたpolicyではpositiveでも、全resource座標でのdominanceではない。T・CX・native1Qを任意の重みで後付け合算し、勝敗を作り直さない。

実際のfault-tolerant時間、デバイス待ち時間、state準備、online古典処理まで含む費用はこの表からは決まらない。全quantum shotsを実行した測定結果でもない。v3の約26.7秒は資源表を生成した古典実行のwallであり、量子アルゴリズムがその時間でtaskを解いたという意味ではない。[R2]

---

## 7. 準備費用・反復・情報アクセスを戻すと何が言えるか

### 7.1 保存された二law間のh交点

採択会計は

\[
G_s(h)=2N_s(\overline C_s+h)
\]

である。Evidence Packetが抽出した同じbest-to-best二lawの保存交点は、ordinaryのTで約27.3132015 T/trial、P5のTで約11.3216176 T/trialである。[R1]

P5については、指定したvertex_0 analytic ISとcommon regularized_1の比較で、hが交点を超えるとcommon側が低Tになる。これは正当に重要な制約である。主な代償が多い試行数だからである。

ただし、**h≈11.32をParent allocation全体の有効範囲の上限としない。** 別の頂点・proposalがその領域で有利かもしれず、全70 lawのh包絡線はまだ作成されていない。同様に、実際の利用場面のhがいくつかも未取得である。未知の準備費用を大きいとも0とも勝手に固定しない。

### 7.2 one-stepの利益をPR全体へ乗算しない

r=1の結果から、長い積へ約19%の改善率をそのまま適用できない。独立なstep weightsを掛ける特定実装では二次momentが積になる一方、回路費用にはD部分・境界・非加法な簡約も入り得る。総時間一定でrを変えれば各stepのx、配分、bias、価格も変わる。

この観察は一般的な独立性の条件付き推論であり、今回multi-stepの新しい費用値・不可能性定理を取得したわけではない。D、分子DF、QPEへの移送は別の科学設計を必要とする。

### 7.3 局所係数取得と全表ISを区別する

parent/root係数を局所的に作れることと、global IS用の全event費用表を安価に取得できることは別である。今回は小さいI1入力で、必要な表と価格を各方式に同じように与えた比較である。[R5,R6,R12]

共通acquisitionやledger CPUは測定されていても、選択されたアルゴリズムのonline sampling費用は`NOT_MEASURED`である。これを0としてend-to-end速度向上を主張しない。逆に、未測定であることだけでlogical T比較を取り消さない。

---

## 8. 新規性と研究価値

任意のparent分割にEuler式を適用する平均保存恒等式は既知式の直接の帰結であり、固定dictionaryの最適ISも既知理論である。[R12,P1] 「normalizationだけでなく回路費用を考える」という一般原理を今回の発見とはしない。

SPSはprocess辞書と線形結合を使い、回路の安さとsampling負担を交換する一般的な枠組みである。[P2] Parent allocationのfirst momentは、そのまま同じfull channel制約ではないが、controlled embedding・off-diagonal projection等を通じて比較可能性を議論できる。phaseの違いだけでSPSを排除しない。[R12]

残る具体的な研究価値は、例えば「構造化されたfinite-RTEを、少数の配分候補と実行可能なsamplerによって費用の低い表現へ変えること」「その利益がsampling・state準備・実装価格を戻すとどの領域に残るかを明確にすること」である。ただし、現在は一つの既知synthetic入力の条件付きpositiveであり、独立新規性や一般的な選択則は未確立である。

利用者の目的は現在、論文化判定ではなくアルゴリズム改善方向の選択である。したがって、世界初の数学primitive、全既知法への普遍的勝利、最低改善率を新しい必要条件として課さない。一方、世界初が未否定というだけで有望性を認定もしない。

---

## 9. 研究方針の比較と今回の選択

| 選択肢 | 今回の判断 |
|---|---|
| 頂点の低T結果を根拠にそのまま分子・r・precision探索へ進む | 今回は採択しない。内部候補の意味論不一致と総資源上の限定が残る |
| Parent allocationを全面終了する | 採択しない。問題のKKTと独立に、common対頂点＋ISの限定positiveが残る |
| KKTのordinary成功・P5失敗を根拠に内部配分を打ち切る | 採択しない。意図したKKTを評価していない |
| 既知Eulerへの包含だけで別アルゴリズムへ全面転換する | 採択しない。具体的な実装利益を評価するという当初の目的にはまだ意味がある |
| 全文献調査・全資料監査を再度依頼する | 採択しない。今回のblocking issueは特定されている |
| **KKTの限定修正と、影響部分だけの補完を優先する** | **採択する研究上の順序。科学実行は別途承認** |

研究の大きな方向は、当面「費用を考慮したpairing・配分とISの組合せ」を維持する。ただしReturn aggregation専用の改善とはまだ呼ばず、ordinaryにも通用する手順として扱う。P5が今のh=0集合で最も低いことと、P5を研究として必須とすることは分ける。

次に必要な補完は、まだ価格を探していない新仮説を次々作る探索ではない。**既に採択した内部候補を、意図した目的で評価するための訂正**である。これは結果を見た後に有利な候補へ救済することとは異なる。ただし訂正はv3の結果を見た後であるため、その経緯を新記録に明示する。修正で利益が増えるという期待を承認理由にしない。

---

## 10. 次の担当：Codexの限定修正・影響確認

### 10.1 直近に行う作業

Codexへまとめて委ねる作業は次である。

**設計とsourceの独立照合。** raw T単価C、sqrt(C)、一般重みwを引数名・単位・目的関数まで区別する。今回の`ci*ci`、探索区間、呼出し元、保存lambdaとラベルの意味を確認し、原因と影響範囲を記録する。同じ誤った式をexpectedにコピーする検査を避ける。

**正しいKKT候補生成への限定修正。** 採択式(1)–(2)へ戻し、固定精度の区間、非負性、exact総和、有理化・距離上界、halfway、重複・失敗処理を整合させる。目的関数を結果に合わせて変更せず、追加候補・seed探索を加えない。必要な登録外testsと回帰を一束で行う。

**影響範囲とreuseの確認。** common・頂点・参照方式、C*、global proposal、alpha、価格の独立性をsourceと保存bindingから確認する。v3のKKT/halfwayは実際に生成された別proxy候補の記録として保持し、正しかったことに改名・改稿しない。旧20該当lawと、影響のない行を明示する。

**補完用source・契約・preflightの準備。** 意図したKKT/halfwayの最大4構成を対象とし、同じ5proposal機会、精度、baseline、C*、alpha意味論を保持する。必要な対照の再利用と新しい価格取得を、source/manifest/guardに結び付ける。

これはCodexへの技術準備の作業範囲であり、今回GPTはrepositoryの修正や実行を行っていない。新しい登録計算・合成・guard・資源表の実行は、凍結sourceと別途承認を必要とする。

### 10.2 補完計算の設計境界

研究上は、**影響のある内部候補だけを補完し、影響のないv3証拠を再利用する方式を優先**する。全70 lawの原runnerを無条件に再実行する必要はない。

新たなroot角の候補は、ordinary/P5 × KKT/halfway × 最大3childなので、重複除去前のpositive primitive keyの自然な上限は12である。これは追加合成が必ず12回必要という意味ではなく、既存cacheとの一致・共通点への一致・端点等で少なくなり得る。非rootの新規取得や他providerを混ぜない。具体的なinventory・guard回数・CPU/RSS等の実行上限はCodexが技術的根拠とともに固定する。

新候補は同じ旧common価格だけから作り、訂正後の新価格を見る前にfreezeする。既存v3の新価格を使ってzを選び直さない。以前のA3/markerは再利用しない。新しい訂正用記録・source・契約・authorization-only直接子・namespaceが必要である。

既存行との結合が科学的に正当化できない依存が見つかった場合は、必要な最小の再評価範囲を報告する。予算・alpha・候補・比較対象を黙って変えて解決しない。

### 10.3 レビューへ戻る条件

今回特定した不一致を、採択された元の意味へ戻す修正と登録外テストについて、軽微なcommitごとにGPTレビューを要求しない。

GPTへ戻すのは、目的関数を変える必要がある、影響範囲が頂点や参照方式にも広がる、保証・比較条件を変更する必要がある、または訂正後の重要な資源結果が揃った段階である。結果レビューの開始は、その時点の明示承認を必要とする。

---

## 11. 訂正後に優先する科学的問い

以下は将来の候補であり、今回その実行を認可したものではない。

1. **内部候補の方法差を確定する。** sqrt(C)目的による意図したKKTとhalfwayの成績を得て、頂点の回転除去と内部の係数移動を正しく区別する。
2. **実装費用と準備費用の有効領域を調べる。** 全h包絡線やkey寄与の必要性を、訂正結果を含めて判断する。単一pairのh*=11.32を全familyの限界にしない。
3. **価格依存・情報取得を含む移送価値を判定する。** sourceの同じルールを別の実装条件へ固定して移す必要があるか、実費用・online生成が主要な懸念かを選ぶ。一斉にseed/provider/precision/rを走査することは採択しない。

訂正で内部候補の利益が出なくても、頂点のpositiveは独立に評価する。逆に訂正で利益が出ても、全方法の最適性や一般化を直ちに主張しない。

---

## 12. 支持・保留する主張の一覧

| 主張 | 今回の扱い |
|---|---|
| 任意の適格parent配分でideal平均を保存できる | 既存数理を支持。KKT最適性は平均保存の必要条件ではない |
| v3は実際の15構成・70lawを正常保存した | 固定実行記録として保持 |
| 頂点＋ISの固定条件でのT低下 | 支持。ordinary約14%、P5約19%は当該common対照との比較 |
| 約19%は配分単独の因果的寄与 | 支持しない。四条件で組合せ依存を確認 |
| 採択したsqrt(C)-KKTのordinary成功／P5失敗 | 保留。実装の目的関数が異なる |
| v3保存名kkt/halfwayの実際の配分に対する費用値 | 原記録として保持。採択候補の検証結果とは区別 |
| 918監査PASSなので科学設計も完全一致 | 支持しない。今回別の対応不一致を確認 |
| 全資源・全hでnoncommonが優位 | 支持しない。N/CX増加と指定law間の反転あり |
| P5を用いることが本質的に必要 | 未確定。ordinaryにも利益、他表現の配分最適化は未完 |
| 全既知法・SPS・一般CTSに優位 | 未確立 |
| 分子PR/QPE全体の費用が約19%減る | 未確立 |
| 次は無条件の大規模実験 | 非採択 |
| 次はKKT限定修正・影響確認・補完準備 | 推奨し、この順序を採択。実行承認は別 |

---

## 13. 確かさと残る不確実性

**確かさが高い事項**：採択KKT式、raw Tを渡す呼出し、`ci*ci`を使う実装、保存proxy_Tが不等であること、および両目的関数が異なるという解析。これはsource名の印象ではなく、入力から式まで対応させた判断である。

**条件付きで支持する事項**：頂点＋ISの保存T利益、資源交換、hの指定pair関係。原resultと既存数値保証・監査の範囲を前提とし、全sourceの独立認証ではない。

**まだ分からない事項**：訂正後KKTの角度・価格・N・順位、全h包絡線、別価格での頑健性、他入力・複数step・分子への移送、online費用、独立新規性。

本レビューは、訂正後の結果がまだないことを曖昧にした最終GOではない。一方、未確定部分があることを理由に現在の頂点positiveを無価値ともしていない。

**最終的には、「費用を考慮した配分＋IS」の研究方向を限定的に維持しつつ、意図した内部候補との不一致を先に正す。** それが現在の証拠に最も整合する次の段階である。

---

## 参考資料（固定URL）

### 内部資料

- **[R1]** Parent allocation v3比較Evidence Packet。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/17bfcaac5a53d648ee94de0067de24d719ed3911/docs/tracks/algorithm_codesign/parent_allocation_v3_comparison_evidence_packet_20261015.md
- **[R2]** Parent allocation v3結果・保存監査handoff。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/docs/tracks/algorithm_codesign/parent_allocation_v3_results_and_gpt_handoff_20261015.md
- **[R3]** 保存law比較索引。今回は関連範囲を取得。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/17bfcaac5a53d648ee94de0067de24d719ed3911/artifacts/track_b_parent_allocation_review_packet/2026-10-15/v1/law_comparison_index.csv
- **[R4]** P5 vertex_0 analytic IS原row。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/rows/row_34.json
- **[R5]** 採択科学設計（特に§4.2、§7）。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/docs/research/track_b_parent_allocation_scientific_design_review_20261014.md
- **[R6]** 固定execution contract v3。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v3/execution_contract_v3.json
- **[R7]** S3 `parent_experiment/design.py`。`kkt_candidate`、`candidates`、`proposals`、`ledger`、`comparisons`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53a36adf9bdd84690e9cdf3e1f97f38a178441dc/src/trottertracks/algorithm_codesign/parent_experiment/design.py
- **[R8]** S3 `parent_experiment/pipeline.py`。`run_pipeline`のraw proxy、C*、freeze、law評価。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53a36adf9bdd84690e9cdf3e1f97f38a178441dc/src/trottertracks/algorithm_codesign/parent_experiment/pipeline.py
- **[R9]** 原候補freeze。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/candidate_freeze.json
- **[R10]** S3 source manifest。critical hash記載の読取り。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v3/source_manifest.json
- **[R11]** 2026-10-14 Track B研究方針レビュー。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d80b45693a1ea7c14d7cbf9035e7358788953bd/docs/research/track_b_algorithm_direction_review_20261014.md
- **[R12]** Parent allocation近接先行研究対応。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf1366ac5832136ab8038d160e9a0285da69e79e/docs/tracks/algorithm_codesign/parent_allocation_prior_art_correspondence_20261014.md

### 外部一次資料

- **[P1]** D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1。§II Theorem 1、§III Theorem 2。
  https://arxiv.org/html/2603.13495v1
- **[P2]** B. Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, arXiv:2402.15550v2。process辞書・線形係数最適化の枠組み。
  https://arxiv.org/html/2402.15550v2
- **[P3]** J. Peetz, S. E. Smart, P. Narang, *Quantum simulation via stochastic combination of unitaries*, 出版社HTML。補助参照。
  https://www.nature.com/articles/s41534-025-01168-w

本書の作成以外にrepositoryへの書込み、source修正、authorization発行、marker操作は行っていない。
