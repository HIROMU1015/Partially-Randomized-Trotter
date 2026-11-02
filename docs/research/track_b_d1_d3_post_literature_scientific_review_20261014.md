# Track B：D1〜D3・最終機構分析・先行研究照合後の統合科学レビュー

- 作成日：2026-10-14 JST
- レビュー開始承認：ユーザーの「レビューを開始して」
- 対象：②「既存のPR研究成果の発展」。特にreturn集約の構成・生成法、D1〜D3の有限精度資源評価、最終機構分析、一次文献照合。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 文献引き継ぎ：`d3a232a446062e6baadef2b69eab87faa0285363`
- 最終機構分析：`534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f`
- D3 v2結果：`f0df9a3b36bc3540f0c83941871fa0d3110ddbf7`
- D3 science S2：`dceccaa774229527a46a1386131583a4f618f8ad`
- D3実行承認・実行開始HEAD A2：`daa2569208a945f8e41b3c4d0de715c1e11b104b`
- 適用する共通運用指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`
- 本レビューの状態：**重要科学レビュー完了。既存成果の発展は継続するが、現在の数値探索系列は一区切りとし、構成・生成法と条件付き資源評価を一体化した原稿の作成を次の作業とする。新しい科学実行は承認しない。**

この判断はGPT側の研究判断であり、repositoryに保存されたD1/D2/D3のclassification、source、contract、authorization、消費済みmarker、STOPを書き換えたものではない。

---

## 0. 結論と、以前のレビューからの変更

**②「既存成果を発展させる研究」は継続を支持する。ただし、P5またはP3+tailが勝つ条件をさらに探す数値系列は、現段階で広げない。**

現時点で最も一体性のある研究の中心は、次である。

> Hermitian involutionの有限Taylor演算子に対するreturn集約を、実際に生成可能な乱択表現として構成する。そのうえで、係数質量・生成情報・有限精度合成・十分測定予算を分離し、集約の利益が実装上の資源利益へ直結するとは限らない条件を示す。

D1〜D3を単独の「P5が何%優れる研究」として位置付けることは推奨しない。既存G系列の構成・有限bit生成・情報アクセスの議論を理論的な中心に置き、D1〜D3を、その構成の資源上の性質と限界を明らかにする具体的な証拠として組み込む。

今回採択するのは**構成・生成法と実装依存性の方法研究としての原稿化**である。「一般的な最適選択アルゴリズムが完成した」「独立新規性が全面的に認定された」「投稿・受理に十分」とは判断しない。

以前のD1/D2レビューでは、保証と価格に対する限定的な頑健性検証を次段階にした。D3によって、その問いには具体的な答えが得られた。さらに機構分析と8論文の照合も完了したため、次も同じ種類の価格・seed・precision調査に進む必然性はない。**検証の採択時点から新たな証拠が加わったことが、実験拡大ではなく原稿化を選ぶ理由である。**

次の担当はCodex。既済の数値解析・claim/evidence表・文献catalogを再作成するのではなく、理論構成から実装評価までつながる本体原稿と必要な証明付録を作る。計算実行のmandatory STOPは維持する。

## 1. 確認した証拠と、確認していないこと

### 1.1 内部資料

本レビューでは、少なくとも次の資料をGitHubの固定refから実際に取得して読んだ。末尾の参考資料に固定URLを示す。

| ID | 資料 | 本レビューでの使い方 |
|---|---|---|
| I1 | D1〜D3一次文献・claim/evidence引き継ぎ | 13主張候補、既知原理、taskとaccessの相違、未確立事項 |
| I2 | D3最終保存機構分析 | key価格寄与、P3/P5の小差、保証・反復数、primaryと全方式、h領域 |
| I3 | D3表示用48行CSV | 全ケースの保存費用・Nの一覧。丸め表示として使用 |
| I4 | G1_P1/r1/closed_P3_tailの原row JSON | 正確な有理数費用、N、moment/range、bias、acceptance、scope |
| I5 | G1_P1/r1/closed_P5_fullの原row JSON | I4と同じ対象での原値照合 |
| I6 | D3数学確認 | Hermitian contraction、積分剰余、非unitary有限平均、finite-bit/native会計 |
| I7 | return集約技術ノートv1.0の§1〜5等 | 全形式return、正値性、local query、zero-fill、P5群構成、複雑性 |
| I8 | G6の既知手法・method delta監査 | 既知の構成部品と結合手順の差、新規性未確立の範囲 |
| I9 | D3 v2 terminal | 36合成、48行、runs=1/retry=0、source/authorizationとSTOP |

文献引き継ぎbranchのremote HEADが`d3a232a446062e6baadef2b69eab87faa0285363`であることを確認した。

本レビューは全348出力、全1890 event、全219303 pair、全合成sequenceを独立に再監査したものではない。以前の2579項目監査・369項目分析・focused testsの完了は、Codexが保存した記録として扱い、本レビューで再実行したとは表現しない。

### 1.2 外部一次資料

Resource-Optimal IS、SCU/CTS、Sparse Probabilistic Synthesisの指定HTML本文を直接再取得し、関連定理・task・費用会計の範囲を確認した。PTSC v2はPDFの解析テキストを取得し、leading pairingの関連箇所を確認したが、webのPDF screenshot取得はcache errorとなった。図を読み取ったとはしていない。

PR元論文v2およびWan–Berta–Campbell v2のHTML/PDF再取得は、この環境のwebで失敗した。該当部分はI1/I8の固定された本文取得記録・式locator・照合に依拠する。Zhao–Yuan、Ross–Selinger、Aomoto–Katoも本回に全本文を独立再読したとはしない。

これはCodexが8論文の本文を取得したことを否定するものではない。**Codexの取得済み範囲と、今回GPTが直接再取得した範囲を区別する。** 網羅的なpriority確定は本レビューの達成事項に含めない。今回の限定的な研究方針判断に必要な内部証拠は取得できている。

### 1.3 実行範囲

新しい科学runner、合成、回路生成、compile、provider/signal行列、sampling、pair評価、N再計算、最適化は実行していない。本文中で約1.7976%、約2.0035%等を補助表示するために、保存された整数Nと表示単価のDecimal比算術だけを行った。これは新しい登録資源表や正確な有理数正本の置換ではない。

## 2. 研究taskを正確に固定する

D1〜D3の数値入力は既知developmentの明示3-system-qubit synthetic involution providerである。

\[
p=(1/5,3/10,1/2),\quad \tau=5/7,\quad \sigma=+1,\quad \rho=|000\rangle\langle000|,
\]
\[
Q_0=Z_0,\quad V_1=R_{XX01}(\pi/4),\quad
V_2=R_{XX12}(\pi/4)R_{ZZ01}(\pi/4),\quad Q_i=V_i^\dagger Z_iV_i,
\]
\[
R=\sum_i p_iQ_i,\qquad U=\exp(-i\tau R).
\]

回転規約は\(R_P(\theta)=\exp(-i\theta P/2)\)。同じr内の理想有限対象は

\[
M_r=[P_5(-i\tau R/r)]^r,\quad r\in\{1,2\}.
\]

比較方式はordinary、partial-return-tail、closed P3+tail、local full、closed P5、matched CTS。**closed P3+tailは同じP5を実現する表現であり、Taylor targetを3次へ落とした手法ではない。**

r=1とr=2では有限平均は異なる。共通なのは物理的target Uとaccuracy/confidenceである。係数・角度・native演算子を有限精度にした平均も、ideal M_rとexact同一とはせず、別のbias上界で扱う。

各axis accuracyは1/200、alpha_axisは49/24000、各rowのresource failureは1/12000。各12-row ledgerの割当和が1/20となる。四ledgerは代替forecastであり、96 axesを同時に測った一つのconfidence 0.05実験ではない。

Nは共通Bernstein式による十分試行予算であり、実量子shot数でも、必要最小試行数でもない。local fullのzero trialは古典draw後・量子初期化前に処理され、全試行数の分母へ含める。

Tはnative 1Qの部分集合なので、T+1Qを独立資源として加算しない。hはaccepted whole quantum call当たりの共通準備・読出し単価。formalな非負h領域と、実回路が持ち得る費用vectorを区別し、古典online費用はNOT_ACQUIREDのままにする。

DF rank、分子geometry、軌道basis、deterministic splitはこの入力に適用しない。別途のDF基底費用診断をD3の寄与率へ混ぜない。

## 3. D1〜D3で得られた結果の評価

### 3.1 D1からD3までの役割

D1は、同じ有限表現の局所比較を、Taylor・native・統計誤差を戻した同一exponential taskの十分資源比較へ接続した。D2は同じ保存law・N・価格のまま、全方式へ共通のliteral境界相殺を追加した。D3は誤差保証と合成価格を二因子として変更した。

D2後も固定r=2でP5のpartial比T費用低下は残った。D1の約5.8358%からD2の約5.8557%への変化であり、境界相殺が5.86%を新たに生んだのではない。既存結果を支持するが、任意compilerへの頑健性とはしない。[I1/I2および既存D2報告]

### 3.2 四ledgerのT最小とprimary

| ledger | 保証 | 保存価格 | 全12構成のh=0 T最小 | primary partial/P5内のh=0 T最小 |
|---|---|---|---|---|
| G0_P0 | \(3x^6/720\) | strict 1e-6 | r2 P5 | r2 P5 |
| G1_P0 | \(x^6/720\) | 同じ旧価格 | r1 P5 | r1 P5 |
| G0_P1 | 旧保証 | strict 1e-7 | r2 P3+tail | r1 P5 |
| G1_P1 | 改善保証 | 同じ新価格 | r1 P3+tail | r1 P5 |

native誤差会計は四ledgerとも旧1e-6に固定されている。P1の余剰精度をN削減へ戻していない。[I2/I3/I6]

primaryで各方式に同じr=1/2の選択を許したP5のT削減率は、partialを分母に次の保存値である。

| ledger | P5の削減率 |
|---|---:|
| G0_P0 | 1.360214% |
| G1_P0 | 1.161236% |
| G0_P1 | 0.712907% |
| G1_P1 | 1.195366% |

したがってP1で全方式の最良がP3+tailへ変わることと、primaryのP5優位が維持されることは両立する。固定r2の5.86%を、自由にrを選べるprimaryの利益へ読み替えない。

### 3.3 G1_P1/r1のP3+tailとP5

| 項目 | P3+tail | P5 |
|---|---:|---:|
| N/axis | 1,103,489 | 1,083,653 |
| 条件付きT単価 P0 | 143.277318797 | 139.489993570 |
| 条件付きT単価 P1 | 159.700056809 | 162.899691745 |
| P0のT切片 | 316,209,890.483 | 302,317,500.005 |
| P1のT切片 | 352,454,511.977 | 353,053,479.316 |
| moment上界 | 1.690512970 | 1.660089378 |
| range上界 | 1.300197281 | 1.288444558 |

費用の小数は保存報告の表示値で、原row JSONにexact rationalがある。[I2/I4/I5]

P5はP3+tailよりNが約1.7976%少ないが、P1の単価は約2.0035%高い。後者が前者を小さく上回り、P3+tailが598,967.339 T低くなる。P5を分母にした差が0.169653%である。

これは固定law・固定価格・固定予算式における確定的な算術差であり、Monte Carlo標本の揺らぎや丸め誤差により「差があるか不明」と扱わない。一方、別の合法sequenceを得たときにも符号が同じという保証はない。

### 3.4 root keyが説明するもの

G1/r1のP5−P3 marginは、旧価格で−13,892,390.478 T、新価格で+598,967.339 T。価格差による移動は+14,491,357.818 Tである。

P5 rootのK32（ratio 2189372/2921709）は68→80 T、P3 rootのK33（310/413）は70→78 Tとなった。両keyのmargin寄与は約+45,675,681.914 Tと−30,666,353.855 T。他keyと合わせて上記移動へ戻る。[I2]

この記述は全eventの一対一因果比較ではない。全表現平均が同じfinite targetであっても、異なる表現の個別root回転は同じunitaryではない。寄与の完全保存は費用会計の整合であり、独立した物理的因果割合の測定ではない。

### 3.5 保証によるNの差

| 量 | G0/r1 | G1/r1 | G0/r2 | G1/r2 |
|---|---:|---:|---:|---:|
| Taylor積上界 | 0.000553376286 | 0.000184458762 | 0.000017293084 | 0.000005764345 |
| axis bias上界 | 0.001138752588 | 0.000400917540 | 0.000098587265 | 0.000075529418 |
| 統計残余幅s | 0.003861247412 | 0.004599082460 | 0.004901412735 | 0.004924470582 |

六方式でr1のNは約29.5%減り、r2では約0.93%減る。回路や価格の改善で生じた差ではなく、同じtaskを保証するための十分予算の改善である。[I2/I6]

## 4. 本レビューの追加的な科学的解釈

本節は元結果のclassificationを変更するものではない。保存値・固定契約から導くGPTの解釈と解析を明示する。

### 4.1 G0とG1は、実行コストが同じなら対等な運用候補ではない

現在の仮定では、G1の上界はG0より弱くない。共通moment/rangeに対し

\[
\delta_1\le\delta_0\Rightarrow b_1\le b_0\Rightarrow s_1\ge s_0,
\]
\[
N_1=\left\lceil\ell\left(2M/s_1^2+4W/(3s_1)\right)\right\rceil
\le N_0.
\]

同じ実装費用を使う固定構成の非負資源forecastでは、G1はG0より大きくならない。整数ceilがあるので同値の場合は許す。G1の仮定を既存情報から満たし、その証明を使える状態なら、G0は保守性の影響を測る対照として重要でも、常に積極的に選ぶべき保証ではない。

これは「G0の旧結果が誤り」ではない。旧の保証も有効である。またG1が可能な限り最強の保証だとは認定しない。

**運用の議論と、保証を切り替える診断実験の議論を分ける。** 約29.5%という数値は「物理的な誤差が29.5%減った」でも「必要最小shot数を29.5%改善した」でもなく、利用可能な保証を強めた十分予算の削減である。

### 4.2 P1でP3が勝つことは、P1を採用すべき証拠ではない

D3は全caseでnative誤差会計を1e-6に固定した。従って、旧P0のsequenceは現在の最終taskに引き続き適格である。

保存48行では、同じ構成・同じGについて、P1はP0よりT/1Q費用が増え、CX・N・prep slopeは同じである。[I3] この保存モデルの量子費用だけを目的にするなら、P1への変更に新しい資源上の利得は生じていない。

例えば、G1_P1/r1のP3+tailは約352.455 million Tだが、G1_P0/r1のP5は約302.318 million Tで、後者も現在の最終accuracy条件を満たす。P1内の順位逆転と、使用可能な全実装の採否は異なる。

ここから新しい48候補の登録最適化結果を作ったわけではない。同じrowの保存値と適格性からの比較上の帰結である。primitive自体に1e-7必須という別taskを設定した場合にはP0を適格と扱えないが、その追加要求はD3の現行最終taskに勝手に加えない。

P1は、実際に取得した合法sequence価格に対する依存性の確認として価値がある。新精度による誤差余裕まで費用へ戻す最適precision選択は、D3が答えた問いではない。

### 4.3 factorial interactionは、固定rowでは乗法会計から生じる

構成i（方式とrを含む）、保証g、価格p、資源kを考える。D3では

\[
C_{i,p,k}=r_i Z_i^{r_i}d_{i,p,k}-S_{i,r_i,k},
\]
\[
F_{i,g,p,k}(h_k)=2N_{i,g}\{C_{i,p,k}+Z_i^{r_i}h_k\}.
\]

Nは価格pに依存せず、Cは保証gに依存しない。この契約の下で差の差は

\[
\boxed{\Delta_{GP}F_{i,k}=2(N_{i,1}-N_{i,0})(C_{i,1,k}-C_{i,0,k}).}
\]

h項は価格変更で不変なので差の差から消える。正の費用に限れば、同一構成・同一hについて

\[
F_{11}F_{00}=F_{10}F_{01}
\]

でもある。整数ceilはN_gに含めたままで成立する。

このinteractionが非零であること自体を、新しい量子相関機構や独立した相乗効果の発見と解釈しない。実際の最小構成を選ぶmin操作では勝者が切り替わるので、選択問題には非自明な離散的変化がある。しかし、その変化と固定rowの積の恒等式は区別する。

D3の二因子設計は、このような区別を可能にした点で妥当である。interactionが既知の算術で説明できることは、実験設計の失敗ではない。

### 4.4 正規化だけでは選べないことの、具体的な証拠になった

G1_P1/r1ではP5のmoment・range・Nは小さいが、総TはP3+tailより大きい。[I4/I5]

同じr、acceptance=1、h=0なら

\[
F_5<F_3\quad\Longleftrightarrow\quad
\frac{d_5}{d_3}<\frac{N_3}{N_5}.
\]

右辺は約1.01830475、保存単価比は約1.02003528であり、P5の許容できる単価増加を実際の増加がわずかに越えている。この式は一般的な費用×試行数の算術であり新定理ではないが、D3の逆転を曖昧な「実装依存」で終わらせず説明できる。

この結果は、**固定した十分予算・nativeモデルの下で、係数質量やmomentの小ささだけでは最小総T方式を選べない**ことの実例である。

「任意proposalを最適化してもP5が不利」「あらゆる実装でP3が優れる」という主張ではない。反例の射程を、固定された比較policyより外へ広げない。

### 4.5 小さい差の価値と限界

G1_P1/r1でP5−P3のT差をh込みに戻すと、保存表示値では

\[
\Delta F_T(h_T)\simeq598967.339-39672h_T.
\]

従って両者のtieは約15.097987で、P3+tail/r1が全方式のT最小となる最初の領域と整合する。[I2]

0.170%は固定モデルでの確定した小差である。一般的な優位性の根拠としては弱いが、単調なproxy選択が常に正しいという主張を崩す実例では、差が大きい必要はない。この二つの評価を混同しない。

また、root keyを連続的に約0.157 T動かせばtieとなる記号感度は、合法sequenceをその価格で合成できる証拠ではない。Tは整数であり、各合成法には追加制約がある。記号的な価格感度と、実現可能な実装集合に対する頑健性は別である。

## 5. 前向き検証と事後説明を正しく区別する

D3は結果前にG0/G1×P0/P1、対象key、精度、比較範囲を固定している。従って、価格を見てから都合のよいケースだけ追加した検証と扱うのは不正確である。

一方、得られた全key価格と全使用係数を代入すれば順位を再現できることは、未取得環境での有効な選択予測を示したことにはならない。これは精密な会計モデルの検証・機構説明であって、情報圧縮された予測器や、未見providerで保証された選択則の検証とは異なる。

今回の評価は次のとおり。

- **達成**：事前固定された保証・価格変更に対し、比較順位の維持・反転を取得し、保存値で機構を分離した。
- **未達成**：新しいprovider、取得前の未知合成価格、別の最適proposalやcompilerに対し、構成選択を保証または実証した。
- **不要な要求**：固定モデルにおける非単調性の存在を示すためだけに、独立providerを必須とすること。
- **必要になり得る要求**：平均的・広範な性能、スケーリング、実用的な予測規則を主張するなら、その主張に合う独立検証を設計すること。

## 6. 既存成果の中で、理論的な中心として残すもの

D1〜D3の順位表だけでは、独立した方法研究の中心として弱い。既存G系列の以下の構成を本文へ戻すべきである。

### 6.1 同じ有限Taylor演算子の全形式return集約

正の確率p_iとHermitian involution Q_iに対し、raw語の隣接iiを消去したreduced word uを使う。

\[
a_u=\sum_{\substack{|u|\le n\le m\\n-|u|\;\mathrm{even}}}
(-1)^{(n-|u|)/2}\frac{x^n}{n!}P_n(u).
\]

ここでP_n(u)は長さnのIID語がuにreduceする確率。これはQ_i^2=Iによる形式returnだけであり、Pauli交換・反交換等の追加恒等式をすべて収集するものではない。[I7]

一般の奇数語をinvolutionと仮定せず、偶数parent uと単一labelのchild iuを組み合わせ、Q_iによる一回転へ戻す。理想平均の同一性とcontrolled phaseの保持を、構成上の明示事項とする。

### 6.2 正値性と局所生成

既存ノートのcover argumentにより

\[
P_{n+2}(u)\le(n+1)\chi P_n(u),\quad \chi=\sum_i p_i^2,
\]
\[
\frac{t_{n+2}P_{n+2}(u)}{t_nP_n(u)}\le\frac{\chi x^2}{n+2}\le\frac12
\]

が得られる。0<x<=1の範囲でalternating sumからa_u>0を示し、局所parent/child分布を構成できる。[I7]

この証明はtoyの数値一致より広い入力を扱う。一方、Euler pairing、Green再帰、zero-fillという各部品の原理を新発明と主張しない。[I8]

一般local器では、全parent表と未知global normalizerを入力に要求せず、既知のordinary envelopeと局所queryで試行を生成する。拒否trialを0として全試行分母へ戻す。controlled Q_i・controlled rotationへのaccessや、その実装価格は無料にしない。

### 6.3 P5の高々L²+1群の具体構成

P5ではeven parentの長さが0/2/4であり、root 1群、ordered pair L(L−1)群、長さ4のfirst-label L群にまとめられる。群数は高々L²+1。[I7]

これは「normalizerを全word/child表からしか計算できない」という実装に対する具体的な別の取得手順である。ただし、ordinary RTEや層別CTSも非列挙生成の手段を持つため、全既知法に対する指数的高速化とは呼ばない。

群構築のO(L²)は有理数演算数の記述であり、任意precisionのbit complexity・実時間・native合成の費用まで含まない。群選択が線形scanなら全samplerをO(L)とは呼べない。Lはinvolution項数で、qubit数やDF rankと自動的に同じではない。

### 6.4 科学的評価

この構成・生成法は、単なる特定合成keyの順位表よりも、方法論として中心に置く価値がある。ただし、今回確認した先行文献に全結合が見当たらないことだけでは、独立priority・非自明性の証明にはならない。

**本レビューが支持するのは、この具体的な構成を提示し、その仮定・保証・生成情報・資源上の限界を一体として論じること。あらゆる構成部品の新規性や、既知法全体を凌ぐ新アルゴリズムを認定することではない。**

## 7. 先行研究との関係と、主張してはいけない新規性

| 系統 | 既に存在する内容 | 今回の構成との境界 |
|---|---|---|
| Resource-Optimal IS [W1] | 固定protocolで費用とweight二次momentを合わせた最適sampling、実装固定時のbias不変 | D3はproposalを固定した表現比較。range・ceil付き十分予算の最適性は同定理から出ない |
| SCU/CTS [W2] | TaylorのPauli収集、実虚部の回転化、反復、統計/bias配分と総回路費用 | matched CTSは同finite operator taskへの特殊化。元channelのleft/right normalizationをそのまま移さない |
| SPS [W3] | process辞書のL1最適化、短い回路とsampling overheadの交換 | 本研究はphase-sensitive first operator mean。joint controlled spaceで接続の可能性はあるが未実装 |
| ordinary RTE・PR [I1/I8] | 次数pairing、乱択unitaryの平均、制御付き位相信号、有限cutoff | returnを先に形式集約した具体lawと必要な情報・角度が比較対象 |
| PTSC・modified Taylor [W4/I1/I8] | leading pairing、identity/低次数への寄与吸収 | 固定finite P5の全形式returnと同一lawかは個別にそろえる必要がある |
| Clifford+T合成 [I1/I8] | 角度・精度に依存するT最小化、条件付き最適性 | 固定tool/seed/capの実取得価格を理論最小Tと認定しない |

W1の単純化された目的では、正のcost c_e、固定coefficient a_eについてCauchy–Schwarzから

\[
\left(\sum_e q_ec_e\right)\left(\sum_e\frac{|a_e|^2}{q_e}\right)
\ge\left(\sum_e|a_e|\sqrt{c_e}\right)^2
\]

となり、適切な正規化の下でq_e∝|a_e|/sqrt(c_e)が等号を与える。この原理だけを今回の新しい選択理論とはしない。zero-cost atom、range、ceil、情報取得、全経路価格を含むD3の実用的な比較へそのまま最適性を移すこともできない。

またSCU/CTSのSupplementary Note4.1には誤差・confidence・sample数・CNOT総費用の会計がある。従って「統計とbiasを分離してgate数×shotsを比べること」をD3が初めて示したとはしない。

**研究の独自性を支える可能性があるのは、特定のfinite operator表現と生成手順をどの仮定で実現でき、その統計的な改善がどこで実装資源へ変換されないかを、対応した証拠で示す部分である。**

## 8. 支持する主張／支持しない主張

| 主張 | 本レビューの判断 |
|---|---|
| D1〜D3の固定モデル内の費用差・順位逆転 | 支持する。原exact表・モデルの範囲を明記 |
| P5はprimary partialよりTが低い | 調べた四ledger・登録r・共通h条件の範囲で支持 |
| 新価格のG1_P1/r1ではP3+tailがP5よりTが低い | 支持。ただし同finite P5、hと価格条件付き |
| normalization/momentが小さければ、実装総Tも必ず小さい | この固定十分予算モデルの反例が得られた |
| G1は旧G0より正しい物理時間発展を生成する | 支持しない。保証だけを変えた比較で回路は同じ |
| factorial interactionは新しい非自明な量子結合効果 | 支持しない。固定rowでは2ΔNΔCで説明できる |
| 新精度P1を採用するのが総資源上最善 | 支持しない。現taskではP0も適格で、精度余裕をNへ返していない |
| P3の一般的優位・P5の無価値 | どちらも支持しない |
| 一般のPR/QPE・分子・DF総資源の改善 | 支持しない。対象が異なる |
| 全体compiler・最適ISを含む最適方式の確立 | 支持しない。取得・最適化していない |
| 局所生成・P5少数群構成の方法としての価値 | 支持する。仮定と既知部品・priority未確立を併記 |
| 独立論文として完成・受理可能性の確定 | まだ認定しない。実際の構成と主張を一体化した原稿で評価する |

## 9. 研究方針の選択

### 採択：構成・生成法＋有限精度資源の成立条件・限界

提案する中心の問いは次である。

> 全形式returnを取り込んだfinite RTEを、位相と有限bitの保証を保ってどのように生成するか。その集約が十分測定予算を減らしても、native総資源を必ず減らすわけではないのは、どの費用因子によるか。

現在の証拠はこの問いへつながる。構成部分は一般入力に対する数理を持ち、D1〜D3は実取得価格による具体的な例と非単調性を与える。

### 今回採択しない方向

| 方向 | 採択しない理由 |
|---|---|
| P3が0.170%勝つことを主題にprecision/seedを増やす | 値上がり差を反復測定するだけでは、方法の独自性や有用な一般条件は増えにくい |
| P5の普遍的総資源優位を実証する方向 | 既に全方式で資源trade-offと順位反転があり、主張が現在の証拠と合わない |
| 別providerを直ちに追加 | 一般化に必要となり得るが、今の原稿のどのclaimを識別するかが先。任意一例追加だけで一般性は得られない |
| whole compiler・最適IS・SPSを全て追加 | 主張を広げない限り一律の必須条件ではなく、複数の新自由度で機構が不明瞭になる |
| 全てのTrack B成果を終了する | 有効な構成・限定positive・機構知見は残っている |
| FRやDF基底費用を新主軸へ自動統合 | 今回のD3証拠でその採否を更新していない。別のtaskと契約を保つ |

本判断は「新しい手法なら何でも必要」という基準を追加しない。一方、単に仕様と表が詳細であることだけを独立科学貢献とも認定しない。

## 10. 原稿としての着地点

仮題：

**有限Taylor RTEのreturn集約：構成的samplingと有限精度資源の成立条件**

英語の作業題例：

**Return Aggregation in Finite-Taylor Randomized Time Evolution: Constructive Sampling and Implementation-Dependent Resource Trade-offs**

### 本文で狙う三つの貢献

1. 同一finite operator meanのreturn集約を、short-step正値性、局所係数取得、明示sampler、P5少数群特殊化、finite-bit/phaseの保証として記述する。
2. coefficient/proposal/weight、予算保証、回転合成価格、境界簡約、準備費用を分離した評価を行い、どの変更で何が変わるかを明確にする。
3. primaryの限定利益と、P3+tail/CTSを含む非単調な資源trade-offを示す。普遍的な勝者の提案ではなく、normalizationや単発gate数だけによる評価の限界を具体例で示す。

これらの書き方は「世界初」を要求しないが、既知理論を再発見したようにも書かない。既存部品を用いた具体構成の価値と、一般原理との重複を本文で正直に記す。

### 推奨する構成

- Introduction：finite-RTEの表現選択、normalizationと実装費用のずれ、既知手法との位置関係。
- Problem and access model：Q_i、controlled primitives、finite first moment、accuracy/confidence、情報取得の範囲。
- Constructive return aggregation：係数正値性、parent/child pairing、local生成、P5特殊化。
- Finite precision and resource accounting：digital係数とproposal、mean bias、Bernstein十分予算、literal/native費用。
- Results：G10を必要なanchorに限定し、D1/D2/D3を問い順で配置する。全48行は付録へ。
- Mechanisms and limitations：root価格、保証の効果、境界相殺、formal h領域、cross-ledgerの解釈。
- Discussion：何を一般化しておらず、どの追加claimにどんな証拠が必要か。
- Appendices：完全な証明、fixed contract、exact値・version・provenanceの索引。

図表は既存値だけから作れる。例えば、構成の流れ、保証×価格の選択表、P3/P5のroot価格margin、主要h領域の四つを核とし、監査項目数や全crossingを主図へ詰め込まない。

## 11. 追加検証を選ぶ条件

現時点では、新しい数値系列を始めなくても本体原稿を作れる。以下は将来のclaimごとに必要性を判断する対応表であり、実行認可ではない。

| 将来の主張 | 必要となる追加証拠の種類 |
|---|---|
| 異なるHamiltonianでも典型的に利益が出る | 結果前に選定した独立入力群と、同じ比較機会・失敗処理 |
| 未取得価格でも有利な表現を予測できる | 使用情報を限定した事前予測と、予測に使っていない価格・実装での検証 |
| 既知samplingを最適化しても新表現の利益が残る | 同target/control/access/finite-confidenceをそろえた強いproposal対照 |
| 古典生成も安い | setup/online/zero試行/random bits/角度取得を分離した測定・複雑性 |
| 実回路全体・fault-tolerant runtimeで有利 | 共通compiler・相対位相・ハードウェア費用・深さ等の実装モデル |
| 分子PR/QPEを改善する | deterministic部分、DFアクセス、state準備、長時間、位相推定まで含む契約 |

一つの限定的な方法・資源ノートを書くために、上の全項目を一括で満たす必要はない。逆に、「限定ノートだから比較は何でもよい」ということでもない。本文で実際に採るclaimの根拠に必要なものだけを補う。

## 12. 次のCodex作業と承認境界

### 直近の担当と成果物

**次の担当はCodex。新しい文献棚卸しや保存値解析ではなく、上記方針に沿った一体原稿の作成である。**

具体的には、既存return集約ノートを土台に、D1〜D3と文献照合を本文の論理順へ組み込み、方法・定理・資源結果・非主張を一つの原稿へまとめる。既存claim/evidence表は利用し、同じ表だけを再び作って完了としない。

原稿上では、次を必ず維持する。

- P3+tailが同finite P5を実現すること、r間のfinite meanの違い。
- Nは十分予算であり、必要最小・実測shotsではないこと。
- G1の保証改善とP1の実装価格の効果を分離すること。
- P1の順位を、precision自体の最適採択へ読み替えないこと。
- factorial interactionを2ΔNΔCの会計的な項として説明すること。
- primaryと全方式、固定rとequal-r自由度、単一metricとquantum Paretoを分離すること。
- coefficient/query/bit complexityと実時間・native取得費用を分けること。
- 既知手法との完全同一性が未確立なことと、独立priorityが証明されたことを混同しないこと。

技術的な整形、引用の配置、既存証明の説明補完、図表作成は一つの作業にまとめる。新しい実験、未登録のN再計算、原結果の再ランキング、precision/seed/provider/r探索、SP/DF/分子への展開は含めない。

### 科学的にGPTへ戻す条件

既存資料では中心命題が証明できない、既知手法と結合構成が完全に同じと判明した、比較の公平性や制御位相に根本的な疑義が見つかった、または本文の主要claimを成立させるために新実験が不可欠となった場合にGPTへ戻す。

新しいcommit、テストPASS、表の整形完了だけを理由に、詳細GPTレビューを要求しない。完成した一体原稿の主張・証拠・位置付けは、次の重要科学レビューのまとまった対象となり得る。その開始には改めてユーザーの明示承認を必要とする。

### 保全と実行認可

本レビューは研究方針の判断であって、Codexを現在起動したものではない。実際の原稿作業はユーザーの指示に従って着手する。

旧source/result/contracts/authorization/消費済みmarker/STOP、親dirty変更、Track A/N3、既存のclosureを保護する。今回の原稿方針を理由に旧runを再開しない。新しい科学実行は別途の明示承認を必要とし、mandatory STOPを維持する。

## 13. 確かさと限界

固定条件下のD3数値関係、保証・価格の分離、P3+tailとP5の単価/N差についての確信は高い。正確な原row、保存値分析、契約と数学的な会計が対応している。

一方、世界的なpriority、最適IS・最良compilerを含む資源優位、一般入力での有効な選択器、独立論文の受理可能性については、本証拠だけで確定しない。これらを未確定とすることは、既存の構成や小さい確定的counterexampleの価値を否定するものではない。

**最終判断：既存成果の発展は継続する。D1〜D3の現在の数値系列は一区切りとし、「構成・生成法＋有限精度資源の条件付き評価」を中心とする一体原稿へ進む。新しい勝者探索や無目的な独立provider追加は開始しない。**

---

## 参考資料・固定URL

### 内部資料

- [I1：D1〜D3一次文献・claim/evidence引き継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md)
- [I2：D3最終機構分析](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md)
- [I3：D3表示用48行CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/resource_rows_display_v2.csv)
- [I4：G1_P1 P3+tail/r1原row](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/artifacts/track_b_d3_result/2026-10-14/v2/rows/G1_P1/row_02.json)
- [I5：G1_P1 P5/r1原row](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/artifacts/track_b_d3_result/2026-10-14/v2/rows/G1_P1/row_04.json)
- [I6：D3数学確認](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dceccaa774229527a46a1386131583a4f618f8ad/docs/tracks/algorithm_codesign/d3_mathematical_confirmation_20261013.md)
- [I7：return集約技術ノートv1.0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md)
- [I8：G6 prior-artとmethod delta](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)
- [I9：D3 v2 terminal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/artifacts/track_b_d3_result/2026-10-14/v2/terminal.json)
- [D3結果・監査入口](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md)
- [D3事前計画](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e1011e3cb7e306a1cc992644aa70af4bf6ff514/docs/tracks/algorithm_codesign/d3_guarantee_price_preregistration_v1_20261013.md)
- [D2結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1c59b28f4370cc7be65cff60d1ec877fd1491f5/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md)
- [D1結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md)

### 外部一次資料

- [W1：Cugini–Atif–Subaşı, Resource-Optimal Importance Sampling for Randomized Quantum Algorithms, arXiv:2603.13495v1](https://arxiv.org/html/2603.13495v1)。本回HTML関連本文を直接確認。
- [W2：Peetz–Smart–Narang, Quantum Simulation via Stochastic Combination of Unitaries, arXiv:2407.21095v2](https://arxiv.org/html/2407.21095v2)。本回Theorem 1、Methods、Supplementary Note4.1等のHTML本文を直接確認。
- [W3：Koczor, Sparse Probabilistic Synthesis of Quantum Operations, arXiv:2402.15550v2](https://arxiv.org/html/2402.15550v2)。本回process・sampling・NMRのHTML関連本文を直接確認。
- [W4：Zeng–Sun–Jiang–Zhao, Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations, arXiv:2212.04566v2](https://arxiv.org/pdf/2212.04566v2)。本回PDF解析テキストの関連箇所を確認。web screenshot取得は失敗。
- [PR元論文 v2](https://arxiv.org/pdf/2503.05647v2)、[Wan–Berta–Campbell v2](https://arxiv.org/pdf/2110.12071v2)、[Zhao–Yuan v2](https://arxiv.org/pdf/2103.07988v2)、[Ross–Selinger v3](https://arxiv.org/pdf/1403.2975v3)、[Aomoto–Kato](https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf)の対応はI1/I8の取得版・関連locatorを継承。今回の独立全本文取得とは数えない。

このMarkdown自体はGPTレビュー原本であり、引用先の原データやCodexの監査資料の代替ではない。
