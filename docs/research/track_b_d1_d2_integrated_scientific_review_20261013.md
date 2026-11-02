# Track B：D1・D2・最終保存値分析の統合科学レビュー

- 作成日：2026-10-13（JST）
- 開始承認：利用者の「レビューを開始して」
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 最新の確認対象：`4cf2827c14fec96613a60bb1427287b30766d5da`
- D2結果：`d1c59b28f4370cc7be65cff60d1ec877fd1491f5`
- D2 source：`28fc3ca17ec51bb7656a498e3eea9b72966eb3fa`
- D1結果：`fc96571881f301435a655f057709df4d6c1678e0`
- D1保存分析：`2dcd56737be35526f6436a52835f59d817adf30c`
- 適用した共通指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`
- 状態：**重要科学レビュー完了。既存成果の発展研究は継続を推奨。次は保証・合成価格に対する限定的な頑健性検証の設計へ進む。科学実行は別途明示承認が必要。**

## 0. 結論

D1・D2を、現在の固定task、指定された誤差・統計予算、literal実装・隣接inverse相殺の範囲の研究証拠として支持する。共通境界相殺を認めても、固定r=2におけるclosed P5のpartial-returnに対するT費用の低さは残った。一方、CXの最小包絡線と、登録したr=1/2間のT選択は変化した。[R1–R4]

この結果は、次のRQを継続する材料になる。

> 同じ物理時間発展を、同じ精度・confidenceで推定するfinite-RTE表現について、集約の統計的利益、有限精度合成価格、境界簡約、利用する誤差保証が、どの条件で資源利益へ結び付くか。

ただし、現段階の結論を「P5が一般のPRで5.86%優れる」「r=2が最適」「約85%のDF基底費用を相関で削減できる」へ拡張しない。最後のDF診断は別の入力・会計であり、本レビューの証拠へ混ぜない。

今回、前回のレビューから追加して重視するのは、次の三点である。

1. **比較自由度を揃えた利益幅。** r=2同士のT差は約5.8557%だが、両方式に登録済みr=1/2の選択を認めたh=0比較は約1.36%である。これは元の結果を否定せず、主張する比較を区別するもの。
2. **選択境界と実用効果の区別。** partialがCX最小となる連続h領域は存在するが狭い。価格vectorの実現可能性を確認せず、実際の実装条件が三領域を取るとは主張しない。
3. **保証の保守性。** 現契約のTaylor上界は有効だが、Hermitianな現在のtargetについて、同じ入力情報からより強い積分剰余上界を導ける。rの順位が約0.20%の差で入れ替わったことを、保証policyを越える結論にしない。

次は、既済の保存値分解や全交点表の反復ではなく、**保証policyと合成価格の二つの軸を分けた小さい前向きな検証計画**をCodexで具体化することを推奨する。D2の深さ別完全帰属、汎用compiler、分子、SP、全r gridを一度に追加しない。

## 1. 証拠の取得と確認水準

### 1.1 実際に確認した主要資料

- 最新の最終統合報告の本文、相殺構造の説明、未取得情報、STOP。[R1]
- D2原結果・監査報告、固定比較と実行範囲、全六方式の旧／新費用。[R2]
- D1 key/event機構分析。literal費用、合成価格、先頭rootとその他event、記号的tie。[R3]
- D1保存分析の資源表とfactor分解。[R4]
- 最終統合の表示用summary。表示用数値とexact正本を区別した。[R5]
- D2の固定source `boundary.py`。inverse規則、境界比較、確率質量の集計、費用評価を静的に読んだ。[R6]
- D2 contractとD1 accounting source。旧N/biasを維持する条件とTaylor上界を確認した。[R7,R8]
- 16人工テスト・82領域包含の保存記録。[R9]
- 最新branchのremote HEADが固定commitと一致すること。[R10]
- 前回のD1科学レビュー原本。今回の変更点が以前の判断とどう異なるかを確認した。[R11]
- 限定した外部一次資料：resource-optimal IS、SCU/CTS、Clifford+T合成。[P1–P4]

### 1.2 再実施していないこと

219,303 pairの再評価、D1の1,890 events・36 keysの全再監査、1,218監査チェック、16 tests、82包含検査、全72資料のbyte/hash照合を独立再実行していない。これらの完了は取得したCodex記録として扱う。

新しい行列・Hamiltonian・event・trajectory・回路は生成していない。scientific runner、compile、T合成、sampling、repository tests、LP、GPU処理を実行していない。新しいN・価格・比較policyによる資源表の再採点も行っていない。

本レビューで実行した数値処理は、保存された表示スカラーの比・差・領域幅のDecimal算術だけである。以下で約1.36%等と記した値は表示精度での算術であり、新しいexact rational正本ではない。理論式は手計算による独立解析として、保存結果から区別する。

GitHub connectorでは主要本文・sourceを取得できた。一方、ローカルcontainerからraw URLのファイルを取得する試みはDNSエラーとなった。これはGitHub資料欠損や未pushを意味しない。原資源表の全raw再保存を行ったとは主張しない。

## 2. D1とD2のtaskを正確に限定する

現在の入力は、3-system-qubitのexplicit synthetic involution providerである。

\[
p=(1/5,3/10,1/2),\quad \tau=5/7,\quad \sigma=+1,
\quad R=\sum_i p_iQ_i,\quad U=e^{-i\tau R}.
\]

状態は`000`、比較する有限近似は

\[
M_r=\bigl[P_5(-i\tau R/r)\bigr]^r,\qquad r\in\{1,2\}.
\]

同じrの表現は同じ理想finite meanを対象にする。finite-bitの係数平均まで厳密同一だとはせず、その差を認証上界へ戻す。r間ではfinite approximationが変わるが、目標の物理Uは同じである。[R1,R7]

同じproviderと確率は既に開封済みのdevelopment入力である。D2は新しい実装条件の検査だが、独立providerへのvalidationではない。多数のpairを網羅したことは、この固定lawのcoverageであり、多数の独立Hamiltonianを確認したことではない。

費用はT/CX/native1Qを分ける。native1QにはTが含まれる。式

\[
G_{i,k}(h_k)=A_{i,k}+B_i h_k,\qquad B_i=2N_iZ_i^{r_i}
\]

において、NはBernsteinの十分trial数、Zはstepのnonzero確率、hは共通prepare/readout単価である。Z<1の方式では全trial数と量子初期化回数を混同しない。古典生成費用は未知として残る。

分子geometry、DF rank、orbital basis、決定論PFとのinterleaving、RPE/QPEの最終資源は本taskの検証対象ではない。

## 3. D2が正当に確かめたこと

### 3.1 仮説と結果

D1はstep内の共通exact inverse passを使ったが、step間の相殺を含まなかった。D2は同じ保存step列を時間順に連結し、全六方式へ同じ境界規則を適用した。proposal、weight、N、bias、precision、保存sequence価格は維持した。[R2,R7]

従って、D2が直接答える問いは、

> D1の資源関係が、共通のliteral境界相殺を加えても維持されるか。

である。sampling lawの改良や、新しい乱択相関の検証ではない。

### 3.2 静的sourceから確認できる同値性

`boundary.py`は各stepが内部でreducedであることを確認し、左末尾と右先頭のliteral inverseだけを比較する。自己逆gate、T/t、S/s、W/w、同一keyで符号反転したRを扱う。非可換因子の並べ替えや角度融合を行わない。[R6]

正負Rに同じ保存sequenceとそのactual adjointを使う契約の下では、除去する対は実装sequenceとしても積がIとなる。従って、理想unitaryだけのprojective同値ではなく、controlled相対位相を保持する簡約として扱える。D2の費用評価に新しいdense matrixが必須だったとは判断しない。

ただし、この静的根拠は、全原sequenceをGPTが再認証したことを意味しない。既存のstrict-phase検証とsource条件を継承する。

### 3.3 固定費用表

以下はh=0の期待native切片、単位10^6 gates。保存報告の丸め表示。[R2]

| 方式 | T・D1→D2 | CX・D1→D2 | native1Q・D1→D2 |
|---|---:|---:|---:|
| ordinary | 515.022952→512.961038 | 27.904618→23.780790 | 1344.170250→1337.355225 |
| partial-return | 456.252299→454.483823 | 23.480168→19.943216 | 1163.682829→1157.837668 |
| closed P3 | 443.162079→441.402602 | 23.591285→20.072330 | 1081.327155→1075.509423 |
| local full | 430.039337→428.282208 | 23.532129→20.017871 | 1108.697460→1102.887640 |
| closed P5 | 429.626133→427.870692 | 23.509518→19.998636 | 1107.632166→1101.827928 |
| matched CTS | 643.093987→643.093987 | 15.227487→14.781004 | 1623.187930→1622.073157 |

r=2のh=0順位は変わらない。T最小はP5、CX最小はCTS、native1Q最小はclosed P3。この限定した関係を支持する。

### 3.4 相対差、絶対差、自己削減率を分ける

P5/partialのT差は−5.835843%から−5.855683%へ変化した。しかしP5自身のT削減は約0.40860%であり、約5.86%を新たに境界相殺で得たわけではない。[R2]

保存表示値ではpartialの絶対削減は約1.768476 million T、P5は約1.755441 million T。P5がpartialより少ないという絶対gapは約26.626166 millionから約26.613131 millionへ僅かに縮小している。両者の分母も減ったため、相対比ではP5の優位が僅かに大きくなった。

この算術は、境界相殺がP5固有の大きな利益を作ったという読み方を排除する。D2の意義は、共通改善を対照にも与えた後に関係が残ることと、他の資源選択が動くことにある。

## 4. 反復数を固定した比較と、選択を許した比較

### 4.1 r=2同士の比較

P5とpartialをr=2で固定すれば、T差約−5.8557%は正当な保存結果である。Nとlawが不変であるため、D2で増減したのは費用側である。

### 4.2 両方式にr=1/2を選ばせる比較

h=0で、partialは保存されたr=1の約433.770906 million T、P5はD2 r=2の約427.870692 million Tを選ぶ。表示値からの比は約−1.36%である。[R1,R2,R4]

\[
\frac{427.870692}{433.770906}-1\approx-0.013602.
\]

これは全r最適化ではなく、登録された二つのr内の公平な選択である。5.8557%と1.36%は異なる問いへの答えであり、どちらかで他方を置き換えない。

新しい表現を採用する利益を主張する際は、同じ比較自由度をbaselineにも与える。hが変われば各方式のr選択も変わるため、h=0の差を全領域へ延長しない。

### 4.3 r=2のT最小は僅差である

P5自身では、D1のr=2はr=1より約0.2088%高かったが、D2相殺後は約0.200632%低い。これにより、12候補のT最小包絡線は全h_T≥0でr=2 P5となった。[R1,R5]

この差は保存モデルでは厳密に符号が決まっており、「計算誤差の範囲で同じ」とはしない。一方、compiler、誤差保証、合成価格を変更しても同じrが選ばれることは検証されていない。数値的な再現性と、モデル変更への頑健性は別である。

## 5. CX三領域の解釈

D2 r=2の最小包絡線は以下である。[R1,R5]

| h_CX | 最小方式 |
|---|---|
| 0≤h<6.332292668 | CTS |
| 6.332292668<h<6.637184831 | partial-return |
| h>6.637184831 | closed P5 |

境界では隣接方式がtieする。CTS/P5の直接交点約6.335384ではpartialが両者より小さいため、二方式の交点を全方式の切替と誤読しない。

### 5.1 三領域が生じる数学的条件

以下は三本の直線に関する解析で、新しいアルゴリズム定理ではない。

\[
G_j=A_j+B_jh,
\quad A_C<A_P<A_F,\quad B_C>B_P>B_F
\]

とする。C=CTS、P=partial、F=P5の場合、

\[
h_{CP}=\frac{A_P-A_C}{B_C-B_P},\qquad
h_{PF}=\frac{A_F-A_P}{B_P-B_F}
\]

があり、h_CP<h_PFならPが両者を下回る区間が生じる。他三方式がそれ以下でないことは、保存された全方式包絡線の確認が必要である。

D2ではBを固定したままAの削減量が方式間で異なるため、この交点の順序が変わった。これは「全方式へ同じ簡約を与えても、最適な表現の領域は変わり得る」という限定した設計知見である。

### 5.2 形式的なh領域と、実際のprepare/readout

partial最小区間の幅は約0.304892である。これはMonte Carloの不確かさではなく、固定直線の形式的領域幅である。

本レビューの追加的な条件付き注意：h_CXが各実行で同じ一つの決定論回路のCX個数なら整数であり、この開区間には整数がない。その限定ではpartialの単独CX最小領域は実現されない。

ただし、hは期待費用、複数prepare policyの平均、またはX/Y axesの平均を意味することもある。その場合6.5のような値もあり得る。元契約は連続hの感度を扱っており、実価格をまだ取得していない。従って「partial領域は無意味」とも「実際にpartialが最良になる」とも断定しない。

Tがnative1Qに含まれることによるh_1Q≥h_T等の制約も、実際の価格vectorを選ぶ際に戻す。非負直交領域でのParetoと、特定装置で実現可能な集合でのParetoを区別する。

## 6. 合成key依存性の評価

D1の追加key分析は、既存のfactor分解から一段下り、literal Tと合成T、event確率、angle key、sequence価格を結んだ。[R3]

| r | partial：literal / 合成 / 合計T単価 | P5：literal / 合成 / 合計T単価 |
|---:|---:|---:|
|1|3.075166 / 133.293214 / 136.368380|3.141354 / 136.348640 / 139.489994|
|2|2.734266 / 143.849804 / 146.584070|2.760119 / 136.014108 / 138.774227|

r=2の全単価差約−7.809843のうち、先頭root三件の差は約−7.571561。約96.949%という比は、対応させたcategoryの記述的内訳であり、一意な因果割合ではない。

先頭rootのprimitive価格は、r=1でP568 T対partial66 T、r=2でP568 T対partial72 Tである。同じcontrolled eventに二つのsigned primitiveが含まれ、同じchildのevent差はr=1で+4 T、r=2で−8 Tとなる。[R3]

### 6.1 D2は何に対する頑健性を示したか

D2は共通境界簡約への頑健性を確認した。元の合成価格は変更していないので、合成条件の変化への頑健性を確認したことにはならない。

相殺構造はprovider基底V1/V2の列と整合し、五方式で保存された削減量を説明する集計式が成立した。ただし、深さ×削減gate・keyのjoint表がないため、全pairの由来を完全に確定したとはしない。[R1,R6]

r=2の利益を「angleの合成価格の偶然」と断定するのも、「構造上必ず8 T安い」と認定するのも早い。合法な新sequenceとその誤差を取得した上で判別すべき問題である。

### 6.2 既存の記号価格感度の正しい位置付け

D1のh=0・他価格固定では、P5 rootのtie価格はr=1で約68.933703、r=2で約72.463861と保存されている。[R3]

これは旧D1会計の境界であってD2更新済み境界ではない。また、任意の整数T価格が当該角度・strict phase・精度で実現できるとは限らない。価格を69や73へ代入した反転例を、実在する合成結果として扱わない。

固定law・N・symbolic passの下では一般に

\[
\Delta G_T(\mathbf t,h)=a_0+\sum_k a_k t_k+\Delta B h
\]

と書ける。もし実価格がt_k^0±δ_kに入るという別の根拠があるなら、

\[
|\Delta G_T(\mathbf t,h)-\Delta G_T(\mathbf t^0,h)|
\le\sum_k|a_k|\delta_k
\]

で条件付きの順位安定性を評価できる。これは感度に関する恒等的な評価であり、未取得の価格幅δを勝手に設定しない。D2のkey別除去係数まで保存済みだとは仮定しない。

Clifford+T合成には確率的・近最適な構成があるが、理論の条件付き最適性を、この固定tool・seed・有限停止条件で得られた価格の最小性へ移してはいけない。[P3,P4]

## 7. 追加の解析的論点：Taylor保証には共通に改善できる余地がある

**本節は保存報告に新しい結果として書かれていた内容ではない。本レビューによる独立した数学的補足である。** 元contractの修正・再実行・新しいN計算は行っていない。

### 7.1 現保証は正しいが保守的

D1/D2は、0≤x≤1でexp(x)<3を使い、

\[
\epsilon_{\rm old}(x)=3|x|^6/6!
\]

をTaylorの一step上界とする。sourceとcontractは一致している。[R7,R8]

これは有効な上界であり、D1/D2の結果をinvalidにする誤りではない。ただし現在のQ_iはHermitian involution、p_i>0、Σp_i=1なので、RはHermitianで||R||≤1である。この構造を使える。

### 7.2 積分剰余による導出

実数xとHermitian Rについて、Banach空間のTaylor積分剰余を使うと

\[
e^{-ixR}-P_5(-ixR)
=\frac{(-ix)^6}{5!}\int_0^1(1-s)^5 R^6e^{-isxR}\,ds.
\]

Rと指数は可換で、e^(−isxR)はunitary。従って

\[
\begin{aligned}
\|e^{-ixR}-P_5(-ixR)\|
&\le \frac{|x|^6}{5!}\|R\|^6\int_0^1(1-s)^5ds\\
&=\frac{|x|^6\|R\|^6}{6!}
\le\frac{|x|^6}{6!}.
\end{aligned}
\]

固有値、exact signal、観測された残差は不要で、全方式へ同じ情報で与えられる。標準的な剰余式からの補強であり、新しい独立定理や新規アルゴリズムの主張ではない。

### 7.3 多stepと実装誤差

δ=|x|^6/6!とし、理想finite meanをM、目標unitaryをUとすると||M||≤1+δ。従って、Mをunitaryと仮定せずに

\[
\|M^r-U^r\|\le(1+\delta)^r-1
\]

とできる。実装平均の一step誤差が共通上界eなら、

\[
b_{\rm impl}\le(1+\delta+e)^r-(1+\delta)^r
\]

も同じtelescopingで保守的に使える。

coherent signalへの変換に使った旧factor2、finite-bit、strict native/control、M2/range、confidenceは、別に対応確認する。本節を理由にそれらを黙って省略しない。

### 7.4 なぜ次の判断に関係するか

r=1とr=2ではTaylor biasが異なる。共通のより強い保証を導入すると、統計残余幅の改善量もrごとに異なる。従って、既存policy内での約0.20%のr選択差を、物理taskに固有の最適rだと読むことはできない。

同じr内の各表現には同じ保証を適用できるので、P5だけに都合のよい情報を足す変更ではない。一方、変更後の整数NとT/CX順位は、本レビューでは再計算していない。新policyで勝者が誰になるかを、計算済みの事実として述べない。

研究としては、**アルゴリズムの違い、合成価格の違い、利用する保証の違い**を分けるべき段階にある。

## 8. 支持するclaimと支持しないclaim

| 主張 | 判断 |
|---|---|
| 固定D2の共通隣接inverse相殺後でもr=2 P5のTがpartialより小さい | 支持 |
| 登録6方式のh=0順位は各resourceで維持された | 支持 |
| 連続prepare/readout価格下でCX最小包絡線がCTS→partial→P5へ変わる | 支持。価格実現可能性は別 |
| 登録12候補のT最小は旧保証・価格下でr=2 P5 | 支持。全r/全compiler最適ではない |
| provider基底相殺モデルとaggregateが整合する | 支持 |
| 全pairのdepth/gate/keyへの完全な帰属を認証した | 未支持 |
| D2でステップ間の乱択相関の有効性を検証した | 不支持。lawは独立で不変 |
| P5は最良partialに対して常に5.86%改善 | 不支持。r自由度とhを揃える必要 |
| P5がどのsynthesizer/precision/providerでも有利 | 未検証 |
| 古典費用込みの総資源最良 | 未検証 |
| D1/D2が不正確だからやり直す必要がある | 本レビューではその根拠を確認していない |

## 9. 新規性・科学的意義・論文化

一般的なcost×moment最適化は既知であり、SCU/CTSは短い回路と測定overheadの交換を扱う。Clifford+T合成も既存の技術である。[P1–P4]

従って、三つの交点、価格感度の直線式、隣接inverse消去そのものを新アルゴリズムとして売り出すことは適切でない。

一方、既存のreturn集約・少数群生成・finite-bit構成を基盤に、**どの表現の利益が、どの保証と合成価格の下で残るかを予測し、同taskで前向きに検証する研究**には価値がある。D1/D2はそのdevelopment証拠として有効である。

現段階で独立論文の主要claimが完成したとは認定しない。理由は、新規性のある構成・条件と、既知一般論から当然のprice trade-offの区別がまだ必要であり、同じprovider・固定precisionの結果だけでは一般的な適用条件を確認できていないためである。

ただし、全分子で勝つこと、一定以上の改善率、新数学primitive、全既知アルゴリズムへの優越を要求しない。順位反転を予測・説明できることも科学的成果になり得る。

仮の論文の筋は、

> 有限RTEのreturn集約はいつ実装資源を減らすか：誤差保証、離散合成価格、境界簡約による表現選択の変化。

とできる。これは題名案であり、投稿先や採録可能性の保証ではない。既存構成ノートを理論的な基盤、D1/D2を機構例、次の前向き確認を適用条件の検証として位置付ける。

## 10. 代替の次作業と優先順位

| 次作業 | 情報価値と問題点 | 今回の扱い |
|---|---|---|
| 全保存値のfactor/Pareto/交点を再分析 | 既済作業の反復になる | 行わない |
| 全pairの深さ別完全帰属 | 主aggregateの利用には必須でなく、欠測を無理に埋める動機になりやすい | 現時点の必須作業にしない |
| より強い共通Taylor保証との比較 | 新しい回路取得なしで、r順位のpolicy依存を識別する | 優先して設計 |
| 一つの前向き合成条件への移送 | root価格に依存するT関係を判別する | 次の限定取得候補 |
| 別provider | 一般化には重要だが、今の価格・保証の影響と一度に変えると説明しにくい | その後、主claimに必要なら選ぶ |
| 汎用whole-circuit compiler | 実装の幅を調べられるが、複数passの影響が入る | 直ちに広げない |
| 古典online sampling費用 | 全資源claimには重要。現在のsetup timingで代用不可 | quantum資源研究の次段階を阻む必須欠測とはしない |
| SP、分子、DF、新r grid、相関law | 未解決要因を同時に増やす | 今回開始しない |

## 11. 次の研究段階：保証と合成価格の限定的な頑健性確認

以後の名称は便宜的にD3とする。既存のD3契約が公開・認可済みという意味ではない。

### 11.1 採択する科学的目的

次の二つを分けて調べる。

A. 同じ保存実装・lawに、同じ情報で正当化できるより強い共通保証を与えたとき、表現とrの選択はどう変わるか。

B. 精度を満たす実際の合成sequenceを、結果前に定めた別条件で取得したとき、現在の価格依存機構から予測した関係が残るか。

D2の単純な再実行ではなく、別の問い・別の契約である。

### 11.2 保証側の最小比較

Codexで第7節の導出を独立に照合し、||R||≤1、実数時間、理想polynomialとrounded mean、native/control、signalの定義を対応させる。

旧保証をlegacy policyとして凍結し、改善保証を別policyとする。全方式へ共通適用し、旧N・分類・原データを上書きしない。exact signalや保存小行列残差を予算oracleへ使わない。旧factor2やnative誤差の扱いは、変更するときだけ別の理由を明示する。

新しいNや資源順位の再会計は、保存値の閲覧ではなく新しい科学的比較として扱う。実行前にscopeを登録し、別途の明示認可を得る。source準備や独立した式照合と、登録実行を分ける。

### 11.3 価格側の最小比較

比較の中心はpartial/P5、r=1/2を双方に許した既存四構成。全六方式・12候補への結論を新条件でも述べるなら、他の基準方式にも同じ価格取得・保証機会を与える。部分coverageで「全方式に勝った」としない。

まず一つの新しい共通合成精度を、結果を見る前に選ぶ案を推奨する。より厳しいprecisionを用いて、旧native誤差予算とNを維持する対照なら、価格のみの変化を切り分けられる。そこから保証を変えた比較を別ledgerへ置ける。

精度の具体値、対象key数、source/tool、seed、CPU・memory・output等の上限はCodexが根拠付きで提案・固定する。今回、勝つまでのepsilon gridやseed探索は採択しない。seedを複数用いるなら、実装の確率性を検査する目的と、全結果を残す規則を先に定義し、安いものだけを選ばない。

保証と価格の効果を分離する一案は、旧保証/改善保証と旧価格/一つの新価格を対応させる小さな二因子比較である。必要以上にprecisionとrを増やさない。これは設計案であり、取得件数や実行認可は未確定。

### 11.4 D2の相殺結果をどう引き継ぐか

lawとsymbolic inverse passは固定する。新sequenceにはstrict位相とactual adjointを要求する。原D2のaggregateを、根拠なく新しい合成価格へ流用しない。

D2のper-key除去係数が未保存である点は保つ。新価格会計に必要な係数を静的に認証できるなら利用し、できなければ必要最小限の新D3会計として取得範囲を明示する。無認可で219,303 pairを再評価しない。旧D2の正誤を再検査するだけの反復は避ける。

### 11.5 検証前に何を予測するか

単に「P5が勝つ」と予測しない。

- 保証変更では、どのrの統計残余幅がより影響されるか。
- 単価差は、どのroot・tail・literal部分の価格が境界を越えると反転するか。
- 同じrの順位と、各方式がrを選ぶ場合の順位を分ける。
- h=0、形式的h感度、実現可能なh vectorを分ける。
- 新条件は同providerの前向きな価格確認であり、独立Hamiltonian validationとは呼ばない。

### 11.6 結果後の進路

P5が勝ち続けることだけを成功条件にしない。固定した機構が順位の維持や反転を説明できれば、設計条件・適用範囲の研究として前進である。

逆に、新条件で結果が変わるたびにモデルを後付けし、予測力や独立した知見が残らないなら、主張を固定実装例へ縮小する。既存の構成ノートへ統合し、それ以上のbenchmarkを優先しない選択も残す。

有意味な機構が残った場合に、初めて独立provider等の未開封条件を結果前固定して確認する。全r最適化、全compiler、全分子を一度に証明する計画へ広げない。

## 12. CodexとGPTの分担・実行境界

**次の担当はCodex。** 第11節の目的に沿って、数理・sourceの対応、必要な新取得、対照・誤差・費用のledger、対象key、資源上限を一つの準備作業にまとめる。

人工fixtureの技術テスト、source保護、byte/hash、保存形式、preflightなどは、明示された作業範囲でCodexに任せる。同じ科学契約内の小修正ごとにGPT詳細レビューへ戻さない。

GPTへ戻すのは、coherent semantics・保証の前提・比較の公平性・主要claimに実質的な問題が生じたとき、または採択済みの限定取得後に重要な研究判断をするときである。

今回のレビュー完了は、D3のsource実装・登録実行を無条件に許可するものではない。次の具体的なCodex指示で技術準備の許可範囲を示し、新しい科学実行・再会計・合成には別途明示的な認可を得る。既存契約が独立source確認を要求していれば、省略しない。

旧source・結果・authorization・消費済みmarker・STOP・個別closureは保持する。新しいnamespaceでの無認可再試行も認めない。Track A、N3、DF費用診断へ介入しない。

## 13. 最終判断の確かさ

- **確かなこと**：固定したD2モデル内で、P5のT関係は残り、CX包絡線とr選択が変わった。旧N・lawを変えない比較である。
- **根拠ある解析**：Hermitian targetに対するTaylor積分剰余、直線包絡線の条件、価格感度の式は前提を明示して導ける。
- **未確定なこと**：改善保証・別合成条件での新順位、一般化できる規則の強さ、未知古典費用、独立論文としての十分性。
- **研究運営上の判断**：現研究を直ちに終了せず、保証と価格の限定確認を次に優先する。これは将来の成功保証ではなく、現在の証拠に対して情報価値が高い次作業の選択である。

**総括：D1/D2は、既存成果の発展研究を継続する根拠になる。ただし、境界相殺を通過したことだけで一般優位を認定せず、保証と合成価格に対する頑健性を次の検証対象とする。保存値分析をもう一巡するのではなく、目的を限定した前向き確認へ進む。科学実行のmandatory STOPは別途承認まで維持する。**

---

## 内部参考資料

- [R1] 最終統合報告。`4cf2827c14fec96613a60bb1427287b30766d5da`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2827c14fec96613a60bb1427287b30766d5da/docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md
- [R2] D2原結果・監査報告。`d1c59b28f4370cc7be65cff60d1ec877fd1491f5`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1c59b28f4370cc7be65cff60d1ec877fd1491f5/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
- [R3] D1 key/event機構・D2準備。最新統合commitから、準備時の状態を履歴として読んだ。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2827c14fec96613a60bb1427287b30766d5da/docs/tracks/algorithm_codesign/d2_key_mechanisms_and_boundary_preparation_20261013.md
- [R4] D1保存資源分析。`2dcd56737be35526f6436a52835f59d817adf30c`。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/docs/tracks/algorithm_codesign/d1_saved_resource_analysis_and_gpt_handoff_20261013.md
- [R5] 最終表示summary。exact authorityではなく表示用。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2827c14fec96613a60bb1427287b30766d5da/artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/display_summary_v1.json
- [R6] D2境界kernel。固定source。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28fc3ca17ec51bb7656a498e3eea9b72966eb3fa/src/trottertracks/algorithm_codesign/rte_development/boundary.py
- [R7] D2契約。準備metadataと、後続の別実行認可を区別した。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28fc3ca17ec51bb7656a498e3eea9b72966eb3fa/artifacts/track_b_d2_boundary_source/2026-10-13/v1/contract_v1.json
- [R8] D1保証・資源会計source。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/874ca12d347f8f35e5d8d64f65f24abc4a2df523/src/trottertracks/algorithm_codesign/rte_development/accounting.py
- [R9] 最終保存値分析の検証記録。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2827c14fec96613a60bb1427287b30766d5da/artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/validation_summary_v1.json
- [R10] branch metadata。取得時HEADが上記最新commitと一致。
  https://api.github.com/repos/HIROMU1015/Partially-Randomized-Trotter/branches/track-b-d2-final-saved-integration-analysis-20261013
- [R11] 前回D1科学レビュー原本。現在の添付原本も参照した。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2827c14fec96613a60bb1427287b30766d5da/docs/research/track_b_d1_results_and_resource_mechanisms_scientific_review_20261013.md

## 外部一次資料と確認の限界

- [P1] Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1。本文§II Theorem 1、§IIIのbias不変の条件を確認。固定ensembleの一般原理を参照し、D1/D2の有限confidence・全実装最適性の証明としない。
  https://arxiv.org/html/2603.13495v1
- [P2] Peetz, Smart, Narang, *Quantum Simulation via Stochastic Combination of Unitaries*, arXiv:2407.21095v2およびnpj Quantum Informationの刊行ページを確認。repositoryのmatched CTSがfamily全体の最適解だとはしない。全文の全定理を再証明していない。
  https://arxiv.org/html/2407.21095v2
  https://www.nature.com/articles/s41534-025-01168-w
- [P3] Ross, Selinger, *Optimal ancilla-free Clifford+T approximation of z-rotations*, arXiv:1403.2975、刊行abstract。最適性・近最適性の条件があることを確認。今回の固定sequenceの最小T数を認定しない。
  https://arxiv.org/abs/1403.2975
  https://doi.org/10.26421/QIC16.11-12-1
- [P4] Selingerの公式GridSynth documentation。確率的入力と近最適性の説明を確認。repositoryのPython移植と完全同一の実行結果だとは主張しない。
  https://www.mathstat.dal.ca/~selinger/newsynth/html/Quantum-Synthesis-GridSynth.html

新規性の網羅的priority調査は今回の作業ではない。近接する一般原理を再発見として扱わないための限定照合である。
