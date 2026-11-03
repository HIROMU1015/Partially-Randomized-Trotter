# Track B 横断的科学レビュー：Return aggregationの構成価値、資源比較、B3実計測の位置付け

- レビュー日：2026-10-19（日本時間）
- 開始承認：ユーザーの「レビューを開始して　丁寧に考えて」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 区分：重要科学レビュー（Level 1）。本レビューの科学的推奨と、ユーザーによる正式採択・科学実行承認は別である。
- 適用指示：`research_project_instructions_unified_2026-10-17.md`。会話に添付された正式版を確認した。
- 直接の最新対象：producer `7ae4bd0b951dae05380538cd5c22a3ead5328ab2`、consumer `bf601f8a9723a61b7a05b534019f8e036a362735`。
- 実施：固定資料・関連SOURCEの静的読解、過去の科学判断との照合、一次文献の関連箇所確認、紙上の数学的・実験設計上の検討、研究記録の作成。
- 未実施：科学SOURCEのimport・実行、sampler、行列計算、数値最適化、保存科学値からの新規数値再解析・再ランキング、回路build/compile/synthesis、repositoryのtests、科学的cold replay、GitHub変更、実authorization/grant/claimの発行・消費。
- 本文の既存数値は取得資料に記録された値の引用であり、新しい資源表の計算結果ではない。

## 0. 結論

**Return aggregationの構成・生成法の研究を継続する価値はある。しかし、「一般Full Returnの資源優位が確立した」「次は準備済み31,710,848 callsをそのまま実行すべき」とは判断しない。**

今回の推奨は以下である。

| 判断対象 | 本レビューの推奨 |
|---|---|
| G6の数学的構成 | 対象・仮定の範囲で支持する。全形式returnの局所取得、正値性、parent/child表現、有限bit接続を研究の核に残す |
| 独立新規性 | 構成上の寄与候補はあるが、世界初・全手順の非自明性を全面認定しない。既知部品との対応を明記する |
| 既存の量子資源結果 | G9/D系列/Parent allocationの限定positiveと、G10・P_U等の限定negativeを両方保持する |
| 改善P1＋fused | 次の実入力技術候補として第一候補に残す。人工benchmarkの改善を量子アルゴリズムの改善率にしない |
| 並列CPUを主指標にするか | 同じ並列実行条件に限定した運用費用としては採用候補。実装・同時実行環境から独立した方式固有のCPU優位としては扱わない |
| Cooperative STOP | 固定trial、予算、未完了結果非採用、未知費用保全を守る設計として支持。逐次R1と同じ意味論ではなく、正式変更は別採択 |
| B3の次の科学取得 | 実G7入力での限定的なcorrectness・全経路費用計測を先に行うことを推奨。実行は別承認 |
| 31,710,848 calls | 固定workload全体の実費・保存・長時間挙動を直接知る必要がある場合に実施する。CPU費用探索の普遍的な最低件数ではない |
| 残るpending | すべてを過去に遡ってPASSへ埋めない。本番で必要な保証と、歴史的に復元不能な証拠を分ける |

既存の正式R1、SOURCE、結果、grant、ledger、marker、mandatory STOPは変更しない。ここでの推奨は新しい実行承認ではない。

## 1. 今回のレビューで更新する判断

### 1.1 技術準備が完了したことと、科学的な次の一手を分ける

サーバーは新v3 Host/Admission、spawn worker、cached observer、coalesced IPC、fused canonicalを技術統合した。consumerは独立したadapterで正常12件と障害35件の公開人工結果を読み取り、保存証拠の範囲を検証した。これらは実G7のcorrectness・実費を取得したという意味ではない。[R15] [R16] [R17] [R20]

以前の会話では、実装が揃った後の全N実行を中心に段階を説明しすぎた。今回、科学的な問いを再評価すると、**実装完成→全N実行を既定にするのでなく、実入力でしか分からない不足を限定計測し、全Nを回す必要性を判断する**方が適切である。

この変更はCPU/RAMを節約するための恣意的制限ではない。計算件数を、明らかにしたい科学的・運用上の問いに対応させるためである。

### 1.2 全研究を無条件に閉じる判断でもない

G10の固定native比較で一般Fullが対照に劣ったことは、G6の構成恒等式を否定しない。Parent allocationの現在の数値探索を区切った正式記録も、今後一切のアルゴリズム改善を否定するものではない。[R4] [R6]

また、研究の価値を「全既知法への勝利」「一定%以上の改善」「独立原著として直ちに投稿できること」のいずれかに限定しない。一方、実装量・監査件数・文書量の増大だけを科学的寄与と呼ぶことも避ける。

## 2. 証拠の範囲と独立性

### 2.1 今回の直接読解と継承

| 範囲 | 今回の扱い |
|---|---|
| G6数理監査、G7 generator/budget、F1/F2、W2b/W2c、P_U反例 | 主要な数式・静的SOURCE・条件を直接読解し、論理関係を紙上で検討した |
| G9 | 固定結果handoffの対象、資源表、native・confidence・normalizerの区別を確認した |
| G10 | 固定過去レビューの主要結果・下界のscopeを継承した。約66.8 MBの全rawを今回再結合・再計算していない |
| D1–D3 | 既存統合レビューと記録された対象・機構・費用条件を確認した。全48行・全eventの新しい独立再集計ではない |
| Parent allocation S4 | 最終70 law Evidence Packetと正式closureを直接確認した。旧KKTと訂正後の区分を保持した |
| BF/BM/SP/G5/DF/Gaussian | 保存された横断レビューの限定的な整理を継承した。各系列の全SOURCE・全rawを今回再審査していない |
| M1-S | 10月17日の有限step pilot結果レビューの存在と主要判断を直接確認した。全後続系列の再レビューではない |
| 最新B3 | producer/consumerの結果、pending、実装設計、driverの関連SOURCEを直接確認した。481/540件のtestsをGPTが再実行したわけではない |
| 外部一次資料 | WBC Appendix C、SCU/CTSの関連本文・補足、SPS、Resource-Optimal IS、Aomoto–Katoの関連テキストを確認した |

本レビューは、異なる実装による新しい科学結果の再現ではなく、資料・SOURCE・既知の数学・独立した論理の照合である。

### 2.2 取得上の限界

長いGitHub資料の一部は関連行範囲を指定して読んだ。全repositoryを全行精読したとは主張しない。サーバー上のworktreeや他ユーザーのprocessを直接観測していない。公開bytes照合・保存監査・testsの完了は、その担当が保存した証拠として区別した。

WBCはarXiv PDFを取得し、Appendix Cの式(C2)–(C5)とAlgorithm 2をPDF画面でも確認した。Aomoto–Katoは関連解析テキストとrepositoryの対応を確認したが、当該PDFの画面取得に失敗し、古い組版の数式全体を独立に視覚照合したとはしない。外部文献の完全なpriority調査・不存在証明はしていない。

## 3. 科学taskを混ぜない

一般の構成対象は

\[
M_m=P_m(-i\sigma xR),\qquad R=\sum_{i=1}^{L}p_iQ_i,
\qquad Q_i^\dagger=Q_i,\quad Q_i^2=I,
\]

である。正の明示有理数p、和1、奇数m、非identityでは0<x≤1を前提とする。[R1]

これはまず有限多項式の**coherent first operator moment**である。channel平均や、有限精度の実装平均、全時間発展、QPEの最終誤差とは同じではない。

| 系列 | 比較している対象 |
|---|---|
| G6/G7/G9/G10 | 主として同次数の有限演算子mean。G9/G10では明示synthetic provider上のnative費用と十分予算 |
| D1–D3 | 同じ物理target \(e^{-i\tau R}\) に、有限多項式の反復・Taylor/bias/confidenceを戻した限定比較 |
| Parent allocation | 同じparent寄与の係数配分・回転辞書・proposalを変えた固定比較 |
| M1-S | H4のshared ±による有限step複素信号、visibilityとphase bias。B3とは異なるtask |
| 今回のB3 | 固定G7 samplerの古典生成・監査・角度enclosure・保存・管理費用の取得準備 |

D系列のP3+tailは同じP5を実現する表現であり、Taylor次数を3へ下げて精度を緩めた対照ではない。[R7]

B3 driverの固定値はp=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1、4方式×A/B×X/Y。root/proposal精度はG7参照、A6は数学的angle enclosureのτ=10^-6 rad、Kφ=32、J=32である。物理RZへの代表値適用ではない。[R18] [R19] [R20]

また、D系列の価格条件名P1と、今回の並列scheduler P1は別概念である。過去の再配分系列R1と、B3の採択済み実行条件R1も文脈を明示して区別する。

## 4. G6の構成は何を実際に達成しているか

### 4.1 全形式returnの意味

raw wordを、隣接する同じlabelの消去関係だけでreduced word uへ写す。長さl=|u|に対し、n-stepのraw語がuへ帰着する確率をP_n(u)とすれば、既存構成は

\[
a_u=\sum_{\substack{n=l,l+2,\ldots\le m}}(-1)^{(n-l)/2}
\frac{x^n}{n!}P_n(u),
\qquad
M_m=\sum_u(-i\sigma)^{|u|}a_uQ(u)
\]

とまとめる。[R1]

ここで集約するのはQ_i²=Iに由来する形式returnである。追加のPauli交換・反交換・特殊な演算子関係まで全て使った最小表現ではない。一般の奇数語をinvolutionと仮定してはいけない。

### 4.2 正値性の論証

既存監査の中心は、最初の隣接iiを削除する対応と逆向きの挿入の上界である。\(\chi=\sum_i p_i^2\)とすると

\[
P_{n+2}(u)\le(n+1)\chi P_n(u).
\]

挿入は同じraw語を重複して数え得るため等式ではなく上界である。Taylor係数を含む隣接項の比は

\[
\frac{x^{n+2}P_{n+2}(u)/(n+2)!}{x^nP_n(u)/n!}
\le\frac{\chi x^2}{n+2}\le\frac12.
\]

従って交代和の下側は最初の2項で押さえられ、\(t_l=x^l/l!\)、\(p(u)=\prod_{j\in u}p_j\)として

\[
t_lp(u)\left(1-\frac{\chi x^2}{l+2}\right)
\le a_u\le t_lp(u).
\]

0<x≤1の有効域で正値性が得られる。この論理は有限mの恒等式を正のsampling構成へ接続する要点である。今回の紙上確認では、この中心論証を否定する矛盾は見つからなかった。ただし、全補題の形式証明検査や全実装の再実行を意味しない。[R1]

### 4.3 Green再帰の役割

形式的first-passage系列F_iとGについて

\[
F_i(z)=\frac{zp_i}{1-z\sum_{j\ne i}p_jF_j(z)},\qquad
G(z)=\frac1{1-z\sum_i p_iF_i(z)},
\]

指定uへの係数はGとF_iの積の切断から取得できる。これは全word宇宙を保持せずに局所係数を取得する仕組みであり、既知の自由積Green関数の特殊化に対応する。新しいGreen関数定理と呼ぶべきではない。[R1] [R14] [P5]

重要なのは、形式的冪級数の切断と、無限walkの解析的収束を混同しないこと、scalarの積が可換でもQ(u)の作用順序を変えないことである。

### 4.4 Parent/child pairing

偶数長parent uに対し、合法child iuの質量和をs_u、\(d_u=\sqrt{a_u^2+s_u^2}\)、\(\tan\phi_u=s_u/a_u\)とする。すると

\[
\sum_i d_u\frac{a_{iu}}{s_u}
(-i\sigma)^{|u|}e^{-i\sigma\phi_uQ_i}Q(u)
=(-i\sigma)^{|u|}\left(a_uI-i\sigma\sum_i a_{iu}Q_i\right)Q(u).
\]

Euler型の恒等式を使うのは単一のQ_iであり、一般のword全体を回転軸と仮定していない。この点はgeneric involution accessに対して本質的である。[R1]

### 4.5 Normalizerを全列挙しない生成

既存ordinary envelopeをB_o、return集合の質量をB_RAとすると、局所上界によりB_RA≤B_oとなる。実装はordinary型の次数・IID label proposalを使い、最初からreducedでないraw語を量子処理前にzeroにする。**raw語をreduceした先へ再配分する別samplerではない。** 合法parentでは局所acceptance、child選択を行う。[R1] [R2]

理想・exact算術の説明として、acceptanceをZ=B_RA/B_o、受理後Hadamardの±1結果をYとすれば、1 trialの推定量は受理時B_oY、非受理時0である。したがって

\[
\mathbb E[X]=\operatorname{Re/Im}\langle M_m\rangle,
\qquad
\mathbb E[X^2]=B_oB_{RA},
\qquad
\mathbb E[K_{\rm quantum}]=NZ.
\]

全trialを分母にする点が重要である。accepted-onlyの平均へ変更したり、B_RAを未知のまま既知としてweightに使ったりできない。古典rejectの処理費用も0ではない。[R1] [R2]

### 4.6 支持する研究上の核

私が支持する構成上の寄与候補は、**同じ有限Taylor演算子について、形式returnを集約した係数を局所取得し、全辞書・全normalizerの明示構築なしに、位相・有限bit・全trial推定へつながる手順を与えること**である。

これは「単に既知のrejection samplingがある」「ordinaryも非列挙である」という理由だけでは消えない。一方、「異なる分布を実装した」だけでも独立科学貢献は確定しない。何を保持し、どの情報の取得を避け、どの精度保証を付けたかが評価対象である。

## 5. 有限bit保証と資源保証の境界

### 5.1 理想Mと実digital平均は別

実dyadic proposal πと近似係数\(\widetilde c\)に対し、\(\widetilde w=\widetilde c/\pi\)で重みを補正しても、一般には得られる平均は\(\widetilde M\)である。元のMとの差は係数biasとして残る。確率丸めの補正が、係数近似誤差まで消すわけではない。[R2] [R10] [R12]

F1のP0–P5は、与えられたroot/probability bit精度が全到達lawのguardを満たすための十分条件である。最小精度や任意入力の自動選択法の最適性を与えるものではない。[R12]

A6の数学的angle半径が10^-6であることと、物理primitive全体のstrict誤差が10^-6以下であることは同じではない。将来C1へ接続するなら、角度誤差・合成・provider等を共通biasへ戻す必要がある。現在B3が数学的enclosureだけを扱う限り、この区別を保てばよい。[R10] [R19]

### 5.2 Bit complexityは実CPU秒の定理ではない

F2は係数のbit長、根、GCDを含む有理演算、selector、law再構築、保存を分けている。例えばFullのparent/child局所queryに加え、現SOURCEのchild_lawの繰り返しsumにはL²型の処理が残る。単純なO(Lm²)等の構成部分だけを全CPU費用と呼べない。[R13]

また、mに多項式という表現と、binaryで符号化されたmのbit長に多項式という表現は異なる。比較モデルの入力・precision変数を明示する必要がある。

### 5.3 W2bの位置付け

同じG7 confidence・bias・root/proposal条件で、Fullは

\[
M_{2,F}^+=\kappa o u,\quad W_F^+=\lambda o,
\]

canonical対照は\(M_{2,a}^+=\kappa b_a^2\)、\(W_a^+=\lambda b_a\)を用いる。oはordinary envelope上端、uはU上端であり、未知B_RA²へ置き換えない。[R10]

\[
N_a=\left\lceil\ell^+\left(\frac{2M_{2,a}^+}{s^2}+
\frac{4W_a^+}{3s}\right)\right\rceil
\]

は既定policyの十分予算である。N_F≤N_Oという条件付き関係は有用だが、Partial/Closed P3との全域順位や、実CPU/native総費用まで自動的に閉じるものではない。

### 5.4 P_U反例の意味

固定反例L=2、p=(1/2,1/2)、x=1、m=7について、保存された初等不等式の連鎖は

\[
U_7-B_{5\oplus o,7}^{\rm form}>13/1024>0
\]

を与える。[R9]

この反証は、Uを使ってその強い対照に普遍的に勝つという特定の命題を否定する。Uは上界なので、この事実を真のB_RAが必ず対照より大きいという結論や、return集約の恒等式の否定へ拡張しない。P_U,3の真偽も、この反例だけでは判定しない。

## 6. 先行研究との関係と新規性の判断

### 6.1 一次資料から維持すべき点

WBCのAppendix C/Algorithm 2は、次数とIID Pauli labelを引いてunitary積を生成する。従ってordinaryを全辞書列挙必須の弱い対照へ置き換えるのは不適切である。[P1]

SCU/CTSには係数集約と回転への変換に加え、補足で分割した高次項のlayer samplingがある。CTS全体を全Pauli表の展開必須と呼べない。ただしlayer化は全展開時の相殺を一般に失い得るため、G6の指定された集約lawと同じ保証・費用かは別問題である。[P2]

SPSはprocess/libraryの線形分解・samplingを扱う。controlled first momentへ比較する際は位相と情報アクセスを戻す必要がある。channelが同じというだけで、制御付き干渉信号も同じとは限らない。[P3]

Resource-Optimal ISの固定費用モデルでは、cost×second momentを最小化する既知のproposal構造がある。その最適性は、辞書変更・range・bias・ceilを含む全てのB3/G7予算設計の最適性ではない。また、量子回路実行後のerror flagと、G6の量子前の古典rejectは支払う費用が異なる。[P4]

### 6.2 本レビューの新規性評価

Green特殊化、Euler pairing、rejection、importance weighting、gate費用×sample数という評価原理は、それぞれ既知として記述すべきである。[R14] [P1] [P2] [P3] [P4] [P5]

それでも、一般involutionの有限target、正値域、局所係数取得、normalizer不要の生成、finite-bit bias/rangeまでを接続した具体手順には、方法研究として提示する内容が残る。

今回確認した一次資料から、これと同じ全接続を、同じaccess・出力・precision・費用でそのまま実行する既存の単一algorithmを特定したとは言えない。しかし、その不特定をpriorityの証明にしない。

**判断は「構成寄与候補として支持、独立新規性は限定付きで未確定」である。** 必要なのは、既知部品を隠すことでも、さらに無制限に論文棚卸しを増やすことでもなく、主張を「何を具体的に構成したか」に正確に合わせることである。

## 7. 過去系列の科学的到達点

### 7.1 G7/G9/G10

G7のFullの限定的な期待T利益は、登録したOrdinary/Partial/P3対照とqueryモデルの範囲で残る。ただし強いClosed P5・CTSを戻すと、研究上の意味が変わる。[R2] [R3] [R4]

G9のdirect比較では、Closed P5がT切片と共通準備呼出し数Kで登録6方式中最小であり、literal matched CTSよりT切片が約29.77%小さい。一方、native CXはCTSが小さい。FullとClosed P5は同じ理想ensembleでも、digital law・normalizer取得・予算が同一ではない。[R3]

G10の同一入力m=7では、一般FullはClosed P5＋tailに対してT切片が約2.91%、Kが約0.615%大きいという記録である。固定辞書・合成・十分予算policyの範囲でのsampling-only救済限界も、普遍的な量子lower boundとは区別して保持する。[R4]

**私の判断：低次数の有効な構成と一般生成器の役割を分ける。G9のpositiveをG10以降の一般Full勝利へ延長せず、G10のnegativeでG6構成全体を消さない。**

### 7.2 D1–D3

D系列は、同じ物理targetにTaylor・反復・native誤差・confidenceを戻し、保証と合成価格によって表現の順位が変わることを具体化した。新価格の条件ではP3+tailがP5より小さいTを持つ記録がある一方、P5がprimary Partialより有利な範囲も残る。[R7]

これは「正規化が小さければ実装資源も必ず小さい」という単純化を否定する具体的な証拠であり、どの表現も一般に最適という主張ではない。D系列で保証を変えたことと、物理回路の正確さ自体が改善したことも区別する。

同じ既知入力で価格・seed・precisionを増やし、望む勝者を見つけることは、追加の科学情報が具体化しない限り優先しない。

### 7.3 Parent allocation S4

parent係数をz_iへ配分して\(\sum_i z_i=a\)を満たすと、回転角と係数を変えて同じmeanを実現できる。common配分は係数L1を最小化するが、回路単価を含む目的まで最小化するとは限らない。[R5] [R6]

訂正後のKKTは\(\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2}\)に対応する。旧raw-C proxyの20 lawを訂正後の証拠として使わない。

ordinaryで約14%、P5で約19%のT改善は、保存されたbest-common対best-noncommonの限定比較である。異なるproposalを含むため、allocation単独の効果ではない。N・CX・準備/readout費用のtrade-offも残る。[R6]

**私の判断：具体的な表現変更が条件付きで資源利益を生む証拠として保持する。既に正式に区切った現在の数値探索は自動再開しない。**

### 7.4 BF/BM/SP/G5その他

BFは固定family内の追加decision valueが示されなかった限定結果、BMは同情報の既知再帰との同値性、SPは短wrapperでの利益と長wrapperでの反転、G5は固定classでの強い対照による閉鎖、と区別する。DF wholeやGaussianの結果も、それぞれのtarget・情報・未取得費用の範囲に留める。[R8]

異なるtaskの試行をまとめて「多数のアルゴリズムが全て失敗した」とは評価しない。一方、既に閉じた固定classを名前だけ変えて再探索することも推奨しない。

### 7.5 M1-Sについての訂正

以前の会話で、M1-S/Gate0の結果解釈が十分レビューされていないように示唆した点は不正確だった。10月17日の有限step pilot結果レビューが実在する。[R21]

そのレビューはH4のvisibility改善とphase bias増加の併存を認め、RPE総資源優位とは区別している。今回、その全後続結果を再解析していない。未レビューの有望候補として自動的に再起動したり、B3へ混ぜたりしない。

## 8. 最新B3技術統合で何が増えたか

### 8.1 正当に評価する技術成果

producerは旧保護356＋統合125＝481人工tests、consumerは旧437＋新103＝540人工testsという別の集合を持つ。consumerは正常12件をVERIFIED_COMPLETE、障害35件をTECHNICAL_FAILUREとして独立に分類した。[R15] [R16]

これは、失敗が成功結果へ混ざる危険、canonical/hash/prefixの取り違え、費用二重計上等を減らす重要な研究基盤である。ただし、481＋540を1011個の独立な物理実験と数えず、旧suiteの重複も無視しない。

### 8.2 高速化の帰属

改善P1の利益にはobserver管理cacheとIPC同梱の改善が含まれる。fusedの追加効果はP→PFの人工比較でCPU5.6/4.0/1.8%、wall4.0/3.9/0.7%（light/mixed/heavy）と記録されている。[R15]

**これは古典的な測定・実行実装の改善であり、Return aggregationそのものの量子資源改善ではない。** また、observer改善は逐次実装にも適用できる。古い重いobserverを用いたS0とだけ比べて、CPU低下の全てを並列化へ帰属させてはいけない。

最終統合版のS/SFは同じscheduler・cached observer・coalesced IPCを用いてpairを逐次実行する対照である。したがってP/PFとの並列効果の比較に役立つが、旧R1のS0や、同じobserverを入れた最小構成の逐次実行と同じではない。[R20]

### 8.3 実入力で未確認のこと

人工endpointで正常経路と障害管理を検査しても、実G7/F3のoperand、local query、guard、実angle処理の費用分布までは得られない。現在のparallel driverは実ModuleTypeを拒否し、実科学入口が明示的に閉じられている。[R18] [R19] [R20]

よって「技術的に接続できた」と「実G7で同じ科学law・費用条件の下で完走した」は別である。実入力での一段の確認を省略することは推奨しない。

## 9. B3を実行する科学的な意味と、全Nの必要性

### 9.1 B3で直接答えられる問い

B3の有用な問いは、例えば次である。

> 同じ固定G7入力・同じ精度・同等の監査/保存条件で、Fullが量子前reject・局所係数・angle requeryのために支払う古典費用はどれほどか。その費用はどの処理に支配されるか。

同じ単位・実装層で、\(C_{cl,a}=S_a+N_a\mu_a+\Omega_a\)と書けば、Fullが対照a以下となる条件は

\[
(S_F-S_a)+(\Omega_F-\Omega_a)+N_F\mu_F-N_a\mu_a\le0.
\]

この条件自体は既にW2cで整理されている。B3の仕事は、この式の未取得の具体的費用を測定することであり、同じ式を再導出することではない。[R11]

μはraw zero・coin reject・acceptedを含む全trial費用である。受理例だけの平均や、setup/hashを無料にした比較では条件を評価できない。

### 9.2 B3だけでは答えられないこと

現在の4方式のB3は、一般の新規性、全入力のclassical scalability、Closed P5/CTSを含む全対照への総費用優位、物理Q provider、state/readout/backend、分子PR/QPE全体を一括で証明しない。[R10] [R11] [R18]

Closed P5/CTSの同等な全trial古典費用は現在の4方式だけでは埋まらない。しかし、B3を限定4方式の実装費用診断として報告するなら、全ての強い対照の新規実装を今回の必須gateにする必要もない。主張を拡大するときに必要な比較を追加する。

FullのCPUが増え、量子費用が減る場合、それはtrade-offである。CPU秒とT数を変換する価格モデルを採択していない以上、どちらか一方だけから全体を勝敗判定しない。

### 9.3 31,710,848 callsの由来

この件数はcorrectness256＋full-N baseline15,855,296＋instrumented15,855,296である。Nは保存されたG7の信号推定に対する十分予算に由来する。[R10] [R18] [R20]

**信号の誤差・confidenceを満たす十分Nと、CPU費用を記述・推定するための観測数は、目的も確率変数も異なる。**

現在のseed関数もFROZEN_G7_SHA256_COUNTERを用い、`IID_claim=False`を要求している。A/B・pass間は同じprefixで、X/Y・armを識別する。これは比較の再現性に役立つ固定workloadであって、全部を独立量子shotとして観測したことにはならない。[R19]

したがって、全Nの古典sampler実行を終えても、それだけで量子信号confidenceの実験的検証ができたとは言えない。また、全Nを二pass実行しない限り古典費用について何も言えないという理由にもならない。

### 9.4 全Nを実行する価値がある場合

固定されたそのworkloadについて、実際の総CPU/wall、全保存量、長時間のmemory挙動、I/O競合、全prefixでの実装安定性を直接測る目的なら、全N実行には明確な価値がある。

しかし、現在最も欠けているのは実G7/F3の短い実行すら含む費用・correctness証拠である。最初から全Nでしか分からない問題へ進む必要性は、まだ示されていない。

**推奨：実入力の限定pilotを先に行い、全Nが必要な問いを残したときに一束の実行として進む。** これは旧R1のNやgrantを黙って縮小する指示ではなく、新しい限定scopeの提案である。

## 10. CPU費用のestimandへの回答

### 10.1 並列CPUは有効な測定値だが、条件付き

4方式を同時に走らせる場合、pairのCPU費用には自身の処理だけでなく、共有cache・memory帯域・I/O・他pairの完了時点・スケジューラの状態が影響し得る。CPU timeは単なる待ち時間とは違うが、計算量の機械独立な値でもない。

同じ8 workerというだけでは、方式ごとに同じ競合条件を受けたことにはならない。例えば長い方式だけが最後に残れば、途中と終盤で競合状況が変わる。

従って、**「この固定並列portfolioを実行した実費」として主報告することは可能だが、「この方式の本来的なCPU費用は他方式より低い」と一般化しない。**

### 10.2 推奨する報告構造

運用上は、採用候補の並列実行を優先してよい。科学的な機構説明では、Layer A/B、各経路、setup/online/output、parent/worker、baseline/instrumentedを分ける。限定pilotでは、同じobserver・codec・入力を持つ比較可能な逐次条件も用意し、競合と管理費用を切り分ける。

全Nを全方式で逐次・並列の二重に走らせることを普遍的な前提にはしない。必要な範囲の対照で不確かさが大きければ、それをそのまま記述し、実際の費用差について言える範囲を限定する。

### 10.3 計測器自身の効果

coarseとfineは別partitionであり足さない。pairはworker内、IPC/起動/finalizationはouterに内包されるので再加算しない。baselineとinstrumentedの差を純observer費用として差し引くことも、環境差・計測器の影響がある限り自動的には正当化されない。[R16] [R18]

比較側にも同じコード最適化機会を与える。Fullだけの高速化と、古い対照SOURCEを固定した費用を比べて、一般的なアルゴリズム改善を作らない。

## 11. Cooperative STOPの科学的な扱い

### 11.1 支持する設計

あらかじめ決めたglobal trial identity、pair内の乱数順・weight・結果の意味が保たれ、失敗後に新しい許可を出さず、開始済みin-flightを記録し、失敗bundleから全Nの推定結果を返さないなら、cooperative STOPは実装候補として合理的である。[R15] [R19] [R20]

一つのworkerの物理的例外発生時刻に全processの既に始まった処理を同時停止させる保証を、必要な根拠なく要求しない。親がSTOPを処理した論理境界と、他workerが既に得たENTER許可を区別する。

### 11.2 推定の正しさと完走確率

以下は本レビューの条件付き論理整理であり、現SOURCEや実量子実験に対する新certificateではない。

固定Nの仮想的な完全標本に対する推定誤り事象をE、abortせず返答する事象をAとする。成功時に必ず同じ完全標本の推定量を返すなら

\[
\Pr(E\cap A)\le\Pr(E)\le\alpha.
\]

ただし、これだけから\(\Pr(E\mid A)\le\alpha\)は従わない。入力や標本に依存したabortは成功条件付きの分布を変え得る。また「誤答しない」ことと「要求された確率で完走して答えを返す」ことは別である。

従って将来の科学実行では、完成した固定Nの推定保証、abortの扱い、未完了費用、必要なら完走確率を別に説明する。失敗を都合よく捨て、同じ予算を再利用して成功runだけを採用する運用は認めない。

現在のB3は古典費用診断であり、上の式を実測済み量子confidenceへ読み替えない。

### 11.3 既存R1との関係

並列STOPは逐次R1と同じ失敗prefixではない。設計を支持することと、旧契約を暗黙に変更することは別である。方式・観測境界・未知creditの扱いをまとめた一つの実行方式として、ユーザーの採択後に新SOURCEへ固定する。6つの途中承認gateへ分割する必要はない。

## 12. Pendingをどう扱うか

| Pending | 科学・技術上の意味 | 本レビューの推奨 |
|---|---|---|
| P07/P11全47件 | 過去のprivate approval/ledger/historyを独立に再確認できない | 今後使用する実装・保存境界に対する新しい人工race/spent/reuse拒否証拠を一束で確認。秘密を公開しない |
| P08全47件 | 全trialのgate履歴がなく過去のSTOP後ENTER不存在を再構成できない | 既存SOURCEと限定人工ordered traceの検証範囲を明示。本番全trialの巨大journalを無条件に要求しない |
| 障害30件contract不足 | 過去の完全binding再計算に不足 | 原本が保存されていれば追加取得。失われていれば後から同一だったと捏造しない |
| P06障害11件 | worker END等の記録不足 | 想定故障での保存限界を残し、必要な将来の前向き検査を絞る |
| P10障害35件 | 正常completion receiptがない | 失敗runとして想定どおり。成功receiptを作ってPASSへ変えない |
| P04/P09親crash | 保存量や総費用が不明 | null/unknownを保ち、そのscopeの費用証明をしない |
| 過去Python実体等 | 当時の環境の完全なbyte再認証が不能 | 保存identityと今回の実環境検査を分ける。別versionを代用品にしない |

これらの不足は、公開47件からproducer欠陥が確定したという意味ではない。同時に、全履歴・atomic性がofflineだけで証明されたという意味でもない。[R16] [R17]

**全pendingを消すこと自体を新しい研究目的にしない。** どの主張・どの将来実行保証が必要かを決め、その範囲で証拠を補う。新たに得た人工履歴を過去47件の実履歴と呼ばない。

## 13. 次の作業：一つの限定実入力計画へ進める

### 13.1 研究上の問い

次の一束で答える問いは、以下に限定するのが適切である。

> 固定G7の4方式を同じ実装層・精度・保存意味論で動かしたとき、生成・監査・angle・保存のどこが古典費用を支配するか。改善P1/PFを用いた運用が実入力でも成立するか。全Nの直接測定を追加しなければ解消できない不確かさは何か。

目的はFullの勝利を出すことではなく、古典費用の空白を埋め、次の研究判断を変える情報を得ることである。

### 13.2 技術準備はまとめてCodexへ

ユーザーがこの方向を採択した場合、サーバーCodexの次の技術bundleを推奨する。

- 固定consumerの導入と公開人工rawでのsmoke。
- 将来の一つの実行namespaceについて、SOURCE/runtime/schema、atomic claim・STOPの必要な人工証拠を確認。
- 実G7入力用の限定pilot契約と実行SOURCEの準備。実科学入口の閉鎖を、承認なしのbool編集で解除しない。
- 比較する実装・観測範囲・保存・失敗処理・科学call範囲を一つのcandidateへ固定する。

worker数・IPC・内部データ構造・人工回帰の細部はCodexに委ねる。CPU/RAM/worker数に恣意的な低い上限を設定せず、実際の空き資源と非干渉を確認する。共有OS・venv・cache・他ユーザーjob等は変更しない。

### 13.3 科学実行承認後の限定pilot

実科学のcall数や候補集合は、pilotの目的・処理負荷・必要な経路coverageに基づき固定し、ユーザーの別途明示承認を得る。本レビューでは具体的な新nを勝手に割り当てない。

pilotでは少なくとも、正しいsource/input/precision、raw zero・coin reject・accepted、canonical側の該当mode、Layer A/Bの費用、observer/codec/scheduleの対応を確認する。未観測の経路を無料・成功済みとみなさない。ごく短い固定traceの平均を母期待値のcertificateに格上げしない。

これは一般入力の検証や、fresh IID量子shotの信頼区間とは別である。既存のdeterministic workloadを使うなら記述的実測として扱う。確率的な費用推定が必要なら、その母集団・標本・不確かさの扱いを別途明示する。

### 13.4 全Nへ進む判断

全Nを行う理由として、実費総量、全出力の保存、長時間のmemory/I/O挙動、pilotからの外挿が不十分な点が残れば、その問いを明示して全Nを実行する価値がある。

逆に、論点が「一trialの支配費用」「条件式に対する大まかな費用差」「実入力の入口が動くか」で既に判別できるなら、全Nを回すこと自体を必須にしない。

少数pilotからの予測は条件付き外挿であり、固定31,710,848 callsの実費を観測した結果と同一視しない。新しいscopeから旧one-shotの自動再開へ移行しない。

### 13.5 次にレビューへ戻す条件

通常の導入・artifact整合・人工バグ修正・schema接続では、今回の科学レビューを繰り返さなくてよい。

再び重要科学判断が必要になるのは、例えば、実入力で理論/実装意味論が食い違う、費用の結果によって主要claim・比較対象を変える、保証や評価taskを変更する、別のalgorithmへ研究資源を移す場合である。

## 14. 原稿化・別方向の位置付け

既存構成・限定positive・限界を自己完結的に記述する価値はある。しかし、今回ユーザーが求めているアルゴリズム改善を、出版準備へ無断で置き換えない。

現時点で一般Fullの優位性を前提とした新たな広域benchmark、GMP/冪漸化式/価格seed探索の追加、Closed P5が勝つ入力探しを優先する根拠は強くない。具体的な実入力費用を知る前に、人工管理部分だけをさらに細かく最適化することも優先しない。

一方、B3のCPU結果がFullに不利でも、構成・精度保証の科学的内容は保持する。量子資源とのtrade-offとして価値が残る可能性と、同じ条件で安い低次数実装へ役割を分ける可能性を評価する。全面終了も自動継続も既定にしない。

## 15. 確かさと未確定事項

| 論点 | 確かさ／未確定事項 |
|---|---|
| G6の基本恒等式・短時間正値性・局所生成の論理 | 主要導出を紙上で支持。全実装/全補題の機械検証ではない |
| 既存G9/G10/S4の限定資源結果 | 固定報告・Evidence Packetの範囲で支持。全rawを今回再集計したわけではない |
| 最新人工技術実装と独立verifier | 公開scopeの証拠として受け入れる。実G7性能・真のatomic実環境証明は別 |
| 一般新規性・最適性 | 全面的には未確定。既知部品を明示した構成寄与候補 |
| B3実G7の所要時間・CPU順位 | 未取得。人工倍率や旧短traceから確定しない |
| 全Nの直近必要性 | 一律の必要性は認めない。実入力pilot後の具体目的で判断 |
| 本番実行の正式採択 | 未実施。ユーザー判断と別途実行承認が必要 |

**最終結論：構成・生成法の研究は保持して進める。次は実入力の限定計測へ向けた一束の準備を優先し、全Nは目的に応じて採択する。並列化の技術的成功を科学的な普遍優位へ読み替えず、残るpendingを過剰な承認gateへ変えない。**

---

## 参考資料・固定locator

以下の参照は本レビューの根拠の入口である。文書全体を取得した場合と関連範囲を読んだ場合は§2に区別した。GitHubの固定refにある過去レビューの記述は、今回の新実行記録ではない。

### Repository資料

[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[R2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[R3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[R4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf601f8a9723a61b7a05b534019f8e036a362735/docs/research/track_b_G10_v3_scientific_review_20261010.md
[R5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md
[R6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf601f8a9723a61b7a05b534019f8e036a362735/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[R7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf601f8a9723a61b7a05b534019f8e036a362735/docs/research/track_b_d1_d3_post_literature_scientific_review_20261014.md
[R8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf601f8a9723a61b7a05b534019f8e036a362735/docs/research/track_b_overall_direction_and_research_pause_review_20261013.md
[R9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/850ac89e1a6ea955def805e5462b8694ec0840b4/docs/research/track_b_g6_pu_analytic_proof_20261018/proof_or_counterexample_certificate.md
[R10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/certified_budget_comparison.md
[R11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/break_even_conditions.md
[R12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md
[R13]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[R14]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf601f8a9723a61b7a05b534019f8e036a362735/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/generation_pipeline_alignment.md
[R15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ae4bd0b951dae05380538cd5c22a3ead5328ab2/docs/research/track_b_g6_w2c_b3_parallel_fused_r1_technical_integration_20261019/handoff.md
[R16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf601f8a9723a61b7a05b534019f8e036a362735/docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/handoff.md
[R17]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf601f8a9723a61b7a05b534019f8e036a362735/docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/pending_evidence_and_server_request.md
[R18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ae4bd0b951dae05380538cd5c22a3ead5328ab2/src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_driver.py
[R19]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ae4bd0b951dae05380538cd5c22a3ead5328ab2/src/trottertracks/algorithm_codesign/g6_w2c_b3_parallel_driver.py#L350-L580
[R20]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ae4bd0b951dae05380538cd5c22a3ead5328ab2/docs/research/track_b_g6_w2c_b3_parallel_fused_r1_technical_integration_20261019/integration_design.md
[R21]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b/docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md

| ID | 資料・主な用途 |
|---|---|
| [R1] | G6数学監査：恒等式、正値性、Green、parent/child、envelopeとzero-fill |
| [R2] | G7 generator：Fullのsample/eventと十分budgetの実SOURCE |
| [R3] | G9結果：強い対照との固定native比較 |
| [R4] | G10過去科学レビュー：固定次数比較・lower-boundのscopeを継承 |
| [R5] | Parent allocation S4最終70 law Evidence Packet |
| [R6] | S4限定成果とD＋E正式採択のclosure |
| [R7] | D1–D3統合レビュー：同一物理task、保証・価格の機構 |
| [R8] | BF/BM/SP/G5/DF/Gaussianの限定履歴を継承 |
| [R9] | P_Uの紙上反例certificate |
| [R10] | W2b：同一条件の認証十分全trial数 |
| [R11] | W2c：古典・量子費用のbreak-even条件 |
| [R12] | F1：有限bitの一様preflight十分条件 |
| [R13] | F2：標準算術モデル・全処理bit費用 |
| [R14] | 先行生成pipeline対応表。完全同値／非同値の証明ではない |
| [R15] | 最新producerの統合・人工性能・障害・未採択境界 |
| [R16] | 最新consumerの独立offline受入範囲 |
| [R17] | 過去contract・ledger・操作履歴等の未取得範囲 |
| [R18], [R19] | 最新driverのpolicy/seed/Layer A/B/監査・angle接続境界 |
| [R20] | v3統合設計、S/SF対照、spawn、閉鎖された実科学入口 |
| [R21] | M1-S有限step結果の既存科学レビューがあることの確認 |

### 外部一次資料

[P1]: https://arxiv.org/pdf/2110.12071
[P2]: https://arxiv.org/html/2407.21095v2
[P3]: https://arxiv.org/html/2402.15550v2
[P4]: https://arxiv.org/html/2603.13495v1
[P5]: https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf

| ID | 資料と今回直接確認した範囲 |
|---|---|
| [P1] | Wan–Berta–Campbell, Randomized Quantum Algorithm for Statistical Phase Estimation. Appendix C、Algorithm 2、有限cutoffの位置付け。PDF第15・16ページ（0始まり14・15）の画面確認も実施 |
| [P2] | Peetz–Smart–Narang, Quantum Simulation via Stochastic Combination of Unitaries, arXiv:2407.21095v2. CTSの構成、補足Note 3のlayer sampling、first-moment/processの文脈 |
| [P3] | Koczor, Sparse Probabilistic Synthesis of Quantum Operations, arXiv:2402.15550v2. Statement 1/2、process/libraryの線形分解とsampling |
| [P4] | Cugini–Atif–Subaşı, Resource-Optimal Importance Sampling for Randomized Quantum Algorithms, arXiv:2603.13495v1. Theorem 1/2、cost×second-moment、error detectionの費用境界 |
| [P5] | Aomoto–Kato, Green functions and spectra on free products of cyclic groups. 関連解析テキストとrepositoryの特殊化対応。PDF画面取得失敗により全数式の視覚再認証は未実施 |

外部本文の取得失敗を、論文が存在しないことや先行法の不存在の証拠にしない。優先性を主張するには、本レビューで限定した比較範囲を越える確認が必要になり得る。

## 記録の扱い

このMarkdownは本レビューの研究記録であり、正式科学SOURCE・契約・authorizationではない。GitHubへ自動commit/pushしていない。ユーザーが科学的推奨を採択するか、Codexへ技術準備を指示するか、実科学計算を別途承認するかは、それぞれ独立した次の判断である。
