# SOS-PF K0.5：追加文献対応表

2026-10-15着手、2026-10-16完了（JST）。一次原文に限定した追跡表。K0の9文献台帳を置き換えず、追加確認した箇所を示す。網羅的な優先権調査・全証明の独立再現ではない。

PDFは原byteを取得してhashを記録した。`pdftotext -layout` は読取り補助であり、版差分の証明ではない。cacheはprovenanceに記した `/tmp` にあり、このディレクトリへ論文を重複保存していない。URL・版・SHAにより再取得照合できる。ACM未取得とNatureのchallenge応答を正常取得本文へ数えない。

## 1. Kang–Su：重点原文

**Christopher Kang, Yuan Su, Quantum Matrix Arithmetics with Hamiltonian Evolution**

- [arXiv:2510.06316v3 HTML](https://arxiv.org/html/2510.06316v3)
- [v3 PDF](https://arxiv.org/pdf/2510.06316v3)、表紙2026-09-01、94 pages。
- [ACM DOI 10.1145/3816436](https://doi.org/10.1145/3816436)。ACM Trans. Quantum Comput. 7(4), 1–86 (2026)、online 2026-08-13。
- PDF SHA256：`d24877bad57e8f797c609c7f6917d3ad50dae6e8583e11f250941a5b29ec4573`。
- HTML SHA256：`34e062fcbcd44980eabceef94c583fcc6e1e5d66fc678ebb1dab358ea5b14313`。

### 定理・回路対応

| 場所 | 数学的仮定・入力 | 保証対象・出力 | K0.5での追加確認と限界 |
|---|---|---|---|
| §2.1、§2.5 Eqs.20–23,62–72、Lemma 7／Cor.8、PDF pp.22–24 | Hamiltonian block encoding。逆向き評価は正しいlog branchと小さいgenerator normが必要 | evolutionのoperator norm誤差と、encoded operatorの誤差の対応 | `||J||,||K||≤1−ξ` の下で逆変換定数1/ξ。Wが適切なencodingであるという条件を省略しない |
| §3.1 Lemma 9 Eq.74、PDF p.24 | J,K Hermitian、実τ≥0 | group commutatorの有限時間operator誤差 | 二つの三重commutator normとτ³/2による明示上界。外側S2やenergy特有の定理ではない |
| §6.1 Cor.27 Eqs.245–253、PDF pp.56–58 | `H=Σ_k B_k†B_k`, `B_k=Σ_j A_jk`, `||A_jk||<π/2`、controlled encoding query | 全stateへのSOS simulation operator誤差ε、query scaling、2 ancilla | 内側和・積・外側和をすべて会計。有限pのEq.252もO記法。SOS offset・励起gap依存は表示されない |
| §6.2 Cor.29、Fig.13、Eqs.254–262、PDF pp.58–59 | H Hermitian、`||H||≤1`、E_H query | H²のHamiltonian evolutionを誤差εで構成、query O(log(1/ε)) | shifted cubicの合成と不要項の相殺を確認。平方fragmentの指数化を常に直接gate1個で行う意味ではない |
| §6.3 Eqs.263–279、PDF pp.59–62 | real DFTHC tensors、normalized u、Hermitian number-preserving quadratic generators | 固定η sectorのcommutator上界とgate scaling、1 ancilla | k-local低energy定理と異なり、係数行列の構造を使う。ただしE_g比例のground bias／overlap保証はこの節から確認されない |
| Appendix C.2 Lemma 43 Eqs.393,399、PDF pp.84–86 | J normal、η-particle sector | `||Quad(J)||_η=max_{|S|=η}|Σ_{k∈S}λ_k(J)|` | Hermitian／anti-Hermitianでは端のη個の和を用いる。任意の多体fragment normが同じ価格で計算できるとは言わない |
| §1.6、§7 | 上記accessを満たすsimulation | QPE等への利用の位置づけ | 一般energy推定への利用は既報。SOS offsetを使う有用なfinite-h certificateまで同値だとは未認定 |

Cor.27は既存SOS表示を入力にする回路生成規則である。PF-nativeな良い表示を係数から発見する独立生成則まで与えると解釈しない。一方、既知SOS生成と同論文を組み合わせられるため、一般的なSOS＋PFやそのenergy推定への利用だけを空白とはしない。

§5.2 Proposition 26のGreen's function問題はground stateとground energyを入力に置く。そこでground関連の量を使うことを、§6のSOS offset依存ground-energy推定保証と読み替えない。

### 出版版の取得と差分の状態

| 対象 | 実際の取得状況 | 照合できたこと | 照合できなかったこと |
|---|---|---|---|
| arXiv v3 PDF／HTML | 原byte取得・SHAあり、関連節を読了 | 上表の定理・仮定・式・circuit | 全証明の独立再現は未実施 |
| ACM landing／DOI | direct browserは403。検索索引のpublisher記述は表示された | title、authors、巻号、頁、publication date、SOS回路の記述 | complete text、全数式、図、supplement |
| ACM PDF／fullHtml／epdf | すべてHTTP 403 | 取得不能を記録 | 定理・アルゴリズム・誤差定数のv3とのdiff、PDF SHA |
| Crossref DOI record | JSON原byte取得、SHAあり | DOIとjournal metadata、VorリンクとCC BY 4.0表記 | 書誌情報から本文同一性を推論しない |

**v3–出版版の数学的・アルゴリズム的diff：UNVERIFIED。** ページ数94対86だけでは差分の内容を判定できない。v3のeven-QSVT等が出版版で同じかは未確認。arXiv v1/v2を出版版の代用品にして比較していない。

有限定数を使う際の追加未確認点は、[報告§2.4](technical_report.md#24-原文の定数を数値certificateに使う前の注意)の平方内shift・一体残余・Eq.271→274のanticommutator係数である。paperの漸近的な結果を否定する指摘とは区別する。

## 2. Childs–Su–Tran–Wiebe–Zhu：有限時間の強対照

**A Theory of Trotter Error**、[arXiv:1912.08854v3](https://arxiv.org/html/1912.08854v3)、2021-02-03、82 pages。[PDF](https://arxiv.org/pdf/1912.08854v3)。関連出版題名は *Theory of Trotter Error with Commutator Scaling*, PRX 11,011020。[DOI](https://doi.org/10.1103/PhysRevX.11.011020)。今回読んだ版はarXiv v3であり、PRX版とのbyte同一性は主張しない。

PDF SHA256：`c76e2980dee799253ac3a5fa6176ac9005f78c844c1243dc2e579433137c7965`。

| 確認箇所 | 仮定 | 保証対象 | 今回への具体的含意 |
|---|---|---|---|
| §5.1 Eq.150、PDF p.39 | A,B Hermitian、実時間t≥0、ABAのS2 | `||S2(t)−e^(−it(A+B))||≤t³(||[B,[B,A]]||/12+||[A,[A,B]]||/24)` | 漸近BCHだけに依存しない、有限時間の明示的対照 |
| Prop.16 Eq.152、PDF p.39 | 有限個のHermitian summands、固定された二次公式の順序 | 二重commutator和によるoperator error | K0と同じ順序へ二分割評価を繰り返すと報告Eq.B。励起gap・PSDを仮定しない |
| Theorem 11周辺 | 指定次数のPFとnested commutator | 高次のcommutator scaling | 高次一般論はprefactorを隠す。二次の明示定数と同じ精度のcertificateとはしない |

新規追加文献だが、広い調査の開始ではない。Kang–Suが誤差理論として参照しており、ユーザー指定の「鋭いcommutator boundと比較」を具体化するために限定取得した。

## 3. Hejaziほか：個別energy摂動と低energy発展を分ける

### 3.1 Better product formulas for quantum phase estimation

[arXiv:2412.16811v1](https://arxiv.org/html/2412.16811v1)、2024-12-22。[PDF](https://arxiv.org/pdf/2412.16811v1)、8 pages。
PDF SHA256：`dd15c958444600b816176aa31dbe99e8ca07d9ff5a5cbf917599e27da8e6e804`。

| 確認箇所 | 仮定 | 保証対象 | 今回への含意 |
|---|---|---|---|
| §I Eq.9、Supplement §VII Eqs.23–30 | 対象eigenvalueの非消失gapを0〜sの全経路で仮定、effective Hamiltonian摂動 | 期待値によるenergy shift展開と高次補正 | 漸近energy係数の既報性は明確。励起gap情報を必要としない保証として転用しない |
| §II Eqs.10–13、Supplement §VI | H=A+B、両側がfast-forwardable、係数条件 | custom PFのenergy error次数改善 | 既知の二分割対照。多fragmentの残り全部を無料の一指数化にまとめない |
| §IV、PDF pp.4–5 | PSDでfast-forwardableな一定数群、k-local、有界degree、一定数のexponential | 低energyでのenergy誤差scaling | 密なnative SOS平方に有利な定数が直接移るとは確認できない |

K0で読んだ範囲から、今回はgapが経路全体の仮定である点とSupplementのprojector評価に戻った。新しい数値表は読取り・再集計していない。

### 3.2 Better bounds for low-energy product formulas

[arXiv:2402.10362v1 HTML](https://arxiv.org/html/2402.10362v1)／[PDF](https://arxiv.org/pdf/2402.10362v1)、2024-02-15、11 pages。
PDF SHA256：`4d1d1533d4cfdd9e993f491a28be139c609d3c283ab8eacb27f709cbc2ca2f30`。

| 確認箇所 | 仮定 | 保証対象 | 今回への含意 |
|---|---|---|---|
| Setting、Eqs.3–9 | commuting k-local interactionからなる群、interaction norm J、degree d、stateの低energy support | retained errorと高energyへのleakageを含む時間発展誤差 | 個別eigenbranchの励起gapを必要としない経路。spectral cutoff差Δ′−Δを励起gapと混同しない |
| fermionic拡張、SupplementのPSD specialization | 各interactionのfermionic次数・mode degree、PSD条件が必要な箇所を区別 | 低energy commutator／leakageの上界 | 非局所平方を展開してdegreeを増やした価格は未取得。一般の利点の証明に置き換えない |

今回主に再確認したのはSettingとEqs.3–9であり、補足全体の再証明は行っていない。K0の調査範囲は元台帳へ残す。

## 4. Mizuta–Kuwahara：励起gapなしの有限時間保証

**Trotterization is substantially efficient for low-energy states**。
[arXiv:2504.20746v1 PDF](https://arxiv.org/pdf/2504.20746v1)、arXiv登録2025-04-29、PDF日付2025-04-30、19 pages。
PDF SHA256：`94ef2cf57963c72f23f087b7ba735578da09d5a8fd8c2291ee70c6ce0497b19a`。

| 確認箇所 | 数学的仮定 | 保証対象 | 今回への含意 |
|---|---|---|---|
| setup Eqs.2–8、Theorem 1 Eqs.13–15、PDF pp.2–3 | PSD k-local terms、extensiveness g_ext、commuting layers Γ、PF次数p | 低energy発展の有限時間誤差、許容step域、leakage cutoff | Δに対する有利なscalingは既知。O定数はk,p等に依存し、非局所化後も一定とは言えない |
| Supplement S2 Theorem S1 Eq.S5、PDF p.8 | 各H_qがPSD・k-local・g_ext-extensive、`ΠH_qΠ≤ΔΠ` | 射影したnested commutator normのΔ一次bound | K0のSOS offset一次という原理自体の新規性を支えない。ΠとH_qの条件が必要 |
| Supplement S3のTheorem／Corollaryの所在確認 | Γの増大、small-step、cutoffに関する条件 | 主定理の有限時間への接続 | 完全な定数抽出・原証明の再現は未実施。密な化学への鋭い値なし |

K0でHTML本文日付とv1 PDFの相違を認識したため、今回も**上記v1 PDF**を基準にする。後のHTML・出版版をv1と同一視しない。g_ext、SOS offset E_g、励起gap g_0の3量を明確に分ける。

## 5. Yi–Crosson：出版本文の対象範囲

**Spectral analysis of product formulas for quantum simulation**、npj Quantum Information 8,37 (2022)。[出版HTML／DOI 10.1038/s41534-022-00548-w](https://www.nature.com/articles/s41534-022-00548-w)。

browserでLemma 3周辺、QPE Theorem 2 Eqs.23–27とDiscussionの関連箇所を確認した。Theorem 2は、対象eigenstate、effective-Hamiltonian条件、定義されたgap lower boundを用いてenergy errorとstep／depthを評価する。K0 Schur評価の個別branch契約と比較できるが、分子用の正のgap証拠を係数だけから供給する定理ではない。

直接HTTP取得したHTML 3,038 bytesはJavaScript challengeページだった。本文SHAとしては採用しない。browser本文の関連節読取りはできたが、出版PDF・supplementの完全再取得／版間diffは行っていない。

## 6. 今回は再調査を広げなかった文献

| K0 ID | 一次資料 | K0から保持する対応 | K0.5での扱い |
|---|---|---|---|
| L5 Lowほか | [2502.15882v1](https://arxiv.org/html/2502.15882v1) §III–IV | chemistry SOS／Gram／DFTHC、下界とnormalizationの競合 | K0本文・台帳参照。今回原文を再取得したと記載しない |
| L6 Kingほか | [2505.01528v1](https://arxiv.org/html/2505.01528v1) Theorems 4,5,7 | spectral amplification、energy taskのaccess・overlap・費用条件 | PF専用の定理と同一視しない。新規数表・版調査なし |
| L7 Rubin–Low–DePrince | [2602.05069v1](https://arxiv.org/html/2602.05069v1) §II、§III.3 | weighted SOS、symmetry ideal、Gram dual、v2RDM対応 | native Hermitian平方への費用保存は未証明というK0の限界を保持 |
| L8 Casaresほか | [2606.30741v1](https://arxiv.org/html/2606.30741v1) §III、§V、Appendix D | 強いPF・分割・processing・gate価格の対照 | empirical leading biasをstrict finite-step certificateに格上げしない |

これらの構成要素が既知であることと、統合した方式が不可能であることは別である。今回の調査で未取得の部分を、新しい論文名の探索で埋めたことにはしない。

## 7. 文献比較の到達点

| 主張 | 状態 |
|---|---|
| 一般SOSをPFで時間発展し、化学一体平方・固定粒子数を使う | 既報：Kang–Su §6 |
| このsimulationをenergy推定へ使い、一般誤差を演算子／出力分布へ移す | 既知理論から接続可能。SOSに特有の新規性とはしない |
| 固有値係数を期待値・gap・projectorで鋭く評価する | 既報：Yi／Hejazi。K0式はその標準技法との関係を持つ |
| PSD低energyで射影commutatorがenergy一次となる | 既報：Mizuta–Kuwahara等。非局所化学の鋭い定数は別 |
| 非局所化学で、同情報の対照を超える有用なSOS-offset依存finite-h ground certificate | 今回の取得範囲では確立を確認できない。未踏の証明とも断言しない |
| 独立した有利なPF-native分割生成R | 今回も未特定 |
| ACM出版版とarXiv v3が数学的に同一 | 未確認。取得できていない本文を取得済みとしない |

**限定調査完了・Mandatory STOP。**
