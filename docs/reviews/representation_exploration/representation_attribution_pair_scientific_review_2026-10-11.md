# Hamiltonian表現探索 A・B′：寄与分解batchの独立科学レビューと研究再設計

- 作成日：2026-10-11（日本時間）
- レビュー開始承認：利用者の「レビューを開始して」
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`representation-attribution-pair-20261010`
- 対象結果commit：`f98050e9402e8c65bb0f3dd27c7bd30d12fe7069`
- 固定実行source：`3975650908c594ce6dddb4fc2086167caf5b6460`
- 比較の基点：`c24dcede10ed9726e1e0b1030eca749ba7bc5cba`
- レビュー状態：**完了。ただし新規中心テーマの採択・一般最適性・出版新規性は未確定。**
- 判断：**現A-core／現B′-pairを主改善として進めるための同型toy性能batchは一旦終了し、研究課題の再設計へ戻る。Cと旧Bの保留を維持する。研究全体や全Hamiltonian変換の可能性を否定しない。**
- 次の科学的担当：**GPT（新しい構成課題と先行研究との差分の具体化）**。この記録は新しい数値batchを認可しない。

## 0. 結論

今回のbatchは前回レビューで残った二つの具体的な疑問に答えた。

Aでは、同じHamiltonianを安価なwhole-Pauli辞書で処理する対照を入れると、core導入を中心改善とする説明は支持されなかった。B′では、物理pair射影・involutionへの変換が数学的にも回路としても成立しても、今回のnative実装の費用がshot削減を上回った。前回の無条件8標本より丁寧な次数別費用推定でも、この差が残った。[R1–R5]

従って、次に同じ構成の角度・seed・toyサイズを増やすことを自動推奨しない。新しい改善機構を特定せずに計算条件だけを増やしても、利用者が目指す新規アルゴリズム研究へ近づくとは限らない。

ただし、これは「DF、core、pair、圧縮、PR協調設計は無意味」「Pauli表現が普遍的に最良」という結論ではない。今回の結論は、固定q1・小さい人工入力・固定compiler・固定信号精度とshot方策に対するものである。異なる自然な物理領域や異なる構成法で優位が生じる可能性は未検証である。

### 判断の区別

| 対象 | 今回の判定 | 継続・再開に必要な内容 |
|---|---|---|
| 現A：共通基底core＋残差 | 主改善候補としての同型性能探索を区切る | Pauli対照では得られない、別の構成機構または別の科学的問い |
| 現B′：pair QR＋QまたはZ/ZZ | primitiveとして保存、主改善候補としての同型性能探索を区切る | 新しい集約・access・入力構造で費用を回収する具体的な根拠 |
| 旧B：拡大空間＋反射RTE | 保留を維持 | 圧縮の実利益を保つ新しい実装経路 |
| C：現在のTHRIFT混合access | 保留を維持 | 既知の条件付きSU(2)より本質的に安い構成などの差分 |
| 研究全体 | 継続、中心課題を再設計 | 単なる既知法比較ではない、入力から出力までの新しい方法論 |

前回Aを継続しB′を比較対象へ追加したのは、coreの寄与が辞書変更と未分離で、B′のnative費用が未取得だったためである。今回その不確実性が狭まったことが、担当を「直ちにCodexの追加性能batch」から「GPTの研究課題設計」へ変える理由である。質問の表現だけを理由に判断を変えたのではない。

## 1. 証拠の取得と今回実施した確認

### 1.1 読んだもの

固定commitの報告書、science sourceの構成と次数別費用推定、一次resultの取得、CSVのq1値とq2/q4未実行区分、保存値寄与表、保存監査、独立verifierの主要部分、manifestの失敗原因を参照した。[R1–R7]

大きいresultは通常のContents取得で本文が空になったが、Git blobによる取得は可能だった。これは未pushではない。前段階で確認されたnative IRの取得経路も存在する。今回のレビューで全1558回路の行列を再構成・再compileしたわけではない。

| 内容 | 識別子・取得範囲 |
|---|---|
| 科学結果JSON | `run1/result.json`、blob `df8eb10c40ebcf5390964b73306e497c01e86339`、約4.93 MB。connectorで取得 |
| native IR | `run1/native_ir.json`、blob `bdffa80c29b986a40f401e9230dbefc34a519a6e`、約21.76 MB。利用可能性確認済み、今回の全件独立再実行はなし |
| 専用science source | `attribution_pair.py`、blob `7b6e70c0590eddbbe6a4f1cc12ea64ba7760a672` |
| 同一精度の保存表示 | `precision_resources.csv`、blob `a9c77e2633a72d28e23536288439be12ca8baa2a` |
| 保存監査 | `saved_evidence_audit.json`、177 source/input blobs、1558 IR、933 conditional draws、30 meanのチェックを報告 |

### 1.2 GPT側で今回新しく実行したこと

保存CSVに表示された数値を明示的に転記し、Python標準ライブラリだけで次を計算した。

- 同一shot方策のNを再計算し、表示値と照合。
- shot比、1回路費用比、総費用比の乗法分解。
- 固定方策内の理想shot数と残る削減余地。
- 現在のbaselineへ届くための1回路費用の損益分岐。

また、本文の多項式の辞書不変性、1粒子／1空孔sectorの表現を代数として検討した。新しい分子、行列スキャン、RTEイベント生成、回路compile、量子shotを実行していない。

補助計算の入力は**CSVフィールドの転記**であり、元result全体のbyte同一コピーや独立SHA再照合と呼ばない。標準ライブラリの再実行は `SCALAR_ARITHMETIC_REPLAY_MATCH` となった。数値と再現コードは付録へ収録する。

containerからのraw GitHub直接取得はDNS制限で失敗した。一方、GitHub connectorでは資料を取得できたため、科学レビューに必要な内容の欠損ではなく、未pushとも判断しない。全blobの追加ローカルhash監査を実施したとは主張しない。

### 1.3 監査結果の意味

保存監査は最大absolute wrapper差約3.13×10^-14、mean差約2.22×10^-16を報告し、位相・shotの改変例を拒否している。verifierはnative gateの直接行更新、explicit JW、exterior-power minorsを使い、Qiskit/projectのscience関数をそのまま呼ばない。[R4,R5]

これはlocalな別実装検査として有用だが、未知入力への外部再現、interval certificate、アルゴリズムの新規性、実資源優位そのものを証明しない。1558回路は1558種類の独立した物理系ではない。

## 2. 今回のtaskを変えずに評価する

今回の比較は固定時間の複素Hadamard信号である。εcomplex=0.05または0.02、各軸ε/√2、二軸合計失敗率0.05を使う。system状態準備、Hadamard ancilla、ordinary control、X/Y読み出しbasisを含み、測定自体はIR外metadataに記録する。[R1–R3]

各candidateのbiasは小系のbinary64演算子norm診断で、事前に使える解析的certificateではない。shot方策は

\[
N(\Gamma,b;\varepsilon)=\left\lceil
\frac{2\Gamma^2\log 80}{(\varepsilon/\sqrt2-b)^2}
\right\rceil,
\quad b<\varepsilon/\sqrt2.
\]

費用は、資源座標μごとに

\[
G_\mu=N\,[\mathbb E C_{X,\mu}+\mathbb E C_{Y,\mu}]
\]

である。RZ、CX、sx、x、size、depthを任意の重みで一本化しない。

RZ countをそのまま論理T countとはみなさない。Clifford角も含み得るため、fault-tolerant価値を主張する際は、その目的に応じた合成・精度・Clifford分類が別に必要である。ただし今回は固定されたRZ/CX比較として評価し、勝敗を変更するために指標を後付けで差し替えない。

actual costは結果前にq1だけと固定され、q2/q4はmean・bias・Γのみがある。従ってqやbackendを全最適化した方法間勝敗と解釈しない。一方、この制限があるだけで、q1で得た大きな費用差の情報価値がなくなるわけでもない。

## 3. A：core固有の改善を支持しないという判定

### 3.1 固定入力

3-mode JW全Fock8、正平方rank2、独立のone-body／constant補正なし。

\[
g_0=\begin{pmatrix}.8&.09&0\\.09&-.35&0\\0&0&.15\end{pmatrix},\quad
g_1=\begin{pmatrix}-.2&0&0\\0&.6&.07\\0&.07&-.45\end{pmatrix},\quad
H=\sum_l d\Gamma(g_l)^2.
\]

T=.4、q1のδ=.4、tail r1/K2、準備X0。分子・化学basis・未知入力の検証ではない。[R1,R2]

### 3.2 ε=.05、q1の主要結果

| candidate | λ | N/axis | X/Y一組のRZ費用 | X/Y一組のCX費用 | RZ総費用 | CX総費用 |
|---|---:|---:|---:|---:|---:|---:|
| native all-R | .900412 | 9285 | 60.6333 | 19.7282 | 562979.749 | 183176.525 |
| native LD1 | .274862 | 7236 | 207.1898 | 119.4428 | 1499225.311 | 864287.748 |
| native all-D | 0 | 7063 | 229 | 148 | 1617427 | 1045324 |
| collected whole-Pauli | .979 | 9668 | 14.3792 | 6.9330 | 139018.099 | 67028.724 |
| 従来core | .0915 | 7151 | 71.3037 | 81.1027 | 509892.718 | 579965.686 |
| 全占有対角core | .085 | 7148 | 71.0113 | 81.3004 | 507588.821 | 581135.568 |

これらは保存値であり、乱択費用は次数別推定による点推定である。SEや全6指標は元資料を参照。[R1,R3]

whole-Pauliはnative all-RよりλもNも大きい。それでも1回路が安く、RZ総費用比0.24693、CX比0.36592となる。従って「低いλほどよい表現」という単一指標では説明できない。

全占有coreはwhole-PauliよりNを約26.1%減らすが、RZの1回路費用を約4.94倍、CXを約11.73倍にしている。結果として総RZ約3.65倍、CX約8.67倍、depth約4.98倍である。元のcoreを少し修正した利益より、安い対照の導入で変わった比較の意味の方が大きい。

### 3.3 何が前回から更新されたか

前回は、coreを作る変更と、乱択辞書をnative DFから直接Pauliへ変える変更が同時に起きていた。今回whole-Pauli対照が入ったことで、この交絡を切り分けられた。

その結果、「core導入の効果としてnative-prefixに対するRZ差を解釈する」という研究仮説は、今回の入力・条件では支持されない。これは、前回の数値が虚偽だったということではない。比較対象が十分でなかった段階の予備的な差に、別の説明が付いたという意味である。

### 3.4 小さな対角補完の意味

全占有対角coreは、以前導出した

\[
.0081(n_0+n_1-2n_0n_1)+.0049(n_1+n_2-2n_1n_2)
\]

を吸収する。既存support内でλが.0915→.085へ下がったが、Nは7151→7148だけである。RZ総費用は約0.452%減り、CXは約0.202%増えた。[R1,R3]

この局所効果は保存してよい。しかし、これだけを新しい一般構成原理やcore研究の主要資源改善とはしない。core全体がwhole-Pauliに大きく劣る現条件で、局所改善を重ねて主張を作るのは適切でない。

## 4. B′：恒等式・回路成立と、アルゴリズムの価値を区別する

### 4.1 成立している内容

二本の非直交軌道を直交化し、

\[
c_x^\dagger c_y^\dagger c_yc_x
=\Delta n_{b_1}n_{b_2},\qquad
\Delta=\|x\|^2\|y\|^2-|x^\dagger y|^2
\]

とする構成、Q=I−2n_b1 n_b2、scalar位相を含むRTE辞書、controlled primitiveの実装は今回検証されている。complex係数のconjugation、QR completion、wedgeによるΔ計算もsourceで扱う。[R1,R2,R4]

この結果は有用なprimitiveの検証として残る。一方、恒等式が正しいこと、係数normが下がること、総資源が減ること、新規算法として成立することは四つの別問題である。

### 4.2 ε=.05、q1の費用

| candidate | λ | N/axis | X/Y一組RZ | X/Y一組CX | RZ総費用 | CX総費用 |
|---|---:|---:|---:|---:|---:|---:|
| 旧拡大・反射 | .9625 | 7544 | 87.2618 | 27.2927 | 658302.948 | 205896.080 |
| 係数から直接Pauli | 1.045353 | 7643 | 17.4139 | 7.0323 | 133094.499 | 53748.197 |
| pair Z/Z/ZZ | .779443 | 7363 | 67.7149 | 22.0702 | 498584.947 | 162502.740 |
| pair Q | .519629 | 7166 | 81.2449 | 30.5781 | 582201.076 | 219122.803 |

Qは直接PauliよりNを約6.24%減らすが、1回路RZは約4.67倍。その積として総RZ約4.37倍、CX約4.08倍である。[R1,R3]

Qの費用SE=0は、有限辞書のorder0とorder2をすべて列挙したためである。量子測定が無雑音、合成が厳密、環境移送でも同じ値という意味ではない。今回のSEの意味を維持する。

### 4.3 直接Pauli対照の位置づけ

新しい直接辞書はquartic係数代数から生成され、以前のdense参照由来31primitiveではなく18primitiveとなっている。微小supportの違いとoperator差を記録し、元の結果を書き換えない。[R1,R2]

ここで直接Pauliを「必ず指数的に高価なので実用的対照でない」と退けることはできない。通常の二体電子Hamiltonianは有限本体次数の係数代数からPauli係数を構成できる。このbatch自身もそれを行っている。ただし大系で全係数を明示保存する古典費用、term数、係数1normや回路費用が最良という意味ではない。

## 5. 共通の費用機構を数式で整理する

### 5.1 同じHamiltonian・同じscalarなら、有限RTEの補正平均は辞書に依存しない

H=cI+Rで、同じRを異なるinvolution辞書で表したとする。K=2の補正されたq反復平均は

\[
M_q=e^{-icT}\left[
I-i\frac{T}{q}R-\frac{T^2}{2q^2}R^2
+i\frac{T^3}{6q^3}R^3
\right]^q.
\]

右辺はRそのものの多項式であり、どの辞書から生成したかには依存しない。λはnormalizationと実行eventの費用を変えるが、同じscalar・q・Kなら補正平均のbiasを直接変えない。

Aのnative all-Rとwhole-Pauliのbiasが一致する理由はこれで説明できる。Bの直接Pauliとpair Z/Z/ZZも同じscalarを持つため同様である。一方pair Qはscalarも変わるため、有限Taylorのbiasが異なり得る。Qのbias改善を、λが小さくなったことだけに帰属しない。[R1–R3]

### 5.2 損益分岐

各資源座標で

\[
\frac{G_1}{G_0}=\frac{N_1}{N_0}\frac{C_1}{C_0},\qquad
G_1<G_0\iff \frac{C_1}{C_0}<\frac{N_0}{N_1}
\]

である。これは単純な会計恒等式で、新規定理ではない。しかし、本候補が実際にどれほどの回路費用を許容できるかを明確にする。

| 比較、ε=.05 | 実際のN1/N0 | 許容されるC1/C0（同じNなら） | 実際のRZ費用比 | 実際のCX費用比 |
|---|---:|---:|---:|---:|
| 全占有core / whole-Pauli | .73935 | 1.35255 | 4.93847 | 11.72651 |
| pair Q / direct Pauli | .93759 | 1.06656 | 4.66552 | 4.34821 |

ここから、今回の主因は小さい数値差や少数標本ではなく、「許容される追加費用を回路実装が大きく超えた」ことと解釈できる。ただし表のCは有限compiler条件下の保存期待費用であり、最適なあらゆる回路の下界ではない。

### 5.3 normalization/biasだけを改善しても、現1回路費用では届かない

今回のshot方策に限ればΓ≥1、b≥0なので

\[
N\ge N_{\min}^{\mathrm{policy}}
=\left\lceil4\log80/\varepsilon^2\right\rceil.
\]

ε=.05なら7012、ε=.02なら43821。この値は量子情報理論的最小測定数ではない。別の推定器、状態情報、振幅推定、異なるconfidence規約に対する下界でもない。

現在のCを固定し、Nだけを理想化した総費用比は次のとおり。

| 比較 | ε | 理想Nでも残るRZ費用比 | 理想Nでも残るCX費用比 |
|---|---:|---:|---:|
| 全占有core / whole-Pauli | .05 | 3.58177 | 8.50499 |
| 全占有core / whole-Pauli | .02 | 3.37022 | 8.00267 |
| pair Q / direct Pauli | .05 | 4.28034 | 3.98923 |
| pair Q / direct Pauli | .02 | 4.27447 | 3.98376 |

従って、同じ費用の回路のままλやbiasだけをさらに小さくする方法では、現在の差を埋められない。これは固定方策・固定Cの条件付き判断であり、q変更や回路構成変更まで禁止・否定するものではない。

実際のNを固定した場合、RZで並ぶためにはcoreは約72.6%、pair Qは約77.1%の1回路費用減が必要である。閾値を研究成功条件に新設しているのではなく、現在の点の損益分岐を計算している。

## 6. 不確かさ・数値条件・一般化

### 6.1 rare eventについて今回答えられたこと

order0は全component列挙、order2は条件付き96 IID draw、pair Qのorder2は4³=64全列挙。候補間およびX/Y間でuniformを共有し、差のSEを対応付けている。[R1,R2]

前回の無条件8標本で高次eventを引けない問題は、今回かなり直接的に対処されている。ただしSEは厳密CIではなく、異なる入力・compilerや探索後の多重比較までの信頼保証でもない。大差を判断材料として使いつつ、一般優位の証明へは拡張しない。

### 6.2 q・時間・系サイズ

costはq1のみ。q2/q4のmean診断を使って全q資源最適性を主張しない。長時間や、同じλが系サイズとともに増える場合はΓの経済性が変わり得る。従って、Pauliが大系でも必ず勝つとは結論しない。

一方、別条件にすれば逆転する可能性が残るという理由だけで、あらゆる方向を無期限に継続する必要もない。今回の目的は既知法の適用域を網羅することではなく、新しいアルゴリズムである。再開には別の自然なtask・構造・構成差分が必要になる。

### 6.3 小さいsectorの意味

Aの準備は1粒子である。1粒子sectorではdΓ(g)はgとして働くため、ΣdΓ(g)²の制限はΣg²であり、その信号は3×3のone-body dynamicsに還元できる。

Bは3物理mode中2粒子で、1空孔sectorである。一般にn-modeのn−1粒子sector上では、one-body operatorの表現は適切なhole基底で

\[
K\mapsto \operatorname{Tr}(K)I-K^T
\]

となる。この写像はn>1で全n×n Hermitian行列への全射である。与えられたsector Hamiltonian Mに対して、

\[
K=\frac{\operatorname{Tr}M}{n-1}I-M^T
\]

を取れる。従ってこのsectorの有限次元時間発展自体はone-holeのGaussian表現を持つ。

これは今回のfull-Fock bias比較が間違いという意味ではない。候補全体のfull-Fock operatorを同じKで置き換えられるとも言っていない。現在の準備状態上の物理taskから、一般の電子相関を含む分子の資源優位へは直結しない、という区別である。

将来、相関した多電子の効果を主張するのであれば、その機構がこの特殊な還元だけでは説明されない入力を選ぶ必要がある。特定の分子、basis、電子数を新たな固定条件にはしない。粒子数・空孔数を増やすだけでも非可解性や優位は保証されない。

## 7. 先行研究照合による新規性の評価

### 7.1 PR原論文に既に含まれること

GüntherらのPR原論文は、DF fragmentの一部を決定論側に置き、残差をPauli表現へ戻す構成を明示している。第1/第2factorizationの切断と乱択処理、基底変更の費用、別軌道基底でのinvolution、λ低減と実装費用の競合も議論する。[L1]

従って「DFとPauliを使い分ける」「native factorよりPauli tailが安い場合がある」というだけでは、この研究の独立した新規性を主張できない。

### 7.2 density fragmentとreflection

Martínez-Martínez–Yen–Izmaylovは、fermionic/qubit partitioning、solvable density fragments、one-bodyの再配分、reflection表現によるLCU norm低減と基底回転費用を扱う。[L2]

重要な留保：同論文のr_i r_j=(I−2n_i)(I−2n_j)と、今回のQ=I−2n_i n_jは同じ演算子ではない。前者は2-modeで固有値+1/−1が2個ずつ、後者は+1が3個、−1が1個であり、単なるunitary共役で同一にはならない。

したがって、この一論文だけから今回のQR＋Q辞書全体が完全同一と断定しない。ただし、基本的なdensity primitiveやprojectorの中心化だけを独立新規性と認定する根拠もない。priorityと実資源価値は別に評価する。今回後者の支持が得られなかったことが、現方法を主題から外す直接の理由である。

### 7.3 圧縮・残差・回路構造の共同設計

RC-DFは資源関連のλや近似誤差を考慮するfactorizationを提案している。[L3] SPRINT/GRADEはfactorization、明示Pauli残差、rotation精度、near-integrable PF、randomization、symmetry protectionなどを組み合わせる。[L4]

「resource-aware factorization」「残差とnative blockのhybrid」「basis費用を目的へ入れる」という広い標語だけでは、既存研究との差分が足りない。新しい入力クラス、具体的構成、古典計算量、誤差・費用の関係のいずれかで、再利用可能な差分を示す必要がある。

### 7.4 何をまだ確認していないか

全世界の先行研究に対する優先性を証明したわけではない。2026年10月11日時点に取得できた一次資料の関連節・本文と書誌を使った限定照合である。全文HTMLが取得できないものは公式PDFを参照し、必要箇所はスクリーンショットでも確認した。

刊行版のページ番号とparsed textの式番号がずれる可能性があるため、PR原論文は節VII.BとPDF内ページ13の該当議論を主な照合点とし、未確認の式番号を新規性否定の根拠にしない。

## 8. 現方向を区切ることと、研究を断念することは異なる

今回成立していないのは、現A-core／現B′-pairを中心にした改善claimである。以下は未証明である。

- Pauli表現があらゆるsize・basis・precisionで最良。
- DF・THC・isometryをPRへ使っても改善できない。
- Hamiltonianの前処理や変換に科学的価値がない。
- qやfinite Kや物理taskを適切に選び直しても差が逆転しない。

これらを結論しない。一方、可能性が残るというだけで現在の候補を残し、次のtoyを自動作成する方針も採らない。

今回の成果は、正しい数学的表現から資源優位へ進む際に、基底変換・central gate・比較辞書・shot余地がどのように障害になるかを分けた点にある。これは今後の設計に利用できる。しかし「λだけでは足りない」という一般論や、数個のtoyで不利だったことだけを独立論文の十分な主要成果とは評価しない。

## 9. 再設計する研究課題の候補

ここで新しい第4のtoyを採択するのではなく、次の**構成問題の形**を比較対象にする。下記は未検証の研究提案であり、新規性・勝利・実装認可を意味しない。

### 9.1 安価な表現を維持したまま重要項を集中させる構成

現在のtoyではdirect Pauliの実行が安い。この利点を残したまま、orbital変換や係数構造を使って、少ない安価な演算で処理できる部分へweightを集中できるかを問う。

単に全λを最小化するのではなく、例えば「実装予算B内で決定論処理できる集合を作ったとき、残差とPF誤差をどこまで小さくできるか」という達成曲線を生成する問題へ置き換える。外側の基底変換、更新時のtensor費用、残差展開、情報の入手方法も含める。

ただしPR原論文はweight concentrationと異なるorbital選択を将来課題として明示し、複合simulationとGRADEにも近い。新しい目的関数だけで新規アルゴリズムとしない。安価な更新則・構成保証・既存法が作れない支持構造などの具体差分が必要である。[L1,L4,L7]

**位置づけ：既存基盤と最も近いが、既知法への還元リスクも高い。候補として再検討、採択しない。**

### 9.2 局所性・低rank構造を使い、1回の作用自体を安くする

今回の損益分岐から、固定条件ではshot低減より回路費用の削減が必要と分かった。従って「任意のfragmentを安くする」ではなく、局所性、距離減衰、係数blockの低rankなど、入力の構造を使って複数相互作用をまとめて実行する構成を探す。

Low–Su–Tong–Tranは、こうした構造を利用した高速Trotter stepと、一般の対角2-local Hamiltonianに対するgate下界を示している。[L5] 単なる可換groupingだけで一般に大幅な削減が得られるとは想定しない。

新しい課題になり得るのは、量子化学の利用可能な係数・軌道からその構造を生成し、近似誤差や残差の費用を制御する手順である。ただし同論文、local composite simulation、GRADEとの差分を先に示す必要がある。[L4,L5,L7]

**位置づけ：回路費用へ直接作用するが、既存方法の再実装だけにならない具体的構造が必要。**

### 9.3 問題表現そのものを小さくする方向

現在は主として同じHamiltonianの正確な表現を変更した。別方向として、active-space／有効Hamiltonian／downfoldingなど、model errorを管理して軌道数や相互作用数そのものを減らす研究も残る。これは元の依頼で許容されていた方向で、PRとの組み合わせを強制しない。

DUCC等のdownfoldingには既に研究があるため、単にそれへPRを適用するだけでは十分ではない。動的signalでの誤差、誘導高体項、classical前処理、状態・observable変換まで含む新しい選択・構成法が必要になる。[L8]

**位置づけ：研究としての変化幅は大きいが、古典費用・model biasという別問題も増える。今ここで採択せず比較に残す。**

### 9.4 現時点で採らない進め方

- Aを勝たせるためだけに、恣意的な時間・精度・重みを探す。
- B′をわずかに再compileするたびに新手法候補と呼ぶ。
- q1の劣位だけからDF・PR全体の不可能性を主張する。
- 既存Track Bのsampling／return改善だけへ戻り、この研究の新規貢献とする。
- 新しい構成を決めず、まず大分子の長時間batchを回す。

これはあらゆる追加計算の禁止ではない。明示された新しい仮説や自然な適用領域があり、その情報価値が説明できる検証なら再開できる。

## 10. 直近の次の作業と役割

### 10.1 次の科学的担当はGPT

現batchは前回の判別目的を果たした。従って直近では、Codexにさらに条件を増やす指示を出すより、GPTが新しい構成課題を具体化する。

次の設計資料が答えるべき問いは以下である。

1. 同じ安い既知対照では解決できない、どの問題を解くのか。
2. 提案の入力・出力・数学的処理は何か。
3. 何を保存し、何を近似し、誤差をどう扱うか。
4. どの費用をどの機構で削減するのか。利益を使い切る負担はないか。
5. いちばん近い既知構成に対して、具体的に何が異なるか。
6. 最小の検証で何が分かれば、その課題を継続・修正できるか。

一般的な定理、全分子での勝利、厳密な大域最適化を最初から必須にしない。経験的改善でも、方法の差分と説明可能な機構があれば科学的に評価する。

### 10.2 Codexの直近の位置づけ

今回の結果・source・原STOPを保持する。現時点で新しいscience batchやTrack A/Bへの統合を認可しない。

通常の資料整理・索引補足・技術修正は、必要があり明示された範囲でCodexへ任せられる。しかし研究方針の新規採択を、技術的な通常作業に偽装して先行させない。新規資料不足を理由とする追加pushは今回必要ない。

### 10.3 次のレビューを繰り返す条件

同じ固定証拠の再承認をルーチン化しない。新しい構成差分、自然な適用領域、重要な実装・科学結果が出たときに、その範囲について再レビューの必要性を判断する。

## 11. 全体manifestの不整合

全体checkerは、基点と今回の双方で `manifest.relevant_commits[7]` にdescriptionがないことにより失敗する。今回追加entryのschema viewはPASS、旧84 result entriesは保持と記録されている。[R7]

これを今回のscience failureやresult欠損と取り違えない。一方でrepository全体のmanifestが完全健全とも呼ばない。固定資料が個別取得でき、対応sourceと専用監査が存在するため、本レビューの阻害要因にはしない。全体manifestを修正する場合は別の通常保守作業として扱い、研究内容を再実行する理由にしない。

## 12. 前回判断からの差分と残す成果

前回：Aの寄与分解を行い、B′のnative比較を実施する。

今回：その判別結果が得られたので、現A／B′を主改善として追うための同型batchを区切る。初等恒等式や小λだけを次の候補選定の根拠にはしない。次は研究課題を設計するGPT側へ戻す。

残す成果は、whole-Pauli/native/core/pairの再利用可能なbaseline、corrected first momentとscalarの取扱い、次数別費用推定、複素pair QRとcontrolled phaseの検査、source/result/native IR/保存監査である。これらを過去結果から消さない。

本研究を既存法の比較だけで論文化する方針に勝手に変更しない。現在の証拠を科学的な設計材料として使い、利用者の目的である新規算法または本質的な改善へつながる構成課題を改めて選ぶ。

## 13. 参考資料

### Repository一次証拠

- [R1] [A core寄与分解・B′物理pairの限定結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f98050e9402e8c65bb0f3dd27c7bd30d12fe7069/docs/research/representation_attribution_pair_results_20261010.md)
- [R2] [固定実行source attribution_pair.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3975650908c594ce6dddb4fc2086167caf5b6460/src/trottertracks/representation_exploration/attribution_pair.py)
- [R3] [同一精度資源CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f98050e9402e8c65bb0f3dd27c7bd30d12fe7069/artifacts/representation_attribution_pair/2026-10-10/precision_resources.csv) / [一次result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f98050e9402e8c65bb0f3dd27c7bd30d12fe7069/artifacts/representation_attribution_pair/2026-10-10/run1/result.json)
- [R4] [保存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f98050e9402e8c65bb0f3dd27c7bd30d12fe7069/artifacts/representation_attribution_pair/2026-10-10/saved_evidence_audit.json)
- [R5] [独立保存verifier](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f98050e9402e8c65bb0f3dd27c7bd30d12fe7069/scripts/verify_representation_attribution_pair.py)
- [R6] [実行source・環境・資源audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f98050e9402e8c65bb0f3dd27c7bd30d12fe7069/artifacts/representation_attribution_pair/2026-10-10/run1/run_audit.json)
- [R7] [manifest検査記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f98050e9402e8c65bb0f3dd27c7bd30d12fe7069/artifacts/representation_attribution_pair/2026-10-10/manifest_check_audit.json)
- [R8] [保存値の寄与表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f98050e9402e8c65bb0f3dd27c7bd30d12fe7069/artifacts/representation_attribution_pair/2026-10-10/attribution_summary.json)

### 一次文献

- [L1] Jakob Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647, v2 metadata dated 2026-07-10, PRX Quantum 7, 020332 (2026). [公式書誌](https://arxiv.org/abs/2503.05647) / [公式PDF](https://arxiv.org/pdf/2503.05647). 第VII節、とくにDF residualとPauli、orbital basis費用の議論。
- [L2] Luis A. Martínez-Martínez, Tzu-Ching Yen, Artur F. Izmaylov, *Assessment of various Hamiltonian partitionings for the electronic structure problem on a quantum computer using the Trotter approximation*, Quantum 7, 1086 (2023), arXiv:2210.10189v2. [公式書誌](https://arxiv.org/abs/2210.10189) / [公式PDF](https://arxiv.org/pdf/2210.10189). density fragment、postprocessing、Eqs.37–42付近、回路費用。
- [L3] Oumarou Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum 8, 1371 (2024). [公式掲載ページ](https://quantum-journal.org/papers/q-2024-06-13-1371/).
- [L4] Pablo A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1 (2026). [公式書誌](https://arxiv.org/abs/2606.30741) / [公式PDF](https://arxiv.org/pdf/2606.30741). SPRINT/GRADE、明示Pauli残差、basisと精度を含む構成。
- [L5] Guang Hao Low, Yuan Su, Yu Tong, Minh C. Tran, *On the complexity of implementing Trotter steps*, arXiv:2211.09133 / PRX Quantum 4, 020323 (2023). [公式書誌](https://arxiv.org/abs/2211.09133) / [刊行論文](https://doi.org/10.1103/PRXQuantum.4.020323).
- [L6] Maxine Luo, J. Ignacio Cirac, *Efficient Simulation of Quantum Chemistry Problems in an Enlarged Basis Set*, arXiv:2407.04432v2 (2025). [公式全文](https://arxiv.org/html/2407.04432v2). isometric THC、auxiliary reset、状態simulationと今回の第一momentのtaskの差。
- [L7] Matthew Hagan, Nathan Wiebe, *Composite Quantum Simulations*, arXiv:2206.06409; Matthew Pocrnic et al., *Composite QDrift-Product Formulas for Quantum and Classical Simulations in Real and Imaginary Time*, arXiv:2306.16572. [前者](https://arxiv.org/abs/2206.06409) / [後者](https://arxiv.org/abs/2306.16572). 異なるsimulation方法や局所分解の組合せという大枠の先行研究。
- [L8] Harjeet Singh et al., *Qubit-efficient quantum chemistry with the ADAPT variational quantum eigensolver and double unitary downfolding*, arXiv:2504.18683 (2025). [公式書誌](https://arxiv.org/abs/2504.18683). 有効Hamiltonian、高体項、外部振幅近似・相関の誤差問題を持つ別方向の例。

## 付録A. 保存数値の再現可能な算術

以下はこのレビューで実行した数値処理の入力とコードである。元科学runを再実行するコードではない。JSON入力は保存CSVの表示フィールドの転記である。独立raw download/hash再照合や全IR再実行を行ったと誤解しない。

実行結果：`SCALAR_ARITHMETIC_REPLAY_MATCH`、比較3組、science runs追加0。


### A.1 再実行方法

下記の三つのコードブロックを、それぞれ指定ファイル名で同じディレクトリに保存する。Python標準ライブラリのみを使う。元repositoryへのアクセスや依存packageのinstallは不要である。

```text
python review_arithmetic.py
```

成功時の出力は以下となる。

```json
{
  "status": "SCALAR_ARITHMETIC_REPLAY_MATCH",
  "comparisons": 3,
  "new_science_runs": 0
}
```

この一致は保存CSVから転記したscalar会計の再現であり、元の1558 native IR、分子Hamiltonian、RTE sampleを再認証した意味ではない。

### A.2 入力

ファイル名：`review_numeric_inputs.json`

```json
{
  "input_provenance": "Manual exact displayed CSV field transcription from committed precision_resources.csv, f98050e9402e8c65bb0f3dd27c7bd30d12fe7069; not a raw blob copy.",
  "computation": "Saved-scalar arithmetic only; no original science functions, matrix campaign, sampling, compilation or quantum execution.",
  "epsilons": [
    0.05,
    0.02
  ],
  "inputs": {
    "A_whole": {
      "N": [
        9668,
        64212
      ],
      "gamma": 1.1512673539304428,
      "b": 0.0006921342096993079,
      "cost": {
        "rz": 14.379199332904026,
        "cx": 6.933049675304585,
        "depth": 20.18606687487432
      }
    },
    "A_complete": {
      "N": [
        7148,
        45838
      ],
      "gamma": 1.001155870173733,
      "b": 0.00029869444525565386,
      "cost": {
        "rz": 71.01130682436433,
        "cx": 81.30044324216627,
        "depth": 136.09081556388188
      }
    },
    "A_native0": {
      "N": [
        9285,
        61665
      ],
      "gamma": 1.1282073852875323,
      "b": 0.0006921342096992005,
      "cost": {
        "rz": 60.633252437086384,
        "cx": 19.72822024390249,
        "depth": 73.23305389001467
      }
    },
    "B_direct": {
      "N": [
        7643,
        47830
      ],
      "gamma": 1.0435297843238907,
      "b": 1.641258542954307e-05,
      "cost": {
        "rz": 17.41390805522787,
        "cx": 7.03234288792114,
        "depth": 21.06932328622251
      }
    },
    "B_pairQ": {
      "N": [
        7166,
        44801
      ],
      "gamma": 1.0107892914794936,
      "b": 4.732594927159785e-06,
      "cost": {
        "rz": 81.24491717818447,
        "cx": 30.57811938034501,
        "depth": 96.98167166801348
      }
    }
  }
}
```

### A.3 再現コード

ファイル名：`review_arithmetic.py`

```python
"""Post-hoc scalar analysis of committed, displayed CSV fields.

No original repository modules, new matrix experiments, circuit compilation,
new trajectories, or quantum measurements are executed. Input numbers were
transcribed from fixed-commit precision_resources.csv; they are not a byte-copy
of the whole original result. This script deliberately preserves that scope.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main() -> None:
    data = json.loads((ROOT / 'review_numeric_inputs.json').read_text(encoding='utf-8'))
    stats = data['inputs']
    epsilons = data['epsilons']
    result = {}
    for candidate, baseline in (
        ('A_complete', 'A_whole'), ('B_pairQ', 'B_direct'), ('A_whole', 'A_native0')
    ):
        rows = []
        for j, eps in enumerate(epsilons):
            n0 = math.ceil(4 * math.log(80) / eps**2)
            nc, nb = stats[candidate]['N'][j], stats[baseline]['N'][j]
            for name in (candidate, baseline):
                value = stats[name]
                margin = eps / math.sqrt(2) - value['b']
                if not margin > 0:
                    raise ValueError('An input is ineligible under the stated policy')
                predicted = math.ceil(2 * value['gamma']**2 * math.log(80) / margin**2)
                if predicted != value['N'][j]:
                    raise ValueError((name, eps, predicted, value['N'][j]))
            row = {
                'epsilon': eps, 'N_floor_under_same_policy': n0,
                'N_ratio': nc / nb,
                'allowed_cost_ratio_actual_N': nb / nc,
                'allowed_cost_ratio_ideal_N': nb / n0,
                'remaining_N_reduction': 1 - n0 / nc, 'metrics': {},
            }
            for metric, cc in stats[candidate]['cost'].items():
                cb = stats[baseline]['cost'][metric]
                wr = nc * cc / (nb * cb)
                row['metrics'][metric] = {
                    'cost_ratio': cc / cb, 'work_ratio': wr,
                    'ideal_N_work_ratio_at_unchanged_cost': n0 * cc / (nb * cb),
                    'fractional_cost_reduction_to_tie_at_same_N': max(0, 1 - 1 / wr),
                    'candidate_cost_threshold_at_same_N': nb * cb / nc,
                }
            rows.append(row)
        result[f'{candidate}_vs_{baseline}'] = rows
    expected = json.loads((ROOT / 'review_arithmetic.json').read_text(encoding='utf-8'))
    if result != expected:
        raise ValueError('Scalar replay differs from the saved review arithmetic')
    print(json.dumps({'status': 'SCALAR_ARITHMETIC_REPLAY_MATCH',
                      'comparisons': len(result), 'new_science_runs': 0}, indent=2))

if __name__ == '__main__':
    main()
```

### A.4 このレビューで実際に得た計算結果

ファイル名：`review_arithmetic.json`

```json
{
  "A_complete_vs_A_whole": [
    {
      "epsilon": 0.05,
      "N_floor_under_same_policy": 7012,
      "N_ratio": 0.7393462970624741,
      "allowed_cost_ratio_actual_N": 1.3525461667599328,
      "allowed_cost_ratio_ideal_N": 1.378779235596121,
      "remaining_N_reduction": 0.01902630106323444,
      "metrics": {
        "rz": {
          "cost_ratio": 4.93847433228557,
          "work_ratio": 3.6512427107134107,
          "ideal_N_work_ratio_at_unchanged_cost": 3.5817730676444373,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7261206446052414,
          "candidate_cost_threshold_at_same_N": 19.448530938796324
        },
        "cx": {
          "cost_ratio": 11.726505224931126,
          "work_ratio": 8.669948215536584,
          "ideal_N_work_ratio_at_unchanged_cost": 8.504991170585132,
          "fractional_cost_reduction_to_tie_at_same_N": 0.8846590573392358,
          "candidate_cost_threshold_at_same_N": 9.377269762289414
        },
        "depth": {
          "cost_ratio": 6.741819315642646,
          "work_ratio": 4.984539146484654,
          "ideal_N_work_ratio_at_unchanged_cost": 4.88970180402216,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7993796476239433,
          "candidate_cost_threshold_at_same_N": 27.30258737357092
        }
      }
    },
    {
      "epsilon": 0.02,
      "N_floor_under_same_policy": 43821,
      "N_ratio": 0.7138541082663676,
      "allowed_cost_ratio_actual_N": 1.4008464592696017,
      "allowed_cost_ratio_ideal_N": 1.4653248442527556,
      "remaining_N_reduction": 0.044002792442951244,
      "metrics": {
        "rz": {
          "cost_ratio": 4.93847433228557,
          "work_ratio": 3.5253501906700615,
          "ideal_N_work_ratio_at_unchanged_cost": 3.3702249379412876,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7163402368801466,
          "candidate_cost_threshold_at_same_N": 20.14305047263042
        },
        "cx": {
          "cost_ratio": 11.726505224931126,
          "work_ratio": 8.371013930424109,
          "ideal_N_work_ratio_at_unchanged_cost": 8.002665941906605,
          "fractional_cost_reduction_to_tie_at_same_N": 0.8805401581801768,
          "candidate_cost_threshold_at_same_N": 9.71213808959069
        },
        "depth": {
          "cost_ratio": 6.741819315642646,
          "work_ratio": 4.812675415661054,
          "ideal_N_work_ratio_at_unchanged_cost": 4.600904258250426,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7922153659592597,
          "candidate_cost_threshold_at_same_N": 28.277580308247085
        }
      }
    }
  ],
  "B_pairQ_vs_B_direct": [
    {
      "epsilon": 0.05,
      "N_floor_under_same_policy": 7012,
      "N_ratio": 0.9375899515896899,
      "allowed_cost_ratio_actual_N": 1.066564331565727,
      "allowed_cost_ratio_ideal_N": 1.0899885909868796,
      "remaining_N_reduction": 0.021490371197320668,
      "metrics": {
        "rz": {
          "cost_ratio": 4.665518901358488,
          "work_ratio": 4.374343640865489,
          "ideal_N_work_ratio_at_unchanged_cost": 4.28033737227865,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7713942748672247,
          "candidate_cost_threshold_at_same_N": 18.573053204871144
        },
        "cx": {
          "cost_ratio": 4.348212234199567,
          "work_ratio": 4.076840098164869,
          "ideal_N_work_ratio_at_unchanged_cost": 3.9892272911431848,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7547119887164239,
          "candidate_cost_threshold_at_same_N": 7.500446091596605
        },
        "depth": {
          "cost_ratio": 4.602979903556323,
          "work_ratio": 4.315707704943688,
          "ideal_N_work_ratio_at_unchanged_cost": 4.222961544385312,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7682882928205519,
          "candidate_cost_threshold_at_same_N": 22.47178870731212
        }
      }
    },
    {
      "epsilon": 0.02,
      "N_floor_under_same_policy": 43821,
      "N_ratio": 0.9366715450554045,
      "allowed_cost_ratio_actual_N": 1.067610097988884,
      "allowed_cost_ratio_ideal_N": 1.0914858173022066,
      "remaining_N_reduction": 0.021874511729648938,
      "metrics": {
        "rz": {
          "cost_ratio": 4.665518901358488,
          "work_ratio": 4.370058797820649,
          "ideal_N_work_ratio_at_unchanged_cost": 4.274465895388466,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7711701269331432,
          "candidate_cost_threshold_at_same_N": 18.591264085211247
        },
        "cx": {
          "cost_ratio": 4.348212234199567,
          "work_ratio": 4.072846671636521,
          "ideal_N_work_ratio_at_unchanged_cost": 3.9837551393447463,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7544714837992692,
          "candidate_cost_threshold_at_same_N": 7.507800279664921
        },
        "depth": {
          "cost_ratio": 4.602979903556323,
          "work_ratio": 4.311480298123078,
          "ideal_N_work_ratio_at_unchanged_cost": 4.217168771769635,
          "fractional_cost_reduction_to_tie_at_same_N": 0.7680610994707939,
          "candidate_cost_threshold_at_same_N": 22.493822298163494
        }
      }
    }
  ],
  "A_whole_vs_A_native0": [
    {
      "epsilon": 0.05,
      "N_floor_under_same_policy": 7012,
      "N_ratio": 1.041249326871298,
      "allowed_cost_ratio_actual_N": 0.9603847745138602,
      "allowed_cost_ratio_ideal_N": 1.3241585852823732,
      "remaining_N_reduction": 0.27472072817542403,
      "metrics": {
        "rz": {
          "cost_ratio": 0.2371503878638873,
          "work_ratio": 0.24693268173053984,
          "ideal_N_work_ratio_at_unchanged_cost": 0.1790951555952157,
          "fractional_cost_reduction_to_tie_at_same_N": 0,
          "candidate_cost_threshold_at_same_N": 58.23125246983317
        },
        "cx": {
          "cost_ratio": 0.35142803504778497,
          "work_ratio": 0.36592420493720895,
          "ideal_N_work_ratio_at_unchanged_cost": 0.2653972408998458,
          "fractional_cost_reduction_to_tie_at_same_N": 0,
          "candidate_cost_threshold_at_same_N": 18.946682350500065
        },
        "depth": {
          "cost_ratio": 0.27564147338701506,
          "work_ratio": 0.2870114986220422,
          "ideal_N_work_ratio_at_unchanged_cost": 0.20816349072587506,
          "fractional_cost_reduction_to_tie_at_same_N": 0,
          "candidate_cost_threshold_at_same_N": 70.33190994712311
        }
      }
    },
    {
      "epsilon": 0.02,
      "N_floor_under_same_policy": 43821,
      "N_ratio": 1.0413038190221358,
      "allowed_cost_ratio_actual_N": 0.9603345169127266,
      "allowed_cost_ratio_ideal_N": 1.4072020264256864,
      "remaining_N_reduction": 0.3175574658942254,
      "metrics": {
        "rz": {
          "cost_ratio": 0.2371503878638873,
          "work_ratio": 0.24694560456524658,
          "ideal_N_work_ratio_at_unchanged_cost": 0.1685261841657894,
          "fractional_cost_reduction_to_tie_at_same_N": 0,
          "candidate_cost_threshold_at_same_N": 58.22820518801676
        },
        "cx": {
          "cost_ratio": 0.35142803504778497,
          "work_ratio": 0.36594335500670344,
          "ideal_N_work_ratio_at_unchanged_cost": 0.24973531052994377,
          "fractional_cost_reduction_to_tie_at_same_N": 0,
          "candidate_cost_threshold_at_same_N": 18.945690857475974
        },
        "depth": {
          "cost_ratio": 0.27564147338701506,
          "work_ratio": 0.28702651891878717,
          "ideal_N_work_ratio_at_unchanged_cost": 0.19587910492649618,
          "fractional_cost_reduction_to_tie_at_same_N": 0,
          "candidate_cost_threshold_at_same_N": 70.32822942951093
        }
      }
    }
  ]
}
```


## 記録の終わり

本レビューの完了は、現候補の主改善claimと次の研究判断についての評価が完了したことを意味する。新しい中心テーマの新規性、実用性、採択、次batch実行の承認までは意味しない。元のsource・結果・STOP分類は変更していない。
