# Hamiltonian表現・PR協調設計：初期探索結果の独立科学レビュー

作成日：2026年10月10日（日本時間）  
レビュー開始承認：ユーザーの「レビューを開始して」  
対象：`HIROMU1015/Partially-Randomized-Trotter`  
結果branch：`representation-exploration-20261010`  
結果固定commit：`39345830ddfe7c3e2a488c284a0623f489764087`  
run2 source：`25d7135f7a0285b6cf415349191b00c00acfb75f`  
レビュー状態：**完了。初期機構の理解と次の研究範囲を判断した。新規アルゴリズムの成立・総資源優位・論文化十分性は未確定。**

## 1. 結論

今回の初期検証は、A/B/Cの構成が何を保存し、どこで改善の期待が崩れるかを調べる検証として有用だった。確認したsource・保存値・検証範囲から、主要な恒等式の結論を覆す具体的な矛盾は見つからなかった。ただし、独立した実装による全結果再現や、全compiler出力の再認証を行ったという意味ではない。

研究方針としては、**AのHamiltonian分解構成とCの実装可能なblock構成を、次の限定探索の二方向として継続する。Bは平均振幅推定の方法として条件付きで残すが、同種toyの追加ではなく、実際の圧縮表現との接続条件を先に調べる。** 一つの中心テーマを現段階で確定する必要はない。

| 候補 | 今回の研究判断 | 次に問うこと |
|---|---|---|
| A：factor／Hamiltonian分解 | 継続。ただし単純な角度探索から、実装費用を含む具体的な分解構成へ進む | 安価な既知実装を対照にして、同じ精度の実PR資源を下げる構成を生成できるか |
| C：近可積分性・mixed block | 並行継続。既知THRIFTの再現ではなく、要求されるblockを作る問題に焦点を移す | 安価な混合propagatorが得られる分解を、どの入力クラスから構成できるか |
| B：補助空間と生成子対称化 | 条件付き保留。第一momentの構成としては正しいが、圧縮の利益がまだない | 真の圧縮factor／isometryと接続し、反射・正規化・準備費用を含めても利益を残せるか |

これはAの優位やBの不可能性を確定した判断ではない。**現在の証拠では、A/Cは次の構成問題を具体化しやすい一方、Bは圧縮という目的に必要な入力構造が未接続**である、という情報価値に基づく判断である。

前回の優先候補A/Bから、今回A/Cに次の検証の比重を移す理由は、新しく得た証拠にある。Bのtoyでは直接実装可能な物理Hamiltonianが得られ、圧縮primitiveの利益を全く試していない。A/Cでは、安価なblockと正確な残差を構成するという具体的な接点が現れた。ただしAとCの統合自体を義務にせず、それぞれ単独で改善できる場合も残す。

## 2. レビュー対象、取得範囲、証拠の境界

### 2.1 対象とした実行

基点は `b2e1bf65e21893b6c617223b42313623d3186f12`。run1は `451bfa523f548ad5e1563428474f62d44e7a3366` から起動し、compiler同値性検査でtechnical STOPした。run2は上記固定sourceから実行した別の結果である。run1を成功に変更せず、失敗と修正の来歴を保持している。[E1,E4,E7,E8]

結果は20 A rows、9 B rows、3 C rows、19 compiled circuits、24,678 B events。すべてsynthetic exact-dataの開発診断で、分子の計算、ground-state solve、未知入力のheld-out検証、量子shotの実行ではない。[E1,E3,E4]

### 2.2 実際に読んだ資料

GitHub connectorで、初期報告と固定scope、`mechanisms.py`のA/B/C全実装、tests、保存値verifier、provenance、run1 failure、run2 audit、一次resultの主要行・保存行列を読んだ。Bの大きいeventファイルは、レビュー前の資料可用性確認でGit blob経由の全文取得に成功している。通常Contents経由の空本文は、未pushやデータ欠落ではなかった。

本レビューでは145 source blobsを全件再取得して再hashしたわけではない。全24,678 eventのunitaryを別実装で作り直し、各eventの証明を再実施したわけでもない。全19回路を独立したcompilerで再合成していない。保存監査が何を検査するかはverifier sourceで確認した。

また、コンテナからraw GitHubへ直接ダウンロードする経路はDNS接続に失敗した。GitHub connectorの必要資料読取は成功しており、これをリポジトリの資料不足とは判定していない。

### 2.3 GPTが今回独立に行った検討

固定sourceの入力を有理数行列として転記し、repository moduleをimportしないSymPy scriptで次を確認した。[D1]

- Aの3-mode入力が保存する占有数と、非自明なsectorの次元。
- Aの共有対角coreに対する残差の厳密な演算子式と固有値。
- Aのisotropic対照が単純なparity projectorに一致すること。
- Bの物理Hamiltonianが単一qubitの既知SU(2)回転へ還元されること。
- Cの固定THRIFT積の時間展開を3次まで求め、先頭誤差を同定すること。

これは固定toy入力に関する独立した代数チェックである。保存浮動小数点rowの全面再現、未知入力による実験、最適化、RTE event生成、量子回路compile、分子計算は行っていない。後述の固定射影coreの不変性は一般の代数として導出したもので、有限個のtoyだけから一般化した主張ではない。

## 3. 実行・保存検証の評価

### 3.1 実行が完了したことと、何を示したか

run2の保存auditはwall `3.8472497100010514 s`、CPU `3.860964 s`、peak RSS `447,684,608 bytes`、約427 MiBを記録する。これらはrunner内計測であり、環境構築・tests・文書化・監査を含む研究作業総時間ではない。[E4]

39 testsは23件の本探索testsと16件の既存RTE testsから成る。39個の独立した科学的機構を検証したという数え方はしない。74,389件の保存監査checkも、その多くはeventごとの確率・範囲・位相などの確認である。[E1,E5,E6,E8]

保存verifierはsource／出力hash、件数、event確率からのscalar集計、保存された補正平均と小行列Taylor多項式の一致を検査する。独立したTaylor再構成は意味のある追加照合だが、eventのラベルから全unitaryを別実装で復元する検査とは異なる。[E6]

### 3.2 compilerのglobal phase修正

run1はBのcontrolled回転を含む回路に対する絶対operator同値検査で停止した。run2では、built circuitとcompiled circuitの差が単位絶対値のscalarであると小行列で認証できた場合に限り、`compiled.global_phase`を補正している。非scalarな差や制御branch間の相対位相誤りを許容する修正ではなく、補正後と追加control後の同値検査を持つ。[E2,E5,E7]

この修正は小規模診断として解釈できる。native gate countはmetadata補正では変わらない。ただし、密行列の比較に依存するため、大規模回路で使えるtruth-free compiler修正の提案ではない。また原因をQiskit一般の不具合と断定する証拠にもならない。

保存cost recordでは`qasm`がnullである。現在のsource・seed・compiler条件は再構成の手掛かりになるが、将来の重要なnative資源比較では、具体的なgate列／IRとphase metadata、合成精度を保存する方がよい。**これは今回の代数的機構レビューを停止する理由ではなく、次の資源比較に対する改善点**である。

### 3.3 二つの警告

警告はqiskit-nature内のComplexWarningである。今回のorbital入力はrealであり、独立JW参照とDF builderの比較が通っているという限定を採用する。複素軌道へ適用範囲を広げる際には、この結果を流用せず別途検証する。[E5,E8]

## 4. A：factor回転の科学的意味

### 4.1 正しい構成自由度

正係数を吸収したHermitian因子を\(F_a\)、実直交行列を\(O\)とすると、

\[
G_\mu=\sum_aO_{\mu a}F_a,
\qquad
\sum_\mu G_\mu^2
=\sum_{ab}(O^TO)_{ab}F_aF_b
=\sum_aF_a^2.
\]

因子の可換性は不要である。\(F_a=d\Gamma(g_a)\)にも適用できる。正係数の平方根吸収をせず、変換後labelに元の不等係数をそのまま残す場合や、符号付きsumへ通常の直交混合を無条件に適用する場合は、この等式の前提を満たさない。[E1,E2]

この操作は、同じ全Hamiltonianの異なるfragment分解を作る。決定論側と乱択側に異なる役割を割り当てる限り、単に同じPFを座標変換しただけではない。一方、すべてのblockを同一のunitaryで一様に共役した場合は、対応するPFも共役され、unitarily invariantな誤差は変わらない。

また、可逆混合はfactor spanと独立rankを保つ。Gramの上位固有値選択で既に最適なFrobenius proxyを、同じ目的の直交混合でさらに改善できるとは限らない。すべてのfirst-level factorを可換化できるなら、逆混合によって元のfactorも可換になるため、非可換なspan全体をこの操作だけで可換化することもできない。ただし因子の平方、選択subset、別の非線形構成は区別する。

### 4.2 保存値から言えること

以下のcostは、\(\delta=.2\)における**controlled exact-tail S2 wrapper**の実装依存の値である。実RTE trajectoryや同一目標精度での総費用ではない。[E1,E2,E3]

| 入力・角度 | identity抽出後tail \(\lambda\) | S2 operator誤差 | RZ / CX |
|---|---:|---:|---:|
| review squares、0 | 2 | 0.0232099947101 | 206 / 150 |
| review squares、\(\pi/4\) | 0.5 | 約\(5.66\times10^{-16}\) | 381 / 282 |
| proxy mismatch、0 | 約1.480189 | 約\(6.9459\times10^{-5}\) | 391 / 280 |
| proxy mismatch、\(\arctan(.1)\) | 約1.388289 | 約\(1.0028\times10^{-4}\) | 762 / 544 |
| proxy mismatch、\(-\pi/8\) | 1.36186443846 | \(3.87536229555\times10^{-5}\) | 未compile |
| proxy mismatch、\(+\pi/8\) | 1.75324004539 | \(1.75527509539\times10^{-4}\) | 未compile |

\(\arctan(.1)\)では抽出後tail係数が約6.21%減る一方、PF誤差は約44.4%増え、保存された回路費用も増えた。したがって「tail係数だけを下げればよい」という単一proxyへの反例として有効である。

ただし、identityを含むfaithful係数は約2.255989から2.235616への変化で、改善率は約0.903%である。identityを正しく位相として分離する方針自体は許容できるが、比較する全armで同じ会計を用い、controlled phaseの実装を無視しない必要がある。

一方、\(-\pi/8\)ではtailとPF誤差の両方が改善している。これを無視して「factor回転は悪化する」と結論するのも不適切である。この点のcompileがない以上、資源勝者は未確定である。\(\pi/4\)のFrobenius weight tieも、丸めによるargmaxの選択を科学的な優位と解釈しない。

さらに、\(-\pi/8\)から\(+\pi/8\)へ移ると、保存commutator normは約0.27266から0.22514へ下がるが、S2誤差は上がる。これも単純な\(\|[D,R]\|\)だけで、S2のnested-commutator誤差を評価できないことを示す固定例である。[E3]

### 4.3 回路費用の悪化を過大解釈しない

isotropic対照では各squareが不変なのに、frameを通す実装によってcountが増えている。今回の入力について独立に計算すると、全Hamiltonianは

\[
H=\frac{I-Z_0Z_1}{2}
\]

に一致する。[D1]

従って、構造を認識すれば既知のdiagonal実装を使える。保存された大きいcountを、そのoperatorに必要な最小費用と扱ってはいけない。逆に、この既知簡約を導入して得た削減を、そのまま新しいfactorization算法の成果と扱うのも不適切である。

またsourceでは、DF回路全体をgate化してcontrolを付けている。一般に

\[
\mathrm{C}[V^\dagger e^{-itD}V]
=(I\otimes V^\dagger)\,\mathrm{C}[e^{-itD}]\,(I\otimes V)
\]

であり、basis変換を無制御で置ける。compilerがどこまでこの構造を回収したかを仮定せず、明示的なstructured implementationも対照にする。この簡約やbasis rotationの連結は既知であり、それ単独を新規性にはしない。[L1]

### 4.4 追加導出：3-mode入力は条件付き二準位問題へ分解できる

sourceのproxy mismatchを二次量子化すると、

\[
F_0=n_0+n_1-n_2,
\qquad
F_1=.1n_0-.8n_1-.7n_2+.04T,
\qquad T=a_0^\dagger a_2+a_2^\dagger a_0.
\]

ここで\(H=F_0^2+F_1^2\)は、

\[
[H,n_1]=0,\qquad [H,n_0+n_2]=0
\]

を満たす。非自明な\(n_0+n_2=1\)のsectorは、\(n_1=0,1\)のそれぞれについて二次元である。この等式とsector次元を有理数行列で確認した。[D1]

従って、この例には占有数で条件分けした既知SU(2)構成という強い対照がある。ただし、基底の条件付けやfermionic sign、制御の回路費用まで無料になるわけではない。必要なのは、密行列指数を無料oracleにすることではなく、この**明示構造を利用した回路**との比較である。

このtoyはproxy不一致の反例として適切である。しかし、このtoyだけで量子化学の難しいHamiltonianに対する資源上の利点を示すことはできない。次は同じ簡約で全問題が終わってしまわない入力へも、構成が適用できるかを見るべきである。

### 4.5 追加導出：固定共通射影coreはfactor回転で変わらない

ここは今後の探索で重要な注意点である。

固定したorbital frame \(V\)で対角成分を取る線形写像を\(\mathcal D_V\)とし、すべての因子から

\[
H_{\mathrm{core}}(V,O)
=\sum_\mu\left[d\Gamma\left(\mathcal D_V\left(\sum_aO_{\mu a}g_a\right)\right)\right]^2
\]

を作る。線形性と直交性から、

\[
H_{\mathrm{core}}(V,O)
=\sum_a[d\Gamma(\mathcal D_V(g_a))]^2
\]

となり、**このcore operatorは\(O\)に依存しない**。全\(H\)も不変なので、残差operatorも同じである。

従って、固定frameですべての因子を同じ線形規則で射影しながら、factor回転だけでcoreと残差のoperator normや対応するexact-block PF誤差を改善しようとする探索は、目的に対して無効である。

ただし、これは「同じoperatorの別の実装に価値はない」という結論ではない。未統合辞書、別basisでのinvolution表現、回路合成を変えれば、operator不変でも費用は変わり得る。その場合、改善はcore/residualのoperator変更ではなく**表現・実装変更**に由来することを明示する。

また、\(V\)自体を選ぶ、因子subsetを選ぶ、複数frameへ配分する、supportやblock構造を選ぶなど、上の前提を外す構成は除外されない。この不変性は探索の不要な自由度を除く簡単な補題であり、それ単独で独立論文の新規定理だとは主張しない。

### 4.6 core＋正確残差への発展と限界

固定入力では、\(D=.1n_0-.8n_1-.7n_2\)、\(H_{\rm core}=F_0^2+D^2\)に対する残差は、厳密に

\[
R=(-.024-.064n_1)T+.0016(n_0+n_2-2n_0n_2)
\]

である。固有値は0が4個、\(16/625,-14/625,56/625,-54/625\)で、operator normは\(56/625=.0896\)となる。保存値と整合する固定入力の独立な代数確認である。[D1]

この式は、近似coreを作った後に残差を単なる大きいoperator同士の差で渡すのではなく、hoppingとoccupancyの構造で明示できることを示す。ただしこの小例が容易なのは上記の保存量にも依存する。

一般に\(\widehat F_a=\widehat D_a+\widehat E_a\)なら、

\[
H=H_1+\sum_a\widehat D_a^2+
\sum_a(\{\widehat D_a,\widehat E_a\}+\widehat E_a^2).
\]

残差を正確に保持する構成は可能だが、\(\|R\|\)が小さいことと、実装する辞書の係数1ノルムが小さいことは別である。同一の明示Pauli基底における係数1ノルムを\(\Lambda\)と定義すれば、三角不等式と劣乗法性から

\[
\Lambda(R)\le\sum_a\left(2\Lambda(\widehat D_a)\Lambda(\widehat E_a)+\Lambda(\widehat E_a)^2\right)
\]

という粗い上界を得る。しかし、この上界が小さいか、実用的に計算できるか、frame変換後に安価なnative primitiveを持つかは別問題である。任意の異なる辞書の\(\lambda\)を混ぜてこの式を使ってはいけない。

また、全factorの対角squareを同一frameで足すと、占有数の二次多項式という既知の可換coreになる。それだけで新規性はない。狙うべきものは、**どのframe／support／blockを選び、残差をどの安価な辞書で返せば、既知の単純分割より有利になるかを決める構成法**である。

## 5. B：補正平均の保存と圧縮の利益を区別する

### 5.1 第一momentについての構成は成立する

\(P\)を既知の物理部分空間の射影、\(Q=I-P\)、\(S=2P-I\)とする。

\[
\overline H=\frac12(\widetilde H+S\widetilde HS)
=P\widetilde HP+Q\widetilde HQ.
\]

従って\(Q\overline HP=0\)で、\(P\overline HP=P\widetilde HP\)である。後者が目的の物理Hamiltonianと正しく対応している必要はあるが、未知のground-state projectorを要求する構成ではない。

involution辞書を元項とS共役項へ半分ずつ分配すると、未統合の係数1ノルムを増やさずに\(\overline H\)を表現できる。各Taylor factorで正しく独立な選択を使うpaired finite RTEなら、

\[
B_K\,\mathbb E[U]=T_{K+1}(-it\overline H)
\]

である。右辺が\(P\)を保つため、補正平均のleakage blockは消える。[E2,E3]

この結果は、前回の構成案と整合する。既知の群平均やLCUの代数を確認したのであって、この等式を初めて発見したことを意味しない。

### 5.2 個々の軌道や平均channelは保護されない

初期状態\(\psi\in P\mathcal H\)について、平均channelの漏洩確率は

\[
\operatorname{Tr}\!\left[Q\sum_\omega p_\omega U_\omega|\psi\rangle\langle\psi|U_\omega^\dagger\right]
=\sum_\omega p_\omega\|QU_\omega\psi\|^2.
\]

右辺は非負の和であり、第一momentの複素振幅のように相殺されない。従って、\(Q\mathbb E[U]P=0\)と、各trajectoryが\(P\)を保つことには明確な違いがある。

これは平均振幅推定としてのBを否定しない。Hadamard test等の信号は第一momentと接続できる。一方、正しい物理状態を一本の量子回路の出力として得る保証や、channel距離の保証へ読み替えることはできない。reset・postselectionを追加した場合も、推定対象と係数補正を再導出する必要がある。

決定論blockを含むPRへ接続する際、各決定論blockが\(P\)を保つことは、全体保存を示す扱いやすい**十分条件**である。一般に必要条件とは限らず、完成した積全体の相殺等は別途あり得る。今回の反例が示すのは「blockの和が保存されるだけでは、任意のPF積の保存は保証されない」ということまでである。

### 5.3 保存値の解釈

\(t=.2\)の保存値は次のとおり。[E1,E3]

| 指標 | K=0 | K=2 | K=4 |
|---|---:|---:|---:|
| 補正係数B | 約1.033247 | 約1.067174 | 約1.067365 |
| 補正平均の漏洩norm | 0 | 約\(5.80\times10^{-20}\) | 約\(4.36\times10^{-20}\) |
| 平均channelの漏洩確率 | 約0.019483 | 約0.032419 | 約0.032500 |
| 個々のeventの最大漏洩確率 | 約0.063320 | 1 | 1 |
| 有限Taylorのphysical bias | 約0.010594 | 約\(1.8721\times10^{-5}\) | 約\(1.3231\times10^{-8}\) |

Kを増やすことはphysical Taylor biasを改善しても、各trajectoryの漏洩を消さない。最大漏洩1の存在は、全trajectoryが必ず漏れることも意味しない。

N=1024のRMSは列挙した二次momentからの予測であり、量子測定やMonte Carloの実測ではない。診断shot数もTaylor biasを精度予算へ戻しておらず、QPE/RPE全roundの総費用ではない。

### 5.4 このtoyでは直接物理Hamiltonianを使える

本入力ではS共役によって\(.4X_{aux}X_{sys}\)が相殺し、物理空間では

\[
H_{\rm phys}=.7Z+.2X,
\qquad H_{\rm phys}^2=.53I
\]

となる。[D1]

従って、

\[
e^{-itH_{\rm phys}}
=\cos(\sqrt{.53}t)I
-i\frac{\sin(\sqrt{.53}t)}{\sqrt{.53}}H_{\rm phys}
\]

という既知の一qubit回転で実装できる。auxiliary qubitや反射RTEを使うことは、この入力の時間発展に必要ではない。

未統合辞書の\(\lambda=1.3\)が増えないことは、比較対象に対する優位ではない。相殺後の辞書なら\(\lambda=.9\)で、さらに上の直接回転もある。したがって、このtoyから「圧縮＋反射RTEが有利」とは言えない。

同時に、これだけで、実際の圧縮factorを使うBが無価値だとも結論できない。現在は、その圧縮構造がそもそも入力に含まれていない。

### 5.5 Bを残す条件

次の問いは、真のenlarged-space factorization／isometryから出発し、物理encoding、実装primitive、補助空間からの復帰、coherent signalの取り方まで具体化できるかである。反射単体のgate数を測るだけでは不足する。

比較には、同じ最終taskを満たす直接block-preserving構成、簡約したHamiltonianへのRTE、適切に接続した既知のreset／echo／symmetry保護等を考える。ただし、channel保護法の論文の費用を、そのまま第一moment推定の費用として流用してはいけない。

isometric THCやGRADEは、補助空間とその誤差処理まで扱う近接研究である。[L3,L4] Bの群平均を再提示するだけでは足りない。既存の圧縮法で高価だった処理を、同じtaskに対して別の費用構造で置き換えられることが必要である。

Bに追加計算を投入するなら、この具体的なoracle／encoding構成の確認に限定する。同種Pauli toyでK・時間・aux数を増やして第一moment保存を再確認する計算の優先度は低い。

## 6. C：THRIFTの再現から実装可能な構造の生成へ

### 6.1 既知機構の再現は正しく位置付ける

sourceは

\[
A=Z_0+.7Z_1+.3Z_0Z_1,\quad B_0=X_0,\quad B_1=X_1
\]

に対し、

\[
U_T(t)=e^{-it(A+\alpha B_0)}e^{itA}e^{-it(A+\alpha B_1)}
\]

を構成している。各mixed propagatorはspectatorの二つのZ値に応じたSU(2)回転で実装し、密行列対角化を無料primitiveとして数えていない。[E2]

\(t=.2\)、\(\alpha=.05,.1,.2\)でTHRIFT誤差は約\(3.98483\times10^{-6},1.59388\times10^{-5},6.37480\times10^{-5}\)。これは既知THRIFTの\(\alpha^2\)機構と整合する範囲である。混合propagatorを通常の分割で置換すると通常一次PFに戻るという代数も正しい。[E3,L5]

### 6.2 追加導出：このtoyには時間次数の特別な相殺がある

この入力は\([B_0,B_1]=0\)を満たす。固定積順序に従って3次まで時間展開すると、

\[
U_T(t)-e^{-it(A+\alpha(B_0+B_1))}
=-\frac{i}{5}\alpha^2t^3Y_0Y_1+O(t^4)
\]

となる（\(\alpha\)を固定した\(t\to0\)の展開）。0次から2次の差はゼロである。[D1]

\(\alpha=.1,t=.2\)の先頭normは\(1.6\times10^{-5}\)で、保存誤差\(1.59388446779\times10^{-5}\)と整合する。ただし、この一致を任意の時間での厳密上界と呼ばない。

従って、このtoyは\(\alpha\)方向の改善だけでなく、time-step方向にも特殊な相殺がある。通常一次PFだけを対照にすると、一般的なアルゴリズムの優位を過大に読み取りやすい。対称二次PF等、同程度の時間次数を持つ簡単な対照も必要である。

これはCの結果が誤っているという指摘ではない。**既知機構をどの条件で確認したのかを、より正確にした**ものである。

### 6.3 安い一歩と安い最終計算は違う

\(\alpha=.1\)の保存countはTHRIFTがRZ294/CX210、通常一次PFがRZ37/CX22である。しかし両者の誤差は揃っていない。[E3]

THRIFTの一歩が高価だから負けとも、同じtで誤差が小さいから勝ちとも判定できない。所定精度を満たす反復数、boundary cancellation、制御、合成を同じ会計で比較する必要がある。THRIFTの元論文自体も、混合項の実装費用が方法の実用性を左右することを扱っている。[L5]

### 6.4 研究対象はmixed oracleを作る方へ置く

例えばdiagonal Ising型coreに\(X_i\)を加えたとき、effective detuningがd個のoccupancy／Z値にだけ依存すれば、単純な構成は\(2^d\)枝の条件付きSU(2)になる。しかしdが系サイズとともに増えると、この列挙は高価になる。算術回路等で枝列挙を避ける可能性はあるが、その精度・算術・制御費用を数える必要がある。指数的枝列挙の存在から、すべての別実装が不可能と推論しない。

また、全Hamiltonian生成子のLie closureが大きくても、個々のmixed oracleは簡単な場合がある。逆に、数値的に小さいclosureが見つかっただけでは、対角化／basis変換／controlを安価に実装できるとは限らない。

従ってCの新しい課題は、**近可積分と書けるかではなく、利用したいsimulation法が要求する混合blockを、費用付きで構成できる分解を生成すること**である。条件付きLie-algebra構成そのものにも先行研究がある。[L6] THRIFTの係数や内部compositionだけを調整する方向へ移るのは、今回の対象ではない。

## 7. 先行研究との距離と新規性の判定

本レビューは近接する一次資料を中心にした調査である。全関連文献を漏れなく調べたという新規性認定ではない。特に次の重複を避ける必要がある。

| 近接研究 | 今回の提案に対する意味 |
|---|---|
| CDF [L1] | factor圧縮、orbital変換、時間発展の回路簡約は既知。既知compiler簡約の再発見だけでは新しい分解算法にならない |
| RC-DF [L2] | 資源関連の係数normを意識したfactor fittingは既知。新しい目的関数を置いたことだけで差分は確定しない |
| SPRINT/GRADE [L3] | near-integrability、randomization、symmetry protection、factorizationを横断する設計が既にある。『core＋乱択残差＋保護』という大枠だけでは新規性にならない |
| Enlarged-basis／isometric THC [L4] | 圧縮と補助空間誤差の扱いが先行している。Bは既知reset／保護との差を同じtaskで示す必要がある |
| THRIFT [L5] | 小さいperturbationの誤差特性とmixed evolutionを利用する方法は既知。Cはそれを使える分解を作る側に貢献を置く |
| Symmetry-extended solvable Hamiltonians [L6] | 保存量で条件付けした可解blockは既知。条件付きSU(2)を作れたというだけでは独自性が不十分 |
| PR原論文 [L7] | 決定論／乱択分担やHamiltonian表現との接続は既知の土台。新研究は内部RTEの再最適化だけに戻らない |

### 7.1 SPRINT/GRADEを特に重視する理由

2026年6月29日公開のSPRINT/GRADEは、Hamiltonian factorization、近可積分構造、randomization、symmetry protection等を同時に扱う。本文と手法概念図から、今回の広い構想にかなり近いことが確認できる。[L3]

従って研究課題を『DFとPRを組み合わせる』『圧縮と保護を併用する』と表すだけでは差が薄い。特定の入力に対して、既存の構成法では得られなかったblock／残差／実装を生成する、またはその費用の問題を解消する必要がある。

一方、SPRINTがすべてのHamiltonian表現・oracle構成を解決したという意味ではない。近接研究があることを理由に研究を終了せず、入力、許されるprimitive、誤差保証、古典取得費用、最終taskのどこに未解決の差があるかを具体化する。

### 7.2 新規性に必要な強さを過剰に設定しない

一般定理や全分子への一様優位を必須条件にはしない。具体的な新しい構成法があり、その仕組み、既存法との差、成立条件、妥当な範囲での資源効果を説得的に示せれば、アルゴリズム研究として成立し得る。

逆に、今回の短い恒等式、既知のTHRIFTスケーリングの再現、数個のtoyでの失敗だけを主要貢献として論文化できると判断する根拠はない。現状はその前段階である。

## 8. 次の研究課題の組み立て直し

### 8.1 Aを中心にした問い

> 与えられたHamiltonianから、安価なbasis変換で実装できる決定論blockと、安価な辞書で表せる正確な残差を生成し、同じ推定精度のPR資源を減らせるか。

固定factorへのラベル回転だけに限定しない。frame選択、subset選択、support制約、共通または近接basisへの配分、diagonal／条件付き可解blockへの再構成を候補にできる。

ただし、最初から全部を同時最適化する必要はない。最初の試作品は、入力から出力block・残差・回路情報を返す小さい構成法でよい。目的はangle gridの最良rowを拾うことではなく、次の入力にも適用できる手順を作ることである。

### 8.2 Cを中心にした問い

> 安価なmixed propagatorの条件を満たすHamiltonian分解を、元の相互作用構造から生成できるか。

Aと独立に進めてもよい。例えば、supportや条件分岐の複雑さに制限を置いた可解blockの構成と、その制限のために残る項の費用を比べる。単なるglobal Lie dimensionや\(\alpha\)の最小化ではなく、返された各blockを実際に実装できることを評価する。

### 8.3 A/Cの接点は候補であって必須ではない

Codex報告にも「bounded-supportの可解core＋正確残差」という追加候補が記載されている。[E1] 今回のレビューは、この案を自動採択せず、Aの残差表現とCの混合oracle費用を接続する一つの構成候補として評価する。

共通の問いは、**どれだけ多くの項をcoreへ入れるかではなく、残すblockを本当に安価に実行でき、そのために生じる残差の負担が小さいか**である。外部だけで良い結果が得られるなら、それをPR内部の変更と無理に結合しない。

### 8.4 Bの問いは圧縮利益に直結させる

> 具体的な圧縮primitiveが、物理部分空間の第一momentを正しく保つ構成と両立し、直接実装／既知保護法に比べて安いか。

この問いに必要なcompressed factorを作れない場合、その理由を『Bの定理が失敗した』ではなく『実用的な入力構成が未接続』として扱う。構成できるなら、小さい実装可能な例で比較する。圧縮の規模を上げることを先行しない。

## 9. 次のCodex作業：限定した構成・比較を一つにまとめる

次の担当はCodexでよい。本レビューで科学的な目的と比較上の注意を整理したため、同じ結果のレビューを直ちに繰り返す必要はない。実装の細かな手順、toy／小規模入力の選択、tests、計算上限の具体値はCodexに任せる。

### Work package A：構成を一段具体化し、実PRの小さい比較を閉じる

既存toyはcorrectness／反例の回帰として保存する。回転だけで変わらない目的を除き、実装しやすいblockと正確残差を生成する具体的な手順を少なくとも明示する。元factorの安い構造化実装をbaselineとし、basis無制御化・既知相殺・identity phase等は新旧へ公平に適用する。

小規模な範囲で、構成されたD/R、有限RTEのcorrected mean、bias、正規化、controlled circuit cost、同じ信号精度に必要なshotを接続する。現在のexact-tail S2 countにB²を機械的に掛けるだけでは、実RTE比較にならない。

入力をすべて同じ既知二準位簡約で解けるものに限定しない。ただし、大きな分子ベンチマークを始める必要はない。新しい適用入力の選択理由と、使った構造を記録する。

### Work package C：適切な既知対照と費用を揃える

混合oracleが明示的に実装できる小さい対象で、通常一次PFだけでなく対称二次PF等を同じ精度で比較する。反復境界、basis merge、制御、合成を同じ条件で数える。

次に、そのmixed oracleを構成できる条件が入力サイズとともにどう変わるかを調べる。単に\(2^d\)枝を列挙できた例だけで一般的な効率性を主張しない。手法の改善箇所は内部PF係数ではなく、Hamiltonianの構成・access側に置く。

### Work package B：小さい接続確認に留める

実際のenlarged-basis／isometric factorを使い、入力encodingと目標physical Hamiltonianの対応、primitive、reflection、測定task、近接baselineを具体化する。小さい構成で確認可能なら行ってよいが、圧縮を伴わない同型toyの再列挙を主成果にしない。

接続が不可能と確定していない場合は、未定義のprimitiveや未取得のcostを列挙して止める。無理に肯定結果を作らず、A/Cの進行もこの準備だけのために止めない。

### 共通の成果物

各候補について、入力から出力への算法、必要なoracle、既知baseline、同一精度比較の成立範囲、古典計算費用、原結果・失敗・sourceの対応を保存する。最良点だけでなく全選定候補と除外理由を残す。新規性や中心仮説の最終採択はCodexが代行しない。

技術的な修正はこの範囲内でまとめて進め、機構の判定や次の大きい科学段階が必要になった時点でGPTへ戻す。一つのbatch内の各小修正を独立承認で分断する必要はない。

## 10. 精度と資源を比較する際の最小限の契約

### 10.1 目的taskを揃える

coherent state出力、channel近似、第一momentの振幅推定、energy／phase推定を混同しない。Bを評価するときは特に重要である。

初期の機構比較では、固定時間の複素信号精度でもよい。最初から全QPE/RPE roundと化学的精度を接続する必要はない。ただし、その場合は局所的なsignal taskの結果と明記し、最終energy予算へ外挿しない。

### 10.2 biasと正規化を分離する

一軸の単純なHadamard推定例で、補正後sampleが\([-B_{\rm tot},B_{\rm tot}]\)、同じ信号に対する系統bias上限が\(b\)、目標誤差が\(\epsilon>b\)なら、Hoeffdingによる十分条件の一例は

\[
N\ge \frac{2B_{\rm tot}^2}{(\epsilon-b)^2}\log\frac{2}{\delta}.
\]

これは推定対象、range、confidenceを指定した一軸の説明用条件であり、任意のRTE方式や長RPEに無条件適用する資源定理ではない。X/Yやround間では失敗確率配分を整合させる。

重要なのは、tail係数や一回路countが改善しても、biasのために残る\(\epsilon-b\)が小さくなるとshot負担が増え得ることである。既知の厳密上界、経験的な誤差見積り、exact-data診断を同じ『保証』と呼ばない。

### 10.3 会計の粒度を揃える

比較する量は、例えば

\[
G=\sum_m N_m\,\mathbb E[C_m]
\]

である。state preparationを含める／除く、control、identity phase、basis変換、反射、rotation synthesis、測定の扱いを明記する。

RZ/CX/count/depth/論理T/Toffoli/qubit数を任意の係数で一つに足さない。まず多資源の結果を示す。特定hardwareでの重み付けを使う場合は、そのモデルを別に定義する。小さなRZ数の改善をfault-tolerant資源改善と自動的に読み替えない。

### 10.4 exact oracleを開発と運用で分ける

小系の正確なmatrix normや厳密解は診断に使える。しかし、それを大量に呼んで候補を選ぶ手順を、そのまま大系向けの古典前処理アルゴリズムと主張しない。oracle-aidedな探索結果と、Hamiltonianの利用可能な情報だけで選ぶ構成を分ける。

大規模held-out campaignは今すぐ必須ではないが、開発toyの最良点だけに依存しない適用確認は必要になる。分子・basis・resource endpointは研究課題に応じて選び、過去のSTO-3G／水素鎖／RZを自動固定しない。

## 11. 継続・修正・終了を判断する基準

| 状況 | 次の研究判断 |
|---|---|
| Aの安い既知実装を入れても、同じ精度の実PRで改善が残り、入力から構成する手順が明確 | Aの構成法を深める。汎用性・古典費用・近接factorizationとの差を検証 |
| Aの改善が既知basis簡約だけに帰着 | 工学上の有用性は別に残すが、新規算法のclaimを縮小。分解構成を修正 |
| Cが対称PF等に対して有効で、mixed accessの生成が小さい費用に保たれる | C単独またはAとの接続を深める |
| Cの改善が通常一次PFとの不公平な比較だけで消える | THRIFT適用の再現として整理し、新しい分解生成がない限り主候補から下げる |
| Bで真の圧縮primitiveと第一moment保存が両立し、同taskの直接／保護baselineを上回る余地がある | Bの実資源検証へ進む |
| Bが同種toyの直接簡約に留まる、または圧縮利益が表せない | 数値拡張を停止。oracle構成が変わるまで保留 |
| どの案も既知手法の適用・簡約に還元される | 成果を無理に新規算法化せず、別の構成自由度・対象taskも探索する |

上記は一律の改善率閾値ではない。微小な数値差だけで優先順位を決めず、効果の原因、比較の公正さ、再利用できる算法の存在を重視する。

## 12. 論文化の着地点

### 12.1 主に目指す成果

有望な着地点は、**Hamiltonianから、実行可能なblockと残差、または圧縮と物理信号保存を両立するprimitiveを生成する算法**である。入力・出力・処理手順・古典費用・誤差条件が明確で、既知baselineでは得られない構造または費用の改善を示すことを目指す。

すべての分子で勝つ必要も、PRだけに有効である必要もない。適切な対象クラスでの新しい構成と、なぜ効くかの説明があればよい。内部のsampling lawを変更せず、外部表現だけで価値が出る成果も対象である。

### 12.2 別の着地点

構成の限界や、どの自由度が有効かを分類する理論・方法論研究も考えられる。ただし、今回の単純な共役不変性／射影不変性やtoyでの失敗をそのまま独立論文の十分な主要成果とはしない。より広い入力クラスや設計判断を説明する非自明な知見が必要である。

資源評価・適用条件の研究へ収束する場合は、それ自体の価値と、当初目指した新規算法研究との違いを明示する。Track Aと重複するベンチマーク研究へ無言で置き換えない。

## 13. Track A／Track Bとの境界

Track Aの資源会計、native DF、control／basis変換の実装は参考にできる。ただし、旧sourceの費用を新compiler系列の費用へ混ぜたり、別研究の入力受理やSTOPを変更したりしない。

Track Bの有限operator、normalization、native primitive、shot会計は接続時の参考になる。一方、この新研究の次段でreturn aggregationやsampling lawだけを再最適化することは目的にしない。外部Hamiltonian構成のために内部側の変更が必要になった場合は、変更と貢献の位置を明示する。

今回のレビューは両Trackの最新結果を再審査するものではない。過去の外部reviewや実行許可を本系列へ流用していない。

## 14. 資料・版・再現性の限界

レビューに必要な主要repository資料は参照可能であり、追加pushを要求する状態ではない。記載したsourceとresultを正本として扱う。最新branchが今後動いても、このレビューの結論は固定commitの範囲である。

SPRINT/GRADEはarXiv 2606.30741v1の本文・概念図を確認した。THRIFTは出版社HTML、isometric THCはv2 HTML、他の近接研究は一次論文のabstract／本文の取得できた範囲を使った。引用したすべての論文の全証明・supplementaryを全面再認証したわけではない。

PR原論文は公式出版情報とarXiv版情報を確認したが、取得PDFの表紙にv1と表示される一方でリンク先metadataがv2を示す取得経路があった。明示v2の全本文を独立に確認したとは主張しない。本レビューの有限RTE判定は、固定repository実装と保存された有限多項式の意味論に基づき、未確認のv2式番号へ依存していない。

今回の独立scriptは、固定source入力の有理数による解釈を用いる。元runのbinary64入力と同じbytesでの再実行ではなく、元のdecimal構成が意図するtoyに対する代数チェックである。floatでの丸め水準の差と、厳密な恒等式を混同しない。

## 15. 一次資料索引

以下のE番号は本レビューのrepository証拠、L番号は近接文献、D番号はGPTが今回行った独立計算を表す。

### Repository（結果commit 3934583で固定）

- [E1 初期報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/docs/research/representation_exploration_initial_validation_20261010.md)
- [E2 固定mechanisms.py（run2 source）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25d7135f7a0285b6cf415349191b00c00acfb75f/src/trottertracks/representation_exploration/mechanisms.py)
- [E3 run2一次result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/artifacts/representation_exploration/2026-10-10/run2/result.json)
- [E4 run2 audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/artifacts/representation_exploration/2026-10-10/run2/run_audit.json)
- [E5 新規tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/tests/test_representation_exploration.py)
- [E6 保存verifier](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/scripts/verify_representation_exploration.py)
- [E7 run1 failure](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/artifacts/representation_exploration/2026-10-10/run1/failure.json)
- [E8 provenance](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/artifacts/representation_exploration/2026-10-10/provenance.json)
- [E9 固定scope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/docs/research/representation_exploration_scope.md)
- [E10 B全event](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/artifacts/representation_exploration/2026-10-10/run2/b_events.json)
- [E11 保存監査結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/artifacts/representation_exploration/2026-10-10/saved_evidence_audit.json)
- [E12 remote取得記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39345830ddfe7c3e2a488c284a0623f489764087/artifacts/representation_exploration/2026-10-10/remote_retrieval_receipt.json)

主要保存identity：

| ファイル | bytes | SHA256 |
|---|---:|---|
| run2/result.json | 127242 | `52f8a3a2914c55c7cb06eb0796332ee6bd9776c5c42d349506da273103ef3562` |
| run2/b_events.json | 10270078 | `80e590ef38368379d95fede4072eb4932f5e0df1b761831fc162c8e8e6fedde2` |

上記は保存auditで宣言されたidentityであり、本レビューが全ファイルをコンテナへ取得して再hashしたという意味ではない。

### 近接する一次文献

- L1. Cohn, Motta, Parrish. *Quantum Filter Diagonalization with Compressed Double-Factorized Hamiltonians*. PRX Quantum **2**, 040352 (2021). [DOI](https://doi.org/10.1103/PRXQuantum.2.040352)
- L2. Oumarou et al. *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*. Quantum **8**, 1371 (2024). [Journal](https://quantum-journal.org/papers/q-2024-06-13-1371/)；[arXiv v3](https://arxiv.org/abs/2212.07957v3)
- L3. Casares et al. *Theory and practice of Trotter product formulas for quantum chemistry*. arXiv:**2606.30741v1** (29 June 2026). [Abstract](https://arxiv.org/abs/2606.30741v1)；[PDF](https://arxiv.org/pdf/2606.30741)
- L4. Luo, Cirac. *Efficient simulation of quantum chemistry problems in an enlarged basis set*. arXiv:**2407.04432v2**；PRX Quantum **6**, 010355 (2025). [v2 HTML](https://arxiv.org/html/2407.04432v2)
- L5. Bosse et al. *Efficient and practical Hamiltonian simulation from time-dependent product formulas*. Nature Communications **16**, 2673 (26 March 2025). [Journal](https://www.nature.com/articles/s41467-025-57580-5)
- L6. Patel, Yen, Izmaylov. *Extension of Exactly-Solvable Hamiltonians Using Symmetries of Lie Algebras*. Journal of Physical Chemistry A **128**, 4150–4159 (2024). [DOI](https://doi.org/10.1021/acs.jpca.4c00993)；[arXiv](https://arxiv.org/abs/2305.18251)
- L7. Günther et al. *Phase estimation with partially randomized time evolution*. [arXiv 2503.05647](https://arxiv.org/abs/2503.05647)；[publication DOI](https://doi.org/10.1103/ynxb-p2xq)。本文版の取得限界は第14節を参照。

### 独立計算

D1. `representation_initial_review_20261010/independent_review_checks.py` と同名JSON。SymPyによる固定入力の有理数行列・時間展開チェック。repository module、runner、合成、sampling、LP、分子計算を使わない。実行方法は同梱READMEを参照。

## 16. 最終判断の要約

初期探索としては、正しい機構と重要な制約が得られた。現状を『新しいアルゴリズムが成立した』とも『有望な方向がなくなった』とも評価しない。

**次は、安価な既知実装と同じ精度で比較できる具体的なHamiltonian構成を作る段階である。A/Cを並行して進め、Bは真の圧縮primitiveへの接続に絞って残す。研究の中心テーマは、その構成と比較から得られる証拠を見て改めて選ぶ。**

本レビューは過去のresult／STOP／authorizationを変更せず、新しい実験を実行していない。次のCodex作業には、本書で示した限定範囲と既存の保全規則を渡す。重要な機構・構成・資源結果が得られた後、必要資料をGitHubで確認して次の独立科学レビューへ戻る。
