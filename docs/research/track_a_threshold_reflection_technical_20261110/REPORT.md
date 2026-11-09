# 元Track A・仮説I：有限格子しきい値反射の技術実装と構造比較

2026-11-10 JST。状態：**人工整数・mock検査PASS／科学的費用・新規性UNASSESSED**。

仮説Iは可逆gate列として具体化できる。ただし、同じ演算子とidentityを保つ単一rank-one DF平方では、
既存I/Z/ZZ辞書との係数mass低下はない。scalarを外出しした場合に残る辞書差、event単価、shot、
強い決定論方式との総費用差は未評価。独立した新しいPR改善機構を認定せず、**現案は費用検証へ
自動進行せず保留し、実装を比較primitiveとして保持する**という技術的提案を返す。正式な採否はユーザー／GPT担当。

## 1. 範囲・配置・固定資料

- ユーザーの今回の指示は、仮説Iの設計、最小実装、科学実行を伴わない人工テスト、静的構造比較、保存・公開。
- 理論入力はユーザー追加の `track_a_new_algorithm_hypotheses_20261110.md` §4。元checkoutの590行を前回読了し、今回§4を再参照。
  この入力MDは基点commitに存在しないローカル提供資料であり、固定Git SOURCEを取得したとは表現しない。
  必要な実装仕様は以下に自足的に記載する。II／IIIは今回実装・実行しない。
- 実装基点は [2e79d80dc9cce2885580e50fe6f1705da8fa41af][base]。
  関連既存branchを確認したが、同目的のしきい値実装branchは確認できなかった。
  最も近い横断調査branchは静的文書用であり、独立した実装と固定成果を分離するため
  `codex/track-a-threshold-reflection-tech-20261110` と専用worktreeを作成した。元checkoutの未commit変更は持ち込まない。
- 正式共通指示2026-11-09版の原文は、この環境の既知path・関連Git検索で未取得。
  読了したとは扱わない。今回のユーザー指示と提供AGENTSの明示範囲を適用し、統合・削除・rebase・force pushは行わない。
- 新規科学計算、保存科学値の再解析、分子load、Hamiltonian／statevector行列、signal計算、量子simulation、
  native科学回路build／compile、synthesis、価格MC、旧runner再実行は行っていない。
  科学SOURCE・input・raw・results・grant・ledger・marker・既存STOPは変更しない。

実装：[df_threshold_reflection.py](../../../src/trottertracks/df_threshold_reflection.py)。
テスト：[test_df_threshold_reflection.py](../../../tests/test_df_threshold_reflection.py)。
科学runnerや結果artifactは作らない。人工PASSを科学validation manifestへ登録しない。

## 2. 再利用と新規部分

| 既存資産 | 適合性と今回の扱い |
|---|---|
| [rte.py][rte] | 既存 `finite_rte_distribution`、`RTEComponent`、`RTEEventApplication`、`RTEEvent`、basis-run規約を再利用。旧ファイルは不変更 |
| `normalize_involutory_tail` | dense行列と全atom列挙を要求するため、巨大な格子辞書には不適合。整数predicateでinvolutionを保証し、この関数を無理に呼ばない |
| [df_rte_tail.py][tail] | DFBlockのeta/lam、I/Z/ZZ式、identity policy、basis ID/hashの意味を参照。`exact_df_diagonal_coefficients` はfloat演算なので、exact grid構築には使わない |
| [attribution_pair.py][pair] | `PairInvolution.apply` の反射回転・ancilla相対位相、実際のbasis逆列を参照。pair専用CZを一般predicateに偽装せず、旧moduleをimportしない |
| [algorithm_design.py][n2] | N2 `increment` の降順carry ladderと、その逆列による減算を小さく移植。任意整数定数・二つのoccupation controlへ拡張。旧 `charge_circuit` はsigned小整数charge・bit対の位相実装であり、任意DF値評価とcomparatorは含まない |
| Gaussian／DF回路基盤 | providerのbasis ID/hashと実adjointを受け取る `CircuitSink` 境界を実装。新Gaussian合成器は作らない。現行I/Z/ZZ専用Qiskit builderにthresholdを渡さない |

N2を直接importするとNumPy／SciPy／Qiskit等の既存研究実装へ依存するため、独立した整数gate計画へ必要なladderだけを移植した。
新規部分はexact grid、一般二次／rank-one値評価、比較、phase-aware反射、暗黙辞書とRTE event bridge。
runtimeに承認証明、grant発行、追加ledger、管理frameworkは導入していない。

## 3. データと数値target

### 3.1 一般occupation二次式

`OccupationPolynomial` は `f0 + sum c_s prod(n_i)`、support高々2を扱う。
同じsupportは正確にcollectし、`n_i^2=n_i`、exact zeroだけを処理する。
`from_number_cdf` の入力は **full-ij** `sum J_ij n_i n_j`。上三角係数形式との混同を避ける。

保存floatは `as_integer_ratio()` でそのbit値をexactなFractionへ写す。有理数もそのまま扱う。
decimal表示への丸め、閾値切捨て、係数の変更はしない。
非定数係数の有理gcdをeta、`c_s/eta=w_s` として、

\[
a=f_0+\eta\sum_{w_s<0}w_s,\quad
z=-\sum_{w_s<0}w_s+\sum_s w_s\prod_{i\in s}n_i,\quad M=\sum_s|w_s|
\]

を使う。これはfull-Fockの包含範囲で、厳密端点の探索ではない。occupation列挙なしで作る。
全係数zeroならM=0のscalar経路となる。

### 3.2 rank-one DF平方

`from_df_square`／`from_df_block` は、**保存factor値のexactな積**として
`f=f0+lam*(sum eta_i*n_i)^2` を定義する。
先にfloatで計算済みの係数表のbit値と同一とは限らない。後者をtargetにする場合は、その係数表を
`OccupationPolynomial` として渡す。実データでの両者の同一性・元Hへの誤差は今回評価していない。
差を無条件に0とせず、例えばoccupation係数差の絶対和によるoperator上界で別途扱う必要がある。

eta_iの有理gcdをu、eta_i=u*v_iとする。整数charge `q=sum v_i*n_i` の範囲は
`L=sum min(v_i,0), U=sum max(v_i,0)`。`B=max(L^2,U^2)` とすれば、

- lam>0：`spacing=lam*u^2, a=f0, M=B, z=q^2`。
- lam<0：`spacing=abs(lam)*u^2, a=f0-spacing*B, M=B, z=B-q^2`。

このfull-Fock範囲はvacuumと符号別occupationから取得でき、スペクトル列挙を要しない。
電子数固定sectorの端点取得・制約利用は実装していない。
factorはO(n)で保持し、一般二次係数への全面展開を避ける。

### 3.3 暗黙辞書・scalar

各gridは `c=a+spacing*M/2` と `Lambda=spacing*M/2` を持つ。
`ThresholdTail` はfragmentとglobal scalar `C=sum c_l` を保持する。M個のatom一覧を作らない。

| policy（既存と同名） | RTE内operator | mass | 外側scalar |
|---|---|---|---|
| `faithful_identity_in_tail` | 元のfragment和 | `sum Lambda_l + abs(C)` | 0 |
| `extract_identity_phase` | 元のfragment和−CI | `sum Lambda_l` | C |

複数fragmentのidentityは正確にcollectする。scalar-only／zero tailも区別する。
二つのpolicyは同じ厳密Hを表すが、有限Taylor近似の引数が違うため同じfinite targetとは呼ばない。
外出し時は外側の `exp(-itC)` と新しい打切り・normalization・半径・RPE条件が必要。
対照へも同じscalarを許す。新atomで得たCだけを提案法専用にしない。

## 4. 可逆gate構成

gate計画はX／MCX、条件付き位相、providerによるGaussian変換で構成する。
MCXを一つのnative gateや無料操作とは扱わず、control本数別の個数を返す。

1. `Gamma(U)^dagger` をsystemへ適用。
2. cleanなvalue registerへzを計算。一般二次は各monomialに制御されたmodular定数加算。
   rank-oneはtwo's-complement chargeを計算し、そのbit平方をshift-addでvalueへ蓄積する。
   chargeは位相まで保持し、最後のinverseで消す。
3. z−Jを拡張register `(value, flag)` へ加算、flagを反転、value下位だけへJを戻す。
   これでzを変えずflagへ `[z>=J]` をXORする。追加carry registerは不要。
4. flagの位相／回転を実行。
5. 比較、値評価を**実際の逆列**でuncompute。
6. `Gamma(U)` を適用。

`z` は非負なので `w_z=max(1,bit_length(M))`。rank-oneのqはLとUを含む最小signed幅w_q。
専用workspaceは `w_z+w_q+1(flag)`、外側controlは別に1 qubit。
一般二次ではw_q=0。これは論理MCXのworkspaceであり、そのnative分解が追加補助qubitを使う可能性は残る。
途中のmodular wrapは許すが、最終zが0..Mにあることを構成で保証する。

flagをpとすると、反射は `R=exp(i*pi)*exp(i*pi*p)`、回転は
`exp(-i*theta*R)=exp(+i*theta)*exp(-2i*theta*p)`。
controlledでは前半もcontrol上の相対位相として残す。`Angle` はpi倍とstored radianを区別し、
piのfloat化やphase fittingをgate計画内で行わない。

同一fragmentの連続eventでは、値とchargeを保持し各Jの比較だけを繰り返す。
Gaussianと値評価はrunに一往復。これは既知のbasis cancellationとcompute共有であり、新しいsampling lawではない。
異なるfragmentや単に近いframeを同一視しない。現実装ではidentityを越すrun融合や異fragmentの同frame融合はしていない。

### 静的操作数の表現

w-bitの定数c加算について、実装が使う抽象X／MCX数を
`A_w(c)=sum_{b: bit_b(abs(c))=1, b<w}(w-b)` と書ける。
monomialのcontrol数dなら、carry ladder内の最大control数はd+w−b−1。

- 一般二次の値評価E：`A_w(origin)+sum_s A_w(w_s)`。
- rank-oneのE：origin加算＋charge評価一回＋signed charge-bit平方の各定数加算。
- 比較D_J：`A_(w+1)(-J)+1+A_w(J)`。
- 一つの反射／回転：`2E+2D_J` の可逆bit gate、中央位相2個、Gaussian二回。
- 同一fragmentのk-event run：`2E+2 sum_j D_Jj`、中央位相2k個、Gaussian二回。

これはSOURCEの静的操作数で、native RZ／CX／T／depth価格ではない。adderもMCX分解も最適性を主張しない。

## 5. finite RTEへの接続と残る境界

`draw_atom(randbits)` はgroupをexact有理massに比例して選び、fragment内Jを一様に選ぶ。
有理massのgcdで整数重みを作り、unbiasedな古典bitからrejectionする。
modulo／固定bit丸め／retry上限後の代替drawは用いない。有限deadlineで中止するなら失敗として返す必要があり、
別の偏ったdrawへ置換しない。旧256-bit proposal契約を継承したとは扱わない。

`order_parameters` は既存 `finite_rte_distribution` へtau=lambda*sを渡す。
exact tauとfloat変換差を返し、overflow／非零値のunderflowは拒否する。
その差、既存order-lawのfloat精度、angle合成誤差を科学誤差予算へ戻す作業は未実施。

旧 `_make_event` は全componentの確率和を計算するため、巨大な暗黙辞書には使えない。
`bind_rte_event` はすでに選ばれたatomを既存event/application classへ結び、exactなatom確率積と
event metadataのfloat変換差を保持する。渡したdrawだけを再正規化してはいけない。
既存と同じ「入力はrotation先頭／回路適用はproductsの後にrotation」という順序、paired phase、basis-run規約を保つ。

`emit_rte_event` は新しいthreshold emitterを使う。I/Z/ZZ supportを偽造しない。
identityの符号を各atomの位相として実行し、paired-order phaseを一度だけ付ける。
ここへ `product_sign_phase` を重ねて掛けない。
register幅が異なっても同じ外側controlと最大workspaceを使い回す。

**実装済みなのはexactデータ・gate計画・emitter interface・RTE event bridge。**
実Gaussian providerとのnative lowering、MCX／多重位相合成、全finite-bit law、実行wrapper、RPE全roundの
成功条件は未資格。既存 `RTETailLike.components` を直接置換するproduction importerではない。
`CircuitSink` の具体的Qiskit／backend実装は今回作成・実行していない。

## 6. 強い対照との構造比較

| 観点 | しきい値反射RTE | 既存DF／Pauli RTE | 強い決定論対照 |
|---|---|---|---|
| 係数mass | 半包含幅。scalarを残すならabs(C)も課金 | collect後のI/Z/ZZ／Pauli絶対和 | fragmentのRTE normalizationなし。fragment間PF誤差は残る |
| 1-event | 値計算、比較、位相、全uncompute | Z/ZZ／Pauli積・回転 | Z/ZZ位相列、又は同じ値計算とbitwise phase kickback |
| basis | predicateが依存する全軌道frameが必要 | DF atomなら高々1/2軌道supportのpartial変換も可能。global PauliはGaussian不要 | 同じfragment全体のframe。隣接frame融合・共有を許す |
| workspace | value、rank-one charge、比較flag。native MCX用は別 | parity ladder等では小さい。controlや合成用は別 | Z/ZZでは算術不要。算術版は同じvalue/charge、比較flag不要 |
| control | 算術とbasisは無制御。中央2位相とscalar相対位相を制御 | 既存diagonal-only controlを同様に許す | 位相とscalarを制御。値評価・basisを同様に無制御にできる |
| 古典生成 | group有理重み、Jのbit rejection。M／2^nの表不要 | 既存atom表とorder生成、必要ならcollect | J生成不要。全fragmentを実行する負担 |
| 精度 | exact integer部分とnative角度／旧float RTE境界を分離 | 保存係数・finite-bit・native誤差 | 同じ係数精度を許す。算術を提案法だけの利益にしない |
| shot | mass低下でnormalizationを減らせる場合がある。新finite mean／半径が必要 | 同じoperatorとcutoffなら補正meanは同じ、normalizationと価格が変わる | RTE由来の減衰なし。PF bias、state、同一成功条件を戻す |

既存 [occupation-diagonal core][pair] の `complete_core` は、各factorの対角平方に
`abs(g_ij)^2*(n_i+n_j-2*n_i*n_j)` を加えたoccupation部分を決定論coreに置き、残差をRTEへ渡す。
今回の一般二次gridはこの種の対角式も表せるが、coreの抽出則や新しいsplitを実装したものではない。
同じcoreを決定論的に実行する対照にはJ生成・比較・core由来RTE normalizationが不要であり、
残差massとfragment間PF biasはそれぞれ別に比較する必要がある。

既存 [pair反射][pair] は `Q=I-2*n_i*n_j` をGaussian frameで共役する。
`omega*n_i*n_j=(omega/2)*I-(omega/2)*Q` は二値predicateであり、しきい値構成の特別な場合に当たる。
既存方式では中央の二軌道predicateを直接CCZ／条件位相にでき、一般値registerも比較も不要。
新thresholdは多値の二次関数へ拡張する代わりに算術費用を払う。pairだけの問題で一般算術を導入する利益は主張しない。

同じ算術を使う決定論実装は、zの各bitに `-t*spacing*2^b` の位相と `-t*a` のscalar位相を付け、
値計算を逆にする。比較不要で、位相は最大w_z+1個。これは既知の対角phase kickbackで、
今回native構築／価格計測はしない。Z/ZZ対照もoccupationの全係数から直接実装でき、通常は算術workspaceが不要。

### 単一rank-one平方・identity保持時のmass一致

ここは既存SOURCE式の記号的整理で、実データの新評価ではない。`f=lam*(sum eta_i*n_i)^2`、
独立scalarなし、全occupation、単一fragmentの**未collect rotated I/Z/ZZ辞書**を比較する。
`S=sum eta_i, A=sum abs(eta_i), Q=sum eta_i^2`、
`P=sum max(eta_i,0), N=-sum min(eta_i,0)` と置くと、既存式は

\[
c_{I}=\frac{\mathrm{lam}}4(S^2+Q),\quad
c_{Z_i}=-\frac{\mathrm{lam}}2\eta_i S,\quad
c_{Z_iZ_j}=\frac{\mathrm{lam}}2\eta_i\eta_j.
\]

従ってidentity込みmassは

\[
\lambda_{I/Z/ZZ}=\frac{|\mathrm{lam}|}{4}(A+|S|)^2
=|\mathrm{lam}|\max(P^2,N^2).
\]

今回のfull-Fock平方gridも、`abs(c)+Lambda=|lam|*max(P^2,N^2)`。
**この条件でnormalizationの改善はない。** 半幅Lambdaだけをidentity込み旧massと比較して利得と呼ばない。
一般number-CDF、多fragmentのscalar／Pauli collect、独立constant、同じ外出しscalarを適用した残差には、
この等式を無条件に広げない。科学入力へ代入した値は算出していない。

## 7. 技術検証

専用環境：Python 3.12.3、NumPy 2.5.4、一時venv。共有Pythonやlibrary設定は不変更。
NumPyは既存RTE classのimportにだけ必要。テストはpackage全体の科学的 `__init__` を通さず、
変更していない `rte.py` を単独loadする。新module自身はstdlibのみ。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /tmp/track-a-threshold-tech-20261110-venv/bin/python \
  -m unittest discover -s tests -p test_df_threshold_reflection.py -v
```

最終結果：**28 tests PASS**（失敗・skipなし）。途中25／27件PASS後に、同fragment共有と負identity符号の検査を追加した。
今回の実装・テストについて失敗後の閾値緩和や科学条件変更はない。

- exact係数、full-ij、signed平方、scalar／zero、巨大Mを展開しないこと。
- 有限整数上の反射恒等式、実±1位相によるHermiticity／involution。
- 1～5bitの全register値、正負加算、inactive control、全J比較とflag XOR、逆列。
- 小さい人工occupationの反射／回転、clean workspace、負角度、実adjoint順序とmock frame phase。
- control単独位相を故意に除くと不一致を検出。負identity符号とorder phaseの二重計上を検出。
- scripted bitのrejectionと有限整数の完全group law。確率的性能samplingはしない。
- 既存event factoryとcompleteなmock辞書でevent metadataが一致。
  非零orderは明示的mock distribution、実canonical関数はzero-time interfaceのみ確認。
- 単一平方のmass等式を小さい整数上のexact Boolean Fourier恒等式として確認。
  価格・信号・科学条件の数値benchmarkではない。

`BitPhaseSink` は一つの古典bit列とexactな位相を追うtruth-table検査。
statevector、密度行列、unitary行列、Born測定を計算しない。Gaussian検査はmock orbital swapとその符号であり、
新しい実DF frameの数値検証を代用しない。人工PASSは科学的GO、native価格、外部再現／immutable CIの証拠ではない。
旧科学testsをまとめて再実行していない。

## 8. 改善機構・残存事項・次の担当

| 到達点 | 位置付け |
|---|---|
| spectrum／Mの表なしにDF式からexact gate計画を作る | 技術的に実装。既知Ξ分解の可逆算術による具体化 |
| rank-oneのcharge表現と連続runの共有 | DF構造による具体的な操作省略。強い決定論対照にも同じ利益を許す |
| sampling側で小さいmassを得る可能性 | scalar・辞書・費用条件付きの仮説。identity保持の単一平方ではmass差なし |
| 反射samplingに固有の新しい有利機構 | 今回の静的比較では特定できず。unitary／算術の実装可能性を新規性と取り違えない |
| 全費用の勝利、強Full／Truncated超え | 未評価。有限RTE、PF bias、native、shot、全round成功と取得費が必要 |

PollaのΞ分解／sign構成自体は既知であり、今回新定理として主張しない。
原PRの一般involution枠組みへの新atom接続で、独立した科学的採択を行っていない。
[Polla III.C–D][polla]／[PR Appendix A・VII.B][pr]。

**技術上の提案は保留。** 新たな分子grid、合成価格探索、最適化自由度を作って研究を継続する根拠は得られなかった。
今回の成果は実装可能性と適用境界、および比較上の落とし穴を具体化したことである。
現段階で科学runを必須の次工程にしない。

次の担当は**ユーザー／GPT**：このprimitiveを比較資産として保持するか、同情報対照でも残る具体的改善機構を
別途研究対象にするかを判断する。もし科学検証を選ぶなら、同一H／state／scalar、係数target、basis、
finite-bit・誤差・成功条件、DF／collected Pauli／強い決定論対照、価格scopeと停止点を先に指定する。
この報告はその科学実行や詳細科学レビューの許可ではない。

**Codex**：今回のコード・人工テスト・報告・必要索引をcommit／pushし、remoteの固定内容を照合して完了。
最終報告後mandatory STOP。新科学計算・重要科学レビュー・正式採択・既存STOP解除へ自動進行しない。

[base]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2e79d80dc9cce2885580e50fe6f1705da8fa41af
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/src/trotterlib/rte.py
[tail]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/src/trotterlib/df_rte_tail.py
[pair]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/src/trottertracks/representation_exploration/attribution_pair.py
[n2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/src/trottertracks/representation_exploration/algorithm_design.py
[polla]: https://arxiv.org/pdf/2207.09479v2
[pr]: https://arxiv.org/pdf/2503.05647v2
