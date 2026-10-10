# Track A：H6 DF Hermitization診断後の独立科学レビュー

- **作成日**：2026-10-10 JST
- **対象**：PR研究 Track A／H6入力生成のHermitization STOPと、その後の保存integrals限定DF診断
- **レビュー開始**：ユーザーの「レビューを開始して」に基づく。
- **結果・索引の固定commit**：`8a3189e69dd461724fa9e2c01ea08562c1c35f8d`
- **診断source commit**：`ff24de4bc410234472a416186b773fc7875ae373`
- **Repository**：`HIROMU1015/Partially-Randomized-Trotter`
- **Branch**：`track-a-ax2b-h4-post-review-20261010`
- **文書区分**：GPTによる独立科学レビュー・研究判断記録。Codexの診断報告とは別文書。
- **認可の境界**：本書は研究上の方針を判断する。新しい入力生成、state solver、H6 pilotの実行grantを発行・変更するものではない。

---

## 0. 結論

**Track Aを継続し、19個のfragmentを保持したまま、Hermitizationの受理条件を「各gの無重み絶対偏差」から「構造検査＋Hamiltonianへの重み付き変更予算」へ改訂することを推奨する。**

今回の診断では、元の`1e-10`検査に失敗するfragment 15–18の非Hermiticityは確認された。一方、重み付き係数変更は各fragmentでおよそ`1e-16`の水準であり、保存された19個のスカラー記録から計算した保守的な変更量の上界式評価は、6電子空間で約`2.492e-13 Ha`、12-mode全Fock空間を覆う評価で約`9.903e-13 Ha`である。[R1–R3]

これらの数値は、厳密な不等式に**保存されたbinary64の診断値を代入した評価**であり、丸め誤差まで封じ込めた数値certificateではない。しかし、物理的な変更量を無視してfragment行列だけで停止する現契約を見直す、十分に具体的な根拠となる。

次の方針を採る。

1. 旧STOPと元契約の違反は、そのまま保存する。過去の失敗をPASSに書き換えない。
2. DFのtol-only政策、19個のlambda、returned order、coefficient cutoff 0を維持する。
3. 保存済みraw decompositionを入力とする新しい受理経路を用意し、全fragmentと補正one-bodyのHermitian projectionを明示的に記録する。
4. 重み付き変更の上界式と独立した係数再構成を確認して、採用DF Hamiltonianを新しいidentityで固定する。
5. 新しいinput-completionの対象・予算・sourceを固定し、ユーザーの明示的実行指示後にstate/snapshotを完成させる。
6. H6の7 cell・36 wrapperの技術pilotは、その後の別実行認可で進める。今回の診断完了から自動認可しない。

**次の担当はCodex。** 受理契約の改訂、重み付きgate、保存rawのimport、入力完成経路の実装・検証をまとめて任せる。通常の実装修正やhash固定ごとに同じ研究レビューを繰り返さない。

---

## 1. レビューの範囲と証拠強度

### 1.1 今回実施したこと

本レビューは、固定commitの診断結果・全fragmentの保存スカラー統計・保存監査・診断source・旧adapter・固定OpenFermion helperを確認したうえで、表現の数学的意味と受理政策を評価したものである。NumPy 1.26の公式`eigh`文書、OpenFermionの公式APIも限定的に照合した。

独自の作業として、保存された19組の`lambda / ||g||F / ||gH-g||F`およびone-bodyの偏差から、本文の解析的上界式を算術評価した。また、既存の係数残差から係数和を介する別の上界式を評価し、fragment削除案が元の切断会計に与える影響を計算した。

### 1.2 今回実施していないこと

SCF、分子integral生成、DF decomposition、固有値計算、sector/Fock Hamiltonianの新規数値構築、state solver、時間発展、trajectory sampling、回路build/compile、実験runnerおよびtestsの再実行は行っていない。

raw NPZの所在・配列layout・hashの来歴は公開receiptと保存監査により確認したが、本レビューではraw NPZ全配列から全診断値を独立再計算していない。保存監査の全bytes照合をGPT側で再実施したとも主張しない。したがって、数値判断には公開summaryの計算が正しいという依存が残る。受理実装では保存rawから本書の必要量を再評価する。

本文の数値表の元データは公開summaryからの転記である。新しく算術評価した値は「本レビューで導出」と表示する。独立性は、sourceを読んだ別の数学的評価・受理判断にあり、外部環境で同じ科学runを再現したという意味ではない。

### 1.3 三つの区分

| 区分 | 本文での扱い |
|---|---|
| 保存事実 | source・一次JSON・実行報告に記録されている値や状態 |
| 理論的導出／保存値に基づく推論 | 本レビューで示す不等式、原因の説明候補、算術評価 |
| 新しい提案・判断 | 受理政策、予算、追加確認、次の作業。旧契約の事実ではない |

根本原因の説明候補が強く支持されることと、丸めの発生箇所を一意に特定したことは区別する。

---

## 2. 固定identityと今回の到達点

| 項目 | identity／値 |
|---|---|
| 今回の診断execution identity | `h6_df_diagnostic_20261010_launch_v1` |
| 公開結果・索引 | `8a3189e69dd461724fa9e2c01ea08562c1c35f8d` |
| raw／結果保存commit | `c4bf9de3af1580dc54ad055f8e3153477aecf6b1` |
| 診断source | `ff24de4bc410234472a416186b773fc7875ae373` |
| 新grant公開commit | `a99e6ba28d0cf26c9d63d131c2d0f620e2880c5f` |
| grant SHA-256 | `5249805956780614c2421994a701caaff300ab05cfe50ecfd9431b5d290ec4e4` |
| manifest digest | `59ff0f13202e25731ad63cb0c47bc5f9600aff9f671b412999279508b3631bd0` |
| 旧入力生成source | `67312f3195aede26e8ba4f5727d89c236772f82e` |
| 旧入力生成STOP結果 | `df3b1f694ceb72a198ab6e3e89706b239e56e1da` |
| 保存integrals SHA-256 | `edd0a618f86011757cacae481eff44dc637c11a3f64c55b0cbfb7ffbe637e51d` |
| raw decomposition SHA-256 | `2943295c2131ce9896fcad83e3dee78763efce492389ce49a6a66b87b57beebb` |
| hypothetical Hermitization SHA-256 | `0b45c9a3910a9229b984cba30d1d642ff22685bac28655c9071e3ce09ea81504` |

新診断はlinear H6／1.00 Å／STO-3Gの保存integralsを使い、明示kwargs `{"truncation_threshold": 1e-8}`だけで分解した。`final_rank=None`と`spin_basis=True`は既存関数のimplicit defaultsであり、追加で渡した引数ではない。[R1,R4]

保存結果はactual rank 19、returned truncation value `2.693610667847679e-9`、元Hermitization検査の違反indexは15,16,17,18である。約2.11秒で診断記録を完了し、raw 25ファイル・1,000,580 bytes、必要記録の欠測0が報告されている。[R1,R3]

**新しいrawは旧失敗runの未保存rawそのものではない。** 旧runのactual rankやfragmentの数値を、新runから遡って確定しない。本レビューが直接評価するのは新診断runの公開データである。

`H6_DF_DIAGNOSTIC_RECORDED`は診断記録の完了であり、H6入力の受理ではない。`H6_NOT_AUTHORIZED`、`DRAFT_NOT_AUTHORIZATION`、N/G null、総数値allowance未認定は保持される。

---

## 3. 旧契約は何を検査していたか

旧adapterは、各fragment `g`について

\[
S=\frac{g+g^\dagger}{2},\qquad d=\|S-g\|_F
\]

を計算し、`d > 1e-10`で例外を発生させる。補正one-bodyにも同じ数値閾値を適用する。全fragmentの検査を通過してからmetadataを作るため、旧runでは失敗fragmentの値を残せなかった。[R5]

この検査は「各行列が非常に近いHermitian行列であること」を要求する局所gateである。`lambda`も、電子数も、その変更がHamiltonianに与える重みも使っていない。

元契約の下でSTOPしたことは適切である。問題は、**その局所gateを物理的な入力の受理・棄却の唯一の基準にしてよいか**である。

### 3.1 無重みgateの表現依存性

実数の非零係数cに対して、

\[
g_l\mapsto c g_l,\qquad \lambda_l\mapsto\lambda_l/c^2
\]

は、`lambda_l dΓ(g_l)^2`を変えない。しかし、無重み偏差`||gH-g||F`は`|c|`倍になる。

現在のproviderは固有vectorの正規化規約を持つので、実runで自由なrescalingをしたという話ではない。それでも、無重み偏差がHamiltonianの不変な誤差尺度ではないことは分かる。

**無重み偏差の記録は残すが、受理判断にはlambda・行列norm・電子数を含めるべきである。**

---

## 4. 新診断で得た主要数値

### 4.1 元gateに失敗した4 fragment

| index | lambda | `‖gH−g‖F` | `abs(lambda) × ‖gH−g‖F`：本レビューの算術 | eigenpair residual F |
|---:|---:|---:|---:|---:|
| 15 | 1.1631213605894103e-7 | 1.100912506122748e-9 | 約1.280495e-16 | 1.9816185612401441e-16 |
| 16 | 5.518005908268932e-8 | 2.38468248263281e-9 | 約1.315869e-16 | 1.252607250460532e-16 |
| 17 | 5.3294894363009497e-8 | 2.00518768872158e-9 | 約1.068663e-16 | 2.8390321880074525e-16 |
| 18 | 1.4985230805595654e-10 | 9.668008256196875e-7 | 約1.448773e-16 | 1.410910164880635e-16 |

lambda、偏差、residualは保存summaryの値。積は本レビューで導出した値である。[R2]

index18ではgの偏差だけを見ると約`9.7e-7`と目立つが、lambdaは約`1.5e-10`である。raw normは約`sqrt(2)`、relative changeは約`6.84e-7`。4個の`lambda×偏差`はおよそ`1e-16`で揃っている。

### 4.2 係数レベルの変更と再構成残差

| 量 | 保存値 |
|---|---:|
| `chemist_interaction_transpose_asymmetry_l1` | 1.1495145113560312e-14 |
| chemist interaction imaginary l1 | 0 |
| chemist plain-square再構成差F | 5.406551739038474e-11 |
| correctionとsource reorderingの差F | 0 |
| raw normal-order one-body残差F | 3.88060528487307e-11 |
| raw antisym two-body残差F | 7.145530411810403e-11 |
| g projectionによるone-body係数差F | 3.901312145612308e-16 |
| g projectionによるantisym two-body係数差F | 7.883635906704019e-16 |
| 補正one-body自体のHermitian projection差F | 5.450455888471451e-16 |

最後のone-body projectionは、gをprojectionした影響とは別の変更であり、総変更会計で加える必要がある。[R2,R6]

### 4.3 数値が支持すること・しないこと

支持されるのは、今回の4違反が「大きいgの偏差と非常に小さいlambda」の組合せであり、保存された重み付き係数影響が小さいこと、そして大きな再構成破綻が報告されていないことである。

支持されないのは、差FをそのままHamiltonian作用素normと呼ぶこと、既存のDF truncation値を丸め込みの厳密certificateとすること、元の化学系のエネルギー精度やPRの資源優位を認定することである。

---

## 5. 非Hermiticityの原因の評価

### 5.1 固定OpenFermion helperが行うこと

固定1.6.1 helperは、chemist tensorを36×36のinteraction arrayにし、実性と行列transpose symmetryを一定toleranceで検査してから`numpy.linalg.eigh`を呼ぶ。返された固有vectorを6×6へreshapeし、`kron(..., eye(2))`でspinを戻す。各fragmentをその場でHermitian化する処理はない。[R7]

`eigh`が対角化する36×36行列のHermiticityと、固有vectorをreshapeした6×6行列のHermiticityは別の条件である。後者は化学積分のpair-index対称性に依存する。

NumPy 1.26の公式文書では、`eigh(a, UPLO='L')`は既定で下三角を使用する。したがって、入力が完全な対称行列でない場合、その両三角を平均した行列の対角化と同一とは限らない。[E1]

### 5.2 小さい固有値による対称性違反の増幅

以下は、実pair-symmetric tensorの理想構造に関する導出である。compound index `(p,q)`を`(q,p)`へ交換する作用をJ、反対称部分への射影を

\[
P_-=(I-J)/2
\]

とする。理想的に`P_- M=0`なら、非零固有値lambdaの固有vectorvは`P_-v=0`、すなわちreshape後に対称となる。

実際に保存された近似固有対のresidualを`r=Mv-lambda v`とすると、

\[
\lambda P_-v=P_-Mv-P_-r.
\]

従って、vが正規化されていれば

\[
|\lambda|\,\|P_-v\|_2
\le \|P_-M\|_2+\|r\|_2.
\]

右辺が丸め水準でも、lambdaが小さいと左辺の`||P_-v||`は大きくなり得る。6 spatial orbitalの実対称matrix空間は21次元、反対称matrix空間は15次元である。この分解は、36次元pair spaceにおける小固有値方向の敏感さを説明する。

今回の4個で`lambda×偏差`とeigenpair residualがともに`1e-16`程度であることは、この説明と整合する。**第一候補は、有限精度で生じたpair対称性成分の混入が小さい固有値方向で増幅されたことである。**

### 5.3 なお未確定の点

この説明は、特定のLAPACK routineの不具合や、積分providerの不適切性を証明するものではない。保存`M-M^T`のnormはpair全体を交換する対称性の診断であり、`p↔q`というpair内部対称性の診断そのものではない。両者を混同しない。

厳密に発生箇所を切り分けるには、保存chemist tensorに対するpair内部交換の偏差や`P_- M`、その固有vector方向への作用を調べる。これは保存配列からの静的数値診断で実施可能であり、追加SCFやDF再分解を必須にしない。

ただし、入力受理に必要なのは「丸めが発生したコード行の完全な特定」ではなく、「採用するHermitian Hamiltonianが意図したtargetへ十分近く、定義と誤差会計が明確であること」である。根本原因研究そのものを新しい終わりのない前提にしない。

---

## 6. 採用するHamiltonianを明示する

`dΓ(g)=sum_pq g_pq a†_p a_q`とする。保存integralsのone-bodyをh、providerのone-body correctionをk、scalarをcと書く。

raw分解が表す代数的作用素は

\[
H_{\mathrm{raw}}=cI+d\Gamma(h+k)+\sum_{l=0}^{18}\lambda_l d\Gamma(g_l)^2.
\]

rawは微小に非Hermitianであり得る。ここから

\[
S_l=(g_l+g_l^\dagger)/2,\quad h_S=((h+k)+(h+k)^\dagger)/2
\]

を作り、採用候補を

\[
H_{\mathrm{acc}}=cI+d\Gamma(h_S)+\sum_{l=0}^{18}\lambda_l d\Gamma(S_l)^2
\]

と明示する。lambdaが実で、各SとhSがHermitianなら、このHamiltonianはHermitianである。

この変更は、元のrawを厳密に不変に保つものではない。**明示した小さな表現変更であり、別hash・別receiptで受理する。** 旧rawを変更しない。

### 禁止すべき代用

- `dΓ(g)^2`を、説明なく`dΓ(g)† dΓ(g)`へ置き換えない。
- 正規順序の`g@g`や`g_pr g_qs`へconjugationを追加しない。
- 再構成残差を0に見せるため、one-body correctionを別の値に取り直さない。
- 小さいlambdaだからという理由でfragmentを削除しない。
- 検査を通すため、rank、ordering、DF tolを結果後に変更しない。
- `hypothetical_hermitization.npz`が存在するだけで、H6入力を受理済みとしない。

---

## 7. Hamiltonianへの変更量：相殺に頼らない上界

### 7.1 電子数を固定した場合

N電子空間において、任意のone-particle matrix Aに対して

\[
\|d\Gamma(A)\|_2\le N\|A\|_2\le N\|A\|_F
\]

である。first-quantizedのN個のAの和を反対称部分空間へ制限した形と三角不等式から従う。AがHermitianでない場合にもnorm不等式として使える。

`D_l=S_l-g_l`、`d_l=||D_l||F`、`s_l=||g_l||F`、`d_h=||hS-(h+k)||F`とする。

行列恒等式`A²-B²=A(A-B)+(A-B)B`と三角不等式から、

\[
\|d\Gamma(S_l)^2-d\Gamma(g_l)^2\|_2
\le N^2d_l(2s_l+d_l).
\]

従って

\[
\boxed{\eta_N=N d_h+N^2\sum_l|\lambda_l|d_l(2s_l+d_l)}
\]

は`||Hacc-Hraw||`の上界となる。各fragmentを加算するので、異なるfragmentの相殺を当てにしない。どのprefixやtail部分集合でも、対応する部分和で同じ方法を使える。

### 7.2 本レビューでの保存値代入

19個の保存スカラーと`d_h=5.450455888471451e-16`を代入すると、

\[
\sum_l|\lambda_l|d_l(2s_l+d_l)\approx6.831339443594272\times10^{-15}.
\]

結果は次のとおり。

| 適用範囲 | 上界式への保存値代入：Ha |
|---|---:|
| N=6電子空間 | 2.4919849350247665e-13 |
| N≤12の全Fock空間を覆うN=12評価 | 9.902534269437409e-13 |

N=12評価は、spin sectorに依存せず12-modeの全粒子数を覆う保守的な評価として使用できる。6電子のtargetではN=6の評価も保存する。

不等式は数学的な結果である。一方、上表の代入値は保存診断の有限精度normに依存する。**上表をそのまま厳密な区間認証・total u・ground-state証明としない。** 受理実装はrawから量を再評価し、丸めの扱いと判定余裕を記録する。厳密certificateを主張するなら、入力normと和の上側評価まで保証する必要がある。

### 7.3 参考：Hermitian partだけを比べる場合

正確なHermitian/anti-Hermitian分解を`g=S+A`、`A†=-A`とすると、

\[
\operatorname{Herm}(d\Gamma(g)^2)=d\Gamma(S)^2+d\Gamma(A)^2.
\]

したがって、`Hraw`のHermitian partと、正確なSによるHamiltonianとの差では、anti-Hermitian成分の一次項が消え、二次項が残る。ただしstored projectionには丸めがあり、この式から直ちにbinary64採用Hamiltonianへ極端に小さい数値精度を宣言してはいけない。

この参考式は「rawの非Hermiticity」と「Hermitianな物理作用素への影響」を分ける説明であり、主たる受理には前節の保守的な一次項を含む評価を用いる。

---

## 8. 正規順序係数による別の整合性評価

診断sourceは

\[
C_{\mathrm{raw}}=k+\sum_l\lambda_l g_lg_l,
\]
\[
T^{\mathrm{raw}}_{pqrs}=-\sum_l\lambda_l(g_l)_{pr}(g_l)_{qs}
\]

を計算する。二体係数は

\[
\mathcal A(T)=\frac{T-T_{pq}-T_{rs}+T_{pq,rs}}4
\]

でcreation/annihilationの両pairを反対称化して比較する。[R6]

符号は`a†p aq a†r as = delta_qr a†p as - a†p a†r aq as`から導ける。`Craw`は元one-body hを引いた後の残差なので、ここへhを再加算しない。

### 8.1 係数normから作用素normへの変換

各fermionic monomialの作用素normは1以下であるため、n modeについて

\[
\|\Delta H\|_2\le\|\Delta h\|_{\ell_1}+\|\mathcal A(\Delta T)\|_{\ell_1}
\le n\|\Delta h\|_F+n^2\|\mathcal A(\Delta T)\|_F.
\]

最後の不等式はn²個とn⁴個の係数に対するCauchy–Schwarzである。これは緩いが、Frobenius normをそのまま物理誤差と呼ぶ誤りを避ける。

### 8.2 projection変更の評価

n=12、g projectionによる保存係数差とone-body自身のprojection差を合わせると、

\[
12(3.901312145612308\times10^{-16}+5.450455888471451\times10^{-16})
+144(7.883635906704019\times10^{-16})
\approx1.2474647869743838\times10^{-13}\ \mathrm{Ha}.
\]

これは別の上界式への有限精度値の代入である。累積係数の差には相殺や丸めがあるため、本書では§7のfragment別三角和を主な安全側の判断材料、係数再構成を規約・符号・表現の独立した確認材料とする。

### 8.3 raw DFと保存integralsの差は別

raw normal-order residualを同じ不等式へ代入すると、

\[
12(3.88060528487307\times10^{-11})+144(7.145530411810403\times10^{-11})
\approx1.0755236427191749\times10^{-8}\ \mathrm{Ha}.
\]

これはtruncation value `2.693610667847679e-9`と同じ量ではない。前者は保存された係数残差に基づく緩い作用素上界式評価、後者はproviderの切断会計の返却値である。

projectionの係数評価を加えた約`1.0755361e-8 Ha`が元thresholdをわずかに超えても、真の誤差がthresholdを超えている証明ではない。緩い上界であり、providerが誤動作したという結論にもならない。数値certificateでもない。

今回の中心的な観察は、**projectionによる変更が、保存integralsからDFへの既存残差より十分小さい**ことである。

---

## 9. finite-time信号への接続と限界

Hermitianな二つのHamiltonian H1,H2なら、Duhamel公式から

\[
\|e^{-itH_1}-e^{-itH_2}\|_2\le|t|\|H_1-H_2\|_2
\]

である。正規化stateのcomplex signal差も同じ右辺で抑えられる。

ただし今回の`Hraw`は微小に非Hermitianであり得るので、rawとprojectedの比較へunitaryの式を無条件に適用しない。採用するHermitian DF targetと、明示的に定義したHermitian integral-referenceを比較する場合に使用する。

例えば、保存integralsの演算子をKとし、参照を`Herm(K)`と明示定義した場合、Hermitian projectionの作用素normに関する収縮性から`||Hacc-Herm(K)|| <= ||Hacc-K||`が成り立つ。ただしこれは参照定義に関する選択であり、今回その新参照の数値計算を実施したわけではない。

§8の緩い保存値評価をT=0.8へ換算すると、およそ`8.6e-9`のsignal水準となる。これはH6 technical案の診断label `epsilon_sig=0.001`と比べ十分小さい水準だが、total u、測定精度、chemical energy accuracyの認定ではない。

### 二重計上を避ける

採用済み`H_DF`をtask targetにする場合、PF/finite-RTE biasは`H_DF`に対して測る。保存integralsから`H_DF`への表現誤差は別ledgerへ置く。後で元integralsを物理targetとする主張へ接続する場合のみ、その層を加える。

DF切断、Hermitization、SCF/integral品質、PF/finite-RTE近似、数値roundoff、測定統計を同じuへ混ぜない。

---

## 10. 代替案の比較

| 案 | 判断 | 理由 |
|---|---|---|
| 無重み閾値だけを1e-6等へ上げる | 採用しない | 今回の最大偏差に合わせた救済に見え、lambdaや物理影響を制御できない |
| 元gateを維持して停止し続ける | 主方針にしない | 小さい物理影響の数値的表現まで一律に棄却する |
| fragment 15–18を削除する | 採用しない | cutoff0・tol-only契約を変え、表現と資源比較を変更する |
| 19個を保持し、重み付き誤差予算内でHermitian化 | **採用推奨** | 保存rawを再利用でき、変更を数値的に説明できる |
| 高精度でDF decompositionをやり直す | 現時点では不要 | 根本原因切分けの補助にはなるが、入力受理の最小手段ではない |
| 最初から21次元実対称pair基底で分解する | 将来の代替案 | 対称性を構成的に保てるが、provider・ordering・rank選択の変更が必要 |
| 別library／別factorizationへ移る | 不要 | 現問題だけでは大きな研究・実装変更を正当化しない |

### 10.1 小さいlambdaを削除する案の具体的問題

providerは`w_l = |lambda_l| (sum_pq |g_l,pq|)^2`を用いてrankを選ぶ。固定helperでは大きいweight順に並べ、残りweightの和がthreshold内になる点で切断する。[R7]

fragment18のweightは`7.966970366011683e-9`。これだけを追加削除すると、元の返却truncation valueとの和は

\[
2.693610667847679e-9+7.966970366011683e-9
=1.0660581033859362e-8
\]

となる。少なくとも元の切断会計では`1e-8`以内とは言えない。これは真の物理誤差超過の証明ではないが、「小さいlambdaだから何も変わらない」として削除できない根拠である。

15–18を全て削除すると、同じ会計は約`1.0703065e-5`になる。今回の問題はfragmentそのものの寄与ではなく、その中の小さい非Hermitian成分なので、fragment全体の削除は修正対象を取り違えている。

### 10.2 対称pair基底案を今すぐ採らない理由

実対称6×6matrixの正規直交基底を使ってinteractionを21次元へ表し、そこで分解すれば、理想的にはreshape後の対称性を構成的に保てる。これは候補として妥当である。

しかし、入力tensorのどの対称化を採用するか、切断weight、返却order、回路prefix、既存結果との対応が変わる可能性がある。現在の保存rawの重み付き変更が小さい状況では、新しいfactorization実装を主課題にする必要はない。新gateで問題が残る、または多系統で大きい修正が必要と判明した場合に再検討する。

---

## 11. 推奨する新しいHermitization受理契約

### 11.1 科学的方針

旧`1e-10`無重みgateは「元契約との比較診断」として残す。新しい受理は次の三条件で行う。

- **構造**：finite/layout/実lambda/sector保存／spin構造などが意図した入力に合う。
- **変更量**：Hermitian projectionがHamiltonianへ与える影響を、全fragment＋補正one-bodyを含む重み付き予算で制限する。
- **表現整合**：projection後も元integralsとの正規順序・符号・one-body補正・tensor再構成が整合する。

重み付き量が小さいだけで構造違反やtensor conventionの誤りを無視しない。構造が正しいだけで大きい物理変更を認めない。

### 11.2 今回の入力に対する予算案

今回の`df_tol=1e-8`に対し、Hermitizationの追加変更予算を

\[
\tau_{\mathrm{proj}}=0.01\,\tau_{\mathrm{DF}}=10^{-10}\ \mathrm{Ha}
\]

とする案を推奨する。**これは本レビューで提案する新しい政策であり、既存契約や原論文の定数ではない。** 1%は表現変更を元DF予算に対して従属的にするための保守的な工学配分である。数学的に唯一の値ではない。

本runの上界式評価は、全Fock評価でもこの予算より約101倍小さい。今回のデータをぎりぎり通すように閾値を調整する設計ではない。

実装は、rawからnormと和を再評価し、数値roundingの扱いを明示する。工程上の受理なら`PASS_ENGINEERING`等のラベルを使い、`representation_error_certified=false`と両立させてよい。厳密な上界認証を主張する場合に限って、各norm・合計の上側評価も保証する。

### 11.3 個別fragmentと合計を両方残す

`lambda`、raw norm、Hermitian/anti-Hermitian偏差、relative change、fragment別の上界寄与、全体和を記録する。個別の大きい寄与を、合計係数での相殺だけで見えなくしない。

lambda=0や小さいlambdaを理由に行を落とさない。全19個を保ち、最大偏差index、最大重み付き寄与indexも記録する。両者は同じとは限らない。

### 11.4 変更しない条件

保存integrals、19個のlambda、returned order、tol-only政策、coefficient cutoff0、H6のspin/number target、scalar、one-body correctionの由来を保持する。

新しく変わるのは、Hermitizationの受理規則と、採用DFの明示的canonicalization・記録経路である。元のSTOP、旧grant、旧freeze、旧source、旧H4結果、Track Bを改変しない。

---

## 12. 追加診断をどこまで行うか

### 今回の保存データで進められる確認

次の確認は、新しいSCFやDF再分解ではなく、既存rawを受理する経路の検証としてまとめる。

1. 保存integrals／raw decomposition／receipt／source identityを照合する。
2. 全19個のSと補正one-bodyを生成し、前後hashと変更量を保存する。
3. §7のfragment別上界式と§8の係数再構成を、rawから再評価する。
4. postprojectionのHermiticity、cross-spin、alpha/beta対応を確認する。
5. pair内部対称性の診断を必要に応じて保存し、原因候補の支持範囲を明示する。

全Fock行列の巨大な構築、MP80/120での分解再実行、別solverの総当たりを必須にしない。新しい受理量が予算を超える、構造が崩れる、保存summaryとの重大な不一致が出る場合に初めて追加方針を判断する。

### 追加科学runを正当化する条件

重み付き変更が予算を超える、pair対称性違反が大きい、raw再構成が規約修正なしで説明できない、postprojectionでsector保存が失われる、独立係数照合が矛盾する場合は、受理をSTOPする。そのときは高精度分解、対称pair基底、入力tensorの対称化政策等をGPTで再検討する。

---

## 13. Codexへ渡す次の作業単位

**「重み付きHermitization受理と、保存DFからのH6入力完成準備」を一つの作業単位にする。**

### A. 契約・source

旧adapterを黙って緩和せず、新policy/versionを作る。科学的に固定すべき事項は§11である。具体的なmodule構成、JSON schema、tests、丸め評価の実装はCodexの裁量とする。

### B. 保存rawの採用経路

今回のraw decompositionを読み取り専用入力として扱う。decomposerを再起動して「良いraw」が出るまで試す運用にしない。新しいDF受理receiptは、旧integralsと新診断runのrawを親identityとして参照する。

`hypothetical_hermitization.npz`はあくまで診断物である。採用時は新policyに基づく受理検査を行い、必要なら同じprojection値とbytesが一致することを確認したうえで、正式DF targetとして別receipt・hashを付す。名前変更だけで受理を代用しない。

### C. tests

少なくとも、弱いlambdaと大きい無重み偏差、同じ偏差で強いlambda、複数fragment寄与の和、符号やlambdaの不正、sector構造の違反、raw変更、旧contractの誤用、one-body correctionの二重加算、projection後の誤差会計をsyntheticで確認する。

具体的なテスト件数はGPTが固定しない。受理することだけでなく、物理変更が大きい場合に拒否することを確認する。

### D. 入力完成

ユーザーが新しい対象・予算を明示承認した後に、受理DFから従来のbounded sector state solverとsnapshot作成へ進む。今回はintegralsとraw DFが保存されているため、SCF/DF decompositionからやり直す必要はない。

入力受理とstate生成の計算上限、実環境、CPU、source/input/output bindingは新計画で固定する。旧480秒の診断予算や消費済みgrantを、そのままstate生成認可へ流用しない。

この作業が成功したら、正式なDF receipt、state receipt、snapshot、terminal、監査をGitHubへcommit/push・remote照合する。H6 pilotは別の明示的実行指示まで開始しない。

---

## 14. H6 pilotと研究全体への影響

研究の主軸RQ-R、補助RQ-P1、H4→H6→H8の枠組みを変更する根拠は、今回のHermitization STOPにはない。これはHamiltonian表現を数値的に整える問題であり、PRそのものの有効性を検証した結果ではない。

新policyで19fragmentを保持して受理する場合、既存のmid-prefix規則`p=(L+1)//2`は`p=10`となる。これは今後採用する場合の算術上の帰結であり、H6候補を既に実行・最適化した結果ではない。

既定の7 cell・36 wrapper、q1/q2、partial R4などをこの診断の結果を理由に増減させない。state、bias、normalization、費用の実際の値は、受理済みDF targetに対して後で評価する。

Hermitian化によりbasis angles、RTE component、lambda_R、回路fingerprintが変わる可能性がある。Hamiltonianへの影響が小さいことから、回路の整数費用や方式順位まで不変と結論しない。新targetからprepared representationを作り直し、参照と回路を同じtargetへ結び付ける。

H4 legacy結果はそのまま残す。H6技術pilotでpolicy接続を確認しただけで、H4とH6を完全に同一表現政策で比較済みとはしない。サイズに関する主張を作る段階では、H4の新policy bridgeとDF政策差を整理する。H8のtruthへ先に触れない。

### 新規性の扱い

Hermitizationを重み付き誤差で管理すること自体を、新しいPRアルゴリズムとして主張しない。本研究では、再現可能な資源比較のための入力品質管理・representation accountingに位置付ける。

input誤差のcertificate完成や、数値固有vectorの研究をTrack Aの新しい主目的にしない。直接資源評価へ戻るために、必要な範囲の問題を閉じる。

---

## 15. GO／STOPと役割分担

| 段階 | 進めてよいこと | 進めてはいけないこと／戻る条件 |
|---|---|---|
| 本レビュー後 | Codexが新受理契約・実装・synthetic・保存raw importを準備 | 旧policyを上書きし、元runをPASSへ書換えない |
| 受理量の再評価 | 構造と重み付き変更が登録範囲なら入力完成候補へ | 予算超過、構造違反、重大なsummary不一致はGPTへ戻す |
| 入力完成実行 | 明示的な新認可後、保存DFからstate/snapshotを作る | SCF/DF再分解や閾値救済へ自動移行しない |
| H6技術pilot | 完成input・source・予算を固定し、別の明示認可後に実行 | 診断完了をpilot認可とみなさない |
| 科学的な次の節目 | H4補完とH6技術pilotの証拠をGPTが評価 | H6本検証・H8へ自動GOしない |

本レビューで決定した方向を実装する通常作業はCodexにまとめて任せる。同じ方針について、source固定・hash照合・synthetic追加のたびにGPTへ戻す必要はない。

ただし、研究意味論、primary target、coefficient policy、比較の独立性を実質的に変更する必要が生じた場合は、GPTレビューを提案し、必要資料を確保したうえでユーザーの開始承認を待つ。

---

## 16. 支持される主張／支持されない主張

### 支持される

- 新診断runでは19fragmentが保存され、15–18が旧無重みgateを超えた。
- lambda・g偏差・Hamiltonian係数への影響は別量である。
- 保存スカラーを用いると、全fragmentを保持したHermitian projectionの物理変更を小さい値で評価できる。
- 小固有値方向で対称性誤差が増幅された説明は、source構造と保存値に整合する。
- 重み付き変更予算と表現整合性を組み合わせて入力政策を改訂することには合理性がある。
- 追加SCF/DF再分解ではなく、保存rawからH6入力を完成させる方針が最小である。

### 支持されない

- 旧失敗runのrawが新runと完全一致していた。
- 全違反が特定libraryのbugによるものだと確定した。
- 元の無重みgateがPASSした。
- 係数Frobenius残差がそのままHamiltonian作用素誤差である。
- 保存値代入の上界評価が、そのまま丸め込みのcertificateである。
- H6 state・信号精度・shot・量子資源順位を認定した。
- fragment削除やrank変更をしても同じ契約である。
- H6 pilotやH8が実行認可された。

---

## 17. 前回判断からの変更と一貫性

前回は、raw DF出力が未保存で、どの程度の偏差とlambdaが対応するかを判断できなかった。そのため、保存integralsから一回だけ診断し、政策を変えず全rawを残す方針を採った。

今回、新しい証拠として19fragmentのlambda・norm・偏差・重み付き係数影響・再構成残差が揃った。これに基づき、**「診断結果を待つ」から「19個保持の重み付き受理政策を実装する」へ進行判断を変更する。**

研究の主軸を変更したのではない。未確定だった入力政策を、新しい数値証拠に基づき具体化した。元gateの失敗を事後的に成功へ書き換えるのではなく、変更履歴を残して別契約へ進む。

---

## 18. 最終判断

**DFとHermitian fragmentを使う回路構成は、今回の診断だけで放棄する必要はない。** 一方、`||gH-g||F <= 1e-10`だけを受理条件とする運用は、今回の小固有値fragmentの物理的影響を適切に表していない。

したがって、19fragmentとtol-only方針を保持し、構造検査・重み付きHamiltonian変更予算・係数再構成を組み合わせて、制御されたHermitian projectionを新しい受理政策として採ることを推奨する。

本レビューの研究判断は完了。次の担当はCodexであり、保存rawからの入力完成経路を具体化する。H6入力受理の記録や実行認可は、この文書を理由に自動変更しない。

---

## 付録A. 上界式計算に使った19個の保存スカラー

以下の値は固定`diagnostic_summary.json`から転記した。raw NPZからの独立再計算ではない。`lambda×d`と`w_l d_l(2s_l+d_l)`は本レビューの算術である。

| index | lambda | s = ‖g‖F | d = ‖gH−g‖F | lambda×d | 上界寄与（N²を掛ける前） |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.1718777128429338 | 1.414213562373095 | 1.0671707795912364E-16 | 1.2505936524e-16 | 3.5372130085e-16 |
| 1 | 0.5613899829190048 | 1.414213562373096 | 2.2147148048383332E-16 | 1.2433187065e-16 | 3.5166363541e-16 |
| 2 | 0.3535291621355899 | 1.414213562373095 | 2.6612407172745247E-16 | 9.4082620102e-17 | 2.6610583466e-16 |
| 3 | 0.23795331065240777 | 1.414213562373095 | 5.363822642631059E-16 | 1.2763393556e-16 | 3.6100328537e-16 |
| 4 | 0.1764860332023603 | 1.4142135623730947 | 5.85261972987713E-16 | 1.0329056400e-16 | 2.9214983294e-16 |
| 5 | 0.14337987322908796 | 1.4142135623730943 | 6.250703627098202E-16 | 8.9622509365e-17 | 2.5349073647e-16 |
| 6 | 0.014817208878490636 | 1.414213562373095 | 6.9243966564566254E-15 | 1.0260023162e-16 | 2.9019727811e-16 |
| 7 | 0.012141886736015716 | 1.4142135623730945 | 1.1951187972486423E-14 | 1.4510997072e-16 | 4.1043297726e-16 |
| 8 | 0.009675337019877632 | 1.4142135623730956 | 9.792575116214506E-15 | 9.4746464542e-17 | 2.6798347028e-16 |
| 9 | 0.008275381862551348 | 1.4142135623730945 | 1.872681975808734E-14 | 1.5497158457e-16 | 4.3832583336e-16 |
| 10 | 0.008566156033672048 | 1.4142135623730945 | 5.782559622674385E-15 | 4.9534308002e-17 | 1.4010418036e-16 |
| 11 | 0.00008860919886359778 | 1.4142135623730951 | 6.906644494505007E-13 | 6.1199223549e-17 | 1.7309754390e-16 |
| 12 | 0.00005735044679891893 | 1.4142135623730956 | 4.624943023214118E-12 | 2.6524254880e-16 | 7.5021921967e-16 |
| 13 | 0.000032095860883971535 | 1.4142135623730951 | 3.3783479884630928E-12 | 1.0843098706e-16 | 3.0668914495e-16 |
| 14 | 0.000013726567916115145 | 1.4142135623730951 | 1.8796175704311202E-11 | 2.5800698237e-16 | 7.2975394731e-16 |
| 15 | 1.1631213605894103E-7 | 1.4142135623730947 | 1.100912506122748E-9 | 1.2804948520e-16 | 3.6217863739e-16 |
| 16 | 5.518005908268932E-8 | 1.4142135623730954 | 2.38468248263281E-9 | 1.3158692029e-16 | 3.7218401491e-16 |
| 17 | 5.3294894363009497E-8 | 1.414213562373094 | 2.00518768872158E-9 | 1.0686626605e-16 | 3.0226344583e-16 |
| 18 | 1.4985230805595654E-10 | 1.4142135623730945 | 9.668008256196875E-7 | 1.4487733515e-16 | 4.0977512457e-16 |

補正one-bodyの変更量は`d_h = 5.450455888471451e-16`。

計算は上表のdecimal表現を高precisionのDecimalへ読み込んで行った。これは保存診断値のprecisionを引き上げたり、元のnormを認証したりするものではなく、転記した値の加算・乗算で余分な丸めを避けるためである。

### 再現用の計算式

```python
# rows: 各行に lambda_abs, delta_frobenius, g_frobenius があるとする。
# 全て元の診断値の文字列。新しい分子計算やDF分解は行わない。
from decimal import Decimal, localcontext

with localcontext() as ctx:
    ctx.prec = 60
    D = Decimal
    total = sum(
        D(r['lambda_abs']) * D(r['delta_frobenius'])
        * (2 * D(r['g_frobenius']) + D(r['delta_frobenius']))
        for r in rows
    )
    dh = D('5.450455888471451e-16')
    beta_N6 = 6 * dh + 36 * total
    beta_full_Fock = 12 * dh + 144 * total
```

---

## 付録B. 資料一覧と参照範囲

- **[R1] [結果・GPTレビュー資料索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/docs/research/track_a_h6_df_diagnostic_result_v1.md)** — 固定identity、実行scope、主要保存値、未認可状態。
- **[R2] [diagnostic_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/diagnostic_summary.json)** — 全19fragmentの保存統計、表現残差、projection変更。本文の数値の主出典。
- **[R3] [saved_diagnostic_audit_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/saved_diagnostic_audit_v1.json)** — 公開raw・receipt・grant・terminal等のhashと監査範囲。
- **[R4] [診断準備仕様](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/docs/research/track_a_h6_df_hermitization_diagnostic_preparation_v1.md)** — 同一保存integrals、tol-only、raw優先保存、normal-order定義。
- **[R5] [旧H6 input adapter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/67312f3195aede26e8ba4f5727d89c236772f82e/src/trottertracks/resource_applicability/ax2b_h6_input.py)** — 無重みHermitization検査とmetadata生成順序。
- **[R6] [diagnostic port](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ff24de4bc410234472a416186b773fc7875ae373/src/trottertracks/resource_applicability/ax2b_h6_df_diagnostic_port_v1.py)** — normal_order、antisymmetrize、weighted係数・hypothetical projectionの計算。
- **[R7] [実行環境に固定されたOpenFermion 1.6.1 helper](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/installed_dependency_sources/openfermion_low_rank_1_6_1.txt)** — chemist変換、eigh、reshape/kron、weight排序、truncation。
- **[R8] [raw_decomposition_receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/raw_decomposition_receipt.json)** — raw file SHA、19×12×12配列layout、returned dtype/valueの来歴。
- **[R9] [raw_decomposition.npz](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/raw_decomposition.npz)** — 保存rawの参照先。今回の本文は公開summaryのスカラーに基づく。
- **[R10] [hypothetical_hermitization.npz](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/hypothetical_hermitization.npz)** — 診断専用のprojection・tensor。入力受理済みではない。
- **[R11] [runtime_environment.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/runtime_environment.json)** — 診断実行環境と外部helper/configのidentity。
- **[R12] [terminal_status.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/terminal_status.json)** — 診断記録完了とH6入力未受理の区別。
- **[R13] [decomposition_call.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/decomposition_call.json)** — 一回call、kwargs、input hash、新旧execution区別。
- **[R14] [source freeze](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a3189e69dd461724fa9e2c01ea08562c1c35f8d/artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/source_freeze_v1.json)** — 診断source187件、validation2件の固定。
- **[E1] [NumPy v1.26公式：numpy.linalg.eigh](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.eigh.html)** — UPLO既定値、Hermitian入力と下三角使用、LAPACK経路。
- **[E2] [OpenFermion公式API：low_rank_two_body_decomposition](https://quantumai.google/reference/python/openfermion/circuits/low_rank_two_body_decomposition)** — 関数の表現、real-integral symmetryとfinal_rankの意味。現行APIと固定1.6.1実装は区別。

### 参照の限界

本文の[R]は固定repository資料、[E]は外部の公式一次資料である。外部の現行API文書だけで、実行時の1.6.1 sourceを置き換えていない。

固定OpenFermion helperの説明文には切断和の添字が紛らわしい箇所があるため、切断判断はsource中の`cumulative_error_sum[-1] - cumulative_error_sum`とretained rankの処理に基づいて説明した。新診断の返却truncation値を、未保存の全discarded eigenpairから独立に再計算したわけではない。

本レビューで採用する新政策の数値・条件は本文で明示した提案であり、出典に既に登録されている値として表現しない。
