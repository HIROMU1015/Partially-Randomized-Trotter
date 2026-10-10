# Hamiltonian表現・PR協調設計：限定構成比較batchの独立科学レビュー

- 作成日：2026-10-10（Asia/Tokyo）
- レビュー実施者：GPT
- 開始承認：利用者の「レビューを開始して」。今回の限定構成比較batchに対する承認。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`representation-construction-comparison-20261010`
- レビュー対象結果commit：`c24dcede10ed9726e1e0b1030eca749ba7bc5cba`
- 結果本体の公開commit：`cfa1ca22494b8b984ff4450e6a159c568bed8a46`
- 科学実行source：`ce99b57166a9f422f151f56fa17ed799cd9909dd`
- 前段階の基点：`39345830ddfe7c3e2a488c284a0623f489764087`
- 状態：**科学レビュー完了。限定継続と研究対象の再整理を推奨する。中心テーマ、新規性、論文の主要claimは未確定。**
- リポジトリへの変更、新しい量子回路compile、sampling、分子計算は実施していない。末尾の独立チェックは固定入力の代数・保存表示値の算術である。

## 0. 結論

今回のbatchは、前回の「構成上の恒等式が成立するか」という段階から、実finite-RTE、制御付きnative回路、同一複素信号精度の会計へ進んだ。その意味で有用な進展である。しかし、現構成をそのまま分子数・系サイズ・角度gridを増やしていくことを、今の時点では勧めない。

**Aは比較要因を切り分ける限定検証へ進める。Cの現行conditional-SU(2)＋THRIFT構成は性能拡張を一旦保留する。Bの拡大空間＋反射RTEも現構成の拡張を保留する一方、投影後の二体項を物理空間で直接実装する別ルートB′には小さい構成比較の価値がある。**

| 対象 | 科学的判断 | 次に解く問題 |
|---|---|---|
| A：共通frameのcore＋正確残差 | 限定継続。資源有利性は未確定 | core導入、辞書変更、基底変換の利益・負担を分離する |
| C：現在の低次数THRIFT＋conditional mixed access | 現構成の性能探索を保留 | 既知構成を超える、具体的な低費用accessの変更があるか |
| B：拡大空間の生成子反射 | 現構成の性能探索を保留 | 圧縮の利益を保持する処理単位があるか。単なるtoy追加はしない |
| B′：投影後の物理空間での2軌道占有射影分解 | このレビューで具体化した比較候補。未採択 | 補助軌道を使わず、projector/involutionとして実装しても費用利益が残るか |

「RZとCXで利害が分かれる」こと自体を失敗とは判定しない。非自明なPareto改善や、明示された実装条件で意味のある改善があれば研究になり得る。一方、指標を後から都合よく選んでwinnerを作ることはしない。

前回はA/Cを同程度に継続する方針だった。今回Cを下げる根拠は、強い対照を導入した保存結果であり、質問の表現が変わったためではない。B′は従来Bの成功認定ではなく、同じ圧縮表現を別の実装へ変換する新しい検討ルートである。

## 1. レビュー対象と証拠の取得範囲

### 1.1 正本と実際の参照

固定source、report、scope、一次resultの主要内容、native IRの取得可能性、保存監査、実行・テスト記録をGitHub connectorで確認した。branch HEADは対象結果commitと一致していた。大きいresult/IRはGit blob経路で取得できる。資料の未pushによるレビュー保留ではなく、追加の再push依頼は不要である。[R1–R4, S1, V1–V3]

レビューの深さを次のように区別する。

- **GPTが行ったこと**：constructor、finite mean、trajectory、制御、scalar phase、shot式、費用集計、Bのisometry投影等のsource読解。主要保存値とreportの意味の照合。固定入力の独立代数と算術。近接する一次文献の確認。
- **Codexの保存記録として採用したこと**：70 local tests、374 IRの別実装照合、176 source/input blobsの照合、各資源観測値、改変拒否テスト。[V2,V3]
- **GPTが今回独立に繰り返していないこと**：70 testsの再実行、374回路全部の行列再構成、全176 blobの再hash、元science runner、量子shot、分子生成、FT合成、全候補の再最適化。

したがって「374回路をGPTが再compileして確認した」「外部環境で元batchを丸ごと再現した」とは表現しない。保存監査が別実装であることと、別主体による科学的再現であることは異なる。

### 1.2 実行状態

保存報告は `LIMITED_CONSTRUCTION_COMPARISON_COMPLETE_AWAITING_GPT_REVIEW`。科学run1は固定sourceで成功しretry=0。pre-freezeの引数名不整合によるtests失敗は保持され、旧探索run1の失敗／run2の成功とは別の履歴である。[R1,V2,V3]

A18・C48・B2の回路比較条件、Bのmean診断6条件、合計374 wrappers（A246/C96/B32）、136精度会計行が保存されている。136行は複数epsilonへ同じ回路を会計し直した行を含み、136独立実験ではない。

wall約16.9223秒、CPU約16.9367秒、peak RSS 561,500,160 bytes、量子shots/GPU/分子load/ground-state solveは0。これらは小さいbatchの古典実行記録であり、constructorの大系scalingや量子計算時間ではない。

### 1.3 数値・実装上の留保

保存verifierはQiskit/project moduleをimportせず、exterior-power minorsとnative gateの行更新で照合している。保存値では最大absolute operator residual約6.08e-12、mean residual約5.61e-16、bias差約1.06e-16。位相・shotの改変を拒否した。[V1,V2]

現在の `real_terms` は実係数の非零項を閾値で間引かず、合計虚部が1e-11を超える場合を拒否し、それ以下の虚部を実数化する。構成再構成誤差が別に検査される。これはbinary64上の処理であり、一般の厳密算術certificateではない。[S1]

compiled circuitのscalar phaseだけを小行列で認証してmetadataで補正する経路も残っている。相対branch誤差を任意に消す処理ではない。保存IRのabsolute phaseを検査しているが、このdense certificateを大系で無料のcompiler修復oracleとして使えるとはしない。

## 2. 共通taskと資源式の科学的評価

今回のtaskは固定時間Tの複素Hadamard信号であり、最終のenergy/QPE/RPE予算ではない。state preparation、Hadamard ancilla、制御時間発展、basis変換、identity相対位相、読み出しbasisが含まれる。Z測定はmetadataとして別計上される。[R1,S1]

K=2のpaired finite RTEは、短時間tailについて

\[
P_3(-i\delta R)=I-i\delta R-\frac{\delta^2R^2}{2}+\frac{i\delta^3R^3}{6}
\]

を正規化補正した第一momentに持つ。\(\tau=\lambda_R\delta\) とすると

\[
B(\tau)=\sqrt{1+\tau^2}+\frac{\tau^2}{2}\sqrt{1+(\tau/3)^2},\qquad
\gamma=B(\tau)^q.
\]

bias診断

\[
b=\|A_{\rm corr}-e^{-iTH}\|_2
\]

を用い、各軸の精度を \(a=\epsilon/\sqrt2\) とすると、採用した十分shot方策は

\[
N_{\rm axis}=
\left\lceil\frac{2\gamma^2}{(a-b)^2}\log\frac4{0.05}\right\rceil,
\quad b<a,
\]

\[
G_m=N_{\rm axis}\big(\mathbb E C_{X,m}+\mathbb E C_{Y,m}\big).
\]

式の役割分担は整合している。normalizationをbiasへ足すのではなくrange/shot側に入れ、biasで精度余裕を減らしている。X/Yの同一trajectoryを使い、費用和のSEを計算している点も妥当である。[S1]

ただし、bは小系dense binary64診断であり、shot式の形式がHoeffding型であることだけで全数値誤差を含む認証になるわけではない。今回の比較には使えるが、未知の大系で同じbを安価に得られる実装は未確立である。これは現段階の診断を無効にする理由ではなく、運用範囲の限定である。

### 2.1 資源改善の分解

同じ方策での比較は、天井関数を除けば

\[
\frac{G_{\rm new}}{G_{\rm old}}
=\frac{\bar C_{\rm new}}{\bar C_{\rm old}}
 \left(\frac{\gamma_{\rm new}}{\gamma_{\rm old}}\right)^2
 \left(\frac{a-b_{\rm old}}{a-b_{\rm new}}\right)^2.
\]

天井関数を含む厳密な会計では \(G_{\rm new}/G_{\rm old}=(N_{\rm new}/N_{\rm old})(\bar C_{\rm new}/\bar C_{\rm old})\) とすればよい。この式は新規アルゴリズムではなく、今回何を改善すべきかを切り分けるbookkeepingである。

### 2.2 同一方策内のshot削減余地

\(\gamma\ge1,b\ge0\) なので、同じ十分shot方策の最小値は

\[
N_{\rm floor}(\epsilon)=
\left\lceil\frac4{\epsilon^2}\log 80\right\rceil.
\]

epsilon=.05では7012、.02では43821。これは**情報理論的な最低shot数ではなく、採用した方策内の下限**である。別の推定器や状態情報を使う方法の下限ではない。

Aのframe_identityの7151 shotsはfloor7012に近い。同じ1-shot費用のままnormalizationとbiasを理想化しても、shot削減は最大約1.94%である。B反射側の7544 shotsでも最大約7.05%。したがって現windowでは、さらにlambdaだけを小さくする探索より、1-shot回路・basis移動を安くする構成が重要になる。この推論を時間・精度の異なるregimeへ無条件に外挿しない。

## 3. Aの評価

### 3.1 保存結果から支持されること

Aの入力は3 fermionic modes、正square rank2で

\[
g_0=\begin{pmatrix}.8&.09&0\\.09&-.35&0\\0&0&.15\end{pmatrix},\quad
g_1=\begin{pmatrix}-.2&0&0\\0&.6&.07\\0&.07&-.45\end{pmatrix},\quad
H=d\Gamma(g_0)^2+d\Gamma(g_1)^2.
\]

frame候補はIと各factorの固有frame。Vで変換したfactorの対角部分を平方してcoreとし、正確なPauli残差を返す。basisは反復全体の外に置く。native対照もGaussian basisをHadamard ancillaに無制御とし、中心対角作用だけを制御している。前回のgeneric controlによる不利をそのまま比較へ持ち越してはいない。[R1,S1]

全候補で今回のRZ point最小はq1。epsilon=.05の主要値は次のとおり。

| 構成 | tail lambda | N/axis | G_RZ | G_CX |
|---|---:|---:|---:|---:|
| native all-R | .900412171 | 9285 | 603525 ± 88364.437 | 194985 ± 30392.297 |
| native prefix L_D1 | .274861527 | 7236 | 1506897 ± 6939.18 | 864702 ± 3618 |
| native all-D | 0 | 7063 | 1617427 | 1045324 |
| frame_identity | .0915 | 7151 | 514872 ± 5405.648 | 582806.5 ± 5233.996 |
| factor0 frame | .054474183 | 7075 | 891450 ± 5348.197 | 689812.5 ± 5178.37 |
| factor1 frame | .123210565 | 7175 | 1015262.5 ± 5251.562 | 703150 ± 5423.79 |

±は保存されたpaired SEで、formal CIではない。RZ/CXで利害が分かれる結果を、一方だけ選んで全面優位と呼ばない。[R1]

保存値をNで割ると、比較の内容が明確になる。

| q1・epsilon=.05 | native all-R | frame_identity |
|---|---:|---:|
| 1軸当たりshot方策 | 9285 | 7151 |
| X/Yの回路1本ずつのRZ費用和の標本平均 | 65 | 72 |
| X/Yの回路1本ずつのCX費用和の標本平均 | 21 | 81.5 |

RZ総費用比は約0.853、shot比約0.770、1組のRZ比約1.108。すなわちframe_identityは、この標本点では「回路自体が短いから」RZ総費用が小さいわけではない。shot側の減少が回路増を上回った結果である。CX総費用比は約2.989。[R1からの独立算術]

最低lambdaのfactor0 frameは、1組RZの標本平均126に対してframe_identityは72。shotは7075対7151と差が小さい。lambdaを最小にする基底と費用を最小にする基底が異なる理由の一部を、費用の積の形で説明できる。

### 3.2 8標本とrare-eventの扱い

native all-R q1では \(\tau\simeq.360165\)。K=2のTaylor order2選択確率は

\[
p_2=\frac{\tau^2\sqrt{1+(\tau/3)^2}/2}{B(\tau)}\simeq0.05790.
\]

独立8抽出でorder2を一度も得ない確率は \((1-p_2)^8\simeq0.62054\)。この値は分布式からの計算であり、元sample列の実現についての断定ではない。

よって、8件の内部で計算したSEが小さくても、高費用の稀なevent群を見落とす可能性がある。Aの小さいRZ差を確定するには、Taylor次数別の費用を組み合わせるなど、構造を用いた期待費用確認の方が、無目的なseed追加より情報価値が高い。

q1の12-involution辞書・K2は、未統合でorder0が12、order2が12^3、計1740 eventである。この有限性は参考になるが、全件compileを自動的な必須条件にはしない。等価classの集約、次数別の厳密部分＋限定標本、上界等をCodexが選べる。乱択法自体をTrack Bの代わりに改造する話ではない。

sourceでは同じtrial/q seedを複数constructorで使うため、候補間費用は独立とは限らない。X/Yのpaired SEと、候補差のSEは異なる。候補差の評価時にはこのcouplingを保持し、独立候補の分散和を無条件に使わない。[S1]

### 3.3 core効果と辞書効果が未分離

native all-Rは、DFの各固有frame内のI/Z/ZZ由来involutionを保持する。一方、frame構成の残差は、共通frameでPauli係数を積算・統合している。[S1]

従って比較には少なくとも、(a) 決定論coreの導入、(b) 乱択辞書の変更、(c) basis移動の変更が含まれる。これは比較が不正という意味ではなく、全体構成の局所改善をどのアルゴリズム上の新要素へ帰属できるかが未確定という意味である。

最も直接的な対照は、同じ係数代数でH全体をPauli辞書へ集めたall-Rである。これは既存sourceのPauli積演算から生成でき、分子を増やす前に確認できる。GPTの有理数計算では、この固定Hのidentity以外のPauli係数1ノルムは **0.979**、identity係数は0.444、非identity項は14。native all-Rのlambda .900412171より大きいので、Pauli all-Rが必ず安いとは推定しない。basis費用が減る側面とnormalizationが増える側面を、実際に比較する。

### 3.4 独立導出：安価な対角項が残差へ残っている

現在のidentity frame coreは

\[
D_0=\sum_l[d\Gamma(\operatorname{diag}g_l)]^2
\]

である。しかし、factorの非対角部分の平方にも対角項が生じる。

\[
T_{ij}=a_i^\dagger a_j+a_j^\dagger a_i,\qquad
T_{ij}^2=n_i+n_j-2n_i n_j.
\]

今回のHのFock基底対角部分 \(D_*=\operatorname{diag}_{\rm occ}(H)\) について、正確に

\[
D_*-D_0=0.0081(n_0+n_1-2n_0n_1)
 +0.0049(n_1+n_2-2n_1n_2)
\]

となる。差は既存coreと同じI/Z/ZZ支持上にあり、新しいPauli支持を追加しない。

残りは

\[
R_*=H-D_*=(0.0405+0.027 n_2)T_{01}
 +(0.0105-0.028 n_0)T_{12}.
\]

この式は同じ固定入力をSymPyの有理数行列として再構成して確認した。元binary64配列をbyte単位で再実行したものではない。

identityを別相対位相へ分けたPauli辞書で

\[
\lambda(R_0)=0.0915,\quad \lambda(R_*)=0.085,
\]

非identity残差項数は10から8へ変わる。これは正確な再配分でHamiltonian近似を導入しない。ただしD/Rの交換子・PF biasは変わるため、総費用の改善を意味しない。

一般にも、Hermitian gの \(d\Gamma(g)^2\) の占有基底対角部分は

\[
\big(\sum_i g_{ii}n_i\big)^2
 +\sum_{i<j}|g_{ij}|^2(n_i+n_j-2n_i n_j)
\]

として係数から計算できる。すなわち、この対照は指数次元のdense diagonalを取得しないと構成できないものではない。

対角成分をcoreへ集めること自体を新規algorithmと呼ばない。まず基本的な基準を揃える操作である。今回の約7.1%の残差lambda低下だけを追って新テーマを作るのも適切でない。前述のshot floorにより、現windowのshot改善余地自体は小さい。

### 3.5 Aの研究上の着地点

今後の問いは「全対角coreを作ればよいか」ではなく、**どのframe・実装可能なblock・残差表現を選ぶと、費用を含めた利益が残るか**である。既知frame選択や係数normだけの最適化を越えた構成差が必要になる。

一つの小さい例においてRZ/CX双方が改善することも、全分子で勝つことも必須ではない。科学的に意味のある入力classや実装条件で、再利用できる構成法と改善の説明を示すことが目標である。

今回のAはstateに一粒子を使う一方、bias診断はfull Fock normを課している。一粒子sectorでは \(H\) は \(g_0^2+g_1^2\) のone-body発展へ還元できるため、固定state信号だけのtaskとfull-Fock operator taskは区別する。これは現在のfull-space恒等式を否定するものではなく、相関した電子系への価値をまだ検証していないという限定である。既定のoperator精度条件を後から黙ってsector条件へ変更しない。

## 4. Cの評価：現在の構成の優先度を下げる

今回の8 contextでは、保存されたq grid内のRZ最良点を比較して、対称S2がTHRIFTよりRZ/CX両方で安い。THRIFT/S2のRZ比は約2.64～8.36、CX比は約2.40～7.57。deterministic circuitなので、A/Bのような有限cost-sampleによるSEはない。[R1からの算術]

ただし、これは次を意味しない。

- すべての時間、精度、alphaでTHRIFTが不利。
- S2が全手法に対して最良。最も易しいold_two/.1/.05では通常一次PFの方が安い。
- 近可積分構造を使う方法に新規研究の余地がない。

現scopeではS2がq1で精度を満たすcontextが多く、THRIFTの小さいbiasで減らせる反復・shotが少ない。一方mixed accessの回路負担を払うため、利益が残らない。

これはTHRIFT原論文の主張と矛盾しない。原論文は、mixed evolutionを安く実装できる場合を重視し、モデルによってはその実装費用で有利性が失われることも明示する。今回のconditional multiplexerでの負けをTHRIFT一般の反証として扱わない。[L2]

**研究判断：現在のknown conditional-SU(2) builderで勝てるalpha/T/epsilonを探すだけの追加gridは優先しない。** 新しいmixed block生成、共有演算の消去、廉価な別accessなどを定義し、その変更が既知法と異なるかを説明できた場合に再開する。既知のcrossoverの再発見だけに終わるなら、今回の新規algorithm研究の中心にしない。

Cの全sourceを破棄する必要はない。安価なmixed primitive、同精度対照、非適格rowの扱いを確認する回帰・比較基盤として残す。

## 5. Bの評価：isometry接続の成功と圧縮利益を区別する

Bは、物理3＋補助1軌道のVと拡大対角二体Dから

\[
\widetilde H=\mathcal U(V)D\mathcal U(V)^\dagger,
\qquad H_{\rm phys}=P\widetilde HP
\]

を作り、独立なnormal-ordered quarticと照合している。単純なPauli toyだけだった前段から、isometric THC型の構造へ接続した点は支持できる。[S1,R1]

ただし、特定分子の圧縮fittingや、圧縮率の規模利益を示したものではない。今回のq1/epsilon=.05の保存点では、反射側のshot7544は直接Pauliの7643より約1.3%少ないだけで、RZ総費用は約4.90倍、CXは約3.95倍。量子bitも5対4である。これらは8cost drawsに基づく点比較であり、期待値の一般的dominanceの認証ではない。[R1]

1組の回路費用は、反射側の標本平均RZ94.25、直接Pauli19。反射側のnormalization/biasを完全に理想化しても、同方策のshot削減は約7.05%までである。したがって現在の1-shot費用を保ったままlambdaのみを改善して、この差を埋める方針は合理的でない。

また、今回補助軌道は1本で反射はZ_auxである。これを多補助軌道へ増やせば自動的に有利になるという推論はできない。

**従来Bの現構成は拡張を保留する。** 平均第一momentの保存は正しいが、それだけでは新規algorithmの資源利益を支持しない。既知resetを比較へ入れる場合も、state/channel誤差と第一moment信号誤差を同じものとして費用を流用しない。Luo–Ciracの基本taskはstate/channelであり、その違いがある。[L3]

## 6. B′：投影後の物理空間で直接扱うルート

### 6.1 何を変えるか

Bの元の問いは「拡大空間で安くできる表現を使い、漏洩を抑える」である。しかし、今回のsourceが既に持つnormal-ordered physical表現から、**物理空間内で実装可能な2軌道項へ分解する**別経路も得られる。

これはBを成功と読み替えるものではなく、外部Hamiltonian表現とPRの接続を作り直す候補である。内部のsampling lawやreturn aggregationを変更する必要はない。

### 6.2 2本の非直交軌道の直交化

isometryの列に対応する

\[
c_\alpha=\sum_i x_i a_i,\qquad c_\beta=\sum_i y_i a_i
\]

を考える。係数ベクトルが線形独立なら、Gram–Schmidtにより正規直交する2モードb1,b2を選べる。次を定義する。

\[
\Delta_{\alpha\beta}=\|x\|^2\|y\|^2-|x^\dagger y|^2\ge0.
\]

CARと反対称性から

\[
c_\alpha^\dagger c_\beta^\dagger c_\beta c_\alpha
=\Delta_{\alpha\beta}\, n_{b_1}n_{b_2}.
\]

証明は、\(c_\alpha=\|x\|b_1\)、\(c_\beta=\eta b_1+\zeta b_2\) と置き、同一モードのcreation/annihilationの平方が0となる項を除くことで得られる。係数は \(\|x\|^2|\zeta|^2=\Delta\)。列が従属ならDelta=0で元quarticも0。

これは「2軌道の同時占有射影」であり、full Fock spaceでrank1の演算子だという意味ではない。二粒子sectorではrank1に対応するが、一般N粒子へその表現を無条件に流用しない。

今回のsourceが使うpair weightsをw_abとすると

\[
H_{\rm phys}=\sum_{\alpha<\beta}\omega_{\alpha\beta}P_{\alpha\beta},\quad
\omega_{\alpha\beta}=w_{\alpha\beta}\Delta_{\alpha\beta},\quad
P_{\alpha\beta}=n_{b_1}n_{b_2}.
\]

各項はGaussian basis変換、2軌道占有へのphase、逆変換として実装できる。補助軌道は不要。ただしHadamard ancillaや、将来のgate synthesis用workspaceまで不要という意味ではない。

### 6.3 PRへ渡すinvolution辞書

\[
Q_{\alpha\beta}=I-2P_{\alpha\beta},\quad
Q_{\alpha\beta}^\dagger=Q_{\alpha\beta},\quad Q_{\alpha\beta}^2=I,
\]

なので

\[
H_{\rm phys}=\frac12\Big(\sum_{\alpha<\beta}\omega_{\alpha\beta}\Big)I
 -\frac12\sum_{\alpha<\beta}\omega_{\alpha\beta}Q_{\alpha\beta}.
\]

scalarを正しく相対位相として戻せば、既存のinvolution RTEへ接続できる。Qは局所のCZにGaussian basisを掛けたものになる。RTE中のQ回転はoccupancy phaseとして実装し、制御相対位相を保持する必要がある。basisをHadamard ancillaに無制御、中心作用のみ制御という既存規則は使える。

この辞書の未統合係数1ノルムは

\[
\lambda_{\rm pair}=\frac12\sum_{\alpha<\beta}|w_{\alpha\beta}|\Delta_{\alpha\beta}.
\]

単一の非自明な射影項 \(\omega P\) に限れば、自由なscalar shift後の最小operator normは|omega|/2であり、任意unitary LCUの係数1ノルムはそれ以上。この1-involution構成はその値を達成する。**これは単一項の既知のspectral centeringに関する事実であり、全Hamiltonianの最適性ではない。** 異なるpair間の相殺を考える余地は残る。

### 6.4 同じisometry入力での独立チェック

sourceと同じ順序でVへ右からGivensを掛け、top3rowsをuとし、exterior-power minorsでFock作用を構成した。normal-ordered quartic、pair projector、involution＋scalarの三通りを同じfixed inputで照合した。Qiskitや元project moduleは使っていない。

| pair | w | Delta | omega=w Delta |
|---|---:|---:|---:|
| 0,1 | .7 | .9761271242968684 | .6832889870078078 |
| 1,2 | .4 | .75 | .3 |
| 2,3 | .25 | .02387287570313154 | .005968218925782885 |
| 0,3 | -.2 | .25 | -.05 |

保存sourceのfloat入力に合わせた独立binary64計算では、quartic投影照合約2.22e-16、pair総和照合約7.06e-17、projector-involution再構成約2.22e-16。これらは形式的証明ではなく、上の代数の固定入力検査である。

| 表現 | lambda（今回固定入力） |
|---|---:|
| batchの拡大反射辞書 | .9625 |
| batchの直接Pauli辞書 | 1.045353063 |
| 物理pairを3本のZ/ZZへ展開した未統合辞書 | .779442904450193 |
| 物理pairのprojector-involution辞書 | .5196286029667954 |

最後の2値は今回GPTが導出・計算したもので、元batchの性能結果ではない。scalar shiftが変わるのでfinite Taylor biasも新しく評価する必要がある。lambdaが低いことだけでshots・total costの改善を認定しない。

### 6.5 新規性と失敗リスク

直交化、Gram determinant、projectorからinvolutionへの変換は標準的な代数である。THCの非直交基底、fermionic fragments、LCUとの先行関係を調べずに、これらの恒等式だけを新規algorithmと呼ばない。[L3,L6,L7,L8]

本レビューでは、**この正確な一連の構成と同一な既報がないことまでは確認できていない**。検索で一致を見付けないことを新規性の証明にしない。

研究的な可能性は、こうした物理pairを入力から安く生成し、同一frameのblockへまとめるか、native pairを乱択するかを構成的に選べる場合にある。一方、pairごとのbasis変換が増えれば、拡大空間で一度に多数のdensity termsを処理する利益を失う。補助軌道を消す代わりにbasis移動・非可換分割を増やすtrade-offである。

Deltaが小さい列の直交化は数値的にill-conditionedになり得る。実装では安定なQR等を用い、近零を無条件に切り捨てない。今回4pairsの結果だけで複素入力や大系へのnumerical robustnessを認定しない。

同時に、direct Pauliが今回dense参照から作られたことを理由に、全ての直接実装を指数計算oracle扱いしない。二体テンソルの係数からPauliや上記pairを生成することは、多項式の係数演算で可能である。ただし古典費用や出力サイズが小さいことを自動保証するわけではない。

**B′の位置づけ：まず既知法との差とnative費用を確認する小規模候補。新規性・資源利益・独立論文の中心課題としての採択は未確定。**

## 7. 先行研究から見た位置づけ

### 7.1 既知の大枠

RC-DFは資源関連の係数normを正則化して圧縮factorizationを求める。従って「factorizationをlambdaで最適化する」という一般論は新規とは言えない。[L5]

Martínez-Martínezらは、fermionic/qubit partitioningの誤差とgate費用を比較し、誤差が小さい分解でもstep当たりgate費用で不利になり得ることを扱っている。今回の「誤差改善＝総費用改善ではない」という観察だけを新規な中心claimにはできない。[L6]

SPRINT/GRADEは、階層的な分解・近可積分性・乱択・symmetry protection・残差の明示Pauli表現を含む。本文のSec.II.Bではfragment fitting後の残差項の保持を記述する。従って「構造のよいcore＋残差」という大枠だけでも不十分である。[L4]

ただしSPRINTの乱択と、本研究のfinite-RTEによる第一moment/normalizationは同一の処理ではない。共通要素があることは、今回のすべてのPR-specific構成が既に解決されていることを意味しない。比較すべきものは入力、生成するoperator、誤差の意味、費用、実装可能なprimitiveである。

### 7.2 Bの動機は残るが、現在の実装利益は別

GRADE本文では、圧縮によるstep費用の改善を、補助空間への漏洩誤差が打ち消す例を報告している。この点はBの問題設定の動機を支える。しかし、その問題が存在することと、今回の反射RTEが最終費用で解決したことは別である。[L4]

Luo–Ciracは拡大基底でのisometryとresetを用いたstate/channel simulationを提案している。Bがfirst momentを目的にすることで別の構成余地はあり得るが、taskを変えた分だけ有利に見せる比較をしてはならない。[L3]

### 7.3 本レビューで用いた文献の確認範囲

2026-10-10時点でPR原論文はarXiv v2（2026-07-10）を持つことを公式abstractで確認した。実RTE意味論は固定repository sourceを直接読んで評価し、原論文v2全Appendixを新たに逐語監査したとはしない。[L1]

SPRINT/GRADEはarXiv 2606.30741v1のabstractとPDF parsed textを確認した。HTML経路が404となったためPDFへ切り替えた。PDFの該当ページのscreenshot取得を複数回試みたが、サービス側Cache miss/Internal Errorで画像を取得できなかった。そのため図の細かい値や視覚的な形の再解釈は行わず、Sec.II.BとSec.V.Bの本文に記載された構成・限界だけを参照した。

ITHCは2407.04432v2 HTML、THRIFTはNature Communications本文、RC-DFはQuantumの一次掲載ページ、partitioning/solvable fragments/THCはarXiv等の一次資料を参照した。網羅的な優先権監査ではない。

## 8. 新しい研究の中心をどう考えるか

現段階で「Aを主論文に決定」とはしない。現実的な中心問題は次の形である。

> **Hamiltonian表現から、決定論でまとめて実行するblockと、乱択に渡すprimitiveを、そのbasis費用と誤差を含めて構成する方法を作れるか。**

Aは共通frameを開く費用を払い、残差量を減らす。B′は補助空間を使わず、物理pairへ分けるが、pairごとのbasisを払う。この両者には関連があるが、一つに統合することを義務にはしない。外部だけで成立する構成法が得られれば、それでよい。

有望性を判断する単位を、以下へ移す。

- 「lambdaが下がった」ではなく、**その低下に払うbasis/primitive費用を含めた価値**。
- 「既知式が再現できた」ではなく、**入力から新しい有用なblockを生成できる手順**。
- 「toyで一つ勝った」ではなく、**比較条件を揃えても残る機構と、適用範囲**。

未知の最適解を完全に予測するselectorや普遍的保証を、最初の必須成果にしない。実装可能な構成、明確な先行差分、限定されたが有意味な適用範囲での効果があれば、算法研究の候補になる。

逆に新規性が残らなければ、Track Aと同様のベンチマーク研究に黙って目的を切り替えない。その時点で別の構成自由度を探索するか、適用条件・限界の研究へ明示的に変更するかを判断する。

## 9. 次のCodex作業の科学的範囲

本レビューからの推奨は、**既存固定入力を中心とした、比較要因の切り分けと最小の代替primitive検証を一つの作業単位にまとめること**である。次担当はCodex。同じレビューを直ちに繰り返す必要はない。

### 9.1 Aの一体作業

1. 同じHを係数代数で集めたall-Pauli RTEを加える。既存native all-Rは残す。
2. 現coreと、Hの全占有対角部分を入れたcoreを比較する。対象演算子・相対位相を保存し、既存source/resultを変更しない。
3. 現8drawの点差を勝敗にするのではなく、次数別の期待費用等を使って必要な範囲の不確かさを確認する。候補間に同じseedを使った依存関係も残す。
4. RZ/CX・depth・basis移動・normalization・bias・Nを分解して、core固有の改善が残るかを示す。

これは現行Aの成功を作るために有利な条件を探索する作業ではない。同じ表現・安い既知対照を入れた結果、差が消えることも重要なoutcomeである。

### 9.2 B′の一体作業

1. fixed isometryからphysical pair/projectorを構成し、上記恒等式とscalar shiftを独立確認する。
2. 可能ならprojector-involutionのnative rotation/productを小さい回路で構成する。RTE内部sampling/returnは変更しない。
3. 従来反射、直接Pauli、physical pairを同じ第一moment信号精度と制御・状態準備の会計で比較する。lambdaだけのランキングをしない。
4. THC、fermionic fragment、LCUの既知構成との差を限定的に照合し、同値ならそのまま記録する。

新規性が否定されれば、B′を比較用primitiveとして残すことはできるが、そのまま新規研究の主テーマとはしない。複雑な分子compression fittingや大規模追加計算を自動的に開始しない。

### 9.3 Cの扱い

既存結果・constructor・強い対照は維持する。現構成で勝つparameterを探すための追加scanは今回の推奨範囲に含めない。具体的な新しいcheap mixed-accessの構成案が得られた時点で再検討する。

### 9.4 共通の保全・裁量・終了条件

source/result/freeze/他Track/既存dirty差分を保護する。新しい独立branch/worktreeで行い、既に公開されている同一証拠を再pushさせない。source、primary result、失敗、audit、remote取得先を結ぶ。科学的な意味を変えない通常の実装・test修正はまとめてCodexへ任せ、細部ごとのGPT承認は求めない。

次のいずれかで研究判断へ戻す。

- 適切な対照と不確かさを入れても、構成固有の資源差が残った。
- 差が既知の辞書・compile上の改善へ還元され、中心仮説を修正する必要が出た。
- B′が既知構成と同値、またはbasis費用によって利益を失うことが判別できた。
- 分子・大系・新しい中心taskへ進む科学的判断が必要になった。

何も新しい改善機構が残らなければ、A/B′を無期限に繰り返すのではなく、候補探索を広げる。その際も、検証していない研究分野全体へ一般的不可能性を主張しない。

このレビューは科学的推奨を示す。実際の次batchの起動・資源割当・Git操作は、利用者からCodexへの作業指示と、その環境における既存運用条件に従う。

## 10. 主張の状態表

| 主張 | 状態 |
|---|---|
| 今回の限定構成・同一信号精度の会計がsourceに実装されている | 確認した |
| 374回路の保存IRを別実装で照合した記録がある | 確認した。GPTによる全件再実行とは区別 |
| Aのframe_identityが一般にPRを高速化する | 未確定 |
| AのRZ/CXのtrade-off自体が研究失敗である | 否定する。採用価値は条件・不確かさ・新規性次第 |
| Aではcore導入と辞書変更が同時に起こっている | sourceから確認した |
| 全対角coreによる追加の正確な再配分が可能 | 独立代数で確認した。total cost未検証 |
| Cの現構成は今回の8 contextでS2より低費用 | 支持されない |
| THRIFT一般が無価値 | 支持されない |
| Bのisometry接続は成立する | 保存検査とsourceに整合 |
| Bの圧縮利益が実証された | 支持されない |
| physical pairを補助軌道なしのprojectorへ変換できる | 条件を明示した代数と固定入力検査で確認 |
| B′が世界で初めて／総費用で最良 | 未確定。主張しない |
| 中心研究テーマを一つに採択した | していない |

## 11. 参照資料

### Repository evidence

- [R1] [限定構成比較report](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/docs/research/representation_construction_comparison_results_20261010.md)
- [R2] [一次result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/artifacts/representation_construction_comparison/2026-10-10/run1/result.json) — Git blob `68f800076847f1f24aa466249885cf096a9aab9d`、保存記録のSHA256 `f7cadf45da0809a7ebf98f12cc14b3dc7e5486ddccb8bf54cedb42b0b3edd821`、1,580,001 bytes。
- [R3] [native_ir.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/artifacts/representation_construction_comparison/2026-10-10/run1/native_ir.json) — Git blob `016bed2f00d90cf9486ea3c861c92f0d862e553c`、保存記録のSHA256 `def7eba9821e766e5f0eb1d85afd9a0c3530e7d9ddf814723640e368ed770be2`、15,649,041 bytes。
- [R4] [remote retrieval receipt](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/artifacts/representation_construction_comparison/2026-10-10/remote_retrieval_receipt.json)
- [S1] [科学source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce99b57166a9f422f151f56fa17ed799cd9909dd/src/trottertracks/representation_exploration/construction_comparison.py) — Git blob `e68b2d9f1896881e6b7867b373cdfe9fda5891dd`。
- [P1] [事前固定scope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce99b57166a9f422f151f56fa17ed799cd9909dd/docs/research/representation_construction_comparison_scope.md)
- [V1] [保存値verifier](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/scripts/verify_representation_construction_comparison.py)
- [V2] [保存監査結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/artifacts/representation_construction_comparison/2026-10-10/saved_evidence_audit.json)
- [V3] [tests v3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/artifacts/representation_construction_comparison/2026-10-10/tests_v3.log)、[run audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/artifacts/representation_construction_comparison/2026-10-10/run1/run_audit.json)
- [M1] [前回科学レビューの保存原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/docs/research/representation_construction_inputs/representation_exploration_independent_review_2026-10-10.md)

### Primary literature

- [L1] Günther et al., *Phase estimation with partially randomized time evolution*, [arXiv:2503.05647](https://arxiv.org/abs/2503.05647)。v2の公開日2026-07-10を公式abstractで確認。
- [L2] *Efficient and practical Hamiltonian simulation from time-dependent product formulas*, Nature Communications (2025), [DOI:10.1038/s41467-025-57580-5](https://www.nature.com/articles/s41467-025-57580-5)。THRIFTのprimitive費用とregime依存性を参照。
- [L3] Maxine Luo and J. Ignacio Cirac, *Efficient Simulation of Quantum Chemistry Problems in an Enlarged Basis Set*, [arXiv:2407.04432v2](https://arxiv.org/html/2407.04432v2)。isometry Eq.(4)–(7)、reset/state task。
- [L4] Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, [arXiv:2606.30741v1 PDF](https://arxiv.org/pdf/2606.30741v1)、[公式abstract](https://arxiv.org/abs/2606.30741)。Sec.II.B、Sec.V.B、Appendix B。
- [L5] Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum 8,1371 (2024), [DOI:10.22331/q-2024-06-13-1371](https://quantum-journal.org/papers/q-2024-06-13-1371/)。
- [L6] Martínez-Martínez, Yen and Izmaylov, *Assessment of various Hamiltonian partitionings for the electronic structure problem on a quantum computer using the Trotter approximation*, Quantum 7,1086 (2023), [arXiv:2210.10189v2](https://arxiv.org/abs/2210.10189)。
- [L7] Lee et al., *Even more efficient quantum computations of chemistry through tensor hypercontraction*, [arXiv:2011.03494](https://arxiv.org/abs/2011.03494)。THCの非直交基底とblock encodingの近接研究。今回B′との完全同値性を認定した文献ではない。
- [L8] *Extension of exactly-solvable Hamiltonians using symmetries of Lie algebras*, [arXiv:2305.18251](https://arxiv.org/abs/2305.18251)。可解fragment構成の近接研究。

## 付録A. 独立チェックの範囲と再現方法

以下のコードは、GitHubで読んだ固定sourceの入力行列・isometryパラメータ、およびreport表示値を手動転記し、SymPy/NumPyで独立に計算する。元result全体のdownload cloneではなく、元runnerの再実行でもない。Qiskit、project module、乱択sample、compile、分子計算を使用しない。

1. Aの0.01単位の小数を有理数として再構成し、完全なH・core・残差のPauli係数を求める。
2. Bの固定isometryを同じ右掛け順序で生成し、normal-order quarticとpair/projector/involutionを照合する。
3. 保存表示値を用いた費用因数分解、RTE order2の選択確率、同一shot方策内のfloorを計算する。

下のPythonをファイルへ保存し、SymPyとNumPyのある環境で実行すればよい。出力JSONの主要計算結果も続けて収録する。コード自体が新規性や資源優位を認証するものではない。


### 再現用コード

保存名：`independent_batch2_checks.py`。実行：`python independent_batch2_checks.py`。同名のJSONを隣に出力する。

```python
"""Reviewer algebra on fixed batch-2 inputs, not the Codex runner.
Inputs were transcribed from source ce99b57166a9f422f151f56fa17ed799cd9909dd.
No Qiskit/project import, sampling, synthesis, molecular data, or new parameter sweep.
A uses exact rational arithmetic; B uses binary64 to check a derived identity.
"""
from __future__ import annotations
from itertools import product, combinations
from pathlib import Path
import hashlib, json, math, platform
import numpy as np
import sympy as sp

def jw(n: int):
    out=[]
    for j in range(n):
        a=sp.zeros(2**n)
        for k in range(2**n):
            if (k >> j)&1:
                a[k ^ (1 << j),k]=(-1)**((k & ((1 << j)-1)).bit_count())
        out.append(a)
    return out

def lift(g,a):
    return sum((g[p,q]*a[p].H*a[q] for p in range(len(a)) for q in range(len(a))),sp.zeros(a[0].rows))

PAULI={'I':sp.eye(2),'X':sp.Matrix([[0,1],[1,0]]),'Y':sp.Matrix([[0,-sp.I],[sp.I,0]]),'Z':sp.diag(1,-1)}
def coeffs(h,n):
    c={}
    for word in product('IXYZ',repeat=n):
        p=sp.kronecker_product(*(PAULI[x] for x in word))
        v=sp.simplify(sp.trace(p*h)/2**n)
        if v!=0:c[''.join(word)]=v
    return c

def rational_dict(c):return {k:str(v) for k,v in c.items()}
def l1(c,identity=True):return sum((abs(v) for k,v in c.items() if identity or k!='I'*len(k)),sp.Integer(0))

def check_a():
    rat=sp.Rational;a=jw(3);ns=[x.H*x for x in a]
    g0=sp.Matrix([[rat(8,10),rat(9,100),0],[rat(9,100),rat(-35,100),0],[0,0,rat(15,100)]])
    g1=sp.Matrix([[rat(-2,10),0,0],[0,rat(6,10),rat(7,100)],[0,rat(7,100),rat(-45,100)]])
    gs=[g0,g1];h=sum((lift(g,a)**2 for g in gs),sp.zeros(8))
    ds=[sp.diag(*g.diagonal()) for g in gs]
    d0=sum((lift(d,a)**2 for d in ds),sp.zeros(8))
    dstar=sp.diag(*h.diagonal());r0=h-d0;rstar=h-dstar
    t01=a[0].H*a[1]+a[1].H*a[0];t12=a[1].H*a[2]+a[2].H*a[1]
    extra=rat(81,10000)*(ns[0]+ns[1]-2*ns[0]*ns[1])+rat(49,10000)*(ns[1]+ns[2]-2*ns[1]*ns[2])
    roff=(rat(405,10000)*sp.eye(8)+rat(27,1000)*ns[2])*t01+(rat(105,10000)*sp.eye(8)-rat(28,1000)*ns[0])*t12
    assert dstar-d0==extra
    assert rstar==roff
    assert dstar+rstar==h
    assert dstar==sp.diag(*h.diagonal())
    spectra=coeffs(h,3);old=coeffs(r0,3);new=coeffs(rstar,3)
    # Particle number preservation applies to the sum, not each individual Pauli.
    N=sum(ns,sp.zeros(8));assert h*N-N*h==sp.zeros(8)
    hc=np.array(h,dtype=complex)
    return {'scope':'exact rational identities on original fixed A input; no costs recalculated',
        'full_H_paulis':rational_dict(spectra),'old_core_paulis':rational_dict(coeffs(d0,3)),
        'complete_diagonal_core_paulis':rational_dict(coeffs(dstar,3)),
        'old_residual_paulis':rational_dict(old),'new_residual_paulis':rational_dict(new),
        'old_residual_nonidentity_l1':str(l1(old,False)),
        'new_residual_nonidentity_l1':str(l1(new,False)),
        'full_H_collected_nonidentity_l1':str(l1(spectra,False)),
        'full_H_identity':str(spectra['III']),
        'extra_diagonal_absorbed_nonidentity_l1':str(l1(coeffs(extra,3),False)),
        'new_diagonal_supports':sorted(set(coeffs(dstar,3))-set(coeffs(d0,3))),
        'full_H_operator_norm_diagnostic':float(np.linalg.norm(hc,2)),
        'one_particle_effective_matrix':[[str(x) for x in row] for row in (g0*g0+g1*g1).tolist()],
        'one_particle_scope_note':'The fixed prepared state has one particle; an effective one-body evolution is task-equivalent on that sector only, not full Fock operator-equivalent.',
        'all_identity_checks':True}

def fock(u):
    n=len(u);d=2**n;out=np.zeros((d,d),complex)
    for i in range(d):
        r=[p for p in range(n) if (i>>p)&1]
        for j in range(d):
            c=[p for p in range(n) if (j>>p)&1]
            if len(r)==len(c):out[i,j]=np.linalg.det(u[np.ix_(r,c)]) if r else 1
    return out

def check_b():
    n=3;m=4;v=np.eye(m)
    for i,j,angle in [(0,1,math.pi/8),(2,3,math.pi/6),(1,2,math.pi/10)]:
        g=np.eye(m);g[i,i]=g[j,j]=math.cos(angle);g[i,j]=-math.sin(angle);g[j,i]=math.sin(angle)
        v=v@g  # right multiplication as in the frozen source
    u=v[:n,:];edges=[(0,1,.7),(1,2,.4),(2,3,.25),(0,3,-.2)]
    a=[np.array(x,dtype=complex) for x in jw(n)];N4=[np.array(x.H*x,dtype=complex) for x in jw(m)]
    diagonal=sum((w*N4[i]@N4[j] for i,j,w in edges),np.zeros((16,16),complex))
    gamma=fock(v);h=(gamma@diagonal@gamma.conj().T)[:8,:8]
    target=np.zeros_like(h);pair_total=np.zeros_like(h);pairs=[];q_reconstruction=np.zeros_like(h)
    for i,j,w in edges:
        x=u[:,i];y=u[:,j];cx=sum((x[p]*a[p] for p in range(n)),np.zeros((8,8),complex));cy=sum((y[p]*a[p] for p in range(n)),np.zeros((8,8),complex))
        term=cx.conj().T@cy.conj().T@cy@cx
        det=float((np.vdot(x,x)*np.vdot(y,y)-abs(np.vdot(x,y))**2).real)
        e=x/np.linalg.norm(x);z=y-e*np.vdot(e,y);f=z/np.linalg.norm(z)
        b1=sum((e[p]*a[p] for p in range(n)),np.zeros((8,8),complex));b2=sum((f[p]*a[p] for p in range(n)),np.zeros((8,8),complex))
        nt1=b1.conj().T@b1;nt2=b2.conj().T@b2;pair=det*nt1@nt2
        residual=float(np.linalg.norm(term-pair,2))
        assert residual<1e-13
        target+=w*term;pair_total+=w*pair
        projector=nt1@nt2; q=np.eye(8)-2*projector
        assert np.linalg.norm(q@q-np.eye(8),2)<1e-13
        q_reconstruction+=(w*det/2)*(np.eye(8)-q)
        pairs.append({'indices':[i,j],'w':w,'gram_determinant':det,'weighted_pair':w*det,'pair_identity_residual':residual})
    assert np.linalg.norm(target-h,2)<1e-13
    abs_sum=sum(abs(p['weighted_pair']) for p in pairs)
    return {'scope':'same fixed isometry, real orbital entries, binary64 verification of derived exact pair identity',
        'isometry_residual':float(np.linalg.norm(u@u.conj().T-np.eye(n),2)),
        'quartic_projection_residual':float(np.linalg.norm(target-h,2)),
        'physical_pair_reconstruction_residual':float(np.linalg.norm(pair_total-h,2)),
        'projector_involution_reconstruction_residual':float(np.linalg.norm(q_reconstruction-h,2)),
        'projector_involution_unmerged_l1':.5*abs_sum,
        'projector_involution_scalar_offset':.5*sum(p['weighted_pair'] for p in pairs),
        'pair_rows':pairs,'pair_coefficient_l1':abs_sum,
        'unmerged_pair_pauli_nonidentity_l1_upper':.75*abs_sum,
        'unmerged_pair_identity_sum':.25*sum(p['weighted_pair'] for p in pairs),
        'compiled_cost_evaluated':False,'dominance_claim':False}

def stats():
    lam=.9004121714431494;t=.4;tau=lam*t
    b0=math.sqrt(1+tau*tau);b2=tau*tau/2*math.sqrt(1+(tau/3)**2);b=b0+b2;p2=b2/b
    floor={str(e):math.ceil(2/(e/math.sqrt(2))**2*math.log(80)) for e in (.05,.02)}
    # Saved report numbers; no claim of a new independent cost estimate.
    l0={'rz':603525,'cx':194985,'N':9285};identity={'rz':514872,'cx':582806.5,'N':7151}
    refs=[(363426,285040,1024628,729872),(2327946,1825840,6412320,4567680),
          (370617,290680,1027402,731848),(2446980,1919200,6455974,4598776),
          (608839,537676,5090400,4072320),(4623619,4083196,32214960,25771968),
          (718179,634236,5220000,4176000),(6291516,5615568,34335360,27468288)]
    return {'rare_event':{'tau':tau,'normalization':b,'order2_probability':p2,'probability_no_order2_in_8':(1-p2)**8,
                         'expected_order2_in_8':8*p2,'unmerged_single_step_event_count':12+12**3},
            'shot_policy_bias0_normalization1_floor':floor,
            'A_report_factorization':{'RZ_ratio_identity_to_allR':identity['rz']/l0['rz'],
                'CX_ratio_identity_to_allR':identity['cx']/l0['cx'],'N_ratio':7151/9285,
                'RZ_cost_pair_allR':603525/9285,'RZ_cost_pair_identity':514872/7151,
                'CX_cost_pair_allR':194985/9285,'CX_cost_pair_identity':582806.5/7151,
                'warning':'Same seeds are reused across candidates; no between-candidate independence or significance assumed'},
            'C_THRIFT_to_S2_ratios':[{'rz':c/a,'cx':d/b} for a,b,c,d in refs],
            'B_report_ratios':{'rz':711022/145217,'cx':226320/57322.5,'shots':7544/7643},
            'fixed_policy_headroom':{
                'A_frame_identity_max_relative_shot_saving':1-floor['0.05']/7151,
                'B_reflection_max_relative_shot_saving':1-floor['0.05']/7544,
                'B_report_mean_pair_rz':711022/7544,
                'B_report_mean_pair_cx':226320/7544,
                'A_factor0_report_mean_pair_rz':891450/7075},
            'new_quantum_shots_or_compilations':0}

def main():
    out={'provenance':{'source_commit':'ce99b57166a9f422f151f56fa17ed799cd9909dd',
            'result_commit':'c24dcede10ed9726e1e0b1030eca749ba7bc5cba',
            'inputs':'manual exact transcription from fetched source and report; not a downloaded full-result clone',
            'scope':'independent reviewer checks, not preregistered Codex results'},
         'A':check_a(),'B':check_b(),'accounting':stats(),
         'review_runtime':{'python':platform.python_version(),'numpy':np.__version__,'sympy':sp.__version__},
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    p=Path(__file__).with_suffix('.json');p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
```

### 実際の独立チェック出力

以下はGPT側の実行結果であり、Codexの原resultへの追記・再分類ではない。実行環境は元batchとは異なる。Aは有理数の恒等式、Bは同じパラメータからのbinary64照合、資源算術は保存表示値の転記に基づく。

```json
{
  "provenance": {
    "source_commit": "ce99b57166a9f422f151f56fa17ed799cd9909dd",
    "result_commit": "c24dcede10ed9726e1e0b1030eca749ba7bc5cba",
    "inputs": "manual exact transcription from fetched source and report; not a downloaded full-result clone",
    "scope": "independent reviewer checks, not preregistered Codex results"
  },
  "A": {
    "scope": "exact rational identities on original fixed A input; no costs recalculated",
    "full_H_paulis": {
      "III": "111/250",
      "IIZ": "-49/200",
      "IXX": "27/1000",
      "IYY": "27/1000",
      "IZI": "3/25",
      "IZZ": "-4081/20000",
      "XXI": "-7/4000",
      "XXZ": "7/1000",
      "YYI": "-7/4000",
      "YYZ": "7/1000",
      "ZII": "-9/160",
      "ZIZ": "21/200",
      "ZXX": "-27/4000",
      "ZYY": "-27/4000",
      "ZZI": "-1637/10000"
    },
    "old_core_paulis": {
      "III": "7/16",
      "IIZ": "-49/200",
      "IZI": "3/25",
      "IZZ": "-1/5",
      "ZII": "-9/160",
      "ZIZ": "21/200",
      "ZZI": "-129/800"
    },
    "complete_diagonal_core_paulis": {
      "III": "111/250",
      "IIZ": "-49/200",
      "IZI": "3/25",
      "IZZ": "-4081/20000",
      "ZII": "-9/160",
      "ZIZ": "21/200",
      "ZZI": "-1637/10000"
    },
    "old_residual_paulis": {
      "III": "13/2000",
      "IXX": "27/1000",
      "IYY": "27/1000",
      "IZZ": "-81/20000",
      "XXI": "-7/4000",
      "XXZ": "7/1000",
      "YYI": "-7/4000",
      "YYZ": "7/1000",
      "ZXX": "-27/4000",
      "ZYY": "-27/4000",
      "ZZI": "-49/20000"
    },
    "new_residual_paulis": {
      "IXX": "27/1000",
      "IYY": "27/1000",
      "XXI": "-7/4000",
      "XXZ": "7/1000",
      "YYI": "-7/4000",
      "YYZ": "7/1000",
      "ZXX": "-27/4000",
      "ZYY": "-27/4000"
    },
    "old_residual_nonidentity_l1": "183/2000",
    "new_residual_nonidentity_l1": "17/200",
    "full_H_collected_nonidentity_l1": "979/1000",
    "full_H_identity": "111/250",
    "extra_diagonal_absorbed_nonidentity_l1": "13/2000",
    "new_diagonal_supports": [],
    "full_H_operator_norm_diagnostic": 1.3418981992179224,
    "one_particle_effective_matrix": [
      [
        "6881/10000",
        "81/2000",
        "0"
      ],
      [
        "81/2000",
        "991/2000",
        "21/2000"
      ],
      [
        "0",
        "21/2000",
        "2299/10000"
      ]
    ],
    "one_particle_scope_note": "The fixed prepared state has one particle; an effective one-body evolution is task-equivalent on that sector only, not full Fock operator-equivalent.",
    "all_identity_checks": true
  },
  "B": {
    "scope": "same fixed isometry, real orbital entries, binary64 verification of derived exact pair identity",
    "isometry_residual": 4.169720964580387e-17,
    "quartic_projection_residual": 2.220446049250313e-16,
    "physical_pair_reconstruction_residual": 7.060243458078294e-17,
    "projector_involution_reconstruction_residual": 2.220446049250313e-16,
    "projector_involution_unmerged_l1": 0.5196286029667954,
    "projector_involution_scalar_offset": 0.46962860296679537,
    "pair_rows": [
      {
        "indices": [
          0,
          1
        ],
        "w": 0.7,
        "gram_determinant": 0.9761271242968684,
        "weighted_pair": 0.6832889870078078,
        "pair_identity_residual": 2.4827926415183016e-16
      },
      {
        "indices": [
          1,
          2
        ],
        "w": 0.4,
        "gram_determinant": 0.75,
        "weighted_pair": 0.30000000000000004,
        "pair_identity_residual": 1.1102230246251565e-16
      },
      {
        "indices": [
          2,
          3
        ],
        "w": 0.25,
        "gram_determinant": 0.02387287570313154,
        "weighted_pair": 0.005968218925782885,
        "pair_identity_residual": 1.734723475976807e-17
      },
      {
        "indices": [
          0,
          3
        ],
        "w": -0.2,
        "gram_determinant": 0.24999999999999994,
        "weighted_pair": -0.04999999999999999,
        "pair_identity_residual": 1.3877787807814457e-17
      }
    ],
    "pair_coefficient_l1": 1.0392572059335907,
    "unmerged_pair_pauli_nonidentity_l1_upper": 0.779442904450193,
    "unmerged_pair_identity_sum": 0.23481430148339769,
    "compiled_cost_evaluated": false,
    "dominance_claim": false
  },
  "accounting": {
    "rare_event": {
      "tau": 0.3601648685772598,
      "normalization": 1.128207385300923,
      "order2_probability": 0.05790168561964706,
      "probability_no_order2_in_8": 0.620540046614217,
      "expected_order2_in_8": 0.46321348495717646,
      "unmerged_single_step_event_count": 1740
    },
    "shot_policy_bias0_normalization1_floor": {
      "0.05": 7012,
      "0.02": 43821
    },
    "A_report_factorization": {
      "RZ_ratio_identity_to_allR": 0.8531079905554865,
      "CX_ratio_identity_to_allR": 2.9889812036823344,
      "N_ratio": 0.7701669359181476,
      "RZ_cost_pair_allR": 65.0,
      "RZ_cost_pair_identity": 72.0,
      "CX_cost_pair_allR": 21.0,
      "CX_cost_pair_identity": 81.5,
      "warning": "Same seeds are reused across candidates; no between-candidate independence or significance assumed"
    },
    "C_THRIFT_to_S2_ratios": [
      {
        "rz": 2.8193579986021913,
        "cx": 2.5605950042099352
      },
      {
        "rz": 2.7544968826596494,
        "cx": 2.501686894799106
      },
      {
        "rz": 2.772139432351997,
        "cx": 2.5177101967799644
      },
      {
        "rz": 2.638343590875283,
        "cx": 2.396194247603168
      },
      {
        "rz": 8.360831024293779,
        "cx": 7.573929280830835
      },
      {
        "rz": 6.9674772077889635,
        "cx": 6.311714647055885
      },
      {
        "rz": 7.26838295188247,
        "cx": 6.584299850528825
      },
      {
        "rz": 5.457406450210092,
        "cx": 4.891453188706824
      }
    ],
    "B_report_ratios": {
      "rz": 4.896272474985711,
      "cx": 3.9481878843386107,
      "shots": 0.9870469710846527
    },
    "fixed_policy_headroom": {
      "A_frame_identity_max_relative_shot_saving": 0.019437840861417977,
      "B_reflection_max_relative_shot_saving": 0.07051961823966069,
      "B_report_mean_pair_rz": 94.25,
      "B_report_mean_pair_cx": 30.0,
      "A_factor0_report_mean_pair_rz": 126.0
    },
    "new_quantum_shots_or_compilations": 0
  },
  "review_runtime": {
    "python": "3.13.5",
    "numpy": "2.3.5",
    "sympy": "1.14.0"
  },
  "script_sha256": "2c787c24a924c2524570ba6f7c6f768a7f8e660cf78607ed6b7ed8191b74c205"
}
```

## 付録B. 今回作成した記録の位置づけ

本Markdownに、科学的判断・条件・先行研究・固定sourceへのリンク・独立計算コード・結果をまとめた。別ZIPの共有は必要ない。元source/resultとこのレビューは異なる証拠であり、Codexが後続作業に使う場合も両者を区別する。

レビュー状態：**完了**。次担当：**Codexによる限定した比較要因の切り分けと代替primitiveの検証**。研究テーマの最終採択、全手法への一般優位、新規性の確定、次の大規模計算は行っていない。
