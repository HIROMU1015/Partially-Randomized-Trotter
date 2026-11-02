# Track B：Return aggregationの発展と、新しいアルゴリズム改善方向の科学的レビュー

- 作成日：2026-10-14（JST）
- 開始承認：利用者の「レビューを開始して　丁寧に考えて」。
- 問い：論文化をいったん置き、Return aggregationを発展させるか、Return aggregationを含む過去の検証から別のアルゴリズム改善を考えるか。
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 方針資料の基準commit：`66c96f8033bcf8392237d820d4a4615f3cebc75a`。個別のsource・結果は後掲の固定commitを使用。
- 正式運用指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`。
- 状態：**科学的方針レビュー完了。次に優先して独立監査する構成を具体化した。新候補のnative資源優位、独立新規性、分子PR全体の改善は未認定。計算実行の承認ではない。**
- 実施範囲：既存資料とsourceの読取り、関連一次文献の確認、解析的な恒等式・最適化条件の導出、研究方針の比較。
- 未実施：新しいHamiltonian・行列・回路・trajectory・登録資源表の生成、compile、T合成、数値最適化、repository tests、GitHub書込み、authorization・marker作成。

## 0. 結論

**直近の第一候補は、Return aggregationを出発点にした「回路費用を考慮したpairing・乱択表現の再設計」とする。既存のfull/P5が勝つ条件を探す検証の延長も、具体的な機構のない別系列への全面転換も、次の作業としては優先しない。**

利用者の二つの方向は完全な排他ではない。今回の優先順位は、Return aggregation周辺でのアルゴリズム改良を先に調べるという意味で第一方向に近い。しかし、Return aggregationを最終的な勝者に固定しない。R1、SP、G5、DF補正などから得た知見を設計へ取り込み、同じ変更がordinaryやpartialにも効くかを比較する。集約を使わない側が有利なら、その結果に従う。

具体的には、現在のparent/child pairingが、parentの実係数をchild係数に比例配分している点を変数へ戻す。**同じ有限operator meanを保ちつつ、childごとの回転角・係数・抽出確率を変える構成**である。単なるproposal変更とは異なる。

今回の解析から、次が得られた。

1. 任意の非負なparent係数配分から、同じparent/child operatorを表すunitary和を構成できる。
2. 現在の共通角pairingは、この配分familyで係数L1質量を最小にする。
3. しかし回路単価を固定した診断モデルでは、その配分がcost×second-momentも最小にするとは限らない。単価に差があれば、共通角配分から費用指標を下げる方向が存在する。
4. closed P5のrootだけを変えるなら、既存の少数群samplerへ接続する経路があり、全形式語・全Pauli表を新たに入力する必要はない。

これは数学的な設計候補と、調べる価値のある機構の提示である。**実際には回転角を変えると合成価格・biasも変わるため、固定単価モデルの改善をnative資源優位へ読み替えない。** また、Euler分解、係数配分、cost-aware samplingは近接する既知原理であり、この結合全体を世界初と認定しない。

次の実務担当はCodex。数式・既存source・過去の再配分class・近接手法・生成費用を一つの限定監査にまとめる。新しい登録比較・合成・分子計算は、その必要性と契約を明確にした後の別途実行承認まで行わない。既存のmandatory STOPと各closureを保持する。

## 1. 今回の判断で変えるもの、変えないもの

直近の10月14日レビューは、既存の構成とD1〜D3の資源評価を一体として整理する方向を推奨していた。その後、利用者の指示で研究ノートが完成した。[R0]

今回は「そのまま整理すれば何が残るか」ではなく、「これからアルゴリズムをどう改善するか」が問いである。したがって、原稿の完成度、投稿可能性、論文になる最低改善率などを優先順位の基準にしない。新しい数学primitiveの発明や、全既知法に対する普遍的勝利も条件に加えない。

変更するのは次の研究作業である。G5の限定class分離、G10の固定full不利、Gaussian候補の方法差未定義、DF generic wholeのclosureなどを撤回しない。技術的停止を性能negativeへ変更せず、旧runを新しい問いの下で再採点・再実行しない。

今回選ぶのは、**独立監査する価値のある一つの構成方向**である。完成した新アルゴリズムの採択、実性能の認定、本番実行GOとは区別する。

## 2. 証拠の確認範囲

### 2.1 内部資料

G6〜G10技術ノートの対象・正値性・pairing・局所生成、`g9_p5.py`全体、`g7_generator.py`のcanonical構成部分、G5のscopeと結果、BF/BM、SP、R1.5、DF補正closure、Gaussianレビュー、時間相関の後続文献監査、D3機構分析とclaim資料を固定refから取得した。[R1–R14]

前の可用性確認ではD3主要原row、48行表などの取得も確認しているが、今回の科学判断で全rawを独立に再集計したわけではない。下記D3数値は、固定された機構分析の保存値を用いている。理想samplerの構造は報告文だけでなく、実際のP5 sourceと技術ノートを対応させた。

全branch、全event、全合成sequence、全テストの再監査ではない。古い全体レビューにある要約のみを用いた箇所と、対応する一次の検証報告・sourceを読んだ箇所を区別する。FRは既存横断レビューの位置付けを継承し、この回に原証明を全面再評価していない。

### 2.2 外部資料

Resource-Optimal Importance Sampling、Sparse Probabilistic Synthesis、SCU/CTSについて指定版のHTML関連本文を直接確認した。[P1–P3]

Wan–Berta–Campbell、Zhao–Yuan、PR原論文については公式abstractを確認したが、指定PDF/HTMLの全文再取得に失敗した経路がある。具体的な式・algorithmとの対応は、G6およびD1〜D3の保存文献監査を併用する。[R10,R14,P4–P6] これを今回GPTが全文再認証したことにしない。

時間相関については、後続Codex監査がPRHS本文とqSHIFT改訂版を取得したことを確認した。[R12] 今回の直接web取得では、PRHS本文を再取得できず、qSHIFT abstractは保存監査のv3とは異なる版表示だった。この差を黙って解消せず、詳細な移送・優先性の認定には使わない。古い「PRHS未取得」のレビューを、後続監査を無視して現在地として扱わない。

PDFへの取得失敗を、論文が存在しない・非公開であるとの判断に使わない。今回PDF図表を独立に読み取ったという主張はしない。外部研究の網羅的な優先性調査も行っていない。

## 3. 過去の検証を、次の構成に効く知見として読み直す

| 系列 | 証拠として残ること | 次の設計への含意 | 今回しない読み替え |
|---|---|---|---|
| B-F | 固定family・budgetではfinite目的Fの最良がleading対照Lにも含まれ、共通参照のS2が安い | 評価式を詳しくすることと、新しい実行法を作ることを分ける | 全PF係数設計の不可能性 |
| B-M | 同じnested列の三次係数と同情報scoreはcompact再帰と一致 | 名前・式の分解より、実行列・生成・アクセスの具体差を求める | 全scheduleが同じという主張 |
| R1/R1.5 | normalization、bias、回転角・precision、native単価は別々に順位へ影響 | 係数と角度を変える構成には、合成と残余統計幅を戻す | Pareto残留を全対照勝利とすること |
| SP | cheap回路化とweightの交換で大きな限定利益があるが、長いwrapperでは反転 | 回路費用を下げるために多少のweight増を許す余地はある | toyの改善率をfinite RTEへ移植・乗算 |
| G5 | 固定prototype/IID shares/非負degree matching/precision/IS classに強いCTS対照が勝る | 同じclassのLPやproposal探索を再開しない | 任意の表現・角度・保証・PRを排除 |
| G6〜G10 | 形式returnの局所取得、finite-bit生成、P5少数群構成がある。一方、一般fullに追加native利益がない固定例がある | 集約量の最大化と資源最小化を分ける。生成可能性は引き継ぐ | Return aggregation全体が無価値 |
| D1〜D3 | 同じphysical taskで、保証・反復数・価格・境界処理を分離できた | 既に観測された価格感度を、表現設計へ戻せるかを問う | 合成seedの勝者探索を新アルゴリズムと呼ぶ |
| DF generic whole補正 | 条件付き高次化は支持。特定generic実装の局所T completion下界がPauli対照を上回る条件がある | 小さい支持・低頻度だけで安さを推論しない | 全whole回転のno-go、Bの全PR勝利 |
| Gaussian積生成 | 既知の反射積・partial Gaussianとの対応。候補固有の差は未定義 | 既知構成への適用で何が変わるかを明確にする | 費用実験に負けたとの表現 |
| 時間相関 | 関連する既知構成があり、joint meanとweight・rangeの移送が課題 | marginal一致やswitch削減だけでは足りない | 相関一般の不可能性 |
| FR | 同情報の境界改善と、予算判断の変化は別 | 保証改善を有効に使えるが、回路構成の改良とは区別 | boundに研究価値がないとの判断 |

根拠は[R4–R13]と既存横断レビュー[R3]。改善率の異なる実験を同一条件の順位表として比較していない。

重要なのは、これらを全て「失敗した候補」とまとめないことである。数学的同値性、限定的な性能negative、実費用未取得、情報アクセスの未完成は、次に必要な作業が異なる。

## 4. Return aggregationを続ける意味と、単純延長の弱点

### 4.1 残っている強み

既存構成は、係数をまとめる形式計算に留まらない。有限Taylor target、short-step正値性、ordered parent/child、制御位相、局所query、実確率とweight、P5の少数群normalizerまで接続されている。[R1,R2]

これは既存投入量が大きいから優先するという理由ではない。新しい変更を加えたとき、**何を変えて何を保ったかを明示できる構成上の足場**があるという理由である。全系の行列や未知真値を先に作らずに、検証する式を定義できる。

### 4.2 追加集約だけで大きな利益が出るとは考えない

既存sourceを解析した前回レビューには、固定P5・x→0で、χ=Σ_i p_i²として

\[
B_{\mathrm{ord}}=1+x^2+O(x^4),\qquad
B_{\mathrm{partial}}=B_{\mathrm{P5}}=1+(1-\chi)x^2+O(x^4)
\]

という先頭項の比較がある。[R3] これは本レビューの新発見ではない。

固定総時間τをn分割し、拒否なし・canonical係数比例samplingの理想momentだけを比べると

\[
2n\log\frac{B_{\mathrm{P5}}(\tau/n)}{B_{\mathrm{partial}}(\tau/n)}
=O(\tau^4/n^3)
\]

となる。したがって、一stepの小さい差を反復回数で単純に増幅して有望性を説明しない。この式はnative価格、finite-bit、bias、異なるn選択を含む資源上界・no-goではない。

m=7の一般fullがP5+tailに負けたG10結果も、指定dictionaryと十分予算policyにおける結果である。そこから、別のpairing・合成・表現を含む全方向を閉じない。[R1]

### 4.3 D3が示した、次の設計に使える具体性

D3のG1_P1/r1、準備T単価h_T=0では次の保存値がある。[R7]

| 量 | P3+tail | closed P5 |
|---|---:|---:|
| 十分試行数N/axis | 1,103,489 | 1,083,653 |
| 一試行の期待T単価 | 159.700056809 | 162.899691745 |
| T総費用切片 | 352,454,511.977 | 353,053,479.316 |

P3+tailは同じ有限P5を表す方式であり、Taylor次数を3へ落とした方法ではない。Nは実測shotsでも必要最小shotsでもない。対象も固定synthetic providerであり、分子PR全体ではない。

P5は統計側で有利でも、合成価格差により総Tで約0.170%逆転する。保存機構分析は、頻出rootに使うP5のK32とP3+tailのK33の価格変化が主要な寄与だったことを示している。他eventも含めた費用分解であり、全差をrootだけに等置しない。[R7]

**ここから得る設計上の含意は、角度を係数normだけから固定せず、実装費用との交換を設計変数にできないか、である。** D3自体は、以下のchild別配分を試していない。その性能証拠として使わない。

## 5. 比較した研究方向と優先順位

| 方向 | 現時点での評価 | 直近の扱い |
|---|---|---|
| full/P5を固定し、次数・seed・providerを増やす | 未確認の適用域はあるが、構成上の改善機構を新たに定めていない | 先に行わない |
| Return aggregationを材料にpairing・係数配分を変える | 同じmeanを保つ明示familyを作れ、既存の角度感度・費用差と結び付く。実費用効果は未検証 | **第一候補として独立監査** |
| 交換・反交換など、Qi²=Iを超える構造を使う | formal returnが使わない相殺に触れる。既知modified Taylor/CTSとの関係と構造取得・合成が必要 | 次点。具体的な入力・生成差ができた段階で比較 |
| SP・確率的合成とのjoint設計 | 接続候補と強い対照として重要。ただし既知原理で、二つの改善率を掛けられない | 第一候補の比較・構成部品として検討 |
| 新しいwhole補正・Gaussian圧縮 | 現候補にない具体的な生成差・費用改善を示す必要がある | 旧系列をそのまま再開しない |
| 時間相関・path再設計 | 可能性は否定されていないが、近接既知法とweight/first-momentの課題が残る | 全面転換の第一候補にはしない |
| PF評価式・FR保証の詳細化 | 保証が資源判断を変える場合は重要。ただしそれだけで回路生成法が変わるとは限らない | 支援的・独立課題として保持 |

この優先順位は、世界中の未発見アルゴリズムに対する最適性判断ではない。現在得ている証拠から、次に手を付ける具体的な構成の情報価値を比較した判断である。

構造的相殺の方向では、完全反交換のような特別な場合にR²がscalarになることが直観的な動機になる。しかし、その条件を利用者のHamiltonianへ勝手に付けたり、一般DF入力にそのまま成立するとしたりしない。反交換利用自体にはZhao–Yuan等の先行研究がある。[P4,R10] 現時点で第一候補より具体的な、同じaccess・native操作に閉じた追加手順は今回示していない。

## 6. 第一候補の構成：parent係数のchild別配分

以下は既存技術ノート[R1]のparent/child表示を出発点とする、本レビューでの解析的な具体化である。独立priorityの主張ではない。数値実験は実施していない。

### 6.1 既存のpairing

偶数長のreduced parent u、child集合C(u)について

\[
a=a_u>0,\qquad b_i=a_{iu}>0,\qquad s=\sum_{i\in C(u)}b_i,
\qquad \zeta_u=(-i\sigma)^{|u|}\in\{+1,-1\}
\]

と略記する。対応する有限operatorの部分は

\[
M_u=\zeta_u\left(aI-i\sigma\sum_i b_iQ_i\right)Q(u).
\tag{1}
\]

既存方式は

\[
d=\sqrt{a^2+s^2},\quad \phi=\arctan(s/a),\quad
c_i^{(0)}=d\,b_i/s
\]

を使い、全childで同じ角度φの

\[
V_i^{(0)}=\zeta_u e^{-i\sigma\phi Q_i}Q(u)
\]

を係数c_i^(0)で足す。P5 sourceではrootの`child_mass=child0/s0`、`ratio=s0/a0`がこの処理に対応する。[R2]

### 6.2 配分を自由にしても、同じmeanを保てる

parentの実係数を、非負のz_iへ配る。

\[
z_i\ge0,\qquad \sum_i z_i=a.
\tag{2}
\]

各childについて

\[
c_i(z)=\sqrt{z_i^2+b_i^2},\qquad
\phi_i(z)=\operatorname{atan2}(b_i,z_i),\qquad
V_i(z)=\zeta_u e^{-i\sigma\phi_i(z)Q_i}Q(u)
\tag{3}
\]

とすると、Qi²=Iより

\[
c_i(z)e^{-i\sigma\phi_i(z)Q_i}=z_iI-i\sigma b_iQ_i.
\]

したがって

\[
\boxed{\sum_i c_i(z)V_i(z)=M_u.}
\tag{4}
\]

異なるQi同士の可換性を仮定していない。一般の奇数語がinvolutionである必要もない。積Q(u)と位相ζ_uを保持したまま、各回転generatorを単一Qiに保つ。

既存方式は

\[
z_i^{(0)}=a\,b_i/s
\tag{5}
\]

という特殊な配分である。従って既存samplerを選択肢から除外しない。

**変更するのはproposalだけではなく、係数c_i、角度φ_i、実行unitary V_iである。** 同じ有限operator meanである一方、同じevent dictionaryの固定回路ではない。

### 6.3 退化ケースと回路規約

b_i=0のchildは純parent寄与と区別する。全child質量s=0なら、純parentを処理し、b_i/sを計算しない。a=0で全z_i非負なら全z_i=0となり、このfamilyに配分自由度はない。

b_i>0かつz_i=0ではφ_i=π/2であり、直接の位相付きQiになる。比b_i/z_iを無理に計算せず、endpoint branchを用意する。数値実装でendpointを通常の巨大ratioに置き換えることは新たな近似である。

回路のR_Q(θ)=exp(−iθQ/2)規約では、θ=2σφ_iになる。指数の角度φ_i、RZ等の物理角度、合成器へ渡すhalf-angleを区別する。ζ_uはcontrolled枝間のrelative phaseとなるため省かない。Q(u)はoperator順であり、回路への適用は右端からである。

## 7. 既存pairingはnorm最小であり、新候補はその利益を犠牲にし得る

配分後のparent質量を

\[
L_u(z)=\sum_i\sqrt{z_i^2+b_i^2}
\]

とする。二次元ベクトルの三角不等式から

\[
\boxed{L_u(z)\ge\sqrt{\left(\sum_i z_i\right)^2+
\left(\sum_i b_i\right)^2}=d.}
\tag{6}
\]

b_i>0の全childについて、等号はz_i/b_iが共通である場合に成立する。制約Σz_i=aより、これは式(5)である。

従って、**この配分family内でnormalizationをさらに下げることはできない。** 現方式がこの意味で合理的であることを認める必要がある。新候補の目的は、追加normと引き換えに回路費用を下げ、最終精度・confidenceで得になる条件を作ることである。

この結果はfamily内のL1最適性である。別dictionary、物理的な相殺、signed係数、別の推定器を含む全RTEの最適性ではない。

## 8. 費用改善の解析：何が示せて、何が示せないか

### 8.1 基本会計

正のevent係数c_e、実際のproposal q_e>0、±1の一回の測定Y_eについて、推定量X=(c_e/q_e)Y_eとする。理想の独立試行では

\[
E[X]=\text{対象axis信号},\quad
\nu:=E[X^2]=\sum_e\frac{c_e^2}{q_e},\quad
W=\max_e\frac{c_e}{q_e},\quad
\overline C=\sum_eq_eC_e.
\tag{7}
\]

状態に依存する真のvarianceはν−E[X]²であり、νそのものと同一ではない。既存研究の十分予算型では、axisの残余誤差幅Δ>0、失敗確率αに対して

\[
N\ge\left\lceil\log(2/\alpha)
\left(\frac{2\nu}{\Delta^2}+\frac{4W}{3\Delta}\right)\right\rceil
\tag{8}
\]

などを用い、必要量をN C̄へ戻す。両axisで同じ予算なら総量は2N C̄であり、一般にはaxisごとに足す。これは既存policyの例示で、将来の新実行契約をこの式に無断固定する指示ではない。

cost×second-momentについて

\[
\nu\overline C\ge\left(\sum_ec_e\sqrt{C_e}\right)^2,
\qquad q_e^*\propto c_e/\sqrt{C_e}
\tag{9}
\]

となる原理は既知である。[P1] このq*は、式(8)のrange、ceiling、bias、古典情報費用まで含む全目的の一般的最適解ではない。

### 8.2 固定単価モデルでの局所改善方向

ここでは**配分を動かしてもC_i>0を一定とする診断モデルだけ**を考える。実際の合成単価を固定と仮定した性能主張ではない。

式(9)の平方根側を

\[
S_u(z)=\sum_i\sqrt{C_i}\sqrt{z_i^2+b_i^2}
\]

とする。既存配分のまわりで

\[
z_i(t)=a b_i/s+t v_i,\qquad \sum_i v_i=0
\]

と置く。a>0かつ全b_i>0なら、十分小さいtで非負制約を維持できる。既存点でz_i/c_i=a/dなので

\[
\left.\frac{dL_u}{dt}\right|_0=\frac a d\sum_iv_i=0,
\tag{10}
\]

\[
\boxed{\left.\frac{dS_u}{dt}\right|_0=
\frac a d\sum_i\sqrt{C_i}v_i.}
\tag{11}
\]

従って、安いchildへparent係数を移し、高いchildから同量を減らせば、C_iが異なる場合は式(11)を負にできる。一方、L1の増加は二次から始まり

\[
\left.\frac{d^2L_u}{dt^2}\right|_0
=\sum_i\frac{b_i^2}{(c_i^{(0)})^3}v_i^2>0
\]

である。

**同じ配分familyでnormは最小だが、費用を含む目的には改善方向がある**という構造が得られる。固定cost条件下で、新配分に対する最適IS指標と、既存配分に対する最適IS指標を比較している。従って単に「既存側にISを与えていなかった」効果だけではない。

ただし、これは実回路の数値結果でも、新しい物理的shot下界でもない。

### 8.3 固定単価モデルの解の形

同じ限定モデルでS_uを最小化すると、a>0、全b_i>0、全C_i>0の場合のKKT条件から

\[
z_i=\frac{b_i\lambda}{\sqrt{C_i-\lambda^2}},\qquad
\sum_i\frac{b_i\lambda}{\sqrt{C_i-\lambda^2}}=a,
\quad 0<\lambda<\sqrt{\min_i C_i}
\tag{12}
\]

を得る。左辺はλについて単調増加し、解は一意である。等単価ならz_i=ab_i/sへ戻る。

これは標準的な凸最適化を、このfamilyに適用した解析である。native角度価格を入力せずに実資源最適解を計算したわけではない。T=0等の単価ゼロを含む場合は別に扱い、利用者の問題へ正単価を強制しない。

### 8.4 最適proposalを実装しなくても、生成経路は存在する

closed P5のrootだけを変え、非root係数は固定する。新全体質量B'=B_rest+L_0(z)、A'=Σ_e c_e(z)C_eとする。全係数比例samplingならν=B'^2、C̄=A'/B'なので

\[
\nu\overline C=B'A'.
\]

既存点ではL'_0=0で、固定単価モデルのA'の方向微分は(a/d)Σ_i C_i v_iとなる。したがって、canonical生成を維持した場合にも、適切な方向でこの指標を下げられる。

この確認は、最適qの存在だけに依存して生成可能性を曖昧にしないためのもの。ただし、実比較では既存側にも適用可能なcost-aware ISを認める必要があり、canonical同士の結果だけで強い対照への勝利を認定しない。

### 8.5 現実の費用に移す際の限界

実際には

\[
C_i=C_i(\phi_i,\varepsilon_{\rm synth},\text{control},\text{basis},\text{compiler})
\]

である。角度を少し変えてもT数が滑らかに変わるとは限らない。新角度の合成列取得にも費用がかかる。C_iを固定できない場合、式(11)だけでは費用減少を保証しない。

D3のroot価格差は、P5とP3+tailの共通root角度の差であり、本候補のchild別配分の効果ではない。同一parent内でどの価格差・角度感度が実在するかは未確認である。

等単価で角度依存もなく、同じbiasを持つなら、既存配分がnormと固定cost目的の両方で最適となる。この候補が常に得になるという結論ではない。

## 9. なぜ最初はP5のrootを選ぶか

rootのa0と全child0_iはP5 sourceで既に得られている。現samplerはroot一群の質量d0と条件付きb_i/s0を使う。[R2]

与えられた新配分zを使う場合、以下で理想生成できる。

1. 非rootの既存群は保持し、root群の質量だけをL0(z)=Σ_i c_i(z)へ変更する。
2. 全群を新B'=B_rest+L0で正規化して選ぶ。
3. rootが選ばれたらchild iをc_i/L0で選び、exp(−iσφ_iQi)を実行する。
4. 非rootでは既存の条件付き生成を使う。理想の係数比例samplingなら全体weightはB'。

式(4)によりtargetは元のfinite P5のままである。**parent係数配分が決まっている場合、rootの再構成に必要な追加係数処理はL個のchildに対する処理であり、全word表は要求しない。** 既存P5のO(L²)群構築、群選択の方式、bit計算量、費用queryや配分探索を無料とはしない。sampler全体をO(L)だとは主張しない。

ここでLはinvolution項数であり、qubit数やDF rankではない。多数parentへ同時に自由度を広げると、既存少数群の共有角度・normalizer圧縮が失われる可能性がある。その場合は別の複雑性問題になる。

rootを選ぶのは、D3で価格感度が具体化され、sourceへの局所変更でmeanと生成を閉じられるためである。今後の研究をrootだけへ永久に制限するものではない。新たなgeometry、seed、precisionを結果に合わせて選ぶ提案でもない。

## 10. 有限精度・bias・長時間で必ず戻すもの

### 10.1 係数とprobability

z_iを有理数で保持しΣz_i=aを厳密に満たせる場合、既存sourceのatan ratio方式との接続が考えられる。一方、c_iの平方根や角度は一般に無理数であり、区間・近似係数が必要である。

実際のdigital係数をc̃_e、実確率をq̃_eとしたら、weightはc̃_e/q̃_eである。q̃_e w̃_e=c̃_eを満たすことと、c̃_eが理想c_eを十分近似することを別に監査する。零確率への丸め、角度endpoint、normalizer enclosure、実range/momentを検査する。

### 10.2 回転合成bias

理想V_eに対し実装V̂_eのoperator誤差がε_eなら

\[
\left\|\sum_ec_e\widehat V_e-M\right\|
\le\sum_e c_e\varepsilon_e
\]

が基本的な上界となる。digital係数・角度誤差は別途加える。c_eやφ_eを変えれば、このbiasも一般には変わる。

proposalだけを変更して同じ実装を使うISではbiasは変わらないが、今回の構成は実装角度・係数を変えるので、その不変性を流用できない。[P1]

### 10.3 時間発展全体

同じfinite meanを保つ変更は、理想的には元のtruncation誤差を変更しない。実装誤差、weight積、moment、range、必要試行数は再評価対象である。高次PF化やTaylor剰余の次数改善を、この候補の効果として主張しない。

Dを挟むPR全体では、境界相殺、basis reuse、準備・readout、native depthも費用へ関与する。one-eventの費用式を全trajectoryへそのまま足して一般的な最適性を示したとはしない。古典秒数とT数を無根拠に足さず、別座標または明示された変換モデルで扱う。

同じ有限P5を保つことは、最初の機構比較を分離するための選択である。Track Bの将来の全アルゴリズムに同じTaylor多項式の使用を強制しない。別の時間発展近似へ進む場合には、同じ物理targetと必要な精度・成功条件へ戻して比較する。

## 11. 既知法との関係と、本候補を評価する基準

### 11.1 既知と扱う部分

Eulerのcos/sin pairingは既知であり、ordinary RTEやCTSも用いる。[R10,P3,P5] 式(4)はその恒等式を係数配分へ適用したもので、恒等式自体の発見を主張しない。

費用×second-momentとq∝coefficient/√costは既知である。[P1] 安価な回路の線形結合と測定負担の交換、dictionary設計・係数最適化にもSPS等の先行研究がある。[P2]

従って「費用を目的関数にする」「角度を変える」「係数を配り直す」だけで独立新規性を確定しない。

### 11.2 今回まだ確認していない差

確認対象は、parent/childのformal構造を使う配分familyが、既存の具体的sampler・最適化・取得手順のどれに相当するかである。一般LCUに含まれることだけで無価値としない一方、同じ入力・target・生成処理・誤差・取得費用まで同じなら、独立の新方法と呼ばない。

SPSの本文はprocess matrixを対象としている。system channelの等式からcoherent first operator meanの一致は直接従わない。しかし、controlled joint systemや適切なobservableとして対応させる経路があるため、一律に比較対象外にもしない。変換に必要なcontrol、位相、dictionary、sampling費用を比較へ戻す。

### 11.3 G5の排除classとの関係

G5は7prototype、固定IID p-word shares、非負degree matching、固定native辞書のprecision配分とfull-support ISなどを対象にしている。[R4] childごとに角度・実係数を変更する式(2)–(4)が、その固定classへ自動的に含まれるとは読めない。

ただし「排除定理の外だから勝てる」とは言わない。CodexはR0/G1/G5の定義と式(2)–(4)の対応を確認し、既存の別pathで実装済みならその事実を示す。今回GPTが全過去branchを検索し、不存在を証明したわけではない。

### 11.4 論文化を置くことの意味

現在の目的は資源効率のあるアルゴリズム改善である。既知原理の適切な適用で実用的な改善が得られる場合、その価値を「世界初ではない」という理由だけで否定しない。

一方、名前を変えただけで実行手順も費用も変わらないなら、アルゴリズム改善の証拠はない。独立した優先性、同一実装との重複、実際の資源利益を別の軸として評価する。

## 12. Return aggregationを守るための比較にしない

最初の機構分離には、概念上、次の四者が役立つ。

|  | 既存共通角pairing | child別配分pairing |
|---|---|---|
| ordinary/partial等の非full集約表現 | 既存構成＋適用可能な既知sampling最適化 | 同じ新配分を適用した比較構成 |
| Return aggregation表現 | 既存P5等＋適用可能な既知sampling最適化 | Return aggregationと新配分の組合せ |

これは結果前比較の考え方であり、今ここで四条件の科学実行を登録・承認したものではない。

child別配分をordinaryへ適用した列も、新しい変更を使う側である。それを「既知対照が初めから無料で持っていた」として新手順の寄与を消さない。一方、Return aggregation側だけに新自由度を与えて、その差を全て集約の利益ともしない。

同じprimitive/control/native条件、同じ費用取得機会、同じ実行誤差・confidenceを用いる。既知のcost-aware ISは対照にも適用可能性を認める。未知の完全最適compilerを無料の対照に要求することはしない。

想定される読み方は次の通りである。

- 両表現で改善するなら、主な機構はpairing設計であり、RA専用の効果ではない。
- RAで追加利益が残るなら、集約後係数と費用対応pairingの組合せに意味がある。
- 非RA＋新配分が良ければ、RAへ戻すための条件探索をしない。
- 既知ISや既知合成だけで同じ利益が得られるなら、追加配分の寄与を小さく評価する。
- 技術的未取得であれば、性能negativeへ丸めない。

## 13. 次の作業：Codexによる一つの限定的な構成・実装監査

GPT側でさらに抽象的な全体レビューを反復するより、今回の具体式を技術的に照合する材料がある。

### 13.1 一つにまとめる内容

**数式と意味論の独立確認。** 式(1)–(6)のoperator順序、位相、endpoint、mean、norm最小性を独立に導出する。固定価格モデルの式(10)–(12)を、native性能定理とは区別して確認する。必要な登録外のsymbolic/mockテストは、依頼された技術作業の範囲でまとめる。

**source・既知構成との差分確認。** P5 rootの係数・角度・lawのどこを変えるか、既存event形式がchild別ratioを扱えるかを確認する。R0/G1/G5の再配分class、Euler pairing、SPS等に同じ生成手順が既にあるかを対応表にする。単なる名称検索や「一般LCUで書ける」だけで結論しない。

**生成・費用情報の閉じ方。** rootだけの変更について、normalizer、finite-bit proposal/weight、実range/moment、native error、追加の係数/角度/query/cache費用を一つの仕様にする。全parent・全角度を先に合成して最良値を探す設計にはしない。費用の未取得箇所は未取得と示す。

**次の比較が何を判別するか。** 元方式＋既知IS、新配分、非RAへの同じ変更をどう比較するかを整理する。利益を判別するために新合成や登録予算計算が必要なら、その対象・精度・上限・失敗処理を結果前に計画する。

### 13.2 今回のレビューから自動的に許可されないこと

旧G5/G10/D1〜D3/DF補正の再実行、結果の再ランキング、新しいregistered parameter最適化、T合成、分子production、新しいone-shot authorizationは行わない。具体的な比較計算は別途ユーザーの実行承認を要する。

Codexへの技術作業依頼と、登録科学実行の許可は別である。今回このレビューを完成させたことだけでCodexが起動したことにもならない。

### 13.3 GPTへ戻す節目

数学的反例、既存構成との完全対応、実費用・biasを戻したときの重要なpositive/negative、比較契約の意味論変更、または予想より大きい情報取得負担が明らかになった段階で、研究判断へ戻す。

sourceの軽微な修正、テスト追加、保存形式の変更、commit作成ごとにGPT詳細レビューを必須化しない。技術的に解決できる点はCodex側でまとめる。重要な科学的結果の次回レビューには、改めて開始承認を必要とする。

## 14. 継続・縮小・切替の判断条件

| 得られた証拠 | 研究としての扱い |
|---|---|
| 同じmeanと生成・finite-bitが閉じ、既知対照との比較で実資源利益が残る | 利益が出た機構を中心に限定拡張を検討する。一般PR優位へ飛躍しない |
| 同じnew pairingがordinary側にも効き、RAの追加利益はない | pairing/representation改善を主対象にし、RAを必須にしない |
| 既知手順と完全に同じだが、未利用だった適用で費用が下がる | 既知法の適用・実装改善として正確に位置付ける。世界初の方法とは呼ばない |
| 既知手順と同じで、費用・情報・保証にも差がない | 別の新アルゴリズム候補としては区切る |
| 固定価格モデルでのみ得で、実角度価格・biasでは利益が消える | 機構仮説が実装で支持されなかったと扱う。価格を後付けして救済しない |
| 費用query・normalizer・古典生成が支配する | 情報取得を含む改良が具体的に作れるかを再判断する。量子Tだけで勝利としない |
| 認証・coverage不足 | technical inconclusive。性能の優劣を作らない |

最低5%や10%といった新しい閾値を設けない。小差であっても再現・認証され、取得費用込みで意味があるかを評価する。逆に大きい表示差でも、不公平なbaselineや異なるtaskから作られた差なら採用しない。

この候補に利益が残らなければ、次点の物理的相殺・別の時間発展表現を検討する理由はあり得る。ただし、候補の終了を合図に別名のrunnerを自動立上げすることはしない。具体的な差を提示できない間は、新規科学実行を止めたままにする。

## 15. 判断の確かさと最終整理

確かさを分ける。

**高い確かさを持つ部分**：取得した固定資料が示す過去の限定結果、P5 sourceにおける共通角pairing、式(4)の理想mean恒等式、式(6)のfamily内L1最小性、明示した固定正単価モデルでの方向微分。

**未検証の部分**：独立コード・実際のsourceにおける新配分の実装、実合成価格での利益、finite-confidenceでの優位、同じ機構の先行実装、費用取得の効率性、分子DF/PR/QPEへの移送。

本レビューの次段階推奨への確信は中程度である。実性能を確信しているのではなく、他の現候補に比べて、変更する自由度、保つmean、利益・損失の機構、限定監査の対象を具体的に示せることを評価している。

**最終判断：Return aggregationを出発点に、回路費用に適した乱択表現を作る方向を先行する。既存P5/fullの延長評価を続けるのではなく、parent係数配分とpairingをアルゴリズムの設計変数へ戻す。Return aggregationの採用自体は固定せず、過去の全系列から得た知見と公平な対照を用いて、どの部分が本当に改善を生むかを調べる。**

論文化作業は今回の次段階に含めない。既存の科学結果・closure・消費済みmarker・mandatory STOPを保持する。

---

## 参考資料：内部固定証拠

[R0] 最新研究ノート・現在地（本会話で取得確認済み）。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/research/研究ノート/2026-10-14_track_b_return_aggregation_G_D_research_note.md

最新のD1〜D3レビュー（本会話で取得・内容確認済み）。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/research/track_b_d1_d3_post_literature_scientific_review_20261014.md

[R1] Return aggregation技術ノートv1.0。今回の中心確認は§1〜3のtarget、正値性、pairing、local generationと、既存資源結果の限定。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md

[R2] P5実装（全体読取り）、canonical generator（冒頭〜150行の構成部分）。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g9_p5.py
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g7_generator.py

[R3] 既存成果発展の横断レビュー。今回特に§3、§5〜7の構成・小step・合成との接続を確認。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/research/track_b_existing_results_development_broad_scientific_review_20261013.md

[R4] G5固定classのscopeと結果。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/tracks/algorithm_codesign/g5_fixed_dictionary_closure_scope_20261010.md
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md

[R5] B-F closureとB-M同値性監査。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md

[R6] SP-1保存結果と範囲。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d2bb1fa439748b02084bd9fbc9b10a705328f8a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md

[R7] D3最終保存機構分析。今回は保存scalarと価格寄与の説明を確認、元数値計算は再実行せず。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md

[R8] R1.5保存値帰属。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md

[R9] DF generic whole系列closure。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4192077723770f4846cc391ea0da96c1e115f2fa/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md

[R10] G6先行研究・method delta照合。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md

[R11] Gaussian積生成の監査後レビュー。今回は冒頭〜§3.2を中心に、closureと性能未確定の区別を確認。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/research/track_b_gaussian_product_generation_scope_review_20261013.md

[R12] 時間相関の後続一次文献監査。PRHS取得後の追記、qSHIFTの版差、mean/weight/accessを確認。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md

[R13] 時間相関の前段レビュー。本文未取得という当時の状態は[R12]の後続確認と区別。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/research/track_b_interstep_correlated_randomization_direction_review_20261013.md

[R14] D1〜D3の文献・claim対応。今回はscopeとclaim境界を確認し、8論文全本文の再監査とはしない。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md

## 参考資料：外部一次文献

[P1] Cugini, Atif, Subaşı, Resource-Optimal Importance Sampling for Randomized Quantum Algorithms, arXiv:2603.13495v1。HTML §II/Theorem 1、§IIIのbias不変を確認。本文での用途はcost×second-momentと固定実装下のIS。有限confidenceの全目的最適性へ拡張しない。
https://arxiv.org/html/2603.13495v1

[P2] Koczor, Sparse Probabilistic Synthesis of Quantum Operations, arXiv:2402.15550v2。HTML §II.1–II.3、§III.1のprocess表現・dictionary・費用交換を確認。coherent first momentへ無条件移送しない。
https://arxiv.org/html/2402.15550v2

[P3] Peetz, Smart, Narang, Quantum Simulation via Stochastic Combination of Unitaries, arXiv:2407.21095v2。HTML §II.3/Theorem 1、関連samplingと資源会計を確認。CTS family全体を全語列挙対照として扱わない。
https://arxiv.org/html/2407.21095v2

[P4] Zhao, Yuan, Exploiting anticommutation in Hamiltonian simulation, arXiv:2103.07988。今回公式abstractを確認。modified Taylorの詳細式は[R10]の本文照合を継承。
https://arxiv.org/abs/2103.07988

[P5] Wan, Berta, Campbell, A randomized quantum algorithm for statistical phase estimation, arXiv:2110.12071。今回公式abstractを確認。Appendix C/Algorithm 2の具体的対応は[R1,R10]の固定監査を継承。
https://arxiv.org/abs/2110.12071

[P6] Phase estimation with partially randomized time evolution, arXiv:2503.05647。今回公式abstractと版情報を補助確認。元PRの全定理・全数値の再評価ではない。
https://arxiv.org/abs/2503.05647

外部文献の式表示に疑義があるという保存監査の記述を、その研究全体の不成立や、新候補の優位を示す根拠にはしていない。本書の新しい解析は上記の明示した条件のもとで独立に式を書き、未実装・未検証部分を分離している。
