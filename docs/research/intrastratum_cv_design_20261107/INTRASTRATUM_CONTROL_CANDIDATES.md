# 層内control候補：metadata・既知平均・費用

2026-11-07 JST。静的SOURCE調査・記号式の設計のみ。新controlの採択・実装・実平均の算出・保存bit再解析はしていない。

## 1. 対象と根拠

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits/全4電子sector、L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、N1 actual finite-grid Q・selected full policy、N2保証ground・認証済みexact engine、理想gate限定。一次targetは同じattenuated複素meanであり、energyや長roundへ拡張しない。

- [理論8d8a290・§3,4,7](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8d8a290fa4d4cf24f968aea27953ad5422e418b3/docs/research/a1_b1ol_theory_20261107/A1_VS_B1OL_THEORY.md)の層内定数・hybrid式を出発点とする。
- [N1 744e1a0・REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/744e1a09661ebba6841fe5faf456f88c41c47780/docs/research/o2_a1_new_target_n1_20261106/REPORT.md)は324 symbolic labels/232 positive grid labels、固定a,b、5層を保存している。これらは既報の引用で、新しいsupport集計ではない。
- [752-shot 3c8e55a・REPORT](../o2_a1_n3_pilot_admission_20261107/REPORT.md)はINCONCLUSIVE。既存control・quota・bit・費用を新候補へ再fit/再解析していない。
- 以下のSOURCE参照は本branchの基点3c8e55aに固定。RNG・関数呼出・model loadはしていない。

## 2. 実際の条件付きlaw

適用順labelを L0,...,Ln とする。n=0ではL0がrotation、n=2ではL0,L1がproducts、L2がrotation。
RTE内部の選択順rotation→productsと混同しない。metadataはproducts→rotationで保存される。

ρ(l)=Q(label=l)、ρ_a=ρ(a)、ρ_b=ρ(b)。N1保存CDFが定義する整数/2^53の質量を用い、理想的な係数比に戻さない。δ_aはaへの点質量、ρ_¬a(l)=ρ(l)1{l≠a}/(1−ρ_a)と書く。

| h | 条件 | 条件付き各位置law（独立） | rotation law |
|---:|---|---|---|
| 0 | n=0,L0=a | δ_a | δ_a |
| 1 | n=0,L0≠a | ρ_¬a | ρ_¬a |
| 2 | n=2,L0≠a | ρ_¬a, ρ, ρ | ρ |
| 3 | n=2,L0=a,L1≠b | δ_a, ρ_¬b, ρ | ρ |
| 4 | n=2,L0=a,L1=b | δ_a, δ_b, ρ | ρ |

これは[ConditionalSampler.draw/_pick](../../../src/trottertracks/algorithm_codesign/a1_n3_connection.py)の制限に対応する。禁止labelを除く「条件付き確率」の分母は必要な条件付けであり、元Qのzero-grid supportを削除して再正規化する操作ではない。

任意の一位置特徴 f の平均・jointは

$$
\nu_{h,j}=\sum_l\rho_{h,j}(l)f(l),\qquad
E[f(L_i)g(L_j)\mid h]
=\begin{cases}
\sum_l\rho_{h,i}(l)f(l)g(l)&i=j,\\
E[f(L_i)\mid h]E[g(L_j)\mid h]&i\ne j.
\end{cases}
$$

一致条件を含む特徴では和の中の共通labelを一つに束ねる。一般には E[u 1_h]/p_h でも計算できる。LocalControlのjoint_mean/pattern_massの積・矛盾排除・包含排除を再利用できるが、今回どの実数値も算出していない。

**h=0はpathが一つに固定される。** 固定Dブロック・wrapperも含め、pathのみの任意のuは一定。ここではmetadata CVで削減できるpath分散はない。測定wordをcontrolに混ぜてこの限界を回避してはならない。

## 3. 第1候補：rotationのfragment群indicator

F(l)を保存df_fragment_id、Sを測定前に固定したfragment群とし、

$$
u_S(\omega)=1\{F(L_n)\in S\},\quad
R_S=\sum_{l:F(l)\in S}\rho(l).
$$

条件付き平均は

$$
\nu_{0,S}=1\{F(a)\in S\},\quad
\nu_{1,S}=\frac{R_S-\rho_a1\{F(a)\in S\}}{1-\rho_a},\quad
\nu_{h,S}=R_S\ (h=2,3,4).
$$

- **層内変化の存在証明**：S={df-fragment-6}を例にできる。N1でbはこのfragmentに正質量を持ち、aはfragment-3。RTEComponent/extractionのI/Z/ZZ仕様では、一fragmentの非identity labelは8+binom(8,2)=36種類以下。既報232 positive labelsは36+1より多いため、aを除いたh=1にもfragment-6外の正labelが必ず残る。したがってこの例はh=1で本当に非定数。rotationが自由なh=2,3,4でもa,bが正なので非定数。h=0だけ一定。この論証は既報support数とSOURCE上限からの記号的帰結で、実catalogの再集計ではない。
- **正確な平均**：保存LAWのρとcatalogのfragment IDの有限和。整数/Fractionのみで表現できる。νそのものの数値取得・平均表作成は次回許可後。νが既存N1の3-control平均として保存済みだと誤認しない。
- **追加量子測定**：pathごとのu取得には0。label→fragmentのlookupのみ。係数を学ぶfresh bitの費用は別途必要。
- **相関の理由**：同fragmentの成分は共通のDF回転基底を持つ。rotationの生成子の族が違えば条件付きHadamard平均が異なる可能性がある。実際の相関・符号・大きさはUNKNOWN。fragment IDと回路価格の相関だけでは測定bitとの相関を保証しない。
- **費用/range**：setupはlabel数に線形の集約、onlineはlookup。u∈[0,1]、既知Σ_h=ν_h(1−ν_h)。1係数/活動層なら少数trainとrange管理が容易。
- **注意**：正質量≠少数trainでの学習可能性。fragment-6例の条件付き質量が十分大きいかは今回未計算。第一検証案は測定前metadataだけで群を適度な質量へ固定するルールを提案する。正式選択はしていない。

## 4. 第2候補：signed support・係数由来の一位置特徴

代表例は保存support S_lと係数sign s_lを用いる

$$
u_{\rm ss}(\omega)=s_{L_n}|S_{L_n}|/2\in[-1,1].
$$

このSOURCEは非identity Z/ZZなので分母2で有界。rotation sign単独 s_{L_n} も候補だが、同じsignだけの層では定数となる。基本角θ_nは層内固定であり、unsigned angle=s_l θ_nはsign controlの定数倍にすぎない。

- **変化**：h=0は定数。h=1のa以外、h=2–4の自由rotation support/signに異なる特徴値が正質量であれば変化する。多数labelがあるだけではこの集約特徴の変化を保証しない。実conditional varianceは未評価。
- **平均**：上の一位置有限和、二次momentも同じ和。saved binary64係数の大きさを使う場合はhex→exact rationalとして定義し、正の事前scaleで有界化する。理想実係数・新たなatanの厳密平均が無料で得られるとはしない。
- **相関**：符号はsigned Pauli生成子とsignalへ入る。supportの大きさは演算子族を区別する。ただし未知のstate依存応答を特徴へ入れない。X/Yで相関の性質が違い得る。
- **再利用/費用**：RTEComponent/DFDiagonalPauliComponentの既存sign/support/coefficient_absをlookup。setup O(label数)、online O(1)、quantum feature取得0。
- **学習/range**：原則scalarのまま別候補として比較。uの幅2なので D_h≤2+2|λ_h|。候補1と同時投入して列を増やす場合は、追加rank・学習費と既知covarianceの検査が必要。今回その組合せを採択しない。

係数の大きさは既にimportance lawにも反映される。特徴に使っても、測定bitの重みやQをさらに係数倍してはならない。

## 5. 第3候補：n=2のproduct label一致・同一基底関係

$$
u_{\rm eq}=1\{n=2,\ L_0=L_1\}\in[0,1].
$$

$$
\nu_{0,\rm eq}=\nu_{1,\rm eq}=\nu_{4,\rm eq}=0,\quad
\nu_{2,\rm eq}=\frac{\sum_{l\ne a}\rho(l)^2}{1-\rho_a},\quad
\nu_{3,\rm eq}=\frac{\rho_a}{1-\rho_b}.
$$

a≠b。232正labelを持つ積lawからh=2,3では一致/不一致が両方正、h=4ではa,b固定のため0。h=0,1も定数0。

- **相関の理由**：完全に同じsigned involution P_lのproductsなら P_l²=I となり、event内の生成子積が単純化する。測定信号は依然state/rotation/Dブロックに依存し、単純化が有用な相関になるかは未確認。
- **平均/費用**：上記はO(label数)の厳密和。same-fragment/basis-hashのgroup一致へ広げる場合は群質量の積和。一般の二位置lookup特徴はO(label数²)でよく、全12,487,400 eventsの列挙は不要。
- **範囲/実装性**：indicator、path内の2label比較、追加量子測定0。既存LocalControlは「特定の二位置label」には対応するが「両位置のlabelが任意に一致」にはそのまま対応しない。有限和/conditional adapterが将来必要。
- **優先度を下げる理由**：作用するh=2,3はN1で希少なorder-2層。全targetへの寄与はp_h²/n_hで重み付けされる一方、conditional train自体はfull価格。強い構造的相関があっても総費用改善とは限らない。

別fragmentのdiagonal support文字列だけで、共通system basisでのPauli一致・可換性を判定してはいけない。basis変換が違う。完全label一致、同じ保存basis hash下の対応など、SOURCEが保証する関係だけを使う。

## 6. 他候補と除外・障害

| metadata候補 | 層内変化 | 正確な平均の方法/障害 | quantum feature取得/追加負担 | 判断 |
|---|---|---|---|---|
| 特定位置label indicator、固定labelの組合せ | 自由位置では可。既存a/b条件そのものは固定 | 積law、包含排除、条件付き比 | 0/lookup。単labelが希少だとrank/学習が困難 | 候補1の基礎。全232 one-hotは低shot設計に不向き |
| Taylor次数、(-1)^(n/2) event phase、基本rotation角 | 全5層で定数 | nが既知なら平均も定数 | 0/ほぼ0 | 新しい層内削減なし |
| productsのsign積 | h=2,3で変わり得る。h=4は固定 | 条件付き一位置sign平均の積 | 0/2lookup | 候補2/3の拡張。signはphaseへ二重適用しない |
| 保存basis群・局所操作数の加法proxy | lookup値が自由label間で違えば可 | 一位置和、有限個の隣接項なら二位置和 | 0/catalog処理。真のcompiled RZとは別 | 平均は計算可能だがbitとの相関理由は弱め |
| full compiled RZそのもの、複雑なcompiler cancellation予測 | 変わり得る | E_Q[価格|h]は既知でない。一般には全supportのcompile/別の平均保証が必要 | 実compileと未知平均の費用 | 安い既知平均controlとは認定しない |
| 基底stateのPauli期待値、Born/path信号、真variance | 変わり得る | 求める応答oracleそのもの | 無料扱い不可 | 禁止 |
| 測定bit/bit-word由来の特徴 | 応答と結び付く | path-only条件・独立性を壊す | — | 禁止 |

## 7. 比較と調査上の推奨順

| 案 | 追加削減余地 | 平均取得 | 再利用 | 有限学習・古典負担 | 全段階RZへの可能性 |
|---|---|---|---|---|---|
| 1 rotation fragment群 | 主なh=1も対象。共分散があれば追加削減 | exact一位置和 | 非常に高い。scalar conditional adapterは新規 | 小。群質量を測定前に確認 | 第一候補。学習費を回収できるかは未検証 |
| 2 signed support | 自由rotationの演算子特徴。相関/変化は未確認 | exact一位置和 | 既存sign/support | 小。ただし幅2と非定数性に注意 | 第二候補。oracle改善を仮定しない |
| 3 product一致 | productの既知恒等関係を利用 | exact平方和/積和 | 既存metadata/indicator計算 | rare層のtrain費、効く層が限定 | 第三候補。小さいpだけで有用/無用を断定しない |

**追加量子測定なしで使え、少なくとも一つの同じB1-OL層内で非定数となるcontrolは存在する。** ただしCov_Q(u,B|h)の非零、有利なrange、train費回収はUNKNOWN。これは候補調査上の優先順位であり、新controlの正式採択や科学的GO/NO-GOではない。
