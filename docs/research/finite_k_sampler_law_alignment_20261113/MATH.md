# Law差の評価方法と適用条件

実H4への数値適用は未実施。以下はSOURCEに対応する数学的定義・条件付き上界であり、既存26値を変更しない。

## 1. 五つの対象を区別する

1. **解析的finite-K law A**：数学的τ、絶対係数から作る実数の正規化law。
2. **Cのsource law P**：SOURCEが生成/保存したbinary64 raw order weightとcomponent.probabilityを、正確な有理数として正規化するlaw。K1数値区間が含む対象。中心値のnominal演算は丸めを持つ。
3. **Q**：samplerがchoiceへ渡すbinary64確率ベクトルvを厳密に再正規化したlaw。
4. **F/G**：その実装CDFを連続一様/一様53-bit格子へ適用するlaw。
5. **実PCG64 joint law**：固定seed列、又は明示したseed ensembleでの列。Gの独立積lawとは別の対象。

P→Q→F→Gを正確に評価してもA→PとGの積law→実PCG64列は別に残る。eventの角度/作用/metadataも同一であることが必要。確率だけのTVは異なる回路wordの同値性を証明しない。

## 2. 正規化・CDF・格子

TV(p,q)=½Σ|p_i−q_i|。raw binary64重みをr_iとし、P_i=r_i/Σr。choice入力v_iをQ_i=v_i/Σvとする。CDF SOURCEの演算をそのまま行い、c_i=fl(fl(cumsum(v)_i)/fl(cumsum(v)_L))、c_0=0、c_L=1とする。

F_i=c_i−c_(i−1)。D=2^53、U=j/D (j=0,…,D−1) が一様なら、right比較の選択確率は

```
G_i = [ceil(D*c_i) - ceil(D*c_(i-1))] / D.
```

既存cdf_lawがこのGを返す。格子massは全て整数/Dで、binary64への変換でも厳密に表現できる。CDF閾値のFraction化とceilingにより、格子全列挙は不要。新diagnosticはP/Q/F/GをFractionで保持し、δ_norm=TV(P,Q)、δ_cdf=TV(Q,F)、δ_grid=TV(F,G)、δ_total=TV(P,G)を分離する。δ_total≤min(1,δ_norm+δ_cdf+δ_grid)。**L1とTVの係数2を混同しない。**

等しい確率がSOURCEから渡されても、丸め済みCDFと有限格子は別の写像。一方、dyadic確率と整列した閾値などではP=Q=F=Gも成立する。実H4でどちらかは今回未判定。

### 保存数値を読まずに導ける一般上界

u=2^-53。binary64のround-to-nearest、非負・有限入力、対象の非零除算がnormal範囲、overflow/underflowなしを仮定する。

v_i=fl(r_i/z)=(r_i/z)(1+ε_i)、|ε_i|≤uなら、共通の正規化定数zの誤差はQの厳密再正規化で消え、δ_norm≤u/(1−u)。この主張はλ/τ/libmや元component生成の誤差を含まない。subnormalを含む場合は絶対誤差項を別に評価し、本相対式だけを使わない。

γ=(L−1)u/(1−(L−1)u)、(L−1)u<1/2（従ってγ<1）とすると、非負の逐次累積和と最後の除算について

```
epsilon = min(1, (1+gamma)*(1+u)/(1-gamma) - 1)
delta_cdf <= min(1, (L-1)*epsilon).
delta_grid <= min(1, (L-1)/2**53).
```

前者は各内部CDF閾値とQの厳密prefixの差をepsilonで抑え、隣接差のTVをprefix誤差和で抑える。後者は各内部prefixの格子round-up誤差が[0,1/D)に入り、両端誤差が0であることから得る。これは保守的な一般式。実際の有限閾値から直接Fractionで得る距離の方が鋭いことがある。実H4でnormal条件や誤差の有用性を確認したとはしていない。

τ/係数の数学的real値との距離には、別途入力intervalとlibm誤差の根拠が必要。元のRTE truncation boundやnormalization Bはこのlaw差の上界ではない。Bを価格へ乗じない。

## 3. Order/componentからeventへ

order分布をp=(p0,p2), g=(g0,g2)、componentのTVをδ_cとする。orderをmaximal couplingし、一致したorder kでは、k+1個の独立component対をcoupleする。すると

```
delta_event <= min(1, TV(p,g)
  + min(p0,g0)*delta_c
  + min(p2,g2)*(1-(1-delta_c)**3)).
```

より単純にはδ_event≤min(1,δ_order+3δ_c)。必要条件は、各law内でorderに条件付けたcomponentが同じ順序で独立、相同event labelが同じ角度/phase/wordを表すこと。component選択順はrotation,product1,product2、作用順はproduct1,product2,rotationで、同一component lawのIID性がこの再配置を許す。実PRNGの相関を無視してよいとの証明ではない。

全component tupleを列挙せずに上界を求められる。新テストのtuple全列挙は2 componentだけの人工入力。

## 4. Trajectoryへの伝播

一致した履歴hに対して、次eventの条件付き分布距離が一様にδ_i以下なら、逐次couplingにより

```
delta_path <= 1 - product_i(1-delta_i) <= min(1, sum_i(delta_i)).
```

同一law n位置ならmin(1,nδ_event)。独立uniform格子drawを全piece/outerに仮定すれば十分。独立でなくても上記の履歴条件が証明されれば足りる。周辺分布だけの一致では足りない。人工例：2bitの各周辺が公平でも、独立二回と同一bit二回ではjoint TV=1/2。

最大n=33,423,360、component数L=324、order数L=2。格子部分だけなら条件付きunion上界の形は `n*(1+3*323)/2**53` だが、正規化/CDF/解析入力/PRNGを除く項であり、実H4の総law誤差や価格保証にしてはならない。巨大event列は一切生成しない。

固定seedの場合はSOURCEが一つの列を決定する。その列のDirac lawとideal product lawは、小さい1選択CDF距離では制御されない。seed ensembleとそのjoint性を定義しないまま、実samplerに上記の積law上界を付与しない。seed hash、PCG64のperiod/再現性、stream非重複はIIDの証明を代替しない。

## 5. 平均価格に必要な追加条件

同じ有限support上の有界fについて

```
|E_I f - E_Q f| <= (max f - min f) * TV(I,Q).
```

S≥0でもE[S]はmax Sではない。D_res=C−Sは符号付きなので、必要なのは全supportのoscillation。標本平均、sample variance、観測min/maxだけを全support boundとしない。

I-targetのmixed推定量`E_I S + mean(D_res under Q)`のbias項は`E_Q D_res−E_I D_res`。Q-targetでは`E_I S−E_Q S`。従ってlaw距離が得られてもoscillationが未知なら価格systematicの上界はnull。本moduleもその場合Noneを返す。

有限cohortの標本誤差は別途必要。X/Yは同一pathのpaired応答で、nを二倍にしない。cross-round・共有fitのcovariance、稀なorder2条件付き残差、長q/R移送は別の未資格事項。forced診断は既知のstratum weightと条件付き推定がなければnormal cohortへpoolしない。

SOURCE一致・同identity・同law_bindingというmetadataだけでは、これらの量が0又は適格とはならない。
