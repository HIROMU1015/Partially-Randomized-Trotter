# 価格推定方法別の検証計画

2026-11-13 JST。**未採択・未実行。** 11候補menuは暫定のまま。以下は必要domain・研究上の精度が決まった後に選ぶ検証契約で、現在のD/Cに新model実装やcompileを指示しない。

## 1. まず検証する対象を固定する

domainは「H/state/preparation・candidateの全schedule・m/axis・q/R/K/allocation/law・compiler/lowering・測定付きfull wrapper・P0」の組。固定PRで全13roundを使うならm12を含む。別R・別formula・Full-H preparation・旧loweringは別domain。13roundの離散domainと連続q/Rへの一般外挿も分ける。

価格targetをideal finite-K law Pとするか、具体的sampler law Qとするかを宣言する。S3のIID shot lawはpath価格のideal/sampler同一性を証明しない。CのE_P[S]とDのsampler標本を組み合わせるには、その違いを扱う必要がある。

## 2. 三つの方法の判定基準

| 方法 | 何を推定するか | 必要な校正・独立検証 | 採用可能条件／不成立時 |
|---|---|---|---|
| 直接paired sampling | 同path・同SOURCEでCとSを取りD=C−S。既知E[S]をcontrol variateとして`c=E[S]+mean(D)`。直接mean(C)も比較可能な設計選択肢 | fit係数の校正は不要。既存pilotは分散/負荷を設計するdata。凍結したn/lawで新しいnormal cohortを取得し、適切なmean CIを使う | 未測定roundのDは埋まらない。必要なroundだけ直接測るか、未評価域を残す。pilot n4の分散を真値にしない |
| 構造量による解析補正 | Dをbasis相殺/head/outer/wrapper等の構造featureから導出。係数が全て演繹的ならfree parameter p=0 | 合法な全path/境界/rareを含む導出と、feature期待値の取得法・誤差。独立実測は実compilerとの対応をchallengeする | 数式がSを足し戻しただけ、局所構造だけ、又はcompiler最適化の条件外なら未資格。根拠のない係数を持てば次のmodelとして扱う |
| 少数parameter残差model | 事前固定featureとθでD平均又はpath残差を予測 | calibration designのrank、校正meanの不確かさ、凍結後の独立holdout、domain全体のmodel discrepancyの根拠 | p個の係数にp点が合うだけでは検証でない。holdout不合格は直接法/狭domainへ戻す。都合のよいrefitで同じholdoutを再利用しない |

直接法のpairedは「同じevent pathのC−S」であって、X/Yを独立標本にする意味ではない。E[S]が数値intervalを持つ固定値なら、その誤差を別に含める。C−SとC−C_hatなど、別の残差を混同しない。

解析式やmodelが`D_hat(x)=θᵀφ(x)`なら、価格meanにはE_P[φ]が必要。C/Dが既存kernel/解析式で供給できるか、samplingが必要かを明示する。Bが新しいfinite-K kernelやfeature期待値solverを作ることはしない。特徴期待値が不足なら、その方法もまだ未完成である。

## 3. calibration・holdoutの最小構成

- **既存P1/P2**：development/calibration候補として再利用できるが、これらを見て方法・特徴・精度を選んだ後の独立holdoutではない。replayは技術同値性の確認で、fresh statistical nへ加えない。
- **p自由parameter**：独立なscalar design rowが少なくともp個、design matrixのrank=pが必要。X/Yが同じfeature列又は同じpathの対だからといってrow数・rankを2倍にしない。平均の反復を増やしてもq/R等の列の共線性は解決しない。
- **校正点の追加**：既存designのrankがrなら不足方向は少なくともp−r。これは最小の識別条件で、model妥当性ではない。内部のlack-of-fit診断も必要ならp+1以上の独立designを候補にするが、p+1を機械的な精度保証にしない。
- **独立holdout roundの最小challenge案**：最大使用round **m12** と、C/Dが構造regime/境界/feature leverageから事前指定する**内部1round**の計2round。内部点は例えばm4又はm10だが、保存価格を見て有利な点を選ばない。m7/8/9を順に測る要件はない。
- **path数**：round数とpath数を分ける。各roundのn_hは第5節のCI幅から決める。n1やn4というfloorだけでは精度を認定しない。p=0の解析式も実装対応を独立にchallengeし、model選択に使ったcellをそのまま唯一のholdoutにしない。

qとRは固定κのschedule上で強く連動する場合がある。q依存とR依存を別々に同定したいmodelがrank不足なら、schedule上だけのmodelに限定するか、ユーザー採択後の別legal designが必要。Bが未承認の別Rを校正点へ追加しない。

**重要：2round holdoutは最小の方法診断案であり、残り11roundの一様認定ではない。** 任意のmodelは未測定roundだけ大きく外れることができる。全13へ使うためには、C/Dの全domain解析的誤差上界、合法feature/regime被覆に基づく移送根拠、又は残る必要roundの直接CIが要る。有限13セルのうち2つが通ったという事実だけで`qualified_for_candidate=true`にしない。

## 4. 最大round・長時間と実行可能性

m0/m6のP2はm12の係数・分散・rare寄与・実行時間を資格付けない。m12のfull-wrapper direct compileが実行できるか、Dが実装・資源見積りを供給する。構造E[S]の計算が速いことはfull compilerの実行可能性を示さない。小roundの時間をevent数に比例させてm12の上限としない。

最大使用roundを直接challengeできない場合は、次のいずれかを選ぶ。

1. 実compilerとの全domainの関係を解析的に囲むC/Dの証拠がある範囲だけを利用する。
2. 実測可能なroundだけ価格を取得し、価格domainをそこまでに限定する。全13roundのGは欠測のまま。
3. 最終比較に必要な未測定roundだけを別の直接実測計画にする。新solver/新compile shortcutの開発が必要ならDの別SOURCEで扱う。

隣接m7の小path負荷診断はDの運用上の選択肢であり、価格model検証の必須段階ではない。m7/8/9を順に実測することでm12までの科学資格が自動的に増えるという設計にしない。

## 5. 独立実測のpath数・confidence・精度

### 直接法／holdout meanの幅

価格又は残差のpath変数Zが独立・同law、全populationでZ∈[L_Z,U_Z]、幅W_Z=U_Z−L_Zであると根拠付きに固定でき、meanの絶対半幅t、二側failure budget αを採択する場合、保守的な有限案は

\[
n\ge \left\lceil {W_Z^2\over2t^2}\log{2\over\alpha}\right\rceil.
\]

Dは負になり得るため幅を使う。標本min/maxや標本SDをpopulation supportへ代用しない。既存公開のbounded-mean/concentration手法をC/Dが選び、適用条件・数値処理を示す。母分散上界を持つ別手法・正規近似/Student区間等を選ぶ場合、その仮定と名目性を区別する。n4のpilot分散plug-inは標本数の見積りに留まり、CI達成の証拠ではない。

凍結予測d_hと、独立holdoutのmean interval `[Dbar_h−t_h,Dbar_h+t_h]`が有効なら、保守的なmean予測誤差上界は

`|E[D_h]−d_h| ≤ |Dbar_h−d_h|+t_h`。

従って**観測差＋holdout sampling幅≤採択したmodel用誤差予算**をmean予測の検証条件にする。単に「予測値が幅広いCIの中に入る」「RMSEが小さい」「相関係数が高い」は精度証明にならない。このboundが既に校正の影響も含むheldout mean対予測の全差を評価する場合、同じfit誤差をさらに足さない。

### G精度からのpath配分（N資格後だけ）

独立cell・分散上界σ_j²・path単価k_j、MC予算τが根拠付きにあると仮定すると、`Σ N_j² σ_j²/n_j≤τ²`の下で仕事量`Σ k_j n_j`を抑える連続設計は`n_j∝N_j σ_j/√k_j`を参考にできる。これはSD尺度τの設計でCIそのものではない。paired X/Yは同pathのN_XD_X+N_YD_Yの分散にcovarianceを含め、shared fit/cross-round依存があればこの独立cell式は使わない。

現在のN/σ/kを代入していない。Nが未確定ならセル別t_j又は相対誤差profileから独立に設計し、後で適格N集合に対して精度を再点検する。未知Nをnominalに置き換えてcompile予算を「最終比較に十分」としない。

### 同時confidence

必要confidenceは最終比較の主張水準と、選択する候補/domainからユーザーが決める。95%を未採択のまま例に使うことと、正式confidenceを与えることは別。対象cell/holdout/fit/law項にα_jを割当て、Σα_j≤α_priceとするunion bound、又は適用可能なjoint法を使う。26軸Bonferroniは1固定候補の一案で、11候補を選ぶ比較へそのまま拡張しない。

X/Y同値の定理があれば必要検証を削減できるが、今回pilotで一致しただけでは1軸のcompileやCIを省略する根拠にしない。正負の共有校正誤差・cross-R covarianceをunknownのまま0にしない。

## 6. rare・law差と二重会計

P2保存REPORTの「fresh全pathでorder2=0」は期待寄与0ではない。必要ならnormal/rareを分け、target stratum確率p_sと条件付き価格meanを使う`μ_C=Σ p_s μ_(C|s)`。forced-rare pathはnormal cohortへpoolせず、正しい条件付きsampler・重み・support・CIを別に供給する。stratum確率が解析的に不明ならその不確かさも含める。

稀な集合の確率上界p_Rと価格/残差の適切なsupportがあれば、未測定tailの寄与をその範囲で保守的に囲める。零観測CPはその集合の確率の診断であり、巨大なcost tailを単独で無視させる証拠ではない。全law bounded MC CIが既にrareを含む場合は、この寄与を二重加算しない。

同じD=C−Sが全wrapper/loweringで定義され、P=ideal、Q=samplerとして、

`μ_C^P = E_P[S] + E_Q[D] + (E_P[D]−E_Q[D])`。

TV(P,Q)≤δ、D∈[L_D,U_D]の根拠があれば、最後の絶対値は`(U_D−L_D)δ`以下。Qをtargetにするなら`E_Q[S]−E_P[S]`の補正が必要。E_P[S]とE_Q[D]を無条件に足してP又はQのcompiled期待値としない。TVはここでは**1 full path**のlawで、MC path数をその期待値biasに掛けない。event TVからpathへ移すときは、そのevent数/依存の適切なcoupling境界をCが与える。

構造S数値enclosure、fit、同時holdout、rare/lawのどれが各区間を覆うかを一度だけ記録する。未取得項はnull、理由を添付し、負Dのclampや旧C1 modelの上乗せをしない。

## 7. 検証の終了とfallback

凍結方法が指定domainと誤差予算を満たせば、その範囲だけ価格受入候補になる。落ちた場合は結果を保存して、直接法・狭domain・必要round追加の別有限案へ返す。holdoutを見て特徴を替えた時点でそのdataはdevelopmentとなり、新しい独立holdoutが必要。途中のCIを見てpathを無制限追加しない。固定n又は事前定義された有効sequential法と最大scopeだけを使う。

この計画自体は価格・モデル・2%/95%の正式採択ではない。ユーザーの有限科学実行承認までは、source metadata/人工確認だけに留める。
