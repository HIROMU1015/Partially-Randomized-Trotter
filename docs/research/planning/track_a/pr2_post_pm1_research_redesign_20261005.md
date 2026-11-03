# PR-2 / Track A：PM-1後の研究方針・新規性・着地点の再設計

作成日：2026-10-05  
研究証拠：`HIROMU1015/Partially-Randomized-Trotter`、commit `194cc604b90c56a0e7e949b91b064a4bcfc846da`  
branch：`pr2-v4-s2-parallelization-20260928`  
位置付け：研究方針・完成条件の提案。既存結果・事前登録・authorizationの変更ではなく、新しい科学実行の認可でもない。

## 0. 結論

**Track Aは、部分ランダム化の新アルゴリズム開発ではなく、固定DF表現・二次PF・canonical finite-RTEの有限時間信号推定について、精度適格性と測定込み資源を区別する実証研究として完成へ向ける。**

PM-1で近接discard rank 4/5を追加しても、保存済みB2の低いprimary RZ点推定は覆らなかった。この結果は現在の比較の内的な信頼性を高める。しかし、新規性そのものを確定したわけではない。[R1]

今後の中心は「partialが勝つ例を増やす」ではなく、次の問いに置く。

> 同じ有限時間の複素信号を同じ精度で推定するとき、残差の破棄・決定論保持・ランダム補完の資源差を何が決めるか。精度を満たしていること、分割を細かくすること、残すfragmentを減らすことが、なぜ必ずしも低資源につながらないのか。

**直近に行うのは、研究の主張と終了条件を固定し、保存済みデータでPM-2の精度依存解析を行うための設計である。新しい分子、trajectory、PF family、synthesis policyを自動追加しない。**

新規性の評価は以下のとおり。

- 部分ランダム化、DF残差のrandom補完、discardとの比較、回路costと測定費用の共同評価：既知の範囲が大きい。
- 現在の実装条件で、有限信号の総bias・normalization・実compile費用をそろえた定量的な適用条件：実証的な差分候補。
- 新しい一般原理、最適selector、truth-free certificate、全RPE資源優位：未成立。
- 独立論文としての十分性：まだ確定しない。H4の改善率だけでなく、既存研究へ追加する具体的な設計上の知見が必要。

最新Track B資料は本レビューの根拠に含まれていない。したがって、Track AとTrack Bの研究価値の優劣や、Track Bへの移行は判断しない。

## 1. 情報の区分

本書は四種類の記述を区別する。

1. **記録された結果**：PM-1、PM-0、M1、M2の報告・保存値。[R1–R4]
2. **今回の算術・代数的整理**：保存値の比、同じPを加える場合の直線差、shot式の変形。新しいHamiltonian simulationではない。
3. **文献照合**：一次文献の具体的な貢献と今回の差分。[W1–W7]
4. **提案・仮説**：今後の主張、終了条件、必要な検査。未実行・未認可。

本レビューでは分子NPZ・runtime・cacheを開かず、新しいsignal、trajectory、回路compile、test再実行を行っていない。全source hash・raw checkpointの独立再監査を行ったとは主張しない。

## 2. PM-1で変わったこと

### 2.1 最良の近接discardは改善したが、B2の点優位は残った

H4 linear 1.00 Å、STO-3G、DF rank12、T=0.8、epsilon_complex=0.05、Qiskit1.3.0 opt1、状態準備除外という条件で、追加したB0 rank4/5×q=1/2/4/8の8構成は全件accuracy適格だった。[R1]

| 構成 | 総解析shot数N | 有効1-shot RZ | G_RZ |
|---|---:|---:|---:|
| 保存B2 rank3 q1 r4 K2 | 20,563 | 6,359.71875 | 130,774,896.65625 |
| 新B0 rank5 q1 | 25,910 | 8,866 | 229,718,060 |
| 旧B0 rank6 q1 | 24,017 | 10,550 | 253,379,350 |
| 保存B1 rank12 q1 | 18,471 | 20,168 | 372,523,128 |

B2と新B0の保存値から今回計算すると、

\[
\frac{G_{B2}}{G_{B0,L=5}}=0.56928435,
\]

すなわちB2のRZ work点推定は約43.07%低い。逆向きの比は1.75659である。「B2が76%低い」とはしない。

また、

\[
\frac{G_{B2}}{G_{B0,L=5}}
=\frac{N_{B2}}{N_{B0,L=5}}
 \frac{C_{B2}^{\mathrm{eff}}}{C_{B0,L=5}^{\mathrm{eff}}}
=0.793632\times0.717315.
\]

新B0に対しては、B2はshot数と1-shot費用の双方が小さい。B1に対してはB2のshot数が多く、1-shot費用の低さが総資源を下げる。比較相手ごとに利益の説明が異なる。[R1,R2]

ただし、B2は既存32-trajectoryから得たcompiled-cost点推定である。今回の比をformal confidence intervalや厳密なwinner判定とはしない。

### 2.2 最小限の主張は更新できる

> 登録した二次DF-prefix・canonical finite-RTEの実装条件では、近接discard rank4/5を追加しても、intermediate partialの低いprimary資源点推定は維持された。残差破棄の精度適格性だけでは、資源上の優劣は決まらなかった。

これは全prefix、全PF、全compilerに対する最適性ではない。未登録rankや未評価の合成policyを不利と仮定しない。

### 2.3 M2の意味を拡張しない

M2はdevelopmentで固定した5構成のH4 1.30 Åへのtransferを支持した。そこで比較したdiscardはrank6であり、PM-1で新たに得たrank5ではない。[R3]

したがって、「強化したrank5 baselineに対してもheld-out優位が確認済み」とは書かない。M2の正式判定は保持するが、新しい比較集合の検証に読み替えない。

## 3. PM-1の、勝敗以外の重要な観測

### 3.1 accuracy適格でも、必要shot数は大きく違う

rank4は全qで適格だったが、q=1でも109,738 shot、q=8では1,733,007 shotを要する解析値となった。rank5 q1は25,910 shotである。[R1]

現行規則は

\[
s_{x,a}=\epsilon/\sqrt2-b_{x,a},\qquad
N_{x,a}=\left\lceil\frac{2B_x^2}{s_{x,a}^2}\log\frac2{\alpha_a}\right\rceil.
\]

正規化した残りの統計誤差予算を

\[
h_{x,a}=\frac{s_{x,a}}{\epsilon/\sqrt2}
\]

とすれば、ceilを除いた式では、bias=0・B=1の場合に対するshot倍率はB_x^2/h_{x,a}^2となる。

これは新しい定理ではない。しかし、精度を満たすかどうかを二値で見るだけでは、費用の大きな差を説明できないことを明示する座標になる。

注意：accuracy-ineligibleは現在の軸別予算・Hoeffding規則における不適格である。他の誤差配分や推定器でも推定不能であることを意味しない。解析shotは十分条件からの費用であり、情報理論的な最小shot数ではない。

### 3.2 qを増やしても、full-Hへの総biasは減らなかった

| L_D | q=1総bias | q=8総bias | q=1 G_RZ | q=8 G_RZ |
|---|---:|---:|---:|---:|
| 4 | 0.026452635 | 0.033647195 | 784,187,748 | 95,311,918,986 |
| 5 | 0.013400419 | 0.020621349 | 229,718,060 | 3,259,129,200 |

観測としては、今回の二rankではq増加に伴いfull-target総bias、shot数、1-shot費用がいずれも増えた。[R1]

ここで「Trotter誤差が増えた」と断定しない。B0はtruncated HamiltonianをTrotter化しており、評価targetはfull-Hである。

\[
z_{L,q}-z_H
=(z_L-z_H)+(z_{L,q}-z_L)
=:d_L+p_{L,q}.
\]

したがって、

\[
|z_{L,q}-z_H|^2
=|d_L|^2+|p_{L,q}|^2+2\operatorname{Re}(d_L^*p_{L,q}).
\]

PF成分が減っても、full-targetへの総biasが単調減少するとは一般には限らない。この恒等式自体は初等的であり、新規性にしない。

**現時点ではexact truncated-H signal z_Lが保存されていないため、今回の非単調性の原因が誤差相殺であることは未確認。** PM-1のpure_discard/pure_pf=nullは維持する。係数フィットやq=8をexactと見なす代用で、この欠測を埋めない。

### 3.3 現在のepsilonで、近接discardのP sweepを追加する必要はない

保存B2 r4と新B0 rank5 q1へ、同じ状態準備RZ相当費用P/shotを加えると、

\[
G_{B0,L=5}(P)-G_{B2}(P)
=98,943,163.34375+5,347P>0\quad(P\ge0).
\]

他の新B0七件も、このB2よりG(0)とNの双方が大きい。従って**現行epsilon=0.05・保存点推定・共通Pモデルでは、新8件のB0はこのB2を越えてRZ lower envelopeへ入らない**。

これは保存値からの今回の代数的整理である。異なるepsilon、異なる状態準備法、MC不確かさを含む優越保証には拡張しない。全候補でB2がPの全域で最良とも言わない。

## 4. これまで修正した帰属は維持する

PM-0により以下が確認済みである。[R2]

- 同じM1候補domainではq=8でもB2 rank3がprimary最小だった。旧S2→M1のrank変化をq最適化だけに帰属しない。
- 64件のproxy frontierはactual Pareto二件とも保持し、旧16件selectorのprimary RZ regretは0だった。secondary frontier損失とprimary意思決定損失を分ける。
- M1/M2のP感度は同じ5構成にそろえると同じ方式順となる。候補集合差をgeometry効果にしない。
- 同じrank/K/T/R=qrではnormalizationとexpected random actionsは一致するが、PF信号とcompiled costは一致しない。
- basis changeは既に無制御、diagonal primitivesだけがcontrolledで、同一blockのstep境界half-sweep融合も実装済み。「既知最適化が何も入っていない」という前提でbaselineを批判しない。

これらを再度主張候補へ戻すことはしない。

## 5. 一次文献による新規性の評価

### 5.1 最接近文献：Güntherら

*Phase Estimation with Partially Randomized Time Evolution*は、部分ランダム化、RTEによる信号、single-ancilla位相推定、量子化学資源評価を既に扱う。Sec. VIIのfactorized representationでは、大きいDF fragmentを決定論側へ残して残差をrandomizeする考えと部分基底変換がある。[W1]

App. D.2、Figs. 19–20では小さい項の単純truncationとそのground-state energy errorも調べている。この特定のtruncation比較ではTrotter errorを含めず、本文はTrotter・phase estimationへも誤差予算が必要であると述べる。[W1]

従って「DF残差をrandom補完した」「discardを比較対象に置いた」だけは新規性ではない。今回の差分候補は、特定の有限信号taskについて、discard+PFの合成bias、finite-RTE normalization、軸別統計予算と実wrapper費用を同じ候補domainで評価したときの、具体的な設計判断である。

これは元論文全体が誤差・費用の接続を考慮していないという批判ではない。estimand、費用指標、比較集合が異なることを明記する。

### 5.2 その他の重要な近接研究

| 一次文献 | 既知として扱う内容 | 今回の非claim |
|---|---|---|
| Hagan–Wiebe, Composite Quantum Simulations [W2] | TrotterとqDRIFTの合成、分割に伴うerror/costの競合 | hybrid原理の初提案ではない |
| Pocrnicら, Composite Qdrift-product formulas [W3] | 実時間・虚時間のcomposite simulation、誤差条件に対する数値的費用評価と分割 | hybrid法の数値的資源比較自体を初めてとはしない |
| Casaresら, SPRINT/GRADE [W4] | factorization、near-integrable PF、randomization、symmetry、QROMと具体的化学taskでのcost/error設計 | 複数の工夫の統合やtask-aware最適化一般の新規性ではない |
| Oumarouら, RC-DF [W5] | regularized compressed DFによる資源削減 | prefix切断を新しい圧縮法と呼ばない |
| Kanasugiら [W6] | single-ancilla Trotter QPE、部分ランダム化、UWC、化学系のearly-FT資源評価 | QPE全体や物理資源の新しい優位性を示したとはしない |
| Cugini–Atif–Subasi [W7] | randomized protocolの回路costとvarianceを含むimportance sampling最適化 | cost×shot一般原理やsampling分布まで含む最適性を主張しない |

### 5.3 新規性の結論

**確立した新アルゴリズムの新規性は、現在のTrack Aにはない。**

残るのは、限定した実装・taskでの実証的知見である。例えば「精度を満たす候補でも統計予算が小さいと総資源が大幅に増える」「discardとPFの総biasを用いるとqの資源判断が単純なPF精度順と一致しない」「無調整で移した構成が所定の範囲で競争力を保つ」といった知見を、定量的な領域・原因・限界として示す方向である。

ただし前二つの一般論は既知の数式から予想可能であり、観測一件だけで強い新規性になるわけではない。独立した論文を狙う場合は、どの既存評価では分からなかった設計上の判断が変わるか、またそれがどの範囲で有効かを示す必要がある。

本調査は主要な近接一次文献との重点照合であり、同一研究の不存在証明や網羅的systematic reviewではない。現在の細部と完全に同じ比較が他にないこと、査読で十分な貢献と判断されることは保証しない。

## 6. 推奨するRQと表題

### 主RQ

> 固定DF表現と二次PF・canonical finite-RTEに基づく有限時間のcoherent-signal推定で、所定の精度を満たす候補の資源競争力を何が決めるか。残差の破棄・決定論保持・ランダム補完の選択は、総bias、normalization、回路費用、要求精度によってどのように変わるか。

### 副RQ

| RQ | 調べる対象 | 既存証拠と不足 |
|---|---|---|
| RQ1：精度適格性と資源効率 | 適格候補内のNと1-shot costの競合 | PM-1とPM-0で一部支持。要求精度を変えた境界は未解析 |
| RQ2：誤差と離散化の役割 | discard+PF総bias、q、R=qrの影響 | 同一R比較はある。discard/PFの純分解は未評価 |
| RQ3：成立領域と移送限界 | 同じ候補domainでのepsilon/P依存、固定構成transfer | M2は元の固定5構成について支持。新rank5を含むheld-out最適性は未検証 |

仮題：

> **DF-prefix時間発展における残差処理と測定込み資源の比較**  
> ――有限時間信号の精度適格性、総bias、回路費用の関係

英語例：

> **Resource trade-offs between truncation and randomized completion in DF-prefix coherent-signal estimation**

比較していない複数compilerや実装policyへの頑健性を表題で示唆しない。「resource-optimal」「chemically accurate ground-state estimation」「新しいpartial algorithm」を表題に入れない。

## 7. 中心claimの階層

### 現在そのまま書けるclaim

> H4の固定DF・二次PF実装について、有限時間複素信号の同じ精度条件で評価したところ、近接discard候補を追加してもintermediate partialの低いRZ-work点推定は維持された。discardの適格性だけでは資源効率は決まらず、総biasによる統計予算の消費と1-shot費用の双方が重要だった。事前固定した元の5構成については、別geometryへの無調整transferも支持された。

### 完成研究の目標claim

> 明示した有限候補・実装classで、どの要求精度と測定負担の範囲にpartialの資源競争力が残るかを定量化し、優劣とその変化を総bias・normalization・回路費用から説明する。固定構成の移送と再最適化、点推定と保証を区別して報告する。

### 現段階で書けないclaim

全prefix/全PF/全sampling法での最適性、強い任意の決定論実装への一般的優位、未知大系で使えるtruth-free選択、状態準備込み物理資源の優位、化学精度の全RPE/QPE性能、r4/r8の厳密winner。

現在のepsilon_complex=0.05、T=0.8は有限信号taskであって、エネルギー推定全体の要求ではない。元の研究目標を満たしたことにせず、独立した限定成果として扱う。

## 8. 研究としての着地点

### 8.1 最小の完結点：機構と限界を備えた限定case study

M1/M2/PM-0/PM-1を統合し、共通の候補domain、N×1-shot分解、現在の精度での点順位と反証、既知研究との差分、未評価範囲をまとめる。

H4二geometryに限ることを前面に出し、再現可能なresource/application noteまたは技術報告として閉じる。形式上の呼称だけで貢献が増えるわけではない。単なる実行履歴ではなく、比較方法と解釈が他の研究者に再利用できる形にする。

### 8.2 推奨する目標：適用条件を説明する限定数値研究

上記に、保存値を用いた要求精度依存の資源境界を加える。代表的な順位変化、または順位が変わらない理由を総bias・統計予算・normalization・費用から説明する。

主因としてdiscard/PF干渉を主張するなら、その二成分を直接分離する最小referenceを取得する。実装に依存しない優位を主張するなら、適用可能な既知の強い実装について一つの公平な対照を置く。

全てのPF・分子・hardwareへ拡大する必要はない。独立論文に十分な知見が残らなければ、最小の完結点へscopeを下げて閉じる。

### 8.3 発展は別成果として管理

化学精度のエネルギー推定には、入力overlap、alias、時間系列、RPE/QPE推定器、roundごとの誤差・成功確率、状態準備・synthesis費用の接続が必要である。現在の短時間結果を長時間へ外挿しない。

Track Bのmethod co-designは別RQであり、Track Aのpublication価値を補うために未検証の新規性を借りない。

## 9. 次の作業：PM-2を保存済みデータから設計する

### 9.1 目的

研究の成立可能性を無期限に調べ続けるのではなく、完成原稿の中心図と説明を作る。

> 固定候補について、要求精度epsilonと共通状態準備費用Pを変えたとき、資源上の選択はどこで変わるか。

### 9.2 データ集合を明示する

- development：M1の210候補＋PM-1の新8候補。218候補を基礎集合とする。epsilon=0.05では記録上214件が適格。
- transfer：M2の元の5構成だけ。developmentの218候補と同じ最適化集合として比較しない。
- M1/M2/PM-1の元status、fingerprint、費用・信号値を変更しない。
- 追加解析はPOSTHOCと明記し、M2を再びfresh held-outとして使わない。

### 9.3 保存値だけで計算できるもの

候補xについて保存済みb_{x,a}、B_x、軸別平均compiled costがあれば、

\[
\epsilon_{\min,x}=\sqrt2\max_a b_{x,a},\qquad
G_x(\epsilon,P)=\sum_aN_{x,a}(\epsilon)(\overline C_{x,a}+P)
\]

を再計算できる。境界ちょうどはs=0で不適格である。

epsilon=0.05を必ず再現し、その上下の採用領域を調べる。表示範囲を結果の見栄えで決めず、task上の関心と保存候補の適用域から結果前に定める。適用範囲外は「未登録候補が必要」「保証対象外」とし、q/r/Kの追加計算で埋めない。

入力として、元のepsilonで不適格だった候補も保持する。他のepsilonで適格になり得るため、最初の206件だけを残すと比較domainが変わる。

### 9.4 Pについて不要な再検査を避ける

epsilon=0.05で新8discardがB2よりNとG双方で大きいことは直線の不等式で処理できる。このケースのP-grid計算や新しい状態準備compileは不要。

他epsilonではNが変わるため、そのepsilonで新たに交点を評価する。Pは共通RZ相当/shotの仮想費用であり、実際の状態準備を測った値と混同しない。

### 9.5 不確かさと成果物

random compiled costは保存されたtrajectory統計の範囲で評価する。formal CIのない領域は点推定感度として示し、選択後の単純±2SEを厳密なfamily-wise保証へ格上げしない。

必須成果物は、(a)共通集合の資源図、(b)適格性と資源最良の違い、(c)代表的なN/B/bias/C分解、(d)既存claimの更新表、(e)未評価domainである。

PM-2では分子NPZ、signal再計算、trajectory、compile、量子測定、新規ground-state solve、全repository testsは行わない。

## 10. 条件付きの最小追加計算：誤差機構の二reference

PM-2後の原稿で、PM-1のq依存をdiscard/PF相殺で説明することが中心claimになる場合だけ、同じdevelopmentのrank4とrank5について

\[
z_L(T)=\langle\psi|e^{-iH_D(L)T}|\psi\rangle
\]

を各一件、計二つ取得する案がある。

これは新しいground stateを求める計算ではなく、同じ保存入力状態に対するtruncated-Hの正確な時間発展信号を得る計算である。ただし新しい科学計算であるため、別の入力・source・出力契約と利用者認可を要する。

取得後は既存q=1/2/4/8の信号を使い、d_L、p_{L,q}、複素内積の交差項を計算する。相殺がなければ仮説を棄却し、別geometryや新しいrankで相殺例を探して救済しない。

原稿のclaimを観測された総bias・費用の非単調性に限定するなら、この追加計算は必須ではない。名前だけの誤差分解を行うより、未確定を明記する方が正確である。

## 11. strong baselineは主張に対応させる

### 11.1 必要な強さを先に選ぶ

- 「現在の二次DF-PF実装classでの結果」：そのclassの定義と妥当な実装確認が必要。全高次PFの網羅比較は不要。
- 「一般的な有力決定論法より優れる」：有力な高次PFまたは既知の低cost実装を含む公平な比較が必要。
- 「実装変更にも頑健」：少なくとも主張を脅かす具体的な実装変更への検査が必要。

### 11.2 文献上の具体的な候補

Simon–Loveの対称Trotter制御化は、標準的な制御化で増える任意角回転を削減する。[W8] ただし総compiled RZが一律半分になるわけではなく、DFでのbasis費用や現在の既存最適化を含めて確認する。

既知の相対Gaussian基底変換融合も候補だが、PM-0で確認済みの無制御basisと同一block融合を「未実装」として二重計上しない。[R2,W4]

選んだpolicyが数学的に適用できる構成には公平に適用する。B0/B1だけに有利な計算モデルへ替えてB2を古い模型のまま比べたり、逆にB2だけへ最適化を与えたりしない。

### 11.3 判別に必要な改善量

固定epsilonで、signalを変えない等価実装によってB0 rank5とB2の1-shot費用がそれぞれk_0、k_2倍になるとき、B0がB2を下回る条件は

\[
\frac{k_0}{k_2}<0.56928435.
\]

これは現在の点推定比からの計算である。B0が43%以上削減されれば必ず勝つという意味ではなく、**B2側にも同じ機会を与えた相対的な改善**が必要になる。

具体的な適用性を静的に監査し、この比較がclaimを本当に変え得る場合だけ、一つのpolicyの最小比較を別計画にする。最初からstrong synthesis・高次PF・新分子を全て必須にしない。

## 12. 停止・完成規則

### A. PM-2で説明可能な境界や失敗条件が得られる

その内容を主結果とし、原因の直接検査が必要な一点だけを追加する。独立条件への一般性を主張しない限り、新分子を自動追加しない。

### B. 広い範囲で既知trade-offどおりで、特定H4の順位表以上が残らない

単純な再現を新原理として売らず、限定case studyとして完成させる。別のepsilon、geometry、seedを繰り返して新規性を探さない。

### C. 既知の強いbaselineでB2の順位が変わる

変更をそのまま報告し、旧M1/M2/PM-1を無効化しない。旧実装classで成立した結果と、比較条件変更後の結果を分離する。優位が消えた理由が研究上有用なら適用限界として残す。

### D. r4/r8などの細かい順位だけが不確か

その不確かさは保持する。新しい32/96trajectoryを自動追加しない。研究の結論を変える差だけを精密化の候補とする。

**本研究の終了線は、「十分多くの系でB2が勝つ」ではなく、固定した問いに対する主張・原因・適用域・限界がそろうことである。**

## 13. 完成原稿の構成案

1. **問題設定**：full-Hに対する有限coherent signal、同一accuracy、残差処理三方式とrandom-dominant参照。
2. **既知方法と比較契約**：DF-prefix S2、canonical finite-RTE、bias/normalization/解析shot、full wrapper、候補domain。
3. **development resource map**：M1とPM-1を統合。元の登録比較と追加反証を時系列で区別。
4. **機構**：N×1-shot、総biasと統計予算、同一Rでのq比較。pure分解の既知・未知を分ける。
5. **適用条件**：epsilon/P依存、必要なら一つの強い実装対照。
6. **固定transferと限界**：M2元5構成、真値依存、MC費用統計、単一family、非FT総資源。

主図は「候補全体の資源比較」「適格性と必要shotの違い」「同一Rのq依存」「epsilon/Pでの採用領域」「M2固定transfer」の五つ程度。図の数自体を達成条件にはしない。

本文ではhash・test数を新規性の根拠にしない。再現性付録にはsource、環境、入力とartifactのidentity、未収録runtimeの範囲を整理する。

## 14. Codexに渡す次の作業範囲

以下は方針を採用した後の**PM-2準備**の指示案であり、この文書だけで実行を認可しない。

```text
Track Aを、固定DF・二次PF・canonical finite-RTEの有限時間coherent-signal資源研究として完成へ向ける。
基準commitは194cc604b90c56a0e7e949b91b064a4bcfc846da。

1. M1/M2/PM-0/PM-1の正式結果・契約・fingerprintを変更しない。
2. 主RQ、現時点のclaim、非claim、最小成果、目標成果、終了条件を文書化する。
3. PM-2は保存済みJSON/CSVの再解析だけとして設計する。
   developmentはM1の210+PM-1の8、M2は元5構成を別扱いにする。
4. 保存された軸別bias、normalization、compiled mean/統計を確認する。
   保存値の欠落を推測で埋めず、元の不適格候補も保持する。
5. epsilon=0.05を再現すること、epsilon_min境界、P交点、N×C分解、点推定/不確かさをschemaへ固定する。
6. epsilon/Pの表示範囲と比較規則を結果に依存させない。
   current epsilonでPM-1新8候補が共通Pの全域で既存B2に点支配される部分は代数で処理する。
7. 新しい分子NPZ、signal、trajectory、compile、QPU、全repo testsを実行しない。
8. exact truncated-H二reference、strong synthesis、高次PF、別分子、Track Bは条件付き別作業。
9. 元M2を新候補集合のheld-out最適性へ読み替えない。
10. 準備資料を保存して停止し、未承認の解析・科学実行へ自動進行しない。
```

## 15. 最終判断

**現在は研究をもう一度八方向へ広げる段階ではない。PM-1で支持された限定結果を、精度適格性と資源効率の違いを説明する完成可能な研究へ収束させる段階である。**

新規性は、構成要素の組合せや成功率の高さではなく、既存研究に追加する設計上の知見で評価する。現在の証拠だけで強い新手法論文と認定しない一方、機構・境界・限界を備えた限定実証研究として閉じる価値は残る。

## 16. 参照資料

### リポジトリ（研究証拠commit 194cc604b90c56a0e7e949b91b064a4bcfc846da）

[R1] `docs/pr2_pm1_discard_result_validation.md`。PM-1 8候補の結果、比較、費用scope、停止状態。

[R2] `docs/research/pr2_post_m2_evidence_attribution.md`。PM-0、同一domain、selectorとproxyの区別、N×C、同一R、共通P、欠測。

[R3] `docs/pr2_matched_accuracy_m2_transfer_result_validation.md`。M2固定5構成、transfer、uncertaintyとP感度。

[R4] `docs/pr2_matched_accuracy_m1_b1_result_validation.md`。M1登録候補、compiled資源と制約。原始evidence commitはb6e65c6123475add5e620ec1064f361378bead95までに固定。

[R5] 添付された `pr2_post_m2_research_redesign_20261004.md`。本書はこの提案をPM-0/PM-1で更新するものであり、以前の原文を改変しない。

### 一次文献（2026-10-05重点照合）

[W1] J. Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332 (2026). DOI: 10.1103/ynxb-p2xq. arXiv:2503.05647. 特にjournal版Sec. VII、App. D.2、Figs. 10,19,20を本文・ページ画像で確認。

[W2] M. Hagan and N. Wiebe, *Composite Quantum Simulations*, Quantum 7, 1181 (2023). DOI: 10.22331/q-2023-11-14-1181. arXiv:2206.06409v3.

[W3] M. Pocrnic et al., *Composite Qdrift-product formulas for quantum and classical simulations in real and imaginary time*, Physical Review Research 6, 013224 (2024). DOI: 10.1103/PhysRevResearch.6.013224. 本書での参照範囲は出版社abstractにある数値的費用評価・composite法の対象。

[W4] P. A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1 (2026). GRADE、near-integrability、randomization、QROM、具体的taskでの誤差・費用設計。

[W5] O. Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum 8, 1371 (2024). DOI: 10.22331/q-2024-06-13-1371. arXiv:2212.07957.

[W6] S. Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv:2603.22778v2 (2026).

[W7] D. Cugini, T. A. Atif, Y. Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1 (2026).

[W8] W. A. Simon and P. J. Love, *Halving the Cost of Controlled Time Evolution*, arXiv:2511.13855v1. 対称Trotterの制御付き任意角回転の削減。DFの総compiled RZ削減率を直接与えるものとは解釈しない。
