# PR-2 主研究契約

日付: 2026-09-27  
状態: `PR2_S0_CONTRACT_AMENDED_PREFIX_IDENTITY_GATE`  
数値計算: 本契約の作成では追加実行していない

2026-09-27の実装監査で、初版SHA-256
`1bddb8d385d2ec3a6cdfb0bf025a3a6780b9917a3670526879151869e4fd553c`の
「通常DF-prefix PRと同一」という表現を訂正した。pilotが直接確認したのは**生成順prefix**との一致であり、
現行`prepare_df_partial_s2`が使うweight再順位付けprefixとの集合一致はpilot artifactに保存されていない。
この訂正は結果を見た再調整ではなく、S1実行前の実装identity監査である。

## 1. 契約時点の判断

PR-2最小pilotの`GO_PR2`は、H4 linear 1.0 Å、STO-3G、rank-12参照に対する
one uncontrolled outer stepの構造screeningである。rank 3/6のhybrid/reference RZ-work比が
0.307753/0.548093となったことから、trade-offがこのscreening単位で消えていないことだけを示す。
controlled wrapper、outer repetition、shot、状態準備、RPE、H12、最終総costは含まない。

同pilotでは、rank-$r$ Hamiltonianがrank-12 DF Hamiltonianの先頭$r$ blockと一致することを
correctness gateとして確認し、残りのblockをexact residualとして取り出した。従って、この条件では

$$
H_D^{(r)}=H_{\mathrm{DF}}^{(12)}[0:r],
\qquad
H_R^{(r)}=H_{\mathrm{DF}}^{(12)}[r:12]
$$

であり、**「prefix圧縮＋random residual」は生成順prefixに対する部分ランダム化と代数的に同じである。**
ただし現行ライブラリの通常PRはfragmentを実装weightで再順位付けしてからprefixを選ぶ。両prefixの
rank 3/6/9での集合・順序一致はS0 identity gateで確認する。一致すれば同一候補として統合し、不一致なら
pilot由来のgeneration-prefix候補と通常weight-ranked prefixを別baselineとして残す。どちらの場合も、
prefix順序の違いだけを新しいmethod deltaとは数えない。

SPRINT/GRADEはfactorization remainderを明示し、qDRIFT/RTE等で処理する構図と、誤差対費用で
remainder strategyを選ぶ手順を既に示す。RC-DFも圧縮DFとnorm低減を既に扱う。従って、現行PR-2を
「圧縮残差randomizationの新発明」または「通常DF-prefix PRに勝つ新手法」とは位置付けない。

## 2. 固定する中心主張

中心主張は次の一文に限定する。

> 固定したDF参照Hamiltonianとcoherent-signal taskにおいて、中間rankで決定論実装を止め、
> exact residualをrandom補完する選択が、residual 1-norm、sample回路、controlled wrapper、
> finite-sampling負担および近似biasを同時に戻した後にも再現可能な資源crossoverを持つ条件と、
> そのcrossoverが消える条件を明らかにする。

これは**限定的な実証・資源研究**である。正の優位性は完成条件ではない。full-scopeでcrossoverが
消えること、または独立条件へ移送しないことも、原因を比較可能な形で切り分ければ完成した
negative resultとする。

## 3. 研究対象とestimand

### 3.1 参照Hamiltonian

- 系: linear H4、STO-3G、8 qubits、4 electrons
- 参照: 各geometryで同じ生成規約を用いるDF rank 12 Hamiltonian
- 元の分子Hamiltonianではなく、rank-12 DF Hamiltonianを本契約の比較対象とする
- one-body、定数項、核反発、orbital/basis順序を全候補で一致させる

rank-12化そのものの分子Hamiltonianに対する近似誤差は別問題であり、PR-2の利益に含めない。

### 3.2 共通task

主taskは、同一の固定入力状態に対する複素coherent signal

$$
z(T)=\langle\psi_{\mathrm{ref}}|e^{-iH_{\mathrm{DF}}^{(12)}T}|\psi_{\mathrm{ref}}\rangle
$$

のcosine/sine Hadamard interrogationとする。入力状態はrank-12参照のsector ground stateを固定して
全候補で共有する。状態準備回路は比較scopeから除外するが、候補ごとのshot数が異なる場合に状態準備
costが自動的に相殺されるとは主張しない。

ground-state energy差はdiscard biasの診断に使うが、energy biasだけでcoherent-signal精度または
random residualの有効性を判定しない。

## 4. development anchorと独立条件

### 4.1 development anchor

- geometry: H4 linear 1.0 Å
- 参照rank: 12
- 主anchor: rank 6
- stress control: rank 3
- near-deterministic/discard-safe control: rank 9
- 基本短時間: $\delta=0.1$ a.u.

rank 6を選ぶ理由は、pilotで48.122%の決定論RZ削減と0.548093のone-step hybrid/reference比を
残しつつ、rank 3よりdiscard biasとresidual burdenが小さい中間点だったためである。rank 3を主anchorへ
昇格せず、極端な圧縮に対するstress controlとする。rank 9はpilotでdiscard biasが既に小さく、
random補完の必要性が消える側のcontrolとする。

### 4.2 独立条件

- geometry: H4 linear 1.30 Å
- basis、sector、参照rank、rank 6 anchor、rank 3/9 control、task、費用指標をdevelopmentから固定移送する
- 独立条件の結果を見る前に、入力Hamiltonian hash、fragment順序、状態規約、compiler条件、seed、
  finite-RTE候補集合および判定閾値を事前登録する
- 独立条件だけに合わせたrank、sampling分布、誤差予算またはcompiler policyの再調整をしない

1.30 Åは現行PR-2 development pilotに未使用のgeometryとして選ぶ。これは別分子・basis・system sizeへの
一般化ではなく、同じH4 family内の最小transfer testである。

## 5. 比較baseline

同じestimand、物理時間、入力状態、制御化、精度判定および費用scopeで次を比較する。

1. **B0: discard residual** — rank-$r$の決定論Hamiltonianだけを使い、元のrank-12 targetに対するbiasを含める。
2. **B1: deterministic reference** — rank-12全体を決定論的に実装する。
3. **B2-G: generation-prefix PR** — pilotと同じ生成順rank-$r$ prefixを決定論化し、残りをrandom補完する。
4. **B2-W: weight-ranked prefix PR** — 現行`prepare_df_partial_s2`の通常prefixをmatched-task baselineとする。
5. **B3: full-random endpoint** — $L_D=0$を同じfinite-RTE候補集合で評価する。
6. **B4: non-prefix compressed residual** — method-delta gate通過時だけ追加する条件付きbaseline。

S0でB2-GとB2-Wのfragment集合・順序が一致すれば一つのB2として統合し、二重計上しない。不一致なら
B2-Wを省略せず、B2-Gが通常prefix PRに勝ったとはB2-Wとのmatched比較なしに書かない。

## 6. 費用と精度のscope progression

追加計算へ進む場合は、別の結果前事前登録で以下を固定し、段階を飛ばさない。

### S1: controlled one-step semantics

- $\delta=0.1$のcontrolled cosine/sine wrapper
- exact rank-12 signal、B0 signal、B2平均signalの差
- exact residual再構成、sampling確率、normalization、identity成分
- expected compiled RZをprimary、CX・size・depthをsecondary metricとする
- 状態準備なしのwrapper全体を数え、中央time-evolution blockだけの値を主結果にしない

S1はcorrectnessと費用scopeの接続であり、資源優位性の最終判定ではない。

### S2: fixed-time development comparison

- 共通物理時間 $T=0.8$ a.u.、基本外側step $\delta=0.1$、$q=8$
- finite residual sampling、outer repetition、cosine/sine wrapper、normalization、必要shot数を含める
- 状態準備を除く1 interrogationの期待compiled costと必要shot数の積を比較する
- candidateごとに異なる誤差・shot条件を使わず、結果前に固定した同一complex-signal精度で比較する
- direct compilationが困難な部分は、適用domainとholdout誤差を固定した既存proxyだけを使う

### S3: independent transfer

S2で固定したrank policy、finite-RTE候補集合、誤差予算、cost metric、判定規則をH4 1.30 Åへ移し、
結果を見て再最適化しない。S3完了まで別geometry、別basis、H12、長RPEへ広げない。

## 7. 判定規則

結果を見る前の事前登録では、primary costの実質差を

$$
\eta_{\mathrm{cost}}=10\%
$$

と固定する。数値的不確かさまたはproxy区間が10%差を跨ぐ場合は優劣未確定とする。
10%は小さなlocal compiler差を資源crossoverと呼ばないためのmateriality thresholdであり、
統計的信頼水準または厳密誤差上界ではない。

### `COMPLETE_POSITIVE_RESOURCE_CROSSOVER`

- S1のcorrectnessを全て通過する
- B2 rank 6がS2とS3の双方で、共通signal精度を満たす非劣frontier上に残る
- B1に対するprimary cost削減が双方で10%以上である
- B0が同じ精度を満たさない、またはB2より10%以上高costである
- improvementの主要因をdeterministic saving、residual norm、sample cost、shot inflationへ分解できる

### `COMPLETE_CONDITIONAL_RESOURCE_MAP`

correctnessは通過するが、正のcrossoverがdevelopmentだけ、独立条件だけ、または一部rankだけに残る。
適用条件と破綻要因を固定して完了し、一般的優位性へ拡張しない。

### `COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN`

one-step screeningの利得が、controlled wrapper、repetition、finite samplingまたはshot負担を戻すと
10%未満になる、区間が重なる、またはB1/B0に支配される。追加geometryや精度sweepで救済せず、
どの費用項が利得を消したかを記録して完了する。

### `STOP_CORRECTNESS_OR_ESTIMAND_FAILURE`

exact residual再構成、control semantics、sampling normalization、共通estimandまたは費用scopeの一致に
失敗した場合は停止する。閾値、rank、stateまたはtaskを変更して同じS1をやり直さない。

## 8. 非prefix圧縮のmethod-delta gate

RC-DF等の再最適化圧縮を使う拡張は、core resource studyとは別branchとする。次を文書だけで全て固定
できるまで数値実行を認めない。

1. $\widetilde H_\theta$がrank-12 DF prefixと数値的にも構成的にも異なる。
2. $\Delta H_\theta=H_{\mathrm{DF}}^{(12)}-\widetilde H_\theta$を、定数・one-body補正を含めexactに再構成できる。
3. residualのsample単位、1-norm、normalization、controlled回路およびsample costを定義できる。
4. B0--B2、とくに最良prefix PRをmatched-task baselineとして残す。
5. SPRINT/GRADE、RC-DFおよび採用optimizerに対する追加の一次文献監査でmethod deltaを一文にできる。

このgateを通っても、新規性は圧縮法またはresidual randomization単独ではなく、特定taskに対する
非prefix表現とrandom補完の共同設計に限定する。

## 9. 完成に必要な成果物

core resource studyは、次を揃えた時点で正・条件付き・負のいずれかとして完成する。

- S1--S3の結果前事前登録とそのhash
- development/independentの入力hash、環境、seed、compiler条件
- B0--B2の同一task recordとcost-scope宣言
- exact residual、signal、normalization、compiled wrapper、shot accountingの自動test
- machine-readable result artifactとfingerprint
- crossoverまたは失敗原因のcomponent breakdown
- 既知事項、新しい観測、未検証範囲を分けた検証文書
- `VALIDATION_STATUS.md`、manifest、研究概要、研究ノート、索引の同期

S2またはS3の結果を得た時点で必ず停止し、固定判定を行う。正の結果でもH12、別分子、長RPE、
最終総costへ自動的に進まない。

## 10. 主張しないこと

- PR-2がfactorization residual randomizationを初めて提案した。
- 現行generation-prefix pilotが、identity gate前からweight-ranked通常DF-prefix PRと同一または異なると
  確定している。
- prefix順序だけが違うことを新しいアルゴリズム上のmethod deltaとする。
- one-step RZ-work比がcontrolled full-task、shot込み、RPE総costの優位性を示す。
- rank 6が別geometry、別分子、別basisまたはH12でも最適である。
- 小さいenergy/Frobenius residualから小さいsampling 1-normまたはsample costが従う。
- 状態準備costが候補間で相殺される。
- 1条件のH4結果から一般的な圧縮許容値選択則が得られる。

## 11. 直近の停止点

本契約により、pilot後に要求されていた中心主張、development anchor、独立条件、scope progression、
GO/STOPおよびterminal completion ruleは固定した。実装監査でgeneration-prefixとweight-ranked prefixの
同一性をS0 gateへ移した。次に許される作業はS1--S3の**結果前事前登録**であり、本契約だけを根拠に
計算を開始しない。

PR-3再調整、PR-4--6、追加rank/geometry、H12、長RPE、noise/backend、最終compiled total costは停止を
維持する。非prefix branchも第8節のmethod-delta gateを通過するまで開始しない。

## 12. 一次資料

1. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*,
   [arXiv:2606.30741v1](https://arxiv.org/abs/2606.30741)。特にEqs. (2)--(5)、Fig. 1、
   Sec. IV Step 7。
2. Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed
   Double Factorization*, Quantum **8**, 1371 (2024),
   [DOI:10.22331/q-2024-06-13-1371](https://doi.org/10.22331/q-2024-06-13-1371)。
3. Jin and Li, *A Partially Random Trotter Algorithm for Quantum Hamiltonian Simulations*,
   Communications on Applied Mathematics and Computation **7**, 442--469 (2025),
   [arXiv:2109.07987](https://arxiv.org/abs/2109.07987)。
