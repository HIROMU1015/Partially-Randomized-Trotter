# PR-2 matched-accuracy resource-map契約 v1

作成日：2026-09-29
基準commit：`a9171d8bea93441afc9b17c7ffeab79af8dbcc95`
現行status：`PR2_MATCHED_ACCURACY_CONTRACT_FIXED_M1_NOT_AUTHORIZED`

## 0. 位置付けと停止状態

本書は、PR-2を新しいprefix法の主張から、DF-prefix部分ランダム化をいつ使うかという
matched-accuracy resource/applicability studyへ狭めるM1前契約である。

旧S2の正式status `S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`、rank 6 primary、rank 3/9 control、
B2/B3 material frontier、10% materiality、artifactおよび全数値を変更しない。rank 3を旧S2の
winnerへ遡及的に昇格しない。旧S0の`STOP_INPUT_REPRODUCTION_MISMATCH`も変更しない。

現時点ではM1のsource実装、signal、sampling、compile、held-out、S3を承認しない。本契約と
[M1前先行研究gate](pr2_matched_accuracy_prior_art_gate_v1.md)だけを固定し、次はM1実装契約の
レビューで停止する。

設計入力として、次のGPT提案を参照した。これらは正本ではなく、本書と先行研究gateを優先する。

- `pr2_research_redesign_a9171d8_20260929.md`、SHA-256
  `5e13b6cc6401b02ec109243eec36e86febb106ecd5194a61bd72c0432f0ff3c2`
- `pr2_codex_redesign_workplan_a9171d8_20260929.md`、SHA-256
  `8f699b489180d87a039ee60c94892b84311e69d3a97fd235ebf8527a40e9907f`

## 1. 研究質問とclaim境界

主RQは次である。

> 同じ固定DF target、state、物理時間、複素signal精度の下で、残差を捨てる、決定論で保持する、
> またはfinite-RTEで補完する方式のどれを選ぶべきか。splitと`q,r,K`を公平に選ぶと、
> 1-shot workと必要shotの交換関係はどの条件で中間partialをresource frontierへ残すか。

これは新しい部分ランダム化algorithm、prefix生成法、finite-RTE定理、global optimizerの提案ではない。
新規性候補は、既知手法を同じfinite implementation、signal task、controlled wrapper、accuracy、
compiled-cost scopeへ接続したときに、簡略評価からの設計判断がどこで変わるかという限定知見である。

M1の結果にかかわらず、一般分子、一般DF表現、H12、fault-tolerant全stack、量子優位、全PFに対する
優位、候補集合外のglobal optimumを主張しない。

## 2. M1で固定する科学条件

| 項目 | 固定値・規則 |
|---|---|
| system | 保存済みdevelopment H4 linear 1.00 Å、STO-3G、sector 8 qubit |
| target | 保存済みrank-12 DF Hamiltonianと同じstateに対するcoherent signal |
| physical time | `T=0.8` |
| complex accuracy | `epsilon_complex=0.05` |
| axis accuracy | `epsilon_axis=0.05/sqrt(2)` |
| axis failure allocation | `alpha_axis=0.025` |
| outer repetition | `q in {1,2,4,8}` |
| step time | 必ず`delta=T/q` |
| discard | prefix rank `k in {3,6,9}`、残差を補完しない |
| deterministic | rank 12、二次symmetric PF |
| partial | prefix rank `k in {3,6,9}` |
| random-dominant | `k=0`、constantとone-bodyは決定論側に保持 |
| finite-RTE base grid | `r in {1,2,4,8,16,32}`, `K in {2,4}` |
| compiler | S2と同じversion、basis、optimization、seed、coefficient tolerance |
| primary cost | 状態準備を除くfull measured Hadamard wrapperのexpected compiled RZ ×解析shot |
| secondary | CX、depth、size、共通state-preparation cost `P>=0`の感度 |

旧q=8 S2はfixed-setting comparisonとして保持する。M1は同じ`T`とaccuracyでqを選べる
matched-accuracy comparisonであり、旧S2を再採点または置換しない。

## 3. estimand、normalization、shot

候補`j=(mode,k,q,r,K)`のraw meanを`mu_j`、実際にmaterializeした全finite-RTE short stepの
normalization積を`B_total,j`とし、補正後meanを

\[
\nu_j=B_{\mathrm{total},j}\mu_j
\]

とする。軸`a`のbias、統計allowance、十分shotは

\[
b_{j,a}=|\nu_{j,a}-z_{H,a}|,\qquad
s_{j,a}=\epsilon_{a}-b_{j,a},
\]

\[
N_{j,a}=\left\lceil
\frac{2B_{\mathrm{total},j}^{2}}{s_{j,a}^{2}}
\log\frac{2}{\alpha_a}
\right\rceil .
\]

`s<=0`、非有限値、branch不整合、correctness gate不通過はaccuracy-ineligibleとする。
raw/corrected mean、outer-PF bias、finite-cutoff bias、total complex differenceは別fieldに保存する。
絶対値の和を実測total biasと呼ばない。解析shotは実行済み量子shotまたは情報理論的最小shotではない。

状態準備感度は

\[
G_j(P)=A_j+N_jP,\qquad P\ge0
\]

とし、全評価済み候補のlower envelopeと全pairwise crossingを保存する。都合のよい`P`だけを選ばない。

## 4. 可変q correctness gate

M1実装は、科学計算より前に次のtestを全て通過しなければならない。

1. 全候補で数値的かつmetadata上`q*delta=T`である。
2. signal経路とcost経路のwrapperにouter stepが厳密にq回materializeされる。
3. random tail seedは`candidate/axis/trajectory/outer_step/tail_occurrence/rte_step`ごとに独立で、
   同一trajectory内の別occurrenceへ再利用されない。
4. `B_total`は実際にmaterializeした全short stepのnormalization積であり、固定指数や旧
   `DELTA_TIME=0.1`を暗黙再利用しない。直接積とlog-domain合成を照合する。
5. signal評価とcost compileは同じcandidate fingerprintを持つ。
6. fingerprintは少なくともsnapshot/state SHA、mode、k、T、q、delta、r、K、identity policy、
   coefficient threshold、wrapper semantics、compiler identity、seed policyを含む。
7. per-trajectory fingerprintだけが具体seed列を追加し、candidate fingerprintを置換しない。
8. q=1/2/4/8についてraw/corrected relation、controlled Re/Im semantics、operator ordering、
   constant/one-body処理を小行列または既存exact経路で照合する。
9. 同じfingerprintでsignalとcostのいずれかが欠けるrecordをresource rankingへ入れない。

一件でも不通過なら`IMPLEMENTATION_GATE_FAILED`で停止し、threshold、候補、seed規則を緩めない。

## 5. signal評価とdirect compileの分離

### 5.1 先に評価する量

基本候補のcorrected/raw signal、bias、normalization、shotsを先に評価する。accuracy-ineligible候補は
compileしない。既存S2 cellはsnapshot、source、candidate fingerprint、compiler、seed policyが
一致する場合だけ再利用し、旧roleとprovenanceを保持する。

direct compile前に、各accuracy-eligible random候補について次の結果非依存proxyを作る。

- `n_det`：full wrapperにmaterializeする決定論DF fragment occurrence数。
- `n_rand`：paired finite-RTE分布から解析的に得るrandom unitary occurrence期待値。
- `n_fixed`：ancilla、axis、constant、one-bodyを含む固定wrapper operation数。
- `W_action=(N_Re+N_Im)*(n_det+n_rand+n_fixed)`。
- `W_tail=(N_Re+N_Im)*n_rand`。

この二つはcompiled RZではなく、候補選抜専用のaction proxyである。係数重みや定義を候補値を
見た後に変更しない。

### 5.2 random direct-compile上限16と選定規則

新規random direct-compile cellは最大16とする。選定にはsignal/bias/normalization/shotと上記proxyだけを
使い、compiled結果は使わない。candidate tieは`(k,q,r,K)`の辞書順で解く。集合は次の順でunionする。

1. **split anchor**：`k=0,3,6,9`ごとに`W_action`最小を1件。最大4件。
2. **q anchor**：`q=1,2,4,8`ごとに、まだ未選択の`W_action`最小を1件。最大4件。
3. **boundary check**：各splitで、選択またはproxy frontierにある`r=32`候補のうち
   `W_action`最小の一件だけ、同じ`q,K`の`r=64`を一段追加する。最大4件。`r=128`へ進まない。
4. **tail challenger**：各splitで、まだ未選択の`W_tail`最小を1件。最大4件。
5. 16未満なら、`(N_total,n_det,n_rand)`の非支配候補をsplit round-robinで追加し、その後
   `W_action`順で埋める。

同じcandidateが複数規則で選ばれた場合は一件として数える。deterministic/discardのexact compileは
このrandom 16件上限に含めず、accuracy-eligibleな全rank/q cellを同一compilerで評価する。

次のいずれかなら正式statusを`SELECTION_LIMITED`とする。

- 16件に入らない候補が`(N_total,n_det,n_rand)`の非支配frontierに残る。
- 一つのsplitで複数の異なる`q,K`のr32境界が結論を変え得るが、一段確認を全て収容できない。
- `W_action`と`W_tail`が選ぶchallengerを上限内で収容できない。
- compile後のwinnerまたは10% frontierが、未compile候補を安全に除外する証拠を持たない。

`SELECTION_LIMITED`でも評価済み有限集合の表は報告できるが、method winner、global optimum、
held-out candidateを確定しない。上限を結果後に増やして救済しない。追加budgetは別契約とする。

### 5.3 trajectory数

選定random cellは独立32 trajectoryで開始する。次のいずれかだけ、別seedの96 trajectoryを一度追加し、
計128へpoolする。

- RZ meanのrelative `2SE`が2%を超える。
- 10% materiality判定のratio intervalが0.9または1.1を跨ぐ。
- 評価済み候補のpoint winnerとのratio intervalが1.0を跨ぐ。

cosine/sineは同じtrajectory列を共有し、相関を無視したformal CIとは呼ばない。32/128以外へ
adaptiveに増減しない。

## 6. M1の出力と停止判定

M1では少なくとも次を保存する。

- 全analytic候補台帳とcandidate fingerprint。
- accuracy eligibilityと除外理由。
- direct-compile選定rule、各候補の選定tier、未選択候補。
- fixed-q8とmatched-accuracyの両比較。
- 1-shot work、`B_total^2`、bias allowance、shot数の分解。
- no-prep primaryと全`P>=0` lower-envelope sensitivity。
- boundary、selection、compiler、samplingの不確かさ。

M1の正式判定は先行研究gateに定めた4分岐だけとする。どの分岐でも一度停止し、結果後にq/r/K、
threshold、rank、geometry、PF familyを調整しない。

## 7. held-outを開く前の別freeze

H4 1.30 Åは引き続き未開封とする。M1が`CONTINUE_TO_FROZEN_TRANSFER_REVIEW`となり、かつ
`SELECTION_LIMITED`でない場合だけ、別のM2 authorizationを作れる。

M2 authorizationは開封前に次を数値で固定する。

1. transferするmethod/split/q/r/K。developmentからretuneしない。
2. candidate数は最大5：P=0 partial代表、必要ならP lower envelopeの別partial代表、best discard、
   full deterministic、random-dominant。重複する場合は減らし、枠を別candidateで埋めない。
3. 各candidateのaccuracy合格式、10% cost category、予測量、許容差、重大underestimateの定義。
4. primary transfer success、partial success、failure、inconclusiveの分岐。
5. held-outで再計算してよいshotと、固定しなければならないmethod settingを区別する。

多数candidateのpass rateを無理に作らず、固定した少数構成の再現性を見る。held-outで候補を選び直さず、
失敗もそのまま報告する。同じheld-outを設計選択と最終評価の両方に使わない。

### 7.1 M1-B1後に固定したM2契約

M1-B1の`CONTINUE_RESOURCE_STUDY`後、上の別freezeを
[M2 held-out transfer契約 v1](pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md)として具体化した。
transferするのはdevelopment actual ParetoのB2二件とB0/B1/B3代表、計5構成だけである。primaryは
`shots × expected compiled RZ`、point Paretoは同じ6 compiled metrics、materialityはB2対best endpointの
ratio 1.10、重大cost underestimateはdevelopment 1-shot costによる事前予測をheld-out actual RZが10%超
上回る場合とした。terminal statusは`TRANSFER_SUPPORTED / TRANSFER_NOT_SUPPORTED /
TRANSFER_INCONCLUSIVE / IMPLEMENTATION_GATE_FAILED`だけで、全statusが研究方針review前の強制停止である。

この契約とzero-compute planはM2科学実行を認可しない。held-outを開く前に、science sourceを先にcommitし、
独立reviewと別result-prior authorizationを必要とする。

## 8. 禁止事項

- M1 source、runner、testまたはauthorizationの実装・実行を本書から自動開始しない。
- held-out NPZ load、signal、cost、ranking、candidate eligibilityを評価しない。
- 新geometry、LiF、別分子、別basis、別PF、H12、長RPE、最終総costを追加しない。
- rank 3を旧S2 primaryへ昇格しない。B2/B3 frontierまたは旧10%判定を書き換えない。
- SPRINT/GRADE、Composite Simulation、RC-DFまたはPR論文の既知貢献を新規claimにしない。
- `C_use`を厳密上界、engineering intervalをformal confidence intervalと呼ばない。

## 9. 次の一件

次の一件はM1実装契約のレビューである。実装契約はsource/runner/test/schema、M0 read-only台帳、
全候補数、16-cell選抜の機械可読dry-run、zero-compute guardを示し、別commitでfreezeする。
そのレビューと別authorizationを通過するまで、M1の科学計算を開始しない。
