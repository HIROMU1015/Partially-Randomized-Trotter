# PR-2 S0/S1前 独立批判レビュー（fixed commit d3e1723）

- 対象repository: `HIROMU1015/Partially-Randomized-Trotter`
- branch: `all-r-coherent-opt2-reoptimization`
- fixed commit: `d3e17239702b56e765ff0a2f8993135015332ea8`
- review date: 2026-09-27
- 数値実行: なし
- repository変更: なし
- 判定対象: PR-2 S0/S1実装開始前の事前登録・研究設計
- 以前の `71169d8` は今回の固定対象ではない。ただしdry-run manifestがそのbase commit上のdirty worktreeで作成された履歴自体はprovenanceとして保持する。

## 判定

**`AMEND_BEFORE_S0`**

研究方針そのものを停止・全面再設計する理由は見つからない。PR-2は「新しい圧縮法」ではなく、固定DF targetに対する**部分ランダム化のresource crossover / breakdown conditionを、coherent signal・finite-RTE・controlled wrapper・shot込みで評価する限定resource study**として成立し得る。

ただし、現在の事前登録のままS0/S1へ進めることは推奨しない。以下の6点を結果前amendmentとして固定してからS0実装へ進むべきである。

1. finite-RTE normalizationを既知のsampling overheadとして扱うprimary estimatorへ修正する。
2. B3の「full-random」表記と実装semanticsを一致させる。
3. no-state-prep primary costに、共通state-preparation costのbreak-even sensitivityを事前登録する。
4. Monte Carlo compiled-cost uncertaintyと10% materiality ruleを接続する。
5. decision ruleのB2-G/B2-W矛盾と「技術bug vs 科学的negative」を分離する。
6. S1をcorrectness stageとして軽量化し、S1後に必ず人手/GPTで研究判断を行い、S2を自動許可しない。

---

# 1. 判定理由の要約

## 1.1 研究としての中心主張

現行contractの中心主張は、次の範囲なら妥当である。

> 固定したrank-12 DF Hamiltonianとstate-conditioned coherent-signal taskにおいて、中間rankでdeterministic実装を止め、exact residualをrandom補完するpartial constructionが、discard、full deterministic、通常weight-ranked partial、random-dominant endpointと比較して、どの条件でresource frontierに残るか、またcontrolled wrapper・finite-RTE・sampling・shotを戻すとどの費用項で利得が消えるかを明らかにする。

この主張は「圧縮残差randomizationの発明」ではない。

SPRINT/GRADEはfactorization residualを明示してqDRIFT/RTE等で処理する設計を既に提示している。RC-DFはcompressed DF自体を既に扱う。Jin–Li、Güntherらによりdeterministic/random partitionも既知である。従って、新規性は**matched coherent-signal taskにおける具体的なresource crossover / failure mapと、その原因分解**に限定すべきである。

この限定なら、positive resultだけでなくnegative resultも成果になり得る。

## 1.2 H4だけで論文になるか

S1/S2/S3だけで自動的に論文価値が確定するとは言えない。一方、今の段階でH12、別分子、precision sweepを追加すべきでもない。

まずS1まで実行し、

- resource questionが実装上きれいに定義できるか、
- corrected finite-RTE estimatorで候補差が残る見込みがあるか、
- ordinary PR baselineとの差が何を意味するか、

を確認する。

S1後に研究方針を再評価する。その時点でS2へ進む価値が弱ければ止める。

---

# 2. 致命的問題

現時点で `STOP_OR_REDESIGN_ESTIMAND` を要求する致命傷はない。

ただし、**finite-RTE normalizationの扱いはS0前に直さないとresource comparisonの意味が変わる**ため、最優先のamendmentである。

---

# 3. S1前に必須の修正

## A01. finite-RTE normalization: raw meanをprimary estimandにしない

### 現状

事前登録 §5, §7 は、finite distribution normalizationを事後補正せず、

\[
\mu_{\rm raw}
\]

をcandidate meanとして使い、そのattenuationをsystematic biasへ含める。

shot outcomeは \(\pm1\) とし、

\[
N_{j,a}
=
\left\lceil
\frac{2}{(\epsilon^{\rm stat}_{j,a})^2}
\log\frac{2}{\alpha_{\rm axis}}
\right\rceil
\]

を使う。

### 問題

RTEではnormalization factorは既知であり、標準的なsignal estimationではattenuated raw meanを既知係数でrescaleし、その代わりsampling overheadを負担する。

Günther et al., *Phase estimation with partially randomized time evolution*, PRX Quantum 7, 020332 (2026) では、RTE Hadamard signalにnormalization/damping factorが現れ、そのsampling overheadが資源へ入る。RTEの利点は、normalizationを考慮するとexact evolution signalのunbiased estimatorを構成できる点として扱われている（finite cutoffではtruncation biasは残る）。

したがって、known attenuationを「補正可能なのに補正しないsystematic bias」として候補を落とすと、randomized baselineに不利なestimandになる。

### 修正

primary estimatorをnormalization-corrected finite-RTE estimatorとする。

candidate \(j\) の全tail occurrenceを通した既知normalization multiplierを

\[
\mathcal B_j = 1/\mathrm{attenuation}_j \ge 1
\]

とする。

Hadamard raw outcome \(X_{j,a}\in\{-1,+1\}\) に対し、

\[
Y_{j,a}=\mathcal B_j X_{j,a}
\]

をprimary estimatorとする。

そのmeanを

\[
\nu_{j,a}=\mathcal B_j \mu^{\rm raw}_{j,a}
\]

とし、systematic biasは

\[
b_{j,a}=|\nu_{j,a}-z_{12,a}|.
\]

有限Taylor cutoffによるtruncation、outer PF、partition等のbiasはこの \(b\) に残るが、既知normalization attenuation自体はbiasではなくsampling rangeへ移る。

\[
\epsilon^{\rm stat}_{j,a}=\epsilon_{\rm axis}-b_{j,a}.
\]

\(\epsilon^{\rm stat}_{j,a}\le0\) ならineligible。

Hoeffdingより

\[
N_{j,a}
=
\left\lceil
\frac{2\mathcal B_j^2}
{(\epsilon^{\rm stat}_{j,a})^2}
\log\frac{2}{\alpha_{\rm axis}}
\right\rceil.
\]

deterministic候補では \(\mathcal B_j=1\)。

raw event mean、attenuation、raw-biasもdiagnosticとして保存するが、resource勝敗のprimaryには使わない。

### 留保

もし研究対象を意図的に「補正しない物理random channelのmean」と定義したいなら、その別研究として明記できる。しかし、それを通常RTE/PR resource baselineのprimaryにしてはいけない。

---

## A02. B3を「full-random」と呼ばない

現行 `prepare_df_partial_s2` は `L_D=0` でもone-body correctionをdeterministic blockとして構成する。

従って現状のB3は、

> **zero deterministic two-body DF fragments; one-body correction remains deterministic**

であり、Hamiltonian全体のfull randomizationではない。

### 修正案

B3の名称を例えば

`B3: LD=0 two-body-random endpoint (deterministic one-body retained)`

へ変更する。

本当にone-bodyもrandomizeするbaselineを新設する必要は、今回のS0/S1にはない。新実装を増やすより、scopeを正確に書く方を推奨する。

---

## A03. state preparationを除くprimary costにbreak-even sensitivityを加える

exact rank-12 ground stateを全候補で共通入力にすること自体は、**Hamiltonian-simulation subroutineのstate-conditioned benchmark**として妥当である。

ただし、候補ごとにshot数 \(N_j\) が違うため、common state-preparation circuitの費用は総costでは相殺されない。

primaryをno-prepに固定してよいが、S2/S3のresource conclusionには、結果前に次をsecondaryとして固定する。

\[
G_j(P)
=
G_j^{\rm no-prep}
+
P N_j^{\rm total},
\qquad P\ge0.
\]

hardware依存のPを勝手に設定しない。

比較候補A/Bについてrankingが変わるbreak-even

\[
P^*
=
\frac{G_B^{\rm no-prep}-G_A^{\rm no-prep}}
{N_A-N_B}
\]

（分母非零の場合）を報告する。

この追加はcandidate selectionのprimaryを変えず、state-prep exclusionの限界を透明化する。

---

## A04. compiled-cost不確かさを10% materiality ruleへ接続する

現状は32 trajectory、RSE>2%なら128へ一度だけ拡張するが、decision ruleに使う「区間」の定義がない。

### 固定すべきengineering interval

trajectory compiled cost \(C_1,\ldots,C_n\) に対し、

\[
I_C=
[\max(0,\bar C-2\,SE),\ \bar C+2\,SE]
\]

を**materiality uncertainty interval**として使う。

これは厳密な95% confidence guaranteeとは呼ばない。adaptive 32→128 samplingを行うため、formal coverage claimをしない。

analytic shot count \(N\) を掛け、

\[
I_G=N I_C.
\]

A/B cost ratioについて保守的に

\[
I_{A/B}
=
\left[
\frac{G_A^{L}}{G_B^{U}},
\frac{G_A^{U}}{G_B^{L}}
\right].
\]

AがBより10%以上安いと判定するには、

\[
\sup I_{A/B}<0.9
\]

を要求する。

0.9を跨ぐ場合はmateriality unresolved。

### candidate selection

method内でpoint estimate最小candidateを選ぶが、128 trajectory後も別candidateのintervalがmaterialなranking reversalを許す場合は `SETTING_UNCERTAIN` とする。

`SETTING_UNCERTAIN` のままS3へ進まない。

---

## A05. correctness failureとscientific STOPを分ける

現行 `STOP_CORRECTNESS_OR_ESTIMAND_FAILURE` は二種類を混ぜている。

### 修正

- `BLOCKED_IMPLEMENTATION_INVALID`
  - code bug
  - adapter不備
  - wrapper mapping bug
  - normalization/probability test failure
  - snapshot serialization bug
  - compiler runner bug

  scientific negativeとは数えない。同一のfrozen scientific designを変えずにbugを修正し、commit/hash/testを記録してS0からやり直してよい。

- `STOP_ESTIMAND_OR_SCOPE_INVALID`
  - common estimand自体が比較不能
  - target/state/taskを変えないと成立しない
  - candidate間で物理時間やcontrol semanticsを揃えられない

  これは研究設計の停止。

- scientific negative
  - correctness通過後、固定条件でresource crossoverが消える場合。

threshold、rank、precision、state等を結果後に変更してscientific negativeを救済しない規則は維持する。

---

## A06. B2-G/B2-Wとterminal decisionを整合させる

現行 §10 は、

> B2-GがB2-Wに10%以上支配される → `COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN`

と読める一方、

> B2-Wを上回らない結果を「新手法の失敗」と呼ばない

とも書いてあり矛盾する。

PR-2はprefix-orderの新手法ではないため、B2-Wが最良partialなら「partial-random resource crossover自体が消えた」とは言えない。

### 修正

rank 6のintermediate partial familyを

\[
\mathcal B_2=\{B2\text{-}G,\ B2\text{-}W\}
\]

（S0でidentityなら一候補へ統合）とする。

- B2-GがB2-Wに負けるがB2-Wがresource frontierに残る:
  `COMPLETE_CONDITIONAL_RESOURCE_MAP`
  またはS2からS3へ進む候補として扱える。
  claimは「generation-prefixの追加利得はない／ordinary PRが同等以上」とする。

- `COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN`:
  rank 6のintermediate partial family全体が、B0/B1/B3 endpointによりmaterialに支配される、またはaccuracyを満たせない場合。

- `COMPLETE_POSITIVE_RESOURCE_CROSSOVER`:
  rank 6のintermediate partial familyの少なくとも一つがaccuracyを満たしfrontierに残り、B1に対するcost ratio intervalが0.9未満、B0/B3にmaterialに支配されず、S3でも同じfrozen policyで再現する場合。

B2-Gだけが勝った場合のみ「generation ordering固有のresource effect」を記述できるが、それ自体を新アルゴリズムとは呼ばない。

---

# 4. 有用だが任意の改善

## O01. S0 prefix identityでorder-only differenceをHamiltonian levelでも記録

集合が同じで順序だけ違う場合、

\[
H_D^G=H_D^W,\qquad H_R^G=H_R^W
\]

をdense small-system normで確認・保存する。

この場合、

- split Hamiltonian自体: 同じ
- deterministic S2 approximation: orderに依存
- compiled circuit cost: orderに依存し得る
- random-tail数学的分布: 同じHamiltonianなら同じ。ただしenumeration/hash/seedは実装上異なり得る

と区別する。

## O02. held-out geometryという語を使う

H4 1.30 Åは同じfamily・同じbasis・同じsystem sizeであり、完全な外部独立検証ではない。

`independent transfer` より、

`pre-frozen held-out geometry transfer`

と書く方が正確。

## O03. exact-state estimandのscopeを明記

\(\psi_{12}\) はH12のexact sector ground stateなので、

\[
z_{12}(T)=e^{-iE_0T}
\]

でunit modulus。

これは循環的なアルゴリズムではなく、simulation subroutineを分離評価するoracle benchmarkである。

したがって、

- end-to-end ground-state preparation
- unknown-energy discovery
- realistic imperfect guiding state

まで検証したとは書かない。

---

# 5. 問題なしと判断した項目

## 5.1 complex error budget

\[
\epsilon_{\mathbb C}=0.05,\qquad
\epsilon_{\rm axis}=0.05/\sqrt2
\]

として各axisの総absolute errorを \(\epsilon_{\rm axis}\) 以下にすれば、

\[
\sqrt{\epsilon_{\rm Re}^2+\epsilon_{\rm Im}^2}\le0.05.
\]

\(\alpha_{\rm axis}=0.025\) を二軸へ使うunion boundも正しい。

修正A01後はoutcome rangeに \(\mathcal B_j\) を入れる必要がある。

## 5.2 candidate-dependent leftover statistical budget

true/benchmark systematic biasを全候補で同じ基準から評価し、

\[
\epsilon^{\rm stat}=\epsilon_{\rm axis}-b
\]

を使うことは、**oracle-informed resource accounting**として数学的に整合する。

ただしonline algorithmがbiasを事前に知っているという意味ではない。論文ではretrospective benchmark/resource lower-envelopeであることを明記する。

## 5.3 fresh IID trajectory

各Hadamard shotでfresh IID trajectoryを引くprimary contractは、current resource estimatorとして明確。

fixed-circuit reuseを別結果として混ぜない判断も妥当。

## 5.4 B0/B1/B2/B3の基本構造

同じsecond-order outer PF、同じphysical time、同じstate、同じwrapper scopeに限定するなら、B1 rank-12 deterministic S2はmatched deterministic endpointとして妥当。

「全deterministic PFの中で最良」とは主張しない。

B3にも同じfinite-RTE gridを与える方針は公平。

## 5.5 rank 3/9 control

rank 3/9をrank6 settingのmechanism controlとして固定移送するのは妥当。

rank3/9を公平最適化されたresource competitorとして扱わないこと。

---

# 6. 中心主張を書き直した案

> 固定したrank-12 DF H4 Hamiltonianのcoherent-signal benchmarkにおいて、DF二体fragmentの中間prefixをdeterministic backboneとして残し、exact residualをfinite-RTEで補完するpartial constructionについて、既知normalizationのsampling overhead、finite-cutoff bias、controlled Hadamard wrapper、compiled one-shot cost、必要shot数を同一scopeで戻す。discard、rank-12 deterministic、weight-ranked ordinary partial、\(L_D=0\) two-body-random endpointと比較し、中間partial splitがresource frontierに残る条件と、one-step screeningの利得を消す費用項を特定する。

この主張はH4/固定精度/固定outer-S2 scopeのresource studyであり、圧縮法やresidual randomizationの新規発明を主張しない。

---

# 7. decision ruleを書き直した案

## S0/S1

- implementation/test failure:
  `BLOCKED_IMPLEMENTATION_INVALID`
  - scientific resultではない
  - frozen scientific designを変更せず修正可

- estimand/scopeが共通化不能:
  `STOP_ESTIMAND_OR_SCOPE_INVALID`

- S1 correctness pass:
  **必ず停止して外部/GPT研究レビュー**
  - S2自動実行禁止

## S2

まずrank6 intermediate partial family \(\mathcal B_2\) を評価。

### S2-N: development negative

以下のいずれかなら

`COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN`

としてS3へ進まない。

- \(\mathcal B_2\) の全candidateがaccuracy不適格
- B0/B1/B3のいずれかが、accuracyを満たした上で\(\mathcal B_2\)をmaterialに支配
- 128 samples後もresource rankingに必要なsettingが unresolved で、development crossoverを主張できない

### S2-C: conditional / ordinary-PR subsumption

B2-GがB2-Wにmaterialに負けるが、B2-Wがintermediate partial crossoverを示す場合、

`COMPLETE_CONDITIONAL_RESOURCE_MAP`

とする。

ただしS3へ進むかは、S2後の外部/GPTレビューで明示承認する。
「generation-prefix新手法の成功」とは呼ばない。

### S2-P: transfer candidate

S3候補にできる条件:

- rank6 \(\mathcal B_2\) の少なくとも一候補がaccuracy適格
- B1に対しcost ratio uncertainty intervalの上端が0.9未満
- B0/B3にmaterialに支配されない
- method内setting selectionが `SETTING_UNCERTAIN` でない
- component breakdownが整合

この場合もS3は自動実行せず、S2後レビューで許可する。

## S3

S2でfreezeしたrank policy / \(r,K\) / accuracy / compiler / estimatorを変更しない。

- held-out geometryでもS2-P条件を満たす:
  `COMPLETE_POSITIVE_RESOURCE_CROSSOVER`

- developmentでは成立したがtransferで消える:
  `COMPLETE_CONDITIONAL_RESOURCE_MAP`

S3後は追加geometry/H12/RPEへ自動拡張しない。

---

# 8. `pr2_s1_s3_preregistration.md`への具体的修正案

| 節 | 現在の記述 | 修正文の要点 | 理由 |
|---|---|---|---|
| §1 | S1 correctness passならS2 | 「S1後に必ず停止し外部/GPT研究レビュー。S2は明示承認が必要」 | 今回の研究運用方針に合わせる |
| §2 | shot accounting新実装 | normalization-corrected estimatorを実装対象へ | RTE standard estimatorとの公平性 |
| §3.2 | prefix identity | order-only時にdense \(H_D/H_R\) equalityも保存 | Hamiltonian差とPF order差を分離 |
| §4 B3 | full-random endpoint | `LD=0 two-body-random endpoint; deterministic one-body retained` | 実装semanticsと名称を一致 |
| §5 | normalizationを補正しないactual mean | raw meanはdiagnostic。primaryはknown normalization corrected mean | attenuationをsampling overheadへ戻す |
| §6 | gridそのまま | gridは維持 | 結果後調整を防ぐ。変更不要 |
| §7 | ±1 shot formula | corrected estimatorでは \(\mathcal B^2\) をHoeffding式へ入れる | 数学的整合 |
| §8 | RSE 2%のみ | materiality interval、ratio interval、SETTING_UNCERTAINを追加 | 10% ruleとMC uncertaintyを接続 |
| §8/§9 | no-prepのみ | \(G(P)=G^{no-prep}+PN_{\rm shots}\) break-even sensitivityをS2/S3 secondaryへ | shot差があるのでprepは相殺しない |
| §9 S1 | 36 random cells×32 trajectory compile | S1はcorrectness stageへ軽量化。full MC expected costはS2で実行 | 過剰検証防止 |
| §10 | B2-G dominated by B2-Wもnegativeに読める | intermediate partial familyで判定し、B2-W subsumptionをconditionalへ | 中心主張と整合 |
| §10 | correctnessとestimand failure同一 | `BLOCKED_IMPLEMENTATION_INVALID` と `STOP_ESTIMAND_OR_SCOPE_INVALID` を分離 | bugをscience negativeにしない |
| provenance | manifest `git_commit=71169d8`, dirty | old provenance保持＋`freeze_commit=d3e1723...`をamendment manifestへ追加 | GitHub固定版の再現性 |

---

# 9. S0/S1/S2/S3の最終進行表

## Phase A — 今回のレビュー後

**数値実行禁止。**

CodexがA01–A06を結果前amendmentへ反映。
old file/hashを残し、amendment version/hashと変更理由を保存。

dry-run manifestはv1を上書きせず、v2またはamendment manifestを作成する。

`generation_base_commit=71169d8...`
と
`freeze_commit=d3e1723...`（またはamendment後の新commit）
を区別する。

## Phase B — S0 implementation

Codexが以下を実装・test。

- development snapshot reproduce/freeze
- held-out 1.30 Å snapshot freeze
- prefix identity gate
- explicit generation-prefix adapter if needed
- corrected finite-RTE complex-signal estimator
- corrected bias-aware shot accounting
- q1/q8 full-wrapper connection
- materiality interval
- artifact fingerprint/tamper tests
- automatic S2/S3 execution guard

S0結果はartifact化。

## Phase C — S1 correctness only

S1の目的は資源勝敗ではなくsemantic/circuit接続。

推奨軽量化:

- fixed gridのexact/raw/corrected meanは必要な範囲で評価
- full-wrapper compile pathは各random cellで**canonical 1 trajectory**を通してbuild/compile可能性を確認
- 32/128 trajectoryによるexpected-cost estimationはS1では行わない
- rank3/9はstructural correctness + preregistered sentinel cellで十分
- S1からpositive resource claimを出さない

S1終了後、必ず停止。

## Phase D — GPTへ戻す

S1 artifact、amended preregistration、S0/S1 test、identity gate結果をこのGPTでレビューする。

ここで初めて、

- `PROCEED_S2`
- `REVISE_SCOPE_BEFORE_S2`
- `STOP_PR2`

の一つを選ぶ。

## Phase E — S2/S3

S2以降は今回自動承認しない。

---

# 10. GPT自身の不確実性と追加確認

## 確認済み

fixed commit `d3e17239702b56e765ff0a2f8993135015332ea8` は `71169d8` より1 commit先で、指定6ファイルが追加されている。

6ファイルの原本をfixed commitから取得して確認した。

dry-run manifestはnumerical execution 0を宣言し、implementation blockerを明示している。

manifestのparent contract SHA-256は

`4d325c0cc28dae08e975ab6311f77258f0cfdefa224e5dfcfb93c1b0a6e8a2b0`

で64桁になっており、以前の暫定レビュー時の不足文字問題は解消している。

## 未確認 / S0で確認すべき

- 1.00 Å pilot Hamiltonianのbitwise hash再現
- 1.30 Å snapshot
- B2-G/B2-W identity
- new explicit partition adapter
- new shot/cost runner
- actual S0/S1 tests
- amended documentのSHA-256

これらは現在「実装済み」と推測しない。

---

# 一次資料の位置付け

- Jakob Günther et al., *Phase estimation with partially randomized time evolution*, PRX Quantum 7, 020332 (2026), DOI 10.1103/ynxb-p2xq.
  - partial randomizationとsingle-ancilla resource accounting、RTE normalization/sampling overheadは既知。
- Pablo A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741 (2026).
  - factorization remainderをqDRIFT/RTE等で扱い、remainder strategyをerror/costで選ぶ構図は既知。
- Oumarou Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum 8, 1371 (2024).
  - compressed DFとresource reductionは既知。
- Shi Jin, Xiantao Li, *A Partially Random Trotter Algorithm for Quantum Hamiltonian Simulations*, Commun. Appl. Math. Comput. 7, 442–469 (2025).
  - deterministic/random partitionとbias/variance tradeoffは既知。

従って、PR-2の価値は新発明ではなく、**指定されたDF/coherent-signal/finite-RTE実装におけるmatched resource crossoverとfailure mechanism**に置く。
