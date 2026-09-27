# PR-2 Codex検証方針 — S0/S1まで

- repository: `HIROMU1015/Partially-Randomized-Trotter`
- branch: `all-r-coherent-opt2-reoptimization`
- review target commit: `d3e17239702b56e765ff0a2f8993135015332ea8`
- 目的: 外部レビューを反映し、S0/S1だけを安全に実装・実行する
- S2/S3: **未承認**
- numerical scope expansion: 禁止

## 0. 最初に行うこと

数値計算・molecular build・signal evaluation・compile・trajectory samplingを開始しない。

まず外部レビュー

`pr2_s0_s1_external_review_d3e1723.md`

を読み、A01–A06へ対応する**結果前amendment**を作る。

旧preregistrationとdry-run manifestを上書きしない。

推奨:

- `docs/research/pr2_s1_s3_preregistration_v2.md`
  または明示的amendment節
- `artifacts/pr2_s1_s3_preregistration/2026-09-27/pr2_s1_s3_dry_run_manifest_v2.json`

旧hash、旧status、変更理由を残す。

---

# 1. 必須amendment

## M1. primary signal estimatorをnormalization-correctedへ

raw finite-RTE meanをprimary resource estimandから外す。

各candidateについて、

- `raw_mean_re/im`
- `attenuation`
- `normalization_multiplier = 1/attenuation`
- `corrected_mean_re/im`
- `raw_bias`
- `corrected_bias`
- `finite_truncation_bias`
- `outer_pf_bias`

を別fieldで保存する。

primary resource decisionは`corrected_mean`で行う。

shot式:

\[
N_{j,a}
=
\left\lceil
\frac{2\mathcal B_j^2}
{(\epsilon_{\rm axis}-b_{j,a})^2}
\log\frac{2}{\alpha_{\rm axis}}
\right\rceil
\]

ただし

\[
b_{j,a}=|\nu_{j,a}-z_{12,a}|,
\qquad
\nu=\mathcal B_j\mu_{\rm raw}.
\]

`epsilon_axis - b <= 0`はineligible。

manualに\(\mathcal B\)を再構成せず、既存RTE normalization/attenuation APIとtestに接続する。

## M2. B3 semanticsを訂正

B3を

`LD=0 two-body-random endpoint; deterministic one-body retained`

と明記。

現行`prepare_df_partial_s2`を使う限り`full-random Hamiltonian`と記載しない。

新しいall-term random adapterは今回作らない。

## M3. state-prep sensitivityを事前登録

S2/S3のsecondary resource recordへ

\[
G(P)=G_{\rm no-prep}+P N_{\rm shots}
\]

を追加。

Pはsymbolic/nonnegative。
hardware値を固定しない。

nearest competitorとのbreak-even \(P^*\) を保存。

candidate selection primaryはno-prepのまま。

## M4. materiality intervalを固定

compiled trajectory costの

\[
I_C=[\max(0,\bar C-2SE),\bar C+2SE]
\]

をengineering uncertainty intervalとする。

正式な95% CIとは呼ばない。

32→128 extension ruleは維持。

128後もmaterial 10% rankingがintervalで反転し得る場合は

`SETTING_UNCERTAIN`

とする。

S3へ進めない。

## M5. decision labelsを修正

以下を分離。

- `BLOCKED_IMPLEMENTATION_INVALID`
- `STOP_ESTIMAND_OR_SCOPE_INVALID`
- `COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN`
- `COMPLETE_CONDITIONAL_RESOURCE_MAP`
- `COMPLETE_POSITIVE_RESOURCE_CROSSOVER`

code bugをscientific negativeへ数えない。

B2-GがB2-Wに負けただけで`NEGATIVE_FULL_SCOPE_ERASES_GAIN`にしない。

## M6. S1後のmanual stopを明文化

S1 correctness passはS2 authorizationではない。

manifest:

`automatic_next_stage = null`

を維持。

S1終了後にsummary artifactを生成して停止。

---

# 2. provenance amendment

dry-run v1が作られたbase:

`71169d817c165b76a9e25fc6f6a16ade28ffe069`

レビュー固定commit:

`d3e17239702b56e765ff0a2f8993135015332ea8`

を区別する。

v2 manifestには最低限:

```json
{
  "generation_base_commit": "71169d817c165b76a9e25fc6f6a16ade28ffe069",
  "external_review_commit": "d3e17239702b56e765ff0a2f8993135015332ea8",
  "amendment_commit": "<new commit>",
  "supersedes_manifest": "pr2_s1_s3_dry_run_manifest_v1.json",
  "results_seen_before_amendment": false
}
```

相当を保存する。

現在parent contractの正しいSHA-256はmanifest記録の

`4d325c0cc28dae08e975ab6311f77258f0cfdefa224e5dfcfb93c1b0a6e8a2b0`

を使用し、再計算して一致確認する。

---

# 3. S0実装

S0でのみ分子生成を許可。

## 3.1 development snapshot

指定recipeを再実行。

pilot expected Hamiltonian hash:

`d8b4aaf21afcc3935d5b5aa4d0805b358c5ec670d8104d25807c7cd0620a3dc3`

と一致確認。

不一致ならS1へ進まない。

hash implementationはarray bytesも含むため、近似一致を同一snapshot扱いしない。

不一致理由を調査する場合も旧pilotと新snapshotを混ぜない。

## 3.2 held-out H4 1.30 Å

S1 resultを見る前に生成・freeze。

保存:

- Hamiltonian snapshot
- Hamiltonian hash
- sector hash
- state hash
- fragment order
- generation recipe
- PySCF/OpenFermion/OpenFermion-PySCF/NumPy/SciPy versions
- relevant source hashes

S3前にsignal/cost/rankingを計算しない。

文書では`held-out geometry transfer`を推奨。

## 3.3 prefix identity gate

rank 3/6/9について保存:

- generation ordered indices
- weight-ranked ordered indices
- unordered sets
- fragment hashes
- fragment weights
- tail hashes

set same/order differentなら追加で:

\[
\|H_D^G-H_D^W\|,\qquad
\|H_R^G-H_R^W\|
\]

を小系denseで確認。

order-only differenceを新method deltaとしない。

## 3.4 adapter

prefixが異なる場合のみgeneration-prefix explicit adapterを実装。

科学条件を変えるためではなく、frozen B2-Gを実装するためのadapter。

test:

- partition covers all fragments exactly once
- deterministic/random disjoint
- stored order respected
- dense \(H_D+H_R=H_{12}\)
- identity extraction consistent
- tail hash deterministic

---

# 4. corrected estimator tests

最低限testする。

1. raw sampled event meanが既存finite-RTE mean operatorと一致。
2. `normalization_multiplier * raw_mean` がfinite truncated polynomial meanと一致。
3. Kを増やす／適切なlimitでtruncationが減る既知small toy。
4. deterministic candidateでmultiplier=1。
5. repeated q=8でtotal multiplierとattenuationが互いに逆数。
6. Re/Im wrapper sign convention。
7. corrected Hoeffding shot formula unit test。
8. `epsilon_stat <= 0` ineligible。
9. no use of legacy RPE unit-radius shot formula。

---

# 5. S1はcorrectness stageへ軽量化

S1でresource winnerを決めない。

## 5.1 実行するもの

- development H4 1.00 Å
- T=0.1, q=1
- B0/B1/B2-G/B2-W/B3 semantics
- rank6 full fixed gridのmean/bias/normalization record
- rank3/9 structural controls
- controlled operator probe
- Re/Im wrapper mapping
- exact residual reconstruction
- probability normalization
- finite-RTE raw/corrected mean

## 5.2 compileの軽量化

v1の

`S1_initial_random_full_wrapper_compiles = 2304`

はcorrectness目的に対して過大。

S1では32/128 trajectory expected-cost MCを行わない。

代わりにrandom grid各cellについて、canonical cell seedの**最初の1 trajectory**だけfull wrapperまでcompileし、

- circuit construction
- axis mapping
- compiler basis
- metric extraction

を確認する。

expected compiled costの32/128 MCはS2まで保留。

rank3/9はstructural correctnessに加え、固定sentinelとしてlexicographic first finite-RTE cell `(r=1,K=2)` を使ってwrapper smoke testする。結果をresource rankingへ使わない。

これによりS1は「正しさの接続」の役割へ限定する。

---

# 6. S1 artifactに必須の項目

S1終了時に一つのsummary JSON/MDを作る。

- fixed commit/amendment commit
- S0 snapshot hashes
- prefix identity結果
- B3 exact semantics
- all correctness tests
- raw/corrected signal convention
- normalization multiplier range
- candidate eligibility count（resource winnerは決めない）
- q=1 wrapper probe result
- compile smoke-test pass/fail
- unresolved implementation issue
- unexpected structural issue
- `automatic_next_stage: null`

S1後に**必ず停止**。

---

# 7. S1後にCodexがしてはいけないこと

- S2を開始
- 32/128 full MC gridを開始
- S3 held-out signal/costを開封
- rank/precision/gridを追加
- H12/別分子/別geometry
- PR-3/4/5/6
- 長RPE
- final total cost
- resultを見てnormalization conventionを戻す

---

# 8. S1後にGPTへ渡すもの

1. amended preregistration
2. v2/amendment dry-run manifest
3. S0 artifact
4. S1 summary artifact
5. prefix identity record
6. corrected estimator tests
7. test log
8. commit SHA
9. dirty/clean worktree status
10. S1中に発生したdeviation一覧

GPT側で次の一つを判断する。

- `PROCEED_S2`
- `REVISE_SCOPE_BEFORE_S2`
- `STOP_PR2`

S2の設計詳細はその判断後に再確認する。

---

# 9. Codex向け短い実行指示

> fixed review commit `d3e17239702b56e765ff0a2f8993135015332ea8` を基準に、外部レビューのA01–A06を結果前amendmentとして反映してください。数値結果を見る前に旧hash・変更理由・new hashを保存してください。特にfinite-RTEのknown normalizationをprimary estimatorで補正し、normalization multiplierの二乗をshot formulaへ戻してください。B3は現実装に合わせて「LD=0 two-body-random endpoint; deterministic one-body retained」へ改称してください。compiled-cost uncertaintyと10% materialityをengineering intervalで接続し、implementation bugとscientific negativeを分離してください。
>
> amendment後にS0を実装・実行し、snapshot freeze、prefix identity、adapter、corrected estimator、q1/q8 wrapper、shot testを通してください。その後S1だけを実行してください。S1では32/128 trajectory expected-cost MCを行わず、各fixed random cellのcanonical 1 trajectoryでfull-wrapper compile smoke testまでに限定してください。S1からresource winnerを決めず、summary artifactを書いて必ず停止してください。S2/S3は開始しないでください。
