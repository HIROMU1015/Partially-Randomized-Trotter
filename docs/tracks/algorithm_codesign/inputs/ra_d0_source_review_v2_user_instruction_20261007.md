Track B / RA-D0について、**現在のsource preparationを修正し、saved-table one-shot実行前の最終source reviewまで進めてください。**

今回はまだRA-D0 registered optimizationを実行しません。

現在の基点：

- branch: `track-b-ra-d0-source-preparation-20261006`
- commit: `0ddf67756516e08f85fed1b987459a5e862676b7`

固定入力：

- R1 result:
  `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b`
- R1.5:
  `af3d014d0a0cfcbbd25bb544f6544652fec92942`
- RA-RTE mathematical audit:
  `8a04c148a66d23dbc1f045086a95a5e19a6372dc`

既存のR1 result / marker / source / authorization、R1.5、過去STOP、Track Aは変更しないでください。

今回の目的は、

> **数値baseline規約、budget freeze、bounded query recipe、execution guardを結果前に固定し、RA-D0 one-shotを実行可能なsourceへ閉じること**

です。

新しいangle、synthesis、science、DF、分子、trajectory、GPU計算は行いません。

---

# 1. B0 / B1 / B2 / B3の数値baseline規約を修正

現在判明している通り、

- `B0_saved`
- ideal representation
- dyadic numerical sampler

を文字どおり一つのnested setとして扱わないでください。

以下に分離してください。

## 1.1 `B0_saved`

R1で実際に保存された9 profiles / x。

役割：

- immutable reproduction anchor
- saved shot count / resource vectorの参照
- budget-vector source

のみ。

`B0_saved ⊂ B1_num`

とは主張しないでください。

R1の既存coefficient rounding / midpoint accountingを変更・再分類しないでください。

---

## 1.2 `B0_ideal`

ordinary / PTSC-K0 / Aの理想的finite-mean decomposition。

これは数学的embedding確認専用です。

ideal arithmeticでは

\[
B0_{\rm ideal}\subseteq B1_{\rm ideal}
\]

を維持してください。

---

## 1.3 numerical comparison classes

実際のRA-D0比較classは

\[
\boxed{
B1_{\rm num}\subseteq B2_{\rm num}\subseteq B3_{\rm num}
}
\]

となるように実装してください。

共通：

- denominator
  \[
  D=2^{60}
  \]
- q:
  largest remainder
- tie:
  index order
- y:
  nearest / half-up
- negative nominal q:
  clipせずreject
- denominatorを結果後に増やさない

を維持してください。

---

# 2. B1 / B2 membership residual

3 precision implementationを持つlogical group \(g\) について、
quantization後のgroup massとideal group membershipの差に、結果非依存の許容幅を入れてください。

固定幅：

\[
\boxed{
\tau_g
=
\frac{3+c_{g,\mathrm{up}}/2}{2^{60}}
}
\]

を使用してください。

ここで \(c_{g,\mathrm{up}}\) は既存outward intervalによるそのideal group coefficientのupper bound。

解釈：

- \(3/2^{60}\):
  3 precision q entriesのworst-case dyadic rounding allowance
- \(c_{g,\mathrm{up}}/(2\cdot2^{60})\):
  y half-up rounding allowance

です。

この式を結果後に調整しないでください。

---

## 2.1 B1_num

固定representation \(r\) について、

\[
\left|
\sum_p q_{g,p}
-
y\,c_g^{(r)}
\right|
\le\tau_g
\]

をinterval-awareにcertificateしてください。

---

## 2.2 B2_num

latent whole-representation variables

\[
z_r\ge0
\]

を使い、

\[
\sum_r z_r=y
\]

としてください。

representation \(r\)、group \(g\)について、

\[
\left|
\sum_p q_{r,g,p}
-
z_r c_g^{(r)}
\right|
\le\tau_g
\]

をcertificateしてください。

\(z_r\) はphysical sampler probabilityではなくnumerical membership witnessです。

zを2^60 sampler lawへquantizeする必要はありません。

physical lawは最終qです。

---

# 3. mean / confidence certificate

B1/B2/B3すべてについて、共通して

\[
\xi\ge\|Dq-yt\|_1
\]

のinterval-certified upperを使い、

\[
\xi\le y\delta_{\rm num},
\qquad
\delta_{\rm num}=10^{-12}
\]

を維持してください。

confidence：

\[
e y-d^Tq-\xi
\ge
\kappa_{n,\rm up},
\qquad
e=1/200.
\]

\[
\ell_{\rm up}\ge\log(10560)
\]

および

\[
\kappa_{n,\rm up}
\]

のoutward certificateを使います。

nominal float feasibilityだけで判定しないでください。

---

# 4. B2 objective lower

strict witness比較用B2 lower boundは、

> **B2_numを確実に包含するouter relaxation**

のcertified dual lowerを使用してください。

outer relaxationのoptimizer自体をphysical B2 samplerとは呼ばないでください。

primary strict witnessは、

\[
\boxed{
U_Q^{B3}
<
L_Q^{B2,\mathrm{outer}}
}
\]

のみです。

point float比較、nominal optimum比較、solver tolerance比較では判定しません。

---

# 5. B2 single-resource minimaからbudgetを先にfreeze

各 \(x,n\) について、B3 solverを一切呼ぶ前にB2で

\[
\min G_T,
\qquad
\min G_{\rm CX},
\qquad
\min G_{1Q}
\]

の3問を解いてください。

各B2 minimizerについて：

1. nominal solve
2. q / y quantization
3. B2 numerical membership certificate
4. mean residual certificate
5. confidence certificate
6. workspace peak check
7. objective/resource upper certificate

を行ってください。

### 重要

3個のsingle-resource minimaのいずれかで、certified feasible implementation pointを取得できなかった場合、

その \((x,n)\) は

`TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION`

としてSTOP対象にしてください。

結果後に：

- denominatorを増やす
- toleranceを変える
- 別solver設定を試す
- second-best solutionを探す

等の救済をしないでください。

---

# 6. budget vectorsはCartesian productしない

元recipeのresource-wise Cartesian productは使用しないでください。

各 \(x,n\) についてbudget-vector sourceは次だけです。

### source A

same-nでconfidence-feasibleな`B0_saved` profileの完全3-resource vector。

最大9 vectors。

### source B

上でcertifiedした3つのB2 single-resource minimizerの完全3-resource vector。

最大3 vectors。

合計最大：

\[
12
\]

vectors。

duplicate vectorはexact/certified resource valuesでdeduplicateしてください。

---

## 6.1 epsilon-constraint query

目的resource \(Q\) を一つ選ぶ。

budget vector \(v\) の残り2座標を、そのまま同じvectorから使用する。

例えばQ=Tなら、

\[
G_{\rm CX}\le v_{\rm CX},
\qquad
G_{1Q}\le v_{1Q}
\]

の下で

\[
\min G_T
\]

をB2/B3双方で解く。

別profile由来のCX budgetと1Q budgetを組み合わせないでください。

---

# 7. budget freezeをB3より前にartifact化

one-shot runnerは二段階にしてください。

## Phase A — `BUDGET_FREEZE`

全対象 \(x,n\) について、

- same-n feasible B0 vectors
- certified B2 T minimum
- certified B2 CX minimum
- certified B2 1Q minimum
- deduplicated complete budget vectors
- derived paired epsilon-constraint queries

を生成する。

その時点で

`budget_freeze.json`

を保存し、

- SHA256
- input identities
- number of vectors
- number of queries

を固定してください。

### Phase Aでは

\[
\boxed{\text{B3 solver calls}=0}
\]

であることをsource-level guardとtestで保証してください。

budget freeze完了後にのみPhase Bへ進める設計にしてください。

---

# 8. Phase B

freeze済みartifactをread-only入力として、

- B2 outer certified lower
- B3 certified primal upper

を取得してください。

B3の結果から、

- budget追加
- budget削除
- grid追加
- solver設定変更

をしないでください。

---

# 9. infeasibility規約

solver statusだけでscientific classificationを行わないでください。

## certified infeasible

exact Farkas certificateがverifyされた場合のみ

`CERTIFIED_INFEASIBLE`

とする。

## certificate failure

nominal solverがinfeasibleでも、

- Farkas acquisition failure
- exact ray verification failure

なら、

`D0_TECHNICAL_INCONCLUSIVE`

です。

retryしないでください。

## B3-only feasibility

B3がcertified feasibleでB2がcertified infeasibleでも、そのqueryは

`B3_ONLY_FEASIBLE_DESCRIPTIVE`

として保存。

primary

`STRICT_DEGREE_LOCAL_WITNESS`

には数えないでください。

primary witnessは必ず

\[
U_{B3}<L_{B2}
\]

のcertified comparisonに限定します。

---

# 10. shot execution sequence

shot-grid definition自体は現在の

\[
n_{k+1}=\lceil201n_k/200\rceil
\]

を維持してください。

ただしone-shot executionは**anchor-first sequential gate**にします。

---

## Phase P1 — anchors

まず、

\[
x=1/8,\;1/4
\]

について、各9個の

`PRIMARY_ANCHOR`

shot数だけ実行してください。

両xでanchor strict witnessが存在した場合、

`D0_STRONG_DEGREE_LOCAL_SIGNAL`

として終了。

coverage gridを実行しません。

---

## Phase P2 — conditional coverage

### 片方のxだけanchor witnessあり

witnessのなかったxだけcoverage gridへ進む。

### 両方anchor witnessなし

両xをcoverage gridへ進める。

### coverage-only witness

`D0_LOCAL_DEGREE_LOCAL_SIGNAL`

の範囲に留める。

coverage結果からSTRONGへ昇格しない。

このsequential ruleを結果前にsource / contractへ固定してください。

---

# 11. classification

以下を維持してください。

### `D0_STRONG_DEGREE_LOCAL_SIGNAL`

両xで、少なくとも1つのPRIMARY_ANCHORにcertified strict witness。

### `D0_LOCAL_DEGREE_LOCAL_SIGNAL`

strict witnessは存在するが、

- 一方のxだけ
- またはcoverage only

### `D0_NO_REGISTERED_WITNESS`

登録されたanchor/必要coverage queryにおいて、

certified

\[
U_{B3}<L_{B2}
\]

が一件もない。

これは

> 今回の固定R1 table・profile-paired budgetsにcertified strict witnessがない

という意味だけ。

degree-local improvement一般の不存在とは解釈しない。

### `D0_TECHNICAL_INCONCLUSIVE`

certificate / source / guard / cap / membership / budget-freeze failure。

---

# 12. LP call upper bound

budget-vector方式へ変更したため、contract上のmain call上限を更新してください。

各 \(n\)：

- B2 single minima：3
- 最大12 budget vectors
- 3 objectives
- B2/B3 pair：2 calls

よって最大

\[
3+12\times3\times2
=
\boxed{75}
\]

main LP calls / shot point。

全737点を使った理論worst-case：

\[
737\times75
=
\boxed{55,275}
\]

main calls。

各main LPにつき最大1つのFarkas auxiliaryを認める保守上限：

\[
\boxed{110,550}
\]

total LP calls。

実際はanchor-first gateにより少なくなり得ます。

---

# 13. execution compute guard

今回のsaved-table development one-shot用に、次を結果前固定してください。

- processes: `1`
- solver/BLAS threads: `1`
- retries: `0`
- total LP calls including auxiliary:
  `111000`
- per-LP wall cap:
  `2 s`
- total wall cap:
  `3600 s`
- total CPU cap:
  `3300 s`
- peak RSS:
  `1536 MiB`
- virtual address:
  `4096 MiB`
- output cap:
  `128 MiB`
- GPU:
  `0`
- synthesis calls:
  `0`
- science/circuit/matrix/trajectory calls:
  `0`

cap到達時は

`D0_TECHNICAL_INCONCLUSIVE_RESOURCE_CAP`

としてSTOP。

prefixだけからpositive / negative classificationを出さないでください。

---

# 14. output policy

全solver vectorを無条件に保存して巨大化させないでください。

全paired queryについて最低限：

- query_id
- x
- n
- PRIMARY_ANCHOR / COVERAGE_GRID
- objective
- source budget vector ID
- two resource caps
- B2 status
- certified B2 lower
- B3 status
- certified B3 upper
- strict witness bool
- certificate hashes

を保存。

完全なprimal / dual / Farkas vectorは最低限：

- 3 B2 single-resource minima / x,n
- strict witnesses
- certified infeasible query
- technical failure

について保存してください。

必要なら圧縮JSONL等を使って構いませんが、schemaは結果前固定してください。

---

# 15. focused tests / static audit

次を追加・更新してください。

最低限：

1. `B0_saved` と `B0_ideal` が別概念であること。
2. ideal:
   \[
   B0_{\rm ideal}\subseteq B1_{\rm ideal}
   \]
3. numerical:
   \[
   B1_{\rm num}\subseteq B2_{\rm num}\subseteq B3_{\rm num}
   \]
4. B1 membership allowance \(\tau_g\)。
5. B2 latent z membership certificate。
6. quantized B2 feasible artificial fixture。
7. B2 outer lower <= certified B2 implementation objective。
8. Phase A中にB3 solver callが発生するとfail。
9. budget freeze hash変更後にPhase Bを拒否。
10. budget-vector Cartesian mixingが発生しない。
11. max vectors=12。
12. max main LP calls=55,275。
13. total with auxiliaries <=110,550。
14. anchor-first gate。
15. coverage-only result cannot become STRONG。
16. uncertified infeasibility cannot become negative result。
17. cap hit => technical inconclusive。
18. denominator変更・retry禁止。
19. sign-table identity control保持。
20. old result/marker/source hashes不変。

既存35 testsも保持し、必要なら追加してください。

---

# 16. artifact / docs

新しいsource-review revisionとして、最低限以下を作成してください。

- amended RA-D0 execution contract
- numerical-baseline amendment
- budget-freeze schema / policy
- bounded query recipe
- execution resource guard
- updated focused verification
- source review report
- GPT handoff
- evidence manifest

推奨branch名：

`track-b-ra-d0-source-review-v2-20261007`

元branchを直接書き換えて履歴を潰さないでください。

---

# 17. この段階で禁止すること

今回まだ禁止：

- registered RA-D0 optimization
- B2 minima取得
- budget実値取得
- B3 result取得
- strict witness取得
- new synthesis
- new angle
- new precision
- IS
- CTS追加
- DF
- molecule
- trajectory
- GPU
- science runner

synthetic/off-domain fixturesだけはtechnical verification用に可。

registered-domain solver callは引き続きsource-levelで拒否してください。

---

# 18. 完了時の報告

commit・push後、

- branch
- full SHA
- remote一致
- worktree clean
- tests数 / PASS
- candidate columns / sign controls不変
- new main LP cap
- new total LP cap
- B1⊂B2⊂B3 numerical nesting確認
- budget-freeze-before-B3確認
- registered optimization calls = 0
- synthesis calls = 0
- science calls = 0
- old R1 result / marker hash不変
- blocking issueの有無

を報告してください。

blocking issueがなければ、

`READY_FOR_SEPARATE_RA_D0_ONE_SHOT_REVIEW`

まででSTOP。

**authorizationは作成しないでください。**

その時点でGPT側が最終source reviewし、別authorization-only childからRA-D0 saved-table one-shotを認可するか判断します。

全作業後mandatory STOPしてください。