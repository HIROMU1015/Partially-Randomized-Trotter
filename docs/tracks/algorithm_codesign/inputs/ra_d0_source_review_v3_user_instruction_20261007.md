Track B / RA-D0について、**one-shot実行前のblocking issueを1点だけ修正し、v3 final source reviewまで進めてください。**

今回はまだRA-D0 registered optimizationを実行しません。

現在の固定source：

- branch: `track-b-ra-d0-source-review-v2-20261007`
- commit:
  `4012cbd167ec10f143fbfddc89feff4dec54bf2b`

固定入力：

- R1 result:
  `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b`
- R1.5:
  `af3d014d0a0cfcbbd25bb544f6544652fec92942`
- RA-RTE mathematical audit:
  `8a04c148a66d23dbc1f045086a95a5e19a6372dc`
- RA-D0 v1 preparation:
  `0ddf67756516e08f85fed1b987459a5e862676b7`

v2で固定した以下は原則変更しないでください。

- B0_saved / B0_ideal分離
- B1_num ⊂ B2_num ⊂ B3_num
- \(2^{60}\) sampler law
- membership allowance
- B2 outer lower / B3 certified upper
- profile-paired budget vectors
- batch freeze-before-B3
- anchor-first sequential gate
- shot grid
- main LP cap 55,275
- auxiliary込み110,550
- hard cap111,000
- runtime/resource guards
- classification definitions
- candidate table
- x / precision / resource coordinates
- old R1/R1.5/STOP evidence

今回の目的は、**B2 single-resource minimum LPがexactにcertified infeasibleだった場合を、technical failureではなく正常な数学的outcomeとして扱えるように修正すること**です。

---

# 1. blocking issue

現v2 `engine.py` のbudget generationでは、B2 single-resource minimum solveについて、

- nominal feasible + implementation certificate PASS
  → minimumとして使用
- それ以外
  → `TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION`

となっています。

このため、

\[
\boxed{
\text{B2 numerical classがその }(x,n)\text{ で本当にinfeasible}
}
\]

であり、exact Farkas certificateまで取得できた場合もtechnical failureになります。

これは修正してください。

特にcoverage gridのlower-shot領域では、RA-D0全体のshot lower boundを満たしていてもB2_numがfeasibleとは限りません。

したがって、

> `CERTIFIED_INFEASIBLE` はsolver/certificate failureではなく、正常な数学的結果

として扱います。

---

# 2. B2 single-resource minimumの3分類

各

\[
(x,n,Q),\qquad
Q\in\{T,CX,1Q\}
\]

について、B2 single-resource solveを以下の3種類に分類してください。

## Case A — certified feasible

nominal solveがfeasibleで、

- q/y fixed quantization
- B2 numerical membership
- mean residual
- confidence
- workspace peak
- resource upper

がすべてPASS。

status例：

`B2_MINIMUM_CERTIFIED_FEASIBLE`

この場合のみminimum vectorとしてbudget sourceへ使用する。

---

## Case B — certified infeasible

main solverがinfeasibleを返し、その後、

- 最大1回のFarkas auxiliary
- exact rational verification

によりB2 LPのinfeasibilityが認証された場合。

status：

\[
\boxed{
\texttt{B2\_CERTIFIED\_INFEASIBLE\_AT\_N}
}
\]

これはtechnical failureにしない。

その \((x,n)\) を**primary B2-vs-B3 strict comparisonの対象外**として記録し、次のshot pointへ進む。

このcaseによってone-shot全体をSTOPしない。

---

## Case C — uncertified / technical

以下は従来どおりtechnical failure。

- solver statusがinfeasibleだがFarkas acquisition失敗
- exact Farkas verification失敗
- nominal feasibleだがquantized implementation certificate失敗
- numerical membership failure
- mean/confidence/workspace certificate failure
- unexpected solver status
- cap / identity / source failure

statusは既存のtechnical classificationへ入れる。

結果後に、

- denominator変更
- tolerance変更
- solver設定変更
- second-best search
- retry

は行わない。

---

# 3. point-level policy

ある \((x,n)\) について、B2 single-resource minima

- T
- CX
- 1Q

のうち**1つでもCase B: certified infeasible**になった場合、そのshot pointではB2の3-resource budget setを完全には定義できない。

したがって、その \((x,n)\) は

`B2_POINT_CERTIFIED_INFEASIBLE`

として扱い、

- B2 minimum budget vectorsを生成しない
- paired B2/B3 epsilon-constraint queryを生成しない
- primary strict witness判定をしない

でください。

same-n `B0_saved` vectorsが存在していても、それだけからB2 primary comparisonを作らないでください。

理由：

`B0_saved` はB2_numの数学的subsetではなく、R1 reproduction anchorだからです。

---

# 4. descriptive information

B2 certified infeasible pointについて、必要なら次を保存してください。

- x
- n
- PRIMARY_ANCHOR / COVERAGE_GRID
- B2 T/CX/1Q minimum status
- certified infeasible certificate hash
- same-n feasible B0_saved profile IDs
- B3 feasibilityを実行したか否か

ただし、原則として**budget生成不能なpointで新たにB3 optimizationを追加する必要はありません**。

もし既存設計上自然にB3 feasibilityだけ取得する場合でも、

`B3_ONLY_FEASIBLE_DESCRIPTIVE`

に限定し、

- `STRICT_DEGREE_LOCAL_WITNESS`
- STRONG
- LOCAL

へ数えないでください。

新しいdescriptive B3 queryによってcall recipeを増やさない方を推奨します。

---

# 5. freeze policy修正

Phase A / budget freezeでは、各pointの状態を明示してください。

最低限：

- `BUDGET_READY`
- `B2_POINT_CERTIFIED_INFEASIBLE`
- technical failure

を区別する。

`budget_freeze.json` のpoint entryには例えば：

- `point_status`
- `minima`
- `certified_infeasible_objectives`
- `budget_vectors`
- `queries`

を持たせてください。

`B2_POINT_CERTIFIED_INFEASIBLE` の場合：

- `budget_vectors=[]`
- `queries=[]`

でよいです。

これはfreeze artifactへ含めてhash固定してください。

Phase Bでは`BUDGET_READY` pointだけpaired queryを実行する。

---

# 6. anchor-first classificationとの関係

既存classificationの中心は変更しません。

## STRONG

両xについて、

少なくとも1つの**比較可能なPRIMARY_ANCHOR**で

\[
U^{B3}<L^{B2,\mathrm{outer}}
\]

がcertified。

B2 infeasible anchorはstrict witnessには数えない。

---

## LOCAL

strict witnessはあるが、

- 片方のxだけ
- coverage-only

の場合。

---

## NO_REGISTERED_WITNESS

次の意味に限定してください。

> B2/B3のcertified comparisonが成立した登録queryについてstrict witnessが一件もなかった。

B2 certified infeasible pointsの存在をnegative evidenceとして数えない。

---

## TECHNICAL_INCONCLUSIVE

Case Cのtechnical failure、incomplete run、resource cap等。

---

# 7. coverage gate

B2 certified infeasible anchorは、

> anchor strict witnessなし

としてcoverage-context判定に扱って構いません。

つまり、そのxでanchor strict witnessが無ければ、既存ルール通りcoverageへ進めます。

coverage gridでは、B2 certified infeasible pointを正常にskipし、次のnへ進んでください。

これにより、

> 低nのB2 infeasibilityのため全runがtechnical STOP

になる現問題を解消します。

---

# 8. call caps

この修正によってquery数は増えません。

従って以下を変更しない。

- main LP:
  `55,275`
- recipe total:
  `110,550`
- hard guard:
  `111,000`

certified-infeasible pointでは後続paired queriesが減るため、この上限は引き続き保守的です。

Farkas auxiliaryは従来どおり、

- main LP 1件につき最大1件
- retryではない

を維持。

---

# 9. resource guardsその他は変更しない

以下を維持してください。

- process=1
- threads=1
- retries=0
- per-LP wall 2s
- wall 3600s
- CPU 3300s
- RSS 1536MiB
- AS 4096MiB
- output 128MiB
- GPU 0
- synthesis 0
- science/circuit/matrix/trajectory 0

per-LP C-call preemption limitの既存注記も保持。

---

# 10. focused tests

既存80 testsを保持し、少なくとも以下を追加してください。

1. B2 minimum main solveがcertified infeasibleの場合、technical failureにならない。
2. exact Farkas verified → `B2_CERTIFIED_INFEASIBLE_AT_N`。
3. Farkas acquisition failure → technical inconclusive。
4. Farkas exact verification failure → technical inconclusive。
5. B2 certified infeasible pointはbudget vectorを生成しない。
6. B2 certified infeasible pointはpaired queryを生成しない。
7. coverage gridでcertified-infeasible pointをskipして次nへ進める。
8. certified-infeasible anchorはstrict witnessにならない。
9. B2 infeasible / B3 feasibleはSTRONG/LOCALへ数えない。
10. certified-infeasible pointが存在しても、他の比較可能pointで両x anchor witnessがあればSTRONGになれる。
11. NO_REGISTERED_WITNESSは比較可能queryだけに基づく。
12. call caps 55,275 / 110,550 / 111,000が不変。
13. denominator / solver settings / candidate table不変。
14. budget-freeze-before-B3不変。
15. old R1 result / marker / source / authorization hashes不変。

synthetic/off-domain fixtureでのみsolverを使ってください。

registered optimizationは0を維持。

---

# 11. source / contract更新

v3で最低限更新してください。

- execution contract
- budget freeze schema
- bounded query recipe
- output schema
- engine logic
- focused verification
- source review report
- GPT handoff
- evidence/source manifest

必要なら新status stringをschemaへ追加。

推奨branch：

`track-b-ra-d0-source-review-v3-20261007`

v2 branchを上書きせず独立branch/worktreeにしてください。

---

# 12. 今回禁止

まだ実行しない：

- registered RA-D0 optimization
- actual B2 minima
- actual budget freeze
- B3 registered solve
- strict witness取得
- authorization作成
- one-shot marker
- new synthesis
- new angle
- new precision
- IS
- CTS追加
- DF
- molecule
- GPU
- trajectory
- science execution

registered-domain solver callはsource-levelで引き続き拒否。

---

# 13. 完了報告

commit・push後、以下を報告してください。

- branch
- full SHA
- remote一致
- clean worktree
- tests数 / PASS
- certified B2 infeasibilityがnormal outcomeになったこと
- uncertified infeasibilityはtechnicalのままであること
- coverage skip behavior
- classificationへの影響
- main / total / hard LP caps不変
- candidate table 21/x不変
- sign controls 18不変
- old R1 protected hashes不変
- registered solver calls=0
- synthesis=0
- science=0
- blocking issueの有無

blocking issueが無ければ最終statusを

\[
\boxed{
\texttt{READY\_FOR\_RA\_D0\_ONE\_SHOT\_AUTHORIZATION}
}
\]

としてSTOPしてください。

authorizationはまだ作成しないでください。

そのv3固定commitをGPTが最終確認し、問題が無ければ、

\[
S
\rightarrow
\text{authorization-only direct child }A
\rightarrow
\text{explicit one-shot execution}
\rightarrow
\text{mandatory STOP}
\]

へ進めます。