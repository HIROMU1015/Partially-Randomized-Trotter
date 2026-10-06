# R0.5 fixed symbolic comparisons: readout v1

2026-10-06 JST. 以下は保存済み [exact artifact](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/exact_symbolic_comparison_v1.json) の表示用再掲。
再実行・追加target・比較追加・precision変更は行っていない。
生成source / 事前domainは [result-prior protocol](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/result_prior_protocol_v1.json) に固定。

## Targetと二つのfixture

targetは P_m(-i sigma x Rhat)、m={3,5,7}、x={1/8,1/4,1}、sigma=±1。p=(1/2,1/3,1/6)。

- 自由word fixture: 3 generators、Q_i^2=Iのみ。自由word多項式でA ordinary/A optimum/PTSC K0のfirst meanを比較。
- I1 Pauli fixture: one-qubit X/Y/Z。同じpでPauli multiplicationをexact有理複素係数で行い、上記3構成とpadded/collected CTSを比較。
  dense matrix・固有値solver・trajectory samplingは使わない。
- 小fixture evaluatorのword列挙はこのdocs/symbolic監査の許可範囲。I0 generatorが全word列挙を要求するという意味ではない。

9 normalization条件、18 sign比較のmeanが一致。一般class証明は既存R0を参照し、このfixtureから外挿しない。

## Normalization

表はartifact midpointの小数9桁表示。判定は96-bit scaleのoutward **exact rational interval** で行った。
丸めた表示値は判定に使わない。全9条件でintervalは分離し、追加precisionや数値再試行は0。

| m | x | Ordinary I0 | A optimum I0 | PTSC K0 I0 | CTS padded I1 | CTS collected I1 |
|---|---|---:|---:|---:|---:|---:|
| 3 | 1/8 | 1.015601497 | 1.015574971 | 1.015920239 | 1.015635146 | 1.010804719 |
| 3 | 1/4 | 1.062134726 | 1.061736986 | 1.064630573 | 1.062661104 | 1.042684029 |
| 3 | 1 | 1.941259839 | 1.900292375 | 2.080880229 | 2.036590743 | 1.563594301 |
| 5 | 1/8 | 1.015611673 | 1.015585097 | 1.015930666 | 1.015645350 | 1.010803185 |
| 5 | 1/4 | 1.062297689 | 1.061897010 | 1.064801472 | 1.062825857 | 1.042659711 |
| 5 | 1 | 1.983751668 | 1.938391372 | 2.130880229 | 2.084594079 | 1.558153998 |
| 7 | 1/8 | 1.015611678 | 1.015585102 | 1.015930672 | 1.015645356 | 1.010803186 |
| 7 | 1/4 | 1.062298029 | 1.061897342 | 1.064801823 | 1.062826199 | 1.042659731 |
| 7 | 1 | 1.985154658 | 1.939616394 | 2.132467531 | 2.086134072 | 1.558227707 |

各条件でA < ordinary、A < PTSC K0、A < padded CTS。一方、collected CTS < A。
**CTS padded列はgeneral I0の実装可能CTSではない。** E-1/Oはraw degree massesで、I1のper-word closure/rephasingにより明示したpadded構成を実現する。
collected CTSのLc/Lsはartifactへexact係数とともに保存。zeroth identityを別に保ち、real/imaginary collectionを行った値。
このI1比較はI0のA-class lower boundを否定せず、Aの全LCU最適性やPauli域での優位を示さない。

## Support: labelsとphase-preserving atoms

各entryは `labelled before identical-atom merge / distinct phase-preserving Pauli atoms`。
xとsigmaを変えても今回の固定fixtureでは以下のcountsが同じだった。相対phaseが異なるatomは同一扱いしない。

| m | Ordinary | A optimum | PTSC K0 | CTS padded I1 | CTS collected I1 |
|---|---:|---:|---:|---:|---:|
| 3 | 30 / 24 | 39 / 33 | 39 / 11 | 39 / 15 | 4 / 4 |
| 5 | 273 / 48 | 363 / 48 | 363 / 11 | 363 / 16 | 4 / 4 |
| 7 | 2460 / 72 | 3279 / 48 | 3279 / 11 | 3279 / 16 | 4 / 4 |

ordinaryのdegree label数は Σ_(j=0)^d 3^(2j+1)。A optimum/PTSC K0/padded CTSは Σ_(n=1)^m 3^n。
これらlabel数は自由generatorの未集約sampling labelsにも適用する。自由wordのatom同一性をPauli closureで判定していない。
表のdistinct列はI1 fixture限定で、一般involutionのdistinct supportや最小supportではない。
同じatomをまとめてもpositive-coefficient normalizationは変わらない。反対phaseのcancellationや別LCU再最適化はこのsupport計測に入れない。
support countはnative gate costやclassical acquisition costの代用ではない。

## Semantic witnessesと資源

- generic W=Q0Q1Q2はW†≠±W。I-iWのunitarity residualが非零。raw odd-degree massをCTS rotationへ直結するI0 shortcutは不可。
- Wan positive-time eventの全体adjointがordinary Aの向きへ一致することを両signで確認。
- A odd X/Z branchはidentity 0かつY/Z双方が非零。literal CTSのsigned-Pauli/single-Pauli-rotation atomではない。
  m=3,x=1,sigma=1のunnormalized numeratorは i(2/7)Y-i(2/9)Z。

technical checker実行回数=1、runtime=5.968321秒、Linux peak RSS=16,640 KiB。
事前上限はCPU 30秒、address space 256 MiB、wall 60秒。cap/error/numeric failureなし。
checker SHA256: `0eb6b56c04af19e1e0c0f22f32df4da3de1e452d0101d90ec9b3ae62553d12df`。stdlib Fractionと固定Pauli multiplicationのみ。
local technical evidenceでありimmutable CI・外部再現ではない。旧checker・科学実験は再実行していない。

限定分類 `METHOD_DELTA_CANDIDATE`、gate `CONDITIONAL-R1` の詳細は [監査](rte_reallocation_r05_equivalence_novelty_audit_v1.md)。
science/sampling/synthesis/compile/分子/DF/NPZ/GPU=0。R1 unauthorized、mandatory STOP、次判断はGPT。
