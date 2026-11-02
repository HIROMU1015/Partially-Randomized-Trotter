# 利用者のBM-0.5 review受領記録

2026-10-05の会話で受領したGPT reviewの**指示抜粋・構造化記録**。
原添付fileのraw snapshotではなく、会話本文をこの公開記録へ整理したもの。
対象はBM-0公開commit `3b2d624adde979f7d8f983bc7fdbbec88f90c500`。

判定ラベル：`REVISE_BM0_METHOD_DELTA_BEFORE_BM1`。

## 指示

研究方針全体は再設計せず、BM-0を維持する。scienceではないBM-0.5監査を先に行う。

1. 固定nested列に対し、Maxwell型compact BCH＋同じDF backendと、
   BMのK_floor+K_A/(4m^2)をsymbolicに比較する。
2. 使う／捨てる情報、計算量、候補間reuse、scoreの差を表にする。
3. 同じG_i/lambda_i/Nのcompact対照から得られない順位または評価cost低減が残ることを
   new-method BM-1提案の結果前条件とする。

代数同値ならBM-1を新手法検証として走らせない。application/engineering studyへの縮小、
または別のmethod deltaを検討する方針判断はGPT側に戻す。
明確な非同値性が残っても、preregistration/source/authorizationなしに実行しない。

## Pilot案への修正

- 初回はleading-order heuristic＋I2 oracle evaluation。finite-time certificateを要求しない。
- I1 selectorへexact finite signalを戻さず、oracleはregret/miss評価専用。
- Primaryはfusion後native logical block countの構造的比較。
- 固定人工cost c_A=1,c_B=8,c_R=4はsecondary stress test。
- 人工costだけの改善はmechanismに限定し、DF-native resource improvementとしない。

## 維持する停止条件

BM-1実行自体は未認可。今回の監査では分子計算・Hamiltonian生成・NPZ・signal・compileを行わない。
BM-1が別途承認された場合もone-shot後は全outcomeでSTOPしGPTで再評価する。
同じcompact選択、oracle regret改善なし、人工costだけの改善をnew-methodの成立と扱わない。
BM-2への自動進行は認めない。
