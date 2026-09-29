# PR-2 matched-accuracy resource study：M1前先行研究gate v1

作成日：2026-09-29
基準commit：`a9171d8bea93441afc9b17c7ffeab79af8dbcc95`
対象：PR-2 S2後のDF-prefix部分ランダム化resource-map再設計
判定：`PROCEED_RESOURCE_STUDY`

## 1. このgateの役割

本書は、M1の新しいsignal評価またはdirect compileを開始する前に、最も近い既存研究との
重複と残る研究差分を固定する。これは同一研究の不存在証明、網羅的systematic review、
投稿可能性の保証ではない。S2の数値、`S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`、rank 3/9の
control区分、B2/B3 frontier、held-out未開封、`S3_authorized=false`を変更しない。

このgateで問うのは、新しい部分ランダム化algorithmを発明したかではなく、既知の方法を
DF-native finite-RTE、normalization-corrected signal、full measured Hadamard wrapper、
matched-accuracyなbaselineへ接続した限定resource studyに、独立して報告できる問いが残るかである。

## 2. 判定区分

| 判定 | 条件 | M1の扱い |
|---|---|---|
| `PROCEED_RESOURCE_STUDY` | 新手法claimを外しても、同一estimand・同一accuracyでの強いbaseline比較から、既存の簡略評価では未確定な設計判断を検査できる | 結果前契約をfreezeした後に限り、限定M1を実装可能 |
| `NARROW_TO_TECHNICAL_NOTE` | 新しいmethod deltaはなく、残る価値が再現可能な実装比較、negative result、またはcost-accounting上の注意に限られる | M1はtechnical noteを閉じる最小範囲へ縮小 |
| `STOP_DUPLICATIVE` | 同じestimand、候補集合、accuracy、cost scope、判断規則が既存研究で十分評価済みで、新しい設計判断も生じない | 新規計算を開始しない |

## 3. claim単位の重複表

| 一次研究 | 既知として扱う内容 | 今回追加を検査する内容 | 今回主張しない内容 |
|---|---|---|---|
| Günther et al., *Phase estimation with partially randomized time evolution* | deterministic項とrandom tailを組み合わせる部分ランダム化、RTEによるsignal、single-ancilla phase estimation、partitionと反復費用を含む詳細resource estimate | 固定DF snapshot上でdiscard／deterministic／partial／random-dominantを同じ複素signal精度へそろえ、finite cutoff、補正後signal、解析shot、実測full-wrapper compiled costを同時に戻したときの設計判断 | 部分ランダム化、RTE、single-ancilla phase estimation、partition optimizationまたはその漸近改善を本研究が初めて提案したとはしない |
| Hagan--Wiebe, *Composite Quantum Simulations* | Hamiltonianをdeterministic simulationとrandomized simulationへ分ける構成、partitionに伴うerror/cost trade-off | composite simulationの一般理論ではなく、DF-prefix、finite-RTE、固定wrapper/compilerでdiscardまで含めた有限候補の適用条件 | deterministic/random hybrid、Trotter/qDRIFT合成または一般partition原理を新規claimにしない |
| Casares et al., SPRINT/GRADE | factorization、near-integrable PF、randomization、remainder処理、具体的化学taskでのerror/costに基づく方式選択 | GRADE/SPRINTを再提案せず、保存済みDF prefixとpaired finite-RTEの特定実装で、固定q評価とmatched-accuracy評価の判断差を測る | 一般的なfactorization-plus-random-remainder戦略、新しいfactorizationまたは大規模化学resource優位を主張しない |
| Oumarou et al., RC-DF | regularized compressed DFとその資源削減 | prefix残差のdiscard／random completion比較を強いbaselineとして置き、圧縮とsimulation誤差を混同しない | weight-ranked prefix切断を新しいcompressed-DF法またはRC-DF代替と呼ばない |

## 4. 残る限定差分

M1を正当化する差分候補は、次の全てを同じtaskで満たす比較である。

1. targetは同じ保存済みrank-12 DF Hamiltonianとstateに対する複素coherent signalである。
2. discard、full deterministic、intermediate partial、one-bodyを保持するrandom-dominant endpointを含む。
3. 固定`T=0.8`のもとで各方式に`q={1,2,4,8}`を与え、`delta=T/q`とする。
4. finite-RTE normalizationを補正したbiasと`B_total^2`を含む解析shotを用いる。
5. costは同一candidateから作る状態準備除外full measured Hadamard wrapperを直接compileする。
6. 1-shot cost、normalization、biasによるstatistical allowance、shot数を分けて説明する。
7. 共通state-preparation costはsecondary sensitivityとし、都合のよい一点だけを選ばない。
8. developmentで固定した少数candidateだけを、別authorization後にheld-outへ無調整で移す。

これは一般algorithmの新規性ではなく、特定のDF-native実装における評価設計と適用条件の差分である。
この差分だけでも、固定qまたは弱いdiscard対照で得た判断がmatched-accuracy比較で変わるかを
再現可能に検査できるため、M1へ進む科学的理由は残る。

## 5. 判定と制約

判定は`PROCEED_RESOURCE_STUDY`とする。ただし次の制約を同時に固定する。

- 表題・abstract・主RQはmethod inventionでなく、限定したresource/applicability studyとする。
- `rank 3が勝つ`、`partialが常に有利`、`global optimum`、`量子優位`を目標claimにしない。
- M1で公平なbaselineを戻した後も、既知trade-offの再現以外の設計判断またはfailure mechanismが
  残らなければ、完了判定を`NARROW_TO_TECHNICAL_NOTE`へ下げる。
- candidate選抜上限のため結論が変わり得る場合は`SELECTION_LIMITED`とし、winnerまたはheld-out
  candidateを確定しない。
- M1の実装・実行、held-out、S3、追加geometry、H12、長RPE、最終総costは本gateでは承認しない。

## 6. M1終了時の再判定

M1結果は次のいずれかへ収束させる。

| M1後判定 | 条件 |
|---|---|
| `CONTINUE_TO_FROZEN_TRANSFER_REVIEW` | 強いbaseline後にも非自明なresource categoryまたはfailure mechanismが残り、選抜制限がない |
| `NARROW_TO_TECHNICAL_NOTE` | 実装・negative result・accounting上の注意は残るが、独立resource claimは狭い |
| `STOP_DUPLICATIVE` | 観測結果が既知評価の直接再現に留まり、限定studyとしても新しい判断が残らない |
| `SELECTION_LIMITED` | compileされなかった候補が結論を変え得るため、winner／transfer candidateを確定できない |

どの判定でもM1完了時に停止する。held-outへ自動進行しない。

## 7. 一次資料

1. J. Günther et al., [*Phase estimation with partially randomized time evolution*](https://arxiv.org/abs/2503.05647), PRX Quantum 7, 020332 (2026).
2. M. Hagan and N. Wiebe, [*Composite Quantum Simulations*](https://quantum-journal.org/papers/q-2023-11-14-1181/), Quantum 7, 1181 (2023).
3. P. A. M. Casares et al., [*Theory and practice of Trotter product formulas for quantum chemistry*](https://arxiv.org/abs/2606.30741), arXiv:2606.30741 (2026).
4. O. Oumarou et al., [*Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*](https://quantum-journal.org/papers/q-2024-06-13-1371/), Quantum 8, 1371 (2024).

今回の重点監査は同一研究の不存在証明ではない。M1後の原稿化判断では、各論文の同じestimand、
候補選択、finite cutoff、controlled-wrapper scope、state-preparation scopeをclaim単位で再照合する。
