# PR-2 matched-accuracy M1前最終amendment v2

作成日：2026-09-29
基準commit：`0ee4d641c10f4e248d2b03ebc59703c56f10e581`
status：`M1_PREEXECUTION_AMENDMENT_V2_FROZEN_SCIENCE_NOT_AUTHORIZED`

## 1. 位置付け

本書は、凍結済みの[M1前先行研究gate v1](pr2_matched_accuracy_prior_art_gate_v1.md)、
[resource-map契約 v1](pr2_matched_accuracy_resource_contract_v1.md)、
[M1実装契約 v1](pr2_matched_accuracy_m1_implementation_contract_v1.md)を変更せず、M1科学計算前に
必要な二点だけを追加する。

1. 2026年3月の近接研究二件をclaim単位で照合する。
2. M1-Aのselectorが`selection_limited=true`を返したら、M1-Bのcompile jobを一件も生成せず停止する。

旧S2の数値、B2/B3 frontier、rank 3/9 control、held-out未開封、`S3_authorized=false`は変更しない。
本amendment自体はsignal、sampling、circuit build、compile、量子shotを承認しない。

## 2. 追加prior-art gate

| 一次研究 | 既知として扱う内容 | M1で残す検査対象 | 今回主張しない内容 |
|---|---|---|---|
| Cugini, Atif, Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms* (2026) | randomized quantum protocolについて、回路一回当たりのhardware-dependent costとestimator varianceを同時に含むnet-costを、classical importance samplingで最適化する一般枠組み。sampling distribution変更後もbiasを保つ解析とqDRIFT等への適用 | importance-sampling分布を新たに最適化せず、事前固定した有限DF-prefix候補の`1-shot compiled cost × normalization-corrected analytic shots`を、discard/full deterministicを含むmatched-accuracy条件で測る | costとvarianceの共同最適化、importance sampling、bias-preservation一般論、randomized protocol一般のresource optimalityを新規claimにしない |
| Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime* (2026) | single-ancilla Trotter QPE、partially randomized time evolution、unitary weight concentration、化学active-spaceへのend-to-end STAR resource estimation | 固定済みH4 DF snapshotで、DF-prefix、finite cutoff、同一complex-signal accuracy、discard baseline、状態準備除外full measured wrapperのdirect compileを組み合わせた有限候補の設計判断 | single-ancilla QPE、部分ランダム化、UWC、化学QPE、early-FTQC/STARのend-to-end resource estimateまたは大規模化学実現可能性を新規claimにしない |

追加照合により、一般的な「random circuit costとvarianceの同時評価」および「部分ランダム化を用いた
化学QPEのend-to-end resource estimate」は今回の差分から除外する。残る問いは、固定DF-prefix候補に限り、
discard／deterministic／partial／random-dominantを同じcoherent-signal精度へそろえたとき、簡略な
fixed-step比較の方式判断がfull-wrapper costまで戻して維持されるか、または消えるかである。

この限定差分はM1で直接検査でき、partialの勝利を前提にしないため、最終gate判定は
`PROCEED_RESOURCE_STUDY`を維持する。ただし投稿可能性、新algorithm、一般的resource optimalityの判定では
ない。M1後に既知trade-offの再現以外が残らなければ`NARROW_TO_TECHNICAL_NOTE`または
`STOP_DUPLICATIVE`へ移る。

## 3. M1をM1-A/M1-Bへ分ける

### 3.1 M1-A：全candidateのsignal側評価

M1-Aは、凍結済み候補集合についてcorrected/raw signal、bias、normalization、解析shot、accuracy
eligibility、action proxy、r64 boundary request、16-cell selectorまでを実行する。M1-Aでは、
deterministic/discardを含むdirect compile、random trajectory compile、full-wrapper compileを全て0とする。

M1-A artifactは候補ledger、signal record、selector入力、selector出力、barrier判定、全counterを保存し、
barrier前にcompile task queue、circuit object、trajectory seed列を作らない。

### 3.2 hard precompile barrier

selector直後に次だけを許す。

```text
selection_limited = true
  -> status = SELECTION_LIMITED
  -> deterministic/discard/random compile jobs = 0
  -> circuit build/compile/full wrapper counters = 0
  -> winner/held-out candidateを確定せずmandatory STOP

selection_limited = false
  -> status = M1_A_COMPLETE_M1_B_ELIGIBLE
  -> 凍結済み選抜fingerprintだけからM1-B planを作成可能
  -> result-prior execution authorizationが許可した場合だけM1-Bへ進む
```

`SELECTION_LIMITED`後に16-cell上限を増やす、proxyを変更する、未選択候補を事後除外する、別outputで
救済することは禁止する。追加compile budgetの科学的価値、またはtechnical noteへの縮小を別reviewで決める。

## 4. machine-readable enforcement

`src/trotterlib/pr2_matched_accuracy_m1_precompile_barrier.py`は次を機械強制する。

- selectorの16-cell cap、selected count、canonical ordinal、fingerprint uniquenessを検査する。
- 未選択proxy frontier、boundary、tail challengerと`selection_limited_reasons`の一致を検査する。
- limited時は`build_m1_b_compile_plan`が`SelectionLimitedStop`を投げ、job listを返さない。
- clear時だけ、最大16 deterministic/discard cellと最大16 random cellのplan identityを返す。
- plan作成自体はcircuit buildまたはcompileを行わず、全compile counterを0で返す。

M1-A schemaはcompile recordを空、direct-compile counterを0に固定する。M1-B schemaは、byte-fixedな
M1-A artifact SHA-256、`M1_A_COMPLETE_M1_B_ELIGIBLE`、同じselected fingerprintを開始条件にする。

## 5. 現在のauthorizationと停止点

今回許可するのは文献gate、standard-library-only barrier、schema、synthetic dry-run、testだけである。
synthetic limited fixtureがSTOPし、synthetic clear controlだけがplanを作れることを確認するが、どちらも
科学値ではない。development/held-out NPZ load、分子計算、signal評価、trajectory、circuit、compile、
量子shotは全て0である。

現行statusは`M1_PREEXECUTION_AMENDMENT_V2_FROZEN_SCIENCE_NOT_AUTHORIZED`。次にM1を実行する場合は、
M1-A budget、M1-A artifact freeze、conditional M1-B budget/process/output、source identity、再開規則を
別のresult-prior execution authorizationへ固定する。held-out、S3、追加geometry、別PF、H12、長RPE、
最終総costへ自動進行しない。

## 6. 一次資料

1. D. Cugini, T. A. Atif, Y. Subasi, [*Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*](https://arxiv.org/abs/2603.13495), arXiv:2603.13495 (2026).
2. S. Kanasugi, R. Toshio, K. Maruyama, H. Oshima, [*Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*](https://arxiv.org/abs/2603.22778), arXiv:2603.22778 (2026).

この二件の照合は各論文abstractと公開本文に基づく限定gateであり、同一研究の不存在証明または網羅的
systematic reviewではない。
