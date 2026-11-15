# A/C・統合担当HANDOFF：LD5 total viewと固定2配分

新規Q3 2件、FULL_PF__B_PRICE=MODEL_BOUND_MEETS_ADOPTED_THRESHOLD、VP=MODEL_BOUND_MEETS_ADOPTED_THRESHOLD。停止理由`TWO_FIXED_ALLOCATIONS_COMPLETED`。LD3は再評価していない。4点のMSE上下界・τとの差・条件付きモデル費用は[REPORT](REPORT.md)、exact正本は[SUMMARY](SUMMARY.json)。

## 受入用locator

- 科学実行SOURCE `b6d4dea49c39f364048af422bc2928ee7c169261` / docs/research/pauli_ld5_two_vectors_q3_results_20261116/INPUTS.json。2配分と評価順・target/loss/law/options・価格参照・exact会計を実行前固定。
- 原API固定snapshot `892a5217b59d8ffc35e4323d84aef21407cb6618` / `docs/research/pauli_ld5_two_vectors_q3_results_20261116/run01/01/LD5__FULL_PF__B_PRICE/API_RETURN.json` と同`run01/02/LD5__VP/API_RETURN.json`（未開始の場合はnullと理由）。
- `C_COMPARISON_JOIN.json::records`：LD5元ID `4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919eb`、total ID `42309d00702fa11c43d62eb27a5ff928657dc1a4d9d8e9401547433af5c25403`、binding、allocation/source、全26軸N/C_ref/積、original/total candidateとlaw、signal/implementation SOURCE、A exact総額の出所、Q3 statusと原APIの固定commit/path。価格kindはMODEL_REFERENCE、qualified_for_candidate=false、E_C/CI/G/G_qualified/sufficient_Nはnull。Cの正式G guardを緩めないこと。
- `TOTAL_BINDING.json`、`TOTAL_SIGNAL_VIEW.json`、`C_SIGNAL_INPUT.json`：A保存packetを保全したmetadata view。新decoder用IDと元physical IDを区別し、元receiptの単純付替えを行わない。
- `run01/RESULT_INDEX.json`、各SELECTION/OUTCOME/API/partials/RESOURCE/PERSISTED/SUPERVISION/PROCESS_EXITS：実行順・数値判定・保存状態・負荷を追跡できる。

価格はb8bcdf97bの26モデル軸。元のN/積はVPなので、FULL_PF__B_PRICEでは新配分のNを使った。両exact総額をA.comparisonに照合済み。LD3の価格・Q3をLD5へ流用していない。

## ロードマップへの影響

LD5の全13signal/26モデル価格に、total decoder接続と共通2配分のQ3証拠が加わる。4点比較は同一decoder/proxy/IID/loss/τの固定配分モデルに限る。各列のmodel適否と、物理的RMSE保証・十分N・実期待価格の資格は別。価格差から正式winnerを認定しない。別配分・価格domain検証・追加科学実行が必要なら別指示。ロードマップ本体は統合担当が更新する。

新規信号/sampling/MC/compile/価格再fit/配分生成/retry各0件。旧成果・旧STOPを保全し、mandatory STOP。
