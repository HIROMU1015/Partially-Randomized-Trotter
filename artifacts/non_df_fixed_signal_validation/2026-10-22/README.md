# 非DF PR/PFの人工技術検証

分岐元 `b81d4096e806efb6523304764d3a0680b7663320`。実分子の新規科学計算は0件。
[技術文書](../../../docs/non_df_fixed_signal_validation.md)を先に読む。

- `inputs/`：人工Pauli 1/2 qubit、人工MO 1 spatial orbital、人工DF 2 qubitと供給済み状態。
- `results/`：新runnerの人工signal・normalization・C/N/GとSOURCE/input/environment。
- `SOURCE.json`：新source commit・ファイルhash・元branchとの関係。
- `validation_summary.json`：人工テスト、対象を限定したDF回帰、5件除外・1件runtime skipと初期失敗の記録。
- `technical_attempts/`：中断前／再開後の人工テストJUnit。初期失敗を保持し、最終PASSと区別する。
- `preservation.json`：旧科学結果・SOURCE等に対する変更なしの検査。
- `STOP.json`：今回の実装・人工技術検証終了。次の実分子科学計算の承認ではない。

Nは宣言した経験的numerical allowance下のHoeffding十分条件、C/Gは人工回路の有限MC平均。
本成果物は実分子の正しさ・科学的優位性・rigorous guarantee・全RPE最終費用を示さない。
