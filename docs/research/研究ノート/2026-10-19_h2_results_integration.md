# H2有限標本結果の受入と資料統合（2026-10-19）

ユーザーはP完了 `f2d1a86c431f835db700f348ce618d8604e7ee23` のε=.05/n80とε=.0005/n12を、PR低費用・REPRODUCEDの有限標本点結果として正式採用した。過去のF/FP/D/A/S_STATS/Pノートと各時点の未承認・STOP状態は書き換えない。

専用branch `codex/h2-results-integration-20261019` に、全6段階のGitHub証拠整理、正式受入記録、H468主表6条件を保持した別8条件表、v2へのH2追記案、paper manifest・CEM・台帳の追加案を作成する。既存v2の別branchでの文書採択 `1f001cfc` は固定参照とし、今回のH2追記案は未採択の別版とする。

理由：結果受入、プロトコル対象拡張、主表への掲載、母平均保証、最終freezeを混同しないため。H4粗精度非再現、H6粗精度PF低費用、H8公開詳細監査不足、H2/各系のsolver・正規化・残差gateの相違を保持する。

科学計算・統計解析・新しい比/SE/CI/順位/nの計算・sampling・compileは行わない。旧D_run01のSTOP・partial・非返還予約、H4/H6/H8主表と原図、H2 private証拠を保全する。

成果物と構造照合は[統合報告](../track_a_h2_results_integration_20261019.md)を参照。次の担当は**ユーザー**：H2本文／補遺掲載、更新案の適用、H2 v2追記案の採択を別途判断する。**Codex**は現在の文書整理完了後mandatory STOPし、詳細科学レビュー・最終freezeへ進まない。
