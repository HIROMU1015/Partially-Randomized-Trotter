# ALL_SAVED458：PF選択・費用順位比較の結果

基点f63065fb、SOURCE19f2c5a9409db69fd8bd6500eb4228fefa00145c。
H6 signed30/H8両bias40の70scope（unique30条件）を明示承認のone-shotで比較した。

最良PF候補変更2、q/PF次数変更0、最良PF総費用変更30、PR/PF方向逆転0。途中順位の厳密逆転146pair、tie変化8pair。scope/倍率の重複を含む。H8非相殺ε=.0005ではp16→p15（q4/S4は同じ）。

[報告書](../../../../docs/research/track_a_pf_selection_cost_comparison_20261020.md)・[70件の結果表](conditions70.md)。

| File | 内容 |
| --- | --- |
| candidate_totals.csv/json | 5,040 PF候補条件行、適格2,594件の基準再現と直接価格置換 |
| selection_comparison.csv/json | 70scopeの最良PF/q/prefix/PF/総RZ、保存PR費用/SE/cohort、PR比、変更flag |
| candidate_ranks.csv/json | 全適格候補の辞書順tie込みordinalとcost tie rank |
| rank_relation_changes.csv/json | 厳密大小逆転146組、tie変化8組の明細 |
| used_price_evidence.json | 使用450新軸＋8既存軸＋102保持旧軸の来歴 |
| summary.json | scope別集計。独立70条件とは解釈しない |
| excluded_unresolved30.json | 未解決H6非相殺30件は未実行で保存 |
| sealed_contract.json / grant.json | SOURCE・入力SHA・算術runtime・承認70件・禁止call |
| launch.marker / one_shot.marker / ledger.json | 一度だけの起動・科学実行とcall記録 |
| baseline_replay_gate.json | 2,594基準費用と70最良PF/PR/比の完全一致 |
| numeric_completion.json | 数値結果SHA、前後入力/SOURCE gate PASS |
| native_started.json / native_exit_evidence.json | 起動親による実wait4・exit0/reap、wall/CPU/RSS |
| structural_validation.json | 人工13件、整数/Decimal検算、順位/tie/比、図と保全監査 |
| figure_data.json / rendering_completion.json | 図座標・caption・6出力SHA |
| packet_manifest.json / publication_scope.json | 成果物hashと未承認の新公開範囲 |
| STOP.json | 承認70件完了、未解決30件/追加stage未実行のmandatory STOP |

| 図 | PNG | SVG | PDF |
| --- | --- | --- | --- |
| 最良PF総費用の変化 | [PNG](PF_cost_change.png) | [SVG](PF_cost_change.svg) | [PDF](PF_cost_change.pdf) |
| PR/PF比の前後比較 | [PNG](PR_PF_ratios.png) | [SVG](PR_PF_ratios.svg) | [PDF](PR_PF_ratios.pdf) |

linear H6 DF19/H8 DF28、1 Å/STO-3G/T=.8/P=0、状態準備なしfull Hadamard wrapper RZ×保存shot。基準は旧モデル/実測混在、PRは固定確認n32。H6 profile差、runtime未証明、S4仮説、初回最終gate欠測、旧profile48の条件を保持する条件付き比較。

新compile/sampling/モデル評価/fit/shot数/PR統計再計算0。元SOURCE/raw/grant/ledger/marker/STOP・論文主表・Figure 2不変。未解決30件へ進まない。新公開未承認、詳細科学レビュー/evidence freezeなし、mandatory STOP。
