# PF q依存モデル：保存450軸の条件付き診断

基点b777ac7e、解析SOURCE274f89774ae3f65b5fd0612f93dfbfe6979db19a。
H6 192/H8 258軸の保存予測と直接実測だけを解析した。新規モデル予測/compile/fit/shot会計/順位変更0、mandatory STOP。

| 系列 | 平均絶対相対誤差 | 最大 | 2%未満 |
| --- | ---: | ---: | ---: |
| H6 DF19 | 0.353456% | 0.805394% | 192/192 |
| H8 DF28 | 0.316507% | 1.488276% | 258/258 |

[報告書](../../../../docs/research/track_a_pf_q_model_diagnostics_20261020.md)、[表示用4図](display/README.md)。相対誤差の分母は実測RZ、差の符号は予測−実測。厳密に2%未満が基準で、母集団保証・profile/runtime差を分離したモデル検証ではない。

| ファイル | 内容 |
| --- | --- |
| axis_errors.csv/json | 450軸。元SOURCE・結果SHA・予測来歴・条件を保持した誤差表 |
| grouped_errors.csv/json | 220 groupの記述集計。group間の重複あり |
| summary.json | 系列別/全体の要約と使用済みrole分類 |
| model_use_coverage.csv | 学習/検証/その他の件数。0件の精度は欠測 |
| H6_B0_p11_q4_2nd.json | 指定候補両軸の詳細 |
| figure_data.json | 初回描画座標、分布bin、英語caption |
| display/ | 目盛文字を整えた最終4図×3形式。数値は初回と同一 |
| sealed_analysis_contract.json / grant.json | SOURCE・入力SHA・定義と今回の科学実行承認 |
| one_shot.marker / ledger.json / numeric_completion.json | 実行範囲・禁止call0・数値終了と結果SHA |
| rendering_completion.json / runtime_environment.json | 初回描画の証拠と実行環境 |
| structural_validation.json | 人工10件、Decimal別計算、220 group、元入力/CSV/図の整合性 |
| packet_manifest.json / publication_scope.json | 成果物SHAと今回の未承認公開範囲 |
| STOP.json | 完了・追加stage未承認のmandatory STOP |

再検証は`python3 -B scripts/check_pf_q_model_diagnostics.py`。これは保存結果の監査であり、モデルを評価しない。元の数値runnerはone-shot markerで再実行を拒否する。SOURCE sealは旧入力・SOURCEのhashを保持し、他worktreeや共有環境に条件を課さない。

初回572軸の最終gate欠測（今回428軸）、H6 profile差、H8等のruntime未証明、S4合成仮説を保持する。残338軸、旧H2 signal修正、旧48軸は対象外。公開未承認、詳細科学レビュー/最終evidence freeze未開始。
