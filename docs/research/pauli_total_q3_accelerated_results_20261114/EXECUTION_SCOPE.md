# ユーザーが明示許可した高速版の代替2評価

2026-11-14 JST。旧run01を保全停止し、同じ固定2構成・各1 control vectorを高速版SOURCEで再実行する。旧runの履歴は`0a6f8e4a5`の`docs/research/pauli_total_q3_fixed_results_20261114/run01/SUPERSEDED_STOP.json`に保存されている。

- B統合SOURCEはeecbcdaed、元kernel710aa69b、C c628e9e84、A714247619のまま。
- 加速backendは `9e294778266c9c867a8b1b98e040c2e94b1cb635:src/trottertracks/rpe_total_dp_parallel.py`。
- 前回caller8652cd873を固定Gitから再利用し、科学gate・API呼出しを変更しない。旧state-major用の進捗観測だけを無効化し、新backendのround観測に置き換える。
- 旧INPUTのcases、target、statistical contract、optionsの科学条件を完全一致で保全。追加fieldはbackendのSOURCEとworker/task設定だけ。
- 各構成8 computational workers、2構成同時。これは人工benchmarkで8と16のwall timeがほぼ同じだったことに基づく実行選択で、科学scopeや恒常的なCPU/RAM制限ではない。
- rho=1、count4225、verification55,043,173、frontier8192、transition141,559,036のまま。
- X/Yとも `[64,60,56,52,47,43,38,34,30,25,21,17,12]`、USER_SUPPLIED。
- EXACT_AFFINE・保存ideal finite-K IID・保存solver proxy・Z1＋T1全事象loss。旧Minkowskiは参照、δE加算なし。
- 新namespaceのrun01を排他的に作成する。旧runの中断を隠さず、この2件を代替として明示する。

人工runner2件PASS。実行中は自分のworkerとその子processだけを観測。parent CPU・全child CPU・parent peak RSS・最大単独child RSS・サンプルされたtree合計RSSを区別して記録する。

UNKNOWN/停止条件・無断retry禁止・残余保全は前回と同じ。終了後はモデルMSE/moments/zero/tie/残余・性能を報告し、十分N/global RMSE/G/G_qualifiedはnullのまま。新shot探索、価格、目標精度filter、主推定量採択へ進まない。結果公開・remote照合後mandatory STOP。
