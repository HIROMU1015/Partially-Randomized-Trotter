# 7候補の実行準備：並列化・高速化

2026-10-27 JST。元の準備SOURCE `1255f6e60656494a45aebceeefe5224313e4355f` から別branch/worktreeで実装。**科学実行は0、Phase Bは未承認のまま停止。** 候補・N・探索1,856本・確認54,833本・全56,689 trajectory・最大113,378 compile・539,044 RTE event・cohort・seed列digest・compiler・統計計画を元PLANと機械的に比較し、不変を確認した。

## 実装した変更

|箇所|変更と目的|
|---|---|
|投入順|元の候補順FIFOから、元work proxyとqubit数に基づく重い構造を先に配る順序へ変更。候補選択・科学的費用予測には使わず、trial IDの単なる並べ替えに限定|
|資源追従|30秒ごとに空きCPU/RAMと**自分の子processのCPU使用**を読み取り、同時投入数を増減。代表trial後も完了trialの最大RSSでメモリ見積りを更新。自分の計算負荷を他ユーザーの負荷と誤認しない。縮小時は実行中taskを完了させて新規投入を控え、空き0なら待機。共有設定や他jobを操作しない|
|context共有|保存入力からのcontext構築を系ごとに1回とし、探索・確認を通じてforkで共有。入力・prefix構成の変更はsignatureで検出して再構築|
|pool共有|最初の代表検査用poolの後、bulk poolを探索から独立確認へ保持。固定標本数の増減やcohortのpoolingは行わない|
|process間通信|元の完全trialは従来どおりdiskへ保存し、worker→controllerにはID・index・RSS・時間・cache統計のみ返す。巨大なprices/checks/lawsの重複転送を省略|
|schedule再利用|元のsigned schedule関数の結果をworker内で再利用し、呼出側には新しいdescriptor辞書を返してmutation伝播を防ぐ|
|同一回路の再構築抑制|workerごとに直前のevolutionを最大1個保持し、全数値入力が一致した場合だけcomposeを再利用。異なるeventは元builderへ委譲。miss時は旧回路を解放してから構築し、回路cacheが増え続けない|
|SOURCE検査|providerをファイルごとに別git processで取得する処理を、単一git archiveの全file byte比較へ変更。検査の内容は維持|

回路再利用keyはcandidateの全field、scalarのexact float表現、preparationのobject/hash、各pieceのsigned time、全eventの数値辞書、component_specsのobject identity、control/ancilla/order/basis-reuse/identity-phase/tail/step数、明示basis planを含む。spec/preparationの参照を保持し、object IDの再使用による誤hitを防ぐ。未対応schemaは元builderへfallbackする。元wrapperが行う数値回路fingerprintの再検査も維持する。

毎回の元sampler、seed/cohort/index、events保存、paired統計は省略しない。独立に引いた2つのevent列が同じ数値回路になった場合の回路再利用であり、同じrandom標本を二重計上する処理ではない。新旧scientific SOURCE本体・native correctness閾値・compile設定・元fingerprint cacheは編集していない。代表trialは各cohortで引き続き検査し、成功前に残りを投入しない。

## 確認したことと未測定事項

人工/mock **21件PASS**。全56,689 trialのID集合不変・欠落/重複なし、資源投入数の増加と減少、最初の失敗後の追加投入抑止、context/pool共有、event/scalar/spec/control/basis/timeの変更時にcache miss、signed zeroの分離、schedule descriptorの変更がcacheへ伝播しないこと、探索・確認のexact会計分離を確認した。

`performance_checks.json` は分子を扱わない確認結果である。7種類の既存scheduleを各1,000回使う小規模測定では、元schedule計算7,000回を初回7回へ減らし、3反復の中央値は約0.0499秒→0.0123秒（約4.06倍）だった。これは**schedule算術だけの測定**で、分子pipeline全体の速度比ではない。

固定task集合を構造work単位で模擬すると、128 worker条件の最終終了workは探索で約14.62%、確認で約0.73%減った。8/32/128は模擬比較の条件であり、本計算worker数の固定上限ではない。構造proxyが実時間に比例する保証はなく、実測短縮率や完了時刻の見積りには使用しない。

実分子の回路cache命中率、compile時間、実throughput、メモリpeak、I/O競合、全体の短縮率は**未測定/null**。将来の許可済み取得では既存trial時間に加え、`*_performance.json` とprivateのresource JSONLに同時投入数・自分のCPU・compose/schedule cache件数を記録する。実データでの性能をこの人工検査から断定しない。

## 保存先と停止

private実行先は `.server-preparation/h46-seven-cost-speedup-20261027/run01/`、公開結果はこのディレクトリの `results/`。両方とも未作成。元準備・元研究の結果領域は操作しない。

再検査は `python3 -B analysis/h46_seven_cost_readiness_20261027/verify_phase_a.py`。実行SOURCE・PLAN・標本数を再固定してcommitする。今回のユーザー指示は並列化・高速化の実装であり、分子sampling/build/compileは開始していない。**Phase Bの別途実行許可までmandatory STOP。**
