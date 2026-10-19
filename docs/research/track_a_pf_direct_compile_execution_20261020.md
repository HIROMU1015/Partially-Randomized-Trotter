# PF直接compile：2契約の本番準備・実行記録

基点d510d7be。ユーザーは現行4系列722軸とH4 legacy DF12の66軸、計788軸の直接compileと契約内controlを明示承認した。旧profile等の再照合待ち48軸は対象外。登録候補・q・軸・T=.8/P=0/symmetric_directionalを維持する。

本番接続は[保存入力のseal](../../scripts/prepare_pf_direct_execution.py)、[pair worker](../../scripts/pf_direct_production_worker.py)、[独立契約runner](../../scripts/run_pf_direct_execution.py)、[保存結果監査](../../scripts/audit_pf_direct_execution.py)で行う。旧providerのnative構築・full wrapper・fingerprint・compiler・preparation復元を再利用する。H2/H4は元と同じ保存配列から元native preparationを復元し、H6/H8は保存済みfull preparationのprefix viewを復元する。SCF/DF生成・状態solverは呼ばない。

試作adapterのmath.infは元providerの整数型契約に適合しないため、今回の新workerはproviderのblock_instruction_boundから候補固有の構造上界を整数算出する。wrapperは実instruction数+4。恣意的な回路規模上限やq除外ではない。旧adapter/SOURCEを変更しない。

各pairで保存input load・preparation restore・native evolution各1、control4、測定wrapper build/compile各2。controlには保存corrected、存在する場合は保存independent_forwardを結合し、branchとX/Yの4probeを検証する。新しいreference/signal計算は0。測定RZは状態準備を除くfull Hadamard wrapperの整数ゲート数で、shot重み付けを行わない。

2契約は別grant・output・ledger・one-shot marker・STOPを持つ。親は起動前に非返却予約し、workerもAPI前に予約する。task異常は当該taskのSTOPとして保持し、自動retryせず他の正常taskを進める。親はowned workerだけをwait4で回収し、exit0・result hash・task ID・整数metrics・compiler/fingerprintを照合した結果を受け入れる。

資源投入は許可CPU、祖先cgroupのquota/空きmemory、FSを毎回観測する。実測H8 p21/q8/S4のpeak RSS26.252 GiBを構造site数でscaleした見積りをadmissionに使い、実際のlive RSSとの差を予約する。見積りはprocess上限ではなく、新しいゲート費用modelやfitではない。CPU/RAM/workerの固定上限なし。1 worker 1 threadは候補間並列との過剰な重複を避ける配置であり、worker数は実効資源で決める。一時的PSIや無関係worktree更新では正常workerを終了しない。

[技術検査](../../artifacts/resource_applicability/track_a_pf_direct_compile_execution_preparation/2026-10-20/technical_checks.json)：新13・既存34の人工検査PASS。別AIの独立レビューではない。既存入力/provider/runtimeの初期35,547ファイルのSHA一致。科学実行前に最終SOURCEをGit commitへ固定し、各契約の入力・runtime・profileを再検証する。この時点では本番compile未開始。

実行完了またはSTOPの結果は後続の保存receiptを引用して追記する。新規sampling/fit/shot再計算/費用再ランキング、詳細科学レビュー、最終evidence freezeは行わない。公開承認は旧d510d7beの22ファイルに限定されるため、今回の新しいSOURCE/結果は追加承認までpushしない。
