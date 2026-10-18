# Mechanism runner SOURCE readiness

2026-10-19 JST。**READY_FOR_USER_AUTHORIZATION**。実科学データへの数値適用は0。

実装SOURCE：`5c256067db29a326bcaa0a0e244b38f6b87bd138`。先行実装d3cd24fbc52cfc85eb41c9db9f0e15b036787932を保持し、実行後のruntime/grant/ledger/marker受入を補強した新commitへ固定した。SOURCEはローカルcommitであり、公開pushしていない。

|file|役割|
|---|---|
|src/trotterlib/mechanism_analysis.py|exact core、cohort/pairing、欠測、normalization/headroom説明|
|scripts/run_mechanism_analysis.py|承認gate、SOURCE/input/runtime byte照合、one-shot、native worker、出力・STOP|
|scripts/prepare_mechanism_analysis.py|標準libraryによるmetadata/byte準備専用。analysis moduleをimportしない|
|tests/test_mechanism_analysis.py|実科学入力を使わない64人工／mock検査|

productionのruntime SOURCE closureは実際に実行するcoreとlauncherの2file。metadata preparerは同commitとそのfile SHAを別に固定する。test・無関係worktree・資料の通常更新をruntime停止条件へ含めない。

stdlibのみ。Python実行file、version、読込runtime葉66fileのbyteをprivate profileへ固定した。Qiskit/NumPy/SCF/DF/native quantum kernelはimportしない。元compile profileの来歴と、今回のscalar解析runtimeは別に記録する。

承認gateは無効grantをcontract/payload読込前に拒否。実行承認receiptと契約hash、32 target IDs、SOURCE・input・runtime・出力先が一致した後だけ数値permitを作る。独立one-shot rootとWORKER_CLAIMをexclusive createし、再入・再利用を拒否する。

将来の本実行では親launcherが1解析workerを起動し、native終了・wait/reap・結果SHAを受け入れる。小さい32比較のscalar処理のためworker増殖はしない。CPU/RAM/時間の強制capは追加せず、共有設定・既存venv・他jobを変更しない。

実科学値でG一致・比・説明分解がPASSしたとの意味ではない。詳細は[契約](mechanism_execution_contract_READY_FOR_APPROVAL.md)／[テスト](mechanism_test_report.md)。
