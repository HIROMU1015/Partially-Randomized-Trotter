# WSLのruntime guard不足3セルだけの実入力dry-check

既存3ホスト12セルとWSL PR-DFの実機PASSは再検査しない。
WSLの旧Full PF・B0I・PR-Pauli記録はPython3.12でのmetadata検査のみだったため、
実行時と同じ固定AST guardを、既に配置済みの専用Python3.11.0rc1で確認する。
SOURCE・REQUEST・入力・guardを変更せず、PR-Pauliは公開33e routeを使う。

[PowerShell実行コマンド](RUN_WSL_RUNTIME_DELTA_V1.ps1)と[bundle hash](BUNDLE.json)。
Windows標準Pythonと既存Ubuntu WSLを使い、SSHは結果返送のqg2だけ。
WSLで新しい専用directoryにchecker・profileを配送し、明示的なdry-checkを3件実行する。
PROFILE.current.json・既存spool・cache・SOURCE・参加状態を書き換えない。
既存PARTICIPATION.jsonが受付停止でなければ処理を拒否する。新worker・relayは起動しない。
科学Q3・signal・価格較正・回路compile・本番queueは0件。

profileは中央optional_wsl/PROFILE.jsonと一致する。各方式を別結果directoryへ保存するため、
HOST_FACTSのroute/runtimeが混在しない。返送先はoptional_wsl/evidence/runtime_delta_v1。
返送済み記録を再転送する場合は `--return-only results` を使い、dry-checkを再実行しない。
中央の受領記録はhashを照合し、異なる既存結果を上書きしない。

synthetic unit 4件PASS（WSL/SSHはmock）。この環境からWSLへ実接続した検証ではない。
