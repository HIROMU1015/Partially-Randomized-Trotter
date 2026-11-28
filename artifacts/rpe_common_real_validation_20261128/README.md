# 共通RPEの実機検証記録（2026-11-28）

総括と検収条件は [レビュー対応記録](../../docs/research/rpe_common_review_fixes_20261128.md)。

- R10: `R10_REAL.json`、RomuTail上の実ファイル計測。
- R18: `R18_HISTORICAL_BASELINE.json`、旧PBS rolloverの観測値。新版の実PBS検収待ち。
- R36: `R36_QUEUE_CORRECTED.log`、実queue読み取り比較。 `R36_LIVE_CPU.json`、既存controllerのCPU標本。 `R36_CONTROLLER_PROFILE_COMPARISON.json`、隔離通常controllerと模擬workerのGIL/CPU比較。 `R36_CONTROLLER_PROFILES/`、4回の生profile・検証ログ。既存科学controllerの直接GIL attachはOSの権限で不可。
- R43: `R43_COMPARISON.json`、実固定DP kernelの出力・cache・PID一致。 `kernel-*/`、初回fallbackと修正後の各結果、入力identity、timing、events。 `R43_CACHE_HASHES.json`、ローカルに保全した生成cacheのhashとサイズ（圧縮・解凍後SHA-256）。既存のignore規則どおり生成cache自体は共有commitへ含めない。科学結果は各RESULT.jsonに保存。
- `fresh/`：通常Q3入口の固定SOURCE不足による技術preflight終了（科学実行なし）。 `persistent/REQUEST.json` は未実行の当該通常入口の比較候補であり、実行済みの直接kernel比較とは別。
- `R36_CONTROLLER_*_LOCATION.txt`、元のローカル一時作業場所（参照用の歴史記録）。共有用の比較証拠は `R36_CONTROLLER_PROFILES/` に保存済み。

この記録は実稼働版の切替、探索開始、PBS予約、旧runの科学結果採用を示さない。R8/R18は新規科学実行や実PBS操作なし。既存の固定SOURCE・raw資料・H8採択コンテキストは保全する。

共有用 `fresh/EXECUTION.log` はCRLFをLFへ正規化した技術ログ。元byte列はローカルの `fresh/EXECUTION.original.log` に保全（SHA-256 `d9b3d65c4fd9c71afacc43548301ff5b6dd19ca7c6b81801d613fa041fa57f43`）。科学結果の変更なし。
