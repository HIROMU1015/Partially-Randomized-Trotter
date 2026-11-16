# 計算用ローカルQ3 worker：準備と未接続項目

2026-11-17。公開branchは `codex/rpe-distributed-q3-local`。
固定基点 `5829a7288d28395d2b3e89bfb676c23c690b9ddb` の子孫である共通仕様
`b9a39567a51bd27ca89e751d4f42e170af7e6c71` を基点にした。
[共通HANDOFF](../../HANDOFF.md)を直接確認した。このSHAには共通worker・relay・分散検収runnerはまだ無い。

**状態：Linux上の準備・人工検査まで。Windows実機準備と実3台接続検収は未完了。**
このセッションはnative Ubuntuで、Windows PCやローカル0へのアクセスは確認できない。
`qg1`・`qg2` はこの環境では未定義で、`ssh qg2` は名前解決で失敗した。
ローカル0のalias・認証設定を計算用ローカルへ引き継いだ扱いにはしていない。

## 今すぐ使える準備コマンド

Windows PowerShellで `wsl.exe --list --verbose` を実行し、既存のVERSION 2のdistroを確認する。
このセッションからWindowsのWSL一覧は取得できていない。以下はそのWSL/Linux内で実行する。
新規WSLの導入、鍵のコピー、SSH設定の変更は実施していない。

```bash
# 専用checkout内。Q3_PYTHONはその端末で確認したPythonへの絶対path。
Q3_PYTHON=/absolute/path/to/python
"$Q3_PYTHON" -B scripts/prepare_rpe_q3_local.py \
  --state-dir .local-q3 --slots 1 --dp-workers 2 \
  --interpreter "$Q3_PYTHON" \
  --smoke-output .local-q3/checks/first-smoke
cat .local-q3/READINESS.json
cat .local-q3/checks/first-smoke/CHECKS.json
```

`--slots` は同時Q3数、`--dp-workers` は1件内のDP worker数。例の1/2は変更できる。
既存 `ConcurrentExecutor` と人工fixtureを再利用する。smokeは人工2方式各2件だけを
単一slot/DP=1の参照と比較する。実H4のproblem/catalogは読まず、新規H4科学callは0。
再検査する場合は `--smoke-output` に別の新directoryを指定する。
設定だけを変更する場合は同オプションを外す。実workerへの設定適用は下記の公開待ち。

`LOCAL_SETTINGS.json` がローカル設定、`ARTIFICIAL_REQUEST.json` が既存adapter用人工入力。
`worker.provisional.json` は共通仕様の予定profileで、稼働確認済みprofileではない。
共通仕様にはまだDP worker上書きのprofileキーが無いため、その設定を黙って渡さず
`LOCAL_SETTINGS.json` に保持する。共通実装SHAで有効な指定方法に対応させてから接続する。

## 実測した配置と検査

このUbuntuの専用checkout:
`/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-distributed-q3-local`。
Python: `/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python`
（3.11.0rc1、既存環境を読取利用）。詳細は [READINESS.md](READINESS.md)。

設定・spool予定場所・cacheはcheckout内 `.local-q3/` 以下（Git対象外）。
`route_views.json` は固定commit→ `.local-q3/views/<full SHA>` の絶対path対応。
`--route-views .local-q3/route_views.json` を準備コマンドに追加するとHEADとtracked変更を検査する。
現在のroute viewは共有objectの専用cloneなので、元repositoryを残すこと。
別PCにはGit objectsを含むcloneと各固定SHAのdetached viewを用意し、path対応を作り直す。

[人工検査](checks/CONFIGURED_SMOKE.json)はslots=2/DP=2で同時2件を観測し、参照と
MSE上下界・判定・方式別採用順・停止理由が一致した。同一Linux上の検査である。
既存DP cacheの10テストもPASS。実workerの転送・再接続は検査していない。

## 共通公開後の普段の4操作

現仕様には参加コマンド案だけがあり、実装もstatus/drain/reconnectのCLIも未公開である。
したがって現時点では4操作を稼働可能なコマンドとして提供できない。

| 操作 | 公開後に確定する内容 |
|---|---|
| 参加 | サーバー2担当の固定実装SHAで `python -B scripts/run_rpe_q3_worker.py --profile <local profile> --controller http://127.0.0.1:18763`（共通仕様案、現SHAにはファイルなし） |
| 状態確認 | 共通workerの正式CLIと保存job状態・未転送結果の表示。コマンド未公開 |
| 新規受付を止めて実行中を完了 | 共通workerのdrain操作。コマンド未公開。トンネル切断をdrainと同一視しない |
| 再接続 | 同じhost ID・spool・cacheで通信を再開し、同じjob/attemptを照合。正式コマンド未公開 |

計算用ローカル自身の既存SSH経路が確認できた場合のトンネル案:

```bash
# <確認済みalias> は計算用ローカル自身で接続確認したもの。
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 -L 127.0.0.1:18763:127.0.0.1:8763 <確認済みserver2-alias>
```

計算用ローカル→サーバー2が無くローカル0へのSSH経路がある場合は、ローカル0側の
サーバー2転送先loopbackへローカル起点で転送する経路を確認する。両経路とも現在未確認。
サーバー1→2の直接接続には依存しない。候補選択・予算・結果採用はサーバー2だけが所有する。

## 次に担当者が行うこと

1. 利用者：計算用Windows/WSLへの既存アクセス先、またはその端末→サーバー2/ローカル0の既存経路を指定する。認証情報を成果物へ記載しない。
2. サーバー2担当：共通worker・ローカル0 relay・固定SHAと4操作のCLIを公開し、最初の2台検収を行う（ローカル0中継→両serverで別の人工Q3→server2が両結果採用→次候補をserver1へ配送）。
3. Codex（ローカル担当）：そのSHAでprofile/DP設定を接続し、新しい人工専用runに途中参加。参照とのMSE上下界/判定一致、切断中の他job続行、復帰時の未転送結果回収/二重採用0、適合jobでのDP cache HITを確認し、この4操作表を実コマンドへ更新する。既許可の人工検収範囲で行い、新規H4は起動しない。

spool・cache・未転送結果は復帰まで保持し、接続断を根拠にjobを再実行しない。
