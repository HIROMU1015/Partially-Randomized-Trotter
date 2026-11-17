# ローカル移行側への引渡し

最初に [REPORT](REPORT.md) と [ACCEPTANCE](ACCEPTANCE.json) を確認する。価格統合SOURCEは `c4e90c9e5ffc7230583567500a99c4ec5c255f7b`。追加releaseは配置・登録済み、**未activate**。共有currentは別セッションの `romutail-d887abdca-r2` のまま保持した。科学実行は今回許可されていない。

## 移行担当Codexの次の作業

1. 本branchのコード差分を移行版へ統合する。主差分は `rpe_price_validation.py`、`rpe_local0_search.py`、`rpe_optimizer_concurrent.py`、controller配置/transfer/reconcile runner。既存PriceRules・worker SOURCE・入力・結果・STOPは保持する。Windows clientは `ee07a2841` / `e197348db` まで本branchにも含むが、別セッションの後続transport修正を優先して照合する。
2. 更新中のWindows clientのnull ExitCode問題を直し、下記Previewを既存入口から再検査する。今回の成功済みSync・旧追加版Previewと混同しない。
3. 複数のcurrent書込みを避け、移行作業の最後に統合済みreleaseを一度だけ採用する。既存runはそのCONTROLLER_CONFIG/RELEASE/固定入力へ戻る。未activateの追加版をそのまま使う場合の登録済みserviceは次のとおり。

```
/home/abe/Project/Partially Randomized Trotter/.rpe-central/
  SERVICE.json
  RELEASES.json
  releases/c4e90c9e5/
  deployment/price-validation-c4e90c9e5/SERVICE.json
  deployment/LAYOUT.price-validation-c4e90c9e5.json
  price-validation/models.sqlite3
  imports/price-validation-server2-20261118/models.sqlite3
```

必要な切替は、移行担当が他の配置変更を終えてから行う。これはコード配置の選択であり、新しい探索やcompileを開始するものではない。

```bash
PYTHONPATH=src python3 -B -m trottertracks.rpe_release_registry \
  --registry '/home/abe/Project/Partially Randomized Trotter/.rpe-central/RELEASES.json' \
  --release price-validation-c4e90c9e5 \
  --service '/home/abe/Project/Partially Randomized Trotter/.rpe-central/deployment/price-validation-c4e90c9e5/SERVICE.json' \
  --activate \
  --evidence '/home/abe/Project/Partially Randomized Trotter/.rpe-central/validation/price-local-staged-0bcc1c23c/READINESS.json'
```

## ローカル0の既存通常入口

`C:\Users\romuk\Desktop\code\project\Partially-Randomized-Trotter` の既存toolsを使う。今回、live clientの設定や共有currentは上書きしていない。

```powershell
# currentを変えず、最終追加版・compile予算0を確認する。
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview `
  -Release price-validation-c4e90c9e5 `
  -RequestFile C:\Users\romuk\rpe-price-validation-local-controller-20261118\PREVIEW.json

# 状態閲覧・既存結果の手動同期。新しいmonitorやrelayは追加しない。
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Status -RunId <既存run_id>
.\tools\local0-result-sync\Invoke-H4ResultSync.ps1 -Action Sync
```

次のcompileを含む探索指示を利用者が出した際、担当Codexがその固定scopeをRequestFileへ反映してから、同じ通常入口を使う。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start `
  -Release price-validation-c4e90c9e5 -RequestFile <次に許可されたscope.json>
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId <そのrun_id>
```

Start/Resumeの実科学呼出しは今回未実施。上記は次回指示時の操作方法であり、今回の実行指示ではない。既存の確認後即時同期・10分ごとのrun monitor・終了時最終同期は通常clientが担当する。

scopeの `preparation_budgets` で `signal` / `compile` / `audit` の非負整数または `UNLIMITED` を明示できる。例として `{"compile":1}` は既存queueのcompile-stage taskを1件許す数であり、wrapper数1を意味しない。標準PR価格検証taskは2経路×X/Yの4wrapperである。Q3予算を増やしてもcompile予算は自動では増えない。今回配置されたtemplateはsignal/compile/auditすべて0、Previewにもcompile 0を指定した。次回の利用者指示に合わせて科学scopeと既存取得contractの両方を確認する。

## workerと共有状態

- ローカル：`deployment/price-validation-c4e90c9e5/profiles/local.json`。worker本体は既存Q3 SOURCE28d、価格adapterは固定a18。local workerも同じqueueへ参加する。
- サーバー2：`/home/AbeHiromu/projects/rpe-price-validation-local-controller-20261118/PROFILE.json`。固定a18 transportと専用CASを配置・metadata検査済み。
- WSL：`/root/projects/Partially-Randomized-Trotter-local0-wsl/.price-validation-local-controller-20261118/` のtransport/PROFILE/views/CAS。固定a18、13ab Pauli helper、326fe DF helperを配置し、保存2要求PASS。受付設定は元 `.local0-wsl/SETTINGS.json` のOFFのまま。可用性と受付許可を分ける。
- サーバー1：disabled。復帰後に実hostname、固定worker/科学SOURCE、Python/compiler、profile、保存metadata入力と受付設定を照合する。停止中のhostを必須待ちにしない。

最終layoutは既存候補組立てSOURCE d887を保持する。controllerの新SHAを理由に、既存の科学組立てprofileを新条件へ置換しない。価格worker a18・Q3 28dも別に固定する。新releaseが使うremote relay設定とworkerごとのprofileはROUND.executor.distributed.relayに保存され、実run開始時のendpoint/lockはそのrunの `distributed/relay/` に置かれる。

storeの再取込は既存runnerを使う。原snapshotをread-onlyで読み、同じIDの進んだローカル状態を巻き戻さない。異なる凍結要求/所有者を同じ要求とみなさずHOLDする。既存jobがあれば、そのqueue・attempt・owner・部分成果を先に照合し、新namespaceやlease切れで再投入しない。今回の13モデルには価格job ownerがなかったが、UNKNOWN/部分成果の保持は人工検査した。

## 公開

固定結果producerは `187c1b088638d9d4cf0a298b56d64e8a91c59105`、C mapは `ca5905e79d7654bdfe42a679b7b5e74aef80b1be`。Cは通常 `sync_saved_rpe_publications.py` がSEARCH_PUBLICATIONを発見する。host別の手動結果表は不要。

ローカル0の実閲覧先 `C:\Users\romuk\OneDrive\ドキュメント\ChatGPT\研究\H4-search-map\INDEX.json` を確認済み。LD2 m5/q32/R33、独立2経路、X/Y平均68,264、+0.044612%と元receiptが保持されている。詳細は [LOCAL0_MAP_CHECK](LOCAL0_MAP_CHECK.json)。

最終的な実compile・新Q3・詳細価格検証は、利用者の次の対象・scope指定後に担当Codexが実施する。今回の技術PASSを科学的価格資格や勝者の確定と扱わず、現在のmandatory STOPを保つ。
