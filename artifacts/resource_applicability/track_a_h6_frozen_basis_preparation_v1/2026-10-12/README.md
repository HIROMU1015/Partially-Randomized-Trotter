# H6 DF準備再現性修正・受領待ち

Branch: `track-a-h6-native-reproduction-fix-20261012`

SOURCE: `b53271f59609f9b633f1771a85c7ca7181fdba8a`
REVIEW: このREADMEを含む別commit（SOURCEを変更しない）。利用者push方針を維持、agent pushなし。

現在は **WAITING_FOR_OLD_ENVIRONMENT_PREPARATION_ARTIFACT**。本計算の再実行0、新grant0。
SOURCE240/source tests、旧科学224・server基盤233・環境profile/input/carryを照合。
`candidate_v2/manifest_candidate_v1.json` が現在のSOURCE binding。旧candidate_v1も保持する。
`execution_plan_sealed=false / approved=false / runtime_authorization=false / allowed_cpus=[] / absolute_launch_command=null`。

## 旧環境へ渡すファイル

[旧環境側Codexへの依頼・実行コマンド](OLD_ENVIRONMENT_CODEX_REQUEST.md)。
ZIP: `/home/AbeHiromu/projects/h6-handoff-evidence/20261012/native-reproduction-fix-v1/h6-old-environment-exporter-v1.zip`
bytes10992、SHA256 `74736b61df9828a8b58407bdc450fd84fc1bba8df2ac0550c48aaff14902091c`。
SOURCEのexporter/codec、依頼、file SHAだけを含む。NPZ/runtime/credential/SSH情報を含まない。

旧環境が使用可能という利用者回答を受領。既存venvを変えず、旧PUBLIC054bb7の独立worktreeから小さな準備値だけをexportする。
出力 `h6-frozen-preparations-v1.json` と `SHA256SUMS` を
`/home/AbeHiromu/projects/h6-handoff-evidence/20261012/native-reproduction-fix-v1/incoming/` へ配置する。
実行失敗時は `export_terminal.json` を保全し、原因を報告する。入力再生成・hash gate緩和で補わない。

## 修正と限定検証

[診断](diagnostic_summary_v1.json)、[source差分/hash](source_binding_audit_v1.json)、[tests](limited_tests_v1.json)。
旧450で停止した理由はPARTIAL_EVENT_REGENERATION_MISMATCH。基底hash・gate数・一部成分符号が異なる。
CPU kernel automatic Zen/HASWELL/SANDYBRIDGEの各7準備比較はいずれも旧hash不一致。各約4.7秒、RSS約409MiB。
CPU差だけが原因とは確定しない。元native binary inventoryは未受領、旧全出力bit同一性も未確認。
保存event2402 applicationには378 component・13 fragmentのbasisしかなく、完全な旧準備を単独復元できない。

新codecは7準備をfloat hex、typed/content-addressed JSON、元RZ/XXPlusYY gateで固定。pickle/eval/不明gate/非finiteを拒否する。
Local gate matrix SHA、旧prepared hashes、Hamiltonian/partition、components/確率/order/phaseを照合する。
B0は元select_blocksによるtruncated Hamiltonian hashを保つ。B1/B2は完全Hamiltonian hashを保つ。
Scoped adapterがown single-thread workerだけへ旧準備を供給し、終了/例外後に元port参照を復元する。
旧科学cost port・seed・trajectory・compiler・full-wrapper・量子費用定義は変更しない。
旧450 event digestは元portでも引き続き厳密照合する。

退出競合も修正。RSS欠測時に最大20msだけWEXITED|WNOWAITを再確認し、definitive退出の場合だけ通常回収へ進む。
生存中欠測はfail-closed、own unreaped identity保持、group cleanup後wait/reap、非退出error・最初のSTOPを維持。
5秒監視・PSI/OOM・resourcesは変えない。

69 local限定回帰PASS。人工2x2 DF/codec/seed、grant/receipt/carry、最大4×128MiB tiny-processの退出/cleanup。
Science NPZをtestsで拒否、signal/transpileなし。独立agent review/CI/実H6 successは未実施。
新native DF診断は3×7準備だけ。科学入力生成、signal、trajectory、transpileは0。

## 受領後の順序

1. 外部SHA/bytes・exporter/codec SOURCE SHA・source224/input/environmentを照合する。
2. 限定server verifierで7旧prepared hashesと450の元seed/eventsを照合する。
3. 完了済みserver447の両軸full-wrapper fingerprintを旧準備で再build/serializeし比較する。transpileしない。
   異なれば447の既存2費用rowは旧系列へ混ぜず、同じq8条件の再測定が必要とする。現在は未検証でsealしない。
4. 技術条件が揃った場合だけSOURCE/profile/input/archive/receipt/carry/outputを結合し再sealする。
5. fresh resource/output/one-shotと新grantを結合してから本計算へ接続する。旧grant/lockの再利用や無断retryはしない。

CPU20/21/24/25 worker4、coordinator22/23、内部thread1。AS32GiB/worker・sampledRSS9GiB、aggregate40GiB、RAM budget64GiBを維持する。
旧447はfeasibility完了証拠として保持。旧449 group・902 wrapper/92 signal/84 B3を保全、旧STOPとserver450–453の失敗証拠も保持。
現在のknown carry: `{'compile': 904, 'control_probe': 292, 'occurrence': 1024, 'primitive': 1470, 'reference_matvec': 400, 'trajectory': 434}`。不明な過去失敗分を0と扱わず、返却/resetしない。
Exportとserver一致診断は各trajectory1/occurrence8。成功receiptでknown carryへ計2/16を追加し、失敗時もterminalの予約・既知消費を保全する。
過去失敗の未記録counterと下限4 trajectory/32 occurrenceは旧stop receiptに保持する。累積上限は変更しない。

本計算・既存H4・venv・他job・共有設定・GPUへ変更なし。最新H4読取状態はRUNNING/590 signals/33800 records/first_stop=[]。

## 利用者がpushする場合

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-native-reproduction-fix-20261012 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h6-native-reproduction-fix-20261012
```
