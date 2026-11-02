# W2c-B3 integrated v3 独立offline人工受入検証

**人工技術scope完了、科学的GOなし、mandatory STOP。** 正常12件は `VERIFIED_COMPLETE`、障害35件は `TECHNICAL_FAILURE`。障害の検出成功を正常完了と同じPASSにしていない。最終ローカル回帰は **540 PASS = 旧437 + 新103、FAIL/ERROR/SKIP各0**。producerの481件・性能60件を加算していない。

[Q1–Q10と次担当](handoff.md)、[machine verification](verification.json)、[47件の結果](published_case_verification.json)、[P01–P11 matrix](integration_acceptance_matrix.json)、[サーバー導入手順](server_usage.md)を参照。

## 固定基点とSOURCE

|用途|固定commit / 根拠|
|---|---|
|新worktreeの直接基点|`b9a34bed5aef601afb193a5e4702e8ad198d0a16`|
|producer公開HEAD|`7ae4bd0b951dae05380538cd5c22a3ead5328ab2`|
|最終人工測定SOURCE|`860fef4a3a8ebb5a46628f4e0364e5934b5af416`|
|共通運用指示|`7eff8a33bef4ee1f29698ab10c6da46988dc8db5` の固定文書、特に§8.4|

[125入力の公開bytes照合](fixed_public_bytes_verification.json)、[測定SOURCE21ファイルのGit object照合](measurement_source_verification.json)、[source/schema inventory](source_inventory.json)を保存した。測定SOURCEのbundleは専用 `/tmp` bare repositoryへ復元し、共有repositoryへmerge/importしていない。125入力の取得commit・原path・bytes/SHA256は [input inventory](published_input_inventory.json) に個別記録。

`formal_digest`、`driver_digest`、`runtime_manifest_sha256`、`source_pins_sha256` はそれぞれ保存contract/candidate/runtime/pinsのexact canonical document identityである。Git commit ID、pretty JSONの物理file SHA256と区別する。[runtime別inventory](runtime_manifest_inventory.json)を参照。実科学authorization credentialではない。

## 実装の入口

新moduleは `src/trottertracks/algorithm_codesign/g6_w2c_b3_v3_offline_{identity,adapter,cost,acceptance,cli}.py`。人工testは `tests/tracks/algorithm_codesign/test_g6_w2c_b3_v3_offline_verifier.py`。旧P1 V1 adapter、global core、pair/F01 verifier、cost partitionを維持して再利用した。固定v3 producer/Host/Admission/schedulerと科学SOURCEは変更・実行していない。旧版Host/Admissionの人工fixtureを用いる既存437回帰は実行した。

これは既存offline技術検証のversion追加であり、新しい科学検証経路・科学結果ではない。研究の総括・数値manifest・既存indexesの科学状態は変更していない。本packet内のsource inventory、再現runner、CLI、tests、raw、reportsを相互に索引した。

## 受入範囲と限界

[検査契約と観測境界](verification_contract.md)、[SOURCE/証拠不足と最小引継ぎ案](pending_evidence_and_server_request.md)、[費用会計](cost_accounting.md)、[回帰と修正記録](artificial_regression_results.md)を参照。

完全contractは17件（正常12＋障害5）のみ。障害30件のcontractは公開されていない。P07/P08/P11は全47件でpending。P08は全操作履歴なし、P07/P11はprivate approval/ledger/historyなし。最新stateやSTOP自己申告からatomic claimや全履歴を認定していない。親crashはunknownを維持し、総CPU/wallをnullにした。

## 再現

Repository root、Python 3.10以降、標準libraryのみ。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/run_artificial_tests.py
```

9 suiteを独立processで並列実行し、新しい専用 `/tmp` にログとmachine結果を保存する。環境・並列化・メモリ実測は [environment](environment.json) と [v3 streaming memory](v3_bounded_memory.json)。再現中も科学module importを監視する。

`build_case_reports.py` は47件を再読する。既存 `offline_reports` を上書きしないので、再現時はこのpacketを新しい専用出力rootへコピーして新規reports先を指定するか、CLIを未存在の外部outputへ実行する。

## 次段階

ユーザーとGPTがpending・SOURCE/estimand・正式実行scopeを判断する。サーバーCodexが担当する場合は、別途指示後に固定consumerを導入し、人工rawでsmoke検査する。今回サーバー接続、正式採択、科学sampling、実authorization/grant/claimの発行・消費はすべて0。自動移行しない。
