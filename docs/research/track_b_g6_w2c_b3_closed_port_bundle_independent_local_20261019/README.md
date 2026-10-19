# Global Bundle × API Port：ローカル独立技術受入

**人工End-to-End統合を限定技術受入。実科学GOではない。Mandatory STOP。**

直接基点 `5e862a70f2d74defe72dd39205d38c2a4f8ebeaf`。統合公開HEAD `62967269e611b61e73b70abe20e998d9c7eaaf50`、人工測定SOURCE `deade78a19f5b23859d16816d3a7bfb37765dd80`。実施2026-10-19〜20 JST。ARTIFICIAL_ONLY・未採択・実科学入口閉鎖。

固定Portがversioned Bridge→scheduler→spawn worker→pair/F01→global state/Storage V2→receipt/callerへ到達することを、SOURCEと保存rawの双方から独立に確認した。前回の「port単体＋Bundle単体」と証拠範囲が異なる。

- 公開53ケース：Bundle正常3、condition-only2、技術失敗22、証拠不足26。
- 公開45 mutation：不整合42、証拠不足3。Port固有10件を全て拒否。
- 追加3 negative：不整合2、証拠不足1。
- 独立tests：既存975＋新63＝**1,038 PASS**。Producer134は別集計。
- 新規偽dispatch14件は既存保存検証を通過するが、新統合checkerで全て拒否。
- 固定資料295件のGit/公開bytes/SHA256一致。10 parts/39,731,962 bytes/2,706 memberを全件検証。
- 公開正常5ケース計21,504 trialを照合。別scopeで人工n16/n256/tag差分3ケースも新規実行し、独立受入。

[Q1–Q10](handoff.md)、[SOURCE](independent_source_verification.md)、[構造](integration_architecture_acceptance.md)、[dispatch](port_dispatch_independent_verification.md)、[会計](global_cost_and_byte_accounting.md)、[回帰](artificial_regression_results.md)、[未決](pending_and_required_decisions.md)。

## 再現

repository rootで `<packet>` をこのディレクトリに置き換える。出力は全て未使用pathを指定する。

```bash
/usr/bin/python3 <packet>/restore_artificial_raw.py --new-directory /tmp/port-bundle-raw-NEW
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 <packet>/run_offline_acceptance.py --raw-root /tmp/port-bundle-raw-NEW --new-output /tmp/port-bundle-offline-NEW
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 <packet>/run_independent_tests.py --raw-root /tmp/port-bundle-raw-NEW --new-output /tmp/port-bundle-tests-NEW
/usr/bin/python3 <packet>/restore_local_artificial_raw.py --new-directory /tmp/port-bundle-local-raw-NEW --public-raw /tmp/port-bundle-raw-NEW
```

新規人工rawと18差分はlocal archiveに保全。閉鎖gateの補助試験は `prepare_artificial_runtime.py --new-runtime /tmp/port-bundle-runtime-NEW` の後、別processの `run_closed_controls.py --runtime /tmp/port-bundle-runtime-NEW --new-output /tmp/port-bundle-controls-NEW` で再現できる。通常offline checkerはProducer/Bridge/scheduler/Admission/self-checkerをimportしない。

次の担当はユーザー/GPT。実科学API接続準備と正式契約の採否・次scopeを判断する。Codexは追加指示までSTOP。
