# H8 bias evidence audit：commit固定資料

[資料入口](docs/research/track_a_h8_bias_evidence_audit_20261017.md)／
[監査結果・未承認計画・source closure](artifacts/resource_applicability/track_a_h8_bias_evidence_audit/2026-10-17/README.md)。

保存証拠のschema・来歴・bytes監査と人工30 testsだけを実施した。
実H8の新数値再解析・非相殺費用・ランキング・signal/compile/sampling/fitは0。
SOURCE：`efc7e00f5032d3003347e843b4929eb57b7ed47e`。SOURCEと資料commitを分け、pushは利用者が実施する。

元の未登録snapshotとprivate raw・logs・反映patchはhome配下にそのまま保全した。
この資料branchにはprivate raw科学値、NPZ、実runtime/checkpoint/ledger、認証情報を追加していない。
B2はSTOP/cleanup記録と元PID不在を読取確認した。B2の成功や次stage実行を認めるものではない。

既定CLIはscope表示のみ。

```bash
python3 -P -B scripts/audit_h8_bias_evidence.py
```

診断prototypeは `fixture_kind=ARTIFICIAL` を要求する。実H8用runnerは未実装。
`approved=false`、`runtime_authorization=false` を維持する。
次の作業：ユーザーが案内されたpush commandを実行する。新しい数値解析は開始せずmandatory STOP。
