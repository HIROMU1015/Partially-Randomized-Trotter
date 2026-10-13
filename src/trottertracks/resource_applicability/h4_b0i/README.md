# H4 B0+I validation-only

`adapter.py` はtail-only constantのコピーと既存B0 builderの利用。
`contract.py` は閉じた6-task予算・新approval・fresh admission・own cleanup。
`runtime.py` とfuture CLIはsource/profile/新grantのgateを通るまで科学処理を行わない。
元H4 sourceは変更せず、method/template/output/namespace/FPを分離する。

[準備資料](../../../../docs/research/track_a_h4_b0i_compile_preparation_20261013.md)と
[未承認bundle](../../../../artifacts/resource_applicability/track_a_h4_b0i_compile_preparation/2026-10-13/README.md)。
人工検査は `tests/tracks/resource_applicability/test_h4_b0i_preparation.py`、
専用scriptは `scripts/resource_applicability/run_h4_b0i_preparation_tests.py`。
現時点でB0+Iの正式採用・実H4 compile・実行許可はない。
