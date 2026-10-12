# N3 OCC v1 固定8入力：一回実行の認可

2026-10-12 JST。利用者の最新指示「次の作業に進んで」を、直前に具体化した固定8入力batch一回、一次seal後の条件付きsecondary、保存監査、GitHub公開・再取得、Mandatory STOPへの実行認可として記録する。
認可の範囲は[authorization JSON](n3_occupation_transfer_v1_execution_authorization_20261012.json)に固定する。過去の準備文書とplanの未認可状態は当時の記録として保持し、書き換えない。

source `83f51cd8a4848572c66a5535c67d0d0a5f3868cd`、plan SHA `a825ece44d9bf8266a9c83d390572139d3799b748419b644515d79230e5c0d0f`、入力SHA `c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4`を維持する。
専用execution branchは公開準備HEAD `e26bcf927ff4b26f39e7294ac3e253b52adb51de`から作成した。凍結source8ファイル、係数、精度、探索予算、比較・分類条件を変更しない。

実対称one-body＋非負unordered density、一外部mode、固定ν。active m=6/7/8/9、ν=2/3/3/4、full modes=7/8/9/10、4base各2方向。δ=.004/.006/.003/.005。
geometry・化学basis・DF rank・L_D・PF/time windowは適用なし。many-body truth、量子compile、samplingは0。
2 spawn workers、全numeric thread1、worker address space4GiB、sampled aggregate RSS8GiB。JSON shard32MiB、総output64MiB、log8MiB、bit cap4096。wall/CPU time capなし。

[実行前schema確認](../../artifacts/n3_occupation_transfer_v1/2026-10-12/execution_preparation/source_bound_preflight.json)は1692 checks、実predicate0。preflight内のauthorized=falseはread-only modeの意味であり、別authorizationが今回の実行権限である。
[resource確認](../../artifacts/n3_occupation_transfer_v1/2026-10-12/execution_preparation/resource_configuration.json)。既存のsynthetic検査と最適化経路を維持し、入力を事前評価して調整しない。

実行前commitでclean worktreeへ固定して一回だけ起動する。失敗も保存し、retry0、追加scan0。実行後は結果・監査・公開receiptを保存してMandatory STOP。
N1追加探索停止、N2保留、中心テーマ未採択、実分子・多外部・一般二体拡張なし。root dirty worktreeと別Trackのsource/results/STOPは保持する。
