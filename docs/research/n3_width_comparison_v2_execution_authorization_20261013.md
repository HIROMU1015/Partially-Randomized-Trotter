# N3幅比較v2：固定pilot一回の実行認可

2026-10-13 JST。利用者の「次の作業に進んで大丈夫」を、直前の固定6入力pilotの明示実行認可待ちに対する承認として受領した。
完了済み科学レビューの比較条件を変更せず、[固定plan](n3_width_comparison_v2_execution_plan_20261012.json)を一回実行する。

## 結合対象と範囲

- source commit：`3eba0814e690e95993db60809c4c3c96784b07c0`
- source fingerprint：`9f7687e74fe93f4b238579ee70b846521f4ea296195491c9f22e536a60066d96`
- input SHA：`4c59ba790c7a1faefa6e60dbfbde3d13b718e168bb6491fa71594103b3ac8395`
- plan SHA：`dc43a8d781e76ba4c3fc92f528c973f225861bf47160c5866bff2bebd9eb9d53`
- authorization SHA：`1ea7cc662478320db5429ced1481711b3be8863ab72df86774ef0d34fc581c96`
- execution branch：`n3-width-comparison-v2-execution-20261012`
- output：`artifacts/n3_width_comparison_v2/2026-10-12/run1`

[認可JSON](../../artifacts/n3_width_comparison_v2/2026-10-12/execution_authorization_v1/authorization.json)、
[利用者指示の記録](../../artifacts/n3_width_comparison_v2/2026-10-12/execution_authorization_v1/user_instruction_intake.json)、
[resource配置](../../artifacts/n3_width_comparison_v2/2026-10-12/execution_authorization_v1/resource_configuration.json)、
[数値適用なしのpreflight](../../artifacts/n3_width_comparison_v2/2026-10-12/execution_authorization_v1/preflight.json)。
準備時のfalse templateとSTOPは履歴として保存し、書き換えない。

3stress family×full5/6の6H、ν2/3、候補2mode/H、非負density、一外部mode、ε_proj=.01。
7method×kernel/cold×3 timing repetitions、計504 mode-method records、managed SCS calls最大432。
2spawn workers/thread1、worker AS3GiB、aggregate sampled RSS6GiB、開始前available8GiB、output128MiB。
固定のcall/iteration/bit/log capを適用し、wall/CPU capは設けない。

疎CSC、普遍CAR template、exact matrix SHA cacheを凍結実装のまま使用する。
source13 blob・環境・入力schema/settingsはpreflight PASS。準備時の合成32件と実parallel/serial証拠照合を参照し、科学比較sourceを変更しない。
cap超過や実装・soundness failureも一次結果として保存する。attempt1/retry0。再実行・追加入力・追加scanは許可しない。

結果取得後に独立保存監査、全結果・失敗・費用・証拠の整理、GitHub公開とremote再取得確認まで行い、Mandatory STOPへ戻す。
新結果の研究価値、中心テーマ、新規性、次の計算・拡張はGPTの科学的判断事項であり、今回代行しない。
N1停止、N2保留、原primary、旧2-base gate、中心テーマnullを維持する。既存global manifest description欠落は別問題として残す。
