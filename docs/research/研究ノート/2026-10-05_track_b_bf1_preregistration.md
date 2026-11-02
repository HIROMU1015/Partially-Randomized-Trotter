# Track B BF-1 preparation — 2026-10-05

利用者はBF-0四文書のcommit `15465b0b856d80f9cfde495fd3d22434825f1cc8`をreviewし、
`PASS_FOR_BF1_PREREGISTRATION`とした。科学実行承認ではない。
研究Bの全面再設計を今繰り返すのでなく、一回の将来のBF-1の判別契約を閉じる方針を維持する。

五条件を[BF-1 preregistration](../../tracks/algorithm_codesign/bf1_preregistration_v1.md)へ具体化した。
primaryはF対全O/L/fixed finite bestの5%、Oは診断、Lがnovelty対照。
全branch・端点・arc・初期点・exact time representation、数値guard、case境界を結果前に指定し、
synthetic semanticsと共有sourceとの一致を限定testで検査する。

保存prefixとone-body込み4-generator countを実sourceから明記した。
保存済みstateを使うため新state solveは不要。GPU preloadを避けるB CPU import境界を用意し、
共通API・A artifact/status/runtimeを変更しない。

これらはB worktreeのuncommitted source-content preparationであり、科学結果、CI固定証拠ではない。
NPZ操作、分子生成、trajectory、circuit/compile、GPU query/use、全suiteは0。
別のsource commitとexecution review/authorizationを満たすまでは科学実行しない。
BF-1終了後は全結果でmandatory STOPし、研究BのRQ・新規性・着地点を全面再評価する。
