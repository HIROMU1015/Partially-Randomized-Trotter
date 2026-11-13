# ベースライン全RPE signalの有限実行案（2026-11-13）

[正本REPORT](../baseline_finite_signal_plan_20261113/REPORT.md)と[有限科学計画](../baseline_finite_signal_plan_20261113/FINITE_SCIENCE_PLAN.md)を追加した。既存C/B APIで3 canonical configuration・全39 physical roundをmetadata固定し、初期seedはFull PF S2/Δ.1/Full10、B0I p7/S2/Δ.1、Full-H K2/q=2^m/R448を維持した。方法内最適性を意味しない。

旧短時間Full PF/B0Iには現gate用power診断が揃わないため、新direct評価へ含める。旧Full PF m6の同candidateだけは保存診断fieldがあるため、元spectral SOURCE/backendを保持した受入再資格1件を提案した。PASSした場合のみ、Full PF残12・B0I13・Full-H13＝新38 meanを一括scopeで計画する。受入失敗時にfresh m6へ自動切替しない。旧screen/STOPを保全する。

今回はmetadata確認だけ。inputファイル読込、受入再資格、新signal/reference/scalar抽出、科学数値再解析、旧人工test再実施、N/価格/Gは行っていない。保存13 reference pairとprefix scalarは承認後の照合で使い、欠測なら停止する。

次はユーザーが受入1件＋PASS時新38件を承認するか判断する。Codex Cは承認後だけ同有限scopeを実行し、Bのrisk/N・Dの価格は別scopeで扱う。公開照合後mandatory STOP。
