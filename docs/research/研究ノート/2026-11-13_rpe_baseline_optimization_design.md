# RPE全体での比較ベースライン設計（2026-11-13）

Full PF(S2/Y4/M5)、B0/B0I切断PF、one-body込みFull-H pure RTEを全RPEのΣN E[C]で探索する計画。旧11案はseed、局所screen/構造proxyと根拠付きrisk/cost除外を区別。既存APIを再利用し新framework0、人工回帰20 PASS。実H4計算/保存再解析/候補最適化/compile/N/G0、正式選定なし、公開照合後STOP。

Track Aの有限時間最良はRPE最良ではない。Full PFは公式/幅、truncatedはp/B0/B0I/公式/幅、pure RTEはK/s/q-R groupingを区別する。同full-H target・state・RPE loss・full-wrapper価格/loweringの比較とし、p7や旧layout係数448を最適条件とみなさない。PFの数値評価は既存direct-expmを使う方針で追加検証を目的化しない。既存C branchを継続し、B/A/Dの実装・作業領域を変更しない。

[REPORT](../rpe_baseline_optimization_design_20261113/REPORT.md)／[計画](../rpe_baseline_optimization_design_20261113/BASELINE_OPTIMIZATION_PLAN.md)／[HANDOFF](../rpe_baseline_optimization_design_20261113/FAIR_COMPARISON_HANDOFF.md)。次の科学処理は明示承認待ち。
