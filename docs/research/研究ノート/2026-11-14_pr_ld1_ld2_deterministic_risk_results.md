# 2026-11-14：保存PR-LD1/LD2の決定論的riskを各1件適用

利用者が5e8f1664dのtarget proxy・raw診断scope・主値loss上界・exact-affine確率・ideal有限K IID・共通vector・ρ=1・全count/隣接検査範囲を採択し、各1件・計2件の科学実行を明示許可した。caller/INPUTSを43898ada1へ固定して既存D.evaluateを2workerで各1回使用し、全体約7.165秒で完了した。旧SOURCE/raw/STOPとsrc/testsは変更していない。

両方で有限good閉包検査が成立し、モデルMSE上界はLD1約0.2398951760583、LD2約0.2396879588138 Ha²。未解決確率は約[9.998e−8,2.127e−4]／[9.969e−8,2.125e−4]。reference-oracle tieは各round exact0だが、実全体tieは前状態依存で厳密値UNKNOWN、範囲だけを報告した。保存exact質量/MSE恒等式・SOURCE/原資格/null保持の確認PASS。

これは採択したモデル・loss familyに対する保守的上界で、実RMSEの点推定、精度達成/未達成、十分N、LDの優劣ではない。rawに含まれ得る近似影響へAδEを二重加算せず、元Minkowskiと物理/数値/lawの未資格を保持した。[REPORT](../pr_ld1_ld2_deterministic_risk_results_20261113/REPORT.md)／[HANDOFF](../pr_ld1_ld2_deterministic_risk_results_20261113/HANDOFF.md)。追加配分/MC/価格/G/最適化0、commit/push・remote照合後mandatory STOP。
