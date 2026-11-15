# Full-H κ2 D7の全較正受入と4予測固定

2026-11-15 JST。D1 e628c0d6の全5 cell・10較正record/18軸を受入、保留0。G36223は完成f98cb963の4軸だけを使い、producerとvalidatorを区別。P2R566の親依存とindex0のみの数値gateを維持した。

SOURCE f46fa120bでX/Y別に各5行3係数・等cell重みのexact最小二乗fit、m5/m6×X/Yの4予測を一回計算しb5165f11bへ固定した。rank3、CALIBRATED_UNVALIDATED/qualified=false。人工5＋回帰3PASS、公開済みJSONを元D1 metadata gateで読取確認済み。新sampling/compile/holdout開封/E[C]/CI/N/G0。

[REPORT](../full_h_d7_model_freeze_20261115/REPORT.md)／[D1 HANDOFF](../full_h_d7_model_freeze_20261115/HANDOFF_D1.md)。次は別指示のD1独立holdout6compile。本体ロードマップは統合担当、公開照合後mandatory STOP。
