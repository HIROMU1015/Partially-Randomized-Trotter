# 2026-10-14 H4 B0+I並列起動

ユーザーの高速化/起動指示により12workerとgroup再利用を実装。driver hard ASの継承による0 compile停止を修正し、旧証拠/carryを保持してrun02起動を確認した。same science/compiler、25限定mock PASS、共有job変更なし。

[起動記録](../track_a_h4_b0i_parallel_run02_20261014.md)。

## 2026-10-14 H4 B0+I全240回路 実測完了・費用地図STOP

12 workerで234新規compile＋6 numeric照合再利用、全240 COMPLETE、stage193.990秒。SOURCE924b779c2bc6f6e844d6283c355f91c73aed6445・closure59とimmutable launch artifactは不変。
実測B0+Iを加えた1812点の費用点最小はB0+I397/B2 1415、ties0。全240軸で旧B0との差はRZ/CX数・各depth・sizeとも0。
ε=.05では.70/.80/1.40 ÅでB0+IがB2点推定比25.97/10.75/3.91%低く、他3距離ではB2が低い。ε=.005では6距離すべてB2が低い。
保存shot×実compiled RZ、状態準備P=0のlocal事後解析。32 paired trajectoryの選抜・推定、希少event、母平均/同時保証、energy/RPE・実backend未評価を保持する。
worker native exit12件全0、observer意図的−15、owned14残存0、monitor最大interval1.01093秒、PSI0/OOM増分0。累積actual73263/74805、旧carry返却なし。
[結果報告](../track_a_h4_b0i_map_results_20261014.md)。公開projection/全候補再最小化/Fraction照合PASS。追加compile/NPZ decode/共有環境・venv・H8変更0、次stageなし。主要claim確定は結果reviewへ戻す。以下は過去の各時点の記録。

起動snapshotを保持したうえで完了結果を追記。ΔRZ=0感度を全120構成の実費用で置換できたため、397点を参考仮定からlocal実測費用による事後比較へ更新した。研究上の主要claimの最終reviewや新しいsamplingは実施しない。
