# 2026-11-13：R3高速decoderとS3統計API

Bbf515の対象統計blobはAが検証したB170047と同一だったため、研究系列の既存branchを継続。A7f1fの独立module/test/fixtureを無変更で取り込み、提案batch hookだけをsimulate/simulate_preparedへ適用した。元scalar/vectorized、binomial/count変換、loss/CIは変更しない。

scalar/vectorizedの完全tie差は両方保全し、B hookはscalar互換を使用。PCG64全count・最終状態、全trial dict・summaryの一致、保存S3 nominal/null・未資格gateを人工/metadataで確認。decoder34＋metadata26＝60技術test PASS。人工13round/512trialの3回中央値はscalar16.651秒→batch.02554秒（約652倍）。peak RSS352.03→355.36 MiBで、メモリ削減は主張しない。

[REPORT](../rpe_decoder_r3_integration_20261113/REPORT.md)／[API](../rpe_decoder_r3_integration_20261113/HANDOFF.md)／[性能原記録](../rpe_decoder_r3_integration_20261113/BENCHMARK.json)。実H4 S3・保存科学値再解析・N/C/G0。S3のtarget/loss/IID/数値不確かさ/統計精度/反復数は未採択、十分N/global bound/G null。ユーザーの有限科学scope別承認後にBが実行する。公開整合確認後mandatory STOP。
