# 2026-10-15：Parent allocation v3の保存処理修正

v2は総CPU3600秒で停止。利用者「次の作業に進んで大丈夫　修正が終わったらその先もできるなら行って」に従い、科学条件/capsを保つParent専用16 KiB bufferを準備。
新15＋既存78＝93 tests PASS。人工48 record/200 fileで旧新bytes一致、旧8.97s/新0.013s。これは本番速度の保証ではない。
旧共通writer/pipeline/guards/予算/seed/精度/比較条件は不変。v1/v2結果/marker/STOP保護。
新sourceとauthorization-only直接子を固定・pushしてから、同じ条件でv3を一回実行する承認を今回の指示に記録する。
[技術準備](../../tracks/algorithm_codesign/parent_allocation_v3_buffered_source_preparation_20261015.md)。実行後はmandatory STOPし、科学的採否判断を自動開始しない。
