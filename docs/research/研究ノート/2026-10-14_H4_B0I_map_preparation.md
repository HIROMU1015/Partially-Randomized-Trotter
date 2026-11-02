# 2026-10-14 H4 B0+I全面評価の準備

利用者の次作業への進行指示で、先行レビューの標準domain（6距離×20構成×2軸）を対象とする閉じた実装・契約を完成した。旧source48件とprofileを保持し、既存6実測の新namespace/sourceへの再利用はcomplete proof＋再build numeric FP一致を必須にした。全240のうち234が新compile、mismatch時の自動fallback・失敗返却・次stageなし。

大きなplanを旧256KiB bootstrapへpickle送信できないため、owned grant file paths＋digestだけを送る形式にした。旧256 inodeでは474 task/reservation filesを収容できないため、新1024 inodeと160MiB observer log、stage4hを具体的な変更案とした。総256MiB/charge17GiB/actual74805を維持する。

19限定stdlib/mock検査、source Git blob/byte/profile/input metadata照合に合格。既存科学レビュー・44人工tests・旧benchmarkは再実施していない。新実行条件に対するgrantは未承認として保持し、本計算・worker・affinity・共有環境/H8変更0。

[実行案とbinding入口](../track_a_h4_b0i_map_preparation_20261014.md)。
