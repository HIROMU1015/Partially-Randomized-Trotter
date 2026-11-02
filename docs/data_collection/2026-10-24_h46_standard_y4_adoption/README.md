# 3環境索引へのH4/H6標準Yoshida4採用差分

[今回の採用版paper manifest](../../../artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/paper_manifest.json)を読む。
[元3環境索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dafed86cc82149d55b4e6395382953d5f3631d7/docs/data_collection/2026-10-23_three_environment_integration/README.md)とf75採用packetは固定する。

[差分](../../../artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/three_environment_overlay.json)は、
旧29 claim・49系列を保持し、新しいH4/H6の2 claimを今回のCEMの同一IDで参照する。
同じ原典・cohortを索引間で再計数しない。既存の再構成PR対PR比較ではS2 componentのみを再利用した。
旧m5_best確認、旧H6採用、探索と標準Y4独立群は別scopeである。

固定条件・shot政策下の独立PR費用方向のみ正式採用。近似区間・母平均未保証・同一q逆転なしを保持する。
主表/Figure2の差替え、外部環境再現、論文中心主張の最終採択ではない。新科学call0、mandatory STOP。
