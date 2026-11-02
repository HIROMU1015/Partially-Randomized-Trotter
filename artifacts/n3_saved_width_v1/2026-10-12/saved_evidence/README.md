# 保存証拠の閲覧用index

元payload e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c のJSONを、丸めずに参照可能な小さいviewへ分割した。
新しい保証幅・採否・PSDをここで再計算していない。元の科学result/STOPは不変。

各baseの2方向は次のindexから辿る。indexのprovenance pagesには、原ファイルpath・JSON pointer・value_refがある。

- [C1 edge pair](cases/C1_edge_pair/index.json) / [distributed three](cases/C1_distributed_three/index.json)
- [R1 edge pair](cases/R1_edge_pair/index.json) / [distributed three](cases/R1_distributed_three/index.json)
- [L1 edge pair](cases/L1_edge_pair/index.json) / [distributed three](cases/L1_distributed_three/index.json)
- [S1 edge pair](cases/S1_edge_pair/index.json) / [distributed three](cases/S1_distributed_three/index.json)

各caseの `c0_NORM_0.json`、`c0_FIVE_0.json`、`c0_OLD_DIRECT_0.json`、`c0_OCC_0.json` 等が直接閲覧用preview。
他の二行も同じ命名。previewは元のκ/μ/τ、保存lower、証拠のvalid/accepted状態、selected Zへの参照を示す。
全spectral_attempts、proofのL/pivots、継承・fallbackはfull_method_refやprovenance value_refに丸めず残る。
proofはmatrix_sha256と原sourceを参照し、元行列を新しく作り直して保存する操作はしていない。

value_refは元の値のcanonical JSON SHA256。[objects/index.json](objects/index.json)の分割indexから値の格納pageを探す。
literal nodeはvalueそのもの。object/array nodeは子value_ref、paged_arrayは分割した子ref配列を持つ。
`Store.get()`が元のJSON値をlosslessに復元し、SHAと原JSON pointerを照合する。
各JSONは8192 bytes以下。小ファイル単位の取得に対応し、元の大きいledgerを再公開しない。

[catalog](catalog.json)はcoverageとinventoryへの入口。[比較仕様](../../../../docs/research/n3_saved_width_v1_comparison_specification_20261012.md)。
