# PR-2 V0：旧pilot入力の回収監査

日付：2026-09-28  
監査基点：`bf5d3b2405b2bec4f88dd0589b75dd737b13e052`  
監査種別：read-only、one-pass  
判定：`OLD_INPUT_UNRECOVERABLE_USE_SEPARATE_NEW_SERIES`

## 1. 目的と境界

旧PR-2 pilotで用いたlinear H4、1.0 Å、STO-3G、DF rank 12の完全入力、すなわちconstant、one-body、ordered lambda、ordered g matrices、sector basis、stateが保存されているかを一巡だけ確認した。

この監査では既存ファイルの列挙、digest計算、JSON・NPZ・HDF5 metadataの読取り、Git object/historyの確認だけを行った。分子生成、SCF/DF、Hamiltonian再生成、signal計算、candidate評価、compile、trajectory sampling、held-outのsignal/cost/ranking評価は行っていない。旧hashへ戻るまで条件を変える探索も行っていない。

## 2. 旧系列の不変な状態

- 旧pilot artifact：`artifacts/pr2_pr3_minimal_pilot/2026-09-27/pr2_pr3_minimal_pilot_v1.json`
- 旧pilot artifact SHA-256：`65d5c12a129cd50f521b069b63be140f7bbee1eddf7ea8b859b18737dbb8d302`
- 旧pilot Hamiltonian hash：`d8b4aaf21afcc3935d5b5aa4d0805b358c5ec670d8104d25807c7cd0620a3dc3`
- 旧S0 artifact：`artifacts/pr2_s0_s1_validation/2026-09-28/pr2_s0_validation_v1.json`
- 旧S0 artifact SHA-256：`cf082a81ed70dcee774906ff2391683811cbdf544c1217127e100a977a11fbd7`
- 旧S0 status：`STOP_INPUT_REPRODUCTION_MISMATCH`
- 旧S0 observed Hamiltonian hash：`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`
- 旧S0 result fingerprint：`6d44888a1b806bc3b6418b49a18fdbcee09b621dd345838dc0d426abb1005182`
- 旧S0 `S1_authorized=false`、`automatic_next_stage=null`

この監査は旧S0をPASSへ変更せず、旧pilotと現snapshotの同一性も主張しない。

## 3. 確認した保存先と結果

| 対象 | 確認内容 | 結果 |
|---|---|---|
| 旧pilot JSON・source | 保存field、生成経路、hash関数 | full Hamiltonian hash、energy、rank別summary等はあるが、one-body、g matrices、stateの完全配列はない |
| Git全到達object | `git rev-list --objects --all`でPR-2、H4 snapshot、pilotを確認 | 旧pilot JSON/source/docsと現S0 snapshotはあるが、旧pilot完全配列はない |
| Git stash | `git stash list` | 空 |
| repository内NPZ/HDF5/cache | H4、rank12、PR-2候補を列挙しmetadata/hashを照合 | 旧pilot Hamiltonian hashと一致する完全入力なし |
| OpenFermion MolecularData既定保存先 | `H4_sto-3g_singlet_df_d100.hdf5` | birth/mtimeが2026-09-28 06:38 JSTで旧S0実行時に生成された現ファイル。SHA-256は`a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958`であり、2026-09-27旧pilot原本とは扱えない |
| `/home/abe`の合理的な名称検索 | exact MolecularData名、PR-2 pilot NPZ、H4 1.00 Å rank12 snapshot | 上記現S0ファイルと現development snapshot以外なし |
| `/tmp`の同検索 | 同上 | 該当なし |
| 既存connected-cluster H4 snapshot | metadataとHamiltonian hash | hashは`56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`で旧pilotと不一致。stateもなく代替原本ではない |
| 既存ground-state cache | state長、energy、Hamiltonian hash | toy/別入力であり旧pilot原本ではない |

## 4. 保存情報だけから回復できない理由

旧runnerは実行中にDF Hamiltonianとstateを構成するが、artifactには完全配列を保存していない。`df_hamiltonian_hash`はconstant、配列bytes、metadata、fragment順序、weight rule等をdigestへ含める一方向の識別子であり、digestとscalar summaryから配列を一意に復元できない。

記録されているPython/package versionだけでは、BLAS/LAPACK、thread、solver経路、当時の中間MolecularDataまで固定されない。現S0とのenergyやlambda summaryの近さもbyte-level同一性やoperator同一性の証明にはならない。

## 5. 回収可否とアクセス範囲

判定は「確認した合理的なローカル保存先からは回収不能」である。repository外の未提示external backup、別端末、クラウド保存、削除済みfilesystem blockは確認対象外であり、「世界のどこにも存在しない」とは主張しない。ただし本プロジェクトで現在利用できる証拠から旧入力を再構成する経路はないため、V0探索はここで終了する。

## 6. 新系列へ渡す固定入力

回収不能時の規定に従い、次を別系列のdevelopment入力候補として採用する。

- path：`artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`
- raw file SHA-256：`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`
- Hamiltonian hash：`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`
- 採用理由：S0で最初に完全保存され、将来のcandidate signal/cost/rankingを見て複数入力から選別されていないため
- 非採用理由ではないもの：旧pilotと同じだから、または性能が良いからではない

held-outは次の保存物を維持し、V1–V3ではsignal/cost/rankingを開かない。

- path：`artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p30_rank12_held_out_v1.npz`
- raw file SHA-256：`ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37`

## 7. 終了宣言

V0は`OLD_INPUT_UNRECOVERABLE_USE_SEPARATE_NEW_SERIES`で終了する。旧S0のSTOPと旧S1非承認を維持したまま、`pr2-rebaseline-de7a5492-v1`を別のdata seriesとして事前登録する。旧pilot数値を新系列のsame-input baselineへ転記しない。

