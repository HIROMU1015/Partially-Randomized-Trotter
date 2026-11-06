# Track C：H4入力照合結果と残る採用事項

2026-11-06 JST / SERVER 2 (`GPUA100-gleap02`)。基準commit `66553d948aa9df684987972789c6f3ee8a741312`。

**H4入力照合：PASS（倍精度の数値診断）。今回許可された元H4積分・保存DF/stateの照合と基底状態参照取得を完了した。3pilotは未実行。** 既存5計画は変更しない。この追補、照合script、結果JSON、基底状態参照を専用branchで公開し、mandatory STOPする。

## 入力と照合方法

優先した物理条件はlinear H4、隣接1.00 Å、STO-3G、charge0/multiplicity1、8 spin orbitals、4電子number-sector（dimension70）。Sz sectorへ追加投影しない。元Hの厳密基底状態を仮定し、状態準備費を除外する方針を維持する。

| 固定入力 | 確認した内容 |
|---|---|
| [既存のDF化前H4 snapshot][original] | `one_mo`、chemist ERI、MO係数、核反発、無閾値spin tensor、元H行列/state、既存FCI energyを保存。[receipt][original-receipt]のfile SHA-256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` と一致。人工modelではない |
| [legacy DF12 snapshot][snapshot] | file SHA-256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。H hash `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`。H4/1Å/STO-3G、rank12、保存metadataのtruncation=0/hermitized=true |
| [保存physical_state][state] | file SHA-256 `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4`。同じH hash、fingerprint `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`、70要素。配列を今回直接評価 |

許可前の静的確認ではmetadataの対応までを確認し、数値配列は評価しなかった。その後、既存の別SCFで保存されたDF化前積分を上記fixed objectから取得し、今回の許可範囲で数値照合した。新SCF・新DF分解は行っていない。

二つのsnapshotのMOはそのまま同一ではない。一体積分の直接差は最大 `1.57123e-7 Ha`。legacy DFのone-body correctionを正しく戻した裸の一体積分と、元積分の非縮退固有系を使い、16通りの実符号を調べ、二体tensorも一致する軌道対応Uを取得した。対称性で等価な4対応から恒等行列に最も近いものを選んだ。これは科学的armの最適化ではなく、同じ物理Hの座標対応である。

Uの向きは `C_legacy = C_source U`、`h_legacy = U† h_source U`、`state_source = wedge(U) state_legacy`。NPZの `orbital_map_source_to_legacy` はこのUを保存する。`||U-I||₂=2.66624e-7`。legacyのDF前入力ファイルとのbyte同一性や旧MO係数そのものの復元は主張しない。

[照合script](../../../scripts/check_track_c_h4_saved_inputs.py)は固定SOURCEのDF excitation table/matvecとH4無閾値spin liftを再利用する。SCF・DF生成・回路runnerをimport/実行しない。CAR作用で元積分から行列を別途組み、Qiskit/OpenFermionのbit順変換、定数、spin順、二体1/2係数、DF normal ordering、determinantへの外積写像を確認した。70次元元Hを対角化し、丸め誤差を除くためのHermitian平均後に得たstateを、平均前の元Hでも残差評価した。

## 数値結果

| 診断 | 今回の観測値 |
|---|---:|
| 核反発/scalar差 | 0 Ha |
| 軌道対応後の一体/二体tensor最大差 | `2.44249e-15 / 9.15934e-16 Ha` |
| 元Hとlegacy DFの4電子sector作用素2-norm差 | `9.48099e-15 Ha` |
| DF既存matvecとnormal-order再構築の2-norm差 | `1.77975e-15 Ha` |
| 元積分からの全256次元行列と保存元HのFrobenius差 | `2.64088e-14 Ha` |
| 外積基底写像のintertwining残差2-norm | `2.25452e-15 Ha` |
| 元H ground energy / gap | `-2.166387448634761 / 0.23263021512018023 Ha` |
| 新ground参照の元H残差L2 | `1.90700e-15 Ha` |
| 保存stateの元H ground残差L2 | `2.80404e-15 Ha` |
| 保存stateのground energy差 | `8.88178e-16 Ha` |
| 保存stateと新groundの `abs(1-|overlap|²)` | `8.88178e-16` |
| 元H energyと保存FCI energyの差 | `3.10862e-15 Ha` |

入力確認の許容差は既存H4 receiptのmatrix agreement `1e-9 Ha`、state residual `1e-10 Ha`、orbital `1e-10`、state norm `1e-12` を再利用し、10確認すべてPASS。これらはpilotのenergy精度や成功確率の採択ではない。overlapの丸めスケールの値や計算上の0を数学的厳密一致・厳密包含保証と解釈しない。新しいPySCF FCI計算は行わず、FCI比較には既存の保存scalarを使った。

**結論：この既存の元H4入力とlegacy DF/stateは、明示した軌道対応後に丸めスケールで一致する。共通pilot入力に使うための数値的根拠が得られた。** 数学的厳密groundの仮定と有限精度stateの診断は区別する。

成果物は [RESULT.json](../../../artifacts/track_c_h4_input_check/2026-11-06/run02/RESULT.json) と [ground_reference.npz](../../../artifacts/track_c_h4_input_check/2026-11-06/run02/ground_reference.npz)。後者はlegacy座標での元H行列、ground state、70固有値、basis indices、energy、Uを含む。再現時はscriptへ未使用の出力directoryを指定する。

技術確認として2-modeのCAR作用（scalar・hopping・quartic係数）を検査した。最初のrun01は数値処理後のJSON出力でNumPy boolのserializationに失敗し、ground NPZのみを残した。boolean出力を修正して同じ入力・数値手順をrun02で再実行し正常終了した。run01のpartial artifactはローカルに保全し、公開結果はrun02とした。科学条件の変更や旧実験の再開はない。

## 未決事項とSTOP

| 項目 | 次に必要な判断 |
|---|---|
| target/成功/RPE | 既存案 `εE=.00015936001019904 Ha`、`α_total=.05`、`β_RPE=.4`、`δ0=.02`、18round m0..17 の採否と全誤差予算・alias条件 |
| compiler/資源 | C-N/C-PのQiskit1.3.0・rz/sx/x/cx・opt2・seed17案、C-Sのstrict Clifford+T profileの採否。RZ・CX・T・depthは別指標として扱う |
| 有限pilot範囲 | [既存3計画](REPORT.md)のformula/step、同位置residual/precision policy、processing kernel/P方向/端点実装と、同じ条件でのbaseline・配分自由度・価格精度 |

位相・半径・normalization・shot・full価格を接続したRPE総コストは今回未評価。PF/RTE信号、合成、回路build/compile、sampling、3pilot本体は実行していない。旧SOURCE・raw・result・grant・ledger・marker・STOP、他worktree、共有設定・他ユーザーのjobは変更していない。旧5計画の実行待ち表示は作成時点の履歴であり、この追補は今回の限定的な入力照合結果を記録する。

**mandatory STOP：許可範囲の入力照合・ground参照取得と公開で終了し、pilotへ自動移行しない。**

次の担当は、**ユーザー**が上記共通条件と有限3pilot範囲を採択し、その実装・科学計算・compileを明示許可する。**Codex**はその範囲で既存コードを最小拡張し、独立pilotの計算processを可能な範囲で並列実行する。今回の入力確認をやり直すための追加承認や承認証明の保存は不要。

[original]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e22d7718c47358ff650dbc04e8e72f6ab1c8bf41/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz
[original-receipt]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e22d7718c47358ff650dbc04e8e72f6ab1c8bf41/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json
[snapshot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz
[state]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e22d7718c47358ff650dbc04e8e72f6ab1c8bf41/artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz
