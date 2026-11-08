# 固定SOURCE・参照資料の索引

2026-11-08。これは静的調査で実際に読んだ既存SOURCEの索引であり、新しい科学実行SOURCE/PASSを発行したものではない。

## 取得・同一性

専用branchの基点は `0d596e7a51a79013bd93cad133b6b14c5b1e2573`、tree `33d188a44c8988f8107d4dd45a6c9108fc239f93`。GitHub Git commit APIから同じSHA/treeを取得した。下記repository codeは基点のimmutable Git objectと作業treeのbytesが一致し、無変更で参照する。

適用AGENTS、PROJECT_MAP、研究概要・現状、研究目的・研究課題、VALIDATION_STATUSとmanifest、固定Track D実H4/C3/C5 PLANを参照した。正式共通指示 `research_project_instructions_unified_2026-10-25.md` は提供された上位共通資料から取得し、2026-10-25統合正式版の§8.4・§10を確認した。未取得扱いではないが、そのGitHub掲載の有無は今回調べていない。内部絶対pathや原本は公開しない。

## 回路・compiler・checkerの固定参照

| ID | SOURCE（全て基点0d596e7a） | 読解対象 |
|---|---|---|
| S1 | [src/trottertracks/c1_h4_strong_baseline.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/c1_h4_strong_baseline.py#L103) | vacuum_phase / transition / sequence |
| S2 | [src/trottertracks/track_d_h4_reference.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/track_d_h4_reference.py#L78) | control_one / directional_head / reference_gadget |
| S3 | [src/trottertracks/track_d_c3_whole_path.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/track_d_c3_whole_path.py#L19) | head_recipe / mixed_head / physical_segment / prepare_steps |
| S4 | [src/trottertracks/pr_dictionary_validation/circuits.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/pr_dictionary_validation/circuits.py#L59) | BasisProvider / append_central / build_event_sequence |
| S5 | [src/trottertracks/track_d_h4_paired_reference.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/track_d_h4_paired_reference.py#L87) | H4 full-frame Provider / scalar / forward/backward |
| S6 | [src/trottertracks/c1_stage2_native.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/c1_stage2_native.py#L93) | exact rotation runs / combined_diagonal / NativeBuilder |
| S7 | [src/trottertracks/c1_stage3_native_bridge.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/c1_stage3_native_bridge.py#L8) | signed RTE bridge / partial_kernel |
| S8 | [src/trottertracks/c_pr_aware_df_stage_b.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/c_pr_aware_df_stage_b.py#L118) | build_kernel / head-tail横断の非実装 |
| S9 | [src/trotterlib/df_partial_s2_repeated.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trotterlib/df_partial_s2_repeated.py#L572) | phase aggregation / _build_boundary_optimized |
| S10 | [src/trottertracks/resource_applicability/ax2b_native_df_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/resource_applicability/ax2b_native_df_v5.py#L94) | directional diagonalのbranch意味論 |
| S11 | [src/trottertracks/c1_h6_compiler_precision.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/c1_h6_compiler_precision.py#L10) | declared phase保持compiler |
| S12 | [src/trottertracks/pr_dictionary_validation/gates.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/pr_dictionary_validation/gates.py#L86) | UnrollCustomDefinitions / P0 transpile |
| S13 | [src/trottertracks/pr_dictionary_validation/fixture.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/pr_dictionary_validation/fixture.py#L54) | 固定CompilerProfile |
| S14 | [src/trottertracks/track_d_c7_2_toy.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/src/trottertracks/track_d_c7_2_toy.py#L98) | 2-mode/full-frame context / exterior lift |
| S15 | [tests/track_d_h4_reference_checks.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/tests/track_d_h4_reference_checks.py#L8) | AB/adjoint/vacuum/coherence/X/Y/postcheck |
| S16 | [docs/research/track_d_c7_2_cx_structure_20261107/REPORT.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/docs/research/track_d_c7_2_cx_structure_20261107/REPORT.md#L88) | 既存局所P0/非加法性反例 |

重点8ファイルのSHA-256（既存SOURCEの識別用）：

| SOURCE | SHA-256 |
|---|---|
| src/trottertracks/c1_h4_strong_baseline.py | `2b6d7e32aa9150fe6cacf88e33883dc4132293d371a4ca7a3ed383752e386b7f` |
| src/trottertracks/track_d_h4_reference.py | `da4c200aec6a7a8917fdbb4c3dda39e1bf649faf3d7e193a72ac0237a70fa4f9` |
| src/trottertracks/track_d_c3_whole_path.py | `ab9a9f3747470af3d4fca87e1a9efb890441b621e05fbd14eb1d1318aa4c9d7b` |
| src/trottertracks/pr_dictionary_validation/circuits.py | `a8881cfec1c93e3a88a4533dbd167d5d481bcf39110bdc5b8e5b79597f9bdbb4` |
| src/trottertracks/c1_h6_compiler_precision.py | `5bc0031e56d62836521e1a4d921930fde400444d834189ad2332c99944d7e2e2` |
| src/trottertracks/pr_dictionary_validation/gates.py | `538c67581c20c7d6b0615f22556c263f79bbae3960c72e99f8d0c9ba1fbd40c3` |
| src/trottertracks/track_d_c7_2_toy.py | `10567e42c4cd76785dd7d9aab5b9080194298c6ce006d5d18b1549f67375f213` |
| tests/track_d_h4_reference_checks.py | `970f07e268b44dc0c3aeccb237cbdd80033a24d038ccf8a586d09e05df1f0986` |

## その他の固定資料

以下は該当commitのローカルGit objectから参照した。O5五文書は21319176のbytesと基点版が一致。固定Bの科学結果は報告の引用のみで再解析していない。新規性の文献レビューも実施していない。

- [21319176：REPORT / CURRENT_ALGORITHM_MAP / IMPROVEMENT_CANDIDATES / NEXT_EXECUTION_PROPOSAL / PRIOR_ART_BOUNDARY。C1とC2の境界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2131917656446bd22f294946d7c1064a47332e4c/docs/research/track_d_o5_algorithm_design_20261108/REPORT.md)。固定commit `2131917656446bd22f294946d7c1064a47332e4c`。
- [a54d0405：候補B人工結果。新しい実行なし](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a54d04059bfc61adef43aa2d40fb9c001b62378f/docs/research/track_d_cut_artificial_20261108/REPORT.md)。固定commit `a54d04059bfc61adef43aa2d40fb9c001b62378f`。
- [2f316fb2：候補B q4_sample00完了](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2f316fb229149117177367720a6eaa5aa8a94b76/docs/research/track_d_cut_h4_q4_completion_20261108/REPORT.md)。固定commit `2f316fb229149117177367720a6eaa5aa8a94b76`。
- [0d596e7a：候補B保存q4・16本の中央cut結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d596e7a51a79013bd93cad133b6b14c5b1e2573/docs/research/track_d_o5_q4_cohort_metadata_20261108/REPORT.md)。固定commit `0d596e7a51a79013bd93cad133b6b14c5b1e2573`。
- [64072ba8：event_features。step内tail run/support/lane特徴はhead-tail端部gaugeを表さない](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64072ba81bb70db1df1b34e94adb08e78926f24b/src/trottertracks/track_d_c7_4_models.py)。固定commit `64072ba81bb70db1df1b34e94adb08e78926f24b`。
- [7ffe8b28：B6bの一般条件・half-ray条件を固定報告と既存O5対応mapで参照。今回新しい一般定理検証なし](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ffe8b281b76cef3a6fd06f43413904d6766d0ec/docs/research/track_d_b6b_bc_integration_20261108/REPORT.md)。固定commit `7ffe8b281b76cef3a6fd06f43413904d6766d0ec`。
- [0edc0fa5：O2とSEの同機構・強い共有対照・適用域](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0edc0fa5ad4656fdd492ca6c96ade4f77cadf412/docs/research/o2_vacuum_reference_cswap_20261108/REPORT.md)。固定commit `0edc0fa5ad4656fdd492ca6c96ade4f77cadf412`。
- [781e155d：O2のlegacy H4/opt2準備。Track D P0と別条件](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/781e155d0907a82c962e41ab8e89f985735bc49b/docs/research/o2_vacuum_reference_h4_preparation_20261108/REPORT.md)。固定commit `781e155d0907a82c962e41ab8e89f985735bc49b`。

## Qiskit公式SOURCE

[Qiskit1.3.0 builtin_plugins.py](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py) をGitHubから取得し、既存実行環境に保存された同fileのbytesと完全一致を確認した（43,969 characters）。init/optimizationのlevel1分岐を読解した。level1.pyのstage構成もローカルで参照した。Qiskit import・compiler呼出しは0。公式pass名を現在の全H4回路での消去完了証拠へ格上げしていない。
