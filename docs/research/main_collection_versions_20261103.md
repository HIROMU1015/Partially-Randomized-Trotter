# main集約：索引と同名ファイルの版差整理

2026-11-03。前回保留した37 pathを、索引・SOURCE・時系列記録・過去の承認記録へ分けた。
目的は研究結果と再現用SOURCEをmainから辿ることであり、新しい科学的採否や実行許可の判断はしていない。

| 扱い | path数 | 内容 |
|---|---:|---|
| 索引の統合 | 7 | 元索引の資料リンクをまとめる。元の全版は別途保存 |
| 公開履歴で後続版を確認できたSOURCE | 12 | 公開ファイル変更の旧blob→新blobの経路を確認し、全候補版から一意に到達する版を原内容で配置 |
| docstringだけのnamespace | 1 | 両版ともmodule docstring以外の文がないことをASTで確認し、原版を配置 |
| 単一の実行pathを選ばず原版を保存 | 17 | 独立に分岐したSOURCE 7、研究ノート4、発表資料4、過去の承認記録2 |

全37 path・88版（1,594,127 bytes）を保存し、出典commit・blob・modeと対応付けた。
SOURCEの後続関係は公開履歴の変更から求め、日付・ブランチ名・ファイルの大きさから版を選んでいない。
いずれのSOURCE配置でも、後続の公開子孫で削除されたファイルを再導入していないことを確認した。
分岐したP1 pipeline一式とCのprice-statistics SOURCEは、単一の整合する実行版を選んでいない。

[全88版の出典と配置（TSV）](main_collection_versions_20261103.tsv) ／ [原版の保存場所](../archive/main_collection_versions_20261103/README.md)

## pathごとの扱い

| 元path | mainでの扱い | 保存版数 |
|---|---|---:|
| `artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/authorization.json` | 全原版を保存。単一版の選択は保留 | 2 |
| `artifacts/track_b_g10_degree_preparation/2026-10-10/authorization.json` | 全原版を保存。単一版の選択は保留 | 2 |
| `docs/manuscripts/README.md` | [統合索引](../manuscripts/README.md) | 3 |
| `docs/progress/track_a/2026-10-21_PR資源評価_スライドQA_Codex参照メモ.md` | 全原版を保存。単一版の選択は保留 | 2 |
| `docs/progress/track_a/2026-10-21_PR資源評価_説明意図明確化版_Codex参照メモ.md` | 全原版を保存。単一版の選択は保留 | 2 |
| `docs/progress/track_a/PR資源評価_スライドQAノート_2026-10-21.md` | 全原版を保存。単一版の選択は保留 | 3 |
| `docs/progress/track_a/PR資源評価_詳細スライド構成案_14枚基本・12枚圧縮_2026-10-21_説明意図明確化版.md` | 全原版を保存。単一版の選択は保留 | 2 |
| `docs/progress/track_a/README.md` | [統合索引](../progress/track_a/README.md) | 3 |
| `docs/research/研究ノート/2026-10-04.md` | 全原版を保存。単一版の選択は保留 | 2 |
| `docs/research/研究ノート/2026-10-06.md` | 全原版を保存。単一版の選択は保留 | 2 |
| `docs/research/研究ノート/2026-10-26.md` | 全原版を保存。単一版の選択は保留 | 2 |
| `docs/research/研究ノート/2026-11-02.md` | 全原版を保存。単一版の選択は保留 | 2 |
| `docs/tracks/algorithm_codesign/README.md` | [統合索引](../tracks/algorithm_codesign/README.md) | 8 |
| `scripts/h8_key_compile_parallel.py` | [原版を配置](../../scripts/h8_key_compile_parallel.py) `a631d63b1bad` | 2 |
| `scripts/restore_pauli_h2_h6_evidence.py` | [原版を配置](../../scripts/restore_pauli_h2_h6_evidence.py) `209ba3875669` | 2 |
| `scripts/run_df_research_pipeline.py` | 全原版を保存。単一版の選択は保留 | 2 |
| `scripts/tracks/algorithm_codesign/README.md` | [統合索引](../../scripts/tracks/algorithm_codesign/README.md) | 4 |
| `src/trotterlib/h4_d1_adapter.py` | [原版を配置](../../src/trotterlib/h4_d1_adapter.py) `7b9869a51bc0` | 2 |
| `src/trotterlib/h4_d2_pauli_control.py` | [原版を配置](../../src/trotterlib/h4_d2_pauli_control.py) `6ca2fa2154b7` | 2 |
| `src/trotterlib/pauli_signal_adapter.py` | [原版を配置](../../src/trotterlib/pauli_signal_adapter.py) `3fd02013360d` | 2 |
| `src/trottertracks/algorithm_codesign/df_correction_comparison/README.md` | [統合索引](../../src/trottertracks/algorithm_codesign/df_correction_comparison/README.md) | 2 |
| `src/trottertracks/algorithm_codesign/df_correction_comparison/native_evaluation.py` | [原版を配置](../../src/trottertracks/algorithm_codesign/df_correction_comparison/native_evaluation.py) `73bc38e353d7` | 2 |
| `src/trottertracks/algorithm_codesign/df_correction_comparison/worker.py` | [原版を配置](../../src/trottertracks/algorithm_codesign/df_correction_comparison/worker.py) `b6f4fb865d5a` | 2 |
| `src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py` | [原版を配置](../../src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py) `2b6f083cb672` | 2 |
| `src/trottertracks/c_pr_aware_df_m4.py` | [原版を配置](../../src/trottertracks/c_pr_aware_df_m4.py) `4c7fd8b4b521` | 2 |
| `src/trottertracks/c_pr_integrated_performance.py` | [原版を配置](../../src/trottertracks/c_pr_integrated_performance.py) `1d3fc707e08b` | 2 |
| `src/trottertracks/c_r32_price_statistics.py` | 全原版を保存。単一版の選択は保留 | 2 |
| `src/trottertracks/c_rc2_pr_stage_b.py` | [原版を配置](../../src/trottertracks/c_rc2_pr_stage_b.py) `29c30f71838c` | 2 |
| `src/trottertracks/resource_applicability/README.md` | [統合索引](../../src/trottertracks/resource_applicability/README.md) | 2 |
| `src/trottertracks/resource_applicability/__init__.py` | [原版を配置](../../src/trottertracks/resource_applicability/__init__.py) `2520332eca1a` | 2 |
| `src/trottertracks/resource_applicability/df_core_stages.py` | 全原版を保存。単一版の選択は保留 | 2 |
| `src/trottertracks/resource_applicability/h4_geometry_pipeline.py` | 全原版を保存。単一版の選択は保留 | 2 |
| `src/trottertracks/resource_applicability/pipeline.py` | 全原版を保存。単一版の選択は保留 | 2 |
| `src/trottertracks/resource_applicability/pipeline_fixture.py` | 全原版を保存。単一版の選択は保留 | 2 |
| `tests/test_df_research_pipeline.py` | 全原版を保存。単一版の選択は保留 | 2 |
| `tests/tracks/algorithm_codesign/README.md` | [統合索引](../../tests/tracks/algorithm_codesign/README.md) | 5 |
| `tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py` | [原版を配置](../../tests/tracks/algorithm_codesign/test_df_correction_comparison_v2.py) `dde0e0af72f3` | 2 |

## SOURCE・証拠・実行状態の読み分け

各実験の結果を新しいmain commitの実行結果へ付け替えない。科学SOURCE・入力identity・runtime・
grant・ledger・marker・STOP・一回実行条件は元の記録と固定snapshotを使う。
過去のauthorizationは保存用の `.original` に置き、実行ディレクトリへ配置しない。
原文内の相対リンクは元snapshotを参照する。統合索引のリンクは、main内で同じblobの資料、
保存原版、元snapshotを区別している。

今回のSOURCEは過去の公開内容の収録である。main全体のAPI整合・科学性能・再現性の検証完了は主張しない。
元runnerや研究testは再実行せず、ファイル同一性、AST・JSON構文、リンク、既存manifest checkerを確認する。
全結果のstatusと既存のquarantine/missingはそのまま保持する。

## 残る範囲

枝全体のGit mergeとブランチ削除は行っていない。既存mainの別版ファイル、分岐したSOURCE一式、
symlink等65 path、ルートの実行claim/marker7件は、元snapshot・保存参照を維持する。
大きな原データは既存の保存先を使い、今回も再計算・再解析・compileは行わない。
