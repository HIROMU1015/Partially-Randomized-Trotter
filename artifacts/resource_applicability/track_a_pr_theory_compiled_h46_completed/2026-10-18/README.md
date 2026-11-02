# H4/H6 PR theory vs compiled：完了入口

**OBJECTIVE_COMPLETED_LIMITED_SCOPE / mandatory STOP**

- [研究結果・全条件表](../../../../docs/research/track_a_pr_theory_vs_compiled_completed_20261018.md)
- [結果summary](result_summary.json)、[期待値vs保存平均84行](expectation_vs_saved.json)、[同一event84行](same_event_comparisons.json)、[PR/PF360行](pf_pr_comparisons.json)
- [モデル/native差の監査](model_native_difference_audit.json)、[保存shots70条件](preserved_shots.json)
- [SOURCE closure](source_closure.json)、[SOURCE commit/mirror証跡](source_commit_proofs.json)、[入力receipt](input_manifest.json)
- [最終gate](final_gate.json)、[科学call ledger](call_ledger_summary.json)、[native7 worker回収](native_worker_exits.json)、各stage/のgrantとSTOP

最終科学SOURCE `536a3671c0ec2939ff3886e40467db26a27ec65a`。branch `track-a-pr-theory-compiled-h46-completed-20261018`。
新compile 0、保存費用再利用14/14、新trajectory0、control28。H4/H6のDF政策・探索/確認phaseを混ぜない。source/raw入力/旧grantは保全。

公開planはprivate絶対pathを除いた監査projectionで、実行用grant再使用の手順ではない。元planのbyte SHA/canonical digestを別保持。Source2/3は同一treeの公開mirrorと原commit headerを保持しており、mirrorのSHAを実行SHAに置き換えていない。

次の担当：ユーザー/GPTが固定範囲の科学的解釈と原稿への反映を判断。追加科学計算は実行しない。

[期待値比較図](expectation_vs_saved.svg)・[同一event比較図](same_event_comparisons.svg)・[図の再描画script](generate_figures.py)。図は保存値の描画のみで、新科学計算を実行しない。

公開sourceの対応（実行SHAと公開mirror SHAを区別）：

|Stage|実行SOURCE|公開source|
|---|---|---|
|2|`98e5c9e61f245dbb9faf381d835a3846ee7e9bf3`|[同一tree mirror bdbd8d5b](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/bdbd8d5b4ca00d2d0b49c6829248419c86b4f26d)|
|3|`34b2da1a6bfc001314e0e31458d803caca80adf2`|[同一tree mirror fe889c4a](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/fe889c4ac070a95ba949370ebf523a0749cc5149)|
|4|`536a3671c0ec2939ff3886e40467db26a27ec65a`|[実行commit自体](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/536a3671c0ec2939ff3886e40467db26a27ec65a)|

Stage2/3の原SHAがGitHubのcommitとして直接取得できるとは報告しない。対応tree全体の同一性と原commit headerのSHA1を照合済み。[各528比較行の来歴索引](row_provenance_index.json)を参照。
