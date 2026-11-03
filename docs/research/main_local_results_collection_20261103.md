# ローカルに残っていた文書・SOURCEのmain収録

2026-11-03。既存の収録計画で把握していたローカル17 checkoutについて、未コミットの追跡ファイルと
可視の未追跡文書・SOURCE等をmainと照合した。比較基準は `f8dc374161e83633a24832b5c41537f2fcdac8c7`。
ignored領域、runtime/cache/raw群、他の未調査worktree、cleanな全履歴の再棚卸しは対象に含めない。

## 今回の収録

| 区分 | 件数 | 扱い |
|---|---:|---|
| 通常配置への追加 | 63 | 独立した報告・原稿・レビュー・研究ノートとmetadata-only SOURCE入口2件、原文が参照する既存の編集確認JSON1件 |
| 同名別版・過去の索引・契約・run記録・研究test等 | 68 | `.original` として原内容を保存。現行文書や実行状態を上書きしない |
| 内容が同一の別path | 1 | 収録済みの同じ原内容へ対応付ける |
| ローカル管理記録 | 7 | 原本のあるローカルに保持。棚卸し・保全baseline・移動台帳の公開は行わない |

重複を除いた原ファイルは131件、6,292,746 bytes。
ここでの件数はpathとblobの組であり、研究数・実験数・ブランチ数ではない。

[全収録内容・原path・blob・関連HEADの対応（TSV）](main_local_results_collection_20261103_files.tsv) ／
[保存原版の入口](../archive/local_collection_20261103/README.md)

## 資料への入口

- Track A原稿v0.2：[本文](../manuscripts/track_a_resource_study_v0_2.md)、
  [補足](../manuscripts/track_a_resource_study_supplement_v0_2.md)、
  [当時のclaim/evidence編集記録](../manuscripts/track_a_resource_study_claim_audit_v0_2.md)。
  草稿と当時のscope判断を保存したもので、今回新たに主張を採用したものではない。
- Track A：[Level 1の受領レビュー](../reviews/track_a/PR_resource_evaluation_Level1_review_2026-10-23.md)、
  [研究課題選定の受領レビュー](../reviews/track_a/PR_Phase2B_研究課題選定レビュー_2026-10-23.md)。
- Track B：[横断レビュー原文](track_b_cross_scientific_review_20261019.md)。
  M1-S・G6・G10等の受領原文は上記対応TSVの `docs/reviews/track_b/` の項目から辿る。
- 研究C：[10月25日の受領レビュー](../reviews/research_c/research_c_c1_scientific_review_h2_h4_h6_20261025.md)、
  [10月31日の受領レビュー](../reviews/research_c/research_c_comprehensive_scientific_review_20261031.md)。
- SOURCE入口：[AX-2A metadata-only準備writer](../../scripts/resource_applicability/prepare_track_a_ax2a.py)、
  [AX-2B metadata-only準備writer](../../scripts/resource_applicability/prepare_track_a_ax2b_preflight.py)。
  既存の対象moduleと関数の存在を静的に確認した。実行・import・科学性能の検証は行っていない。

## 原本・出典と版の読み分け

今回の対象は未コミットのバイト列である。対応TSVの `source_snapshot_oid` は `UNCOMMITTED` とし、
関連HEADは元checkoutの文脈だけを示す。そのHEADに今回のファイル内容が含まれるとは扱わない。
今回の収録commitを、過去の科学SOURCE・入力identity・実行時runtimeの代わりに使わない。
各原文内に記録されたSOURCE・条件・status・STOPを参照する。

原ファイルの内容とmodeを保持した。rootで受領したレビューを分類先へ置いた場合も、原pathと収録pathを
TSVで対応させている。元のローカルファイルは移動・編集・削除していない。
過去の総合索引・研究概要・validation manifest・同日研究ノートの別版等は保存原版とし、
現在のmainの正本、既存の証拠status、入力条件、実行許可・grant・ledger・marker・STOPを置き換えない。
過去の実行状態を含む結果JSONと契約も保存原版であり、新しい実行権限やCI再現証拠ではない。

原文の相対リンク・ローカル環境の再現手順は当時の文脈を保持しており、mainで全リンクや全runtimeが
成立することを保証しない。対応TSVを使って同じ原内容の収録先を確認し、収録対象外の原データは既存保存先を使う。
原文にある時点の「現在」「次段階」と、最新の研究判断を区別する。

原稿が参照する既存の小さなeditorial audit JSONは、原文のまま追加した。記録された過去の検査を再実行したものではない。
M1-B1の14,917,348 bytesの結果JSONは既存の保存先を維持し、
[固定公開snapshotの原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json)
から辿る。原稿補足中の同JSONへの相対リンク1件は、main内にpayloadがないため、この固定原本を使う。
この所在確認で結果値を再解析したり、原稿の科学的主張を新しく認定したりしていない。


## 確認範囲と残る作業

原blob/mode一致、Python AST、JSON構文、新規案内のリンク、公開差分を確認した。
これは既存文書・SOURCEの技術的な収録であり、重要科学レビューや数値再解析の新規実施ではない。
新しい科学計算・数値再解析・回路build/compile・元の研究runner/test実行は0件。
原文に記載された過去の科学結果やPASSを、今回の実行・独立再現結果と称しない。
全体API整合・科学的採用・原結果の再現は今回検証していない。
既存のvalidation manifestと証拠statusは変更しない。

サーバー1・2の追加収録は各担当の完了報告と最新mainで別途照合する。
このローカル収録は、サーバー側の原本・bundle・rawの受領や新しいバックアップを要しない。
ブランチ全体のGit merge・ブランチ削除は行っていない。
