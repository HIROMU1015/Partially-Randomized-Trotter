# mainに収録した既存研究成果

2026-11-03 JST。枝ごとに保存されていた成果を、研究単位でmainに収録する。
以下は既存成果の配置と出典の案内であり、新しい科学的採否・保証・再実行の判断ではない。

## 第1回：Track B G6（2026-10-10）

| 入口 | main内の資料 |
|---|---|
| 結果・取得範囲・未解決点 | [G6報告](../tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md) |
| 保存結果・STOP・SOURCE binding | [原証拠ディレクトリ](../../artifacts/track_b_g6_return_generator_audit/2026-10-10/)と[evidence manifest](../../artifacts/track_b_g6_return_generator_audit/2026-10-10/evidence_manifest_v1.json) |
| 数学・実装・文献の元監査 | [独立数学監査](../tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)、[finite-bit/access監査](../tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)、[prior-art表](../tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md) |
| 再現用の原SOURCE | [形式prototype](../../src/trottertracks/algorithm_codesign/return_aggregation.py)、[元technical runner](../../scripts/tracks/algorithm_codesign/g6_return_generator_audit.py)、[元test SOURCE](../../tests/tracks/algorithm_codesign/test_g6_return_aggregation.py) |
| 入力レビュー | [G5レビュー原本](track_b_G5_research_direction_review_20261010.md)と[入力identity](../../artifacts/track_b_g6_return_generator_audit/2026-10-10/review_input_identity.json) |
| 当時の工程記録 | [2026-10-10の研究ノート](研究ノート/2026-10-10_track_b_g6_return_generator_audit.md) |

元の公開snapshotは `28cfabb1d47e0e1824bce1e154a9c289738fa2b9`。
上記22ファイルはそのsnapshotのGit blobと同じ内容・実行modeで収録している。
元の `evidence_manifest_v1.json`、SOURCE hash、status、marker、STOP、既存の数値や判定は変更していない。
`src/trottertracks` は研究別SOURCEの名前空間であり、共通 `trotterlib` APIへの組込みを意味しない。

原報告の技術状態は `G6_IDEAL_IDENTITIES_VERIFIED_NATIVE_AND_METHOD_VALUE_UNRESOLVED`、mandatory STOPである。
保存された16 testsの記録は当時のsource-bound検査であり、今回のmain収録で再実行したものではない。
今回の確認はファイル同一性・配置・リンク・manifest構造だけで、native実装・性能・独立新規性・外部科学再現を新たに認定していない。

元runnerは当時のSOURCE、runtime、保護対象、既存markerに結び付く。mainへコピーしたことは、その一回実行やSTOPの解除ではない。
再現を検討する場合は元snapshotと既存scopeを確認する。Git管理外のrawやserver内のcacheはmainへの収録対象に含まれていない。

## 第2回：競合のない追加ファイルの収録

2026-11-03 JST。main以外の公開48枝を、先端コミット日付の古い順に通常のGit mergeで確認した。
全48枝に索引・manifest等の衝突があったため、枝全体をmerge済みとは扱っていない。
そのうえで、既存mainにないpathについて、公開snapshot間で内容とmodeが一致するファイルを追加収録した。
既存ファイルの上書き、異なる版の選択・混合、既存結果の再計算は行っていない。

- 追加収録：49,543ファイル、1,489,695,949 bytes。報告・SOURCE・図表・小さな記録を含む。
- 既存保存先に保持：18,947ファイル、8,635,154,300 bytes。QPY/NPZ等のpayload、生成inventory/rawディレクトリ、5 MiBを超えるJSON/CSV/TXT等のデータ記録を含む。
- 同名で複数版のある37 pathと、symlink等65 pathは選択・移動せず保留。後続snapshotで削除されたものも再導入しない。
- 統一索引のうち、元snapshotとGit blobが一致する18リンクをmain内の相対リンクへ切り替えた。科学SOURCEの指定やidentityリンクは固定参照を維持する。

5 MiBはmainのデータ配置を分けるための基準であり、科学実行の資源上限・入力条件・精度・採否を変更するものではない。
元ファイルの内容は変更していない。原文中の相対リンクがmain未収録のraw等を指す場合は、下表の原snapshotから同じ文書を開くか、原データ所在表を使う。

| 対応表 | 用途 |
|---|---|
| [枝と固定snapshot・収録範囲](main_collection_20261103_branches.tsv) | 48枝の照合OID、通常mergeの衝突path、追加収録の範囲。件数はsnapshot間で重複する |
| [原データの既存所在](main_collection_20261103_raw_locations.tsv) | 大きな原データを変更せずに辿る固定commit・directoryのリンク。新しいbackupの作成や完全性再検査を意味しない |
| [同名別版・特殊pathの保留](main_collection_20261103_deferred_paths.tsv) | 今回内容を選択しなかったpath |

収録したSOURCEは過去の公開内容であり、共通ライブラリとしての採択や、main全体での科学的な再検証を意味しない。
当時のSOURCE commit・入力identity・runtime・grant・ledger・marker・STOP・一回実行条件は各原記録を使う。
新しい科学計算、保存結果の数値再解析、回路build/compile、元runner/testの再実行は0回である。
Git管理外のraw・cache・他hostの原本は、その既存保存先を保全する。今回の所在表はGitに公開済みのデータだけを対象とする。

## 第2回の補足：ルートの報告・入力定義・ライセンス

上記の研究ディレクトリに加え、ルートにあった過去の報告・計画、`references/` の入力定義、
`schemas/`、`LICENSES/` の39ファイル（1,478,587 bytes）も、公開版の内容とmodeを保ったまま収録した。
[ファイル別の原snapshot・blob対応](main_collection_20261103_additional_files.tsv)を使って原記録へ戻れる。
第2回と補足の追加原ファイルは合計49,582件（1,491,174,536 bytes）である。

ルートの旧計画や `REPORT.md` は当時の研究別記録であり、現在の統合仕様ではない。
`PROJECT_MAP.md` に従い、現行の `docs/research/` の統合文書を優先する。
入力定義の配置は新しい科学実行の許可、既存契約の改訂、STOP解除を意味しない。
ルートの `.h4_execution_claims/` と `.h4_precision_b_claims/` の7件は原snapshotに保持し、
現在のcheckoutの実行状態へ移植しない。これらも保留path表に区別して記録した。

## 残る統合

索引・manifestの構造的な衝突、同名SOURCEの異なる版、科学的に意味の異なる記録を区別して解決する。
固定された原本を改変したり、一方の版を黙って捨てたりしない。
枝の削除は今回行っていない。追加収録だけで枝全体の履歴がmainの祖先になったとは判断せず、既存保存参照を維持する。

## 第3回：索引と同名SOURCEの版差整理

[37 path・88版の整理結果](main_collection_versions_20261103.md)を反映した。7索引を統合し、公開履歴で後続版が一意なSOURCE12件とdocstringのみのnamespace1件を原内容で配置した。
全88版をmain内に保存し、独立に分岐したSOURCE・研究ノート・発表資料・過去の承認記録17 pathでは単一版を選んでいない。
これは追加の内容集約であり、枝全体のGit mergeではない。

## 第4回：P1・研究CのSOURCEと依存版

[P1の6ファイル・研究C helper・P1依存core 5件の選定](../integration/main_p1_c_price_cohorts_20261103.md)を反映した。すべて公開SOURCEの原内容を使用し、変更前のcoreも保存している。
P1の人工・mock検査は79件PASS。研究Cの統計処理・科学実行は行わず、元のSOURCE記録・契約・STOPを保持する。
