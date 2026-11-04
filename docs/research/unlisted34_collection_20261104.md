# 追加34枝の報告・再現用SOURCEと保存原版の収録

2026-11-04。比較基準mainは`8ee8cd6686d5f7861ab96cd80b240b26a28ffafb`。
前回の一次分類で特定した34枝に対象を固定し、照会時の公開先端snapshotを比較した。
同じpath・別path・保存版に同一blob/modeがあるものを除き、既存の報告・小さな集計・再現用SOURCEをmainに収録した。

## 収録した内容と出典

追加した原版は1,079項目、36,108,619 bytesで、元のblob・modeを保つ。
元pathへの追加は858項目、保存原版は221項目である。
[全ファイルの出典・配置対応](unlisted34_collection_20261104_files.tsv)から、
原snapshot/path/blob/modeとmainの保存先、元の公開文書への固定リンクを辿れる。
これは閲覧・再利用に必要な内容の集約であり、枝全体のGit mergeではない。

| 系列 | main内で読む入口の例 | 読み分け |
|---|---|---|
| H8長時間・高次PR | [実行報告](h8_long_time_execution_20261103/REPORT.md)、[準備時の入力・SOURCE対応](h8_long_time_adapter_preparation_20261103/VALIDATION.json) | 保存時点の途中経過・HOLDを含む。後続の実行完了・科学的採否を今回認定しない |
| 研究Cの比較・精度・近実用条件 | [既存証拠の対応](research_c_evidence_map_updated_20261103.md)、[精度scopeの統合](research_c_b1_b2_accuracy_scope_integration_20261103.md)、[fixed-word pilot報告](research_c_fixed_word_pilot_execution_20261104.md) | 各報告のSOURCE・入力・対象lawと条件を別々に保つ。準備と実行済みを混同しない |
| RPE測定・mean・数値精度 | [固定要求B報告](rpe_fixed_request_measurement_b_20261103/REPORT.md)、[mean検証報告](rpe_m17_mean_validation_execution_20261103/REPORT.md)、[virtual bridge準備報告](rpe_m17_virtual_measurement_bridge_preparation_20261104/IMPLEMENTATION_REPORT.md) | 保存済みの診断・準備SOURCEと科学的な保証・採否を区別する |

出典のsnapshotは資料を閲覧するための公開状態であり、各計算の科学SOURCEとは限らない。
科学SOURCE・入力identity・runtime・statusは原記録の値を使用し、この収録commitへ付け替えない。
原文の「現在」「次段階」、過去の許可・契約・STOPは、それぞれの研究と記載時点に限る。

## 保存原版と実行context

同名の現行mainと異なる内容、公開snapshot間に複数版がある内容、
異なる依存SOURCEを必要とする内容は、
`docs/archive/unlisted34_collection_20261104/<blob>/<原path>.original`へ保存した。
研究testと研究別configurationも、自動実行や意図しない採用を避けて保存原版とした。
既存mainの共有SOURCE・正本の概要・manifest・AGENTSは上書きしていない。
保存版の収録だけで、共有APIの採択・実行互換性・科学的結論を認定しない。

Python SOURCEはASTと固定元snapshotのproject/sibling import先のblob/modeを静的に確認した。
今回、元pathへ追加したPythonは33件、保存原版は74件である。
静的検査はdynamic import・外部依存・runtime互換性の保証ではない。
再現の際は対応表の固定snapshotと各実験のSOURCE manifestを使う。
patchは未適用、lockの環境構築、GDB trace、研究test・runnerの実行は行っていない。

原報告の内容・相対pathを改変しないため、main内の相対リンクがraw・保存版・元の実行contextを必要とする場合がある。
その場合は対応表の`source_context_url`から原snapshotの文書・SOURCEを開く。
`.original`は当時の原版であり、元pathへそのまま置き戻せば実行できるという意味ではない。

## 原データ・運用記録の所在

大きな記録、NPZ/QPY/ZIP、生成回路・trajectory・raw・log等はコピーしない。
[今回残した公開原データ等の所在](unlisted34_collection_20261104_raw_locations.tsv)は、
元の公開snapshotとtree metadataによる所在を示す。
各行の件数・bytesはそのsnapshot・directory・除外理由の組での値であり、
同じ原本が複数snapshotへ継承されているため、行を合計して独立した原本数・データ量としない。
[既存の原データ所在表](main_collection_20261103_raw_locations.tsv)で管理済みの同一path/blobは再列挙しない。
この照合で外部の私有データ、ignored/未追跡原本、backupの完全性や別媒体保存を確認したとは扱わない。
私有・SOURCE withheld・seed・内部path・実行時claim/grant/marker等の制限は維持し、公開範囲を拡大しない。

## 比較した範囲と技術確認

[34枝の固定snapshotと照合範囲](unlisted34_collection_20261104_branches.tsv)に、
前回の観測先端、今回の固定先端、共通祖先、同一内容と今回の追加件数を記録した。
H8実行枝1本は一次分類後に先端が進んでいたため、両OIDを残して今回の観測値で比較した。
各snapshotの比較対象はmainとの共通祖先以後に内容・modeが変わったpathであり、
継承した別系列の成果も含む。各枝で新たに実施した計算の件数とは解釈しない。
元枝の削除pathをmainへ伝播させる処理は行わない。

全1,079項目を原blob・modeと照合し、Python 107件のAST、JSON 727件の構造、CSV 3件の表構造を確認した。
新しい案内・対応表の構造と案内リンクも確認した。元の研究test・runner・compilerは実行していない。
新規案内の`git diff --check`はPASS。元の保存patchにある3行の空白は、原blobとの一致を保つため変更していない。
科学計算、保存結果の科学的再解析、回路build/compile、重要科学レビューは0件である。
元SOURCE・入力・原結果・契約・grant・ledger・marker・STOPの内容は維持する。

今回、ブランチ・タグの追加削除は行わない。
内容の選択収録と、先端のGit履歴保持・現在のworktree/未push作業・利用予定・固定ref依存の確認は別の作業である。
この収録だけで34枝を削除可能とは判定しない。特に固定verifierが照会する
`track-b-g6-w2c-b3-v2-independent-acceptance-server-20261019`のリモート名は保持する。
