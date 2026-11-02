# H4/H6高次PF研究データ統合・付属プロトコル作成報告

2026-10-20。**保存結果の統合整理と付属プロトコルの作成を完了した。新しい科学計算・数値再解析は0件。mandatory STOP。**

専用branch：`track-a-h46-higher-pf-dataset-protocol-integration-20261020`。基点：`5764aacb0714c6be42cb6dc7c7d8fdf1c8d98bdb`。新しいコードは既存値のreader・索引生成・構造検査だけで、scientific moduleのimportや実行grantを持たない。

## 成果物と固定証拠

- [統合研究データセットREADME](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/README.md)、[JSON索引](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/dataset_index.json)、[統合JSON](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/dataset.json)。候補88、精度対応440、軸来歴176、実測単価96のCSVも提供する。
- [高次PF付属実験プロトコル・記録v1](track_a_h46_higher_pf_supplementary_protocol_v1.md)。入力、PF版・順序、outer32/K4、signal/経験的適格性、native/compile、再利用、標本設計の未決事項を明文化した。実験前固定資料と事後文書を区別する。
- [共通プロトコルv2](track_a_pr_resource_evaluation_protocol_v2.md)の§11から参照を追加した。照合した最新公開版は`3826f566b04ad9c2ead9abf1b80c864f7b34b785`。既存本文は同commitと完全に同一のprefixとして保持し、H2完了追記も含めて上書きしていない。
- [固定SOURCE/結果catalog](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/source_catalog.json)、[原本照合範囲](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/private_metadata_verification.json)、[検証結果](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/validation.json)。

signalの科学SOURCEは`bec018c64e7a8458dbf13b1c0e9655d628292857`、結果commitは`a960a41da5ebc564b0306e798572257091c8509c`。費用取得の科学SOURCEは`635ba563766875d688ad6912499b7f90397d0cc5`、元終了監査SOURCEは`c1b419cde004d2b07b41f5727cd4722b5e0e0bc4`、公開結果は基点commitに固定する。各ファイルのSHA・Git blob・軸別receipt/result SHAを索引で追える。

## 保存値の結合と範囲

| 保存対象 | 統合内容 |
| --- | --- |
| signal88候補 | 新規82＋保存再利用6。元JSONへ固定参照、原結果SHAを候補別に結合 |
| 適格性880行 | 5精度×real/imagの既存値。同じεの両軸ラベルを照合し、式は再評価しない |
| 決定論m5単価 | H4新38＋保存2、H6新54＋保存2＝48候補96軸 |
| ゲート指標 | RZ/CX count、RZ/CX depth、total depth、circuit sizeを全96軸へ転記 |
| native検査 | 新46pairの保存368gateに参照結合。保存4軸は旧pilot/CT01受入の区分を保持 |
| 非active16候補 | 保存領域に残し、今回の両軸適格εなしと表示 |
| PR24群 | n_E、単価、shot、総費用は未取得/未採択のnull |

signal・880適格性・368gateの数値payloadを重複保存せず、固定commit/SHA参照で再利用する。新しい単価96軸表では、既存active表で不足していた再利用4軸のCX/depth等を、SHA確認した原receiptから正確に転記した。H4保存2軸のcompiled QPY SHA欠測はnullのままである。

新規/再利用、実測/model/欠測、原取得SOURCE/独立受入SOURCEを分離した。H6 full/q2の旧STOPは保持し、CT01受入を過去のPASSへ書き換えていない。既存強化S2/S4対照は別系列として保全し、H4 DF12・PF788・ordinary-control旧研究を統合表へ無条件に混ぜていない。

## 技術検証・保全・公開範囲

作成者自身による人工・改変拒否テスト31件がPASSした。主な検査は重複/欠落ID、異なる精度の軸join、数値転記の改変、compiler/preparation/SOURCE不一致、新規/保存の誤表示、欠測の0補完、旧STOP変更、native FAIL/phase alignment、exit異常、retry、AD03/shot政策の暗黙採択の拒否である。独立科学レビューではない。

構造検査では固定Git source27件のSHA/bytes/blob、統合表の全key・保存値対応・件数、共通v2公開snapshot27fileと本文prefix不変を照合した。小さな独立Git読取をthreadでまとめ、sourceはコマンド内でcacheする。重い科学workerは存在せず、数値kernelを起動しない。

今回使用した原JSON233file（signal88、費用92新軸・46pair結果、保存4軸、入力JSON3）をSHA確認した。QPY/NPZは復元も再読込もせず、元の終了監査を参照する。過去の大規模原本・全runtimeを今回再監査したとは主張しない。既存公開ファイルは索引追記以外変更せず、原SOURCE/raw/grant/ledger/marker/STOPには書込を行っていない。

基点branchには最新v2がなかったため、その公開本文とリンクに必要な公開済み依存資料を固定commitから移入した。これは他branchのmerge・科学成果の再採択ではなく、[一覧](../../artifacts/resource_applicability/track_a_h46_higher_pf_dataset_v1/2026-10-20/common_v2_imports.json)で追える最小のMarkdownリンク閉包である。過去DRAFT・採択・H2更新manifestの記録はそのまま保持した。

公開対象は研究metadata、SHA付き参照、値の転記表、文書、reader・人工テストに限定する。private原本、QPY/NPZ、seed値一覧、有効grant、認証情報、内部絶対パスは新規成果物に含めない。過去公開bundleを変更したり新しいraw inventoryを複製したりしない。詳細科学レビュー・最終evidence freezeは行っていない。

## 未解決事項と次の担当

1. **ユーザー**：PR24群について、固定探索標本の目的・n_E、または別の同law設計pilotを採択し、sampling/compileを独立承認する。現在のn_Eは全てnull。
2. **ユーザー・Codex**：会計前に正式shot政策、強化S2/S4とのnative/preparation/profile接続、保存値再現gateと数値実行範囲を別契約で固定する。経験的allowanceの診断承認をshot採択へ読み替えない。
3. **ユーザー**：探索後の独立確認とm5の主比較採用AD03を別に判断する。今回の整理から自動移行しない。

新規signal・compile・sampling・fit・shot計算・費用再会計・再ランキング・科学的数値再解析は全て0件。この資料整理をもってmandatory STOPする。
