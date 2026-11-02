# Track A paper preparation — integrated report

2026-10-19 JST。**H4/H6/H8のデータ索引、定量機構分析契約案、図表仕様、H8確認状態、証拠freeze準備を作成し、mandatory STOP。新しい科学計算・保存値の数値再解析は実施していない。**

## 成果物

- [paper_data_manifest.csv](paper_data_manifest.csv)／[説明](paper_data_manifest.md)：1,568索引行。保存主系列360、旧D240、新A全endpoint960、H8固定確認8。科学値は原文字列からの転記。
- [mechanism_analysis_contract_DRAFT.md](mechanism_analysis_contract_DRAFT.md)：DRAFT_UNAUTHORIZED。3系共通の正確な軸重み付き分解、入力参照、丸め、欠測、照合、schema、承認・STOP境界。
- [固定対象244枠](mechanism_targets_DRAFT.csv)／[入力readiness](mechanism_input_readiness_DRAFT.csv)：保存済み選択だけを列挙。core存在確認と数値一致を区別。
- [figure_table_specification.md](figure_table_specification.md)：3図＋主表／補足、H4/H6/H8を同じ分析対象として保持。完成figureは未生成。
- [main_table_transcription.csv](main_table_transcription.csv)：主系列45行とH8確認8行の原表。
- [h8_confirmation_status.md](h8_confirmation_status.md)：両n48 scopeがCOMPLETED。主2点再現と母平均未確定を分離。
- [paper_evidence_freeze_readiness.md](paper_evidence_freeze_readiness.md)：保存値で固定できる部分と未完了部分。

## 統合所見

A：H4 legacy DF12・測定subset16、H6 tol-only DF19・PF136、H8 tol-only DF28・PF184を区別した。旧PR n32と新A探索n8をpoolせず、H4 n32は選択・推定用fixed anchorと明記。主要5ε・通常factor1を主表にし、他会計／倍率／比較scopeは補足で区分した。

B：G=N_real C_real+N_imag C_imagから、軸別回路比・shot比とPF費用重みによる正確な分解を定義した。Ceffを使う積の恒等式も定義し、二軸単純平均の積を採用しない。実科学値へ式を適用していない。主系列30 contrastのC/N参照を結合したが、bias/B/headroomの全説明入力は未完了。H4旧full-PFの軸単価欠測は補足readinessへ残した。

C：Figure 1/3と主表は保存値転記を入力にする。Figure 2のデータは正式解析の別承認後のみ。レビュー概算を実測値として流用せず、機構図はPENDING_NUMERICAL_AUTHORIZATION。

D：e6eea538…の最新H8完了結果を追加取得・接続した。元設計434e67a8…、起動d21647d0…、resume be2e5475…との候補・cohort・SOURCE・control方針・旧STOP来歴を照合。公開完了報告の作者によるnative終了確認と、このtaskで行った公開表の構造検査を区別した。n48の保存削減率表示2.935924%／3.302597%は転記であり今回算出していない。

H6非相殺は旧eligible:null検出漏れを反映した。旧Dの有限値は取得済みsubset比較だけに使用し、全登録domainの比は未確定。H6高次PF独立確認6e197973…は別系列として所在と補足掲載候補のみ記録した。

## SOURCE・取得証跡

次の固定commitはローカルGit objectの存在を確認し、GitHub git/commits endpointのSHA/treeと一致した：

| 用途 | 固定commit |
|---|---|
| 最終静的監査 |bf8c19057fd82c900c6234152e152e79cb52e707|
| 主系列新A費用 |64c85f7a833b5950ef2ccae623708b73081edcd6|
| A取得／旧B-D会計 |fc6bd039dc7171d5674ac04f0b3ad427f1b6495b|
| H8希少層／選択効果 |cf8e93eef50333bef41f6d75803f6b819d679343|
| α感度 |343e0c591d297f99174459d750206b01bf7b9412|
| H6高次PF独立確認 |6e197973d4d4183d4e27b1b8cc95cb02f311bb92|
| H8確認設計 |434e67a8bb205711d22d5bf1aa5d143fd08b7765|
| H8承認起動 |d21647d07bc86343b292b1e1154ba05f887b1593|
| H8並列resume |be2e5475f4dbb5a07e947b053e8c419109f1eda9|
| H8最新完了結果 |e6eea5387a6e84d1e39a9adb3b61975acc74e200|

GitHubの既定branch recent-commit検索は古い履歴を返したため、指定commit endpointとH8公開branchを直接照合した。短縮SHAをgit/commitsへ渡した初回d216参照は404だったが、ローカルでfull SHAへ解決して再照合し一致した。未取得を読了済みと扱わない。

共通AGENTS、固定AGENTS、PROJECT_MAP、研究概要・現状、固定監査A〜D／E matrix、保存表と関連SOURCEを取得。長大な履歴索引は現行・規範・関連範囲を確認し、全歴史資料の再監査／全SOURCE closure再認定を行ったとは記録しない。各fileの取得・参照範囲、blob、SHA256、bytesとprivate snapshotは内部source inventoryへ保存した。

既存の未完了記述やSTOPを後から上書きせず、新しい保存値／完了結果は出典時点ごとに区別した。既存の大型索引は変更せず、この専用追補と新研究入口だけを追加した。

## 検証と保全

人工17検査PASS。異なる両軸単価・shotの正確な恒等式、単純平均が恒等式でないこと、整数shot、null／不適格／欠測の分離、paired共分散、丸め因子、実データ／cohort pooling拒否を検査した。人工専用moduleに実データloaderやproduction実行経路はない。作者本人の技術検査であり独立科学レビューではない。

全1,568索引行の転記・出典文字列一致、H8公開192軸の構造、42公開source snapshot、41 private入力roleのbyte保全、10固定commit/treeのremote一致を確認した。出力privacyとリンクもPASS。Git差分は新規成果物のみとし、既存fileの変更を含めない。技術検証の詳細receiptは内部に保全し、公開projectionにprivate path・有効grant・seed台帳・詳細運用hashを含めない。

新sampling/signal/SCF/DF/state生成/回路生成/compile/保存費用比・shot比・機構算術/新平均・SE/ランキング/rare Tier R/非Clifford集計/新科学レビュー/既存job操作/共有環境変更/公開pushは全て0。今回の人工入力算術とmetadataの件数・hash計算を科学データの新解析と混同しない。

## 次の担当とSTOP

**ユーザー**：Mechanism Contractとreadinessを確認し、対象scope、説明欠測の扱い、paired SE再構成の要否を含めて保存値数値解析を承認するか判断する。必要ならH6全登録非相殺・rare母平均・非Cliffordを追加要求するか別途判断する。

**Codex**：今回の準備を完了してmandatory STOP。追加承認まで数値機構表生成・再ランキング・科学実行・公開push・新科学レビューを開始しない。既存科学jobは停止しない。
