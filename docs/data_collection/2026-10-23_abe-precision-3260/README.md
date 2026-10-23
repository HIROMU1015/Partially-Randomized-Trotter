# PR資源評価：計算環境別の既存証拠索引

2026-10-23 JST。調査hostname：`abe-Precision-3260`。
Repository：`HIROMU1015/Partially-Randomized-Trotter`。
専用branch：`codex/pr-paper-data-abe-precision-3260-20261023`。
状態：**ローカルcommitまで。今回の9ファイルの公開承認は未確認、pushなし、mandatory STOP。**

既存のPR資源評価結果、SOURCE、条件、監査、不足入力を固定commitに結び付けた。
原科学表を再登録せず、保存数値、単位、候補順位、cohort、科学SOURCE、grant、ledger、marker、STOPを変更していない。
新しい科学計算、原実験の再実行、数値再解析、再ランキング、論文執筆は行っていない。

## 論文データ整理担当の入口

既存の公開データは以下の固定commitから取得する。既定branch `main` の先端は今回の最新結果の入口ではない。

|用途|既存の固定公開参照|
|---|---|
|H4/H6/H8の論文用機械可読manifest・採用範囲|[paper_data_manifest.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d4ff27cb630939f87e08ab5881404c72ad2fa19/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/paper_data_manifest.csv)|
|保存主表・negative resultを含む採用案|[main_table_proposed.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d4ff27cb630939f87e08ab5881404c72ad2fa19/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.csv)|
|新A探索n8の保存費用表|[comparisons.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d4ff27cb630939f87e08ab5881404c72ad2fa19/artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/comparisons.csv)|
|H2独立Pの公開projection|[confirmation_result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2d1a86c431f835db700f348ce618d8604e7ee23/artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/confirmation_result.json)|
|H8独立n48の保存比較|[comparisons.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d4ff27cb630939f87e08ab5881404c72ad2fa19/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv)|
|旧H4技術pilotとexact SOURCEの公開来歴|[v5結果・監査入口](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_ax2b_h4_pilot_v5_result.md)|
|旧H6のMemoryError停止と欠測|[matched v2結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_h6_matched_accuracy_result_v2.md)|
|旧H6補完runのWORKER_EXIT停止|[supplement結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_h6_matched_supplement_result_launch_v2.md)|
|現行S2の保存値と未回収入力|[現行packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d4ff27cb630939f87e08ab5881404c72ad2fa19/docs/progress/track_a/figures/pr_pf_s2_theory_compiled_main_20261022/README.md)|

表中の主表案や採用記録は各原資料の承認scopeを保持する。この索引作成で新しい掲載判断をしていない。
新しい集約ファイル自体は公開承認待ちのため、上記GitHub参照と、このローカルbundleの公開状態を区別する。

## 今回の9ファイル

|ファイル|内容|
|---|---|
|[README.md](README.md)|固定公開入口、調査範囲、制約、公開状態|
|[evidence_index.csv](evidence_index.csv)|既存429ファイルのcommit・path・Git blob・SHA256・bytes・取得検証段階|
|[series_manifest.json](series_manifest.json)|17系列のmodel、geometry、basis、DF政策、split/時間条件の参照、cohort、SOURCE、限界|
|[source_result_links.csv](source_result_links.csv)|原JSON fieldまたは報告の明示SOURCEと保存結果の対応。関係を新規認定しない|
|[required_input_candidates.csv](required_input_candidates.csv)|既存回収依頼の27候補を文字列転記。候補追加・値補完なし|
|[missing_evidence.csv](missing_evidence.csv)|O00002、元preparation、shot原表、host来歴、private監査等の不足と担当|
|[definitions.md](definitions.md)|bias、allowance、shot、費用、cohort、欠測の読み方と規範出典|
|[verification_receipt.json](verification_receipt.json)|固定commit/tree、実取得8ファイル、SOURCE163件、保全・公開検査のscope|
|[SHA256SUMS](SHA256SUMS)|上記8ファイルのSHA256。自身は対象外|

CSVは結果値の新しい統計表ではなく、所在・来歴・検証状態のmetadata表である。
SHA256は固定Git blobの原bytesから計算した。`source_result_links.csv`のSOURCE記述は原資料のfieldまたは行番号へ戻って解釈する。
SOURCE commitが存在することと、全入力・全SOURCE closure・科学条件の再現が確認できることは異なる。

## 調査範囲と環境帰属

対象repositoryに登録された211 worktree・204種類のHEADを静的に調べた。
`artifacts/resource_applicability/` のtracked path/blobの全組は、確認した2つの公開snapshot
`2d4ff27cb630939f87e08ab5881404c72ad2fa19` と `054bb7b2b2ca788b47ebdc5c14c115f24c8b6379` の和集合に含まれた。
これはこのprefixと観測したHEADの範囲の結論であり、repository全履歴・他環境・全private保存物の網羅証明ではない。

untrackedの同prefixでは1,861ファイルをbyte照合し、1,838件は上記公開snapshotと同じpath/blobだった。
残る23件は複数worktreeのコピーを含むlog差分、監査helper、旧編集metadata、運用registryである。
これらをGitHubの全branchに存在しないと断定していない。未加工log、運用registry、2つのruntime cache領域は公開候補へ取り込んでいない。
既に公開されている失敗結果・terminal・coverageからnegative resultを参照する。

このhostnameで既存資料を保持・照合した事実と、各元科学計算をこのhostnameで実行した事実は分けた。
今回、原runごとのhostnameを一律には再確立していない。別環境から取得したH2/H4/H6/H8の公開projectionを、このhostで新たに実行した結果とは記載しない。
所有する研究用保存領域での前段O00002探索も参照したが、原設定と実行rootのtoken mapは未回収である。
内部絶対パスとprivate原本の詳細hashは公開候補に含めない。

## 検証の意味

GitHub connectorで固定50 commitのSHA/treeを取得し、ローカルcommit/treeと照合した。
主要な結果表・監査8ファイルはraw bytesを実取得し、SHA256とGit blobがローカル原本と一致した。
その他の行は確認済みcommit/treeのローカル固定blobを照合した段階であり、全429ファイルを今回remoteからダウンロードしたとは記載しない。
`github_status`列で実取得、commit/tree照合、公開snapshotの祖先SOURCEを分離する。

旧H4 v5の原公開manifestにある163 frozen SOURCEを、指定された公開commit/pathのSHA256と照合した。
実行時はdirtyなfrozen bytesを使った履歴を保持し、後日のSOURCE公開commitを実行時HEADへ置き換えない。
保存したSOURCEとhashの一致は、原実験の再実行、外部再現、immutable CI、独立科学レビューを意味しない。
scientific moduleのimport、NPZの科学的decode、compiler実行、sampling、shot再会計は行っていない。

## 時点・cohort・未回収資料

H2はS_STATS時点のP未完了記述だけで止めず、後続の固定P完了projectionを別cohortとして索引化した。
H2 outer32取得報告は元H2 A/Pとは別系列として扱う。
H4 legacy DF12とtol-only DF10、H6 DF19、H8 DF28、探索n8、旧union/fixed-anchor n32、独立確認を結合しない。
旧H6のSTOPは後続の別runや別cohortの完了で上書きしない。

現行S2 packetは原preparation等が不足しており、現在の理論Gは0/20条件のままである。
27候補の順序付き基底操作・対角係数・scalar・完全なPR lawと、H6/H8新Aの対応shot原表を推測で補っていない。
O00002のprivate設定原本は未発見で、SHA256の一致・不一致を判定できない。公開projectionをprivate原本と同一視しない。
H8のprivate詳細監査やH6非相殺の全登録domainについて、この索引で新しい充足宣言をしていない。

## 公開候補とSTOP

新規公開候補はこのdirectoryの9ファイルだけである。主な開示情報は、依頼で指定されたhostnameと、既存公開資料のpath・commit・hash・不足状態の対応である。
private config、認証情報、seed値・seed原本、内部絶対パス、private raw/archive、非公開grant/receipt、個人情報は含めない。
既存公開snapshotを基点にした専用ローカルcommitを作成するが、新ファイルの公開承認は未確認のためpushしない。

次段階（提案・未実施）：**ユーザー**がこの9ファイルと開示範囲を確認し、公開branchへのpushを承認するか判断する。
承認後の**Codex**はこのmetadataだけをpushし、remote commitとファイルbytesを照合する。
不足原本の回収は**元実行環境の保持者**が、同じSOURCE・候補・cohortの非機密なreceiptまたは承認済みprojectionで対応する。
いずれも今回自動実行しない。**mandatory STOP。**
