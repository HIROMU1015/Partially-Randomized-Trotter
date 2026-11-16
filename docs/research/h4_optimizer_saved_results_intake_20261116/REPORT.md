# H4方式別比較の更新と公開成果の定期統合

Full PFの保存集合内最安を**M5・要求幅16h、480,258,064 RZ**へ、PR-Pauliを**LD5・元κ×2、2873295767/7 RZ（410,470,823.86）**へ更新した。B0I p7は開始時のB公開branchで探索継続中のため、保存済み粗幅Yoshida4 A_SHAPE・**372,035,410 RZ**を使用した。旧S2の559,457,386 RZ、旧LD5/LD3配分、当時のPR探索予算は保持した。

主張は**指定した保存候補集合内の最安適合点**であり、母期待compiled価格・十分N・正式G・全候補winnerではない。方式別の最新値・差額・全候補は[C比較表](../h4_saved_rpe_cost_catalog_20261116/run01/REPORT.md)と[CATALOG](../h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json)を正本とする。本報告の上記数値は今回の固定受入時点の記録である。

## 今回受け入れた2点

|項目|Full PF|PR-Pauli|
|---|---|---|
|元公開|438d6c9a43351d45c31921a5676359dbc2c80b32|f01ce68ba918bf372acbefbf5152ccd9be5df390|
|configuration|70f7b31744ecd87fe277511bad7d1fc7dace9b1a097c315729c566f706c75edd|cb1aa7487fbfb938c629961014ea5911cc5f3e29b9fc45e16b872ac8fb60c052|
|公式・幅・分割|M5、要求16h、Full10|S2、要求h、LD5|
|X/Y共通配分|[20,25,26,24,16,20,17,15,12,11,8,7,2]|[48,45,42,39,24,27,14,20,16,16,8,7,2]|
|保存Q3|適合、元exact上下界を転記|適合、元exact上下界を転記|
|探索状態|Full PFの有限探索完了、p7は別作業|INTERRUPTED_SEARCH_POLICY_CHANGE、収束ではない|
|価格証拠|直接16軸＋条件付き閉包10軸、26単価の固定locator|26 MODEL_REFERENCEのcoverage summaryと保存exact総額。26個別単価は未保存/null|

Full PFは`C_RESULT_JOIN`から原API_RETURN・RESULT・PRICE_SOURCE_INDEXへ戻り、configuration/total/binding、配分、raw signal SOURCE、Q3契約・exact MSE、26軸のphysical/canonical ID・元単価・compilerを照合した。直接価格と採択済み固定compiler閉包価格の区分を継承し、新たな価格保証は付けていない。

PR-Pauliのκ絶対値は保存configurationの

`26225228882306721296844821347007773673162352858535179072118456320 / 539496659426909122096437389293855970805348475396548904610992201`

であり、`kappa_multiplier=2/1`とは別fieldに保存した。元κ×1のbase configurationとexactに対応させた。元のtotal view `f411ec466a47ba221376a9705afa569e9317db7c8aef0f9b287e33b5ddbf5173`、binding `38d4a055730b2ad61e5cc8a7c99f6fda06c359d74a60ec0b76a60a9e581f52d4`、law・signal packet・13roundのq/R/実step・配分を照合した。

### B2のexact値と、個別単価がないこと

公開`EVALUATIONS.json`の最安点1件、job `94f17683f6356a159708`に対応する、利用者が指定した保存OUTCOMEだけを読んだ。公開sha256

`124881075f189e15a028440c45b49d7d1104301912118435f2f97b07d83bfd74`

との一致を確認し、46,400 bytesをbyte一致で[B2_BEST_OUTCOME.json](B2_BEST_OUTCOME.json)へ保全した。元fileは変更せず、他の進行中jobのrawは読んでいない。これは新しいQ3ではなく、元OUTCOMEのexact MSE上下界・identityの保存である。

個別26単価はB2がtemporary quoteとして処理し、公開結果には残していない。したがって、`price_binding_level=SAVED_TOTAL_AND_26_MODEL_COVERAGE_SUMMARY`として、通常の26単価receipt受入と区別した。各round/axis/N/lawは`model_coverage_axes`へ接続し、`C_ref=null`、`price_axes=[]`を保持した。元κ×1の価格をコピーせず、価格規則の再適用・再積算もしない。

公開`NEEDS.json`の同candidateは`PRICE_RULES_MODEL_COMPLETE`、`MODEL_REFERENCE:26`、外挿26軸。D2 `c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa:configs/pr_price_rules/h4_saved_rules.json`の`PAULI_LD5`を参照する。元モデルはq8/R4較正、Pauli D2 lowering、scalar一回、保存fragment順・compiler/full wrapperを継承。`qualified_for_candidate=false`、全law E[C]/CI/Gは未認定である。

保存総額同士の差は、Pauli−Full PFが`−488510681/7 RZ`、Pauli−p7が`269047897/7 RZ`。採用された価格規則での数値比較であり、未知のモデル誤差を0とする順位保証ではない。

## 継続運用

既存の`Catalog.add/make/result`、Git blob `Reader`、価格・Q3別着のadapterを再利用した。追加したのは公開refの検出・既存adapterの呼出し・段階依存の文書更新・利用者自身のcron接続だけであり、optimizer・decoder・価格計算器を実装していない。

- `scripts/sync_saved_rpe_publications.py`：公開branchをfetchし、固定commit/path/blobで成果を検出。既存のC_RESULT_JOIN、C_COMPARISON_JOIN、価格HANDOFF、SUMMARY、ROUND_RESULTを確認する。対応schemaを集計し、未対応・欠測・結合不一致を[SYNC_STATUS](../h4_saved_rpe_cost_catalog_20261116/SYNC_STATUS.md)へ残す。1件の失敗で他成果を止めない。
- `SYNC_CONFIG.json`の担当branch群とbranch名patternを監視する。新しいcommit/pathの利用者転送やSkill起動は不要。同じ系列名patternの後続branchも検出する。無関係な全projects/rawの走査は行わない。
- `PUBLICATIONS.json`は各固定locatorと受入/参照/保留理由を記録する。既存の保存入力へ固定sourceを追加するため、通常の集計コマンドでもネットワークなしで再現できる。
- `RESEARCH_ROADMAP.md`はC branch内で現在地・進行状況・次工程を管理する。目的・採択済み精度規約・証拠の区分は旧正本ad6ab45bbから保持し、古い現在地と費用の重複記載を置き換えた。費用・point ID・HEAD・時刻の変化だけでは本文を更新しない。採択待ちの提案を自動採択せず、異なる評価契約は保留する。
- 利用者自身のcronを10分間隔で使用する。systemd user busと既存公開後hookは利用できず、稼働中のcron daemonを再設定せずに使う。共有OS設定・他担当worktreeは変更しない。専用lock、dirty checkout時のskip、通常pushとremote SHA照合を使う。

初回の16公開branchで65個の集計入口を検出し、51候補（適合37・不適合13・価格待ち1）を保持した。旧sourceの別HANDOFF再掲も重複登録しない。未対応schema等は27入口で、旧較正/単path/診断など「現在の比較点へ独立登録しない資料」も含む。保留は科学的FAILではなく、具体的なschema・原recordとのbinding不足である。最新の件数はPUBLICATIONS/SYNC_STATUSを参照する。

対応済みschemaは自動で反映される。未対応schemaはCのconsumer拡張、exact元結果が公開されない場合はproducerの固定公開、比較契約の変更は採択済み文書・設定への反映が必要である。これらを自動科学実行で埋めない。

## 検証

保存カタログ14、公開検出/順序独立/冪等性4、cron人工4の計22テストPASS（[TESTS.txt](TESTS.txt)）。Full PFと中断Pauliの原SOURCE調査はprimary agent、B2最安点のmetadata・hash調査とcron実装/人工4件はsub-agent、統合と残る検証はprimary agentが実施した。独立した科学レビューとは呼ばない。

3例を検証した：①Full PF完了点、②PR-Pauli途中終了点、③粗幅Yoshida4のQ3→価格後着。③は逆順でも同じ1点となる。公開refからの検出、保存SOURCEの受入、同じ入力の再実行で候補・費用・fileが増えないこと、費用だけの更新でロードマップ本文が不変であることを確認した。未知/missing sourceは保留し、他の2点は反映できた。

新signal/sampling/compile/fit/Q3/配分探索は各0件。旧SOURCE・raw・凍結モデル・Q3判定・STOPは保全した。定期同期は利用者が明示的に割り当てた保存成果の統合だけを継続する。科学実行への自動進行は行わない。
