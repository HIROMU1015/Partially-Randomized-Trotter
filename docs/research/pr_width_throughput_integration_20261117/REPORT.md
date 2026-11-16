# PR複数幅・共通queue／DP再利用・B0I自動p選択の統合

2026-11-17 JST。通常入口`run_rpe_schedule_search.py`へ、複数要求幅のPR-Pauli/PR-DFと、B0I未評価pの選択を接続した。PRの初期READYと監査済み部分価格の到着後READYが同じqueue・予算・job記録を使い、実際のDP prefix HITと逐次参照のexact一致を人工入力で確認した。B0Iは固定H4 metadataからp無指定で**p9／標準Yoshida4／32h/5**を選び、人工応答で補完→shot→Q3→方式別結果→再開まで通した。科学的な最安幅・最安pを認定した結果ではない。

新しい実H4信号・較正・sampling・compile・実Q3・round05本番は**すべて0件**。実H4入力の起動はplan-onlyに限定した。人工成果は本番catalogから分離し、旧run・SOURCE・STOPを保全した。

[HANDOFF・起動/再開](HANDOFF.md)／[SOURCE](SOURCE.json)／[検査一覧](CHECKS.json)。

## 固定成果と差分の再利用

|役割|固定commit|
|---|---|
|共通B2基点|`d8e806265c3ded8e1184181c37c2d7ec71f08ade`|
|PR複数要求幅|`d69516191f3126a076b23bc55f6818b3f866f10e`|
|共通queue/DP再利用|`e839b0d3f5469c21f5944aec5d978f91d193efc8`（実装`a50aadc1f430e8c8ee6f8917a8024f1121ec913a`）|
|B0I自動選択|`5efa6e97faf5906320c14d6fff58220f259b6670`|
|B0Iの元D provider|`36ad751d4ebf0ec160532615da6425b268a0c18a`|

着手時に関連branch・後続公開ref・両HANDOFFを照合し、同等の統合は確認されなかった。複数幅版を基点とする専用branch`codex/pr-width-throughput-integration-20261117`で、必要なコード・テスト・固定入力だけを選択的に取り込んだ。元B2、幅担当、高速化担当、Dのworktree/runは変更していない。各版の科学契約・precision・price qualificationを変更していない。

## 統合した実行経路

`WidthDriver.batch`にあった旧`O.Executor`直接生成を、通常method CLIと共有する`make_executor`へ変更した。scheduler、並列数、DP workers、prefix cache、prior結果再利用を同じ設定から解決する。`continue_selected_preparation(..., optimizer=...)`へ同じCompletion/context/routes/executor/optimizerと累積予算を渡し、戻り時に旧optimizerや別queueへ落ちない。完了後はfinallyで所有executorを閉じる。

外側WidthDriverが幅・L_D・κを選び、内側V2はshotを選ぶ構造を維持した。既存の`kappa_neighbourhood`抑止スイッチを使い、準備後候補に対して第二の内側κ chooserを起動しない。新しい探索規則は設計していない。適応HYBRID・shot振替・準備待ち中の先行Q3を統合のために既定有効化していない。元単一幅WideDriverの任意設定は保持する。

追加の技術確認で、幅版が明示的な`overlap_ready_batch=true`を無視している箇所を発見し、明示設定時だけ対応した。準備前に固定したREADY contextをQ3へ渡し、Completionが作業する間は共有stateを読み直さない。準備futureを待ってから新READYを同じexecutorへinstallする。初期READYと後着候補は同じqueue invocation/予算を共有し、UNKNOWNなら後続Q3や継続処理を開始しない。既定false、plan-onlyでは先行実行しない。

要求幅は元の表現とexact値を保持し、mapperのq・T/q・Rを使う。binary64 `0.2`と厳密`"1/5"`は別family。requested configuration IDとinstalled evaluation IDを分け、同一性が成立するschedule aliasだけ既存機構で集約する。価格のpartial registryは方式・辞書・law・実装・実step・検査済みwrapper domainを保持する。別幅/分割へ係数や価格を移送しない。

PFを指定した同じ通常入口は、既存method CLIの`round_mode`へdispatchする。B0Iのp選択・D補完・shot生成・Q3・比較は公開済みの`pf_candidate_selection`等を使用する。C catalog readerとPF registry readerの差分も取り込み、人工recordを科学consumerへ受理させない。元round05の単一PR-Pauli入力はWideDriverへ、複数幅宣言はWidthDriverへdispatchする。

## 受入結果

|検査|結果と記録|
|---|---|
|PRの初期READY＋価格待ちからのREADY|両方式で各2job、計4 ENQUEUE/LAUNCHが**1 queue invocation**を共有。scoped監査/receipt検査・実Completion refresh/install・累積予算各2を確認。[queue](checks/queue/SUMMARY.json)|
|実DP途中状態|人工4構成・8job中、後半m12だけ異なる4jobで**HIT、各12 round再利用**。donor EXITがsuccessor LAUNCHに先行。[prefix](checks/prefix/SUMMARY.json)|
|逐次参照|同じ8入力をlegacy・1 slot・prefix無効で実行。concurrent・2 slot・prefix有効とMSE上下界/判定がexact一致。異configuration/異signalはHITなし。[比較](checks/prefix/CLI_SUMMARY.json)|
|PR再開|同じCLIで完了済み信号/価格leaf/Q3の追加起動0。保持STOPは開かず、未対応幅が他のREADYを妨げない。[queue/prefix記録](checks/queue/INVOCATIONS.jsonl)|
|先行Q3の明示設定|3検査PASS。準備待機中の人工Q3、後着候補も同じqueue、既定無効、非空input_requestsを伴うUNKNOWNでも継続しない。[log](checks/OVERLAP_INTEGRATION.log)|
|B0I保存入力|固定H4入力・p無指定でp9/Y4/32h/5、ID`9a495a223e254747a34f6b31f2457bf7b0498578f1dc49053f915652a92494f2`。D planのみ、Q3起動0。SELECTION/REQUEST/RESPONSEは再開で不変。[B0I](checks/b0i/SUMMARY.json)|
|B0I人工応答|legacy/concurrentの双方でprovider1回→通常shot seed→人工Q3 1件→方式別結果。同run再開のprovider/Q3追加0。p9を規則へ埋め込まず、公開scopeから選択。[検収](checks/b0i/artificial_concurrent/CHECKS.json)|
|旧経路/保存入力|保存9＋9構成、round04の8要求→4 cell、DF mapperを再現。元round05由来は開始/再開とも70点、再開0起動。幅版plan-onlyは開始16候補/70 metadata stage、再開で8候補/32 stageを追加し、旧stage再起動0。[plan記録](checks/plan/PLAN_ACCEPTANCE.json)|

DP検査は、通常queueとQ3 adapterのprefix contextに、元guard付きDPを人工tableで接続したもの。分子信号や実科学routeの性能測定ではない。人工numerical leaf/test interpreterを使った範囲を公開fixtureに明記した。小さいholdout誤差・人工費用・人工速度から科学資格を与えない。

## 回帰・実行環境・限界

結合/回帰84件、元prefix回帰10件、先行Q3の結合3件がPASS。人工DP結合2ケース、B0Iの保存/人工各経路もPASS。逐次・並列比較、実HIT、予算、再開、SOURCE/law/価格誤用拒否を既存テストと最小差分で検査した。controllerは既存Python3.12、固定DP/route用環境は既存Python3.11.0rc1。既存AST/hash guardや数値閾値を緩めていない。

SOURCEの相対出力制約に従い、実H4 plan-onlyのrunは所有integration worktree内、D planは所有する固定36adのview内へ保存した。最初の/tmp配置は科学call前に拒否され、その技術setup記録は保全した。保存計画の入力2JSONを固定5efaからbyte一致で収録し、元の回帰ケースも実行可能にした。

既完了Q3の再利用は新規起動0。prefixを用いて別配分を評価するjobは新規Q3予算1件を消費する。cacheの保存/復元は科学状態の再利用であり、新しい実行許可ではない。新しい実H4の幅探索・価格較正・Q3・最安候補選定には別の科学実行範囲が必要。

rootが統合glue・回帰と公開を担当し、担当sub-agentが深いCompletion/queue検査、実DP人工検査、B0I保存/人工検査を分担した。作者検査と別担当の技術確認を区別し、重要な独立科学レビューを行ったとは主張しない。公開照合後mandatory STOP。既存のC catalog定期同期の運用には介入していない。
