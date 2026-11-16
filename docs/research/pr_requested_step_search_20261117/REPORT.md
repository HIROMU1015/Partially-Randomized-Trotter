# PR-Pauli / PR-DFの要求PF幅を通常RPE探索へ接続

2026-11-17 JST。実装SOURCEと基点は `SOURCE.json`。専用branchは `codex/a-pr-step-width-search-20261117`。新しい実H4信号・path sampling・compile・価格fit・実Q3・round05本番は実行していない。

## 完成した経路

通常入口 `scripts/run_rpe_schedule_search.py` に複数要求幅を宣言すると、方式・L_D・公式・exact要求幅・Kをfamilyとして既存SCHEDULE_WIDE規則でκを選び、共通Completionへ渡す。入力が揃った候補は同じMETHOD_MINIMUM_PRIORITY_V2・shot配分・Executor・Q3応答検査へ戻り、その候補自身の13 round信号と26軸価格で比較する。保留幅があってもREADY候補は進む。

要求幅はhやTを変更しない。Pauli/DFの固定mapperがq、actual Δ=T/q、R、allocation、lawを決める。要求幅のraw値とexact表現を保持し、binary64 `0.2` と厳密 `"1/5"` を別familyにする。κの境界・評価点は幅間で転用しない。全物理入力・辞書・SOURCE・law・評価契約が同じscheduleだけ、既存登録機構に厳密照合を追加してaliasとして扱う。requested configuration IDを別IDへ書き換えず、`evaluation_configuration_id` を別途記録する。

保存基準構成のない分割は、同方式の固定templateをmetadataの基点として、既存のdescriptor/dictionary準備入口へ渡す。対象LD自身のλ・辞書・mapperからfamily定数を得る。templateのκは定数を読むためだけのmetadataであり、科学評価候補のκを採択しない。欠測descriptor/P0は具体的な準備要求として残す。今作業でそれらのH4科学leafは起動していない。

## 別幅の価格を探索へ戻す処理

`pr_step_price_registry.py` が、既存Pauli auditまたはDFのfrozen model/独立holdoutを検査する。価格の適用domainは方式固有の実装契約、H/state/snapshot、split・辞書・law、signed pattern、exact actual step、検査したwrapper時間、較正q/R範囲を含む。q/R外挿とevent時間変更は各軸に明示する。直接価格は同じ物理点にだけ使う。係数を別LDや別実幅へ移さない。

- Pauli：Cの既存実step別要求を使い、選択された各cellについて従来driver → P1/P2/P3/P5/P6監査 → 固定receipt付きpackage → run専用の部分価格登録表へ自動接続した。監査用metadata adapterは該当roundだけのwrapper時間・価格を作る。既存数値監査helperは無変更。
- DF：同じactual stepの欠測roundをまとめ、既存calibration/acquisition/freeze/holdoutを再利用する。scoped modelは対象roundだけを出力する。1物理点しかないgroupはfitや外挿を行わず、`DIRECT_FINITE_COHORT` としてreceiptからの2軸直接価格を返す。rankや独立holdoutが不足するgroupは必要条件を返す。
- Completion：返ったpackageを取り込み、価格証拠の更新に応じて待機候補を再照会する。部分価格を全26軸へ広げない。自動取得経路では利用者による結果の転記を要しない。外部から同じ固定packageを受ける入口もある。既存科学FAIL/曖昧な起動/STOPは再開しない。

## 保存資料で確認した範囲

`SAVED_ACCEPTANCE.json` と保存receiptを読む回帰testが根拠。

|確認|範囲・結果|
|---|---|
|hのみの宣言|PR-Pauli 9構成、PR-DF 9構成のcatalog候補・単価・保存point IDが従来と同一。両mapperで基準configuration全体を再現。原round05設定はbyte同一|
|Pauli round04|LD1/2/3/5、κ=2のbinary64 .2でm0価格再利用、m1–12を各分割1 cellへ集約。計4 cell。保存request ID・action・configuration IDを再現|
|Pauli exact 1/5|元値を保持しm2–12の実幅非binary64表現を理由付きで返す。m0/m1の既存価格再利用を維持|
|DF mapper|LD7/S2/K2でbinary64 .05を受理しq=2×2^m、actual step=.05。binary64 .2は元mapperの非整数PF格子条件で拒否、exact 1/5は非binary64入力として拒否|
|監査結果の読取|公開済みPauli LD4 auditとDF LD6 model/独立holdoutを同じ受入APIで読み、元26軸単価をexactに再現|

上表は保存metadata・価格規則の技術的照合であり、新しい信号診断や科学的優劣の判断ではない。

## 人工入力で確認した範囲

検査結果は `TESTS.json`（unit 80件＋既存schedule 7検査がPASS）、詳細は `checks/`。通常CLIのmainから、両方式の複数幅宣言 → 不足入力/保存再利用 → 監査済み価格の取込 → 既存V2のshot提案 → 既存人工workerのsubprocess/OUTCOME検査 → 方式別費用表まで通した。別幅が安い人工例ではPauli .2、DF .05を選択でき、同run再開の完了済み処理/Q3起動は0だった。

科学leafだけを人工化したCompletion検査では、実際の部分registry・receipt検査・候補再判定・installを使用した。Pauliの複数group driver orchestration、DF直接価格、audit失敗拒否、旧SOURCE/辞書/target混入拒否、部分26軸、物理alias、descriptor欠測、旧STOP保全も検査した。人工結果は本番catalogへ登録していない。旧Pauli経路に新auditを無条件要求する回帰を発見し、scoped経路だけに限定して旧3 testを保全した。

`entry_plan/ENTRY_SUMMARY.json` は、両方式・複数幅を指定した実際のH4通常CLIの **plan-only** 検査。計画のmetadataを保存しただけで、信号・価格較正・Q3の起動は0。計画専用2回目はさらに8候補のmetadata 32 stageを準備し、完了済みstageの再起動は0。全分割の広い初期集合を列挙した先行technical checkは途中で対象を絞り、部分metadataをrun01に保持した（INITIAL_BROAD_CHECK.json）。科学runの中断やretryではない。冗長なrun全体は所有worktreeに保持し、公開は要約とSOURCE付き入力に限る。

## 再利用・並行作業

基点B2 d8e806265はD2 98b40e439の共通Completionを統合済み。C 02e7cc6a1の実step別要求、D2 c1f95e080/7cd8ea139のPriceRules、A 9f00ddd86の辞書/descriptor/信号入口を再利用した。取り込んだ無変更ファイルは `REUSED_SOURCE_FILES.json`。新しい科学アルゴリズム・候補管理framework・approval証明機構を追加していない。

並行高速化成果 e839b0d3f（実装a50aadc1f）のHANDOFFを読取照合した。当branchには並行queue/DP実装を重複移植していない。共通変更箇所は `WidthDriver.batch`、`Completion.install/continue_selected_preparation`、`rpe_method_optimizer`。統合時は高速化側の共通executor/optimizer factoryへWidthDriverのcontext/routesを渡す。候補ID・評価契約・stage receipt・価格登録domainを保つ。具体的差分はHANDOFF。

## 実H4で次に必要なもの

方式別の保存一致roundは既存入口で再利用できる。新幅の欠測signal、新実幅の較正cell/独立holdout、その候補に対する実Q3は未取得で、対象menuとSOURCEに対する別途実行許可が必要。今回の人工的な費用順位や正常終了を、実H4のwinner・精度達成・十分N・global RMSEの根拠にしない。技術成果の公開・remote照合後、mandatory STOP。
