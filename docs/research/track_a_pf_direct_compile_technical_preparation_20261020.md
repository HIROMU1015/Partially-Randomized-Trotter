# PF直接compile：技術準備と実行境界

2026-10-20（JST）。**技術準備段階の実装・人工検査を完了。本番取得は未認可・未結合。**
ユーザーが[計画db06128a](track_a_pf_direct_compile_expansion_plan_20261019.md)を研究目的に適合と判断し、技術準備へ進む指示を与えたことを記録した。[技術packet](../../artifacts/resource_applicability/track_a_pf_direct_compile_technical_preparation/2026-10-20/README.md)を参照。

788軸は保存計画の提案のままで、取得範囲の承認ではない。実際の選択は`selected_task_ids=null`、選択済み軸0、有効grantなし、科学SOURCE/入力/実行環境の最終sealなし。旧profile等を照合すべき48軸も自動追加しない。今回、実入力decode・分子回路生成・control数値検証・compile・sampling・fit・費用再会計は全て0。

## 実装したもの

- [metadata契約](../../scripts/pf_direct_compile_contract.py)：固定788提案のbytesと候補行digest、DF系列を含むtask ID、完全なcosine/sine組、厳密な選択集合とcall予約数、科学実行専用の承認種別、SOURCE/input/provider/runtime/profile結合を検査する。研究上の適合判断や公開承認だけでは科学実行を認可しない。
- [取得adapter](../../scripts/pf_direct_compile_adapter.py)：既存providerの`build_deterministic_native`、`build_native_hadamard_wrapper`、fingerprint、`transpile_and_measure_cost`を受け取る依存注入型API。保存済みの同じ入力・preparation・自身のprefixに対応するscalar・独立control oracleを要求する。新しい分子入力生成・signal格子・shot会計・候補選択・モデル評価は含まない。
- `Journal`は専用出力の再利用を拒否し、許可callを呼出し前に予約する。失敗したcosineの予約を返却せず、sineへ進まない。STOPは当該pairの専用出力に保存し、無関係なtaskの共有STOPを作らない。
- 実測にはRZ/CX/depth/RZ_depth/CX_depth/size、全gate counts、compile前後fingerprint、compiler hashを含める。低level APIからnative exit/reapを成功と推測しない。実workerの終了証拠とresult digestを検査する別の受入関数を設けた。
- 資源admission関数は、与えられた実効空きmemory/FSと許可CPU、taskごとの見積りから投入候補を決める。固定worker/CPU/RAM/q上限を置かず、未知のfootprintは推測で埋めない。実worker schedulerや資源観測はまだ起動していない。
- [metadata preflight](../../scripts/check_pf_direct_compile_preparation.py)は案と原本SHAの照合だけを行う。`--execute`などの科学起動CLIは提供せず、未知の引数を拒否する。

旧H2/H4/H6/H8 driverはそのまま起動しない。H4 setupには科学primitive/action検査があり、H6/H8 driverには固定候補・旧grant・旧call上限・モデル誤差2% STOPが含まれる。今回これらを削除・変更せず、新しい取得用adapterの接続候補と、呼ばない旧driver処理を[静的対応表](../../artifacts/resource_applicability/track_a_pf_direct_compile_technical_preparation/2026-10-20/backend_binding_DRAFT.json)に記録した。

新adapterはモデル式を評価しない。保存予測から外れたという理由だけで技術的に正しい直接観測を失敗扱いにせず、将来の別解析へ保存できる。これは新しい取得契約の設計であり、旧契約の2% STOPを遡及変更するものではない。

## 検証した範囲

[人工検査](../../tests/test_pf_direct_compile_preparation.py)はstdlibとfake providerだけを使う。架空の契約・小さい整数metrics・一時directoryで、承認前のfactory/input/output到達拒否、公開承認との区別、候補改変・半pair・DF系列混同・無許可call・file変更の拒否、control/compile失敗時の停止と予約保全、fingerprint・整数metrics・終了証拠、q16保持、動的admissionを検査した。科学packageのimportを拒否した別Python processでも新moduleのimportが成功することを確認した。

これは実分子やQiskit circuitを使ったcompile検証ではない。fake providerの成功をH2/H4/H6/H8での制御意味論・性能・compiler互換性の確認と読み替えない。別AIによる独立レビューではなく、主担当自身の技術検査である。

## 本番結合前に残る事項

[readiness境界](../../artifacts/resource_applicability/track_a_pf_direct_compile_technical_preparation/2026-10-20/readiness_boundaries.json)に完了/未開始を分けて保存した。

1. ユーザーが実際の取得対象pairと、48軸の再照合/再取得の扱いを決める。現行722軸と別legacy66軸を区別し、788軸全量を暗黙に選択しない。
2. Codexが選択範囲に対して分子別の保存入力loader・preparation復元・scalar/control oracleを最終結合する。新しい独立oracle等が必要ならcall範囲へ明示する。既存hashから実ファイルの存在やprofile一致を推測しない。
3. owned worker/spawn・native wait/reap・SOURCEのGit blob照合・実入力/provider/runtime/profile bytes照合・直前資源gateを結合し、最終SOURCE/出力/予約数をsealする。今回のcallback受入口だけを本番witness完成と呼ばない。新adapterの無制限instruction allowanceを既存providerで扱えるかも実行前の技術確認事項である。
4. 科学実行の明示承認を得て、新しいgrantとone-shot領域で実行する。計画確認・技術準備・公開承認は代用にならない。

最終の取得範囲が未決定なので、今回は実入力復元・本番loaderの動作確認、production dispatcher、科学実行sealまで進めていない。旧SOURCE・原データ・raw結果・grant/ledger/marker/STOP・旧論文主表/Figure 2は不変。詳細科学レビュー・最終evidence freezeは開始しない。

**mandatory STOP。** 次の担当は、取得範囲を判断する**ユーザー**。**Codex**は範囲決定後の最終技術結合・sealを担当し、別の科学実行承認までcompileを開始しない。
