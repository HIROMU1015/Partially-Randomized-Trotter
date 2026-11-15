# C2 LD2 m9：観測前SOURCE準備

ユーザーは2d513c案Bを採択し、m9専用入口・重複照合・2予測の計算/観測前公開・fresh2 path・
最大4 compile・2軸のexact比較と公開を明示許可した。本書は科学SOURCEと技術差分の説明で、承認証明ではない。

元13abはC2専用detached checkout `c2-pauli-m9-source-13ab` から無変更で使用する。
D2 PR-DF worktree/jobは使用・変更しない。旧validatorを緩和せず、新driverでm9だけを正確に束縛する。
元CAL planのfrozen_model_gate→canonical round_request/price_context→既存acquire_path/
pilot_core/normal sampler/native/wrapper/compiler→compile_receipts→canonical G.checkを接続。
新driverとtestはC2 branchへcommitし、実行SOURCEとraw公開SOURCEを分ける。

§10.5：計画後のC2 tip=2d513c、D2 Pauli tip=705f59ea8。関連m9/LD2の公開履歴に同等予測・raw/receiptなし。
同ユーザーの関連processに同一jobなし（検索自身除外）。C2 public出力と専用privateは未存在だった。
旧705fの監査を引継ぎ、B0/W/CAL/holdoutは元用途のまま保全。同名新cohortの保存価格を開いていない。
この範囲で再利用可能な独立m9価格は0、予定新規4 compile。外部非公開実行まで未取得と断定しない。

元packet714247のprefix2/m9、元b74eのX/Yモデル、043cの旧LD2 CAL/holdout seedとW seedを照合。
新cohortはD_PAULI_LD2_M9_LONG_NORMAL_N2_V1、index0/1各512 step。新1024 seedは全てunique、
旧cohortとoverlap0。metadataだけを読む既存canonical経路を使い、H4 sampling/buildはまだ0。

性能：独立2 pathをprocess並列。小さい256次元matvecにはBLAS1 thread/processを選び、
大きなmatrixに不向きなthread起動を避ける（OS設定や他jobは変更しない）。元event_matrixを完全なevent JSON・
labels・nでworker内memoizeし、同じ行列の重複生成を削減。元oracleの行列積順は変更しない。
既存h46_b3_fast_fingerprintの同一byte writerを元coreのcontextへ局所的に接続し、
wrapper内部のcanonical再hashも同じcontextに含める。sampling、D2 emitter、simulator、compiler、gate閾値は変更0。
2軸は既存core順にcompileし、各path最大2（計4）を超える呼出しを拒否する。

4 test PASS。toy4-modeの旧core/新contextでprices/fingerprints/events/event_metadata/scalar/seeds/checksが完全一致。
元receipt exporterまで人工接続しwrong-axis/SOURCE/seedsを拒否。m9 bindingと新旧seed重複、
旧validatorのm9拒否、欠測/0分母・軸別exact算術、実observerのtimerと子exit7記録を確認。
fake compilerのみで、実H4 build/compile/価格試験ではない。provider-first import preflightもscience0で確認する。

予測公開commitがremoteで解決できるまでsamplingしない。実行はfresh private/public一回限り、
no retry。両pathは開始済み独立workerとして終了まで記録し、失敗を補う再sampling/追加compileはしない。
部分cache/QPY、進捗・子exit/signalを保全。数値許容差なし、モデル判定なし。取得後の原rawはlossless gzip JSON。
新実行の結果は旧m5/m6やb74eを変更せず別保存し、公開/比較後mandatory STOP。
