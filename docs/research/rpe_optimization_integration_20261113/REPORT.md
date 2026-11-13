# PR・baseline・shot配分・compiled価格会計の技術統合

2026-11-13。**AのPR signal、Cのbaseline signal、Bのshot配分・cost/selectを接続する薄いadapterを実装した。** 人工Hamiltonian/mean/N/価格だけで4方式の全round接続と費用比較を確認した。実H4のsignal・保存値再解析・MC/N最適化・compile・E[C]/G・科学的候補選定は0。

[接続仕様](INTEGRATION_HANDOFF.md)／[人工・回帰テスト](TESTS.txt)／[人工結果](ARTIFICIAL_E2E.json)／[次の有限科学案](NEXT_SCIENCE_PLAN.md)。既存C branch `codex/r-variable-compiled-price-design-20261111`、基準commit `86f37d25ffd5eee3daf7f25ce001b090b1f2cf33` を継続。A/B/Dのbranchや作業中領域は変更していない。

## 一連のflowと再利用

`B.generate → integration.plan → A.SignalBatch / C.execute → B.W.prepare → 明示的な既存W.diagnose又は外部N証拠 → W.compare → B.select`。

|固定SOURCE|使う既存処理|今回追加した接続|
|---|---|---|
|A `fefd548ecd56f60d72031b3d385f6f5e6ee95d06`|`pr_rpe_signal_plan`、`pr_rpe_signal_provider`、direct head/finite-K raw独立伝播|Aのwhole-schedule job/結果をB meanへcopy。資格を昇格しない|
|B `b9b9ffa6233db9dd2b413dc07b27e7039324c666`|V2 controller、decoder/statistics/fast decoder、`rpe_shot_allocation.prepare/diagnose/compare`、cost/select|同一物理候補の複数allocationを保ち、実際に使用したsignal入力へN証拠を結ぶ|
|C `86f37d25ffd5eee3daf7f25ce001b090b1f2cf33`|Full PF/B0/B0I/Full-Hの全round pipeline・共有cache・数値gate|同じV2/target形式で呼出し。既存Cコードは無変更|
|D `24b0a6eb1d806391a3c6ad454fb0c018148d8545`|公開HANDOFFの価格mean/区間・profile extension・同時box|未評価26軸request生成とphysical bindingだけ。価格取得・残差・compilerは実装しない|

新コードは `src/trottertracks/rpe_optimization_pipeline.py` と人工test。新しい共通optimizer、decoder、N資格solver、価格計算器、探索gridは追加していない。

Cの小さいcheckoutで依存graph全体をmergeせず、固定Git objectから必要moduleをprivate namespaceへ読み込む。Bのcontroller/common/accountingはC参照版と今回B版のbytes一致を確認する。AのRTE importとBの元scalar decoder importの2箇所だけを同じ純粋関数の読み先へ向け、eagerなcircuit package初期化を避けた。元SOURCEの演算・閾値・law・decoder規則は変更していない。allowanceは元extractor/算術とSOURCE定数だけを再利用し、旧scientific runner graphを起動しない。

## 人工接続結果

新10＋固定B metadata回帰2＋既存C/M5回帰24、**36 test method PASS**。作者自身の技術確認で、独立科学レビューではない。

- 2×2人工非可換行列を実A/C numerical providerへ渡し、PR/Full PF/B0I/Full-Hそれぞれ全13roundを評価。52個の人工complex meanのidentity/law/rawをB統計側まで保持した。これらは実H4の52値ではない。B0は既存C回帰でも確認。
- テストfixtureだけでmean/N/価格を適格と宣言した2round例では、同physical configurationの12 shot配分のG=1002、18 shot配分のG=216。既存exact costが後者を点選択し、shot総数最小とG最小を区別した。4方式の人工menuも元B.selectへ接続した。
- 同時価格boxが重なる場合、点最小があっても `strict_interval_winner=null`。外側Bはcross-configurationの厳密価格優越を認定しない。
- 人工PR rawを既存decoder/shot診断へ渡した。診断のN packet/Gはnullで、人工MCからも十分Nを生成しない。
- 欠測raw/価格、未資格mean/RMSE、nominal N、有限cohort価格、SOURCE/law/physical/compiler/lowering不一致、Nとsignalの取り違えは会計をblock又は入力不一致として拒否。synthetic/scientific evidenceの混在も拒否した。
- A数値失敗の観測残差・未開始round・旧STOPを保つ。新価格requestはmean/profile extension/law距離資格がnullで、適格価格recordには使えない。

人工fixtureのQUALIFIED宣言はテスト専用で、production providerの出力はUNQUALIFIEDのまま。元科学結果の資格・state仮定・sampler距離を変更していない。

## 現在実行可能な部分と不足

|部分|現在の状態|
|---|---|
|候補・全round metadata|固定B V2の合法構成をcallerが生成して接続。旧11候補やTrack A winnerを最適集合にしない|
|PR signal|AのS2/M5・K2/4/6・非空head/tail。H4保存factoryのprefix制限・SOURCE条件は元HANDOFFのまま|
|baseline signal|CのH4 DF10、Full/Truncated S2/Y4/M5、global prefix/Δ/K、Full-H明示q/R。未対応構成は科学的劣位ではない|
|実入力factory|明示callbackが必要。Aのoptional保存factory依存graphはこの最小namespaceに一括取込みしていない。実snapshot/reference locator・bindingを選択/固定し、A公開factoryを完全な環境から渡す等の接続を最初の科学pilot前に確認する|
|shot配分・診断|既存Bで可能。loss/IID/反復数/confidenceはcallerが指定。自動診断・自動N最適化はしない|
|科学的に十分なN|**未取得。MC summaryを適格N packetへ変換する科学producerは未実装**。外部契約/証拠が必要。meanの物理資格も別途必要|
|D平均compiled価格|並行adapterの公開待ち。同law/full wrapper/lowering/domain・mean/区間/不確かさが必要。Cは人工価格以外を供給していない|
|全G選択|適格な外部mean/N/Cが揃えば既存Bへ通る。欠測ならpending。点最小・同physical配分間区間優越・科学的最適性を別に保持|
|外側探索範囲|既存Bのglobal h、method内K/κ定義、formula menuの制約を保持。K横断・新h探索等を今回実装したとしない|

今回の初期fixtureで、Aのbinary64 physical timeとexact decimal rationalのcanonical timeは同一でないことも確認した。人工入力を元A規約のbinary64で揃え、canonical IDを後から書き換えなかった。Bの旧人工価格fixtureにlaw_contextがない部分は、そのhelperのN/price形状を再利用しつつ、**新統合fixtureに明示した架空law context**を付けた。古いB全testがこのcheckoutで再PASSしたとは主張しない。無変更回帰は上記2 metadata methodとC/M5の24 method。

全event展開やCartesian gridは不要。元providerのmacro-step/cache/行列累乗を使い、要求候補だけを処理する。資源の恒常上限や共有環境設定は変更しない。旧SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPは保全。commit/push・remote SHA/tree/主要成果物照合後、**mandatory STOP**。新科学計算には進まない。
