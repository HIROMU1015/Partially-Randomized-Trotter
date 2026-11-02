# F3完成範囲と次の担当

**FullのP0〜P5をG6/G7へ接続し、一般有理入口・identity・terminal failure・angle enclosure・event保持を実装できた。最終63人工テストPASS。技術statusは`F3_TECHNICALLY_VERIFIED`。** 旧理論/アルゴリズム/科学状態は変更しない。

## 完成した内容

[F1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md)の固定H/K十分条件をexactに評価し、全parentを列挙せずsampling前にfail closedする。P5はsource/input/policyと実lawを結ぶ。x=0はpositive generator/bit/query/weight/angleを呼ばない独立identity。raw非reduced/coin不受理だけを正当zeroとし、精度不足・不正入力・実行障害は原因を保存してSTOPする。

既存half-angle＋atan交代級数を固定K_phi/Jで実装し、要求τに対する代表値半径、unsigned/signedφ、parent/child/σ/精度来歴を保存する。旧実π/c_tilde/W_tildeとπW=c_tilde、word/phase/time-orderを保持する。sampler/proposalを変更せず、source/hashとevent証拠を照合する。

[人工報告](focused_test_report.md)と[test_catalog](test_catalog.json)、[4回のraw記録](test_runs/run04.json)が実行証拠。[管理検査](verification.json)は別のprovenance/保全検査。run04と新SOURCE/testのbytes/SHA256を一致させ、[SOURCE索引](source_inventory.json)、[実装索引](implementation_inventory.json)、[公開receipt](publication_receipt.json)から固定版へたどれる。

## 未保証・未実装

canonicalの非identityは未対応として拒否する。これは仕様が許す限定実装で、Fullの一様下界を転用していない。H/K selector/refinement/retryはない。人工testから一般数学証明、fresh IIDの経験的証明、実環境の無故障を主張しない。P0のcontrolled-Q仮定は物理certificateではない。

angle evidenceはnative RZ/T/CX/depth/strict phase/adjoint/物理error証拠ではない。DF/PR/RTE adapter、全trial estimator、測定、state、backend、全N confidence、任意α、総資源比較は今回の完成範囲外。既存G7 budgetのfixed policyは不変で、F3から呼んで新forecastをしていない。F3のhash/serialization/bit-map監査/angle時event再計算の追加費用も[実装契約](implementation_contract.md)に明示した。

新しい非自明な保証が必要となる問題は今回特定されなかった。故意のguard注入は分類テストであり、F1と旧SOURCEの実矛盾ではない。実際の固定条件とguardの矛盾を将来発見した場合は、旧SOURCE/反例を保全してSTOPし、F1・law・precisionを勝手に変更しない。

## 次の担当・承認

**ユーザー**：公開F3を既存仕様の技術成果として受け取り、次の限定作業を行う必要があるか判断する。現時点で追加作業を採択していない。

**Codex**：明示された次指示の範囲だけを担当する。canonical preflight、H/K selector、adapter/estimator、回路/native、分子/新科学計算、P_U/W2b/W2cを自動開始しない。新規科学計算と重要GPTレビューには、それぞれ別途ユーザーの開始承認が必要。論文執筆・独立新規性・科学的GOを認定しない。

## 公開・保全

基点`dd03eb08b6d768f5578d743ab5ad390bd9c93b07`、branch`track-b-g6-f3-preflight-angle-implementation-20261018`。新3SOURCE、F3専用test1本と本資料だけを追加。旧基点の全blobと、親tracked/直下Markdown原本/既存index・exclude、以前の自分の公開資料、既知のconsumed ledgerを限定hash照合する。Codex Aのlive P_U worktree/index/lock/process/outputにはアクセスしない。他研究を全面監視した意味ではない。

payload commitをpushして公開bytes/SHA256を照合後、receiptだけの子commitを公開する。最終remote HEAD・全変更file（receipt含む）も再取得して照合する。自己hashの循環を避け、実装commitはreceiptに記録する。

旧D＋E、研究④保留、A2 NOT_ADOPTED/A2現経路追加投資見送り、A1 CLOSED、K1-F未承認、N1 STOP/N2 HOLD/N3 CLOSED/THRIFT CLOSED等を維持。新規科学計算、回路、backend、新証明、authority/grant/marker/claimは0件。**公開・保全確認後mandatory STOP。**
