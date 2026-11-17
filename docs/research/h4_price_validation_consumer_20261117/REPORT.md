# 保存PR-Pauli LD2価格検証の通常探索マップ接続

`92b3e617a0a0ede94d90ac2582b058f5bf9f67a3`の読取り・表示差分を、既存公開branch `codex/h4-saved-rpe-cost-catalog-20261116`へ選択的に接続した。[探索マップ](../h4_search_map_20261117/index.html)のPR-Pauli・LD2で「独立実測あり」の配分詳細を開くと、保存検証と元receiptへ戻れる。

表示するm5は **q=32、R=33、独立2経路、各軸実測平均68,264 RZ、符号付き相対差+0.044612%**。差の定義は `(予測−実測)/実測`。保存exact値2751307649/6167206567816を保全し、百分率は表示用。X/Yは同じ2経路を共有し、4独立経路とは数えない。実測は経路ごとに68,228・68,300 RZ。m6の独立経路数1も原記録のまま保持する。

## 接続範囲

- 元SOURCEの `h4_search_run_evidence`・`h4_search_map`・HTML renderer、共通publisherのoptional sidecar指定、関連consumer testを再利用。
- controllerの`PRICE_MODEL_VALIDATIONS.json`を、元`SEARCH_PUBLICATION.json`の固定hashで読む。モデルはINDEX.price_validation_modelsへ一度だけ置き、各配分から参照する。
- 測定時のq/Rと、価格モデルを適用する各roundのq/Rを区別。sidecarだけが更新された場合も元Q3 SOURCEを保存したまま更新する。
- 表示を人が読めるように平均へ桁区切り、符号付き差へ小数6桁の百分率を追加。元exact数値も併記。
- acquisition worker、model fit、共有科学queue、探索器、local0の実行SOURCEは取り込んでいない。既存consumerだけの変更。
- 条件付き費用・Q3・既存モデル資格・最安・原結果・旧STOPは変更しない。表示上の「独立実測あり」は全law・長qの資格ではない。

## 確認

consumer/通常同期/マップ回帰13件＋保存LD2照合1件がPASS。保存された同一軸receipt4件へたどり、q/R・軸・compiled RZを照合した。元receiptのGitHubリンクはHTTP 200。

V8上で実rendererと保存LD2値を使う人工DOM検査を行い、q32/R33、独立2経路、68,264、+0.044612%、元receiptリンクを確認した。実ブラウザの描画試験とは区別する。

通常の同期で10 run・4,877一意配分点、共有検証モデル1件を生成。再同期と公開の検収は[CHECKS.json](CHECKS.json)へ記録する。元C catalogの判定・費用表は不変。新規信号・sampling・build/compile・fit・Q3は各0件。

## 同じ公開先

local0が参照するリモート・branch・出力pathは従来のまま。`scripts/run_h4_catalog_sync.sh`の10分同期へ追加される。各環境の通常controller公開にsidecarを含めればよく、CへSHAを毎回転記する手順は追加しない。

元保存例は[92b3e617aのREPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92b3e617a0a0ede94d90ac2582b058f5bf9f67a3/docs/research/h4_price_validation_integration_20261117/REPORT.md)。実測の正本は`705f59ea863e1b9fe90e023f1cb1057d00366b16`のCOMPARISONと、`0e1ce866e8f8d4e042801260fb8d543d0b1bee53`の軸別COMPILE_RECEIPTS。

公開・remote照合後にmandatory STOP。保存結果の表示以外へ進まない。
