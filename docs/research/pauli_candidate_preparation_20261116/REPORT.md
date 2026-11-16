# B2要求からのPR-Pauli分割準備

H4 DF10の保存辞書・configurationを価格catalogから独立して解決し、B2が選んだ1候補を準備する入口を追加した。新しいH4辞書・signal・normalizer・価格・compile・Q3は実行していない。

- 固定技術SOURCE：`6e4b6bd6a491815c2a3cb8f809cbfc0240b60fdb`
- 専用branch：`codex/a-pauli-on-demand-preparation-20261116`
- [B2向けHANDOFF](HANDOFF.md)、[保存SOURCE一覧](SOURCE_INDEX.json)、[受入結果](ACCEPTANCE.json)
- 入口：[prepare_pauli_candidate.py](../../../scripts/prepare_pauli_candidate.py)
- 本体：[pauli_candidate_preparation.py](../../../src/trottertracks/pauli_candidate_preparation.py)、[descriptor bridge](../../../src/trottertracks/pauli_split_descriptor.py)

## 保存範囲と§10.5照合

snapshot receiptの`arrays.lambdas.shape=[10]`と`g_matrices.shape=[10,8,8]`を照合し、fragment数10、one-body常時head・tail非空の範囲を **L_D=1〜9** とした。H/state配列を読む処理ではなく、固定receiptのshapeを読む処理である。

| L_D | DF descriptor | Pauli辞書 | 基準canonical configuration | 今回確認した範囲 |
|---:|---|---|---|---|
| 1 | 保存あり | `4220f3006` | `d905d219…` | 保存JSON・13round信号再利用。26軸価格なしでも構成を返す |
| 2 | 保存あり | `4220f3006` | `d7dd4f52…` | 保存JSON・13round信号再利用 |
| 3 | 保存あり | `50640b74c` | `dd5b881d…` | 保存JSON・13round信号再利用。公開κ×2も再現 |
| 4 | 未保存・導出SOURCEあり | 調査範囲では未保存 | 未保存 | 実H4では準備要求のみ。未登録fixtureの全経路は人工系で確認 |
| 5 | 保存あり | `6255fbc42` | `4d6e2c9c…` | 保存JSON・13round信号再利用 |
| 6 | 未保存・導出SOURCEあり | 調査範囲では未保存 | 未保存 | 準備要求のみ。共通経路は人工系 |
| 7 | 保存あり | 調査範囲では未保存 | 未保存 | 保存descriptorの所在・準備要求。辞書経路は人工系 |
| 8 | 未保存・導出SOURCEあり | 調査範囲では未保存 | 未保存 | 準備要求のみ。共通経路は人工系 |
| 9 | 保存あり | 調査範囲では未保存 | 未保存 | 保存descriptorの所在・準備要求。辞書経路は人工系 |

descriptor正本は`7a8bc7f`のSOURCE_INDEXが指す保存`H4_context.json`（SHA `8396605f…`）。辞書/P0 gate・configuration・packetの正確なcommit/pathはSOURCE_INDEXと応答inventoryに残した。Cの価格catalogでの未登録を、辞書・configuration欠測の根拠にはしていない。

関連成果の観測：A `08812cd30`、B `0a6f8e4a5`、B2 `98b6f8e1a`、C `dbeb6480d`、D2 `1eda4ba3f`。A信号入口の実装は指定`4e5b8bef6`、B2 generatorは`98b6f8e1a`、D2契約は`c1f95e080`へ固定した。Aの`08812cd30`はその信号入口の報告更新。B2ではoptimizerのdirty変更と実行中processを観測したが、変更・停止・freeze・再投入はしていない。他サーバーの非公開runを確認したとは主張しない。

## 実装と再利用

1. 保存Pauli configurationを、A/Bの固定locatorから解決する。LD1はB2の公開NEEDSで`configuration_id=null`でも見つかる。
2. 同じ分割の辞書を再利用し、B2の公開`generator('pauli_native')`をそのまま呼ぶ。canonical ID、mapper、整数化、signed allocationはB/P1が所有する。A独自のmapperはない。
3. `formula`・`delta_over_h`・`kappa_multiplier`はB2既存requestを維持する。絶対`delta_pf`と`K`も受ける。元の要求表現を保存し、float→Fractionはbinary64の厳密値を保持する。厳密`1/5`への丸めはしない。
4. 辞書がなければ、通常呼出しはdescriptorの保存/導出可否とP0準備要求を返す。全分割を先に計算しない。
5. 将来の承認済み`--prepare-missing`経路は、既存`split_stage1.execute_p0`を呼ぶ。未保存descriptorのみ既存`matrix_only_tail_coefficients`を使い、元contextを変更せず一時metadataへ供給する。P0成果を限定commitし、固定locatorを得てから既存P1変換を呼ぶ。新しい承認証明書は作らない。
6. 分割cacheはH/state/snapshotと分割・辞書規則に束縛する。κ・幅変更でP0を再実行しない。local metadata configurationは公開固定locatorと区別し、未公開のcommitを捏造しない。

Pauli scheduleは辞書の**λ_P**を使う。spec内の互換field名`split.lambda_R`にもλ_Pが入る。元DF descriptorのλ_Rは応答の`lambda_DF`として別に返す。one-body、fragment順、scalar一回会計、law、h/M/target/prior/誤差policyは元specを保持する。

既存信号入口には、旧contextにないdescriptorを渡す任意の`--tail-descriptor`だけを追加した。binding・prefix・固定SOURCEの値を照合し、contextがplanだけでなくexecute/assembleまで有効であることを人工確認した。mean kernel・normalizer・mapper・旧P0/P1は変更していない。

## 完成区分と価格契約

応答は`readiness.configuration_complete`、`signal_13round_complete`、`price_26axis_complete`を別々に返す。保存26軸の有無はSOURCEと軸・ID・q/Rの技術照合で確認し、新しい価格quoteは実行していない。

- LD1：構成・保存13round信号あり、保存26軸価格なし。
- LD2/3/5基準：構成・保存13round信号・登録された26軸価格carrierあり。この構造照合を新しい価格モデルやE[C]の資格と解釈しない。
- LD3 κ×2：公開構成を完全再現し、指定された追加packetから13roundすべて再利用。価格の新しい適用計算は未実行。
- 価格または信号SOURCEが欠ける場合も、検証できたconfigurationを捨てない。信号証拠の所在が未解決なら新規取得要求へすり替えず、`SOURCE_UNRESOLVED`を返す。

D2へは`PriceRules.quote({configuration, implementation, policy:'MODEL'})`の入力を返す。既存LDでは当該registry entryの契約とSOURCEを付ける。新LDには他LDの契約・価格係数を採用せず、候補固有contract、control/lowering/scalar/order/sampler/wrapper、価格根拠の不足を列挙する。implementation=nullならD2の候補契約解決が先である。

## 検証

保存資料・人工系のみ、計23テストPASS。

| 検査 | 件数 | 記録 |
|---|---:|---|
| B2の実request入口：保存LD1/2/3/5、公開κ変更、精度表現、signed PF/K、境界、欠測、cache再開、人工未登録分割 | 11 | [TEST_PREPARATION](TEST_PREPARATION.txt) |
| descriptor：shape由来範囲、既存helperと人工保存値の一致、metadata context/restore | 3 | [TEST_DESCRIPTOR](TEST_DESCRIPTOR.txt) |
| 既存信号入口のplan/取得stub/packet組立て・保存fixture回帰 | 7 | [TEST_SIGNAL_REGRESSION](TEST_SIGNAL_REGRESSION.txt) |
| 既存stage1の人工P0 operator gateと保存configuration回帰 | 2 | [TEST_STAGE1_REGRESSION](TEST_STAGE1_REGRESSION.txt) |

主担当が実入口・回帰検証を実施。補助agentはB2/D2契約と保存inventoryの読取調査、descriptor helperと人工3検査を担当した。独立科学レビューを実施したとの主張ではない。

未登録fixtureでは、既存係数helper→P0のJW global collector→P1/B canonical→既存信号plan→D2の人工空registryへ通した。これは実H4のLD4/6/8等の辞書・operator gateが取得済みであることを意味しない。H4で未保存descriptorを再導出する検査も実施していない。

新規fixtureは2mode・6fragmentのLD4、既存stage1回帰は3mode・7fragmentのLD5。H4の未登録LDを個別にoperator評価した検査ではない。固定技術SOURCEで既存signal/stage1のSOURCE確認関数も実際に呼び、[SOURCE_CHECK](SOURCE_CHECK.json)へ保存した（科学call 0）。

## 次の処理とSTOP

B2が候補を選び、本入口の`GENERATED`応答を登録できる。未登録LDのP0が必要なら、その**選択された1分割**の科学実行許可を得て`--prepare-missing`を呼ぶ。既存stage1のLD2対照・SOURCE gate・fresh output・失敗時STOPを保持する。保存signalで足りないround、新規価格根拠、Q3は別の担当・実行範囲である。

H4の新descriptor・辞書・mean・normalizer・較正・compile・Q3は0。B2探索規則、shot選択、他担当のrun、既存raw/KEEP/grant/ledger/marker/STOPは変更していない。技術成果の公開・remote照合後にmandatory STOP。
