# m3保存価格のconsumer binding引渡し

[REPORT](REPORT.md)／[12行CSV](ADMISSION.csv)／[ADMISSION.json](ADMISSION.json)。**12受入・0保留、既存36軸と合わせ48軸の直接価格**。

## 読取

- 元C表：`0b4b2d941cd669aaa45231e1236bc2d3158a9d12:docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json`。
- Cの各`ADMISSION.rows[i].alignment_source`に、元表のrow selectorを保持。`configuration_id,m=3,axis`で対応付ける。旧表全体の書換え・方式名だけによる結合はしない。
- 受入priceは`PRICES.json`、固定evidenceへ結合したsidecarは`SIDECARS.json`、実guardの個別結果は`GUARD_CHECK.json`。metadata証拠SOURCEは`826f7c9230af7fbfbcc8501d236c44f5eefe2135`。
- `price.source`は元receiptのcommit/path/axis selector＋generation_SOURCEと元trial。新しいcompile結果を発行したものではない。旧canonical未記録はnull、旧trial IDとcandidate tupleはoriginに保持する。
- 原cohort未記録はprice/sidecarともnullで保持する。現行C guardの選択keyは`cohorts[(canonical_id,axis)]=None`。新しいsampling cohort名を作らない。X/Y各receiptを固定fingerprintへ結び付け、他軸の値で補完しない。
- wrapper_sourceはDの同等性確認済み参照で、原caller生成SOURCEとは分離した。sidecarのroleとD正本へのSOURCE linkを保持する。

本作業は既存guard・tier・plannerの再利用。C/Bのprice accounting・fitを変更せず、新共通providerを実装していない。異なるSOURCE/scalar/physical/axisは同一価格として受け入れない。

## m4～m6とモデル

[PREVIOUS36_REFERENCES.json](PREVIOUS36_REFERENCES.json)は元24較正と12独立holdoutのreceipt locatorだけ。元cohort、生成SOURCE、guard、Stage2報告を維持する。新しい平均・相対誤差・判定・価格予測はない。

m3のroleは`SAVED_M3_DIRECT_OBSERVATION_NOT_CALIBRATION_OR_HOLDOUT`、model_group=null。actual幅が異なるので12モデルの較正へ追加しない。m3を追加holdout PASSへ昇格させない。

凍結価格モデル`12372fa6b53c878dc5dd124e036572ba0cd82061`、固定CELL_PLAN36、Stage1/2、C旧対応表は不変。Dのm6既存結論と、新しい長q/別幅モデル資格は別問題。

## 統合担当・次の境界

ロードマップには「m3 C bindingの未完了が解消、m3～m6計48軸の直接価格」と反映する。残る短round36軸と長q72軸、signal/risk/十分Nの資格は未解消。D1 Full-H作業と独立で、Full-Hへの価格/条件移植はない。

m9検証案はD既存計画として参照するだけ。本成果はm9予測固定・実行許可を含まない。再fit、build/compile、MSE/N/G、winnerへ自動進行しない。

**mandatory STOP。** 公開commit/pathは最終報告の固定GitHubリンクを使用する。
