# B V2へのC受動価格入口と次の科学scope

[REPORT](REPORT.md)／[METHODS](METHODS.md)／[SCIENCE_PLAN](SCIENCE_PLAN.json)／[PRICE_COVERAGE](PRICE_COVERAGE.json)。実装SOURCE`58382c54c`、固定B `6a9d7fd36b687e58aecbed1278c6d7927d6827d6`、新12技術test PASS。実S/C/保存数値再解析/sampling/compile/fit/N/Gは0。

## 取込み・呼出し

```python
# B所有の固定pr_rpe_rmse_integrationをI、既存V2 configurationをconfigとする。
from trottertracks.bv2_mean_price import V2MeanPriceLookup
prices = V2MeanPriceLookup(config, supplied_records, integration=I,
                          selections=explicit_cohort_selections)()
```

Cファイル`src/trottertracks/bv2_mean_price.py`だけを取り込める。testは固定Git Bモジュールと旧fixtureを読んで現在Cへ結合するため、B worktreeを操作する必要はない。Bへcontroller/optimizer変更を要求しない。inventory scriptは任意のfresh出力へ固定metadataだけを書く。

既存C MeanPriceProvider/既存B MeanPriceLookupを変更せず再利用する。呼出し側はB/C SOURCEを固定すること。schemaが同じ将来版すべての互換性を宣言していない。新C入口では値ありrecordにquantity=`FULL_WRAPPER_MEAN_COMPILED_PRICE`の明示が必要。Bのlegacy compiled recordのquantity省略ルールは変更せず、旧recordへquantityや新IDを自動追記しない。

PRはA-source-bound physical viewだけを既存B adapterへ通す。identity/law/SOURCE/元physical ID、measurement/profile、cohort/値/uncertaintyを保つ。MODEL false・null、複数cohort選択、FINITE_COHORT_ONLYはそのまま。Full PF/truncated PFは既存PF価格routeへ、Full-H RTEと未接続native PRはOUTSIDE_DOMAIN/nullで、価格law/idealとの距離もnull。S0aのS、local D、単回路priceを全law meanへ昇格しない。

## 保存metadataの要点

118候補/236軸は歴史的round別R menu。新M/共通κ V2 identityや十分N・mean Cのcoverageではない。H4 truncated13、H6 full13、PR92。DIRECT存在は順に6/8/20軸で、すべて新RMSEでは未資格。

H6 m3/prefix5/δ.8には保存有限価格がある一方、C regular mean-providerは対象外。m0..2は価格missing。prefix7 Affineを初期roundへ移さない。Full-H RTEには別T=2.0の旧finite cohort価格・SOURCEがあるが、新RMSE event_count/law/全H preparationとのbridgeは未接続。DF B3を代用しない。

C1の8セルは元role/n/cohort/SOURCEのまま。q/Rモデル、S0a run01失敗とrun02 8/8を保全。S0aを価格provider recordにはしていない。

## 次に必要なユーザー判断

1系・外側条件・M/共通κ・実physical候補・price law・必要精度をA/Bと固定する。A ee15のpilot pinは旧B f09なので、新V2追従はA/Bが担当する。CはAのRMSE/N理論を実装しない。

第一科学候補は、選択した1 preparation/law domainのsupport・flush・状態同値性診断（新S/compile0の案）。実metadata再構成と物理hypothesis検証のSOURCE/対象を固定し、別の明示許可後だけ行う。proofが得られなければUNKNOWNでSTOP。

次いで定義済み平均S法を保存2cohortでdevelopment比較する案、paired C−S/全境界の校正とfresh holdout、新B候補の不足価格取得を個別に選べる。これらは同時にすべて実施する計画ではない。具体候補/n/compile上限が未固定なら実行可能としない。40/16/32件の旧案は自動採用しない。

適格なB価格にはfull-wrapper平均C、同じlaw/SOURCE/domain、校正/独立検証・q/R移送・rare/systematic/covariance/CIの根拠が必要。現C1 falseを変えず、finite-cohort scenarioとqualified expectationを区別する。

担当：ユーザーが方法・有限scopeを判断、A/BがV2全schedule/RMSE条件を管理、Cが価格内部と選んだ検証を担当する。今回のcommit/push・remote照合後mandatory STOP。追加科学計算へ進まない。
