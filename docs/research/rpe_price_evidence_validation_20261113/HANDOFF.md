# B/A/D向け：価格証拠と有限検証の接続

正本は[COVERAGE.json](COVERAGE.json)。主対象はA固定LD1/LD2の52軸セル。Full PF156軸は旧6 recipeの照合catalogで、Aの正式選択menuではない。[REPORT](REPORT.md)にSOURCEと資格、[VALIDATION_PLAN](VALIDATION_PLAN.md)に条件付き最小セルと精度入力を記載した。

## 今回の技術差分

production moduleは変更していない。C1914 `rpe_mean_price_model`、mean_compiled_price、I/B会計、C3802 scalar接続を再利用する。新しいSOURCE readerは文書内の受動BUILD_COVERAGEだけで、S/C値の算術・新価格予測・fit・実input読込みはない。必要なら`tests/check_rpe_d_price_connection.py`の新test14だけを選択取込みする。

D e827の`input_source.scalar_binding`付きrequestを受ける際も、既存呼出しをそのまま使う。

```python
pc = externally_materialized_corrected_D_request['physical_context']
pending = I.price_requests(root, C_request,
    input_source=pc['input_source'], execution_source=pc['execution_source'],
    expected_scalar=pc['expected_scalar'], axes=externally_materialized_corrected_D_request['axes'])
```

Dのselectorはsnapshotのconstantを読むので、今回Cでは呼んでいない。Cはtotalの再加算・再抽出・推測をしない。既存constructor pinは748のままでよく、D最新版の該当3 metadata関数とのAST一致と人工total転送を確認した。observerを含むmodule全体のbyte一致は主張しない。

scalar/入力SOURCE変更でrequest/context IDは変わる。COVERAGEに保存した旧D keyは旧metadataのlocatorに過ぎず、新科学runへcopyしない。旧path/meanを新SOURCEへ付け替えず、cohort・role・scalar・law・profile・V2受入を別添の根拠で確認する。

## 担当ごとの次の入力

- **B**：d717の全13round risk/whole-vector案を承認後に評価し、同taskの資格N又はN集合、price target law、比較family、margin/regret・価格confidence/必要精度を供給する。今回の6セル案やnominalからNを設定しない。
- **A**：2f0deのFull PF診断から選定が必要になった際、公式・actual PF幅・全round physical ID・scalar/sourceを渡す。S2/.1、旧6 recipe、レビュー候補を既定の最適構成へ採択しない。
- **D**：新compiled取得が承認された場合にのみ、同law・full measured wrapperでpaired S/Cを取得する。LD1旧P2はDEVELOPMENTのまま、consumerの校正入力roleを別に保持し、holdoutと呼ばない。旧PF receiptは同profile/domainの再利用候補で、loweringが違う場合は同値性又は新directが必要。
- **C**：同条件の供給値が揃い、fit/比較/受入再解析が明示承認された場合だけ既存quote/combiner/paired fitへ渡す。今は全domainのprice qualification=false、mean_C/prediction/CI/G=nullを維持する。

価格単位はfull controlled measured X/Y wrapperのRZ gate_count、P=0。量子shot N、古典path n、normalization Bを分離し、同path XYの共分散を保つ。S、E[S]、cohort平均、MODEL予測、全law E[C]を別の証拠として供給する。BのNがPASSしても価格を自動昇格しない。

## テスト・保全・終了

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests python -B -m unittest check_rpe_d_price_connection -v
```

既存専用venvで14件全PASS（新1＋旧13）。C1914の38件とDの既存29件は記録を参照し重複実行していない。新testのconstant/tail/totalは人工dyadic値で、実H4の値を解析・修正したものではない。独立科学レビューではない。

旧C1/C2、26 E[S]、S0a/失敗run、T2/診断、input/raw/grant/ledger/marker/KEEP/STOP、未承認baseline計画を保全した。現在の新科学実行は0。**ユーザー**がB/A/Dの不足入力と有限科学scopeを判断するまで、公開・remote照合後mandatory STOPを維持する。
