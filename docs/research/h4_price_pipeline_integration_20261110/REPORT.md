# H4 DF10価格providerのB統合pipeline接続

2026-11-10 JST。**Aの凍結Mq価格providerをBの統合branchへ取り込み、B自身のAPI・依存module・Python環境で技術的な結合を完了した。** 保存direct/Mq・PR有限cohort・未取得・未検証外挿・全law期待価格の未資格を区別し、null価格を取得済みへ数えず、資格不足を理由に同じproviderを自動再呼出ししない。実H4/H6の新しい科学計算・価格予測・N×C/Gは行っていない。

## 取り込んだ成果と変更

A正本は[`2a521b156f8ae957d981516ab8a0031fcafae0a5`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2a521b156f8ae957d981516ab8a0031fcafae0a5)。[A REPORT](../h4_df10_price_provider_20261110/REPORT.md)・[HANDOFF](../h4_df10_price_provider_20261110/HANDOFF.md)・[詳細interface](../h4_df10_price_provider_20261110/INTERFACE.md)は当時の引継ぎ記録として、そのbytesを保持して取り込んだ。本REPORTがB取込み後の状態である。

- `h4_df10_price_provider.py`、元テスト、上記資料をそのまま取り込んだ。必要な既存`h4_df10_price_v1.py`・`h4_df10_price_v2.py`・`h4_df10_price_v2_independent.py`もA固定版と一致。既存`track_d_native_resources.py`は同版だったため変更なし。固定係数bPF=606.75、bPR=176.88888888888894、V1会計本体と科学条件は不変。
- Bは既存`codex/pf-pr-rpe-cost-integration-20261110`を再利用。指定統合成果`3b742fff26381514e12c07b51701276b7d1b26ec`を含む着手時HEADは`f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a`。後続H6 N結果を巻き戻さない。
- [`pf_pr_rpe_pipeline.py`](../../../src/trottertracks/pf_pr_rpe_pipeline.py)に`h4_saved_price_provider`を追加し、AへB module自身を渡す。loader・identity・measurement・task validationを再利用し、共通frameworkやN/G会計器を複製しない。
- `plan/run_missing`は価格値の存在、返答recordの存在、同一cohortのpaired X/Yの利用資格を分離。有限cohort policyではMqもdomain資格を検査する。`evaluate`は選択価格がnullなら既存積和へ進む前に`MISSING_SELECTED_C_VALUE`を返す。
- C batchの全recordを照合してから統合。同一recordは再追加せず、部分欠測のnullだけを補完する。取得済み価格と矛盾する同一scenarioの値は自動上書きせず例外を返す。過去の結果fileは変更しない。

## B APIからの接続

以下は接続方法の例であり、この例による新科学実行・実積和は今回行っていない。

```python
from trottertracks import pf_pr_rpe_pipeline as p

store = p.SavedArtifacts(repository_root)
provider = p.h4_saved_price_provider(store.read, selections={
    "64e4f7e9ec7fba28815e": {
        "kind": "model_prediction",  # 保存済みMq値だけを選択
        "price_unit": "cohort_mean",
        "cohort": "H4_DF10_V2_INDEPENDENT_Q_20261110",
    },
})
batch = p.run_missing(task, supplied_N_records, supplied_C_records,
                      c_provider=provider)
prices = batch.get("prices", supplied_C_records)
```

PFの既定は保存direct。PRはunit/cohortを明示しなければunavailable。`kind='model_prediction'`も新Mq計算ではなく固定JSONの保存値選択だけである。N不足時は`N_PROVIDER_REQUIRED`、保存N不成立時は`STOP_UNQUALIFIED_N`でCへ進まない。新N providerは今回接続・実行していない。

`run_missing`の戻り値の`shots/prices`を次の呼出しへ渡す。null応答も`prices`へ保持する。別のprovider・selector・科学条件を試す必要がある場合は、明示的に別の要求を組み立てる。同じ入力に対する無限再試行や、既存の未資格modelを新modelへ黙って置換する動作はない。

固定Git JSON readerを使うためAのcarrier objectも必要。新しいcheckoutでは必要に応じて`git fetch origin codex/resource-eval-h4-df10-longtime-price-design`で取得する。object不足をcompile/予測で補うfallbackはない。Aの明示数値API`predict_path`は残すが、Bのlookup入口・`run_missing`から自動起動しない。

## Coverageと科学的資格

| field | 意味 |
|---|---|
| `C_axes` / `missing_C` | 有限・非負の価格値がある軸／両軸の値が揃わない候補。nullはここへ数えない |
| `C_record_axes` | unavailableを含め、既に返答recordを持つ軸。取得済み価格とは区別 |
| `expected_C_axes/cohorts` | 既定の期待価格policyで資格した軸と、同じscenarioで揃ったX/Y |
| `eligible_C_cohorts` / `unqualified_policy_C` | 明示policyで利用可能なpaired scenario／資格不足候補 |
| `C_provider_requests` | 未返答の軸が残り、paired価格も揃わない候補だけをbatch要求 |
| `blocked_C` | 両軸の返答はあるが、値・domain・期待価格・paired scenario等が不足。自動再呼出し対象にしない |
| `price_contexts` | kind分類、expectation、valueの有無、unit、validation、reason、model資格、SOURCEを保持 |

`provider_quote`と`uncertainty`は元recordのまま返す。既知S1、PF qPF32/PR qPF16の有限独立q、未検証q外挿、モデル対象外を区別する。qRPEとqPFは別field。価格は一回の測定付きfull Hadamard interrogation、P=0のcompiled RZ/gate_countで、normalizationを価格へ再乗算しない。

科学domainはLinear H4/隣接1Å/STO-3G・保存tol-only1e−8 DF10/sector(2,2)・保存H/state/snapshot/fragment/scalar、S2/δ.1、PF B0 p7、PR p1/R4/K2、元finite RTE law、symmetric_directional、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17・元compiler hashである。Full10や別δ/prefix/lawへ拡張していない。

PR cohort平均は`FINITE_COHORT_ONLY`、pathは`FINITE_PATH_ONLY`。既定`qualified_expectation`では有限cohortを受け付けず、明示`finite_cohort_scenario`でも有限条件のscenarioに留める。独立qのMq資格があっても全law期待値や全13roundの正式採用ではない。path単体をcohort/law期待値へ変換しない。

Mqの共有校正group・paired X/Y・保存誤差summary/残差参照を維持。係数不確かさ・長q systematic・全law・joint covariance・total CIは未評価のnull。観測maxを上界にせず、標本SEやαtotal=.05を価格の95%保証へ読み替えない。

## 技術検証

B専用`h6-environments/20261011/venv`、Python3.11.0rc1/NumPy1.26.4/Qiskit1.3.0で**34 test method PASS**。A時点のsnapshotだけでなくBの実module・実accounting依存を使用した。

[静的保全・SOURCE照合](CHECKS.txt)では、A取込みbytes・B旧結果/数値kernel・過去manifestを確認し、試験したB SOURCEのSHA256が最終実装と一致した。

| 試験 | 件数 | 記録 |
|---|---:|---|
| A providerの元人工・保存golden試験 | 8 | [A_PROVIDER_REGRESSION.txt](A_PROVIDER_REGRESSION.txt) |
| B pipelineの元人工/mock試験 | 12 | [PIPELINE_REGRESSION.txt](PIPELINE_REGRESSION.txt) |
| 新しいB環境結合試験 | 7 | [INTEGRATION_TESTS.txt](INTEGRATION_TESTS.txt) |
| 既存H6 N専用の人工回帰 | 7 | [H6_N_ONLY_REGRESSION.txt](H6_N_ONLY_REGRESSION.txt) |

元の68軸direct/Mq計136値照合を再利用。新試験はB factory→`run_missing`へ保存lookupを実接続し、Nはmock、実価格の新予測・積和を禁止して確認した。null/有限cohort/外挿/対象外の再呼出し抑止、部分軸の重複と補完、矛盾するcached価格、外国bindingのnull応答、paired cohort・model資格、不確かさ伝達を検査。人工の価格7はmetadata用の架空値でありH4の新予測ではない。旧moduleのDeprecationWarningはログに保持する。

今回、新しい実H4/H6のsignal/N・価格予測・数値再解析・fit・sampling・実回路build/compile・law/CI推定・実N×C/Gは全て0。人工試験PASSは科学的な精度保証・独立科学レビューではない。旧A/BのSOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOP、他担当tree/process、共有環境を変更していない。Bの新技術版でpipelineだけを拡張し、旧SOURCE版はGitに保持する。

## 残る科学項目と次の担当

- **ユーザー**：次の科学実行が必要なら対象・SOURCE/inputs・price policy・外挿採用・必要精度を決め、明示許可する。今回の接続完了は実数値会計の実行許可ではない。
- **Codex B**：別途許可された範囲でH4不足N（既存D1 carrierではm3以外）と、そのうち保存m4 signalからのscalar再利用を扱う。fresh H4 N leaf・未評価候補の科学資格は本作業に含めていない。H6の取得済み全13round Nは保全済み。
- **価格担当A/C・Codex B**：必要なら長q/全law期待C・共有校正/系統誤差・全体CIの不足を、別の有限科学範囲として設計する。保存lookup missを新compileの指示としない。全N/C・科学policyが揃った後の実N×C/Gも別許可が必要。

必要な技術部品・結合テストは完了。commit/push後にremote SHAと主要成果物を照合して**mandatory STOP**し、新しい科学計算・最適化へ自動進行しない。
