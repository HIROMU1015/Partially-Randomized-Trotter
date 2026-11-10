# Phase 1結果：Codex B向けHANDOFF

固定科学SOURCE `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`、係数凍結commit `7eeabf0ae482ee610d667e28470692d209f2bb21`。[REPORT](REPORT.md)の8セルを実行済み。検証取得前に係数をpush・照合し、validation再fitなし。通常64＋forced16 compile、保存C/S59 path再利用。実行は[mandatory STOP](run01/STOP.json)に到達した。

## 保存recordの取込み

C branchをfetchし、この結果REPORTを含むpublication commitを40桁SHAで固定する。Bの科学SOURCE保全期間/STOPと自分の変更を確認してから、結果directoryを選択して取り込む。branch全体mergeや共通N/C/会計moduleの過去版restoreは不要。

```bash
git fetch origin codex/r-variable-compiled-price-design-20261111
# 確認したpublication SHAを使い、B自身のworktreeで実行する。
git restore --source=<publication-SHA> -- docs/research/r_price_phase1_results_20261111
```

数値の供給元は次の2配列。これは保存値lookupであり、fit/prediction関数を再実行しない。

| 保存file | JSON配列 | record数 | 内容 |
|---|---|---:|---|
| [CALIBRATION_EVALUATION.json](run01/CALIBRATION_EVALUATION.json) | `prices` | 24 | 6校正セル×XY×direct/model |
| [VALIDATION.json](run01/VALIDATION.json) | `prices` | 8 | 2独立検証セル×XY×direct/model |

```python
import json
from pathlib import Path
root = Path("docs/research/r_price_phase1_results_20261111/run01")
price_records = [
    record
    for filename in ("CALIBRATION_EVALUATION.json", "VALIDATION.json")
    for record in json.loads((root / filename).read_text())["prices"]
]
# 保存recordだけをc_provider(task, requested_rows)へ供給する。
# candidate全体・qPF・R・law・compiler・axis・cohortで照合し、欠測はmissing。
```

現B固定APIは`7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae`。Aは`5ca2dce3234f3e4a971f4a8c852da676f050e315`。準備の公開A/B metadata人工検査23件を再利用している。B最新版の全runtimeを結合検証したという意味ではない。CはA/Bの作業中領域を変更していない。

`identity.binding.T`とmodel.binding.Tは旧B task参照値.8を保持する。実時刻は**identity.candidate.T**（H4 .8/1.6、H6 1.6/3.2）で、回路時刻はこれを用いた。固定共通APIはその参照bindingを含めてidentityを比較するので、C recordだけのfieldを変更しない。必要な正規化はBのmenu/N/Cで一貫させ、元結果を保存した派生metadataとして扱う。candidate.T/qPF/R/lawを省いたcache keyは禁止。

## 資格と不足の扱い

- `DIRECT_COMPILED_FINITE_SAMPLE`と`MODEL_PREDICTION`は別cohort/scenario。両方を同時加算・poolしない。PR価格は全て`FINITE_COHORT_ONLY`。量子shot Nと価格path nは別物。
- 直接有限cohortを使う場合も、Bの明示`finite_cohort_scenario`と対応N側の資格が必要。`qualified_expectation`を満たす全law期待価格は得ていない。
- modelは校正または`FINITE_INDEPENDENT_POINT_EVALUATED_NOT_ADOPTED`。採用閾値未設定のため`qualified_for_candidate=false`。誤差を見て閾値やfitを後付け変更していない。
- 係数nominal covariance、paired価格/残差covarianceは保存済み。direct校正点の`calibration_data_group`は対応fitと共有し、独立検証directへfit由来の係数共分散を付けない。未知joint/total CI/longq-R systematicはnull。
- normalization BをCに再乗算しない。full wrapper/P0・元compiler・元Hamiltonian/prefix/formula/δ/K・law bindingを保つ。
- H4 q8/16・R4/8、H6 q1/2・R16/32の**上表8セルだけ**。補間や外挿、H6短roundのp5/別δは自動供給しない。旧固定RのMq/Affine providerを別Rへ拡張しない。
- モデルのSには各pathの構造価格が必要。今回の保存価格lookupを未取得q/R・全law期待Sの計算器と扱わない。
- [rare診断](run01/RARE_DIAGNOSTICS.json)は通常価格供給へ入れない。normal新32 pathでorder2未観測、forced4 pairのΔCはH4 268/192、H6 141/1096（各軸）。全law重み付き寄与はnull。

保存`source`に加えてpublication SHA、frozen SHAを参照する。元価格・係数・marker/STOPを上書きしない。private rawはC保全先のまま、共通pipelineの実行にコピーを要求しない。

## 次の作業とSTOP

ユーザー：許容価格誤差と、全law/rare/追加q-Rについて次に必要な科学範囲を判断する。今回の独立点観測だけで全13round/R menuを採択しない。

Codex B：自身の許可scope内で保存recordのlookup、identity/law/有限cohortの資格を接続する。N不足・モデル未資格・価格欠測を保持する。N×C/G・科学的R最適化はこのHANDOFFによって許可されない。

Codex C：公開・remote照合後mandatory STOP。追加sampling/compile/fit/価格予測/再解析/N/G・詳細科学レビューは開始しない。
