# A → B：人工PF power調査と次の診断入口

2026-11-13 JST。[REPORT](REPORT.md)／[人工測定値](RESULT.json)／[集計](SUMMARY.json)／[次の有限案](NEXT_SCIENCE_PLAN.md)。実H4計算・保存数値再解析0。旧科学SOURCE256caad34とrun01/run02を変更せず、B/C/Dのworktreeにも変更していない。

## 得られた技術証拠

36次元・11人工block・固定seed2種の192条件。元EVRは7/48、EVD4/48、EV1/48で1e−9のunitarity gateを失敗し、直接expmは48条件とも通過した。ordinary fixtureでは全state norm gateは通過した。driver切替だけを修正完了としない。直接expmは次の実照合候補であり、採用済みproduction backendではない。

零eigenspaceへ手動でGram defectを入れた人工対照により、spectrum再構成gateを通過してもpower後のgateが失敗する機構を確認した。実H4の失敗値を取得した証拠ではなく、run02の原因は未確定。

## 再利用API

新`src/trottertracks/pf_power_stability.py`は次だけを提供する。

- `spectral_metrics(matrix, spec)`：同じeigentripleの再構成、固有方程式、Gram、zero-time identity defect。既存`algorithm_codesign.numerics.Spectrum`の条件付きobserverを供給eigensystemで再利用。observerの独自入力条件・boundを科学契約へ採択しない。
- `explicit_spectrum(matrix, driver)`：EVR/EVD/EVを明示して比較。fallbackなし。
- `expm_step(matrices, weights, delta)`：元forward/reverse順とsigned weightsの独立数値対照。scalarは外側callerが旧規則で一回会計する。
- `power_diagnostics(step, q, state)`：powered matrixとnorm/unitarity値・各gateのboolを返す。**gate失敗でも残差値が先に返る**。stateやoperatorの投影・再正規化なし。非有限は例外、結果は未認証。

元の`h4_four_method_screen.execute`やmean leafへは接続していない。将来これらを用いる実診断は、別の固定SOURCE・fresh出力として旧rawを保全する。global RMSE、decoder/shot統計、価格C/GのAPIを追加していない。

人工再現は次の入口。実分子入力を渡すoptionはない。

```bash
python -B scripts/check_pf_power_stability_artificial.py --out <fresh-artificial-RESULT.json>
python -B tests/check_pf_power_stability_artificial.py -v
```

NumPy1.26.4/SciPy1.14.1で新8＋旧9の17技術test PASS。driver/default仕様は[使用versionのSciPy公式仕様](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.linalg.eigh.html)で確認した。

## 次の担当と制約

**ユーザー**：代表1点F2-COARSE/m12の旧spectral対direct-expm、二経路だけの診断を実行するか決定する。**A**：承認後だけ、両経路を各一回実行し、gate値を保存して局所結果として返す。旧gate失敗は診断対象として保持し、閾値変更・自動fallback・新referenceを行わない。

**B**：元4局所record・未評価16セル・全RPE資格未取得を保持する。今回の人工結果だけでFull PFの末期signal、十分N/global RMSE、最適方式を認定しない。実H4の診断が成功しても、残16セルや全13roundへの再走は別scope。技術成果公開・照合後mandatory STOP。
