# B/C HANDOFF：固定H4 S0の価格request

技術SOURCE `bdd2b93565c0994ab5682e131dbccc74912197d3`。branchは`codex/r-variable-compiled-price-design-20261111`。新科学計算・価格値なし。実行中B worktreeを変更していない。

## Bが取り込む部分

- 新`src/trottertracks/h4_s0_price_plan.py`、`tests/check_h4_s0_price_plan.py`と本ディレクトリ。既存`bv2_mean_price.py`、Bのcontroller/integration/MeanPriceLookupが依存先。Cのbranch全体を強制mergeする必要はない。
- `prepare(repo_root, integration=B_integration_module)`は固定A JSONをGit objectから読み、B所有の`generate(a_spec(...))`でV2 metadataを生成する。A.evaluate、物理preparation、law数値評価、S、sampler、compiler、fit、Gは呼ばない。
- 固定B coreは`6a9d7fd36b687e58aecbed1278c6d7927d6827d6`。指定最新`af284d853890155bacd19e61850c196bd1d5c660`のcontroller/integrationとbyte一致。新共通APIを作っていない。
- C WTの旧依存moduleでB最新版を代用しない。技術テストは既存`check_bv2_mean_price`の固定Git loaderを再利用し、B exact-scalar accounting/pipelineとA/B module定義を同一test processへ読み取る。B作業ファイルは変更しない。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests /path/to/project-python -B tests/check_h4_s0_price_plan.py
```

既存専用venvは`/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python`（Python3.12.3/NumPy1.26.4/Qiskit1.3.0）。今回テストは7件PASS、実H input arrayは読み込んでいない。

## 暫定metadataと正式B publicationの照合

A S0 `7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`のCONFIGURATIONはV1。元packetを変更せず、そのexact inputsからCがV2を再構成した。`PRICE_REQUESTS.json`は**C_RECONSTRUCTED_FROM_FIXED_B_NOT_YET_B_PUBLISHED_REQUEST**である。

- V2 configuration: `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`
- m0 physical: `53e00d8b07d1547d19a7`、B: `RMSE:e9ac70f2efddf2b25d621da1a2ee0f3d43c1248eb15ef8ddabeaae3c840954c8`
- m6 physical: `2a8c6718220eec59d419`、B: `RMSE:a4ed2fac90ddbd6bf8e7d5e0077157a66eade4a657d8db01ebf4ad973781def0`
- M12/13round、κはA正本のexact文字列。h、split、normalization、profile、元law/source宣言も変更しない。

Bが正式configurationを公開したら、C側の`verify_B_publication(plan, published_configuration, integration=B_integration_module)`でB自身のvalidate・configuration IDとm0/m6 identity/law/physical recipeを照合する。不一致なら値をrelabelせず差分を確認する。一致はmetadataだけの結果であり、価格資格や全RPE保証ではない。

今回の13round coverageはC/Aの暫定projectionで、Bの正式coverage作成を代替しない。m0〜6の旧索引recipeには価格なし。m7〜12は固定exactκのRが旧索引と異なるため、その欠測を維持した。全26軸mean C、full Gはnull。

## 将来の価格record

既存B `integration.MeanPriceLookup`とC `V2MeanPriceLookup`形式をそのまま使用する。新しい共通record schemaは導入しない。

1. 正式Bの全identity、physical ID、configuration/M/κ、law_binding、source、profileを一致させる。bindingの元参照T=.8で各候補Tを上書きしない。
2. X=`cosine`、Y=`sine`、RZ count/測定1回のcontrolled PR + full Hadamard + measurement、P0。scalar −cT一回、head/tail/outerの元順を保持。normalization Bは価格乗数にしない。
3. `quantity=FULL_WRAPPER_MEAN_COMPILED_PRICE`。保存/新DIRECT cohort平均とMODELを区別し、cohort・path数・seed/event provenanceを持たせる。複数cohortをpoolしない。
4. PRの有限cohort平均は`FINITE_COHORT_ONLY`。同path X/Yのpaired関係とnを維持し、全law期待値を宣言しない。samplerとideal法則の距離は未資格。
5. MODELはdomain、凍結SOURCEと校正/holdoutの役割を保持。C1 `qualified_for_candidate=false`。R2/q1とR128/q64への固定R係数の自動fallbackは禁止。
6. rare寄与、長q/R systematic、係数/round間共分散、total CI等の未取得はnull。S0a Sや平均S/local内部Dを平均compiled Cとして受け入れない。

現recordのprice/cohortはnull。値を供給していないので、上記quantityは要求対象を示すだけである。値ありrecordの適合性テストは前段12件を再利用し、新7件では固定S0との接続を確認した。価格許容誤差・採用policy・全体十分Nは未確定であり、Cから勝手に設定しない。

## 次の承認範囲

第一候補Aは同H4 preparation 1個とR2/R128の2lawの合法support/価格状態/flushの診断。新trajectory/full-path S/compile/fit/mean C/Gなし。物理解析のSOURCEと手順を固定してから、ユーザーの明示科学実行許可の範囲にする。

代替Dは各候補normal2path、同path XY、計4path/最大8 wrapper requestとpaired S/Dの有限pilot。新しい全law平均やfitを目的にしない。これは条件付きの提案で、今回の実行許可ではない。sampling/compile/S/D/有限cohort記述の各処理を実行scopeに明記し、seed/cohort/leaf SOURCE/STOPを固定する必要がある。n=2は価格精度保証のために決めた標本数ではない。

B/C残差モデルの校正・fresh holdoutはその後の別計画。Dを実施したら同path S/Cを再利用して重複取得を避ける。旧C2 40/16案、S0a再実行、H6や他11round、N/Gへ進まない。

次の担当：**ユーザー**が有限科学案を選択・許可、**Codex C**が選択scopeのSOURCE/入口を確定して実施、**Codex B**が正式V2 configuration/coverageを公開してmetadataをjoinする。今回の成果公開・remote照合後mandatory STOP。
