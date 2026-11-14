# A → D/C：Full-H P-2・Pauli B1の実装と実験の引継ぎ（いずれも未承認）

[REPORT](REPORT.md)／[A1](A1_FULL_H_P2_PLAN.md)／[A2](A2_PAULI_B1_PLAN.md)。

科学実行（sampling・compile）はユーザーの明示的な承認が出てから行う。ここに書くのは技術上の準備だけである。

## 1. 共通：取得前の設計検査（`src/trottertracks/finite_price_design.py`）

```python
from trottertracks import finite_price_design as D
summary = D.check_plan(plan)          # rank・領域・較正/holdout分離・wiring除外・paired規則を取得前に検査
out = D.fit_identified(root, records, plan=plan, axis='X')  # Cのfit（c628e9e8 blob、無変更）へ渡す前に照合
M = D.c_model(root)                   # Cのprice_model_validation（validateは無変更のまま使う）
```

- **plan**：`feature_map`（`FULL_H_GROUP_V1`または`PR_STEP_V1`）、`cells`、`design_source`の3つを持つ。
  - 各cellのfield：`cell_id`、`role`、`q`、`R`、`n_paths`、`route`、`law`、`implementation`（`lowering_id`・native・scalar quantity）、`profile`、`quantity`、`kind`、`expectation`。
  - 任意のfield：`composed_of`、`cohort`、`declared_reuse`。
- **record**：Cの`price_model_validation.record`形式で、軸ごとにmodelを分ける。
- 設計候補は[BUILD_DESIGNS.py](BUILD_DESIGNS.py)で再生成できる（metadataだけを使う）。

## 2. D（取得）：Full-H P-2のcaller拡張

P-1 caller（`scripts/run_full_h_kappa2_p1.py`、`8258b2200`）を基にする。

1. **cellの指定**
   - 非configuration cell：q=1、T=0.1、任意のRの単一group。q=8、T=0.8、R=566の複数group。
   - configuration round（m5・m6）。
   - physical rowは既存の`FULL_H_` cidの規則で作る。
2. **paired composition**
   - G566のpath0・path1のgroup seedを、`native.trial_seed`の差し替えでそのまま連結する。
   - T=0.2、q=2として1 pathだけ組み立て、compileする。
   - 連結した回路のevent列が、両group pathのevent列と一致することを検査する。
3. **index0の作用検査**：既存の7 gate、新ZZ、parity、`scalar_event_phase_gate`（`8258b2200`）は、各cellで保持する。
4. **記録**
   - 非configuration cellは`price_model_validation.record`で記録する。
   - roundのholdoutは、既存のB-kind record・sidecar・C guardで記録する。
   - いずれもn_paths・law・cohort・seedを持つ。
   - pathごとのevent metadata（order、成分、basisの再利用）を保存する。
5. **資源**：段階的に取得する。
   1. s≤9056のcell（G566、P2R566、Q8R566、m5）を先に取得する。
   2. 時間とRSSを観測してから、G4528・G36223・m6へ進む。
   - 独立なcell・軸はprocess並列にできる。共有環境の設定は変更しない。
6. **規模**（提案）
   - D7は24 compile。
   - holdoutの判定にH-dを使う場合は、m6をn2にして26 compile。

## 3. D（取得）：Pauli B1のcaller拡張

Pauli pilot（`62ae1bd5`／`9abbcc56`）を基にする。

1. prefix（LD1/LD2）とcellを引数にする。非configuration cell（1 step・8 step）とroundのholdoutの両方を扱う。
2. paired 2 stepのための明示的なstep seed表を受け付ける。
3. 独立したfull-2^8 oracleとindex0のgateを、各cellで維持する。
4. pathごとにevent metadataを保存する（label、order、積、step位相）。
5. 順序：B1-W（LD2 m0、2 compile）→ B1-CAL（LD1、24）→ 資源を観測 → B1-CAL（LD2、24）。B1-V（m0の追加path）は任意。

## 4. C（受入・model）

- 取得の**前に**、次の3つをSOURCE付きで固定する。
  - feature design（`FULL_H_GROUP_V1`／`PR_STEP_V1`）
  - 較正cellの一覧
  - holdoutの許容差（A8、ユーザーが決める）
- fitは`D.fit_identified`経由で、Cの`fit`を無変更で呼ぶ。holdoutは`validate`を一回だけ行い、失敗してもrefitしない。
- 単一pathのrecord（P-1、B0）は、`classify_saved_price`で`SINGLE_PATH_DIRECT_FINITE_SAMPLE`と判定される。これらは較正・holdout・E[C]・Gに使わない。
- 外挿（m7〜m12）の扱いは、`UNQUALIFIED_MODEL`のまま変えない。

## 5. ユーザーが判断すること

1. A1：較正設計の採択（D7、D6、別案）と、P-2の実行承認（compile数・段階）。
2. A2：B1-W・B1-V・B1-CALのそれぞれについて、要否と実行承認。
3. holdoutの許容差（A8）と、interval contract（n・SE・coverage）。
4. 比較policy（A9）。
