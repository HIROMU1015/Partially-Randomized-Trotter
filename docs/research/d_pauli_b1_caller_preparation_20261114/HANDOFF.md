# Pauli B1：A/Cへの接続

実装SOURCE：`bd1b1b2862da6d174fa521e7612051810a636e7f`。A正本：`7a0c9468b2499fac9fa1cd279ae2980c9dab048e`。基点B0：`00f74f86d16cc6878e1d4c5977eb9abd9ce9c076`。C planner/model/guardは `c628e9e8447c11c984064b922cbce6c52dc72d48`。科学計画は全て未採択、mandatory STOP。

## metadata caller

repository rootのQiskit 1.3.0を持つproject専用Pythonで実行する。以下はmetadataだけを表示する。

```bash
python -B scripts/run_pauli_b1_prices.py --prefix 2 --kind B1-W
python -B scripts/run_pauli_b1_prices.py --prefix 1 --kind B1-CAL
python -B scripts/run_pauli_b1_prices.py --prefix 2 --kind B1-CAL
python -B scripts/run_pauli_b1_prices.py --prefix 1 --kind B1-V --declare-b0-path0
python -B scripts/run_pauli_b1_prices.py --prefix 2 --kind B1-V
```

[PROPOSALS.json](PROPOSALS.json) にrequest・seed表・再利用件数を保存した。[JOB_TEMPLATES.json](JOB_TEMPLATES.json) は `adopted=false` / duplicate照合false、LD2 W参照未設定という実行不能の例である。実行する計画の採択・科学実行許可を代替しない。

将来の明示許可後の入口は `--execute-approved-job JOB.json`。一回につき一cell、明示n_pathsとX/Yだけを扱う。現在実行しない。完了/失敗でSTOPし、次cell・path・seed・retryへ自動進行しない。

## APIとphysical binding

- `proposal(prefix,kind,reuse_b0=False,reuse_wiring_source=None)`：AのCAL cell表を利用。W/Vはrank1配線/ばらつき用途で、CALへのrecord混入を禁止する。
- `request_for(plan,cell)`：canonical m0/m5/m6はC `price_context` のIDと照合。非configurationはcanonical anchorのH/state/snapshot、split、辞書、profile、route/lowering、q/R/T/Kを結ぶ `PAULI_PRICE_CELL:<sha>` を使う。Bの別configuration IDへ置換しない。
- `seed_table(plan,cell_id)`：元 `H46_LONG_TIME_20261025` 系のcandidate/cohort/index/outer/piece規則を再利用。paired q2は1 step cellのpath0/1のseedを順に使用。
- `acquire_path`：保存prefix別P0とB0 coreへ接続。`sample_steps(...step_seeds=...)`、`compose(...step_diagnostics=...)` は旧引数の既定動作を保つ。

nonconfigurationのTは元binary64の0.1*q、step_time=T/qである。metadataではnormalizationを新計算せず、既存finite-Kルールの実行入口を保持する。実行時は辞書のλ、normal sampler、native DF head、Pauli D2 tail、scalar `c_H+s_R`、full測定wrapperを既存経路で構成する。異なるcandidateへscalar契約を横展開しない。

## 再利用・paired・stage

LD1 B1-VのB0再利用は事前宣言がある場合だけ。元B0 raw `8502dd33ad16dc012fb3558668ef3e4cd8a4405c:.../run02/RESULT.json`、cohort `D_PAULI_LD1_M0_WIRING_V1`、index0、seedを保持する。既存B0は再sampling/compileしない。宣言なしのfresh n2をB0の独立再取得として自動実行しない。

LD2 B1-VはB1-Wの続きとする。公開Wの`{commit,path}`を `reuse_wiring_source` に結び、Wのcohort/index0/seedとrequest、作用/compiled-vector gateのmetadataを確認する。Wがない現状では実行不能である。W path0は保存標本、path1だけが新規であり、新独立標本2件と数えない。cohortを変更したWを使う場合、後続Vも同じcohortに整合させてからSOURCEを固定する。

B1-CAL LD2は `ld2_wiring_source` の公開W metadataを要求する。CALとHOLDOUTはAの完全planで分離し、HOLDOUT前はCの凍結X/YモデルSOURCE・domain/features/CAL cell/cohort/seedをmetadataだけで確認する。モデルのfit/predict/判定はDでは行わない。

paired用`parents`は、1 step path0/1の完全なreceiptをその順で渡す。親request・cohort/index/seed、event label/order/product/step phaseを確認する。連結後のscalarをtotal Tで一回に保つ。pairedと親の共分散を独立性で置き換えてはいけない。

## Cへの価格record

成功した将来のcellだけが `RESULT.json`、`C_INPUTS.json`、`PRICES.json`、`SIDECARS_UNBOUND.json`、`IMPLEMENTATION_EVIDENCE.json` を出力する。今回はこれらの実科学結果は作っていない。

| 対象 | C/Bへの入力 |
|---|---|
| CALの非configuration cell | C `PRICE_CALIBRATION_RECORD_V1`、feature `PR_STEP_V1=(one,pf_step_count=q,event_count=qR)`、X/Y別 |
| CALのm5/m6 | 同形式HOLDOUT＋canonical round価格/sidecar |
| W/Vのm0 | canonical roundのfinite recordだけ。C_inputsは空、モデル較正禁止 |

`DIRECT_COMPILED_FINITE_SAMPLE` / `FINITE_COHORT_ONLY` / n_paths / law expectation falseを保持する。少数pathの平均を適格E[C]へ昇格させない。Pauli構造kernelや残差fit、Sのcompiled上下界化は実装していない。

`source.commit` は実行code SHA、`result_path` は結果出力先。結果のpublication `{commit,path}` は公開後に別途渡す。implementation evidenceをcommitした後、`check_round_records(...evidence_commit=公開SHA)` で固定C guardを検査し、返却 `bound_sidecars` を新しい公開sidecarとして保存する。未結合sidecarを受入完了と扱わない。旧raw/recordの改変は不要。

A planのprofile記法とcanonical configurationのprofileは各既存APIの正本を使い、字面だけを揃える書換えをしない。下流比較ではroute/law/dictionary/native/scalar/lowering/compiler/wrapper全体を照合する。compiler hashだけで同一quantityと認定しない。

## 技術検証と範囲

```bash
python -B -W ignore tests/check_pauli_b1_prep.py
python -B -W ignore tests/check_pauli_b1_circuits.py
python -B -W ignore tests/check_pauli_ld1_m0_pilot_prep.py --metadata-only
```

最後のsuiteはdefaultも保存H4数値caseを除外する。今回の最初のunfiltered実行による範囲逸脱は [SCOPE_DEVIATION](SCOPE_DEVIATION.md) に保全済み。`--include-saved-numerical` は将来の別途許可がある範囲専用であり、今回使わない。検証の内訳と既存コード保全は [REPORT](REPORT.md) を参照。
