# B・D向け：全方式の供給価格モデルと受動provider

[REPORT](REPORT.md)／[次の科学条件](NEXT_SCIENCE_PLAN.md)。新コードは`rpe_mean_price_model.py`、testは`check_rpe_mean_price_model.py`。基点C3802の上へこの二つを選択取込みすればよく、B/A/Dのbranch全体のmergeは不要。既存mean_compiled_price、I/Bの価格会計、D request定義は変更していない。

## 1. 明示的なモデル評価

```python
from trottertracks import rpe_mean_price_model as P

scope = P.context(root, request, m, axis, profile_extension=source_bound_lowering)
model = dict(mode='ADDITIVE', name='EXTERNAL_PRICE_HYPOTHESIS',
             context=scope, source=coefficient_source, terms=external_terms)
record = P.quote_model(root, request, m, axis, model=model,
                       profile_extension=source_bound_lowering,
                       features=supplied_expected_features)
```

`terms`は`feature/coefficient/covers`の列。PFの簡単な人工例は、one×a（covers=SIGNED_PIECE_BOUNDARY/SCALAR/WRAPPER）＋pf_step_count×b（covers=PF_BODY/OUTER_BOUNDARY）。合算項のcoversを重ねず、未知の項は省略して価格nullとするか、既知zeroの証拠とともに明示する。q−1のouter_boundary_countを別項に使う場合は、step係数のcoversからOUTER_BOUNDARYを外す。

PRではRTE_BODYとPF_RTE_BOUNDARYも必要。Full-HではPF_BODY/PF_RTE_BOUNDARYを使わず、RTE_BODY＋境界・scalar/wrapperを指定する。CONTROLは別追加項ではなく元のcontrolled profileに含む。各寄与を独立fitする要求ではなく、全体の係数に含まれる寄与の二重計上を防ぐ定義である。

builtin featureは`one/q/pf_step_count/event_count/outer_boundary_count/signed_piece_boundary_count`。pf_step_countはFull-Hに存在しない。PFの内部相殺とRTEの内部相殺を同じ係数と仮定せず、q/Rにより変わる局所確率は外部feature側に定義する。外部feature entryは`{context: scope, quantity: EXPECTED_STRUCTURAL_FEATURE, value: supplied_mean_or_null, source: descriptor, ...}`。元q/R/countを上書きするfeatureは拒否する。

E[S]との結合はmode=`STRUCTURAL_PLUS_RESIDUAL`でterms無し。`structural`はquantity=`EXPECTED_FULL_WRAPPER_STRUCTURAL_RZ_COUNT`、`residuals`はquantity=`EXPECTED_FULL_WRAPPER_C_MINUS_S_RZ_COUNT`、同context/source、value、mode=`C1_AFFINE`又は`LOCAL_C_MINUS_S`、coverage=`FULL_WRAPPER`の**一つだけ**を渡す。RTE内部だけなら価格null。MeanKernelや旧cluster formによるfeature生成は別の明示API・科学scopeであり、quote/lookupから自動実行しない。

これは数値APIなので、実H4/H6の係数/feature/E[S]/残差を使う新しい評価には別途科学実行許可が必要。今回の人工係数は移送しない。今のrecordは全て未資格MODELで、SOURCEやdomain一致を科学的な外挿資格へ昇格しない。

## 2. Bへの受動接続

```python
provider = P.PriceProvider(root, request,
    records=already_supplied_direct_or_model_records,
    profile_extension=source_bound_lowering,
    selections={(B_candidate_id, 'X'): selected_cohort, ...})
inputs = provider()  # copy/selection/missingのみ。quote_modelもD取得も呼ばない。
result = I.compare(root, request, signal_packet, allocations, N_evaluations,
    **inputs, statistical_contract=existing_statistical_contract)
```

`inputs`は既存I/Bの`prices/cohorts/price_bindings/price_extension`。全0..M×X/Yについて元B identity/law、physical source/candidate、full compiler/lowering profile、measurement/unitを確認し、元price_sourceとphysical contextのsidecarを作る。区間は生成せずmean_interval=None。供給recordの未知CI/covarianceはそのまま残す。実入力でI.compareを行うことは今回の許可範囲外。

DIRECTは外部producerが供給し、quantity/mean/cohort/n_paths/expectation/qualification/source/uncertaintyをそのままcopyする。providerはreceiptを再監査せず、科学的な受入を新しく認定しない。finite-cohortを全lawへ変更せず、既存Bがpolicyに従って判定する。外部meanやsource bindingの不足がある場合はUNKNOWNで残す。

MODELには新しいexact contextとfalse資格が必要。BはUNQUALIFIED_MODELを正式Gへ使わない。モデル水準screeningの費用を別途評価する場合は、ユーザーが許可範囲・主張水準を定め、Bが明示scenarioを管理する。Cのproviderに正式G guardを回避する機能はない。

複数cohortのpool、q/R/別formula/prefixへの暗黙移送、X値のYへの補完、normalization B再乗算をしない。S/path/local residualは平均compiled quantityとして受け付けない。scalarはIの既存split/physical SOURCEを保ち、Dのexpected_scalarや実preparationの値をCで修正しない。

## 3. 残る担当

人工・回帰の再現コマンドは次のとおり（既存プロジェクト専用のNumPy/SciPy利用環境、テスト以外の実入力不要）。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests python -B -m unittest check_rpe_mean_price_model check_mean_compiled_price check_rpe_optimization_pipeline -v
```

今回の実行には既存専用venv `/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python` を使用した。38件全PASS、scientific input factoryを新しく起動していない。H6の新domain接続や未対応formulaを本テストで資格化していない。

- **B**：N/risk/主張水準・候補family・必要な価格精度を供給し、モデルと適格価格を別に扱う。
- **D**：SOURCE・scalar/law/lowering整合後、別承認の有限範囲でpaired full-wrapper direct/残差・校正/holdoutを取得する。
- **ユーザー**：未資格の加法性・残差移送を検証候補として扱い、科学実行範囲と採用条件を判断する。
- **C**：今回の人工・技術成果のremote照合後STOP。保存26 E[S]、C1係数/qualification、S0a/C2/T2とベースライン計画のSTOPを保全し、新S/C/fit/Gを開始しない。
