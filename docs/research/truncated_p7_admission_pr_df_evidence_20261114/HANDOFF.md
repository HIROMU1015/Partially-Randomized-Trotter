# D/Bへの受渡し：保存4価格とPR-DFの資格

## D Stage 0 / metadata更新

正本は[SAVED_RECEIPT_SOURCES.json](SAVED_RECEIPT_SOURCES.json)。4件の元price・C入力record・bound sidecar・guard結果を、すべて固定commit＋path＋axis selectorで指定した。

- 新版計画：`43c15bfa34a22f313e55b824223b200332b7f23f:docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json`
- p7 price/入力/assertion bundle：同`43c15bfa…`。p7 bound sidecar/guard：`2b94539784995790c8ff1ba7dee578137bfc6034`の同ディレクトリ。
- Full PF保存2件は従前どおりC `2fa88065318f0f2ab13937ef610828200b35874c`。原price SOURCEは両方式とも155529e2/EXPLORATIONのまま。
- actionは旧interfaceの`SAVED_RECEIPT_FOR_C_ACCEPTANCE`を踏襲。新p7のC受入は既に完了し、再compile対象ではない。

Dの固定参照 `c848bbefa30c23a9ee69170482fb02accae31601` は既に較正/holdoutを分離している。今回Dへ必要な差分は以下だけである。**CはDのファイルを変更していない。**

1. `pf_price_stages.CELL_PLAN`を新版へpin。旧planも履歴/固定SOURCEに保持する。
2. p7 `plans()`でM5/m4を保存actionにし、上記4件の参照をroute別に読む。`saved_calibration`のFull PF専用ID/route前提を、p7の既存C recordもコピーできるようにする。元SOURCE/cohort/record値を新cohortへ移さない。
3. Truncated較正の新規wrapper期待数を12→10へ。Full PF較正10、各holdout6はそのまま。CALIBRATION20＋HOLDOUT12＝32新wrapper、保存4、全36価格。
4. C_INPUTSの12 model_groupは不変。両方式のM5/X/Y groupで、保存m4 recordと新m5 recordを明示的に結合する。他groupは新m4/m5の2点。
5. 各groupのSOURCE・domain・公式/actual幅/軸/roleを検査する。旧guardの不一致を無効化しない。新版planと旧T3 actionの組合せが停止することは今回の人工testで確認済み。

保存p7の入力2レコードは、D c848の現`fit_inputs(roles={4:'CALIBRATION'})`で整形した結果とも完全一致した：[C_INPUT_COMPATIBILITY.json](C_INPUT_COMPATIBILITY.json)。これはrecord変換だけで、係数fitではない。

保持する条件：D62のclassification/expectation/schema、native/controlled/測定wrapper、元Hamiltonian/state、B0I scalar/tail identity、Qiskit/profile、index0の元gates、fresh output、SOURCE/入力保全、retryなし。p7保存receiptのfingerprint・元trial・cache locator/hashはADMISSION/SAVED_WRAPPER_CHECKSを参照し、単なる同数価格で別回路を結合しない。

## CALIBRATION取得後のC入力・凍結

既存D `C_INPUTS.json`の `saved_records[*].record + new_records`を、groupごとにCの`price_model_validation.fit(features=['one','q'])`へ渡す経路は維持できる。今回はrecord整形だけで、fitはしていない。

- m4/m5の2点のみでfit。方式2×公式3×軸2＝12 model、poolなし。
- DのHOLDOUT gateに渡すJSONには、既存fit出力の`PRICE_LINEAR_FEATURE_MODEL_V1`、`CALIBRATED_UNVALIDATED`、qualified=false、同じdomain/design_source/calibration(cell_id, source)をそのまま固定する。
- modelと許容差/比較policyのSOURCEを固定後にだけm6を取得する。HOLDOUT判定は別ファイルへ保存し、元の凍結modelを変更しない。
- 観測値分母2%は[前回提案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2fa88065318f0f2ab13937ef610828200b35874c/docs/research/m5_m4_receipt_admission_20261114/POLICY_PROPOSAL.md)のまま未採択。結果を見て閾値を変えない。
- m6でFAIL/UNKNOWNなら、部分保存STOP。成功してもm7〜12の外挿は未資格のまま。

新価格取得/実fitに関するDの既存承認をCが拡張解釈しない。今回C scopeは技術受入と計画更新のみで、D実行・実fit・Gを開始していない。

## B / PR-DF

p7保存価格は元configuration・物理candidate・profileの範囲でDETERMINISTIC_DIRECT。decoder由来のcanonical IDが変わる場合はB eecの`rpe_total_binding.bind_prices/check_prices`を使い、元receipt/assertionを保持する。単純なID付替えをしない。正式比較policyは未採択なので、本件のprobe policyをG用policyとして流用しない。

PR-DFは価格recordを新発行していない。[PR_DF_COVERAGE.json](PR_DF_COVERAGE.json)は証拠索引で、Bの価格provider packetではない。

- LD1：E[S]26軸、m0/m6の新ZZ paired C/SはP1とP2の別有限cohort。consumer bridge未受入。after-admissionの第1水準はFINITE_COHORT_SCENARIOまで。
- LD2：同domainのE[S]/paired価格は未取得。
- C1：旧prefix1/R4,8/旧RZZの校正・検証。現R/新ZZ/prefix2へ係数を移植しない。
- `STRUCTURAL_PLUS_RESIDUAL`は同じS observable、同path/axisの残差、同target law/contextの**単一残差**だけを組み合わせる。law差・finite cohort・rare・共有systematicが未資格ならMODELはfalseのまま。
- ES単独をE[C] recordとして返さない。欠測はnull。Bに適格Nがあっても、正式G/winnerを決めない。

残る検証の具体案は[PR_DF_VALIDATION_PLAN.md](PR_DF_VALIDATION_PLAN.md)。AのPauli/Full-Hモデル、BのN solver、Dのcompiler/callerを重複実装していない。公開後 **mandatory STOP**。
