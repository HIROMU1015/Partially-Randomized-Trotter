# Track A AX-1a：評価・不確かさ・終了仕様

2026-10-09、version 1。**評価値は未計算、AX-1b未認可。** [結果前契約](track_a_ax1a_preanalysis_contract.md)、[モデル・fit仕様](track_a_ax1a_model_comparison_and_fit.md)と一組。

## 1. 入力、join、保存参照の意味

allowlistは `artifacts/resource_applicability/track_a_ax1a/2026-10-09/input_allowlist_v1.json`。各entryのpath/evidence commit/source provenance/bytes SHA-256/schema/必要field/join/情報区分を検査する。JSON内のpathはデータとして扱い、read authorityを付与しない。allowlist外のNPZ/runtime/registry/cacheへ追跡しない。hash/schema/identity不一致はSTOP、欠測したscience値をseedから再生成しない。

M1-A candidate ledgerとsignal recordsをfingerprintで1対1joinし、M1-B1のcompile mapと同fingerprintでjoinする。candidate_idだけではjoinしない。Hamiltonian/state/vector/snapshot hash、T/delta hex、prefix/q/r/K、outer formula/identity extraction、compiler/wrapper identityが一致していることを確認する。PM-1のsignal/compiled fingerprintも検査する。M2はdevelopment fingerprintを費用由来の識別に使い、geometry実行fingerprintと混同しない。M2 signalとcompiledは実行fingerprintでjoinする。M2 candidateにはT_hex/delta_hexとsnapshot hashesがないため、保存T/delta literalとrootのinput_snapshot_identityを使う（hex未保存はnull、geometryの違うdevelopment hashと一致させない）。candidate集合はallowlistに明記したmembershipを固定する。

direct H4 development集合はM1 210 + PM-1 8 = 218。M2の5は別geometry集合。PM-0とPM-2は既存direct値を再編した派生証拠であり新独立標本でない。PM-2の精度点でone-shot Cを重複fitしない。

参照Cはfull wrapperの保存cost標本から求めるaxis平均（random32、deterministic1）であり、randomで未知の真の期待費用と同一ではない。参照signalは当時の決定的な平均作用計算で、cost trajectoryのMC平均ではない。現在のcontract auditは参照値の正しさを新計算で検証した意味を持たない。

## 2. 四つの要因分離case

| case ID | 定義・情報 | availability |
|---|---|---|
| REF_SHOT_DIRECT_COST | `Σ N_ref,a C_ref,a`、保存bias/Bに基づくbenchmark reference | SAVED_REFERENCE |
| REF_SHOT_PRED_COST_CONDITIONAL_ORACLE | `Σ N_ref,a C_pred,a`、参照eligibility/shotを使用 | CONDITIONAL_ORACLE_COST_ONLY |
| PRED_SHOT_DIRECT_COST | `Σ N_pred,a C_ref,a`、shot modelだけを検査するoracle-cost case | N_A_NO_OPERATIONAL_BIAS_PREDICTOR |
| PRED_SHOT_PRED_COST_OPERATIONAL | `Σ N_pred,a C_pred,a`、prediction時に参照bias/Cを使わない | N_A_NO_OPERATIONAL_BIAS_PREDICTOR |

最後二caseをBだけから埋めない。M2 `predicted_work_by_metric`はdev compiled C × held-out参照Nの条件付き予測であるという既存分類を維持する。図表・filename・column・regretに `conditional_oracle` / `operational` / `in_sample` / `internal_group` / `observed_geometry` を付け、無修飾の「総資源予測精度」に合算しない。

## 3. RQ-P1のcost指標

C_ref>0の同scope/unit cellに対し、axis別に定義する。

`e_signed=(C_pred−C_ref)/C_ref`、`e_absolute=|C_pred−C_ref|`（RZ）、`e_abs_relative=|e_signed|`、`e_log=log(C_pred/C_ref)`（両者>0）、`under_fraction=max(0,−e_signed)`。

過小評価率は `count(e_signed<0)/count(valid relative-error cells)`、material underestimation率は `count(e_signed<−.10)/同分母`。保存M2の異なる分母定義はそのまま残し、対応時に表示する。誤差の平均/median/maximumと分母をaxis/method/prefix/q/R/K別に示す。missingとzero-refをdenominatorから除いた理由・数を必ず併記する。

C_ref=0はabsolute errorのみ、relative/log N/A。C_pred=0かつC_ref>0はsigned=-1、abs-relative=1、logは `ZERO_PREDICTION` でnull。負cost/nonfiniteは通常値でなくSTOP/NUMERICAL_UNDEFINED。ineligible候補もone-shot cost診断に残す。

coverageはdefined predictions/registered rows、欠測率はINPUT_MISSING rows/registered rows。別にidentity failure、unidentifiable fit、zero-ref、extrapolationを区別する。structural model全件N/Aが他modelのcommon supportを空にしないよう、available model間のpairwise supportを明示する。結果に応じてcandidateを削除してcoverageを上げない。

action indexは単位が異なるためRZ relative/log errorを出さない。A_ceilとRZの順位一致はindex診断（tieは平均rankのSpearman）とし、RZへの係数1代入はしない。保存W_actionとRZ-workのrankは参照Nを使うoracle indexと表示する。

入力取得費用はfeatureごとにprovenance、必要なclassical処理、CPU/wall/RSS/disk、compile/source-oracle使用の有無を申告する。今回は保存field読取・source算術であり、新サイズのλ_R/DF/basis取得費用はUNKNOWN。未知を0と記録しない。quantum G_RZとclassical分析時間/メモリを合算しない。

## 4. Shot、normalization、eligibility

既存PM-2再現はbinary64、u=0、axis failure α=.025を維持する。保存absolute axis bias b_a、Bから `h_a=ε/√2−b_a`、両axisでh>0のとき `N_a=ceil(2 B² log(2/.025)/h_a²)`。境界は `ε_min=√2 max(b_real,b_imag)`、等号は不適格。通常の0 shotにしない。新しい数値uの導入で旧ledger/statusを修正しない。

主anchorはε={.05,.01,.005,.001}。既存PM-2 grid（.005〜.1）との保存照合を分けて出し、.001は同じ保存bias/Cに対する新しい**条件付き会計点**、新高精度signal検証ではない。.0001はAX-1bに追加しない。許可なしでq/r/K候補を追加しない。

finite b/Bはsource-accounting再現と保存一致を評価できる。原論文bound `b≤exp(τ²)` はassumptionとbound/ref ratio・log slackを報告し、heuristic prediction errorへ混ぜない。bound違反なら丸め/版/入力差を確認してSTOP、理論反証と即断しない。

operational bias predictor未登録のため、axis-bias error、N_pred error、false acceptance/rejection、operational classificationは全て `N_A_NO_OPERATIONAL_BIAS_PREDICTOR`。参照biasを代入したNをpredictedと呼ばない。将来のpredictorはprediction情報のみでb_pred/B_predを凍結し、その後N_refとのsigned/absolute/log errorとacceptance confusion matrixを測る別契約が必要。

future confusion statesはTRUE_ACCEPT/TRUE_REJECT/FALSE_ACCEPT/FALSE_REJECT/UNDETERMINED/N_A。判定不能をfalse rejectへ合算しない。現conditional caseはoracle eligibilityを共有するのでfalse acceptanceはNOT_APPLICABLE_ORACLE_ELIGIBILITY（0と表示しない）。marginが非finite、ceil overflow、未解決uはNUMERICAL_UNDEFINED/UNDETERMINEDとして保持する。|ε−ε_min|≤1e−12はBOUNDARY_SENSITIVE flag、旧strict判定をtoleranceで変更しない。

## 5. Selectionとregret

primary selectionは固定direct集合X_direct（development218、M2別5）に限定する。ε、scope、metric、foldを揃える。cost conditional caseは `eligible_ref` から `G_conditional_oracle` 最小を選ぶ。operational caseは将来predictorが用意された時だけ `eligible_pred` から `G_operational_pred` 最小を選ぶ。

`regret = G_ref(x_selected)/min_{x∈X_direct,eligible_ref}G_ref(x)−1`。

全available cost modelの共通complete prediction subsetでのregretは `regret_conditional_oracle_common_support` とし、集合hashとfull218からの除外を表示する。full218についてcoverageが不完全なら `INCOMPLETE_PREDICTION_COVERAGE`、full-set optimumを予測できたという主張は不可。STRUCT_ACCOUNT N/Aをcommon subsetへ強制参加させない。M2五構成のregretはその五件内だけ。

cost OOF primaryは210の全4q-foldで得るpredictionを合わせ、PM-1の8を追加せず `regret_conditional_oracle_internal_group_210` を別に出す。full210 fitを218へ適用するregretは `in_sample_plus_observed_prefix`。H4 calibration後のin-sample/group/既観測M2を独立held-out性能と表現しない。

tieは保存されたcanonical candidate identity（sorted-key compact JSON）文字列の辞書順。同一stringならexecution fingerprint順。最小Gとの差 `|G−G_min|≤1e−9` かつ `|G−G_min|≤1e−10 max(|G|,|G_min|)` を数値tieとして固定する。toleranceを結果後変更しない。

| selection status | regret処理 |
|---|---|
| REF_ELIGIBLE_EMPTY | N/A。registered searchで到達なし、方式全体の不可能性ではない |
| NO_PREDICTIONS / MISSING_COST_PREDICTIONS | N/Aとcoverage。missingを∞/0で黙って置換しない |
| MODEL_ALL_REJECTED | N/A、abstention/missed opportunity。eligible_refがある場合を分ける |
| SELECTED_OUTSIDE_DIRECT_SET | N/A、supplementary candidateとして別契約なしに評価しない |
| SELECTED_REFERENCE_INELIGIBLE | false acceptanceを先に記録、通常の有限regretはN/A |
| REFERENCE_COST_UNDEFINED / ZERO_REGRET_DENOMINATOR | N/A、absolute excess costは定義可能時だけ別記 |
| SELECTED_ELIGIBILITY_UNDETERMINED | N/A、eligibleへ救済しない |
| VALID_CONDITIONAL_ORACLE / VALID_OPERATIONAL | 指定集合内のregret、情報区分・coverage・uncertainty付き |

## 6. Trajectory費用のpaired統計

保存retained rowsからtrajectory index/seed/step_seedsとsemantic/evolution fingerprintsを確認し、cosine/sineのpairを保持する。cost scalarごとにcanonical IID unweighted mean、n−1 sample variance/covarianceを使用する。保存event probabilityをsampling importance weightとして掛けない。n32はcost sample数でありquantum Nではない。

固定参照Nについて `SE(G_ref)²=(N_real² s_cos²+N_imag² s_sin²+2N_real N_imag s_cos,sin)/n`。deterministic n=1はcompile値が固定でcost-sampling SE=0、一般のdevice確実性ではない。pair identity欠測/不一致時はSE/covariance N/Aまたはidentity STOPとし、無断でcovariance=0にしない。

±2SEは **ENGINEERING_INTERVAL_ONLY**。formal CI、simultaneous family CI、family winner保証ではない。model fitとreference Cが同じH4標本に依存し、cross-candidate covarianceも不明なので、error/regretへ独立SEを機械的に伝播してformal区間を作らない。負engineering下端はそのまま表示してphysical costと区別する。candidate間orderingを統計的勝者へ昇格させない。

32 samplesで未観測rare order/eventの費用寄与は保証できない。finite probabilitiesの存在とfull event列の未保存を区別する。pre-transpile boundが欠測なのでrare-cost tailはUNRESOLVED。新sampling/bootstrap draw/seed replay、order-stratified追加、compileをAX-1bに含めない。

## 7. 出力schema、失敗、凍結

将来の予測/metric rowsは少なくとも `schema_version, dataset_id, candidate_fingerprint, model_id, model_version, fold_id, evaluation_case_id, prediction_information_class, oracle_flags, unit, scope_identity, training_membership_sha256, C_pred_by_axis, B_pred, axis_bias_pred, N_pred_by_axis, G_operational_pred, G_conditional_oracle, availability_status, missing_reason, extrapolation_flags` を持つ。参照/診断fieldsは別objectとする。operational欄をconditional値で埋めない。nullに理由/statusを必ず付ける。

metric rowsは数値とvalid denominator、support/membership hash、information class、in-sample/internal/observed-geometry区分、uncertainty kindを持つ。action index、bound slack、RZ error、shot誤差、conditional regret、operational regretを別metric namespaceに置く。

分析planはdraftであり、実装・環境・予算・launchが未固定のため実行できない。実行前identity checkはallowlist全件をhash/schema/key照合、出力dir新規、source commit/hash固定を要求する。失敗時は新namespaceのfailure auditを残してSTOP。自動retry/resume、旧science cacheの再利用は禁止。再開はreview済み同一仕様と明示launch、新attempt namespace、完了入力/出力hash照合が必要。

AX-1bのterminalは `AX1B_COMPLETE_WITH_DECLARED_NA` または `AX1B_STOP_{AUTHORIZATION,INPUT_IDENTITY,SCHEMA,CONTRACT_CONFLICT,ENVIRONMENT,BUDGET,IMPLEMENTATION,NUMERICAL_FIT,PAIR_IDENTITY,OUTPUT_COLLISION}`。正常完了でもmandatory_stopとnext_stage_authorized=falseを維持する。false acceptanceやモデル改善なしは科学的negative診断であり、都合のよいretryの理由ではない。
