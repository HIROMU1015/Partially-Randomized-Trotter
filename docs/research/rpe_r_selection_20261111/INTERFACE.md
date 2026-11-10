# A/C向け：有限R menuのN(T,R)・C(T,R) interface

実装は[`rpe_r_selection.py`](../../../src/trottertracks/rpe_r_selection.py)。旧Bのdict形式、canonical candidate ID、task/identity、signed schedule/allocation、測定符号・単位を再利用する。実N/C producerは呼ばない。

```python
from trottertracks import rpe_r_selection as r

# fixed_taskは既存のconfiguration/binding/contract/profileとPF/PR候補。
# この例は接続仕様。実データでの新計算・評価の実行許可ではない。
menu = r.build_menu(fixed_task, {3: [4, 8, 16], 12: [4, 8, 16]},
    sampling_law={"model": "ideal_finite_RTE_law",
                  "sampling": "upstream_explicit_sampling_definition",
                  "source": {"commit": "full_source_sha", "path": "law_definition"}})
request = r.request_context(menu, menu["candidates"][0])
inventory = r.coverage(menu, supplied_N, supplied_C,
    cohort_selection={(candidate_id, "X"): scenario, (candidate_id, "Y"): scenario})
# 別途許可された数値範囲でだけ使用。今回の実行は人工/mockのみ。
result = r.evaluate_menu(menu, supplied_N, supplied_C, cohort_selection=selection)
```

`build_menu`はPRのR_outer、canonical ID、R依存allocation/event_countだけを更新する。T/qPF/δ/prefix/K/formula、qRPE、h/M/prior/decoder/α/β、H/state/snapshot、compiler/完成wrapper/P=0は保存する。PFは複製・変更しない。Rはroundごとの有限・重複なし正整数列。H4の変更対象δは.1、H6は1.6。H6 m0..3の別δ/p5/R32等は元taskに残し、R変更対象へ含めない。H6 m5_bestの11 signed pieceへのR配分は元`h46_m5_plan`を使用し、不可能な小Rを受け付けない。

## A：N record

B既存形式を継続する。

- `identity = request['identity']`：m/qRPE/qPF/full candidate/structure/binding/contract/profile/configuration。candidateにはsystem、method、formula、T、q、δ、prefix、R_outer、K、IDを含む。
- `law_binding = request['law_binding']`：法則の定義SOURCE/sampling、candidate ID/R/K/qPF。証拠が一致したときだけ返す。既存N recordへ機械的に付与して新Rの科学的資格を作らない。
- `plan`：元の`CONDITIONAL_ROUND_PASS`等のstatus、上記candidate/binding、m/qRPE/qPF/base_time/target_id、αX/Y、**正のPython整数N_X/N_Y**。JSON integerも桁を維持して読む。浮動小数点Nやboolは不適格。
- `actual`、`diagnostics`、`source`：旧仕様を保持。精度・phase/radius/allowance・整数検証とその条件はA担当。roundの条件付きPASSは全RPE成功ではない。

## C：価格record

旧B形式の`identity,axis,classification,cohort,n_paths,mean,source,expectation,model,uncertainty,normalization_is_price_multiplier`と測定metadataに、同じ`law_binding`を加える。`normalization_is_price_multiplier=False`、X=cosine/Re、Y=sine/Im、P0−P1、RZ gate_count。N側attenuationや古典価格標本数n_pathsをN×Cへ再乗算しない。

`mean.rz_count`は供給済み非負整数または有限binary64。欠測はnull。direct/modelを代用せず、`UNVALIDATED_EXTRAPOLATION`を除外する。X/Yは同じcohort/scenarioを明示選択する。path単体は期待価格として使わない。

modelには旧Bのqualified_for_candidate/q_variable=q_PF/q/coverage/source/validation_sourceに加え、`domain`として**その価格が資格したfull candidate object**、同一`binding`、`compiler`を要求する。R4/R16校正のdomainを別Rへ読み替えない。旧providerにdomainやlaw_bindingがなければ不足のまま返す。本実装はA/Cの固定modelやproducerを変更・自動補完しない。

`law_context`ではN_lawと実際のprice_lawを保持する。既定期待価格policyでは両者一致と`ideal_sampler_distance_qualified=True`を要求する。PR有限cohortとideal Nの違いは、明示`price_policy='finite_cohort_scenario'`に限って条件付きproxyとして扱う。同じtarget law bindingの宣言は、有限samplerとideal lawの距離の新証明ではない。`FINITE_COHORT_ONLY`は全law期待値ではなく、model domainが未資格なら有限policyでも使用しない。

不確かさは保存済み誤差証拠・shared calibration group・paired XY・SOURCE等をそのまま渡す。未知のR依存残差、係数/systematic、全law、cross-R covariance、全体CIを0にしない。

## B：metadataと数値出力

`coverage`は元`p.plan`を使うread-only metadata検査だけで、N/C計算・積和・選択を行わない。各候補の不足N、精度/radius不成立、価格未資格、law/domain不一致・不足、paired axis不一致を区別する。identityや明示law_bindingの矛盾は例外を返す。

`evaluate_menu`は供給済みN/Cの積和と選択を行う明示数値API。eligible候補ごとに既存`aggregate_supplied_axes(..., exact=True)`を呼ぶ。価格は供給されたbinary64の正確な有理数として解釈し、Nは一度もfloatへ変換しない。出力cost/totals/差/区間は`{numerator: integer, denominator: positive_integer}`。これは算術の精度であり、価格modelの科学的精度を追加保証しない。旧会計の既定numeric pathは互換のまま、新R入口は常にexact pathを使う。

各round・方式で点費用を最小にする候補を選ぶ。α/β/h/M等は固定なので供給点目的関数は加法的。tieは全IDを残し、最小R/IDで表示用代表を決める。PF/PR比較は旧比較器の`POINT_ESTIMATES_ONLY`を保持し、費用差もexactで返す。

不成立が確認されたN候補は除外する。欠測/未資格C/未確認law等があると`menu_complete=False`、`selection_scope=AVAILABLE_QUALIFIED_SUBSET`になり、全menu最適とはしない。選べる候補がないroundは`INCOMPLETE_R_MENU`で全Gをnullにする。missingを0として勝者を作らない。producer呼出し・再試行・R gridの自動拡張はない。

## 成功条件と不確かさ

全round NのPASSとは別に、選ばれたscheduleに対するupstream成功証拠を受け付ける。`schedule_success.context`は返却された`schedule_success_context`（configuration/binding/profile/contract/law/選択ID順）と一致させる。sourceと`prior_alias,horizon_decoder,all_round_phase_radius,failure_budget,implementation_and_law`の各資格を要求する。これは上流の科学的評価の受渡しであり、Bがboolから証明を生成する仕組みではない。未供給や不完全scheduleは`rpe_success_qualified=False`。priceの全RPE採用・Rの科学的採択は別で、`full_RPE_qualified/scientific_R_adopted=False`を維持する。

任意の`simultaneous_bounds`は全menu/全軸のlabel順・source・全stat/model/law/menuを覆う明示資格と価格区間を供給する。BはN×区間をexactで伝播し、勝者の上端が全代替候補の下端を下回る場合だけ`strict_menu_dominance=True`を返す。欠測候補があればfalse。区間は新推定せず、観測max/SE/αtotalから作らない。

共分散の空欄・共有校正・cross-R依存は旧joint uncertaintyの未資格表現とともに保持する。本段階はcross-R joint matrixの新推定・その数値伝播や選択後CIを実装しておらず、`cross_candidate_covariance/total_confidence_interval=None`。未知の独立性を仮定してR順位の統計的確定をしない。
