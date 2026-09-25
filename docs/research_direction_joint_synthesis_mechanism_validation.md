# P-A非退化mechanism validation

最終更新：2026-09-25 JST

## 結論

事前登録したH4 forced-support、Taylor order 2検証を完了した。training 15、blind 15の全30 taskで
`interval_union_dp`は明示的な`one_segment_per_source_run` baselineと同じplanを選び、run内分割は
0件だった。全compiled metricも両policyで完全に一致した。

従って事前固定した7 gate中、order-2 coverage、最大個別RZ悪化、operator同値性の3 gateだけが通過し、
分割移送、plan変更、pooled RZ改善の4 gateは不通過だった。decision ruleを変更せず、statusを

`stop_pa_interval_dp_as_primary_and_return_to_pc`

とする。P-Aのrun-level full/support-union選択で得た既存RZ改善は保持するが、interval subdivisionを
研究上の独立寄与として主張しない。今後の主研究候補はP-C geometry energy differenceとする。

## 1. 事前登録と完全性

計算前の条件は
[事前登録文書](research/pa_joint_synthesis_mechanism_validation_preregistration.md)へ固定した。

- expected tasks：30/30
- training：fragment 3、5、7 × 5 profile = 15
- blind：fragment 4、6、8 × 5 profile = 15
- 全event：Taylor order 2
- 全task：一つのsource-basis run
- expected-task fingerprint：
  `e8b064e9821fae5c2e7a44d0c98d3f9a0ed973a4cd87945fb051151e446a96fc`
- threshold変更：なし

比較した2 policyは同じ4段lexicographic proxyを使い、candidateだけがrun内を複数segmentへ分けられる。

## 2. 結果

| partition | row | run内分割 | plan変更 | RZ改善row | pooled RZ変化 |
|---|---:|---:|---:|---:|---:|
| training diagnostic | 15 | 0 | 0 | 0 | 0.000% |
| blind holdout | 15 | 0 | 0 | 0 | 0.000% |

blindの集計値は次のとおりで、全指標がbaselineとcandidateで一致した。

| metric | baseline sum | candidate sum | 相対変化 |
|---|---:|---:|---:|
| RZ count | 4,378 | 4,378 | 0.000% |
| RZ depth | 1,088 | 1,088 | 0.000% |
| CX count | 1,476 | 1,476 | 0.000% |
| CX depth | 746 | 746 | 0.000% |
| total depth | 2,395 | 2,395 | 0.000% |
| circuit size | 8,505 | 8,505 | 0.000% |

これはcompilerが差を消した結果ではない。compile前のbasis plan自体が30/30で同一だった。
強制的にsupportを遠隔blockへ変化させても、v1のbasis-operation-count proxyでは一区間解が常に選ばれた。

## 3. gate

| gate | 結果 |
|---|---|
| blind全eventがTaylor order 2 | pass |
| run内分割が2 blind basis以上へ移送 | fail（0 basis） |
| run内分割が2 blind profile以上へ移送 | fail（0 profile） |
| blind rowの25%以上でplan変更 | fail（0/15） |
| blind pooled RZが2%以上改善 | fail（0.000%） |
| 最大個別RZ悪化5%以下 | pass（0.000%） |
| operator同値性・relative phase | pass |

5 operator probeの最大operator差は$8.327\times10^{-16}$で、baseline/candidateともrelative ancilla
phaseが一致した。

## 4. 解釈

既存blind transferのH5 -17.076%、H4 opt2 -6.598%は、source-basis runごとにfull basisと
support-union completionを選ぶrun-level policyの効果として維持する。一方、

- 自然sampled order 0 stream
- 強制support変化を持つorder 2 stream
- training basisと未使用blind basis

の全てでinterval DPは一run一segmentに退化した。従って現行v1のinterval partitioning layerを
独立した研究テーマとして広げる根拠は得られなかった。

この結果は「一般に区間分割が無効」という定理ではない。固定H4 snapshot、5 profile、現行completion、
現行basis-operation-count proxyにおける停止判断である。ただし、結果後にprofileやthresholdを調整して
同じP-Aを延長することは事前登録decision ruleに反するため行わない。

## 5. 次の研究方向

P-Aは主研究候補から外し、既存run-level policyを実装上の候補として保存した。この時点ではP-Cへ戻り、
geometry間のsigned PF errorについて次を事前登録した。

1. orbital／DF fragmentをgeometry間で連続追跡する規則
2. 追跡あり／各geometry独立分解の比較
3. 未使用geometry regionとgeometry pair
4. error cancellationの成立・破れを予測する停止条件

H12、長RPE、full-wrapper総costはP-Cの機構が成立してから必要性を判断する。

## 6. scope
この後続検証は[完了](research_direction_geometry_tracking_breakdown.md)し、stretch側予測と診断が
固定gateを通らなかったため、current H4 familyのP-Cも主研究候補から停止した。
現時点でA/B/Cに確認済みの主研究候補はない。


本検証は有限RTE分布で合法なforced order-2 eventを使うが、自然sampleでの発生頻度を推定していない。
新しい分子、geometry、Hamiltonian、state、backend/noise、routing、full wrapper、RPE総costまたはH12を
評価していない。literature novelty、global circuit optimalityまたは科学的優位性も主張しない。

## 7. 証拠

- expected task：
  `artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25/pa_forced_support_order2_expected_tasks_v1.json`
- final artifact：
  `artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25/pa_forced_support_order2_mechanism_validation_v1.json`
- content fingerprint：
  `fdc89974e89a4a6809cecd2c5608a36d684d40d76d9b3055fbbe6ec9276abbaf`
- file SHA-256：
  `26d2913c8f6812162a4e6f12cd54f75ea92d561e094f51674bf6615d217cc43b`
- implementation：
  `src/trotterlib/research_direction_joint_synthesis_mechanism_validation.py`
- runner：
  `scripts/run_research_direction_joint_synthesis_mechanism_validation.py`
- test：
  `tests/test_research_direction_joint_synthesis_mechanism_validation.py`
- 関連test：18 passed
- evidence status：local dirty worktree、外部再現／immutable CIなし
