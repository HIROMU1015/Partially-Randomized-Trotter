# 2026-10-16 M1-S Gate 0 v4最終実験契約の正式採択

ユーザーはcommit44277e03のv4契約を、17項目・絶対残差品質・条件付きcoverage・runtime/caps・one-shot/failure STOPを含め正式採択した。
新しい正式契約JSONと採択原文を固定し、候補との差分をstatus/adopted/adoption_record/coverage_model_adoptedの4項目に限定した。
SOURCE・入力・精度数値/式・runtime・scope・STOPは同一。旧候補・SOURCE・レビュー・結果・採択記録・auth/grant/markerは保全。

[正式契約](../../tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v4_formally_adopted_20261016.md)、
[採択記録・監査](../../../artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1)を参照。
coverage受理は条件付きengineering policyの採択で、未検証のBLAS/IEEE/fsum/underflowを認証するものではない。
JSON schemaのexecution_readyは契約完備性のみ。科学実行はユーザーが明示的に承認していない。
新科学計算・Pauli生成・模擬/実入力数値test・formula評価・実行authorization/grant/markerはいずれも0。
次は別の明示Gate 0実施承認。採択だけでは初回数値correctnessも開始しない。mandatory STOP。
