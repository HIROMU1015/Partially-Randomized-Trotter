# H4 resource observer source再固定

`H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`。科学未実行、公開後STOP。
起点 `5245a29ca26cad7421640410934907647459b822` はfetch後remote SHAと一致。
旧認可v1 bundleを保持し、新source固定後に別commitで草案v2を再bindingする。

- [observer修正・root判定・不変条件](../../../../docs/research/track_a_h4_geometry_resource_observer_fix.md)。
- [修正前identity/remote確認](preflight_identity_v1.json)、[source段manifest](source_stage_manifest_v1.json)。
- [observer33検査](guard-observer-tests-attempt-01.json)、[全ログ](observer-tests-attempt-01.log)。
- [実環境read-only観測](live_observation_v1.json)、[観測guard](guard-observe-attempt-01.json)、[ログ](live-observation-attempt-01.log)。
- [固定前source/環境/旧証拠監査](pre_freeze_audit_v1.json)。
- [SOURCE固定後の外部blob/hash監査](source_freeze_v1.json)。actual SOURCE_COMMITはこの別資料へ記録する。
- [更新した入力生成認可草案v2](../../track_a_h4_geometry_input_generation_authorization/2026-10-06-v2/README.md)。

rootは初期cgroup namespace、全root mount、shadow欠測なし、所属/可視性の観測前後一致で判定する。
真のv2 rootだけ非root memory interfaceを要求せず、全非root制限/events/pressureとhost PSIを維持する。
非rootの欠測・読み取り不可・不正値・隠れた上位制限はSTOP。host-only fallbackなし。

実observerは全3非root祖先を読み取り成功、観測effective available約981.826GiB、PSI/OOM0。
この値は準備時点のmetadataで、live admission/CPU許可ではない。
CPU未確定allowed_cpus=[]、review approved=false、有効実行認可0、最終review/明示launch未実施。
source15件＋親2件、旧bundle・科学証拠は不変。追加transpile0、旧系列28/64を維持する。
