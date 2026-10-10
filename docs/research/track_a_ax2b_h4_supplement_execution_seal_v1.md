# Track A AX-2B：H4補完の実行前seal v1

2026-10-10 JST。ユーザーの「作業を進めて」を、直前に説明したH4限定の2単位一回実行・保存監査・STOPへの指示として記録する。
GPTレビューや準備manifest自体を実行認可として扱わない。
CPU 1 / science worker 1 / BLAS thread 1をCodexが割り当てた。ホスト全体でのCPU専有を保証する記録ではない。

[準備資料](track_a_ax2b_h4_supplement_preparation_v1.md)の科学的条件・source・予算・保存先を維持する。
source commitは`67aa6bb54dd5385eb3c56def1b6052da12643447`、実行前baseは`55f1af9d9e128947d05cd8e7ce5087abd0a826e9`。
旧6 correctness・12 MPの結果、過去のSTOP、source freeze、未コミット差分は変更しない。

| 単位 | 新seal | 別grant | 固定範囲 |
|---|---|---|---|
| EVENT_CONTROL | [manifest](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/sealed_event_control_v1.json) | [grant](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/authorization_event_control_v1.json) | B2 K2 / B3 K6 × order 0/2、4群 |
| S4_MP | [manifest](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/sealed_s4_mp_v1.json) | [grant](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/authorization_s4_mp_v1.json) | B1 S4 q1/q4、correctness 2件・MP80/120 4件 |

[資源確認](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/resource_assignment_v1.json)と
[metadata preflight](../../artifacts/resource_applicability/track_a_ax2b_h4_supplement_execution_v1/2026-10-10/seal_preflight_v1.json)に、
CPU affinity、メモリ観測、source180件・環境・入力・全coverage・grant/output結合の確認を保存した。
この時点では科学実行を開始していない。cgroup上限・ホスト全jobの確認は未成立と明記する。

両manifestは`execution_plan_sealed=true`、CPU割当済みである。
manifest内部の`science_authorized=false` / `launch_allowed=false` / `H4_SUPPLEMENT_NOT_AUTHORIZED`は、
別grantを必須とするrunnerの契約上の値であり、grantの代用にはならない。
今回の別grantは各単位一回に限り、retry/resumeなし。順序はEVENT_CONTROL→S4_MP。
各単位で既存input/referenceと179 primitive/time組×3 probeを先に照合し、必要数値比較を保つ。

上限は準備時のまま、EVENT_CONTROL total900秒・validation300秒、S4 total2400秒・validation1800秒、
各AS8GiB・output128MiB・log64KiB。sampling・transpile・compileは0。
EVENT_CONTROLの限定代表回路build/state actionだけが今回のH4範囲に含まれる。
結果を見て上限・精度・stage/probe数を変更しない。失敗記録を残し、再実行しない。

数値不一致や独立性・対象変更が必要ならGPTへ早期に戻す。
保存監査後はmandatory STOP。`N/G=null`、`accuracy_eligibility=UNDETERMINED`、
`numerical_allowance_certified=false`、`H6_NOT_AUTHORIZED`、`DRAFT_NOT_AUTHORIZATION`を維持する。
H6入力生成・pilot・本検証、H8、科学的GO/STOPは今回の認可に含まれない。
