# 2026-10-06 Track B: narrowed R1 source preparation

利用者/GPT reviewがR0.5のmethod-delta候補を受け、
`PROCEED_TO_R1_PREPARATION_WITH_NARROWED_CLAIM`としてR1の契約・実装準備だけを認可した。
R0.5 commit `61dd534567fda5c7348fdc688814089eb26a3561`を基点とする独立branchで作業。
旧R1 v1 proposalの履歴は変更しない。

[v2 preregistration](../../tracks/algorithm_codesign/rte_reallocation_r1_preregistration_v2.md)へ
ordinary/PTSC-K0/A、Pauli contextのみcollected CTS、distinct-basis controlled primaryを固定した。
canonicalのみ、resource vector primary、共通finite-confidence forecast secondary。
group rounding後のIID law、odd phase/complement、basis cost/errorを保持するB専用sourceを追加。

CTS controlの角度により旧78 key案を結果前に126へ改訂。同じ3 context、2 x、2 sign、3 precision。
264予定resource rows。合成器/version identityは維持し、phaseまで近似するため新contractはup_to_phase=false。
strict guardとjoint-unitary測定biasのfactor2を使う。旧channel projective guardを流用しない。

27 focused tests PASS、登録domain外のsynthetic operator照合とlaunch拒否のみ。
登録cost表・synthesis・science・trajectory・分子/DF/NPZ/GPUは0。
distinct-basis toyも原理的にはPauli展開でき、アクセス境界の一般的不可能性は主張しない。
単一scalar GOを設けず、vector/trade-offの研究解釈をGPTへ戻す。

source Sをcommit/pushして[GPT source review](../../tracks/algorithm_codesign/rte_reallocation_r1_source_review_request_20261006.md)へmandatory STOP。
R1実行はpending。source review後のauthorization-only direct childと利用者の別明示実行指示が必要。
