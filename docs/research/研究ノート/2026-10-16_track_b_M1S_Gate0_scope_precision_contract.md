# 2026-10-16 M1-S Gate 0：限定scope採択・精度契約の具体化

ユーザーがH4 linear1.00 Å/STO-3G/8 spin orbitals/保存DF rank12、生成順G3/G6/G9、JW符号集約の非冗長Pauli law、identity→HD、保存数値oracle、W/v/rho、同task・同概念event比較、全分割報告・rho自動GO/STOPなしを正式採択した。採択主体はユーザーで、実験契約全体・数値精度・実行制限・科学実行は未採択/未承認。

17項目を定義/norm/単位/誤差源/伝播/候補根拠/不適格/technical STOP/科学判断/未採択理由まで整理した。構造に基づく候補5件、分解能提案3件、絶対値未決9件。誤差予算はengineering会計で、exact固有stateや定理domainを認証しない。小分母・raw点・区間・精度不足を区別した。

CPU1 worker/各thread1、Python3.11.0rc1/NumPy1.26.4、4 GiB address space、result/log各256 MiB、context4 MiB、Pauli65536、全寄与1,000,000を候補化。静的loop包絡は706,837。benchmarkではない。関連NumPy/BLAS file6件のmetadata/hashだけを確認した。

旧v1 SOURCEを変更せず、copyを基にv2 namespace/runnerを追加した。scope-record binding、exact-zero gate、build hash、raw区間/精度分類、context/native log captureを修正。SOURCE固定は94e6089eba37b022a79eb262a2aa5025f5e8ddcd。追加静的検査67件PASS、旧30/79件は保存記録を参照し全面再実行していない。数値・runtime correctnessは未検証。

[精度設計](../../tracks/algorithm_codesign/m1s_gate0_precision_error_budget_design_20261016.md)、[runtime候補](../../tracks/algorithm_codesign/m1s_gate0_runtime_and_resource_contract_candidate_20261016.md)、[最終契約v2候補](../../tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v2_candidate_20261016.md)、[証拠索引](../../../artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/evidence_index.json)を公開する。旧候補・レビュー・S0/T1a/T1b・STOP・結果・authorization/markerを遡及変更しない。

新規科学計算、保存state/Pauli/H/W/v/rho評価、実/模擬数値correctness、simulation/回路/compile、benchmark、新定理、新science authority/markerは全て0。IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED、限定scopeのみ採択。精度9項目のmaterialization/coverageと全17値の最終受理、runtime/caps、全契約の正式採択をユーザーへ戻す。その後さらに別科学実施承認が必要。公開照合後mandatory STOP。
