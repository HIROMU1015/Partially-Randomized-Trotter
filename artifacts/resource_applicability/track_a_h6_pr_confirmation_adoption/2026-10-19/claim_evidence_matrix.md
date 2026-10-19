# Claim–Evidence Matrix：H6固定独立確認の正式採用を反映した更新版

[CSV](claim_evidence_matrix.csv)／[JSON](claim_evidence_matrix.json)。旧CE1–CE6の全cellを固定commit9036d8a8…から保持し、H6の2主張だけを正式採用として追加した。旧主張のstatusを遡及的に一括変更していない。

| ID | 主張 | 採用の範囲 |
|---|---|---|
|CE1|決定論側の比較対象を強化するとPRの点費用利益が縮小する|歴史的な証拠対応：元の限定を保持|
|CE2|PR側の探索domain拡張で、特にH8両端の点費用の結論が変わる|歴史的な証拠対応：元の限定を保持|
|CE3|中間精度でのPR点費用削減は強化した比較集合でも残る|歴史的な証拠対応：元の限定を保持|
|CE4|回路費用とshot負担のトレードオフが勝敗を左右する|歴史的な証拠対応：元の限定を保持|
|CE5|signed／経験的非相殺の会計区分が結果に影響する|歴史的な証拠対応：元の限定を保持|
|CE6|結果は有限時間signal・保存DF・P=0・Qiskit compiled RZ費用に限られる|歴史的な証拠対応：元の限定を保持|
|CE_H6P_Q2|H6の事前固定したPR不利方向を独立標本の点費用で再現した|今回正式採用：固定H6独立Pの点方向再現|
|CE_H6P_Q8|H6の事前固定したPR有利方向を独立標本の点費用で再現した|今回正式採用：固定H6独立Pの点方向再現|

旧6行のaudit_linksは固定元のE_claim_evidenceディレクトリを基準に解決する。今回の2行のaudit_linksは本bundle基準。新しい表の場所を根拠に旧リンクやSOURCEを読み替えない。

## CE_H6P_Q2：H6の事前固定したPR不利方向を独立標本の点費用で再現した

- claim_candidate：H6の事前固定したPR不利方向を独立標本の点費用で再現した
- fixed_commits：3cccfcb2051dacf29e91473e2e1152acef1574bd;f79366f68caf8585cc36d2f2b0fefcb1a4bd0d1f
- result_files：artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json;artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/native_completion.json;artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/ledger_audit.json;artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/STOP.json
- SOURCE：bf0c26dae4477de5fe7264a7c91e077661ef9b38
- conditions：linear H6/1Å/STO-3G/tol-only1e-8 DF19/T=.8/P=0/signed bias/factor1/Qiskit1.3.0 rz,sx,x,cx opt1 seed17/full symmetric-directional wrapper; B2_p4_q2_2nd_r4_K2 vs B0I_p7_q1_2nd
- existing_value_quote：ε=.05/n48：PRが78.055%高費用、保存PR削減率−78.055463%、paired SE=4054009.1059934883。 REPRODUCED; PR_CHEAPER=false.
- measured_or_model：両固定候補のcompiled RZ単価はDIRECT_MEASURED。Gは保存解析shot×両軸RZ。PRはpaired trajectory有限平均、PFは保存直接実測対照。
- search_or_confirmation：H6_INDEPENDENT_UNCONDITIONAL_CONFIRMATION_V1; scope=H6_CONFIRM_P4_Q2; n=48; 探索n8・旧union n32・H4/H8と非混合。候補間のpooling/pairingも行わない。
- unresolved：H6P_POPULATION;H6P_RARE_SUPPORT;H6P_EMPIRICAL_BIAS。CI/p値は保存null。厳密母平均順位・全support分散精度・全候補最適性・定量機構説明を認定しない。
- evidence_strength：FORMALLY_ADOPTED_FIXED_SCOPE; REPRODUCED。ユーザーが固定2条件の独立Pを完了扱いで正式採用。COMMON_CORE_READYや最終freezeの一括認定ではない。
- audit_links：completion_status.json;evidence_catalog.json;STOP.json

## CE_H6P_Q8：H6の事前固定したPR有利方向を独立標本の点費用で再現した

- claim_candidate：H6の事前固定したPR有利方向を独立標本の点費用で再現した
- fixed_commits：3cccfcb2051dacf29e91473e2e1152acef1574bd;f79366f68caf8585cc36d2f2b0fefcb1a4bd0d1f
- result_files：artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json;artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/native_completion.json;artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/ledger_audit.json;artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/STOP.json
- SOURCE：bf0c26dae4477de5fe7264a7c91e077661ef9b38
- conditions：linear H6/1Å/STO-3G/tol-only1e-8 DF19/T=.8/P=0/signed bias/factor1/Qiskit1.3.0 rz,sx,x,cx opt1 seed17/full symmetric-directional wrapper; B2_p4_q8_2nd_r4_K2 vs B0I_p11_q4_4th
- existing_value_quote：ε=.0005/n104：PRが9.644%低費用、保存PR削減率9.644416%、paired SE=65539320043.303085。 REPRODUCED; PR_CHEAPER=true.
- measured_or_model：両固定候補のcompiled RZ単価はDIRECT_MEASURED。Gは保存解析shot×両軸RZ。PRはpaired trajectory有限平均、PFは保存直接実測対照。
- search_or_confirmation：H6_INDEPENDENT_UNCONDITIONAL_CONFIRMATION_V1; scope=H6_CONFIRM_P4_Q8; n=104; 探索n8・旧union n32・H4/H8と非混合。候補間のpooling/pairingも行わない。
- unresolved：H6P_POPULATION;H6P_RARE_SUPPORT;H6P_EMPIRICAL_BIAS。CI/p値は保存null。厳密母平均順位・全support分散精度・全候補最適性・定量機構説明を認定しない。
- evidence_strength：FORMALLY_ADOPTED_FIXED_SCOPE; REPRODUCED。ユーザーが固定2条件の独立Pを完了扱いで正式採用。COMMON_CORE_READYや最終freezeの一括認定ではない。
- audit_links：completion_status.json;evidence_catalog.json;STOP.json

母平均／rare support／厳密biasの保証、新cohortの機構説明、H4/H8との最終統合や最終evidence freezeは認定していない。[完了status](completion_status.json)と[採用判断](adoption_record.json)が今回の範囲を定める。

更新版CSVの旧6行は改行だけCRLF→LFへ統一した。全cellは元の固定CSVと一致し、元CSVのbytes/SHAは変更していない。
