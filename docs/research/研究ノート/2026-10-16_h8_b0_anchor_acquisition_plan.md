# 2026-10-16 H8 B0新anchor取得の計画

直前B1のSTOPを受け、利用者は旧B0を新条件で最小限取得する計画を依頼した。今回はcompileの承認ではない。

S2 TRAIN p14/q1・p21/q1・p14/q8、HOLDOUT p21/q8、各cosine/sine、計8軸を提案した。3点だけでは飽和fitになるため4構成を最小とした。設計座標の行列式−49で非共線性を確認したが、実RZのfit/残差は0。HOLDOUTは固定して後続B1でもfitに入れず、各軸2%以上でSTOPする。

保存export操作列からbasisを復元し、旧B0の来歴を補完したと見なすことなく新cost用来歴を直接保存する計画である。最大4 compile worker/thread1、2 wave、AS/RSSと累積CPU/wall上限を固定した。既存B0I16軸は再利用、H6係数移送・旧B0の無条件利用・B1/B2/C自動開始はしない。

265 provider filesと7入力のhashを照合し、metadata/人工receiptの10 unittest methods PASS。実adapter/launcherと新runtime SOURCE sealは未完了で、grant案は無効。新compile・科学計算・数値再解析0。旧673保護/113worktreeを前後照合し、commit/push・remote SHA確認後mandatory STOP。

[現在の計画と未完了事項](../track_a_h8_b0_anchor_acquisition_plan_20261016.md)。
