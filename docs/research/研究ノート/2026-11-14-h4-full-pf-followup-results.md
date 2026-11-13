# 2026-11-14：承認済みFull PF追加5点の実行

ユーザーが計画d48d8732の新5cellとM5窓・保存M5/.8スペクトル検査を承認したため、科学SOURCE `a3005d72349e9a91c3229215b189e6cf8bed0000` で一回実行した。[REPORT](../h4_full_pf_followup_results_20261114/REPORT.md)。保存Linear H4/1Å/STO-3G/tol-only DF10/sector(2,2)/Full10、同H/state/scalar/target、h.1/M12/P0・元誤差予算を保持。

5新step/Schur、直接/spectral各10 mean、保存M5/.8窓整合検査1が全数値gate通過。局所予算内はS2/16h/7（+6.99614984259e−4 Ha）、Y4/16h/3（−5.90726632746e−4）、M5/16h（−1.04921622697e−5）。S2/8h/3とY4/32h/5は予算超過を有効な結果として保存。fitや追加探索は行わない。

M5/1.6のENERGY_WINDOW [-2.5,1.0]で1モードに整数枝−1が付いた。元最低groupとの最大trace overlapは全点でgroup0、局所の対応観測でground/alias認定ではない。最大mean差2.811e−12、最大power unitarity1.271e−11、元gate1e−9内。元H新eigh/reference/旧signal再解析/N/MC/C/G/retry0、M5/3.2未実行。入力3hash一致、人工新5＋既存8＝13 PASS。

C/Bへ3局所候補・65round metadataと固有系SOURCEを渡すが、十分N/global RMSE/G/winnerはnull。公開・remote照合後mandatory STOP。次はユーザーが価格/risk接続の有限範囲を決定し、承認後に各担当が実行する。
