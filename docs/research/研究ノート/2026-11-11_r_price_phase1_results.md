# 2026-11-11：R依存compiled価格 Phase 1実行

[結果REPORT](../r_price_phase1_results_20261111/REPORT.md)／[HANDOFF](../r_price_phase1_results_20261111/HANDOFF.md)。ユーザーの「実行してよい」を受け、準備公開c596e88dの固定PLAN/SOURCE1a87e5331を変更せず実行した。

Linear H4/H6・1Å・STO-3G・保存tol-only DF10/19・元completion/state、H4 S2/δ.1/p1/K2、H6 m5_best/δ1.6/p7/K2、元compiler/full wrapper/P0を保持。保存C/S59 pathと新校正16 pathでsystem別3係数をfitし、rank3を確認した。独立検証取得前に7eeabf0ae482ee610d667e28470692d209f2bb21へ係数・校正値をcommit/pushしてremote確認した。

独立H4 q16/R8・H6 q2/R32は各n8、paired XY。保存相対RMSEは順に1.039841%、0.229959%。検証を見た再fitなし。通常64 compile、別legal forced order0/order2診断16 compileで上限80を完了。新通常32 pathのeventはorder0のみ、診断ΔCはH4 268/192、H6 141/1096（両軸）だった。診断を通常平均・fitへpoolしない。

採用閾値なし・3セル3係数のlack-of-fit DOF0・有限cohortであるため、model資格false、all-law/longq-R/systematic/joint/totalCI/fullGは未資格のまま保持。旧B binding.Tはtask参照.8、実回路時刻candidate.Tというmetadata制約をHANDOFFに残した。NやG、信号、状態再生成、新statevector作用検証・詳細科学レビューは実施していない。

固定科学SOURCE/原入力/校正freeze/private-public価格・hash・fingerprint/旧provider等の照合PASS。作者の技術確認であり、独立科学レビューではない。過去のSOURCE/raw/grant/ledger/marker/KEEP/STOPは保全し、A/B作業領域へ非介入。新private rawはC専用repo外run01で保持。

公開・remote照合後mandatory STOP。次はユーザーが許容価格精度と追加検証scopeを判断し、Bが許可範囲内で保存recordを接続する。Cは追加q/R/compile/fit/N/Gを自動開始しない。
