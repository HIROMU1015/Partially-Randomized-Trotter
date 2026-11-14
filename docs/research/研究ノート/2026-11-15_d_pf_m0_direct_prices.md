# 決定論PF m0の直接価格取得（2026-11-15、Codex D）

利用者の明示承認を受け、固定C alignmentのFull10 / B0I p7 × S2 / 標準Yoshida4 / M5 × m0 / X/Yの12軸を取得した。保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、T=.1、q1、actual幅.1、P0除外、scalar total T一回、元native・compiler・測定付きcontrolled wrapperを保持。旧ID・別scheduleをphysical metadataで照合し、受入可能な保存価格0、重複保留0、実compile12。人工4 testの後に実行SOURCE729fe5370を固定した。

Full X/Y RZはS2 8090、Y4 23656、M5 85568、p7は5820 / 16846 / 60950。各軸別compile、元7作用gateとC guard全PASS、cache-hit0、scientific retry0。取得wall100.876 s、6 cell並列、最大worker RSS575.16 MiB。raw a7d066cae、bound471f0c3fa。[REPORT](../d_pf_m0_direct_prices_20261115/REPORT.md)、[C HANDOFF](../d_pf_m0_direct_prices_20261115/HANDOFF.md)、12軸JSONに固定locatorを保存した。

較正時とactual幅が違うため、m0の直接価格の有効性までが結論。m9の2%判定・予測再計算・refit・他roundへの移送・MSE/N/G/winnerは行わない。m6/m9・Stage1/2・元凍結JSON・raw・STOPは不変。ロードマップへの差分は「m0直接coverage＋12、モデル追加判定なし」で統合担当へ渡す。commit/push・remote SHA/tree/主要成果物の照合後mandatory STOP。
