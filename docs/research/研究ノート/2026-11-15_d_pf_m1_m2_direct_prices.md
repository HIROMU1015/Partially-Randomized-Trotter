# 決定論PF m1/m2直接価格（2026-11-15、Codex D）

利用者の明示承認により、Full10 / B0I p7 × S2 / 標準Yoshida4 / M5 × m1/m2 × X/Yの不足24軸を取得した。保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、元H/state/snapshot、P0除外、scalar total T一回、元native / compiler / 測定wrapperを保持。m1はT=.2/q1/幅.2、m2はS2 T=.4/q2/幅.2、Y4/M5 T=.4/q1/幅.4。旧ID・別scheduleを含むmetadata重複照合で受入候補・保留0。新3＋m0回帰4人工/mock test後に実行SOURCE a3cbe70e4を固定した。

新規実compile24、作用gateとC guard全24軸有効、cache-hit・科学retry0、round別各12軸が増加した。raw bf8a7f042、bound eea0bfab9。取得wall102.512 s、12 cell並列、最大worker RSS576.74 MiB。[REPORT](../d_pf_m1_m2_direct_prices_20261115/REPORT.md)、[C HANDOFF](../d_pf_m1_m2_direct_prices_20261115/HANDOFF.md)、24軸JSONへ固定locatorを保存した。

直接証拠coverageは72/156→96/156で、6候補のm0～m6と既存m9を含む。旧m3の現在consumer向けbindingは別課題のまま。m6/m9の2%規則、予測・モデル判定・refit、他round移送・追加取得、MSE/N/G/winnerは行わない。旧SOURCE / raw / 凍結モデル / STOPを保持し、ロードマップは本体を編集せずHANDOFFで統合担当へ差分を渡す。commit/push・remote確認後mandatory STOP。
