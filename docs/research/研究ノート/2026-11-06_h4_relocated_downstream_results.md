# 2026-11-06：H4再配置下流比較の登録結果と終了

## 2026-11-06：研究C・線形Z再配置後の下流は主比較でENTRY低費用

固定H4直鎖1.00 Å/STO-3G、number CDF8/sector36/元順LD2、保存C0/64更新のP(PR-aware γ=.1)、E1(ENTRY γ=.1)、E2(ENTRY γ=.03)へ同じ線形Z再配置を適用。partial S2/s1/δ=h=.1/K0/m0..12、own-DF ground oracle・状態準備除外、P0/fenced serial/shell込みの条件付き総RZを評価した。

共通r32のP対E1は92.464828249対86.352772151 billion、P/ENTRY=1.070780080。selection32で全候補r16を固定した公平retuningのP対E2は87.514417699対75.621395471 billion、比1.157270600。独立final各64、主family各名目97.5% Welch差区間で両NOMINAL_MODEL_ENTRY_LOWER、登録precision/分布診断はPASS。旧辞書の45.47%PR低費用は再配置後へ流用できない。

P/r16の個別PR/Dとbest-policy二比較は、名目95%のdelete-one影響度.25119>.25によりUNRESOLVEDを維持。近似IID coverage・ideal-law価格transfer・actual CA・全law/FT・一般性は未資格。保存候補の結果であり、再配置後の再最適化やPR-aware一般の否定ではない。

800標本（再利用288＋新512）、D3、全512 raw/compiled/実逆/XY検査PASS・correctness FAIL0。明示許可された一回の引継ぎで元SOURCE/結果/乱数domainを保全し、継続runnerはretry/resume0・442.15秒で完了、登録STOP済み。新科学条件・H2/H6・重要レビューは追加せず、公開照合後MANDATORY STOP。

[詳細結果・来歴・統計の限界](../research_c_h4_relocated_downstream_results_20261106.md)。以下は各時点の履歴を保持する。

既存ノートや旧判定は上書きしない。本記録は固定範囲の実行・保存結果の確認であり、重要科学レビューや次の実験の採択ではない。公開後は追加作業なし。
