# B：S3全RPE統計の最終実行計画

2026-11-13 JST。**既存実装を変更せず、ユーザーが科学条件を採択できる具体案を整理した。未採択・未実行。** 実H4 simulation、保存meanの数値再解析、shot最適化、compiled価格/G、新しい科学テストは0。

[最終科学計画](SCIENTIFIC_PLAN.md)／[採択・実行HANDOFF](HANDOFF.md)／[静的照合記録](VERIFICATION.txt)。旧計画・S3判断表・R3統合を再利用し、新controller/runner/CI/承認機構は追加していない。branch `codex/rpe-decoder-statistics-20261113`を継続した。

## 推奨する最初の有限実験

固定canonical `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、H4 DF10/S2/L_D1/ΔPF=h=.1/K2/M12・元exactκ/H-state/prior/decoder/law/profile/P0・元budgetを保全。全13roundの各軸nominal64,60,56,52,47,43,38,34,30,25,21,17,12を1配分として使う。

推奨は保存solver energyを条件付きproxyとするideal-law IID binomial評価。exact整数count callerとR3 batch hookを使用し、元scalar規則を保持する。主評価は未解決lossをUNDEFINEDとする**経験的な解決条件付きRMSE＋未解決率CI**。全試行の特定energy estimatorのRMSEは定義しない。

別途ユーザーが採択する場合に限り、未解決loss∈[0,U]のrisk感度区間を同じtrialから出す案を付けた。U=4096 Ha²、η=.01U、各CI97.5%・joint≥95%、零観測CP解像度10^−4に対して、汎用設計算術から**n=43,819固定・seed20261113**を提案する。H4 mean/確率/lossからnを計算したものではない。

η=40.96 Ha²は粗いpilot用のMSE端点の余裕幅で、化学精度・十分N・global boundの判定精度ではない。副評価を採らなければMSEのCI/精度を主張しない。条件付きRMSE点推定のCIは現APIにない。より厳しいηを希望する場合は既存trial_budget式から別nを事前決定し、今回のnへcapしない。

## 必要なユーザー採択と未資格

proxy、ideal-law IID、count caller、高速decoder、主loss、任意の副loss/support、数値誤差を伝播しないplug-in scope、精度/confidence/反復数・seedを[HANDOFF](HANDOFF.md)に列挙した。条件の採択と科学実行許可が必要であり、今回の計画公開はどちらの代用でもない。新しい証明書やgrantは不要。

Aの経験的allowance・signal bias・m3/m4観測norm欠測・target/law差unknownは0にせず、L2/energy誤差へ新変換しない。保存proxyはground_state_certified=false。全十分N/global RMSE bound/compiled C/Gはnullを維持する。A/C/Dの作業・process・rawを操作していない。

確認はSOURCE/identity/nominal/未資格metadata、計画算術、相対リンク、差分の静的照合のみ。R3人工テスト60件はab12の既存証拠として参照し、今回再実行していない。実行SOURCEはab12、入力はbf515/A固定refs。本資料の公開commitは計画SOURCEとして区別する。

次は**ユーザー**が推奨セット又は変更点を採択し、有限1実験を明示許可する。**Codex B**はその許可後だけ実行する。今回の計画をcommit/pushし、remote SHA/tree/主要資料一致を確認後mandatory STOP。科学実行へ自動進行しない。
