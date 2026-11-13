# 2026-11-14：RPEのunsafe一律loss上界を状態集約で細分化

利用者の技術実装指示に従い、B6106e4d96の既存risk/decoder/shot/costを調査した。unsafeは誤branch/未解決と同義ではないため、元R3のbinary64推定状態を完全一致で集約し、解決履歴の実終端lossを計算する任意DPを追加した。zero/tieは元の共通loss familyで吸収し、追跡しない残余には旧最大lossを残す。旧/DPの有効な上界のminにより悪化を防ぐ。既存good閉包・loss契約は迂回しない。

新12＋回帰45＝57人工/回帰PASS。完全DPは独立comb/Fraction確率と元scalar全履歴列挙のloss区間/zero/tie/未解決確率にexact一致した。人工M1の旧15.0003855に対し完全DP0.03238464、部分DPが粗い例では旧上界を選択した。これは実H4の改善予測ではなく、保存rawや旧結果の新数値再解析は行っていない。

[REPORT](../rpe_mse_state_refinement_20261114/REPORT.md)／[数式・API](../rpe_mse_state_refinement_20261114/HANDOFF.md)／[有限科学案](../rpe_mse_state_refinement_20261114/FINITE_PLAN.md)。次はユーザーが追跡範囲と元LD1/LD2各1件の新適用を許可する場合だけBが実行する。whole-shot探索族・model閾値・N資格は未採択、AδE二重加算や元Minkowski変更なし。実H4新値/追加shot/MC/実C/G0、公開照合後mandatory STOP。
