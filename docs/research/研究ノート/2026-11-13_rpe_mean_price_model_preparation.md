# 全方式の平均compiled価格モデル技術準備（2026-11-13）

[REPORT](../rpe_mean_price_model_preparation_20261113/REPORT.md)のとおり、既存MeanKernel/combine_mean、I価格metadata/B会計を再利用した。旧PR専用lookupのdomainを無理に広げず、外部係数の加法モデルと全方式の供給record接続を最小moduleへ追加した。coversを相殺/境界/測定一回の定義に使い、適合性証明とはしない。モデルは常にfalse資格、受動providerは式を計算しない。

人工新11＋旧15＋統合10＋pinned B metadata2＝38 PASS。最初のfixture不足と人工R対照の整数丸め領域を修正し、production guardを緩和しなかった。実H4/H6の新S/C、保存数値再解析、sampling/compile/fit/G0。D request/scalar、C1係数/資格、26 E[S]、既存SOURCE/結果/STOPは不変。

Claudeレビュー§4.3/T3はユーザー指定原本から参照したが、加法性/比率安定性/閾値を採択していない。次はBが必要domain・N/価格精度、Dが別承認で同条件direct/paired残差・独立holdoutの有限案を供給し、ユーザーが科学範囲と採用を判断する。Cは公開照合後mandatory STOP。
