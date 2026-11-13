# 2026-11-14：固定H4 LD1/LD2の部分状態DP 2評価を完了

ユーザーが4139bed3cの有限範囲（保持8192/round・遷移141559036/構成）と各LD1件・計2評価を明示許可した。入力/callerをce0517de6へ固定し、06db9aefeの元kernel・原raw/target/loss/IID/whole-vector/ρを維持して実行した。両workerを一度だけ起動し、チャット終了中も継続、全体8840.282秒で完了。両方86,036,445組、全roundで組数不足なし、元kernel変更なし。

モデルMSE上界はLD1約9.039527981195e−4、LD2約9.032441317634e−4 Ha²、旧上界より約99.62%縮小。追跡外質量は約7.24e−27/5.64e−27で、DP上界の約99.9477%は既知未解決loss寄与だった。残余を0にせず、実tie/未解決のexact点値はnull、幅rの区間として保存した。保存SOURCE/旧上界一致・exact質量とloss恒等式の検算PASS。

[REPORT](../h4_dp_risk_fixed_results_20261114/REPORT.md)／[HANDOFF](../h4_dp_risk_fixed_results_20261114/HANDOFF.md)。上界の縮小であり実MSE削減率や必要N/物理精度保証ではない。元Minkowskiとstate/target/数値/lawの未資格を保ち、AδEを二重加算しない。次はユーザーが未解決loss/decoder/risk水準と有限whole-shot候補の要否を判断し、Bは別指示までSTOP。追加配分/MC/価格/G0、原SRC/input/raw/旧結果/STOP・他担当を保全し、公開照合後mandatory STOP。
