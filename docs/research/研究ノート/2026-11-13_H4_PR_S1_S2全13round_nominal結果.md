# 2026-11-13：H4 PR S1＋S2・全13round nominal

明示承認された固定SOURCE c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73eのS1_S2を一回実行。S1は保存m3/m4 mean/対照を再利用し、独立reference4 actionを追加。観測norm scalar欠測はnullのまま、承認されたSOURCE gateに条件付けたempirical診断とした。S2は欠測8mean/対照をspectral powerで取得し、保存reference8 pairを再利用した。

m2のみ逐次actionと照合してsignal差約2.3e−14が固定allowance内、logB一致。m12の33,423,360 logical eventは4096outerのblock処理で完了した。速度比やrigorous同値性の主張ではない。m2直接診断のmaximum_norm_domainに共通adapterの既定文字列が残る点は、数値rawを変更せずCHECKSでSOURCE上の実意味（逐次観測）を注記した。

全13roundの経験的nominal各軸は64,60,56,52,47,43,38,34,30,25,21,17,12。旧m0/m1/m6はcopyのみ。十分N/global RMSE/actual law・実装・ground誤差は未資格/null。S3 decoder simulation、compiled価格/G、候補最適化は0。固定codeと保存入力hash前後一致、13人工/metadata testと出力scope/packet検査PASS。

[REPORT](../h4_pr_s1_s2_rmse_results_20261113/REPORT.md)／[HANDOFF](../h4_pr_s1_s2_rmse_results_20261113/HANDOFF.md)。Bは新10round scopeを受入、ユーザーはS3の未解決trial/target・誤差契約・統計精度/信頼度と有限scopeを判断し、Aは別承認後だけ実行する。今回公開照合後mandatory STOP。
