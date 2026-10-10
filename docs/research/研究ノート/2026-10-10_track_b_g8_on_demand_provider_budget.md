# 2026-10-10 Track B G8

## 採用と結果前固定

利用者がGPT G7 reviewを採用。§11.1–11.3のproof/contract/on-demand pathwayを新branchで準備。
G7の限定期待T結果を保持し、native key全表のproduction依存を外す。
finite CQ誤差は仮想delta=common Rz epsilon=10^-6とsymbolic許容範囲。物理providerは選ばない。
推定0.049/資源0.001を分離、bounded cache対称、miss cap32/一key一回/retry0。
17 off-domain tests、native取得前。補助cost-aware oracleはproduction完了後の別I2診断。
全support取得/合成termination/physical delta達成の保証は未取得。bundle完了/失敗後STOP。


## 一束完了と保存値監査（追記）

source cb60a4a1336f1803ad49d890f4e74aafd6ad7c61をclean HEAD/remote一致で実行。
20 live positive native key/8 row/2048 interface trials、strict guard全通過、retry0、status `G8_ON_DEMAND_PATH_AND_CONDITIONAL_PROVIDER_BUDGET_COMPLETE`。
wall1.860446 s/CPU1.860393 s/peakRSS180776 KiB。54 source hashと旧918 path/prefix保持、保存値監査14項目PASS。
短trace/on-demand cache経路、仮想deltaと非列挙accepted cap、保存G7 Rz価格のI2診断を別証拠として記録。
物理provider/全support/backend termination/実quantum accuracy/新規性の採択には到達していない。
[結果/GPT引継ぎ](../../tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md)。
mandatory STOP。追加取得や研究方針判断を行わず、資料公開後GPT側へ返す。
