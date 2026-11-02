# H4 conditional C/N/G saved-data results, 2026-10-24

[研究結果・条件・限界・次の判断](../../../docs/non_df_h4_conditional_cng_results_20261024.md)。新SOURCE `b1984dcde06a874d0fd6e9fe6395b7f3fb7504d8`。20/20非空、100仮定シナリオ・200軸・40閾値・30領域/主比較10・514旧価格join・PR2 tail解析を一回完了。実rho/母平均順位は未確定。元H4 null/native欠落とC/L/旧compiler状態は保全。Mandatory STOP。

主要ファイル：

- [20条件関数](run01/conditional_functions.json)、[100シナリオCSV](run01/conditional_results.csv)／[JSON](run01/conditional_results.json)、[200軸会計](run01/axis_results.csv)、[40 guard閾値](run01/guard_thresholds.csv)。空欄は欠測値で、0ではない。
- [4方式の単価・paired共分散](run01/cost_summary.json)、[30比較領域](run01/comparison_regions.json)、[PR2方式の希少event・反転感度](run01/rare_cost_sensitivity.json)。Population stratum meanはnull、observed maximumはupperではない。
- 図：[回路単価](run01/cost_comparison.png)、[仮想theta断面G](run01/conditional_G.png)、[guard予算](run01/residual_guard.png)。同名SVG/PDFも保存。G図は各方式の表示断面であり、全域の順位は独立box比較JSONに基づく。
- [結果要約](run01/result_summary.json)、[検証](verification.json)、[保存出力checker](verify_saved_results.py)、[人工16件ログ](artificial_tests_final.log)。Checkerは科学再実行を行わない。
- [今回の明示承認](approval.txt)、[GitHub取得](github_verification.json)、[SOURCE/input/runtime seal](seal.json)、[runtime](runtime.json)、[active contract](active_contract.json)。固定550入力と40 pointerは基点の計画directoryを参照。
- [新grant](run01/grant.json)、[消費](run01/CONSUMED.json)、[call ledger](run01/call_ledger.json)、[今回のnative exit](run01/native/worker_native_exit.json)、[native回収](run01/native/STOP.json)、[mandatory STOP](run01/STOP.json)。旧runの監視証拠の代替にはしない。
- [公開状態](publication_status.json)。専用branch `codex/non-df-h4-conditional-cng-results-20261024`。旧SOURCE/grant/STOPの再利用・retry/resumeは禁止。

再解析runtimeは既存project SciPy環境の読み取り専用利用、package追加0。重い分子計算を含まない約12秒の小規模処理であり、共有環境の変更やCPU/RAMの人工制限は0。旧C/Lの局所診断は再実行せず、実残差へ無証明で足し込まない。
