# 2026-11-14：Full-H κ2 P-1 run02のSTOP（作用検査はPASS、`scalar_once`判定の偽陰性）

利用者のチャット承認により、run01のSTOP（実行treeの`references/`欠落）を修正してP-1を1回だけ再実行した。

- caller：`72d4f4ff2`。固定SOURCE `04bead115`からの変更は、出力label `run02`と子processのimport preflightだけである。科学条件は変えていない。
- 結果：`STOP_WIRING_CHECK`。retryはしていない。

**得られた観測**
- 実H4の1 path（568 event、order2が3個）について、作用検査7件はすべてPASSした。
  - control=0は恒等、leakageは0、独立oracleとの差は1.2e−14。
  - Hadamard X/Yは≤1.1e−15、compiled作用は≤9.9e−12。
- 新controlled ZZは有効だった：`crzz` 0、`crz` 568、置換172、parity PASS。one-bodyは205 application。
- 2 compileの直接価格（X/Y同値）：RZ 92,448、CX 27,719、total depth 55,352、9 qubit。STOPのため未資格で、Cへは引き渡していない。
- 実trajectoryの`log_B`はS-1のm3と一致した。

**STOPの原因**
- callerの`scalar_once`判定は「ancillaの`p`がちょうど1個」を要求していた。
- 実際の`p`は4個だった。
  - scalarの`p(−scalar·T)`が1個（最後、角度はbit一致）。
  - provider builderがrequestごとに付けるevent位相`p(π)`が3個（order-2 eventを含むrequest 0・3・7）。
- 保存回路の構造診断で、この4個がbuilderの規則どおりであることを確認した。判定式の偽陰性と判断するが、固定gateを書き換えてPASS扱いにはしていない。

**次の判断**
- 判定式の修正（scalar角の`p`が1個で最後にあり、残りはeventから計算した累積位相と一致すること）。
- 修正後の判定を保存run02回路に適用して受理するか（推奨、再計算なし）、run03を実行するか。

[REPORT](../d_full_h_kappa2_p1_results_20261114/REPORT.md)。mandatory STOP。P-2・S-2・N/MSE/Gへは進まない。
