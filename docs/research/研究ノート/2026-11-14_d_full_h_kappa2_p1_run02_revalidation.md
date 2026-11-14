# 2026-11-14：Full-H κ2 P-1 run02の再検証（方式A）— PASS、価格をCへ引渡し

利用者が方式A（保存run02を再計算なしで再検証）を承認した。

**判定の修正（SOURCE `8258b2200`）**

- 旧`scalar_once`（ancillaの`p`がちょうど1個）を、`SCALAR_EVENT_PHASE_ANCILLA_SIGNATURE_V1`に置き換えた。
- 新判定は、ancillaに作用する命令列（product cz、回転crz、requestごとの累積位相p、最後のscalar p）を、保存eventと固定provider（c31ef424 `_build`）の規則から作り、実回路とexactに照合する。
- X/Yでは、制御進化が同一であることも要求する。
- test：toyと負の対照を含む11件＋4件＋7件がPASS。

**保存run02への適用（commit `6d188defd`、23条件PASS）**

- 新しいsampling・build・simulation・compileはいずれも0。
- 原本の照合：
  - publicの原本は`cd56422a8`とbyte一致。
  - privateの11 fileはすべて存在。
  - QPYから再計算したwrapper fingerprint、数え直したcompiled価格は、compile record・trialと一致。
- 位相：event位相`p(π)`3個がrequest 0・3・7（order-2 event）に対応し、scalar `p(0.26518…)`は1回で最後の命令。旧判定はfalseを再現した。
- 原run02の`STOP_WIRING_CHECK`記録は無変更。

**価格**

- X/Y：全compiled RZ 92,448、CX 27,719、depth 55,352、9 qubit。
- record：`DIRECT_COMPILED_FINITE_SAMPLE`／`FINITE_COHORT_ONLY`、n_paths=1。
- sidecar・evidenceを付け、Cの`c628e9e8` guardを無変更で通した。2件受理し、負の対照6件は拒否された。
- 単一pathのdirect価格である。E[C]、P-2の較正点、Gには使わない。

[REPORT](../d_full_h_kappa2_p1_results_20261114/run02_revalidation01/REPORT.md)、[HANDOFF_C](../d_full_h_kappa2_p1_results_20261114/run02_revalidation01/HANDOFF_C.md)。mandatory STOP。
