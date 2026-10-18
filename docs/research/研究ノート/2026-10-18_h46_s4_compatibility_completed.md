# 2026-10-18 S4互換性修正・再会計

[最終報告](../track_a_h46_s4_counter_native_compatibility_results_20261018.md)。利用者が3 Stageを明示承認した。元結果を保持した新branchでpf_scheduleだけを変更し、Stage 1の68 tests、Stage 2の8整数行・33block照合、Stage 3の84/84/360行再会計が全てPASSした。

Stage間ごとにSOURCE/resultを固定・公開し、one-shot grant・STOPを分離した。旧grant/ledger/STOPの流用・reset・科学retryは0。A/B/mixedのS4理論値を確認し、期待値/same-event各24行・PR/PF168行が変化した。S2/PR/shot/compiled値は不変。

元のH4主条件は理論0.871668→0.871632、compiled1.039336で方向不一致が残る。全表の方向変化0、一般的優位性や母平均保証は認定しない。旧UNATTRIBUTED分類は改変しない。

次の担当：ユーザーが原稿・科学的主張への反映を判断。追加科学計算・merge・詳細科学レビューを開始せず、最終mandatory STOP。
