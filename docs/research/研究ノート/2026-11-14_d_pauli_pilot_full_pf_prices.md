# 2026-11-14 D：Pauli LD1/m0 pilot callerとFull PF同条件価格caller

Pauli LD1/m0 pilotは、A P0辞書・保存operatorと全round packetをSHA照合して読む。
native DF head（prepare(ham,1)）とD2 Pauli tailを、既存seed family・samplerで構成する。
保存mode-0-MSB→Qiskit little-endianの変換はAのreverse_bitsとし、full-2^nの独立oracleでindex0を照合する。
toyでは、独立なJordan–Wigner構成の保存順headとnative回路が一致した。変換の省略や二重変換、headの取り違え、辞書・s_Rの不一致、scalarの二重適用は検出された。
保存P0（LD1/LD2）のbit順はAへの照合結果として全項目PASS。C cfc0fae6の `PR_PAULI_TAIL` guardがsidecarを受理した。

Full PFはTruncated p7と同じround・q・実幅で、S2/Y4/M5のm4〜m6を全11 block・定数scalarで準備した。
保存M5 m4を除いて8 cell・16 wrapper。FULL_PFとTRUNCATED_PFのsidecarを同じpolicyでguardに通し、Cのfit入力形式に接続した。

新規16、回帰30のtestがPASS。凍結したS-1/P-1/T3 callerは不変。実sampling・build・compile・価格・fit・N・Gは0。
[REPORT](../d_pauli_pilot_full_pf_prices_20261114/REPORT.md)。公開照合後mandatory STOP。
