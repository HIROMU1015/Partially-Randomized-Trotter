# Codex C向け：Full PFの保存compiled価格と新しい価格取得の条件

**更新（2026-11-13）**：B f0c4bb55への最終接続と保存価格17件のreceipt照合は[d_full_pf_full_h_v2_final_receipts_20261113](../d_full_pf_full_h_v2_final_receipts_20261113/REPORT.md)。

[REPORT](REPORT.md)／[COVERAGE.csv](COVERAGE.csv)（`X_status`、`X_rz_count`、`Y_status`、`Y_rz_count`、`price_kind`、`price_carrier`、`mismatch_or_conditional_reasons`の列）。価格は、1回のRPE interrogation（controlled時間発展＋full Hadamard wrapper＋測定、P0）のRZ gate数で、cosine（X）とsine（Y）を別々に扱う。

## 1. 区分

| 区分 | 件数 | 内容・所在 |
|---|---:|---|
| ① directがあり、物理条件が完全一致 | 56 | 49件は長時間研究の`exploration_costs`／`extension_parallel_costs`（`155529e2`、決定論でn=1、trial `result.json`はprivate）。7件はpf_common T=0.8で、candidate receipt（method、prefix、formula、q、compiler `1239304c`、rz_count）が一致 |
| ② 保存価格はあるが、SOURCE/receiptの追加確認が必要 | 17 | pf_common T=0.8の再利用。12件は`result_sha256`しかない（NEW_COMMON 6、AUDITED 5、RECOVERY 1）。5件はaxis receiptにcandidate fieldがない。pf_commonの`native_results.json.gz`との照合が必要 |
| ③ 保存価格がない（signal・referenceはある） | 63 | T=0.8 q16の全公式。m5_bestは各系15候補。COVERAGEの`SIGNAL_ONLY_PRICE_MISSING` |
| ④ 価格はあるが不適格 | 0（grid内） | grid外の参考：C providerのH4 PFはB0/p7（truncated）。H6 FULL19・m5_best・δ1.6のAffine予測（q2/q4）はMODELであり、direct価格ではない |
| ⑤ 保存gridの外（すべて欠測） | – | T∉{0.8,1.2,1.6,2.0}、または保存されていないq。Dの`for_packet()`で`OUTSIDE_SAVED_GRID_ALL_MISSING`と表示される |

- X/Y：保存価格がある73候補は、すべて両軸がそろっている。片軸だけのものはない。軸の間で補完していない。
- 他の保存系列：H6 FULL19・m5_best・δ1.6のdirect（q1 T1.6、q2 T3.2、q4 T6.4）は`CONDITIONAL`。このうちT1.6 q1は、長時間研究の価格（EXACT）と同じ物理IDである。別研究なので、native構成が同一かを確認してから使う。

## 2. 保存価格を再利用する条件

- 一致すること：compiler（Qiskit 1.3.0、basis rz/sx/x/cx、opt1、seed17、hash `1239304c…`）、cost policy `FULL_COSINE_SINE_HADAMARD_WRAPPER_MEASUREMENT_INCLUDED_P0`、symmetric-directional control、scalar一回。
- 物理候補の一致：system、FULL、全rank、formula、T、q。
- 保存の再利用値には、preparation hash、fingerprint、compiled SHA（receiptがある場合）を対応させる。
- D-only・uncontrolled・局所costは使わない。

## 3. 新compileが必要になる場合の元SOURCE

- 回路のbuild：`h46_long_time_native.compose`（`requests=None`の決定論経路。pieceは`core.schedule(formula, q, 'PER_PIECE', 1, T)`、`ax2b_native_df_v5.build_deterministic_native(formula='2nd')`）。
- wrapper：`build_native_hadamard_wrapper(axis=cosine/sine, include_measurement=True)`。
- 価格取得：`cost_trial`。trial0では、low/high control・独立occupation vector・Hadamard期待値・compiled vectorを検査する。
- compileの再利用：private `compiled_cache`がfingerprint単位で利用できる。照合してから使う。
- 新しいcompileは、上記③と⑤のうち選んだ候補だけを、ユーザーの明示承認後に行う。
