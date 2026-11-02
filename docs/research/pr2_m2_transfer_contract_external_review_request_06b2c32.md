# GPTへのPR-2 M2 held-out transfer契約レビュー依頼

## 依頼

PR-2 matched-accuracy resource studyについて、M1-B1 development結果から固定したM2 held-out transfer契約、
schema、zero-compute planをレビューしてください。今回は結果前契約だけを確認し、H4 1.30 Å held-outの
resolve/stat/hash/load、signal評価、trajectory sampling、circuit build、compile、transfer実行は行わないでください。

## 固定identity

- repository：`HIROMU1015/Partially-Randomized-Trotter`
- branch：`pr2-v4-s2-parallelization-20260928`
- M1-B1 evidence commit：`8e0814e70c14ecf526444fac8a2142799610dc96`
- M2 contract source commit：`06b2c32528713a5270ee4915432bcd3898e0e5e1`
- M1-B1 result SHA-256：
  `71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4`
- M1-B1 validation SHA-256：
  `c9a05babed99cd1e80eaec5b58e47f25d74513c7ba2e5a00775cfd5959c37a0f`
- transfer contract SHA-256：
  `a78d70c493125ef5321a989468dc41961ee5d3e053e6879a75db4b68d73e2f67`
- plan schema SHA-256：
  `6c06764a9caaf3f21b95988dbde55c56893f47d3a8d24443eee53367992fc68c`
- reserved result schema SHA-256：
  `2b653212377142070d61cd3e00a013fa5692ab3004b3d471454a2638bbd0c51d`
- zero-compute plan SHA-256：
  `7b9a2c4e9fbf4e9c58b6a5e5402239b9527c49d27ccf8e8f0f1a5ae080097132`
- zero-compute plan fingerprint：
  `e6b6ae0bcf9ff7b97eb61b2703305d793f7d531460456baf2337960fb36fb198`
- status：`M2_TRANSFER_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED`

主要ファイルは次です。

- `docs/research/pr2_matched_accuracy_m2_held_out_transfer_contract_v1.md`
- `src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`
- `scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`
- `tests/test_pr2_matched_accuracy_m2_transfer_contract.py`
- `artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`

## 暫定claimと非claim

暫定claimは次に限定します。

> H4 1.00 Å development条件では、matched-accuracy化とactual full-wrapper compileによりfixed-q/proxy比較から
> 方式判断が変わり、intermediate DF-prefix partial randomizationがresource frontierに残った。

rank 3、`q=1`を一般的最適構成とはせず、`r=4,K=2`対`r=8,K=2`の厳密winnerも決めません。M2はこの
development結論のtransferabilityだけを判定します。

## frozen candidate

次の5件だけを移します。

1. `B2-rank3-q1-r4-K2`
2. `B2-rank3-q1-r8-K2`
3. `B0-rank6-q1-r0-K0`
4. `B1-rank12-q1-r0-K0`
5. `B3-rank0-q8-r32-K4`

held-outでmethod、rank、`q,r,K,T`を再探索せず、candidateの追加、除外、置換を行いません。

## 判定規則

- accuracy/shotはM1と同じcorrected finite-RTE式でheld-out上だけ再計算する。
- primaryは`N_real*E[C_cosine,RZ]+N_imag*E[C_sine,RZ]`。
- point Paretoは`rz_count, rz_depth, cx_count, cx_depth, total_depth, circuit_size`の6指標。
- B2 materialityは`min(B2 G_RZ)/min(B0,B1,B3 G_RZ) <= 1.10`。
- random candidateのratio uncertaintyはpaired 32 trajectoryのdelta-method `point ± 2SE`。formal CIとは呼ばない。
- 重大cost underestimateは、held-out shot数×development 1-shot costの予測よりactual primary RZが10%超
  大きい場合。該当B2はsupportに使わない。
- 状態準備`P>=0` lower envelopeはsecondaryであり、terminal規則を結果後に変えない。

terminal statusは次だけです。

- `TRANSFER_SUPPORTED`
- `TRANSFER_NOT_SUPPORTED`
- `TRANSFER_INCONCLUSIVE`
- `IMPLEMENTATION_GATE_FAILED`

どのstatusでも停止し、runnerは次段階を認可しません。

## resource cap

- random：B2二件＋B3一件、各32 trajectory、cosine/sineで同じtrajectoryを共有。
- deterministic/discard：B0/B1各二軸。
- future full wrapper cap：`3 × 32 × 2 + 2 × 2 = 196`。
- worker最大5、BLAS thread各1。
- 追加96、adaptive extension、別seed救済は0。

zero-compute planではM1-B1 result/validation JSONだけを読みました。held-out pathはliteralとして記録しただけで、
resolve/stat/hash/load、signal、trajectory、circuit、compiler、quantum shot、molecular calculation、GPUは全て0です。
focused testsは30 passedで、そのうちM2専用は9 passedです。

## 確認してほしい点

1. 5構成の選択がdevelopment結果のtransferに限定され、held-out再探索を許していないか。
2. primary RZ、6指標point Pareto、1.10 materialityの役割分担が明確か。
3. development 1-shot cost×held-out shotsを事前予測とする重大underestimate定義が妥当か。
4. point Paretoまたはrobustな10% materialityをsupportとする4分岐が恣意的でないか。
5. 32 trajectory、196 wrapper、最大5 workers、追加0がbounded one-shot transferとして妥当か。
6. state-preparation感度をsecondaryに限定し、結果後の判定変更を防げているか。
7. このbundleを基礎に、science module/runner/testを別commitで実装する段階へ進めるか。

## 回答形式

次のいずれか一つを先頭に示してください。

- `APPROVE_M2_EXECUTION_SOURCE_IMPLEMENTATION`
- `REVISE_M2_CONTRACT_BEFORE_IMPLEMENTATION`
- `STOP_OR_NARROW_BEFORE_HELD_OUT`

その後、重大な問題、必要な最小修正、science source/authorization前に追加固定すべきidentity/resource/testだけを
列挙してください。このreview回答だけではheld-outを開かず、science source commit、result-prior authorization、
最終pre-execution reviewまで停止してください。
