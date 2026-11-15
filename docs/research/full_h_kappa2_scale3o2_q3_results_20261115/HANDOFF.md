# Full-H κ2・固定1.5倍shot：B/C・統合担当への引渡し

今回の科学SOURCEは **47ccd1f3759b1dfe45f464421f89a669d1e3eb27**。caller `scripts/run_full_h_kappa2_scale3o2_q3_fixed.py`、[INPUTS](INPUTS.json)を実行前固定した。結果公開commitとは区別する。数値判定・保存状態・ロードマップ差分は[REPORT](REPORT.md)を正本とする。

## 入出力と元candidateの対応

allocation ID `FULL_H_KAPPA2_TOTAL_Q3_USER_SCALE_3O2_01`。

```text
m = 0..12
N_X = [96,90,84,78,71,65,57,51,45,38,32,26,18]
N_Y = [96,90,84,78,71,65,57,51,45,38,32,26,18]
```

USER_SUPPLIED control、各round `ceil(3*N_control/2)`。整数列を途中変更していない。新しく識別したのはallocationとrunのみで、13physical candidate・元/total configuration・binding・law・normalization・scalarは2倍配分と同じ。

- 元configuration：`54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`。
- total configuration：`265b90dc4a9b81011736664205e23833c420d569e57c923ef45c39eb59489612`。
- decoder binding：`579c5de23a2336d402a1d5b195061b34b3c7a1b49e99ea2e1a674c1a086189a5`。
- 元13round packet：ce6aacfaca8249c510baa2d958849cffc7649cdcの`docs/research/d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/`。Full-H κ2/K2はone-bodyを含み、R448は対象外。
- decoder/kernel：Z1＋T1 `consecutive_dyadic_nearest_branch_principal_energy_total_v1`、710aa69b。高速backend9e294778を再利用。

`run01/FULL_H_KAPPA2/API_RETURN.json`が元返却値、RESULTと3種PARTIALが科学記録。SUMMARYはexact端点の25桁表示、[C_RESULT_JOIN](C_RESULT_JOIN.json)は元physical/元canonical/total candidate・m/axis・整数N・allocation・結果SOURCEの26軸索引。[COMPARISON](COMPARISON.json)に元control a82b0f0・今回・2倍422db93fcを並べ、元結果の参照を保持する。元controlと2倍は再評価しない。

## 契約と資格の境界

保存ideal finite-K raw・shots/X/Y/round間IID二項モデル、EXACT_AFFINE_OF_BINARY64_RAW、保存DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、十進εE=0.00159360010199040のexact二乗τによる直接モデルMSE filterを維持した。原packetのtarget/ground認証欄は変更せず、normalization/scalarを再適用しない。AδEを加算せず、元Minkowskiは別参照で保全する。

ρ1、非終端frontier8192、残余幾何loss・principal guard・engineering allowanceは既存契約のまま。今回vectorの全count上限9409、隣接検査273175586、transition353666241を実行前固定した。旧配分の実測状態数・ρ閉包は今回の証拠へ移送しない。m0を含むpruning質量を残余へ会計し、終端に非終端capを適用しない。

zero/tieは元HOLD/argmin規則を保持。配分変更でcount格子・偶奇も変わるので、モデルriskやzero/tieの変化を回数だけの効果としない。unsafeを実誤branch率やround別MSE寄与と同一視せず、未評価配分への単調性・補間・最小Nを主張しない。

`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。この配分の条件付きモデル判定と、物理的十分N・ground RMSE保証は別。ground/state/target包含、実sampler law移送、厳密数値certificateは未認証。価格receiptのcompiler/lowering/full-wrapper/cohort/SOURCE資格はCの既存guardで別途確認し、allocation変更だけで元receiptを無条件に付け替えない。今回価格・Gを計算しない。

## 保存状態・資源の意味

原API return・partials・RESULT・RESOURCE_FINAL・PERSISTEDはworkerが保存してから親へreceiptを返す。親のCOMPLETIONと科学判定を分けて記録する。worker/child CPU、worker peak RSS、最大単一child peak RSS、subtree RSSの観測合計ピークは別の測定範囲。subtreeのRSS和は共有pageを重複して数える場合があり、厳密な物理RAM最大ではない。

各roundの伝播境界はACCEL_PROGRESS、伝播間/終端処理はSEGMENTSに記録。PROCESS_EXITSは科学workerとDP子のOS終了code/signal、PROCESS_RETURNはcaller全体の終了code/signal。非同期frame dump/threadは追加していない。観測ログは計算進行の確認用で、科学的gateの代替ではない。

今回の判定は**MODEL_BOUND_MEETS_ADOPTED_THRESHOLD**、MSE区間は`[0.000002098434139000992158910785, 0.000002105802648067548936567344] Ha²`（表示）。原API return・資源はWORKER_ARTIFACTS_SAVED、callerと105子processは終了code0で完了した。

統合ロードマップ本体は統合担当が更新する。**元control不適合・今回1.5倍適合・保存2倍適合**という3本の固定配分結果が揃ったことと、今回の完全保存をREPORTから反映できる。最小N・単調性・未評価配分の適否やG最適性は推論しない。追加倍率・round別shot探索・新signal/価格・Gへ進まない。**固定1件完了・mandatory STOP。**
