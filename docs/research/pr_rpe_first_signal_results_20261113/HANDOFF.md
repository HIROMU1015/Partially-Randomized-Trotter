# A → B/D：LD1/LD2全13roundの物理signal

2026-11-13 JST。**26/26局所数値gate通過、78complex・finite normalizer26、新reference0・旧mean再解析0・retry0。十分N/global RMSE/G/winnerは未資格/null。** [REPORT](REPORT.md)／[全round表](ROUND_SUMMARY.csv)／[SUMMARY](SUMMARY.json)。

## 固定SOURCEとidentity

- 計画：`a9abf8de75bf3432fcd714e868e5c14c8884b8f8`、`docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json`と`CONFIGURATION_LD2.json`。
- 数値provider：`fefd548ecd56f60d72031b3d385f6f5e6ee95d06`。最小caller：`041ac435a38afec3cf2a426661351871a1bac8ef:scripts/run_pr_rpe_first_signal.py`。
- **raw結果carrier：`16210932e5dabe7d12d41f35fea0e0a0d7a9b271`**、`docs/research/pr_rpe_first_signal_results_20261113/run01/LD{1,2}/m{m}.json`。元recordは変更していない。
- B正本：`437ba4acdb82708b6fc98e3cd119a2a734f7cf64`、V2 fixed-A configuration/whole-shot評価設計。

LD1 ID=`a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d`、LD2 ID=`1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`。元physical IDとB canonical candidate IDは各record・CSVで別々に保持。H/state/snapshot、元exactκ、S2/Δ.1/K2/h.1/M12、prior/decoder/誤差予算、P0は計画どおり。

## Bの既存統計入口へ

[B_MEANS_LD1.json](B_MEANS_LD1.json)／[B_MEANS_LD2.json](B_MEANS_LD2.json)は`configuration_source, configuration_id, target, means`を持つ。meansの`identity/law_binding`は正本rowのcopy、`source`は上記raw carrierへのimmutable commit/path。新configurationを作り直す必要はない。

```python
# 供給データの読込。Aのsignal/normalizerを再呼出ししない。
configuration = read_pinned(bundle['configuration_source'])
means = bundle['means']
target = bundle['target']
# shots/statistical contract は今回未指定。
# 別の有限科学承認後にB既存prepare/simulate_prepared/summarizeへ渡す。
```

| 入力 | 意味と制約 |
|---|---|
| raw | ideal finite-Kの未補正complex mean。X=Re(raw)、Y=Im(raw)、p(+)= (1+coordinate)/2。correctedから測定確率を作らずBを再乗除しない |
| corrected/exact_tail/reference | 有限normalization補正・同PF内部対照・保存stateのfull-H reference。referenceは厳密groundの単一位相と自動同定しない |
| normalization | producerのfinite logB/B/attenuation、元empirical margin/normalizer ULP感度。再計算せずその資格を保持 |
| bias | corrected−reference、exact-tail−reference、corrected−exact-tailのsignal差。phase/Tや単純合算でenergy RMSEへ変換しない |
| uncertainty | empirical uと元数値診断・最大norm観測境界。raw rigorous bound/ground/actual law/finite-K RMSE移送はnull |
| target | 同保存DF-H/同sector、shift0、同binding/source、保存solver proxyの出典。quantity=`ORIGINAL_DF_H_TARGET_ENERGY`はDF前分子H保証を意味しない |
| qualification | `UNQUALIFIED`を維持。局所numerical gate passはRMSE資格とは別 |

[B_READINESS.json](B_READINESS.json)ではB437の既存`prepare`を**shots=[]**でmetadata確認しただけ。両方でblockerは`MISSING_N_X`・`MISSING_N_Y`・`PHYSICAL_MEAN_UNQUALIFIED`、statistics_executed=false。これはNを0にした実験ではなく、shot未指定の状態を保つ受入確認である。nominal Nも生成していない。

次のB科学実行には、whole-shot family・claim/loss・target proxyの扱い・IID law/actual lawのscope・MC又は別risk法の精度と有限範囲を別途固定する必要がある。旧S3のN/seed/反復数・nominal配分を自動採用しない。元fixed-A専用N packetへ今回のmeanだけを入れて十分Nに昇格しない。

## D：平均compiled価格のmetadata

[D_PRICE_REQUESTS.json](D_PRICE_REQUESTS.json)に2configuration・26candidate・X/Y52軸を列挙した。各candidateのq/R/K・allocation/event数・physical ID、V2 identity/law source、profile/compiler/control/full measured Hadamard wrapper/P0を使用する。LD1 m12はq4096/R8160、LD2はq4096/R2559。価格は未評価/nullで、新compile/fit/C/Gは0。

headを一つ増やす価格と、共通κから派生するtail event数変更を両方扱う。λ/scalar/sourceやq/R groupingの違いをevent数だけで置換せず、既存price domainに適合する範囲と追加検証が必要な範囲を分ける。古典matrix power/cacheを量子回路の縮約と解釈しない。normalization/B²や価格標本数をN×Cへ追加乗算しない。

## 数値限界と次の担当

closure最大はLD1 2.538524340812354e−10、LD2 1.311018718304664e−10で固定1e−9を通過。長時間のu/logB marginはlogical event数に依存するempirical値。小さなfinite-RTE差を精密に認証したとは扱わない。旧m3/m4欠測の原証拠や旧169d結果はそのまま保全している。

**B**：許可された同claim/targetの全round shot/risk評価。**D**：許可された同identityの平均compiled C/domain精度。**ユーザー**：それらの有限scopeと、必要になった追加prefix/κ/幅/formula/K探索を判断/承認。**A**：今回の公開・remote照合後mandatory STOP。N・G・winnerを今回認定せず、二候補で探索を完了したとしない。
