# C1 Stage 3：feasibility-only限定科学実行

2026-10-25 JST。固定公開SOURCE `6ea5c6a9c10cc4b5411039a0d61a9f19acc4f34f`。
Branch `codex/research-c-c1-stage3-feasibility-science-20261025`。

**限定実行は正常完了。260セル、20候補、639/639 logical units。候補はPOTENTIALLY_FEASIBLE 14／INFEASIBLE 6／UNRESOLVED 0。科学的correctness FAIL 0、retry／resume 0。mandatory STOP。**

[原本・要約・CSV・検証](../../artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/)／[実行要約JSON](../../artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/execution_summary.json)／[260セルCSV](../../artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/round_values.csv)／[20候補の分類・根拠JSON](../../artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/candidate_summary.json)。

## 固定条件と実行

ユーザーが当該SOURCE・[限定契約](research_c_c1_stage3_feasibility_contract_unapproved_20261025.json)に対する科学実行をチャットで明示許可した。承認証明書・承認記録・新grantは作成していない。契約の準備時点の`unapproved`ファイル名とmetadataは履歴として無変更で保持する。

- 保存P-H4：直鎖1.00 Å、STO-3G、Nα=Nβ=2、sector36／full256。
- CDF：保存rank8、通常number表現、元gで補正したκ・scalar・元frame順を保持。XDF：保存rank10。
- CA=0.00159 Ha、DF reserve CA/8=0.00019875 Ha、δ=0.1、13 round m=0..12、q=2^m。
- 決定論：ordered symmetric S2、s∈{1,4}、L_Dは全frame。PR：L_D=2、s∈{1,4}、r∈{4,16}、K∈{0,2}。既存順序の20候補に限定。
- 各DF Hamiltonianのsector ground-state oracle、状態準備除外、ideal finite-RTE IID law。実装有限ビットlaw誤差は未評価。
- βtotal=π/6、βstat=π/12、alias窓[-10,10] Ha、既存toleranceを無変更で使用。中間状態のsectorへの再射影はしない。

独立worktree `/tmp/prt-c1-stage3-feasibility-science-20261025`、fresh output `artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/run01`で専用entryを1回実行した。既存CAR・一般K値adapter・保存operator・ground／alias・finite-RTE・matrix power／norm・call計上・resource planをそのまま再利用し、SOURCE修正は0。

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_stage3_feasibility.py --mode feasibility-only \
  --contract docs/research/research_c_c1_stage3_feasibility_contract_unapproved_20261025.json \
  --input /tmp/prt-c1-stage1-compatibility-20261025/.external/P-H4.npz \
  --execute --output artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/run01
```

これは実行記録であり、再実行用の許可ではない。当該outputは既に使用済み、retry／resume 0。

## Preflight・環境・正しさ

| 項目 | 実績 |
|---|---|
| 固定SOURCE 50ファイル、保存入力17点のbyte照合 | PASS、実行後も無変更 |
| 元P-H4 input | SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 保存fit export | SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| 保存fit SOURCE | `da2edb465d5ae6432439fbd1df8e57512ee59c11`、今回の評価SOURCEと独立保持 |
| 契約SHA256 | `23a2e24f5162b57b38cdbf61095b9cfab2481f100dae68e10a21e6769c1a72ad` |
| 評価SOURCE manifest SHA256 | `b6df9c1a22571e098650b52ca93e7314507dd71d0836e6fd0f9f028e77989de0` |
| 正式共通指示2026-10-25版 | SHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720` |
| Python・依存 | Python3.11.1、lock52 packageのversion一致、NumPy1.26.4／SciPy1.14.1 |
| 人工回帰 | 16 PASS、5.85秒。実H4の判定を代用しない |
| 保存sector Hermitian・ground | 3 eigenpair call COMPLETE、3 ground NPZ保存 |
| CDF／XDF full ground埋込・sector整合・alias | 2 call COMPLETE、PASS |
| round reference／分類 | 各260 COMPLETE、全数値finite、欠測0 |
| 科学call journal | 639 attempt／639 COMPLETE、重複label0、FAIL0 |

保存Stage1の物理sector DF operator誤差はCDF `7.666027761392501e-5 Ha`、XDF `3.3751558354542566e-15 Ha`で、今回normを再計算していない。登録の元engineering allowance込みで使用したDF reserve値はCDF `7.666215916301914e-5 Ha`、XDF `1.4812631843704909e-9 Ha`。ともにCA/8以内。これらは数値pilot用の値であり、厳密区間証明ではない。CDFとXDFの物理DF精度は異なる。

## 候補別結果

PF列は`||P^n − exp(−iHt)||₂`、補正RTE列は`||C^n − P^n||₂`、Aは既存normalizationによるattenuation。PRでn=s・2^m、kernel時間δ/s。決定論ではC=P、A=1。operator誤差は無次元であり、Hartreeのエネルギー誤差ではない。

固定十分条件はA>0かつPF誤差＋補正RTE誤差≤sin(βtotal−βstat)（約0.2588190451）。全13 roundと保存DF reserveを確認する。**合成誤差は未知のままであり、POTENTIALLY_FEASIBLEはCA達成を意味しない。INFEASIBLEは登録の十分条件を満たさない意味で、物理的な不可能性を示さない。**

以下のmax／minは保存13セルの記述集計だけで、資源順位付けではない。

| 候補 | 判定 | 不適格round m | max PF誤差 | max 補正RTE誤差 | min attenuation |
|---|---|---|---:|---:|---:|
| `CDF_D_s1_r0_K0` | POTENTIALLY_FEASIBLE | — | 0.069136931 | 0 | 1 |
| `CDF_D_s4_r0_K0` | POTENTIALLY_FEASIBLE | — | 0.0043213383 | 0 | 1 |
| `CDF_PR_s1_r4_K0` | INFEASIBLE | 9,10,11,12 | 0.071626068 | 22.598672 | 1.453685e-11 |
| `CDF_PR_s1_r4_K2` | POTENTIALLY_FEASIBLE | — | 0.071626068 | 0.00017725532 | 2.1446054e-22 |
| `CDF_PR_s1_r16_K0` | INFEASIBLE | 11,12 | 0.071626068 | 1.2043526 | 0.0019352871 |
| `CDF_PR_s1_r16_K2` | POTENTIALLY_FEASIBLE | — | 0.071626068 | 2.769891e-06 | 3.7462038e-06 |
| `CDF_PR_s4_r4_K0` | INFEASIBLE | 11,12 | 0.0044769601 | 1.2043526 | 0.0019352871 |
| `CDF_PR_s4_r4_K2` | POTENTIALLY_FEASIBLE | — | 0.0044769601 | 2.7700517e-06 | 3.7462038e-06 |
| `CDF_PR_s4_r16_K0` | POTENTIALLY_FEASIBLE | — | 0.0044769601 | 0.21848717 | 0.20971318 |
| `CDF_PR_s4_r16_K2` | POTENTIALLY_FEASIBLE | — | 0.0044769601 | 4.3279539e-08 | 0.043979777 |
| `XDF_D_s1_r0_K0` | POTENTIALLY_FEASIBLE | — | 0.06843041 | 0 | 1 |
| `XDF_D_s4_r0_K0` | POTENTIALLY_FEASIBLE | — | 0.0042770929 | 0 | 1 |
| `XDF_PR_s1_r4_K0` | INFEASIBLE | 10,11,12 | 0.068782476 | 3.086685 | 1.950525e-06 |
| `XDF_PR_s1_r4_K2` | POTENTIALLY_FEASIBLE | — | 0.068782476 | 6.1335499e-05 | 3.8201734e-12 |
| `XDF_PR_s1_r16_K0` | INFEASIBLE | 12 | 0.068782476 | 0.4219124 | 0.037278935 |
| `XDF_PR_s1_r16_K2` | POTENTIALLY_FEASIBLE | — | 0.068782476 | 9.5840239e-07 | 0.0013898082 |
| `XDF_PR_s4_r4_K0` | INFEASIBLE | 12 | 0.0042991154 | 0.42192156 | 0.037278935 |
| `XDF_PR_s4_r4_K2` | POTENTIALLY_FEASIBLE | — | 0.0042991154 | 9.5847311e-07 | 0.0013898082 |
| `XDF_PR_s4_r16_K0` | POTENTIALLY_FEASIBLE | — | 0.0042991154 | 0.091991432 | 0.43938889 |
| `XDF_PR_s4_r16_K2` | POTENTIALLY_FEASIBLE | — | 0.0042991154 | 1.4971545e-08 | 0.19306279 |

20候補の分類は両armで同じ設定別7潜在適格／3不適格だった。決定論S2のs1／s4は両方が潜在適格。PRはK2全設定とK0・s4・r16が潜在適格、残るK0の3設定が不適格。round単位では247潜在適格／13不適格／0未解決。

6候補の不適格は長roundの登録PF/RTE十分条件の超過であり、correctness FAILとして扱わず、契約どおり他の登録候補を継続した。attenuationのbinary64ゼロは今回0セルで、既存A0早期returnによるnorm欠測はない。

一方、潜在適格でも減衰は小さくなり得る。CDF PR s1/r4/K2のmin Aは約2.14e-22、XDFの同設定は約3.82e-12だった。固定条件には正のAへの追加の下限を設けていないため、この分類だけでは測定負担・実用性を判断できない。shotsや総費用への換算は今回行っていない。

## 科学call・資源・保全

| 科学callカテゴリ | 実績／上限 |
|---|---:|
| input_load／fit_export_load | 1／1、1／1 |
| saved_operator_load | 13／13 |
| ground_eigenpair／model_prepare | 3／3、2／2 |
| car_context／xdf_full_generator／xdf_full_sum | 1／1、11／11、1／1 |
| state_alias_sector_checks／split_native_model | 2／2、2／2 |
| rte_configuration／exact_model_evolution | 16／16、26／26 |
| kernel_reference | 20／20 |
| round_reference／feasibility_assessment | 260／260、260／260 |
| candidate_feasibility_summary | 20／20 |
| 合計 | **639／639** |

新fit／optimizer／retry／resume／回路build／compile／synthesis資格付けはすべて0。shots／総費用／資源順位／CA達成は未評価のnullで、0費用や達成扱いにしない。

開始前CPU idle snapshot99.6%、affinity32 CPU、空きRAM約52 GiB。既存resource planが20独立candidate workerを選択し、26 exact evolutionも20 workerで実行した。process内BLASをworker当たり1 threadへ調整し、多重並列を回避した。共有設定・既存venv・他ユーザーのjobは変更していない。wall13.379秒、process CPU51.767秒、peak RSS804,664 KiB。速度比較は実施せず、speedup倍率は主張しない。

実行後の整理ではJSON／call journal／CSV／byte照合だけを使い、追加のoperator評価・norm・PF/RTE計算は行わなかった。260セルのprogressとcandidate原本、ground3 NPZ、SOURCE／契約／runtime／call／resources／STOPを保存し、旧SOURCE・結果・grant・ledger・marker・STOPは保全した。保存結果はこの固定SOURCEのlocal scientific evidenceであり、外部独立再現ではない。

## 残る研究判断とSTOP

今回得られたのはPF/RTEと減衰のfeasibilityで、全lawで有用な合成誤差上界、実装finite-bit law誤差、費用support、shots・同CA総資源比較は未解決／未評価である。潜在適格14候補に資源優位を付与しない。

次はユーザー／研究Cが、候補別のPF/RTE余裕とattenuationを確認し、どの候補について合成誤差資格・費用support等を準備するか判断する。Codexはその具体的な依頼後にだけ技術準備を行う。追加科学計算・回路build／compile・shot／総費用評価には別途明示許可が必要。今回commit・push・remote照合後は**mandatory STOP**。
