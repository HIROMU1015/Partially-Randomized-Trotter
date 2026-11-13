# A → B：H4 PR全13round nominal・S1＋S2結果

2026-11-13 JST。科学SOURCE **c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73e**、固定S1_S2を一回実行し局所診断完了。全13roundの経験的nominal N_X=N_Yは **[64,60,56,52,47,43,38,34,30,25,21,17,12]**。十分N/global RMSEは未資格/null。

[REPORT](REPORT.md)／[V2 packet](run01/N_HANDOFF.json)／[result](run01/result.json)／[CSV](ROUND_SUMMARY.csv)／[SOURCE](SOURCE_INDEX.json)／[checks](CHECKS.json)。

## 固定contextと新scope

- B参照3216d27a69ac8532d2a22e78f5f94491d202d238、canonical ID **587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911**。
- H4 linear1Å/STO-3G/保存DF10/sector(2,2)/36、S2/L_D1/K2/Δ.1/h.1/M12/S0 exactκ・元H/state/snapshot/law/profile/prior/decoder/scalar/P0/minkowski/finite_Bは不変。
- previous_packetはA3e035c0d…のm1結果。m0/m1/m6の64/60/38と各元SOURCEをcopyし、新たなmean/shot再解析はしていない。
- `diagnostic_scope.newly_evaluated_rounds=[2,3,4,5,7,8,9,10,11,12]`、`saved_mean_rounds=[3,4]`、`new_mean_rounds=[2,5,7,8,9,10,11,12]`、`preserved_rounds=[0,1,6]`、`whole_schedule_RMSE_qualified=false`。
- 各rowは全canonical identity/law/source_request_id/physical source_candidate_idを保持。新10roundのdiagnostic_SOURCEはc76ffe5d、旧3roundは元SOURCE。root sourceは今回の組立SOURCEで、旧3roundを再計算した意味ではない。

B3216d27aのm1専用validatorへscopeを偽装していない。既存`n_from_a`で変換した**新10round scope**なので、B担当が対応する受入条件を追加する。旧S0/m1 packetを上書き・再ラベルしない。十分N_X/Yは全26軸null、global bound=null、model/global/finite-K/signed資格はUNQUALIFIED。価格record・Gは供給しない。

## 計算と証拠水準

| 段階 | 完了範囲 |
|---|---|
| S1 | m3/m4保存mean/対照を再利用、独立reference4 action、局所診断とnominal2組 |
| S2 | 高速mean/内部対照8候補、保存reference8 pair再利用、新reference0、nominal8組 |
| 同値性 | m2のみ逐次action＋内部対照で照合。corrected/raw/exact-tail差最大約2.303e−14、比較allowance2.828e−9内、logB一致 |
| 保全 | m0/m1/m6未再解析、固定code/保存入力hash前後一致、retry0 |

m3/m4 observed norm scalarはnull/欠測を維持し、承認されたSOURCE gateを使った**条件付きempirical**診断。S2の新norm観測値は元gate内。raw/correctedは独立伝播、scalar一回、R/K/law/order変更なし。R-aware allowanceとnormalizer感度はempiricalで、rigorous boundではない。

m2 direct診断の`allowance.maximum_norm_domain`には共通adapterの既定ラベルが残るが、実maximumはD2の逐次half-step/event `checked()`の観測値である。元rawは保全し、CHECKSのtechnical_annotationsにSOURCE上の意味を記録した。主8高速経路のmaximumはblock境界である。

全signal・bias・位相・半径・u/margin・reference・SOURCEはresult/個別mean/CSVを参照。局所disk/βdetから全energy RMSEを推論しない。actual law/実装/ground誤差はnull。S3 decoder simulation・trajectory・compiled価格・G・Full PF/8次PFは0。

## 次の担当・STOP

**B**：この新scopeを受入し、13round nominal coverageと十分N/global資格の欠測を分けて表示。

**ユーザー**：S3の全trialに対するloss/未解決trial、target/state証拠、Minkowski成分、ideal/actual law、MSE精度・信頼度・有限反復数を判断し、具体的S3実行を別途承認。

**A**：承認後だけS3へ進む。今回13人工/metadata test PASS、実行scope/packet/input保全を確認。commit/push・remote照合後mandatory STOP。
