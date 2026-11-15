# Cへ：PR-DF LD1 F0保存確率診断

2026-11-15 JST。**F0_DIAGNOSTIC_COMPLETE：予定14／再利用0／新規14完了／保留0／異常0。条件付きevent/path上界は各13件、X/Y26軸で共有。**

1. **段階差とsupport**：componentの正規化TVはexact0、CDF丸めTV≈3.9132533×10⁻¹⁵、格子TV≈8.3802388×10⁻¹⁶、P→G全TV≈4.5412949×10⁻¹⁵。CDF段階で25、格子段階で追加7、合計32 componentが正→0、逆変化0。order2のsupport消失は全13表で0、order表の格子TVは全てexact0。
2. **条件付きpath上界の有用範囲**：m0〜m12の全13roundで1未満。最大はm12の約1.5248229×10⁻⁷で、1に飽和するroundはなかった。これは独立一様53-bit格子等の仮定下でのlaw差の情報であり、実PCG64又は価格の採用精度を認定するものではない。
3. **次に必要な証拠**：実運用の確率モデル/seed ensembleとconditional-gridモデルの接続を明示し、ideal E[C−S]へ移すための全support残差幅又はobservableに即した移送根拠を得る必要がある。有限残差/rare/長roundの価格資格も残り、今回の差だけからE[C]補正は出さない。

[REPORT](REPORT.md)／[機械可読HANDOFF](C_HANDOFF.json)／[元RESULT](run01/RESULT.json)／[段階TV exact CSV](TABLE_TV.csv)／[order2 exact](ORDER2_MASS.csv)／[bound exact](ROUND_BOUNDS.csv)／[support変化](SUPPORT_CHANGES.csv)。

## 固定参照

- 実行SOURCE **`fad0c1e4fe41c0e4c6d0690158056572295ed15a`**。同commitの[INPUTS](INPUTS.json)に保存input locator・元file SHA・順序/hex/τ/normalizationを固定。
- 採択計画 `950577f416c17a20d309a8053a4bb016027b724d:docs/research/d_pr_df_ld1_ec_plan_20261115/NEXT_F0_PLAN.json`。
- helper/rule `d72dc4e6228bb380f4d1d000e5e030794fce7164`、元CDF `155529e20a8fac39e00786c40b57830b71203c4f:scripts/h46_long_time_native.py#cdf_law`。NumPy1.26.4/元SOURCE正規化/Fractionを維持。
- raw公開SHA **`ec78fde57e117e9bbf79a26bfaa2e1575c1f5784`**、同commit `docs/research/d_pr_df_ld1_f0_law_diagnostic_20261115/run01/`。
- 最終公開SHAは本HANDOFFを収録するcommit。remote一致後、完全SHAを最終回答で報告する。

`C_HANDOFF.json#tables` の14行は元診断とSOURCE入力へ、`rows` の26軸は共通component診断・round別order診断・conditional boundへ `{commit,path,selector}` で戻る。SOURCEを結果公開commitへ付け替えていない。

## round別上界（表示値、exactは元JSON）

| m | q / R | 保存event_count | 条件付きevent TV上界 | 条件付きpath TV上界 |
|---|---|---|---|---|
| 0 | 1 / 2 | 2 | 4.649452253013e-15 | 9.298904506027e-15 |
| 1 | 2 / 4 | 8 | 4.616519709452e-15 | 3.693215767561e-14 |
| 2 | 4 / 8 | 32 | 4.642276552142e-15 | 1.485528496685e-13 |
| 3 | 8 / 16 | 128 | 4.591916352138e-15 | 5.877652930737e-13 |
| 4 | 16 / 32 | 512 | 4.591276392244e-15 | 2.350733512829e-12 |
| 5 | 32 / 64 | 2048 | 4.580668970379e-15 | 9.381210051337e-12 |
| 6 | 64 / 128 | 8192 | 4.585487605951e-15 | 3.756431446795e-11 |
| 7 | 128 / 255 | 32640 | 4.616790493994e-15 | 1.506920417239e-10 |
| 8 | 256 / 510 | 130560 | 4.579192577462e-15 | 5.978593829135e-10 |
| 9 | 512 / 1020 | 522240 | 4.545030818161e-15 | 2.373596894477e-09 |
| 10 | 1024 / 2040 | 2088960 | 4.592177349693e-15 | 9.592874796414e-09 |
| 11 | 2048 / 4080 | 8355840 | 4.616117860756e-15 | 3.857154226562e-08 |
| 12 | 4096 / 8160 | 33423360 | 4.562147283611e-15 | 1.524822910332e-07 |

全13roundで `NONVACUOUS_TV_BOUND_ONLY_NO_PRICE_THRESHOLD`。同じlabelが同じword/angle/phaseを表し、各モデル内のorder条件付きcomponent独立、全event/outerの独立一様53-bit入力又は一致履歴で一様な条件付き距離という前提を明示している。**実PCG64のjoint距離、E[C]への補正、全support残差幅はnull。** 現在のQ3 target `ideal_finite_RTE_law` は維持し、helperのQ/Gをactual path law/総量子資源と混同しない。

component P→Gで失われた32 supportは、単に観測されなかったという意味ではなく、固定CDF/grid写像で質量が0となったもの。元Hamiltonian/重みは削除せず、idealをgrid標本から評価する際には欠ける寄与の扱いが要る。order2は全roundで非零であり、実eventの新観測ではない。

次の不足は実運用の確率モデル/seed ensembleとgrid条件の接続、およびideal E[C−S]への移送を支えるfull-support残差幅又はobservable別の方法。有限path/rare/共有model誤差・長round検証と必要価格精度/信頼水準も別に残る。F0結果だけでC guardを価格資格へ昇格しない。m4を較正へ戻さず、凍結価格モデルと保存E_I[S]を保全する。

資源：wall0.50秒、内部0.075383830秒、max RSS41552 KiB、swap0、exit0。新人工reader試験14 PASS、既存helper記録17件を再利用。新path/H4 build/compile0、E[S]再計算/E[C]/CI/価格予測/fit/Q3/N/G0、science retry0。

ロードマップ差分：F0保存確率診断の待ちは解消。条件付きgrid-law差は全roundで数量化できたが、actual-law/残差range/価格精度・長roundモデル資格は未解消。ロードマップ本体は統合担当が更新する。

C consumerによる受入は未実施。D2はcommit/push・remote/公開成果物照合と本引渡しで **mandatory STOP**。結果を受けた追加取得・価格補正・別law解析は別指示。
