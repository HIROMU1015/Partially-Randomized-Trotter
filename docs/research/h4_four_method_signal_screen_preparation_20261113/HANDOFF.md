# A → B：4方式限定signal screenの接続

2026-11-13 JST。B51500b7の11候補を増減せず、[PLAN.json](PLAN.json)へ新20セルと保存40セルを固定。[SCREEN_PLAN](SCREEN_PLAN.md)は未承認の科学実行案。今回実H4のsignal/N/RMSE再解析・simulation・C/Gは0。

## physical identityとV2の扱い

- 新20セルは`label/m/candidate/structure/reference`を持つ。H/state/snapshot・profileはplan共通値。PFはmethod FULL/prefix10、truncatedはB0I/prefix7、Full-H RTEはlegacy candidate内method PRでも**comparison_method=FULL_H_RTE**、family FULL_H_PURE_RTE・no-head・one-body randomizedを明示する。
- Full-Hのq/R/K・allocation・preparation hash・κなしを価格側へそのまま渡す。event数だけでq/R groupingを省略しない。旧q/R価格、PR tail λ、DF B3で代替しない。
- Full PF12＋Full-H6はD e7dcaの確定physical_scheduleによる全field・ID一致を技術テスト済み。B0I2は元cid/signed scheduleと元scalar規則を使用。
- Bの11案はまだ正式V2 schedule群として採択されていないため、**新canonical configuration/request IDをAで発行・偽装しない**。PLANのcanonical_configuration_idsはnull。Bが採択後のconfigurationを既存APIで生成し、physical tuple・全schedule・law・SOURCEで結ぶ。PR-KEEPの旧canonicalとSOURCEは保持する。
- Bの共通identity/planner/N統計器は再実装していない。旧HP型NをRMSE Nへ転用せず、PR nominalをbaselineへ押し付けない。

## 実行後の受渡し形式

承認されたexecuteの個別recordには次を供給する。

| field | 意味 |
|---|---|
| candidate / structure / label / m | 固定20セルと一致する物理recipe・signed順・R layout |
| signal.reference / raw / corrected | 同H/state/Tの保存reference。Full-H RTEは独立伝播raw/corrected、PFはB=1で同一 |
| signal.log_B / B / attenuation | 全interrogationの有限K normalization。PFはB1/logB0 |
| total_signed_bias / outer_PF_or_discard_bias / finite_RTE_bias | 適用可能なsignal空間分解。Full-Hのexact_tailは保存full-H reference、PFのfinite-RTE寄与は不使用の0 |
| local | bias絶対値、corrected/raw phaseとradius。zeroの場合phaseはnull |
| numerical / allowance | 旧gate・power境界norm・spectral residual・closure、R-aware empirical allowance、未観測exact-tail norm=null |
| reference_source / backend / source | 保存pairの出典、数値方式、今回の科学SOURCE。旧結果のSOURCEを変更しない |
| N_X/Y / global_RMSE_bound / scientific_PASS_FAIL | 全てnull。局所screenから全RMSE資格へ昇格しない |

Full-Hではpositive Bの補正を事後raw定義に使わず、group operatorの段階でnormalizationして独立にq乗する。qが非dyadicでも元のdt演算順を保つ。scalar/control/profileの価格同値性は、この平均値計算で新たに証明されるものではない。

`raw_corrected_independent_propagation`はFull-H RTEでtrue。PFではfalseとし、`raw_equals_corrected_at_B1=true`で同一になる理由を残す。

## 保存セル・比較の範囲

Full PF m3/m4の12、B0I m3/m4の2、PR/B0各13の合計40はlocatorと既存資格を保持し再計算しない。保存meanを同じtarget/referenceへ新たに数値比較・再診断する行為は別科学scope。最初の20cell許可を得る際にも、新しいsaved再解析を含めるなら対象と処理を明示する。

新recordは同じ保存full-H referenceへのbiasを既に返すため、m6/m12の長時間傾向を示す材料になる。Bは既存値をcopyする表示と新しい比較判定を区別し、独自の閾値で自動選抜しない。数値不安定・欠測を0又は方式の「敗北」として扱わない。

## 再現入口・次の担当

metadataだけの再現：

```bash
python3 -B scripts/prepare_h4_four_method_screen.py --out <fresh-PLAN.json>
python -B tests/check_h4_four_method_screen.py -v
```

実行後のB接続は、既存Full PF/Full-H request API・B0/B0I retained-H契約でphysical tupleをcanonicalへ結合し、A97146481の全事象loss/population/target/lawの不足を保持する。20screenの成功は全13round評価、同RMSEでの比較可能性、十分Nや最適候補の認定ではない。

**ユーザー**：固定20 mean＋Full-H6 normalizer・局所診断、新reference0の有限科学scopeを別途承認する。**A**：承認後のみ固定SOURCEで一括実行。**B**：結果を受けた次の有限候補案と同RMSE評価計画を整える。**C/D**：今回新価格作業の依頼はなく、DのP2を継続scopeのまま保全する。

準備成果のcommit/push・remote SHA/tree/主要成果物確認後、Aはmandatory STOP。全13roundや実科学実行へ自動進行しない。
