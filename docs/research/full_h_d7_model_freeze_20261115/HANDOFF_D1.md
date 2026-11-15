# D1へ：Full-H κ2 D7の2モデル・m5/m6の4予測

**固定結果commit：`b5165f11bb63f9eb2a6bfe385ba75d8003b7e724`**。

固定path：`docs/research/full_h_d7_model_freeze_20261115/run01/FROZEN_MODELS.json`。

科学実行SOURCEは`f46fa120b4cddcaac3edcf21e3731a0d9020b2b8`。同SOURCEの`docs/research/full_h_d7_model_freeze_20261115/INPUTS.json`は、採用10 recordと元18軸のlocator・producer/validator・gate範囲を固定する。[REPORT](REPORT.md)／[D1_READER_CHECK](D1_READER_CHECK.json)／[TESTS](TESTS.txt)。

## 受入とfit

| 項目 | 結果 |
|---|---|
| 5較正cell | 全受入 |
| 軸別較正record | 10受入、0保留 |
| 元path/axis | 9 path/18軸を一度だけ受入 |
| X/Y rank | 3 / 3 |
| fit | 5行・3係数のexact有理数最小二乗、X/Y各1件 |
| 重み | 各cell1、path数による追加重みなし、X/Y poolなし |
| status / qualification | CALIBRATED_UNVALIDATED / qualified_for_candidate=false |

元G566/G4528/Q8R566/P2R566はD1 `7d0c3906…`、完成G36223は`f98cb963…`を採用。旧partialと再開時再利用4軸は追加計上していない。G36223のproducer p0=`d323fee1…`、p1=`887c3b9d…`と、最終validator=`8cce5406…`を別保存。P2R566は親G566 path0→path1の連結で、独立な第3親標本ではない。各cell index0の7数値gateだけを継承し、index1へ作用保証を拡張しない。

## D1 metadata readerからの読取

```python
reference = {
    'commit': 'b5165f11bb63f9eb2a6bfe385ba75d8003b7e724',
    'path': 'docs/research/full_h_d7_model_freeze_20261115/run01/FROZEN_MODELS.json',
}
frozen_model_gate(plan, reference)  # 元D1の固定D7 planを使用
```

元D1 `e628c0d6…:scripts/run_full_h_kappa2_p2.py`の無変更`frozen_model_gate`と元metadata依存関数（validate/proposal/physical_row/seed_table/P1.seed_function）で、**公開済みJSONの固定commit/pathを実際に読取済み**。GitHub公開bytesとGit固定blobも一致した。結果は`FROZEN_METADATA_CHECKED_NOT_MODEL_QUALIFICATION`。fit/predict/holdout価格readerの呼出しは0。この接続は科学launcherをimportせずに確認しており、D1 branch/worktreeを変更していない。

各モデルは`models.X`／`models.Y`。`PRICE_LINEAR_FEATURE_MODEL_V1`を各1件だけ収録し、features/domain/design_sourceと`calibration[*].source`のcohort・step_seedsを保持している。元PRICE_CALIBRATION_RECORDへの`input_locators`もモデル別にある。係数と較正絶対残差はexact分子・分母。

特徴量順は`one,event_count,outer_boundary_count`＝`1,qR,q−1`。モデル式は`a+b*s+c*(q−1)`。元design_sourceのA提案時statusは履歴として維持し、D1 gateに合わせて付け替えていない。今回の利用者指示/固定D7 planによる採用と、母lawモデル資格は別である。

## 予測4件の固定selector

| cell / axis | JSON selector | exact予測RZ | 後続path数 |
|---|---|---|---:|
| M5_ROUND X | predictions[0] | 27894546580642075/20115111656 | 2 |
| M6_ROUND X | predictions[1] | 111815248527103419/20115111656 | 1 |
| M5_ROUND Y | predictions[2] | 27894546580642075/20115111656 | 2 |
| M6_ROUND Y | predictions[3] | 111815248527103419/20115111656 | 1 |

M5はq32/R283/s9056/T3.2・physical `FULL_H_bc8aab34256b73b68025`。M6はq64/R566/s36224/T6.4・physical `FULL_H_4ec8d73571760a6830b8`。各予測に元cohort/step_seeds、physical、features、n_pathsを保存。丸め表示から再構成せず、`predicted_RZ.numerator/denominator`を使用する。追加fitや再予測は不要。

後続compileはm5各軸2path＋m6各軸1path＝6件。**今回のholdout取得/開封は0件**。q方向は較正最大8から32/64への未検証移送であり、sが近いことだけで価格資格を与えない。許容差・PASS/FAIL・他round予測は追加しない。

Full-Hの元one-body込みlaw、全scalarの総T一回、新ZZ、full controlled measured wrapper/P0、元compiler/control/profile/SOURCEを維持する。有限cohort平均、paired依存、sampler/ideal距離、rare、共有不確かさと母law E[C]・CIは未資格のまま。N/G/winnerはない。

ロードマップ差分：D7全較正のC受入・2fit・4予測固定待ちは解消。次はD1の独立m5/m6取得（別指示）と後続検証。モデル性能・母law/長時間/CI/N/G資格は未解消。本体は統合担当が更新する。

**mandatory STOP：公開JSONのD1読取確認、commit/push・remote照合と本HANDOFFで終了。holdout6 compileは別指示。**
