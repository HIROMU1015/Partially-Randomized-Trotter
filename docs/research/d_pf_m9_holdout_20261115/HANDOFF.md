# C/B/A向けHANDOFF：m9限定の直接価格と凍結モデル判定

**m9 campaign PASS。再利用0、新規実compile12、有効直接価格12。モデルPASS/FAIL/UNKNOWN=12/0/0、全12予測と観測RZが完全一致（相対誤差0/1）。mandatory STOP。** 取得・作用/実装の成立とモデル適合を別に扱う。

対象はFull10/B0I p7 × S2/Y4/M5 × m9/T51.2 × X/Y。q=224/96/32、P0除外、測定付きfull controlled wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。各軸個別compile、cacheによる片軸補完0。[12行表・資源・制約](REPORT.md)。

## 固定受渡し

- C予測：`b1a7cd358daeab51ce538c747d0cdb8ad86c97d4:docs/research/pf_m9_prediction_freeze_20261115/run01/PREDICTIONS.json`、SHA e5e00558669fd9568d40fcc2deba5e157a2c21c5505c5bb3268ea1a185701fc5。
- 規則：`375f026a30e3c1ed3179a73cbbe333578abff00f:docs/research/pf_m9_prediction_freeze_20261115/M9_POLICY.json`。元分子/分母を使い、observed分母のexact相対誤差<=1/50。UNKNOWN/FAILを欠測補完やpoolで隠さない。
- 判定実行SOURCE：`2cf4a415a3d96dc8a018f5c4b74c65cd49adc496`。codeは観測前固定のm9 reader/exact comparatorと同一。fit/予測再計算0。
- 判定データSOURCE：`389ae157371b267a819357da1212f3115f7350f9:docs/research/d_pf_m9_holdout_20261115/run01/RESULT.json`（rowsの12行）。同SOURCEの`run01/HOLDOUT.csv`と`PUBLICATIONS.json`も利用できる。

| route | 実行SOURCE | raw公開SOURCE | bound公開SOURCE |
|---|---|---|---|
| FULL_PF | `c02e2416ad8ffd63b800a4d1039e8ec1e6cc75aa` | `2334b0d42fd5a7b00e756d251512445a97514352` | `2cf4a415a3d96dc8a018f5c4b74c65cd49adc496` |
| TRUNCATED_PF | `7ed93d24dbd2c9d49d8951549c976c9eb463d4e5` | `6b193982cf211b46d0de39cc09ba7efb18e5b2d2` | `fcb87d556da12c4f601008ae9c49e90513814a69` |

各routeのpathはFull `docs/research/d_full_pf_shortlist_prices_20261114/m9_holdout/run01`、p7 `docs/research/d_truncated_p7_t3_prices_20261114/m9_holdout/run01`。bound SOURCEのPRICES/SIDECARS/GUARD_CHECKと、raw SOURCEのRESULT/IMPLEMENTATION_EVIDENCEへ対応する。各行のreceiptに配列selector/axis/physical ID/実行SOURCEを保存した。

Cは元prediction rowsの `(model_group_id, configuration_id, physical_id, axis, m)` を維持して接続する。価格classificationはDIRECT_COMPILED_DETERMINISTIC、expectationはDETERMINISTIC。lowering/native/scalar/control/measurementの根拠はsidecarと固定実装証拠、実作用はtrialの7 gate。compiler hashだけで同一性を判断しない。

Fullはc02e2416aのまま完了。未開始p7だけを7ed93d24dで加速し、Fullと非重複の3 cellを並列化した。構築・hash・template再利用の技術SOURCEをprice_source.accelerationへ明示し、compiler/回路の物理quantityは保全した。p7は約1545 s、Fullは約5301 sだがprefixが異なり、同条件の速度倍率ではない。

## 引き継ぐ主張・残るUNKNOWN

直接観測12軸が増えたことと、12モデルがm9点でPASSしたことは別の成果。m9の2%は統計的95%保証ではない。保存状態に対する元作用gateの結果を全入力状態の新証明にしない。

元FROZEN_MODELSやm6の判定・SOURCEを更新しない。m7/m8/m10～m12、短round別幅、一般qへのaffine性/誤差上界、N/G/winnerは未資格・未評価のまま。BのNとの積算やCのモデル全domain資格付与へ自動進行しない。ロードマップへの反映は統合担当へ渡し、本体は編集しない。

新しい科学実行は別途明示承認を要する。旧raw/STOP、今回private cache/QPY/markerを保全し、**mandatory STOP**。
