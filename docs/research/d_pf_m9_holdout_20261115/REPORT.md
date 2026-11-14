# 決定論PF m9長q検証：取得・独立判定結果

**campaign PASS。モデルPASS/FAIL/UNKNOWN＝12/0/0。全12軸で固定予測と観測compiled RZが完全一致し、exact相対誤差は0/1。mandatory STOP。**

取得・実装とモデル適合を分ける。予定12軸の保存再利用0、新規実compile12、有効直接価格12、作用gate完了12軸、C実装guard受入12軸。科学retry0、未取得0。X/Yは各軸を個別compileした。モデルが外れた場合も有効な直接価格を保全する判定経路だが、今回はFAILなし。

## 固定入力・観測前確認

linear H4/1.00 Å/STO-3G、保存tol-only DF10、Full10（一体項＋fragment0..9）又はB0I p7（一体項＋fragment0..6）。全点m9/T=51.2、qはS2 224、標準Yoshida4 96、M5 32。actual幅は .2285714285714286 / .5333333333333333 / 1.6。P0除外、測定付きfull controlled Hadamard wrapperの全compiled RZ数である。

- C HANDOFF：`04a68f5932f53377595b2d69bbf8f46e02e82a01:docs/research/pf_m9_prediction_freeze_20261115/HANDOFF_D.md`。
- 固定予測：`b1a7cd358daeab51ce538c747d0cdb8ad86c97d4:docs/research/pf_m9_prediction_freeze_20261115/run01/PREDICTIONS.json`。SHA-256 `e5e00558669fd9568d40fcc2deba5e157a2c21c5505c5bb3268ea1a185701fc5`。各行の元numerator/denominatorをそのまま使い、予測再計算・refitは0。
- m9限定policy：`375f026a30e3c1ed3179a73cbbe333578abff00f:docs/research/pf_m9_prediction_freeze_20261115/M9_POLICY.json`。`abs(predicted-observed)/observed <= 1/50`、有効な正の観測を分母とし、丸め前の有理数で判定。取得/出所/実装不成立・欠測・分母未定義はUNKNOWN、2%超過はFAIL。
- 元係数：`12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`。m6専用reader/Stage 1/2の契約・予測・原結果は無変更。
- 物理点・計画：`bcb67b2b2bb78776db1dd8d0625a3facb37cd298:docs/research/d_pf_allround_price_coverage_plan_20261115/NEXT_VALIDATION_PLAN.md`。C alignment 0b4b2d94の行36/37,38/39,88/89,90/91,140/141,142/143を再利用した。

[PREOPEN.json](PREOPEN.json)で12行を(model_group_id,configuration_id,physical_id,axis,m)に一意対応させ、q/T/幅のbinary64、prefix、scalar、H/state/snapshot binding、fragment順、signed structureを確認した。CのPREOPEN_CHECK/PUBLICATION_PRECHECKとD coverageを引き継ぎ、[差分照合](DUPLICATE_AND_INPUT_PREFLIGHT.json)・[p7非重複照合](P7_FAST_PREFLIGHT.json)で保存receipt/alias/正確なtrial path/関連jobを確認。受入可能な同等観測0、不明重複0。未列挙archiveや別サーバーの網羅は主張しない。

## 実行SOURCE・公開locator

Fullとp7は入力・出力・実行SOURCEを分離した。Fullの開始後、利用者の並列化/高速化指示により、未開始p7だけを新SOURCEで加速して同時実行した。実行中Fullのファイル・process・raw/cacheを変更・停止・再実行していない。

| route | 実行SOURCE | raw公開SOURCE | bound公開SOURCE |
|---|---|---|---|
| FULL_PF | `c02e2416ad8ffd63b800a4d1039e8ec1e6cc75aa` | `2334b0d42fd5a7b00e756d251512445a97514352` | `2cf4a415a3d96dc8a018f5c4b74c65cd49adc496` |
| TRUNCATED_PF | `7ed93d24dbd2c9d49d8951549c976c9eb463d4e5` | `6b193982cf211b46d0de39cc09ba7efb18e5b2d2` | `fcb87d556da12c4f601008ae9c49e90513814a69` |

public namespaceはFull `docs/research/d_full_pf_shortlist_prices_20261114/m9_holdout/run01`、p7 `docs/research/d_truncated_p7_t3_prices_20261114/m9_holdout/run01`。各routeのRESULT/PRICES/SIDECARS_UNBOUND/IMPLEMENTATION_EVIDENCEをrawで固定し、その後SIDECARS/GUARD_CHECKを束縛した。C guardは実装受入であり、モデル判定ではない。

厳密判定の実行SOURCE：`2cf4a415a3d96dc8a018f5c4b74c65cd49adc496`、判定codeは観測前固定c02e2416aとbyte同一の `scripts/run_pf_m9_prices.py`（既存m6 exact comparatorを再利用）。[判定データの固定公開SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/389ae157371b267a819357da1212f3115f7350f9/docs/research/d_pf_m9_holdout_20261115/run01/RESULT.json) は `389ae157371b267a819357da1212f3115f7350f9`。

## 12個別結果

取得・実装欄は元7 gateとC guardを確認した直接観測。receiptのF/P7[index]は各固定PRICES.jsonの0-based配列selector。両軸の同価格は観測結果であり、片軸による補完ではない。

| model_group_id | physical ID | axis | q | 固定予測RZ n/d | 観測RZ | exact相対誤差 | 取得・実装 | モデル判定 | 元receipt |
|---|---|---|---:|---:|---:|---|---|---|---|
| FULL_PF\|2nd\|16/7\|X | a4ec53ff4e60d60a2930 | X | 224 | 1745491/1 | 1745491 | 0/1 | 有効 | PASS | [F[0]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2cf4a415a3d96dc8a018f5c4b74c65cd49adc496/docs/research/d_full_pf_shortlist_prices_20261114/m9_holdout/run01/PRICES.json) |
| FULL_PF\|2nd\|16/7\|Y | a4ec53ff4e60d60a2930 | Y | 224 | 1745491/1 | 1745491 | 0/1 | 有効 | PASS | [F[1]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2cf4a415a3d96dc8a018f5c4b74c65cd49adc496/docs/research/d_full_pf_shortlist_prices_20261114/m9_holdout/run01/PRICES.json) |
| TRUNCATED_PF\|2nd\|16/7\|X | 1bd0a5dd4bbf876da1d6 | X | 224 | 1235219/1 | 1235219 | 0/1 | 有効 | PASS | [P7[0]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcb87d556da12c4f601008ae9c49e90513814a69/docs/research/d_truncated_p7_t3_prices_20261114/m9_holdout/run01/PRICES.json) |
| TRUNCATED_PF\|2nd\|16/7\|Y | 1bd0a5dd4bbf876da1d6 | Y | 224 | 1235219/1 | 1235219 | 0/1 | 有効 | PASS | [P7[1]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcb87d556da12c4f601008ae9c49e90513814a69/docs/research/d_truncated_p7_t3_prices_20261114/m9_holdout/run01/PRICES.json) |
| FULL_PF\|4th\|16/3\|X | b2c479e7805c4d1f0b30 | X | 96 | 2259667/1 | 2259667 | 0/1 | 有効 | PASS | [F[2]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2cf4a415a3d96dc8a018f5c4b74c65cd49adc496/docs/research/d_full_pf_shortlist_prices_20261114/m9_holdout/run01/PRICES.json) |
| FULL_PF\|4th\|16/3\|Y | b2c479e7805c4d1f0b30 | Y | 96 | 2259667/1 | 2259667 | 0/1 | 有効 | PASS | [F[3]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2cf4a415a3d96dc8a018f5c4b74c65cd49adc496/docs/research/d_full_pf_shortlist_prices_20261114/m9_holdout/run01/PRICES.json) |
| TRUNCATED_PF\|4th\|16/3\|X | 9924c226ee246283ec54 | X | 96 | 1588051/1 | 1588051 | 0/1 | 有効 | PASS | [P7[2]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcb87d556da12c4f601008ae9c49e90513814a69/docs/research/d_truncated_p7_t3_prices_20261114/m9_holdout/run01/PRICES.json) |
| TRUNCATED_PF\|4th\|16/3\|Y | 9924c226ee246283ec54 | Y | 96 | 1588051/1 | 1588051 | 0/1 | 有効 | PASS | [P7[3]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcb87d556da12c4f601008ae9c49e90513814a69/docs/research/d_truncated_p7_t3_prices_20261114/m9_holdout/run01/PRICES.json) |
| FULL_PF\|4th(m5_best)\|16\|X | a53547a5fb707b9550d5 | X | 32 | 2741203/1 | 2741203 | 0/1 | 有効 | PASS | [F[4]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2cf4a415a3d96dc8a018f5c4b74c65cd49adc496/docs/research/d_full_pf_shortlist_prices_20261114/m9_holdout/run01/PRICES.json) |
| FULL_PF\|4th(m5_best)\|16\|Y | a53547a5fb707b9550d5 | Y | 32 | 2741203/1 | 2741203 | 0/1 | 有効 | PASS | [F[5]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2cf4a415a3d96dc8a018f5c4b74c65cd49adc496/docs/research/d_full_pf_shortlist_prices_20261114/m9_holdout/run01/PRICES.json) |
| TRUNCATED_PF\|4th(m5_best)\|16\|X | 60111109e0024d1a1064 | X | 32 | 1940883/1 | 1940883 | 0/1 | 有効 | PASS | [P7[4]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcb87d556da12c4f601008ae9c49e90513814a69/docs/research/d_truncated_p7_t3_prices_20261114/m9_holdout/run01/PRICES.json) |
| TRUNCATED_PF\|4th(m5_best)\|16\|Y | 60111109e0024d1a1064 | Y | 32 | 1940883/1 | 1940883 | 0/1 | 有効 | PASS | [P7[5]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcb87d556da12c4f601008ae9c49e90513814a69/docs/research/d_truncated_p7_t3_prices_20261114/m9_holdout/run01/PRICES.json) |

機械可読表：[HOLDOUT.csv](run01/HOLDOUT.csv)、exact値・元予測selector・元receipt：[RESULT.json](run01/RESULT.json)。有効直接価格の件数とモデルPASS件数は別field。全12個別PASSなのでm9 campaign成功である。2%規則は統計的95%信頼保証ではない。

## 実装・作用gateと保全

Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、symmetric-directional、full_Hadamard_with_measurementを維持。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。native/compose/wrapperはaf284d853890155bacd19e61850c196bd1d5c660、provider c31ef424175c02379eafebe8fc6c7548c1d9aabe。`H46_LONG_TIME_NATIVE_COMPOSE / NOT_APPLICABLE_NO_RTE`で、RTE用旧/新ZZ loweringの価格を混在させていない。

Full scalar=2.29310124732、p7 scalar=2.293146549384713。元規則のc_H又はc_H＋tail identityをtotal Tで一回適用し、C固定scalarとのhex完全一致をbuild前に確認した。fragmentは保存生成順、signed piece・制御・測定・X/Y符号を元処理で保全。

全6 cellでlow-control、high-control leakage、independent occupation vector、Hadamard X/Y、compiled-vector X/Yの各7 gateを完了。native/測定は元のstrict <1e-9、compiled-vectorはstrict <1e-7。保存入力状態に対する既存検査であり、全入力状態の新しい作用証明へは拡張しない。wrapper fingerprint・RZ/CX数・各depth・circuit sizeは各routeのPRICES/RESULTとprivate cache/QPYに保存。compile cache hitは両route全軸0、全12軸を新規compileした。

C凍結JSON、Stage 1較正、Stage 2 m6 holdoutのbytesを取得前後で確認し不変。旧SOURCE/input/raw/grant/ledger/marker/KEEP/STOP、今回private QPY/cache/CHILD_STARTEDも保全した。sampling/event生成、予測再計算、fit、他roundへの価格外挿、N/G/winner計算は0。

## 実H4の実行資源と高速化

| route | stage wall s | user / system CPU s | 最大worker peak RSS MiB | workers |
|---|---:|---:|---:|---:|
| FULL_PF | 5300.597 | 12048.54 / 84.32 | 6439.31 | 3 |
| TRUNCATED_PF | 1544.982 | 3689.67 / 66.36 | 4841.39 | 3 |

wallはchild起動からrecord作成まで。GNU timeの外側wall/CPU/RSS/I/OはFULL_PF_time/P7_time、ログは同prefixの_logに保存。RSSは最大process peakであり、同時メモリ合計ではない。最大6 workerが重なり、p7はFull処理中に完了した。affinity64、利用可能RAM約508～517GBを観測し、恣意的な恒常cap、共有設定変更、他process操作はない。

[ACCELERATION.md](ACCELERATION.md)の高速fingerprintと同一signed S2 template再利用をp7の実入口へ適用した。実p7 composeはS2 17.145 s、Y4 22.545 s、M5 33.305 s。builder呼出し/再利用/実生成は224/223/1、288/286/2、352/346/6。詳細はRESULTのtrial.technical_accelerationに残した。人工4-modeで旧/新のcanonical fingerprint・compiled全6指標・作用一致を確認済み（初期metadata/旧回帰28、加速test4）。

Fullとp7はprefixが違うため、wall比を実H4の同条件高速化倍率としない。同条件の新旧H4比較は今回追加しておらず、倍率は未測定。主要段階の観測は元Fullで粗かった。CPUの生存だけを正常完了やgate PASSとせず、保存trial/guardで確定した。今後の新規SOURCEでは開始前に段階計測を用意する。[再利用索引](../../PERFORMANCE_METHODS.md)。

## HANDOFF・ロードマップへの影響

- **直接価格側**：新規12軸の有効直接価格が増えた（Full6/p7 6、再利用0）。
- **モデル側**：指定m9点の12軸で適合、全予測と観測RZが完全一致。m6の証拠を再取得・再判定したものではない。
- 未資格：m7/m8、m10～m12、別幅・別compiler/lowering、全qのaffine性・誤差上界、総G/winner。元モデルの全candidate資格を自動昇格させない。
- 統合担当には上の二つを分けて反映する根拠を渡す。ロードマップ本体は変更しない。

[HANDOFF](HANDOFF.md)の固定locatorでCへ引き渡す。**今回の結論は指定m9の12軸まで。commit/push・remote照合後mandatory STOP。**
