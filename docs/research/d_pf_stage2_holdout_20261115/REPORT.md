# 決定論PF Stage 2：m6独立holdout取得・開封・個別判定

**campaign：PASS。12 group中PASS=12、FAIL=0、UNKNOWN=0。mandatory STOP。** 予定12 wrapper、保存再利用0、新規compile12（Full6/p7 6）、取得・作用gate・実装guardは12/12完了。モデル判定と取得/実装の成功は別に扱う。

## 固定SOURCEと開封順

H4 linear 1.00 Å、STO-3G、保存tol-only DF10、Full PF（one-body＋fragment0..9）/B0I p7（one-body＋fragment0..6）。S2 16h/7、Y4 16h/3、M5 16hのm6、T=6.4、q=28/12/4、X/Yを個別に取得した。状態準備P0を価格から除外した測定付きfull controlled Hadamard wrapperのcompiled RZ数である。

C HANDOFFは `f0fc056464b63551d8f7795766bc0983d5863f95`。凍結正本は `12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`、SHA-256 `845c705082565a2b44d198bddb3034497620d991a00604454394ae5acda9651a`。Cのfit実行SOURCEは79c951b38bf9d8f1c80f28b2e7f82846d85c67e5、DのStage 1索引はb190bd815、判定policyは91365dcfのCOMPARISON_POLICYを保持した。

観測前にhash、各routeのStage 1 C_INPUTSを渡した既存 `pf_price_stages.frozen_models`、全12 groupのconfiguration/physical ID・actual幅・q・axis・CELL_PLAN selectorを確認した。[開封前記録](FREEZE_PREOPEN_V2.json)。新しいcheckerがCの価格式fieldとPF公式fieldを取り違えた最初のmetadata STOPは、科学call/観測読取0の段階で修正し、[経緯](PREOPEN_SCHEMA_CHECK_STOP.json)と空出力を保全した。正本・科学条件の不一致ではない。

判定前に全予定receiptを取得・公開・C guard束縛し、途中の価格誤差に応じて条件や取得対象を変えていない。元JSONのpredicted_RZの分子/分母をそのまま使用し、再fit・予測再計算は0。凍結JSONとStage 1の7ファイル/routeは取得前後で不変だった。

| route | 実行SOURCE | raw価格公開 | bound公開 |
|---|---|---|---|
| FULL_PF | `e276ab70fdd6ff988928703ec79fd3a9aa8c1603` | `cd25fd4cb7c655021a50e1be52249fdcf97f7f42` | `f9755e73dacb6f9efbffd452f2e1ac550eed22ec` |
| TRUNCATED_PF | `f9755e73dacb6f9efbffd452f2e1ac550eed22ec` | `c2f6f5078f5d1bb5dea1a2f286ef8eb9b0570d40` | `b2aef9faff475e65da40fef9764d21ce35386c7c` |

厳密判定SOURCE：`f8f8c5db0b78018baa0181a9d5e051b7f01489cc`、新helper `scripts/pf_m6_holdout_evidence.py`。判定codeはe276ab70fのSOURCEで観測前に固定済みで、実行時のcode byteも同じ。元stage runner/native/wrapper/compilerはStage 1から無変更。

## 12個別結果

相対誤差は `abs(predicted_RZ-observed_RZ)/observed_RZ`。PASSは厳密に `<=1/50`、campaign成功は12個別PASSのみ。下表は丸め前の有理数であり、表示値やX/Y平均を判定に使っていない。2%超過はFAIL、receipt取得・SOURCE・実装不明はUNKNOWNと区別する。

| model_group | m6 physical ID | axis | q | 凍結予測RZ | 観測RZ | 相対誤差（exact） | 判定 | 元receipt |
|---|---|---|---:|---:|---:|---|---|---|
| FULL_PF\|2nd\|16/7\|X | 4f3984c6d381fa0ca848 | X | 28 | 218455 | 218455 | 0 | PASS | [PRICES[0]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9755e73dacb6f9efbffd452f2e1ac550eed22ec/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/PRICES.json) |
| FULL_PF\|2nd\|16/7\|Y | 4f3984c6d381fa0ca848 | Y | 28 | 218455 | 218455 | 0 | PASS | [PRICES[1]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9755e73dacb6f9efbffd452f2e1ac550eed22ec/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/PRICES.json) |
| FULL_PF\|4th(m5_best)\|16\|X | 8db0bd3c1a5e1eeb541f | X | 4 | 342919 | 342919 | 0 | PASS | [PRICES[4]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9755e73dacb6f9efbffd452f2e1ac550eed22ec/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/PRICES.json) |
| FULL_PF\|4th(m5_best)\|16\|Y | 8db0bd3c1a5e1eeb541f | Y | 4 | 342919 | 342919 | 0 | PASS | [PRICES[5]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9755e73dacb6f9efbffd452f2e1ac550eed22ec/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/PRICES.json) |
| FULL_PF\|4th\|16/3\|X | 7727a3fec25e033629f1 | X | 12 | 282727 | 282727 | 0 | PASS | [PRICES[2]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9755e73dacb6f9efbffd452f2e1ac550eed22ec/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/PRICES.json) |
| FULL_PF\|4th\|16/3\|Y | 7727a3fec25e033629f1 | Y | 12 | 282727 | 282727 | 0 | PASS | [PRICES[3]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9755e73dacb6f9efbffd452f2e1ac550eed22ec/docs/research/d_full_pf_shortlist_prices_20261114/holdout/run01/PRICES.json) |
| TRUNCATED_PF\|2nd\|16/7\|X | fb512f2d5762df64bfae | X | 28 | 154671 | 154671 | 0 | PASS | [PRICES[0]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2aef9faff475e65da40fef9764d21ce35386c7c/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/PRICES.json) |
| TRUNCATED_PF\|2nd\|16/7\|Y | fb512f2d5762df64bfae | Y | 28 | 154671 | 154671 | 0 | PASS | [PRICES[1]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2aef9faff475e65da40fef9764d21ce35386c7c/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/PRICES.json) |
| TRUNCATED_PF\|4th(m5_best)\|16\|X | 4bc967fe8dbc6f814146 | X | 4 | 242879 | 242879 | 0 | PASS | [PRICES[4]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2aef9faff475e65da40fef9764d21ce35386c7c/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/PRICES.json) |
| TRUNCATED_PF\|4th(m5_best)\|16\|Y | 4bc967fe8dbc6f814146 | Y | 4 | 242879 | 242879 | 0 | PASS | [PRICES[5]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2aef9faff475e65da40fef9764d21ce35386c7c/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/PRICES.json) |
| TRUNCATED_PF\|4th\|16/3\|X | 5e19b5ded7386572684f | X | 12 | 198775 | 198775 | 0 | PASS | [PRICES[2]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2aef9faff475e65da40fef9764d21ce35386c7c/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/PRICES.json) |
| TRUNCATED_PF\|4th\|16/3\|Y | 5e19b5ded7386572684f | Y | 12 | 198775 | 198775 | 0 | PASS | [PRICES[3]](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2aef9faff475e65da40fef9764d21ce35386c7c/docs/research/d_truncated_p7_t3_prices_20261114/holdout/run01/PRICES.json) |

機械可読の12行は [HOLDOUT.csv](run01/HOLDOUT.csv)、予測SOURCE・receipt selector・exact分子/分母・個別判定は [RESULT.json](run01/RESULT.json)。各元PRICESにはRZ/CXと各depth等6 metricsがあり、RESULTにはwrapper fingerprintと実作用gateを保存した。各軸を別々にcompileし、片軸で補完していない。

## 取得・実装・資源

§10.5の既存照合を継続し、固定table/catalog、旧trial/receipt、別名physical schedule、関連branch参照、当route出力、実行中jobを差分確認した。既知範囲の同等m6 receipt/重複候補は0、再利用0、新規12。未列挙archive/別サーバーまで網羅した主張ではない。[重複/input preflight](DUPLICATE_AND_INPUT_PREFLIGHT.json)。科学call直前にも既存runnerのduplicate/source/Stage1/frozen gateを通過した。

Hamiltonian/state/snapshotの実bytes・receipt、provider-root import、Qiskit1.3.0/NumPy1.26.4を再確認。native/wrapper SOURCEはaf284d853890155bacd19e61850c196bd1d5c660、provider c31ef424175c02379eafebe8fc6c7548c1d9aabe、compiler hash 1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b。rz/sx/x/cx、opt1、seed17、symmetric_directional、H46_LONG_TIME_NATIVE_COMPOSE、NOT_APPLICABLE_NO_RTEを維持した。

Full scalar=c_H、p7=c_H+s_tailをtotal Tで一回、fragment順・signed pieces・actual幅・measurementを既存経路で保持。全6 cellが各7 gate（low/high-control leakage、independent occupation vector、Hadamard X/Y、compiled vector X/Y）を通過。元のnative/測定1e-9、compiled-vector1e-7を緩和していない。保存状態に対する既存作用検査であり、全入力状態の新証明ではない。C guardのPASSは実装受入、上の2%判定とは別である。

| route | stage wall s | user / system CPU s | 最大worker peak RSS MiB | workers |
|---|---:|---:|---:|---:|
| FULL_PF | 663.902 | 1518.31 / 19.79 | 1140.92 | 3 |
| TRUNCATED_PF | 440.316 | 1064.82 / 25.58 | 955.46 | 3 |

stage wallはchild起動からrecord構成まで。GNU timeのwall/CPU/RSS/I/OはFULL_PF_time/P7_time、stdout/stderrはFULL_PF_log/P7_logに保存。RSSはprocess peakであり、同時物理メモリ総量ではない。route間を順序化し、route内は資源観測による3 cell workerを使用。恣意的な低い資源cap・共有設定変更・他process操作はない。科学retry=0。私有QPY/cache/cost/jobと、旧raw/Stage1/STOPを保全した。

## 引渡し・ロードマップへの影響

取得・開封・12個別判定まで完了した。統合担当へ、決定論PF価格の進捗を「C凍結12モデルのm6独立holdout：PASS、PASS/FAIL/UNKNOWN=12/0/0」として反映する根拠を渡す。ロードマップ本体は編集していない。

結論は今回の12モデルのm6点まで。m7〜m12、別幅/別compiler/lowering、全qのaffine性、統計的信頼保証やsystematic/covarianceの資格は追加していない。凍結正本のqualified_for_candidate=falseを変更せず、refit・条件変更・追加compile・MSE/N/G・winnerへ進まない。[HANDOFF](HANDOFF.md)。公開commitとremote SHA/tree/成果物の最終照合は最終チャットに記載する。

**mandatory STOP。**
