# 決定論PF m0：6候補・測定付きX/Yの直接価格

**12軸すべて有効。再利用0、新規実compile12、保留・欠測・UNKNOWN各0。元7作用gateとC implementation guardを全軸で通過した。mandatory STOP。** 今回はm0の直接価格取得であり、凍結モデルの予測・2%判定・再fitを行っていない。

| route / 公式 | m0 physical ID | X compiled RZ | Y compiled RZ | 軸別判定 |
|---|---|---:|---:|---|
| Full PF / S2 | `5669f142d693eee92583` | 8,090 | 8,090 | 有効 / 有効 |
| B0I p7 / S2 | `d2da7de820833a4c6334` | 5,820 | 5,820 | 有効 / 有効 |
| Full PF / 標準Yoshida4 | `03df026eb51a56241eef` | 23,656 | 23,656 | 有効 / 有効 |
| B0I p7 / 標準Yoshida4 | `6c100b1fcb67060431a8` | 16,846 | 16,846 | 有効 / 有効 |
| Full PF / M5 | `715f0d9f996ec0e62951` | 85,568 | 85,568 | 有効 / 有効 |
| B0I p7 / M5 | `fc389f45fb639698ae84` | 60,950 | 60,950 | 有効 / 有効 |

各X/Yは別の測定wrapperを個別compileした。価格が一致したことを根拠に片軸を補完していない。機械可読の[12軸JSON](run01/AXIS_PRICES.json) / [CSV](run01/AXIS_PRICES.csv) はconfiguration ID・physical ID・axis・T/q/実幅・取得区分・全6指標・fingerprint・gate・固定trial/receipt/SOURCEを持つ。C向けの正本は[HANDOFF](HANDOFF.md)。

公開metadataの再照合は `python3 -B docs/research/d_pf_m0_direct_prices_20261115/check_metadata.py`。12軸のalignment / trial / receipt / proof / guardと実行SOURCE、元凍結bytesを検査してPASSした（[出力](METADATA_CHECK.txt)）。科学関数・保存vector・回路の再生は行わない。

## 固定条件と元pipeline

保存linear H4 / 1.00 Å / STO-3G / tol-only DF10 / 元H・state・snapshot、m0、T=0.1、q=1、actual幅0.1（binary64 `0x1.999999999999ap-4`）、P0除外。Fullは一体項＋fragment0..9、B0I p7は一体項＋fragment0..6を保存順で保持する。C alignment `0b4b2d941cd669aaa45231e1236bc2d3158a9d12` の行0/1・2/3・52/53・54/55・104/105・106/107に一意に対応する。

元のsigned S2/Yoshida4/M5分解とsymmetric_directional controlled evolutionを保全した。Fullのscalarはc_H、p7はc_H＋discarded-tail identity。total Tへの一回適用を維持し、公開Stage 2に束縛したscalar/prep hashと実contextが完全一致した。実値はFull 2.29310124732、p7 2.293146549384713。数値をcallerへhard-codeせず、元公開recordを固定commitで読む。SOURCE・input bytes・fragment順は[実行時preflight](run01/PREFLIGHT.json)、[route別JOB_METADATA](../d_full_pf_shortlist_prices_20261114/m0_direct/run01/JOB_METADATA.json)を参照する。

比較quantityは測定を含むfull controlled Hadamard wrapper全体のcompiled RZ数。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。native `af284d853890155bacd19e61850c196bd1d5c660` のcompose / cost_trial、元wrapperとcompilerを変更していない。lowering識別は `H46_LONG_TIME_NATIVE_COMPOSE`、RTEは `NOT_APPLICABLE_NO_RTE`。RTEの旧/新controlled-ZZ価格やPauli価格を混入していない。

元index0のlow-control、high-control leakage、独立occupation作用、Hadamard X/Y、compiled-vector X/Yの7 gateを元許容差（native <1e-9、compiled <1e-7、位相合わせなし）で通過した。保存誤差の最大はFull native 4.64e-13 / compiled 3.70e-11、p7 native 1.48e-13 / compiled 7.26e-12。これは元保存状態に対する作用検査であり、全入力状態への新しい証明ではない。

## 重複照合と技術差分

[事前の照合・実行仕様](EXECUTION.md) と[重複preflight](DUPLICATE_PREFLIGHT.json)を引き継ぎ、実行入口で再確認した。5 carrier計746 record・Full catalog136候補をphysical metadataで照合し、同条件の保存価格0。旧schedule名・IDの違いだけで排除していない。旧PF commonのq1はT=.8、旧direct106表には対象Full10/p7 q1がなく、他rank/prefixの価格は不適合。正確な既知private trial pathに追加結果なし、関連running jobなし。既知関連remote3枝のSHAも確認した。未列挙archive・別サーバーまで不在を主張しない。

新caller [run_pf_m0_prices.py](../../../scripts/run_pf_m0_prices.py) はm0の12軸selectorと計測・process終了観測・公開接続だけを追加した。元Full/p7のtrial関数、Stage1/2、m9、scalar・compiler・wrapper・native・fingerprint helperは変更していない。人工テストは4件PASS（metadata誤結合、provider process分離、元7 gate・高速化と段階計測の実入口、異常終了時の部分保存とretryなし）。[保存test出力](TECHNICAL_TESTS.txt)。開発時にmetadataを先にimportするとnative provider解決が変わる問題を人工入力で検出・修正し、実科学前にSOURCEを固定した。

## 実H4の時間と資源

2 route各3 cell、計6 workerを同時投入した。開始時affinity64、観測idle CPU64、利用可能RAM524,540,059,648 byte。worker数は科学対象6 cellと実資源から決まり、恣意的な恒常上限を設定していない。各cellのevolutionはX/Yで共有する。既存 `accelerated_native` のsigned部品再利用とcanonical bytesを保つ高速fingerprintを元trial全体へ適用した。

| route / 公式 | cell wall s | compose s | X compile＋保存 s | Y compile＋保存 s | worker peak MiB |
|---|---:|---:|---:|---:|---:|
| Full / S2 | 14.335 | 3.151 | .739 | .714 | 394.48 |
| Full / Y4 | 30.457 | 5.820 | 2.218 | 2.182 | 435.25 |
| Full / M5 | 96.831 | 16.868 | 9.609 | 9.646 | 575.16 |
| p7 / S2 | 10.046 | 2.029 | .589 | .543 | 374.74 |
| p7 / Y4 | 21.597 | 4.329 | 1.650 | 1.615 | 408.57 |
| p7 / M5 | 69.229 | 11.114 | 7.560 | 7.645 | 499.99 |

取得parent wall **100.876 s**、route wall Full99.056 / p7 71.543 s、context各約2.13 s。公開・binding・報告の時間はこの取得wallに含めない。stage時間はnested/inclusiveであり、compile欄はarchive保存を含む。RSSはprocessごとのpeakで、同時aggregate peakではない。同条件の旧経路再実行はせず、高速化倍率は未測定。[TIMING.json](run01/TIMING.json) とrouteごとのprogress JSONLに全段階を保存した。2 route親・6 workerは終了code0、signalなし。cache-hit0、sampling0、scientific retry0、残存科学worker0。

## 公開・結論の範囲

- 今回実行SOURCE：`729fe5370f53cea218f26e32ee262fee10e748ab`。両routeとも同じ固定caller SOURCEを使い、元Full / p7関数と入力・出力namespaceを分離した。
- raw / trial / implementation proof：`a7d066caed56a0c7d8e64c617606f99d0fe4e634`。
- 軸別sidecar / C guard：`471f0c3fa06bc99fc969eb6a9e291fa15446c76e`。
- [PUBLICATIONS.json](PUBLICATIONS.json) と12行の固定locatorをCへ渡す。結果表・本報告は最終公開commitで固定する。

m0の有効な直接価格coverageは0→12軸、増分12。較正時と異なるactual幅の直接証拠を得たのであり、凍結モデルの資格を拡張していない。旧m6/m9結果・C凍結JSON・Stage1/2 raw・旧STOPは不変。m1/m2以降の取得、他roundへの価格移送、未観測価格予測、MSE/N/G/winnerは実行していない。ロードマップ本体は変更せず、差分をHANDOFFで統合担当へ渡す。**mandatory STOP。**
