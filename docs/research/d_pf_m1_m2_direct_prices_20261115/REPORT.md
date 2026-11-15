# 決定論PF 6候補：m1/m2の測定付き直接価格

**対象24軸すべて有効。再利用0、新規実compile24、保留・欠測・UNKNOWN各0。元作用gateとC implementation guardを全軸で通過した。直接証拠coverageは72/156→96/156、増分24。mandatory STOP。** 価格モデルの予測・2%判定・fit・MSE/N/G・winnerは実施していない。

| round | 再利用 | 実compile | 有効 | 保留 | 欠測 / UNKNOWN | direct coverage増分 |
|---|---:|---:|---:|---:|---:|---:|
| m1 | 0 | 12 | 12 | 0 | 0 / 0 | +12 |
| m2 | 0 | 12 | 12 | 0 | 0 / 0 | +12 |
| 合計 | 0 | 24 | 24 | 0 | 0 / 0 | +24 |

全候補・両軸の正本は[AXIS_PRICES.json](run01/AXIS_PRICES.json) / [CSV](run01/AXIS_PRICES.csv)。m0形式を引き継ぎ、configuration/physical ID・m/axis・T/q/actual幅とbinary64 hex、取得区分、RZ・全6指標、wrapper fingerprint、gate状態、元trial・receipt・sidecar・SOURCEの固定locatorを保存した。C向けは[HANDOFF](HANDOFF.md)。

## 物理点と直接RZ

保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、元H/state/snapshot、P0除外。Fullは一体項＋fragment0..9、B0I p7は一体項＋fragment0..6を保存順で保持する。各公式の符号付き分解、symmetric_directional control、測定付きfull controlled Hadamard wrapperを元pipelineから再利用した。

| route / 公式 | m | physical ID | T | q | actual幅 | X RZ | Y RZ |
|---|---:|---|---:|---:|---:|---:|---:|
| Full / S2 | 1 | `eb745e6e4759d15f75bc` | .2 | 1 | .2 | 8,090 | 8,090 |
| p7 / S2 | 1 | `79a56d848e8320740165` | .2 | 1 | .2 | 5,820 | 5,820 |
| Full / S2 | 2 | `4433563c3a4ef7a61702` | .4 | 2 | .2 | 15,873 | 15,873 |
| p7 / S2 | 2 | `ae9d8bf4a8899fd30eab` | .4 | 2 | .2 | 11,333 | 11,333 |
| Full / 標準Yoshida4 | 1 | `ce89c91c95889ed4a657` | .2 | 1 | .2 | 23,680 | 23,680 |
| p7 / 標準Yoshida4 | 1 | `c841e095f0f12a2fcdbe` | .2 | 1 | .2 | 16,846 | 16,846 |
| Full / 標準Yoshida4 | 2 | `21951422c4c3cb249055` | .4 | 1 | .4 | 23,680 | 23,680 |
| p7 / 標準Yoshida4 | 2 | `9531e5eb957c0b00e08a` | .4 | 1 | .4 | 16,846 | 16,846 |
| Full / M5 | 1 | `67211ee2c1c1aa51f15f` | .2 | 1 | .2 | 85,600 | 85,600 |
| p7 / M5 | 1 | `20fffbb6bfbc62a79e22` | .2 | 1 | .2 | 60,950 | 60,950 |
| Full / M5 | 2 | `d7b8233d8219e2575ebb` | .4 | 1 | .4 | 85,920 | 85,920 |
| p7 / M5 | 2 | `6dcfa24120296b4b0266` | .4 | 1 | .4 | 60,950 | 60,950 |

X/Yを各軸で個別compileした結果であり、同数だった軸やroundを補完していない。同じactual幅のS2 m1/m2も、T・q・scalar位相が異なる別physical ID・別fingerprintを持つ。C alignment `0b4b2d941cd669aaa45231e1236bc2d3158a9d12` の6固定configurationと24行のselectorへ一意に対応した。

Fullのc_H、p7のc_H＋tail identityをそれぞれtotal Tに一回適用する元規則を保持した。scalar / prep hashは公開済みStage2固定recordから取得し、実contextにbinary64完全一致で照合した。callerへの数値hard-code・tolerance追加はない。実scalarはFull 2.29310124732、p7 2.293146549384713、fragment順とprep hashも一致した。[実行時preflight](run01/PREFLIGHT.json) とroute別JOB_METADATAに入力・scalar SOURCEを保存した。

価格はQiskit1.3.0 / rz,sx,x,cx / opt1 / seed17での測定付きwrapper全体のRZ数。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。native / wrapperは元`af284d853890155bacd19e61850c196bd1d5c660`、native policy `H46_LONG_TIME_NATIVE_COMPOSE`、RTE `NOT_APPLICABLE_NO_RTE`。RTE/Pauli/旧loweringの別quantityを混在させていない。

## 重複照合・技術変更・gate

m0公開c95353a44と既存全round coverageを起点に、5 carrier 746 record / Full catalog136候補、Stage1/2・m9・m0・今回namespace、既知private trial path、関連running jobをmetadataで確認した。旧ID・別schedule名もsystem/method/prefix/formula/T/q/actual幅/R/Kで検索し、対象への保存一致0、未解決候補0、同等の稼働job0。関連remote3枝もm0時点と一致した。旧PF commonのT=.8等の除外根拠は既存SOURCE監査を継承し、保存値の再計数・再fitをしていない。未列挙archive / 別サーバーは調査範囲外UNKNOWNであり、全履歴の不在証明ではない。[EXECUTION](EXECUTION.md)、[事前preflight](DUPLICATE_PREFLIGHT.json)。

既存`run_pf_m0_prices.py`に有限scope `(1,2)`、別namespace、m依存のC guard接続だけを追加した。既定m0の選択・物理点は人工回帰で保持を確認し、元実行SOURCE729fe5370と旧m0 raw/STOPは保存した。native / wrapper / compiler / Full/p7 trial / fast helper / 監視経路は変更していない。旧m0のSOURCE同一性checkerは元固定checkoutでの検査であり、新callerのSOURCEと取り違えない。

新3人工/mock test＋m0回帰4 testがPASS。固定24軸とscope拒否、T/q/actual幅/ID/axis/structure誤結合の拒否、同幅S2の別作用・fingerprint、元7 gateの実入口と観測を確認した。[新test](TECHNICAL_TESTS.txt) / [m0回帰](M0_REGRESSION.txt)。悪いstructureを既存処理がTypeErrorで拒否する場合のtest期待型だけを修正し、科学guardは緩和していない。

実H4ではlow-control、high-control leakage、独立occupation作用、Hadamard X/Y、compiled-vector X/Yの7 gateを全12 cellで通過した。許容差は元native <1e-9 / compiled <1e-7、位相合わせなし。記録された最大誤差はFull native4.58e-13 / compiled3.84e-11、p7 native1.53e-13 / compiled7.27e-12。保存状態に対する元作用検査の結果であり、全入力状態への新証明ではない。新空cacheで24 compile、cache-hit0、sampling0、科学retry0。C guardも24軸を個別に確認した。

公開metadataの再照合：`python3 -B docs/research/d_pf_m1_m2_direct_prices_20261115/check_metadata.py`。24軸のalignment / trial / receipt / proof / C guard、SOURCEと凍結bytesを検査してPASS（[出力](METADATA_CHECK.txt)）。科学関数・保存vector・回路を再生しない。

## 資源と性能

2 route各6 cell、計12 workerを同時投入した。開始時affinity64、観測idle CPU約63.84、available RAM525,167,755,264 byte。対象12 cellと実資源でworker数を決め、恣意的な恒常CPU/RAM上限や共有設定変更を行っていない。route contextをforkで共有し、各cellのevolutionをX/Yで共有する。元trial全体に既存fast fingerprint / 同一signed部品再利用を適用した。別cell間で科学的contextや角度を無条件共有しない。

| route | context s | route wall s | 最長cell s | 最大worker RSS MiB |
|---|---:|---:|---:|---:|
| Full | 2.074 | 99.738 | 97.460 | 576.74 |
| p7 | 2.247 | 73.782 | 71.387 | 503.76 |

取得parent wallは**102.512 s**。報告・binding・公開は別。最大のFull M5 m2はcompose16.445 s、X/Y compile＋保存9.908 / 9.846 s、cell97.460 s。stage時間はnested/inclusive、RSSはprocessごとのpeakで同時aggregateではない。同条件の旧経路を再実行していないため高速化倍率は未測定。[TIMING.json](run01/TIMING.json) と各routeのcontext / cell progress JSONLに全段階を保存した。2 route子processと12 workerは終了code0・signalなし。非同期stack読取や他processへの操作は行っていない。

## 固定成果と結論の範囲

- 実行SOURCE：`a3cbe70e4afb0948188403f48cf4086bb419d8bf`。
- raw / trial / implementation proof：`bf8a7f0428a7efd45c3888e288b3e2095f5e4c34`。
- sidecar / C guard：`eea0bfab986221040165f2cc9c5f81cc094023f1`。
- [PUBLICATIONS.json](PUBLICATIONS.json)、24行表、HANDOFFを最終公開commitで固定する。

同条件の直接証拠として6候補すべてのm0～m6が揃い、m9を含め96/156軸となった。これは全96軸の現在consumerへのbinding完了や正式G入力資格を一括認定した意味ではない。旧m3のmetadata binding等は別課題のまま。m7/m8/m10～m12の60軸、価格モデルの誤差や適用domain、MSE/N/G・winnerは今回資格付与・計算していない。ロードマップ本体は編集せず、HANDOFFで統合担当へ差分を渡す。旧SOURCE / raw / 凍結モデル / STOPを保全し、commit/push・remote照合後**mandatory STOP**。
