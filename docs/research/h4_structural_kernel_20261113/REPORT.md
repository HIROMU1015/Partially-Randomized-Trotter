# H4 K0/K1：構成runnerと入力参照STOP（2026-11-13）

**run01はB configurationの参照先を誤ったため、物理K0開始前にUNKNOWNで停止した。K1は未実行で、実平均Sの4値は全て欠測。** 参照先を技術修正し、実際の固定metadataで照合testを通したが、指示§4のUNKNOWN時mandatory STOPを維持し、科学処理は再実行していない。これは閉状態方式の不成立を示す結果ではない。

正本：[K0 UNKNOWN](run01/K0.json)／[STOP](run01/STOP.json)／[実行log](EXECUTION.txt)／[技術修正](TECHNICAL_FIX.md)／[coverage](COVERAGE.json)／[HANDOFF](HANDOFF.md)／[再開案・未実行](RUN02_PLAN.json)。

## 固定条件・SOURCE

対象は前段と同じLinear H4/1 Å/STO-3G/保存tol-only DF10、保存H/state/snapshot、S2/L_D1/δ.1/K2/h.1/M12/P0、symmetric_directional・測定付きfull Hadamard X/Y。元fragment/basis/qubit順・scalar一回・元exactκを維持する。数値stateのground仮定は条件付きのままで、state/HやSCF/DFを新生成していない。

| 対象 | 固定値 |
|---|---|
| C基点 | `a637ed1b813dc3952339cb05f36beb79168ace23` |
| 成功した前段診断 | `eeccdf18ee839ccddb824afbc5c8d0aa8b52b7b6` |
| 前段縮約SOURCE | `2b3b0ee7ffea92de25a2c0c549d64f258c34a611` |
| 保存preparation hash | `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7` |
| 固定support | 324非identity component・9 basis（前段保存metadataを参照） |
| B最新 | `f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8` |
| B configuration ID | `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911` |
| 今回の初回実行SOURCE | `0f7d65d5f20de26c449b33c8da9d110188245808` |
| 技術修正版SOURCE | `19b95d8b3c88063d2dbae4e5bb47455b52b00903`。科学実行なし |

| m | physical candidate | T | qPF | R_outer | K0 | X/YのE[S] |
|---|---|---:|---:|---:|---|---|
| 0 | `53e00d8b07d1547d19a7` | .1 | 1 | 2 | 未開始・入力STOP | null / null |
| 6 | `2a8c6718220eec59d419` | 6.4 | 64 | 128 | 未開始・入力STOP | null / null |

κは[d458b5e4f固定PRICE_REQUESTS](../h4_s0_mean_price_plan_20261113/PRICE_REQUESTS.json)内の元exact文字列をそのまま参照し、近似値から新scheduleを生成していない。今回のK0開始前照合はmetadataの参照だけで、N/normalization/RMSE/Gの再解析を含まない。

## 実装済みの経路と科学的な未検証部分

新[h4_structural_kernel.py](../../../src/trottertracks/h4_structural_kernel.py)と[runner](../../../scripts/run_h4_structural_kernel.py)は、元Accounting・前段のclosure module・既存MeanKernelを再利用する。B controller/provider、元Accounting/MeanKernel/Streamingを変更していない。

- K0は、basis/role分岐を元SOURCEのASTから読み、有限wire状態のgraphを構成する設計。各local nativewordについて、元Accountingのliteral Z演算/整数reward/出口を照合し、entry/head/close/reverse/outer/scalar/XY/terminalを検査する。
- 元C1はe.to_dict()をJSON化する前に構造計数するため、tuple/list・matrix_sha256=Noneを含む元表現を保持する。前段のnative幾何metadataを、literal basis比較の表現として代用しない。元_make_eventとのmetadata照合を実装した。
- order2はphase-positive→product→product→rotationへ因数分解し、全324³ tuple/全event列の列挙を避ける。共有active basis・phase flagと正確なwire別Zを残し、wire間の独立性は仮定しない。
- 元binary64 Z加算/literal zeroは順序を変えず、local/terminalの整数parityは完全一致だけを許す。確率matrixは元nominal値を補正せず、数学的ideal weight比と浮動小数点伝播を方向丸め区間で囲む。既存mass tolerance=1e−10を維持し、longdouble probability powerも確認する設計。[実行前数値契約](NUMERICAL_CONTRACT.md)を固定した。
- 全K0 PASSとkernel配列の保存後に限り、K1がm0/m6×X/Yの4つのE[S]を評価する。K0 FAIL/UNKNOWNからK1を呼ばないtestがある。出力quantityはFULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNTで、compiled C recordを作らない。

以上は**実装と人工検証の結果**である。実物理のtransition/reward parity、全boundary、mass/精度、実E[S]は今回一つも確認できていない。前段の十分条件と実kernelのPASSを混同しない。KERNELS.npz/K1.jsonは存在しない。

## 初回失敗と修正の範囲

初回`check_inputs`は、Aの`docs/research/pr_rpe_s0_v2_connection_20261113/metadata01/V2_CONFIGURATION.json`をB最新commit f0c4bb55内から読もうとして失敗した。Bは成果を複製せず、[B公開intake](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8/docs/research/abc_d_v2_integration_20261114/PUBLISHED_INTAKE.json)から別commitの正本を参照している。

実際のB正本は`030f3f548a35d98a88e62e92a35f132e0243a293:docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json`、SHA256 `22a1bd1af8704006568c545eb8bcbce5bd476202ad59aaeaf79f50850815f943`。修正版はB intakeのSOURCE索引と元bytesを照合し、同canonical IDと重点2件の全identity/law/physical recipeを確認する。

このmetadata照合は修正後の追加testでPASSした。修正はlocatorのみで、科学条件・角度・law・数値閾値や価格意味を変更していない。初回SOURCEとrun01 UNKNOWN/STOP/logは不変。物理descriptor復元0、実kernel構成/物理parity0、実確率伝播0、K1 0値である。

## 技術テスト・保全

**新8method（人工7＋固定metadata locator1）と既存10methodの計18 PASS**。[人工7](TESTS.txt)／[修正metadata1](SOURCE_FIX_TEST.txt)／[既存10](REGRESSION.txt)。新7件には人工324 component replicaでの全K0/K1経路、Fraction基準との区間照合、元RTE metadata、nominal確率の非補正、UNKNOWN/FAIL時のK1非起動を含む。これは人工入力であり、実H4 K0の科学的PASSではない。今回の確認は作者自身のもので、独立科学レビューではない。

新trajectory sampling/full-path S/実平均S/回路build/compile/transpile/compiled C/fit/N/G/RMSEは全て0。直接価格8wrapper、H6、他11round、S0a、8次PFは実行していない。旧C1/C2・connected-cluster・固定SOURCE/input/raw/cohort/結果/grant/ledger/marker/KEEP/STOP、S0a run01失敗/run02成功、A/Bの作業領域/process/cacheを保全した。

## 再開条件とSTOP

[RUN02_PLAN](RUN02_PLAN.json)は技術修正版から同一1 preparation・2lawでK0→PASS時のみK1の4値を実施する未実行案。run01を上書きしない新出力先を指定した。**指示§4のUNKNOWN時mandatory STOPに従い、科学処理は再実行していない。** 再開にはこのSTOPを踏まえたユーザーの実行指示が必要で、承認証明や新grantは不要。

次は**ユーザー**が修正版の再実行を指示するか判断する。**Codex C**は指示後に同一scopeのK0を実施し、PASSの場合だけ追加承認なしでK1へ進む。UNKNOWN/FAILならK1なしでSTOPする。**Codex B**へは入力STOPと技術修正のmetadataだけを渡し、平均S/compiled Cとも欠測に保つ。

成果をcommit/pushし、remote SHA/tree/主要成果物照合後mandatory STOP。
