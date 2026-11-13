# Full-H κ2 P-1（m3、新controlled ZZ）run01：実行前段でSTOP

2026-11-14 JST。利用者が承認したP-1を固定callerで1回実行したが、**状態は`STOP_WIRING_CHECK`（終了コード2）である。**

**原因**：実行に使ったworktreeのcheckoutに、tracked file `references/h46_m5_tol_pilot_inputs_v1.json`が含まれていなかった。

- 私はsparse checkoutを使い、top-levelの`references/`を含めていなかった。
- このfileは、元Full-Hの`context()`がimportする`h46_m5_tol_contract`が、import時に読み込む。
- 子processはimportの時点で`FileNotFoundError`になった。保存H4入力の復元より前である。

**科学計算は行われていない。** sampling、compose、compile、statevector検証はいずれも0件。

固定計画と指示に従い、**retryせずにmandatory STOPとした。** 再実行には新たな判断が必要である（§4）。

- 実行したのは主agent（Claude Code）で、独立した査読ではない。
- 実行記録：[EXECUTION.txt](EXECUTION.txt)（tests、preflight、終了コード、資源、子processのtraceback）。
- callerの出力：[run01/SOURCE.json](run01/SOURCE.json)、[run01/P1_RESULT.json](run01/P1_RESULT.json)。無変更で保存。
- private：`/home/AbeHiromu/projects/d-full-h-kappa-20261114-private/P1-KAPPA2-M3/run01/job.json`。

## 1. 実行までに通過した確認

| 確認 | 結果 |
|---|---|
| callerのbyte | 固定SOURCE `04bead115`、Dの後続`f2a2c3afd`、実行tree（`ce6aacfa`、S-1結果を含むD準備系列）で同一（`24146d583a`）。layout helper・P-1 testも同一 |
| 人工・metadata test | P-1 parent 3件、P-1 circuit 4件（toyで子processのprovider root経路をend-to-end）、S-1 caller 7件：すべてPASS |
| metadata preflight | `PREFLIGHT_METADATA_ONLY`、PASS |
| preflightの確認内容 | native・wrapper・provider・native script 6本が固定blobとbyte一致（`af284d85`、`c31ef424`） |
| preflightの確認内容 | κ2計画のm3行と固定wiring cellが一致：`FULL_H_4de6130f0d3a66e5d695`、q8/R71/s568/K2、T=0.8、cohort `D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1`、lowering `CONTROLLED_ZZ_CX_CRZ_CX_V1` |
| preflightの確認内容 | 8 outer seedを元Full-H `trial_seed`（9b83）で再計算し、固定値と一致 |
| execute直前（親） | HEADとのbyte照合（caller、S-1 caller、integration、C module）、source checks、計画行とseedの再照合：通過 |
| 子process | `install()`を最初に実行 → provider root moduleのc31ef424 byte照合 → 元`setup()`：通過。その次の`context('H4')`内のimportで停止 |

**子processで停止した位置**

`pilot → full.context('H4') → from h46_m5_tol_native import restore → import h46_m5_tol_contract → read(ROOT/'references/h46_m5_tol_pilot_inputs_v1.json')` → `FileNotFoundError`。

**実行件数と資源**

| 項目 | 値 |
|---|---|
| 科学実行の試行 | 1回（STOP） |
| context・preparation | 0 |
| trajectory | 0 |
| compose | 0 |
| compile | 0 |
| retry | なし |
| 時間 | callerの計測2.94 s、wall 4.12 s |
| 最大RSS | 345 MB（子processを含む） |
| 環境 | h6 venv（Python 3.11.0rc1、Qiskit 1.3.0、NumPy 1.26.4、SciPy 1.14.1、rustworkx 0.18.1）。既存の新ZZ P1 run02のcompileと同じvenvである。S-1で使ったPython 3.12 venvも同じpackage版だった |

## 2. 原因の切り分け

- **科学的な不一致ではない。** SOURCE・ID・seed・scalarの不一致、作用検査の不成立、新ZZの無効化、数値gateの不成立はいずれも起きていない（それらの段階に到達していない）。
- **実行環境の欠落である。** 実行treeのcheckoutの不備（tracked fileの欠落）が原因である。
- **Dの実行準備に残っていた穴**：callerのmetadata preflightとtoy testは、子processが実入力の経路でimport時に読むworking-tree fileを確認していなかった。Dのworktreeも`references/`を含まないsparse構成なので、同じ設定なら同様に停止したと考えられる（推定）。
- **STOP後の技術確認（科学計算なし）**
  - 実行treeに`references/`（tracked、0.75 MB）をcheckoutした。
  - 子processと同じ順序で、`install()`の後に`context()`・`cost_trial`が使うmodule 12個をimportだけして確認した。
  - 12/12がimportでき、`h46_m5_contract`系のSOURCE_PATHSにも欠落はなかった。
  - 保存H4入力の復元、sampling、回路の作成はしていない。

## 3. 報告項目の状況

| # | 項目 | 状況 |
|---|---|---|
| 1 | 実H4 P-1の作用検証（index0の各gate） | **未取得**（到達前にSTOP） |
| 2 | X/YのcompiledのRZ・CX・depth等 | **未取得**（compile 0） |
| 3 | SOURCE・seed・physical IDの照合 | 親processと子processのprovider照合まで、すべてPASS（§1） |
| 4 | scalar、one-body、新ZZ loweringの検証 | **未取得** |
| 5 | sidecarとCへの価格HANDOFF | **未作成**（価格がないため。sidecar builderは実行されていない） |
| 6 | 実行件数・時間・peak RSS | §1 |
| 7 | 残るUNKNOWN | 上記1・2・4・5がすべてUNKNOWN。P-2には、少なくとも実H4での配線（作用検査PASSと新ZZの有効化）と、単一pathのdirect価格が必要で、どちらも未取得 |

## 4. 再実行に必要な判断（ユーザー）

run02は**新しい実行**である。今回の承認・retry禁止の範囲では行わない。

**必要な変更点**

1. **実行tree**：tracked fileが欠けない構成にする（full checkout、またはsparseに`references/`を含める）。後者の構成でimportの連鎖を確認済み。
2. **出力先**：固定callerは`OUTPUT=.../run01`、`PRIVATE_OUTPUT=.../run01`を`exist_ok=False`で作る。run01のSTOP記録は保全する必要があるので、run02には出力labelだけを変える最小のcaller修正（新SOURCE）が必要になる。
3. **推奨（任意）**：子processのimport preflightを追加する。`install()`の後に、実入力経路のmoduleをimportするだけの段階を置く。これにより、科学実行前に同種の環境欠落を検出できる（I1型の配線preflight）。

**変えないもの**：caller・native・provider・新ZZ・seed・cell・compile上限（2）。科学条件は変えない。

**判断を求める事項**：上記の技術修正（D担当、または本担当）と、run02の実行承認。

## 5. 保全

- run01のcaller出力（SOURCE、P1_RESULT）とprivateの`job.json`は、そのまま保存した。
- 固定SOURCE・input・raw・S-1結果・grant・ledger・marker・KEEP・STOP、D・B・Cのbranch・worktree・processは変更していない。
- **mandatory STOP**：P-2、S-2、N/MSE/Gへは進まない。
