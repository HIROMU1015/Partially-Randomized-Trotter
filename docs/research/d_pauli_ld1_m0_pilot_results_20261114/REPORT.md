# Pauli LD1/m0 B0：入力bridge importでSTOP（compile 0）

2026-11-14 JST。**承認された固定callerを一度起動したが、実H4入力復元より前のimportで停止した。normal path/event、実preparation、native/full-wrapper build、新compileはすべて0。retryは行っていない。** Pauli/DF headの実接続・実scalar gate・価格は未確認のままである。

[固定callerの結果](run01/RESULT.json)／[実行SOURCE・scope](run01/SOURCE.json)／[traceback](EXECUTION.txt)／[技術preflight](PREFLIGHT.json)／[照合と停止箇所](CHECKS.json)／[実行資源](RESOURCE_TIME.txt)／[人工テスト](TESTS.txt)／[HANDOFF](HANDOFF.md)／[次回条件](NEXT_STEPS.md)。新しい結果専用branch `codex/d-pauli-ld1-m0-b0-results-20261114`を固定SOURCEから作成した。科学コード・他担当worktreeは変更していない。

## 固定条件と実行

| 項目 | 固定・照合した内容 |
|---|---|
| 実行SOURCE | **`62ae1bd56718287de5d958cf0ed6411f5b7395c9`**。実行時HEAD・caller/connection byte一致 |
| caller | `scripts/run_pauli_ld1_m0_pilot.py --execute-approved-pauli-b0` |
| C planner/guard | `c628e9e8447c11c984064b922cbce6c52dc72d48`。planner/D接続/guardの3 moduleが固定blobとbyte一致 |
| A全round signal / P0 | `714247619afb4714fd663eb15c0470ef32bdc7af` / `4220f3006d0d8d413e195ed948e250578513b557` |
| 物理条件 | Linear H4/1.00 Å/STO-3G、保存tol-only DF10、元H/state/snapshot、Pauli LD1、m0、T=.1/qPF1/R_outer3/K2、P0 |
| 元configuration | `d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27` |
| C canonical physical ID | `RMSE:62a9a6b68c2c88951f1055d1da7f9b6a89da94a91092c9488cf5fce503288e54` |
| 辞書 | 保存184項、SHA-256 `dbdde5a12791cd7b31b00d8ec6ae5074d56964b467389688163ece8bf186945c`。順序/係数/identityは既存selectorが照合 |
| normal cohort / index / seed | `D_PAULI_LD1_M0_WIRING_V1` / 0 / **14788159701244282917**。seedは未使用（sampling前STOP） |
| route / lowering | `PR_PAULI_TAIL` / `PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`。旧形式ID・DF ZZ・B新Z1＋T1/Q3構成へ置換していない |
| 予定scalar | c=2.29310124732、保存s_R=0.4177175120626939、予定total=2.7108187593826942。metadata上の値であり、実native抽出scalarとの一致は未実施 |
| compiler | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/target/coupling/layout/routing追加指定なし。full measured X/Y、状態準備費除外。**compiler呼出し0** |

実行環境はPython3.12.3、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0。既存専用venvを利用し、共有設定・ライブラリ・他者processを変更していない。CPU/RAM capを追加せず、providerの元installを使った。

metadata preflightはPASS。親の人工/metadata test **8件PASS（8.889秒）**、そのうち一件がprovider-root childで人工回路test **5件PASS**を確認している。fake compile counterだけを使い、実価格を取得しないテストである。保存P0 bit-orderの限定確認もこの許可されたpreflight/testに含まれる。人工PASSを実H4 context成功とはしていない。

## 失敗箇所と原因

```text
child
  -> native.native_context('H4', (1,))
  -> import h46_m5_tol_native.restore
  -> import h46_m5_tol_contract
  -> TARGETS = read(ROOT / 'references/h46_m5_tol_pilot_inputs_v1.json')
  -> FileNotFoundError
```

必要なJSONは固定SOURCE内に存在する（Git blob `59f21d1a853fcf10331c2dd2093c48f5d4e65a04`）。一方、新worktreeは既存のsparse checkout patternを継承し、`references/`を展開していなかった。`git ls-files -t`は当該fileに **S（skip-worktree）**を示し、実filesystemには存在しない。Git blobの内容は読取確認したが、失敗後のmaterialize・修正・再import/再実行はしていない。[停止時pattern](SPARSE_CHECKOUT_AT_STOP.txt)を保全した。

**実行準備で私が、このimport時の参照file依存を見落とした。** metadata preflightとtoy providerテストは実H4 restore契約のimportを通らないため、その欠測を検出しなかった。科学SOURCEの破損やscalar不一致が確認されたわけではなく、worktree materializationに起因する失敗である。missing fileを0や別SOURCEで補完していない。

固定callerはchild exit1を検出し、parent exit2、`STOP_PAULI_B0`とresult=nullを保存した。失敗を受けて入力を展開し直して続行することは、今回禁止されたretryになるため行わなかった。normal samplingへ到達していないので乱数pathは生成されず、unused compile枠も次回へ自動転用しない。

## 必須gateと価格の状態

| gate / 処理 | 状態 |
|---|---|
| SOURCE/canonical ID/辞書/seed/route/emitter metadata | preflight/人工テストPASS |
| 保存mode-0-MSB→little endian | 保存P0の限定preflight/testはPASS。**当該実runtimeのgateは未実施** |
| native DF head順・s_R一致・scalar c+s_R一回 | 実preparation前STOP、未実施 |
| low-control identity / high-control独立full-2^n oracle / leakage | 未実施 |
| X/Y符号・測定付きwrapper・compiled vector | 未実施 |
| normal path/event | **0 / 0**（承認上限1 path/3 event） |
| native evolution / wrapper / 新compile | **0 / 0 / 0**（compile承認上限2） |
| 価格・sidecar・implementation evidence | **未生成**。PRICES.json/SIDECARS.jsonは存在しない |

実C/S/D、RZ/CX/depth、fingerprint、非Clifford数は未取得/null。予定recordの`DIRECT_COMPILED_FINITE_SAMPLE / FINITE_COHORT_ONLY / n_paths=1 / law_expectation=false`は将来の成功pathに対するschemaで、今回はpathが0のためその分類の価格recordを発行していない。0価格や架空sidecarへ置換しない。

count0の根拠はtracebackでrestore関数のimport時に停止したことと、privateにjob.jsonだけがあり、event/QPY/cache/child resultがないこと。libraryの全callをinstrumentした記録ではない。実context関数のattemptは1、入力restore関数のbody・実prepare・pilot_coreは未到達。

## 資源・保全・STOP

parentが計ったchild呼出し区間は **2.5981824488844723秒**。GNU timeによる全command wallは **6.72秒**、max RSSは **604,088 KiB**。これは失敗した起動処理の観測で、回路build/compile性能や長時間価格の負荷ではない。個別compile時間も未取得。

公開run01のSOURCE/RESULTとtraceback、time原記録、preflight/環境、private `/home/AbeHiromu/projects/d-pauli-ld1-m0-20261114-private/run01/job.json`を保全し、SHA/bytesをCHECKSに保存した。旧成果・SOURCE/input/raw/cache/grant/ledger/marker/KEEP/mandatory STOPは変更していない。

科学コード修正・新mean/CI/fit/N/G・B1は0。結果をcommit/pushしremote SHA/tree/主要成果物を照合後 **mandatory STOP**。復旧準備や新しい一回実行は、[NEXT_STEPS](NEXT_STEPS.md)の条件をユーザーが確認して別途判断する。
