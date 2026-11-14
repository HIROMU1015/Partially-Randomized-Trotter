# PR-Pauli LD2 B1-CAL：5 cell・18実compileの取得と引渡し

2026-11-15 JST、Codex D2。利用者の較正取得指示により、技術修正公開後のSTOPを**LD2 CALIBRATION 5 cellだけ**再開した。予定18、再利用0、新規実compile18、PASS18、FAIL0、取得UNKNOWN0。8 fresh normal path＋S66からのpaired 1 pathで、X/Yは同一pathの別軸である。**C向け10 recordの公開receipt照合を完了し、ここでmandatory STOP**する。fit・係数凍結・独立holdout・期待価格・N/Gは行っていない。

[HANDOFF_C](HANDOFF_C.md)／[C_HANDOFF.json](C_HANDOFF.json)／[18軸個別価格CSV](OBSERVATIONS.csv)／[公開raw一覧](RAW_PUBLICATIONS.json)／[静的引渡し確認](DELIVERY_CHECK.json)。rawのRESULT/C_INPUTS/PRICES/COMPILE_RECEIPTS/IMPLEMENTATION_EVIDENCEと、公開後binding/checkは別ファイルである。

## 1. 固定条件・計画との差分

保存H4 linear 1.00 Å、STO-3G、tol-only DF10（rankは保存観測、再生成argumentではない）、LD2、one-body＋fragment0,1のDF head、保存Pauli tail184項、S2、step time0.1、K2、元normal sampler（PCG64/NumPy float64 CDF）を維持した。Pauli D2 lowering `PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、Qiskit1.3.0、basis `rz/sx/x/cx`、opt1、transpiler seed17、symmetric-directional、測定付きfull controlled Hadamard wrapper、P0・状態準備費除外。価格quantityは当該wrapperのcompiled RZ数である。

- 実装SOURCE：`13ab1acfaebe5f33958e1db67862f5c45725ab91`。本作業でcode変更0。実行HEADがraw公開で前進しても使用SOURCEは同実装とbyte一致することを各段階で固定・確認した。
- cell/physical ID/m=None/cohort/seedの正本：`043c7567451e8f502a1cf0f425d32efe0a50f5dc` の `docs/research/d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_METADATA.json`。`PROPOSED_V1`を含むcohort名は元識別子として保持した。今回採択されたのはCALIBRATION5 cellのみである。
- 技術HANDOFF：`355e7074df806b60a31160a51899ec15d6b54e33` の [HANDOFF](../d_pauli_b1_receipt_pair_repair_20261115/HANDOFF.md)。旧計画のseed再生成説明は履歴で、今回のpaired経路は固定13abの保存親event直接連結を使用した。
- 配線前提：LD2 B1-W `1e2639a1be46ebea618b15591fb1e886bc499d33`、LD1 B0 `00f74f86d16cc6878e1d4c5977eb9abd9ce9c076`。いずれもCAL標本0件、元raw/STOP保全。

[PREFLIGHT](PREFLIGHT.json)にH/state/snapshot、入力file hash、辞書、実行環境と元cell一覧を保存した。Hamiltonian `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`、state `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`、snapshot `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`。辞書SOURCEはP0 `4220f3006d0d8d413e195ed948e250578513b557`、辞書SHA-256 `dd986e52057aa01a330a83dd147316696dbae669dc8a51828ee2a54734016ea7`、λ_P=2.0550998769244275。c_H=2.29310124732 と保存s_R=0.2223759094107018を合わせたscalar=2.515477156730702を**各子の総Tで一回**だけ適用した。

| cell | q / R / T | path | 軸 | 役割 | 予定 / 再利用 / 実compile |
|---|---|---|---|---|---|
| S66 | 1 / 66 / 0.1 | 2 fresh | X/Y | CALIBRATION | 4 / 0 / 4 |
| S526 | 1 / 526 / 0.1 | 2 fresh | X/Y | CALIBRATION | 4 / 0 / 4 |
| S4205 | 1 / 4205 / 0.1 | 2 fresh | X/Y | CALIBRATION | 4 / 0 / 4 |
| Q8R66 | 8 / 66 / 0.8 | 2 fresh | X/Y | CALIBRATION | 4 / 0 / 4 |
| P2R66 | 2 / 66 / 0.2 | 1 paired (S66 p0→p1) | X/Y | CALIBRATION | 2 / 0 / 2 |

既存callerが完全designを検査するためJOBには元7-cell計画が残る。`plan.adopted=true`は選択したCALIBRATION job用の既存flagであり、元24件全体の採択ではない。各JOBはCAL5 cell限定、`holdout_adopted=false`、最大18を明示し、M5/M6 jobは作成・投入していない。

## 2. §10.5の取得前差分照合

固定計画の辞書/law/head/tail/scalar/physical/cohort/seed/lowering/wrapper/quantity照合と既存catalog/別ID同等性調査を再利用した。今回は計画以後の関連公開ref・同等receipt・local実行job・出力namespace・実行SOURCEの差分だけを確認した。確認した関連公開tipに同等CAL receiptなし、同一候補のlocal price jobなし、新出力と専用private namespaceは未使用だった。受入可能な再利用0、判断保留対象0。外部hostの非公開jobまで確認できたとは主張しない。

S66前の[PREFLIGHT](PREFLIGHT.json)、独立3 cell前の[INDEPENDENT_STAGE_PREFLIGHT](INDEPENDENT_STAGE_PREFLIGHT.json)、paired前の[PAIRED_PREFLIGHT](PAIRED_PREFLIGHT.json)を保存した。使用する入力は保存H/state/snapshotと辞書から復元し、caller内の入力・作用gateを実行した。既存26人工/toy PASSは[TEST_RESULTS](../d_pauli_b1_receipt_pair_repair_20261115/TEST_RESULTS.json)と保存logを再利用し、suite再起動0。旧SCOPE_DEVIATIONにある保存P0演算子の**独立数値再検査は0**。後述の今回実pathに必要なbit順・作用・compiled-vector gateとは区別する。

## 3. 全18個別compileの観測価格

| cell | path | 軸 | RZ count | RZ depth | CX count | CX depth | total depth | size | 状態 / receipt selector |
|---|---|---|---|---|---|---|---|---|---|
| S66 | 0 | X | 2643 | 814 | 1508 | 996 | 2132 | 5683 | PASS / `p0X` |
| S66 | 0 | Y | 2643 | 814 | 1508 | 996 | 2132 | 5683 | PASS / `p0Y` |
| S66 | 1 | X | 2641 | 807 | 1514 | 1004 | 2139 | 5683 | PASS / `p1X` |
| S66 | 1 | Y | 2641 | 807 | 1514 | 1004 | 2139 | 5683 | PASS / `p1Y` |
| S526 | 0 | X | 5522 | 2357 | 5558 | 4681 | 7460 | 13921 | PASS / `p0X` |
| S526 | 0 | Y | 5522 | 2357 | 5558 | 4681 | 7460 | 13921 | PASS / `p0Y` |
| S526 | 1 | X | 5523 | 2339 | 5502 | 4594 | 7360 | 13853 | PASS / `p1X` |
| S526 | 1 | Y | 5523 | 2339 | 5502 | 4594 | 7360 | 13853 | PASS / `p1Y` |
| S4205 | 0 | X | 28830 | 14760 | 38122 | 33958 | 50133 | 80240 | PASS / `p0X` |
| S4205 | 0 | Y | 28830 | 14760 | 38122 | 33958 | 50133 | 80240 | PASS / `p0Y` |
| S4205 | 1 | X | 28878 | 14827 | 38306 | 34373 | 50532 | 80490 | PASS / `p1X` |
| S4205 | 1 | Y | 28878 | 14827 | 38306 | 34373 | 50532 | 80490 | PASS / `p1Y` |
| Q8R66 | 0 | X | 18948 | 5992 | 11550 | 7864 | 16122 | 41512 | PASS / `p0X` |
| Q8R66 | 0 | Y | 18948 | 5992 | 11550 | 7864 | 16122 | 41512 | PASS / `p0Y` |
| Q8R66 | 1 | X | 18987 | 5970 | 11664 | 8012 | 16223 | 41683 | PASS / `p1X` |
| Q8R66 | 1 | Y | 18987 | 5970 | 11664 | 8012 | 16223 | 41683 | PASS / `p1Y` |
| P2R66 | 0 | X | 4977 | 1546 | 2942 | 1964 | 4118 | 10810 | PASS / `p0X` |
| P2R66 | 0 | Y | 4977 | 1546 | 2942 | 1964 | 4118 | 10810 | PASS / `p0Y` |

各行は個別にcompileした観測値で、全18件 `cache_reused=false`。今回X/Yの各指標は同pathで同値だったが、別のfingerprint・個別receiptを保存している。CSVにfingerprint、実行code SHA、元raw公開SHA、receipt path/selectorを収録した。全wrapperはsystem8＋ancilla1 qubit、測定付きである。

### C向け10較正record

| cell | 軸 | n_paths | 有限cohort RZ値 | 個別receipt |
|---|---|---|---|---|
| S66 | X | 2 | 2642 | `p0X`, `p1X` |
| S66 | Y | 2 | 2642 | `p0Y`, `p1Y` |
| S526 | X | 2 | 11045 / 2 | `p0X`, `p1X` |
| S526 | Y | 2 | 11045 / 2 | `p0Y`, `p1Y` |
| S4205 | X | 2 | 28854 | `p0X`, `p1X` |
| S4205 | Y | 2 | 28854 | `p0Y`, `p1Y` |
| Q8R66 | X | 2 | 37935 / 2 | `p0X`, `p1X` |
| Q8R66 | Y | 2 | 37935 / 2 | `p0Y`, `p1Y` |
| P2R66 | X | 1 | 4977 | `p0X` |
| P2R66 | Y | 1 | 4977 | `p0Y` |

これらは元callerが出力した有限cohort値を転記したもの。`PRICE_CALIBRATION_RECORD_V1`、kind `DIRECT_COMPILED_FINITE_SAMPLE`、quantity `FULL_WRAPPER_MEAN_COMPILED_PRICE`、expectation `FINITE_COHORT_ONLY`、law_expectation=falseのままである。raw `RESULT.mean/fit/N/G` はnullで、記録値は `C_INPUTS.json` にある有限cohortのRZ算術平均だけである。全lawのE[C]、統計CI、科学的なモデル資格へ昇格しない。

## 4. 実際に観測したeventとpairedの対応

| cell | path | order0 / order2 | 先頭3 component (順序保持) | 先頭label | D2 step phase |
|---|---|---|---|---|---|
| S66 | 0 | 66 / 0 | 47, 30, 12 | `IXYIYZZX` | [0.0] |
| S66 | 1 | 66 / 0 | 26, 16, 18 | `IXZXIYZY` | [0.0] |
| S526 | 0 | 526 / 0 | 6, 22, 15 | `IIXZZZXI` | [0.0] |
| S526 | 1 | 526 / 0 | 9, 40, 17 | `IYZZZYII` | [0.0] |
| S4205 | 0 | 4205 / 0 | 55, 65, 1 | `IYZIZYII` | [0.0] |
| S4205 | 1 | 4205 / 0 | 91, 0, 77 | `ZYZZZYII` | [0.0] |
| Q8R66 | 0 | 528 / 0 | 1, 7, 77 | `ZIIIZIII` | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| Q8R66 | 1 | 528 / 0 | 41, 112, 97 | `IIYZIZYI` | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| P2R66 | 0 | 132 / 0 | 47, 30, 12 | `IXYIYZZX` | [0.0, 0.0] |

fresh8 pathのeventは計10650、すべてorder0、event phase=(1,0)、D2 aggregate source phase=0。pairedはそのうちS66の132 eventを再利用した派生pathで、新たなdrawや独立標本ではない。表は先頭だけを表示し、**全順序列・Pauli label・符号・rotation係数・order/product・event phase・D2 token/phase owner/mapping**は各raw `RESULT.json → child.paths[i].result.event_metadata` に保存している。[EVENT_COUNTS](EVENT_COUNTS.json)には各元rawへの固定locatorがある。元RESULTにあるlaw probabilityはそのまま保持し、確率の再推定やorder2出現保証は行っていない。

**実LD2のorder2 product・非自明event/D2位相は未観測UNKNOWN**。order0の回転係数に負符号があることや、Hamiltonian scalarによる相対位相を、未観測のorder2/product位相と混同しない。toy order2/位相PASSは実LD2 coverageに繰り入れない。追加seed/path/強制sampling0。

P2R66の入力親はS66の公開raw `ff536ed2a6effc6c01899a7dcde76e5361448b85` の `run01/S66/RESULT.json`、selector `['child','paths',0]` → `['child','paths',1]`。両親の`execution_commit`は `355e7074df806b60a31160a51899ec15d6b54e33` で、親公開SHAと区別した。完全な4-field locatorは [JOB](JOBS/P2R66.json)／[PAIRED_PREFLIGHT](PAIRED_PREFLIGHT.json)／子RESULTに保存されている。

固定callerが親SOURCE/request/辞書/cohort/index/seed/順序/gate、保存eventのphaseとD2 metadataをbuild前に照合した。`sampling=SAVED_PARENT_EVENTS_NO_DRAW`、`paired_check.status=PAIRED_STEP_METADATA_IDENTICAL`、`scalar=ONCE_AT_COMPOSED_TOTAL_T` を保存。保存stepの`outer`だけを子0,1へ対応付け、event列とその他metadataは親p0→p1の直接連結に一致した。paired経路のsampler呼出し0、子T=.2へのscalar一回、子自身のX/Y wrapper個別2 compileを実施した。親2本と子の相関を維持し、独立標本を9本や18本へ増やさない。

## 5. 作用gateと公開readerの確認範囲

| 固定caller gate | 件数 | 保存誤差の最大値 | 固定tol | 結果 |
|---|---|---|---|---|
| low_control | 9 | 1.9447391e-12 | 1e-09 | 全PASS |
| high_control_leakage | 9 | 0 | 1e-09 | 全PASS |
| independent_full_space_vector | 9 | 1.977113e-12 | 1e-09 | 全PASS |
| Hadamard_cosine | 9 | 3.8682391e-12 | 1e-09 | 全PASS |
| Hadamard_sine | 9 | 8.5406682e-13 | 1e-09 | 全PASS |
| compiled_vector_cosine | 9 | 5.302282e-12 | 1e-07 | 全PASS |
| compiled_vector_sine | 9 | 5.3111139e-12 | 1e-07 | 全PASS |

全9 pathでbit_order.pass=true。固定caller内の保存stateに対するfull-space vector／control／scalar／Hadamard／compiled-vector確認で、全入力演算子の新認証や全law保証ではない。許容差は元callerのままで緩和0、各誤差は保存RESULTからの要約で再simulation0。

公開後に既存 `bind_calibration_records` → `read_calibration_records` を各cellに使用した。全5 cell×X/Yの10 record、構成18個別receiptが `CELL_RECEIPTS_CHECKED_NOT_SCIENTIFIC_QUALIFICATION`。cell、axis、path index/集合、元physical ID/m=None、辞書、cohort/seeds、metrics/fingerprint、作用gate、実装証拠、実行SOURCE、元raw公開locatorの対応を照合した。rawを上書きせず各cellの `C_INPUTS_PUBLISHED.json` / `SIDECARS_PUBLISHED.json` / `C_CELL_CHECK.json` に保存した。[C_HANDOFF](C_HANDOFF.json)に10 record、10 sidecar、18 receipt locatorをまとめた。C別worktreeの変更0。**D2 readerの整合PASSと、C自身のfit・モデル受入は別**である。

## 6. 実行SOURCEとraw公開SHA

| cell | 実行code commit（実HEAD） | 元raw公開commit |
|---|---|---|
| S66 | `355e7074df806b60a31160a51899ec15d6b54e33` | `ff536ed2a6effc6c01899a7dcde76e5361448b85` |
| S526 | `ff536ed2a6effc6c01899a7dcde76e5361448b85` | `74955a11ce955f49341183bd5aa933f6af707104` |
| S4205 | `ff536ed2a6effc6c01899a7dcde76e5361448b85` | `74955a11ce955f49341183bd5aa933f6af707104` |
| Q8R66 | `ff536ed2a6effc6c01899a7dcde76e5361448b85` | `74955a11ce955f49341183bd5aa933f6af707104` |
| P2R66 | `74955a11ce955f49341183bd5aa933f6af707104` | `2669ad11da9d116f2834ee49a9f4a5b0654d91ef` |

S66を保存・commit・pushしてからP2へ公開親locatorを渡した。独立3 cellは同一実行HEADで各一回起動し、S66親公開後に並行取得した。P2は独立3 cell raw公開後に一回起動した。全SOURCEは固定13ab実装とbyte一致し、親SOURCEも付け替えない。元raw30 file（5 cell×6 file）は各raw公開commitとのbyte一致を静的確認した。private QPY/cache原本は専用 `d-pauli-ld2-b1-cal-20261115-private/run01/<cell>` 以下に保持し、公開COMPILE_RECEIPTSに所在・hashを保存した。C入口は公開JSONを読めばよく、QPYの再compile/再simulationを要求しない。

実行commandは各 `RESOURCE_<cell>.txt` の先頭に保存した既存 `scripts/run_pauli_b1_prices.py --execute-approved-job JOBS/<cell>.json`。各cell job一attempt、無断retry/resume0、seed差替え0、別軸補完0。空の `EXECUTION_<cell>.txt` はexit0時のstdout/stderrで、結果はJSONに出力されている。

## 7. 資源実績・運用記録

| cell | wall (m:ss) | user / system (秒) | peak RSS (KiB) | exit / swap |
|---|---|---|---|---|
| S66 | 0:16.93 | 48.38 / 63.55 | 630528 | 0 / 0 |
| S526 | 0:38.69 | 91.40 / 293.30 | 636364 | 0 / 0 |
| S4205 | 2:34.53 | 597.71 / 1649.42 | 681972 | 0 / 0 |
| Q8R66 | 1:17.24 | 155.33 / 247.45 | 652560 | 0 / 0 |
| P2R66 | 0:16.18 | 43.60 / 50.26 | 637704 | 0 / 0 |

GNU time -vのcell単位実測。wall欄はm:ss（S4205=154.53秒）、import/context準備/gate/compileを含み、軸別compile秒ではない。S526/S4205/Q8R66は3 job並行のため、cell wallの和を全体経過時間としない。peak RSS最大は681972 KiB/単jobで、並行全体の同時peakは未計測。CPU affinity64、取得前MemAvailable約479 GiB、disk空き約459 GiB、Python専用venv・NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0。独自resource capや共有設定変更0。ログのCPU競合/context switchは観測として保持し、正常jobを中断していない。

metadata-only環境収集では任意のthreadpoolctl/NumPy旧config APIが利用できず、その補助診断を省略した。取得前の固定条件確認には影響なく、これによる科学起動0。引渡し段階では既存write helperの排他的file作成が既存RAW_PUBLICATIONS索引の更新を拒否したため、reader成功済の出力を再利用して同索引だけを通常更新した。raw/readerの失敗や科学retryではなく、新規科学投入は行っていない。

## 8. 完了件数・未確定事項・ロードマップへの影響

| 項目 | 予定 | 再利用 | 実施 | PASS | FAIL | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| real compile (axis単位) | 18 | 0 | 18 | 18 | 0 | 0 |
| cell取得 | 5 | 0 | 5 | 5 | 0 | 0 |
| C reader (record単位) | 10 | 0 | 10 | 10 | 0 | 0 |
| 固定作用/compiled-vector gate | 63 | 0 | 63 | 63 | 0 | 0 |

取得のUNKNOWN0と、未観測order2/product・非自明位相のcoverage UNKNOWNを区別する。実価格の有限cohort引渡しは完了した。モデルfit・係数凍結、独立m5/m6 holdout6 compile、全lawの期待価格/CI・N・G、長q外挿・5方式の総費用順位は未判定。Pauli用holdout許容差は未採択で、決定論PFの2%を流用しない。

固定[CAL_AFTER_W_PLAN](../d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_PLAN.md)の「技術接続不足→CAL保留」は13ab技術修正と今回の限定取得により、**LD2の較正入力10 recordをCへ渡せる状態**へ進んだ。LD1側の較正不足は残る。B0/W価格は標本に加えず、同時48件やV先行の前提も追加していない。次の依存順序は **較正取得（今回完了）→別途指示によるC fit・凍結→さらに別指示で独立holdout**。有限cohortのfit用入力が揃ったことから、全law期待価格の保証や統計区間が得られたとはしない。ロードマップ本体は統合担当が更新するため本作業では無変更。

旧SOURCE/raw、SCOPE_DEVIATION.md/JSON/log、PROPOSALS/JOB_TEMPLATES、CAL_AFTER_W_METADATAと過去STOPを保全した。新旧rawの差分と実装byte一致は [DELIVERY_CHECK](DELIVERY_CHECK.json) に記録。最終HANDOFFと索引を既存D2 branchへcommit/pushし、remote照合後にmandatory STOP。**LD1、B1-V、fit/凍結、holdout、E[C]/CI/N/Gへ進まない。**
