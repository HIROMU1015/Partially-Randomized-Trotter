# PR-DF LD1 m4：独立holdoutの記述的比較

2026-11-15 JST。**m4 fresh4 path、X/Y計8実compileと8 receiptの照合、凍結予測に対する2件のexact比較を完了した。終了状態は `DESCRIPTIVE_COMPLETE`。** 予定8、再利用0、新規compile8（試行8）、有効8、保留/欠測0、retry0。各軸4 pathで、X/Yをpoolしていない。

両軸とも観測平均O=129692.5 RZ、P−O≈−706.178399206563 RZ、符号付き相対誤差≈−0.544502110150211%。この指定cohortでは過小予測だった。`model_verdict=NOT_ASSESSED_NO_ADOPTED_TOLERANCE`、数値許容差null。誤差の大小によるモデルPASS/FAILは付けない。

[HANDOFF](HANDOFF_C.md)／[C入力（8 receiptと2比較）](C_HANDOFF.json)／[exact比較](COMPARISON.json)／[8行CSV](PATHS.csv)。

## SOURCE・公開段階

| 段階 | 固定参照 |
|---|---|
| 今回の科学実行SOURCE・INPUTS | `7bdfc6f629aa74a5c1347abf84ad6486c669c9d8` |
| 元P1/P2新ZZ・S/C pipeline | `39ddd59edf54a3a901e5302207b2765ddbd03fbb` |
| m4取得前seed/物理点/規則の正本 | `2f423befbba5075bad4ede3057f89edd788d2661:docs/research/d_pr_df_ld1_m3_results_20261115/INPUTS.json#future_holdout` |
| 凍結予測正本 | `3cac53da1a4a8b2228245593c74c1ff9ec1f5212:docs/research/pr_df_ld1_residual_freeze_20261115/run01/FROZEN_MODELS.json#models[axis].m4_prediction.predicted_RZ` |
| C HANDOFF | `890755f0aeea3abfcdd5c849422bb198666a477e:docs/research/pr_df_ld1_residual_freeze_20261115/HANDOFF_D2.md` |
| 元raw公開SHA | `3042e6e66ad693ed514274334adecd3d1b42dc02` |
| 最終公開SHA | 本REPORT/HANDOFFを収録するcommit。remote照合後の完全SHAを最終回答に記載し、下記コマンドでも解決できる。 |

```sh
git log -1 --format=%H -- docs/research/d_pr_df_ld1_m4_holdout_results_20261115/HANDOFF_C.md
```

branch `codex/d2-pr-df-ld1-m3-acquisition-20261115`、既存D2独立worktreeを使用した。m3のraw/receipt/INPUTS/HANDOFFと旧SOURCE・STOPを保全し、m4は別public/private出力へ保存した。Dのm9 branch/worktree/job、PR-Pauli LD2の完了STOPには書込み・停止・再開操作を行っていない。

## 観測前の固定と重複差分

[INPUTS](INPUTS.json)と[PREOPEN_CHECK](PREOPEN_CHECK.json)を科学call前にSOURCE commitへ固定。Cの固定JSONをcommitから直接読み、X/Y各予測のphysical ID、axis、q/R/T、元law/exactκ、独立cohort/seed表、比較規則と対応付けた。Pの再fit・再予測、表示値からの再構成はない。

既存§10.5のC索引/旧metadata照合を再利用し、その後の関連remote refsの名前差分、受入済索引、同物理点の旧ID/保存request、local同一取得job、出力先を価格未開封で確認した。受入可能な独立m4 compiled receipt0、出所未解決候補0、matching取得job0、出力namespace未使用。旧ZZやq/Rの異なる参照・ideal構造値はholdout価格ではない。外部のアクセス不能jobまで不存在とは主張しない。無関係なm9側の資料更新は読取観測のみで、HEADの一括凍結は要求していない。

seedは正本 `future_holdout.cell.seeds[index]` の16個をそのまま取り出した。元normal sampler bodyの実際のglobal `trial_seed` 参照へliteral lookupを渡し、PCG64/CDF・law・event生成本体は維持。pathごとに16回の参照が表と同順で一致した後にbuild/compileへ進んだ（`run01/PROGRESS_p*.jsonl` の `fixed_seed_lookup:CHECKED`）。元future planの `execution_authorized=false/HOLDOUT_NOT_AUTHORIZED` は履歴としてそのまま保持し、新INPUTSのactive `cells[0]` roleのみHOLDOUTである。

## 物理・実装条件とgateの範囲

Linear H4/1Å/STO-3G・保存tol-only DF10、LD1/S2/Δ=.1/K2、m4 T=1.6/q=16/R=32、physical ID **`3df0cfd3356c358aeda8`**。cohort **`D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1`**、path index0〜3。元exactκ・H/state/snapshot・DF fragment/basis順・P0を維持。元価格configurationは `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、元価格canonical IDは `RMSE:e217a74fb3981b43f6951fcfb24ae4683c5d95b688b38ab798fcd839a35b15dc`。Cの新元/total decoder IDは別欄 `frozen_prediction_identity` に残し、receiptの生成時IDへ付け替えていない。

scalar c+s_R=2.7108187593826942（hex `0x1.5afc1bee774aap+1`）、総Tで一回適用。preparation `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`、deterministic fragment順 `[0]`。Hamiltonian/state/snapshot hashesとexactκ/lawの正本はSOURCEのINPUTS `menus.H4.binding` / `menus.H4.candidates[0].law_binding` / `saved_input_files`。保存入力のbytesを固定値へ照合した。

`CONTROLLED_ZZ_CX_CRZ_CX_V1`、symmetric-directional control、測定付きfull controlled Hadamard wrapperをX/Y別に生成。compilerはQiskit1.3.0・rz/sx/x/cx・opt1・seed17、settings hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。元P2のaccelerated structural/compose/fingerprint経路を同一event列で実行し、固定SOURCE/入力、prep/fragment/scalar、request/time/allocation、通常seed、wrapperのqubit/measurement、compiler/metrics、axis fingerprint、compile時の入力mutationを確認した。

これは元P2/m3と同範囲の取得・identity・実装検査である。**数値action gateは `NOT_RUN_AS_IN_ORIGINAL_P2_FRESH`、full-wrapper statevector検証=false**。保存H4 P0演算子の独立数値再検査を含むsuiteを起動していない。過去のP0/人工検証を再利用し、新たな全入力作用同値性の保証を付けない。

## 個別receipt：同一pathのS/CとX/Y

以下の全行は同じphysical ID/cohort・各indexの固定16 seedを持つ。各行のSOURCEは `7bdfc6f629aa74a5c1347abf84ad6486c669c9d8`、raw publicationは `3042e6e66ad693ed514274334adecd3d1b42dc02`。全8行の取得・receipt状態は `D2_RECEIPT_BINDING_CHECKED`、valid=true。C自身の受入は未実施である。

| path | axis | S | C | 固定seed selector | 元axis receipt |
|---|---|---|---|---|---|
| 0 | X | 124411 | 129955 | `cells[0].seeds[0]` | [receipt](run01/receipts/p0/cosine.json) |
| 0 | Y | 124411 | 129955 | `cells[0].seeds[0]` | [receipt](run01/receipts/p0/sine.json) |
| 1 | X | 125935 | 131685 | `cells[0].seeds[1]` | [receipt](run01/receipts/p1/cosine.json) |
| 1 | Y | 125935 | 131685 | `cells[0].seeds[1]` | [receipt](run01/receipts/p1/sine.json) |
| 2 | X | 128133 | 133389 | `cells[0].seeds[2]` | [receipt](run01/receipts/p2/cosine.json) |
| 2 | Y | 128133 | 133389 | `cells[0].seeds[2]` | [receipt](run01/receipts/p2/sine.json) |
| 3 | X | 118424 | 123741 | `cells[0].seeds[3]` | [receipt](run01/receipts/p3/cosine.json) |
| 3 | Y | 118424 | 123741 | `cells[0].seeds[3]` | [receipt](run01/receipts/p3/sine.json) |

各固定locatorは [ADMITTED_RECEIPTS](ADMITTED_RECEIPTS.json) / [C_HANDOFF](C_HANDOFF.json) `rows[*].axis_receipt` の `{commit,path,selector}`。同じ行から `request`（固定物理request）、`path_result`（保存path）、`original_receipt`、`structural_receipt`、seed表、event digest、wrapper fingerprint、gateへ戻れる。公開axis receiptは元private bytesのコピーで、SHA256を保存・照合した。axis receipt自体にaxis名がない既存形式も、保存pathの `fingerprints[0/1]`、native cosine/sine、別軸fingerprintの違いにより接続している。

## 2件の記述的比較（exact有理数）

Pは軸ごとの凍結JSON `predicted_RZ` のみを使用。既に構造項と残差を含むPへ再加算していない。Oは同一軸のcompiled RZ 129955+131685+133389+123741を4で割ったexact算術平均である。正は過大予測、負は過小予測。分母はOである。

| axis | P (exact RZ) | O (exact RZ) | P−O | 絶対誤差 | (P−O)/O | 絶対相対誤差 |
|---|---|---|---|---|---|---|
| X | 217164876899444503/1683627180032 | 259385/2 | -1188941146855657/1683627180032 | 1188941146855657/1683627180032 | -1188941146855657/218353818046300160 | 1188941146855657/218353818046300160 |
| Y | 217164876899444503/1683627180032 | 259385/2 | -1188941146855657/1683627180032 | 1188941146855657/1683627180032 | -1188941146855657/218353818046300160 | 1188941146855657/218353818046300160 |

[COMPARISON.json](COMPARISON.json) / [CSV](COMPARISON.csv) がexact値の正本。全2単位で4 receipt有効・O>0なので `DESCRIPTIVE_COMPLETE`。欠測/無効時はその軸のOと誤差をnull、O=0なら相対誤差のみnullという取得前規則を維持した。今回は該当0。許容差必須validate APIに仮値を渡していない。

## 実際のeventと未観測

| path | events | order0 | order2 | ZZ rotation | Z rotation |
|---|---|---|---|---|---|
| 0 | 512 | 512 | 0 | 465 | 47 |
| 1 | 512 | 512 | 0 | 473 | 39 |
| 2 | 512 | 512 | 0 | 476 | 36 |
| 3 | 512 | 512 | 0 | 472 | 40 |

4独立pathの合計2048 eventはすべてorder0。X/Yは同一pathの別軸であり8独立pathには数えない。order2 product・非自明位相は今回未観測でUNKNOWN、未観測のrare event価格/寄与は未評価。追加seed・forced event・補完samplingは0。保存event原本はprivateに保全し、[ARCHIVED_RECEIPTS](run01/ARCHIVED_RECEIPTS.json) に元path/SHA256/event digestを保存した。

## 資源・人工検証・公開照合

元P2のcontext一回準備と4 path並列を使用し、compiler内poolは1。GNU time wall **61.27秒**、user235.07秒/system8.17秒、CPU397%、max RSS **593212 KiB**（各process最大の計測であり同時aggregateではない）、swap0、exit0。caller内部wall **60.382917秒**。取得開始時affinity64 CPU、MemAvailable512298716 kB、終了時512063272 kB、hostGPUA100-gleap02。GPUは使用していない。stageごとのSTART/END・RSSと60秒周期stack用ファイルを保存した。各workerは60秒以内に終了しstack dumpなし。CPU生存のみを進捗やgateの根拠にはしていない。

| path | wall秒 | 構造S秒 | compose秒 | X compile秒 | Y compile秒 | worker peak RSS bytes |
|---|---|---|---|---|---|---|
| 0 | 56.513109 | 5.184109 | 10.187771 | 12.857406 | 12.829990 | 589946880 |
| 1 | 57.246180 | 5.299491 | 10.174633 | 13.050732 | 13.039840 | 583577600 |
| 2 | 58.190950 | 5.401165 | 10.800056 | 13.072900 | 13.180889 | 586801152 |
| 3 | 54.069844 | 4.964732 | 9.906008 | 12.220463 | 12.179128 | 578363392 |

[EXECUTION](EXECUTION.txt)／[GNU time](EXECUTION_RESOURCES.txt)／[stage progress](run01/PROGRESS_p0.jsonl)。[性能手法集](../../PERFORMANCE_METHODS.md)へ再利用条件を記載した。追加benchmarkは行っていない。

取得前の限定人工/toy試験は **13 PASS**（m3既存5＋m4追加8、[log](tests/TECHNICAL_TESTS_01.txt)）。保存seedを使う元sampler bodyは人工prepで検証し、同一event S/C取得はfake compilerで検証した。exact比較・符号・ゼロ・欠測/無効/重複でpath数を変えない処理も確認した。既存文字列のSyntaxWarning5件のみ。人工PASSを科学性能や全law価格の保証に転用しない。

raw公開後、[BIND_AND_CHECK.py](BIND_AND_CHECK.py) が固定commit上の8個別receiptと元path/構造record、SOURCE、cell、軸、physical/cohort/index/seed、指標/fingerprint/hashを照合して8/8受入、失敗0を保存した（[check](RECEIPT_BINDING_CHECK.json)）。これはD2によるreader/binding照合で、Cの消費者受入とは別。比較後にもexact P/O/各誤差をreceiptから独立したFraction式で照合した。検査の詳細は [FINAL_CHECK](FINAL_CHECK.json) を参照する。

## ロードマップへの影響とSTOP

「m4未取得・凍結予測との比較待ち」は今回の固定4 path・2軸について解消した。比較対象は**保存ideal IのE_I[S]とsampler Qの有限較正cohort由来残差の混合予測**、および今回指定Qの有限holdout平均である。小標本変動、I/Q law差、未観測rare event、係数/モデル不確かさ、全law E[C]とCI、長roundへの移送、N/G・winnerは未認定のまま。差が小さいという理由で価格資格やモデル合格へ昇格しない。ロードマップ本体は統合担当が更新する。

Cへ8個別S/C receipt＋2件exact比較を引渡す。残差再集計/再fit、事後閾値、追加path/seed、別round、再compileは行わず、commit/push・remote照合後 **mandatory STOP**。
