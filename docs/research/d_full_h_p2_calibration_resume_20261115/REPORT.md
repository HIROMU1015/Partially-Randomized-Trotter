# Full-H κ2 D7 CALIBRATION取得

現在は4 cell・14 compileが完了し、G36223を許可済みcheckpointから継続している。D7全体の引渡し・mandatory STOPは未完了。

## 固定条件とSOURCE

H4 linear 1 Å/STO-3G、保存tol-only DF10、one-bodyを含むFull-H pure RTE、L_D=0、K=2・κ2。各group時間0.1、元physical ID・D7 cohort/seed・scalarを維持した。Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17、P0、測定付きfull Hadamard wrapper、CONTROLLED_ZZ_CX_CRZ_CX_V1。状態準備費は含めない。

- 元SOURCE `6069605d50270111d1510478b08e11b8e5a18d86`、旧STOP `df0e11fd1e8d0f09166db71646d3df101ab13233` は保全。
- 限定修正と最初の取得SOURCE：`d323fee1ca8cd6893c4b705f990cdc931bdb5c87`。[SOURCE_PREPARATION.md](SOURCE_PREPARATION.md)と[FROZEN_JOBS.json](FROZEN_JOBS.json)。
- 許可済みcheckpoint継続SOURCE：`887c3b9d04d15be96637b6634265424e7dd50339`。[高速化報告](../d_full_h_p2_acceleration_20261115/REPORT.md)、[固定継続入力](../d_full_h_p2_acceleration_20261115/FROZEN_CONTINUATION.json)。
- A設計checker `7a0c9468b2499fac9fa1cd279ae2980c9dab048e` のrank3/domain検査PASS。D7全24 compile中、今回の対象はCALIBRATION18だけ。

## 取得済み価格

各値はそのpathのX/Yで個別にcompileしたRZ数。有限cohort観測でありlaw期待値ではない。

| cell | q/R | path0 X/Y | path1 X/Y | 状態 |
|---|---|---|---|---|
| G566 | 1/566 | 82785 / 82785 | 87375 / 87375 | 4 PASS |
| G4528 | 1/4528 | 695637 / 695637 | 691505 / 691505 | 4 PASS |
| G36223 | 1/36223 | checkpoint保存済み | 実行中 | 最終receipt未取得 |
| Q8R566 | 8/566 | 696593 / 696593 | 689039 / 689039 | 4 PASS |
| P2R566 | 2/566 | 169888 / 169888 | 該当なし | 2 PASS |

## pairedの依存

G566 path0/group0（seed11290319720649780064）、次にpath1/group0（seed9014355981526777692）の親receiptを直接復元し連結した。親SOURCE/physical/cohort/index/seed・event順序・phase・scalarをbuild前に照合し、samplingは `SAVED_PARENT_EVENTS_NO_DRAW`、対応は `PAIRED_EVENTS_IDENTICAL`。子のX/Yは新たにcompileした。親価格のcopyや同seed再samplingはしていない。P2R566は親2本に依存する1つのpaired観測であり、独立な第3標本には数えない。親と子の各receipt・個別compile recordはrun01に保存する。

## correctness・重複・証拠

index0の各cellでlow/high control/leakage、independent occupation vector、Hadamard X/Y、compiled vector X/Yの既存7数値gateを実施する。index1では作用gateを再実行せず、全pathでparity・新lowering・scalar/event phase・one-body・同一evolutionのX/Y検査を実施する。元閾値とphase_alignment=falseを保持する。人工検証はmetadata11、paired toy2、高速化9 PASS（小toyの実Qiskit X/Y含む）。追加H4試運転0。D1本人のlocal検査であり、独立レビュー・CIではない。

[DELTA_PREFLIGHT.json](DELTA_PREFLIGHT.json)はdf0e11fd1の§10.5調査を継承し、関連ref/job・新出力・保存価格とSOURCE差分を照合した。旧light18/T2.0 90はphysical条件・旧loweringが一致せず、P1 m3 q8/R71はcohort/R/pathが異なる。受入可能な事前保存価格の再利用0。G36223の内部checkpoint2軸は今回同scopeで先に取得した結果を再利用するもので、18上限へ再加算しない。

各cellのPRICES.jsonはDIRECT_COMPILED_PATH_PRICE_V1、C_INPUTS.jsonはPRICE_CALIBRATION_RECORD_V1、IMPLEMENTATION_EVIDENCE.jsonは実際のchecks・compile記録・価格・固定SOURCEへ結ぶ。COMPILE_RECORDSには元native compiler JSONのbytesをhash照合して保存する。巨大RESULTは元rawをlocal保全し、lossless gzipで公開する。RAW_RECEIPT.jsonに原bytes/hash・gzip hash・復号一致を記録する。結果commit `7d0c39066d4ed900cfe7f5de6f4c7c5a3b2ec41f` をpushした。4 cell・14軸について公開READ_RECEIPT.pyで明示復号し、元recordとの完全一致とC metadata readerの読み取りをPASSした。固定locatorは [HANDOFF_C.md](HANDOFF_C.md)／[HANDOFF_C.json](HANDOFF_C.json)。

Cのc628e9e8 evidence readerは小さいJSONを読む。D1がこれを使ってlocatorと証拠対応を検査することと、C担当の科学的受入は区別する。C科学的受入は未実施。非configuration cellへcanonical round guardや架空round IDを適用しない。

## checkpoint引継ぎ

旧G36223 path0のnative trial PASS（X/Y2 compile・7数値gate）が保存された境界で、path1未着手を検査し、ユーザーの明示許可を受けて旧workerを終了した。旧parentのSTOP_CELL/exit2とrawをそのまま保全し、CONTROLLED_AFTER_NATIVE_CHECKPOINTと区別する。科学gate失敗・無断retryではない。新namespaceでpath0のsampling/compile/数値gateを再利用し、元fingerprint/metrics照合と残りphase/parity/XYを実行し、未着手path1を一回だけ取得する。新compile最大2・D7累計最大18。

## 資源・未取得・終了境界

cellごとに独立jobとし、親G566の保存後にpairedを実行、独立cellは重ねて取得した。G566全job wall4:47.06 / peakRSS606688 KiB、P2R566 5:34.31 / 933960 KiB、Q8R566 38:39.61 / 2219544 KiB、G4528 39:05.27 / 2210804 KiB。正確な元計時はrun01/*.time.txt。高速化はpath/process、X/Y compiler/process、exact controlled-gate cache、同一canonical bytesのfast fingerprintを使う。共有設定は変更していない。toy primitiveの速度比をH4全体の速度比へ外挿しない。

現在G36223の最終receiptとC向け公開後locator結合が未取得。全較正引渡し、または重大gate失敗でmandatory STOP。HOLDOUT6 compile、fit/係数凍結、長時間外挿、E[C]、N/Gは実施しない。finite-cohort CI/confidence、sampler/ideal-law距離、rare寄与、group境界の一般化、m7–m12資格は未確認。

ロードマップへの影響：D7のrank3較正データを取得する段階であり、Cの後続fit/凍結とD1の別指示によるHOLDOUTの依存順序を維持する。ロードマップ本体は統合担当が更新する。
