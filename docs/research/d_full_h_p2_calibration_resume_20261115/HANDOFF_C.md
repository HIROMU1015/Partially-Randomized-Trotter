# C向け部分引渡し：D7 CALIBRATION

**G36223取得中・全較正未完了。C fit/凍結は実施しない。** 4 cell・14軸の公開後対応検査はPASS。完全な各cell/path/X/Y locatorは [HANDOFF_C.json](HANDOFF_C.json) の `cells.*.path_axis_bindings` にある。

結果SHA：`7d0c39066d4ed900cfe7f5de6f4c7c5a3b2ec41f`。取得SOURCEはd323fee1c、G36223の許可済みcheckpoint継続SOURCEは887c3b9d0。D7のplan/cohort/seedはFROZEN_JOBSから変更していない。

## Cが読める証拠とD1が確認したこと

- C_INPUTS.json：既存 `PRICE_CALIBRATION_RECORD_V1` のX/Y別有限cohort平均・feature・physical/law/profile/SOURCE。有限平均はlaw期待値ではない。
- PRICES.json：各path/X/Yの直接compiled価格、fingerprint、cohort/seed、埋込み元compile record。
- IMPLEMENTATION_EVIDENCE.json：軸ごとのcalibration recordと、全pathの価格・実際のphase/parity/XY/one-body/scalar checks、数値gate、資源。
- COMPILE_RECORDS/p<i>_<axis>.json：元native compiler JSONのbyte保存。SHAとmetrics/fingerprint/compilerを価格receiptへ照合済み。
- RAW_RECEIPT.json→RESULT.json.gz：rawはlossless gzip。公開READ_RECEIPT.pyの `read_receipt(commit,manifest_path)` を使ってhash検査し、JSONを復号する。既存C metadata readerにgzipを直接渡さない。JSON selectorは `child.paths[i]`。
- PUBLICATION_CHECK.json／SIDECARS_BOUND.json：固定公開SHAへの結合結果。D1は元receiptから固定callerのrecordsを作って公開小JSONと完全比較し、C c628e9e8の `evidence` readerで各locatorを読み取った。

これはD1本人による技術照合であり、**C担当の科学的受入はNOT_PERFORMED**。非configuration cellへcanonical-round guardを呼ばず、架空round IDを作っていない。index0の各cellに既存7数値gate、全pathにphase/parity/XYの適用範囲を保持する。index1をindex0の数値検査済みと扱わない。

## pairedと継続の依存

P2R566はG566 path0/group0（seed11290319720649780064）→path1/group0（seed9014355981526777692）の順に、実取得親event receiptを直接復元・連結した。親SOURCE/physical/cohort/index/seed・順序・phase・scalarをbuild前に照合し、子X/Yはそれぞれcompileした。sampler呼出し0、同seed再生成0、親価格copy0。親2標本と相関するpaired1観測であり、独立な第3標本として数えない。親の公開manifest locatorとdecoded selectorはHANDOFF_C.jsonのpairedに固定した。

G36223のpath0は旧SOURCEでnative7 gateと2 compileが保存済み。許可された計画的引継ぎにより旧workerを終了したため、旧parentのSTOP_CELL/exit2を科学FAILへ数えない。旧record/raw/QPYは保全した。新SOURCEではpath0をsampling/compileなしで復元し残検査を行い、未着手path1の2 compileだけを初回取得中。全体上限18は維持する。

未取得：G36223の最終path receipt・公開後binding。未確認：C受入、finite-cohort CI/confidence、sampler/ideal距離、rare寄与、境界一般化、長時間資格、E[C]/N/G。全較正引渡し後又は重大gate失敗でmandatory STOP。後続のfit/凍結・HOLDOUTは別指示の作業である。
