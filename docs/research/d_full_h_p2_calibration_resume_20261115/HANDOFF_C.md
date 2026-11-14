# C向け部分引渡し：D7 CALIBRATION

最新：利用者許可によりSOURCE8cce54064でG36223 path0残検査を再開中。人工負荷で監視dumpのSIGSEGVを再現し、親process観測へ修正した。追加sampling/compile0。下記STOPは履歴、現受入は16軸PASS/2軸UNKNOWNのまま。[修正と実行](../d_full_h_p2_completion_20261115/RECOVERY.md)。

**MANDATORY STOP・全較正未完了。C fit/凍結は実施しない。** 18 compileの価格は全て保存したが、最終path検査は16軸PASS/2軸UNKNOWN。4 cell・14軸の公開後対応検査はPASS。G36223 path1は保存済みPASS、path0残検査のworkerが異常終了した。完全な各cell/path/X/Y locatorは [HANDOFF_C.json](HANDOFF_C.json) の `cells.*.path_axis_bindings` にある。

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

G36223のpath0は旧SOURCEでnative7 gateと2 compileが保存済み。許可された計画的引継ぎにより旧workerを終了したため、旧parentのSTOP_CELL/exit2を科学FAILへ数えない。旧record/raw/QPYは保全した。新SOURCEではpath0をsampling/compileなしで復元し残検査を行い、未着手path1の2 compileを初回取得し、全path検査PASSで保存した。その後SOURCE0dcの0 compile継続へ切り替えたが、同workerは4:14.99でchild receiptを書かず異常終了した。全体上限18は維持した。新runは計画的STOPではない。

未取得：G36223 path0の最終path receipt（2軸）。compile価格の未取得0。G36223の完全な較正平均recordは作らず、4 cellの8較正recordとG36223の部分証拠を区別する。SOURCEd323のpath0 X/Y各5552901、SOURCE887のpath1 X/Y各5572406は保存済み。新SOURCE0dcでの追加sampling/compile0。未確認：C受入、finite-cohort CI/confidence、sampler/ideal距離、rare寄与、境界一般化、長時間資格、E[C]/N/G。全較正引渡し後又は重大gate失敗でmandatory STOP。後続のfit/凍結・HOLDOUTは別指示の作業である。

## 異常終了と部分証拠の扱い

[終了報告](../d_full_h_p2_completion_20261115/REPORT.md)／[STOP_DETAIL](../d_full_h_p2_completion_20261115/STOP_DETAIL.json)。source0dcの最後の段階はbuild_new_evolution。例外・child receipt・子終了signalがなく原因未特定。cgroup OOM記録0、単純人工timer診断では再現できなかった。数値gate不一致は観測していないが、必要な残検査のPASSを確認できないためmandatory STOPとして扱う。科学retryはしていない。

G36223のCOMPILE_RECORDSは元JSON bytes/hashを保存し、OBSERVED_COMPILESで軸・producer SOURCE・測定RZ・受入状態を照合する。path0のX/Yは保存QPY末尾の測定軸と固定hashで確認した。元native7 gateは7d0のNATIVE_CHECKPOINT_PATH0.jsonに固定される。原eventはSAVED_PATH0_EVENTS、完了path1原receiptはSAVED_PATH1にlossless gzipで保存する。これらは部分証拠であり、G36223の完成PRICE_CALIBRATION_RECORD_V1や全D7の受入を意味しない。

既存4 cellの固定locatorは7d0c39066のまま維持する。G36223部分証拠の公開SHAとD1の公開後読取検査は、HANDOFF_C.jsonのG36223_partialへ結合する。C担当の科学的受入は全て未実施。非configuration cellへcanonical round identityを割り当てない。

## 最終公開locatorの結合

G36223の部分結果SHA：`1fdb5dd42f65d6286848dc76e686f62c5b3accf8`。各path/軸の元compile JSON・観測価格・producer SOURCE・raw manifest・実装証拠の固定locatorはHANDOFF_C.jsonの `G36223_partial.path_axis_bindings`。完成4 cellの8個のPRICE_CALIBRATION_RECORD_V1は `available_PRICE_CALIBRATION_RECORD_V1_locators` に列挙し、元7d0のlocatorを維持する。

D1は公開SHAから両raw archiveを復号し、path0の元event hash/cohort/seed、path1の完了receipt hashと全実装証拠、4つの元compiler JSON bytes/hash/metrics/fingerprint、元native7 gateの公開file hashを確認した。[公開後検査](../d_full_h_p2_completion_20261115/PUBLICATION_CHECK.json)。C c628e9e8のgeneric evidence readerで部分JSONが読めることも確認した。D1技術検査であり、C科学的受入はNOT_PERFORMEDのまま。

G36223のC_INPUTSは存在せず、完成較正recordを作っていないことも照合した。path0最終検査2軸UNKNOWN・全D7未完了・mandatory STOPを維持する。
