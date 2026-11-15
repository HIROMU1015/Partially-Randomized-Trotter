# C向け引渡し：D7 CALIBRATION全5 cell

全5 cell・9 path・18軸の最終検査PASS。X/Y別のPRICE_CALIBRATION_RECORD_V1は10件。全較正を引き渡しmandatory STOP。有限cohort観測でありlaw期待値・モデル資格ではない。

先行4 cellの固定結果SHAは7d0c39066d4ed900cfe7f5de6f4c7c5a3b2ec41f、完成G36223はf98cb96384b775bec2c2b9e1e63fe01a6ccdf2b1。
全axisの固定locatorは [HANDOFF_C.json](HANDOFF_C.json) の cells.*.path_axis_bindings、10較正recordは available_PRICE_CALIBRATION_RECORD_V1_locators。

## Cが追跡できる証拠

- C_INPUTS：各cell/X/Yの有限cohort平均・feature・physical/law/profile/SOURCE。
- PRICES：各path/X/Yの直接RZ価格、cohort/seed、fingerprint、埋込み元compile record。
- IMPLEMENTATION_EVIDENCE：対応する全pathの実checks・数値gate・scalar/one-body・compile記録・価格・資源。
- COMPILE_RECORDS：元個別compiler JSONをbytes/hashで保存。producer SOURCEを検証SOURCEと区別する。
- RAW_RECEIPT：hash付きlossless gzip（G36223は2 part）。READ_RECEIPT.py:read_receipt(commit,manifest_path)で復号し、child.paths[i]へ辿る。既存C metadata readerにgzipを直接渡さない。
- SIDECARS_BOUND/PUBLICATION_CHECK：上記を公開SHAへ結合したD1の照合結果。

D1は公開SHAからreceiptを復号し、固定callerのrecordsを再生成して公開JSONと完全比較。各path/軸の価格・元compiler bytes/hash/fingerprint/metrics/compiler・実装証拠・PRICE_CALIBRATION_RECORD_V1を追跡できることを確認した。C c628e9e8のgeneric evidence readerで公開JSONを読んだ。**C担当の科学的受入はNOT_PERFORMED**。このD1本人の技術照合を独立reviewやC科学受入と呼ばない。

非configuration較正のためcanonical-round専用guardは呼ばず、架空round IDを付けていない。Cはこのrecord/evidence対応を読み、必要な較正受入と後続fitを別途判断する。

## correctnessと依存関係

各cell index0の既存7数値gateはPASS。index1にindex0の数値検査を拡張した主張はしない。全pathのparity・scalar/event phase・one-body・X/Y・元compile fingerprint対応はPASS。

P2R566はG566 path0/group0（seed11290319720649780064）→path1/group0（seed9014355981526777692）の実親eventを直接連結。親SOURCE/physical/cohort/index/seed・event順序・phase・scalarをbuild前に確認。sampling/同seed再生成/親価格copyは0、子X/Yは各々compileした。親2観測と相関するため独立な第3標本に数えない。親manifest/selectorはHANDOFF_C.jsonのpairedに固定した。

G36223 path0価格のproducerはd323fee1ca8cd6893c4b705f990cdc931bdb5c87、path1は887c3b9d04d15be96637b6634265424e7dd50339。今回の検証SOURCEは8cce540645b99e515d72888b817782a1b9f97334。元native7 gate・4 compileを再利用し追加sampling/compile0で残検査を完了した。旧SOURCE/STOP/raw/QPYは保持する。旧部分証拠G36223_partialはhistorical_onlyであり、cells.G36223の完成receiptに置き換わる現受入状態とは区別する。

## 件数・未確認・終了境界

予定18、事前同等価格再利用0、実取得18、最終PASS18/FAIL0/UNKNOWN0、価格/必要receipt未取得0。再開時再利用4は18件の内数。D1公開後対応照合18軸、較正record10件。

未確認：C科学的受入、母law CI/confidence、sampler/ideal距離、rare寄与、境界一般化、長時間資格・モデル採用。fit/係数凍結はCの後続作業。D1はm5/m6 HOLDOUT6 compile・外挿・E[C]/N/Gへ進まずmandatory STOP。[全報告](REPORT.md)。
