# H6不足metadata追加取得：loader初期検査STOP・修正と保全

2026-10-22 JST。**METADATA stageはsnapshot_restoreを1回消費した直後、scalar配列のshape検査により例外終了。指定の失敗時STOP条件に従いmandatory STOP。63候補の再観測・H6後続取得・3系最終比較は未完了。**

入力の破損や科学条件の不整合は検出されていない。loader側が0次元scalarをcontiguous化してshape=(1,)に変え、そのshapeを原NPY receiptのshape=[]と比較したため失敗した。修正は別SOURCEへ保存し人工5件PASS。ただし修正後の実H6 loader/metadata取得は再実行していない。

## 承認範囲と固定条件

利用者は、旧H6 STOPが列挙するB0I 63候補の不足norm/closure再計算を別科学予算で明示承認し、その成功後の固定276候補の共通PF取得・PR/PF比較まで包括承認した。対象はlinear H6、1 Å、STO-3G、tol-only1e-8 rank19、T=.8/P=0、B0/B0I L_D=5..15、Full19、q1/2/4/8、δ=.8/.4/.2/.1、S2/Yoshida4/m5_best。full cosine/sine Hadamard、symmetric-directional、scalar once、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17を固定。

基点branch `pf-common-domain-execution-20261022` / commit `0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c`、新branch `pf-common-h6-metadata-execution-20261022`。旧H6 INPUT_STOPと旧H2/H4完了結果は変更しない。[事前実行契約](track_a_h6_pf_metadata_execution_20261022.md)。

## 取得方法の調査・封印

厳密算術ではB0I scalar位相がblockと可換でnorm不変。一方、旧B0は各merged primitiveとouter scalar後にnormを観測し、B0I serializerはscalar signalのみを回転して中間vector情報を保存していない。binary64上の途中norm/closureを保存scalarから復元したと主張せず、63件を新たに観測する計画を固定した。

固定providerの元compact_signal・recurrence_paths・merged scheduleを使用し、native corrected/rawを別実行、独立400次元occupation forwardと照合、各primitive sector leakage<1e-12、終端vector closureを実測する。B0Iのscalar位相は元overlayと同じ位置、full-H referenceは回転しない。52保存signalは旧corrected/reference/bias/Cと1e-9未満で照合して旧signalを保持、残る11件は新取得と区別する。0補完・観測定義の別指標への置換はしない。これらは**今回の実行では数値取得まで到達していない**。

63候補の元scheduleから事前封印した内部上限：native corrected63/raw63、独立forward63、native primitive生成/作用各18736、forward block作用9368、native outer scalar476/forward238、最終vector位相作用189、closure63、旧signal照合52、独立sector matrix/eigh各16、tail scalar11、context snapshot/reference/oracle/full preparation各1、prefix preparation0。SCF/DF/state/reference新生成・compile・sampling0。これは102候補の本来の新signal予算と別枠。

旧後続予算102=51base+51overlay、forward51、1380 endpoint、最大224 compile候補448軸は変更せず、依存するMETADATA成功が得られなかったため、後続契約/grantも未発行。

## 実行経過と失敗原因

| 区分 | SOURCE | 科学call・状態 |
|---|---|---|
| 初回準備 | 53af59a0e | package contextなしの相対importで準備停止、科学grant0 |
| 準備修正 | 689654ce2 | tuple scheduleとJSON arrayの比較相違で準備停止、科学grant0 |
| 正規化済み実行pack | 3cff9a4ba67d61b3d2fed04f18ad39d3cb0a2c0b | METADATA grant発行・snapshot_restore1後に例外、exit1 |
| STOP後loader修正 | 458ea00dd | H6は原NPY shapeを比較、H2/H4移行receipt定義を保持。人工5PASS、実H6再実行0 |

2件の準備停止はgrant前で、科学retryではない。各準備rootも保全した。実行pack `METADATA63_run02` は名前にrun02を含むが、科学grantを消費したmetadata取得はこれが最初の一回である。

NPY原snapshotのSHA256は `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005` で旧binding・receiptと一致。constantのNPY headerはshape=()、receiptはshape=[]、dtypeは<f8、payload SHA `2b3651c6d759385f18dad4b447e3b0789fe4e179892a1519998e2fccfcdb2b4b` も一致。`np.ascontiguousarray`により一時配列shapeが(1,)となる挙動を検査コードが考慮できていなかった。原因確認はstdlibでNPY headerとpayload SHAを読んだだけで、追加分子作用を行っていない。

修正版はH6 receiptのshapeを原value.shapeと比較する。dtypeとpayload hash検査は保持し、H2/H4移行receiptのcontiguous-shape規約は変更しない。scalar数値/Unicode scalar/非scalar、SHA改変・dtype改変拒否を含む人工5 test method PASS。事前には新4＋既存人工10PASSだったが、原H6 scalar shape差を事前人工テストで捕捉できなかった。

## 非返却実使用・終了監査

| 科学処理 | 実使用 |
|---|---:|
| snapshot_restore | 1 |
| occupation oracle/reference read/preparation | 0 |
| metadata候補・native作用・forward・closure | 0 |
| 旧52signal照合 / 新11signal | 0 / 0 |
| 本来の新signal102候補 | 0 |
| 資格/shot endpoint / compile軸 | 0 / 0 |
| PR sampling・再統計・再価格取得 | 0 |

ledger1件=reserved counter=STOPのactual_calls、全上限内。一回のsnapshot_restoreを返却せず、grant/CONSUMED/ledger/STOPは原byteを保全。終了receiptは科学成功とはせず、失敗stageの終了監査を別に記録した。native commandはexit1で回収済み。context中の停止でworker生成0、worker残存0、resource選択前である。計算を小さいworker上限で制限した結果ではない。共有設定や他ユーザー操作0。

ユーザー指示§6の「科学call消費後に実行が失敗した場合は、対象stageをmandatory STOP」に従い、修正後もこのstageを再起動していない。same grant retry/resume0、後続grant0。成功したことにして次stageへ進めない。

[終了監査](../../artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/ending_audit.json)、[新契約の読み取り投影](../../artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/execution_contract_APPROVED_projection.json)、[grant](../../artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/grant.json)、[CONSUMED](../../artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/CONSUMED.json)、[ledger](../../artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/ledger.json)、[原STOP](../../artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/STOP.json)、[診断](../../artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/loader_failure_diagnosis.json)、[人工検証](../../artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/tests.json)を保存した。実行SOURCEの全依存ファイルはGit commitからarchive化して保全し、実行SOURCEと修正SOURCEを区別した。入力/provider/runtimeのseal370件照合、旧結果75ファイルのbyte一致を確認した。

## 588候補の位置づけ

H2 84・H4 228の計312評価完了を維持し、再計算0。H6 276は引き続きUNRESOLVED、原観測111・不足63・本来の新signal未実行102。H2/H4の元結果と旧H6 STOPへの参照結合だけを[状態表](../../artifacts/resource_applicability/track_a_h6_pf_metadata_STOP/2026-10-22/integrated_588_status.json)に保存した。新しいH6最良PF・PR/PF費用比・統計・順位は取得していない。

共通政策・保存PR cohort・mean/variance/SE・SOURCEは不変。経験的allowance、有限標本選択効果、未観測rare event、母平均保証なしという旧限界も維持する。今回の終了監査PASSは、科学取得PASSやH6比較完了を意味しない。

**修正・人工検証・監査・結果保存を専用branchへローカルcommit後mandatory STOP。GitHub push、新規候補、q16、PR sampling、詳細科学レビュー、論文本文作成は0。**
