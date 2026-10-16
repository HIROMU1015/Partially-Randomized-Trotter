# Track A V2：保存RZ残差runnerの実装・人工検証と未解決の来歴結合

**runner・人工検証・未承認契約を準備した。実データのfit・残差解析は未実行である。現在の来歴台帳では262 scope entriesすべてが`UNRESOLVED_PROVENANCE`で、fit適格な層はない。実行承認だけで有効な残差評価へ進める状態ではない。**

branchは`track-a-v2-residual-analysis-readiness-20261016`。前段の計画commitは`0b573f0bbb922c1cb8bf851241d11816162a1653`、今回の公開実装SOURCEは`5703dbda58ebb8b7a9d18dce18ba33ed45f3c11c`。本文とartifactを追加する文書commitとは区別する。[前段の構成対応・最小計画](track_a_v2_residual_analysis_readiness_20261016.md)を維持し、[今回の機械可読証拠](../../artifacts/resource_applicability/track_a_v2_residual_implementation/2026-10-16/README.md)へ結合した。

## 実装した範囲

|成果物|役割|
|---|---|
|[library](../../src/trotterlib/track_a_v2_residual.py)|来歴分類、反復・PF再利用の重複排除、固定holdout、uniform A/B別・軸別の端点fitとSTOP判定|
|[runner](../../scripts/track_a_v2_residual_runner.py)|既定は計画表示。新grant・fresh gate・bytes検査・外部observer・one-shot予約を通過した場合だけ保存値を読む|
|[metadata準備](../../scripts/prepare_track_a_v2_residual.py)|公開済み入力のhash照合、保守的な来歴台帳、未承認config/grant案の作成|
|[人工tests](../../tests/test_track_a_v2_residual.py)|分子入力を使わず、dedup・欠測・数値境界・grant拒否・停止順序を検証|

既存理論の正本は`7e41670565d9284da9aded94063be591fdcaa3dd`。理論の実行SOURCE`7e59df8ac3c356007d85bfd9218ce1a224a40d52`、832理論行、保存接続10,480行、candidate/source索引262 entriesを再利用する。新primitive、law、DP、compiled結果を作らない。

対象はlinear H6/H8、1.00 Å、STO-3G、保存DF rank19/28、T=.8、旧prefix格子・q1/2/4/8・登録S2/S4/PR集合。uniform A/Bの保存RZ単価診断だけを実装し、実nativeのA-prefix/B-tail構成と区別する。混合理論列、CX、shot重み付き総費用、PR削減率、候補順位、V1追加格子・p7実測、H5/H7は対象外である。

## 来歴照合の結果と、数値実行に先立つ不足

今回の実データ処理はindex・execution bindingのmetadata分類とbytes/hash検査までである。理論費用・接続費用の数値欄をparseしてfit・減算・比を計算していない。公開auditの262行はmetadataの判定結果で、残差行ではない。

|独立に結合できていない項目|対象entries|
|---|---:|
|compiler profile、basis variant、control、scalar policy、native policy、lineage compatibility、preparation binding、law binding|各262|
|origin SOURCE|16|
|PRのdataset/seed SOURCE|158|
|wrapper identity|7|

これらは既存研究に条件の記述が全くないという意味ではない。今回固定した台帳に、個々のcompiled lineageへ結合された独立receiptが不足している。indexにあるSOURCE列挙・wrapper記述・compiler hashと、理論側のH/state/DF/orderは保持したが、それだけで旧profile・旧H6 basis variant・準備/lawの同値性を推定しない。PFのdataset/seedは明示的に非該当とし、PRの欠測と区別した。

次に必要なのは**保存receiptとcompile経路の読取専用照合**である。profile/basis/control/scalar/nativeの証拠、準備/lawへの参照、originとdataset/seedの結合、scopeを越えて再利用したPF wrapperの同一性を記録する。証拠が揃わないentryは未解決のまま除外する。台帳・configを変更した場合はhashと契約を改めて固定し、旧draftを上書きしない。

現在の台帳を数値stageへ渡してもfitは0件で、実装は`NO_ASSESSABLE_STRATA`・`all_strata_passed=false`を返す設計である。この動作は人工値で検証しただけで、実データでは起動していない。完全数値envelopeの認証、母平均保証、mixed構成の分解、厳密bias保証は未確立である。

## 固定した解析・STOP条件

独立来歴を持つ同一stratumで、保存単価に対して共通の`C_compiled = a + b*C_theory`をfitする。epsilon/factor/modelによる同一観測の反復は照合して一つにする。PFのscope間再利用は独立したreuse cohortの証拠を要求し、単価が同じという理由では統合しない。欠測・零/負単価・非有限値を残差0で補わない。SOURCE/profile/variantの矛盾や反復値の不一致はSTOPする。

holdoutは前段の`(system,prefix,q,formula)`のcanonical ASCII JSON→SHA256 big-endian `mod 5 == 0`に固定する。train/holdout双方にPR/PF各2以上、trainの異なるxが2以上、rank2を要求する。不足時は`INCONCLUSIVE`、分割探索はしない。lower→upperの別fitとし、midpointへ置換しない。scaled float64 least squares、`rcond=1e-12`、condition number上限`1e10`。負の傾き、非正予測、非有限値はSTOPする。

holdoutの`e=(C_compiled-fit)/C_compiled`について、`abs(mean(e_PR)-mean(e_PF)) >= 1/20`で即STOP。閾値より下でも差が`1e-12`以内なら`BOUNDARY_UNRESOLVED`でSTOPする。これは平均signed相対残差差の5 percentage pointsであり、V1の2% compile検証とは別である。H6→H8、A→B、cosine→sine、stratum辞書順、lower→upperの順序を固定し、fit前に予約をfsyncする。最初の不一致・例外を保存して次fitへ進まない。再fit、retry、resume、次stageへの自動移行は禁止する。

## 人工検証とSOURCE/input seal

固定Python/NumPy環境で新規36件と、再利用するobserver/cleanupの限定7件がPASSした。fail/skipは0。metadata・costは小さな人工fixtureで、実H6/H8数値解析ではない。primary agent自身のlocal checksであり、独立レビュー・CI・外部再現とは呼ばない。[検証receipt](../../artifacts/resource_applicability/track_a_v2_residual_implementation/2026-10-16/artificial_validation.json)に選択した検査を記録した。

新36件は、反復とPF再利用、SOURCE/profileの混合拒否、欠測・不正単価、固定holdout/rank不足、.05と近傍、lower/upperの即STOP、fit例外の保存、fit適格層0の非PASS、grant/CPU/fresh gate、gzip上限、SOURCE/runtime/input改変を検査した。既存7件はobserverのexec peak/lifetime区別、RSS/AS・総量/thread・PSI/OOM、observerとdriverの制限分離、正常終了・失敗driverのowned orphan回収を検査した。

実SOURCE9ファイル、固定runtime34,273ファイル、5入力（theory・connections・index・binding・新metadata registry）のbytes/hashを照合した。既存4入力は理論結果commitのGit blobにも一致する。gzip展開の上限は16 MiB、既存exportの展開14,579,546 bytes/SHAを承認後にも照合する。SOURCEの完全9ファイル一覧と入力SHAは[source binding](../../artifacts/resource_applicability/track_a_v2_residual_implementation/2026-10-16/source_input_binding.json)にある。

private config SHAは`37d81ab7bfaa0bfb1cae995cf7938230b71a3ba792e011bc528a72037003dd05`、policy SHAは`0422806c0721b7814ae9edcd3356848713205eb7ad9735980562cc71560dbafa`。公開済み実装SOURCEへの結合前に作ったdraftもprivateに保全した。新しい有効grantとfresh gateは発行していない。公開draftは承認false・SOURCE=null・CPU=[]・command=null。実行にはcleanな実行SOURCE・入力台帳と別途承認を再結合する必要があり、文書HEADや公開draftだけでは起動できない。

実際の未承認private draftを`--launch-approved`へ渡す拒否検査も実施し、`NEW_APPROVED_RESIDUAL_GRANT_REQUIRED`で終了した。これはnegative gate testであり、数値stageの起動ではない。control/output/one-shot lockは存在しない。

## 資源上限・保全・公開後STOP

driver1＋observer1、thread各1、別CPU2枠を上限とする。driver AS4 GiB/RSS1 GiB、observer AS/RSS512 MiB、総RSS1.5 GiB、総CPU600秒、wall1800秒、output64 MiB、stage128 MiB。上限は所要時間の予測ではない。最大832 theory rows、10,480接続行、262 index entries、1,048単価/model行、64 strata・計128 fitsで固定した。小さなfitは登録順で逐次実行し、即STOPを守る。実際のCPU予約は0である。

既存外部observerの1秒間隔・最大gap5秒、PID/start/parent/UID所有確認、live RSS/AS/CPU/thread、PSI/OOM、headroom16 GiB、storage reserve2 GiB/inode4096を継承する。起動前30秒以内の資源・SOURCE/input/profile・保護job非干渉確認を要求する。所有確認した自分のprocessだけを回収する。

既存grant/ledger/carry/one-shot等665ファイルのbytes/hash、V1・旧V2・理論worktreeのHEAD/status、旧V2の未追跡scriptを前後照合して保全した。旧V1/V2/H5/H7のSOURCE・結果・rawを変更せず、旧grantを流用していない。今回の追加は準備draftで、有効grantの発行は0である。

公開対象は実装・人工tests・報告書・policy・未承認grant案・metadata監査・hash台帳・関連索引に限定した。private config/registry本体、runtime一覧、分子配列、credential、他job一覧は公開しない。既存832理論行・compiled接続を新branchへコピーしない。

**新科学計算・数値再解析・compile・signal・sampling・primitive/law/DP・実データfit/残差/費用比/順位更新は0。公開後にremote SHA/treeを確認してmandatory STOPする。数値stageは未承認であり、自動移行しない。**
