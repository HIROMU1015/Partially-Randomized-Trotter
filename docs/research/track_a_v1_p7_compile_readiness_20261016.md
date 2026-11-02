# V1：B0+I p7/q1 S2の2軸実測検証・実行準備

後続の利用者実行承認により、SOURCE `84b9ea22…`の登録2軸compileは正常終了した。[実測結果・終了監査](track_a_v1_p7_compile_results_20261016.md)：両軸14,492 RZ、各0.222191554%でPASS、mandatory STOP。総費用再会計は未実行・別承認待ち。以下は準備完了時点の記録。

**実装・人工検証・入力照合・未承認grant案を準備した。新規compile・native build・control probe・signal再計算は0。**
今回の利用者指示は準備と公開までであり、実計算の実行承認ではない。公開・remote SHA確認後にmandatory STOPし、別途の明示的な実行承認を待つ。

起点は指定結果commit `25bbdd33fe758a0e730138cb91ddac8c41b0a298`。元branchのremote SHA一致を確認し、専用branch `track-a-v1-h6-p7q1-compile-readiness-20261016` を作成した。元のV1結果branch・科学SOURCEは変更しない。
利用者指定のGPT科学レビュー `track_a_v1_strengthened_baseline_scientific_review_20261016.md` 第8節の第1優先に従う。文書全体を新しい科学的保証や実行承認とは扱わず、入力文書のbytes/SHAを監査receiptに保存した。

[機械可読の計画・grant案・入力監査](../../artifacts/resource_applicability/track_a_v1_p7_readiness/2026-10-16/README.md)。[固定契約](../../src/trotterlib/track_a_v1_p7_contract.py)、[future compile adapter](../../scripts/track_a_v1_p7_compile.py)、[grant限定runner](../../scripts/track_a_v1_p7_runner.py)、[metadata materializer](../../scripts/prepare_track_a_v1_p7.py)、[人工tests](../../tests/test_track_a_v1_p7.py)。

## 対象と科学的な位置付け

linear H6、間隔1.00Å、STO-3G、保存tol-only DF rank19、12 modes、sector400、保存Hamiltonian/state、T=.8、prefix7、q1、δ=.8、S2、B0+I。対象は **`B0I_p7_q1_2nd` のcosine/sine各1軸、合計最大2軸**だけ。
通常signed signal bias・経験的allowance・P=0の有限時間complex signal benchmarkを維持し、energy/QPE/RPEやFT費用の保証へ拡張しない。

旧再会計では、この候補はε=.05の最良決定論対照となり、PR点費用削減率−13.04%の条件付き逆転を支えていた。ただし費用は固定modelの外挿である。[再会計結果](track_a_v1_reaccount_results_20261016.md)、[Phase A保存signal](track_a_v1_phase_a_run02_results_20261016.md)、[Phase B全4軸の局所検証](track_a_v1_phase_b_run01_results_20261016.md)を参照。

Phase Bが検証したのは別のS4候補であり、今回のp7 S2は未測定。今回もまだcompileしていない。旧model値を実測値として表示したり、実測前に−13.04%を確定した勝敗と呼んだりしない。

## そのまま再利用するSOURCE・入力・phase

科学providerは `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`。旧実行コード17ファイルを指定baseのGit blobと照合し、provider230ファイルをworking bytes/Git blobへ、runtime34,273ファイルを既存hashへ照合した。
snapshotは150,052 bytes、SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`。既存archiveは5,670,237 bytes、SHA `ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010`。今回はbytes/SHAとcodec構造だけを読み、配列やprepared科学objectをdecodeしていない。

Phase A run02の公開gzipをexactに復元し、元bytes/SHAへ照合してp7保存recordをcopyした。選択recordのdigestは `937e886f0dc9cb693365e8f456b0bb2fdb54dfdc351e8ba01b8c9f32829b119b`。Phase A/Bのnative exit0・完了STOP・全回収とPhase Bの全4軸PASS、再会計の正常終了を前提入力として固定する。新しいsignalやshotは計算しない。

将来の回路はarchiveの`faithful:p19`の先頭8 block（one-body＋fragment0–6）を使う新prefix viewであり、存在しない旧discard:p7のpreparation hashを流用しない。既存native builderでT=.8/q1/formula2nd/symmetric_directionalを指定する。
scalarはfull preparationのconstantにPhase Aで保存したfull-Fock tail identity C=`0.01946666126068529`を加え、全scheduleの後にancilla relative phaseを1回だけ適用する。Cの再導出・位相回転によるsignal再生成は行わない。
full-H referenceは保存値real=`−0.8510893550745252`、imag=`0.5250208659461343`のままで、回転・再計算しない。保存allowanceのpolicy・値・非認証という意味を保持する。

compilerはQiskit1.3.0、rz/sx/x/cx、optimization1、seed17、topologyなし。固定hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。full-wrapperのmeasurementを含めてcompileする。
保存アンカーのprofile `622ce4a8…`、archive export時のprofile `4e32bb9b…`、現在runtime closure `850568c8…`を区別する。既存の旧binary output bit同値未確認は取り消さない。

## 固定予測と各軸STOP

固定model SHA `406c6f36f93dac317f9709e68b042c4563edf5ec4b06219e1b60de660b695211` を再利用する。B0+I/S2/cosineとsineのstratumを別々に確認し、各々の保存係数a=797、b=269、d=9613/5を変更しない。
`a+b*q+d*p*q`から、p7q1の予測は両軸とも **72621/5＝14,524.2 RZ/軸**。未測定model値として固定した。

各軸で`abs(72621/5 - measured_RZ)/measured_RZ`を有理数で判定し、**2%以上なら即STOP**。2%ちょうどもSTOPする。合否を軸平均・総費用で判定せず、cosineを合格と確認した後だけsineのwrapper/compileへ進む。
測定値とcompiler/input fingerprintを保存してから判定するため、停止時にも取得済みの証拠を残す。compiler/phase/inputのidentity変更、欠測、非正値/非整数RZ、命令数cap違反、control不一致もSTOP。再fit・追加compile・別候補探索・自動retry/resumeは禁止。

将来のcontrol検査は、native evolutionのancilla0/1 branch各1回と、cosine/sine wrapper各1回の最大4 probes。保存Phase A signalとのnative回路同一性検査だけに使い、保存signal・full-H reference・allowance・shotを置換しない。元signal評価pipeline、matrix exponential/reference solver、PR samplingへ接続する経路はない。
固定のtechnical agreement guard（absolute1e-9＋relative1e-10×max(1,norm)）を維持する。この局所一致を厳密bias certificateとは扱わない。

## 予算・grant案・実行前gate

|call|最大数|
|---|---:|
|snapshot load|1|
|frozen archive decode|1|
|native evolution|1|
|measured wrapper|2|
|native control probe|4|
|compile|2|
|新signal/reference/scalar/primitive/SCF/DF/state/sampling/fit/費用再会計/V2|0|

即STOPを保証するためcompileは1軸ずつ。snapshot/archive/evolutionは共有し、再load/buildを重複させない。外部observerは別processで並行動作する。driver1＋observer1、各thread1、fresh gateで別idle CPUを割り当てる。
未承認資源上限はdriver AS32GiB/RSS9GiB、observer AS/RSS512MiB、総RSS9.5GiB、総CPU3600秒、wall7200秒、output128MiB、stage512MiB。これらは停止上限であり、所要時間や高速化の実測ではない。
保存codec構造からのevolution命令上界5525・wrapper命令上界5529は、既存native100万命令cap内。transpiled命令capは500万。これはメモリ入場用の構造検査であり、新しいprimitive receipt/理論費用計算ではない。

未承認private configと[公開grant draft](../../artifacts/resource_applicability/track_a_v1_p7_readiness/2026-10-16/grant_UNAPPROVED_DRAFT.json)を作成。`approved_by_user=false`、`runtime_authorization=false`、SOURCE=null（承認後に最終commitへ結合）、CPU[]、command=null。**有効grant・fresh実行gate・科学output・one-shot lockは作成していない。**
別途の明示承認後に、新grantだけをSOURCE/config/input/profile/model/p7 recordへ結合する。起動直前30秒以内のCPU idle・既存job identity/affinity・memory/PSI/OOM・disk/inode・外部監視ready・全bytes/SHA gateを要求する。今回の静的資源snapshotを実行直前gateの代わりにしない。
監視間隔1秒・最大gap5秒、CPU/RSS/AS/thread/process/output・PSI/OOM/disk違反でSTOPし、own processだけをTERM/KILL/reapする。旧grant・ledger・carryをrefund/resetしない。

## 実測後の費用比較は別承認

[費用比較計画](../../artifacts/resource_applicability/track_a_v1_p7_readiness/2026-10-16/future_accounting_plan.json)。ε=.05で保存shotはcosine=10216、sine=14356、合計24572。倍率1/10で同じ保存shotを照合したが、新しいshot算術は実行していない。
2軸取得後の式は`G_det=10216*C_cosine+14356*C_sine`、比較先はPR保存確認候補B2 p5/q1 r4/K4の`G_PR=403423200 RZ`。各軸の実測が異なっても、各々のshotで重み付けする。PRは32標本の保存点平均であり、母平均保証へ変更しない。

compile phaseのrunnerには総費用再会計を実行する経路を接続していない。両軸完了・両軸2%未満PASS・入力不変の証拠を報告してSTOPした後、**別途の数値再解析の明示承認**を受けて、この候補と保存PRの比較だけを行う計画。全136候補の再ランキングや他精度への拡張は含まない。

## 人工検証・保全・終了状態

p7専用19件、既存71件と合わせてpinned Pythonで90人工tests PASS。未承認grant/CLIの拒否、範囲・予測・phase/compiler固定、2%ちょうどの停止、1軸目失敗後の2軸目予約/compileゼロ、2軸目を独立判定すること、保存signal/referenceの変更拒否、compiler変更・入力回路mutation・control不一致の停止を人工metadataだけで検査した。実H6 native build/probe/compileは呼んでいない。
これはprimary agent自身によるlocal技術検証であり、別担当の独立レビュー・外部再現・CI全体の成功ではない。初回のmock vectorの軸属性不足はfixtureを修正した。初回の既存人工Permit検査でRSS_CEILINGが1件出たが、その回のRSS値は未保存で原因は未確定。最終確認はpinned Pythonの軽量fork/exec起動で全90件PASS、既存科学コードや資源guardは変更していない。

metadata照合で旧SOURCE17/provider230/runtime34,273ファイル、入力15receipt、H5/H7を含む既存grant/ledger/carry/marker等660ファイルの不変を確認。新科学import・配列decode・native build/control/compile・新grant発行は0。既存SOURCE・raw・結果・V2・H5/H7へ書き込んでいない。
公開成果物は実装・計画・既存公開値のbinding・人工検証receiptのみで、新しい実測結果を含まない。commit・push・remote SHA確認後、**準備完了・未実行・mandatory STOP**とする。
