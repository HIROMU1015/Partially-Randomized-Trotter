# H8 native B0+I 全登録候補：実装・実行準備

**利用者の「4ワーカーにして」に従い、2構成・最大4軸の並列compileへ更新した。限定人工検査45件PASS、SOURCE `54ed98c5fa5b7190f77255271f4e3bec8db15630`。H8の新scalar/signal/shot会計・native build/probe/compileは未認可・未実行。**
[契約案・入力監査・tests・closure入口](../../artifacts/resource_applicability/track_a_h8_native_b0i_readiness_v1/2026-10-15/README.md)。
新worktree/branch `h8-native-b0i-readiness-20261015` / `track-a-h8-native-b0i-readiness-20261015` をH6結果commit `34efc914b2f485ec694c7d6016025d5c76829be6` から作成した。
H6結果branchのorigin SHAも同commitと一致したため、H6の手動pushや追加計算は不要。既存H6/H8実行worktree・source・grant・証拠は変更しない。

## 科学範囲とH8固有の接続

対象はlinear H8/1.00Å/STO-3G、保存tol-only DF1e-8/rank28/cutoff0/weighted Hermitian projection/signed generation order、16 spin modes、Na=Nb4、sector4900。
T=.8、S2、prefix L_D=7/14/21 × q=1/2/4/8、δ=.8/.4/.2/.1、symmetric_directional、12構成・最大24測定軸を固定する。
compilerはbasis rz/sx/x/cx・optimization1・seed17、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` を保持する。

B0+Iはdiscarded fragmentのfull-Fock Pauli identity
`C = Σ_tail λ/4 * ((Tr G)^2 + Tr(G^2))`
を取り、native final scalarを`original_constant+C`へ一度だけ置換する。
固定sector4900のtraceへ置き換えず、signed λ・生成順・retained blocksと内部phase・ancilla相対位相を保持する。
保存B0 corrected/exact-truncated signalだけを`exp(-iTC)`で回転し、full-H referenceは回転しない。
この実入力C/位相/shot計算は今回行っていない。

H6のmath helper6関数はAST body一致を確認した。H8の保存signalも固定provider内の同じ`h6_matched_accounting_v1.empirical_allowance`を使用していた。
元の経験的uをB0 signalとunrotated referenceの各寄与へ保守的に適用し、回転・stored binary64 scalar算術の誤差項を追加するH6方針を保持する。未知の真の数値誤差や化学モデル誤差のcertificateにはしない。

H8専用adapterは既存`load_h8_snapshot`とbasis bridgeをlazy接続し、16 modes/DF28/sector4900を確認する。H6の12/19/400は拒否する。
新SCF/DF/state生成やexponential reference/PR trajectory生成は実装経路に追加しない。保存独立forward-overlapを回転した参照とnative control/X-Yを照合する。
実16-mode B0+I回路の正しさやcompile成功はまだ未検証である。

preparationは入力/provider/profileに結合した固定semantic recipeを12件保存した。
qに依存しないprefix preparationは最大3回だけ行い、生成順prefix、optional one-body-first、unoverlaid scalarを確認する。
観測`preparation_hash`を実行中にexpectedとして学習せず、mutation/provenanceにだけ使う。recipe hashはnumeric回路fingerprintではない。
各wrapperのnative検証後のnumeric SHAとcompile前後SHAは別に必須とする。H6の旧profile preparation hashをH8へ転用しない。

## 入力・provider・環境候補

provider SOURCEは`c31ef424175c02379eafebe8fc6c7548c1d9aabe`、265 source fileについてGit固定bytesとdisk SHAが一致した。元providerは変更していない。
凍結NPZ、snapshot/DF/projection receipt、signal_summary、固定D0候補会計、旧wrapper metadataの7件・合計6,460,006 bytesをstreaming byte SHA照合した。
NPZ array読込は0。保存metadataからB0全12件、独立forward reference12件、logB=0/b=1・EMPIRICAL/certified=falseを確認した。新しいC・bias・eligibility・shotは計算していない。

H8 D0固定commit `3d910ee1de4597b8895e8506500553dbad5a5085` の`H8_CONFIRMATION_UNION6`（26候補×10条件＝260行）を比較入力にする。
PRはunion6の独立32 trajectoryだけを用い、探索n8やH6結果を混合しない。
比較分母は適格なB0/B0+I/full S2/full S4最良、主比較はP=0のshot-weighted RZ、他5metricと仮想共通Pは補助解析。

環境候補は旧H8と同じprivate Python3.11.0rc1/Qiskit1.3.0を使う無変更venv。
固定profile digest `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`（34,273 file）の参照を保持し、起動前にmaterializeしてfresh byte検査する。
今回install/upgrade/新venv・共有設定変更は0。旧host compiler outputとの完全bit同一性は未確認のまま。

## 実行順序・最初のnative確認・静的上限

全12構成×ε=.05/.01/.005/.001/.0005×factor1/10をscreenし、全10条件で確定不適格ならbuild/probe/compileを省略する。未解決は不適格へ丸めずSTOPする。
最初の実native確認は「固定prefix/q registry順で最初に適格となる1構成」。対象の選択規則と4probe上限をSOURCEと契約案で事前固定し、今回実データから対象を選んではいない。
最初の適格構成から登録順に最大2構成をまとめ、各構成のcontrol-left/control-right/X/Yを順番に確認する。まとまり内の全構成PASS後にcosine/sine最大4軸を4 workerでcompileする。最初の構成はnative検証でも先頭に保つ。2番目の検証が失敗した場合も、そのまとまりのcompileは0でSTOPする。
各まとまりのcompileとcleanup後に親の大きな回路を解放する。最後に1構成だけ残る場合は2 workerだけを使う。親が保持するnative回路は最大2構成で、回路生成・正しさ検証そのものは逐次である。

旧2 worker SOURCE `65a9d801d297ba5402a4fe647d1ece4d5a60ae93` とREVIEW `7a113b33d88c5862014f15ad4eb9bdf84e227a8a` はGit履歴に保全した。後続のまとまりで検証が失敗した場合は以前の費用を履歴として保持するが、比較COMPLETEにはしない。
全対象が終わった場合だけ12構成のcoverageと費用欠測を検査し、登録範囲内の6metric/P比較を保存する。

|実操作|最大|適格な構成がn件の場合|
|---|---:|---:|
|保存snapshot load|1|1|
|prefix preparation|3|3|
|tail scalar / analytic overlay|各12|各12|
|native evolution build|12|n|
|measured wrapper / probe copy|各24|各2n|
|control/X-Y probes|48|4n|
|compile|24|2n|
|新SCF/DF/state/PR trajectory/reference exponential action|0|0|

n=0〜12。最初のcase検証失敗ならcompile0。旧実測費用の自動B0+I reuse、matched B0再測定、q16、prefix範囲拡大、次stageは認可しない。
まとまりごとに重複しないcleanup receiptを保存し、wait4 CPUを次waveへ加算する。observerも最大4 workerのidentityを確認し、exact完了carry＋未完了taskの観測CPUとして計数する。リセットと二重計上を避ける。
own PID identity/pidfd、ESRCHだけの退出扱い、first STOP、bounded TERM/KILL/reap/pipe closeとPDEATHSIGを保持する。

## 資源・容量案：未承認

|項目|未承認案|
|---|---|
|数値driver / worker|1 / 最大4、各thread1、freshな異なるphysical coreを5つ選ぶ|
|observer|既存metadata launcher1、driver core共有、1秒観測・5秒欠測STOP|
|数値role AS / RSS|各64GiB / 32GiB|
|metadata AS / RSS|512MiB / 256MiB|
|memory admission案|5×64GiB＋.5GiB＋headroom16GiB＝336.5GiB（各AS上限の保守的な合算。実測RAM消費・確保量ではない）|
|CPU上限|driver＋全子の当stage累積86,400秒、metadata1,800秒|
|phase / total wall|null / null|
|pressure / OOM|host full PSI>=1%・cgroup PSI非zero・OOM増分等でSTOPする案|
|scientific JSON / combined logs|64MiB / 512MiB|
|journal / terminal|最大約1MiB予約 / 1MiB別予約|
|temporary/pending / block-inode余裕|2GiB予約 / 512MiB予約|
|FS / inode admission|fresh8GiB / 40,000 inodes、output20,000 inodes|
|quota|有効ならfresh effective user/group/project quotaを照合し、証明できなければSTOP|

H6の資源認可をH8へ自動移送しない。host PSI1%案を含め、上記は新しいH8契約での承認対象である。
64GiBは各数値processの仮想アドレス空間上限、32GiBは別のRSS上限案であり、同時に上限まで消費するという予測ではない。物理RSS上限の単純合計はdriver＋4worker＋metadataで160.25GiBで、共有ページの重複を含み得る。336.5GiBは従来の保守的な起動判定方式を4 workerへ換算した未承認案として保持し、実測必要量と混同しない。
[旧2 worker案採取時のread-only host snapshot](../../artifacts/resource_applicability/track_a_h8_native_b0i_readiness_v1/2026-10-15/read_only_resource_snapshot_v1.json)は履歴として保全し、4 workerのfresh gateやCPU認可には使わない。
temp/control全treeの独立hard byte capは未実装で、容量予約・FS headroom guardと区別する。追加の共有quota設定は行わない。
H8実速度・native最大RSS・compile完走時間は未測定で、この予算での完走を保証しない。

## 人工検査・固定・残る承認

単一process/thread1、AS2GiB、RSS512MiB観測上限、CPU/wall180秒、log16MiB、home配下の専用fixtureで45件PASS（旧38件に4 worker対応7件を追加）。
tiny2-mode行列、string回路、mock provider/worker/16-mode traitsだけを使用し、実科学import・NPZ・fork/execをharnessで拒否した。
旧2 worker時の第一試行は旧FakeMonitorに初期child CPU fieldがなく9件失敗し、fixtureを補正した。4 worker更新時の最初の2試行は新observer fixtureのlog上限/output field不足を補正し、3試行目で45件PASSした。CPU繰越・実行中task加算、全screen障壁、first/late native失敗、scope/input/source/compiler/recipe改変、費用proof、confirmation n32、非finite/非Hermitian・負λ・phase・cleanupに加え、4 CPU割当、奇数件の最後2軸、最大2回路保持/解放、2番目の検証失敗、4番目のworker起動失敗時の回収、4 identityのobserver記録を検査した。
本番CLIへ未承認draftを渡す拒否検査もPASS（exit2、output/one-shot作成なし）。最初の相対path指定もcanonical home path guardで拒否されたことを保全した。
実H8回路・compiler equivalence・実fork/pressure stressは検査していない。主担当の限定確認であり、別担当の独立reviewや新GPT科学reviewではない。subagents0。

終了terminalに全出力bytes/SHAを保持する補正と費用coverage回帰も追加した。
code/testsを上記SOURCE、監査・契約案・研究ノートを別REVIEW_BUNDLEへ固定する。
`approved=false`、`runtime_authorization=false`、`grant_created=false`、`execution_plan_sealed=false`、`allowed_cpus=[]`、`absolute_launch_command=null`。
今回の4 worker化は実装変更の承認であり、本計算の実行承認ではない。次回は固定SOURCE・H8新会計/最大24compile・最初の4probeと各caseのnative確認・4 worker/profile/compiler候補・CPU/資源案について明示実行承認を受ける。
その後、旧H8のknown/held/unknown carryを返却せず、新output/control/one-shot・font metadata seed・fresh input/profile/provider/CPU/memory/pressure/OOM/容量/inode/quotaを結合して一度実行する。
完了またはfail-closed STOP後にmandatory STOPし、新しい科学的解釈が変わる場合だけ別途科学reviewへ戻す。

今回の新実科学計算・新grant・GPU query/use・共有環境/他job変更・原稿改訂は0。準備資料はローカル固定であり、未pushを公開済みと扱わない。
