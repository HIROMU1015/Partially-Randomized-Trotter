# H6 fixed δ：T依存性の実行直前準備

2026-10-15。指定起点 `34efc914b2f485ec694c7d6016025d5c76829be6` から独立worktree/branchを作成した。
**実装・人工検証・byte bindingを完了。新しい実科学計算は未承認・未起動でmandatory STOP。**

SOURCE：`90a3c06d16559d132d8fcb04364868c919c2d546`。
branch：`track-a-h6-fixed-delta-t-dependence-readiness-20261015`。
[固定bundle](../../artifacts/resource_applicability/track_a_h6_fixed_delta_readiness_v1/2026-10-15/README.md)を資料入口とする。
主担当が技術確認した。サブエージェント0、別担当の独立review・GPT詳細科学reviewは実施していない。

## 固定する比較

Linear H6、1.0 Å、STO-3G、12 system+ancilla、保存Hamiltonian/DF rank19/stateを再利用する。
H/DF/stateの再生成、SCF、GPU、共有設定変更は行わない。

| T | q | δ=T/q | PR局所step δ/r | R=qr |
|---|---:|---:|---:|---:|
| 0.4 | 2 | 0.2 | 0.05 | 8 |
| 0.8 | 4 | 0.2 | 0.05 | 16 |
| 1.6 | 8 | 0.2 | 0.05 | 32 |

各TでB0 p15/S2、B0+I p15/S2、full PF p19/S2、full PF p19/global Yoshida S4、PR p5/r4/K2/S2の5構成、計15セル。
精度 .05/.01/.005/.001/.0005、主指標P=0のshot-weighted full-wrapper RZ。
cosine/sineを同じtrajectoryに対応させ、RZ/CX/RZ-depth/CX-depth/total-depth/sizeを保存する。
固定panelの決定論的最良との比であり、各Tで全候補を最適化した結果と扱わない。

B0+Iは既存full-Fock discarded-tail scalar Cに対する exp(-iTC) をnative ancilla位相へ一度適用する。
full-H referenceを回転しない。Cは保存p15受領値 `1.1246847212849682e-7`、元scalar誤差receiptを結合する。
補正signalとexact truncated-H signalを回転し、unrotated referenceの経験的不確かさを別に加える。
局所δ/rが同じでも、log B=R log bとoccurrence総数は増える。PF、finite-RTE、discardのsigned complex成分を分ける。
絶対誤差の加法性は仮定しない。既存binary64独立Horner/forward・occupation/expm/eigh比較とempirical allowance規約を維持する。

## 既存T=.8の再利用

保存signal92とD0再解析commit `8dbdb68293c988a999645e21a82b53966f9ef6cd` から固定q4の4構成を抽出した。
PR p5/q4/r4/K2は探索8、独立確認32の両方が保存されている。主比較は確認32、探索8は別layer。
B0+I p15/q4は結果起点commitに含まれるSOURCE `ac6d28e35cf01398f169ce59577182a38da9b959` の2軸実測を再利用する。
旧B0Iの他prefix/qを新panelへ混ぜず、元全12構成の最適化headlineも転記しない。

H/DF/state/input SHAとcompiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` を既存receiptへ照合した。
[保存値projection](../../artifacts/resource_applicability/track_a_h6_fixed_delta_readiness_v1/2026-10-15/T08_saved_panel_v1.json)に元phase/sample count/allowance/欠測を保持する。
B0I実測metricsは[元のaxis表](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/axis_compiled_metrics_v1.csv)のp15/q4両軸を参照する。
科学providerは `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` のnative lowering、旧science SOURCEは `5a903aea2492c2d211c1e15e7737df0d50694c12`。
旧ローカルhostとserver compiler outputのbit完全同一性は未確認。元のroundoff受理済み準備archiveと現在profileを固定し、新旧gate数を無根拠に同一視しない。

T=.8の新reference/signal/trajectory/compileは0。PRのε=.0005でのINELIGIBLEを0費用として埋めない。
新Tでは全固定回路を取得する上限を用意するため、費用取得と精度適格性を区別できる。

## 新しく取得する範囲と高速化

T=.4/1.6だけに新規reference各1、基本signal各4とB0I analytic overlay各1を用意する。
PRは各T独立32標本、計64 trajectory。今回は候補選定がないため新探索8を追加せず、固定構成の独立費用推定に直接進む案である。
元の探索・確認・条件付きpilotと異なるnamespace/64個のseedを結果前固定した。
新条件付きpilotは追加せず、全Taylor次数を元の無条件finite lawからsampleする。

| 新規処理 | 上限 |
|---|---:|
| signal worker | 2（Tごと、各thread1） |
| full reference sector matvec | 800（400×2） |
| primitive probe | 750（375×2、負Yoshida時間を含む） |
| 独立PR trajectory | 64（32×2） |
| sampling API call | 320（32×q2 + 32×q8） |
| RTE events | 1280（32×R8 + 32×R32） |
| 決定論的cost group | 8（4×2） |
| 全cost group | 72 |
| full-wrapper compile | 144（cos/sin各72） |
| native branch/measurement probe | 40（各Tの5構成×4） |

既存の機構を以下のように再利用する。

- 受理済み準備archiveをdecodeし、DF基底・係数を作り直さない。signal worker内で4構成のprimitive spectra/native actions/tailを共有する。
- 独立metadata coordinatorが最大8個のdisposable group workerを監視する。科学処理のGILで観測を止めない。
- 同一groupのcos/sinは同じsampled stepsと1個のevolutionを共有し、両軸分を再sample/buildしない。wrapperとcompiled circuitは各軸完了後に解放する。
- 既存content-checked canonical fingerprint再利用を有効にする。最大32MiB/worker、guardは毎回current graphを確認し、mutable object IDを永続keyにしない。
- 大きな環境file inventoryは起動時に一度照合する。workerごとには軽量sourceと入力/archive bytesを確認する。
- full S4は2groupを単独workerで順に取得し、その後normal 70groupを最大8並列とする。heavyとnormalを同時投入しない。

標本数、compiler seed17/opt1/basis rz,sx,x,cx、backend/coupling/layout/routing=None、symmetric_directional/raw concatenationを変えない。
高速化率・新Tのpeak memoryは未測定。今回実worker stress、科学circuit build/transpile、性能campaignは0。

## 環境・資源案

既存private Python：`/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python`。
Python3.11.0rc1、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、Numba0.60.0、OpenFermion1.6.1、PySCF2.7.0。
SOURCE5file/provider230file/environment34,273file/input/archiveのbytes/SHA照合PASS。既存venvへのinstall・設定変更0。

候補CPU：worker [20,21,24,25,26,27,28,29]、metadata coordinator22（同一processがobserverを兼ねる）。
各数値role AS32GiB、normal/signal RSS9GiB、heavy RSS16GiB。normal合計RSS76GiB/job RAM96GiB。
coordinator soft AS2GiB/hard32GiBを継承し、子がAS32GiBを設定できる。heavy/signal合計RSS22GiB。
起動時host available248.25GiB/effective96GiB以上、filesystem10GiB/100,000 inodes以上を要求する。
候補CPUは予約・使用許可ではない。直前の1秒負荷sampleでbusy10%超やphysical sibling重複なら起動しない。

read-only時点ではeffective available約467GiB、home filesystem空き約297GiB、約2.19億inodes。
user/group/project quotaは無効と観測した。値はsnapshotであり、起動直前に再検査する。
host full PSI 1%以上、可視cgroup full PSI非zero、新OOM増分、1秒観測の5秒超gap/duration、role/aggregate上限でfail-closed。
既存user.slice oom_kill=1は過去baselineとして保持し、新増分と区別する。

出力は全stage/control/temp/block余裕込み8GiB案、stage毎2GiB、各worker64MiB、journal128MiB/stage、log8MiB/stage。
100,000 available inodesのadmission、通常計画約30,000、trace/progress/cache込み保守的90,000以内。full-QPYや巨大compiled cacheを保存しない。
private runtime/cache/controlはGitへ入れない。完了cost/trajectory/progress/resource/native終了記録はhome内で保存する。

旧T=.8のfull S4/q8は一軸約616秒、RSS約7.93GiBという実測がある。新Tの実測とは異なる。
全体2〜8時間は旧caseの工程/8並列とsignal/native検証を見込む工学的予測であり、完了保証ではない。
worker CPU cap86,400秒、wall capはnull案。停止時に自動retry/resume/次stageへ進まない。

## 累積会計と承認対象

旧条件付きpilot終了時known compile1716/trajectory842、保留予約20/10、合計上限1736/852を保持する。
その後のnative B0Iはknown compile24とunknown予約6を追加した。旧failed run03の未記録actual/exitは補完しない。
compile carryは保守的1766、trajectory carry852、sampling call carry3264。
新144/64/320を加え、**累積上限案compile1910、trajectory916、sampling calls3584**。
予算返却・リセット0。新grantはSOURCE/config/carry digestと上記上限・stage demandに結合し、旧grantを拒否する。
[会計詳細](../../artifacts/resource_applicability/track_a_h6_fixed_delta_readiness_v1/2026-10-15/budget_and_carry_v1.json)を参照する。

残る一括承認は、この15セル固定panel、新規2Tのsignal/reference、PR各32、計144 compile、資源案、累積上限案と一度の起動である。
実行許可の採用後に新authorizationだけを発行し、固定SOURCEは変更せず最終bindingとfresh gateを確認する。
現在 approved=false/runtime_authorization=false/allowed_cpus=[]/absolute_launch_command=null、grant未発行。
技術的metadataの結合は済んでいるが、科学実行planの最終sealは新承認とfresh gateに結合する。

## 実行手順（未実行）

最終private config：
`/home/AbeHiromu/projects/h6-handoff-evidence/20261015/h6-fixed-delta-readiness/materialized03/config.json`。
output01/control01は未使用。grant/source/input/profile/carryの最終整合後、native waitを記録するown launcherから次を一度だけ実行する。
新authorizationは承認後に作成するもので、この段階では存在しない。

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 QISKIT_NUM_PROCS=1 QISKIT_PARALLEL=FALSE \
  /home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -P -B \
  /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-fixed-delta-t-dependence-readiness-20261015/scripts/h6_fixed_delta_runner_v1.py \
  --execute \
  --config /home/AbeHiromu/projects/h6-handoff-evidence/20261015/h6-fixed-delta-readiness/materialized03/config.json \
  --authorization /home/AbeHiromu/projects/h6-handoff-evidence/20261015/h6-fixed-delta-readiness/materialized03/authorization_approved.json
```

起動gateは新grant/CPU/環境/全入力/carry/未使用rootを先に照合し、新controlのexclusive作成をone-shot claimとする。
子はrun-local grant/claim/stage/task/CPU/SOURCEと親start ticksを再照合し、PDEATHSIGを設定する。
各groupは成功terminal、paired axes、task/event/compiler/metrics一致後だけ受理する。失敗時は追加dispatchを止め、own unreaped groupだけを終了・waitする。
最初のSTOP理由と全予約を保存し、すべて完了してもmandatory STOP。旧partialからの自動resumeは実装しない。

実行後は15セル×5精度についてbias/u/logB、axes別shots、回路6metrics、paired総費用平均/SE、限定panel内のPR/最良deterministic比を集計する。
不適格・未測定はnull、母平均保証はfalse。T×精度の費用比図とbias/normalization/shots/circuitの分解図を保存値から作成する。
この結果集計・図はまだ実行していない。実測がないTの優位性は結論しない。

## 検証と限界

最終46件のstdlib人工/mock tests PASS。scope/64 seed/72group/144 compile、δ/R、3Tの位相・full reference不変、error closure、旧grant/改変carry/上限拒否、RSS/親hard AS、partial pair拒否、exclusive出力を含む。
既存providerへのAPI importとmetadata plan接続PASS（科学数値呼出0、thread1）。check-config exit0でSOURCE/provider/profile/input byte binding PASS。
同じ主担当による技術確認であり、独立review・外部再現・CI・科学成功とは呼ばない。
中間SOURCE用の公開bindingが既存だったため一度exclusive writeが拒否された。中間資料をprivate homeへ保全し、新materialized03に最終configを作成した。science0。

未確認なのは新Tの実signal/controlled-wrapper一致、実測速度とpeak RSS、rare Taylor次数の有限標本coverage、母平均費用、旧hostとのbit同一性。
32標本で未観測の次数の費用を0扱いしない。旧s2 pilotはq8/r2で今回のr4とは異なり、その費用を流用しない。
B/shotの増加と平均回路費用の変化を分け、PR優位がなくなる結果もそのまま報告する。
新しい科学定義・候補最適化・H8計算・GPT詳細reviewは開始しない。準備公開後にSTOPする。

## 2026-10-15 利用者の開始指示・実行中

利用者「計算を開始して」を、直前のscope/資源/累積上限案の実行承認として固定した。SOURCE `669842c85e2c04a90dcdcfb774b323a2737cb8ba`、run `h6-fixed-delta-t-panel-20261015-run02`、coordinator1603119／signal workers1603135,1603136。承認済みCPU予算内で空いているworker34/35/36/37/39/40/41/43・driver44へ自runだけを配置した。

初回CPU admission停止はscience0/one-shot未消費。後のbootstrapはsnapshotパスを文字列で渡し、NPZ array読込前のAttributeErrorで停止した。両workerのexit1/reap・native wait1と予約を保持し、Path変換の1行修正＋人工47件PASS後に新SOURCE/新outputへ結合した。数学・input/compiler・64trajectory/144compile・累積1910/916/3584は不変。

最後のfresh CPU/memory/pressure/OOM/filesystem/inode/quotaとSOURCE230provider/34273environment/input/carry byte bindingに合格して起動した。[開始確認](../../artifacts/resource_applicability/track_a_h6_fixed_delta_readiness_v1/2026-10-15/execution_start_v1.json)。2signal worker/thread1/CPU時間増加・STOP理由なしを確認。完了証拠ではない。runtimeはprivate homeのmaterialized05、native logはlaunch02。旧grant/証拠を保持し、共有設定・他job・venv変更0。runnerの自動retry/次stageは無効のまま。

## 完了結果

この準備・起動記録は履歴として保持する。[最終結果・図・終了監査](track_a_h6_fixed_delta_t_dependence_results_20261015.md)に144/144 compile完了を記録した。新stage未起動、mandatory STOP。
