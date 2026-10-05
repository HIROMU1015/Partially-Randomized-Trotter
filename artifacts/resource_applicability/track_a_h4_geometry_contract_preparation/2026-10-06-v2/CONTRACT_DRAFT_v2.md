# Track A H4 geometry 契約v2・レビュー採用案

`H4_GEOMETRY_CONTRACT_V2_PREPARED_AWAITING_REVIEW_SCIENCE_NOT_AUTHORIZED`

起点commitは `7c1a3d43f61c5501a9e79206b7c60933f94b1077`。
本書、[設定案JSON](review_decisions_v2.json)、[zero-compute plan](zero_compute_plan_v2.json)、
[二段階認可契約](stage_contract_v2.json)を一組としてレビューする。
数値は具体的な採用案であり、条件の最終承認・科学実行authorizationではない。
D1〜D4のレビュー承認は未解決。契約完成、source port完了、実行可能planとは宣言しない。
science/source port/input generation/next stageの認可はすべてfalse、plan未seal、research_decision=null、mandatory STOP。

## Scopeと旧証拠

H4 linear、STO-3G、requested/actual DF rank12、8 system qubits、**ancilla 1個**（index 8）、合計9 qubits。
追加距離はcanonical Angstrom strings `0.70/0.80/0.90/1.10/1.40/1.60`。
H_i=(0,0,(i-1.5)d) Å、i=0..3、原子ID順を固定しdecimal dを一度だけbinary64へ変換する。
T=0.8、二次DF-prefix PF、canonical finite-RTE、delta=T/q、q=1/2/4/8。
L_D=0/3/4/5/6/9/12。登録済み218 templateをそのまま移す。
B0 20、B1 4、B2 145、B3 49、random194とbaseline24。既存r64二件だけを含み、追加・除外・置換をしない。

randomは各32 trajectories、cosine/sineは同trajectoryを共有する。
1点194×32×2+24×2=12,464 logical wrappers、6点74,784、signal slots 1,308。
actual science transpile reservation/invocationも最大74,784。cacheはlogical sample/weightを減らさない。
accuracy-ineligibleを監査に保持し、該当shots/workはnull。追加96、anchor、8点化、GPU/実shotsは禁止。
旧1.00 Å218候補、旧1.30 Å元5構成は別identity layer。fresh blind、厳密winner、最終RPE総costとは呼ばない。

旧v1の26 bundle filesと旧manifestはbyte-identicalのまま別日付ディレクトリにv2を追加する。
v1 manifest33件は**起点commitのblob**で照合する。更新する索引・研究概要・当日ノートはv2 manifestだけへ収録する。
v1 manifestの索引hashを現在のworktreeに合わせて書き換えない。
旧247 source、公開準備25 files、保存6 JSON、旧status/validation manifest、原稿・図・Track Bを保存する。

## D1：SCF/DF採用案

installed PySCF2.7.0、OpenFermion1.6.1、OpenFermion-PySCF0.5の**source text**を根拠とする。
config-sensitiveなclass defaultsの存在を確認したので、将来の別sourceでは明示値と読み戻し検査を必要とする。
今回の環境でSCF objectを生成したりdefaultをimportして確認したりしていない。

- RHF、charge0、multiplicity1/spin0、symmetry=false、STO-3G、Angstrom、cart=false、incore_anyway=false。
- initial guess=minao、dm0=null、geometry間density/guess移送なし。minao内部の別guess fallbackや例外もSTOP。
- 通常のCDIISを初回から使用：enabled=true、space8、開始zero-based cycle1、damp0、rollback0、diis_file=null。
  これは通常の収束手順である。不収束後のDIIS変更、Newton、restart、別guess、cycle延長による救済はしない。
- conv_tol=1e-9 Ha、max_cycle=50、conv_tol_grad=sqrt(1e-9)=3.1622776601683795e-5、raw gradient L2。
  tight-gradient規約をprocess内で確認する。damp0、level_shift0、direct_scf=true、direct_scf_tol=1e-13。
- check_convergence=null、conv_check=true。PySCFの通常main-cycleはenergy差とgradientのAND、
  収束後extra確認は10倍energy/3倍gradientのORを使う。この後者だけを合格根拠にしない。
  mf.converged、main-cycleのabs(delta E)<1e-9かつraw gradient<sqrt(1e-9)、
  extra確認後のraw gradient<sqrt(1e-9)、有限energy/MO/integralsを独立gateとする採用案。
- PySCF advisory max_memory=4000 MB、chkfile=null。これはAS/RSS上限とは別。
  旧run_pyscfはcontrolsがimplicitで最後にsaveするため、そのままfuture runnerへ流用しない。

全4 spatial orbital・all-electron、frozen core/active-space削減なし。AO hcore=T+V_nuc、S、Coulomb ERI、
real canonical MOでC.T@hcore@C、ao2mo.kernel→restore(1)→transpose(0,2,3,1)のOpenFermion (ps|qr)規約。
nuclear repulsionを含み、MOは昇順energy・PySCF generalized scipy.linalg.eigh(h,S)。
最初の最大絶対real AO成分を正にし、localization・geometry trackingをしない。
MO relative energy gap<=1e-12の縮退はSTOPする採用案。
spin orbitalは2p=alpha、2p+1=beta。OpenFermion既存EQ_TOLERANCE=1e-8 zeroingとInteractionOperatorのtwo-body factor1/2を保持し、新screeningを導入しない。

DFはspin_basis=true、final_rank=12、truncation_threshold=1e-8（final_rankがrank選択を上書き）。
installed sourceのreal/symmetric tensor absolute L1検査1e-8を保持する。
返されたlambda/Gのcountが厳密に12、全有限であることを要求する。
**actual rank12は返されたfragment数であり、微小固有値を除いた代数的rank12の主張ではない。**
zero/small lambdaを変更・padding・再screenせず、count不足、SCF不収束、gate不合格はSTOP。
今回の案は分子計算では試していない。

## D2：順序・縮退・sector・solver

旧S0/M1のgeneration_ranked_fragmentsのenumerationを継承し、
OpenFermion生成のabs(lambda)*(sum(abs(G_spin)))²とnumpy.argsort(weights)[::-1]の返却順を保持する。
汎用関数のFrobenius再ソートを使わない。全16 raw eigenpair indexと全返却permutationを保存する。
equal-weight sortingがstableとは主張しない。tieがあっても返却順を別の安定sortへ置換しない。
有意な同重みが予定prefix3/4/5/6/9/12の境界を跨ぐ場合はsignal/cost前STOP。
ここで有意性は下記lambda relative閾値に従う。その他のtieとnull tieは一回の返却順をfreezeする。

DF eigenvectorはreal spatial成分のC-flatten順で最初の最大絶対成分を正にし、Gにも同符号を適用する。
G²を変えない符号固定案だが、新bytesを旧snapshotと同一とは呼ばない。
full16 eigenpairのrelative gap |lambda_i-lambda_j|/max(1,max|lambda|)<=1e-12で、
少なくとも一方がrank12に含まれ、少なくとも一方の|lambda|/max(1,max|lambda|)>1e-12ならSTOP。
null縮退はpinned solver一回のbasis/permutationを符号固定してfreezeし、回転・再試行・交換しない。
分類閾値でsmall lambdaをゼロ化・削除しない。host間の縮退basis再現性は未保証でありレビュー事項。
別案のcanonical degenerate-projector basisを採るなら別の結果前契約とsource検査が必要で、今回は実装しない。

sectorはNalpha=Nbeta=2、N=4、dimension36。
OpenFermion JWではspin-orbital pがinteger bit(7-p)、該当alpha2/beta2 occupationを8-bit整数昇順にする。
Qiskitへの8-bit reversalを明示し、qubit pはlittle-endian。sector並べ替えを結果後に変更しない。
dense complex128 36×36のnumpy.linalg.eigh(UPLO='L')を採用案とする。
元HのHermiticityを確認してから(H+H†)/2を解き、昇順固有値index0だけを選ぶ。
phaseはordered sectorで最初の最大絶対振幅をreal positiveにし、同phaseをfull stateへ適用する。

| gate | 採用案 |
|---|---|
| abs(norm2(psi)-1) | <=1e-12 |
| 元HとRayleigh real Eのresidual L2 | <=1e-9 Ha |
| spectral-norm(H-H†)/max(1,spectral-norm(H)) | <=1e-12 |
| abs(Im(psi†Hpsi)) | <=1e-11 Ha |
| sector gap E1-E0 | <=1e-10 HaならSTOP |
| outside-sector振幅・full/sector一致 | absolute max<=1e-12 |

縮退、gate不合格で別state/prefix/rank/thresholdを選ばない。上記追加のDF/MO縮退1e-12とnull保持規約はレビュー未承認。
原子・積分・MO・full DF・order・sector・stateのbytesは一回の承認済み入力生成後にfreezeしてSTOPする。

## D3：seed

master seed採用案は20261006。actual master seed、actual source/input identityはnullのまま。
旧domain h4-trajectory-v1、axis共有、step/occurrence別domain h4-step-occurrence-v1、
geometry/H/DF/state/template/source/compiler/environment/semantics/index bindingを維持する。
軸・epsilonをtrajectory seedへ入れず、duplicate seedはSTOPで救済seedを作らない。
今回は実trajectory seedを一件も生成しない。検査に使うseed7とhashは既存の架空fixture専用。

## D4：admission・AS・RSS・wall・output

GiB=2³⁰ bytes、worker AS8 GiB、driver AS8 GiB、fixed host headroom16 GiBを採用案とする。
own-process RLIMIT_ASとRSS監視を分離し、PySCF advisory max_memoryとも区別する。
ASはvirtual address spaceの上限、RSSは実際のresident memory。104 GiBは最大12 workers＋driverのAS budget和であり、physical RAM予約ではない。

```text
required_available(w) = 8 GiB + w * 8 GiB + 16 GiB
1 <= w <= min(requested_workers, 12, explicitly_allowed_cpu_count)
effective_available_bytes >= required_available(w)
```

開始前に条件を満たす最大整数wを選ぶ。64 GiBなら最大5 workers、12 workersには120 GiBが必要。
最小1 workerにも32 GiBが必要で、足りなければSTOP。観測availableを資源予約と呼ばない。
host /proc/meminfo MemAvailableのkB×1024と、有限のcurrent cgroup memory.max-memory.currentの最小をeffective availableとする。
値欠落・負値・malformed・5秒より古い観測はSTOP。
CPU permissionは明示許可集合と/proc/self/status Cpus_allowed_listの交差。nprocやaffinityだけから利用許可を推定しない。
今回のallowed CPU countはnullで、実hostのworker admissionを実行していない。

開始後はown process treeのVmRSS、effective available、memory PSI/cgroup OOMを最大5秒間隔で監視する案。
available<16 GiB、role RSS>8 GiB、admitted own total超過、allocation/AS失敗、PSI full avg10>0、OOM delta>0ならown runを停止してレビューへ戻る。
worker減少は**開始前**だけ。開始後のpressureを暗黙retry/resume/restartにしない。
他ユーザーのjob/priority/affinity、共有cgroup、swap、OS/CUDA/package/settingsを変更しない。

wall停止上限72時間はgeneration＋mapの累積time.monotonic消費時間として引き継ぐ（停止間のdowntimeではbudgetをresetしない）。完了ETAではない。
output上限10 GiBはsnapshot/log/record/manifest/tempを含むown run総量。書込前にbyte budgetを確保し、超過する次の書込はSTOP。
旧outputを消してbudgetを空けず、symlink escape・上書きを拒否する。

- fixed run ID案：`track-a-h4-geometry-v2-20261006-run01`
- absolute project root案：`/home/AbeHiromu/projects/partially-randomized-trotter`
- absolute output root案：`/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run01`

exclusive run registrationとowned monitoringはfuture sourceの別検査が必要。今回、このoutput directoryやregistryは作成・statしない。

## 二段階の認可・seal順序

[machine-readable順序](stage_contract_v2.json)は以下の順を飛ばせない。今回どちらのauthorizationも発行しない。

```text
契約v2レビュー・条件承認
  → 別指示でnew science source/runner/testsを実装
  → actual source commit/hash固定
  → source-bound入力生成専用plan
  → 入力生成専用の別authorization・review・明示launch
  → 承認6距離の入力生成だけ
  → H/DF/state/sector/order/coordinate bytesをfreezeしてSTOP
  → input-bound signal/compile planをseal
  → 別result-prior signal/compile authorization・最終review・明示launch
  → signal/compile/resource map
  → mandatory STOP
```

生成前に新input hashはない。生成専用source-bound planはそれを前提とし、input-bound sealedを名乗らない。
入力生成認可はsignal/sampling/science circuit build/compileを一切認可しない。
入力freeze後も別認可・review・launchまでsignal/costを取得しない。
source実装段のsynthetic semantic testsと、認可後のactual-input science gateを区別する。
実signal/costを先に取得して「意味論gateの準備」と呼ぶ経路を拒否する。
生成結果は明示STOPし、signal stageに認可を流用しない。plan/source/input/authorizationの変更は別レビュー。

## Identity・cache・検査の限界

v1 record wire format、wrapper key、numerical full declaration、全global phase/parameters、
candidate/axis/H/DF/state/source/compiler/environment/semantics、外部completion digestと独立registryを維持する。
COMPLETE非cache ownerだけ、同scopeかつ数値fingerprint完全一致だけ再利用可。seed/indexの違うsampleもweight1/32を保持。
baseline seed/indexは両方null。自己参照・cache chain・欠測owner・RESERVED/AMBIGUOUS・digest改変は拒否する。
compile前reservation、record/ledger atomic IO、消費済みAMBIGUOUS STOPのv1規約を変更しない。

新validatorはpure JSONの**レビュー用**で、起動controller・authorization emitter・atomic runtime・科学serializerではない。
段階例はARTIFICIAL_NOT_AUTHORIZATIONとSYNTHETIC_CONTRACT_TEST_ONLYを必須にし、actual authorizationsは空。
合成acceptanceが科学的な意味論、実SCF、物理operator、RSS上界、serializer閉包、独立再現性を証明したとはしない。
旧129件は保存検査記録。今回の検査は[結果JSON](contract_tests_result_v2.json)の320件、fail/skip0。
開発中の起動cache lookup拒否とfixture期待値不整合の2失敗ログも保存し、最終suiteのfail数に隠して混ぜない。
最終suiteはprotected access/import試行0。分子・signal・sampling・science build/compile/transpile・新synthetic transpile・GPU・環境/他job変更0。

精度表示302点、strict headroom、paired32/ddof1 covariance、±2SE engineering interval、exact ties、null ineligibleと共通P>=0はv1を継承する。
新GO/winner/materiality閾値や自動研究判断を追加しない。公開後STOPし、本計算、source port、authorization、Track B、追加距離・trajectoryへ進まない。
