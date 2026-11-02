# H5中性開殻：実装・人工検証・限定pilotの実行準備

2026-10-16 JST。**PREPARATION ONLY：実装・人工/mock検証とsource固定を完了。実H5計算・grant発行/消費は0。**

- branch：`track-a-h5-neutral-open-shell-readiness-20261016`
- SOURCE：`f906a61415d40ec7f362d00aa76e26420e25e7dc`
- 実装基点：H8公開結果 `681ec23c93fdd1090d325908f4b155a79115afdf`
- [資料bundle](../../artifacts/resource_applicability/track_a_h5_open_shell_readiness_v1/2026-10-16/README.md)

採択済み[科学レビュー第9節](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)と[設計第4節](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ccc79502f043b4885e2337ffa98ff45f1875ae4/docs/research/track_a_resource_evaluation_expansion_design.md)に従った技術準備である。
研究方針・科学比較条件の再採択やGPT詳細科学reviewは行っていない。主担当のみ、subagents0、独立reviewではない。

## 固定条件

Linear neutral H5、1.00 Å、STO-3G、charge0、ROHF spin1/multiplicity2、共通real spatial MO、5 electrons、Nα=3/Nβ=2、5 spatial/10 spin orbitals、sector100、T=.8。
旧odd-chain helperのcharge+1/tripletや旧charged H5成果を新inputとして流用しない。
DFはtol-only1e-8/cutoff0、signed returned generation order、weighted Hermitian projection、全returned rank保持。
実rank Lは未知、prefixは受領後のLに対するceil(L/4)。rankを他系へ合わせない。
Reference stateはDF sector solver stateであり、ROHF determinantは軌道/solver初期値の用途に限る。

B0、B0+I、full PF S2、full PF S4、B2/r2/K2、B2/r2/K4の6構成。q1、PR prefixはceil(L/4)、fullは全rank。
Full S4は既存global Yoshida/負時間係数を保持する。B0+Iはfull-Fock discarded-tail identityのexp(-iTC)を保持し、full-H referenceは回転しない。
有限RTEの次数/component確率、component選択、sampling API、回路構成、normalizationとshot規約を変更しない。
要求signal精度 .05/.01/.005/.001/.0005、P=0のshot-weighted full-wrapper RZを主、CX/各depth/sizeを補助とする。
Pilotはcorrectness/feasibility用であり、全精度の最良候補探索・母平均順位の保証ではない。

## 実装

新package `src/trottertracks/resource_applicability/h5_open_shell/` を追加した。旧source・raw結果・grant/ledgerは編集していない。

| Module | 実装内容 |
|---|---|
| input.py | charge/spin明示、ROHF共通MO、単一kernel、AO/MO積分変換、MO直交性、tol-only DF、signed order、weighted projection、array identity |
| sector.py | α/β別occupation basis/eigensystem、interleaved JW↔grouped tensorのfermionic sign、one-body/(eα+eβ)²、S²、独立DF行列とmatrix-free全column照合 |
| science.py | DF solver、primitive/native検証、Horner/forward/exact-tail signal、B0I位相、allowance/shot会計、既存native/compiler接続、paired cost取得 |
| contract.py | 固定6構成、source/input別seed namespace、2段階task registry、call caps、新H5 grant bindingと旧grant拒否 |
| execution.py | 独立coordinator監視、own-run affinity/停止、pidfd、wait/reap、予約・first STOP保持、最大4group並列、one-shot、terminal保存 |

`run_h5_open_shell_pilot.py` は実行に新manifest/明示H5 grantを必要とする。
`prepare_h5_open_shell_pilot.py` はbytes/hash・package metadataのみでsource closure/profile/未認可draftを出力する。準備scriptは実行済みで、科学計算は行わない。

H6結果branchにはnative provider sourceの全体が含まれなかったため、実コードを含む固定H8結果を実装基点とした。最初に作った未使用H5 worktree/branchも保全し、切替/reset/clean/削除していない。

## 開殻の技術的整合性

αとβのbasisは別々に作成し、各small one-body matrixを独立に対角化する。H5では両方10次元でも、それぞれ3占有/2占有であり同一basisとして扱わない。
DF平方はsigned λ(eα+eβ)²を使い、2eαeβを省略しない。spin-cross係数はexact zero、共通空間MOに対応するα/β係数blockはexact equalityを要求する。
Hermiticity、basisの完全性/順序、Nα/Nβ/Sz保存を検査する。Gaussian変換の途中をsectorへ射影せず、native full-system actionの最終結果とsector actionを照合する。

S²は S²=Ms(Ms+1)+S−S+ を、異なるspin sectorへ移るS+も含めて構成する。人工小系で独立OpenFermion full-Fock S²と一致した。
DF stateのnorm・eigen residual・lowest-sector energy・S²期待値/variance/target residualを検査する。S²が成立しない場合はdoubletと呼ばずSTOPし、stateを別のものへ自動変更しない。
これはfixed Ms-sectorのengineered検査であり、全spin/全Fock空間の基底状態証明ではない。

実input後に100×100の独立occupation DF Hamiltonianを構成し、production matrix-free actionの全100 columnを照合する。
Norm1e-12、reference split agreement1e-10、state/action/nativeは既存abs1e-9+relative1e-10、S² target residualは1.1e-9というengineering gateを用いる。
Weighted projectionは既存式 N||Δh||+N²Σ|λ|d(2s+d)、primary N=10/補助N=5、9.9e-11 Hartree gate、投影後Hermiticity1e-12を継承した。
Finite precision norms・numerical allowanceは経験的評価で、厳密interval certificateや新しい精度保証ではない。allowance factor1/10を保持する。

## 高速化・並列化

- α/β別small eigensystemをprimitiveごとに一度作り、時間別位相を最大32 entryのLRUで再利用する。primitive検証の3 probeも同じprepared blockを使う。
- signal処理はdiscard/partial/fullの3 preparationを共用する。K2/K4でpartialを共有し、検証済みfull referenceを再利用する。B0+IはB0 signalとphaseから導出する。
- cost前にidle coordinatorでpreparationを一度作り、disposable fork workerへread-only/COWで引き継ぐ。同一trajectoryのevolutionをcos/sin両軸で共有する。
- 既存のcontent-checked streaming fingerprint/LRUを使用する。numeric bytesやphaseが変われば新fingerprint、mutable object IDだけによる再利用はしない。
- 通常groupは最大4並列・内部thread1。S4 heavyは単独。別groupのtrajectory/seedは独立、両軸で同じeventsを保持する。
- coordinatorは科学workerとは別processで監視するため、workerのGIL保持で監視経路を塞がない。

これらは実装と人工/mockの検証結果であり、実H5でのspeedup/所要時間/peak RSSは未測定。
GPU、install/update、新venv、共有環境・他jobの設定変更は0。

## 人工・mock検証

**32 tests PASS、約6.45秒。単一test process・内部thread1。**

- α/βの異なるbasis、complex one-body、signed DF平方cross term、fermionic signを独立occupation expmと照合。
- 人工100次元sectorのDF matrix-free/独立行列、state norm/residual/reference、S²を確認。高spin stateをdoubletへ誤分類しないことを検査。
- 6 system qubits以下＋ancillaの人工full-systemで、native primitive、S2/S4負時間、ordinary-controlled target、relative phase、X/Y wrapperを照合。
- K2/K4の人工PRは、同じeventsによる既存explicit circuit builderとnative loweringを照合。元のfinite lawの確率を維持。
- compiler identityと6metrics受領interfaceはmock compileで検査。実full-wrapper cost compileは0。
- 最大4 mock slotsの完了/STOP、最初の理由保持、所有pidfd、ESRCHが最初/途中/全対象のcleanup、非ESRCH error保持、wait/reapとFD閉鎖、旧/改変grant拒否を検査。
- H7はsector1225の組合せ列挙だけであり、H7のHamiltonian/state/reference/native計算はしていない。

[検査log](../../artifacts/resource_applicability/track_a_h5_open_shell_readiness_v1/2026-10-16/artificial_tests_v1.txt)、[計画](../../artifacts/resource_applicability/track_a_h5_open_shell_readiness_v1/2026-10-16/artificial_test_plan_v1.json)、[技術結果](../../artifacts/resource_applicability/track_a_h5_open_shell_readiness_v1/2026-10-16/technical_validation_v1.json)。
31件合格後にrealized input bindingと改変grant検査を加え、最終32件を合格した。技術確認は主担当が行ったものでCI/外部再現/別担当reviewではない。

計画表示を意図したrunnerのdefault呼出しは、自動承認reviewが「実H5科学計算には別承認が必要」と判断して拒否したため未実行。迂回して起動せず、専用metadata preparation scriptでclosure/profile/draftの生成を完了した。
実runnerのOS fork・native終了の実動作、実H5全経路のend-to-endは未実施であり、PASS扱いしていない。

## Pilot scope・call会計

| 処理 | 最大の新規取得数 |
|---|---:|
| ROHF SCF / integral build / DF decomposition / state solve | 各1 |
| solver matvec（post-solve residual含む） | 10000 |
| 独立referenceとのproduction column照合 | 100 matvec |
| full/truncated sector reference matrix | 2 |
| ensemble signal（B0I overlay含む） | 6 |
| 最初のcorrectness/feasibility group | 6 / 12 compile |
| PR総trajectory（各構成8本） | 16 |
| sampling API requests | 16 |
| RTE events/occurrences（r2） | 32 |
| cost group / full-wrapper compile | 20 / 40 |
| native control probe | 32 |

最初の6 group完了後にのみ、経験的allowance下でいずれかの要求精度に適格なPRへ残り7本を追加する。
全PR不適格であれば12 compileで終了し、不適格を費用0として扱わない。q/r/K/prefixを勝つまで増やすfallbackはない。
PR16本はpilot標本であり、探索・独立32確認の再実行やpoolingではない。namespaceは `H5_NEUTRAL_ROHF_NATIVE_PILOT_V1`、seedはSOURCE/input/cell/replicaへ結合する。

primitive検証数は実rankから既存S2/S4/負時間scheduleのunique (primitive,time)を列挙した数×3 probeで固定する。
例としてL=9/17/25なら195/339/483 actions。これらのLは予測・input受領値ではない。
Tail matvec/決定論的action上限は各500000、超過前に予約/停止する。
[rank依存call plan](../../artifacts/resource_applicability/track_a_h5_open_shell_readiness_v1/2026-10-16/rank_dependent_call_plan_examples_v1.json)を参照。
旧carry/failed/unknown/reservedを保持し、H4/H6/H8のledgerを新H5でreset/refundしない。H5の新増分は上表で別記し、将来grantのcarry digestへbindingする。
現時点のH5 actual/reservation/grant consumptionは0。旧lineageの未知actualを0と補完していない。

## 環境・監視・起動前gate

既存private venvを変更せず利用候補として固定した。
Python3.11.0rc1、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、Numba0.60.0、OpenFermion1.6.1、PySCF2.7.0、OpenFermion-PySCF0.5。
Compilerはbasis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing None、symmetric_directional、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
旧hostとのcompiler output bit完全一致は未確認のまま。
SOURCE closure264 file（dynamic importも含む全library/providerの保守的superset）、環境14817 file（pycache/pyc除外）をbyte SHAへ固定した。2回のprofile digestは同じで、venv変更0。
Profile digest `70ebab6aec9b79b21592da2d928f3b68c15ba5b152be52560014596a8dca6992`。
全profile inventoryはprivate homeに保存し、Gitにはsummary/digestだけを公開する。

利用者が元指示の計算資源指定・詳細見積りを撤回したため、CPU番号・所要時間・容量の詳細予算を新しい科学的承認条件としていない。
コードはprocess限定thread1、1秒観測/最大5秒、own-run限定affinity/pidfd STOP/cleanup、pressure/OOM、role/aggregate memory、FS/output/journal guardを維持する。
実行時のCPU・資源数値は現在の負荷に合わせてmanifestへ固定する。CPUは未割当、資源予約ではない。読取snapshotでは有効空きRAM約469GiB、home空き約293GiB、affinity64 CPU、quota読取KNOWNだったが、起動の保証には使わない。

将来の起動前に必要なのは以下である。

1. 利用者による「H5 SCF・DF・state生成＋この限定pilot」の別途明示承認。
2. 固定SOURCE checkout、現profile/全source bytes、target、旧carry、未使用home output、空きCPU/physical core、memory/pressure/OOM、FS/inode/quotaのfresh照合。
3. 新H5専用manifest/authorizationをSOURCE/profile/target/carry/output/contract digestへ結合。旧grantを拒否し、exclusive outputをone-shot claimとする。
4. 起動後のinput/state/primitive checks合格時に実rank/input SHA/registry/call caps/seedを `realized_input_binding.json` へ固定してからcostをdispatch。
5. Coordinatorのnative exitを呼出側でwaitして保存する。完了またはfirst STOP後に自動retry/resume・H7・全面mapへ進まない。

SOURCE固定後のREVIEW commitではHEADがSOURCEと異なる。将来の実行はSOURCEをcheckoutした独立runtime worktreeから行う。
未実行のcommand templateは以下であり、現時点のabsolute launch commandはnullである。

```bash
# 新manifest/authorization・process限定thread/cache設定・fresh gateを揃えた後だけ
<private-python> -P -B <SOURCE-worktree>/scripts/run_h5_open_shell_pilot.py \
  --execute --manifest <new-H5-manifest.json> --authorization <new-H5-authorization.json>
```

準備profile/bindingは `/home/AbeHiromu/projects/h5-handoff-evidence/20261016/open-shell-readiness/fixed-source-preparation02/` に保存した。
`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`sealed=false`、grant未発行、one-shot未消費、`absolute_launch_command=null`。
科学定義の未採択事項は新たに見つかっていないが、実H5のROHF収束・DF rank・spin純度・精度・回路費用・feasibilityはまだ未知である。
それらを人工PASSから認定せず、実dataがgateを満たさなければSTOPする。

## H7への再利用と不足

α/β別SpinPrimitive、JW/sign、S²、DF構造検査、native loweringと会計は再利用候補。
H7は新しいneutral ROHF/DF/state、sector1225のbounded solver/CSR/reference adapterと固有のinput/namespace/registry/資源実測が必要であり、H5入力関数は意図的にH5固定である。
H5の正しさ・実資源の取得後に別途H7の準備/実行判断を行う。H5でPRが勝つことをH7のcorrectness条件にしない。
H7の実装全体が完成した、実H5が成功した、全面resource mapが完成したとは報告しない。

## 到達点とSTOP

中性ROHF input・不等spin sector primitive・DF state/S²・native回路・二段階pilot・簡単な高速化/最大4並列の実装と32人工/mock testsを完了した。
**実H5は未実行であり、起動には別の利用者承認とfresh bindingが必要。**
source/testsと軽量準備資料のみを公開し、実NPZ/runtime/cache/credential/内部SSH情報はcommitしない。
GitHub remote確認後、mandatory STOP。追加の科学review・H7・pilot launchへ自動進行しない。
