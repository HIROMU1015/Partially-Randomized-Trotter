# DF補正比較：native/T評価・実行計画 v1

2026-10-11 JST。[事前契約](df_correction_comparison_preregistration_20261011.md)の従属計画。**今回execute0**。
比較runner／新しいangle adapter／resource containment／全semantic testは未実装。この文書は実行authorizationではない。

## 固定実装方針と順序

1. 将来のsource準備：CD/E input生成を既存recipeと照合。source/API変更なし、新B namespaceへ最小adapterを置く。
   データrecipeを並列worker内で再drawしない。親でtask-specific inputsを固定し、immutable共有する。
2. off-domain semantic gate：Pauli pairingのweighted mean、moment、controlled phase、wrong sign、identity、rank-one閉包。
   新登録結果を見ないsynthetic controlsを用いる。数学的counterexampleなら停止。
3. CD domain6taskのmean/error/moment/sufficient-shot会計。8×8平均、exact signalは評価専用、shots0。
   安全なsource spectral guardの組立／roundoff伝播を流用候補とし、裸のexpm/normをcertificateにしない。
4. E r3/r6 active-frame中心のnative構成と費用。既存whole履歴4件を保持し、再構成4件を別artifactで照合。
   exact source fingerprintに一致しない場合、same-input再現とcompiled履歴再現を区別し、T対象へ進む前にSTOP。
5. 有限T対象inventoryを出力・exclusive freezeする。その後のみ、別途認可された同じone-shot内で最大64 targetを合成する案。
   targets選択は下記の固定位置規則で、resource gainや合成成功によって置換しない。
6. raw result＋外側process exit/status＋resource receipt＋saved-value auditを別々に保存。全結果でSTOP、GPTへ返す。

現在はstep1〜6の計画のみであり、off-domain testsやregistered matricesを今回生成しない。
future sourceを準備する認可と、実行する認可も今回から自動継承しない。

## E native回路

NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0、既存 `.venv311` をread-only使用する案。
all-to-all、basis rz/sx/x/cx、optimization0、seed20261011、internal parallel FALSE。共有API `transpile_and_measure_cost` を用いる。

A: exp(iT/128)またはdiag(I,exp(iT/128))をgeneric unitary decompositionで4calls。
B: exp(i signβ atan(ℓ/128)P)のClifford basis→parity ladder→RZ/controlled RZ→uncompute。
weightは古典測定値に掛け、制御qubitをmeasurement outcomeでpostselectしない。
uncontrolledはCX=2(w−1)、controlledはCX=2w、任意角RZは1/2個（w≥1）。identity stringのphaseは別扱い。
Pauli係数はtrace identity=0というexact数学と数値residualを区別する。数値near-zeroを勝手に削除しない。

local Pauli decompositionは全4^r係数を保持するが、4,096本のcompileはしない。
各rについて、各support weight1〜rでlexicographic最初の非zero Pauli labelを一つ選び、control有無の代表を検査する。
最大2(3+6)=18 B calls、A4callsと合わせて最大22。欠けるweight classはskip、別classで補充しない。
全eventのcostはこの固定parity構成のIRを解析的に会計する。代表compileとsource IR countsが一致しない場合はINCONCLUSIVE、別compilerへ切替えない。
符号反転とcontrolゼロbranchの位相は小行列semantic gateで検査する。
identity/commuting controlはsemantic用、成功例の追加fixtureではない。

Gaussian frameは両者同じactive-frame identity。local centerの費用だけを測る。
global frame取得・thin embedding・routing・prep/readoutは未取得変数であり、local cost表へ密かに加減しない。
particle-sector専用compiler、rank-one入力への変更、optimization-level探索は今回の計画に含めない。
whole generic費用を全whole方式のlower boundにしない。

## 限定Clifford+T targets：固定選択規則

arm/r/control/taskの固定順序でnative circuitsを生成し、RZ gateの発生順を0-basedに列挙する。
φは元binary64のexact dyadic radiansと、意味論上のangle intervalを共に記録する。2πでの無条件moduloはしない（RZはstrictには4π周期）。

| 対象 | 登録枠 |
|---|---:|
| A r3/r6 controlled | 各16位置：floor(j(N_RZ−1)/15)、j=0…15。合計32 |
| A r3/r6 uncontrolled | 各8位置：floor(j(N_RZ−1)/7)、j=0…7。合計16 |
| B r3/r6 | ±atan(ℓ/128)、±2atan(ℓ/128)。合計8 |
| native phase accounting | 必須追加RZがある場合のみ、r3→r6／uncontrolled→controlled／A→Bの順で最初の8。通常global Clifford phaseはsymbolic記録 |

合計最大64 logical target keys。Nが枠より小さい場合は全位置、重複はdeduplicateしても空き枠を補充しない。
未測定位置のangleが同じkeyなら既測定sequenceをreuseできる。identityはexactfast path、証明済みCliffordも共通適用する。
strict guardを満たすClifford sequenceに限る。π/4 RZをphase自由の1T fast pathとして扱わない。
exactfastで使わなかったcall枠を他targetへ振り替えない。新target追加・precision変更・第二synthesizerは禁止。

BのすべてのPauli labelは同じ±α（controlled）／±2α（uncontrolled）を使うため、有限4angles/rで対応できる。
Aの全unique RZがこのinventoryに含まれなければ、**全A TはINCONCLUSIVE**。測れたangleのactual T、未取得のangle費用変数、RZ-countによる推計を別fieldへ保存する。
部分sampleの平均Tで全回路を埋めない。eventのexpected Tは必要な全keyがcoveredか、certified rangeがある場合だけ出す。
CDのlocal correction anglesはEと異なるので、このinventoryのT値をCDへ流用しない。

## synthesizer・精度・phase

pygridsynth==2.0.0、mpmath==1.3.0、dps100、GridsynthConfig(up_to_phase=False,dloop=10,floop=10,verbose=0,show_graph=False,measure_time=False)。
seedはsha256("20261011:"+canonical angle key)の最初の8hexをintegerにする。dedup keyにはarmを含めず、同じangle/precisionは同じsequenceを使う。
実数anglesとεをmpmathへexact rational/interval経由で渡す。裸のfloat変換を厳密入力と称さない。
共通gate-level strict operator tolerance ε_gate=1/40000000000=2.5e−11、合成器へはε_gate/4を要求し、100dps interval Frobenius upperでε_gate以下を確認する。
失敗時にε/dps/seedを変更しない。sequence中のW/global phase、TとT†、ordered H/S/Xを全て保存する。

全local centerのmaximum unitary誤差予算1e−6：synthesis8e−7、matrix/angle/compiler数値2e−7。
最大32,000 RZ occurrence×ε_gate≤8e−7。追加のphase rotationもこの32,000へ含める。
compiled numerical residualを裸のfloat値で認証せず、独立誤差boundが必要。32000超または未認証phaseならT_INCONCLUSIVE。
この同じε_gateをA/B、r3/r6、control有無に使う。Bにprecisionを緩めて有利にしない。
CDのnative信号5e−6 budgetに必要なglobal gate allocationは別台帳。CD実回路費用未取得なので今回の計画では記号化する。

RZ(φ)=exp(−iφZ/2)規約は[pygridsynth2.0.0公開仕様](https://pypi.org/project/pygridsynth/2.0.0/)とlocal sourceを確認した。
wheel SHA256 `30b5b15e9383a8ea8510d54f28e2de0385d102cb54fd4440071abf4525027568`、source tree identityは[準備runtime記録](../../../artifacts/track_b_df_correction_comparison_preparation/2026-10-11/runtime_identity_preparation.json)。
既存SP05隔離runtimeをread-only使用する案。packageをinstall/updateしない。共有Python>=3.11環境へ旧Python3.10 projectをinstallしない。
R1のstrict guard/sequence evaluatorを再利用候補とするが、R1 angle型はπ/atan専用なのでraw dyadic radiansへの小adapterが必要。
これは未実装のblocking item。古いR1 runnerを起動して代用しない。

## resource capsと並列化（将来の別実行案）

現在affinity32 CPU、約64GiB RAM。過去native代表5callsは15.824s／557.465MiB peakで、律速候補はgeneric compileとarbitrary-angle synthesis。
native worker2、synthesis worker2、matrix worker1、各BLAS/OpenMP/Rayon1、Qiskit internal parallel FALSE、GPU query/use0。
phaseは重ねず、各workerのRSSを集計する。seedとtask labelはworker数に依存させない。speedup測定は未実施。

| phase | wall cap | aggregate CPU cap | per-worker RSS | parent＋workers aggregate RSS |
|---|---:|---:|---:|---:|
| CD/math | 300s | 150s | 512MiB | 768MiB |
| E native | 600s | 450s | 1024MiB×最大2 | 2560MiB |
| T synthesis | 1200s | 900s | 512MiB×最大2 | 1536MiB |
| 全one-shot | 2400s | 1800s | 上記 | 3072MiB |

T per-key wall30s/CPU20s、target keys≤64、actual synthesis calls≤64、native compile≤22、1 compiled circuit≤200,000 instructions。
1sequence≤20,000文字、aggregate output64MiB、log4MiB、remote-review files原則≤512KiB。
今回利用者がwall上限の固定を明示したため、この新案ではwall capを置く。通常のno-wall-default方針の一般変更ではない。
新capsを旧G10512MiB contractへ適用しない。

将来は外側parent監視＋POSIX process-group kill、per-key子process、aggregate RSS/CPU監視、exclusive markerを実装する。
内部SIGALRM/post-checkだけをhard containmentと称さない。失敗時はqueue未投入分を実行せず、進行中workerを停止、prefix/provenanceを保存。
最初のactual synthesis前にinventoryをfixed artifactとして保存。retry0、timeout0回再試行、partial結果から勝敗を選ばない。
resource containmentのnon-science synthetic validationは将来のsource準備で必要。今回はguard実装もテストも実行していない。

## source / authorization / STOP

現在はcontract review待ち。runnerなし、source_commit=null、runs0、science_execution_authorized=false。
契約確認後に利用者が認めた範囲でfuture実装をprepareし、source Sへfreeze。全cap/semantics/source/runtime hashをfocused reviewで照合する。
science executionはSの直接子authorization-only Aと明示one-shot指示を別途受けてから行う。今回はAもmarkerも作らない。
全math/phase検査が通ってもalgorithm採択・科学的GOはCodexで出さない。
G10/RA-D0／分子production／広範grid／fixture変更／次stageは0。計画・公開後はmandatory STOP。
