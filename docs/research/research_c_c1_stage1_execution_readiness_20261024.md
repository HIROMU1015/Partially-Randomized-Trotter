# C1 Stage 1 実行準備報告（2026-10-24）

**条件設計と接続設計まで準備した。実装・環境・数値gateは未完成であり、科学実行は未承認。** 推奨する初回はP-H4、Nα=Nβ=2の36次元sector、通常C-DF number表現、κ保持、`max_vecs=8, tol=1e-8, L-BFGS-B` の1 fit。詳細は[科学条件案](research_c_c1_stage1_scientific_conditions_20261024.md)、機械可読の未承認状態は[契約案](research_c_c1_stage1_contract_draft_20261024.json)。

基準commitは `31705b2caddc8bee316ff241055dcc23b18ba8bc`。専用branchは `codex/research-c-c1-stage1-readiness-20261024`、worktreeは `/tmp/prt-research-c-c1-stage1-20261024`。今回の成果物は新規3文書だけであり、コード・runner・環境・grant・共通研究状態を変更していない。

## 1. 静的に確認した再利用資産

| SOURCE / tests / scripts | Stage 1での使い方・限界 |
|---|---|
| [c1_general_k_adapter.py](../../src/trottertracks/c1_general_k_adapter.py) | `from_ffsim_number_arrays`、GeneralKModel/Frame、`general_k_coefficients` を再利用できる。κ/c・J/U・K/V・frame順序・hashを保持。shape/finite/規約の確認は実装済みだが、κ Hermiticity・U unitarity・実operator同値性は未検証 |
| [adapter人工tests](../../tests/management/test_research_c_c1_general_k_adapter.py) | 前回13件PASSの出所。データ/係数/mock bridgeの技術証拠であり、実H4/CAR/Gaussianの資格を代替しない。今回は実行0 |
| [df_hamiltonian.py](../../src/trotterlib/df_hamiltonian.py) | 保存X-DFのDFHamiltonian、PhysicalSector、`df_linear_operator`、既存basis-column dense化を再利用。sector36本のmatvecでよい。一般Jをsquare Gへ代入しない。新規ground-state solve/分解/再hermitizationをしない |
| [df_trotter/](../../src/trotterlib/df_trotter/)、[df_partial_s2.py](../../src/trotterlib/df_partial_s2.py) | Gaussian/diagonal/Partial-S2の既存経路を静的確認。Stage 2資産として維持し、今回のStage 1では呼ばない |
| [df_rte_tail.py](../../src/trotterlib/df_rte_tail.py)、[df_rte_qiskit.py](../../src/trotterlib/df_rte_qiskit.py) | tail・RTE・wrapperは後段。Stage 1へλ、sampling、partitionを持ち込まない |
| [pr_dictionary_validation/reference.py](../../src/trottertracks/pr_dictionary_validation/reference.py) | generic `car_annihilator(ctx,modes,p)` のbit作用は8-modeにも適合。`exterior_frame` は数学的にはgenericだが全minor・高精度dense積を初回の必須経路にしない。`original_operators` はsquare fixture、保存reference validatorは16/32次元固定なのでH4用ではない |
| OpenFermion v1.6.1 conversions / symbolic_operator / sparse_tools | InteractionOperatorの通常の変換はFermionOperator `+=` の `_issmall` による係数削除を含む。係数を無閾値のterm mapへ渡して既存 `get_sparse_operator(FermionOperator,n_qubits=8)` を使うglueが必要。sparse処理自体は非ゼロ係数を扱い、数値閾値によるcompressを要求しない |
| P-H4生成SOURCE、下記固定blob | unthresholded spin-liftの純粋処理と係数配置を再利用。`build_h4_inputs` はSCF/積分/DFを再実行するため呼ばない。旧state/energy/保存PASSを新しいreference normとして読まない |
| [test_df_hamiltonian.py](../../tests/test_df_hamiltonian.py)、[test_df_rte_tail.py](../../tests/test_df_rte_tail.py) | 既存sector、InteractionOperator、python/numba、number/rotated densityの比較設計を再利用。今回はtestsも数値部分も実行していない |
| [run_df_ground_state.py](../../scripts/run_df_ground_state.py)、[benchmark_df_ground_state.py](../../scripts/benchmark_df_ground_state.py) | sector/backendの使い方の参考。ground-state/benchmark全体を今回のrunnerに転用すると不要な科学callが増えるため使わない |

既存adapterの `diagonal_primitives` / emitter / `BoundGeneralKFrame` / registry bindingはStage 1で不要。実registryはOperator fingerprintを計算し得るため呼ばない。κ/J/U/cから演算子を構成して独立に比較する接続処理がまだなく、既存mock testsだけでは不足する。旧C4 executor全体、grant、4-mode reference、controlled検証結果は流用しない。

## 2. inputとSOURCEの固定情報

| 対象 | 固定identity・今回の確認 |
|---|---|
| P-H4 input | commit `86b8b002704aced8ca239414080f53793de81658`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz` |
| packet | SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、Git blob `5828b6b225bb4990a46934edba152b82ec97782f`。今回byte hash一致を再確認、数値decode/演算なし |
| input生成SOURCE | commit `0c329b30a81accb35c212be8defa66fee0dd0f12`、`src/trotterlib/h4_molecular_inputs.py`、blob `105c41dbd51e34d74026cfa5f7a382d76678a1f7`、SHA256 `fb53bf44459dc6ef18223d55594511c03766c53537f6c445dad7a0b7a42225ea` |
| ffsim | v0.0.83、commit `558d3517385778c733f1aa226bb61b8f30f4b0f3` |
| ffsim Hamiltonian / fit | `python/ffsim/hamiltonians/double_factorized_hamiltonian.py` blob `fd87cc70f1a3b3d37349ab0bc6790b68166e9f45` / `python/ffsim/linalg/double_factorized_decomposition.py` blob `2eac9e1690ddf5fb7779136aa7a94b04b02de526` |
| ffsim dependency宣言 | `pyproject.toml` blob `eb3f1ad809393f3f58c285ba0c20889809650bb0`。全依存の解決済みlockではない |
| SciPy solver候補 | v1.14.1、`scipy/optimize/_lbfgsb_py.py` blob `42ad9038ef0ce4c29b0cf22c5c9d2a1c029827c3`。`options={}` の実効既定を静的確認 |
| OpenFermion reference経路 | v1.6.1、`conversions.py` blob `95f45b63c6092cc8c542f66f003a5a47268633db`、`symbolic_operator.py` blob `fbd8d6a81a7e647e6703fa58edf62d4c88699c6d`、`polynomial_tensor.py` blob `a2908d95a0c806971f2cb0ff416b8f5a819784c6`、`sparse_tools.py` blob `453db13d3746047c5101d16e52a1bb8485dfff87`。経路の詳細とリンクは科学条件案§6 |
| 共通運用 | `research_project_instructions_unified_2026-10-23.md` SHA256 `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f` |

P-H4 packetの旧生成metadataは同一h/g/cからPauli/DFを生成した来歴と実rank10を含む。保存κ/λ/G、spin_one/spin_twoを初回fitのh/gと取り違えない。歴史的DF/referenceのPASSは保存証拠の資格に留まり、新規sector normは次回契約で明示する。[入力来歴報告](research_c_c1_input_ffsim_readiness_20261024.md)

将来Science SOURCEは上記基準だけでは足りない。接続wrapper/reference glue/runner/tests、dependency lock、schema、allowance、call policyを含む新しいcommitを凍結し、そのfull SHA・file hashes・contract hashをSOURCEへ記録する。現在そのcommit・実行command・grantは存在せず、draftにnull/falseで保持する。

## 3. 環境分離とarray境界

接続方式は **固定MO packet → 独立ffsim fit環境 → NPZ+JSONのκ/J/U/c → 既存DF側評価環境 → adapterと独立CAR検証** とする提案。Python object/pickle/QPY、ffsimの回路を環境間で渡さない。

固定ffsimの[pyproject](https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/pyproject.toml)はPython≥3.11、Qiskit≥2.0、PySCF≥2.12、JAX、NumPy、SciPy等を要求する。歴史的P-H4側のQiskit1.3.0/PySCF2.7.0をupgradeして合わせない。

| 環境 | 固定・確認する内容（まだ構築していない） |
|---|---|
| fit専用venv | ffsim固定commit、Python 3.11系のpatch版、候補SciPy1.14.1。JAX/jaxlib/NumPy等を依存解決後にexact version・wheel/source hash付きlockへ固定。Qiskit/PySCFは依存してもSCF/回路を呼ばない |
| 既存DF側評価 | 既存NumPy1.26.4/SciPy1.14.1/OpenFermion1.6.1を第一候補とし、実環境identityを確認。共有venvを変更しない。必要なら既存lockを複製した独立評価venvを別承認で作る |
| 精度・device | fitは固定ffsimのJAX x64を起動時に確認。小さい4軌道ではCPUを第一候補とし、実device/backend/BLAS/OpenMP設定を凍結。演算前に実dtypeを確認する |
| cache / 出力 | task専用venv/cache/outputを新規作成する将来案。既存共有package/cacheを変更しない。SOURCE/input/export hashをcache keyへ含める |

JAX/jaxlibの互換version、Python patch、全lockは未選定。依存解決・import smoke testは今回禁止のため未実施である。SciPy1.14.1が解決できなければ無断で別版・別既定へ移らず、科学契約案を改訂して採択へ戻す。dependency宣言をruntime検証済みlockと呼ばない。

export schemaの候補は `C1_STAGE1_FFSIM_NUMBER_EXPORT_v1`。今回は生成していない。

| export項目 | 規約・検査 |
|---|---|
| array packet | `kappa[4,4]`、`diag_coulomb_mats[R,4,4]`、`orbital_rotations[R,4,4]`、`constant[]`。初回の実fitは全てfloat64、C order、little-endian保存、pickleなし。演算子は評価側complex128 |
| 物理identity | 元packet commit/path/SHA256、member名・shape/dtype・member hash、MO C/geometry/basis/charge/spin/ordering、sector |
| 表現と帰属 | `z_representation=false`、`one_body_role=ffsim_corrected_kappa`、κは元g補正、cは元nuclear constant、Jにspatial1/2なし、K側で1/2を適用 |
| fit | 全設定・初期化・実R、OptimizeResult success/status/message/nit/nfev/njev/fun、x/jacのarrayまたはlossless export、tensor residualの別記録 |
| SOURCE | ffsim commit、wrapper SOURCE、fit/evaluation両環境lock hash、実version/device/x64、input/output/metadata hash |
| frame | 元outer indexを維持した一意frame_idとgroup_id。Stage 1の識別子でありD/R partitionの採択ではない |

shape/dtype/finite/実対称性を検査し、投影・閾値削除・丸めての修正をしない。adapter `from_ffsim_number_arrays`へ明示role/number flagを渡す。独立referenceは同exportのraw κ/J/U/cを読む。adapter出力由来のZ係数をreference入口にしない。各runはraw exportとmetadataを保存し、同じsource/input/dtype/orderのidentityが成立する場合だけ参照を共有する。

## 4. 別承認後に必要な最小差分

新しいfitter・CAR algebra・linear algebraを実装する必要はない。欠けているのは以下を接続して、科学entryを明示する薄い処理である。新規file名は設計例であり、今回追加していない。

1. **fit/export接続**：例えば `src/trottertracks/c1_stage1_validation.py` 内の独立fit entry。lower-level `double_factorized(...return_optimize_result=True)` を1回呼び、factoryと同じ元gのκ補正式を適用してexportする。JAX objective/optimizer/Choleskyはffsimを再利用し、factoryによる二重fitを避ける。
2. **reference接続**：同moduleに元h/g/c→OpenFermion、raw J/U→generic CAR、adapter K/Z→OpenFermion、保存X-DF→既存sector matrix-freeの薄いglue。OpenFermionのsymbolic `+=` を通さない無閾値term map、既存CAR helperへNumPy allocation contextを渡す部分、bit/sector順序の明示変換が必要。4-mode referenceの寸法定数やsquare組立ては転用しない。
3. **限定runner**：例えば `scripts/run_research_c_c1_stage1.py`。fit/evaluateの別process呼出、SOURCE/contract/schema照合、call journal、partial evidence/STOP、read-only入力と出力配置だけを担当する。実行commandは実装後に固定する。
4. **必要な技術確認**：人工の非対角J・対角項・負係数・非自明かつ非対称なU・κ/c、bit ordering、既存squareへの退化例で、独立経路とallowanceの形成を確認する。微小係数がOpenFermion経路で削除されないこと、source/API、有限値、status失敗、hash/schema拒否も確認する。これは別承認の実装検証案であり、今回testsを追加・実行していない。

数値allowanceの演算graph係数とSVD項はこのglueが無い状態では確定できない。人工入力での確認を経て、**実H4 fitより先に**固定する。reference精度を結果に合わせて引き上げるretryは初回契約に含めない。表現gateを根拠付きで設定できなければ科学runを開始しない。

## 5. 科学callの事前登録案

以下は**採択後の正常終了経路の算出規則**であり、今回の実績は全項目0。Rはfitが返した実outer rank（候補上限8、正の整数）とする。開始前にR依存の規則と許容上限を固定し、結果後にcallを追加しない。内部BLAS呼出や全演算回数と、表の論理callを混同しない。

| 処理 | 初回候補の論理call数 / 計上規則 |
|---|---|
| 固定inputの科学decode | fit側h/g/cで1、評価側h/g/c+保存X-DFで1、合計2。論理的packetは1個。各process内はcache |
| fit export decode | 評価側で1。metadata/hash照合は別の技術I/O |
| C-DF fit | **1**。内部Cholesky初期化1はこのfitに含む。独立X-DF分解0 |
| solver内部評価 | `nfev/njev/nit` を別記録。ffsim/SciPyの固定反復政策で決まり、別fitとして数えない |
| fit資格・tensor residual | OptimizeResult資格1組、g_fit再構成1、max-absoluteとFrobenius残差を各1。有限な未収束出力はpartialとして記録してoperator評価へ進まない |
| adapter取込 | 1。modelのデータ変換のみ、basis registry/circuitは0 |
| H_phys sector reference | 1。元h/g/cから作成。内部のfull sparse256は中間表現で、full物理誤差の評価はしない |
| 保存X-DF sector operator | 1。既存linear operatorに36本のbasis action。全electron sectorやenergyは計算しない |
| raw CARとKのbase | full256のκ+cを各1、計2 |
| raw CARとKのframe | full256を各R、計2R |
| raw CARとKの合計 | 各1のfull積算、計2。sector36への制限は各1、既存行列の再利用 |
| Hamiltonian construction job小計 | **6+2R**。H_phys sector、X-DF sector、base2、frame2R、合計2。上記以外の中間配列・診断行列はこの小計外 |
| spectral norm | full base1 + full frame R + sector adapter1 + sector DF-CDF1 + sector DF-XDF1 = **R+4** |
| 構造診断 | κ Hermiticity1、J対称性R、U†U−IがR、sector H_phys/CDF/K/XDFのHermiticity4、CDF/K full合計とNα/Nβのcommutator4、κ_orig/exportおよびκ_orig/保存κの差2。Frobenius norm **2R+11**。exact J対称/schema/constant検査は別の技術check |
| scale/allowance | 固定した係数絶対和・行列形成scaleの算出を各経路/寸法ごとに計上。wrapperの演算graph未固定のため総数はnull |
| provenance | packet byte hashは共有可、各member/出力/metadata/SOURCE/lockのhashとledgerを記録。実file inventory未固定のため総数はnull |

未確定の総call数はnullのままにする。必要な前提は、採択条件、実装済みglueの演算graph・出力file一覧・allowance規則、Science SOURCE/両環境lock、実Rである。契約にはR依存formulaを事前登録できるが、架空の固定総数や評価結果を入れない。

reference共有は同input/SOURCE/κ政策/sector順序/精度/hashが一致するこの1条件内だけ。H_physはCDF/XDFに共用し、CAR bilinearはframe間で再利用し、full frameはsector projectionにも使う。identity検査後のcache hitと実生成callをledgerで分ける。過去のenergy/保存PASSを新normのcache hitにしない。

SCF/積分再生成、新X-DF分解、energy/ground-state solve、Gaussian/Qiskit回路、expm/controlled/adjoint、compile、sampling/shot、Stage 2/3はすべて0。fit失敗・不正出力・gate FAILの場合は終了したprefixだけを記録し、追加診断科学callを自動で増やさない。

## 6. 性能・共有環境の方針

将来runの前にCPU affinity/使用可能core、RAM/pressure、GPU/backend、他job負荷を読み取り、process数とBLAS/OpenMP/JAX threadを合わせる。資源上限、phase/total wall-time上限、恣意的な低worker数を設けない。solverの評価/反復政策とone-fit/call/STOPは資源設定と独立に維持する。

静的には4軌道fitのJAX初回JIT、generic dense CARの重複積、frameごとの同じbilinear生成が候補bottleneck。ffsimの構造別JIT cache、既存CARの±1/bit作用、immutable bilinear cache、vectorized/sparse演算とframe対の並列処理を優先する。complex128の256×256は1行列約1MiBだが、全中間配列とworkerのaggregate RAMで判断する。全frameの高精度dense exterior-minor積は必須にしない。実装技術検証後にbackend/threadsを凍結し、speedup値は測定比較なしに主張しない。

この作業では重いrunも環境構築もないためresource構成を選定していない。将来の性能変更も凍結runに上書きせず、新しいSOURCEで科学条件を保つ。共有server設定、共有cache/venv、他userのprocess/fileを変更しない。

## 7. 残件、GO/STOP、公開後の停止

| 種類 | 未完了事項・担当 |
|---|---|
| 科学条件の採択 | 統括・ユーザー：P-H4/sector/full base・frame、κ政策、保存X-DF、rank上限8/tol/solver、DF誤差は測定報告・Stage 2適格性は別判断という提案を採択する |
| 技術blocker | Codex（別承認後）：最小glue/runner、互換dependency lock/import確認、人工経路確認、allowance係数、call/出力inventory、SOURCE凍結。現段階はcommandなし、grantなし |
| 物理誤差budget | 統括・研究A/B/C：同精度比較で必要なDF budgetを後に判断。初回はCDF/XDF sector normを別々に報告する案で、未知の絶対閾値を新設しない |
| 科学実行承認 | ユーザー：凍結したSOURCE/input/contract/call/STOPを指定する新たな明示承認。実装承認と科学run承認は区別する |

技術的STOPはhash/schema/補正/非有限/status欠落/reference欠落/call超過/共有環境干渉/承認欠落。科学的STOPまたは保留はfit未収束、対象sector/規約不整合、adapter/構造gate FAIL、X-DF比較不成立、後に採択したDF budget違反。両者を結果に記録する。raw出力/partial evidence/ledger/STOPを保全し、失敗後の自動rank/tol/seed/optimizer/precision変更・再fit・restartをしない。中断後も自動resumeせず、完了済みcallと出力identityを照合して統括・ユーザーへ戻す。

今回の許可はread-only調査と新規3文書のcommit/push/remote照合まで。固定SOURCE・旧結果・旧grant/ledger/marker/mandatory STOP、元workspace、共通採択事項を保全する。公開後はmandatory STOPし、未承認の実装・環境・科学実行、C4/Gaussian再開、Stage 2/3へ進まない。

## 8. 統括チャットへの引継ぎ文

> C1 Stage 1の科学条件案・接続準備報告・未承認JSON契約案を作成しました。初回推奨は保存P-H4（直鎖H4、1 Å、STO-3G、Nα=Nβ=2、sector36）、ffsim v0.0.83通常C-DF number表現、元g由来のκ保持、max_vecs=8/tol=1e-8/L-BFGS-B/SciPy1.14.1既定の1 fitです。保存rank10 X-DFを第一対照とし、CDFとXDFの物理sector normを別々に報告、adapter同値性はsector合計とfull256のbase/各frameで確認する案です。rank/tol/solver/κ/誤差政策はいずれも未採択で、DFの絶対精度閾値やC-DF≤X-DF必須gateは置いていません。既存adapter・CAR・OpenFermion・sector matrix-freeを再利用できますが、Codexの最小fit/export/reference glueと限定runner、独立環境lock、allowance係数とSOURCE/call inventoryの固定が科学run前のblockerです。統括は科学条件と誤差報告政策を採択し、必要な技術実装を別途承認してください。その後、固定SOURCE/input/契約/call/STOPを指定した新たな明示承認が科学実行に必要です。今回は追加実装・環境構築・科学call 0、旧SOURCE/結果/grant/STOPと共通研究状態を維持し、公開照合後mandatory STOPします。
