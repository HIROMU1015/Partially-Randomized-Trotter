# 研究C・C1：一般対角K adapterの最小実装・人工検証・科学実行準備

2026-10-24 JST。状態：**C1_ADAPTER_PREPARED_POLICY_DEPENDENCY_REMAINS**。

一般実対称Kを元frameのまま保持するopt-in adapterを追加し、人工unit/mock **13件がPASS**した。既存primitive/emitter・basis registry・component型へ接続する技術接口を用意した。partial-S2の実preparation、RTE tail正規化・sampling、実Gaussian/controlled operatorの同値性は未検証であり、科学条件待ちのまま保持する。P-H4の採択、C-DFの性能優位や実H4同値性を意味しない。

## 1. 固定資料と今回の承認境界

起点は `e8a0cd683f65e189f5900ce002af51fce2c53059`。[入力回収・接口調査](research_c_c1_input_ffsim_readiness_20261024.md)、[前段C1調査](research_c_compressed_df_interface_survey_20261024.md)、[C4終了・資産引継ぎ](research_c_c4_closeout_handoff_20261024.md)を再利用した。ffsim v0.0.83／commit `558d3517385778c733f1aa226bb61b8f30f4b0f3`の[Hamiltonian SOURCE][FFH]と[fit SOURCE][FFFIT]のnumber表現を対象にする。

ユーザーが今回承認したのは最小adapter、係数・型・metadata・mock dispatchの人工検証と文書・未承認科学契約の公開まで。実H4科学配列のdecode、元Hの再構成、fit、実QuantumCircuit build/compile、Operator/Statevector、tensor残差・energy・operator norm、保存結果の科学的再解析は実行していない。

正式共通指示 `research_project_instructions_unified_2026-10-23.md` のSHA256は `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f`。§8.4・§10・§10.3を適用し、実装・検証を必要な数学的差分に絞った。専用worktree／branchで作業し、旧SOURCEや共通研究状態を変更しない。

## 2. 追加したファイルと旧実装を変えない理由

| 新規ファイル | 役割 |
|---|---|
| [`src/trottertracks/c1_general_k_adapter.py`](../../src/trottertracks/c1_general_k_adapter.py) | 一般K/frame/model、mock ffsim array取込、係数、primitive/emitter、basis/componentの橋渡し、reference元データ |
| [`tests/management/test_research_c_c1_general_k_adapter.py`](../../tests/management/test_research_c_c1_general_k_adapter.py) | 人工13件。実回路・operator・fit・分子生成を禁止したmock環境で既存式/型/emitterを再利用 |
| 本書 | 仕様・技術結果・再利用範囲・未接続事項 |
| [未承認科学契約案](research_c_c1_general_k_science_contract_draft_20261024.json) | Stage1/2/3、入力候補・未採択条件・担当・call算出規則。実行grantではない |

既存 `DFHamiltonian`／`DFBlock` は `λ(a†Ga)²` とeta/lamを意味するため、一般Jをそのまま入れられない。追加した型が必要なのはこの表現差だけで、既存Gaussian、RZ/RZZ合成、wrapper、compiler、科学executorを新設していない。module importは標準libraryだけで完了する。既存trotterlibへのbridgeは明示的な関数呼出しでlazy importする。大きな既存 `trotterlib.__init__` や旧route exportを変更せず、技術用moduleを独立namespaceへ置いた。

## 3. データ型・取込schema

`GeneralKFrame`は `frame_id,group_id,original_index,K,orbital_rotation,source_identity,input_identity` を持つ。ffsim入力では元 `spatial_J,spatial_orbital_rotation` も保存する。matrixはimmutable tupleへコピーし、元arrayの後変更に影響されない。Kは非空正方・実・有限・厳密対称を要求し、非対称を黙って対称化せず、負係数・不定値・zero/sparseは許す。U/Vはshape・finite・向きの宣言のみを確認し、unitarityやGaussian作用を認証しない。

`GeneralKModel`は補正済みspin κとconstantをframe列から独立して所有する。元hを再補正する関数はない。κはshape/finiteを確認し、Hermiticityの科学許容値・暗黙のprojectionは導入しない。将来の科学gateで確認する。frame IDの重複・元outer順序の入替・model/frame間のdimension/provenance不一致は拒否する。`original_outer_rank`は元frame数であり、補助square rankを計算しない。

`from_ffsim_number_arrays`はNumPy arrayまたは人工tupleを受ける。必須の `one_body_role="ffsim_corrected_kappa"` と `z_representation=False` を照合し、`enable_general_k=True`がない呼出しを拒否する。Z入力を自動number化せず、元hとκ、表現属性を推測しない。frame/group IDとsource/input identityはcallerが明示する。ffsim自体をimport/実行しない。

`SCHEMA=C1_general_K_number_v1` のcache identityはframe順・group・係数・回転・κ/c・source/inputを含む。basis requestにも `c1-general-k-basis:` の別namespaceを使い、旧DF/C4科学cacheを読む・書く処理はない。basis request IDはGaussian作用の検証済み宣言ではない。

## 4. J→K、U→Vとprefactor

ffsim number表現の中心 `1/2 Σ_lijστ J_lij n^l_iσ n^l_jτ` を、interleaved `a=2i+σ` へ

`K_(2i+σ),(2j+τ)=J_ij/2`、`V_(2p+σ),(2i+τ)=δ_στ U_pi`

で写す。Kは全prefactor込みで、`D(K)=Σ_ab K_ab n_a n_b`は対角と両順序を含む。元J/Uを保持し、取込後のK/Vとの不一致を拒否する。interleaved→block permutationは `a→floor(a/2)+(a mod 2)N`。これはsingle-particle indexの配置であり、Qiskit state indexのbit reversalとは別。

`gaussian_request()`はV、helper名 `U_to_qiskit_ops_jw_givens`、内部 `BogoliubovTransform(V.T)`、作用順inverse→中心→forwardとCAR検証pendingを記録するだけで、合成器を呼ばない。目標 `Γ(V)exp(−iτD)Γ(V)†` とffsim Uの関係は、将来の独立CAR・全Fock-space/controlled検証で確認する。今回自動的に一致済みとはしていない。

## 5. 係数・角度・scalarの所有

`general_k_coefficients`は純粋な係数算術で、

`c_K=(1ᵀK1+tr K)/4`、`z_a=−(Σ_b K_ab)/2`、`z_ab=K_ab/2 (a<b)`

をI/Z/ZZ supportと共に返す。対角K_aaはn_a²=n_aとして扱い、同じ空間軌道の異spin密度積も含む。row/全係数和はmath.fsumを使い、nonfinite結果・算術overflowを拒否する。科学的cutoffは持たない。

`diagonal_primitives(frame,τ)`は既存 `DiagonalEvolutionPrimitives` を返す：phase `−τc_K`、RZ/RZZ角 `2τz`。primitiveではexact-zero角のみ省く。旧square primitiveのzero-RZZ placeholderは残る場合があるが、旧emitterもexact-zeroを省くので、人工退化testでは発行される非零データを比較した。

τは有限実数の入口に限定し、symbolic parameter bindingは今回未実装。これは技術接口の範囲であり、科学的な時間窓の採択ではない。

`append_general_k_diagonal`は既存 `append_diagonal_primitives`を呼ぶだけで、回路は作らない。controlled時はancilla指定を要求し、system register内のancillaを拒否する。既存emitterは中心controlとancilla PhaseGateへscalarを保持する。今回はgate/circuitをtoken recorderへmockし、このdispatchを確認した。

| 所有する層 | 含める項・含めない項 |
|---|---|
| model | 補正済みκと元constant。再補正・Z変換・対角化はしない |
| frame primitive | そのD(K)だけのc_K/Z/ZZ。κ/cを読み込まない |
| component候補 | 同じframeのI/Z/ZZを正規化前の既存型へ写す。identityは保持し、移動先を決めない |
| 将来のfull preparation | 元constant、one-body diagonal scalar、frame scalar、tail identityの各1回計上を検証。policy未採択なので今回組み上げない |

## 6. partial-S2／RTEへの接続準備

| 既存SOURCE・資産 | 今回の再利用・差分 | 保留する境界 |
|---|---|---|
| `df_hamiltonian.py`／`df_trotter/model.py` | 旧square意味を保全。一般Kは別型 | G² kernelへの無変換格納をしない。符号付きsquare化は今回未実装 |
| `df_trotter/ops.py` | 既存primitive型とemitter。rank1退化を旧square係数/primitiveと比較 | 実Gaussian synthesis、実controlled gate buildは未実行 |
| `df_partial_s2.py`／`df_partial_s2_repeated.py` | 新 `BoundGeneralKFrame` は元frame/order/basis definitionを保持するdeterministic-spec準備用sidecar。中心は既存emitterへ渡せる | 旧specのeta/lamを変更せず、旧builderへ新型を偽装しない。実preparation/dispatchとboundary融合はpartition政策が決まってから限定拡張 |
| `df_rte_tail.py` | `bind_frame`が既存 `DFBasisRegistry.register` にcaller提供opsを渡す。`diagonal_components`は既存 `DFDiagonalPauliComponent`を返す | 既存registryのgate fingerprintはlocal Operatorを評価し得る。今回そこはmock。実ops bindingは別承認の科学gate対象 |
| `df_rte_circuit.py` | `component_specs`が既存Z/ZZ specとidentity specを返す。group情報はbound frame sidecarに保持 | DFTailExtraction／NormalizedRTETailは作らず、atom選択・identity移動・閾値・λ_R・sampling未決定 |
| `df_rte_qiskit.py` | 将来このspecを採択済みtailへ接続すれば、ordered eventとbasis共有・中心controlを再利用可能 | 新しいevent列・law・finite meanを今回作らない |
| `df_partial_randomized_pf.py` | prefix・group・入力同定の設計参照 | square重みを一般Jへ流用しない。新ranking/L_D/PF係数を作らない |
| C4 `reference.py`／gates/circuits | CAR/exterior、absolute phase、control0/1、actual adjoint、X/Y、compilerの再利用設計 | 旧4-mode fixtureのPASSやreferenceを分子H4へ転用しない。B2も合法な型・境界で両方式へ同じ機会を与える場合だけ |

`bind_frame`は明示opt-inであり、既存Gaussian opsはcallerが別途用意する。component変換は**候補となる型変換**までで、研究Aのtail atomization政策の正式採択ではない。旧full preparation・lawを仮定してend-to-end経路を通す実装はしていない。

bound recordは `validation_status=NOT_NUMERICALLY_VALIDATED` を保持する。既存component specのrepresentation/type名は規約上の型名であり、今回の数値検証PASSや物理的unitarityを表すstatusではない。

## 7. 人工unit/mock結果と技術修正

既存環境のPython3.11.0rc1・NumPy1.26.4を使い、追加installなしで次を実行した。Qiskit1.3.0／Nature0.7.2のpackage metadataは確認したが、テスト中のQiskitはmockのみ。

```text
python -B -m unittest discover -s tests/management -p test_research_c_c1_general_k_adapter.py -v
Ran 13 tests ... OK
```

テストは小さい人工行列・係数とmock tokenだけである。既存SOURCEの係数式、primitive型、emitter、registry/hash、component型を読み込み、広いlibrary初期化とQiskitをpassive module/mockへ置換した。実QuantumCircuit、transpile、Operator/Statevector、Gaussian synthesis、eigh/normを禁止するguardを置いた。RTE正規化・sampling・旧研究CLIは呼ばない。

| coverage | 確認内容 |
|---|---|
| K | 非空実対称・shape・有限値、非対称/complex/bool拒否、一般符号、diagonal/offdiagonal、zero/sparse、overflow |
| 数学的退化 | 人工rank1 Kで既存 `exact_df_diagonal_coefficients`／`df_squared_diagonal_primitives`の係数・非零angle/phaseと一致 |
| spin/表現 | J/2、interleaved/block permutation、異spin、complex Uの配置、number/Zの二重変換防止 |
| scalar/phase | κ/cをframeへ二重追加しない。controlled相対phaseとuncontrolled global phaseのmock dispatch |
| provenance/境界 | immutableコピー、元順序・group・元R、別cache namespace、未指定科学policyを推測しない |
| bridge | NumPy mock出力、実既存型、basis conflict、candidate component/spec、未評価reference元データ |

初回12件では、(a) τ=0時の旧zero-RZZ placeholderと新exact-zero省略の表現差、(b) registryのlocal Operator fingerprintをmockすべき境界、でFAIL/ERRORが出た。係数算術の誤りではなく、emitterが使う非零データを比較し、gate descriptor境界をmockするよう修正した。guardが実Operator評価を阻止しており、実operatorは実行していない。その後NumPy moduleを同一lifetimeに保ってreload warningを解消し、array境界testとκの科学gate保留を加えた最終13件がPASSした。技術修正・再testの履歴であって科学runのretryではない。

実行可能CPU/affinityは32、物理RAM約64GiBを確認したが、CPU/RAM/worker capは設定していない。tiny算術と一時的sys.modules mockが主処理のため、標準unittestの単一processで完了した。重い計算を小workerに制限したものではなく、GPUや追加並列化を要するworkloadは今回ない。speedup倍率は主張しない。

これはcommitされるローカル人工test記録であり、実H4科学同値性、実runtime統合PASS、CI・外部再現や資源改善の証拠ではない。

## 8. 独立referenceと6つの誤差層

`GeneralKModel.reference_input()`は元のnumber-density κ/cとraw spatial J/U（直接general Kの場合はraw K）を返し、candidateのZ係数やprimitiveをreferenceへ再利用しない。physical h/gは保存input identityから別に取得する必要がある。CAR数値評価・reference生成はしていない。

| 層 | 将来の独立対象 |
|---|---|
| 1 元H→C-DF | 同じMO h/g/cからH_physを作り、採択したone-body政策のH_CDFとの近似誤差を評価。tensor残差とoperator/sector誤差を分ける |
| 2 C-DF→general K | raw κ/J/U/cのCAR referenceとK-density作用を照合。J/2、spin、U向きとscalarを確認 |
| 3 general K→native回路 | 独立CAR/number作用対raw circuit、control0=I/control1=U、full-space相対phase、actual adjointとHadamard X/Y |
| 4 PF/finite-RTE | 新しい分割・atom列のordered finite target。研究B/Aと条件決定後に照合 |
| 5 compiler | 同じfinite targetに対するcompiled operatorとsignal。global phase fittingで差を隠さない |
| 6 sampling/統計 | 研究Aのlaw/shot/confidenceを別評価し、同精度総資源へ接続 |

実H4入力P-H4は前回の区分Aとして利用可能だが、今回初回対象に採択していない。packet `86b8b002704aced8ca239414080f53793de81658:artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz` のSHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`を来歴として引き継ぐだけで、科学arrayはdecodeしていない。

## 9. 未承認科学契約と担当

[契約案JSON](research_c_c1_general_k_science_contract_draft_20261024.json)は `UNAPPROVED_UNEXECUTED_NOT_A_GRANT`、execution_authorized/grant_issued/dispatch_enabledは全てfalse、commandなし。新science grantを発行せず、未選択値はnullにした。

| 決定者 | 未決定事項 |
|---|---|
| ユーザー・研究C | P-H4採択、一般frame接口の次scope、one-body政策、X-DF対照、C-DFのrank/tol/optimizer設定 |
| 統括 | 対象sector/full-space、DF/総誤差予算、比較主張、Stage1/2の採択と実行承認 |
| 研究Aとの共同判断 | tail atomization/identity/law、normとfinite mean、shot/confidence |
| 研究Bとの共同判断 | D/R group/split、L_D、PF時間窓/step/係数とbias、finite target |
| Codexの技術判断 | 採択後の独立環境・既存helper/検証器再利用、provenance、資源と並列化、確定した実R/frame数からcall一覧を作る |

Stage1は表現/operator、Stage2は回路/native費用、Stage3は別の後続判断で有限PRと総資源。fit/reference/build/transpile数は実R・unique frame/time/control・ordered case/axis・compiler/cacheが確定してから列挙する。架空の固定call数、rank10/12の流用、旧tol/sector/δの自動採択はしない。Stage1/2にも別明示承認が必要。

将来の重いfit/reference/compileではSOURCEとtimingから支配的処理を特定し、利用可能CPU/GPU/RAMに応じJAX・matrix-free・Numba・cache・独立case並列を使う。processとBLAS/OpenMP/Rayon等を協調させ、科学条件・精度・seedを維持する。恣意的なCPU/RAM/worker/phase・total wall capを設けず、call/output/log/STOP境界を保つ。共有venv/cache/設定や他jobを変更しない。

## 10. 完了範囲、次段階とSTOP

今回追加したのは技術adapterと人工test、報告と未承認契約の4ファイル。旧SOURCE・raw・結果・grant・ledger・marker・STOP、C4/Gaussian一時終了、研究A/B契約、共通研究文書と元workspaceを保全する。

**次はユーザー・研究C・統括**が入力・sector・one-body/DF誤差・対照と段階を選び、研究A/Bへpolicyを照会する。**Codexの次作業案（別依頼・別承認）：** 選定条件を科学契約へ固定し、必要な独立fit環境／実preparation dispatch／CAR referenceを接続する。今回の型準備から実fit/build/compileへ自動進行しない。

新規fit、実H4数値計算、Hamiltonian/積分生成・再構成、実QuantumCircuit build/compile、実Operator/Statevector、科学残差/energy/norm、sampling、旧結果再解析は0。RC-DF/SCDF、C4/R2、Gaussian singletonも再開しない。専用branchのcommit/push・remote SHA/全変更blobの照合で**mandatory STOP**する。

[FFH]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/hamiltonians/double_factorized_hamiltonian.py
[FFFIT]: https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/linalg/double_factorized_decomposition.py
