# Track D：実PR条件への移送に向けた静的調査

2026-11-06 JST。**第一候補は実H4・number-CDF8の保存固定word02、予備はlegacy DF12の保存8-event path。真空Referenceを適用できる構造と来歴は追えるが、実PRでの削減・correctnessは未検証。** 第一候補もPR分布から新たに抽出した標本ではなく、既存研究が事前固定した有限wordである。

本調査はSOURCE本文、固定JSON、Git object、入力bytes/hash、NPZのmember名・headerだけを照合した。旧H4 NPZでは保存metadata文字列も読んだ。科学配列の展開・演算、QPY decode、回路生成、compile、simulation、sampling、数値再解析、実装変更は行っていない。既存SOURCE/raw/結果/grant/STOPは変更していない。

## 1. 固定資料と判定範囲

| 資料 | 固定版・役割 |
|---|---|
| [Track D人工検証](../track_d_minimal_reference_validation_20261106/RESULT.md) | `102e3c2f29999fb9fbdfa6f85020370df228a7f3`、実行SOURCE `1cc8c87c297e3b526b4e18f20f44da3b70f86157`。2-modeの有限技術検証であり、分子結果ではない |
| [Track D技術調査](../track_d_reference_control_feasibility_20261106/REPORT.md) | `c3cd64755944be4aad2e032affd1ed0308cf5743`。Referenceの一shot条件と強いcontrol baseline |
| [3環境統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105) | `3571477093441ac4852166fe829f8ccc5db5ec4f`。索引・来歴の入口。carrier commitを各runのSOURCEと同一視しない |
| 正式共通指示 | 利用可能な `research_project_instructions_unified_2026-10-25.md` の§8.1/8.2/8.4/10/10.3、適用AGENTS、PROJECT_MAP、研究概要・現状を確認。旧契約とSTOPを保持 |

初期systemは**元Hamiltonianの厳密基底状態**を仮定し、状態準備コストを除外する。Referenceはfull-Fockの真空。保存数値stateを厳密基底状態と認定したり、own-DF groundを元H groundへ読み替えたりしない。一shotの任意入力に対するobservable一致を確認できれば、この初期状態仮定にも適用でき、ground solveを追加する必要はない。

## 2. 候補の絞込み

| 候補・元条件 | 保存入力・ordered path | 真空・位相・比較可能性 | 判断 |
|---|---|---|---|
| **C-H4：number-CDF8、LD2、S2/s1/r32/K0、h=δ=0.1、8 system qubits** | 元h/g/scalar、最終J/U、独立CAR generators/H、atom定義、固定word02の全32event、raw/compiled/actual-adjoint QPYが固定Gitに存在 | number-conserving Gaussian＋signed Z/ZZ。元g補正κ、tail identity、scalarを追える。相対Gaussian境界融合済みbaselineあり | **第一候補**。異frame・Z/ZZ・負符号を含む既存 `alternating_frames` をそのまま使用。全law・母平均の代表性は主張しない |
| **A-H4：legacy DF12、generation prefix6＋one-body、B2/K2/R8/q4、T=0.8、δ=0.2/r2、8 qubits** | 元NPZ、157-file byte-freeze manifest、replica0の全8eventとpaired control費用JSONを追跡 | 保存eventのbasis gateはRZ/XXPlusYYのみ。ordinary/symmetricが同じpathを使用。ただし調査したNPZ/結果には完全なdeterministic準備値を確認できない | **予備**。短い実sampleだが、元D準備とSOURCE closureの回収・一致確認が先。seedから再samplingしない |
| 旧A-H6：legacy DF19、prefix5/10/15、12 qubits | [旧基底固定準備](../track_a_h6_frozen_basis_preparation_v1.md)は旧全準備archiveの未受領を記録。部分eventだけでは完全復元できない | 型としては真空候補でも、旧prepared hash/全D基底の一致が未確認 | 今回選定しない。この履歴資料の未受領状態を全環境の現在状態と断定しない。新計算で旧準備を補わない |
| non-DF Pauli H6：1Å/STO-3G、12 qubits、T=0.8 | [保存model/provenance](../../../artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/provenance.json)、918非identity項・all-Z prefix78、H32/H64等の既存cost evidence | 保存項に `XXYYIIIIIIII` がある。個々のPauli pathは真空rayを一般に保存しない。全Hの粒子数保存では保証できない | 真空Referenceの初回対象から除外。特別な全word相殺・別referenceは未調査であり、不可能と断定しない |

両H4はlinear H4・隣接1.00 Å・STO-3G・Nα=Nβ=2・sector36。ただしlegacy DF12とnumber-CDF8は別Hamiltonian表現・別split・別SOURCEであり、費用や精度結果を混ぜない。C-H4は1 h-stepだけを対象とし、親研究の13 RPE round、CA、shots、T=0.8のA-H4全pathへ外挿しない。

## 3. 第一候補C-H4のSOURCE・入力

科学実行SOURCEは [`ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7)、実装SOURCE `6e8475cc871916473d48fd38fb4b4a0297e84d15`、結果は [`3649b75af191916daf44abb91e14d34c43b83e8f`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3649b75af191916daf44abb91e14d34c43b83e8f/artifacts/research_c_fixed_word_pilot/2026-11-04/run01)。[元結果説明](../research_c_fixed_word_pilot_execution_20261104.md)と[固定契約](../../../artifacts/research_c_fixed_word_pilot/2026-11-04/run01/contract.json)に戻った。

今回のcheckoutでは主要NPZ/QPYと `c_fixed_word_pilot.py` が収集されていない。しかしSOURCE commitのGit objectから、契約の**保存入力16件すべてとSOURCE143 fileすべて**のSHA256一致を確認した。欠損入力を再生成したのではない。SOURCE manifest SHA256は `22284b98e3d2fa37fc134e734f402d96b3f3989f5c0e3fba5a2098fb03d7847d`。新実行時はこの固定版から必要範囲を回収し、現checkoutの似た関数へ黙って置換しない。

以下のpathはrepository相対。入力4件の固定bytesは上記SOURCE commit、word/QPYは結果commitにある。

| 入力 | 保存path・同一性・役割 |
|---|---|
| 元H | `artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`、SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。headerはh(4,4)/g(4,4,4,4)/constant scalar。κは**元gからのcorrected_kappa**、nuclear/model scalarは一度だけ |
| CDF endpoint | `artifacts/research_c_near_practical/2026-11-03/run01/PR/endpoint.npz`、SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。J/Uは各(8,4,4)。新fit・rank変更を行わない |
| 独立reference | 同run `physical/PR_update016.npz`、SHA256 `ebdebe8f2c3f156c8202834181e143298d667490b3f33fc611dc0fab97b976cd`。同J/U、9 generators、sector Hを保存。`physical_reference.npz` は元Hとsector indices、SHA256 `92a8d27bdebb53937f6361c106a13ef90b9110930d4bf4ddb577796cfb208739` |
| signed atoms | `artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/atoms.json`、SHA256 `8b5bcadb458e2eb345aa779e171f8cc4f28467a648c73c4bc9ec4f1e8f70bfbf`。216 atomのcoefficient/frame/supportと固定wordを束縛 |
| deterministic部分 | `c_pr_aware_df_stage_b.py::prepare_arm` → corrected κのone-body、元frame順F0,F1。`c1_stage3_native_bridge.py::partial_kernel` が前半 `(one_body,F0,F1)` 各h/2、tail32event、後半逆順各h/2を接続。先頭3 blockの対称headであり、tailの並びは反転しない |
| scalar/identity | 上記constantと `plan.extracted_identity` を `P(-(constant+identity)h)` で一度だけ会計。保存 `artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json` はidentity `0.041217410629575235` を記録。normalization・event確率とは別 |
| 固定ordered word | 結果run `samples/word02_events.json`。`CDF_proposed:F2:0`（+、Z0）と `CDF_proposed:F3:0,1`（−、Z0Z1）を交互に16回。K0、product列空、event.phase=+1。保存rotation_angle `0.0015626369893203945`。これはunsigned保存角で、係数符号は別に適用 |
| 元実回路 | 結果run `circuits/word02_{raw,compiled,raw_adjoint,compiled_adjoint}.qpy` が存在。raw SHA256 `f87aa606c917a146ef553f20a1782223bad5bc434017cf56a9bf002aa97c7692`、compiled `af5ddc5b6aae9924dcd0f417976e3c65b2e17acd42ea912b2501e07b75915713`。今回はbytes照合のみ、decodeせず |

eventの `basis_change_operations=[]` は恒等basisの意味ではない。C系列は保存Uとproviderからfull frameを供給する。basis hashだけでGaussianを復元したことにしない。one-body κのspin-adapted固有基底とGaussian gate列は上記NPZに独立memberとして保存されていないため、将来のloweringでは元SOURCE/精度規約と保存raw回路の位相・演算子一致を確認する。元J/Uを再fitしたり、別固有基底規約へ変更したりしない。

元検証の状態は**own-DF numerical ground**で、元H groundとのoverlapはnull。元runのX/Y PASSを今回指定の元H厳密groundの実測結果とは呼ばない。またStage C ready=false、whole-law bound=null、actual CA=null、元mandatory STOPをそのまま保持する。

## 4. 予備A-H4の来歴と不足

[元H4 v5結果](../track_a_ax2b_h4_pilot_v5_result.md)の実行SOURCEはbase `b2e1bf65e21893b6c617223b42313623d3186f12` 上の**未commit bytes凍結**。[preparation manifest](../../../artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/h4_preparation_manifest_v5.json)の `local_uncommitted_byte_hashes` /157-file closureが正本で、baseを全SOURCEと呼ばない。既存Track D調査でnative v5等3部品を `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` と照合済みだが、全closure回収の証明ではない。

元snapshot `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz` は元workspaceに存在し、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` がmanifestと一致した。前回[静的調査](../track_d_reference_control_feasibility_20261106/MINIMAL_VALIDATION_STATIC_CHECK.md)の「隔離checkoutにない」という記録を、全環境で未保存という意味へ拡張しない。

metadataのHamiltonian hashは `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`、constant_hexは `0x1.25845777d3feap+1`。保存memberはconstant/one_body/lambdas/g_matrices/sector indices/full・sector state/metadataで、完全なD固有基底・gate準備は含まない。`_prepare` はgeneration partition、identity抽出、coefficient_atol=0の準備処理を呼ぶ。今回これを実行して補完していない。保存stateは元結果でもground_state_certified=false。

保存pathは `artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/H4_B2_K2_rep0_trajectory.json`。seed2343775440は来歴だけに使い、ordered event digest `07d86db03ab3544ae9e3fb77b2ad173fbe5fa45f8219bb3a5c0277100e1a922f` を維持する。

| outer step | event 0 → event 1 |
|---|---|
| 0 | fragment6:Z1Z3 (−) → fragment6:Z0Z4 (−) |
| 1 | fragment6:Z0Z1 (+) → fragment6:Z1Z3 (−) |
| 2 | fragment6:Z5Z7 (−) → fragment7:Z6Z7 (+) |
| 3 | fragment6:Z0Z6 (+) → fragment6:Z4Z6 (−) |

全8eventは保存上K=0、product空、phase=+1、rotation_angle `0.002207888449316714`。設定K2だからといってproduct coverageがあるとはしない。保存basis gate列はRZ/XXPlusYYで真空rayを保存する。既存ordinary/symmetric×cosine/sineは同digestだが、元correctnessは保存stateとsectorの先頭・末尾basis stateに対する両ancilla branchの比較で、全operator認証ではない。旧入力の生成・rebaseline・停止履歴を新namespaceで迂回しない。

## 5. Reference条件・path位相

物理的number-conserving Gaussian Wとそのactual inverseによる `W Z/ZZ W†` は真空を保持する。C-H4のnumber-density headとA-H4のnormal-ordered one-body/DF-squared headの真空energyはscalarを除き0。C系列の `c1_h4_strong_baseline.py::vacuum_phase` はRZの `−angle/2`、XXPlusYYの真空作用、circuit.global_phaseを解析的に追跡し、relative Gaussianにも位相補正を入れる。単独basisの位相を無視してよいという意味ではない。

各eventの真空作用は、signed rotation角をφ_j、product各因子の真空符号をs_jkとして、`event.phase × exp(−i φ_j) × ∏s_jk`。全pathのχはこれらと元scalar/extracted identity、露出したbasis/global phaseの合成で取得する。選定2pathはK0/product空なので、complete sandwich規約では `χ = −(c + c_I)t − Σ_j φ_j (mod 2π)`。Cではt=h、Aではt=T。**今回χの数値を新たに算出していない。** 元Hの真空energyだけでrandom pathのχを代用せず、B/attenuation/確率を位相へ混ぜない。

同じ保存pathをVω=AωBωに切り、実gadgetの分枝を `Aω†ψ ⊗ Bωr` と `Bωψ ⊗ Aω†r` にすると、coherenceは `〈ψ|Vω|ψ〉〈r|Vω†|r〉`。`Vωr=e^{iχω}r` とancilla P(χω)で、fresh-shotのX/Y条件付き分布が直接制御と一致する。任意ψに対するこの条件整理は既存Track D結果の適用であり、coherent QPE全体・反復後のsystem back-action・全PR estimatorの実装検証ではない。

## 6. 公平なbaselineと再利用

| 資産 | 実装済みの範囲／移送時の境界 |
|---|---|
| diagonal-only control | legacy `df_rte_qiskit.py` とC `pr_dictionary_validation/circuits.py` はbasis/parityを非制御、中央rotation/productと位相を制御。Gaussian全制御を主baselineにしない |
| symmetric-directional | `resource_applicability/ax2b_native_df_v5.py` は対称Dの前半uncontrolled・逆順後半directional、中央tailはordinary-controlled。tailのpalindromeは不要。**Cのgeneral-J NativeBlockへ接続済みではない**。generic diagonal primitivesと `append_directional_diagonal` の再利用を検討し、DF-squared型builderへCDFを無理に渡さない |
| basis共有・融合 | legacyは同ID/hashの隣接event共有とrepeated D0半step融合。CのStrongBuilderは同rotation runのdiagonal fsum、phase補正付きrelative Gaussianをhead内部に使用。tailは既存 `build_event_sequence(optimized=True)` の共有。C元SOURCEはcross-head/tail relative fusion=falseであり、全境界融合済みとは呼ばない |
| actual inverse・符号 | 既存event順序/negative atom対応は修正版 `c1_stage3_native_bridge.py`、独立CAR/event referenceは `c_pr_aware_df_stage_b2.py`。A†は実回路inverseで、角度だけの反転や独立drawではない |
| compile・wrapper | 人工版 `track_d_minimal_reference.py::{hadamard_wrapper,metrics}` とF2 `fixture.py::CompilerProfile` を再利用。Cでは `c1_h4_native_fair.py::compile_checked` → `c1_h6_compiler_precision.py::compiled_wrapper` → F2 `gates.py::compiled_wrapper` の既存経路を優先する。MPFR160でdeclared phaseを蓄積し、nested定義を位相保持でunrollする。人工版のF2直接呼出しだけではCの位相精度規約を満たさない。全armへ同じ処理を適用 |
| 既存tests | 人工 `tests/track_d_minimal_reference_checks.py` のχ欠落negative check、observable/phase check。C固定版 `tests/management/test_research_c_fixed_word_pilot.py` の固定word/CAR/P0/adjoint/X-Y、`test_research_c_c1_h4_strong_baseline.py` のrelative向き・解析位相・符号・fusion回帰、Aの `tests/tracks/resource_applicability/test_ax2b_native_df_v5.py`。今回は読んだだけで実行していない |

CのSOURCE版 `c_fixed_word_pilot.prepare` / `c_pr_integrated_performance.native(fixed_words=...)` が保存wordの受け渡しと検査の入口。ただし既存runnerは一式科学実行と元fresh-output/STOPに結びつくため、そのまま再起動しない。現在checkoutの `native` にはfixed_words引数がなく、固定SOURCEに戻る必要がある。旧grantや実行許可は移送しない。

人工adapterの最小変更点は、(a)2→8 modeとregister割当、(b)同一S² fixtureから保存32-eventの一つのpathへの置換、(c)Cのuncontrolled A/Bとdirectional headを既存primitive/位相規約で接続、(d)χのmetadata読取り・actual adjoint、(e)correctnessの入力固定部分とobservable評価の拡張。`reference_gadget` は2-mode限定、`induced_observable` は3/5 qubits・先頭4列限定なので、そのまま実H4へ適用できない。巨大な17-qubit全Operator生成もそのまま拡大しない。既存state-action/CARと必要な入力部分空間に適した検査を選び、full-operator/有限state/observable検査の資格を区別する。新Reference実装は今回行わない。

## 7. 次に必要な最小検証案（未承認・未実行）

**対象はC-H4のword02だけ、1 h-step、保存event16の直後というcomplete-event境界で一度だけA/Bに分割する案。** 初期状態は元H厳密ground、準備費除外を維持する。前半/後半を新たにsampleせず、同じordered pathのscalarを一度だけ配る。これは固定構造からのcutで、cost最適化やS²の仮定ではない。

1. **入力/SOURCE**：上記143-file SOURCE・16保存入力と結果word/QPYを固定版から回収。元κ/F0/F1の順、J/U、Gaussian/phase規約、K0/r32/h、signed atoms、identity/scalar、basis keysを保持。保存raw kernelとの位相を含む一致を確認できない場合は進めず、保存不足を再fit・別basis・新sampleで埋めない。
2. **correctness**：同じVωに対しordinary、symmetric-directional、Referenceを照合。rawとcompiledのX/Y induced observable、およびχを除いた負対照を検査。direct controlのcontrol=0 identity/control=1 V、A†のactual inverse、真空ray・χ・global phase・normalizationを別々に確認する。元Hgroundに依存しないobservable検査を優先し、own-DF保存stateを代用しない。
3. **比較**：basis共有・head relative融合を保持したordinary、同じ最適化機会を与えたsymmetric-directional、Referenceの3arm×X/Y、**full-wrapper compile計6件**を提案。既存Cの保存価格はcomplete controlled kernel価格で、ancilla shell/測定込みのwrapper価格ではない。保存kernel価格と新wrapper価格を直接比べない。融合した強いbaselineを作れない場合は、弱い対照で優位判定せず不足を報告する。
4. **精度/compiler**：C元pilotのbinary64、component abs 1e−10、unitarity abs 1e−10、complex signal 2e−10、rtol=0、global phase fitなしを開始条件とする。2-modeの3e−12を実H4に無検討で移さず、閾値の事後緩和もしない。P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initial-zero false、Gaussian `nature_jw_exact_zero_lapack_atan2_v1`、MPFR160 declared phase。数値PASSを厳密包囲・化学精度・全law保証に昇格しない。
5. **資源**：RZ count/depth、CX count/depth、total depth、qubitsを別々に報告し、主指標はまだ固定しない。direct9 vs Reference17 qubits、前後2層の8対CSWAP、P(χ)、cutで失う相殺と配線条件を含める。topology-free結果と将来routing費を区別する。初回は期待価格/N/全RPE総費用を求めない。
6. **実行準備**：承認後に実CPU/RAM/GPUと既存timingから支配処理を確認し、独立arm/axis・既存cache・BLAS/Qiskit並列設定を調整する。17-qubit dense全Operatorへの単純拡大を避ける。恣意的CPU/RAM/worker/wall上限や共有環境変更を追加しない。今回資源probeや計算benchmarkは実施していない。

**STOP条件**：元SOURCE/入力/word/phaseの不一致、必要準備の欠損、真空保存未確認、wrong sign/order/adjoint、phase fitを要する差、correctness/compile後検査FAIL、公平なbaseline不成立、共有環境への干渉懸念で新しい科学callを止め、既存結果を保全する。retry/resume・別word・別候補・精度変更・再sampling・ground solve・新DFへの切替を自動で行わない。完了/FAILとも公開照合後mandatory STOP。

## 8. 今回の公開・引渡し

専用branch `codex/track-d-real-pr-transfer-static-20261106` に本Markdownだけを追加する。公開後、remote branch SHAとcommit SHA、公開ファイルとlocal bytes/Git blobを照合する。新しい科学結果ではないため、既存研究正本・validation manifest・grant・STOPへの更新は行わない。

**本調査の公開照合後mandatory STOP。** 次担当はユーザー：上記「C-H4/word02・固定cut・有限correctness＋6 full-wrapper compile」という範囲と精度を採択し、新科学実行を明示的に許可するか判断する。Codexは許可後に必要最小限のadapter拡張・固定SOURCE準備と指定検証を担当する。重要科学レビューが必要な場合は実行許可とは別に許可を得る。新grant/承認証明の発行・承認原文保存は今回も次段階の通常要件にしない。
