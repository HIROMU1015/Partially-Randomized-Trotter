# 研究C：PR-aware DF C2・Reference Bridgeと全law Synthesisの統合準備

2026-10-27 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。

**Bridgeの最小adapter／限定runner、誤差の統合schema、全law primitive／P0 coverage要件を準備し、人工17＋既存回帰68＝85 testsがPASSした。実H4の新規科学callは0。** 次の最小実験として、保存CAR referenceとK0平均eventを比較する**C2-Bridge・47 logical units**を推奨する。これはsector36の数値対応pilotであり、厳密Bridge boundやwhole-kernel P0の一様合成上界を取得済みとは扱わない。

## 1. 起点と維持する資格

起点commitは`ca078350a5eb35fa8abaece21971c05deb3d755e`。[C2-B結果](research_c_pr_aware_df_stage_c2b_execution_20261027.md)は35/35 COMPLETE、FAIL/retry/resume0、各216 atom表の再生PASS。IID一様53-bit格子モデル下のTV上向き表示はBaseline `3.647073528232258e-15`／Proposed `3.226176237345202e-15`、m12のraw-law条件付き上界は`1.912116885969834e-9`／`1.6914454871252413e-9`。今回は保存値を引用しただけで、係数・確率表の再解析はしていない。

[C1](research_c_pr_aware_df_stage_c1_execution_20261027.md)の約83.1%総RZ削減は2つの明示仮定シナリオの点推定。実装証拠経路の全26roundはUNRESOLVED、保証shots/total/actual CAはnull。C2-BはPCG64のIID性、元積分→atom係数、B1 reference、native synthesisを囲っていない。これらの結果、旧SOURCE/raw/grant/ledger/marker/STOPを保持した。

対象は保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8のBaseline/Proposed、Nα=Nβ=2／sector36、元frame順LD2、元g補正κと核scalar一度。Partial symmetric S2、δ=.1／s4／h1/40／r16／K0、m0..12。Own-DF ground oracle／状態準備除外、CA=.00159 Ha／DF reserveCA/8、共通Gaussian/P0、既存phase/failure条件を変更しない。

## 2. 再利用したSOURCEと新しい差分

| 既存SOURCE／関数 | 適合範囲と今回の扱い |
|---|---|
| `c1_general_k_adapter.from_ffsim_number_arrays/general_k_coefficients` | J→interleaved K/V、元frame順、I/Z/ZZをそのまま使用。新しいDF分解や係数公式を実装しない |
| `c_pr_aware_df_stage_b.prepare_arm/atom_operators/configure` | 同じsplit、signed CAR atoms、λ／τ／K0 lawの再生。NativeModelはデータであり回路をbuildしない |
| `SectorCAR` | 既存spin保存bilinearとsector36。新CAR algebraは作らない |
| B1保存`generators/corrected/ground/full_input_indices` | 科学実験時に再利用予定。Physical H、DF norm、ground eigensolver、PF誤差は再計算しない |
| `prototype.MixedAtom/make_tail`／C2-B`replay_saved_table` | 同一保存catalog・符号・確率を再生。旧C2-B TVは再算定しない |
| `rte._make_event/event_unitary` | 全216 K0 atomのeventを一度ずつ評価する有限平均。Samplingや16-event全word列挙なし |
| `rte.finite_rte_corrected_operator/finite_rte_attenuation` | 元CARの同じ有限Taylor定義とnormalization。保存corrected referenceの再生照合、PF/exact-Hの誤差再計算はしない |
| `c1_stage3_reference._ordered_exponentials` | 保存G0/F0/F1の二つのhalf-headをキャッシュ付きで再利用 |
| `c1_stage3_molecular.component_check` | 既存1e-10・rtol0のcomponent correctness gate。新しい丸めcertificateと扱わない |
| C2`categorical law/synthesis_preview`、既存`repeat_bound/round_synthesis` | 条件付きlaw入力、局所proof入力のcoverage検査、telescopingの出所。未取得boundを生成する処理ではない |
| B2`SharedQiskitCalls`／共通precision/P0 | 人工controlled-phase/compile/actual-inverse試験でも既存mutexを使用。実H4のbuild/compileは0 |
| C1 accuracy pipeline | 旧SOURCEは不変。統合schemaだけ準備し、C1 entry／shots／資源関数を実データに呼ばない |
| `Calls/resource_plan/threadpool_limits` | 将来47-callのjournal・依存STOP・実資源に応じた2arm並列とjob内BLAS調整 |

新規は`src/trottertracks/c_pr_aware_df_c2_bridge.py`、`scripts/run_research_c_pr_aware_df_c2_bridge.py`、同名management test。既存B1はPF/RTE誤差全体のrunner、B2はsampling/native runner、C2-Bは確率表だけのrunnerで、平均event→保存raw referenceの差を分離するendpointはない。そのため既存部品を呼ぶ新しいopt-in接続だけを加えた。旧科学SOURCEと旧契約を編集していない。

## 3. Reference Bridgeの数学的対応

### General-K／spin／frame／符号

空間軌道Jをspinへliftした`K_ab=J_(a//2,b//2)/2`、`V_(2p+σ,2q+σ')=U_pq δσσ'`を使う。Mode順はinterleaved `2i+σ`、Uの列が回転後creation operator。`D=Σ_ab K_ab n_a n_b`の実対称Kに対し、既存係数は

- I：`c_I=(Σ_ab K_ab + tr K)/4`
- Z_a：`c_a=−Σ_b K_ab/2`
- Z_a Z_b（a<b）：`c_ab=K_ab/2`

元frame F0..F7のうち、κ/F0/F1が決定論head、F2..F7がtail。Tail Iだけをdeterministic phaseへ抽出し、λはnonidentity |c_l|の和。符号は`P_l=sgn(c_l) Γ(V) Z_support Γ(V)†`へ一度だけ吸収する。Supportはsingle Z／ZZで、微小nonzeroを閾値で落とさない。

Exact unitary VならP_lはHermitian involution。Binary64のUとCAR縮約行列がその恒等式を厳密に満たすとは扱わない。将来実験では全atomのHermiticity/involution、保存CAR tailと係数和の数値対応を明示する。Sector36ではspin保存bilinear／完全なdensity factorを扱える。**このsector検証を、Gaussian gate個別の射影やnative full512の証明へ転用しない。** 保存`full_input_indices`と現SectorCARのbit mappingも照合する。

### K0 eventと有限Taylor reference

K0ではorder0が確率1で、1microstepにrotation atomを1つ選ぶ。`τ=λ_float*(h/r)`、`φ=atan(τ)`、`b=sqrt(1+τ²)`。Referenceではsigned P_lに対し`E_l=cosφ I−i sinφ P_l`を作る。Native emitterはunsigned Z supportに`sgn(c_l)*φ`を使う。同じ符号を両方へ掛けて二重反転させない。K0のTaylor product／(-1) phaseはなく、r=16回の独立eventを仮定する。

Exact arithmeticかつ同じφ/P_lで、`p_l=|c_l|/λ_exact`なら

`M_atom=Σ_l p_l E_l = cosφ I−i sinφ Σ_l p_l P_l`。

さらに`cos(atanτ)=1/b`、`sin(atanτ)=τ/b`から、`M_atom=(I−iτ H_atom/λ_exact)/b`となる。一方、B1は保存raw CAR tail `H_R`と**configのλ_float**を使い、既存finite Taylorで`(I−iτ H_R/λ_float)^r`を構築する。両者を同じb^rでrawへ戻すためには、`H_atom/λ_exact`と`H_R/λ_float`、φ/b、実際の浮動小数点演算が対応する必要がある。確率表の一致だけでは十分でない。

Identity phase `exp(−ih c_tailI)`は両式の外へ一度だけ掛ける。B1保存G0にはκと核scalarが含まれ、forward/reverseの二つのh/2 headで核scalarが一度分入る。Bridgeでさらに核scalarを掛けない。B2 native側はheadのscalar_time=0と最終ancilla phaseで同じscalarを持つ。この対応をSOURCEで追った。

### Raw kernel／全roundの対象

`K_atom=exp(−ih c_tailI) S_rev M_atom^16 S_fwd`。B1の保存corrected kernelをCとし、raw一歩は`R=A_1 C`、roundでは`A_m C^n`、`n=4*2^m`、`A_m=exp(−16n log b)`。Bridgeでは次の差を別々に数値観測する。

1. Atom係数和と保存raw CAR tail。
2. Explicit全atom K0平均と、そのcos/sinの線形恒等式。
3. 同じfinite Taylor処理のcorrected再生と保存C。
4. `K_atom`と`A_1 C`。
5. `(A_1 C)^n`と`A_m C^n`のnormalization power／exp-log経路差。
6. `K_atom^n`と`A_m C^n`のsector operator2／Frobenius／own-ground complex-signal差。
7. 再生した`<ψ|A_m C^n|ψ>`と保存B1 raw signal。

保存PF operator誤差・corrected RTE operator誤差は引用するだけで、PF kernel／exact evolutionとの差を再評価しない。新しいreference pathの検証なので、旧B1のtail sum PASS・B2のfull512 sample PASSを丸ごと繰り返す実験ではない。Finite atom/eventの正しさgateは、新平均へ接続する入力のframe／符号／基底対応を確認するためにまとめて含める。

## 4. C2-BのTVをB1 raw signalへ使える条件

TVの約10^-9のround上界は、**同じexact unitary trajectory map**へpとqを載せ替えたときのraw complex mean差であり、reference bridgeやsynthesisを含まない。格子の各drawがIID一様という明示モデル、discardされるK0 order drawを含むstreamの独立性、absolute matrix element≤1が前提。固定seed PCG64がそのモデルを数学的に実現する証明はない。

統合のtriangle分解は次の対象間に置く。

`B1 raw CAR reference → exact saved-atom IID mean → IID-grid same-event mean → native raw/compiled measurement mean`。

| 項目 | 共通raw complex-signal単位への対応 | 現時点の資格 |
|---|---|---|
| DF | Phase→energy後に一度だけHaで計上 | 保存数値pilot。ここへ混ぜない |
| PF operator error | A_m倍してraw signalへ | 保存B1 numerical pilot |
| corrected RTE operator error | A_m倍してraw signalへ | 保存B1 numerical pilot |
| Categorical law | C2-Bのconditional raw boundをそのまま使用。Aを再乗除しない | IID53bitと共通unitary-mapに条件付き |
| Atom／angle／λ／normalization／identity bridge | 同じraw対象間のoperator差はnormalized own-ground signal差を抑える。ただし計算した行列の丸め誤差が別途必要 | 実H4では新評価未実施。将来実験もまずobserved distance |
| Native synthesis | 全word／全occurrence／shellを覆うraw-signal boundが必要 | 未取得 |
| State／alias | 同じown-DF groundと保存alias条件 | 保存numerical PASS、実機状態準備は対象外 |
| Numerical reference uncertainty | CAR、cos/sin、行列積、expm／powers／norm等の有効な誤差 | 未取得。0を代入しない |

`signal_schema`は上の項を別欄に保持し、Bridgeへreference uncertaintyを含めた値と、同じuncertaintyの別加算を混用しない。Unknown／observed residual／scope違いは合計null。`categorical_input`と`uniform_synthesis_preview`を接続したが、schemaがproofを検証したわけではない。全入力が人工的に揃う場合も`CONDITIONAL_MODEL_INPUTS_COMPLETE`というpreviewに留め、実PCG64・B1/DFの資格を昇格させず、actual error_evidence／保証shots/totalはnull。C1は再実行しない。

## 5. Whole-kernel P0の追加監査と全law coverage

[Primitive要件](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/primitive_coverage.json)／[現P0 pass構成・cancellation内容](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/compiler_coverage.json)。固定preset managerを構成してSOURCE／task属性を読んだだけで、実分子回路を渡してrunしていない。前段で取得したQiskit1.3.0のRust/Python SOURCEを再利用した。

各armで有限入力集合は、9 Gaussian basis、head内部の4 directed relative products、6 tail bases×36 support/sign、K0の共通unsigned angle、二つのdeterministic half-head、scalar/identity phase、共通ancilla H/X/Y shell。16eventのbasis runは最大16で、既存のexact basis-key共有だけを適用する。入力行列・係数・phaseは保存J/Uと元SOURCEから取得可能だが、今回はdecodeしない。

Raw emitterについて、各有限要素の**full controlled operator誤差enclosure**が得られれば、ideal unitary積のtelescopingで全wordを覆う構成経路はある。Basis回路はsystem作用だが、central controlとの合成後の相対phaseを維持する必要がある。ReferenceのU／density／angle丸めは別の誤差源である。

P0は空EquivalenceLibraryでofficial definitionをbasis+uへ展開後、whole-kernel opt1を行う。TranslationはUnitarySynthesis／HighLevelSynthesis／BasisTranslator、最適化loopは1q run再合成／inverse cancellation等。Source上、1q runのf64行列積、arg/atan2、mod2πとphase補正、default `ANGLE_ZERO_EPSILON=1e-12`による簡略化がある。`approximation_degree=1`は丸めゼロの保証ではない。Event境界を越えるrunがあるので、個別event P0残差の和でwhole-P0を囲わない。

追加確認としてRust InverseCancellationの`gate_eq`はparameter比較に1e-10を使うが、この固定presetが登録するself-inverseとinverse pairは**すべてparameter-free**だった。したがって、それを根拠なく任意のRZ近似キャンセル誤差へ数えていない。主な未資格はrun再合成／角度・phaseのuniform誤差、basis変換、全pathの変形回数とreferenceの囲い込みである。

Actual inverseは既存の`QuantumCircuit.inverse`を使い、ControlledGate.inverseがbase inverseを再controlする経路もある。Qiskit共有equivalence状態へアクセスする人工testでは既存B2 mutexを再利用した。Phaseをfitして誤差を隠していない。

**到達分類：有限のraw入力集合と数学的伝播経路は明確だが、現whole-P0に有用な一様上界を得られるかは未解決。** Whole-word列挙を既定にしない。Primitiveの観測残差表だけを追加すれば完成するとも断定できない。最終roundのn倍伝播に耐えるuniform compiler/ref誤差が必要で、緩いunitary上界2だけでは有用な精度資格にならない。Compilerを変更する、barrier/event別compileへ変える、未実装のfusionを採用する作業は行っていない。

## 6. 最小実分子実験の比較と推奨

| 案 | 得られる情報／残る不足 | 規模・準備状態 |
|---|---|---|
| **C2-Bridge（推奨）** | C2-Bが前提にするevent mapとB1 referenceの数値対応、角度／符号／normalization／phase差の所在。厳密enclosureとP0 synthesisは残る | 2arm×216 atom/event、36×36平均・kernel、2arm×13round。保存G/C/ψ/PF値再利用、実回路0。47-call runner／人工E2E完成 |
| C2-A all-law native synthesis | Uniform raw primitiveとwhole-P0・reference誤差を囲えればlawと統合可能。しかし現在はcompiler proof/coverageが不足 | 科学対象／call上限未固定。実行可能契約にせず、追加compileやword列挙を自動開始しない |

先行B1/B2の表現・tail・sample検証で数学的定義と基本実装の対応は確認されている。一方、全finite平均eventと保存raw corrected-RTEを26roundで同じ対象へ接続する距離は未保存であり、strict boundはない。したがって**SOURCEと人工検証だけで実H4の残差量まで確認済みとは言えず**、Bridgeの小さな数値pilotを先に提案する。この結果が小さくても、自動的なゼロ認定や厳密proofとはしない。

## 7. 未承認47-call契約とSTOP

[固定契約案](research_c_pr_aware_df_stage_c2_bridge_contract_draft_20261027.json)。Scope `C_PR_AWARE_DF_C2_REFERENCE_BRIDGE_V1`、fresh `artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/Bridge-run01`、retry/resume0、mandatory STOP。

| 計上単位 | 最大数 |
|---|---:|
| Saved JSON／provenance／元snapshot load／SectorCAR context | 各1、計4 |
| 2armのarrays load／model-table replay／atom context／atom audit／K0 event mean／head reference／kernel bridge | 各2、計14 |
| Round bridge：2arm×13 | 26 |
| Arm summary | 2 |
| Overall summary | 1 |
| **合計** | **47** |

Atom audit/event meanは各armの有限216件をまとめた単位で、432件を未記載の裏callとして実行する意味ではない。各関数の内部評価を契約の定義に明記した。全16-event wordやfull512回路の生成はない。

SOURCE／input／frame／sign／spin-bit mapping／normalization／phase／保存reference・raw signal replayのcorrectness FAILで依存STOPしpartial evidenceを保全する。**両armの局所kernel gate完了後だけ**26roundへ進む。保存component基準1e-10／complex-signal基準2e-10／rtol0・phase-fitなし、normalization1e-12、mass rel1e-12/abs1e-14を再利用。これらは技術的なbinary64 pilot gateで、厳密誤差enclosureではない。

保存PF値は再生成しない。元h/gは既存corrected-κ model再生のためだけに使い、physical HamiltonianやDF誤差は作らない。新ground/alias計算、新fit、sampling、native build/compile、shots、総費用、C1再解析は0。Roundのobserved Bridge距離が得られても、unknown reference/synthesisと保証欄はnullを維持する。条件・閾値の事後変更、fallback、追加rank/fit/retryはしない。

科学実行には**この新SOURCE・契約への別途明示的なチャット許可**が必要。承認証明書・承認原文保存・新grantは不要。今回の作業は準備だけで、Bridge-run01を作成していない。

## 8. 人工・回帰結果、保全・環境

| 検証 | 結果 |
|---|---:|
| 新Bridge：人工CDF/CAR/sign/frame/bit/identity/K0 mean、raw/corrected／round信号、referenceと合成unknown、TV接続、二重計上拒否、P0 coverage、phase欠落／actual inverse、47-call E2E、SOURCE/input/fresh/STOP | **17 PASS** |
| 既存B1（微小nonzero・spin・full controlled等）、B2 mutex、C1 scenario、C2-B確率・coverage回帰 | **68 PASS** |
| 合計 | **85 PASS** |

[構造化test結果](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/tests.json)／[Bridge log](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/bridge_tests.log)／[回帰log](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/regression_tests.log)。人工array／probability／小回路・mock SOURCEだけで試験し、実H4数値や旧結果をfixtureへ転用していない。人工残差を実H4の科学上界にしない。

[Passive preflight](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/passive_preflight.json)はSOURCE82・保存入力20、B1/C2-Bのstatus／STOP／来歴とhashだけを確認した。実NPZ load／model replay／atom／event／kernel／round／SectorCAR構築を禁止するmonkeypatch下でもPASS。[保全確認](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/preservation.json)では元1787件のうち1785件がbyte不変、残る2件はscript/library READMEへの入口追加だけ。旧科学SOURCE/raw/STOP・C1 evidence／C2-Bの条件付き結果は変更していない。

既存read-only evaluation venv（Python3.11.1、NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0等、[lock](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/evaluation_requirements.lock)）を使用。Task-local cache、sys.dont_write_bytecode、pytest cacheprovider無効。CPU affinity32、空きRAM約52 GiB、memory pressure avg0、CPU pressureは記録時avg10=.45。科学実験時も実資源を再確認する。準備済み2 arm chainをresource_planで並列化し、BLASは自job内だけ調整する。Sector36の小行列を使い、shared immutable CAR/atoms／head exponential cacheを再利用する。CPU/RAM/worker/wallの人工上限や共有環境設定変更、他job操作はない。

## 9. SOURCE・Q1〜Q5・次の担当

- Branch：`codex/research-c-pr-aware-df-c2-bridge-preparation-20261027`
- [SOURCE82](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/SOURCE.json) SHA256：`9dbfe8a3e7886a73fcac79238e4f432f308ae4c7456080e2ad5734fa93c97408`
- 契約SHA256：`e4f0326c1769f27b615da558af8f48370c6b02fb3192a410d502e8c3ce41ed8f`
- [技術STOP](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/STOP.json)：実H4科学call0、Bridge/C2-A/C1再解析なし。

**Q1：残る差は、atom/CARの正規化対応、λ_exact/λ_float、φ/cos/sin/b、identity/scalar、head、normalization power、数値reference誤差である。** Categorical TVへこれらを含めたと扱わない。

**Q2：Exact arithmeticの定義とSOURCE経路は追跡できたが、実H4での距離／丸めenclosureは未取得。** 保存referenceを使う最小Bridge数値評価が必要で、今回の47-call案はその数値対応までを対象とする。

**Q3：有限primitiveのraw telescoping経路はあるが、whole-P0の有用なuniform上界は未解決。** 有限入力集合が分かることと、全compiler run・referenceの誤差を囲えることは別である。

**Q4：C2-Bridgeを先に推奨する。** 47-call・二arm／全216atom平均／26round、保存G/C/ψを再利用し、最初のcorrectness FAILで依存STOP。C2-Aは未固定・未承認のまま。

**Q5：実行で対応が確認されれば、C2-B lawをどのreferenceへ接続できるかという不足を数値pilotとして縮小できる。** その結果だけで83%削減が実装CA／保証資源へ昇格するわけではない。Synthesis、reference enclosure、IIDモデル、cost support等を揃えた新しいC1相当評価は別scope・別許可が必要である。

次は**研究C／ユーザー**が47-call Bridge案の実行可否を判断する。**Codex**は明示許可後だけ固定SOURCE/input／fresh outputを再照合し一度実行する。今回commit/push・remote SHA／SOURCE／成果物照合後は**mandatory STOP**。Bridge科学実行、C2-A、C1再解析、追加fit/sampling、別分子、重要なGPT詳細科学レビューへ自動進行しない。
