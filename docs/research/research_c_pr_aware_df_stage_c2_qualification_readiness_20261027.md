# 研究C：PR-aware DF Stage C2・全law誤差資格付けの技術準備

2026-10-27 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。

**技術調査・opt-in接続・人工26＋回帰73＝99 tests PASS。新規実分子科学callは0。** 最小の次実験として、保存2 armの有限確率表だけを調べる**C2-B・最大35 logical units**を準備した。全law native合成誤差、実PCG64のIID性、CAR/atom/角度・normalizationのreference bridge、費用supportは依然未資格。C2-Bの表レベルの成功だけではC1 evidenceをGOにしない。

## 1. C1の固定結果と保全対象

起点は`99fcaca44f6b015ec64714a8552641dd044b246a`。[C1実行報告](research_c_pr_aware_df_stage_c1_execution_20261027.md)／[原本96件](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/)。164/164 COMPLETE、correctness FAIL/retry/resume0、mandatory STOP。C1契約と`CONDITIONAL_DESIGN_NOT_EXECUTABLE`の旧C2設計草案は変更していない。

保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8のBaseline/Proposed、Nα=Nβ2／sector36、元frame順LD2、元g補正κ・核scalar一回、partial symmetric S2、δ=.1／s4／h1/40／r16／K0、m0..12。共通Gaussian/P0、own-DF ground oracle、状態準備費用除外、CA=.00159 Ha／DF reserve CA/8、既存phase/failure条件を維持する。

| 保存シナリオ | Baseline総RZ点推定 | Proposed | 比較の資格 |
|---|---:|---:|---|
| ideal exact native／ideal IID | 3,194,333,274,946 | 540,290,160,109 | 明示仮定下、約83.086%低い |
| synthesis/law各.001Aを仮定 | 3,209,642,597,218 | 542,769,406,705 | 未検証感度仮定下、約83.089%低い |

上表は保存値の引用。新しいshots/総資源再解析は行っていない。C1 evidenceは全26roundが`UNRESOLVED_SIGNAL_ERROR`、保証shots/total/actual CAはnullのまま。

## 2. 既存コードの再利用と最小差分

| 既存処理 | C2で確認・再利用した範囲 |
|---|---|
| `c_pr_aware_df_stage_b.py`、`stage_b2.py` | 同じrank8二arm・signed atom law・native policy・full512検証と、有限標本のcoverage |
| `c1_stage3_connection.split_native_model/configure_rte` | exact Fraction化した元順のI/Z/ZZ係数、identity抽出、既存law生成の出所 |
| `pr_dictionary_validation.prototype.MixedAtom/make_tail` | 保存atom→component確率の**実際の既存経路**を将来C2-Bで再生。新law生成法なし |
| `rte.iter_sample_rte_events` | fsumによるcomponent確率和、再division、PCG64、order/atomの`choice`順をSOURCEで追跡 |
| `StrongBuilder`／`PrecisionBuilder`／native Gaussian | 決定論head内のrelative融合、tail full-frame bases、位相保持、raw primitiveの構造 |
| signed-event bridge／`build_event_sequence` | K0 signed rotation、同basis runの共有、元順序、ancilla相対phase |
| P0 gates wrapper／MPFR160 phase wrapper | definitionsをbasis+uへ展開後、whole-kernel opt1。既存SOURCEを変更せず監査 |
| B2 `SharedQiskitCalls` | 人工compile／actual-adjoint構築でも同じprocess-local mutexを使用 |
| `repeat_bound`／`round_synthesis` | 有効な一様入力boundの伝播則。入力coverageを生成する処理ではない。人工結果の照合に再利用 |
| `support_cost_envelope`／`bounded_cost_intervals` | 全support／有効rangeと標本極値を区別。人工の構造的supportだけを既存CI関数へ接続 |
| C1 adapter／`assess_round`等 | 旧SOURCEを保全。C2はnullを保持する入力previewのみを作り、shots/total関数を実データへ呼ばない |
| 既存`Calls`／`resource_plan` | C2-Bの35-call journal、依存STOP、実資源による2 arm並列を再利用 |

新規は`c_pr_aware_df_stage_c2.py`、静的`c_pr_aware_df_stage_c2_audit.py`、専用runner、人工test。既存伝播関数の値はfloating-point pilot labelで、exact discrete random-grid距離を計算する処理がないため、標準`Fraction`による整数区間計数とTVだけを追加した。新fitter、optimizer、Gaussian合成器、数値計算libraryは実装していない。Local-bound previewの有理L1伝播は既存`2*hypot(dX,dY)`を安全側に囲み、schema検査を証明そのものと扱わない。

## 3. C2-A：全law synthesisの構造と未解決点

### 有限要素とwordの構造

各armの決定論headはκとF0/F1の3block。各半sweep内の隣接Gaussianをrelative transitionで融合する。Tailは残る6frameのfull-frame basisと、8 mode上のZ/ZZの216 signed atom。K0では1eventにrotation atomを1つ選び、1kernelには16eventsがある。Taylor productはなく、K0 order phaseは1。Scalar＋tail identityはancilla Pで一度だけ入り、systemに無条件global phaseを加える操作と混同しない。

Tailの同basis runは既存方式で共有する。Head/tail間の新しいrelative融合は導入しない。有限のbasis/primitive alphabetを調べる方法はあり、216^16 wordの列挙を前提にする必要はない。一方、Gaussianの行列→角度、境界のvacuum phase、signed rotation、controlled relative phase、actual inverse、X/Y shellをすべて同じノルム・精度・referenceへ対応付ける必要がある。

Full512の64標本PASSは各word内の全operator検証であり、全wordの検証ではない。人工test PASSやtransfer残差、per-gate受理閾値も、一様boundにはならない。

### 固定P0の静的監査

[Pass構成・local Python SOURCE hash](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/technical/compiler_static_audit.json)。回路を渡さず、固定preset pass managerを構成して階層を読んだ。`PassManager.run`／実H4 build/compileは行っていない。

| 段階 | このprofileでの主処理 |
|---|---|
| wrapper前処理 | empty EquivalenceLibraryの`UnrollCustomDefinitions`、basis `rz,sx,x,cx,u` |
| init | InverseCancellation |
| layout／routing | backend/target/couplingなし、この構成で空 |
| translation | UnitarySynthesis、HighLevelSynthesis、BasisTranslator |
| optimization | Depth/Size FixedPoint、DoWhile内のOptimize1qGatesDecomposition・InverseCancellation・必要時translation等 |
| phase | 既存task-local MPFR160 symbolic offsetでdeclared global phaseを追跡し、最後にbinary64へ戻す |

固定Qiskit1.3.0の[Rust Euler SOURCE](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/euler_one_qubit_decomposer.rs#L45)を取得した。`ANGLE_ZERO_EPSILON=1e-12`、1q runのf64行列積、Euler角への再合成、微小角・特殊角の簡略化を確認した。`optimize_1q_gates_decomposition`はsimplify=true/default toleranceで再合成し、targetなしでは回路長等を基に置換する。ここでいうerror評価はoperator residual certificateではない。**approximation_degree=1.0はこの経路の丸め・簡略化誤差ゼロの証明ではない。**

[InverseCancellation Rust](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/inverse_cancellation.rs)も取得した。Basis変換やキャンセルの理想的な恒等式と、実装された行列計算・角度変換の丸め、controlled化時のphase保持は別に評価する。既存MPFR phase対策も全operatorの囲い込みではない。

**判定：必要な一様丸め・reference・全pass経路の評価条件が不足し、実用的な一様P0上界は未解決。** Telescopingの数学的な構成経路はあるが、有限primitiveの追加残差評価だけで完成するとは断定できない。Whole-kernel 1q runはevent境界を越え得るため、event別compile residualの和を保証へ流用しない。閾値1e-12にgate数を機械的に掛けても、Euler角復元・mod2π・phase・run再置換等を含む上界にはならない。有用性を失うほど緩い上界しか得られない可能性も残す。

局所proof入力adapterは、full controlled operator 2-norm、位相感度、reference誤差、compiler coverageを必須にし、sample maximum／phase-fit／event-P0だけの記録を拒否する。現在のH4へ与えられる入力proofはない。正しい伝播関数を作れたことを、科学的なbound取得と扱わない。

## 4. C2-B：確率表／finite-grid／PRNGの分離

### SOURCEで確認できた処理

[NumPy1.26.4 Generator.choice](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx#L806)のSOURCEを取得した。入力pはdouble連続配列へ変換し、Kahan和・非負性・形状・和の許容範囲等を確認する。`replace=True,p=...`では累積和を最終値で再正規化し、random値を`searchsorted(side='right')`へ渡す。和の受理許容差と実際の選択確率の誤差は別物である。

既存`make_tail`は保存されたexact Fraction atom係数の絶対和λを作り、p_i=float(|c_i|/λ)へ丸める。Positive係数／確率が0へ落ちれば明示的に拒否する。Samplerはcomponent確率をmath.fsumで合計し再divisionした後、NumPyへ渡す。ゼロ確率／cumsumの平坦区間／正の微小確率が格子点を持たない場合を区別する。

[PCG64のdouble変換](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_pcg64.pyx)と[`uint64_to_double`](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd)、[uniform fill](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/src/distributions/distributions.c)も取得した。Uint64から下位11bitを捨て、2^-53を掛ける。従って**入力wordがIID一様という明示モデルの下では**、doubleはk/2^53（k=0..2^53−1）の一様格子となる。固定seedのPCG64がそのモデルを数学的に実現するという証明ではない。

### 追加samplingなしのexact table評価

Binary64で実際に計算したCDF境界をexact Fractionとして保持する。D=2^53、右側searchのbin iに入る格子点数は

`n_i = ceil(D*C_i) − ceil(D*C_(i−1))`、`C_(-1)=0`。

よってq_i=n_i/Dはexactで、全D点を列挙せずO(atom数)で計算できる。P_i=|c_i|/Σ|c_j|を保存atom係数に対してexactに定義し、`η=Σ|P_i−q_i|/2`を有理数で直接計算する。最終binary64表示だけ上向きに丸め、巨大整数・微小値をclipしない。最小positive係数が0へ丸められた場合も、元Pに対する質量を黙って削除しない。

第一確率丸め、samplerの再division、連続CDF、finite gridの各TVも別に記録するが、直接のP対q距離へ再加算しない。NumPy自身の入力検査は`choice(size=0)`で再利用し、乱数wordは一つも消費しない。K0のorder distribution=[1]は必ずorder0になるが、実samplerはorder用乱数も消費する。この捨てられるdrawを含むstream全体がIIDであるというモデルを明記する。

### Roundへの条件付き伝播と適用限界

同じexact unitary trajectory mapを両lawへ用い、1trajectoryのcomplex matrix elementの絶対値≤1、各eventのIID性を仮定する。n=rs2^m=16×4×2^m eventのproduct lawのTVはmin(1,nη)以下なので、raw complex mean差は`min(2,2nη)`以下となる。**Aを掛けたり、Aで割ってnormalized errorへ置き換えたりしない。** 合成誤差はideal trajectoryと実回路の差として別に足す。

これは保存atom係数で定義したlawと、固定CDF写像＋IID一様格子モデルとの距離である。次の未資格項目を消さない。

- 原g/J/U→atom係数の丸め・省略と、B1 raw CAR tailとのoperator差。
- λのbinary64化、tau/atan/normalization/phaseの計算とB1参照との対応。
- PCG64決定論streamのjoint lawとIIDモデル、shot／occurrence間の独立性。
- Native emitter／whole-kernel P0／X/Y shellの合成誤差。

特にB1は`raw CAR tail / config.lambda_r`からfinite Taylor referenceを構築する。Exactに正規化したatom表の平均と同じ演算子だという丸め込みの証明ではない。表のTVだけでB1とのraw signal差を全て囲ったと表示しない。`c1_preview`はconditional table boundを別欄へ保持し、actual implemented-law／synthesisのround入力をnullのまま返す。旧C1を再起動しない。

## 5. C2-C：費用supportと母平均

既存32標本/armからは固定cohortの平均、分散、記述的SE、観測範囲が得られる。母平均の保証CI、全law極値、ideal-law平均への転送は得られていない。追加samplingだけでは全law synthesis／law／supportの不足を解消しない。

Finite native alphabet、有限Gaussian emitと定義展開の最大サイズ、basis translation、1q再合成、loopの置換条件まで追跡して、**最終whole-kernel**のgate数rangeを構成する可能性はある。今回そのrange proofは完成していない。Raw countsやevent個別compiled価格の和、64sampleの最大値をwhole-P0 upperへ転用しない。Depth/count/sizeはそれぞれのscopeを保ち、未知upperを0にしない。

Opt-in support入力はwhole-kernel/all-word coverage、pass iteration proof、provenanceを要求し、観測extremaやevent価格和を拒否する。人工のexact構造的rangeを既存`bounded_cost_intervals`へ渡す接続だけを試験した。現実のH4 supportはnull。

C2-Cで有効なrangeが得られた場合の案はcost alpha total .01を2arm×6metricsへ各.01/12。これは未採択の提案であり、C1の候補別RPE failure .01を変更しない。同じ32標本を13roundで共有しても独立cost confidence eventは12個で、round数で統計幅を縮めない。2armのenergy事象を同時に扱うunion upperは.02、別cost failureを組み合わせる必要があり、「同時失敗率.01の総優位」と呼ばない。C2-CはC2-Bの実行前提にしない。

## 6. 人工検証・静的SOURCE確認

| 検証 | 結果 |
|---|---:|
| 新C2：exact TV、small-grid全点照合、実Generator.choiceへの境界uniform注入、zero/tiny/underflow、raw norm／event数、PCGモデル拒否、relative phase/reference/compiler範囲、support／既存CI、null preview、人工35-call E2E・失敗STOP・fresh/no-retry、人工P0／actual inverse | **26 PASS** |
| 旧C1 scenario／accuracy／connection／H4 CA／B2 mutex | **73 PASS** |
| 合計 | **99 PASS** |

[Tests機械記録](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/technical/tests.json)／[C2 log](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/technical/c2_tests.log)／[回帰log](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/technical/regression_tests.log)。最初の人工test harnessで既存saveの上書き拒否とCompilerProfile名の誤りがあり、人工fixtureの別path／既存classへ修正した。最終SOURCEに対する全試験は上記の通りPASS。科学SOURCEや旧rawを直す作業ではない。

上界・確率解析を一切禁止するmonkeypatch下でも、実固定bytesに対する[passive preflight](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/technical/passive_preflight.json)がPASS。SOURCE76／保存入力12、C1 STOPとschemaだけを確認し、実H4の係数・確率表の値は解析していない。Fresh C2B-run01は未作成。元入力・J/Uはhashだけを照合し、NPZ/QPY decodeは0。

NumPyとQiskitのupstream9SOURCE/licenseは[provenance](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/technical/upstream/provenance.json)と共に取得し、GitHub blob SHAと保存bytesを照合した。固定tagのSOURCEと、local wheelのversion・人工動作を対応付けた技術検証であり、wheel binaryをupstreamから再buildして一致証明したものではない。共有packageを書き換えていない。

Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、threadpoolctl3.6.0、symengine0.13.0。[Read-only環境lock](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/technical/evaluation_requirements.lock)。Affinity32 CPU／空きRAM約52.7 GiB、pressure avg=0。重い行列workloadはなく、将来2小表を既存resource_planで並列処理する。恣意的なCPU/RAM/worker/wall上限を追加せず、共有venv/cacheや他jobを変更しない。Task-local MPL/cacheを使用した。

## 7. 推奨する最小の科学実験：C2-B、35-call

**C2-Bを先に提案する。** 2 arm×216 atomの保存表だけで、samplingの丸め/CDF/grid部分をexactに評価でき、追加sampling・回路・operatorなしで、この誤差源が律速となり得るかの情報を得られる。Aの主要blockerを無視する選択ではなく、現在実行範囲を確定できる最小のbranchである。Aはuniform P0/参照丸めの技術proof、Cは構造的range proofが未完で、科学callと実験範囲がまだ確定していないため実行契約に含めない。

[未承認C2-B契約](research_c_pr_aware_df_stage_c2_contract_draft_20261027.json)。Scope `C_PR_AWARE_DF_C2B_K0_TABLE_IID_GRID_V1`。保存atom表と既存component fieldをexact再生し、初回確率／sampler再正規化／CDF／53bit gridの誤差を分離し、26roundのconditional raw-law上界を保存する。Numerical allowanceはexact rational/binary64 replay一致、NumPy自身の受理条件、exact grid counting／outward表示で事前固定。CAやphase thresholdは再評価しない。

| 科学call分類 | 最大数 |
|---|---:|
| grouped saved load | 1 |
| C1 SOURCE/status identity | 1 |
| 既存make_tailによるtable replay | 2 |
| choice CDF/grid/TV解析（表全体を1unit/arm） | 2 |
| raw-law round伝播：2×13 | 26 |
| arm qualification/null C1-preview | 2 |
| output summary | 1 |
| **合計** | **35** |

成功は`EXACT_CATEGORICAL_TABLE_DISTANCE_UNDER_IID_GRID_MODEL`であり、actual全law誤差の資格ではない。Overallのactual law、synthesis、CA、保証shots/totalはnull。入力/SOURCE/再生/正規化/順序にcorrectness不一致があれば依存STOPしpartial evidenceを保存。未知のreference／PRNG資格はUNRESOLVEDとして他の登録表処理を継続する。Retry/resume0、fresh `artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/C2B-run01`、公開照合後mandatory STOP。新fit/rank探索、sampling、build/compile、PF/RTE operator、shots／総費用／C1再解析はすべて0。

**この契約の科学実行には別途チャットでの明示許可が必要。** 承認証明書、承認原文保存、新grantは要求しない。現在は科学call0。

```bash
# 現在実行可能なpassive確認のみ
"$EVAL_PY" -B scripts/run_research_c_pr_aware_df_stage_c2.py \
 --contract docs/research/research_c_pr_aware_df_stage_c2_contract_draft_20261027.json
```

## 8. SOURCE・完成範囲・Q1〜Q5

- [SOURCE76](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/technical/SOURCE.json) SHA256：`2cfb05ea3764c220fdc5ffdc3eae1f1bc54bfc107b45a6ffacac0c6a09e3b45b`。
- C2-B契約SHA256：`735ee857e549abd086d5085120cddc023708fec71329886921eb2a0dd3aa0150`。
- Branch：`codex/research-c-pr-aware-df-stage-c2-preparation-20261027`。
- [技術STOP](../../artifacts/research_c_pr_aware_df_stage_c2/2026-10-27/technical/STOP.json)：C2科学実行未開始。旧A/B1/B2/C1のSOURCE/raw/grant/ledger/marker/STOPは保全。

**Q1：数学的なuniform telescoping経路はあるが、現P0で実用的な一様誤差boundを得られるかは未解決。** 個別primitiveの残差を増やすだけで解決するとは扱わず、全pathの数値変換・reference囲いが必要である。

**Q2：固定NumPy SOURCE・実CDF・exact grid countの下でcategorical TVは評価可能。** IID一様53bit wordモデルと同じunitary mapを条件とし、実PCG64のIID性やB1とのreference gapは未資格として別管理する。

**Q3：現32標本/armは記述的な費用平均／SEまで。** 構造的whole-P0 rangeと明示した統計モデルが得られれば既存CIへ接続できるが、そのrangeはまだない。観測extremaをsupportにしない。

**Q4：固定SOURCE76／保存入力12／2表／26roundのC2-B、最大35-callを推奨。** 実行経路・人工E2E・fresh/STOPが準備済み。A/Cの未固定計算や旧C1再解析を束ねない。

**Q5：C2-Bだけでは83%を保証付き結果へ引き上げない。** まずtable/grid由来のlaw誤差を明示モデル下で定量化できる。Synthesis、CAR/atom/normalization bridge、randomness仮定を揃えた新しい限定C1相当実行を別許可後に行えば、より実装に近い条件付きnumerical pilotへ進める可能性がある。厳密CAにはDF/PF/referenceの丸めenclosure、保証付き母平均優位にはさらにcost support・同時failure等が必要である。

次は**研究C／ユーザー**がこの35-call案・IID格子モデルの意味を確認し、限定科学実行を許可するか判断する。**Codex**は許可後だけ固定SOURCE／入力／fresh outputを再照合して一度実行する。今回commit/push・remote SHA／SOURCE／成果物照合後、**mandatory STOP**。C2-A/C、C1再解析、追加fit/sampling、別分子、重要な詳細科学レビューへ自動進行しない。
