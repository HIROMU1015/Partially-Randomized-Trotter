# 研究C：F2-K1 exact Hermitian κ＋polar U・技術準備

2026-10-27 JST。起点 `8e53df35e4f2e78fb07cd5b8b409ff69dedb868d`、独立branch `codex/research-c-F2-K1-preparation-20261027`。

**K1方針を採用し、実装・人工検証・未承認7-call契約を完成した。新規実H4科学callは0。** 実H4のκH・κ bridge・U bridge・CAR reference・raw/CAR certificateは今回新評価していない。J/Uの新decode、旧F2再起動、F3、sampling、C1再解析、shots/総資源、詳細科学レビューも0。旧F2のSTOPと旧SOURCE/raw/契約は保全する。

## 固定対象・SOURCE・再利用

将来の対象は保存直鎖H4・隣接1.00 Å・STO-3G、Baseline通常number CDF8の `baseline_00` 一件。Nα=Nβ2/sector36を含むfull8-mode/256、last ancilla/full512。元順LD2、partial symmetric S2、δ=.1/s4/h=1/40、各half-head=1/80、保存16 events/r16/K0、元g補正κ、scalar、tail identity、signed atoms、angleを保持。Own-DF oracle／状態準備除外は文脈であり、新ground/PF/RTE/DF誤差を評価しない。

P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx・既存Gaussian/declared phase政策は保存回路の条件として保持し、今回build/compileしない。Arb192、python-flint0.8.0／FLINT3.3.1、位相込みoperator2、allowance exact `1/100000000`、global phase fitなし。

| 実装 | 再利用・差分 |
|---|---|
| `c1_stage1.corrected_kappa` | 元h/gからの返却値を変更しない |
| `c_pr_aware_df_kappa_diagnostic` | exact dyadic実虚部、加減算、record生成を再利用。保存診断値はhash/metadata確認と旧結果引用のみ |
| `fixed_car._block_task`、`SectorCAR.one/raw_frame`、`car_bilinears` | 独立B0のCAR代数・25 spin-count blocks・matrix exponentialを無変更再利用 |
| `_ordered_exponentials`、`event_unitary`、`stage_b.atom_operators` | forward/reverse、scalar、符号、event順序を無変更再利用 |
| `fixed_car.polar_bridge` | exact κHを固定したV1→V2として使用。実装・right polarの定義は不変 |
| `fixed_car._init_compare/_compare_column`、`fixed_certificate`、ZSXX enclosure | 全基底・recenter/cache・Arb・raw gate意味論を無変更再利用 |
| `fixed_car.compose`、旧F2 prior/reuse/trajectory gate、`Calls` | 同一raw/位相/full512のtriangle、科学call・partial/STOPを再利用 |

新しいのは[数学adapter](../../src/trottertracks/c_pr_aware_df_fixed_k1.py)、[専用runner接続](../../src/trottertracks/c_pr_aware_df_fixed_k1_execution.py)、[CLI](../../scripts/run_research_c_pr_aware_df_fixed_F2_K1.py)、[人工test](../../tests/management/test_research_c_pr_aware_df_fixed_k1.py)の4ファイル。既存SOURCEの変更0。旧compareはtargetとunitary norm capを結合しているため、新targetのaggregationだけをopt-in adapterで分離し、column評価は共用した。新しいCAR algebra、分解法、fitter、Gaussian emitter、数値線形代数libraryは作らない。

SOURCE142は旧診断inventory138＋新4。[manifest](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-preparation/SOURCE.json) SHA256：`de1fcf1b9bcaf227301687b921ced582c906658ad866370c5c9a362379cf6fd6`。固定起点と各file hashを保持し、新公開commitは最終チャットから参照する。

## Q1：exact κH／κAとCAR head

κは `corrected_kappa(h,g)` の返却binary64そのものをexact dyadic行列として解釈する。実虚部を有理数対にし、`κH=(κ+κ†)/2`、`κA=(κ−κ†)/2`を**有理数演算で**構成する。各成分について `κ=κH+κA`、`κH†=κH`、`κA†=−κA` を厳密照合し、異常時STOPする。

`ball_coefficients`は各有理数を既存`z._real`→acbへ接続する。Dyadic constructorは192 bitsを超える入力mantissaもexactに保持でき、その後のArb演算が必要な区間丸めを保持する。Exact κHをfloat64へ再丸めした行列を代用しない。人工の非表現可能な平均 `(2^53+1)/2`、subnormalの半分2^-1075、1075 bitsにまたがるdyadicも確認した。

`head`はこのacb係数を既存`SectorCAR.one`へ渡す。Full-Fockでのliteral/H/A liftと和、Hermitian/anti-Hermitian関係を人工確認した。元のprepare_arm／native modelには元κを渡し続け、κHを書き戻さない。

新target IDは **`CAR_exact_kappa_H_polar_U_fixed_trajectory_v1`**。Qは保存exact Uの唯一のright polar因子 `Q=U(U†U)^(-1/2)` で、旧定義から変えない。

| Map | 定義 | 数値評価／役割 |
|---|---|---|
| V0 | CAR(元κ,元U) | 独立acb B0 referenceとして全25 blocksを実評価する。非unitaryを許す |
| V1 | CAR(exact κH,元U) | 数学的中間target。κ bridgeの終点、U bridgeの始点。Tail等の非unitarityを残す |
| V2 | CAR(exact κH,polar Q) | 新しい理想unitary target |

V1/V2は係数・CAR規約・証明付きbridgeで定義し、同じtrajectoryを3回数値指数化する重複計算を避ける。κHのCAR head helperは人工検証済みで、実科学時の重いblock指数化はV0だけで十分な設計である。旧target `CAR_polar_of_exact_saved_U_exact_J_kappa_phase_v1`とは別物。

## Q2：κ bridgeの証明と非unitary伝播

Spatial κAに対し `a=sum_pq (|Re κA_pq|+|Im κA_pq|)` をexact有理数で求める。`||κA||2≤||κA||F≤a` であり、SVDの数値残差を保証へ昇格しない。

既存one-body CARは `dΓ(κ⊗I2)=sum_pq κ_pq(E_pα,qα+E_pβ,qβ)`。N粒子の作用素は各粒子に作用するone-body項の和なので `||dΓ(κA⊗I2)||2≤N||κA||2`。Full8-mode Fockの全sectorでN≤8、したがって **μ=8a** を用いる。sector36のN=4をfull256へ流用しない。人工n=2では同じ式2n=4を全Fockで確認し、n=4は25 blocks／全512入力でcoverageを確認した。

HをHermitian head generator、Aをanti-Hermitian lift、`||A||≤μ`とする。Generator `−i(H+A)` のHermitian部分は `−iA` で、logarithmic normはμ以下。任意の実tについて `||exp(−it(H+A))||≤exp(|t|μ)`。Duhamelの積分とHermitian側のnorm1から、

`||exp(−it(H+A))−exp(−itH)|| ≤ integral_0^|t| μ exp((|t|−u)μ) du = exp(|t|μ)−1`。

μ=0では差0の連続延長を使う。非可換なH/Aでも成立する。実scalarはHにのみ加え、このnorm増大を変えない。

既存scheduleを展開すると、scalar phaseを除き **V0=A0 M A0、V1=B0 M B0**。A0/B0はそれぞれκ／κHを用いた時間h/2のbase factor、MはJ frameとtail eventsの変更しない積である。Jが実対称なら、非直交UのdensityもHermitianで、`(1/2)sum_ij J_ij ρ_iρ_j` は対称なi/j対の和としてHermitian。したがってliteral J headの指数因子はunitary。ただしliteral tail atomはunitaryと仮定しない。

Old U bridgeが局所誤差e_jから構成する `εU=prod_j(1+e_j)−1` は、全tail-local norm増分を含む。各ideal Q eventはunitaryで、literal eventのnormは1+e_j以下。正のhead-error因子も含む積なので、**L=1+εUはMのnormの有効な上界**となる。これは完成したV1のglobal normから部分積Mのnormを推測したものではない。

b=exp(|h|μ/2)、e=b−1として、application order `[A0,M,A0]` 対 `[B0,M,B0]` に一般telescopingを適用すると

`εκ ≤ e L b + e L = L (b²−1) = L[exp(|h|μ)−1]`。

両half-headを一度ずつ、各時間h/2で計上する。コードは一般telescoping式をArb192で評価し、上向きendpointを保存する。V0のnormにもb²Lを保持し、norm=1へ置換しない。有効なU-local積上界がない場合はκ bridgeをnull／UNRESOLVEDとする。実科学でeta条件不成立等が起きた場合は依存処理をSTOPし、未測定のboundをnullのまま保全する。

## Q3：U bridgeと誤差計上

旧`polar_bridge`をそのままV1→V2へ再利用する。前提は今回明示して検査する：exact κH、実対称J、実constantとidentity phase、eta≥||U†U−I||2かつeta<1、K0の実angleとsigned Z/ZZ support、固定frame対応。

EtaはArbで囲ったGram欠損のFrobenius上界。`||U−Q||≤eta/[1+sqrt(1−eta)]`、`dn≤[sqrt(1+eta)+1]||U−Q||`。CAR annihilatorのnormが列normに等しいことからdensity差を囲う。Spin-summed densityは `||ρU||≤2(1+eta)`、`||ρQ||≤2`、`||ρU−ρQ||≤2dn` なので、既存のJ frame差 `2 sum|J| dn(2+eta)` が成立する。両J generatorはHermitianであり、exponential差には|t|を一度だけ掛ける。

Literal Zのnorm≤1+2eta、Z差≤2dnからsupport長kのsigned atom差を `2dn sum_{l=0}^{k−1}(1+2eta)^l` と囲う。Q側の異なるdensity projectorは可換で、signed Z/ZZはHermitian involution。K0 event `cosθ I−i sinθ P` はQ側でunitary、差は|sinθ|×atom差。これが既存product boundの適用条件である。κHを固定しているためκの変更分はこのU bridgeに入らない。

Raw→V2は **εraw/B0＋εκ＋εU**。Compiled→V2は **εcompiled/raw＋εraw/B0＋εκ＋εU**。比較結果へ3項／4項を分けて保存し、それぞれ一度だけ足す。

- B0のreference区間半径はacb column差に含める。
- Raw gate評価のrecentered radiusはcolumnごとに一度加える。
- それらのcolumn上界をoutward Frobeniusでoperator2上界へ集約する。
- κ/U bridgeはその後それぞれ一度加える。
- 同一rawの旧compiled/raw certificateを最後に一度加える。

**Raw/B0には距離2のcapを適用しない。** 最終raw/V2およびcompiled/V2は両endpointが数学的unitaryと確認された比較なので、その終点距離に限りmin(2,上界)が有効。大きいκAの人工例ではraw/B0>2をそのまま保存することも確認した。

## Q4：全基底reference・certificateへの接続

独立referenceはraw QPYから逆算せず、元h/g→κ、保存J/U、保存signed eventからCAR代数で構成する。Full256は25のspin-count invariant blocksに完全分割され、最大36次元。Non-Hermitian one-bodyも粒子数／spin数を保存するので分割は有効である。

最後のancillaに `diag(I256,V0)` をembedし、rawの全512入力列・全512出力に対する位相込み差を既存Arb backendで囲う。Sector36だけの比較や有限probe外挿ではない。Vacuum/scalar/control0/control1の位相を保持する。Rawとcompiledのtriangleは同一raw QPY hash、dimension512、全512 columns、norm、phase_fit=false、precision192、新TARGETを照合してから組む。

旧`build_reference`入口の厳密κ gateは削除していない。新adapterで元κのfinite/shape/provenance→exact H/A→J/frames/events→polar条件と両bridge→literal B0 blocksという別経路を定義した。重い指数化より先にbridge資格を確認し、Partial evidenceとしてκ input proof、両bridge、reference、decoded raw、certificate、failure、Calls、STOPを段階保存する。

## Q5：残る実入力条件

今回J/Uは新decodeしていない。次の科学実行で必要な検査は契約へ固定した。

| 条件 | SOURCEで分かること／将来の実値gate |
|---|---|
| κH | Exact有理数構成でHermitian。返却κと保存診断literal/H/A entriesの同一性もCAR_reference内で確認 |
| J | 固定ffsim0.0.83の`df_tensors_from_params`は`real_symmetrics_from_parameters`を使用し、修正codecもexact symmetric Jを要求。保存exportの実対称・finite・float64/8×4×4をinput_arrays内で再確認 |
| U | Float64/8×4×4・finiteをload時に確認。eta<1はCAR_reference内のvalidated値で判定。eta≥1は未資格であり、Q不存在の証明ではない |
| CAR head | κHと実scalar、実対称Jの証明からHermitian。元κ側のanti-Hermitian liftを別に計上 |
| Signed events | 保存source/arm/atom/phase/orderを既存restoreで照合し、support1/2・実angle・spin-lift basisを新gateでも確認 |
| Basis/phase | 元順、interleaved spin、全Fock、last ancilla、scalarとtail identityを維持 |

Jの対称化やUの再生成、追加target変更が必要なら無断で行わずSTOPする。現時点で人工実装上のblockerはないが、実入力条件のPASSと最終1e-8以内のcertificateは**未取得**である。

## Q6：未承認科学契約・call範囲・STOP

[契約](research_c_pr_aware_df_F2_K1_contract_draft_20261027.json) SHA256：`bc7a63d6ba0dd3355c9ea8b4aa3231d6187a1cd5ee97f1c8e76d44d3185c4b63`。Scope `C_PR_AWARE_DF_FIXED_F2_K1_V1`、`executable=true`／`status=UNAPPROVED`。新しい科学実行許可は未取得である。

| Logical call | 上限／処理 |
|---|---|
| input_arrays | 1：h/g/cとBaseline J/U、schema/finite/real exact-J gate |
| arm_replay | 1：元κの既存prepare_arm。native metadataだけで回路は作らない |
| trajectory_load | 1：保存16 signed K0 events復元、samplingなし |
| CAR_reference | 1：exact H/A・診断identity・acb係数・κ/U bridge・全25 B0 blocks・exponentialを含む |
| qpy_decode | 1：保存baseline_00 raw/9 qubitsのみ |
| raw_CAR_certificate | 1：full512全列、reference/raw半径、両bridge、登録allowance |
| triangle_composition | 1：同一rawの旧compiled/raw certificateを再利用 |
| **合計** | **新規独立枠の最大7 units** |

Fresh output：`artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-run01`（現在未作成）。Retry/resume0。旧F2残3 callsは流用しない。新fit/optimizer/build/compile/sampling/DF-PF-RTE/shots/総資源・C1再解析なし。

最初のcorrectness FAILで依存callを停止する。Enclosureがallowanceをまたぐ場合はUNRESOLVEDとしtriangleへ進まない。資格条件不成立では未測定boundをnullにしてSTOPする。数値閾値の変更、fallback、別target再実行は行わない。許可はチャットで足り、証明書・承認原文保存・新grantは要求しない。

## 人工・回帰・passive確認

**新28＋旧125＝153 distinct tests PASS、29 subtests PASS、skip0。** 新経路は有理数H/A、微小／大きいκA、複素H、subnormal/Arb半径、full-Fock lift、非可換factor、両half-head、exact/near-unitary U、既知のexact polar Q=Iを持つ人工U、両bridge和、非unitary raw/B0>2、相対位相、全512基底、raw/compiled triangle、null、source/input不一致、partial/STOP/no-resumeを検証した。

旧一般Kは既存import隔離条件に従い別fresh processで実行。旧C2の並列STOP非決定性は今回未修正・未採用で、修正済みとは扱わない。初期人工fixtureのPython複素冪の表記と、Arb dyadic constructorが入力bitをexact保持する点に関するテストの誤った仮定を修正し、最終suiteはPASSした。分子データの試行・retryではない。

[技術証拠](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-preparation/)にtest logs、SOURCE、passive、preservation、resource、technical STOPを保存。SOURCE142・旧入力・κ診断raw9・旧F2 raw6をbyte照合し、旧結果不変。Actual-resource planningは既存block/process-column・per-process Arb/cache/recenterを使用し、CPU/RAM/worker/wallの人工capを加えない。共有環境／共有cache／他jobは変更しない。

## 主張の限界・次担当

今回確認したのは技術経路と人工条件下の数学的実装である。実H4の新εraw/B0／εκ／εU／compiled/V2上界は**すべて未取得/null**。将来の固定一trajectoryがPASSしても、元物理積分の厳密性、DF新保証、全trajectory/全law synthesis、PCG64 IID、CA、保証shots、cost support、母平均／総資源優位は未資格。Stage C1の約83%は元の明示仮定付き点推定のままである。

次は**ユーザー／研究C**が固定SOURCEと新7-call契約の実行可否を判断する。**Codex**は別途明示許可後だけfresh F2-K1を一度実行し、結果公開後にSTOPする。今回はcommit/push・remote SHA/SOURCE/input/成果物照合後に**mandatory STOP**。F2-K1/F3、sampling、C1再解析、shots/総資源、別分子、詳細科学レビューへ進まない。
