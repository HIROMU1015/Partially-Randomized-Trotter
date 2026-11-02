# 研究C：固定回路validated保証のCAR接続・両arm展開の技術準備

2026-10-27 JST。起点 `da66a2d322bab40a4a0130628fe4497aea79a021`、独立branch `codex/research-c-fixed-car-reference-preparation-20261027`。新しい実H4科学callは **0**。技術実装と人工検証までで、F1/F2/F3の実科学計算・重要科学レビューは実施していない。

**結論：Proposed固定回路のF1は実行準備済み・未承認、最大7 calls。独立CARをArbで評価する接続と人工検証も完成した。ただし、保存dyadic Uが厳密unitaryでない場合の「理想CAR unitary」のtarget選択は未採択であり、F2はexecutable=falseを維持する。** Literal dyadic CAR mapと、polar基底を使う理想unitaryを同一視しない。

## 固定B1の成果と今回の対象

[前回の固定B1結果](research_c_pr_aware_df_c2a_fixed_b1_execution_20261027.md)は19/19 calls・9/9判定PASS・retry/resume0。科学SOURCE `518a730510a57d0cafa1575a91f5f1979c203be3`、manifest `e8180d8e0a5479df4bfda527fbe7610a7a3d812a6b7ff7c3fc077427d0d68e80`、baseline_00 raw/compiled operator2上界 `3.4694675824e-12` は既存結果の引用であり、今回は再計算していない。

保存直鎖H4・1.00 Å・STO-3G、通常number CDF8、Nα=Nβ=2/sector36、8 system qubits/full256＋last ancilla/full512。元frame順LD2、partial symmetric S2、δ=0.1/s4/h=1/40/r16/K0/m0..12、元g補正κ、scalar一度、own-DF ground oracle、状態準備除外、CA=0.00159 Haを保持する。B2共通Gaussian/relative-head融合、P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false/MPFR160 declared phaseも変更しない。新しいnative生成・compileはない。

今回の数学的比較は **保存された一つのh-step trajectory**。平均finite-RTE operator、B1のattenuation、C2-B categorical TV、13roundの平均信号と混同せず、単一trajectoryへattenuationやLCU normalizationを掛けない。

## 再利用と差分

| 既存実装 | 再利用範囲／新しい差分 |
|---|---|
| `c_pr_aware_df_fixed_certificate.record/compare/_column/_local/adjoint/resource_plan` | 保存QPYの独立gate意味論、Arb192、全基底column、recentered vector2 radius、outward Frobenius、process-local cacheを無変更再利用 |
| `c1_stage1.car_bilinears` → 既存 `car_annihilator` | 全Fock bit ordering・CAR符号をそのまま使用。新CAR algebraは実装しない |
| `SectorCAR.one/raw_frame` | 既存one-body／density contractionを再利用。opt-in `arithmetic` 引数だけ追加し、既定NumPy/SciPy経路は維持 |
| `stage_b.prepare_arm/atom_operators` | 保存J/U・元g補正κ・元frame順・signed Z/ZZ planを再利用。atom_operatorsのarithmetic引数だけ追加 |
| `c1_stage3_reference._ordered_exponentials` | forward/reverse half-head順、cache、時間係数を再利用。matrix exponentialのbackendを選べる引数を追加 |
| `trotterlib.rte.event_unitary` | 既存の符号・phase・product順・rotation式を再利用。sin/cos/行列だけをArbへ切替可能にする |
| `stage_b2.reference_context/trajectory_reference/raw_gate`、C2-Bridge | 時間・scalar・順序・対象の照合に使用。既存binary64残差やBridge最大観測差を保証に転用しない |
| `c1_stage3_integrated.Calls` | 既存call journal/first failure/STOPを使用。承認証明の仕組みは追加しない |
| `SharedQiskitCalls` | 人工native/compile/adjointの共有equivalence利用を既存mutexで保護 |

新実装は `c_pr_aware_df_fixed_car.py`、`c_pr_aware_df_fixed_car_execution.py`、`run_research_c_pr_aware_df_fixed_car.py`、人工test。新しいfitter、Gaussian emitter、分解法、sampling lawはない。4既存SOURCEの変更は新commitにおけるoptional arithmetic接続だけであり、旧commit・旧実行済みB1 runner/契約・raw結果は書き換えない。

## 数学的targetと数値保証の区分

### A：保存binary64 referenceそのもの

`compare_saved_numeric` は保存complex128のbit値をexact dyadic行列として読み、全controlled基底でその固定数値への距離を囲う。保存行列は厳密unitaryとは限らないので距離を2でclipしない。結果は `VALIDATED_FIXED_SAVED_NUMERICAL_MATRIX_ONLY`、`ideal_CAR_bound=null`、`reference_uncertainty=null`。これだけでは理想CAR保証にならず、F2の7-call案にもこの追加診断は含めない。

### B0：literal dyadic CAR map

固定SOURCE・保存J/U・返されたbinary64 κ・constant・signed atoms・保存event角度を数学的係数として解釈する。κは既存 `corrected_kappa(h,g)=h−0.5 einsum(prrq,g)` の返値を用い、元g補正を維持する。κ計算自体の丸めと元積分への対応まで保証する意味ではない。

CAR密度は既存式 `D_a=Σ_pq V_pa conjugate(V_qa) a†_p a_q`。headは `H0=κ+constant` のCAR liftと最初の2 frame、各h/2、tailは既存signed atom P_Uの `cos(θ)I−i sin(θ)P_U` を保存順で適用する。tail identity phase `exp(−ih c_tail)`、nuclear scalarはH0の二つのhalfに一度だけ含む。hは登録rational 1/40、保存角度／identity coefficientはexact dyadic。binary64による時間乗算の丸めもrawとの直接比較へ含まれる。

**保存Uは近似unitaryである。exact dyadicとして読むとZ/ZZは厳密involutionとは限らず、B0のevent polynomialも厳密unitaryとは限らない。** B2のbinary64 referenceを高精度で再現したことだけで、この問題は消えない。

### B1案：polar基底で定義する理想CAR unitaryと明示的model bridge

これは **提案であって未採択**。各保存spatial Uに対して `Q=U(U†U)^(-1/2)` を唯一のright polar factorと定義する。J、κ、event/sign/angle/order、scalar、tail identityは変えず、CAR densityのUだけをQにしたunitary trajectoryをtargetとする。Qを数値的に近似生成したり、元Uを書き換えたり、raw回路から逆算したりしない。

Arbにより `η≥||U†U−I||2` をFrobeniusで上から囲い、η<1を必要とする。解析的に

- `||U−Q||2 ≤ η/(1+sqrt(1−η)) = dU`
- `dn=(sqrt(1+η)+1)dU` とすると、一つのspin densityの差はdn以下
- frame generator差は `2 Σij |Jij| dn (2+η)` 以下
- support長kのsigned Z-product差は `2dn Σ_{j=0}^{k−1}(1+2η)^j` 以下
- Hermitian headのexponential差は `|t| ||H_U−H_Q||2` 以下
- event差は `|sin θ| ||P_U−P_Q||2` 以下

Ideal因子はunitaryなので、各因子差上界εjから完全trajectoryのmodel bridgeは `Πj(1+εj)−1`。これは解析上界であり、標本最大値ではない。Literal mapのunitarity defectも `2ε+ε²` で囲う。η≥1、非finite、厳密Hermitian入力条件不成立では明示STOPする。Hermiticityを近似symmetrizationで修復しない。

このtargetは元の保存dyadic Uをliteralに使うB0とは異なる。**どちらを理想referenceとして採用するかは研究C／ユーザーの判断が必要**。勝手に「保存Uは厳密unitary」としたり、polar修正を採択済みとしたりしない。このためF2契約は実行不可である。

### Enclosureとfull512への接続

B0を既存CAR algebraからArb/acbで構築し、全spin-count invariant blockを評価する。H4では25 block、最大36次元。これはsector36だけの試験ではなく、全256 Fock基底の厳密な直和分割である。Gaussian gate単体のsector射影は行わない。既存 `acb_mat.exp()` をheadの完全CAR generatorへ適用し、signed eventとheadの行列積でもinterval radiusを保持する。[python-flintの行列exponential](https://python-flint.readthedocs.io/en/latest/acb_mat.html#flint.acb_mat.exp) は現行公開API資料、採用runtimeは旧B1と同じ0.8.0/FLINT3.3.1を人工試験で確認した。

最後のancillaを制御とする `C(V)=diag(I256,V)` を明示的に構築する意味論で、全512 columnのraw gate actionを既存verifierで評価する。control0 identity、control1 evolution、sector外成分、full-Fock ordering、vacuum phase、相対位相を含む。Gate列からCAR targetを生成せず、global phase fitもしない。

Columnごとの `raw−B0` ballに、raw actionのrecenter半径を一度だけ加え、全column上界をFrobenius集約する。Referenceのconstruction/exp/product/serialization丸めはball内に一度だけ含まれ、さらにreference誤差を二重加算しない。B1案を採る場合のみanalytic model bridgeを最後に一度加える。これで `||Uraw−UCAR,Q||2` を囲い、同一raw identity・dimension・位相規約を確認した既存 `compiled/raw` certificateとのtriangleに接続できる。有限probeや高精度点値をenclosureへ昇格させない。

## Proposed・共通shellと保存identity

実在するB2 run02の `circuits/proposed_00_{raw,compiled,raw_adjoint,compiled_adjoint}.qpy`、`samples/proposed_00_trajectory.json`、`references/proposed_00.npz` をhashで固定した。NPZ/QPYの新規科学decodeは0。対応する元J/U export、Stage A input、B2 SOURCE/STOP/元契約も独立に固定する。

Sampling SOURCEは `cdf79ed569127f5cb6c7ee017b76bce6422166a656bb3ae866de6e86917c45bf`、B2 execution SOURCEは `a1cf0046967c206f414dbad568a75884ee076ebf`。baseline exportは `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02`、proposed exportは `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`。保存seedはそれぞれ2939838941121001441／12704117278500360798。同じindex0でも別arm・別law・別streamであり、paired trajectoryとは扱わない。

共通head/X/Y raw/compiledの6QPYと6 certificateを旧B1と同じpath/hashで固定し、同一last-ancilla-only／terminal measurement除外規則の場合だけ再利用する。新decode・再certificateは不要。両armのArb192／operator2／phase規約／allowance1e-8は共通で、index0の固定選択を維持する。誤差値を見た標本再選択はしない。

## 人工検証・回帰

[テスト記録](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/technical/tests.json)：新27＋関連既存143＋旧general-K/native40＝**210 distinct tests PASS、旧29 subtests PASS**。新試験は28.20秒。全て人工入力・mock・人工QPYで、分子値の再解析を含まない。

新試験は、独立native/CAR一致、negative signed atoms、noncommuting event順序、一般Gaussian basis、scalar/vacuum/control相対位相、全Fock mapping、full512全512基底、interval幅、A/Bの区別、triangle identity/null、不正入力、arm/source/law混同拒否、旧certificate再利用、失敗時partial STOP、F1/F2 E2E、passive no-decode、F2非実行条件を確認した。既存B1のadjoint・X/Y・人工P0 compile、長非可換word・process-column試験もPASS。

旧general-K testには「qiskit等がsys.modulesに存在しない」という隔離前提がある。実Qiskitを使う新testとの同一process実行は14 failure/1 teardown errorとなったが、旧テストを個別のfresh processで実行し13/13＋29subtests、旧nativeは27/27 PASSとなった。productionコードを変えてテストを回避したものではない。旧C2の別の並列STOP非決定性は未修正で、その不安定試験は今回実施していない。

## 次の最小科学実験とcall

| Scope | 新規logical units | 状態／対象 |
|---|---|---|
| **F1 推奨** | QPY decode4＋全基底certificate3＝**7** | Proposed_00 raw/compiled、raw actual adjoint、compiled actual adjoint。共通shell6保証は再利用。`UNAPPROVED`、`executable=true` |
| F2 | input/JU load1＋arm replay1＋saved events1＋validated CAR reference1＋raw QPY1＋raw/CAR certificate1＋triangle1＝**7** | Baseline_00、target方針未採択のため `UNAPPROVED`、`executable=false` |
| F3 | F2と同じ構造で7を想定 | Proposed_00。F2 targetの判断と、F1で取得するcompiled/raw certificateの固定hashが必要。まだ実行契約を発行しない |

F1が現時点で最小の実行可能scope。両armのcompiler/actual inverseの有限証拠を揃える。一方、CARへの独立接続の情報はF2で初めて得られる。F2の数学的targetが採択されたら、その方針を新しい固定契約へ反映してから別途科学実行を許可する必要がある。F1～F3を同時実行しない。

[未承認F1契約](research_c_pr_aware_df_fixed_proposed_contract_draft_20261027.json)／[実行不可のF2 draft](research_c_pr_aware_df_fixed_car_contract_draft_20261027.json)。各referenceの25 blockや512 basisは該当logical unit内で評価し、別trajectoryを暗黙追加しない。旧B1の19-call枠は再利用しない。Fresh `F1-run01`／`F2-run01`、retry/resume0、first correctness FAIL・unregistered semantics・入力/SOURCE不一致・nonfinite referenceで依存後続STOP。区間がallowanceを跨げばUNRESOLVED、triangleを作らない。実行結果を見たprecision/allowance変更やfallbackはない。

F1の3件はいずれもfull512比較。前回の34分はkernel3比較＋shell等の実績であり、同程度の可能性はあるがProposed回路長によるので今回の所要時間を確約しない。F2は25 independent block（最大36）のexp/productと512 raw actionを分けて実行する。既存worker計画はaffinity/quota/実空きRAMに従い、CPU/RAM/worker/wallの人工上限は置かない。task-local cache、process-local Arb context、共有venv/cache/他jobの変更なし。

## SOURCE・環境・passive gate・保全

実装SOURCE commit：`e7d22a404442234662ef6d117c55dd9868217c9e`。新SOURCE132 filesを[manifest](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/technical/SOURCE.json)へ固定した。SHA256：`0b10324fd59835dc371a65290dbd52e8909d9d85fbb44120421907f107f77701`。Python3.11.1、python-flint0.8.0／FLINT3.3.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0等は[依存lock](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/technical/dependency_lock.json)へ記録。task-local venvから既存依存siteをread-only参照し、共有環境へインストール・変更しない。

F1 input29／F2 input30、source132、package11、trajectory/source/arm/event metadata、共通shellの既存証拠をpassive照合しPASS。NPZはhash・zip内のarray header/schemaのみ、QPYはhashのみ。実H4のarray payload decode・operator/reference/enclosure/certificate計算は0。旧B1 raw25件はbyte hash不変、旧artifacts/契約/grant/ledger/marker/STOPの変更はない。[passive preflight](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/technical/passive_preflight.json)／[保全記録](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/technical/preservation.json)。

## 得られる結論と限界・次担当

今は人工入力で接続を確認した段階で、実baseline/proposedのraw/CAR誤差はnull。F2が将来PASSしても、対象は **固定1 trajectory、指定されたdyadic/polar数学的targetへの位相込み距離** に限られる。元物理積分・DF近似・PF/RTE平均誤差、実装law/IID、未観測trajectory、all-law synthesis、保証shots、cost support、母平均費用、実装CA・総資源優位は認定しない。C1約83%の仮定付き点推定も変更しない。

**ユーザー／研究C**：最初にF1・最大7 callsの実行可否を判断する。CARを優先する場合はB0/B1のtarget定義を判断する。**Codex**：F1は別途明示許可後だけ固定SOURCE/契約で実行し、F2はtarget採択後に新契約を固定する。重要科学レビューを自動開始しない。

SOURCE・技術証拠・契約をcommit/pushし、remote SHA・Git tree・SOURCE/input/成果物の照合後に **mandatory STOP**。今回の技術準備からF1/F2/F3、sampling、C1再解析、shots/総資源、別分子へ自動進行しない。
