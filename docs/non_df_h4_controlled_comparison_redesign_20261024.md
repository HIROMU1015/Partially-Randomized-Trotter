# H4 D2後：Controlled演算対応とPauli/DF資源比較の再設計

2026-10-24 JST。D2は、**同じ保存Pauli bodyの理想controlled演算を保つSOURCE対応に基づく、control合成実装＋compiler応答の比較**として位置付けられる。product・符号・次数phase・constant phaseを落とす変換ではない。局所恒等式の前提と固定SOURCEへの適用を確認し、既存D2人工12件に非可換PR product等の新8件を加え、計20件PASSを得た。実H4全回路・compiled誤差の厳密保証を得たという意味ではない。

既存D2だけで実装方式の大きい影響と、観測済みPF単価のPauli/DF順位変更は確認できる。Pauli central PRの256本平均、両軸G、母平均tailは未確定のまま。**次の最小案は、保存order2を含む最初のtrajectory index1のcosine/sine各1回、計2compileの限定gate**である。正確な既存cohort平均が研究判断に必要になった場合だけ、残るcosine計測へ進む。D3は後順位のままとする。

今回の実H4回路生成・QPY読込・compile・sampling・科学データの数値再解析・C/N/G計算・科学grant発行/消費はすべて0。原本を変更せず、無効契約と技術報告を保存してmandatory STOPする。

## 1. 固定資料・検証の範囲

基点branch `codex/non-df-h4-d2-execution-20261024`、基点commit `b9183b42bf667df2853a05ba8033c92eb1c1e056`。専用branchは `codex/non-df-h4-controlled-comparison-redesign-20261024`。

| 役割 | 固定SOURCE |
|---|---|
| 元H4 | `0c329b30a81accb35c212be8defa66fee0dd0f12` |
| D1 | `9843b803878db6c3aee7ab5e7c6bb4818572c944` |
| D2 | `74515d8eb4f073c40fc783681a3775dbca89e9b1` |
| 条件付きC/N/G | `b1984dcde06a874d0fd6e9fe6395b7f3fb7504d8` |

基点を含む5commitをGitHubから実取得し、D1/D2結果およびSOURCE実装・旧統計8fileのremote/local Git blobを照合した。[取得記録](../artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24/github_verification.json)。[D1結果](non_df_h4_d1_source_execution_20261024.md)、[D2結果](non_df_h4_d2_execution_20261024.md)、[構造調査](non_df_h4_cost_reversal_structure_20261024.md)、[旧条件付き会計](non_df_h4_conditional_cng_results_20261024.md)を参照した。以下の数値は既存出力の引用で、新しい科学集計ではない。

対象はlinear H4、隣接1.00 Å、STO-3G、system8＋ancilla8、S2、T=.8、q8/r4/K4。Pauli184非identity項・PR prefix27、DF tol-only1e-8/rank10・PR prefix1＋one-body。将来の精度評価もε=.05/.01/.005/.001/.0005の5条件を維持する。状態準備を除くfull Hadamard wrapper、Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。algorithm・分割・law・状態・compilerを今回変更していない。

現行研究指示の§8.4・§10.1/10.2を適用した。参照された§10.3は読んだ版には存在しないため、その本文を仮定せず、今回の直接指示にある既存コード再利用・必要十分な証拠取得・共有環境保護を適用した。これは技術調査の妨げではない。

## 2. 数学的な変換とSOURCEへの適用

ancillaを最高位qubitとし、`Π_b=|b><b|`、`C(U)=Π_0⊗I+Π_1⊗U` とする。Bがsystemだけに作用するunitary、Dが同じsystem空間のunitaryで、control=1にD、control=0にIを適用する場合、

`C(B†DB) = (I⊗B†) C(D) (I⊗B)`。

`Π_0 Π_1=0` から両branchを展開すれば、0側はB†B=I、1側はB†DBになる。同様に `C(U_m…U_1)=C(U_m)…C(U_1)`。これらは理想matrixの恒等式であり、Bの不完全な逆、control wire/stateの変更、phaseの欠落を許す規則ではない。

### 2.1 PF：単一Pauli exponential

元[append_pauli_evolution](../src/trotterlib/pauli_signal_adapter.py:161)は `PauliEvolutionGate(cP,time=t)`、すなわち `exp(-ictP)` を置く。固定runtimeのPauliEvolution説明とSuzukiTrotter.expandはPauli rotation角度を **2ct** とする。非identity単一項では多項間Trotter近似はない。D2は既定LieTrotter・単一項・finite realのc/tだけを受け付ける。

各active wireでXにはH、YにはSdgの後Hを置く。このbasis changeをB、各active wireから最後のactive targetへのparity CNOT列をVとすると、`P=(VB)† Z_target (VB)`。したがって

`exp(-ictP) = (VB)† RZ_target(2ct) (VB)`。

[D2 _rotation](../src/trotterlib/h4_d2_pauli_control.py:56)はbasis/parityを非制御でcomputeし、中央RZだけをancillaで制御し、parity/basisを逆順にuncomputeする。0側ではcompute/uncomputeが相殺し、1側では元回転となる。Pauli label右端がsource qubit0で、実際のqargsを保存し、並び替えない。

forwardの保持項順、event位置、reverseの保持項順、outer step順は元[body builder](../src/trotterlib/h4_d1_adapter.py:92)から保持する。非可換な異なる項を交換していない。Constant Hamiltonianは各step末に元SOURCEと同じfloat更新順でglobal phaseへ入る。保存bodyの**最終集約値**を使い、`-constant*T` や`scalar_phase.angle`の別演算順で再計算した値へ置き換えない。

### 2.2 PR：rotation以外の演算とphase

元[rte._make_event](../src/trotterlib/rte.py:1599)のapplication_sequenceはproduct k個の後にrotationであり、kは0/2/4。実際の作用matrixは、時間順のproductを右側から掛けて

`E = (-1)^(k/2) exp(-i s_rot θ P_rot) (s_k P_k)…(s_1 P_1)`。

| SOURCE要素 | 元bodyでの処理 | D2転送 |
|---|---|---|
| Pauli product | labelごとのX/Y/Zを保存順で適用 | 同じ順・wireのCX/CY/CZ |
| negative product sign | global phaseにπを加える | 最終集約phaseのP(φ)に含める |
| rotation | signを係数、保存θをtimeとしてPauliEvolution | CRZ角度2 sign θ＋basis/parity |
| order phase | `angle((-1)^(k/2))` をglobal phaseへ | 同じ集約P(φ)に一度含める |
| constant phase | 元outer step末のfloat更新 | 保存bodyの集約値をそのまま使用 |
| intrinsic product phase | 非可換Pauli積自体が±i等を持ちうる | Pauli gate列の積が保持。別のscalarへ落として捨てない |

`U=e^(iφ) U_m…U_1` なら `C(U)=(P_ancilla(φ)⊗I) C(U_m)…C(U_1)`。新controlled circuit全体のglobal phaseにφを付けるだけでは0branchも変わるため、同じ操作ではない。D2は[central_control_body](../src/trotterlib/h4_d2_pauli_control.py:81)で、system bodyのglobal phaseを**ancillaの相対phaseへ一度だけ**移す。eventごとにphaseを再加算しない。productやphaseを除いた「rotationだけcontrol」の主張ではない。

identity componentと、product全体が偶然±Iになる場合は別である。Fixed Pauli SOURCEはidentity Hamiltonian項をconstantへ分離し、PR componentは非identity labelのみであることを保存modelで確認した。D2のall-I PauliEvolution拒否は維持する。仮にidentity rotationが入れば `exp(-isθ I)` のancilla scalar転送が別に必要であり、このruntimeのall-I synthesis経路を既存の変換へ無断追加しない。一方、複数の非identity productが全体で-Iや±iPとなる場合は、ゲート列と集約phaseで既に表せる。人工identity productのnegative signも確認した。

### 2.3 Hadamard wrapper

元[hadamard_wrapper](../src/trotterlib/fixed_signal_evaluation.py:164)のcosineはH–C(U)–H–measure、sineはH–C(U)–Sdg–H–measure。D2は `already_controlled=True` として同じwrapperへ接続し、controlを二重に付けない。人工検査は両branchと両wrapperをmatrixのまま比較し、system bodyのphase差を許すphase alignmentはしていない。D2の実H4費用計測自体はcosineのみである。

## 3. 証拠の層と未保証部分

| 層 | 確認できた範囲 | 確認していない範囲 |
|---|---|---|
| 数学的規則 | projectorによる共役・積・scalarのcontrol lift | floating演算・compilerを含む形式的証明 |
| 固定SOURCE | c/t/sign/wire/順序/phaseを対応付けた。D2既存stage AはPF2944/PR464命令のSOURCE token対応を保存 | 追加255bodyのreplayは未実行 |
| 保存D2 | input/compiled-input fingerprint一致、結果・native・保全記録。PR index0はorder0のみ | H4全9qubit Operatorを新たに求める検証 |
| 人工 | 既存12＋新8 PASS、system2/3qubit、phaseを含む両branch・cos/sine | 実H4 compiled全演算の誤差上界 |
| compiler/数値 | 固定profileでのRZ/CX/depth実測 | 新compiled programのρ、丸め上界、PR law-mean誤差、正式certificate |

[新人工8件](../tests/test_h4_controlled_comparison.py)は、異なる非可換productのk2/k4、imaginary product phase、product全体-I、identity productのsign、非可換PF/PR schedule＋constant、両軸wrapperを確認した。product・order/sign phaseの欠落やproduct順序変更を検出するnegative witnessもある。sampler・event生成・real body builder・compile・shot evaluatorをmockで拒否して実行した。[新8件](../artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24/artificial/additional.json)、[再利用12件](../artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24/artificial/reused_d2.json)。有限浮動小数点許容値による、主agent自身の技術検査である。

したがってD2は、**理想的な同じcontrolled bodyをSOURCE規則で比較する目的には支持される**。実測compiled回路の精度を保証するという別の主張には未被覆のρが残る。元Pauli-PR historical input hash FAILは現在QPYやSOURCE-Bの対応とは別で、旧recursive program identity・排他的原因UNKNOWNを維持する。D2監査の32件int/float JSON表現差を、この旧hashの原因として扱わない。

## 4. 比較体系と観測値の来歴

今後はアルゴリズムだけでなく、**表現×PF/PR×controlled合成規則**を実装版として識別する。

| 表現・方法 | control版 | 既存cosine RZ | 保存cohort/軸 | 出所 |
|---|---|---:|---|---|
| Pauli PF | generic | 255500 | singleton、旧両軸一致 | D1 full/旧cost_summary |
| Pauli PF | central | 15028 | singleton、sine未測定 | D2 PF metrics |
| Pauli PR | generic | mean9321.1015625、index0は9652 | 256本、旧両軸一致 | 条件付き解析cost_summary／D1 |
| Pauli PR | central | index0の1145 | **1本のみ**、256平均・sine未測定 | D2 PR metrics |
| DF PF | native | 135355 | singleton、旧両軸一致 | D1 full/旧cost_summary |
| DF PR | native | mean23678.44140625、index0は23021 | 256本、旧両軸一致 | 条件付き解析cost_summary／D1 |

旧PR標本SE/nominal t95半幅はPauli13.847523668697294/27.270074727002452、DF46.718033123081106/92.00231643184831を保存値として転記した。[機械可読引用](../artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24/comparison_quotes.json)。nominal半幅をtail上界にしない。

同じindex/IDでのPauli generic–central比較と、同じIDのcosine–sine価格はpairedにできる。Pauli cohortとDF cohortは異なるlaw・分割・event粒度を持ち、同じindexという理由でcross-representationのpaired標本にしない。DF nativeも、Pauli centralと同じgeneratorを使っているわけではない。

| 研究上の問い | 現時点で答えられること | 追加証拠を要すること |
|---|---|---|
| Q1 centralでPauli PF/PR差はどうなるか | 固定代表ではPF15028、PR1145。両方がgeneric版より小さい | central PRのcohort平均・分布。index0を256平均に代入しない |
| Q2 Pauli/DF順位 | 保存PF単価ではgeneric Pauli255500がDF135355より大きく、central Pauli15028は小さい。実装版に依存する観測順位 | PR central mean、両軸G、母平均順位 |
| Q3 分割構造と合成のどちらか | D2の同一body差はPF−240472、PR−8507 RZで、実装依存は直接大きい。Pauli保持27がZ/ZZという構造は継続 | 異なる基準の差を百分率で合算して因果寄与率にしない。必要ならcentral ablationを別契約で測る |
| Q4 全trajectoryで何が分かるか | 同じ256に対するpaired価格分布・正確な有限標本平均・PFとの経験的比較を更新可能 | law全体のtail・母平均や選択後保証は解消しない |
| Q5 N/G更新 | ideal意味が同じなら式・誤差/normalization対象の一部を再利用できる | 新compiled residualと両軸価格、証拠gateを別に確認する必要 |

D1のRZ retained/event/interactionはPauli6524/3110/+18、DF13483/5773/+3765。これは元control版のoperational ablationで、新central版の寄与率ではない。新Pauli full価格にそのまま足し直さない。保持集合を変えない今回の設計では、分割構造の一意の因果効果を同定したとはいえない。

## 5. 256 trajectoryはいつ必要か

**最初から256本をcompileする必要はない。** 実装差の存在・観測PF順位変更の確認にはD2が既に十分で、追加compile0で説明できる。既存Pauli PRのorder2 event35/order4 event0という数は旧rare_cost_sensitivityの引用である。D2 index0はorder2を含まないため、保存product経路を実H4 compilerで測る最小gateに研究価値がある。

価格を条件にせず「保存order2を一つでも含む最小trajectory_index」と固定した結果、**index1**、ID `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa` となった。[256索引と選択](../artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24/cohort_manifest.json)。これは保存cohortを見た後の構造coverage選択であり、事前random抽出や代表標本とは呼ばない。新抽選・RNG replayは不要。

次に**既存256の正確なcentral標本平均そのもの**を答える場合、未計測値を決定する証明済みcost恒等式/envelopeがないため、D2 index0を再利用し残る255 cosine compileが必要になる。少数のよい結果、source gate数の上界、旧generic平均から、未知255件のcompiled平均を確定することはできない。小標本の近似だけを目的とするなら、許容する研究判断幅と価格非依存の選択規則を別途固定すべきで、今回その標本数を恣意的に設定しない。

全256計測にも限界がある。保存cohortにorder4がない事実は変わらず、未観測law tailや母平均上界は得られない。新samplingを自動的に追加する設計でもない。

## 6. N/Gとの接続：再利用できる対象と条件

同じevent law・B尺度・状態/ray・phase・ideal controlled演算が維持される場合、各trajectoryの理想Hadamard signal、従ってそのlaw平均と理想signed biasの対象は同じになる。Pauliの旧C/L証拠を別表現へ転用するという意味ではない。

| 項目 | 条件付きで再利用可能な内容 | 新実装で自動継承しない内容 |
|---|---|---|
| ideal controlled unitary | 上の局所恒等式を同じ全event-domainへ適用する規則 | 255 replay済み・全compiled error0という認定 |
| bias b0 / u_enc | 同じideal演算・基準・尺度に対する保存値/包含 | 新compiled biasやactual residual |
| B_saved / log_B上界 | law/重み/normalization定義が同一なら同じ対象。旧B_up marginも保持 | certificate_verified=trueへの変更、PFだからB_up=1への縮小 |
| range/confidence | ±1測定・Bによる重み付けと元α_axis=.025、ε_axis=ε/√2という十分条件の形 | 全20条件同時95%や経験的allowanceの厳密化 |
| 必要shot N | 同じb0/u_enc/B/guardに**新ρ**を入れる関数形式 | 元genericのρやNをactual central値としてコピー |
| 総費用G | 新価格と同じtrajectoryの両軸covarianceを使う会計式 | cosine価格=sine価格という旧経験則、index0価格=cohort平均 |

元条件付き会計の式は、`h_a=ε_a−|b0_a|−u_enc,a−ρ_new,a`、`N_a=ceil(2 B_up² log(2/α_a)/h_a²)`、元guard `u_a/h_a≤.01`、`Ghat=mean_i(N_cos C_i,cos+N_sin C_i,sin)`。これを設計として示すだけで、今回はN/Gを評価していない。

ideal演算が同じというだけでは、compiler丸め・controlled分解・ray/normalization転送のactual residualが0とはいえない。`ρ_new`を独立unknownとし、仮想ρ=0断面を将来評価するなら**仮定**と明記する。旧genericの条件付き順位を新価格へそのまま差し替えない。新しいCの完成と、実N/Gの科学的適格性は別のgateである。

新PRの精度をlaw平均まで主張するには、全allowed trajectoryに対する一様measurement-error envelope、または正当なlaw-weighted envelope等が要る。2本や既存256本の小さい数値残差だけでは未観測eventを含む平均誤差を保証しない。例えばper-trajectory observable差δが正しく囲める場合、同じnormalizationの下で `ρ_a ≤ B E_p[δ_a]` という転送を設計できるが、その全域上界を今回得たわけではない。C/L等の重複項を無断加算せず、誤差の対象を対応付ける。

## 7. 最小計画と科学call数（すべて未実行）

[非実行契約JSON](../artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24/contracts_DRAFT.json)は全候補のexecution/dispatch/grantを無効とし、SOURCE/seal/grant/launchをnullとしている。候補を一括実行する契約ではない。

| 候補 | 目的・新規call | 進める条件・得られる判断 |
|---|---|---|
| A_GATE（推奨する最初の有限実行） | index1のsource body emit1・central lift1・wrapper2・compile2（cos/sin） | 保存order2のSOURCE対応とcompiler価格を確認。平均/ρ保証ではない。完了STOP |
| A_COHORT_COS | 新PR body/lift/cosine compile各最大255＋cost統計1batch | 正確な既存256平均が必要な場合だけ。A_GATE cosineを保全入力として使えれば残254。標本平均・paired差・分布を保存しSTOP |
| A_AXES | 共通bodyを再利用しPF1＋PR256のsine wrapper/compile最大257 | complex Gが必要な場合だけ。A_GATE sineを使えれば残256。両軸価格は別に記録しSTOP |
| A_SPLIT（任意） | central PR index0のretained/event/同じpathのemptyを計3compile | Q3のoperational分解が意思決定に不可欠なら別承認。旧empty0を新pathの測定値へ代入しない |
| B | compile0、paired価格統計1batch、新Pauli10精度行/20軸shot/10G行、指定3pair×5精度=15比較 | 必要な両軸価格を得てから、一つの明示した仮想ρ断面に限定。DF行は既存結果をコピー。実N/Gはnullのまま |
| C（条件付きの診断案） | 新compile0、保存D2 cosine PF/PR2programのsignal residual診断2。明示precision schedule採用時の最大tier call6 | phase/scale不整合等が残るか精度主張に必要な場合だけ。selected2本の局所診断で、PR mean保証ではない |

両軸まで全central価格を揃える表は(PF1＋PR256)×2軸で514 entries、D2のcosine2 entriesを再利用するため新compileは**計512**。A_GATEの2件はこの512の一部であり、上乗せしない。Cosineだけの問いなら追加sine257件を要求しない。将来、固定compilerでの費用同一性を正当に証明できれば契約変更で減らせるが、少数の軸価格一致から全cohortへ外挿しない。

将来のbody-only replayには技術実装が要る。既存A.buildはgeneric wrapperまで構築するため、丸ごと呼んで「bodyだけ再利用した」とは扱わない。Pauli primitive/event emitterと元step順・float phase更新だけを限定的に共通化し、人工検査後、新SOURCEへ固定する。この作業・実H4 replayも今回は行っていない。

各実行の前に、対象ID・input/SOURCE・compiler/runtime・明示承認を新seal/grantへ結合し、callを起動前に非返還予約する。旧grantの残callを流用しない。結果の再利用とgrant再利用は別である。native wait/reap・保存SHA・STOPを保ち、他worktreeの変更やPSI上昇だけで正常workerを止めない。独立targetは利用可能資源に応じ並列化し、CPU/RAMへ恣意的な低い上限を置かず共有環境を変更しない。

## 8. D3・保全・引渡し

D3（DF-native PRのbasis reuse無効化）は今回実装せず、契約上も未承認のままとした。D2後の中心課題に対して、まず新Pauli cohort価格を確認する方が直接的である。D1のDF interaction+3765はbasis reuseだけの効果ではない。DFのbasis reuseが意思決定に重要な未解決要因としてなお残る場合だけ、後でD3の要否を再検討する。

[静的監査script](../scripts/audit_h4_controlled_comparison.py)はstdlibだけを用い、QPYをparseせず、価格統計・shot・物理演算を呼ばない。既存SOURCE・raw・grant・ledger・marker・STOP等の既知1805fileを保全し、[最終検証](../artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24/verification.json)でSHA一致、人工20 PASS、科学call/grant/N_G0を確認した。新しい科学SOURCE・evidence freeze・正式certificateを作成していない。

元Pauli-PR historical hash FAIL／原因UNKNOWN、元H4 native UNKNOWN、旧N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCE、正式certificate未受理は変更していない。[公開記録](../artifacts/non_df_h4_controlled_comparison_redesign/2026-10-24/publication_status.json)に専用local commit、push結果、remote SHAを記録する。基点commitは実際にremoteから取得できたので、過去のpush失敗記録だけを根拠に再pushを要求しない。

**mandatory STOP。次の担当はユーザー**：D2のSOURCE対応と比較設計を確認し、追加計測へ進むならA_GATEのindex1・両軸・最大2compileだけをまず明示承認する。**Codex**：承認された範囲だけbody-only replay接続・人工検査・新SOURCE/seal/grant・one-shot・終了監査を行う。A_COHORT/B/C/D3へ自動連続実行しない。
