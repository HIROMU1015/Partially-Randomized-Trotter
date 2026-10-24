# H4 Pauli central-control：実装残差ρの限定診断準備

**PF・PR index0・PR index1の両軸、計6候補の保存body/input/compiled QPY・state・SOURCE・fingerprint来歴を接続した。最小pilot案はPF cosine＋PR index0 cosineの2論理診断。今回は実H4のQPY/NPZ decode・状態作用・signal/ρ計算を一切実行していない。**

重要な不足は、PRの個別trajectoryのC整合理想信号が未保存であることと、saved compiled program全体のvalidated enclosure接続が未実装であること。保存PR law meanをindex0/1の理想信号に代入しない。未compile入力とcompiled出力の差だけでも、C/N/Gのρ全体を評価したことにはならない。

[非実行契約](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/contracts_DRAFT.json)はR0/R1/R2すべてexecution_authorized=false、dispatch_enabled=false、grant_valid=false。grant・科学runner SOURCE・launch_commandはnull。新科学SOURCE/seal/grant/jobなし。**準備完了後mandatory STOP。**

## 固定条件・来歴

Linear H4、隣接1.00 Å、STO-3G、system8＋ancilla1（wire8）、Pauli-JW184非identity項、PR prefix27。S2、T=.8、q8/r4/K4、元event lawと保存256 trajectory。固定D2 central control、Qiskit1.3.0、rz/sx/x/cx、optimization1、seed17、backend/coupling/layout/routing指定なし。元のHadamard wrapper、保存state/ray・normalizationを固定する。新しい保持規則・精度窓・Hamiltonian分解は導入しない。DF診断・H2のS2/Yoshida4費用順位逆転は今回の対象外。

基点branch `codex/non-df-h4-central-conditional-cng-20261024`、commit `388b1e525d3cfbeba8f231acd42c813f67a7f531`。専用branch `codex/non-df-h4-central-rho-preparation-20261024`。基点を含む7commitと主要7blobをGitHubから実取得し、local Git blob SHA1と一致した。[取得照合](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/github_verification.json)。基点の以前のpush認証失敗記録は時点履歴として保持し、現在GitHubで基点を取得できることを別に記録した。

| 保存科学SOURCE | commit |
|---|---|
| 元H4 | 0c329b30a81accb35c212be8defa66fee0dd0f12 |
| D2 | 74515d8eb4f073c40fc783681a3775dbca89e9b1 |
| A_GATE | ee7d281b5daead1d0d5e81e00c9a0ed4727792b0 |
| A_COHORT_COS | 34ef0a7ca2412a61f5215a4ce1192e99f3226b2a |
| A_AXES | a37232f85812688c3b9ab957c5073c5ce143b17e |
| 条件付きC/N/G | d96dfe5ca0a6c950a5f1791ff61659e6dc8f1c34 |

Cの既存実装SOURCEはe80d7510880f8db993b777c627bb8e95b05a070d。Cの保存model/resultと旧sealのruntime参照を読み、C/L engineは起動していない。今回のcommitは**技術的準備の版**で、科学実行SOURCEの封印ではない。

## 6候補の保存物

固定ID：PF `d761627d56a4454e644358fef8907b8a7db28c4feeb44130224c20e4e50aae4d`、PR index0 `93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97`、PR index1 `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa`。

| 候補 | 保存run | source body命令数 | 保存RZ | 入力fingerprint先頭 | 出力fingerprint先頭 |
|---|---|---:|---:|---|---|
| PF0_cosine | D2 | 2944 | 15028 | a5807762c1da | ac65c4de7d57 |
| PF0_sine | A_AXES | 2944 | 15028 | 8fa3606c021d | 56ede2d2f721 |
| PR0_cosine | D2 | 464 | 1145 | 890f1bd2550f | f7e855e7560c |
| PR0_sine | A_AXES | 464 | 1145 | 7d99238234fa | a29f63a4fdbc |
| PR1_cosine | A_GATE | 469 | 1129 | 156bcc95a46d | e7411e76d88c |
| PR1_sine | A_GATE | 469 | 1129 | 072518c2e59b | 88287619a8c2 |

[全6行の来歴・完全fingerprint・QPY SHA](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/targets.csv)、[機械可読表](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/targets.json)。PF/PR0 cosineのsource_body.qpy・controlled_body.qpy・input.qpy・output.qpyはD2に保存。PR1のbody/controlledはA_GATE run01、cosine/sineのinput/outputは同runのvariants下。PF/PR0 sineはA_AXES run01/trajectories/PF, PR0000に保存され、D2 controlled bodyとの対応がsource_binding.jsonに記録されている。

入力QPY SHAとinput.json、入力fingerprintとmetrics、SOURCEとpaired価格行、compiler profile、event/trajectory hashesを照合した。既存verify_joinと保存保全SHAを使用した。**今回はQPYをdecodeしてfingerprintやlayoutを再計算していない**。将来の最初の数値実行前に実QPYの構造・fingerprintを再検査する。新しい数値一致を確認した記録ではない。

source body aggregate phaseはPF/PR0が0.2651822507334482、PR1が0.26518225073344787という保存値。後者を前者へ合わせず、各SOURCEの集約順序を保持する。新しい位相再集計はしない。

StateはCの`run01/pauli_jw_model.json:state`に256個のbinary64複素pairとして保存されている。既存array_hash純関数で元state hash `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`との一致を確認。元NPZ/receiptも存在し、SHAだけ確認した。[state/ray引用](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/state_ray.json)。normやstateを再計算・置換していない。

## 比較信号とρの定義

ψを保存vector、ν=ψ†ψ>0とする。入力arrayのnormやglobal phaseを勝手に直さず、同じrayのobservableを一貫してνで割る。m_C,i,aはCのexact encoded operandsで定義されるtrajectory iの理想Hadamard信号、m_in,i,aは保存uncompiled central wrapper、m_out,i,aは保存compiled QPYの数学的測定期待値。実機noise・状態準備誤差・shot揺らぎはこの局所simulator診断の対象外。

C保存値は `corrected_encoded_mean` がraw ψのcorrected平均であり、normalized expectationそのものではない。従ってμ_C,aはそのcomponentをνで割った量。PFはB_saved=1のsingleton、PRは同じeffective finite event law pについてμ_C,a=B_saved E_p[m_C,i,a]となる。[C SOURCE](../src/trotterlib/h4_ball_evidence.py)のnative scalar・rotation角・product順序に対応させる。

- compilerだけの差：δ_i,a=B_saved |m_out,i,a−m_in,i,a|。
- Cから保存入力までの差：η_i,a=B_saved |m_in,i,a−m_C,i,a|。
- 必要なselected gap：d_i,a=B_saved |m_out,i,a−m_C,i,a|≤δ_i,a+η_i,a。
- PFのρ_a=d_PF,a（同一対象の包含を確立した場合）。
- PRのρ_a=|B_saved E_p[m_out,i,a−m_C,i,a]|≤E_p[d_i,a]。

**δが小さいだけではηを0にできない。** C retained evolutionはexact encoded coefficient×delta/2で定義される一方、body/D2のrotationパラメータはbinary64積・global phase集約を通る。D2の理想的control恒等式が支持されても、その数値辺の差が消えたとはいえない。新計画ではdirect C-target対outputを主対象にし、input対outputを原因切分け用に併記する。[誤差所有範囲](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/error_ownership.json)。

PR index0/1だけのdが小さくてもE_p[d]の一様/正当なlaw-weighted上界にはならない。保存256は全event domainを覆わず、order4は未観測。CのPR corrected_encoded_meanはlaw meanであり、特定trajectoryのm_Cとして使えない。

## 位相・control・測定意味論

[固定D2実装](../src/trotterlib/h4_d2_pauli_control.py)と[元wrapper](../src/trotterlib/fixed_signal_evaluation.py)を静的に確認し、既存人工テストを再利用した。

| 論点 | 固定する意味 |
|---|---|
| control | ancilla nを最上位wireに置き、diag(I,U)。初期vectorは(ψ,0) |
| cosine | H → controlled U → H → Z測定で+Re〈U〉 |
| sine | H → controlled U → **Sdg → H** → Z測定で+Im〈U〉 |
| source global phase | system bodyのe^(iφ)はcontrol後P(φ)としてancilla相対phaseになる。削除不可 |
| full-wrapper global phase | 全9qubitに共通なphaseはobservable不変でも、QPY/fingerprintは保全 |
| Pauli product | application_sequence順、係数sign、identity product signを保持。順序変更不可 |
| order phase | (-1)^(k/2)など保存event.phaseを一回だけ反映 |
| constant phase | 元SOURCEの各step更新と集約済みphaseを保存。event/constantを再加算しない |
| ray | 元raw vectorを保持し、observableを同じνで割る。出力normで再正規化しない |
| scale | selected unweighted信号差にB_savedを一回。B_upはshot rangeでありρ尺度の代用にしない |

保存RZの両軸一致からsignalや誤差の一致を推論しない。Sineはその実際の保存sine QPYを評価する。Control後の相対phaseをsystem global phaseとみなし、Operator.equiv等で落とす検証も行わない。

旧fixed_signal_evaluationのcompiled checkはancilla確率vectorの差で、Z期待値差そのものではない。正規化された2-outcomeで同じ総和ならZ差は確率の最大差の2倍。新診断では直接p0−p1を使い、旧1e-9 gateをρ上界や新guardとして転用しない。

## 何が保存され、何が不足するか

| 対象 | 現在の証拠 | 次の数値実行に必要なこと |
|---|---|---|
| QPY/source/compiler | 6組すべて存在、SHAと保存fingerprint/profile接続 | actual QPYの構造・layout・parameter/fingerprint確認後に状態作用 |
| state/ray | encoded vector、hash、C norm interval、元NPZ/receipt | 変更せず再利用。同一νでobservableを定義 |
| PF C理想信号 | corrected_encoded_meanのreal/imag包含＋norm保存 | 保存区間を同一rayへ変換。新PF ideal伝播は不要 |
| PR0/PR1理想信号 | model/events/phase operandsは保存、個別信号は未保存 | 選択trajectoryのC整合state action。law meanは代用不可 |
| compiled signal/ρ | 新D2/A_GATE/A_AXESには未保存 | 別承認後に実際の保存programを限定評価 |
| interval backend | C Arb primitives、rational helpersあり | selected Cとcompiled local-state-actionの接続・人工検証・新SOURCE固定 |
| PR全law/費用tail | 未確定 | R2転送設計。必要な追加計算は別契約。費用tailも別問題 |

[不足一覧](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/missing_evidence.json)。保存C totalからの減算、C/L上界の無条件加算、旧generic残差の移植は行わない。u_enc=`bias_distance.upper`、b0、B_saved/log_B上界、B_upは既存定義のまま。[全20軸のguard予算引用](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/guard_inputs_quoted.json)はコピーだけで再計算していない。

## 数値手法と再利用判断

[機能別の再利用表](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/reuse_matrix.json)、[非実行adapter](../src/trotterlib/h4_central_rho_plan.py)、[metadata専用runner](../scripts/prepare_h4_central_rho.py)。

1. **最初の経験的pilot**：既存Qiskit Statevector.evolveを再利用し、保存input/outputの最終measurementだけを外して同じraw vectorへ作用させる。ancilla各sectorの振幅二乗和の差をνで割る。512振幅の伝播でよく、9qubit全OperatorやHamiltonian expを作らない。C-targetを別に対応付け、δとdを区別する。binary64結果はEMPIRICAL、numerical_error_upper/rho_upper=null。小さい値から保証PASSにしない。
2. **PF targetの再利用**：Cの保存corrected_encoded_mean/νを使う。保存target幅が判別を妨げた場合はSTOPし、元Cの再実行をこの契約に混ぜない。
3. **PR selected target**：既存trajectory_reference/event_unitaryのchronological scheduleとCのpauli/real/phase/signal規則を、保存model/eventへ最小接続する。全law・reference・Hamiltonianを再生成せず、selected vector作用だけを対象とする。full matrixの既存親entrypointをそのまま呼ばない。現在のadapterはこのreal-H4 executorを意図的に持たず、future scientific SOURCE固定前に最小接続と人工検証が必要。
4. **包含が必要な次段階**：CのArb実数・複素数・endpoint処理を流用する。compiled rzはencoded binary64角のinterval sin/cos、sxは有理複素係数、x/cxは置換として同じwire順で扱うlocal-state作用のbridgeが必要。巨大dense full-unitaryをgateごとに再構成する設計は避ける。これは現在完成したvalidated simulatorではない。実H4での性能・幅は未測定。
5. **Lの範囲**：LはDF-nativeのrz/XXPlusYY・basis/supportを対象とし、Pauliのsx/cx/全wrapper compiler証明ではない。誤差合成規則は参照できるが、DF実測上界をPauliに転用しない。C/L COMBINATION_BLOCKEDは保持。

今回の新規実装は、metadata接続・実行拒否・人工小系のobservable縮約・mock判定だけ。独立した科学simulator、Hamiltonian builder、compilerや区間基盤を新規作成していない。

## 数値精度・判定・資源

最厳ε=.0005の保存guard予算はPF cosine2.64900e-6/sine3.36139e-6、PR cosine2.59361e-6/sine3.35232e-6。これは許容予算で、測定したρではない。

提案する**数値判別幅の目標はB_saved尺度で絶対幅1e-8**。新しいnumerical allowanceやguardへの置換ではなく、診断intervalをどこまで細くするかの設計値である。必要bit数は事前に確定できない。binary64（mantissa53bit）はgross error診断に有用でも、同じ精度を二回計算した一致だけでは丸め保証にならない。

有効な包含backend接続後に限り、元草案に対応するArb 64→96→128bitを有限な任意選択scheduleとする。**現在はscheduleも実行無効**。有効なenclosureで幅>1e-8のWIDTH_INSUFFICIENTだけが次tierへ進める。semantics不一致・実残差の予算超過を理由に精度を上げてPASS探しをしない。最大128でも幅不足は不十分STOP。十分細いがguardを跨ぐ場合もBUDGET_OVERLAP_STOPとし、分類変更のための無制限精度追加はしない。

ρの包含上端≤予算ならLOCAL_BUDGET_MET、下端>予算ならLOCAL_BUDGET_EXCEEDED。ただしPRはそのselected trajectoryに限る。ω_C/ω_outなどの数値不確かさはdを囲むための内部項として扱い、`u=u_enc+rho`に第三のallowanceを再加算しない。依存関係を利用した差引きや重複除去を未証明のまま行わない。

Runtimeは[既存分離環境](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/runtime_plan.json)を利用可能。point用native-dependency-v2はPython3.11系/Qiskit1.3.0/numpy1.26.4/scipy1.14.1でpython-flintなし。C用.c-reference-fix-venvはPython3.12系/python-flint0.9.0でQiskitなし。package変更0。将来はQPYをQiskit側で読み、wire/phase/binary64 operandのexact IRをArb側へ渡す。JSON round-trip/hex/rational同一性を新SOURCEで固定し、既存runのroutingを上書きしない。

独立なPF/PR診断を別processで並行可能。Arbのprecision contextはprocessごとに管理し、同じケースのtierは逐次。local-gate作用と同じideal複素信号の両軸再利用で重複を減らす。CPU/RAM/worker数への恒常的な上限・wall/RSS制限・affinity指定は設けない。保存QPY長から規模を判断するが、実H4未実行なので時間予測の実測根拠はない。共有設定/他user/jobを変更せず、無関係なworktree HEADや一時PSIだけで正常計算を止めない。

## R0/R1/R2：別承認する有限package

| 候補 | 対象・順序 | 論理診断 | binary64段階での最大新state作用 | 支持範囲 |
|---|---|---:|---:|---|
| R0 | PF0 cosine＋PR0 cosine。独立に処理しR0完了STOP | 2 | input2＋compiled2＋PR0 C-target1＝5。PF targetは保存値 | PF/selected order0の局所差。経験的/包含を区別 |
| R1 | 追加1件ならPR1 cosineを優先。その後、明示的に選んだPF0/PR0/PR1 sine | 最大4 | input4＋compiled4＋PR1 C-target1＝9（R0複素idealを適合確認後再利用） | order2 product/phaseと両軸。order4は対象外 |
| R2 | 全allowed eventへのuniform/正当なlaw-weighted転送設計 | 現在0 | 0 | 追加証拠の要否と最小契約の設計だけ |

R0とR1を合わせた候補上限は6論理診断。論理診断とstate作用を区別する。R0のC selected complex signalを保存すれば、R1のPR0 sine idealを同じSOURCE/bit数/scaleで再利用できる。別compiled sine QPYの作用は省略しない。再利用cacheが不適合/欠落なら勝手にcall予算を追加せず、新しい範囲判断へ戻す。

区間scheduleを**別途明示採用した場合のみ**R0最大6 logical-tier/15 state作用、R1最大12 logical-tier/27 state作用。これは経験的pilotに無条件で上乗せする予算ではない。実行時はpointのみかenclosure scheduleかを一つの明示契約へ選択・固定し、未承認の分岐を走らせない。

R0の結果を見てR1/R2へ自動移行しない。構造coverageを一つ増やすならPR1 cosine、同じ演算の測定符号を実programで確認するならsine追加に価値がある。最初から6件全部を走らせる必要はない。R2では未観測order4を含むcompiled program mapとC-target/角度/control/compiler辺全域の上界を明示する必要があり、selected2件や保存256本の小さな平均誤差では代替できない。これでRZ母平均費用tailまで解決したとも扱わない。

## 人工検証・技術回復・保全

新[人工15件](../tests/test_h4_central_rho_plan.py)と既存D2人工12件・controlled意味論8件を実行し、最終35件PASS。全て主agent自身の技術検査であり、実H4の科学的数値検証や独立レビューではない。[初回人工失敗](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/artificial_first_failure.log)、[修正後15件](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/artificial15_PASS.log)、[既存D2](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/reused_D2_12_PASS.log)、[既存意味論](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/reused_semantics8_PASS.log)。

初回はraw normの大きい人工rayで旧probabilities APIをそのまま使い、15件中1件失敗。固定Qiskit実装にnp.clip([0,1])があることを確認した。Statevector.evolveは流用し、伝播振幅からunclipped Zを縮約して元νで割る最小修正で15件PASSとなった。[技術回復記録](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/technical_recovery.json)。入力stateを正規化/書換えしていない。この人工所見から旧実H4結果の不具合を推定しておらず、旧結果を再解析していない。

人工fixtureだけで2〜3system qubitの±control branch、global/relative phase、sine符号、Pauli順序、raw ray、B一回適用、compiler-only差とC-target差の相違、QPY構造、SOURCE/fingerprint/profile不一致、9qubit shapeの数値呼出前拒否、CLI実行拒否、無効契約、mock precision分類、非返還callと重複拒否を検証。新adapterのnumeric経路はARTIFICIALかつsystem≤3に限定される。

metadata inspectorはstdlibだけで実行し、実QPY/NPZ decode0・数値backend import0・norm/state/signal/shot計算0。対象入力83file、保全163fileを確認した。前段C/N/G namespace全体とselected既存証拠をSHA保全し、旧artifact treeへの差分がないことも確認する。大量の別worktree/履歴横断や全cohort数値再検証は行わない。

[入力manifest](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/inputs.json)、[保全snapshot](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/preservation_before.json)、[準備STOP](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/STOP.json)、[主agent metadata57項目PASSの検証](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/verification.json)。STOPは**科学jobを起動していない準備終了の記録**で、native exit/reapを捏造・元H4 native UNKNOWNを補完するものではない。今後は旧CallBudget/native witnessを新契約へ再利用し、call前予約、失敗回収、消費済みgrant保全、成功部分再利用を実施する。

最新版運用指示§8.4/10/10.1/10.2を適用。参照された§10.3は利用可能な2026-10-23版に存在せず、版不一致を記録した。未取得規則を読了扱いにせず、今回の直接指示にある再現性・必要十分な検証・過剰管理回避を優先した。

## 次の承認・未解決状態

**ユーザー**：次に進むなら、まずR0の2論理診断（PF/PR0 cosine、保存QPY/保存state、compile/sampling0）について、pointのみか、backend実装・人工検証後の有限enclosure scheduleかを選んで明示承認する。必要な新state作用は上表で区別した。R1/R2は別の範囲判断とし、最初から6件を一括必須にしない。

**Codex**：承認後、selected C-target/real-QPYへの最小connectorと必要なinterval bridgeを検証し、科学SOURCE/runtime/input/output schema・未使用one-shot grantへ固定してから承認されたcallだけを実行する。今回のtoy entrypointをflag変更だけでH4へ流用しない。source/理想演算/scaleに重要な不一致が残る場合は科学call前にSTOPする。

維持する未確定状態：Pauli-PR historical hash FAIL/原因UNKNOWN、元H4 native UNKNOWN、旧N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCE、新compiledρ UNKNOWN、PR母平均費用tail UNKNOWN、正式certificate未受理。前段条件付きC/N/Gを変更していない。局所ρの検証ができても、PR全law・総費用保証・正式certificate受理は別判断である。

Commit/push/remote SHAは[公開状態](../artifacts/non_df_h4_central_rho_preparation/2026-10-24/publication_status.json)へ記録する。**mandatory STOP。実H4の残差診断・新compile/sampling・N/G再解析・certificate受理へ自動移行しない。**
