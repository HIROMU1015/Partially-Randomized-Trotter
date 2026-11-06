# Track D Stage B3：元H向けRPE統計・必要shot数の接続準備

2026-11-06 JST。**B1の元H向けoperator十分条件を、既存のradius→coordinate誤差→Hoeffding shot数へ接続するための部品はある。元H専用の状態・energy会計接続、採択した目標でのalias／全round成功条件、実装law・合成誤差の資格が未接続である。** 本書は静的調査と次の最小作業案であり、Nの新規算定・科学実装・科学検証・詳細科学レビューは行っていない。公開・照合後mandatory STOP。

## 1. 範囲・固定前提・資料の取得状況

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。system8 modes／sector36。入力は**元Hamiltonian Hの厳密基底状態**を仮定し、状態準備費は研究範囲から除外する。保存own-DF groundを同じ状態として扱わない。

既存m=0..12／q=1..4096は再利用候補の保存scheduleである。13-round、旧energy=.00159 Ha、failure=.01、βtotal=π/6、βstat=π/12、αaxis=.01/26、DF reserve=1/8を、今回の新しい研究条件として採択していない。これらの数字は既存記録の引用である。

適用する共通・repository AGENTS.md、PROJECT_MAP、研究概要・現状、関連する研究目的・解析手順、VALIDATION_STATUS／manifest、正式共通指示 `research_project_instructions_unified_2026-10-25.md` を参照した。正式共通指示はlocal原本を読んだもので、repositoryに存在しない公開リンクは付けない。旧2026-10-23版を運用正本として併用しない。

専用branch `track-d-b3-rpe-statistics-preparation-20261106` をC2結果commitから作成した。以下の4固定commitはlocal Git objectから取得し、前3件がC2の祖先であることも確認した。C3の進行・未公開結果には依存せず、そのbranch／worktree／SOURCE／入力／結果／STOP／processには変更・操作をしていない。

| 固定資料 | 取得・参照した内容と役割 |
|---|---|
| [B1/C1結果][B1FIX] `d3913866e6ade4ab97ac02645fed91b2d5fcfa75` | RESULT、PLAN、B1全13行JSON、SOURCE metadata、STOP。実行SOURCEは `8c6eeb02e89f6cf14870789719493e75482f62f8` |
| [B/C静的準備][BCFIX] `5f94972adf5a10e96e9400d3919c00d14e7a718a` | REPORT、STAGE_B、STAGE_C。B1前の不足記述は履歴として読む |
| [Stage A結果][AFIX] `3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0` | RESULT、SOURCE metadata、STOP、有限cohort価格の定義。実行SOURCEは `2dd012dc77dcc877d5918c7d212819eb79b97557` |
| [C2結果][C2FIX] `c8fe5091c42ec786688b6931105ad64d10c87d18` | RESULT、SOURCE metadata、STOP。固定sample00→01／q2／step境界cutの4 full-wrapper。実行SOURCEは `e0befa619a14b956ad64349a86ec2f261962883c` |
| [CDF8 r32報告][CDFREPORT]・[個別固定契約][CONTRACT] | 同endpointの13-round、旧state／target／lane、one-shot、retry=resume=0、mandatory STOP。旧PR対ENTRY比較をReference対symmetric比較へ転用しない |
| [CDF8科学SOURCE][R32SOURCE] `4df8eca29c59ffd55e2dd898f05da16bb246ddbe` | `c_near_practical_r32.py`、`c_update064_r32_downstream.py`、`c_rte_r_retuning.py`、`c_pr_integrated_performance.py` の関連call siteをGit objectから静的閲覧。実装commitは `703938aba8cc38080e29e75acde8e0edc8dce868` |
| [CDF8 SOURCE manifest][CDFSOURCE] | 139 fileのclosureを記録した既存manifest。今回はclosure全体の再監査・再hashをしていない |
| 保存carrier `3649b75af191916daf44abb91e14d34c43b83e8f` | 数値・events・CAR・QPYの所在。元の科学SOURCEと区別。physical_ground JSONはこの固定Git objectから取得 |
| Stage A/B1の再利用元C SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` | 各SOURCE metadataの143 SOURCE・16 input・96 cohort bindingの来歴を確認。今回全payloadを復元・再検査したとはしない |

元h/g/scalar packetの保存SHA256は `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、PR J/U endpointは `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。B1 input bindingは既存[SOURCE.json][B1SOURCE]にある。law・endpoint・scalar／identityを別のlegacy DF12や別endpointへ置き換えない。

今回は文書／source／保存JSONとGit metadataを読んだ。科学module import、runner／test実行、NPZ配列展開、QPY decode、phase／radius／energyの数値再解析、N算定、decoder成功率計算、sampling、ground solve、DF refit、回路build／compile、総費用集計は行っていない。科学結果・既存契約・grant／ledger／marker／STOPを変更・再消費していない。以下の数値は保存欄からの引用で、新しい測定結果ではない。

## 2. B3-1：B1保存schemaと既存pipelineへの対応

### 2.1 B1が既に得た条件付き結果

[B1/result.json][B1JSON]は `status=CONDITIONAL_OPERATOR_SUFFICIENT`、`model=exact_original_H_ground_in_invariant_Nalpha2_Nbeta2_sector36`、`law=ideal_finite_RTE_IID_exact_native`、`rigorous_enclosure=false` を保存する。各rowの `N=null`、packetの `N_evaluated=false`、`actual_RPE_success_qualified=false` を維持する。

[transport_round][TRANSPORT]は同じHermitian H／H_CDFと共通不変sectorを前提に、保存DF norm＋engineering guardをD_Hとして、既存Duhamel輸送を保存PF／RTE operator誤差へ接続した。

\[
d_{H,m}=\min(2,t_mD_H),\quad
\epsilon_m=\epsilon_{PF,m}+\epsilon_{RTE,m}+d_{H,m},\quad
\Delta_m=A_m\epsilon_m.
\]

exact-H groundに対する中心は `A_m exp(-it_m E0)`。B1の保存定義は `rho_lb=A_m-Delta_m`、`beta_sys=asin(epsilon_m)`（epsilon_m<1）である。partial PFの実信号半径を1に固定した評価ではない。

| 保存row欄 | 意味・単位／再利用時の注意 |
|---|---|
| `m,q,time` | round／coherent occurrence数／t_m=hq。r32 eventは1 occurrence内に含まれる |
| `attenuation` | raw first momentの正のnormalization係数A_m。signal errorそのものではない |
| `saved_pf_operator_error` | 保存CDF-relative PF operator 2-norm。energy誤差[Ha]ではない |
| `saved_rte_corrected_operator_error` | normalization-corrected finite-RTEとPFのoperator 2-norm。同じnormで加算 |
| `df_hamiltonian_error_premise` | D_H [Ha]。同じendpoint／sectorに対する保存norm＋guard |
| `df_evolution_error` | min(2,t_m D_H)。dimensionless evolution誤差 |
| `original_H_corrected_operator_error` | 上記epsilon_m。元H exact evolutionまでの条件付き誤差 |
| `raw_signal_error` | Delta_m。A_mを掛けたraw complex-signal単位 |
| `observed_radius_lower` | A_m−Delta_mという下界。元Hで観測した実半径の数値ではない |
| `deterministic_phase_upper` | 元H理想位相に対する系統phase上界[rad] |
| `historical_*`、`extra_raw_implementation_error_room_historical` | 旧配分に対する診断。新しい誤差割当や実装誤差boundではない |

B1の保存D_Hは `8.060251994885292e−9 Ha`。m12の保存欄はA=`.852119825515583`、epsilon=`.12062707239775255`、raw error=`.1027887198840285`、radius lower=`.7493311056315545`、phase upper=`.1209215432235359 rad`。全13 roundの旧phase条件が成立し、保存された最小slackは `.14087784457561353 rad`。追加raw誤差余裕 `.11775611966884127` も旧配分の診断引用であり、未知synthesis／law誤差がこれ以下とは認定しない。[B1固定結果][B1FIX]

DF差はここでexact-H targetへの信号誤差に一回入っている。旧own-DF energy評価のWeyl加算を後から重ねてはいけない。保存norm＋engineering guardとbinary64計算をpremiseとしたnumerical pilotであり、丸め誤差の厳密包囲には格上げしない。

### 2.2 既存関数の入力・再利用境界

| 関数・SOURCE | 入力と既存処理 | 元H向けの扱い |
|---|---|---|
| [AccuracyTarget][ACCURACY] L25–52 | energy、DF reserve、h、βtotal／βstat、αaxis／αtotal。0<βstat<βtotal<π/3。horizonはDF reserveを引いたenergyから計算 | 現状のまま採用しない。元H輸送後のenergyへ旧DF reserve政策を自動適用しない |
| [RoundEvidence／assess_round][ACCURACY] L55–120 | m、A、PF／RTE誤差、raw synthesis／law誤差、target_id、state、alias／IID、evidence_kind／law_model。radius、phase、coordinate、Nへ変換 | state-neutralな幾何・shot部分は再利用候補。wrapper全体はown-DF専用 |
| [assess_energy][ACCURACY] L123–152 | horizonの全row、同一target、両軸failure和、own-DF状態とaliasを要求。DF[Ha]＋最終phase幅/hq | 元H専用接続が必要。DFを0と偽装したり、own-DF state名で通したりしない |
| [transport_round][TRANSPORT] L13–52 | 同sector誤差輸送・raw変換。Nを算定しない | B1保存rowと定義を再利用。旧targetの履歴slackを新配分として再使用しない |
| [reference_kernels／reference_round][REFERENCE] | ordered PF、corrected kernel、matrix powers、operator 2-norm、任意stateのraw信号 | 将来B2が必要な場合の部品。今回起動不要。own-DF保存raw信号は元Hの信号でない |
| [_hoeffding_shots][RPE] L553–561 | 正のcoordinate allowanceと0<alpha<1（呼出側でも検査）→ceilの整数 | 元Hのqualified coordinateを与えれば式自体は再利用可能。今回呼び出さない |
| [evaluate_rpe_round_candidate][RPE] L564以降 | unit-radius effective-PF eigenstateモデルとPF係数を用いる候補評価 | B1の元H operator誤差をそのまま入れるAPIではない。全関数を流用して元H資格を得たとしない |
| [reconstruct_rpe_phase][DECODER] L74–123 | 測定round位相列と連続dyadic q。前round推定に最も近いbranchを選択 | 復元関数はstateに依存せず再利用候補。科学的成功条件を関数単独では検証しない |

現行 `assess_round` L81は `state_model != 'own_DF_ground_state_oracle'` を拒否し、`assess_energy`も同状態を要求する。`c_pr_aware_df_stage_c::{assess_round,energy}` のcall siteもown-DF／alias=true／IID=trueを固定している。これを元Hの証拠として無変更で使用しない。[lane処理SOURCE][LANES]

accuracy、reference、connection、lane処理、resource accounting、decoder、[horizon関数][HORIZON]の7ファイルは、今回の基点とCDF8科学SOURCE `4df8eca2…` でGit blobが一致した。これは対象部品の版照合であり、closure全体の検査や科学的な再検証ではない。

**今回提案する最小接続（未実装）：** 将来の新SOURCE revisionで、raw error→radius／phase→coordinate／Hoeffdingの小さい状態非依存部分を既存accuracyから共通化する。旧own-DF wrapperのstate guard・energy定義は残す。元H用の明示wrapperはB1 row、元H／CDF／sector／law identity、実装誤差の資格、alias／IID条件を受け取り、original-H stateを明記して共通部分を呼ぶ。元H用energy評価は既に信号側へ含めたDFを再加算せず、既存horizon関数と全round／failure検査を再利用する。固定済み過去SOURCEを書き換えない。

## 3. B3-2：N・decoder・全round成功条件の資格

### 3.1 raw誤差からshot数への接続案

以下は[既存accuracy][ACCURACY]と[解析手順§5][METHOD]の式を、B1の元H-targetへ接続するための入力仕様である。**数値代入・新規N算定はしていない。**

元H用の全raw complex-signal誤差を

\[
d_m=\Delta_m^{B1}+e_{m,\mathrm{implementation}}+e_{m,\mathrm{law}}
\]

として、同じnorm・同じraw単位へ換算する。implementationにはcompile／synthesis、actual adjoint、Referenceのχ補正・reference leakage等を含め、重複するboundを二重加算しない。PF／RTEのepsilonへraw誤差を直接足したり、raw項へAをもう一度掛けたりしない。

既存の幾何手順は `rho_m=A_m-d_m>0`、`beta_det,m=asin(d_m/A_m)`、`beta_det,m+beta_stat,m<=beta_total,m`。統計座標を同じ幅にする既存選択は

\[
\epsilon_{m,X}=\epsilon_{m,Y}
=\rho_m\sin(\beta_{stat,m})/\sqrt2,\qquad
N_{m,b}^{\mathrm{suff}}
=\left\lceil\frac{2\log(2/\alpha_{m,b})}{\epsilon_{m,b}^2}\right\rceil.
\]

軸別の幅を選ぶ場合は、既存一般式の `sqrt(epsilon_X²+epsilon_Y²)<=rho_m sin(beta_stat,m)` を守る。最小案は同じ座標幅と明示した軸別alphaを使い、不要な配分最適化を追加しない。これはHoeffdingによる**十分なshot数**であり、情報論的な最小Nではない。

現在のaccuracy wrapperは1 roundに1つの `shots`／`alpha_axis` を返し、energyは `2*alpha_axis`、総費用は両軸に同じshotsを使う。将来N_X≠N_Yやround別配分を採用するなら、出力・failure和・費用加算も軸別に小さく拡張する必要がある。今回その配分を採択せず、コードも変更しない。

### 3.2 確定・未確認・追加条件

| 項目 | 今回確認した既存状態 | 次の接続に必要な条件 |
|---|---|---|
| deterministic phase error | B1のideal-law／exact-native元H向け13行に保存済み | 採択したtargetでの比較。追加実装誤差を含むlaneは別資格 |
| positive radius | B1の保存十分条件は成立 | 実装／law項を含めたd_m<A_m。未知項はnullのまま |
| statistical phase allowance | B1は旧βstatとのslackだけを保存 | 新しいβtotal、βstatとbranch余裕の明示採択 |
| X/Y coordinate誤差・N | 式と既存helperはある。元H用Nは未算定 | qualified radius・βstat・軸別alphaから別許可下で算定 |
| round／axis failure配分 | 旧均等配分は保存履歴 | 全m,bのalpha和が採択したalpha_total以内。浮動小数点の許容超過でPASSへしない |
| dyadic decoder | 連続q=1,2,4,…を検査し最近branchを選ぶ既存実装あり | 全roundのphase条件・符号・初期aliasを結ぶ。保存accounting PASSだけをdecoder成功実験と呼ばない |
| alias／phase branch | 保存physical groundの旧alias gateはPASS | 同じ元H・sector・h・新βtotal・energy prior windowへのbindingと、wrap境界を含むenergy区間の扱いが必要 |
| 全round成功 | 既存strict branch guardはβtotal<π/3、energy解像度はβtotal/(h q_max) | 採択したhorizon全row、全系統誤差、全coordinate成功事象を同じ元H位相へ接続。今回成功率を新評価しない |
| implemented sampling law | 保存lawは `implemented_finite_bit_law_error=null` | finite-bit marginal biasとshot間／occurrence間の依存を区別し、実際に用いるjoint law／IID bridgeを示す |
| synthesis／compile／phase補正 | Stage A/C2は有限fixtureの数値postcheck | 使用domainのraw signal errorまたは適切なoperator/observable誤差からのbound。観測最大残差は全law boundでない |

全coordinate成功事象へのunion boundはround間独立を要しない。一方、Hoeffdingの各軸内の独立な±1 shot条件、二項productによる成功率計算で用いるX/Y独立性は別条件である。lawの平均bias boundだけでは相関shot列をIIDにできない。全列のlaw距離で成功事象を評価する方式を選ぶなら、その誤差は成功確率予算へ適切に入れる必要があり、単なるraw平均誤差との混同を避ける。方式・数値は今回選定していない。

各shotはfreshな元H状態を用い、各occurrenceのtrajectoryを対象lawどおりに生成する。同じ保存wordをN回測る、保存32本を巡回する、同じ短pathをq乗する方法ではfresh whole-path lawを置き換えられない。seed非重複はideal IIDの証明でない。価格標本数32は量子shot数Nへ掛ける係数でもない。

### 3.3 decoderとaliasの具体的な不足

`reconstruct_rpe_phase` は `arg(mean_X+i mean_Y)` の列を受け、E0を入力にせずbranchを選ぶ。U(t)=exp(−itH)ならbase phaseはφ=−hE0、energyの符号は−φ/h。scalar／identity補正を二度加えない。主値phaseのwrapと物理energyへのunwrappingは区別する。scalar版はtieを `argmin` で一つに選び、vector版は1e−12でtie診断を記録するが、ゼロ座標や非有限入力を含む成功条件の代用品ではない。[decoder SOURCE][DECODER]

[旧ground_gate][GROUND]は固有対residual／normalization、energy window包含、`π−(max(|lo|,|hi|)*h+beta_total)>0` を検査する。[保存physical_ground JSON][PHYSICAL]にはenergy=`−2.1663874486347625 Ha`、residual=`1.7642254846760187e−15`、alias_margin=`1.6179938779914944 rad` がある。これは旧window [−10,10] Haと旧配分に対する保存数値pilotである。JSONのstate文字列は共通helper由来の `own_DF_ground_state_oracle` だが、固定r32 SOURCEはphysical Hとcandidate Hを別々に渡している。ラベルだけを根拠に元H資格やcandidate同一性を決めない。

次のalias接続は保存physical H／packet／sector basisと対象の結び付きを確認して再利用する案を優先し、新ground solveを自動必須にしない。厳密ground oracle仮定と保存binary64 eigenpairの証拠水準も区別する。B1ではalias／horizonを新資格化していない。旧の数値groundやE0を正解branch選択のoracleとしてdecoderへ渡さない。

元Hへの系統誤差を全てphase側へ含めた上記経路では、選択した共通βtotalについて既存horizon関数の `beta_total/(h*2^M)<=epsilon_E` を使う接続案になる。旧own-DF経路のDF energy加算・reserveを重ねない。round別βを変える場合は、既存の共通βによるbranch条件・最終幅を無変更で利用できるか確認してから採用する。今回はMも新しく計算していない。

legacy DF12／LD3,12／δ=.02／18-roundの[measurement adapter調査][LEGACY]はinterface設計の参考に限る。CDF8／LD2／sector36へそのまま実行せず、旧18-round NやV2成功率を転用しない。

### 3.4 ideal laneとimplemented lane

保存CDF8 [references.json][CDFREF]のevidence laneは `UNQUALIFIED_ROUNDS`／N・費用null、ideal laneの `total_shots=14738` は**own-DF ground**での旧条件値である。元HのNとして採用しない。assumed laneの合成・law誤差仮定も実測boundへ格上げしない。

将来の元H ideal laneはideal IID／exact nativeという**モデル定義**を明記すれば、B1と新targetから条件付きの十分Nを算定する候補になる。これはactual finite-bit samplerやcompiled Reference実装の誤差を0と認定することではない。actual laneでは未知項をnullとし、N／全成功保証を未資格に残す。B1の余裕だけでこの不足は埋まらない。

## 4. B3-3：Referenceと強いsymmetric-directionalでNを共通化する条件

[既存測定対応][SIGNAL]の一shot恒等式を、各roundの同じwhole ordered pathに適用する準備である。

\[
V_{\omega,m}=U_{\omega_{q-1}}\cdots U_{\omega_0}=A_{\omega,m}B_{\omega,m},\quad
V_{\omega,m}r=e^{i\chi_{\omega,m}}r.
\]

actual A†と正しいpath全体のP(+χ)を使う一回のReference gadgetは、既存恒等式の条件下でdirectと同じcoherence z=〈E0|V|E0〉を読む。同じH→core→YのみSdg→H、bit0→+1／1→−1なら各軸の条件付き分布は `Pr(Y_b=y|omega,m)=[1+y f_b(z)]/2`、f_X=Re、f_Y=Imである。これは既存条件の転記であり、全lawで新しくcorrectnessを確認した結果ではない。

| 共通化条件 | 必要な証拠／未確認事項 |
|---|---|
| 同じwhole ordered path | 同じq occurrence・全r events・head／逆順head・符号・scalar／identity。短gadgetのq回反復や固定wordのq乗で置換しない |
| 同じp_m・normalization・attenuation | 同endpoint／K0／r32／h／軸のtrial定義。B_K補正を±1 outcomeへ重ねて乗算しない |
| vacuum ray・χ・actual adjoint | 全pathのray条件、path全体のχ、実際のA.inverse、Gaussian位相・scalar／identity一回の処理。平均χや一stepのχを流用しない |
| 同じfresh条件 | 各shotでfresh system／reference／ancilla、lawに従うfresh trajectory。途中reset／post-selection／weight変更を無断追加しない |
| 同じ測定分布 | 各round・axisの条件付きX/Y対応と同じsamplerを結ぶ。system出力状態や任意multi-gadget channelの一致は不要かつ未主張 |
| 同じdecoder・目標 | 元H用N自体が資格を持ち、同じround domain・alpha／beta・alias・energy区間を使用する |
| compiled有限精度 | 両方式の全対象domainの追加誤差が共通予算内。両方式のより保守的なradius／phaseを使う共通Nも候補だが、新算定が必要 |

idealに同じ全測定列のlawが成立すれば、同じN列・decoderの成功分布を共通化できる。有限精度で誤差が異なる場合、同じNの採用は追加boundと共通予算の確認後であり、自動ではない。「同じN」は方式間で同じ `{N_m,X,N_m,Y}` を使う意味で、全roundへ同じ整数を使う意味ではない。

現在の保存証拠はStage Aのq1・32 path／cut16、C1/C2のsample00→01・q2／step境界cutである。C2のcompiled paired X/Y PASSもその一fixtureに限る。全256 system入力列の検査はそのfixtureの元H入力にも対応する数値observable証拠だが、全trajectory・全roundの誤差boundではない。残る証拠は、使用全domainに適用できる構造対応／law bridge、全law実装誤差と元H予算への接続、元H N／alias／全round成功条件である。C3のq4/q8結果が得られても、そのscopeを越える資格を自動付与しない。

## 5. B3-4：全RPEコスト会計への接続

### 5.1 共通入力と資源の定義

| 入力群 | BとCの間で一致させる情報 |
|---|---|
| 科学identity | 元H packet、CDF J/U、sector／encoding／bit順序、元frame順LD2、scalar／identity、SOURCE版 |
| round・測定 | m/q/t、s/r/K/h、whole path／cut、X/Y符号、fresh law、weight／trial規約、元H状態と準備費除外 |
| 精度・成功条件 | energy精度、failure総量と軸別配分、deterministic／statistical誤差、実装誤差単位・coverage、alias／decoder |
| Cの価格 | 方式・axis別full-wrapper価格、P0／routing／synthesis精度、valid q domain、sample selection、law、期待値かfinite fixtureか、SE／covariance／区間資格 |

同じ定義で加算する資源gについて、接続する目的関数は既存の

\[
G_g=\sum_m\sum_{b\in\{X,Y\}}N_{m,b}
\mathbb E_{\omega\sim p_m}[C^{\mathrm{full,no\text{-}prep}}_{g,m,b}(\omega)]
\]

である。今回は数値和を計算していない。

| 資源 | 全RPEでの意味と境界 |
|---|---|
| RZ count／CX count | 同一精度・compiler・wrapper scopeのgate数期待値×Nの和。RZ countをT countへ自動換算しない |
| total depth／RZ depth／CX depth | 1 interrogationの指定されたdepth期待値×Nの和ならserial実行負荷指標。whole-wrapper compile depthとfenced scheduled serial sumを混ぜない。実時間・T-depthとも同一視しない |
| qubits | 逐次shot実行なら対象domainで必要な**peak width**を別記。Nやqを掛けず、観測最大を未検証全lawの最大幅としない。shotを並列実行するなら同時実行数に応じた別のpeak定義が必要 |
| full-wrapper | H／Y readout／測定、Reference register・CSWAP・χ・actual A†・cut損失を含む。system状態準備費を除外。実機reference初期化・維持／reset、routing／合成のscopeは別途明示し、不明を0にしない |

### 5.2 再利用できる会計と誤って使いやすい点

[rpe_resource_accounting][RPE]は軸別N×期待価格、全round連続性、同一Hamiltonian／compiler／scope／cost model、failure unionを検査する。`build_rpe_resource_summary` は専用candidate型と既存保証モデルを要求するため、B1 rowを型だけ合わせて渡さない。

[accuracy.aggregate_totals][ACCURACY] L184–229は全m×X/Y coverage、target_id、scope、long_round_coverageを確認し、6資源を加算する。`expected_scope`を明示できても、文字列を変えるだけでReference価格の資格を得たことにはならない。qubitsはMETRICSに含まれないためpeakを別保持する。軸別N採用時は同じ小拡張が必要である。

CDF8固定SOURCEの実際の経路は `c_near_practical_r32.execute → c_update064_r32_downstream::{references,calibrate} → c_rte_r_retuning.assess_lanes`、費用は `c_pr_aware_df_m3::{serial_cost,calibration_uncertainty}` を使う。[r32 downstream][R32DOWN]／[retuning][RETUNING]／[M3][M3]

**このM3経路は `compose_serial_cost(...,substeps=1)`**。1 kernelはr32 eventsを既に含むcomplete h-stepであり、roundのkernel multiplicityはqである。汎用旧 `c_pr_aware_df_stage_c.serial_cost` のsubsteps=4を誤移送したり、r32・classical sample32を追加乗算したりしない。

[compose_serial_cost][CONNECTION] L128–153は `native_kernel_repeat_with_fences` と別head/readout shellの政策に限る。countはnative連結、depthはscheduled serial sumで、whole-wrapper再compileの価格ではない。Reference full wrapperをkernel欄へ置くとshell重複やgadget反復の意味変更を生む。whole-path Referenceの価格はfull-wrapperとして受け取り、旧kernel/shellを再加算しない。既存builderのmodes+1 interfaceも17-qubit Referenceの無変更接続先ではない。

### 5.3 価格不確かさ・shared calibration

M3の旧serial政策では同じcalibration平均を全roundへ再利用し、保存実装は `W=sum_m 2*N_m*q_m`、`SE_total=W*SE_kernel` とする。roundごとに独立calibrationを得たとみなすsqrt(round数)の縮小はしない。shell不確かさがあれば同様に保持する。ただしこのWは旧の軸同N・kernel政策の係数であり、whole-path Referenceのq依存へ無条件に流用しない。[M3 shared calibration][M3]

whole-wrapper価格やproxyで校正を共有するなら、再利用した同じparameter／sampleの共分散を保持して全和へ伝える。方式間paired差は同じordered requestに対する差から評価し、X/Yや同じ保存pathの再使用を独立標本数の増加と数えない。quantum failure alphaとclassical価格区間の信頼度は別に記録する。

既存 `bounded_cost_intervals` はanalytic／exhaustive等で正当化したsupportと同時failure配分がある場合の候補で、観測min/maxはsupportでない。保存SE、Welch／tの名目的区間、Stage Aのcohort SDをrigorous boundやvalidated coverageとしない。implemented-lawの価格標本をideal-law期待価格へ移す資格も別に必要である。

### 5.4 Stage A/C2の価格資格とC3からの受取項目

Stage Aはq1／32保存pathで各軸の有限cohort平均、C2はq2の**単一固定fixture価格**である。C2の各X/Y価格はsymmetric→ReferenceでRZ `10773→11114`、CX `4100→3860`、total depth `6096→3014`、qubits `9→17` と保存されている。この引用からp_m母平均・長qの順位・総資源優位を外挿しない。[Stage A][AFIX]／[C2][C2FIX]

| C3から将来受け取るもの | 接続時の確認 |
|---|---|
| 固定SOURCE／結果／STOP／公開commit | 実取得できるものだけ採用。進行中結果や口頭のPASSを固定証拠にしない |
| q4／q8の構成・path／cut／χ／actual adjoint | full ordered path、一回gadget、強いsymmetricの共有／融合、scalar／identity、same raw／law |
| correctness record | raw／compiled／paired X/Y、reference leakage、全system入力か特定stateか、norm・tolerance・精度、phase fitなし、実際のcoverage |
| 資源g別のaxis価格 | full-wrapperかkernel/shellか、depth意味論、qubit peak、P0・topology・合成精度、含む／未評価費用 |
| selection・価格統計 | 固定fixtureか対象lawの標本か、nとsample単位、平均／SE／paired差／shared covariance、support・CIの資格 |
| valid q／law domainと不足 | q4/q8 fixture PASSをq16..4096や全law保証に拡大しない。未確認域は明記 |

本書作成ではC3結果を取得・評価していない。B1保存row・統計接続案・decoder条件整理はC3の価格なしで進められた。後続で必要な長q価格coverageが得られなければ、取得済み域の限定費用または未確認として報告し、全Gを出さない。最大qの直接compile・新samplingを自動必須にせず、既存provider／holdout／境界補正を同scopeで使えるか検討し、追加評価が必要な範囲だけ別許可へ含める。

## 6. B3-5：次の最小実装・科学検証計画（未実施）

以下は提案であり、本書の公開許可はこれらの実行許可を含まない。B3完了とStage Bの科学的完了を区別する。

| 段階・担当 | 最小作業と再利用先 | 必要な判断／実行許可・成果 |
|---|---|---|
| 1 ユーザー／GPT | 共通energy精度・成功確率・claim水準、候補13-roundを使うか、beta／alpha配分、ideal条件付き評価とactual実装評価の範囲を選ぶ | 旧値の自動採用なし。重要な研究判断の採択と詳細レビュー開始を混同しない |
| 2 Codex（許可後） | §2の元H専用薄いwrapperとstate-neutral共通部分、元H energy／horizon／軸別failure・必要なら軸別N会計を最小拡張 | 新SOURCE revisionを固定し、旧SOURCE／guard／契約を保全。科学実装の許可を受ける |
| 3 Codex（同じ実装scope内） | 既存人工テストを再利用・必要な差分だけ追加：未知誤差null、DF一回、誤state拒否、ゼロ半径／phase不足、branch／alias境界、軸別failure・scope・coverage | 分子入力を使わないsynthetic regression。人工PASSを科学性能証拠にしない |
| 4 Codex（別の明示科学実行許可後） | 固定B1全rowと採択target／alias証拠を入力に、radius／phaseの必要な再評価とHoeffding十分N、全round／energy／failureの条件付き会計を行う | 保存値の新解析・N算定として許可に明記。新ground solve・DF refit・sampling・compileは含めない最小案 |
| 5 Codex（actual保証に必要な許可範囲で） | implemented-law／IID bridgeと使用domainのcompile／synthesis／Reference補正誤差を同じraw単位へ接続。Cの取得済み資格を利用 | 未確認ならactual laneはnull。仮定scenarioは分離。必要な追加評価だけ対象・SOURCE・入力・scopeを固定 |
| 6 Codex（価格・会計の許可後） | 公開されたC3と将来のlong-q価格coverageを照合し、qualified N×同scope期待価格を資源別に会計 | C3への介入なし。欠落項ゼロなし。全Gの数値評価には別の明示許可が必要 |
| 条件付きB2 Codex（必要時のみ、別許可後） | 保存physical vectorと既存kernel／reference_roundで元H signal pilot、必要なら既存decoderへ接続 | operator十分条件では採択targetを通せない、よりtightなstate依存評価が研究上必要、または実信号に基づくdecoder成功率を明示的に要求した場合に検討 |

**B1が十分ならB2は不要。** B1の不通過も物理的不可能性の証明ではない。B2はstate依存のsignal／radiusを調べる候補であり、law／全path合成bound・alias資格・厳密丸め包囲の不足を自動的に解消するものではない。B2を行う場合も保存physical vectorのH／sector／basis bindingと数値groundの精度資格を確認し、初期案で新ground solveを必須にしない。

再利用する既存人工test候補は [check_track_d_b1_c1.py][B1TEST] のscalar輸送検査、[management/test_research_c_c1_stage3.py][ACCTEST] の誤差単位／未知誤差／state／radius／energy／coverage検査、[test_rpe_four_round_phase_validation.py][DECTEST] のbranch wrap・scalar/vector一致、既存resource accountingの軸別／union／scope検査である。ファイル全体を無差別実行せず、同じファイル内の回路build・samplingを含むtestを今回の許可へ混入しない。今回test実行は0。

解析十分条件によるN・成功条件の資格に、量子shot simulationやMonte Carlo成功率推定は自動必須でない。実際の成功確率を新しく数値評価する場合は別の科学scopeとする。詳細科学レビューも今回実施せず、必要と判断して開始する場合の許可は科学計算と独立である。

将来のheavyな評価は実空きCPU／RAM等と非干渉を確認し、独立round／pathの計算worker、既存cache・matrix-power／state-actionの再利用で適切に高速化する。CPU／RAM／workerへ恣意的上限を設けず、過去runの数字を恒常上限へ転用しない。AI sub-agentは使用せず、共有venv／OS／他job・他利用者の環境を変更しない。

## 7. 明示的なSTOP条件と引渡し

| 条件 | 後続の科学実行での扱い |
|---|---|
| 元H／J/U／sector／law／SOURCE／入力binding不一致 | dependentな新規callを止め、partialを保全。別endpoint・条件に差し替えて続行しない |
| stateモデル偽装、DF二重計上、corrected／raw単位やscalar／χの不一致 | N／energy／費用を出さず技術修正の範囲を明示。既存STOP／固定SOURCEを無断変更しない |
| 未採択target、alias未資格、horizon全row不足、ゼロ半径・phase／failure予算不通過、overflow／underflow | 未資格／不成立を保存。Nをclipしたり配分・precision・roundを無断変更しない。B2へ自動移行しない |
| law／IID bridge・全law実装誤差が未取得 | ideal条件付きとactual未資格を分ける。未知項を0にしてactual成功保証を出さない |
| AB／actual adjoint／vacuum／χ／raw・compiled X/Yのcorrectness不一致 | dependentなcompile／会計へ進まずpartial保全。phase fitting・閾値緩和・無断retryでPASSにしない |
| 公平な強いsymmetric、同scope価格、valid q／law coverage、価格不確かさが不足 | その不足に依存する全費用・優位claimを止める。finite fixtureや点推定として可能な限定報告と区別 |
| 新sampling／新compile／B2／別target／別q／重要な契約変更が必要 | 既存許可scope外ならユーザーへ戻す。旧one-shot・retry/resume・mandatory STOPを別namespaceで迂回しない |
| 実OOM・入力破損・自process異常等で継続不能 | 自分のjob内で証拠を保全。一時的CPU競合・PSI上昇・他worktreeの通常更新だけで正常科学workerを不要停止しない |
| 許可scope完了またはmandatory STOP | 成果物の許可された公開・照合を終え停止。次の科学段階・詳細レビューを自動起動しない |

今回の変更は本書とPROJECT_MAP／docs索引の案内だけである。科学結果・研究上の採択は変えていないため、科学manifest・VALIDATION_STATUS・過去の研究概要や結論は更新しない。文書のUTF-8／参照先／diffを技術確認し、専用branchへcommit/push、remote SHAと公開文書の内容・bytesを照合する。公開照合の実測は最終チャットで報告する。

**公開・照合後mandatory STOP。次担当はユーザー／GPT**：共通target・誤差配分・claim水準と次の実装／科学実行範囲を判断する。**Codexはその明示許可後**、元H用最小接続と必要な条件付きN評価へ進む。本書だけでは新規N、B2、sampling、compile、全RPE総費用、C3介入、詳細科学レビューを開始しない。承認は対象を明示したユーザーの許可そのものであり、新grant・承認証明・承認ledgerを作る必要はない。

[B1FIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3913866e6ade4ab97ac02645fed91b2d5fcfa75/docs/research/track_d_b1_c1_validation_20261106/RESULT.md
[BCFIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f94972adf5a10e96e9400d3919c00d14e7a718a/docs/research/track_d_bc_static_preparation_20261106/REPORT.md
[AFIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0/docs/research/track_d_h4_paired32_reference_20261106/RESULT.md
[C2FIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8fe5091c42ec786688b6931105ad64d10c87d18/docs/research/track_d_c2_q2_compile_20261106/RESULT.md
[B1JSON]: ../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/B1/result.json
[B1SOURCE]: ../../../artifacts/track_d_b1_c1_validation/2026-11-06/run01/SOURCE.json
[CDFREPORT]: ../research_c_near_practical_r32_downstream_execution_20261103.md
[CONTRACT]: ../research_c_near_practical_r32_downstream_contract_20261103.json
[CDFSOURCE]: ../../../artifacts/research_c_near_practical_r32/2026-11-03/run01/SOURCE.json
[CDFREF]: ../../../artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/B/references.json
[TRANSPORT]: ../../../src/trottertracks/track_d_b1_c1.py
[ACCURACY]: ../../../src/trottertracks/c1_stage3_accuracy.py
[REFERENCE]: ../../../src/trottertracks/c1_stage3_reference.py
[LANES]: ../../../src/trottertracks/c_pr_aware_df_stage_c.py
[RPE]: ../../../src/trotterlib/rpe_resource_accounting.py
[DECODER]: ../../../src/trotterlib/rpe_four_round_phase_validation.py
[HORIZON]: ../../../src/trotterlib/rpe_target_round_horizon_validation.py
[GROUND]: ../../../src/trottertracks/c_pr_aware_df_stage_b_runner.py
[CONNECTION]: ../../../src/trottertracks/c1_stage3_connection.py
[M3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4df8eca29c59ffd55e2dd898f05da16bb246ddbe/src/trottertracks/c_pr_aware_df_m3.py
[METHOD]: ../研究方法・解析手順.md
[SIGNAL]: ../track_d_reference_control_feasibility_20261106/SIGNAL_PRESERVATION_MAP.md
[LEGACY]: ../rpe_18round_measurement_connection_20261103/IMPLEMENTATION_CONNECTION_REPORT.md
[R32SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4df8eca29c59ffd55e2dd898f05da16bb246ddbe/src/trottertracks/c_near_practical_r32.py
[R32DOWN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4df8eca29c59ffd55e2dd898f05da16bb246ddbe/src/trottertracks/c_update064_r32_downstream.py
[RETUNING]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4df8eca29c59ffd55e2dd898f05da16bb246ddbe/src/trottertracks/c_rte_r_retuning.py
[PHYSICAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3649b75af191916daf44abb91e14d34c43b83e8f/artifacts/research_c_near_practical_r32/2026-11-03/run01/shared/physical_ground.json
[B1TEST]: ../../../tests/check_track_d_b1_c1.py
[ACCTEST]: ../../../tests/management/test_research_c_c1_stage3.py
[DECTEST]: ../../../tests/test_rpe_four_round_phase_validation.py
