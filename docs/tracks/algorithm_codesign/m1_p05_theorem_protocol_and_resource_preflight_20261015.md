# M1 P0.5：既存定理・controlled protocol・資源記録の追加照合

2026-10-15 JST。**P0.5の資料照合は完了、M1採択は保留、mandatory STOP。** 新しい数学的証明・科学計算・費用集計は行っていない。

## 1. 範囲とP0からの追加情報

基点はP0 commit `fa91ffb3d4880d767e55f0f661e8ade294b4c9fe`。[P0 Decision Packet][P0]と、その[文献locator][LIT]・[evidence index][P0INDEX]を再利用した。共通指示は[2026-10-12版][COMMON]、研究候補の位置付けは[既存GPTレビュー][REVIEW]による。P0の9一次PDFと21 GitHub資料を再取得・再監査せず、cacheの該当箇所を読んだ。

追加価値は次の三点に限定する。

1. **定理を使う際の入力条件を具体化した。** Guの隔離距離はchannel固有値間の距離、S8のnormはchannelのinduced trace normである。LiのSec4はIID unitary stepと状態上のmean consistency、register-QPEの集中条件を置く。いずれもM1の平均振幅へ無条件に移せない。FRについては未完の候補契約だけでなく、既存の定理別レビューの判定へ到達した。
2. **H6の実行sourceを一段深く追った。** P0の`ax2a_native_df.py`は実装系譜の参照。H6 portの直接参照は`ax2b_native_df_v5.py`である。後者でもordinary tail／決定論部のdirectional loweringが明示され、これを独立／共有±qDRIFTの実装済み証拠とは扱えない。
3. **費用内訳の保存有無をfield単位で区別した。** 旧screeningには`g_det/g_rand/g_total`がある。一方、確認したH6 resource map・wrapper CSVと旧H4長round記録には、同じ意味の決定論／乱択のnative費用内訳は保存されていない。既存のevent数proxyやround割合を費用割合へ変換していない。

追加で参照した固定GitHub資料12件のpath・bytes・SHA256・remote照合結果は[追加参照索引](m1_p05_reference_index_20261015.json)。原データを複製せず、参照先のschemaと数学的対象を整理した資料である。P0の結論・既存classificationは変更しない。

## 2. 既存保証とM1の必要保証の対応

以下のページは**PDFの1始まり番号**。著者・公開日・取得版・PDF hashはP0 locatorを正本とし、ここでは定理の適用条件を補う。「M1への追加事項」は未実施の適用証明または対象の定義であり、新しい定理の採択ではない。

| 文献／定理・式 | 原文の対象 | 主要仮定 | 保証される量 | 演算子・channel・signal | 位相／振幅／誤差／confidence | 部分乱択・D blockとの関係 | M1へ移す前に残る事項 | 固定版・箇所 |
|---|---|---|---|---|---|---|---|---|
| PR：Eq24、Lemma A.1／A14–A17 | 完全qDRIFTのHadamard信号 | λ=1に規格化、確率分解、符号付きPauli/involution、独立step、等角回転 | meanは規格化された`(I−iτH)^r`。H固有basisで減衰とarctan位相を明示 | `E[U]`とscalar信号 | 位相・振幅の正確な式。単独では統計confidenceでない | 非可換Dを挟まないfull-qDRIFT | Dとのinterleaving、実際の±branch、反復roundの対象信号への対応 | [2503.05647v2][PR] pp6–8,20 |
| PR：Lemma A.3／A32–A41 | partial **RTE**のHadamard平均 | `H=Σ H_l+H_R`、H_RのPauli確率分解、順序を保つPF、各RTE occurrenceの独立draw、論文のexact RTE平均・B | `E[BZ]=<ψ\|S_p(δ)^s\|ψ>`。D／R primitive数も記述 | coherent first moment | 既定PFに対するunbiased numeratorと減衰。PF biasは別 | H_Rを一項とするPFにDが入る | qDRIFTへ置換しても同じ式とはしていない。有限cutoff・±lawを含めた対象が必要 | [PR][PR] pp22–24 |
| PR：ThmB.2／B13–B29 | guiding stateからのsingle-ancilla RPE | ground overlap `c0≥η>4−2√3`、規格化H、RPEの位相branch、指定round/depthとshot規則。qDRIFT証明のB18–B21ではenergyの絶対値に関する便宜上の仮定も記述 | exact／RTE／qDRIFTの**RMS energy error**と最大・総rotation数。qDRIFTはinverse-tan補正 | full-qDRIFT/RTEのcoherent信号からRPE | damping・overlap・round失敗確率を経由したRMS保証 | partial RTEのPF誤差会計とは別段階。非可換partial-qDRIFT全体の定理として読まない | M1のround signalがこの定理の仮定を満たすか。RMSと高確率保証の選択 | [PR][PR] pp27–29 |
| Hagan–Wiebe：Def3／27、Thm4／28–29、Thm5／31–39 | qDRIFT channel、A＋B composite channel | 正係数norm分解、規格化local term、IID qDRIFT、AのTrotter近似、commutator/norm条件、指定ε domain | diamond距離とoperator-exponential query上界 | `E[UρU†]` | channel error。coherent位相やRPE confidenceを直接保証する出力ではない | Aを決定論、BをqDRIFTにする構成そのものを扱う | ancillaを含むcontrolled問題へのlift、M1の位相精度へ戻す条件・強さ。未証明だが「利用不能」とも判定しない | [2206.06409v3][HW] pp8–9；高次はThm2.1 pp3,13 |
| Composite QDrift：Def3／11–13、14–16 | real-time qDRIFT／Trotter composite | IID label列、A/B partition、内側・外側orderを区別、norm/commutator入力 | 既知channel boundとquery費用、partition設計の根拠 | channel。imaginary-timeは別のmap/norm | real-time diamond誤差等。universal gate setの厳密T費用でない | inner/outer symmetrizationとD/R選択は既知 | M1のcoherent signal、branch sharing、energy位相保証との対応 | [2306.16572v1][CQP] pp6–7。imaginary-time/localの定理をM1へ混ぜない |
| 富士通2026：10–18、A3–A6、B1–B5 | qDRIFT信号、partial randomizationとSTAR/RPE資源モデル | 規格化Pauli分解、guiding overlapとRPE規則、対称PF、Trotter係数による誤差モデル。B1/B2では経験的C_gs利用も明記 | full-qDRIFTの信号式、RPEの誤差・費用モデル、D/R別rotation数 | coherent信号／RPE／費用式を分けて記述 | A6はmean-square phase、B1はenergy誤差quadrature。B5のγは費用係数 | partial qDRIFTを費用式で扱い、γ=1、RTE/TE-PAIは2 | 既採用の費用式と、M1の非可換平均・±law・spectral weightの保証が同じ仮定かを確認する適用論点 | [2603.22778v2][FUJ] pp4–6,21–22 |
| Günther博士論文：Ch2 §§2.4/2.5/2.9、§2.10 | 初期PRのqDRIFT／RTE／±制御と、その後の費用修正 | 再録部分の規約と後記修正を区別 | full-qDRIFT信号、partial RTE平均、独立±乱数、HWP等の修正記録 | coherent信号とモデル費用 | 原論文の別版履歴。独立再現でない | p85はU/U†へ独立RTE。p119は外部FeMocoの全round費用表 | thesis／PR v2の式を混ぜず、M1への追加保証を特定する。表の割合をTrack Aへ移送しない | [公式PDF][THESIS] title June2025、番号付き版なし；pp59–66,77–85,116–119 |
| Gu：Thm1、SI S1–S8／S9–S16 | ideal unitary channelにnoiseを後置した固有位相、測定信号のspectral展開 | CPTP noiseのHermitian Kraus、weak perturbation、SIのnoisy map diagonalizability。S7は**異なるchannel固有値**間のisolation、S8はinduced trace norm条件 | 一次固有位相補正なし。縮退subspaceもSIで扱う。左右固有operatorによる信号展開 | Hilbert–Schmidt operator空間上のsuperoperator | 一次位相／振幅、摂動収束条件。M1のfinite-step RPE成功保証そのものではない | D blockをinterleaveする平均振幅を直接対象にしていない | M1をこのnoise配置・norm・スペクトル条件に対応させられるか。単純固有値だけでなく縮退との差も必要 | [2208.04100v2][GU] p2, SI pp7–9 |
| Li：Thm2.2/3.1、Sec4／43–56、Thm4.1 | residual/gap付きQPE、random-unitary register-QPE | Sec4はIID unitary W_j。43は状態上のone-step mean consistency、q>1；44は独立性；46のQ_kはr·2^k個のproduct。49はgap、53–56は集中とregister数の条件 | 位相精度2^(−n)、成功確率1−ε、rとregister数tの許容範囲 | mean/random productとregister-QPE状態 | mean誤差・gap・成功確率。single-ancilla RPE定理ではない | §4は支配項を決定論にする一般化へ言及するが、M1のD/±仕様ではない | meanのspectral表現・状態overlap・正規化・random queryの依存をM1で満たすか。平均振幅をunitaryと仮定して転記しない | [2111.10430v3][LI] pp8,11,12–16 |
| 既存FR：候補C1/C2、T1–T4の**既存レビュー** | finite paired-Taylorの平均振幅、同情報でのphase/radius certificate | 独立occurrence、正scalar、順序保持、signal floor、情報層／許容クラス。一般Qとfinite-RTE実現集合を区別 | 既存レビューはT1/T2/T4の具体化をdirect corollary、T3をunresolvedと整理。数値gateは変更なし | `E[U_ω]`、状態matrix element | phase/radius。一般のqDRIFT/RPE energy保証でない | 参照は同じouter PF/split。outer PF biasは別会計 | 正scalar分離・位相と減衰の区別だけを新成果にしない。M1固有の対象と追加保証が必要 | [FRレビュー][FRREVIEW] §§2,4–6；[候補契約][FR] §§2–6 |

### 適用時に誤読しやすい三箇所

- **Gu S7/S8**：isolationは`e^{i(λ_a−λ_b)}`どうしの距離で、bare Hamiltonianのground gapではない。S8は`||E−I||_(1→1) < r_ab/2`というchannel条件。平均振幅行列のspectral normへ置き換える証明は今回行っていない。S11–S16は左右固有operatorの重みも記述しており、non-normalなspectral weightを扱う一般論自体をM1の新発見とはしない。
- **Li 43/44/46/56**：43の評価対象は`(E[W]−U^(1/r))|ψ>`、44はmean product、46はcontrolled-power queryの構成、56はr・register数と成功保証の条件である。M1の固定平均行列を反復する信号へ、この四つを一括転用できるという適用証明は未実施。本文のspectral展開があることだけから、任意の非normal平均に対する保証まで確認済みとはしない。
- **PR B2**：full-qDRIFTのenergy位相補正とoverlap付きRPEは既知。partial RTEのA41とも、富士通のγ付き資源式とも同一statementではない。B18のround選択とp29の別depth選択の説明も別であり、都合のよい箇所だけを組み合わせたM1保証を作っていない。

これらは適用条件の整理であって、論文の誤りの判定でも、M1へ適用できないという証明でもない。同じ定理が確認箇所で未特定であることから、新規性は認定しない。

## 3. Controlled protocolの仕様照合

一般の二branch Hadamard信号`<ψ|B0† B1|ψ>`はP0と既存レビューの照合規約を再利用する。以下の式は既存資料にある対象を示すもので、新しいM1 first-momentの導出ではない。

| protocol | controlled演算子 | 乱数条件 | 順序・逆演算・時間符号 | controlled relative phase | Hadamard信号と既存平均 | 対応source／資料 | 未決定・未証明 |
|---|---|---|---|---|---|---|---|
| ordinary | `diag(I,U_ω)` | 一つの列。shot間IIDの保証は別の実行条件 | sourceが受け取るordered U。負branchの生成なし | Uのsystem global phaseもbranch間では保存必須 | `Z=<ψ\|U_ω\|ψ>`、平均は`<ψ\|E U_ω\|ψ>` | [RPE wrapper][WRAP] `build/_build_validated`、[仕様][WRAPDOC] | builderはresamplingしない。fresh-shot IIDを単体で検証した意味ではない。M1用lawは未固定 |
| PRの独立± | ideal blockは`diag(U†,U)`。A46でbranch別ordinary controlに分けてRTE | U側／U†側を**独立**にdraw | exact symmetric PFではA45。random blockのU†側は正しい逆時間law。一般random wordの同順sign flipとは別 | A44/A46の相対位相を保持。projective一致だけでは不可 | paperはRTEの正確な平均と、対称PFの`<ψ\|S_p(δ)^(2s)\|ψ>`を扱う。BとPF誤差は別 | [PR][PR] p24 A43–A46、p23 A41；[thesis][THESIS] p85 | qDRIFTに替えたM1のinterleaved平均・物理時間・RPE保証は別に必要。random費用はpaperの半減対象でない |
| shared full-block ± | `diag(U_ω†,U_ω)` | 同じfull wordと**literal adjoint** | 負branchは積全体の逆順／各gateのinverse | 二branchの相対phaseもinverseに対応させる | 既存レビューが指摘した`E[U_ω²]`型。`(E U_ω)²`とは同一視不可 | [既存GPTレビュー][REVIEW] §8.2、[P0][P0] §4 | M1として正式採用されていない。multi-blockの共有範囲、物理時間、平均保証は未固定 |
| gate単位のshared label | 各gateに`C± U_j`を使う案 | gateごとにlabelを共有。全体のadjoint共有とは別 | 各gateの符号を反転しても、同順productは一般にfull-word adjointでない。A45は対称性条件付き | gate phase／scalar phaseもbranch別に維持 | 実際のB0†B1が対象。一般random列の共通mean式は既存資料から確定しない | [PR][PR] A42–A45と直後の注意、[P0][P0] §4 | 対称PFの時間半減を任意random wordへ使えない。M1の生成順序・共有規則は未採択 |

### H6の`directional`が意味すること

今回新たに追跡した固定sourceはTrack A commit `0dd5c3d204dab9b731268a1aa2add621080f3d20`の次の箇所である。

- [h6_matched_port_v1.py][H6PORT] L28、L298–302は`ax2b_native_df_v5`から`partial_native_from_step_requests`等を呼ぶ。L305–321には既存のbranch probeとX/Y確認の処理がある。今回は**読んだだけ**で実行していない。
- [ax2a_control_plan.py][H6CONTROL]の`ControlStage.branch_time`はUNCONTROLLED=t/t、DIRECTIONAL=−t/+t、ORDINARY=0/tを定義する。`partial_s2_control_plan`はforward uncontrolled、central ordinary tail、reverse directionalを指定する。
- [ax2b_native_df_v5.py][H6NATIVE] L94–123はdirectional diagonalとbasis変換、ancillaの`RZ(2*global_phase)`を保持する。L169–239はsupplied tailを`diag(I,U_event_sequence)`として扱い、explicit requestをreplayする。最後のscalarはancillaへのphaseとして残る。
- [h6_matched_memory_v3.py][H6MEMORY]はowned wrapperへH／S†／測定を付け、whole wrapperをcompileしてmetricを保存する経路である。sourceには回路内容を変えない所有権管理と記載されており、今回は新たな同値検証・compileをしていない。

したがって、P0の祖先builder参照を直接のH6 port/v5参照へ補強できた。**決定論部のdirectional loweringは、乱択tailの二つの独立drawやfull-word共有±を実装したことを意味しない。** ここで分岐の積を新たに証明したり、M1のprotocolを採択したりしていない。

## 4. 既存資源記録の保存状況

費用数値の再計算・割合算出は0。以下は固定fileのfield／sourceの存在確認である。「内訳がない」は対象fileの記録形式についての結論で、repository全履歴を網羅した不在証明ではない。

| 記録 | 保存されている値／場所 | task・protocol・precision／cost model | その記録から未取得のもの／必要な作業の種類 |
|---|---|---|---|
| 旧DF screening | [JSON][SCREEN]の`/candidates/*`と`/best_by_molecule/*`に`g_det,g_rand,g_total`。plotの[helper][PLOT] L2283–2299はそれらからfractionを**その場で計算**する | `df_reduced_screening_cost_minimization_v1`、`randomized_method=qdrift`、`g_rand_input=1.0`、保存`cost_rule`はanalytic D-only RZ-depth、C_gsはanchor流用。旧screeningであり現在の検証結果として使わない | **内訳fieldは存在する**が、M1／H6 compiledの実測比率でない。geometry/basis/δ・controlled protocol・synthesis精度の一致を確認した値でないため割合や費用数値を転載しない。helperも実行しない |
| 旧H4 G08 | [G08][G08] `/candidates/3/rounds/*`に`compiled_rz_point_estimate,cost_fraction`。上流[compute][G08COMP]には`expected_component_applications_per_short_step,shot_weighted_component_application_proxy`と各axisの`predicted_rz_count_per_interrogation` | H4 linear1Å/STO-3G/8qubit、DF12、L_D3（対照12）、δ=.01/.02、r/K schedule、ordinary Hadamard compiled-RZ proxy。P0引用したδ=.02/r32/K2は特定候補の条件。直接holdoutと長round proxyの範囲は[P0][P0] §5 | `cost_fraction`は**round間**配分。component applicationはevent適用数proxyでnative費用でない。確認したround recordにD/R別RZ費用はない。既存providerのcomponent情報から新しく配分すれば再集計となるため今回は行わない |
| H6 confirmation | [resource map][H6MAP] `/rows/*`は`G_point,N_total,metric,axes,common_preparation_slope,preparation_cost_P`。各axisはmean/min/max/n/SE等。[CSV][H6CSV]のheaderは`cell_id,phase,replica,seed,axis,control,event_digest,compiler_hash,fingerprint,RZ,CX,RZ_depth,CX_depth,depth,size` | [scope正本][H6RESULT]：linear H6/1Å/STO-3G/12 modes/tol-only DF19/cutoff0/sector400/T=.8、prefix5/10/15/full19、δ=.1–.8。選択8cell・32paired trajectory。Qiskit1.3.0、rz/sx/x/cx、opt1/seed17、`symmetric_directional`、P=0、complex-signal誤差の経験的allowance | 保存費用はwrapper全体。**D/R別metric列はない**。RZは任意角native gate数で、Clifford+T合成精度付きT数でない。既存raw eventとwhole-compile metricだけから、cross-boundary最適化後のD/R一意帰属を読み取れない。追加の会計規則／取得が要るかは将来の契約事項 |
| PR論文・thesis | [PR][PR] E21は`G_det`と`G_rand`をprimitive費用として分ける。thesis Table2.2 p119に既存の`G_det/G_tot`がある | PR E.2のcommuting rotations／HWP＋catalyst・Toffoliモデル。thesisはReiher/Li FeMoco外部benchmarkの全round集計であり、Track AのDF-native資料でない | model内訳や外部保存割合はあるが、M1・同じDF入力・同controlled protocolの実測比率ではない。既存割合の補数も計算しない |
| 富士通2026 | [B3–B5][FUJ]でroundごとのdet/random logical rotation会計 | STARの`C_gate=1`、γ=1/2、対称PFのdet半減。Trotter C_gsの経験的モデル・RPE予算を使用 | gate一回のlogical non-Clifford rotationと、angle別Clifford+T T数／CX数は別。γの差を総T改善率に変換しない |

### 記録不足と、新しい科学取得を要する不足

**記録の探索で解消できたもの**は、旧screeningに内訳fieldがあること、H6では全wrapper metricを保存すること、H4のcomponent fieldが費用ではなく適用数であること、H6実装portの直接参照である。

**既存の値を抜くだけでは解消しないもの**は、M1の対象protocol・同精度・同状態・同round scheduleに整合するD/R別の期待総費用である。H6の保存NPZ/eventや別runの単価から新しい費用を構成していない。additive attributionで足りるか、whole-circuit最適化後の因果的帰属を求めるかで必要な情報が異なるため、直ちに「再compileが必須」とも決めない。

| 付帯費用 | 確認できる扱い | 未確定の点 |
|---|---|---|
| state preparation | H6は`preparation_cost_P=0`で除外し、共通Pの係数・固定比較の感度を保存。RPE wrapper自身もstate preparationを生成しない | 実際のstate preparation費用を0と測定した意味でない。M1条件の値は未取得 |
| readout | H6 owned wrapperはancilla H/S†と測定命令を含む。G08/H6のscopeは理想的なgate metric／signal accounting | 物理readout時間、readout error/reset、hardware shot overheadは別で、今回対応する測定費用は未取得 |
| classical | H6 sourceはwrapper compileのwall/RSS、summaryは実行resourceを記録。P0やG系列の既存setup/query記録もある | setup・compile時間をquantum shotごとのonline生成費用にしない。M1のclassical online costは`NOT_MEASURED`。全研究で古典時間が一切無いという意味ではない |

## 5. P0で完了しており、繰り返していないこと

qDRIFTの既知の位相写像、Composite構成、PRの独立RTE ±、富士通γ、既知FRとの重複、H4のlate-round集中、H6の有限時間とRPEの違いは[P0][P0] §§3–6を参照する。文献著者・日付・版・取得hash、GPTレビュー原本identityは既存indexへ戻る。9論文の再検索・再取得、旧97項目の文書監査、FR/G/D/S4の数値監査は再実行していない。

今回新たに参照した[FR定理別レビュー][FRREVIEW]は`TECHNICAL_NOTE`を推奨した既存判断である。旧[候補契約][FR]に残るOPEN／未完表示を、そのまま現在の独立新規性の証拠にはしない。一方、finite-RTE許容クラスの未解決項目をM1が解決したとも扱わない。

## 6. GPTの採択判断に残る事項とG0の境界

| 採択論点 | P0.5までで使える既存事実 | なおGPTに委ねる判断／採択された場合の数学的課題 |
|---|---|---|
| D1：新規性 | full-qDRIFTの位相/RPE、Composite、Hermitian-noise一次位相安定性、FRのscalar/phase/radiusは既知 | 何を既存定理の直接適用以上のstatementとするか。M1を新アルゴリズム、新保証、既知法の応用のどれとして検討する価値があるか |
| D2：対象protocol | ordinary／独立±／shared full-word／shared gateは異なる。H6 directionalはordinary tail | どれを対象とするか、積順序・branchごとの時間・sign/phase、round間の実効周波数、物理時間、初期状態、信号・誤差量を正式に固定する必要がある |
| D3：数学的実現可能性 | Gu/Li/PRの前提と出力を上表で区別した | M1のmeanに対し、既存のnormal/diagonalizable条件、gap/isolation、left/right weights、signal floor、overlap、alias-free/phase-unwrapping条件をどう満たすか。これらの適用証明や新boundは今回未実施 |
| D4：資源条件 | HWP/Toffoli、STAR logical rotations、native RZ/CX、T合成価格は別会計。選定したH6/H4表に整合するD/R費用比率はない | 対象taskで何をcost coordinateとし、どの情報を取得する必要があるか。γやlate-round比率だけで総費用改善を採択しない |
| D5：G0との境界 | 一次文献の式・仕様・保存fieldの確認まではP0/P0.5 | **採択する場合だけ**、protocolのmeanへの定理適用、phase bias／overlap／RPE成功条件の証明義務を固定する。G0の担当・契約・新証明は今回開始していない |

これは必要な判断の索引で、具体的な新protocol・定理・実験条件を提案契約として確定したものではない。未特定の定理を不存在としたり、M1の成立性・資源優位・採択可否をCodexが認定したりしない。

## 7. 完了範囲と停止

実施したのは、cache済み文献の限定読解、既存sourceの静的参照、JSON/CSVのfield確認、引用索引と本Markdownの作成、取得bytes/SHA256照合だけである。検証は資料の整合性の範囲であり、理論・数値の独立再現ではない。

**新証明、M1採択、G0、数値simulation、行列評価、operator guard、sampling、trajectory、compile、synthesis、LP、新しい費用・比率・shot forecastはすべて0。** 科学source・旧contract/result/classification・authorization・marker・STOPを変更しない。親worktreeのHEAD・index・既存dirty fileを保護し、公開は専用branchの新規整理資料だけとする。

**次は、利用者が明示的に開始を承認した場合のGPTによるM1採択判断。資料公開だけをレビュー開始・科学実行の承認と扱わず、mandatory STOPを維持する。**

## 固定参照

[P0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa91ffb3d4880d767e55f0f661e8ade294b4c9fe/docs/tracks/algorithm_codesign/m1_p0_prior_art_control_and_resource_decision_packet_20261015.md
[LIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa91ffb3d4880d767e55f0f661e8ade294b4c9fe/artifacts/track_b_m1_p0_preflight/2026-10-15/v1/primary_literature_locator.csv
[P0INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa91ffb3d4880d767e55f0f661e8ade294b4c9fe/artifacts/track_b_m1_p0_preflight/2026-10-15/v1/evidence_index.json
[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa91ffb3d4880d767e55f0f661e8ade294b4c9fe/docs/research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md
[REVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa91ffb3d4880d767e55f0f661e8ade294b4c9fe/docs/research/track_b_next_direction_gpt_scientific_review_20261015.md
[PR]: https://arxiv.org/pdf/2503.05647v2
[HW]: https://arxiv.org/pdf/2206.06409v3
[CQP]: https://arxiv.org/pdf/2306.16572v1
[FUJ]: https://arxiv.org/pdf/2603.22778v2
[THESIS]: https://www.math.ku.dk/bibliotek/arkivet/phd-theses/2025/phd25jg.pdf
[GU]: https://arxiv.org/pdf/2208.04100v2
[LI]: https://arxiv.org/pdf/2111.10430v3
[FR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/fr_research_claim_and_manuscript.md
[FRREVIEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/tracks/algorithm_codesign/inputs/fr_theorem_prior_art_review_71169d8.md
[WRAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/src/trotterlib/rpe_hadamard_interrogation.py
[WRAPDOC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/rpe_hadamard_interrogation.md
[H6PORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/src/trottertracks/resource_applicability/h6_matched_port_v1.py
[H6CONTROL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/src/trottertracks/resource_applicability/ax2a_control_plan.py
[H6NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/src/trottertracks/resource_applicability/ax2b_native_df_v5.py
[H6MEMORY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/src/trottertracks/resource_applicability/h6_matched_memory_v3.py
[H6MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/confirmation_resource_map_v1.json
[H6CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/wrapper_metrics_v1.csv
[H6RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/result_summary_v1.json
[G08]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/artifacts/research_direction_round_dominance/2026-09-22/g08_round_cost_risk_proxy_dominance_v1.json
[G08COMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_full_scope_optimization_compute_v2.json
[SCREEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json
[PLOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/src/trotterlib/df_cost_plotting.py
