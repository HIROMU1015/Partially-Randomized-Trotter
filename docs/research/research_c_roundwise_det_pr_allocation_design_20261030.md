# 研究C：Round別・決定論PF／PR割当の共通energy接続と最小検証

2026-10-30 JST。起点`0cb2e5ea986320283465ea51e5b273d0b3313b96`、独立branch `codex/research-c-roundwise-det-pr-design-20261030`。

**技術判断：共通energyへの接続は、以下の十分条件の下で成立する。保存値だけの189-call検証を一件準備した。** 既存RPE復元器・accuracy/Hoeffding・serial会計を変更せず再利用する。混合方式のH4での性能GO／dominanceは未判定。今回は保存値による新しいround価格表、割当、shots、総費用を一切算出していない。

人工新42＋回帰75＝**117 distinct PASS**。旧M2/M4 SOURCE・契約・raw・grant・ledger・marker・STOPは不変。実H4の数値配列/QPY decode、PF/RTE/operator/ground、sampling、build/compile、fit、科学的な保存値再解析はすべて**0**。Passive preflightのみ実行し、科学run01は作成していない。正式共通指示2026-10-25版§8.1/8.2/8.4/10/10.1/10.3を実ファイルで確認した。承認証明・承認原文保存・新grantは不要であり追加していない。

## 1. 固定対象・既存研究との重複：Q7/Q8

保存直鎖H4・1.00Å・STO-3G、Nα=Nβ=2/sector36、**同一の保存PR-aware number CDF8 γ=.1 J/U**。元g補正κ、核scalar一回、spin規約・frame順0..7、同じDF ground oracle/状態準備費用除外を維持する。

| 項目 | 決定論D | PR |
|---|---|---|
| 基準 | 保存M2 Proposed | 保存M4 Proposed＋M4-native |
| PF | 全frame ordered symmetric S2、s1 | LD2、partial ordered symmetric S2、s1/r128/K0 |
| 時間 | δ=h=.1、n=2^m、t_m=.1·2^m、m0..12 | 同じ |
| 推定対象 | 同じH_DFの同じground energy E_DF | 同じ。別PF固有値へ置換しない |
| CA/配分 | .00159 Ha、DF reserve CA/8、βtotalπ/6、βstatπ/12、αaxis=.01/26、αtotal=.01 | 同じ |
| Native費用 | 保存固定compiled kernel・共通shell | 保存32 trajectoryのkernel平均/SE・同じshell |
| Compiler | P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false、既存precision/declared phase | 同じprofile。各方法の既存合法融合scopeを保つ |

保存Proposed J/U SHA256は`f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`、input snapshotは`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、元P-H4来歴は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。Ground/reference NPZも両契約で同じpath/hashを持つ。今回はhashとschema/metadataだけ確認し、配列を復元していない。

| 必須報告・既存検証 | 再利用と今回との差分 |
|---|---|
| [直前native構造調査](research_c_pr_native_structure_optimization_design_20261030.md) | Native追加実験はNO-GO提案。今回はnativeを変更せず、共通energyのround割当を設計する |
| [M2 同J/U強決定論](research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md) | 全frame S2のPF、round合成pilot、kernel/shell、ground/aliasの保存証拠。旧numeric synthesis資格をideal laneと混同しない |
| [M3 s1/r64修正後](research_c_pr_aware_df_M3_s1_r64_repair_execution_20261028.md) | s4/r16→s1/r64のhead反復/減衰交換は既存。再度r/sを探索しない |
| [M4 feasibility/損益分岐](research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md) | PF/RTE/A、13round、2仮定laneのshots、固定accuracy。高m shots置換counterfactualも既済で、新しい混合方式の成果と呼ばない |
| [M4-native128](research_c_pr_aware_df_M4_native128_execution_20261028.md) | Complete kernelの有限32校正・SE・共通shell。全law supportやideal-law価格への保証は移送しない |
| [性能・機構整理](research_c_pr_aware_df_performance_mechanism_readiness_20261028.md) | 原因帰属と会計寄与を分離し、強baselineと同条件で比較する。λ単独・小rank・native局所gate数から総資源を予測しない |
| [OPT-01/OP-III2](pr_algorithm_improvement_landscape_20261023/PR_IMPROVEMENT_OPPORTUNITIES.md)・[WP11固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_wp11_synthesis.md) | Round別r/K・β/α最適化は既存。本件は固定2方法・固定配分の、共通energy bridgeと保存証拠の適用比較 |
| [Track A/AX-1b一次資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/execution_audit_v1/result_review_report.md) | Oracleとoperational、条件付き選択と外挿、finite-time signalとenergy精度の区分を再利用。旧rank12/T=.8の性能や許可は移送しない |
| [Track B Parent allocation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md) | 既知Euler pairing、mean保持、cost-aware IS、normalizer/価格の交換は既存。本件はsampling law自体を変えない |

Branchにない資料は上記の実在する固定commitから確認した。全branchの網羅監査や、新たなGPT詳細科学レビューではない。**Round別の手法選択、独立最小化、nearest-branch復元を新アルゴリズムと主張しない。**

## 2. 共通complex signalへの十分条件：Q1/Q2

理想的なHermitian H_DFの正規化ground state |ψ_DF⟩と、同じE_DF・同じδを固定する。θ=−E_DF δ、q_m=2^mとする。二つのPFが同じ近似unitaryや同じ有効固有値を持つ必要はない。

決定論では、既存のround全体のoperator誤差を

`εPF,m,D = ||U_D(h)^q_m − exp(−i H_DF t_m)||₂`

とする。PRでは、理想finite-RTE IID lawにおける一kernel平均が正のscalar bと補正operator Fの積であり、**coherent occurrenceごとに独立なtrajectory**を使うと、平均round operatorは`b^q_m F^q_m`となる。従ってA_m,PR=b^q_m>0で、既存referenceが評価する`||F^q_m−U_PF(h)^q_m||₂`をεRTEとすれば、

\[
|z_{m,a}-A_{m,a}e^{iq_m\theta}|\le
d_{m,a}:=A_{m,a}(\epsilon_{\rm PF,m,a}+\epsilon_{\rm RTE,m,a})
+\epsilon_{\rm syn,m,a}+\epsilon_{\rm law,m,a}.
\]

DはA=1、RTE/law非適用=0という定義である。**PR r=0からDを生成するのではない。** PRのAは正の実数として振幅を縮め、energy位相を変更しない。Synthesis/lawはraw complex-signal単位の誤差。PF/RTEは既にround全体の誤差なので、ここでq_mをもう一度掛けない。M2の合成誤差も既にcoherent反復を伝播したround量である。

`d<A`なら、円板の幾何から`|arg z−q_mθ|circ ≤ asin(d/A)`、かつ`|z|≥ρ=A−d>0`。X/Yの標本平均が各座標η以内ならL2誤差は√2η以内。既存η=`ρ sin(βstat)/√2`を用いると、追加位相誤差はβstat以下になる。既存Hoeffding実装の`ceil(2 log(2/αaxis)/η²)`は±1観測量の各軸の十分shotsであり、普遍的な必要最小shotsではない。

従って、各roundで`asin(d/A)+βstat ≤ βtotal`を満たせば、観測位相は**同じq_mθ**からβtotal以内にある。各shot・各coherent PR occurrenceでの独立性は条件として残す。固定32校正回路の繰返しをこのIID法則の代わりにしない。

### Nearest-branch復元が方法切替を許す理由

[実際の復元器](../../src/trotterlib/rpe_four_round_phase_validation.py) `reconstruct_rpe_phase`は、q=1,2,4,…を要求し、前round推定値に円周距離で最も近い候補枝を選ぶ。入力は各roundの位相だけで、methodや同一回路という条件は参照しない。

前roundのbase-phase誤差がβ/q_(m−1)、今回の正しい枝の誤差がβ/q_m以内なら、両者の距離は最大`3β/q_m`。候補枝の間隔は`2π/q_m`なので、**3β<π**なら正しい枝が一意に最も近い。帰納法により最終base-phase誤差はβ/2^12以下となる。本件βtotal=π/6はこの条件内である。Roundごとの位相偏りが互いに同じである必要もない。

初期branchは保存alias条件と同じenergy window [−10,10] Haに固定し、E=−θ/δの符号を保つ。δ=.1でこのwindowはbase-phaseの一意範囲内にある。X+iY規約の不一致、別ground、別DF、δ/q gridの変更、alias不成立はSTOPする。最後の1 roundだけのPASSや、別PFの固有位相への置換では十分でない。

この議論は既存decoderに対する直接の十分条件の証明である。RPEの一次資料[Kimmel–Low–Yoder, arXiv v3](https://arxiv.org/pdf/1502.02677)には2021年の訂正が含まれ、枝誤りの判定でπ/(2q)を使う旧解析の問題とπ/(3q)の条件が記載されている。[Erratum](https://doi.org/10.1103/PhysRevA.104.069901)も確認した。旧π/2条件や原論文の資源定数をそのまま採用せず、上の独立した3β<πの議論と既存Hoeffding配分を使う。

### 全13round・物理energyへの戻し

各軸の失敗事象の和は`Σ_m 2αaxis=.01`。Union boundにround間独立は不要だが、各軸のHoeffdingに必要なper-shot IID条件は維持する。実行前に固定したscheduleに対する条件であり、複数候補の同時保証ではない。

同じsectorの物理HamiltonianとのDF operator誤差からWeylのenergy差を一度だけ加え、

\[
|\widehat E-E_{\rm phys}|\le \epsilon_{\rm DF}
+\frac{\beta_{\rm total}}{\delta\,2^{12}}.
\]

DF[Ha]をraw-signal誤差へ足さず、13回計上しない。この数学は誤差上界等の前提に条件付けたもの。保存binary64残差・engineering guardを厳密enclosureへ昇格せず、κの新たな対称化やpolar U置換も行わない。実H4については従来通り数値pilotとoracle仮定の範囲に限る。

## 3. Identity・既存コードへの最小接続：Q3/Q7

新SOURCEは[純scalar adapter](../../src/trottertracks/c_roundwise_det_pr.py)、[保存値専用接続](../../src/trottertracks/c_roundwise_det_pr_saved.py)、[明示実行CLI](../../scripts/run_research_c_roundwise_det_pr.py)、[人工test](../../tests/management/test_research_c_roundwise_det_pr.py)の4件。旧科学コードは一切変更していない。

共通energy IDは、元input/JU/ground/reference/物理inputのhash、geometry/sector、frame順、κ/scalar/spin、oracle/位相/δの一致から構成する。さらにM2/M4/nativeの固定SOURCE・契約・STOP・compilerとground/aliasを照合する。**旧target_idの文字列置換だけを同一性の根拠にしない。**

| Identity | 保存・接続の方法 |
|---|---|
| Energy | `same_saved_DF_energy:<binding digest>`。全round共通のH_DF/ground/phaseを示す |
| Method | 各roundに`D`または`PR`。同じroundのX/Yは同じ選択 |
| Evidence | 旧SOURCE/契約、結果file/m、旧`source_target_id`、lane、実際に使用したsyn/law仮定を併記 |
| Cost | 同じenergy ID＋method/m/axis＋scope。Kernel、shell、snapshotの出典は元bundleに残す |

M2の旧IDは`D_proposed`、M4は`proposed:<JU hash>`。両者を消さずsource_target_idとして保存し、共通入力をbindした後で新しい`RoundEvidence`を作る。旧M4のunknown-error行にはtarget_idがないschemaも、固定SOURCEのpacket来歴から扱い、旧行へIDを挿入しない。

[AccuracyTarget/RoundEvidence/assess_round/assess_energy/aggregate_totals](../../src/trottertracks/c1_stage3_accuracy.py)、[compose_serial_cost](../../src/trottertracks/c1_stage3_connection.py)、[既存RPE/Hoeffding](../../src/trotterlib/rpe_resource_accounting.py)、[horizon](../../src/trotterlib/rpe_target_round_horizon_validation.py)をそのまま呼ぶ。全round同一targetの検査は緩めない。PRの再接続後shots/statusは保存M4と一致することも検査し、異なればSTOPする。

[既存short-round optimizer](../../src/trotterlib/rpe_short_round_optimization.py)は直接construct可能なq≤4、DFPartialS2Preparationと(r,K)候補を扱う。非零tailでr=0を拒否し、deterministic-only tailでのみcanonical r=K=0を認める。本件の全frame S2と非零LD2 tailを、そのr=0で同一視できない。旧optimizerを拡張・再実行せず、同じ独立最小化の原理を、保存scalarと明示的なD/PRラベルへ適用する薄いselectorを追加した。

PF/RTE/CAR/SectorCAR/native処理は既存Stage B/M2/M3/M4/reference/moleculeコードと誤差定義を確認したが、今回および次の189-callで再計算しない。新fitter/simulator/compiler/RPE decoderはない。

## 4. 公平なlaneと費用・分離可能性：Q4/Q5/Q6

保存M2はPF＋full-operator synthesisの数値pilot、保存M4は全law synthesis/implemented-law未資格であり、そのまま同じ保証レベルとはいえない。次の3区分を固定する。

| Lane | D | PR | 結論の限界 |
|---|---|---|---|
| evidence | 保存PF＋保存round synthesis pilot、A=1、RTE/law非適用 | 保存PF/RTE/A、syn/lawはnull | PR未資格を保持。費用最小化・dominanceはUNQUALIFIED |
| ideal_exact_native（主） | PFは保存値、syn=0は明示counterfactual | 保存PF/RTE/A、exact native/ideal finite-RTE IIDの仮定 | 両方法に同じexact-native仮定。実装の未知誤差を0と認定したものではない |
| assumed_raw_signal_budget（感度） | syn≤.001 A_D、law非適用 | 保存M4と同じsyn≤.001 A_PR、law≤.001 A_PR | 同じraw-signal単位の合成仮定。Random-law誤差はPRだけ。新たな保証ではない |

Dのconditional laneでは合成誤差の仮定が旧M2 numeric laneから変わるため、**旧M2のshots/totalを黙って流用しない**。保存PFだけを再利用し、許可後に既存accuracyでscalar shotsを再評価する。PRは旧M4と同条件であることとshots/status一致を確認する。今回その新計算はしていない。

各round・方法の費用は

\[
C_{m,a}=\sum_{b=X,Y}N^{(a)}_{m,b}
\left[2^m\mu_a+S^{(a)}_{m,b}\right].
\]

μはcomplete controlled h-step kernel価格。Kernel内の決定論headはμに含まれ、RPEのancilla head/readoutはSに一回だけ入る。状態準備は除外する。Depthは既存のfenced serial scheduled sumで、全round一括compileのdepthではない。Classical32標本を量子shots倍率にしない。

固定β/α、共通energy/alias、固定13round、round間の価格結合・global compileなし、かつ各選択が共通の最終energy条件を満たすなら、適格集合F_mの直積上で

`min Σ_m C_(m,a_m) = Σ_m min_(a∈F_m) C_(m,a)`。

左辺の任意scheduleは各項の最小値の和以上で、各roundの最小値を選べば等号を達成する。従ってO(13×2)で十分。価格mean/SEを共有することは不確かさを結合するが、**固定点推定目的の分離可能性**を壊さない。β/α再最適化、global compilation、量子観測を見て方法変更する場合は別問題である。

主指標RZで厳密に小さい方法を選び、同値ならD。副5指標はその**同じRZ選択schedule**を評価し、指標別に安いscheduleへ変更しない。Known phase/radius不適格だけを除外する。Unknown/null価格やsyn/law未資格を不適格として捨てたり0にしたりしない。非有限値、target/time/method/価格scope不一致はcorrectness STOP。

全roundでDが適格かつ`C_D≤C_PR`なら（PRが登録十分条件で不適格な場合も含む）、全Dが最小となり`MIXED_ROUND_DOMINATED`。**保存点価格モデル内のNO-GO**であり、母平均の支配性ではない。未知競合があればUNQUALIFIED。少なくとも一部roundでPRが安く、full13 energy/failure条件も成立し得る場合だけ`MIXED_ROUND_POTENTIALLY_USEFUL`とする。All-PR選択とstrict mixedも別表示する。現時点で実H4の各roundがどちらに有利かは判定していない。

## 5. 校正不確かさ・選択と検証

同じPR32標本平均を全roundへ再利用するので、選択したPR roundの`W_PR=Σ_(selected PR m) 2 N_m 2^m`に対して、conditional total SEは`W_PR SE(μ_PR)`。sqrt(13)で縮小しない。All-Dとmixedの差にも同じ共有meanを使うため、差のSEは両者のPR重み差の絶対値を掛ける。保存固定deterministic価格にPRの校正SEを追加しない。

これは選択されたscheduleに条件付けた**記述的感度**であり、最小値選択によるoptimismを補正したCIではない。Cost support、ideal-lawへの価格移送、native全law synthesis、implemented-law、actual CA、保証付きshots/資源優位は未資格/null。実機状態準備費用を0と認定したわけでもない。

将来population改善を検証するなら、今回の保存32でscheduleを選び、そのschedule・統計量を凍結して、別に許可された独立native校正cohortで確認する設計が必要。いま32を恣意的に分割して新結果を出したり追加samplingしたりしない。Candidate failure .01は、全D/全PR/混合×3laneすべての同時保証を意味しない。

## 6. 人工検証と技術状態

[最終log/XML](../../artifacts/research_c_roundwise_det_pr/2026-10-30/technical/tests_final.log)、[SOURCE/環境/技術照合](../../artifacts/research_c_roundwise_det_pr/2026-10-30/technical/)。新42＋既存75＝117 distinct PASS。人工metadata fixtureの不足と、旧unknown laneのID省略を科学実行前に修正し、最終SOURCEで再確認した。旧科学SOURCEの修正や実分子によるdebugはない。

| 範囲 | 確認内容 |
|---|---|
| 共通target | 入力/JU/ground/sector/phase/time、旧SOURCE/ID、nativeとのbinding。別DF/不一致/aliasは拒否 |
| Accuracy | A=1/A<1、PF/RTE/raw-signal、phase/radius不適格、nullを維持、failure合計、DF予算/一回計上、round欠損 |
| Allocation | 全D、全PR、実際に両方法を使う人工混合、D支配、同値D、unknown価格、非有限値、独立最小値和と既存総会計の一致 |
| RPE復元 | 実際の既存decoderで13round、異なる正attenuation/round位相偏り、5人工energy、branch越え、最終β/2^12内 |
| End-to-end | 人工JSON接続で189/189、3lane×26 accuracy/cost、partial STOP/fresh拒否、保存PR shots不一致STOP、同じ32meanのSE共有 |
| 回帰 | Stage C、Stage3 accuracy/serial/reference、旧phase復元、short-round optimizer。すべて人工データ |

117件は約2.3秒規模。32利用可能CPU/空きRAM約52GiBを確認し、小さいscalar/NumPy経路と既存vectorized decoderを使用。科学runnerにもCPU/RAM/worker/wall capは追加せず、支配する大行列処理は存在しない。Task-local venv/cache/tmp、既存依存read-only、共有環境・他jobへの変更0。Local人工実行でありimmutable CIや外部独立再現ではない。

## 7. 一件だけの未承認保存値検証：Q9

[固定契約案](research_c_roundwise_det_pr_contract_unapproved_20261030.json)は**最大189 logical units**、fresh `artifacts/research_c_roundwise_det_pr/2026-10-30/run01`、retry/resume0。新SOURCE4件＋再利用114件、保存input23件をhash固定した。新しい科学計算の追加承認を要求するのは、この189-call保存値再解析の開始時だけである。実行中の段階ごとの承認証明等は要求しない。

| 種別 | 最大units | 一単位 |
|---|---:|---|
| Saved bundle／common energy binding | 1＋1 | 固定JSON一式／SOURCE・H・ground・phase・aliasの照合 |
| Qualification | 3 | evidence／ideal／assumedの宣言 |
| Round accuracy | 78 | 3lane×2methods×13round、既存assess_round/Hoeffding |
| Round serial cost | 78 | 同上、X/Y一組を一単位 |
| Allocation | 3 | 各laneの独立13選択 |
| Full13 energy／total | 9＋9 | 3lane×全D/全PR/混合、既存energy/failureと6指標集計 |
| Shared uncertainty／comparison | 3＋3 | 各laneで全scheduleの共通SE／主baseline比較 |
| Summary | 1 | 状態・限界・STOP |
| **合計** | **189** | Correctness FAIL時は依存callを停止 |

新fit・optimizer・DF/ground/operator/PF/RTE再計算、分子NPZ数値decode、QPY decode、新sampling/build/compile、量子shots、新r/K/候補/β/α探索は0。CSV価格表や最適scheduleなどの科学的保存値再集計は、この次の許可後だけ実施する。

順序はpassive preflight→保存bundle→common bridge→laneごとのqualification/26 accuracy-cost→allocation→3つのenergy/total→uncertainty/comparison→summary/STOP。最初のcorrectness exceptionでpartial証拠を保全する。登録十分条件の不適格やUNQUALIFIED自体は他の固定laneを改変する理由にしない。Source/閾値の事後変更・fallback・retryはない。

将来の科学entryは`run_research_c_roundwise_det_pr.py --contract ... --execute`。**今回は`--execute`を付けずにpassive preflight PASSだけ確認した。** 未承認ファイル名・metadataは許可の証明や認証手続きではない。Scientific calls=0、科学run01なしを技術STOPに保存する。

## 8. 技術結論・次担当：Q1～Q10

| 問い | 回答 |
|---|---|
| Q1 | 同じDF ground/energyに対して各方法のround信号を上の正attenuationモデルへ結べれば可能 |
| Q2 | Common θ/q grid、raw-signal誤差/正radius、各round β以内、β<π/3、alias、IID/union boundが十分。既存decoderで対応する |
| Q3 | 共通energy bindingとmethod/evidence/costのidentityを分離し、旧IDと出典を残す。既存同一target検査は維持 |
| Q4 | 同じCA/β/α/DF/stateと揃えたlaneで可能。旧M2 numericとPR仮定の単純結合は不可 |
| Q5 | 固定配分・独立serial価格なら分離可能。2^13探索不要 |
| Q6 | 全round D優位なら固定点モデル内でNO-GO。Unknown競合を支配されたと扱わない |
| Q7 | 保存PF/RTE/A/ground/native/shellを再利用。新しいscalar accuracy/cost/選択/totalは次の科学再解析である |
| Q8 | 既存round最適化の新しい適用比較。PR r=0をDと同一視せず、Track A/Bの別task成果も移送しない |
| Q9 | 保存189-call一件で、どのroundにもPRを使う余地があるか、全D支配か、証拠不足かを区別できる。まだ実結果はない |
| Q10 | 支配された場合は、この固定H4のschedule探索を区切る。次に一つなら、**tail平均を保つ辞書/samplingの自由度でnormalizationとnative価格の交換を改善できる余地が、既存Euler/IS/Parent allocationを超えて残るか**を紙上で先に判定する。新しいγ/r/s/rank/nativeの追加探索を自動的に増やさない |

Q10は新しい手法の採択ではない。既存Track Bのcommon配分のL1最小性やcost-aware ISを再発明せず、具体的な新自由度・成立条件が見つからなければ、その改善系列を区切る。今回その設計・実装・計算・詳細レビューへ進まない。

**ユーザー／研究C**：保存値189-callの範囲を確認し、科学的な保存値再解析を許可するか判断する。**Codex**：許可後だけ固定SOURCEとfresh outputで一度実行し、各round・3比較・誤差資格・不確かさ・STOPを公開する。今回は実装・人工証拠・契約案をcommit/pushし、remote SHA/tree・SOURCE/input・旧原本の照合後に**mandatory STOP**する。
