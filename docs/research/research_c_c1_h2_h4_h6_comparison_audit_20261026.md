# C1：H2／H4／H6の結果統合・baselineとnative条件の技術監査

2026-10-26 JST。起点・証拠snapshot：`5385c6ef01879ee8a70734bfc0d5ae8999dd6186`。
Branch：`codex/research-c-c1-comparison-audit-20261026`。

**保存結果の比較は成立するが、3系を同じrank選定・同じnative実装によるサイズ依存実験とは扱えない。** 特にH4のXDF10は、生成時のOpenFermion閾値に由来する保存対照であり、CA/8を初めて満たすrankとして選定されたものではない。確認したC1証拠にはprefix1〜9のsector誤差評価がない。rank10が必要か、過剰かは`UNRESOLVED`である。

技術的な次の一実験には、**案A：保存H4 XDFのprefix1〜9のDF適格性だけを調べる**ことを推奨する。H4比較のbaselineに直結し、新fit・回路・shots計算が不要だからである。これは未採択の技術提案であり、実行許可・研究方針の正式採択ではない。[案A／案Bの独立した技術設計](research_c_c1_minimal_experiment_design_20261026.md)に入力・関数・call案・STOPをまとめた。

## 1. 実施範囲と証拠

正式共通運用指示2026-10-25版の§8.1／8.4／10と、今回の添付指示に従った。固定GitHub commitの報告・raw scalar/CSV・生成SOURCEを取得し、ローカルGit blobと照合した。分子NPZ・QPYは数値decodeせず、既存JSON／CSVの報告済み数値を引用した。新しいnorm、rank判定、費用集計、順位計算は実施していない。

成果は技術監査と未採択実験設計である。新しい詳細科学レビュー、主要claimの採否、研究継続判断は実施していない。実分子科学call、新fit、PF、回路build／compile、QPY再評価、shots／総費用の新規算定はすべて0。以下の表は過去の実行値の転記である。

主要出典は[H2実行報告](research_c_c1_h2_execution_20261025.md)、[H4精度pilot](research_c_c1_stage3_deterministic_accuracy_execution_20261025.md)、[H6共通CA結果](research_c_c1_h6_ca_resources_execution_20261026.md)。[監査証拠](../../artifacts/research_c_c1_comparison_audit/2026-10-26/evidence_map.json)にremote blob・SOURCE差分を、[12候補転記CSV](../../artifacts/research_c_c1_comparison_audit/2026-10-26/saved_comparison.csv)に各値の出典pathを保存した。local evidenceの公開であり、CI・外部独立再現ではない。

## 2. 共通条件と相違点

全系は直鎖・隣接距離1.00 Å・STO-3G・中性singlet。通常number CDF／XDF、元gから補正したκ、核scalar、元のframe順を維持する。決定論全frame ordered symmetric S2（2次）、s=1主対象／s=4対照、δ=0.1、m=0..12、時刻0.1×2^m、kernel時間δ/s、反復n=s×2^mである。PRのL_D split、有限RTE打切り、ランダムtrajectoryは今回の決定論比較に非適用。

各DFのsector ground-state oracle、system状態準備費用除外、ideal IID Born測定を仮定する。CA=0.00159 Ha、DF reserve=CA/8=0.00019875 Ha、β_total=π/6、β_stat=π/12、候補ごとのfailure上限0.01、各round・X/Yの配分0.01/26は共通。4候補同時や事後選択の成功保証は含まない。実際の量子samplingは全系0。

| 条件 | H2 | H4 | H6 |
|---|---|---|---|
| 電子数・軌道 | 2電子・2空間／4 spin軌道 | 4電子・4空間／8 spin軌道 | 6電子・6空間／12 spin軌道 |
| sector | Nα=Nβ=1、4次元 | Nα=Nβ=2、36次元 | Nα=Nβ=3、400次元 |
| full／controlled | 16／32 | 256／512 | 4096／8192 |
| 元入力 | 当該pilotでPySCF RHF／MO積分を生成 | 旧非DF H4 pilotから保存P-H4を回収 | 当該H6 pilotの保存RHF／MO入力 |
| CDF | ffsim0.0.83、通常圧縮DF、JAX x64解析gradient | 同左 | 同左 |
| CDF初期化 | modified-Cholesky→旧real codec、2D logm bridge | 同左 | 検証済み保存XDF14→real_schur_column_gauge_v1→修正pack |
| optimizer | L-BFGS-B、tol1e-8、options=None | L-BFGS-B、tol1e-8、options={} | L-BFGS-B、tol1e-8、options=None |
| CDF rank政策 | requested1..3、first-passing、各rank1fit；1不適格→2採用、3未実行 | 上限8を事前指定した1fit、rank探索なし | 旧requested1..21は全予算未達で停止；後に修正codecのrank14限定1fitを別実行 |
| CDF採用rank | 2 | 8 | 14 |
| XDF分解 | ffsim0.0.83、非最適化modified-Cholesky | OpenFermion1.6.1、eigh→重み順square | ffsim0.0.83、非最適化modified-Cholesky |
| XDF rank政策 | max_vecs3で1分解→元順prefix1..3を評価、3が最初のPASS | 生成時df_rank=None／df_tol1e-8→保存rank10を再利用 | max_vecs21／actual18で1分解→元順prefix1..14を評価、14が最初のPASS |
| XDF採用rank | 3 | 10 | 14 |
| CDF停止の保存例 | rank2 success、nit5／nfev7、projected gradient条件 | success、nit68／nfev74、relative reduction条件 | 修正fit success、nit1／nfev3、relative reduction条件 |

ffsimのtol1e-8は初期分解の条件であり、SciPy minimizeのtolやsector誤差保証ではない。全系の目的関数はtensor Frobenius残差の二乗の半分であり、sector operator normを直接最小化していない。H6ではtensor残差減少とsector誤差増加が同時に観測されている。[修正fit報告](research_c_c1_h6_corrected_fit_execution_20261025.md)

H2／H4の旧codecを使った事実から、その最終DF表現が不正だとは推定しない。保存された最終表現には独立CAR／一般K・物理誤差の実測証拠がある。一方、忠実な初期化で同rankのfitをした場合にも同じ解・rank選択になるかは未評価。H6で見つかった初期化問題をH2／H4へ数値的に外挿しない。

## 3. DF誤差・予算・PF・合成誤差

以下は既存sector operator normとguardの個別値。異なる物理DF精度を完全一致させた比較ではなく、共通CA予算内の登録候補比較である。

| 系・arm | rank | sector DF norm（Ha） | 保存guard／allowance（Ha） | 保存CA/8判定 |
|---|---:|---:|---:|---|
| H2 CDF | 2 | 1.3176231587153952e-6 | 2.4289938062665934e-12 | PASS |
| H2 XDF | 3 | 4.8652059505012625e-16 | 3.0465959187204976e-12 | PASS |
| H4 CDF | 8 | 7.666027761392501e-5 | 1.881549094133745e-9 | PASS |
| H4 XDF | 10 | 3.3751558354542566e-15 | 1.4812598092146554e-9 | PASS |
| H6 CDF（修正fit最終） | 14 | 1.0689213236556284e-4 | 1.1384527736890936e-6 | PASS |
| H6 XDF（selected） | 14 | 1.0247385485275444e-4 | 1.1384527781887395e-6 | PASS |

H2／H6は`sector_operation_scale_guard8_proposal_v1`、H4は`binary64_operation_scale_guard8_v1`を使用し、同じguard式ではない。H4 Stage1時点のDF値は非gateの測定であり、CA/8の採否は後のStage3で評価された。XDFの数値floor近傍の値を厳密な誤差0と扱わない。

| 項目 | H2 | H4 | H6 |
|---|---|---|---|
| PF reference | SectorCAR＋S2 kernel、反復二乗とDFのeigenpairによるexact evolution差norm2 | 独立CAR／保存operatorとStage3 sector reference、保存13round norm2 | SectorCAR400、保存DFのeigenpair＋反復二乗、52セル保存 |
| 最大PF誤差 CDF s1／s4 | .0132802154112／.000829829308650 | .0691369307091／.00432133825219 | .1222267558487／.00764308172000 |
| 最大PF誤差 XDF s1／s4 | .0160182315159／.00100095521050 | .068430410／.0042770929 | .1222311163101／.00764335475930 |
| 合成の入力coverage | full32 controlled operator、4 kernel＋3 shell | full512 controlled reference、保存s1とs4追加局所reference、4 kernel＋3 shell | S=ancilla×sector400の全800入力、出力はfull8192、4 exact-IR group＋shell |
| kernel数値距離 CDF s1／s4 | 1.30484408581e-13／1.03854159854e-13 | 7.64459292172e-10／7.66177713833e-10 | 7.96974388305e-10／8.18575441612e-10 |
| kernel数値距離 XDF s1／s4 | 1.57363840906e-13／1.56475972791e-13 | 1.28122521537e-9／1.22567412486e-9 | 1.56020424340e-9／1.54554657260e-9 |
| coherent反復 | nηとhead／X-Y、unitary telescoping＋Frobenius代用 | 同左 | reference norm-growth αも保持し、αの幾何和で伝播 |
| 資格 | 数値full-operator pilot、rounding enclosure未資格 | 数値full-operator pilot、rounding enclosure未資格 | 完全sector入力の資格B、丸めenclosure C・full8192全入力global boundは未資格 |

PF誤差は無次元、DF誤差はHa。PFと合成は複素signal単位で各一度だけ合わせ、位相と統計余裕を検査した後、DFをenergyへ一度だけ加える。異種誤差をそのまま足していない。H6の有限probeのみの旧104行は未資格のまま保全され、後段の800基底評価とは別証拠である。[H4 PF保存表](research_c_c1_stage3_feasibility_execution_20261025.md)／[H6 PF](research_c_c1_h6_saved_pf_execution_20261025.md)／[H6全基底](research_c_c1_h6_synthesis_bound_execution_20261026.md)

## 4. 保存shots・条件付き総資源

全12候補は過去の報告で`NUMERICAL_PILOT_CA_PASS`。H2 rawの`CONDITIONAL_CA_PASS`は、その報告が上記の数値pilot資格に限定して表示している。H4は前段の楽観的Nを固定して合成誤差込みのmarginを再検査した。H2／H6はその系のPF／合成を入力としてNを算定した。この手続きの違いを、3系で同一のshots最適化を実施したと表現しない。

| 系・arm | s | 算定shots合計 | 数値energy誤差評価（Ha） | 条件付き総RZ | 条件付き総CX | 条件付き総depth |
|---|---:|---:|---:|---:|---:|---:|
| H2 CDF2 | 1 | 13352 | .0012796349488257434 | 3990125812 | 2694155960 | 4893901728 |
| H2 XDF3 | 1 | 13364 | .0012783173262851165 | 6274985632 | 4194706740 | 7593362128 |
| H2 CDF2 | 4 | 13294 | .0012796349488257434 | 15699690593 | 10600614368 | 19255632763 |
| H2 XDF3 | 4 | 13294 | .0012783173262851165 | 24589446313 | 16437661520 | 29755608387 |
| H4 CDF8 | 1 | 13606 | .0013549794824010532 | 111798946393 | 67828478608 | 117389854735 |
| H4 XDF10 | 1 | 13600 | .0012783188045012185 | 153454725962 | 87693651360 | 149364852770 |
| H4 CDF8 | 4 | 13314 | .0013549794824010532 | 408641594551 | 247923144128 | 429077084077 |
| H4 XDF10 | 4 | 13314 | .0012783188045012185 | 560693479327 | 320492595456 | 545181461829 |
| H6 CDF14 | 1 | 13880 | .0013863479083772862 | 520491751108 | 306659760960 | 507573487852 |
| H6 XDF14 | 1 | 13880 | .0013819296308689773 | 510393076228 | 301300464192 | 498694652908 |
| H6 CDF14 | 4 | 13326 | .0013863479083772862 | 1761144891569 | 1037643479376 | 1717292262491 |
| H6 XDF14 | 4 | 13326 | .0013819296308689773 | 1727240281697 | 1019642229424 | 1687651106555 |

RZ depth、CX depth、sizeも[転記CSV](../../artifacts/research_c_c1_comparison_audit/2026-10-26/saved_comparison.csv)に出典列付きで保持した。主指標はΣ_round,axis N×（n kernel＋head＋axis tailのRZ）。depthはserial fenceを持つstageの加算値で、全長回路のglobal compile最小depthではない。決定論費用supportがsingletonなのはこの費用モデルの性質であり、誤差の厳密enclosureとは別である。

既報の比較はH2でCDFが36.412192%／36.152728%少なく、H4で約27%少ない。H6ではCDFがXDFより1.978607%／1.962935%多い。同sのH6 shotsは全round・X/Yで一致する。これらは各系の登録baselineに対する保存結果で、rank・実装・分解policyを揃えたサイズ依存傾向やCDF一般の優劣ではない。今回新しい割合・順位を算出していない。

厳密`CA_achieved`、保証付きactual shots、状態準備込み費用は未資格／null。energy値は登録十分条件の評価で、実測推定energyではない。ideal測定、oracle、numerical reference、serial費用に条件付けられた比較として引用する。

## 5. H4 XDF10 baselineの選定来歴と公平性

### 生成時のrank10とCA/8の相違

[生成SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py)は、同じRHFのh/g/cから`df_hamiltonian_from_integrals(df_rank=None,df_tol=1e-8)`を1回呼んだ。そのhelperはOpenFermion1.6.1の`low_rank_two_body_decomposition`へ渡し、one-body補正・λ・Gを保存してhermitizeする。

[固定OpenFermion SOURCE](https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/circuits/low_rank.py)は二体arrayのeigh後、`abs(λ)*(Σ|G|)^2`の重みで並べ、残りの重み和が生成閾値以下になるprefixを選ぶ。これは**CA/8の物理sector norm判定ではない**。その処理の結果が保存rank10で、C1は再分解せず引き継いだ。第4戻り値truncation_valueはscalarに加えていない。

確認対象はP-H4生成SOURCE／receipt、入力回収・Stage1条件、Stage1保存全rank10 sector値、Stage2、Stage3 feasibility／cost／精度契約・raw・SOURCE。ここには同じP-H4／保存順序でのrank1〜9のsector norm・guard・CA/8結果がない。PRのL_D prefixは、残りをランダム側に保持する分割であり、XDF rank切詰めの評価ではない。古い別snapshotのDF screeningやPF検証を代用しない。全repository・全branchで不存在を証明したとは主張しない。

| 問い | 監査結果 |
|---|---|
| rank10はCA/8を満たすか | 保存Stage3の数値判定でPASS |
| rank10はCA/8でのfirst-passingか | その選定をしていない。最小適格rankはUNRESOLVED |
| prefix1〜9の実測誤差は | 今回確認した当該C1証拠にない。値を推定しない |
| CDF8も同じrank探索か | 事前指定1fitであり、first-passing探索ではない |
| H2／H6との分解の違い | XDFはmodified-Cholesky生成順prefix、H4はeighの重み順square |
| 分解表現だけの因果比較か | 初期化・分解・rank政策・native入口も異なるため、そうは言えない |

CDF8／XDF10は、同一元P-H4・共通CA・同一P0／費用定義の**登録候補同士**として比較できる。XDFの最も強い低rank baselineを既に排除した比較、CDFが一般に最適なXDFより27%少ない比較とは言えない。

より小さい適格prefixが見つかれば、DFの比較対象を再検討する材料になる。ただしrank低下だけでPF・合成・shots・compiled費用の改善や順位反転を認定しない。新targetではそれらの最小限の再資格付けが必要であり、旧27%の数値を上書きしない。全prefix不適格なら「保存順prefix1〜10中で10が最初」と言えるが、別順序・別分解法を含むXDF全体の最小rankにはならない。

## 6. native Gaussian・P0・共有規則の整合性

### 同一の共有コードとH6専用差分

H4 Stage2 SOURCE a977a439…、H4 cost SOURCE9ce6703c…、H4 accuracy SOURCE002d0dc2…、H2 SOURCE33c380d8…、H6 native SOURCE3d7b56b4…で、共有`c1_stage2_native.py`、`circuits.py`、`gates.py`、`df_trotter/ops.py`のGit blobは4ファイルとも同一だった。[blob表](../../artifacts/research_c_c1_comparison_audit/2026-10-26/evidence_map.json)

H2／H4はNativeBuilder→native_gaussian→`BogoliubovTransform(U.T)`というNature0.7.2経路。H4 XDFは保存λ/Gを`prepare_xdf`のsquare primitiveへ、H2／H6 XDFはffsim J/UをCDFと同じ一般K入口へ接続する。この入口の違いも保持する。

H6は共有SOURCEを変更せず、`PrecisionBuilder._basis`とtask-local compiler wrapperを使用した。schedule・中央I/Z/ZZ・κ・scalar・actual adjoint・制御の規約は再利用する。

| 差分 | 技術的効果 | 費用と既存結果に対して言える範囲 |
|---|---|---|
| Gaussian：内部isclose／arccos→exact-zero／LAPACK zlartg／atan2 | 微小角度の省略・復元精度低下を回避。U.Tと消去順序・gate種は維持 | 新たにgateが残り、count／depthやP0の簡約が変わり得る。H2/H4での実際の変化はUNRESOLVED |
| P0：declared phase加算だけMPFR160で追跡 | 同じP0 passを呼び、symbolic markerを正確に相殺して最後にbinary64へ戻す。出力へのphase fitなし | 人工短／長回路でgate・qubit・angle列、countが旧P0と同一。一般的なgate列不変の証明や、H2/H4実回路での確認とはしない |
| compile／inverse生成の共通mutex | Qiskitの並行生成競合を避ける | 物理targetや最適化profileを変えるものではない |

[Gaussian修正報告](research_c_c1_h6_native_precision_repair_20261025.md)には人工1e-9回転の旧省略と修正後保持、[phase修正報告](research_c_c1_h6_native_stable_phase_repair_20261025.md)には長い人工回路で旧4.352757e-10→修正5.360574e-13という保存値がある。今回の再試験29件もPASSした。これらをH2/H4の実データ差分や既存PASSの取消しへ置き換えない。

P0の公称条件は全系同じ：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、backend／target／coupling map／layout／routing／schedulingの指定なし。既存の公式gate definition先行展開によるphase欠落対策は全系共有であり、H6の**加算精度対策はその追加差分**である。両者を混同しない。

### 同basis共有と一般相対Gaussian融合

NativeBuilder.runsは、隣接blockのrotation tupleが**完全一致**する場合だけ同basis runへまとめる。combined_diagonalは同じrun内のphase／RZ／RZZをmath.fsumで加算し、exact-zeroだけを省略する。異basis間の相対回転V_next†V_prevを新たに合成する最適化は、このC1 NativeBuilderには実装されていない。別のffsim Trotter実装に相対basis処理があることは、当該保存回路での実施証拠ではない。

H4 Stage2およびH6の保存literal／shared metricsは各候補で同じで、H6ではIR aliasも確認済み。H2はoptimized=Trueのみを本pilotで評価したため、H2 literal費用との差は未測定。未観測の共有削減を加えない。

現行規則は各比較の両armへ対称に適用されている。一方、一般相対融合を導入した場合の効果は、rankだけでなくbasisの構造・隣接性に依存し得る。CDF／XDFのどちらが多く得をするかは未測定で、現在のrank差の節約を相対融合後にも保証しない。今回は融合を実装・採択しない。

### 将来、完全に同じ実装で比較する場合

保存J/U／λ/G、κ・scalar・frame順を変えず、H2／H4双方のarmにもH6のprecision basis providerとstable-phase P0を注入する薄い接続が最小変更となる。旧runner／旧契約を変更せず新SOURCEで人工回帰を行い、その後に別許可で必要な回路correctness・metricsを取得する。

gate列やmetricsが変わらず保存精度条件を維持する場合だけ対応する費用の再利用を判断できる。変わる場合は合成誤差の13round伝播・固定shotsのmargin・費用を再資格付けする。H6と同じ修正codecでH2/H4を再fitすることは、native実装の整合化とは別実験であり、この最小変更に含めない。現時点の未評価を理由に3系を自動再計算しない。

## 7. 2案の情報価値と技術上の注意

| 観点 | 案A：H4保存XDF prefix適格性 | 案B：H6 CDF13修正codecの1fit |
|---|---|---|
| 解消する不明点 | 約27%比較の対照rank10が、保存順CA/8選定でも必要か | rank13で忠実な初期化から予算へ圧縮できるか |
| 新しい計算 | 36次元のprefix1..9 norm／allowance。fit・回路なし | codec・tensor・sector400診断、非最適化initializer13照合、L-BFGS-B 1fit |
| 再利用 | 元P-H4、保存λ/G、physical sector行列、rank10評価、既存xdf matvec／allowance | 元H6、XDF_full prefix13、旧rank13 scalar、修正codec／ffsim接続、SectorCAR |
| 実装負荷 | 小さい。既存評価関数への薄いprefix入口 | 中程度。rank14固定entryの条件と開始gateを新rank13契約へ分離する必要 |
| 結論への影響 | H4の強いbaselineが未確認という点を直接解消。低rank適格なら費用再評価の必要性を判断できる | 現H6同rank14の約2%差を変え得る候補だが、DF PASSだけでは費用優位不明 |
| 不適格時の知見 | 全1..9がNOT_METなら保存順の最初の適格が10 | この初期値／固定optimizerの単一fitではCDF13予算未達。全rank13の不可能性ではない |
| 負荷 | 36×36、324 basis matvec案。新fit不要 | 400×400、13 frame、1fit。旧349 iterationを修正fitの所要時間へ移送しない |

**優先提案は案A。** 既存H4比較のbaseline監査に直接応え、新しいfitの成否に依存せず、どちらの判定でも情報が得られる。案Bは次の独立候補として設計を残す。2案を一括科学実行にせず、rankや候補の自動追加もしない。

案Bで特に重要なのは、保存XDF13が既に予算外である点。旧rank14 runnerの「開始時DF予算PASS」をそのまま使うとfit前に停止する。新しいrank13案では、**codecの忠実性を必須gate、開始DF予算を診断値、終了DF予算を採否条件**とする変更を、実行前にユーザーが明示採択する必要がある。旧rank14契約を緩和したり、黙ってgateを迂回したりしない。

## 8. 未解決事項・SOURCE・完了状態

| 未解決事項 | 現在必要な証拠／判断 |
|---|---|
| H4 XDF10の最小適格性 | 案Aの別許可後の測定。現状UNRESOLVED |
| H4低prefixのPF／合成／native費用 | 案Aで選定しても未取得。別設計・別実行が必要 |
| H6 rank13のcodec前後・修正fit | rank14／18での原因確認を代用できない。案Bで初めて測定する |
| H2/H4のprecision版との費用差 | 実データの同一実装再比較は未許可・未実行 |
| 一般相対Gaussian融合の利益 | 当該builder未実装・未評価。両armに同じ機会を与える別研究判断 |
| 厳密CA／実hardware／状態準備込み優位 | 全系で未資格。数値pilotを昇格させない |

| 対象 | 科学SOURCEと結果commitの区別 |
|---|---|
| H2 | 科学SOURCE `33c380d8aa792e5119bb3adec5dd07d1e1377214`、公開結果 `cc0cdb52570993fa4f14f6eb199b583d0411d585` |
| H4元入力／fit | 入力生成 `0c329b30a81accb35c212be8defa66fee0dd0f12`、packet公開 `86b8b002704aced8ca239414080f53793de81658`、CDF fit `da2edb465d5ae6432439fbd1df8e57512ee59c11` |
| H4評価／native／費用／精度 | CAR評価 `bec41ac7dc7e034bd664817acfdcfed858ddadd0`、native `a977a4390668bc107360c8ff05f43d851dc81dc9`、cost `9ce6703c89874c142c64ad934354bdff0db703d1`、accuracy `002d0dc2e1dafd8cb71735c43253fa054e675f6d`、精度結果公開 `6202f036457940532aeaa2873cd6d9e6f123c0ca` |
| H6保存入力／修正fit／PF | 旧入力SOURCE `33c380d8aa792e5119bb3adec5dd07d1e1377214`、修正fit `33432e78659b8b3e654e39f1392328cb0ba5c487`、PF `87ea924ae28353707bd2eb3adf94231042b67361` |
| H6 native／全基底／共通CA | native `3d7b56b44832a2a90a8d3e500bf55e856f36da76`、全基底 `e36e0560624c66df0aedf619fc5c8c33ecb4f837`、CA `55b3d86817308a8d676221982aae13e1a6e3bed4`、最終結果公開 `5385c6ef01879ee8a70734bfc0d5ae8999dd6186` |

同じsource fileでも旧runと新runの許可・STOPは別である。旧失敗、grant、ledger、marker、SOURCE／入力／結果を変更していない。新scientific runner・承認管理・新grantは作らず、必要な資料と29人工回帰の証拠だけを追加した。

独立worktree `/tmp/prt-c1-comparison-audit-20261026`、既存eval venv read-only・task-local cacheで人工検証。共有環境・他ユーザーjobは変更していない。29 tests PASS／FAIL0、既存分子配列を読まないことをSOURCE確認した。

**次の担当はユーザー／研究C**：案Aを先に実施するか、その科学条件を採択・明示許可する。**Codex**：採択後、案Aの最小entryと人工検証を完成し、新SOURCE／契約を実行前に固定する。今回の報告や未採択call案を実行済み契約とみなさない。実験後の科学的解釈・主要claimの採否はユーザー／GPTへ戻す。

commit／push・remote SHA／SOURCE／成果物照合後、**mandatory STOP**。新科学計算・詳細科学レビューへ自動進行しない。
