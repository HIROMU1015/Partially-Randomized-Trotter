# 研究C：性能・機構・一般化へ移行する技術整理と最小実験M1

2026-10-28 JST。起点は公開commit `0075f660b1e3eaa3ca93347e6249d3c8c5ffe987`、独立branchは`codex/research-c-performance-mechanism-preparation-20261028`。正式共通運用指示2026-10-25版§8.1/8.4/10/10.1/10.3と今回の依頼を適用した。

**次の一実験には、保存PR Stage B1/B2/C1を使ったshots×価格の記述的寄与分解M1を推奨する。独立20-call契約・最小adapter・人工検証まで完成し、新規科学callは0。** 追加の厳密保証は当面保留する。F2-K1/F3-K1の保証・旧STOPは保全するが、保証完成を性能研究の開始条件にはしない。

本書は保存報告の引用、比較条件の技術調査、実験設計であり、重要なGPT詳細科学レビューではない。実科学JSONから新しい比率・順位・寄与・shots・総費用を算出していない。以下の実数値・割合は明記した既存報告からの引用である。分子NPZ/QPYの数値decode、新fit、PF/RTE、sampling、build/compile、追加certificateは0。

## 1．確認した公開証拠と、現在支持される範囲

GitHubの起点commit/treeとローカルGitを照合した。依頼で指定された10報告をGitHubから取得し、本文bytesが起点のローカル報告と一致することを確認した。SOURCEと既存結果のJSON schemaを静的に確認し、rawの新集計は行っていない。各報告は公開ローカル実行証拠であり、独立外部再現・CI結果とは区別する。

以下のPR研究は共通して**保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ2/sector36、通常number CDF8、元frame順prefix LD2、元g補正κと核scalar一回、partial ordered symmetric S2、δ=.1/s4/h1/40/r16/K0/m0..12、own-DF ground oracle・状態準備費用除外**を対象とする。Stage Aでは時間発展条件は後続の背景条件で、fit自体にPF評価はない。

| 証拠・科学SOURCE | 既報の指標（転記） | 支持すること／残る限界 |
|---|---|---|
| [Stage A V2](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)、`e1085c95f5c01b68566e17d543ace5832cb94061` | 同一x0、γ0/.1。exact tail mass 2.2097618367788057/1.4424298855756243 Ha、既報34.7246449%減。DF norm+guard 1.0297636237465247e-8/1.229561498918692e-8 Ha、両CA/8内 | この初期値・目的・停止規則でのproxy改善と予算保持。単一γ対・初期値であり、一般化・大域最適性なし |
| [Stage B1](research_c_pr_aware_df_stage_b1_execution_20261027.md)、`7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9` | 両13/13潜在適格。m12 A=.20959759/.51386660、補正RTE誤差.1643051229/.1256120553、PF誤差.0044769543/.0044762458 | より大きいA・小さい補正RTE誤差の数値観測。PFはほぼ同程度。Aはraw信号の絶対値ではない |
| [Stage B2 run02](research_c_pr_aware_df_stage_b2_execution_20261027.md)、`a1cf0046967c206f414dbad568a75884ee076ebf` | RZ標本mean5198.3125/4765.875、既報8.3188%減。各32標本。6指標の保存mean/SEあり | finite-bit独立cohortのwhole-kernel価格差。別law・seedで非paired。標本極値はsupportでも母平均CIでもない |
| [PR Stage C1](research_c_pr_aware_df_stage_c1_execution_20261027.md)、`b1c735da394d4b0a444cbc618ead9781ef4d9134` | ideal_exact_nativeでshots53136/19574、総RZ 3,194,333,274,946/540,290,160,109、既報83.08598%減 | 明示仮定付きserial資源の点推定。evidence laneはUNRESOLVED、actual CA・保証shots・母平均優位は未資格 |
| [F2-K1](research_c_pr_aware_df_F2_K1_execution_20261027.md)、`3742ec10ddc0df5faaaa2a6856a9cc60ea013eb9`／[F3-K1](research_c_pr_aware_df_F3_K1_execution_20261028.md)、`9b0f4d2b40159b77f28caf3f215ad22d0d075879` | Baseline/Proposed各固定1 trajectoryから独立K1 targetへのvalidated operator2上界、いずれも登録1e-8内 | full512/Arb192/位相保持の個別保証。全law・C1平均誤差・CA・費用supportへの昇格なし。今回追加評価しない |

Stage Aは両arm success/status0だが、停止は`REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`。保存終了gradient L2は3.104515097647238/4.673093479443538、iterationは97/1129で、gradient収束や同一反復予算の比較ではない。`tol=1e-8`はffsim初期分解側で、SciPy minimizeのtol指定ではない。Baselineもf/f*で正規化した新fitであり、旧通常CDFの再現実験ではない。

**RQ1は、この保存H4の2-arm数値pilotと明示仮定付き資源推定という範囲で支持される。RQ2（強い決定論PFにも競争的か）は未判定。** 約83%を研究全体の成功、同精度保証、独立した新規性の証明とは扱わない。

## 2．強決定論baselineとの条件差

[強baselineの共通CA報告](research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)の科学SOURCEは`1aeedd96178a3f50a15bc2701c3c332f4f391b64`。旧CDF8のroute B2 relativeとXDF8のroute B3 spin+relative、全frame ordered S2・s1・δ=.1/m0..12、PR split非適用。以下のB2/B3はnative route名で、PR Stage B2とは別である。

| 保存指標 | 強決定論CDF8 B2 | 強決定論XDF8 B3 |
|---|---:|---:|
| DF norm+guard [Ha] | 7.666215916301914e-5 | 9.533901455091894e-5 |
| 1 h-step RZ | 11062 | 10885 |
| 全13round X/Y shots | 13606 | 13600 |
| 条件付き総RZ | 102,056,444,569 | 100,267,421,030 |

これらも既報値の引用で、PR値との新しい比率や順位は計算していない。[強baseline機構報告](research_c_c1_h4_strong_baseline_execution_20261026.md)のSOURCEは`880006396326a4d46f4a4030564279e92fadb9cf`、B3科学SOURCEは`b40fce5c62c9fff97f68073cbc02182e4bbf2b9b`。その保存比較ではXDFの基底自由度・相対融合で旧CDF優位が変化しており、弱いXDF実装だけを対照にしない。

| 観点 | PR Baseline/Proposedと共通な部分 | 一致しない部分・比較時の処理 |
|---|---|---|
| 元Hamiltonian | 元P-H4 SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、geometry/basis/sector、元g補正κ・scalar政策 | PRはStage Aの新しい2fit。強baselineのCDFは旧保存fit、XDFはOpenFermion元順prefix8。DF Hamiltonianを同一と扱わない |
| DF予算 | CA=.00159、CA/8=.00019875 Ha | 実DF誤差は異なる。誤差完全一致は必須ではなく、共通予算・正しい最終energy接続が必要 |
| PF・時間 | ordered symmetric second order、δ=.1、13round | PRはprefix LD2+random tail、s4/h.025/n=4·2^m。強baselineは全frame、s1/h.1/n=2^m。kernel数や価格をそのまま同一単位として順位付けしない |
| reference | 独立CARとown-DF oracle、元のspin/frame順 | PR C1はB1の数値PF/RTEと仮定付き誤差、決定論は保存full512合成残差・norm guard伝播のlevel B。K1の新targetを自動で旧PF/DFのtargetへ移送しない |
| RPE | βtotalπ/6、βstatπ/12、candidate failure.01、axis.01/26、既存Hoeffding十分shots | PRはAと補正RTEを含み、合成/lawの明示仮定あり。決定論はA=1のideal IID Born数値pilot。実装CAを両者が保証したとは言えない |
| Gaussian・合法融合 | 共通PrecisionBuilder系、exact spin-lift κ、MPFR160 declared phase、actual adjoint | PRはhead内部relativeとtail隣接同一basis共有、cross-head/tail一般相対融合なし。強決定論は全sweep境界relative、XDFはexact spin基底。適用した規則・scopeを明示し、未観測削減を足さない |
| P0・precision | Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false、共通配置政策 | 同じcompiler名だけではtarget/回路構成/誤差資格の一致を意味しない。今回compilerを変更しない |
| 費用と状態準備 | `Σ N(n·kernel+head+readout)`、fenced serial、状態準備除外 | PRは32校正平均、決定論は保存固定gate価格。depthはscheduled sum、global最小depthではない。PRにlaw平均転送/価格CIの追加仮定がある |

共通CAでの候補比較は可能な研究設計であり、DF近似精度を完全一致させる必要はない。ただし**この保存2契約の総RZを直接割って、PRが強baselineに勝つ／負けるとは結論しない**。

RQ2の将来最小設計は、現PRの保存Baseline/Proposed J/Uについて全frame S2 s1対照を作り、同じprecision/P0・合法fusion政策・own-DF oracle・RPE/serial費用で比較すること。これで旧CDFと新CDFのfit差を除いた対照が得られる。さらに保存XDF8 B3を強い外部対照として扱い、政策が一致する部分だけ再利用する。`build_kernel(config=None)`/`StrongBuilder.ordered`は既存接続候補だが、その2分解の決定論PF・native価格は未取得。新fitは不要でも新PF・実回路・shots/費用が必要で、**M1契約には含めない**。共通合法規則を登録する前にPRへcross-head/tail融合を導入しない。

## 3．候補機構と、まだ切り分けられていないもの

依存関係は`J/U→DF残差と非可換性`、`Jと固定split→非identity tail係数mass λR→RTE normalizationとA`、`PF/RTE/Aと仮定誤差→信号room→N`、`J/Uとsampling lawとbasis/compiler→whole-kernel価格 μ`、`N,n,μ,shell→serial費用`。UはλRの係数写像そのものではなく、PFやbasis価格等を通じて作用する。

| 仮説 | 保存証拠 | M1で分かること／分からないこと |
|---|---|---|
| A：tail mass減少 | Stage Aのexact λ、B1のA改善。固定K0では既存式A=(1+(λh/r)^2)^(-rn/2) | 既存式の関係と保存値を参照。λだけを介入した再計算はしないので因果寄与率は未確定 |
| B：shots低下 | C1の保存Nは高mで大きく異なる | 同じ価格なら費用差がどうなるか、高mが何を占めるか。AとPF/RTEの内訳の因果分離まではしない |
| C：kernel価格低下 | B2のwhole-kernel標本平均は6指標とも小さい | 同じshotsならどれだけ変わるか。Gaussian・head・event/law別の因果効果には分解しない |
| D：DF/PF変化 | 両DF予算内、PFほぼ同水準、ProposedのDF誤差は僅かに大きい | DF[Ha]、PF/RTE[無次元]、A、保存room/Nを同じround表に並べる。別単位を単純加算しない |
| E：optimizer/x0依存 | 同一x0の単一γ対、相対目的減少停止 | 保存停止理由・gradientを再掲するだけ。再現性やγ頑健性には新fitが必要 |

J/U・law・head/tail構造は同時に変わっている。M1の価格項とshots項は会計上の寄与であって、A〜Eの因果効果ではない。

## 4．推奨する一実験：保存スカラーのM1記述的寄与分解

既存C1は総費用・Nの一覧・kernel meanを報告したが、**同一shots/同一価格の4 corner、高mの費用寄与、交互項と6指標の寄与分解は未評価**。同じ83%を再計算するだけの実験にはしない。新fitも新compileも要さず、次にshots側と価格側のどちらを調べるべきかの材料を得られるため最優先とする。

保存した両シナリオ`ideal_exact_native`（主）と`assumed_raw_signal_budget`（既存の感度対照）だけを使用する。新scenarioや候補は追加しない。未知合成/law誤差を実際に0と認定せず、0と.001Aはそれぞれ旧シナリオのcounterfactual仮定として保持する。未資格のevidence laneは数値集計へ入れず、そのnullを保持する。

shots armをa、価格armをbとして、保存のNとserial axis価格だけから

`C[a,b,k] = Σ_m,axis N[a,m] (n_m μ[b,k] + shell[axis,k])`

を評価する。a,b∈{B,P}の4通り。主指標RZ count、副指標RZ depth/CX count/CX depth/total depth/circuit size。crossed cornerでは双方のtarget_idを保存し、物理的に実現した新候補や新しいCA適格候補とは呼ばない。

| 出力 | 定義・意味 |
|---|---|
| 総点推定差D | CBB−CPP。負の値もそのまま残す |
| 価格を先に交換 | 価格項CBB−CBP、後続shots項CBP−CPP |
| shotsを先に交換 | shots項CBB−CPB、後続価格項CPB−CPP |
| 交互項I | CPP−CBP−CPB+CBB。D=(CBB−CBP)+(CBB−CPB)−I |
| 対称的配分 | 価格項P=((CBB−CBP)+(CPB−CPP))/2、shots項S=((CBB−CPB)+(CBP−CPP))/2、P+S=D |
| Round寄与 | 全13round×6指標、累積差、固定群m0..9/m10..12とm12単独。高m区分を結果後に変えない |
| tradeoff | 保存DF norm/guard、fit停止・gradient、PF/RTE/A、各scenarioの保存Nを転記。新energy判定はしない |

shellは各axisで一回だけ、coherent kernel反復はn=4·2^m。32標本をquantum shots倍率にせず、同一sample回路を全occurrenceに繰り返す物理モデルへ置換しない。旧C1のfresh独立occurrenceを仮定した期待価格モデルを維持する。

同じ32校正標本は全roundに共通なので、W_a=Σ_m,axis N_a,m n_mとして平均価格のSEをW倍する。Baseline/Proposedは非pairedの独立cohortとして、Nと物理誤差を固定した条件付き校正SEのみを伝播する。価格項とshots項は同じμから得るので共分散も保存する。`SE(D)=hypot(W_B SE_B,W_P SE_P)`、PとSの係数はそれぞれ(W_B+W_P)/2と(W_B−W_P)/2、共分散は`(W_B+W_P)(W_B−W_P)(SE_B²−SE_P²)/4`。sqrt13の独立化やSE単純加算をしない。これは保証CI・母平均順位・fit不確かさを含む統計保証ではない。

## 5．接続SOURCE・契約・20 call・STOP

[adapter](../../src/trottertracks/c_pr_aware_df_mechanism.py)、[専用entry](../../scripts/run_research_c_pr_aware_df_mechanism.py)、[人工test](../../tests/management/test_research_c_pr_aware_df_mechanism.py)の3ファイルを追加した。旧科学コードは変更していない。

| 既存資産 | 適合性・再利用／今回呼ばない部分 |
|---|---|
| `c1_stage3_accuracy.aggregate_totals` | own-arm cornerにそのまま使用。crossed cornerはtargetが異なるため同関数のidentity guardを偽装せず、両identityを残す最小の記述的sumを追加した |
| `c1_stage1_execution.read/save/sha`、`c1_stage3_integrated.Calls` | 既存JSON I/O、call計上、partial evidence/STOPを再利用。古い承認証明関数は呼ばない |
| `c_pr_aware_df_stage_c`／`c1_stage3_connection.compose_serial_cost` | saved serial_costs26と2scenarioのplansをそのまま読み、価格・shell・Nを再生成しない。旧C1 executorは再起動しない |
| `c1_stage3_accuracy.assess_round/assess_energy`／`rpe_resource_accounting._hoeffding_shots` | 誤差単位・配分・元式をSOURCEで確認。M1では呼ばない。新shot式は実装しない |
| Stage A objective/fitter・B1・B2 | 目的/gradient、law、CAR、sampling、whole-kernel校正のSOURCEを調査した。M1では結果JSONの診断値のみ使う |
| `c1_molecule_extension.SectorCAR/rank_assessment/PF_rounds`、修正codec・H6 precision | 将来移送の再利用先。今回新しい分子演算・codec/fit評価をしない |

[未承認M1契約](research_c_pr_aware_df_mechanism_contract_unapproved_20261028.json)はscope `C_PR_AWARE_DF_SAVED_MECHANISM_M1_V1`、`status=UNAPPROVED/executable=true`。SOURCE7（新3＋再利用4）と実際に使う保存JSON9件をhash固定した。元NPZ・QPYを追加decodeする経路はない。契約・SOURCEのhashと人工検証は[技術証拠](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/technical/)に保存する。

| Logical call | 上限 | 内容 |
|---|---:|---|
| saved_bundle_load | 1 | 保存JSON9件のみ |
| identity_gate | 1 | 旧SOURCE/STOP、arm/export、scenario、serial価格coverageとpolicy |
| scenario_attachment | 2 | 旧2scenarioの保存N・適格性・仮定 |
| counterfactual_corner | 8 | 2scenario×4corner、各13round・X/Y・6metricのscalar集計 |
| descriptive_decomposition | 2 | ordered paths、交互項、対称配分、round群・累積 |
| tradeoff_table | 2 | armごとの保存診断値の対応表 |
| calibration_uncertainty | 2 | 保存SEと共通校正重み・共分散 |
| reconciliation | 1 | 元C1のself endpoint・Wへ整合するか |
| result_summary | 1 | 結論scopeと未資格null |
| **合計** | **20** | 新fit/PF/RTE/sampling/build/compile/shot式/追加保証0 |

Fresh outputは`artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1-run01`、retry/resume0、完了時mandatory STOP。別途明示許可後だけentryの`--execute`を使う。既存C1 call枠を再利用しない。承認証明・承認原文保存・新grantは不要。

SOURCE/input/来歴・arm・scope・coverageの不一致、非有限・負の価格、成功planに欠損/非整数N、元endpointの会計不整合はcorrectness FAILで依存処理を停止し、partial/failure/call/STOPを保全する。欠けた適格性はUNRESOLVED、未知値を0へ置換しない。負／ゼロの効果や仮説と逆の結果は有用な科学結果として保持し、候補を追加して探索しない。

Scalar会計の照合だけ`rel_tol=1e-12/abs_tol=1e-6`を用いる。大きな保存集計値の加算順差を検査するためのもので、旧DF/PF/phase/certificate allowanceやCA条件を変更するものではない。研究GOや統計的優位の閾値には使わない。

## 6．人工検証・実行規模・保全

**新30＋既存PR Stage C回帰23＝53 tests PASS、skip/deselect0**（2.09秒、ローカル実行）。人工13round・2scenarioを使った20-call E2E、既知の4corner/交互項/寄与、価格のみ・shotsのみ・差0・悪化、shell一回、非paired共分散、unknown/null、誤arm・SOURCE・coverage・誤反復数・偽資格、欠損/N不正/NaN/Inf、途中FAIL・partial/STOP・fresh拒否を確認した。実H4の数値を入力したテストではない。

Passive importはNumPy/SciPy/Qiskit/ffsim/FLINTをimportせず、preflightは契約/SOURCEのJSONと保存inputのbytes hashだけを読む。科学dataのdecodeや集計はない。実入力の将来gate結果・効果の正負は未取得であり、人工PASSを科学PASSに置換しない。

旧C1 SOURCE60と現在のファイルを照合すると、`rte.py`、`c1_molecule_extension.py`、`c1_stage3_reference.py`、`c_pr_aware_df_stage_b.py`の4件は今回の起点以前に更新済みだった。起点からの今回の変更はなく、旧manifest/rawはそのまま保全する。M1はこれらを呼んで旧PF/RTEを再生せず、保存スカラーだけを使う。今回再利用する4つのscalar/I/O/callモジュールは旧C1のSOURCEとも一致する。この版差を旧C1の再現実行・新しい誤差資格として扱わない。

独立worktree/task-local venv/cache、Python3.11.1、既存依存site-packagesはread-only参照。実行コードはstdlibと固定scalar moduleだけを要し、数値libraryを新規導入していない。実際のaffinity32・MemAvailableを技術記録へ保存した。今回の入力は小さいscalar表で、依存gate→集計→照合の順序とI/Oが主処理となるためprocess生成を追加しない。CPU/RAM/worker/wall上限は設けない。重いfit・compileを将来契約に入れる場合には改めて空き資源に応じた並列・既存加速を準備する。

旧科学SOURCE・契約・raw・grant・ledger・marker・STOP、F2-K1/F3-K1の原本を変更しない。SOURCE/input hash、Git差分、F2/F3原本hashを技術記録で照合する。今回の科学output M1-run01は作成しない。

## 7．アルゴリズムablationの比較（未採択・今回非実行）

既存目的は`Fγ=f/f*+γ(Mε/M*)²`。Tensor loss・gradient・Schur codec・qualified x0引渡しは既存実装を使う。新しいoptimizer、full RPE loss、Gaussian価格の微分器は実装しない。[初期設計§7](research_c_pr_aware_df_optimization_design_20261026.md)の固定U・凍結価格weighted mass案も確認したが、旧event価格の和がP0 whole-kernel価格と同一でない限界を維持する。

| 候補 | 問い・固定条件 | 必要な新fit／評価 | 交絡・不成功でも分かること |
|---|---|---|---|
| A：限定γ感度 | 同じH4/rank8/LD2/qualified x0、同じf*/M*/ε・optimizer options。追加γ .03/.3を候補として事前選択 | 最大追加2fit。DF/tensor/exact λの診断後、予算内の登録対象だけB1 PF/RTE。native単価の頑健性まで問う場合だけB2。M1後の別契約 | γが同時にJ/Uとlawを変える。固定x0だけの頑健性。λが減ってもPF/DF/価格が悪化し得る。旧γ0/.1はidentity一致時だけ保存再利用 |
| B：初期値依存 | 追加1個の独立初期値でγ0/.1を対にする。同じrank/LD・初期化政策・物理予算・停止規則 | 追加2fit。新x0の定義/seed/scale・初期DF gateを事前固定し、必要なB1/B2だけ別登録 | 列符号gaugeだけでは新しい物理初期値にならない。失敗も局所解・初期値依存の証拠であり、最適化法全体の不可能性ではない |
| C：cost-aware | U固定、同じJ/x0と共通の凍結非負価格。既存AD/solverを再利用 | 純tensor・unweighted mass・weighted massの固定Uで3fitを候補。先に適合価格calibrationが必要、PF/RTEとwhole-kernel nativeで独立検証 | 現行J/U可変結果との直接差をweighted効果と呼べない。event proxyは融合・shotsを表さず、λ増でAが悪化し得る。価格不足は0補完しない |

いずれも今すぐ採択・実行しない。**最初はM1一つ**に絞る。M1でshots側が大きければλ→A/phase-roomの頑健性を問うA/Bが候補、価格側が逆効果ならCやbasis/head構造を調べる動機となる。記述的な大小だけで因果メカニズムを決めず、追加γ・seed・rankを自動追加しない。

## 8．一般化と競争力の順序（設計案）

第一の移送候補は**H4・1.30 Å/STO-3G、CA=.00159を維持しgeometryだけ変更する小さい対比較**。現H4のγ・x0依存と対照の競争力を踏まえて別途採否を判断する。これは過去PRのtransfer資料との接点であって、今回のPR-aware最適化の移送成功を示すものではない。

| 候補 | 固定／変更するもの | 再利用と必要な新しい科学処理 |
|---|---|---|
| H4の1つの距離 | geometryだけ変更、basis/CA/PR/RPE/P0/費用政策は共通。新条件内で両armに同じ初期値・rank/LD政策 | `chemistry_hamiltonian.call_geometry`とPySCF RHF/ao2mo workflow、修正codec・objective・SectorCAR・B1/B2/C1を再利用。新integrals/paired fit/PF-RTE/必要native校正が必要。旧H4 J/Uや校正価格をそのまま移送しない |
| H4の精度条件 | 同じ保存H4とJ/U、CAだけ変える候補 | 新horizon/round誤差/Nの資格が必要。既存13roundを無条件外挿しない。Hamiltonian一般化の検証ではなくaccuracy-policy移送 |
| H6 | molecule/sector400・rank政策を変更し、両armで共通初期化/予算を採用 | 保存H6・Schur codec・SectorCAR400・PF/precision資産。旧決定論CDF14成功はPR-aware fit成功ではない。dense4096/controlled8192全unitaryを前提にしない |
| rank/LD | 探索を初回移送へ混ぜない。共通rank政策か明示的な相違を事前固定 | H4 rank8/LD2をH6へ機械移植しない。予算未達時は登録結果としてSTOPし、新rank/LDは別判断 |

新条件での初期化は同じ入力から共通J/U・qualified x0を作り、同じ規則でscaleを一度固定する。旧`c_pr_aware_df_pilot.prepare_worker`は保存H4/rank8、`c1_molecule_extension.specification/integrals`はH2/H6・1Åに固定されており、**旧runnerを引数だけ変えて再起動できない**。将来geometry/rank bindingの薄いadapterを新契約で準備する。過去PR2の設計資料は経緯として参照できても、その固有SOURCE・rank/law・held-out成功を今回の証拠として再利用しない。

## 9．今回可能になる判断・残る問い・次担当

| 研究判断 | 現在の証拠／M1で変わり得る点 |
|---|---|
| 1：tail mass以上の効果か | B1/B2は異なる診断量の改善例を支持する。M1は条件付き費用への寄与を整理するが、一般的有効性・因果性は未確定 |
| 2：shotsか単価か | 保存N/μと総費用だけでは重み付き寄与は未分離。M1の2つの順序・交互項・対称配分・高m表で記述的に答える |
| 3：強い既存手法にも競争的か | 未判定。M1では強決定論との新順位を出さない。現PR分解の決定論対照を同じ政策で用意する別実験が必要 |
| 4：γ目的を改善する価値か | M1がshots/価格の次の焦点を選ぶ材料になる。λ、価格、DF/PFのtradeoff次第で追加γ/初期値/固定U案を選ぶ。全部実行しない |
| 5：一般化へ進むか | 単一初期値・γ・geometryの限界は残る。小さい移送を優先し、強baseline比較と再現性の不足を明示して判断する |

**ユーザー／研究C**：本技術設計を確認し、M1の固定SOURCE・JSON9入力・旧2scenario・20-call範囲による保存値再解析を別途明示許可するか判断する。**Codex**：その許可後だけfresh M1-run01で一度実行し、実寄与・全round・6指標・不確かさ・STOPを公開する。新fitや移送、強baselineの新科学比較、重要な詳細レビューはその許可に含まれない。

今回はSOURCE・技術報告・契約・人工証拠をcommit/pushし、remote SHA/tree・SOURCE/input・保全を照合後、**mandatory STOP**。M1/C1再解析、追加保証、別分子・新しい科学計算へ自動進行しない。
