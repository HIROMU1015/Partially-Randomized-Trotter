# PR–RPEのRMSE型Nモデルと共通κ schedule（2026-11-12 JST）

**論文v2の経験的RMSE資源モデルを旧Hoeffding評価と別policyで追加し、共通paper κからラウンド別event構成を生成する入口を実装した。** Bが作業中に公開した共通interface草案 `d562bc38d3b63df51f775aa4eb0998afbf1873e2` を確認し、共有mapperと必須fieldへ接続した。実Hamiltonianの新signal/N、保存値の数値再解析、P2/P3、sampling/compile、価格/G/κ最適化、詳細科学レビューは0。

成果：[module](../../../src/trottertracks/pr_rpe_rmse.py)／[人工例専用CLI](../../../scripts/demo_pr_rpe_rmse.py)／[人工テスト](../../../tests/check_pr_rpe_rmse.py)／[ARTIFICIAL_EXAMPLE](ARTIFICIAL_EXAMPLE.json)／[HANDOFF](HANDOFF.md)／[CHECKS](CHECKS.json)／[SOURCE_INDEX](SOURCE_INDEX.json)。Aの既存branch `codex/pr-r-dependent-n-model-20261111`を継続し、B/Cのworktree・実装を変更していない。

## 原論文v2で確認した定義

一次資料は [arXiv:2503.05647v2 PDF](https://arxiv.org/pdf/2503.05647v2)。Appendix B.1/B.2、E.1/E.3とA.2/A.3を確認した。今回扱う原論文の主な式は次の通り（式の転記・定義確認であり、新しい分子数値ではない）。

- m=0..M、Mは最大index、round数はM+1。Xm/Ymを各Nm回測り、1 roundのHadamard testは2Nm回。
- B.2の経験的配分は Nm=ceil(B²[11+4(M−m)])。E.3では共通上界 B=exp(1/κ)を使う。B.2で述べる厳密解析側のD≈4.11と、資源推定用の経験的D=4を区別する。
- 経験的な最大時間との関係は、PF側ではεqpe≈.08π/Tmax、RTE側では≈.10π/Tmax。hを明示するとTmax=h·2^M、候補Mはceil(log2(Cmaxπ/(h εqpe)))に対応する。今回の実装は供給Mを検査し、Mを自動探索しない。
- E1/E23の誤差会計はε²=εqpe²+εtrot²。QPE推定が有効Hamiltonianのエネルギーに対して不偏であること等が前提であり、finite-Kのbias・保存state・branch失敗へ無条件に適用できない。
- E21のr=κpaper λR²T²はランダム部分の抽象Pauli回転総数。κはconfiguration全体で共有し、round別の独立R選択にはしない。

以上の定数は論文の経験的資源推定であり、現在のfinite-K・DF10/DF19・保存数値stateについての厳密十分shot数ではない。論文のRPE理論（B.1/B.2）は、早期roundでのbranch errorと最終角度誤差の二乗を扱う。旧αtotal=.05のunion-bound条件をRMSE条件に読み替えていない。論文のoverlap不足量α、B.1の減衰指数αも旧失敗確率αtotalとは別。

E22の費用は概念的にexp(2/κ)(A+Bκ)という形を持ち、定数A/Bを固定すれば停留条件はBκ²−2Bκ−2A=0。原論文の解析的κ最適値は、現在のcompiled価格の最適κではない。今回はκ optimizer・係数fit・数値最適化を実装/実行していない。

## 原論文rとfinite-K eventの対応

`rte.py`のpaired eventは、偶数次数nのPauli積n個と、角度atan(τ/(n+1))の抽象回転1個からなる。K2ではn=0/2で、選ぶcomponent数は1/3だが、抽象回転はどちらも1個。原論文A.2のr回転と実装のevent数は、同一のHermitian involution分解・λR・scalar分離・sampling lawに対応する場合に、この単位で結び付く。一般の非involutory DF fragmentを丸ごと1個のPauli回転とみなしていない。DF componentはbasis変換付きinvolutionであり、元の計算基底で単一Pauliであるとは仮定しない。そのbasis/control実装の費用はC側で扱う。

static native SOURCE `h46_long_time_native.sample_requests`も各signed pieceのevent数をrte_stepsに照合し、composeはそのtail sequenceを一回ずつ使う。よって今回のmapperは全event数E=qPF R_outerを抽象回転数の整数実現として保持する。ただしcompiled RZ、basis change、Pauli積の実装、full measurement wrapperの価格とは別単位である。2測定軸やA.4のdeterministic半時間化からevent数へ追加の2倍/半分を入れない。物理wrapperの価格資格は従来のfull identity/profileに従う。

共有定義IDは `PAPER_V2_E21_KAPPA_ROTATIONS_OVER_LAMBDA_R_SQUARED_TIME_SQUARED`。B公開草案と同じcore mapper `PAPER_TOTAL_UNIFORM_OUTER_DRAFT` を供給する。

```
T_m = h 2^m
r_nominal = κpaper λR² T_m²
r_target = ceil(r_nominal)
R_outer,m = max(J, ceil(r_target/qPF,m))
E_m = qPF,m R_outer,m
(r_j) = 既存ABS_TIME_PER_OUTER min-one/largest-remainder配分
```

ceil(ceil(x)/q)=ceil(x/q)（q正整数）なので連続r_nominalから直接outer数へ丸める式と同じ整数候補になる。正のκ/λ/Tを要求し、κ=0の特異なnormalization式を実行しない。Fractionによる整数境界を使い、元allocationでΣr_j=R・r_j≥1を検査する。最小event数はqPF Jで、r_nominalより多い回転を実行する場合も明記する。極端値で元binary64 allocationや正規化指数を扱えなければ、数値未解決として止める。恣意的なR/κ上限は設けない。

**H6の11 signed pieceには追加の注意が必要。** A38のnormalization上界はqPF λR² δ² Σ_j(w_j²/r_j)であり、絶対時間長と整数配分に依存する。E22のS2では正時間だがm5_bestには負時間がある。r=κ λR²T²という回数規則だけではB≤exp(1/κ)を導けない。κに(Σ|w|)²を黙って乗算したり、normalization上限型κ_Bへ置換したりしない。実finite normalizer/信号の資格を別に確認する。

## 初期roundとPF時間grid

現在のH4/H6へ接続するdesign_from_taskは、旧taskから物理構成・h・prior・decoder・H/state/profile/sourceをcopyするだけで、旧Nは読まない。H4のS2/δ=.1/prefix1/K2、H6のm5_best/δ=1.6/prefix7/K2を保持する。PF対照はH4 B0/p7、H6 Full19。旧horizon M12・α/β・energy_precisionを新RMSE policyへ自動採用しない。

H6 m0..3は保存qPF1・δ=T・PR prefix5/R32/K2とPF Full19をそのままcopyし、κ_applied=falseとする。prefix7のλをこれらのroundへ使わず、実際のmean/normalizerはprefix5の保存bindingから供給する必要がある。

regular roundは固定δを保つためqPF=T/δが正整数であることを要求し、非整数なら拒否する。B草案の一般則ceil(T/δpf)は実幅を変えるが、現在のdyadic通常roundでは両者が一致する。Bへは `pf_grid=EXACT_FIXED_DELTA_WITH_EXPLICIT_SAVED_EARLY` と、H6用 `PAPER_TOTAL_UNIFORM_OUTER_SAVED_EARLY_V1`、元early overrideとSOURCEを明示する。Bのdefault generatorに暗黙変換させない。

## RMSE評価器の仕様

新policy IDは `PR_RPE_PAPER_V2_EMPIRICAL_RMSE_V1`、familyはB共通の `PR_PAPER_RMSE`。旧高確率型Hoeffding API・P1のN packet/結果/SOURCEは変更しない。新RMSE requestは全schedule・M/h/κ・tail norm SOURCE・physical recipe・policyをconfiguration IDに含め、同じ丸め済Rでも異なるκ/scheduleを混在できない。

1. rmse_policyへtarget/qpe/evolutionのRMSE budgetとruleを明示する。paper_quadratureの数値制約だけで適用仮定を真にしない。Minkowski型の線形予算も明示選択できるが、今回どちらも分子へ採用しない。
2. scheduleへM/共通κ/外側design/λR bindingを渡す。canonical candidate ID、signed allocation、qRPE/qPF、round別のκ適用有無を生成する。候補生成成功は適格性ではない。
3. 既存action-only finite meanをcallbackで供給する。existing_finite_meanは保存ctx/half-step/referenceを前提に旧action/independent-reference gateを再利用し、referenceの新計算へfallbackしない。旧Hoeffding N solverやGを呼ばない。
4. normalization ruleは明示二択。`paper_kappa_envelope`はlog B=1/κのnominal配分を使うが、供給finite logBが上界を超えた候補を不適格とする。`finite_B_diagnostic`は各roundの既存finite logBを使う経験的plug-inで、論文の校正が成立したとは扱わない。両者は別identityである。
5. Nは各軸ceil(exp(2 logB_used)[11+4(M−m)])。Decimalの2精度でceil安定性を検査し、巨大整数をfloat化しない。この数値診断もrigorous上界ではない。旧NやA0のshot係数を新Nとして再ラベル/再乗算しない。
6. state/effective evolution、alias/decoder、involution mapping、finite-K→RMSE移送、誤差合成、actual lawをSOURCE付き適用条件として保持する。未知はUNQUALIFIED、反証済はINAPPLICABLE、未計算はNOT_EVALUATED、数値不能はNUMERICAL。全入力が報告されても経験的モデルであり、scientific_PASS_FAILはnull、rmse_guaranteed=false。

`nominal_N_X/Y`がモデル値。`sufficient_N_X/Y`と`actual_N_X/Y`はnullのまま。B packetのN_X/Y・supplied_rmse_bound_haも未資格としてnullで、旧高確率型Nと合流しない。現時点でB costへ適格な実Nを供給したという意味ではない。

decoderは既存reconstruct_rpe_phaseを使用し、φ=arg(z)=−ET、E=−φ/hを維持。zero座標・branch tieを未解決とし、供給座標の復元成功からalias資格・エネルギーRMSE保証を作らない。実測/成功率simulationは行っていない。

## 再利用と変更範囲

| 既存SOURCE/関数 | 再利用/変更 |
|---|---|
| pf_pr_rpe_pipeline：candidate_id/structure/identity | 純metadata処理を再利用。旧validate/高確率N/Gは不変 |
| h46_m5_plan：weights/allocation | 元11 signed pieceと整数tie規則のみ。旧SYSTEMSのH4 DF12設定は使わない |
| rte：finite_rte_distribution/finite_taylor_operator | 旧actionを経由して再利用。paired eventの一回転という定義も確認。変更0 |
| h6_df19_rpe_d2：evolve/half-step/reference | 科学leafを再利用するadapterを追加。real call 0、人工2-levelだけ確認 |
| h4_df10_rpe_batch_provider：evolve_h4 | H4 B0のscalar規則を保持するrouteを再利用。real call 0 |
| pr_r_detailed_n_p1：checked_reference | 数値pairの既存gateを再利用。RMSE資格としては扱わない |
| rpe_four_round_phase_validation：reconstruct_rpe_phase | 最近branch decoderをそのまま利用。人工座標のみ |
| 旧RBatch/analyze、Hoeffding、補正済N_HANDOFF | 不変・新policy内でN solverを起動しない |
| 新pr_rpe_rmse / demo / tests | schedule・モデルN・B草案packetの最小追加。optimizer/C/Gなし |

## 人工検証とB接続

[TESTS.txt](TESTS.txt)：**17人工/mockテストPASS**。Fraction整数境界、event最小値・signed allocation、H6早期4round、M=0/13・qの区別、非整数PF grid、極端κ/数値範囲、Nm per-axis/normalization、policyとconfiguration分離、未知/不適格、2-level action、既存decoder、旧SOURCE保全、B草案必須field・packetを確認した。人工例はTOY bindingとSYNTHETIC_ONLYで、実H4/H6のλ・信号・保存Nを使った再解析ではない。作者自身の技術確認で、独立科学レビューではない。

B公開先端は調査開始時7dd61438b、その後公開されたd562bc38dを最終interface基点とした。core回数mapperは同一式、B必須configuration/N_packet fieldを人工照合済み。Bのgenerate/assess本体はこの草案pinでは未公開なので、B生成identityとの完全一致・実RMSE error contractの採用は未完了。PF/PR別packetへ全schedule IDを付け、未知bound/Nはnull、人工packetはSYNTHETIC_ONLY。FULL_H_RTEは今回未実装であり、one-bodyを保持するDF B3へ付け替えない。[HANDOFF](HANDOFF.md)を参照。

## 未確定事項・最小の次実行案

未確定は、実行するM/共通κ、RMSE targetとQPE/時間発展誤差予算、finite-Kのbiasが全decoder RMSEへ与える影響、保存stateと有効PF固有状態の関係、H6 signed normalization、早期overrideのRMSE校正、actual law/implementationと全schedule資格。旧α=.05/P1 PASSをこれらの代用にしない。

**最小の科学pilot案（未承認・未実行）**：ユーザーが1系・固定外側条件・1組(M,κ)とRMSE budgetを指定し、H4ならm0/m6、H6ならm4/m8の2時点に限定する。生成candidate IDが一致するP1 signal/referenceは保存値を再利用できるが、新policyでの数値評価は別の科学許可が必要。欠測mean/内部対照は最大2候補、新referenceは必要な場合だけ最大2時点×2経路を承認範囲へ明記する。新RMSE診断は2round×X/Y、C/G・R最適化は含めない。このpilotだけで全energy RMSEを認定しない。

その後の全schedule RMSE検証には、選んだM/κの全round mean/normalization、state/target/alias契約、および実finite lawに基づくdecoder統計の科学検証が必要。反復数・seed・誤差/CI条件は将来固定し、今回勝手にsimulationを始めない。C側も同じR/q/配分・compilerの価格domainが必要で、Nだけでκを採択しない。

**Codex B**は公開A mapperとpacketを草案controllerへ接続し、外側/全schedule単位のidentity照合を行う。**ユーザー**は上記有限科学scopeと未確定条件を判断する。**Codex A**は明示承認後だけ保存再解析/新signal・RMSE検証へ進む。今回はcommit/push・remote SHA/tree/成果物確認後mandatory STOPする。
