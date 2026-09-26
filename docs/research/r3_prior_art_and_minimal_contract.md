# R3先行研究監査と条件付き最小研究契約

日付: 2026-09-26
status: `r3_s0_complete_stop_r3_no_method_delta`

## 0. 結論

P-D S1事後再解析の次候補として、R3「固定DF表現の分割と部分ランダム化／決定論の選択」を
scopedに監査した。

広いR3、すなわち

> fragment norm、Product Formula誤差、実装費用、randomized remainderの負担を合わせ、
> 最も安い分割・公式・remainder方式を選ぶ

という問いは、現在のままでは独立した研究差分にならない。部分ランダム化自体、誤差と回路費用を
合わせたHamiltonian partitioning、摂動的誤差推定によるpartition選択、回路費用を意識したterm配分、
およびSPRINTの統合設計手順と強く重なるためである。

広いR3から差分を残せるか確認するため、次の狭い候補R3-Sまで監査した。

> 固定したordered DF fragment列と有限のsimulation候補集合に対して、安価な解析量から、
> 高価な共通詳細参照と同じ選択を**認証するか棄却して詳細評価へ戻す**selective ruleを作り、
> 未使用条件で選択regretを制御しながら詳細評価数を減らせるか。

しかし、certified multi-fidelity optimizationとbest-arm identificationは、異なるcostの近似評価、
選択費用、fixed-confidence identification、data-drivenなoptimization-error上界を既に扱う。現行資産には、
これら一般手法を越えるquantum-specificなbias boundまたは構造定理がない。R3-S0の三GO条件は全て
不通過であり、`STOP_R3_NO_METHOD_DELTA`とした。R3-S1、新しい対角化、RTE sampling、compileは
開始しない。

## 1. 今回の監査範囲

これはsystematic reviewではない。R3の一文主張に最も近い次の系統を、一次資料を入口として確認した。

| 系統 | 既に扱われている内容 | R3への含意 |
|---|---|---|
| partially random Trotter | 大きいHamiltonian項を決定論、小さい項をrandomにし、splitting biasとsampling varianceを釣り合わせる | deterministic/random split自体は既知 |
| phase estimation向けpartial randomization | 一部を決定論、残りをsampleし、単一ancilla phase estimationの詳細resource estimateを行う | PRの優位領域とresource比較だけでは不足 |
| Hamiltonian partitioning | fragmentの選択によるTrotter errorとT-gate costの交換関係を比較する | 誤差と実装費用の同時比較は既知 |
| perturbative error estimator | norm boundでなくeigenvalue errorを予測し、time step・partitionを選ぶ | 安価な誤差予測によるpartition選択は既知 |
| circuit-aware term allocation | commuting groupへtermを部分配分し、greedy algorithmで非Clifford costを下げる | 実装費用を用いた配分最適化は既知 |
| SPRINT/GRADE | factorization、Trotter/leakage error、per-step cost、norm-based grouping、near-integrable formula、randomization、remainder strategyを統合して選ぶ | broad R3に最も直接的に重なる |
| multi-fidelity optimization | 安い近似と高価な参照へ評価資源を適応配分してbest armを同定する | 「詳細評価を半分にする」だけでも一般的方法論と重なる |

特にSPRINTは、fragment normに基づくgroupingと再配分、BCH/Trotter error estimatorによる
near-integrable formula選択、error reductionと実装費用に基づくremainder strategy選択を一つの手順として
明示している。このため、既存のR3記述にある「$\lambda_R$、PF誤差、決定論・random componentの
実装費用からsplitを選ぶ」だけでは差分にならない。

一方、今回確認した量子simulation資料だけからは、ordered DF prefixについて、安価な診断の適用可能性を
条件付きで認証し、曖昧な場合は棄却し、高価な詳細評価数とselection regretを同時に評価する特定手順が
既に同じ形で示されているとは確認できなかった。しかし、この問題設定そのものは一般のmulti-fidelity
best-arm identificationに近い。量子simulation固有の構造または保証がなければ、単なる既知optimizerの
適用例に留まる。

## 2. R3の採否

### 2.1 No-Goとする広い主張

次はR3の新規性として主張しない。

- fragment normや$\lambda_R$だけでdeterministic/random splitを作る。
- PF errorとper-step action/gate proxyを足して最安候補を全探索する。
- H4の$L_D$別cost曲線またはPR有利領域を描く。
- SPRINT型のerror--cost手順を固定DF prefixへ移しただけの比較を行う。
- 一般のmulti-fidelity optimizerを使い、評価回数が減ったことだけを成果とする。

現行のRQ3は、将来の総resource評価に必要な**工学的目的**としては残る。しかし、その達成だけを
独立した論文上の新規性とは扱わない。

### 2.2 条件付きで残すR3-S

R3-Sの候補差分は、単なるsplit最適化ではなく次の三点の組合せに限る。

1. ordered DF prefixとpartial-RTEに固有の、誤差・tail・実装構造を分離した安価な診断を使う。
2. 診断を常に信じず、適用範囲外または候補差が小さいときは`abstain`して詳細評価へ戻す。
3. 未使用条件で、選択regretと高価な詳細評価数を同時に評価する。

この一文差分がclosest prior artに対して残ることをR3-S0で確認できなければ、R3-Sも停止する。

## 3. R3-S0：計算前の新規性gate

### 入力

- 本文書で確認した量子simulationのclosest work。
- multi-fidelity best-arm identification、Hyperband/successive halving、multi-fidelity Bayesian
  optimizationの一次資料。
- 本リポジトリの既存PF、RTE、compiled-cost surrogateの保証範囲と既知の破綻例。

### 必須の差分

次の全てを一文で特定できることをGO条件とする。

1. 一般multi-fidelity法へ渡すだけでは得られないquantum-simulation固有の構造または誤差証明。
2. SPRINTのnorm/error/cost手順と異なる選択、認証、または安全な棄却機構。
3. 既存のpartition error estimatorと異なる、partial-RTE tailまたはordered prefix固有の情報。

一つでも欠ける場合は`STOP_R3_NO_METHOD_DELTA`とし、R3-S1を実行しない。

## 4. R3-S1：GOの場合だけ使う最小検証契約

これは条件付き契約であり、現時点の実行指示ではない。R3-S0を通過した場合も、実行前にsource hash、
task manifest、候補集合を別の事前登録artifactへ固定する。

### 4.1 問い

> 固定ordered DF fragment列の有限候補について、事前固定したcheap diagnosticから、共通詳細参照の
> 最良候補を5%以内のregretで選ぶか、判断不能として棄却できるか。また、候補を全て詳細評価する場合に
> 比べ、高価な詳細評価数を少なくとも半分にできるか。

### 4.2 development scope

- H4 linear chain、1.0 Å、STO-3G、8 qubit、4電子、固定DF rank 12。
- 同一Hamiltonian snapshot、fragment順、係数threshold、時間、誤差予算、cost scopeを固定する。
- eligible prefixは$0\leq L_D\leq12$とするが、過去に参照したprefixを新しいblind evidenceとは呼ばない。
- H4の既存結果はretrospective developmentと機構診断だけに用いる。
- H12、長RPE、noise/backend、最終compiled総costは対象外とする。

### 4.3 候補と公平性

- deterministic endpoint、fully randomized endpoint、partial-randomized interior prefixを同じ候補表に置く。
- 各候補へ同じ許容PF family、$\delta$、内部精度、RTE cutoff/allocationの再最適化機会を与える。
- tailなし候補は$R=0$、normalization 1の専用経路を使い、存在しないrandom costを課さない。
- high-fidelity referenceは全候補に共通のfinite-error・realized-work定義を使う。
- 現在のcomponent-action/B4は詳細**モデル参照**であり、実測回路または最終総costとは呼ばない。

### 4.4 比較baseline

最低限、次を同じholdoutへ適用する。

- exhaustive high-fidelity oracle。
- normまたは$\lambda_R$だけの単純split規則。
- SPRINTに対応するnorm + error + per-step cost規則。
- 一般のmulti-fidelity/best-arm allocation baseline。
- 提案するquantum-specific selective rule。

提案法が一般baselineと同じ選択・評価数になる場合は、独立方法としてGOにしない。

### 4.5 cheap diagnosticの上限

特徴量は結果を見る前に少数へ固定し、少なくとも次の区分を混同しない。

- deterministic PF phase/eigenvalue error proxy。
- tail係数L1と符号付きPF係数から得るabsolute tail time。
- deterministic component actionとrandom event costの分布。
- finite cutoffまたはinteger allocationがleading modelから外れることを示す診断。
- 既存compiler proxyの適用domainと不確かさ。

多数特徴の事後回帰、holdout結果を見た閾値調整、候補ごとの別目的関数は禁止する。

### 4.6 holdoutとdata leakage

- 既存H4 prefixはPFまたはcompiled-cost検証で既に広く参照されているため、未使用分割と自動的に呼ばない。
- 事前登録前にevidence inventoryを作り、development、既参照、真のholdoutをpath/hash付きで分類する。
- 真のholdoutが残らない場合、retrospective pilotとして閉じる。独立検証には、結果未参照の固定snapshot、
  geometry、または表現を別途事前固定する。
- holdoutの詳細参照を見た後にrule、feature、閾値を変更した場合、その条件はdevelopmentへ戻す。

### 4.7 主判定

GOには次を全て要求する。

1. certified decisionのhigh-fidelity regretが5%以下。
2. certificationが誤ったfeasibility acceptanceを起こさない。
3. 高価なhigh-fidelity評価数がexhaustive比50%以下。
4. `abstain`を含む判定を事前規則どおり返す。
5. SPRINT型規則と一般multi-fidelity baselineの両方に対し、quantum-specific部分の追加価値を示す。

単一decisionでこれらを満たすことはmechanism pilotに過ぎず、一般化証拠ではない。論文上の主張へ進むには、
事前固定した複数の未使用decision contextで再現する必要がある。

### 4.8 STOP条件

- R3-S0の一文差分が成立しない。
- cheap diagnosticだけでは安全な認証ができず、ほぼ全候補の詳細評価が必要。
- 5% regretを守るにはholdout後の閾値調整が必要。
- SPRINT型または一般multi-fidelity baselineと実質同じ。
- H4既存結果をblindと呼ばなければ結果を支えられない。

この場合、追加$L_D$ gridでR3を延命せず、R6またはR8も別RQとして新規性から再評価する。

## 5. 直近の作業境界

R3-S0は完了し、`STOP_R3_NO_METHOD_DELTA`となった。R3-S1、数値pilot、追加$L_D$、H12、
長RPE、full compile、別分子への展開は開始しない。

R3は主研究候補から外す。現行RQ3は将来の工学的resource accounting目的としてのみ保持する。
次に進む場合は、R6/R8へ自動移行せず、新しいRQを先行研究差分から改めて設計する。

## 6. 一次資料

1. Jin and Li, *A Partially Random Trotter Algorithm for Quantum Hamiltonian Simulations*,
   [arXiv:2109.07987](https://arxiv.org/abs/2109.07987).
2. Günther et al., *Phase estimation with partially randomized time evolution*,
   [arXiv:2503.05647](https://arxiv.org/abs/2503.05647), PRX Quantum 7, 020332 (2026).
3. Martínez-Martínez, Yen, and Izmaylov, *Assessment of various Hamiltonian partitionings ...*,
   [arXiv:2210.10189](https://arxiv.org/abs/2210.10189), Quantum 7, 1086 (2023).
4. Mehendale et al., *Estimating Trotter Approximation Errors to Optimize Hamiltonian Partitioning ...*,
   [arXiv:2312.13282](https://arxiv.org/abs/2312.13282).
5. Mukhopadhyay, Wiebe, and Zhang, *Synthesizing efficient circuits for Hamiltonian simulation*,
   [arXiv:2209.03478](https://arxiv.org/abs/2209.03478), npj Quantum Information 9, 31 (2023).
6. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*,
   [arXiv:2606.30741](https://arxiv.org/abs/2606.30741)（SPRINT/GRADE）。
7. Li et al., *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*,
   [JMLR 18(185)](https://www.jmlr.org/papers/v18/16-558.html), 2018.
8. Poiani et al., *Optimal Multi-Fidelity Best-Arm Identification*,
   [arXiv:2406.03033](https://arxiv.org/abs/2406.03033).
9. Fan et al., *Multi-fidelity Bayesian Optimization with Multiple Information Sources of
   Input-dependent Fidelity*, [PMLR 244](https://proceedings.mlr.press/v244/fan24a.html), 2024.

文献の不存在を証明したものではない。以下のR3-S0完了判定は、このscoped auditと現在の
project資産に基づく停止判断である。

## 7. R3-S0完了判定

### 7.1 追加で確認したclosest method

条件付きR3-Sの中心語である「安価な近似評価」「高価な真値」「選択の認証」「評価費用の削減」を
一般方法論側から追加監査した。

- Certified Multi-Fidelity Zeroth-Order Optimizationは、異なるcostの近似評価を用いて目的関数を
  最適化し、data-drivenなoptimization-error上界を出力するcertified algorithmを既に定式化する。
- Multi-Fidelity Best-Arm Identificationは、armごとに低精度・低cost評価を選び、高精度でのbest armを
  fixed confidenceかつ低costで同定する問題を直接扱う。cost-complexity lower boundと整合するalgorithmも
  既に提案されている。
- Multi-Fidelity Multi-Armed Bandits Revisitedは、異なるcostとaccuracyを持つfidelityの選択と、
  fixed-confidence best-arm identificationのcost upper/lower boundを扱う。
- quantum simulation側でも、摂動領域のrigorous error boundを効率的な古典cost functionへ変え、
  Hamiltonian simulation sequenceを事前最適化する方法が既にある。

従って、`abstain`、5% regret、詳細評価半減を導入すること自体は方法論上の差分にならない。

### 7.2 三つのGO条件の照合

| R3-S0 GO条件 | 照合結果 | 判定 |
|---|---|---|
| 一般multi-fidelity法だけでは得られないquantum-specificな構造または誤差証明 | 現行のPF/RTE/compile proxyはlocal empirical modelで、cheap/high-fidelity差をprefix横断で拘束する新しい上界がない | fail |
| SPRINTのnorm/error/cost手順と異なる選択・認証・棄却機構 | selective evaluationとcertificationは一般multi-fidelity研究が直接扱う。SPRINTとの差は適用対象の狭さだけで、固有機構を固定できない | fail |
| partition error estimatorと異なるpartial-RTE/ordered-prefix固有情報 | $\lambda_R$、absolute tail time、finite cutoff、integer allocationは特徴量として存在するが、それらをcertified selectionへ結ぶ新しい関係を得ていない | fail |

内部証拠もこの結論を支持する。S1事後再解析では、選択近傍でB2/B4は一致した一方、全候補では
B2受理/B4不適格がcombined 45件、比較可能候補のobjective相対差の最大絶対値が約77.15%だった。
また既存compiled-cost proxyは$L_D=6$などで既知の破綻例を持ち、rigorousな5%保証ではない。
これらはlow-fidelity sourceが有用でないという意味ではないが、certificationを名乗るには候補別bias bound
またはcoverage保証が別途必要である。その保証自体を現在の資産から導く一文差分は固定できなかった。

### 7.3 Gate結果

R3-S0は`STOP_R3_NO_METHOD_DELTA`と判定する。

- R3-S1の数値pilot、source-hash付きtask manifest、追加$L_D$ gridは作らない。
- 既存H4結果をblind evidenceへ読み替えない。
- R3は主研究候補から外す。RQ3は将来のsystem engineering/resource accounting目的としてのみ保持する。
- R6またはR8へ自動移行しない。次は新しいRQを、先行研究差分から再設計する。

このSTOPは、将来quantum-specificなbias boundや構造定理が得られる可能性を否定するものではない。
現時点の計画と証拠では、それを独立研究として約束できないという判断である。

### 7.4 追加一次資料

10. de Montbrun and Gerchinovitz, *Certified Multi-Fidelity Zeroth-Order Optimization*,
    [arXiv:2308.00978](https://arxiv.org/abs/2308.00978), SIAM/ASA Journal on Uncertainty
    Quantification 12(4), 2024.
11. Wang et al., *Multi-Fidelity Multi-Armed Bandits Revisited*,
    [arXiv:2306.07761](https://arxiv.org/abs/2306.07761).
12. Mansuroglu, Fischer, and Hartmann, *Problem specific classical optimization of Hamiltonian
    simulation*, [arXiv:2306.07208](https://arxiv.org/abs/2306.07208), Phys. Rev. Research 5,
    043035 (2023).