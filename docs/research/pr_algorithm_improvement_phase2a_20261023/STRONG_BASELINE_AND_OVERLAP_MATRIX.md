# 強いbaseline・過去研究との重複

同情報、同じ元H/target、精度/失敗確率、controlled相対位相、合法な回路共有、情報取得費をそろえる。表現が異なるL_Dは同一の分割量ではない。baselineの名前が同じだけで比較可能とはしない。

|課題|既に扱われた範囲|必須の強い比較対象|残る区別／公正性|
|---|---|---|---|
|RQ-01|REP-03, REP-04, REP-08, REP-09, QC-05, QC-08, REP-07|同じ入力情報・sector・model誤差枠を持つcollected Pauli、既存DF/圧縮DF、適用できる既知orbital/grouping・BLISS・SPRINT/GRADE系。各表現にdiscard/full deterministic/partial/random endpoint、同じ合法fusion/control/compilerを許す。外部論文の別task費用を数値対照として流用しない。|同じ元Hamiltonian・sector・状態アクセス・要求精度を保つとき、表現／軌道／可解coreの変更は、normalizationやPF誤差の低下を上回る基底移動・controlled生成費を払わず、PRの測定込み資源を下げられるか。DFを出発時点の必須形式とせず、実行可能な表現の成立条件を問う。|
|RQ-02|PR-01, EVAL-01, OPT-03, OPT-04, OPT-05, RB-01, RB-02, QC-05|PR原法/composite partition、SPRINT/GRADE、同じ候補domainとDF backendを持つcompact BCH再帰・cache、PR-2近接discardを含むendpoint。全方式へ同じ情報・探索機会・精度保証とlegal circuit最適化を与える。prefixだけに制限した対照を最強としない。|単項weight順prefixより自由な決定論／random分割やgroupingは、同じ情報を使う既知のpartition・compact BCH・cost評価にも許したとき、PF誤差・tail統計・実装境界を通じて残る費用差を作れるか。一般探索を広げることと独立した方法差を区別する。|
|RQ-03|CIR-01, CIR-02, PR-01, RB-12, RB-13|full-basis共有、既存support_run_le_1、one_segment_per_source_run、既存diagonal-only control、同block boundary fusion、全方式共通actual-adjoint/literal inverse cancellation、適用可能な既知partial Gaussian compiler。WとZを別回路にする弱い対照を強制しない。|同じordered event列とcontrolled coherent meanを保持し、既存run-level full/support選択・同basis相殺・step境界融合を与えた後、異basis間やwrapper全体の生成に削減可能な大きい費用が残るか。何を共有できるかの構造条件を問う。|
|RQ-04|RB-18, RB-14, RB-06, RB-09, RB-16, RB-17, RB-15, PR-02|同じ情報での既知whole-path IS/cost-aware IS、canonical ordinary/partial/P3+tail、同finite targetのGeneral Full/Closed P5/collected CTS。全方式へ同じ合法cache・selector・前処理を許し、全trialの量子測定と古典生成を含める。branch相関でtargetを変える方式は同mean対照と混ぜず別のbias契約を必要とする。|同じcontrolled first momentを生成するlawについて、非加法なwhole-path量子費用と、weightのmoment/range・bias・有限bit・古典生成／取得費を含めても、既知ISとcanonical RTEを超える実行可能な節約条件はあるか。最適分布の式と安い実装可能性を分ける。|
|RQ-05|RB-03, RB-09, RB-13, RB-14, RB-17|common/非commonと同じ既知ISの機会、ordinary/partial/P3+tail/Full/Closed P5/CTS、exact cheap角とactual-adjoint、全方式同じliteral inverse cancellation、適用可能な既知probabilistic synthesis。native保証とprepare/readoutの費用枠を一致させる。|同じfinite target・精度を生成する回転辞書／allocation／合成配置の選択に、既知Euler pairing・cost-aware IS・共通fusion後も残る資源機構はあるか。加法primitive価格での利益が、全wrapper・複数step・準備費の条件で残る範囲を問う。|
|RQ-06|OPT-01, PR-01, EVAL-01, RB-15, PF-05|固定表現/splitに既存round別r/K・beta/alpha/shot配分を最適化した対照、同じcost/error情報の既知adaptive/multifidelity運用。真のE0やexact spectral oracleを候補だけへ与えない。|時刻／roundごとに表現・rank・splitが異なる信号を、同じ元Hの絶対energy taskへ接続し、既存round別r/K・beta/alpha配分を超える費用節約を得られるか。変更した有効位相・状態overlap・branchをどの情報で整合させられるかを問う。|
|RQ-07|CIR-01, PR-01, REP-08, RB-17|適用可能な既知Gaussian/fermionic mappingとrouting/compiler、同じgraph・layout探索budget・control・fusion・精度を持つPauli/DFのPF/PR endpoint。Kivlichanらのfermionic swap network/Givens準備も適用classでは比較候補。ただし同論文の表現前提を任意DF-PRへ自動移送しない。どの既知法を必須とするかはPhase 2Bで判断する。|明示した接続性と同じmapping/routing最適化を全方式に与えたとき、PRのbasis移動・演算子分割・表現の選択は、既知compilerへの単純接続を超える資源差を作れるか。|

## 前回10候補の再評価

各候補の8論点を省略せず記す。統合・分離は課題の重複を整理するためで、採否・順位ではない。

### OP-I1 → RQ-03

**1. 過去研究の繰返し：** 基底共有・literal相殺は既存。P-A interval固有差は否定的。

**2. 具体的失敗条件を変えるか：** run共有を壊さない必要がある。追加分割のplan同一を越える機構が未提示。

**3. 強いbaselineでも同じ改善があるか：** diagonal-only control・同basis/step融合・既知Gaussian compilerを含める。

**4. 費用寄与は重要か：** 短列局所削減は観測。full wrapperのbasis寄与内訳は未取得。

**5. 情報・前処理は実行可能か：** registry/planを利用可。追加compiler/情報取得費は未測定。

**6. 同H/task/精度にできるか：** 同event列で比較可能。ただし位相を含む同値性が必要。

**7. 何が未解決か：** 回路生成と費用寄与が未解決。

**8. 既存証拠で問いを定義できるか：** 既存成果の再利用として保持し、残る増分の有無を問う。

### OP-I2 → RQ-04 / RQ-05

**1. 過去研究の繰返し：** Closed P5はRA内部特殊化。G9とH4 C1移植は既に実施。

**2. 具体的失敗条件を変えるか：** C1ではparentがほぼ空でもchild回転が残る。条件差の根拠が要る。

**3. 強いbaselineでも同じ改善があるか：** General FullだけでなくClosed P5/CTSと同じtarget・情報で比較。

**4. 費用寄与は重要か：** G9の条件付きT機構は保存。H4のlogical差は小さくnative/N未取得。

**5. 情報・前処理は実行可能か：** P5 setupとonlineを分ける。C1 CPUはcache/境界の交絡あり。

**6. 同H/task/精度にできるか：** 同finite targetなら可能。G9とC1のN/費用を混合しない。

**7. 何が未解決か：** アルゴリズム移送、finite-bit、回路、統計、古典費の接続。

**8. 既存証拠で問いを定義できるか：** 単純移植を独立課題にせずsampling実行可能性とnative機構へ分離。

### OP-I3 → RQ-01

**1. 過去研究の繰返し：** Pauli/DF四方式はH2固定比較済み。H4は準備資料あり。

**2. 具体的失敗条件を変えるか：** norm低減を基底/central費が相殺する既存例を越える必要。

**3. 強いbaselineでも同じ改善があるか：** collected Pauli・DFと各PF/PR endpoint、同control/fusionを許す。

**4. 費用寄与は重要か：** 同じHでも表現で1-shot費用が違う証拠。大系の寄与は不明。

**5. 情報・前処理は実行可能か：** 入力identity基盤は利用可。表現取得費は無料にしない。

**6. 同H/task/精度にできるか：** 元H/sector/状態/精度を共通化できるがL_Dの数値は同じ単位でない。

**7. 何が未解決か：** 表現・生成機構とrare-order費用scope。

**8. 既存証拠で問いを定義できるか：** 比較基盤自体は新算法でない。表現変更の成立条件へ統合。

### OP-I4 → RQ-02 / RQ-06

**1. 過去研究の繰返し：** PR2/PM/P-Dの既存協調軸。R3は既知選択法との独立差なし。

**2. 具体的失敗条件を変えるか：** P-D/B-Fの選択一致を越える差が必要。広いgrid追加では足りない。

**3. 強いbaselineでも同じ改善があるか：** 同情報partition・compact BCH/cacheと公平な再最適化。

**4. 費用寄与は重要か：** q/backbone/shotの交換は保存。評価法追加だけの増分利益は未確立。

**5. 情報・前処理は実行可能か：** 探索・誤差情報・compile取得費が必要。

**6. 同H/task/精度にできるか：** 同元H/task/精度に整合させる必要。旧stale DFを使わない。

**7. 何が未解決か：** 具体的算法差とround間energy整合。

**8. 既存証拠で問いを定義できるか：** 評価体系そのものは別チャット。構造分割と異round構成の問いへ分離。

### OP-II1 → RQ-01

**1. 過去研究の繰返し：** orbital DF・UWC/BLISS・RC-DF/GRADE等との重複。

**2. 具体的失敗条件を変えるか：** core/pairのnative相殺・N1 fallbackを変える条件が未提示。

**3. 強いbaselineでも同じ改善があるか：** 同情報の既存orbital/grouping・Pauli/DFと同精度全task。

**4. 費用寄与は重要か：** normだけでなく生成費を変える必要。UWC値は未照合。

**5. 情報・前処理は実行可能か：** 旧UWCは候補7件の来歴未確定。軌道法はSOURCEで未実装。

**6. 同H/task/精度にできるか：** sector shift、model/DF誤差、stateアクセスの整合が要る。

**7. 何が未解決か：** 表現取得と全実装の機構。

**8. 既存証拠で問いを定義できるか：** 表現名を固定せずRQ-01へ統合。旧UWC成功を根拠にしない。

### OP-II2 → RQ-02

**1. 過去研究の繰返し：** partition/grouping/compact BCHは既知。

**2. 具体的失敗条件を変えるか：** BMの係数・cache同値、P-C追跡の選択同一、R3差分不足を踏まえる。

**3. 強いbaselineでも同じ改善があるか：** 同domain・同backend・同集約規則・同cacheを既知法に与える。

**4. 費用寄与は重要か：** commutatorとbasis境界の寄与は重要だが非prefix増分は未測定。

**5. 情報・前処理は実行可能か：** group構造・誤差scoreの取得と探索費を支払う。

**6. 同H/task/精度にできるか：** 同元H/precisionで可能だが新split評価は未実施。

**7. 何が未解決か：** 具体的な生成/情報差と回路・誤差の接続。

**8. 既存証拠で問いを定義できるか：** 非prefixという名称で再開せず、同情報対照との差を問う。

### OP-II3 → RQ-04

**1. 過去研究の繰返し：** cost-aware/whole-path IS・history法は既知。

**2. 具体的失敗条件を変えるか：** mean破壊、moment/range増、取得費を避ける具体的手順が不足。

**3. 強いbaselineでも同じ改善があるか：** canonical RTE、既知whole-path IS、P5/CTSへ同じ情報を与える。

**4. 費用寄与は重要か：** 切替数減だけでは費用支配性不明。C1 child費を保持。

**5. 情報・前処理は実行可能か：** likelihood/normalizer/価格取得とonline生成が中心の未解決。

**6. 同H/task/精度にできるか：** 同coherent finite targetなら可能。channel/observable法の自動移送不可。

**7. 何が未解決か：** アルゴリズムアクセス・有限bit・統計・実装。

**8. 既存証拠で問いを定義できるか：** 既知最適lawの再記述を除き、安い実行可能性を中心にする。

### OP-II4 → RQ-05

**1. 過去研究の繰返し：** SP/S4/D2/P5/Euler/ISに既存成果。

**2. 具体的失敗条件を変えるか：** 多step・prep・whole compileへの移送と価格marginが不足。

**3. 強いbaselineでも同じ改善があるか：** 共通fusion・actual-adjoint・common/P5/CTSと同IS機会。

**4. 費用寄与は重要か：** synthetic加法費用の利益あり。実PR全wrapperの主要寄与は未確立。

**5. 情報・前処理は実行可能か：** 新角度価格の取得、bit費、online費を無料にしない。

**6. 同H/task/精度にできるか：** 同finite target・bias/native精度・confidenceで比較する。

**7. 何が未解決か：** 回路/合成と統計の交換、全wrapper移送。

**8. 既存証拠で問いを定義できるか：** 同toy再探索を避け、強対照後に残る機構の条件へ具体化。

### OP-III1 → RQ-07

**1. 過去研究の繰返し：** routing/Gaussian/mapping自体は既知。世界的未探索としない。

**2. 具体的失敗条件を変えるか：** 現資料にはrouting後の性能がなく、克服機構未確認。

**3. 強いbaselineでも同じ改善があるか：** 同graph/layout budget/compiler/controlを全方式へ。

**4. 費用寄与は重要か：** topology-free費用は観測、routing支配性は未測定。

**5. 情報・前処理は実行可能か：** 配置探索/取得費とstate/control追加費が未確定。

**6. 同H/task/精度にできるか：** 同H/mean/accuracyは必要。backend未選定。

**7. 何が未解決か：** 先行研究上の独立差と回路費寄与。

**8. 既存証拠で問いを定義できるか：** 問いは保持するが、証拠不足を有望さに読み替えない。

### OP-III2 → RQ-06

**1. 過去研究の繰返し：** round別r/Kとbeta/alpha配分は既存。

**2. 具体的失敗条件を変えるか：** 構成別phase/model biasとenergy bridgeを先に解く必要。

**3. 強いbaselineでも同じ改善があるか：** 固定構成＋既知round配分、同情報adaptive運用。

**4. 費用寄与は重要か：** late round負担の旧局所証拠あり。表現/split変更の追加利益は未測定。

**5. 情報・前処理は実行可能か：** 更新・準備・誤差情報・compile較正費が必要。

**6. 同H/task/精度にできるか：** 共通E0・branch/overlap/phase整合が未確立。

**7. 何が未解決か：** 数学的target接続、統計、構成更新費。

**8. 既存証拠で問いを定義できるか：** 単なるschedule追加から切り離し共通energy条件を問う。

## 統合・分離の理由

- I3とII1は、比較基盤と表現変更の問いを分けてRQ-01に統合した。DFへ固定しない。
- I4は評価枠組みそのものを新研究とせず、II2と重なる構造分割をRQ-02、III2の異構成energy接続をRQ-06へ分けた。
- I1は既存技術の再利用を保持し、その先に残る回路生成の問いをRQ-03とした。P-A intervalの再開案ではない。
- I2のP5単純移植は既にC1で試されている。安い生成・finite-confidenceの接続をRQ-04、native辞書と全wrapperの交換をRQ-05へ分けた。
- II3、II4はそれぞれRQ-04、RQ-05の中心。法則の生成アクセスと、生成した辞書のnative価格を混同しない。両課題には共通証拠があるが独立実験として重複計数しない。
- III1はRQ-07に保持した。ただし未測定・先行研究差未確定であり、優先すべき候補と採択したものではない。

## 近接一次文献の扱い

PR/composite simulation、既存Hamiltonian partitioning、RC-DF・BLISS、SPRINT/GRADEは比較対象の出発点。文献のLCU norm、channel error、measurement、qubitization費用を、そのまま本PR coherent-signal費用へ移さない。[文献と確認範囲](PRIMARY_SOURCE_GUIDE.md)を参照。whole-path ISの最適lawの存在は既知として扱い、安い情報取得・生成・有限confidence・位相整合の問題を区別する。独立新規性はPhase 2Bで未判断。
