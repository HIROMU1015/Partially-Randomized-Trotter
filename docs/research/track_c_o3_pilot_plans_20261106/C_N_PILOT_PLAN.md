# C-N pilot案：PFのnative境界価格

2026-11-06 JST。**DRAFT / 未実行・未採択。** [共通契約案](SHARED_PILOT_CONTRACT_DRAFT.md)の元H/state・target・native profileが確定した後、別途許可された範囲だけ実行する。旧P-D/BF/BMの再開ではない。

## 1. 問い・入力・固定する層

問うのは、既知PFに対する**D/R境界を含むfull回路価格**が、強い加法proxyによる有限集合内の選択を変え、その差が同じenergy精度・成功条件の全RPE費用へ残るかである。新しい係数solverや非prefix法を作らない。

入力案はH4 linear1Å/STO-3G/4電子、元H ground、既存DF/native表現とLD3。LD12のFull deterministicを強対照とする。共通RPE時刻は既存δ0=.02、18round。元HとlegacyDF/stateの対応が閉じなければ主要pilotは未発効。DF-model診断に縮小する場合はその別目的を明示する。

合成policyは一つのnative profile内で固定、processorは全armでなし。basis共有・diagonal-only control・fusion・compiler機会は全armで同じ。C-Sの合成残差やC-Pのprocessorを混ぜない。

## 2. 有限比較集合と同じ物理時間

| arm family案 | 実装定義 | 役割・版の区別 |
|---|---|---|
| S2 | one-body/correction、順序付きD fragment、Rの対称基本列 | canonical PRおよびFull baseline |
| Yoshida4 | 同じS2基本列の既知triple-jump composition | 旧P-Dの`4th`と同じ係数family。Suzuki5再帰と混同しない |
| Morales Y8m10b | 公開版の係数を使う同じS2 composition | 旧`8th(Morales)` m8とは別。今回の高次強対照候補 |

最小集合は上の**3formula**。各formulaのPF microstep候補は既存h=.02/.01だけ、LDはPartial3/Full12だけで、計画上の構成上限は12。これは新数値の恣意的な探索域ではなく既存2刻み・既知係数からの採用案である。new_2、旧m8、Yoshida8、10次をさらに全件追加しない。公開Y8m10bを移植しない判断なら、実行前に旧m8への置換と限定比較であることを明記し、結果後に候補を差し替えない。

RPE時刻`t_m=q_m δ0`を全armで固定する。h=.01なら同時刻までのPF反復数を増やす。PFのhを変えたからといってRPE round/alias windowも同時に変更しない。qはRPE power、PF step数、tail rは別変数として保存する。

### nativeに落とせるPF target

基本列はD全体を無料のexact exponentialとせず、実装可能なone-body/DF elementary blockと、中央R evolutionの列で定義する。全formulaで同じblock順・scalar規約を用い、`iter_pf_steps`相当のliteral列を出す。高次化はその同じ基本列を係数付きでcompositionする。D内部だけの高次化と全D/R列の高次化を混同しない。

R occurrenceのsigned time `s_j=b_j h`に対するcanonical finite-RTEを使う。整数配分の初期案は、既存r_mを基準に`r_mj=max(1,ceil(|s_j| r_m/δ0))`、K2。これはsigned時間を保ち、各occurrenceの時間幅を管理する既知の比例配分案で、新lawではない。全stepでの累積bound/normalizerは実際の列から評価する。Fullはr0/K0のtail-free表現とする。

この配分が同精度を満たさないarmは、まずこの有限recipeでは不適格と記録する。coefficients、h、r/Kを無制限に探索してPASSを作らない。再最適化を許す場合は全armに同じ追加の有限範囲を**実行前**に与える。既知高次PF一般の劣位とは結論しない。

## 3. 再利用SOURCE・必要な最小差分

| 固定SOURCE・入口 | 再利用するもの | 必要差分の計画 |
|---|---|---|
| `f29eafbf4412193e8e8660d9f051889c2cba684b`：`product_formula.py`、`pf_decomposition.py` | S2/Yoshida4、`iter_pf_steps`、係数順・負時間展開 | compact係数入力から同じnative block列を得る小さい接続。registryの科学評価全体は再実行しない |
| [公開PF SOURCE][ypf] `20d64c2f…` | `morales_2025_y8m10b_list` | 公開係数を出所付きで渡す。係数探索・丸め再最適化なし |
| 同f29版：`rte.py` | `compose_finite_rte_occurrences`、signed config、corrected/raw moments、attenuation | 高次macro内の異種occurrenceを同じ順序で接続。各Rを単一分布で代用しない |
| 同f29版：`df_partial_s2.py`、`df_rte_qiskit.py`、`df_partial_s2_repeated.py` | elementary DF/basis/controlとS2 reference | repeated builderはS2専用。無理に型を書き換えず、既存elementary builderをつなぐ高次macro adapterだけを計画 |
| `4b625e5b…`収録の`rpe_hadamard_compiled_cost_benchmark.py`、`rte_order_stratified_cost_validation.py` | full測定wrapper、paired trajectory record、order条件付き価格の考え方 | 元benchmark requestもS2条件を持つ。literal PF macroのfull wrapperと価格providerの必要差分のみ |
| f29版：`rpe_resource_accounting.py`、`research_direction_full_opt2.py` | 軸/round/価格provider・同条件集計 | S2/Cδ²/半径1を一般PFへ移さない。共通契約案の一般signal/error/Nデータを受け渡す |

再利用test候補は`test_research_direction_energy_tail_pareto.py`、`test_research_direction_pd_realization.py`、`test_df_rte_qiskit_builder.py`、`test_rpe_resource_accounting.py`。旧科学評価を呼ぶtestは今回起動しない。将来はsigned/phase/orderと新接続に必要な人工controlだけを選び、旧結果の全再現を必須にしない。

既存scriptの参照入口はf29版`scripts/run_research_direction_energy_tail_pareto.py`と`scripts/run_finite_rte_signal_validation.py`、基点版`scripts/run_rte_order_stratified_cost_validation.py`。各scriptは旧固定科学条件へ入るのでそのまま起動せず、採用されたTrack Cの有限caseを既存関数へ渡す薄い入口が必要。旧script/SOURCEを上書きしない。

## 4. correctness・信号・shot

将来の最小検査は、各macroのliteral積と独立sector参照、正/負stage、zero/identity成分、actual adjoint、control0/1とX/Yの符号を対応させること。full compiled wrapperは全joint行列または対応state-actionで検査し、global-phase fitで不一致を消さない。正しい数学PF biasとcompilerの実装残差は別記する。

同じ元H groundに対して、元H exact、ideal PF、finite-RTE corrected/raw、native meanの複素信号を区別する。全18roundは小さいsector meanのpowerで評価する案とし、全random trajectory列挙やshot samplingを初期必須にしない。粒子数を保存しない個別native pathがある場合、sector70だけの検査に閉じない。

phaseは共通E₀への円周残差と、PF固有位相bias/混入を別々に保存する。実半径/根拠ある下界から軸別十分Nを計画する。Cδ²の旧近似や旧Nを高次armの保証にしない。E₀は診断にのみ使い、実行可能selector/decoderへ渡さない。

## 5. 価格取得と、最初に差を識別する比較

同じrealized trajectoryと同じcompilerを使い、以下の価格をpairedに取る。

| 価格model | 定義と役割 |
|---|---|
| stage/action proxy | literal stage・rotation/DF action数。既存の粗い診断 |
| **強い加法proxy** | 各D run・R occurrence内では既知共有/融合を済ませてcompileし、共通wrapper費を加算。basis/control/負時間・内部D workを含む。event個別和は既知の過大評価なので唯一のbaselineにしない |
| **full-wrapper実価格** | 同じliteral列をinterrogation全体として共有・融合・compileする。D/R境界・反復境界・readoutまで含む |

一次の価格情報は共通既存域からq1/2を較正、q8を未使用holdoutとする案。各armについて先にphase/半径の適合を確認し、同じNを各価格modelに掛けた診断で**価格modelだけによる選択差**を測る。その後、全armが同precision/successを満たすNと再最適化機会でのGへ接続する。粗いproxyとだけ比較して新method差を作らない。

r/K/符号/occurrenceが変われば価格lawも変わる。既存S2のanchorを高次へ流用せず、各新回路族の価格を取得する。K2のrare orderは条件付き層別で確率重みを戻し、未観測を0補完しない。n、paired covariance、model discrepancyを保存する。

q>直接域の全Gは、検証した価格構造と移送誤差を用いる。q8 holdoutが通っても終端q131072を保証しない。終端価格を直接大量compileすることはこの最小pilotの必須条件にせず、未取得域の幅が結論を左右するなら全費用主張を保留する。追加の長q情報を取る場合は、対象arm/回路族と問いを限定する。

## 6. 科学call・保存物・継続条件

| call群 | 計画する有限範囲 | 保存する科学的内容 |
|---|---|---|
| 入力/preparation | 共通H/state/basisを共有して一度取得 | H_phys/DF対応、state誤差・sector・SOURCE/input |
| PF/finite mean | 最大12構成、各18round。literal macroはcache。全点はsourceが選ばれた後の新評価 | formula版、h/step数、signed stage/r_j/K、scalar、ideal/finite/native signal・phase/半径・各誤差 |
| 回路・価格 | 各適格構成のq1/2/8、両軸、登録した価格標本/rare strata | full/分割を同trajectoryで対応、gate counts/depth、n/seed/uncertainty、全lawの残る未知費用 |
| 会計 | 同target・同情報の有限候補選択と全18round G | model別選択、相対差とその不確かさ、feasibility、N/α/β、長qscope |

新係数探索、旧P-D S1再探索、BF bridge穴埋め、BM symbolic再証明、別geometry/rank/新compiler探索はcall範囲に入れない。科学call数の具体的n/precision/価格精度は未決で、CPU等の低い人工上限で代用しない。

**継続材料**：強加法対照とfull価格の選択差が、同条件・不確かさを含めても残り、全energy/RPE会計へ接続できる。改善が既知fusionの応用だけならその適用結果へ限定する。

**縮小/保留**：価格modelだけで選択が変わらない、強対照で差が消える、或いは長q/rare/N/元H接続の未知幅に依存する。負結果はこの有限集合に限定する。correctness不一致・入力違いは費用比較を止めて原因を記録し、結果後のarm差し替えをしない。

## 7. 独立性・未決事項・次の担当

C-NはC-S/C-Pの結果を待たず、固定合成・processorなしで評価できる。共有できるのはH/state/primitiveと一般会計だけ。C-Nで選ばれたPFをC-S/C-Pへ事後移植すると独立ablationではなくなるため、初期pilotは別々の固定armを維持する。

**具体化済み**：3formula、2既存h、Partial/Full、同t_m、signed RTE配分案、強加法対照、paired full価格、全round接続と縮小条件。
**未決**：元H/state、共通target/profile/主資源、配分案の採否、誤差・価格precisionとmateriality、再最適化範囲。ユーザーがこれらを選び、別途実装・科学実行を許可した後にCodexが必要差分を作る。現文書から実行を開始しない。

[ypf]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
