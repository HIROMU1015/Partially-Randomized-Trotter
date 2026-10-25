# C1：H6合成誤差の部分空間上界・技術準備（2026-10-25）

保存compiled QPYに対し、**必要な800入力基底をすべて作用させ、出力8192成分を保持したFrobenius上界を作る方式**を実装した。回路の再build／compileやPF再計算は不要。全列が揃えば資格B（数値的operator上界）を得る経路であり、丸めを囲った資格Cではない。

今回は技術準備だけである。実H6のNPZ／QPYの数値decode、新operator評価、13 round再解析、CA／shots／総費用計算は**0**。旧104 profileの`UNRESOLVED_FINITE_PROBES`を変更していない。

## 1. 固定する対象と既存資産

起点は`3551e40d30ac9c9fa6d42dc586163f8bd87165c4`。直鎖H6・1.00 Å・STO-3G、Nα=Nβ=3、sector400、CDF／XDFとも保存rank14。全15因子のordered symmetric S2、s=1／4、literal／sharedの8経路、δ=0.1、m=0..12、CA=0.00159 Ha、DF reserve=CA/8=0.00019875 Haを保持する。PR splitは用いない。own-DF ground oracleと状態準備費用除外も保持する。

[native実行報告](research_c_c1_h6_native_stable_phase_execution_20261025.md)と[PF実行報告](research_c_c1_h6_saved_pf_execution_20261025.md)を来歴として再利用する。今回のhash照合は数値の再解析ではない。

| 再利用先 | 使用範囲 |
|---|---|
| `c1_h6_state_action.StateActions` | Aer CPU double、局所gate変換、declared phaseの保持、回路変換cache、full8192 state-action |
| 同`state_gate` | 既存component／norm-drift各1e-10、位相fitなし |
| `c1_stage1`／既存`PhysicalSector` | CARと同じsector列順とbit reverse。bilinears／Hamiltonianは作らない |
| 保存PF kernels4件・PF結果52行 | 1 h-stepのsector400 Wと、各roundの既存PF operator誤差。再計算しない |
| native QPY52件中のcompiled kernels8件 | 保存済みP0回路を読む。新build／compileは0 |
| 保存shell評価3件 | `numerical_segment`による**ancilla全2×2**の距離を再利用。有限probeと区別 |
| `round_synthesis`／既存RPE精度設計 | head／tailとkernelのtelescoping、complex signalからphaseへの接続規則。今回shots関数は呼ばない |
| 既存`Calls`／journal／`resource_plan` | 科学call、最初のFAILで依存call停止、資源に応じた並列化 |

追加SOURCEは[`c1_subspace_synthesis_bound.py`](../../src/trottertracks/c1_subspace_synthesis_bound.py)、[`c1_h6_synthesis_bound.py`](../../src/trottertracks/c1_h6_synthesis_bound.py)、[専用runner](../../scripts/run_research_c_c1_h6_synthesis_bound.py)の3件だけ。既存シミュレータ、CAR、Gaussian合成、PF、compiler、RPEは再実装していない。新補助関数は全列coverageの確認とstreaming norm、数値Wのnorm-growth補正に限定した。旧`round_synthesis`はexact unitaryの仮定を持つため、数値Wの欠損を保持する式を小さく追加した。

## 2. 3方式の比較

以下の「方式A/B/C」は、後述する「証拠資格A/B/C」と別の分類である。

| 方式 | 成立条件・得られる範囲 | 必要な計算／変更 | 判断 |
|---|---|---|---|
| A：構造・局所誤差 | 同じtargetへ局所差を対応付け、unitary列なら全kernel差≤局所operator差の和。Gaussian basisの向き・controlled scalar／相対位相・compiler global phaseも含む | gate単位の作用素上界、compilerによる全体rewriteの対応、角度／phase集計とreferenceの数値誤差評価が必要 | 現存の局所probeや単粒子再構成チェックだけでは全compiler差を資格付けできない。厳密化には有望だが今回の最小経路ではない |
| **B：全sector基底streaming** | 800入力列の完全性、出力8192保持、ideal targetのS不変性、actual回路のfull-space unitary性 | unique kernelごと800 state-action、4個の400×400 referenceチェック。full8192²不要 | **推奨・実装済み**。既存QPYとAerを直接再利用し、総compiled/reference差を測る |
| C：own-ground限定 | exact DF evolutionならancilla×groundの2次元は不変。compiled対PFの分離にはPFの到達空間も必要 | PF cyclic spanの不変性／truncation残差、またはexact DF比較へtargetを変更する設計が必要 | exact DF groundをPF固有状態と扱えない。現状の分離を維持するなら最悪sector400が必要 |

方式Aで、unitary single-particle行列の差がηなら、固定N粒子へのlift差はtensor productのtelescopingから≤Nη、basis共役には前後2側の誤差が入る。ただしこれだけでは中央I/Z/ZZの係数・scalar位相・controlled操作やP0全体rewriteの誤差を覆わない。P0のMPFR phase集計修正も、それ自体でkernel norm certificateを与えない。

方式Cには別案もある。compiledと**exact DF evolution**を直接比較するなら、ideal groundの2次元上で反復できる。ただし一段誤差がPFと合成誤差を併せたものになり、既存のround PF誤差との二重加算は禁止される。保存近似eigenpairを使うなら固有状態残差の扱いも要る。科学的なtarget／誤差配分を変更せずに済む方式Bを優先する。有限random probeや未収束Krylov spanを全到達空間と見なさない。

## 3. 全列上界の定義と資格

Sをancilla control=0/1と物理sector400の積空間とし、Eをその800列の計算基底embeddingとする。保存sector PF行列Wから

\[
U E=E\operatorname{diag}(I_{400},W),\qquad D=(V-U)E
\]

を定める。S外でのUはidentityに拡張できる。Vは保存compiled回路のideal gate semantics。actual出力はS外へのleakageを含むfull8192成分を保つ。

800個すべての直交入力列について、

\[
\epsilon_F=\sqrt{\sum_{j=1}^{800}\|D e_j\|_2^2}
           =\|D\|_F\ \geq\ \|D\|_2
\]

をstreaming集計する。各列のL2・最大成分差・coverageを保存し、同じ列の二重計上や欠落を検出する。全列を保持した8192×800差行列も不要。Frobeniusとspectral normの定義は[NumPy 1.26公式資料](https://numpy.org/doc/1.26/reference/generated/numpy.linalg.norm.html)に対応する。反復SVD／power iterationの点推定は上界の代用にしない。[SciPy 1.14.1のsvds](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.sparse.linalg.svds.html)もpartial iterative SVDであり、`tol=0`だけで厳密上界にはならない。

| 証拠資格 | 本経路での扱い |
|---|---|
| A：有限probe | 旧raw/reference、compiled/raw観測はそのままA。800列未完了も全域上界を出さず`UNRESOLVED_INCOMPLETE_BASIS` |
| B：数値的operator上界 | 全列から上式を構成。必要な入力Sについて任意のsuperpositionを覆う数学的規則を用いる。ただしシミュレーション／reference／norm等の丸めを厳密に囲っていない |
| C：厳密enclosure | 今回は取得しない。Aer演算、局所gate行列、保存CAR/PF reference、差・norm・伝播・phase計算の誤差enclosureが追加で必要 |

最終値への`nextafter`による1 ULP上方移動は最終cast程度の処置であり、全丸め誤差の囲い込みではない。零観測値も厳密零の証明ではない。`rigorous_operator_bound`とfull-Fock入力全体の`global_full_Fock_operator_bound`は常にnull。800入力に対する数値上界を8192入力全体のglobal normへ昇格させない。

raw/referenceとcompiled/rawを別々に再度全列評価する必要はない。今回の目的には**compiled/referenceを直接覆う総合成差**があれば足りる。旧有限probeの内訳は診断として残し、分離されたglobal上界とは呼ばない。

## 4. 不変性・leakageと13 round

Vのstructural checkはofficialなRZ/SX/X/CX、有限実角、bound parameters、classical操作なし、非identity layoutなしを要求する。これはexact ideal gate semanticsのunitary性を確認するものであり、Aerが丸めなしでunitaryを作用させたという証明ではない。

UのS不変性はsquare WをS内へembeddingする構成による。PF ground不変性は仮定しない。VはS外へleakしてもよい。

数値Wを厳密unitaryと見なさず、

\[
\Delta=\|W^*W-I\|_F,\quad
\alpha\geq\max(1,\sqrt{1+\Delta}),\quad \|U^kP_S\|\leq\alpha^k
\]

という**数値guard**を残す。reference欠損の最大成分が既存pilot許容1e-10を超えたらSTOPし、Wの再unitarizeや黙示的polar補正はしない。

\[
(V^n-U^n)P_S=\sum_{k=0}^{n-1}V^{n-1-k}(V-U)U^kP_S
\]

より、Vのunitary性とUのS不変性があれば、

\[
K_n\leq\epsilon_F\sum_{k=0}^{n-1}\alpha^k,
\qquad K_n\leq1+\alpha^n.
\]

α=1の場合だけ`n*epsilon`、unitary両者ならcap2となる。一般の場合は`expm1(n log(alpha))/(alpha-1)`とcap`1+alpha^n`を使う。n=s·2^m、最大16384を維持する。数値伝播は資格Bであり厳密Cではない。

targetがSを不変にしない場合、S上の一段誤差だけではこの式は使えない。人工反例では、一段差0でも二段差2となる。actual leakageを投影して取り去る検証も不可。

## 5. head／X/Y・PF・phaseとの接続

保存shellの3距離はancilla全2×2行列に対する既存`numerical_segment`のFrobenius評価であり、有限probe最大値ではない。ancilla作用をsystem identityへliftしてもoperator normは同じなので、S外leakageにも適用できる。shellがsystemを触らないことは前段で確認済み。今回はその固定結果を再利用する。

Aₙ=αⁿ、head距離h、各tail距離tₓ/tᵧとすると、入力を正規化した状態で

\[
d_X=h+K_n+A_n t_X,\quad d_Y=h+K_n+A_n t_Y.
\]

actual状態のnormは1、数値target状態のnormは≤Aₙなので、Z期待値差には

\[
|\langle a|Z|a\rangle-\langle b|Z|b\rangle|
 \leq(1+A_n)\|a-b\|,
\quad e_{syn,complex}\leq(1+A_n)\sqrt{d_X^2+d_Y^2}
\]

を用いる。α=1では既存`round_synthesis`の2·hypotに戻る。非unitaryな数値targetを途中で正規化し直さない。

保存PF誤差は`E_m=||W^n-exp(-it_m H_DF)||₂`である。正規化own-DF groundのcomplex matrix element差は≤Eₘであり、ideal Hadamard X/Yのcomplex signalに対応する。したがって**同じtarget・同じcomplex signalへ変換した後**だけ、`e_complex <= E_m + e_syn,complex`とできる。PFに係数2を重複して掛けず、合成誤差を2回加えない。

`e_complex<1`で位相誤差≤asin(e_complex)を既存の決定論phase room π/12と比べる。13 roundすべてが通れば`POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL`、登録十分条件を満たさなければ`INFEASIBLE_REGISTERED_BOUND`。後者は物理的不可能の断定ではない。

DF operator誤差はHa、PF／合成差はdimensionless、RPE統計誤差は別である。DF誤差をこのcomplex誤差へ直接加算しない。今回のrunnerはCA／shots／cost関数を呼ばず、対応結果はnullとする。δ=0.1・13 round・既存failure配分・共通CA目標は変えない。

## 6. 人工・回帰テスト

**異なるtest case計134件PASS、FAIL／skip 0**。新規39件＋既存回帰95件。途中の旧版38件を含む115件run、native単独18件run、新規最終39件runを保存した。重複runは件数へ加算していない。

- 独立dense spectral norm（2／5／9次元）とFrobeniusの照合、部分基底未資格、tiny値1e-200のnorm underflow対策。
- full-output leakage、controlled相対位相、正しいcontrol／CAR列順、入力dtype／shape／column重複／非finiteのFAIL。
- n=1／2／17／16384、target不変／非不変反例、数値reference欠損とnorm-growth、非unitary入力のSTOP。
- head／X/Yと反復を含む小dense oracle、m=0／4／12のstate差・complex signal差が導出上界以内。
- 人工QPYから全列runnerのend-to-end、同一IR重複排除、位相不一致、reset、reference不一致、メモリ不足、fresh output保護、FAIL後の依存call停止。
- SOURCE／入力hash／登録route／条件／allowance／call不一致のpassive preflight。新fit／PF／build／compileはmockで禁止。
- 既存H6 native／state-action／Gaussian・compiler precision／保存PF、既存決定論RPE精度接続の非回帰。nativeのmockを他テストへ波及させないよう独立processで実行した。

証拠：[technical](../../artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/technical/)。Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Aer0.15.1を既存task venvからread-only再利用した。既存SOURCE46件は無変更、旧protected727件＋直前raw122件＝849件をhash照合した。人工PASSは実H6上界の小ささ・CA達成・科学的GOを示さない。

## 7. 未承認科学実行契約・計算規模

[固定契約](research_c_c1_h6_synthesis_bound_contract_unapproved_20261025.json)のscopeは`C1_H6_COMPLETE_SECTOR_SYNTHESIS_BOUND_V1`。新SOURCE manifest51件、保存input38件を固定し、fresh outputは`artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/run01`とする。

| 科学call | 上限 |
|---|---:|
| 保存bundle load | 1 |
| sector基底index生成 | 1 |
| reference check | 4 |
| saved compiled QPY構造確認 | 8 |
| 同target・同compiled IRのgrouping | 1 |
| full-output basis-column action | 6400（800×最大8 group） |
| route kernel bound | 8 |
| round bound | 104 |
| route summary | 8 |
| **合計** | **6535** |

列actionは**バッチ数と独立に1入力列=1 logical unit**。G unique groupなら総数`135+800G`。4組のliteral/sharedがすべて同一と確認できればG=4、3335 unitsになる。arm／s／保存PF hashに加え、ordered official gate class・qubit・parameter float値・declared phaseが完全一致した場合だけ共用する。回路名の差は無視できるが、RZ数・最大残差・近似的同値では共用しない。現時点では保存QPYをdecodeしていないため、G=4を仮定しない。

順序はbyte/environment preflight→保存bundle→sector index→reference→compiled構造／grouping→全列streaming→8 kernel／104 round／8 summary→保存／公開／STOP。初回correctness FAILで新しい依存callを停止し、開始済みのものを回収する。部分列・journal・traceback・資源・STOPを保存する。registered boundが大きいだけの候補は他候補を停止させない。retry／resume=0。

1 stateは8192 complex128=128 KiB。8192²密unitaryは1 GiB、全差行列でも100 MiBになるため、両方を作らずバッチと列L2のみ保存する。保存reference1件は400² complex128≈2.44 MiB。long gate-body cacheが支配的になり得るので、既存resource planの約0.655 GB/groupのworkspace見積りと実際の空きRAMを照合する。

準備時affinityは32 CPU。将来は実行時のquota／空きRAM／pressureを再確認し、unique groupsを並列化、CPU budget/groupをAer threadsとbatch列数に割り当てる。threadpoolctlで現job内のBLAS過剰並列だけを調整する。[Aer 0.15.1固定SOURCE](https://github.com/Qiskit/qiskit-aer/blob/0.15.1/qiskit_aer/backends/aer_simulator.py)のCPU statevector・parallel experimentsを既存経路で用いる。共有venv／cache／設定／他jobは変更しない。

新規actionは3200～6400列、全roundの長い回路buildは0。前段のkernel action bundle（raw5＋compiled5 probes）の公表運用時間は約185～252秒であり、単純列数比例なら1 laneに約4～6時間程度を見込む。ただし回路変換cacheの償却・batch・thread配分が変わるため、これは粗い計画値であって実測H6 benchmarkやspeedup主張ではない。wall／CPU／RAM／workerの恣意的上限は追加しない。資源不足なら現jobをSTOPし、旧結果へfallbackしない。

SOURCE manifest SHA256：`61eaad15866a643a9fca34d686e4cf4c9357d81ba1ba1294625516b9f20a36d9`

契約 SHA256：`39a2fe9e6f5fa60a05c7b0c965127f87a53b58d9d831eb7c1b95d263cf35e734`

## 8. 残る資格と次の判断

既知の技術的blockerは残っていない。**実H6の全列評価が未実行であり、上界が13 roundで有用かは未確定**。Frobeniusによる緩さとn倍伝播で登録phase roomを超える可能性がある。旧有限probeの小さい残差から結果を先取りしない。

数値pilotとして次のCA／shots／条件付き総RZへ進むには、全列coverageとcorrectness、同じtargetへのPF接続、全13 roundのphase roomを確認し、その後に既存RPE統計・energy・DF誤差の条件を別途評価する必要がある。資格Bでも、その限界を明示した数値pilotは検討できる。論文での厳密CA保証・certificateには資格Cと保存DF／PF／ground・数値referenceの必要なenclosureが別途必要である。

**次の担当はユーザー**：この固定SOURCE／契約を確認し、最大6535 unitsの限定実行を明示的に許可するか判断する。許可後の担当はCodexであり、fresh run01を1回だけ実行・公開してSTOPする。今回は許可を得ていないため科学実行しない。承認証明書・承認原文保存・新grantは不要。公開remote照合後mandatory STOP。
