# C1固定macrocycle pilot v1：一回結果と監査

2026-10-13 JST。ユーザーの[明示実行認可](qc_c1_macrocycle_execution_authorization_v1_20261013.md)に従い、
source `ddaa05d2b51d575c3cfea1194bbfdffa182f9636`、認可commit `9c3b5c82942edfef3df27aeadf41a2299dc39270`で一回だけ実行した。
入力・contract・plan・runtime・比較条件を変更せず、940 eigensolver callsを完了した。retry0。
独立保存監査は936 unitaryと8,064比較行を照合してPASS。監査で追加eigensolveは行っていない。

保存結果の集約statusは `FIXED_PILOT_AUDITED_NO_SELECTED_C1_ADVANTAGE_OBSERVED`。
通常近似stateの16比較で、empirical/joint/boxのどのselectorも、登録した5%の費用改善基準を満たすC1優位を観測しなかった。
固定toy・有限h grid・登録価格モデルでの観測であり、全PF/全分子の不能性やC1研究の正式終了判断ではない。
新規算法/中心テーマを採択せず、追加科学計算とGPT科学レビューを開始せずMandatory STOPする。

## 1. 評価scope

[固定入力](../../artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/inputs.json)、
[結果前契約](qc_c1_macrocycle_preregistration_v1_20261013.md)、[数学・費用](qc_c1_macrocycle_technical_design_v1_20261013.md)。
4件の2qubit Pauli model、三つのpairwise非可換fragment、各fragment内は可換。

\[
A=\tfrac17I-aZI-bIZ-jZZ,\quad B=\lambda XI+\tfrac\lambda3IX,
\quad C=\mu XX+\kappa\mu YY.
\]

| input | a | b | j | λ | μ | κ |
|---|---:|---:|---:|---:|---:|---:|
| S1_weak | 1 | 1/2 | 1/4 | 1/16 | 1/32 | 1/3 |
| S2_balanced | 3/4 | 1/2 | 1/4 | 1/4 | 1/3 | 1/3 |
| S3_signed | 1 | 3/4 | 1/8 | 1/8 | 1/8 | −3/2 |
| S4_strong | 1/4 | 1/4 | 1/8 | 1 | 1/2 | 1/3 |

分子geometry、basis、DF rank/rank policy、split L_D、固定粒子数sectorは適用なし。
case名S4_strongは分子H4や手法S4を意味しない。target10^−5/10^−6はmodel energy単位で、Hartree/化学精度ではない。
Λはfragment係数L1の合計、h=1/(8Λ2^s)、s=0..5。PFとstatistical予算は各targetの1/2。
3代表S2、各m2/3/4 subdivision、24macro、3 Suzuki S4で39unitary methods。
39×6×4=936 PF eig callsとH reference4で940。C2はm1/m2保存phaseから72recordを導出した。
state |00⟩、悪い情報control |11⟩、価格κprice3/6、2targetで32比較。独立Hamiltonianは4件であり、32入力ではない。

## 2. 固定method選択の費用比較

[保存集約](../../artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/saved_results_summary.json)は、既に保存したfrontierと選択の集約だけである。
最良対照は保存single/subdivision、S4、C2のfrontierの最小値。C1/baselineのratio<0.95を登録改善とする。

| selector | 通常stateの比較可能件数 | 登録5%改善 | cost ratio最小 | cost ratio最大 |
|---|---:|---:|---:|---:|
| empirical近似期待値 | 16 | 0 | 約1.000 | 3.798 |
| joint state幅 | 16 | 0 | 約1.000 | 3.612 |
| independent box幅 | 16 | 0 | 約1.000 | 3.275 |

丸め前のratioとmethod/hは集約JSONのtableに全保存する。数値roundoff程度の差を優位と解釈しない。
|11⟩ controlは16比較全てineligible。ground-branch overlapの最大は全caseで約0.02120で、登録閾値0.25を満たさなかった。
|00⟩は全caseで最小overlap約0.60928以上。936 probes全て登録branch/residual条件を満たした。

上記frontierのh選択は、保存真biasから選ぶ**事後oracle診断**である。
reference前にsealした運用method選択と、事後の最良precision hを分ける。
ratioはcontrolled RZ/CX/basis Cliffordを価格3/6で換算したconditional Heisenberg energy-query費用であり、
QPE/RPEのconfidence/branch判別/量子回路compile/T/routing/分子state準備を含む最終総費用ではない。
product-state toyの情報measurement zeroを、分子ground-state情報の無料取得へ拡張しない。

## 3. 主要誤差・情報幅・既知法の診断

referenceを運用selectorへ戻さず、保存真ground vectorで評価した3 kernelのd_jは、4入力全てで正だった。

| input | d_ABC | d_ACB | d_BAC |
|---|---:|---:|---:|
| S1_weak | 0.000949185 | 0.000962224 | 0.000986176 |
| S2_balanced | 0.018710943 | 0.019982086 | 0.021570118 |
| S3_signed | 0.030805091 | 0.029871666 | 0.031294580 |
| S4_strong | 0.025161906 | 0.027176782 | 0.034050471 |

signed係数を持つinput名から、異符号ground-energy誤差を得たと推定しない。
今回の真dでは異符号機構は観測されなかった。これは有限入力の数値診断で、全Hamiltonianの同符号定理ではない。
same-sign/additive・周期sharingの既知制約は[設計](qc_c1_macrocycle_technical_design_v1_20261013.md)の条件付き理論として保持する。

joint selectorが真値なしに取得したcoefficient-only PF precision hは6/32比較でvalid、26/32でnull。
validはS1_weakの両target・両価格4件、S3_signedの10^−5・両価格2件、いずれもs=5。
exact rational PF boundと同じpointの保存numerical eligibilityも記録し、6件全てeligibleだった。
大半のnullを数学的不能とは呼ばず、係数由来gap/幅と有限gridの未判定として保存する。
PF boundは量子計測でのground枝識別・統計confidenceを保証するものではない。

同一catalog/情報/sharingを使用するstrong-known-composition controlとの生成器同値性は32比較で保持される。
これを独立新規算法の実証としない。原empirical分類は通常16比較全てでstrong baselineと細分化に改善が説明される側、
悪いstateの16比較でeligible点なしとなった。joint/boxの費用比も別に全保存する。

## 4. 実行と保存監査

一次出力directory：`artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1/`。
[completion](../../artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1/completion.json)、
[selector seal](../../artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1/selector_seal.json)、
[全比較](../../artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1/comparisons.json)、
[独立監査](../../artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1/independent_audit.json)。
raw completionの `FIXED_PILOT_COMPLETED_PENDING_INDEPENDENT_AUDIT` は生成時のまま保持し、後続監査と集約statusを別fileで追加した。
source独立のunmerged stage/Strang BCH/保存残差/有理幅/controlled counts/costのcheckerがPASS。
これは同一担当による実装独立監査で、外部研究者の再現やimmutable CIではない。

[実行監査](../../artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/execution_audit.json)。2spawn workers/各1thread、4×4 Pauli閉形式/cache。
runner wall8.73s、保存checker1.42s、BCH/情報/selector前処理8.03s。
残る主要負担はserial exact前処理で、4caseのmatrix phaseは各約0.11–0.12sだった。
今回のsourceを変更して再試行・再benchmarkはしない。将来採択時にだけ別versionで再利用/cacheの改善を検討できる。
GNU time maxRSS56,020KiB、cache384/case、保存一次出力7,020,288bytes<16MiB、console642bytes<1MiB。
同期aggregate RSSは未計測、速度向上率の比較測定なし。phase/total wall-time capなし、calls/resource/instruction制限を維持した。
20 locked files/plan/旧229filesは実行後もbyte一致、source/input/旧認可/STOPを変更していない。

## 5. 現在の停止と次の担当

[停止記録](../../artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/execution_stop.json)。一回認可は消費済み、retry0、追加scienceなし。
N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、他Track、global manifest既存description欠落FAILを維持する。
C4、分子拡張、候補追加、compileを自動起動しない。

次は**GPTによる科学的継続判断の開始承認待ち**。
未達理由の研究上の意味、C1を終了/再設計する情報価値、C4等へ移るかの判断はCodexの数値集約だけで採択しない。
新しいGPTレビューはまだ開始していない。
