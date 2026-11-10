# Q4-0：強baselineと公平な比較条件

2026-11-10 JST。既存値の引用と静的な数理・SOURCE対応だけを記録する。新shot最適化、count/信号走査、価格再会計は未実行。文献IDは[一次文献表](PRIOR_ART_AND_NOVELTY.md)、次の科学callは[未実行計画](NEXT_NUMERICAL_VALIDATION_PLAN.md)。

## 1. 共通の推定問題・情報

固定入力は[`08f9cbe7`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/08f9cbe77456ec29d50a992d7165c4d726df7bf7)に保存されたQ0/Q3成果。Linear H4／1.00 Å／STO-3G／DF12、Partial S2／LD3／h=.02／K2、元H厳密ground供給・準備費除外、canonical independent finite bits、Compiler A/Qiskit1.3、ideal native/reset/measurementを保つ。m0–16は保存schedule、m17はq131072/r32/bits88。energy精度1.5936e−4 Ha、failure .05、phase cap c=2/5。

各prefix成功履歴に対する中心Ecと `u=−t17(E−Ec)` を用い、`|u|≤U=4/5`、数学的hullの径≤4/5を比較基準とする。実装資格には、中心の区間包含と最終復元丸めまで含む**出力energy区間の径≤(4/5)/t17**を別途要求する。従ってenergy半幅≤c/t17。中心位相は `ψ=−t17 Ec (mod 2π)`。全方式が同じhistory、Q3-Bの区間中心・丸めcertificateとfailure配分を利用できる。

| 情報 | 主要比較での扱い |
|---|---|
| 全m0–16の実count履歴 | 両側に同じものを渡す。初手Aはその証明済みinterval summaryを使う。full-history likelihoodを使う方式は、その新しい同時尤度・nuisance関係を資格化すれば別baselineとして許す |
| prefix energy/phase interval | Qだけの特権にしない。ψ依存のshot/failure配分も対照に許す |
| ρL、ρU=1 | 同じ保存armの値。真のρではない。無重みphysical coherenceの範囲であり、weighted PRへ流用しない |
| bias上界 B | `2/5 − F.from_float(statistical_phase_lower)`。PF/native/finite-bit等を既に含むので再加算しない |
| 真のphase、E0、gap、groundの古典記述、真のρ | policy・推定器へ渡さない。oracle診断は主比較と分ける |
| calibration data | 主要比較では追加なし。使うなら双方へ同じ機会を与え、量子shot・時間・回路費を計上 |
| 内部乱数・事前密度 | 真値と独立。prefixのconfidence集合をuniform Bayesian priorとすることは追加の設計選択であり、真の分布と認定しない |

保存端点はQ0 INPUTSから読む：B′のρL=.5726873297263416、β=.38598699711850315、disk=3.6648120444602677e−6。Reference-B1はρL=.5726872813784947、β=.3859869126962799、disk=3.713159891479787e−6。両armの旧m17 Nは626/axis。ここでは値を読み引用しただけである。

## 2. 比較するbaseline

| 手法 | 定義・測定能力 | ρ/bias・成功基準 | 判明していること／必要差分 |
|---|---|---|---|
| 保存L2 X/Y | 固定X/Y、座標HoeffdingのL2位相十分条件。nX,nYとαX+αY=1/360 | 保存ρL/B、有限標本の一様十分保証 | **1252がこの証明family内の最小総数**。Q0の結果を再計算しない。全推定器の下界ではない |
| 前段依存の単一Pauli | ψからX又はYを一つ選び、その方向だけ測る | 同じ未知ρ/B、全ψに対するhull幅 | Q0の中心π/4の等投影反例で、この保存許容集合の一様条件は無限標本でも満たせない。任意角の単一方向や実H4全到達信号の不可能性ではない |
| **A-H：強いX/Y集合推定器** | 前段区間とannulusを使い、両軸CIを**同じlatent信号**に同時反転。n配分・α配分はψ依存も許す | Qと同じtwo-sided Hoeffding、同じworst-case hull/failure | 最優先。Q0は連続中心・理想Pauliモデルで総数≤340を排除、最小総数は**341〜1252で未確定**。L2の最良α式をhullの最良α式と取り違えない |
| **A-E：二項CI版X/Y＋matched Q** | 同じAの幾何に、等尾Clopper–Pearson型の厳密二項反転を使う。Qも同じCIへ変更して比較 | 同じ一様coverage、未知ρ/B、全count幅。MLEの点誤差やWilks近似で代用しない | 新しい数値資格が必要。A-Hの≤340下界は流用できない。旧scipy ppfの点計算を厳密な区間保証へ昇格しない |
| **B：既知adaptive PE** | P1/P5/P7のreference位相制御。固定時間・prefix後1batchへ制限した版と、m17内逐次adaptive版を分ける | Bayesian posterior/credible regionや期待損失が原型。一様有限保証へは別certificateが必要 | Qと同じ任意角測定family。既知法のpolicyを制限・変更した場合は「原論文そのもの」としない。Qとのpolicy同等性は測定原理だけから断定しない |
| **C：未知visibility推定** | P8のphase＋visibility joint posterior／nuisance射影。固定時間でX/Yや複数角を使える | Bayesian median/MSE、asymptotic情報境界と、frequentist hullを分ける | 同時推定は既知。Qはvisibilityを点推定しない。ρ既知化・追加校正を対照だけに無料提供しない。最初の公平なfinite-CI版はA-E |
| **D：cost/time-adaptive** | P6/P7のk・角・測定費用の設計 | kごとのρk/bk・source/price資格が必要 | 今回の固定m17とは自由度が異なる。別PF/lawを使わなくても、別interrogation長の資格・価格が未取得なら同等比較は未資格 |
| Q固定policy | `φ*=ψ+π/2`をprefixから一度計算、同じbinary角で340 fresh trial。m17内で変更しない | Q0未知ρ/B＋Q3の受理/算術/native/丸め保証 | 固定モデル内で資格済み。A-Eを比較するならQ-Eも同等に強化し、その新nと旧340を区別する |

この集合を「世界の最強baseline全て」とは呼ばない。Aは具体的に定義したCI-hull familyであり、joint likelihood-ratio region、asymmetric tail配分、全prefix joint推定、m17内adaptive停止等の全方法の最適値ではない。

## 3. Joint信号を共有するためのSOURCE条件：静的補題

**Q3-Cの各角φごとのvirtual-state保証だけから、異なる軸が同じρ/bを持つとは推論しない。** そこで保存されたnative readout templateから次の強い対応を直接導く。これは今回の静的な代数・norm論証であり、新しい科学数値検証をしたものではない。

保存Hのnative列は `RZ(a) SX RZ(a)`、`a=float.fromhex('0x1.921fb54442d18p+0')`。保存SdgはRZ(−a)、P(−φbin)はRZ(−φbin)。coreと初期HはQ3-Cの全context閉包により共通である。common global phaseは既にcontrolledなjoint全体に限って落とす。

ideal-native SXを使うと、終端H̃のobservableは厳密に

`H̃† Z H̃ = sin(a) X + cos(a) Y = Xδ,  δ=π/2−a`。

P/Sdgを含む一般readoutは `X_(δ+φbin)`。終端readout直前の共通coherenceをzbodyとし、`z'=exp(−iδ) zbody`と定めれば、同じtrajectory lawのmeanについて

`μX=Re z'`、`μY=Re(exp(−ia)z')`、`μQ=Re(exp(−iφbin)z')`。

z'はρ≤1を保つ。**native Yを数学的π/2へ丸めず、既知の軸aとして扱う**。これはoracle calibration情報ではなく固定SOURCE literalである。理想Pauli版との違いを数値0としない。

旧body/preparationのjoint errorをEbody、終端Hの保存上界をeHとする。observable差のnormから `2|sin(δ/2)|≤2 eH`、回転によるcoherence差も≤2eH。bodyの差は旧2√2 Ebodyで囲むので、共通z'のdiskは `2√2 Ebody+2eH ≤2√2(Ebody+eH)`。旧chargeは初期H・core・終端Hに加えSdgの非負分も含むため、その全安全側budgetを残せばρL/Bを継承できる。PF/finite-bit/SOURCE/vacuum項は一度だけ保持する。

従って、この**特定のreadout列・同じcore・ideal-native primitive**に限り、Aの共有latentモデルを既知の非直交軸(0,a)で構成できる。一般のH̃、別compiler、weighted signal、axis依存の追加noiseへは移送しない。後続実装では上のgate列・符号・norm chargeと、角aの区間処理を人工fixture／SOURCE照合で固定する。資格が閉じなければ共有ρ/bを強制せず、軸別nuisanceの大きい集合を使って別結果として報告する。

## 4. A-H/A-Eの推定器を明示する

モデル点は `z'=ρ exp(i(ψ+u+b))`、`ρ∈[ρL,1], |b|≤B, |u|≤U`。理想版はθX=0, θY=π/2、登録native版はθX=0, θY=a。実際の中心はQ3-Bの区間包含を受け取り、幅を消さない。

各軸のmean区間Iaを得た後、最終集合は

`S(kX,kY;ψ)={u∈[−U,U] : ∃ρ,b, Re(exp(−iθa)ρ exp(i(ψ+u+b)))∈Ia, a=X,Y}`。

**同じρ,bが両条件を同時に満たす**必要がある。軸ごとに別々にρ/bを消去した集合の交差は一般に外側集合であり、強いjoint推定器と同一ではない。正しい強baselineは同時実現可能性を解く。集合が分裂した場合も全成分を残し、最後にhull径を測る。空集合は`INCONSISTENT_COUNTS`であり、0誤差成功へ置き換えない。prefix成功と両CIのcoverage event上では真のuがSに含まれ、空集合にならないことを示す。従ってその拒否は既存の末尾failure予算に含められるが、成功履歴の追加post-selectionや未計上の再試行は認めない。

A-Hは±1 sample mean c=2k/n−1に `ε(n,α)=sqrt(2 log(2/α)/n)`、Ia=[c−ε,c+ε]∩[−1,1]。αX+αY≤1/360、n=0の軸は未測定としてIa=[−1,1]、CI式へn=0を代入しない。配分はprefixに依存してよいが、m17 countを見て変更するなら別のsequential保証が必要。

A-EはBernoulli p=(1+μ)/2の等尾CIをinterval binomial tailで反転する。Q-Eも同じ等尾CI・全count保証で反転する。αXの連続最適化は許すが、等尾の規約は両側で同じ。非対称tail、joint LR、sequential e-processは次の独立familyで、今回計画の外に置く。将来それらを加えるならQ側にも同等の強化機会を与える。

## 5. 中心phaseの全域と成功基準

Q0強X/Y下界のπ/4 witnessは、**連続した中心phase・保存annulus**に対するもの。実N347のcountから得る中心や、実H4の到達coherenceが必ずその点を取るとはしていない。Q3-Bに全count-domain証拠があることから、π/4がある成功履歴で厳密に到達するとも推論しない。

第一の比較はQ0と同じ連続ψ全域のrobust taskとする。これは全実履歴を含む外側問題であり、一様上界は実履歴にも安全側に使える。一方、その最悪点での**下界**は、この外側問題の下界であって実H4で必要なshot数とは限らない。native (0,a)へのwitness移送も別に確認する。実count中心に限定して強化する場合は、全有限中心のcoverageと成功履歴との適合を別scopeで資格化する。

主要指標は全ψ・全countの**非空S**についての `diam hull S ≤4/5` と有限標本coverage、さらに出力energy区間の丸め込み幅。空集合の扱いは§4のfailure会計に従う。Bayesian expected loss、credible interval、MSE/RMSE、asymptotic CR、平均caseは別表にする。confidence集合のcoverageと、全countでの径保証も別義務である。

## 6. 成功確率・費用の公平性

全方式でprefixのfailure34/720、末尾A=1/360を固定する。Aの固定配分ではconditional Hoeffding/二項CIのunion boundを使う。m17内の測定方向や停止を逐次変える方式に、固定nのCIを無条件適用しない。Bayesian posterior massだけをfrequentist coverageと呼ばない。

費用jは同じ保存prefix費を加え `NX μX,j+NY μY,j` と `NQ μQ,j+A_j` を比較する。Q3-C/Dにより現nativeモデルではμQ,j=μY,jだが、追加RZは単価に既に含み、古典feedback／校正／回路生成費を0にしない。history依存policyでは、history分布を無料で仮定せずconditional worst-case費用、又は明示した確率モデルでの期待費用を区別する。最小shotの配分と最小RZ/depthの配分は同一と仮定しない。

ここでµは保存lawのfresh trajectoryに関する期待native資源の上下界を継承する。主比較のworst-caseは**prefix履歴について**取るものであり、全trajectoryのpathwise最大費用や実行時間の保証へ変更しない。履歴の平均を取るには別途その分布が必要である。

Qの50.381058%は従来X/Y費用に対する保存結果。強baselineの費用へ置換しても同じ率が残るという数値はない。B′とReference-B1はそれぞれ自分のρL/B／価格を使い、同じ最適化機会を与える。Full/Truncated、別q、CTへの無条件移送はしない。

## 7. Full／Truncated・Clifford+Tへの静的接続

Full／Truncatedでも、元H energyに対する複素coherence、正の半径下界、bounded phase bias、prefixのbranchと受理が同じ形に資格化できれば、prefix後のquadrature測定という原理は利用できる。PR専用の原理ではない。一方、現在のρL/B、N347 prefix、340、m17単価はこのQ系列の固定入力であり、別PFへコピーしない。

再利用候補は[`da8b2c3f`の固定比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPORT.md)、同commitの`artifacts/track_c_df_truncation/2026-11-08/ordering/{ROUNDS.csv,ERROR_BUDGET.csv,VALIDATION.json}`と`SOURCE.json`、`src/trottertracks/track_c_df_truncation.py`の`det_qualification/circuit_piece/price_piece/choose`。元H・state/sector・DF順序・時刻・normalization・α配分・compiler・wrapperの一致を先に確認し、相違があればadapterと再資格が必要。保存Full/Truncatedの各roundのρ/βと失敗配分から新prefixの成功eventを作り、Q3-Bの全履歴受理・丸め論証をそのprefixへ接続してから、Q readout単価と新shot数を評価する。Truncatedでは元Hに対するcombined PF/drop位相保証を保持し、既知E0による補正を追加しない。

同じTrack CのClosed P3 PRはweight積をrange Lで正規化する推定量であり、今回の無重みphysical ±1 likelihood／共有density coherence補題へ直結しない。bounded-mean concentrationだけの移送と、二項likelihood／unknown-visibilityモデルの移送は別である。従って保存総額の数値を今回のQ価格と並べて新しい優劣を判定しない。

CT側ではC0・43基本word・85 certificateのkey／strict仕様／actual adjoint／consumer対応という既存監査資産を再利用できる。ただしprefixから生成する任意binary feedback角のtarget literal・domain・precision・relative phaseをその辞書が扱えるかは別問題。新しい角のwordと全support誤差、shell等を含むjoint disk、必要shot数、両arm共通policyのT価格が揃うまで、nativeのµQ=µYをT価格へ移さない。Q3-C/Dの[T未資格の記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08f9cbe77456ec29d50a992d7165c4d726df7bf7/docs/research/pr_reference_feedback_q3cd_20261110/RPE_AND_COST_RESULT.md)と旧CTの負結果・STOPを保持する。これらの後続計算は今回もH/E/C案にも含めない。
