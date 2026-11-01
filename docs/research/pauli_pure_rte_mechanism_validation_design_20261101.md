# Pauli純RTE低費用の原因分離：検証設計案

2026-11-01 JST。対象は HIROMU1015/Partially-Randomized-Trotter。**文書・SOURCE・保存descriptorの技術調査と、未実行の設計案**である。新しい信号、差分値、N/G、順位、統計、Hamiltonian、回路を計算・生成していない。科学SOURCEの実装・変更・新規固定、grant発行・消費、詳細科学レビュー、原因の正式採択も行わない。[実行準備・再利用対応表](pauli_pure_rte_mechanism_execution_readiness_20261101.md)と対になる。

## 1. 目的と今回の判断材料

研究の中心はDF Hamiltonianに対する部分ランダム化PFの測定込み資源評価である。Pauliの純RTE低費用は、決定論的に何を保持するべきかを理解する比較材料とする。[全体棚卸し][LAND]・[U01〜U04][GAP]を継承する。

既存の最新比較は有限候補・保存確認標本・仮想compiled残差ρ=0に条件付けられる。純RTE低Gは14/15精度セルだが、保持部分、tail/law、bias、normalization、単回価格の因果寄与を分けた結果ではない。[四方式統合][PAULI]、[図表・claim範囲][PAULIFIG]、[既存原因調査][MECH]。

準備調査で重要だった点は次のとおり。

- **H6 prefix78は保存モデルに all_Z_prefix=78 と明記される。** 公開Hybrid archiveの固定specにある保持78ラベルもI/Zのみだった。理想演算では保持項が可換なので、後述BとCが一致することを先に検査する。この構造的性質を、実compiled誤差ゼロや「観測biasは全てsplit由来」という新しい数値結論へ格上げしない。[モデル][H6MODEL]、[archive索引][H6ARCH]、[保持SOURCE][ADAPTER]
- DFの既存 exact_tail は「保持部全体も厳密」という意味ではない。実際の保持PFを使うCに相当し、外側分割と保持PFをまとめた誤差しか分けていない。[evolve_batch][DFSIG]
- Pauliの保存targetは**有限bitの実効law**を評価する。理想Taylor多項式との差を別に残す必要がある。[law][LAW]、[tight評価][TIGHT]
- DF builderの one_body は元一体積分に low-rank の one_body_correction を加えたもの。同一full Hの確認と、一体／残差の物理的分割の確認は別である。[DF生成SOURCE][DFH]

以下の優先順は今回依頼された設計順であり、研究方針の正式採択ではない。

## 2. 既存結果で説明済みのことと再実行しない範囲

| 問い | 既存証拠の到達点 | 今回追加しないこと |
|---|---|---|
| NとCの競合 | H2/H4では大Nを低Cが補う条件、H6厳精度では小bias・広headroomによりNも小さい条件が保存される。[MECH][MECH] | 比率・寄与率・最良候補の新計算 |
| H6 event拡張 | Hybrid p78/q8/K2のr16/32/64は確認各n64、純RTE q8/r64/K2は確認n48。r32がHybridの保存family nominee/minで、r64の純RTEはなお低G。[H6表][H6CSV]、[RTE内訳][EVENTCSV] | 全parameter再探索、既存価格の取り直し |
| 同じq/r/K | H4 L0/L1のq8/r8/K4、H6旧L78/L0のq8/r8/K2が既存。新対象にもH6 r64/K2/q8の両familyがある | family bestのr32対r64をmatchedと呼ぶこと |
| DFのA/C/D診断 | 公開長時間signal recordに reference、exact_tail、corrected、raw、outer_PF_or_discard_bias、finite_RTE_bias がある。[保存record][DFRECORDS] | Pauliや別stateへの値の転用 |
| Track A Mechanism | 既存N/C/G恒等式、normalization/headroom/alpha/丸めの説明因子は完成。A82後続を含め固定32比較64軸が完備。[最終統合][MFINAL] | 古い63side/31比較の欠測を現在状態にすること、因子の再計算 |
| 部品価格・control | generic D1の保持/event/emptyと非加法性、D2のgeneric→central差は測定済み。[D1][D1]、[D2][D2] | generic価格を最新central保持費へ代入、D1/D2全体の再実行 |
| 全law実ρ | 6局所包含とR2-C未完を区別。保存Gのρ=0仮定は実ρの保証ではない。[局所][ARB]、[R2][R2] | 全law証明の自動再開 |

DF B2長時間の33/40、DF B3の0/40という取得域の結果は別々に保持する。B3は一体項を保持する。公開ebe9e2aa版を使い、未公開local最終版とのraw同一性は仮定しない。[B2][LONG]、[B3][B3]、[一体保持の来歴][B3PROV]。

## 3. 第一優先：複素信号としてのbias分離

### 3.1 共通の入力と定義

保存raw stateをφ、保存exact normをν=〈φ|φ〉とし、全ての中間信号を Z(U)=〈φ|U|φ〉/ν に揃える。状態を新しく正規化しない。Aは保存referenceを使い、新しいfull-H referenceを作らない。raw referenceをνで割る経路と、既にphysical_referenceである経路を混在させない。

H=cI+H_D+H_R、δ=T/q。H_Dは**選択prefixの非identity Pauli項の和**で、物理的なbare one-bodyとは限らない。行列積は右から作用する。保持SOURCEのforward適用順をP1→…→PLとすると、

~~~
E_j = exp(-i δ h_j P_j / 2)
F   = E_L ... E_1                 # forward halfの行列
R   = E_1 ... E_L                 # reverse halfの行列
P_{K+1}(X) = sum_{m=0}^{K+1} X^m/m!
~~~

このF/Rは逆演算ではない。同じ正時間の対称half sweepである。負の時間を検査する人工例でも、符号付きδをそのまま使い、勝手にabs(δ)へ置き換えない。

| 信号 | 1 stepの演算（scalar以外） | 判別対象 |
|---|---|---|
| Z_A | 保存されたfull-H厳密reference | 共通の比較基準 |
| Z_B | exp(-iδH_D/2) exp(-iδH_R) exp(-iδH_D/2) | 保持／tailの外側対称分割 |
| Z_C | R exp(-iδH_R) F | SOURCEどおりの保持PFを追加した効果 |
| Z_Dpoly | R [P_{K+1}(-iδH_R/r)]^r F | ideal finite paired-Taylor打切りの追加効果 |
| Z_Dgrid | 保存された有限bit実効lawの補正平均target | 実際の保存N会計が参照したtarget |

B/C/Dpolyは上記stepをq回適用し、scalar exp(-icT)を一度だけ含める。SOURCEのper-step scalar実装と数学的に同じ位相を表すが、浮動小数点のscalar形成差は記録する。Dpolyは既に補正後の多項式平均なので、追加のB乗算はしない。

Dgridは保存targetの physical_corrected_mean を再利用する。これはraw meanに保存Bを**全体で1回**掛け、exact νで割った量である。再度Bやνを適用しない。K2のideal多項式次数は3であり、PF次数2や最大偶数event order2と区別する。

### 3.2 加法分解とfinite-bitの扱い

~~~
Δ_outer = Z_B     - Z_A
Δ_hold  = Z_C     - Z_B
Δ_trunc = Z_Dpoly - Z_C
Δ_grid  = Z_Dgrid - Z_Dpoly

Z_Dgrid - Z_A = Δ_outer + Δ_hold + Δ_trunc + Δ_grid
~~~

各Δの実部と虚部をcosine/+Im sineとして保存する。閉合は複素差・両軸で確認する。絶対値の和は必要なら三角不等式の上界として示すだけで、総biasの内訳率・独立因果寄与率とは呼ばない。相殺があるため、絶対誤差やNはこの線形分解に従わない。

Dgridのcomponent/order確率は、保存CDFの53bit uniform grid上のexact masses、保存binary64 rotation angle、scalar_step、B_savedを使う。[LAW][LAW]・[TIGHT][TIGHT]はその平均を評価している。ideal polynomialと同じと決めつけない。Δ_gridにはそのlaw・角度・正規化の意味の差を含め、丸めたbias中心の差だけを精密なfinite-bit誤差と呼ばない。

Δ_gridは**実compiled残差ρではない**。中央角形成、実ライブラリのbinding、transpile後の全law meanまでの差はU04として別に残る。Dgridのu_encもρではない。保存targetの包含と新しいB/C/Dpolyの数値診断は、保証の強さを揃えたと仮定しない。

### 3.3 H6初期対象と可換保持部

| ID | 固定候補 | 用途 |
|---|---|---|
| H16 | PR_p0078_q8_r16_K2 | 保存Hybridのr系列 |
| H32 | PR_p0078_q8_r32_K2 | 保存Hybrid family best |
| H64 | PR_p0078_q8_r64_K2 | 純RTEとの同q/r/K対照 |
| R64 | H6_p000_q8_r64_K2 | L_D=0純RTE |

全てlinear H6/隣接1Å/STO-3G、12qubit、Pauli918＋constant、T=.8、ε_sig=.0005。保存同一model/referenceを使う。[H6 SOURCE][H6SOURCE]、[RTE SOURCE][EVSOURCE]。

prefix78の保持項がI/Zのみであることから、理想保持halfは exp(-iδH_D/2) と同一になる。**B=Cが成立するはずというcorrectness条件**を設ける。数値的な不一致があれば保持PF誤差と解釈せず、順序・係数・scalar・数値精度・SOURCE bindingを調査する。可換性はH_DとH_Rの可換性を意味しない。

H16/H32/H64でB/Cは共通で、rごとに取り直さない。R64はH_D=0なのでB=C=Aを構造的対照に使う。初期batchは共通B/C各1とDpoly4の**最大6種類の新しい信号伝播**、保存A/Dgridの読取と新しい差分評価に限定する案とする。高精度再評価・独立数値cross-checkを実行する場合は、その対象・回数も実行範囲に明記する。

同じq/r/Kの主対照はH64対R64。H32対R64は実際の保存family best比較であり、rも違う。RTEのr16/32は保存探索点があっても、Hybrid各候補にmatched独立確認が全て揃ったとはしない。初期計画へ新しいL0/r16,32を追加しない。

### 3.4 数値精度・shot影響の判定

- 既存DgridはArb scalar128bit・vector中心53bitのtight L2包含であり、保存raw norm、reference方法、B適用回数を継承する。referenceはraw stateのDuhamel残差を使い、厳密固有状態を仮定しない。
- B/Cの「厳密」は数学的ターゲットを指す。実装候補は既存sparse exponential actionとPauli作用・保持rotationを再利用する。H6はfull 4096次元を既定とし、DFのspin-sector加速を自動流用しない。全ての中間演算の不変性を証明・確認した場合だけsector化する。
- 既存DF診断の1e-10整合閾値を初期の数値cross-check目安にするが、その値をrigorous uへ転用しない。新B/C/Dpolyには手法・精度・数値不確かさの取得方法を付ける。初期結果を包含保証にするには別途妥当なexponential-action boundが必要で、SciPyのdouble精度計算だけで認証済みとしない。
- 差の符号や大小が数値不確かさを超えなければ UNRESOLVED。小さいΔ_truncやΔ_gridを0と確定せず、精度を上げる必要性だけを報告する。精度の追加を無制限に自動実行しない。
- 両軸の保存headroomは h_a=ε_axis−|b_a|−u_a−ρ_a、N_a=ceil(A_upper,a/h_a²)。保存B・α・ε_axis・guard政策を維持する。実biasを差成分の絶対値の和へ置換しない。

shot影響を定量化する場合は、例えばb_Dから複素Δ_truncを引いた仮想biasを両軸へ戻し、同じDのB/αを固定した条件付きN感度を別承認範囲で評価できる。これは「その誤差だけを除いた場合の診断」であり、B/Cを実装する新推定法の必要Nや一意な寄与率ではない。変更したbiasの数値不確かさも更新し、元uを根拠なく流用しない。初期bias batchの必須出力は符号付き差とheadroom関係までとし、新しいG順位は作らない。

## 4. 既存A/C/D・B対照とコードの再利用

DF e417ffc9版 evolve_batch は、保持各blockのforward/reverse half、厳密tail列、補正多項式列、raw列を同時伝播する。exact_tailはC、correctedはDpoly、referenceはAに対応し、rawはnormalization診断である。public record 3195cd358b8f5056f0ba（H6/p3/T1.2/q6/r4/K2）と6d2993e77eb62263a899（H4/p1/T.8/q8/r4/K4）で該当fieldの存在を確認した。保存値の差は今回計算していない。

旧DF finite_rte_signal_validationも保持PF＋厳密tailでC型であり、保持全体の厳密指数Bとは違う。[旧検証][DFOLD]。一方 finite_rte_phase_amplitude._condition_record にはh_d=.7Zを厳密指数とするB型の2×2 toyがある。[toy SOURCE][TOYB]。これは分子H6の保存Bではない。関連するM1-Sにもexact H_D half/quarter sandwichがあるが、shared±/ordinaryの別平均・別lawを対象とし、今回の有限paired-Taylor H6 Pauli Bの保存結果とは同一視しない。[M1-Sの適用域][M1S]。Track A Mechanismのnormalization/headroom因子もB信号を新たに提供するものではない。限定検索範囲で、同じH6 Pauli tupleの保存B/C分離結果は未確認である。

最小拡張は、既存Pauli sparse_operator/rotationを使うB/Cのtail-exact切替と、既存degree recurrenceを使うDpolyの補正平均出力である。DF contextはDF fragment・sector・入力生成に結び付くため、そのrunner全体をPauliへ呼び替えない。旧run_signalは他時間reference生成や多数候補評価を含み、今回の将来batchにも直接使わない。

H2は16次元、H4は256次元の保存表現を維持できる。保存H2 p1、H4厳精度p1は単一保持項、H4粗精度p36も保存descriptor上I/Zのみである。ただしH4のtargetは別FLINT経路・precision schedule、H2のq/Kも異なる。方法の移送にはraw norm、reference、effective law、位相・bit精度の再照合が必要であり、H2/H4の追加計算を初期batchへ加えない。[H2][H2PAULI]、[H4][H4PAULI]、[H4モデル][H4PM]。

## 5. 第二優先：同じ保存trajectoryの回路価格診断

### 5.1 4構成とphase所有

| 構成 | 含むもの | 位相 |
|---|---|---|
| F Full | 保持forward→元event列→保持reverseをq回 | scalar＋event product/order phase |
| R Retained-only | 元位置・時間の保持half sweep | scalar phase |
| E Event-only | 元event順序・角度・符号だけを同じstep順で並べる | event product/order phase |
| 0 Empty | 同じcosine/sine Hadamard wrapper＋measurement | scalar/event phaseなし |

全構成はcentral-rotation制御、同じ12 system qubit＋ancilla、P=0、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、同じrouting境界。scalarはRへ割り当て、eventの負係数・(-1)^(k/2)はEへ割り当てる。Fは両者を一度ずつ持つ。body global phaseはcontrol後のancilla相対phaseなので捨てない。R64のRにもconstant phaseがあり、L0だからEmptyと同じとは限らない。

保存QPYからFの価格・metricsを再利用し、R/E/0だけを新build/compileする案とする。Fを含む全構成の未compile program、roles、位相所有、compiler設定を保存する。R/Eのsemantic programがFの該当部分に対応することを先に確認する。

~~~
ΔC_interaction = C_F - C_R - C_E + C_0
contextual retained increment = C_F - C_E
contextual event increment    = C_F - C_R
~~~

RZ/CXについて既存D1の定義を再利用する。depthは同じ加法式の原因配賦には使わず、各構成の記述指標として保存する。各差はその固定program・削除操作・compilerに条件付けられる。Gaussian、control、pass別の独立寄与率ではない。**ablationへ元Nを掛けず、新しいG・方式順位を作らない。**

### 5.2 最小の対象と取得範囲

保存confirmation index0000を、H32/H64/R64で価格を見ずに固定する案とする。H32/R64はfamily比較、H64/R64はmatched比較を補完する。RのHybrid保持回路はr32/64で同じなので再利用し、Emptyも両軸ごとに共通化する。

既存F6価格を使える場合、新規compileはR（Hybrid共通・純RTE scalar）4本、Eの3trajectory×両軸6本、Empty両軸2本、**最大12本の異なる入力program**。候補追加・sampling・RNG replayは0。Fの再compileは既定にせず、保存Fと再構成program/境界が合わなければcost段階をSTOPして原因を報告する。

これは1program/候補の機構診断であり、平均価格の標本数設計ではない。保存SEや希少event確率から母平均を判別できるnを今回推定していない。index0で希少orderがなければ、その作用を観測したと主張しない。平均・tailを追加対象にする場合だけ、保存分散・層確率・判別幅に基づく別設計を行う。

### 5.3 再利用できるもの／必要な拡張

D1のevent復元、grouping、phase付きparts、interaction式とその人工testsは再利用候補。ただしD1 Pauli buildはgeneric control・cosine固定なのでそのまま使わない。[D1 adapter][D1ADAPTER]、[tests][TESTREPLAY]。D2 central_control_bodyと既存両軸hadamard_wrapperを使う。[CTRL][CTRL]、[tests][TESTCTRL]。

公開H6 Hybrid archiveにはFのinput/compiled QPY、spec/law、target、price resultがある。固定r32/index0の保存resultはcompact event schema（order/products/rotation/angle/phase/signs）で、control_sidecarは永続化されていない。一方R64はfull event schemaである。D1 event_from_recordへcompact recordを直接渡せない。最小拡張は2schemaを元のSOURCE通りのbody tokenへ復元し、保持/event tagを付けてpartsを出す処理である。確率や角度を再生成せず、保存値を使う。[Hybrid索引][H6ARCH]、[RTE索引][EVARCH]、[実際のprice_job][PRICE]。

QPYの保存byte SHAと、再構成した命令・qubit・parameter・phaseの意味的一致を分ける。新QPYの名前やserialization差だけで科学的同一性を否定しないが、元契約の固定input条件を緩めることもしない。F一致に失敗したprogramで旧Cの説明を続けない。全compiler pass traceの新取得は必須にしない。

## 6. 第三優先：DF対Pauliのmatched比較

### 6.1 H4の静的対応と不足

既存H4共通input runはlinear1Å/STO-3G、8qubit、中性singlet、同じ未threshold MO積分でPauli184とtol-only1e-8/DF10を生成した。receiptのfull-H一致誤差8.462237051574415e-15は既存の数値観測であり、厳密同一性ではない。[receipt][H4REC]。

Pauli/DFのCモデルのstate配列とstate_hashは同一だった。保存snapshotにはone_mo、eri_chemist、spin_one、spin_two、df_one_body、DF factors、両full matrixとstateが残る。referenceは両表現の独立診断があり、後続enclosure・raw norm・uはそれぞれの意味で扱う。[Pauli model][H4PM]、[DF model][H4DM]、[入力生成SOURCE][H4INPUT]。

元inputはinterleaved spin、chemist ERI→transpose(0,2,3,1)、two-body係数1/2、little-endian JW。DFは同じspin tensorから生成するが、一体補正を吸収する。既存Pauli prefix27とDF prefix1＋one-bodyは物理的に同じ分割ではない。scalar/identityも別途対応させる。legacy DF12、長時間H4 DF10の別snapshotをこの共通inputと自動joinしない。H6は別Pauli入力生成とDF weighted-Hermitian completionがあり、初手で統合しない。

### 6.2 分割の定義を先に決める

bare電子Hamiltonianを cI+T_bare+V_normal、DF平方表示を cI+T_eff+V_square と書く。SOURCEではT_eff=T_bare+one_body_correctionであり、V_squareは二次量の平方なのでnormal-order後に一体・identity成分も含み得る。

- **既存B3に合わせる案**：同じT_effを保持し、同じ演算子V_squareをPauli/DF辞書で表す。実装再利用が多いが、「bare二体項だけをrandom化」と呼ばない。
- **bare一体／二体を合わせる案**：T_bareを保持しV_normalをrandom化。DF側に平方表示の一体補正を戻す必要があり、既存B3そのままではない。追加辞書・生成法の設計が必要。

どちらを採るかは未決定で、H6初期bias/cost batchの前提にしない。full Hの一致だけでは、この選択は解消しない。

| 比較arm | 固定する物理operator | 判別できること／なお交絡すること |
|---|---|---|
| M_A Pauli全非identity純RTE | common full H、constant保持、H_D=0 | 従来L0のanchor。M_Bとの差には保持とlaw/normalizationの両方が入る |
| M_B 一体保持Pauli-tail | 選んだ同じTと残差VをPauli表現 | 物理的保持の変更を比較できるが、B/bias/N/角度も変わる |
| M_C 一体保持DF-tail | M_Bと同じT/V、DF residual辞書 | M_B対M_Cは辞書・law・Gaussian/control/compileの総合差。表現単独の効果ではない |

M_B/M_Cでは同じTの保持block列・scalar・controlを共通化する案を優先検討する。既存Pauli prefix APIはGaussian一体blockを注入できないため、小さなblock接続が必要。共通保持を「厳密指数診断」にする場合と、同じGaussian実装にする場合は別taskであり、実行前に選ぶ。

normal-ordering・係数1/2・orbital order・JW label・scalar移管・identity抽出・sectorを照合し、**H一致**と**T/V別の一致**を別gateにする。数値的なoperator再構成／一致評価やM_A〜C生成は新しい科学計算であり、今回は実施していない。

## 7. 実行順序・受入・停止

1. **固定版の再利用と最小実装・人工検査**：将来許可後にだけ実施。共通SOURCE/input/lawの結合、A/D読取、B/C/Dpoly切替とcost replayを最小拡張する。旧研究runner/grantを起動しない。
2. **H6 bias batch**：B/C各1＋Dpoly4、A/Dgrid再利用。まず可換保持B=CとL0 B=C=Aを確認し、複素閉合・両軸差・数値分解能を記録する。
3. **必要なら条件付きshot感度**：分解が数値的に解像された場合だけ、同じB/ε/α/ρ政策でNの感度を求める。独立した方法のG順位は作らない。
4. **central価格batch**：H32/H64/R64の固定index0、既存F再利用、最大12program compile。bias batchとは別に承認範囲を選べる。
5. **H4 matched表示研究**：一体補正・scalarを含むT/V定義をユーザー/GPTが選んだ場合だけ別scopeを準備する。H2/H4追加bias、H6 DF統合、全探索は自動追加しない。

受入は特定仮説の勝利ではなく、比較条件の一致・再現可能な信号／programの対応・不確かさを超えた判別の有無を正しく報告できることである。B=C、閉合、norm/B適用、cosine/sine符号、scalar/relative phase、保存Dgridとの共通条件を確認する。

SOURCE/input/law不一致、reference/normの二重補正、期待する保持可換性の破れ、精度不足、非有限値、F再構成不一致、未保存event/角度、compiler境界の不一致では該当batchをSTOPする。精度不足は仮説の否定でなくUNRESOLVED。他worktreeの通常変更や一時的負荷だけでSTOPしない。旧STOPは維持し、新たな実行・retry・拡張を自動開始しない。

**本設計の公開・remote本文照合後mandatory STOP。次の担当：ユーザー/GPTが、bias・N感度・価格・matched比較のどの範囲を必要とするか判断し、実施する作業を明示許可する。Codexの現時点の科学実行作業なし。**

<!-- 固定結果carrierと科学実行SOURCEを本文で区別する。 -->
[PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md
[PAULIFIG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_claim_figure_map_20261101.md
[MECH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5432857b94c3714ea9f4d6031e493e8dfc308a91/docs/research/pauli_pure_rte_cost_mechanism_inventory_20261101.md
[LONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md
[B3PROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/scientific_provenance.json
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_d1_source_execution_20261024.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_d2_execution_20261024.md
[H6SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/SOURCE.json
[H6CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CANDIDATE_RESULTS.csv
[EVENTCSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/selected_cost_breakdown.csv
[H2PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/docs/h2_pauli_rte_validation_20261026.md
[H4PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/docs/non_df_h4_pauli_pr_search_20261025.md
[R2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_r2c_compiler_phase_proof_20261025.md
[ARB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_central_arb_pf_pr0_sine_20261025.md
[LAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md
[GAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md
[ADAPTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/pauli_signal_adapter.py
[LAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_pauli_pr_search.py
[CTRL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_d2_pauli_control.py
[TIGHT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/pauli_vector_tight_enclosure.py
[D1ADAPTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_d1_adapter.py
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/scripts/pauli_h2_h6_study.py
[H6ARCH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/data_archives.json
[EVARCH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/data_archives.json
[EVSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_rte_event_extension/2026-10-26/SOURCE_e755ecf581600192fd2745d7066e410cebd933d3.json
[H6MODEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/model.json
[DFSIG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e417ffc9f52b13c18f5493193ce510d858664259/scripts/h46_long_time_signal.py
[DFRECORDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/signal_records.json.gz
[H4INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py
[DFH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_hamiltonian.py
[TOYB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/finite_rte_phase_amplitude.py
[DFOLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/finite_rte_signal_validation.py
[H4REC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json
[H4PM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json
[H4DM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/df_native_model.json
[MFINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/artifacts/resource_applicability/track_a_mechanism_a82_factor_results/2026-10-19/integrated_report.md
[TESTCTRL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/tests/test_h4_d2_pauli_control.py
[TESTREPLAY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/tests/test_h4_d1_adapter.py
[M1S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md
