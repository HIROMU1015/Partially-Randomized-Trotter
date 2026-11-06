# 新A：Clifford+T資源評価に向けた限定技術調査

2026-11-07 JST、SERVER 2。固定参照 `a095a225ceae383027b300e1f31ec5ccbb1522fc`、専用branch `codex/pr-reference-clifford-t-survey-20261107`。

**strictな単一RZ合成・actual adjoint・区間guardは再利用できる。ただし新A向け全体pipelineとしての適用は未資格で、全support位相key、precision、誤差配分、期待価格の会計に追加の資格が必要である。** 保存7,603 recordが新Aを覆うとは認定していない。新しいT資源の価格・削減率・科学的GO/NO-GOは取得・判断していない。

本作業は**静的調査のみ**。新規合成、guard再評価、価格再集計、angle catalogue生成、数値再分類、optimizer、回路build/compile、simulation/sampling、shot再計算、科学testは実行していない。SOURCEをimportしてrunnerを動かさず、テキスト・既存JSON/schema・保存archiveの一つのrecordを読んだ。既存件数は保存auditから引用した。変更は本書・[REUSE_APPLICABILITY.md](REUSE_APPLICABILITY.md)と最小限の索引だけ。

## 固定条件と引用する結果

Linear H4、1.00Å、STO-3G、DF12。元Hの厳密groundを各shotで供給し、状態準備費を除外。Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、保存18round q/r。energy精度1.5936e-4 Ha、全失敗率.05、phase cap .4rad。Qiskit1.3.0/compiler A/native RZ-SX-X-CX/全結合、B′4-CX controlled-RZZ、reference F/G・actual inverse・vacuum P・2-register・8-CX CSWAP、D融合・basis共有・scalar・event符号・X/Y wrapperを変更しない。

登録14-CX Simon–Love型を参考に含め、主比較をB′対referenceとする。引用する既存結果は次の通り。再計算・再判定は行っていない。

| 既存のnativeモデルの指標 | ReferenceのB′比・状態 |
|---|---|
| 18round期待総CX | ≥4.681782%削減 |
| 18round期待総native DAG depth | ≥40.995952%削減、V5補強で維持 |
| 18round期待総RZ | ≥0.150597%増加 |
| 確定した任意角RZ部分 | ≥2.818200%減少。未確定を含む全任意角優越は未資格 |
| qubits | 9→17 |
| nativeでの十分shots | 8,598→8,606 |
| 総T・T-depth | 未取得 |

参照した固定資料：B′ `43bfc5918703e8dcd020be4fcf96bb8dfd99f909`、V3 `224440f740051ed5794e8db87a57908fc02dd02f`、V5 `285b77023542c3b39caea767e2a937dffa046a3b`、V5補強 `a095a225ceae383027b300e1f31ec5ccbb1522fc`。各REPORT・SOURCEは本branchの同じpathで保全される。特に[V3分類規約](../pr_reference_rotation_v3_20261107/CLASSIFICATION_RULES.md)、[V3 REPORT](../pr_reference_rotation_v3_20261107/REPORT.md)、[V5 REPORT](../pr_reference_depth_v5_20261107/REPORT.md)、[V5補強 REPORT](../pr_reference_depth_v5_reinforcement_20261107/REPORT.md)を参照した。

## A：既存strict合成資産

[track_c_h4_synthesis.py](../../../src/trottertracks/track_c_h4_synthesis.py)の `exact_word_matrix`、`strict_guard`、`acquire` が主な核である。

- 対象は `RZ(theta)=diag(exp(-i theta/2),exp(i theta/2))`。binary64 thetaを `as_integer_ratio()` で正確に取り込み、wordをcyclotomic ringで評価した後、外向き区間Frobenius上界を得る。これはspectral operator normも上から抑える。phase fitを用いない。
- cacheのspecは `abs(theta).hex():10^e:pygridsynth2.0.0:strict:seed17`。precision exponent、backend/version、strict規約、seedを区別する。2πへの数値正規化はしていない。
- backendは `GridsynthConfig(seed=17,up_to_phase=False)`。wordのH/S/X/T/t/W、sequence hash、T/T†数、global W、guard上界・精度・route・状態を保存する。取得時のepsilon/4のheadroomだけで保証せず、最終guardのepsilon包含を判定する。
- `strict_clifford_fast_path` も、近いπ/2倍角を**所定epsilon内で近似できるか**guardで判定する経路である。exact Clifford分類の証明ではない。

V3の[保存audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a095a225ceae383027b300e1f31ec5ccbb1522fc/artifacts/pr_reference_rotation_v3/2026-11-07/SYNTHESIS_REUSE_AUDIT.json)には7,603 record、963 positive binary angle key、pygridsynth route 5,376件、strict fast path 2,227件が記録されている。今回はその値を引用し、全件の再guard・再集計はしていない。raw archive先頭の1 recordについて、spec/status/route/guard仕様とfield schemaのみを確認した。

保存wordは `artifacts/track_c2_b_pilot/2026-11-06/raw_synthesis.tar.xz`、precision requestは `PRECISION_REQUESTS.json.xz`。採用された会計は **`verified_final/`** であり、初回結果や `verified/` 中間結果ではない。C2-B原pilot `78004943ed5b718c12841e80f55fddf1d1644364`、強baseline `602ca4b67b1037e221e910a2733444630c5add62` の保存資料・SOURCEを参照した。別条件Full Y4/LD12で得たT価格・予算・normalization=1を、新A Partial S2/LD3へ移さない。

`acquire` はcache miss時に合成を実行し、lock/outputも扱う。既存pathからの戻りではそれ自体が全record条件を再検証するわけではない。将来の読取再利用では、strong-baselineの `load_inputs` にあるspec/word hash/PASS/guard≤要求epsilon/T数・bundle provenanceの照合パターンを使う必要がある。旧moduleのOUTを付け替えて旧runを再開しない。今回 `acquire`・検証runner・保存check scriptを呼んでいない。

## B：新Aのkeyとの対応

読取確認した[V3角度台帳](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a095a225ceae383027b300e1f31ec5ccbb1522fc/artifacts/pr_reference_rotation_v3/2026-11-07/ANGLE_KEY_MULTIPLICITIES.json.xz)は、`tokens` と `rows` を持つ。tokenは分類・literal分類・signed angle key、rowはm/q/r/bits/arm/axis/shotsと構成部分ごとの有理multiplicityを保持する。D融合の差分はsignedな会計補正で、負のgateではない。

| 角度・役割 | 対応・未解決事項 |
|---|---|
| 数値が既知のsigned RZ | finiteなbinary値をpositive `abs(theta).hex()` keyへ対応させ、元の符号・role・roundを別に保持できる。負角は同じpositive wordのactual adjoint。signed zeroは同一identityだが、他の符号・周期まで消さない |
| B′/Referenceの共通部分 | 同じ保存Gaussian frame/inverse等には構造的な共有機会がある。実際の同一binary key・要求precisionを照合した場合だけwordを共用する。今回はcacheとのintersection件数・新Aの完全coverageを計算していない |
| RTE中央回転 | 登録/B′のcontrolled回転はlower後の±theta/2、referenceはtheta。sourceの制御付きthetaへ非制御RZ(theta)の価格を直接当てない。CXとrelative phaseも保持する |
| 数値は既知だがexact分類が未確定 | V3の未確定分類のまま、binary RZをstrict近似合成する入力にはできる。Clifford/Tへ再分類することは合成の必須条件ではない。予算内でC/T近似wordを採用してもexact型へ昇格させない |
| 解析的π由来の固定slot | 理想角と実native binary角は別target。literal nativeを合成するなら、そのbinary値へのguardが必要。理想SOURCE角を使う別評価には、置換誤差・compiler経路・joint gaugeを改めて定義する必要がある |
| event符号・scalar | numeric scalarは実keyを保持。symbolic integer-pi event phaseは実binary累積値と同じではない。単一のπやparity keyへ勝手に置換しない |
| reference真空位相 | V3では全supportの `symbolic:vacuum_phase[...]`。保存代表pathの数値を全support価格にしない。既存naive位相計算法を変えず、実key／価格上界・期待価格を保証する追加手段が必要 |

RZ(theta+2π)は符号が変わる。native joint回路全体の共通phaseと、control前のsource相対phaseを分ける。既存keyを2π moduloへ変えない。

理想T型RZはT/T†とscalarで表せるが、scalarにはπ/8位相が必要な場合もある。既存W（π/4単位）wordだけでstrict exact実装と宣言せず、全joint gaugeとrelative phaseを照合する。[SP-0.5 economics](../../../src/trottertracks/algorithm_codesign/synthesis_placement/economics.py)のsymbolic `exact_sequence/error_guard` は**projective**な経路であり、strict binary RZ cacheの代用品ではない。固定π情報を利用することと、近いfloatを無料のexact gateへsnapすることを区別する。

`tokens.origin` は `setdefault` で登録された最初のラベルであり、全出自の完全な一覧ではない。rowのpartsはrole別multiplicityを持つが、全gateのwire・時系列・conjugate pairを一意に復元できる台帳ではない。T-countの線形会計候補として使う場合にもsymbolic keyを解決し、非負の物理出現数へまとめる必要がある。block別精度・T-depthには、固定SOURCE/local loweringから必要な配置・pair対応を補う必要がある。

## C：precisionとRPE保証

以下は**後続作業の技術条件・提案**であり、今回の新しい精度・予算・optimizerの採択ではない。

### 誤差の対象と単位

理想SOURCEをU_S、compiler A後のnativeをU_N、合成後をU_CTと分ける。ここでUは測定前のunitary部分で、reset/測定は両側で同じ準備・読出しとして扱う。既存のSOURCE/finite-RTE・proposal・compiler・位相生成の保証を保持し、追加するのはU_N対U_CTの誤差である。strict wordの局所Frobenius guardからspectral operator上界を得て、unitary telescopingで全joint演算子へ接続する。制御・leakageを省いた粒子sectorだけの評価に縮めない。

新Aの [transfer_error.qualify](../../../src/trottertracks/pr_reference_transfer_error.py) と [B′ error_and_shots](../../../src/trottertracks/pr_reference_bprime_accounting.py) は、joint operator誤差を `2sqrt(2)` でraw complex-signal diskへ移す規約を持つ。将来のword置換が同じjoint測定を保つことを確認したうえで、合成operator boundにも接続できる候補である。古いnative compiler diskは一度だけ、既存のvacuum位相算術diskも既にraw単位のまま一度だけ使う。

新A既存式に合わせるなら、nominal raw半径ρ₀に対して、SOURCE等の共有disk・native compiler disk・位相生成disk・新しいsynthesis raw diskを同じ単位で合算したdを使い、

`rho_lower = rho0 - d`

`beta_stat = .4 - PF_phase_upper - asin(d/rho0)`

を評価する。d<ρ₀、正の半径・統計幅、既存alias条件が必要。既にcompiler分を差し引いた保存βへcompiler分を再び加えない。native angleのπ残差をstrict guardが評価済みなら、その同じ差を別名で再加算しない。評価targetを理想π角へ変更する場合は、U_S/U_N/U_CTのどの辺を評価し直すかを明示する。

C2-B補正後は追加compiler diskと合成diskを**raw単位で合算して一度**phaseへ変換した。初回補正前の幾何やgamma_4096補償の機械的流用はしない。新Aの旧disk・既存丸めモデルが何を包含するかを対応付ける必要がある。

誤差保証には全supportの回路上界を使う方法と、固定lawのcoherent meanへの誤差伝播を別途導出する方法が考えられる。後者で期待multiplicityを用いるなら、triangle inequality、word policy、位相key、測定range、固定sampling lawの条件を証明する必要がある。保存された平均個数・cohort価格だけから保証を自動的に移さない。C2-Bのtail-free normalization=1をPR信号へ与えず、既存attenuationを保持する。

### 公平な精度・shot

候補は、共通精度、既知の非一様配分、又は成立条件を満たす共役block配分である。どの有限precision候補・raw予算・alpha自由度を用いるかはGPT／ユーザーの判断事項。両armに同じrecord・新規価格取得の機会と、同じ程度の配分最適化機会を与える。C2-Bの旧2精度・2 cap・alpha scheduleを新Aの採択値にはしない。

共役誤差式は、戻りが**合成したBの実adjoint**であることなどが条件。V5補強のinverse span PASSは、`Compile(F.inverse())` と `Compile(F).inverse()` のword/operator同一性の保証ではない。F†/Gは別registerの異なる演算であり、C2-BのBDB†重みをF†/G全体へ適用しない。構造的配分を使うなら、固定native構成に適合するpairと全空間誤差式を個別に資格する。

さらに、新Aの期待Tは有理multiplicityを持つ。C2-Bの `allocation_lp_lower` のceil、strong baselineのgcd格子・整数ceilは整数価格の前提を持ち、そのまま移植できない。再利用できるのは有限選択・有理feasibility・実離散word価格を使う考え方であり、新Aでは有理期待価格に適合する下界が必要。LP下界は実行可能baseline価格でもない。C2-B強baselineの判定自体も `OPTIMALITY_UNDETERMINED` であり、新Aの最適性や構造配分の利益を示す結果ではない。

nativeの8598/8606 shotsは追加合成誤差込みの資格ではない。将来は、同じNで十分条件を再確認するか、同じenergy精度・全失敗率.05の下で整数Nを再評価する。既存 `shots` helperは候補だが、rho/beta/alphaの規約と外向き丸めを揃える必要がある。alpha再最適化を認める場合も両armへ同じ自由度を与える。一様alphaのLP下界を別alphaの最適性証明へ使わない。

## D：T-countとT-depthの比較対象

`track_c2_b_verify.lower_native` はwordを逆順にgate列へ展開し、負角にactual adjoint、Wにglobal phaseを保つ。SX→HSH、X/CXも明示する。単位block向けなので、new A wrapperのmeasure/barrier/reset、role・precision・wire対応を保持する拡張が必要。Wはphase bookkeepingであり物理T gateではないが、control前のrelative phaseとして勝手に捨てない。

全構成を含める必要がある。D・境界融合、RTE rotation、basis/inverse、product、scalar/event phase、vacuum P、CSWAP、X/Y wrapperとfresh reference初期化を落とさない。

- B′のZZ中央は4 CXと±theta/2の2 RZ。登録14-CX経路は24 RZ、reference中央は1 RZ。既存のsource T型slot数だけでは各literal-native RZの合成費を決められない。
- 固定8-CX CSWAPはV3で11 RZ（解析的Clifford由来4、T由来7）と記録され、16 CSWAP/shot。これを未評価T=0にしたり、別7-CX gadgetへ変更したりしない。
- native SX/X/CXは論理Cliffordとして扱える。一方、SOURCEでH/Sdgと呼ばれる部分にも、compiler後にはbinary RZがある。literal-native評価でそのRZを無条件に無料Cliffordとせず、target/gauge/誤差規約に従って扱う。
- referenceの追加8 qubit・8 resetと測定は保存構成に含める。stabilizer初期化のT費用と、実際の初期化時間・qubit費用は別指標である。

wordをcacheで共用しても、回路中の全出現とactual adjointはそれぞれ課金する。`T` は実wordのT/T†数、最小T数ではない。

既存recordの `depth` はWを除く1qubit wordのgate数、`t_metrics.depth` は合成後gate depthで、どちらもT-depthではない。T-depthは実wordのchronological配置とwire依存、Clifford処理・reset/測定の規約を定めた評価が別途必要。V5のnative frontierを再利用する余地はあるが、RZをwordへ置き換えた回路のT-depthへ40.995952%を移せない。

| 今後埋める指標 | B′ | Reference | 今回の状態 |
|---|---|---|---|
| 十分shots | native 8,598、合成込み未評価 | native 8,606、合成込み未評価 | 保存値の引用のみ |
| 総T-count | 未取得 | 未取得 | 同一成功条件での期待価格／上下界が必要 |
| T-depth | 未取得 | 未取得 | word DAGと全supportの資格が必要 |
| 任意角の合成費 | 未取得 | 未取得 | 既知numeric keyもprecision未採択 |
| 固定Clifford/T由来slotの実費 | 未取得 | 未取得 | binary targetとphase規約を照合 |
| CX・native depth | 既存結果 | 既存結果 | 旧native shot条件での参照値 |
| qubits | 9 | 17 | 固定 |

合成込みNが変わればshot加重総CX等も変わるので、旧百分率をFT全体費用の保証として据え置かない。RZ・CX・T・depth・qubitを恣意的な重みで合算しない。

## E：最小限の後続案とSTOP

以下は未実施の提案で、今回の許可には含まれない。

1. **GPT／ユーザー**がliteral-native対理想SOURCE/gaugeの評価target、有限precision候補・回路/round予算、通常加算対構造配分、alphaの自由度、期待T対安全上界、T-depthの対象範囲を判断する。数値は今回固定しない。
2. **Codex（別途明示承認後）**がV3の既存key/schemaと固定SOURCEから最小限のcoverage照合を行う。新しいcatalogueが必要な場合はそのscopeを限定し、numeric keyとsymbolic phaseを分ける。既存recordだけで可能なのは、実binary角・precision・specが一致し有効guardを持つ部分のword再利用。新規合成が必要になるのは、その照合で欠けた有限key/precisionであり、現時点の件数は未確定。
3. 合成する前に、既存naive真空位相・event位相の**全support**に対して、実key又は保証された価格/誤差上界を用意できるかを判別する。平均位相や代表pathへの置換はしない。有限cohortだけならその資格に限定し、全18round値へ格上げしない。
4. 取得が承認された有限keyだけに既存strict engineを使い、word順序・±角・W・actual adjoint・control後のjoint演算・wrapper/resetを最小の人工/局所検査で確認する。旧科学suiteを丸ごと再実行しない。
5. 非負の実出現数／保証された期待multiplicity、未知位相部分、実word価格、有理目的量に適した配分・下界を接続する。両armに一様・強い非一様baselineを与える。T-countは先に資格できる部分を区別し、T-depthは配置情報と非線形期待値を独立に資格する。
6. raw diskを重複なく合算し、必要整数shotと全18round `sum N * E[T]` を評価する。旧native結果と新FT会計を分け、得られた差を有限候補・仮定の範囲で報告する。

後続でのSTOP候補は、strict guard不成立／word・phase規約不一致、必要key/全support位相価格の欠損、正半径・統計幅・alias/失敗率条件不成立、baselineと候補で情報/価格機会が不公平、変更条件や承認外候補が必要になった場合、又は許可範囲完了。未取得を0埋めせず、全support資格がなければUNQUALIFIEDとして戻す。結果を良くするためのprecision・law・gadget・PFの自動拡張はしない。

## 終了記録

指定の静的調査を完了した。次の数値的判断には、上記の評価target・precision・予算・実行scopeの採択と新しい実行承認が必要である。**mandatory STOP。** 旧新A、C/C2、新B/C、P-D、BF/BM、PR-3等のSOURCE・raw・結果・grant・ledger・marker・STOPは変更していない。

SERVER 2のaffinity64、利用可能RAM・memory pressure・diskを読取確認した。処理は小さなSOURCE/保存schemaの読取で、科学workerを起動せず、共有設定・他者process/fileにも変更していない。次担当は**GPT／ユーザー**。本調査はT資源削減・科学的新規性の認定ではない。
