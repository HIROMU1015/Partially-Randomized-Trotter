# A1 科学実行契約草案：保存DFモデルのm0・X/Y診断

2026-11-06 JST。**実行前の提案。科学計算の承認・実施済み結果・正式な科学的GOではない。**
固定引き継ぎ元は `6e8d099a5252d3b424045fb22aeb112d5ebad5c9`。
取得と版照合は [HANDOFF](A1_SERVER1_HANDOFF.md)、残る判断とmandatory STOPは [REPORT](REPORT.md) にまとめる。
本書の数式は既存A1計画と既知の有界標本保証を具体化するもので、新定理・独立科学レビューではない。
新しいCDF/moment/variance/N/費用の数値評価はしていない。候補shot数も新しい科学結果から逆算していない。

## 1. 対象・state・情報アクセス

| 項目 | 提案する限定scope |
|---|---|
| モデル | 保存linear H4、隣接1.00 Å、STO-3G、legacy DF rank12、8qubit、4電子sector |
| snapshot | [元carrier][CARRIER]内の既存NPZ。SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| Hamiltonian hash | `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` |
| 分割・round | `L_D=3`、最終opt2 m0/q1、δ=t=.02、r1/K2、X/Y両軸、一つのRTE occurrence |
| state | 保存DF Hamiltonian全体のphysical-sector ground。記録fingerprint `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`。H_D groundやPF有効固有状態controlと区別 |
| 回路 | canonical PR、ordinary controlled `diag(I,U)`、既存 `support_run_le_1` とboundary処理、測定付きfull no-state-preparation wrapper |
| 主指標 | compiled RZ count。CX/RZ depth、CX count、total depth、size等は補助。T/nativeは未取得 |
| primary target | opt2の固定SOURCEで定まる有限・attenuated canonical PR target。actual-qとの差は§2で別会計。normalization補正済みsignalへ変更しない |

研究全体では元の物理Hamiltonian Hの厳密基底状態を利用できると仮定し、状態準備費を除外する。
今回の候補は**保存DFモデル限定**であり、未近似の物理Hとのoperator/state bindingは未閉鎖である。この仮定を満たした実装の保証とは呼ばない。
元H→保存DF、DF tail threshold、PF、finite RTE、actual finite-grid law、native/数値実装を別の接続として保持する。

測定engineが内部的にstateを必要とすることと、推定器がその完全vectorを古典情報として利用できることは別である。
真E0、gap、state vector、path別の未知期待値、真variance/covariance、最適λ/配分をfeature選択・fit・N決定へ渡さない。
既存V1/V2の保存meanは条件・来歴の照合用であり、fresh outcomeの代わりにしない。E0を使った事後の参考評価を求める場合は別scopeとして明示する。
新しい分子、geometry、basis、rank、round、δ/r/K/degree、feature bankの探索は含めない。

## 2. 固定lawと実CDF・既知平均の取得経路

### 2.1 静的に確認したcanonical generator

`src/trotterlib/rte.py` の `finite_rte_distribution` → `make_rte_config` → `iter_sample_rte_events` → `_make_event` を再利用する。
K2ではeven order `n=0,2`。各eventについて先にorder、次に復元抽出でn+1個のlabelを選ぶ。
内部抽出配列はrotationが先、残りがproducts。保持される `selected_component_ids` とA1の `L_j` は **products → rotation** の適用順である。
n0のL0はrotation、n2のL0/L1はproducts、L2はrotation。n2 controlをrotationの条件と取り違えない。
phaseは `(-1)^(n/2)`、coefficient_signはsign-absorbed involution/controlled circuitへ保持される。

orderには保存distributionのbinary64 `order_probabilities`、labelには `_validate_components` が返す和で各 `component.probability` を割ったbinary64配列を使う。
生成器は `Generator(PCG64(seed))`。NumPy 1.26.4の[choice SOURCE][CHOICE]は累積和→最終CDF要素による正規化→right searchである。
[PCG64 SOURCE][PCG]と[double変換][DOUBLE]は `uint64 >> 11` による53-bit格子へ対応する。
固定seed列の決定論性をIIDの証明とはせず、既存のfresh-IID random-wordモデルを明記する。

### 2.2 実行許可後に取得するもの

1. 固定NPZとopt2の `_worker_preparation` に対応するpreparation、fragment順位、identity policy、`coefficient_atol=0.0`、tail hashをbindする。完全な順序付き `symbolic_tail.components` を保持する。
2. 同じSOURCE/演算順でorderとlabelのchoice入力配列・normalized binary64 CDFを取得する。既に保存された完全payloadを取得できれば再利用する。今回それらへの科学callは0。
3. その実CDFだけを既存 `law_from_choice_cdf` へ渡す。境界c_jに対する格子質量は `(ceil(2^53 c_j)-ceil(2^53 c_(j-1)))/2^53`。order質量をq0,q2、label質量をq_lと書く。
4. `OccurrenceLaw` → `ControlBank` でν・joint momentを得る。binary64境界をexact Fractionとして扱う範囲と、実行時CDF/NumPy/SOURCEの対応を記録する。Parentのdyadic proposalやideal probabilityを実qに置換しない。
5. 支持消失、CDF/生成順の不一致、同一positionの誤解、宣言外の相関があればSTOP。独立stream名だけで独立性を認定しない。

canonical unweighted streamは元のweight w=1を維持する提案で、adapterには明示的にbindする。一般の補正weightを使う場合も元定義を固定し、`|w|≤W`を保証する。
`Z_b=wY_b`、`Y_b=1−2 bit`。phase/signやSineの符号を後処理で二重に掛けない。

### 2.3 target・threshold・centeringの残課題

同じH hashでも、WP04保存signalは `coefficient_atol=1e-12`、preparation hash `89145d…`、218 components、opt2回路側は既定 `0.0`、preparation hash `50c77f…` である（完全hash・保存欄はHANDOFF §5）。
**WP04の218-label法をopt2の実lawとして使わない。** 推奨はopt2の既存lawを維持し、保存signal/予算とのthreshold/model差を許可後に上界化・接続する経路である。
差が小さいとの推測で0としない。あるいは対応する保存済みsignal/CDFを回収する。どちらも今回数値評価していない。

実qの平均を `μ_q=E_q[Z]`、元algebraic finite targetをμ_finとし、全方式に共通の `b_law≥|μ_q−μ_fin|` を保持する。
同じbounded path応答・支持が対応するなら `W Σ_ω |q(ω)−p_alg(ω)|` 等の既知評価を検討できるが、νを実qで正しくcenterするだけではこの差は消えない。
補正weightの無断変更・支持外補完で解決しない。正しい差の上界が取得できなければ測定開始を保留する。

νの実装誤差をe_jで上界化した場合、CV固有の `b_center≤Σ_j |λ_j|e_j` を控除する。
正しいCDFと独立factorized lawのexact rational入力なら、その有限格子モデルのcenteringは厳密に扱える。未確定のSOURCE bindingやPRNG相関までe_j=0とみなさない。

## 3. 結果前に固定するcontrol bankの提案

### 3.1 label IDの選択規則

完全catalogを今回計算していない。利用できた既存[WP06a metadata][CATALOG]の `representative_events` には、元snapshot/preparation hash `50c77f…` に結ばれた次の2 IDがある。
`df-fragment-6:Z0` と `df-fragment-3:Z0Z1`。保存 `coefficient_abs>0` であり、符号・未知path平均・測定応答を使わず候補にできる。

**提案規則**：この固定metadata配列のpositive coefficient_absかつ非identityの既存IDだけを取り、ID文字列を昇順に並べ、最初をa、次をbとする。
従って提案IDは **a=`df-fragment-3:Z0Z1`、b=`df-fragment-6:Z0`**。
格納されたcompiled cost、operator residual、signal、coefficientの大きさによる順位は使わない。全catalogの「最良label」とは呼ばない。
元の代表例選択を新しいCV効果の証拠にはしない。

この規則・IDは未承認の実行計画案である。許可後のopt2実CDFにおいてq_a,q_b>0、q0,q2>0を確認し、満たさなければSTOPする。
結果後に別labelへ差し替えない。完全catalogが取得できなかったという不足をC1の別324-label packetで埋めない。

### 3.2 一つの固定bank、3 features

occurrence=0だけを使い、X/Yは同じbank、係数だけ別に学習する。

| feature | `LocalControl`相当 | 既知平均（実q） |
|---|---|---|
| g1=`1[n=0,L0=a]` | `(0,0,((0,a),))` | ν1=q0 q_a |
| g2=`1[n=2,L0=a]` | `(0,2,((0,a),))` | ν2=q2 q_a |
| g3=`1[n=2,L0=a,L1=b]` | `(0,2,((0,a),(1,b)))` | ν3=q2 q_a q_b |

各gは[0,1]。二乗は自身、g1g2=g1g3=0、g2g3=g3なので、joint momentsは `ControlBank.joint_mean` で既知lawから得られる。
同一positionは一度だけ掛け、矛盾条件は0にする。未知量子共分散をこの既知古典momentと混同しない。
order-only `1[n=0]` は層別対照の補助metadataであり、CVの第四featureへ結果後に追加しない。
bank・feature数・fit規則・label IDをoutcome取得前に固定する。

## 4. 比較集合と強い層別対照

比較はB0、B1-O、B1-OL、A1の4方式。全て§1の同じfinite target・q・w・selected quantum policy・X/Y・ε/αを用いる。
条件付き取得はqを層で分解して元質量で戻すもので、canonical PRを別proposalへ置換する研究ではない。

| 方式 | 推定・利用情報 |
|---|---|
| B0 raw canonical PR | `raw_mean`。CV用trainは不要。calibration＋variance-sensitive保証も認める。旧range-only rawだけを対照にしない |
| B1-O | n=0/n=2の全正層。既知q0/q2による層平均合成、実行可能な有料pilot/配分 |
| B1-OL | 下表のorderと3featureの共通細分。全正層の飽和層平均、同じ古典情報からのcost-aware配分 |
| A1 | `fit_and_freeze`で固定bankのλをfitし、`X=Z−λᵀ(g−ν)`。train/cal/prod全費用 |

B1-OLの分割は次の5層。零質量層だけを**lawによって**除き、観測されなかった正層を除かない。

| 層 | 質量 |
|---|---|
| n0, L0=a | q0 q_a |
| n0, L0≠a | q0(1−q_a) |
| n2, L0≠a | q2(1−q_a) |
| n2, L0=a, L1≠b | q2 q_a(1−q_b) |
| n2, L0=a, L1=b | q2 q_a q_b |

補助order-only indicatorを持つ `ControlBank` と既存 `pattern_mass` / `stratified_mean` により表現可能。qのfactorization/supportの確認は必要。
gは各層で定数なので、この分割で飽和した層平均へ同じCV定数を足しても、正確な層質量で合成すると補正が相殺する。
そのためorder-onlyに対する改善だけをCV固有利益と呼ばず、B1-OLを主要な強い対照に含める。

**取得法の提案**：既存canonical generatorの古典pathをまず生成し、測定前metadataで層を選別する。受理したpathだけ同じ量子回路・測定を行う。
棄却pathの古典生成・選別時間とmemoryを課金する。量子測定後の選別、空層の削除、未補正IID poststratificationは行わない。
直接条件付きsamplerは今回新規に作らない。将来必要なら実q条件付きlawとの同一性を確認する小adapterに限る。
全正層の受理shotを取り切れなければ不完了としてSTOPし、別targetへ再正規化しない。

## 5. Train／calibration／productionの有限scope案

以下は**採択前の候補予算・規則**であり、新しい十分Nの数値発行ではない。旧510はどの段階にも流用しない。

| 方式・段階 | 提案 |
|---|---|
| A1 train | 各axis 1024 fresh canonical shots、1回だけ。intercept付き既存OLS、3 slopesを凍結。3featureに対する有限pilotを明示する値で、精度・rare feature観測の保証値ではない |
| B0 train | 0。不要なCV学習費を課さない |
| B1 train/pilot | 各正層・各axis 64 conditional shots、1回だけ。層内sample varianceと価格metadataによる配分用。真variance/最適配分は使わない |
| calibration | 各評価変数（B0/A1はaxisごと、B1は層/axisごと）で独立fresh標本。後述の事前固定n_cal式。追加較正や結果を見た増量なし |
| production | 独立calibrationから十分N/層配分を決めて凍結し、fresh trialで1回。候補の測定scope上限は**各方式・各axis合計8192 production shots**。これは実行回数の上限案で、必要NやCPU上限ではない |

calibrationの候補設計は `η=1/8` とし、§6の該当δ_calに対し

`n_cal = max(2, 1 + ceil(2 log(1/δ_cal)/η²))`

を**outcome前**に固定する。ここでlogは自然対数。
理由は§7の標準偏差上界の加算項をηD以下に設計するため。真varianceや効果量を予知した数ではない。
層別のδ_calは層数で分けるので、そのn_calはB0/A1より増え得る。総較正費を必ず単独方式費用へ含める。
ηを緩くすればcal費が下がる代わりにproduction上界が緩くなり得る。初期値1/8を効果最適と主張せず、ユーザーが別値を選ぶ場合は全outcome前に一つへ固定する。
この式への科学入力による数値代入は今回しない。

OLSのrank不足は既存 `lstsq` のminimum-norm解のまま記録し、未観測rare featureを学べたと呼ばない。feature削除・ridge探索・再fitを行わない。
提案するrangeの事前guardは `D_CV≤4W`。超えれば係数をclipせず、その候補を不成立としてSTOPする。guardも未承認の運用案であり、効果の定理ではない。
1024でもn2/controlの変動を十分観測できない可能性がある。結果後のpilot延長・有利なseed選択で救済しない。

条件付き取得の有限性には、正層質量p_hと必要受理数M_hを用い、各層の古典生成attempt capを `ceil(2M_h/p_h)` とする候補を提示する。
train/cal/prodを含むM_hの確定後、測定前にcapを決める。これは完走確率の保証ではなく、棄却による無限取得を防ぐscope制約。
cap到達時は不完了STOP。極小p_hのため実際の取得が許容できなければ、費用を隠さず取得法/予算を実行前に判断する。
失敗・未完了の取得費も記録し、成功したcampaignだけを費用比較へ選ぶことはしない。

axis×方式×stage×stratumごとに独立stream/trialを設ける。既存PCG64/fresh-trajectoryの規約に従い、測定noiseのstreamも分離する。
trial/stream/trajectory/wrapper/axis/weightをbindし、`Batch` / `require_disjoint` / `evaluation_inputs` を再利用する。
旧seedや同じbitをtrain/cal/prodに使い回さない。各段階で新しい承認手続きは要求せず、最初の明示許可の有限scope内で進む設計とする。

## 6. 精度・失敗確率配分の提案

対象は§1の保存DF12/L_D3/δ.02/m0/q1/r1/K2、opt2 selected policyだけ。
[opt2][OPT2]の保存欄には、coordinate ε=`0.266146579872968`、各axis α*= `2.8897922699779505e-08`、旧N=510がある。
これらは18round資源モデルの一部で、PF有効固有状態/単位基準半径等の条件付きモデル由来である。旧NはCV用の十分Nではない。

**推奨案**：εを保存DFのfinite-target座標比較の参照幅として採り、X/Yは各α*、4方式を同時に報告できるようaxisごとに等分する。

`A_method,axis = α*/4`

各方式・axisで `δ_cal=A/4`、`δ_prod=A/2`、`δ_price=A/4` とする。
B1はH個の正層に対し `δ_cal,h=δ_cal/H`、`δ_prod,h=δ_prod/H`。複数価格上界が要る場合もδ_price内で層へ等分する。
4方式・2axisのunionは2α*以内。これは本診断の同時報告用配分で、α_total=.05を新たに全部使ったり、18roundへ再配分したりするものではない。
trainを条件付けた後の保証を使うため、学習成功に別の仮説検定αは使わない。学習精度保証やvariance改善の有意検定を行ったとの主張はしない。

ここでprice用配分は母平均費用の保証を付ける場合の予約であり、実際のcompiled count合計そのものの丸め誤差を確率化する意味ではない。
母平均価格を未知のまま実現campaign費用だけを報告する選択肢もある。その場合は期待総費用改善のconfidenceを主張せず、未使用αを結果後に再利用しない。
別のα/εを選ぶ場合も、全方式・X/Y・層・段階の公平性を維持し、測定前に一つへ固定する。

## 7. range・variance・十分N

各axisで `|Z|≤W`、凍結済みλ、既知νに対し、既存 `FrozenCV.output_range` のbox enclosureを使う。
下端は `−W+Σ min(λ_jν_j,−λ_j(1−ν_j))`、上端は `W+Σ max(λ_jν_j,−λ_j(1−ν_j))`。
従ってrange長は `D_CV=2W+Σ|λ_j|`。B0と未補正の各層はD=2Wを使える。sample最小最大でrangeを狭めない。
indicatorのjoint制約を使ったtight rangeは初期契約には追加しない。

独立固定n_calの不偏sample varianceをs²とし、[Maurer–Pontil v1 Theorem 10][EB]をrange変換して

`V_U = min(D²/4, (sqrt(s²)+D sqrt(2 log(1/δ_cal)/(n_cal−1)))²)`

を使う計画。λ固定条件下でIIDなcalibrationを必要とする。s²=0でも真variance=0としない。
全てのcalibration上界の失敗配分を§6へ戻す。

元finite targetへの座標許容幅をε_finとし、

`ε_rem = ε_fin − b_law − b_center − b_numeric − b_native − b_other_unincluded > 0`

を要求する。保存εをそのまま採るだけで正残余が保証されたとはしない。
PF/RTE位相配分を既に控除して作られた保存εをRPE用途へ戻すとき、同じPF/RTEをcoordinateへ無換算で二重控除しない。
threshold差の扱い、native/noiseを対象外とする理想gateモデルの採否、未包含biasをどこで予算化するかを測定前に閉じる。
T/native価格がUNKNOWNであることと、モデル上の理想gate仮定は別である。

fresh productionの二側bounded Bernsteinによる保守的な十分数の候補は

`N_B = ceil(log(2/δ_prod) (2V_U/ε_rem² + 2D/(3ε_rem)))`。

同じδ_prodに対するHoeffdingの `N_H=ceil(D² log(2/δ_prod)/(2ε_rem²))` も全方式に認め、calibration後・production前に十分条件として小さい方を選べる。
選択は独立calibrationだけに依存し、productionのpeeking/任意停止はしない。Nが上限を超えるなら切り詰めて成功を宣言せずSTOPする。
対数・sqrt・ceilingを含む将来実装は保守丸めを行い、閾値付近は外向き評価等で十分性を確認する。今回この数式の実装/数値計算はしていない。

B1では `μ_hat=Σ p_h mean_h`、`Σ p_h ε_h≤ε_rem` とする。
既知p_h、pilotからの層価格、独立calibrationのV_U,hを使い、上の十分数を各層で満たすようε_h/N_hをproduction前に選ぶ。
全正層にN_h≥1を課し、cap内で実行可能なcost-aware配分を選ぶ。真層内variance・oracle最適配分は使わない。
整数配分の最適性が証明できなくても保証を満たす実行可能配分を報告できる。ε_hの選択・N_hはcalibrationに条件付けるので、fresh productionで各層の保証を適用する。
各層 `δ_prod,h` をunionし、各層誤差をp_hで加重する。cal/pilotをproduction平均に混ぜない。

B0には不要な学習を課さず、固定nのempirical Bernsteinを使う別の正当な運用も認める（既存A1計画参照）。
複数routeを測定後に都合よく選ぶ場合は別の同時保証が必要。初期契約では上表のcal/prod routeを一つに固定する提案とする。
旧V2のbinary PMF/tailを、非binary CV推定器へ無条件適用しない。
sample varianceの低下は記述的結果として報告できるが、本設計だけで厳密なvariance reductionの検定を実施したとはしない。

## 8. 保存RPE信号と系統誤差の条件

以下は§1のH4/DF12/L_D3/δ.02/m0/q1/r1/K2についての**保存欄の引用**。新しい計算値ではない。

| 保存欄 | 値・解釈 |
|---|---|
| opt2 beta_PF / beta_RTE / beta_stat | `.014050012886294481` / `9.899584668737594e-08` / `.38594988811785885` rad |
| opt2 attenuation | `.9998639385601628` |
| V1 physical PF radius | `.9999999999999549`。厳密な下界1とはしない |
| V1 physical observed radius | `.9998639385285691`。実行済み保存pointの数値 |
| V1 saved physical radius lower bound欄 | `.9998639377868259`。数値的model診断欄で、今回の厳密な証明ではない |
| V1 physical PF円周残差 / finite-RTE残差 | `1.0672645894827473e-07` / `2.5393576131542425e-13` rad。既存保存値、threshold差を残す |

本診断のfinite mean推定には正の信号半径を仮定する必要はない。RPE位相/energy保証へ変換する場合は別途、正当な正のρ_lbとPF/RTE/native/law biasを要求する。
CVはmeasurement estimatorを変えるだけで、attenuation・PF/RTE biasを自動的に消さない。
保存near-one、PF controlとの高overlap、元H ground利用仮定からρ_lb=1を導かない。
X=cosine=real、Y=sine=imaginary、Y readoutはSdg→H、bit0→+1/bit1→−1。
`exp(−itH)`、`energy=−phase/δ`。constant/identityは元controlled circuitに含まれ、二重加減しない。

V1の条件付き性とV2の `CONDITIONAL_SUCCESS` を維持する。V2の保存18round branch成功は旧binaryモデルに条件付けられている。
将来全RPEへ進むにはalias-free初期energy window、同一phase convention、nearest-branchの適用条件、全roundへの誤差/α合成が別途必要。
今回1roundで分散や費用が改善しても、全18round成功・energy精度改善・最終総コスト優位を示したとは扱わない。

## 9. full価格取得と費用会計

Qiskit1.3.0、`rz,sx,x,cx`、opt level2、seed17、backend/coupling/layout/routing指定なしを固定する。
全D half-steps、中央RTE、ordinary control、相対scalar/identity phase、ancilla、X/Y readout、測定を含む `single_hadamard_interrogation_without_state_preparation` が価格scope。
元opt2 `support_run_le_1` / boundary policyを共用し、標準full-basis provider価格・opt1・δ=.1価格を代用しない。

**保存価格の役割**：[opt2 JSON][OPT2]は該当q1直接較正を含むが、有限8 trajectoryのmean/SEであり、全fresh path/各stratumの価格表・rigorous信頼上界ではない。
この平均を全方式の1shot価格に固定すると、条件付き層取得やpathと価格の対応を失う。旧SE/5% proxy gateを厳密confidenceへ昇格しない。

**将来の取得案**：同一trajectory/wrapper/policy/compiler fingerprintの既存価格cacheがあれば使い、未知pathは許可された範囲内だけfull build/compileで取得する。
selected repeated builder→Hadamard wrapper→`transpile_and_measure_cost` の既存処理を用いる。`_compile_paired_requests` 全体はfull/selected両方を計算するので、不要な比較callを自動追加しない小接続を検討する。
この新build/compileは今回未許可・0件。native synthesisは初期scopeから除外したまま。
将来のcall範囲は、承認された各axis・方式・stageの**全実行trialに必要なselected full価格**まで。重複pathのcompileはcacheで省けても、量子shot費を省略しない。

各trialの実compiled countを合計した実現費用と、期待単独運用費用を区別する。
期待費用にconfidenceを付けるなら、独立の価格標本またはstage標本における正当な有界価格上界・δ_price配分が必要。
必要な価格range/検証誤差が未取得なら期待費用優位は未確定として止め、UNKNOWNを0にしない。
calibrationからランダムにNを決める設計で無条件に `E[N]E[C]` としない。段階・層別条件付きの費用を会計する。

`TrialCost/account_costs` を方式ごとに呼び、train+cal+prodを全て課金する。B0の不要trainは0。
元6指標のうちadapterにないRZ/CX depthやsizeは既存full cost recordに残す。T/1Qは取得できたものだけ、未取得はNone/UNKNOWN。
wrapper/readout込み価格へH/Sdgを重複加算しない。状態準備のみ除外。
棄却生成、CDF/moment、fit/Fraction計算、記録、cache/setup、measurement simulatorの古典実費とpeak memoryも別座標で記録する。
古典setupを偽の量子trialへ載せない。深さのshot合計を並列wall-timeと呼ばない。
比較データを共用する場合は実験全体の取得費と各方式の単独運用費を分ける。

## 10. 再利用・実行前接続

固定A1 moduleと人工testを変更せず継承する。law/control/OLS/凍結/range/data分離/raw/strata/費用の各接口を再実装しない。
既存RTE、partial/repeated、selected basis plan、Hadamard builder、compiler、V2の決定論decoderを再利用対象とする。
科学実行前に必要な小接続は、実CDF/完全catalogの取得、path/bit provenance、独立stream、条件付き取得・有限配分、confidence/N、selected full価格である。
これらは未実装・未実行であり、新汎用pipeline・grant管理機構を作る必要はない。
SOURCEを変更する必要が生じたら、元SOURCEを保存したまま独立scopeで必要差分だけ説明し、科学的意味の変化を隠さない。

## 11. 実行前・実行中のSTOP

**科学取得前**：モデル限定scope/state利用法が未採択、threshold/finite target差が未処理、actual CDF/support/νが未対応、centering/系統biasの正残余が不明、価格policy不一致、比較/α/finite scopeが未固定なら開始しない。
実state payloadがなく再構成が必要な場合、許可のない対角化・H/state生成へ入らない。

**実行中**：SOURCE/input/wrapperの不一致、NaN/範囲外bit/weight、trial混入、data再利用、正層の欠落、range guard違反、N/attempt/価格call scope超過、データ破損、実際の資源不足・他者への支障で自分の処理を安全停止する。
旧結果・失敗pilot・途中費用を保存し、feature/seed/精度/δ/r/K変更、別round、無断retryで救済しない。
一時的な負荷や無関係worktree更新だけを理由に正常処理を再起動しない。共有環境・他ユーザーprocess/files/venv/cache/settingsを変更しない。
CPU/RAM/workerの恣意的な低い上限は課さず、実際の空き資源と非干渉に応じて自分の処理を効率化する。旧固定契約の資源条件は変更しない。

## 12. 次に必要なユーザー判断と明示許可

1. 保存DF model・opt2 threshold0 lawを主対象とするm0/X/Y診断を採るか。元物理Hへの未閉鎖・WP04との差を条件として認め、どのbias接続を要求するか。
2. 上記2label/3control、B0/B1-O/B1-OL/A1の4方式、精度/α配分、train/cal規則、8192 production cap、range/attempt guardを採るか。
3. 状態payloadを回収できない場合の限定state再構成・科学CDF/moment/bias取得・fresh Hadamard simulation・train/cal/prod・variance/十分N算出・費用評価をどこまで許可するか。
4. 同じselected full価格のcache利用/不足build・compileを許可するか。期待費用の価格confidenceを取得するか、まず実現campaign費用だけに限定するか。T/nativeは別の許可scopeとする。

不足が解消しても、今回の文書作成・公開指示だけでは上記科学callを開始しない。
新しい明示的な実行許可が必要で、承認原文保存・certificate・新grant発行を要求しない。
**今回のmandatory STOPは3文書の公開・本文照合まで。その後、科学測定・新実装・別Trackへ自動進行しない。**

[CARRIER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/4591b8d75a2a6a55c74a51225f0c61648023c8ee
[OPT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[CATALOG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json
[CHOICE]: https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx
[PCG]: https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_pcg64.pyx
[DOUBLE]: https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd
[EB]: https://arxiv.org/pdf/0907.3740v1
