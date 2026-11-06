# Track C：並列pilotの実行条件・科学call範囲

2026-11-06 JST / SERVER 2。基準commit `7db4792d4cd0ed9a7abc255565f207dd31a437bd`。
**採択用の最終実行案。今回は静的確認と本追補の作成まで。新規科学計算・実装・合成・回路build/compile・test起動はしていない。mandatory STOP。**

既存[共通契約案](SHARED_PILOT_CONTRACT_DRAFT.md)、[C-N](C_N_PILOT_PLAN.md)、[C-S](C_S_PILOT_PLAN.md)、[C-P](C_P_PILOT_PLAN.md)、[REPORT](REPORT.md)と[H4照合結果](H4_INPUT_READINESS.md)を引き継ぐ。旧5計画・科学SOURCE・結果・grant・STOPは変更しない。以下の追加数値・限定目的は採用を提案する条件であり、既にユーザーが採択した、又は実行を許可したとは記録しない。

## 1. 共通条件と到達可能な主張

| 項目 | 実行案 |
|---|---|
| 物理入力 | Linear H4、隣接1.00 Å、STO-3G、charge0/multiplicity1、8 spin orbitals、4電子number-sector dimension70。legacy DF12、生成順prefix LD3。Full LD12は強対照 |
| state/reference | 元Hの厳密groundを供給する仮定、状態準備費除外。数値代表は基準commitの `artifacts/track_c_h4_input_check/2026-11-06/run02/ground_reference.npz` のstate/basis/energy。保存されたlegacy座標で使用し、再対角化・H4入力照合の再実行・新SCF/DF生成はしない |
| DF回路入力 | 既存共通案の `4591b8d…` snapshot、H hash `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`。新規DF fitやrank探索なし。H4照合の丸めスケールの差を数学的な0と改称しない |
| energy / 成功 | 今回の表記どおり `epsilon_E=1.5936e-4 Ha`、`alpha_total=.05` を提案値とする。旧 `.00015936001019904` と混在させない。各armが同一条件を満たす必要がある |
| RPE | `beta_RPE=.4 rad`、`delta0=.02`、m=0..17、q=2^m、t=q delta0。PF microstep h、反復数 t/h、RTE short-step rは区別する |
| primaryの証拠水準 | 固定H4・指定coherent mean・fresh IID有界shotモデルに対する**条件付き数値pilot**。経験的PF誤差や倍精度の参照を厳密包含保証に昇格しない。厳密な全実装保証を初期pilotの必須成果にはしない |
| 費用境界 | ancilla準備、物理processor、ordinary controlled evolution、basis/Givens、scalar相対位相、合成、X/Y readout・測定を含むfull interrogation。除外は元H groundの準備のみ |

H4入力照合PASSは再利用する。新PF/有限meanの構築に必要なDF primitive準備・basis変換は将来の科学callに含むが、元H4の同一性確認を再試行するcallではない。

## 2. 18round・全誤差予算・alias

[既存V2結果][v2]のnominal幅は `beta_RPE/(q_max delta0)=.000152587890625 Ha`。今回の `1.5936e-4 Ha` より小さいので、**この整合性のためにround追加やdelta0変更は不要**。ただしこの式にはPF等を含む全位相誤差を入れる。0.4 radを統計だけに使ってから合成誤差を外付けしない。

各roundで同じ元H referenceへ、次を別々に評価する。

`b_H/state + b_PF + b_original_RTE + b_residual_RTE + b_synthesis + b_numerical + b_stat <= .4`。

旧係数thresholdによる誤差、有限bit/係数誤差、参照stateの有限精度も該当欄へ含める。旧 `coefficient_atol=1e-12` を再利用する場合も、その非ゼロ作用素誤差を落とさない。C-Sのcoarse化は同位置で補完するのでPF分割を変えない。C-Pのprocessorは理想共役によるsignal変化とnative誤差を分ける。相互作用する誤差をnorm差引きで「寄与率」にしない。

実行案の配分は、各stage間の位相上界（又は条件付き数値診断とその誤差幅）を先に求め、残りを `b_stat=.4-sum(b_systematic)` とする共通規則。残りが正でないarmは不適格。円周残差だけでなく各中間signalの正半径を確認し、complex diskの半径が中心の絶対値未満の場合に限り `arcsin(error/radius)` で位相幅へ換算する。単位半径を代入しない。最終mean対元Hの全位相差も別途照合し、誤差の二重計上や欠落を検査する。

新たに採用を提案するC-Sの有限policyは、追加residual位相上限 `.01 rad/round`、合成位相上限の2候補 `{.01,.02} rad/round`。S0/S1/S2の全armに同じ2候補を与える。上限内の実際の誤差を会計し、不使用のresidual予算をS0から差し引かない。Q profileは理想連続角native gateを価格対象とするのでClifford+T合成誤差は対象外であり、浮動小数点lowering等の誤差は別途残す。

符号は `U(t)=exp(-itH)`、base phase `phi=-E delta0`、`E_hat=-phi_hat/delta0`。decoderは既存nearest-branchを再利用し、E0を引数やbranch offsetの補正へ渡さない。`.4 < pi/3` のstrict条件と全roundの十分事象を確認する。

外部alias条件はbase phase window `[-pi,pi)`、energy window `(-pi/delta0,pi/delta0]`。energyへprincipal branchを戻す際の端点飛越えを避けるため、初期案では対象phaseがwindow端点から `.4 rad` より内側にあるという保守的な外部条件も置く。既存V2の保存referenceがwindow内部である事実は診断であり、E0を用いた実行可能なalias排除証明ではない。外部window仮定を認めない場合は、その根拠の取得範囲を別途決める必要がある。

## 3. 独立した有限比較集合

| pilot | 固定arm・policy | 主対照と変更しないもの |
|---|---|---|
| C-N | S2 / Yoshida4 / 公開Morales Y8m10b × h=.02/.01 × LD3/12、最大12構成。R occurrence `s_j=b_j h` ごと `r_mj=max(1,ceil(abs(s_j) r_m/delta0))`、K2。Fullはtail-free | processorなし、Q profile固定。3formula・2hのFullを含む。粗いstage proxy、**occurrence内部を最適化済みの強加法価格**、同trajectoryのfull-wrapper価格を比較 |
| C-S | canonical S2、LD3、h=.02、processorなし。S0通常合成 / S1 eligible class一様residual / S2 D側residual。上記2合成予算で最大6構成 | PF/元tail/回路位置を固定。S0にも同じprecision・alpha配分・exact fast path・fusion機会を与える。S1は配置選択の対照。T profile内で比較 |
| C-P | h=.02、D-context raw YP8m8 kernel、n_D=1、各D half-runのtau=h/(2 n_D)。P0なし / P1全kernel共役。P=P_pub† と固定 | 同じkernel・元tail・Q profile。processorの逆方向やn_D gridを事後に追加しない。C-Nに登録したunprocessed強対照も共通結果から利用 |

元tailの固定r列は `[1×8,2×2,4×1,8×2,16×2,32×3]`、K2。C-Nのsigned時間へは上記recipeで移す。基準recipeで不適格なarmを救うための無制限r/K・係数・step探索は含めない。負結果はこの有限recipeに限定する。

Y8m10bは[固定公開SOURCE][ypf]の `morales_2025_y8m10b_list`（取得blob `70d68691a1dcdedea5447aac4b8bbfa51141d4cb`）を使用する。旧 `morales_8th_list` やYP8m8へ置換しない。今回は関数本文を静的に確認しただけで、係数評価・formula実行はしていない。

### C-S：eligible class・coarse・残差の確定案

eligibleは既存DF/RTE primitiveの**非identity Pauli involutionをgeneratorとするdiagonal phase rotation**。Dと元RTEの両roleをS1、D roleだけをS2とする。basis/Givens、scalar/identity相対位相は全armで通常合成し、同時にrandomizeしない。native低水準RZを無差別に選ぶmask探索はしない。

target `R_A(theta)=exp(-i theta A/2)` を、**同じsystem operator位置、control前の意味**で `R_A(theta_tilde) R_A(theta-theta_tilde)` に分ける。coarseは最寄りの `k*pi/4`（同距離は偶数k）、周期短縮は4piとし、2piの符号を捨てない。既存SP catalogueと同じ刻みを使うが、system-channel PAI式は流用しない。coarse角がClifford/T表現を持つことから、制御化後も無料又は厳密Tだけで実現できるとは仮定しない。相対位相・scalar補正を保持してlowerし、必要な合成も課金する。

残差はsigned `theta_res/2` を時間係数とするsingle-involution canonical operator-RTE、K_res=2。r_res候補 `{1,2,4,8,16,32}` から、当該round全位置の合成truncation boundがresidual予算に適合する最小整数を選ぶ共通規則とする。r=32でも不適合なら当該構成を不適格とし、粗角刻み・K・sampling lawを変更して救済しない。scalar bound評価も将来の科学callに含む。中央Rへ残差を移す別split、PAI/TETRIS channel実装、旧SP mask sweepは含めない。

### C-P：whole-kernel processingと実費

各D run内でprocessingする旧DRun列をprimaryへ連結しない。raw kernel Qに対する全体共役 `P† Q^q P` を比較する。pure conjugationは固有値を改善しない。評価対象は元H groundから見た有限時間signal、半径・混入・shotと費用。

有限meanのQは元tailのr_m等でroundごとに変わり得る。同一round内の固定recipeを反復し、異なるroundを同一Qのpowerと取り違えない。Pは指定D-context/tauの同じ実現を使う。

primary価格は**ancilla X/Y測定等価lowering**とする。最初の物理system Pを実行・課金し、controlled Q^qを実施する。最後の共通system P†はancilla readoutに不要なので省く。これはP state oracleの無料供給ではない。ordinary controlled `diag(I,P†Q^qP)` をoperator参照とし、短qのfull joint-spaceで両X/Y期待値を照合する。二つのjoint operatorが同一だとは主張しない。Qiskitが残す全P費用・basis往復を数え、native Pの誤差は端点に応じて伝播する。RTE障壁を越えたper-D-run相殺を仮定しない。

## 4. compiler・合成精度・最小実装差分

| profile | 条件・資源 |
|---|---|
| Q：C-N/C-P | Qiskit1.3.0、`rz,sx,x,cx`、opt2、seed17、backend/coupling/layout/routing指定なし。主指標RZ、CX/depthを併記。T数への換算なし |
| T：C-S | 同じphase保持control lowering・共通の最適化機会の後、strict Clifford+T/CXへlower。pygridsynth2.0.0、`up_to_phase=False`、seed17を提案。負角は実取得sequenceのactual adjoint。TとT†を分記し合計Tを主指標、CX/Clifford/depthを併記。残った連続RZをT取得済みとして数えない |

T合成は旧1e-6/1e-7を固定移送しない。finite meanの正半径下界をrho、合成位相capをb、full interrogation内の各実現のgate作用素誤差をeps_gとすると、corrected信号の誤差について `2 sqrt(2) sum_path p_path abs(W_path) sum_g eps_g <= rho sin(b)` を満たすよう取得する。各X/Y期待値の差をjoint-unitary誤差の2倍で抑え、二座標をcomplex diskへ合わせる保守的な規則である。controlled-system operatorの誤差へ直接換算できる場合のより鋭い係数は補助診断に留め、primaryでは同じ保守係数を全armに用いる。raw側を使う場合は同じ式をnormalizerで割って表す。全18round・使用回数・元tail/residualのweight・control lowering・basis/端点を含め、同じangle keyのprecisionは使用箇所が要求する最も厳しい値へ合わせる。

precision取得policyは、この不等式を満たす最大の10進power epsilonを各keyに割り当てる等分gate-error方式。通常合成baselineにも同じpolicyを与える。有限supportの最大gate数からの保守的boundを使えるので全trajectory列挙を必須にしない。取得後はstrict guardの実際の誤差幅で再会計し、coarse・通常・残差用rotationの全誤差を含める。projective phase fitをcontrolled system gateの合格判定にしない。exact fast pathは相対位相を含む対応が成立する場合だけ全arm共通で用いる。

再利用と新規差分は次の4接続に限定する。旧toy runner・固定inventory・grantをH4用に改変又は起動しない。

| 固定SOURCE | 再利用 | 必要最小限の差分 |
|---|---|---|
| `f29eafbf4412193e8e8660d9f051889c2cba684b`：PF decomposition、DF control/repeated、rte、finite signal、RPE provider/decoder | S2/公開係数のliteral列、signed heterogeneous RTE、DF primitive/control、full wrapper、round集計 | 既知3formulaを同じDF elementary blockへ接続するmacro adapter。S2専用repeated runnerへ高次次数を偽装しない |
| `4ccc1832f63cd323b23fc05623f7e0b58cf34f37`：`src/trottertracks/algorithm_codesign/parent_allocation.py` | event statistics、誤差/weight、actual native/adjoint順序 | H4 same-position residualと元tailのordered composition |
| `546bca7ed2cece59b714b986a264d7b0b0871a01`：同algorithm_codesign配下の `parent_allocation_native.py`、`rte_reallocation/native.py,numeric.py`、`synthesis_placement/economics.py` | strict key/guard、phase保持lowering原理、取得済みkey照合 | 8 system qubit用DF providerと実H4角度への小さい接続。旧2-system-qubit providerやatan/pi限定角度APIへ実数H4角度を黙って通さない。SP projective guard/PAIはcoherent targetに不適合 |
| `b190fe28b98d2f94e1509e19cd3d36a63b2bb99e`：b2d_adapter / native-v5 | DContext、公開YP8m8、実append順、actual inverse、basis/control | raw kernelとwhole-interrogation Pの接続、測定等価wrapper。共通会計は一般mean/正半径/誤差を入力できる最小拡張 |

科学SOURCEは将来の実装を新しい版として保存し、使用した固定由来を添える。環境準備・必要な人工control検査・通常debugも別途許可後の同じ範囲に含む。CPU/RAM上限や承認証明の新機構を作らない。今回、module import・環境変更・実装・test起動はしていない。

## 5. signal → shot → full価格 → 全RPE

C-N/C-Pのprimaryはcanonical finite lawのraw ±1 Hadamard測定とする。実現に共通な正のBについて `Zraw=Zcorr/B` を使い、Bは観測半径に一度だけ反映する。

C-Sの全armは**corrected weighted推定**に揃える。元tail実現xを固定proposalで引き、各同位置residualの既存canonical lawを条件付きで引く。追加residualのnormalizerはx中の角度・位置に依存し得るため、一般には `W(x,y)=B_original product_j B_residual,j(x)` を実現ごとに保持し、`Zcorr=E_p[W V]` を評価する（coefficient phaseはcontrol側で保持）。全supportから解析的に有限な `L >= sup abs(W)` を得て、各軸の測定結果にweightを掛けた有界変数を使う。S0では追加residualがなくW=B_originalなのでraw会計と等価になる。全体で一定のBを証明していないC-Sに `Zraw=Zcorr/B` を仮定せず、unweighted測定平均とcorrected平均を別欄にする。

normalization負担をraw半径とweighted rangeへ重ねない。C-Sで有限Lが閉じなければ必要N/Gは未資格とする。momentだけを単位range式へ代入しない。この方針は元tailのproposalをnormalizerでtiltする新lawの導入を避け、既存のcoefficient/proposal/weight会計を使う。

特にT合成でcontrol0=Iや粒子数保存が厳密には残らない場合、native joint-channelの反復を70次元system平均operatorのpowerで代用しない。primaryは**理想finite meanを中心とするfull joint-spaceの合成誤差disk**からnativeの位相幅・半径下界を得る。native平均を直接計算できる構造が確認できた場合だけ点推定も併記する。full joint-spaceの短q検査と全interrogation gate誤差の伝播を使い、巨大なnative superoperatorの構築を初期必須にしない。nativeの点値未取得はdiskと区別し、0誤差で補わない。

正のnative推定平均の半径下界rhoと残る統計位相幅b_statから `epsilon_coord=rho sin(b_stat)/sqrt(2)`、
`N_m,b=ceil(2 L^2 epsilon_coord^(-2) log(2/alpha_m,b))` を使う。rawは観測半径とL=1、C-Sはcorrected半径と上記Lを組にする。36軸のalpha和を.05以下にし、全roundの系統＋統計位相条件から既存strict branch十分条件とenergy幅へ接続する。full mean/誤差幅の根拠が条件付きならN/成功/Gも同じ条件付き表示にする。

全armに **一様alpha=.05/36** と **同じcost-sensitive alpha規則** の両機会を与える。後者は各軸のHoeffding係数 `c_i=2 L_i^2/epsilon_i^2` とcalibration価格C_iから `alpha_i proportional to C_i c_i` とし、C_i=0の軸には一様配分と同額を先に確保して残額を正重みへ配る。整数Nを再評価し、二policyのうち安価なものを使う。旧Partial/Full別の保存alpha/Nを新armへコピーしない。

`G_resource=sum_m,b N_m,b E[C_full,no-prep,m,b,resource]` を全18roundで計算する。RZ、CX、T、depthは別の合計。sum N E[depth]はdepth費用指標であり、実wall timeや実並列実行時間へ換算しない。Q/T profile間の絶対値で3候補の総合順位を作らず、各pilot内の強baseline比を示す。

C-Nは同じarm・同じNを価格model間で使う比較を先に行い、価格modelだけの選択差を切り分ける。その後、全armへ同じ配分機会を与えたGを比較する。exact E0・gap・exact signalによるarm選択は**oracle diagnostic**と表示し、実行可能selectorの利得に数えない。実行可能選択には入力から利用できるboundとcalibration価格だけを使い、それが不足する場合は選択法の資格を保留する。

## 6. 科学call範囲と価格精度の有限案

| call群（全て将来の明示許可後） | 含む範囲 | 含めないもの |
|---|---|---|
| 共通準備 | 保存H4 state/DFをload、DF primitive/native loweringの準備を共有cache。一度の共通取得を3pilotで再利用 | H4入力照合再実行、SCF、新DF fit、新ground solve、別geometry/rank |
| operator/mean | C-N最大12、C-S最大6、C-P2構成のideal/finite signalとnative点値又は誤差diskを各18round。C-PのP/kernel診断、C-Sの有限r bound選択を含む | 旧toy全runner、係数/P solver、新sampling law、全実現列挙、仮想量子shot大量sampling、巨大native superoperator |
| correctness | 必要な正/負時間・scalar/control・実adjoint、C-S同位置mean、C-P両X/Yの短q full joint-space参照 | sectorを保存しないnative中間gateを毎段70次元へ射影してleakageを消す処理 |
| 合成 | C-Sのunique angle/precision/tool key。2合成cap、通常/coarse/residual/basis/相対phaseを含む。適合する旧keyだけ共有 | 旧23sequence等の無差別再取得、projective guardによるstrict結果の代用 |
| 直接価格 | 適格構成・各distinct round recipe keyについてq=1/2をcalibration、q=8を未使用holdout、両軸。元tail/residualでlawやprecisionが変われば別key | 全roundのr/precision変更を無視した1本のq1/2較正の流用、終端巨大回路の自動compile |
| 会計 | 全18roundの位相/半径/B/range/N、資源別G、未知費用・差の不確かさ、古典取得/合成/生成費 | q8だけで終端まで価格保証、未取得strata/長q価格の0補完、gateへの根拠のない古典費換算 |

価格標本数の提案は、q1/2で**元tailの各非空stratum32実現**、q8で**各非空stratum128実現**。科学的な有限標本設計であり、CPU/worker制限ではない。元tailのstratumは「order-2が一度以上あるか」の2群で、複数order-2も含める。C-Sは各元実現xに対して「residualにorder-2が一度以上あるか」の非空2条件をそれぞれ1本条件付き生成し、xごとの既知確率で平均してから元tailの平均へ戻す。したがって最大4経路群に相当する。元tail群とresidual群の**周辺確率の積を使わない**。角度・位置がxに依存しても元tailのproposalを保つ階層的な条件付き層別である。

確率0の群のみ省略する。群内の全label/位置分布を保ち、単一order-2だけで群全体を代用しない。group確率の有限bit誤差・underflowで未観測rare群を0にしない。価格は実proposalの `E_p[C]` であり、estimatorのabs(W)を価格へもう一度掛けない。C-Sの階層層別の分散は元実現単位で扱う。

比較可能なarm間は同じ元tail実現をpairedにする。追加residualは明記した独立drawとし、異なるlawの乱数を同一と偽装しない。C-Nのfull/強加法は同一trajectory、C-PはP以外同一kernelを使う。calibrationだけで比較対を選び、holdoutで選び直さない。

既存の価格診断基準である予測相対半幅2%、q8の点誤差5%を初期目安として報告するが、rigorous保証にはしない。paired差の区間・共分散と未知tail/model幅を併記し、多候補の点wise区間を同時保証にしない。固定標本で差を識別できなければUNDETERMINED。自動で標本を増やす追加batchは本scopeに含めない。

q8が通ってもq131072の価格は検証済みにならない。全round Gは直接取得域・構造に基づく移送域・仮定だけの域を分記する。長qのモデル仮定に依存する値は条件付きG、未知幅が閉じなければ総費用優越は保留する。大差のない結果を救うための長q追加compileは別scopeとする。初期pilotは「全費用の保証済み勝者」を必ず出す契約ではない。

継続判断は各pilotに限定する。C-Nは強加法対照からの選択差と全G差が残るか、C-Sはweight/rangeによるshot増加を払ってもS0/S1に対する全G差が残るか、C-Pは固定Qのsignal/shotの改善が物理P費用を上回るかを判別する。差が消えた場合は登録範囲の負結果、未知幅に依存する場合はUNDETERMINEDとする。correctnessや実装の不一致は科学的負結果と分けて記録し、影響するarmの比較を止める。他の独立pilotは共通入力・接口に影響しない限り継続できる。

## 7. 未決事項・次の担当・STOP

定義と有限call範囲は上で具体化した。数値結果・適格arm・unique synthesis key数は、実行後に得る出力であり、今回は計算していない。次の3点は、採択と別途実行許可の対象として明示する。

| 判断 | 推奨する採択案 | 別の判断をする場合の影響 |
|---|---|---|
| 証拠水準・alias | 元H ground仮定、外部alias window/端点margin付きの条件付き数値pilot | 無条件の厳密成功・厳密長q価格を要求する場合、包含/alias根拠/価格call範囲を追加で定める必要がある |
| 新たな有限policy | C-S residual cap .01、合成cap {.01,.02}、最寄りpi/4、K_res2/r_res有限集合、C-P n_D1/P方向/測定等価loweringを上記どおり採用 | 主仮説・比較条件を変える別設定を、実行結果後に混ぜない |
| 価格と探索範囲 | 32/128本stratum設計、q1/2/8、uniform/cost-sensitive alpha、追加batchなし、未知は保留 | 標本追加や長q直接情報を最初から要求する場合は対象と判別したい差を指定する |

**ユーザー**：この案の採否・変更点を示し、実装・人工検査・科学計算・合成・compileの実行を別途明示許可する。採択と実行許可を同じメッセージで与えてよく、承認証明やgrant発行は不要。

**Codex**：その一つの許可範囲内で関連実装・debug・検証をまとめて実施し、共通入力/preparationを共有後、C-N/C-S/C-Pの独立計算process・合成key・価格標本を十分に並列化する。CPU/RAM/workerへ人工上限を置かず、実際の空き資源・OS制御・他ユーザーへの非干渉を守る。共有設定・他ユーザーjob・旧封印SOURCE/証拠は不変更。AI sub-agentは使わない。

誤差不適合・負結果・技術失敗・未確定事項を区別し、各pilotの継続/STOP材料を保存する。承認範囲完了後、専用branchへ結果を公開してremote SHA/成果物を照合しmandatory STOPする。今回は本追補をローカルに作成してSTOPし、実装・科学call・compile・公開commitへ移行していない。

[v2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7db4792d4cd0ed9a7abc255565f207dd31a437bd/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[ypf]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
