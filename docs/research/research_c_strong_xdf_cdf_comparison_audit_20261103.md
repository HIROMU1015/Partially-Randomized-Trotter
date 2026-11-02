# 研究C：保存XDF/CDFと最新PRの公平比較条件・不足証拠の技術監査

2026-11-03 JST。Codex-AのSOURCE・保存JSON/CSV・実装経路の静的監査。新科学計算、保存配列の数値再解析、回路生成/compile、新しい統計量・CI・検定、重要科学レビューは実施していない。

## 1. Executive Summary

**update064/r32が同じJ/Uの強決定論S2より条件付き総RZで約4.48%低いというnominalな結果は維持する。一方、別DF表現を許す保存XDF8より低費用という結論は、現在の証拠からは得られない。** 最新PRの保存95%近似総RZ区間は保存XDF8の点費用を含む。同等性、非劣性、XDF優位も認定しない。

元の物理問題、CA/DF予算、13round、RPE政策、P0、状態準備除外とserial会計は対応している。したがって別DF表現の**限定的なtask-level保存値比較**には意味がある。ただし、各DF Hamiltonian/ground/PF誤差は異なり、決定論の数値合成誤差を含む `NUMERICAL_PILOT_CA_PASS` とPRのideal/assumed laneは同じ証拠資格ではない。保証付きの実装同CA比較は成立していない。[保存強baseline][OLD]・[最新下流][NEW]・[独立価格][PRICE]

また、共通precision/P0と相対Gaussian実装を再利用する一方、旧CDFのκは8モード直接対角化、最新PR/同J/U Dのκはspin対応対角化を使う。保存XDF B3はκと各Gにspin対応基底＋相対融合を適用済み。これは明示すべき合法な実装差であり、未観測の削減を補正したり、保存XDFを除外したりする理由にはしない。

今回の成果物は本書一件。共通起点 `ba0f280c8be9e7038c5fa6b5fd390a235b113833` から独立branchで作業し、共有索引・Evidence Map・研究概要・manifestは変更しない。数学的な新規性・近接正則化の評価はCodex-Bの担当であり、本書では再評価・採択しない。

## 2. 固定commit・SOURCE・candidate identity

| 記号 | 保存証拠 | 科学SOURCE | 結果収録commit |
|---|---|---|---|
| OLD | B2 CDF/B3 XDFの共通CA・総資源、[JSON][OLDJ]・[契約v2][OLDC]・[SOURCE][OLDS]・[STOP][OLDSTOP] | `1aeedd96178a3f50a15bc2701c3c332f4f391b64` | `54ceff9981d9bc53210db97c743a7d1cf694f362` |
| B2 | CDF relative native、[報告][B2]・[結果][B2J] | `880006396326a4d46f4a4030564279e92fadb9cf` | `e0c89f611eac92c939edde67890f5cc4391d098e` |
| B3 | XDF spin＋relative native、[報告][B3]・[結果][B3J] | `b40fce5c62c9fff97f68073cbc02182e4bbf2b9b` | `866d3ba327abd5ca38af5d7a9347f6c91294958f` |
| NEW | update064 PR/r32と同J/U強D、[JSON][NEWJ]・[契約][NEWC]・[SOURCE][NEWS]・[STOP][NEWSTOP] | `928fda0a6df260b66eba1bbea63b51f2734e7b54` | `cca6b487c8e1ed3362f8825e95c937ebbcc87755` |
| PRICE | update064/r32の独立256価格、[JSON][PRICEJ]・[契約][PRICEC]・[SOURCE][PRICES]・[STOP][PRICESTOP] | `9837eac59b94113bf057fcc311a27b35e60343f5` | `3718d369b3db515b2dd13b2cdbb6c49d57880121` |
| U64 | PR候補の追加64 U更新・[終端物理判定][PHYSICAL] | `025d495166311a58ec9c24c43f07ed6e40f65f42` | `ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6` |

科学SOURCEを閲覧用の共通起点や結果commitへ付け替えない。OLDは保存B2/B3のnative証拠を再利用した86-call評価で、新しい回路compileをしていない。NEWは617-call、PRICEは2374-callの別scopeである。今回これらを再起動していない。

| 候補 | 保存表現・identity | 参照 |
|---|---|---|
| B2_CDF_relative | 通常number CDF8。旧fit export `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`、元κ/J/U/constant。元fit科学SOURCE `da2edb465d5ae6432439fbd1df8e57512ee59c11` | [native契約のexport binding][NATIVEC]・[Stage1来歴/評価][CDF1] |
| B3_XDF_spin_relative | 元P-H4のOpenFermion1.6.1 eigh由来rank10 λ/Gの**元順prefix8**。identityは `snapshot SHA256 + :prefix8` とSPIN recipe。CDFのJ/U hashを流用しない | [native契約][NATIVEC]・[B3 prepare][B3CODE]・[XDF8 PF/DF結果][XDFPFJ] |
| PR064、同064 D | 通常number PR-aware CDF8、terminal export `53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3`。NEW/PRICEは同じU64終端をpeer commit＋hashで参照 | [NEW契約][NEWC]・[PRICE契約][PRICEC]・[終端物理結果][PHYSICAL] |

B3の歴史的U配列のbitwise一致は保存されていないため主張しない。元G/符号付きλ/recipeの保持と独立full512作用検証がその資格である。PR064の同一J/U比較には保存J/Uと独立CAR factorのidentity検査がある。

## 3. 物理Hamiltonian・軌道・spin・入力来歴

全候補は直鎖H4、隣接1.00 Å、STO-3G、charge0/singlet、Nα=Nβ=2、sector36、system8 mode/full256、ancilla込みfull512を対象とする。元SCF/MO入力は、生成SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`、公開 `86b8b002704aced8ca239414080f53793de81658` の[receipt][INPUTMETA]とsnapshotを参照する。

| 入力・規約 | 照合内容 | 範囲・限界 |
|---|---|---|
| 元P-H4 | `artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` | OLD/native契約とPR初期契約が同じ元inputを指定 |
| PRのprepare/input.npz | SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。`h=one_mo, g=eri_chemist, constant=nuclear_repulsion` の保存コピー | [PR初期契約][PRINIT]・[prepare_worker][PRPREP]からNEW/PRICEの固定inputへ接続。NPZ容器hashの違いを物理入力の違いとは扱わない |
| 空間軌道 | 保存MO integralを使用、新SCF/MO再生成なし。PR prepareは積分の並べ替え・別軌道変換をせず上記memberを保存 | 今回NPZ memberをdecodeして配列値を再比較していない。固定来歴・SOURCE・保存gateを照合した範囲 |
| κ | 共通 `corrected_kappa(h,g)=h−.5 einsum("prrq->pq",g)`、**元gから補正**。DF近似gから再補正しない | CDF保存κとXDF df_one_body、最新PRの再構成κにはそれぞれ保存同値性gateあり。[共通コード][KAPPA]・[旧identity][NATIVECODE] |
| 核scalar/phase | Nuclear scalar一回、controlledの相対位相として保持。各対角I項も保持し、PR tail identityは確定位相へ分離 | global phase fitなし。旧/新SOURCEは同じscalar所有規約 |
| Spin/bit | mode2p=α、2p+1=β、空間→spin lift。CARのindex規約からQiskit little-endianへのbit-reversal/sector埋込みを明示 | Gaussian単体のsector射影で代用せず、旧/最新とも対応するfull controlled作用を保存検証 |
| DF/frame | CDFの一般対称JとXDFのsigned λ/G squareは別表現。各自の元frame0…7順を保持 | 同じラベルF0でも異なるDF間で同じfragmentを意味しない |

**同じ元物理Hamiltonianを近似することは来歴として対応するが、同じ近似Hamiltonianではない。** 近似固有状態の用意を無償oracleとするtaskで、候補ごとのenergy biasをDF誤差として接続する。共通の物理状態を全候補へ入力した実験、状態準備込み比較とは異なる。

## 4. DF誤差・予算・groundの対応

単位はHa。以下は保存値の転記で、新しいnormや和を計算していない。全候補の予算はCA/8=.00019875 Ha。

| 候補 | 保存sector operator norm | 保存guard/allowance | 保存norm＋guard | 予算・資格 |
|---|---:|---:|---:|---|
| 旧CDF8 | 7.666027761392501e-5 | 1.881549094133745e-9 | 7.666215916301914e-5 | PASS、Stage1数値norm＋engineering allowance |
| XDF8 | 9.533753411107529e-5 | 1.480439843662034e-9 | 9.533901455091894e-5 | PASS、保存prefix8 norm＋allowance |
| PR064/同064 D | 5.673413458637335e-9 | 3.2244465214668673e-9 | 約8.89785998010e-9 | PASS、SectorCAR数値norm＋engineering guard |

出典：[CDF1 records/sector_DF_CDF][CDF1J]、[XDF8 saved_DF][XDFPFJ]、[OLD候補CSV][CANDCSV]、[064物理JSON][PHYSICAL]。旧allowance `binary64_operation_scale_guard8_v1` と新guard `sector_operation_scale_guard8_proposal_v1` は同じ厳密enclosureではない。いずれも数値pilotで、厳密物理DF上界を認定しない。

共通CA予算を満たす異なるDFを比較する**比較II**は、この達成誤差差を保ったまま設定できる。達成DF誤差を一致させる**比較III**は別の問いである。PRの誤差が小さいことを根拠に、後から比較IIへ達成誤差一致を要求してXDF8を排除しない。

Own-DF groundも別である。保存energyは旧CDF −2.1663690490737824、XDF8 −2.166402428462818、064 −2.1663874493318342 Ha。XDF8/064のalias marginは1.6179938779914944 rad、旧CDFも保存H4 branch/ground gateを使用している。[CDF prepared][CDFPREP]・[XDF8 ground][XDFPFJ]・[064 ground][GROUND64]

CDF prepared内の別 `XDF` fieldは当時のrank10対照であり、**XDF8のground/DF値に使わない**。旧強baseline loaderは専用XDF8結果・groundを参照する。最新PR/同064 Dは同じ064 own-groundを共用するが、旧CDF/XDF groundを流用しない。

## 5. PF・RPE・誤差政策

| 項目 | B2 CDF8 / B3 XDF8 | PR064/r32 | 同064強D |
|---|---|---|---|
| 時間と順序 | 全9因子ordered symmetric S2 | 元順LD2のpartial ordered S2、中央tailは有限RTE | 全9因子ordered symmetric S2 |
| s、δ=h | s1、.1 | s1、.1 | s1、.1 |
| Round/時間 | m0…12、n=2^m、T_m=.1×2^m | 同左 | 同左 |
| LD/r/K | PR split/r/Kは非適用。旧JSONのr=0/K=0は決定論ラベル | LD2/r32/K0 | PR split/r/K非適用 |
| PF誤差 | 候補固有のsector36 `norm₂(PF(T_m)−exp(−iH_DF T_m))` | exact-tail partial PFと同じH_DFの比較。補正RTE誤差は別field | 同じ064 H_DFを基準とした全frame PF |
| A / 補正RTE | A=1、RTE/law差は決定論として0 | Aと補正RTEを候補固有に評価。finite-bit実装lawのsignal差は未知 | A=1、RTEなし |
| CA/DF reserve | .00159 Ha / CA/8 | 一致 | 一致 |
| β/phase room | βtotal π/6、βstat π/12、残る決定論room π/12 | 一致 | 一致 |
| Failure | 候補ごと.01、13round×X/Y各.01/26 | 一致 | 一致 |
| Oracle/準備 | own-DF ground oracle、準備費用除外 | 一致。ただし状態自体は異なる | PRと同じ064状態 |
| 保存十分shots合計 | CDF13606 / XDF13600 | ideal15008 | ideal/numerical13706 |

OLD `accuracy_target` とNEW/PRICE `target` は全keyの値が一致し、同じ `AccuracyTarget` / RPE horizon / Hoeffding / energy会計を再利用する。s1/r32と決定論r非適用という方法差自体は不公平ではない。各方式を共通taskへ接続したときの負担を比較する。

誤差は同じraw complex-signal normへ換算してから扱う。既存式は、PRでは `δsignal=A(εPF+εRTE)+εsynth+εlaw`、`ρ≥A−δsignal`、`βdet≤asin(δsignal/A)`。決定論はA=1/RTE=0。DFのHa単位誤差はWeyl energy biasとして最終energy評価で一回加算し、dimensionless PF/RTEへ直接足さない。Attenuationもenergy誤差へ加算しない。[共通accuracyコード][ACCURACY]

| m12の保存値（無次元、位相はrad） | B2 CDF8 | B3 XDF8 | PR064 ideal | 同064 D |
|---|---:|---:|---:|---:|
| PF operator誤差 | .06913693070912988 | .06842795072754271 | .09095865272735126 | .09080980745079276 |
| 補正RTE operator誤差 | 0 | 0 | .043175938105442346 | 0 |
| Attenuation | 1 | 1 | .8282103212683136 | 1 |
| Raw-signal半径下限 | .9308627080514446 | .9315715002619364 | .7171186687014919 | 候補固有保存判定を使用 |
| 合成complex誤差pilot | 3.61239425568958e-7 | 5.490105208994203e-7 | **0を仮定** | 数値合成profileを使用 |
| 位相bias | .06919248973495769 | .06848201485765056 | .13454010965229243 | 候補固有保存判定を使用 |
| Shots/axis | 590 | 589 | 994 | 候補固有保存整数を使用 |

表は保存roundの代表表示で、今回新しい全round比較や優劣判定を計算していない。[OLD round CSV][ROUNDCSV]・[NEW JSON][NEWJ]

候補別failure.01は、4候補全体や事後選択を含むfamilywise保証ではない。予定十分shotsは実量子outcome抽選の実績でも普遍的な最小shotsでもない。

## 6. Native compiler・合法な融合・費用scope

| 対応項目 | 静的照合結果 | 比較上の意味 |
|---|---|---|
| P0 | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、approximation_degree1、qubits_initially_zero=false。OLD/NEW/PRICEのcompiler dict一致 | 別compiler設定の費用を混ぜていない |
| Precision/phase | 共通 `nature_jw_exact_zero_lapack_atan2_v1`、`qiskit_P0_mpfr160_declared_phase_v1`。MPFR160 declared phase、位相fitなし | 同じ精度処理を使用。実機/FT synthesisモデルではない |
| Controlled作用 | ancilla-last、control0 identity/control1対象evolution、κ/scalar/対角I項を保持 | operatorや相対位相を省いた安い回路との比較ではない |
| Gaussian/融合 | 共通PrecisionBuilder/StrongBuilder。B2/B3は全S2 sweepの相対境界融合。PRはhead半sweep内部相対融合、tail exact basis_key共有。cross-head/tail一般相対融合なし | 各方式に既存の合法な最適化を適用。PRへ未実装融合の節約を足さない |
| κ basis | **旧CDFは8モード直接eigh**。XDF B3はκ/各Gをspatial eigh→exact spin lift。最新PRと同064 Dはκをspin対応eigh | 全routeの基底選択が同一ではない。凍結実装のtask比較はできるが、J/U更新だけの純粋ablationではない |
| CDF/XDF frame basis | CDFは保存Uのspin lift、XDF B3は元Gを保持するspin対応対角化 | B3は弱い旧B0 XDFではない。旧CDFと同じ基底配列を強制する必要はない |
| 正しさ | B2/B3はfull512 raw/P0/reference・actual adjoint・complex X/Y PASS。PRは初期32と別独立256の各対象trajectoryでfull512検証 | 対象回路の数値correctness。PR全lawの一様誤差上界ではない |
| Shell | head/X/Yをkernelと分離。共通の保存shell recipe/metrics、RZ head2/X2/Y1 | Head/scalar位相、測定前操作を総費用から落としていない |
| 会計 | 各軸 `N_m,b × [2^m C_kernel + C_head + C_readout,b]`、s1、全wire fence、scheduled serial sum | 1 kernelのgate数だけの比較ではない。全round一体最適compileの最小depthではない |
| PR occurrence | 各coherent occurrenceに独立trajectoryを要求。校正256本を全shotへ同じsampleで使い回すモデルではない | 平均価格をWで重み付ける期待値会計。校正標本数をshots倍率にしない |

実装出典：[StrongBuilder/spin_adapted_eigh][BUILDER]、[旧CDF prepare_cdf][CDFMODEL]、[最新prepare_arm/build_kernel][PRMODEL]、[旧serial/RPE接続][OLDCODE]、[最新price adapter][PRICECODE]。OLD SOURCE31件のうちNEW manifestと共通する30件は同じhash。差分1件 `trotterlib/rte.py` は `event_unitary(..., arithmetic=None)` のopt-in演算backend対応であり、既定binary64経路を変えていない。決定論にRTEを適用したという差ではない。

| h=.1 controlled kernel、準備/shellを除く | RZ | RZ depth | CX | CX depth | Total depth | Circuit size |
|---|---:|---:|---:|---:|---:|---:|
| B2 CDF8（固定） | 11062 | 5577 | 7008 | 6280 | 12127 | 21011 |
| B3 XDF8（固定） | 10885 | 5524 | 6892 | 6192 | 11973 | 20606 |
| PR064/r32（独立256の標本平均） | 7497.4765625 | 2584.62890625 | 3359.8046875 | 2337.15625 | 5657.14453125 | 14733.390625 |
| 同064 D（固定） | 11026 | 5568 | 7000 | 6276 | 12098 | 20919 |

出典：[CDF packet][PACKETCDF]・[XDF packet][PACKETXDF]・[PRICE primary.calibration][PRICEJ]・[NEW D.metrics][NEWJ]。副指標の独立CI/総順位を本書で作らない。PRの安い1 kernelを、そのままshots/反復込みの優位と読まない。

## 7. 合成誤差・CA・価格の証拠資格を分ける

| 項目 | 保存決定論B2/B3 | 最新PR064 |
|---|---|---|
| 数値合成誤差 | full512全入力/出力のcompiled/reference Frobeniusとnorm growth、shellを13roundへ伝播 | 校正対象各trajectoryのfull512残差はあるが、全lawに有効な合成上界なし |
| CA判定 | 合成＋PF、候補DF誤差、RPE統計を接続し `NUMERICAL_PILOT_CA_PASS` | ideal/assumedなら `CONDITIONAL_CA_PASS`。Evidence laneは `UNRESOLVED_SIGNAL_ERROR`、shots/費用null |
| Ideal/assumed | OLDの公開総費用は数値合成誤差を**含む**lane | Idealはsynthesis/law追加誤差0を仮定。Assumedは各.001Aという感度仮定。実測した上界ではない |
| 厳密性 | CAR/expm/Operator/norm等の丸めenclosureなし | 全law/implemented-law signal差/ideal価格転送なし。未知項を0と認定しない |
| Native価格 | 固定gate metricsからの条件付き整数会計。費用intervalが点と一致しても科学誤差の厳密保証ではない | 固定finite-bit実装lawの擬似乱数IIDモデル、固定N256・近似Student-t。Nominal95%、被覆率のvalidated保証なし |
| actual CA | `CA_achieved=null`、rigorous enclosure null | `actual_CA_achieved=null`、guaranteed_resource_superiority null |

既存energy評価値はCDF .0013549794824010532、XDF .001373656337788953、064 ideal .0012783262210980142 Ha。実測energy誤差ではない。共通目標内という数値判定と、実装全体が実際にその精度を達成するという保証を区別する。

PRICEは初期native32をpoolせず独立256件を使用し、precision半幅目標は同064 Dの1%、登録判定は `NOMINAL_MODEL_PR_LOWER`。観測最大値は全support上界ではない。13roundで同じ価格平均を使うため、保存価格SEはW倍で伝播し、sqrt13で縮めない。母平均CIとRPE failure.01は別の不確かさである。

## 8. 保存値から直接言えること・言えないこと

全行はH4/1.00 Å/STO-3G、rank8、s1/δ=.1/m0…12、own-DF oracle/準備除外、上記RPE/P0/serial条件で読む。PRだけLD2/r32/K0。

| 方式・証拠lane | 保存条件付き総RZ | 出典・資格 |
|---|---:|---|
| B2 CDF8・決定論S2 | 102056444569 | OLDの数値pilot、固定kernel |
| B3 XDF8・決定論S2 | 100267421030 | OLDの数値pilot、固定kernel |
| 最新PR064/r32 | 100330684089.25 | PRICE独立256、ideal laneのモデル内期待価格推定 |
| 同じ064 J/Uの決定論S2 | 105038112319 | NEWの同J/U強D、ideal/numerical保存値 |

OLDの[候補CSV][CANDCSV]・[26round][ROUNDCSV]・[52軸][AXISCSV]・[6metric比較][COMPCSV]はJSONと対応する。最新PRは初期native32の点費用101118440083と区別し、主要価格証拠にはPRICEを使う。

保存PR総RZのnominal95%近似区間は **[99428513537.38805, 101232854641.11195]**。同064 Dに対する保存比区間 **[.9465946344829994, .9637726003078625]** はその同J/U分母にだけ属する。[PRICE結果][PRICEJ]

- 保存XDF8の100267421030は、このPR総RZ区間内にある。PRがXDF8より低費用という根拠にはならない。XDFの統計的優位・同等性・非劣性も認定しない。
- 保存CDF8の102056444569は、区間上端より大きい。ただしこれは保存scalarの位置関係の記述であり、新しいPR/CDF比区間、検定、登録PASS判定を生成したものではない。Laneの資格差も残る。
- PR/D約4.48%というnominal低費用の結果は、**同064 J/Uの比較I**として残る。別DF表現を含む決定論全体、最強XDF/最強CDF、hardware実時間、FT T/Toffoliへの優位には拡張しない。

## 9. 比較I〜IVと差異の分類

| 比較 | 現在できること | 未比較・未資格 |
|---|---|---|
| I：同一J/U PR対D | 同064のHamiltonian/ground/DF誤差を共有。固定r32対強Dについてnominalモデル内の主RZ低費用を支持 | Actual CA、全law、一般PR優位、hardware、最適r |
| II：共通task、別DFを許すPR対XDF/CDF | 元問題・CA/予算・RPE/P0/費用scopeを対応させ、保存値を資格付きで並べられる。XDFを正当な対照として残す | 同一lane/保証水準での新しいcross-DF判定は未登録。PR対XDFの差の符号・同等性は未確定 |
| III：同じ達成DF誤差で方法比較 | 保存値が一致していないことを特定 | 誤差を一致させた候補・対応native/合成評価。今回の共通CA taskの必須条件へ変更しない |
| IV：各方法を最適化した一般比較 | 現在の固定候補・実装・政策の射程を明示 | 全rank/r/δ/PF/基底自由度の最適性、最強決定論法への一般勝敗 |

| 観測差 | 分類 | 技術上の扱い |
|---|---|---|
| 元inputとPR prepareの容器hash | 来歴/metadata差 | 同じ元snapshotからの直接保存経路を確認。違う物理問題とはみなさない |
| J/U、signed λ/G、DF誤差、ground、PF/RTE/A、shots | 科学条件の実質差 | Task-level方式差として保持。別候補の数値を流用しない |
| numerical合成lane対ideal/assumed、guardの方式 | 証拠資格の差 | 結果を同じ厳密保証へ昇格させない |
| κ basisと相対融合scope | Native実装/会計範囲の差 | 固定された合法なrouteを比較。純粋な目的/U効果のablationではない |
| r=0/K=0対非適用、CA_achieved対actual_CA_achieved | 表示/schema差 | 決定論の意味とnull資格を保ち、再計算・旧SOURCE修正は不要 |

主要な共通target/compiler設定の不一致は見つからなかった。ただし、これは各方式の数値保証・最適性が同一という認定ではない。異なる方法・表現を許す比較と、実装差を全て消した因果ablationを分ける。

## 10. 不足証拠と最小の追加確認案（未採択・未実行）

**今回の限定的な監査結論に新fitや追加native校正は不要。** 強い主張へ進む場合だけ、下表から研究Cが問いを選ぶ。新しいXDF fitや大量校正を既定工程にしない。

| 判断したいこと | 不足証拠 | 最小確認案・再利用 | 何が変わるか・実行境界 |
|---|---|---|---|
| 共通の反実ideal条件で保存XDF/CDFとPRを比べたい | OLDは数値合成誤差を含み、PR主laneは理想合成を仮定。完全に同じlaneの保存比較がない | 保存PF13/DF/metricsだけでXDF/CDFのideal laneを別途評価する案。既存 `c_same_ju_strong_d.ideal_plans`、`AccuracyTarget/assess_round/assess_energy`、serial集計を再利用。NPZ/QPY/fit/build/compile不要。OLD数値laneは保全 | モデル条件を明示した比較IIへ揃うか。保存値の新規科学再解析なので別明示許可。今はN/費用を算定しない |
| 既存XDF/CDFへの価格母平均の主張を増やしたい | PRICEの登録比較相手は同064 D。XDF/CDFへの新しい比較・選択範囲は未登録 | まず主比較相手・lane・選択/多重性・必要な精度/主張を事前に決める。保存 `primary` と既存affine会計/近似t実装を再利用可能。ただし現在の区間を新しい事前登録結果へ改名しない | Point記述からどのモデル限定推論へ進めるか。新しい比/CI/検定は別科学許可。旧標本再利用の探索後の扱いを判断する前に新256/640を自動追加しない |
| 実装同CAを主張したい | PR全law synthesis/implemented-law signal差。双方の厳密reference roundingも未資格 | 主張に必要な信号誤差資格を選び、既存独立CAR/reference/accuracy接続を利用する技術設計。有限sample残差・旧F2/F3固定保証を全lawへ流用しない | 反実資源から実装精度の主張へ進めるか。今回は有用な上界が準備済みとは認定しない。追加保証計算は別指示/許可 |
| Native差をJ/U/目的だけへ因果帰属したい | κ basis、DF、head/tail、PF/RTE、融合scopeも変わる | 既存StrongBuilder/SPIN/relative optionと保存metadataで問いに必要な対照だけを設計。未測定節約を費用へ足さない | 因果ablationの範囲を定める。実回路比較は新SOURCE/条件/別許可後 |
| 達成DF誤差一致/一般最適法比較 | 対応候補・共通収束/探索政策・汎化証拠 | 比較III/IVを本当に中心課題にするか先に判断。保存prefix等を候補資料として調べても、native/誤差資格を無条件流用しない | 比較IIとは異なる研究課題の採択。新rank探索・fitを自動開始しない |

新規実行契約は本書では作成しない。どの比較laneと統計的主張を正式対象にするかが未選択であり、現時点でその選択をCodexが代行しないためである。選択後のCodex担当は必要な最小adapter/人工検証と、実際のSOURCE・入力・call範囲・STOPを固定する技術準備。既存runの残枠を転用しない。

## 11. 再利用コード・静的検証・保全

以下は読解・照合に再利用した資産であり、科学関数を実行した一覧ではない。

| 資産 | 確認した内容・既存試験 |
|---|---|
| `c1_stage1.corrected_kappa/load_input`、`c_pr_aware_df_pilot.prepare_worker` | 元input member・κ・spin/bit来歴。保存配列をdecodeせずSOURCEを追跡 |
| `c1_h4_strong_baseline`、`c1_h4_b3`、`c1_h4_native_fair` | Spin対応、真空位相、relative境界、full512/actual adjoint/X-Y、compile profile。既存B3/強baseline人工testsの対象範囲を確認 |
| `c1_h4_strong_ca_resources`、`c1_h4_ca_resources`、`c1_h6_ca_resources` | 保存scope/identity gate、PFと合成の一回計上、整数N検証、scheduled serial/aggregation。既存人工testにmissing PF/未資格/不正identity/STOPを含む |
| `c1_stage3_accuracy`、`rpe_resource_accounting`、`c1_stage3_connection` | Raw signal→phase→Hoeffding→energy、candidate failureとserial費用scope。共通SOURCE/hashを確認 |
| `c_update064_r32_downstream`、`c_pr_aware_df_stage_b`、`c_update064_r32_price_statistics`、`c_r32_price_statistics` | 同064 J/U、ground/alias、独立occurrence、affine W/shell、固定独立cohort/nominal区間の資格 |

OLDの報告・4種CSV・result/候補JSON/packet・契約/SOURCE/STOP・summary、計14件の公開本文はローカル固定Git blobと一致。最新PR/PRICEの固定報告・JSON・SOURCE・STOP等は共通起点の[11月3日統合照合](../../artifacts/research_c_latest_evidence_integration/2026-11-03/static_verification.json)を再利用し、必要な固定本文も取得して照合した。今回の公開本文照合は合計28件。旧SOURCE31件と契約manifest hash、共通accuracy/compiler/phase/Gaussian、候補export/input hashのbinding、CSV→JSONの転記とcoverageを静的確認した。

標準ライブラリによる734箇所の保存field・条件・SOURCE・リンク等の一致確認がPASS。CSVは候補2/round26/axis52/比較6行を対応付け、44本の絶対commitリンクをそれぞれのGit treeで確認した。これは科学関数を再評価したPASS数ではなく、既存値の転記・構造検査である。文献の追加網羅調査、新CI/検定、候補間の新ランキングは0。独立資料の取得は並列化し、共有環境・他jobの変更や人工的なCPU/RAM制限は行っていない。

今回用の確認は標準ライブラリでJSON/CSV/Git objectを読み、保存値への一致だけを検査する一時処理。科学moduleをimportせず、NPZ/QPYをdecodeせず、norm/新shots/価格/CIを再計算しない。新しい研究runner・検証framework・承認証明は追加しない。文書だけの変更なので、既存人工/科学テストは再実行しておらず、旧PASSを今回の新試験実績とは表示しない。

既存manifestのstrong-baseline/strong-CA recordと科学SOURCEの対応を確認した。新科学validation recordは作らず、既存manifestは無変更。旧SOURCE/raw/契約/grant/ledger/marker/STOP、元dirty worktree、Codex-Bの作業には触れない。保全確認の射程は本branchの一文書のみの差分で、他担当の全作業領域を不変と認定するものではない。

## 12. 次の研究判断への引渡し・mandatory STOP

- **ユーザー／研究C**：同一J/Uの限定肯定結果Iと、有力な別DF対照を許す比較IIを分け、中心の問いと必要な証拠laneを選ぶ。現在のXDF8を正当な対照として残す。
- **統合担当Codex**：Codex-A本書とCodex-Bの方法差/近接正則化資料が揃った後、共通Evidence Map・概要への最小統合を担当する。本branchでは共有ファイルを編集しない。
- **後続担当Codex**：具体的依頼後に必要最小限の比較準備を行う。保存値再解析・新CI・fit・compile等の科学実行、重要科学レビューは別の明示許可後のみ。現scopeに未完了計算はない。

独立新規性、論文化GO/NO-GO、研究Cの終了・継続は採択しない。Commit/push後にremote SHA/tree・公開本文・リンクと一文書差分を照合し、**mandatory STOP**する。

## 固定公開参照

[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md
[OLDJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/result.json
[OLDC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/docs/research/research_c_c1_h4_strong_baseline_ca_resources_contract_binding_v2_20261026.json
[OLDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/technical/SOURCE.json
[OLDSTOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/STOP.json
[B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0c89f611eac92c939edde67890f5cc4391d098e/docs/research/research_c_c1_h4_strong_baseline_execution_20261026.md
[B2J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0c89f611eac92c939edde67890f5cc4391d098e/artifacts/research_c_c1_h4_strong_baseline/2026-10-26/run01/result.json
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/866d3ba327abd5ca38af5d7a9347f6c91294958f/docs/research/research_c_c1_h4_b3_execution_20261026.md
[B3J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/866d3ba327abd5ca38af5d7a9347f6c91294958f/artifacts/research_c_c1_h4_b3/2026-10-26/run01/result.json
[NEW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/docs/research/research_c_update064_r32_downstream_execution_20261103.md
[NEWJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/artifacts/research_c_update064_r32_downstream/2026-11-03/run01/result.json
[NEWC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/docs/research/research_c_update064_r32_downstream_contract_unapproved_20261103.json
[NEWS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/artifacts/research_c_update064_r32_downstream/2026-11-03/technical/SOURCE.json
[NEWSTOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/artifacts/research_c_update064_r32_downstream/2026-11-03/run01/STOP.json
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/docs/research/research_c_update064_r32_price_statistics_execution_20261103.md
[PRICEJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/result.json
[PRICEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/docs/research/research_c_update064_r32_price_statistics_contract_unapproved_20261103.json
[PRICES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/artifacts/research_c_update064_r32_price_statistics/2026-11-03/technical/SOURCE.json
[PRICESTOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/STOP.json
[PHYSICAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/artifacts/research_c_u_step_control/2026-11-03/run01/physical/update064.json
[NATIVEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/docs/research/research_c_c1_h4_native_fair_contract_unapproved_20261026.json
[CDF1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/docs/research/research_c_c1_stage1_eval_only_execution_20261025.md
[CDF1J]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/result.json
[B3CODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b40fce5c62c9fff97f68073cbc02182e4bbf2b9b/src/trottertracks/c1_h4_b3.py
[XDFPFJ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/run01/result.json
[INPUTMETA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86b8b002704aced8ca239414080f53793de81658/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json
[PRINIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e1085c95f5c01b68566e17d543ace5832cb94061/docs/research/research_c_pr_aware_df_pilot_contract_x0_v2_unapproved_20261027.json
[PRPREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e1085c95f5c01b68566e17d543ace5832cb94061/src/trottertracks/c_pr_aware_df_pilot.py
[KAPPA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aeedd96178a3f50a15bc2701c3c332f4f391b64/src/trottertracks/c1_stage1.py
[NATIVECODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aeedd96178a3f50a15bc2701c3c332f4f391b64/src/trottertracks/c1_h4_native_fair.py
[CANDCSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/candidates.csv
[CDFPREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_stage3_feasibility_science/2026-10-25/run01/prepared.json
[GROUND64]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/artifacts/research_c_update064_r32_downstream/2026-11-03/run01/A/candidate_ground.json
[ACCURACY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aeedd96178a3f50a15bc2701c3c332f4f391b64/src/trottertracks/c1_stage3_accuracy.py
[ROUNDCSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/rounds.csv
[BUILDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aeedd96178a3f50a15bc2701c3c332f4f391b64/src/trottertracks/c1_h4_strong_baseline.py
[CDFMODEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aeedd96178a3f50a15bc2701c3c332f4f391b64/src/trottertracks/c1_stage2_native.py
[PRMODEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/928fda0a6df260b66eba1bbea63b51f2734e7b54/src/trottertracks/c_pr_aware_df_stage_b.py
[OLDCODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aeedd96178a3f50a15bc2701c3c332f4f391b64/src/trottertracks/c1_h6_ca_resources.py
[PRICECODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9837eac59b94113bf057fcc311a27b35e60343f5/src/trottertracks/c_update064_r32_price_statistics.py
[PACKETCDF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/candidates/B2_CDF_relative.packet.json
[PACKETXDF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/candidates/B3_XDF_spin_relative.packet.json
[AXISCSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/axes.csv
[COMPCSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54ceff9981d9bc53210db97c743a7d1cf694f362/artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/comparison.csv
