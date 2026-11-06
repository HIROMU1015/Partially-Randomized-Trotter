# O2-R1：既存証拠による研究候補の選別

2026-11-06 JST。現セッションのサーバーで、固定文書・SOURCE・保存summaryを静的に確認した。入口は[REPORT](REPORT.md)、版・結果・再利用対応は[BASELINES_AND_REUSE](O2_R1_BASELINES_AND_REUSE.md)。ここでいうO2-R1は構造化Return Aggregation（RA）のfull-wrapper接続であり、旧Track BのR1 stageとは別である。

**判定：`NO_SUPPORTED_R1_CANDIDATE`。第一候補なし。**

これは今回の固定資料から、追加の最小科学検証へ進む理由を持つ具体的な「入力＋方式＋追加改善機構」を支持できなかったという技術的選別である。候補であるために総費用改善の実証を要求したわけではない。既存Closed/CTSにまだ含まれない効果と、その効果を判別する対象を、既存証拠から特定できるかを基準にした。RAの一般no-go、正式な研究終了、最適性、新規性、科学的GO、論文化可能性の判定ではない。

## 1. 共通taskと評価の境界

元の物理Hamiltonianの厳密基底状態 `H|E₀⟩=E₀|E₀⟩` を共通に利用できると仮定し、その状態準備を除外する。H_D/近似Hのgroundへの置換、未知E₀・gap・完全state vector・真のpath別signalの無料利用は認めない。目標は各資源aについて

\[
G_{\mathrm{total}}^{(a)}=\sum_m\sum_{b\in\{X,Y\}}N_{m,b}\,
\mathbb E[C_{m,b}^{\mathrm{full,no\text{-}prep},(a)}].
\]

T/CX/RZ/1Q/depth、古典setup/online時間、memoryは別座標である。Tはnative 1Qの内数で、加算しない。D halves、child、control、basis、ancilla準備・X/Y readoutは、物理入力状態準備の除外を理由に無料にできない。下記の旧hは準備/readoutをまとめた旧モデルの変数であり、今回の状態準備除外へ数値を移し替えていない。

PF/有限RTE/finite-bit/nativeのbiasとattenuation、測定統計、正のsignal半径、全roundのbranch/aliasとenergy成功条件を保持する。厳密元H groundであっても近似PFの固有状態・基準半径1は自動的に成立しない。[研究目的][PURPOSE]と[全RPE費用地図][RPEMAP]の留保を適用する。今回は新しい分子・精度・成功確率・r/degree・compiler・主資源を選ばない。

## 2. 非自明returnの構造と既存実装

G6は `Q_i²=I` だけを使い、raw wordをreduced wordごとに係数集約する。既存testには `(0,1,1,0)→()` と `(2,1,1,0)→(2,0)` がある。これらはreturnの構造例であり、新しい入力評価ではない。Taylorの異なる次数から来る符号付き寄与を先にまとめ、even suffixをparent、先頭に付けるlabelをchildとしてpairする。回転generatorはchildのQ_iであり、一般のodd word全体をinvolutionとする構成ではない。[G6][G6]、[kernel][KERNEL]、[既存test][TEST6]

`ReturnKernel.parent` のa・child mass・sからparent依存角度が決まり、reduced parent積とchild rotationを実行する。empty parentでもchild rotationは残る。係数項数の削減は、回転数・normalization・必要Nの削減と同じ意味ではない。

| 層 | 既存証拠が与えるもの | 今回の候補判断に効く限界 |
|---|---|---|
| G6 数理構成 | even-suffix/parent-child、符号・位相・語順、ordinary envelopeによるrejection/zero-fill、finite-bit supportとrational weight | Green再帰は既知free-product構成のZ₂特殊化、pairingは既知Euler、zero-fillも既知部品。再提案する新アルゴリズムではない。[G6ART] |
| G7 生成・統計 | 非列挙local生成、moment/rangeによる十分N、helper＋条件付きCQ、取得RZ価格 | P5の登録旧3対照に対するT利益はあるが、後続Closed P5対照が必要。P3 FullはClosed P3より不利。[G7] |
| G8 on-demand | live miss合成、actual adjoint、cold/warm cache、accepted-call cap | 有限traceであり全support終端保証・分子native価格ではない。P3 Full/Closed差は既存I2 IS診断では消える。全normalizer/価格表の取得を無料にしない。[G8] |
| G9 strict native | direct controlled、phaseを保つactual adjoint、matched CTS、Closed P5 | Closed P5とGeneral Fullは同じ理想full-return ensemble。量子改善機構を二重計上しない。実digital law/予算/古典取得経路は別。[G9] |
| G10 degree別 | m=3/5/7の各finite target内のnative T/CX/1Q・moment/range・N | 低次数Closedを許すとFullにT負結果がある。異degreeを同exponential精度の比較としない。m5はG9再利用。[G10] |

G6の条件は正有理p、Hermitian involution、odd cutoff、short-step domainである。以下のG7〜G10引用は既知development条件で、分子geometry/basis/DF rank/L_D/δ窓は非該当。G7/G8のP3 controlはp=(3/7,4/7), x=2/5, m=3、P5はp=(1/5,3/10,1/2), x=5/7, m=5で、p/xも異なる。両者の差をdegree単独の因果効果にしない。

## 3. 強いbaselineが既に説明する利益

### G9：Closed P5とFullを別機構に数えない

共通providerは3 system qubits、Q0=Z0、V1=R_XX01(π/4)、V2=R_XX12(π/4)R_ZZ01(π/4)、Qi=Vi†ZiVi。G9は同じfinite P5 operator meanを比較し、特定の分子ground-state taskではない。p=(1/5,3/10,1/2), x=5/7, m=5、probability160/root256 bit、η=ρ=10⁻¹²、strict primitive誤差10⁻⁶、axis誤差1/200、22軸α=49/22000と11資源行の失敗配分でfamilywise0.05。

次表は[G9保存CSV][G9CSV]の値の引用で、新しい順位計算・丸め差評価ではない。Tは二axisのnative-event切片で、未知旧prep/readoutのaffine項が別に残る。

| 方式 | N/axis | T切片 | CX native |
|---|---:|---:|---:|
| General local Full | 1,076,085 | 257,506,695.137339354 | 15,748,232.374372307 |
| Closed P5 | 917,129 | 255,860,636.626162171 | 15,647,565.042511262 |
| matched CTS | 1,596,220 | 364,311,583.986430824 | 9,766,516.586824298 |

既存報告はClosed P5を登録direct集合のT切片・共通prep/readout係数で最小とする。これはRAの有効な限定成果であり取り消さない。しかしFullとClosed P5のaccepted event当たりTは同じ理想ensembleに対応する。両者の差はnormalizerアクセス、zero-fill、保守的budget、有限bit生成等にあり、「General Full固有の追加回路利益」にはならない。CTSのCX上の強さも保持する。[G9]

### G10：高次数returnが残るだけでは次候補にならない

G10は上記p/x/providerでm=3/5/7を別々に比較する。primitive/bit精度は同系統、axis誤差1/200、34軸α=49/34000と17資源行を合わせfamilywise0.05。次は[G10保存CSV][G10CSV]の表示値そのままである。

| degree内の比較 | 方式 | 二axis T切片 |
|---|---|---:|
| m=3 | Full | 279999178 |
| m=3 | Closed P3+tail | 279459012.3 |
| m=7 | Full | 280249067.7 |
| m=7 | Closed P5+tail | 272320319.3 |

m7 Fullは、P5 prefixの外にあるreturnも集約できる構成である。それでもこの保存native会計ではClosed P5+tailが強い。追加高次returnの存在だけから有望性を推定しない。保存fixed-dictionary lowerは取得済み最適lawではない。他degree・新r・別precision・新compilerの自動探索へは進まない。[G10]

### D系列・Parent：既に検証した接続を新機構にしない

D1〜D3は上記synthetic provider、state000、τ=5/7、P5、r=1/2。fixed r内は同じ `P5(-iτR/r)^r`、r間は有限targetが異なる。D1の独立step加法会計に対し、D2は実sequenceのliteral inverse境界相殺を既に全方式へ適用した。保存D2統合ではr2のT最小はClosed P5、CXにはCTS/partialの領域、1QにはP3の領域がある。新たに単一metricへまとめない。[D2]

D3は保証と合成価格を分離し、旧価格ではP5、新価格の全方式比較ではP3+tailが強い場合を保存している。G1_P1 r1のP3/P5差は約0.170%という既存記述で、価格依存の小差である。degreeを下げたtarget比較ではない。別価格へ頑健なRA利益の証拠にはしない。[D3]

Parent/S4はidentity shareの変更と既知ISを含む、既存の別自由度である。common配分のfamily内L1最小性、pure端点での安いprovider、訂正sqrt(C) KKT、70 lawの条件付きT成果を保持する。一方、best-to-bestの差をallocation-only効果とせず、未知h・online費・多step・分子全RPEを残す。同developmentの数値探索は正式に区切られており、RAとの組合せという名称で再開しない。[PARENT]

## 4. C1：保存logical pathは再開の根拠を与えていない

対象はlinear H4、1.00 Å、STO-3G、legacy DF12、R1 generation-prefix L_D=3、8 system qubits、13 bases/324 labels。T=4/5, q=1, r=4, degree5, microstep1/5, σ=+1、H_prob2048/root4096、η/ρ/angle tolerance10⁻¹²、angle root256/terms64。測定対象は `TAIL_ONLY_PLUS_SCALAR`。

Ordinary/Fullは旧C1-5Aの完走substage、Closed P5は別SOURCEのP5R完走であり、旧全job failureを成功へ変更しない。[C1LIMIT][C1LIMIT]が引用する3方式各1024 whole paths/4096 microstepsでは、valid zero pathsは全方式0、child rotations/pathは全方式4.0。raw degree0はOrdinary4070、Full/P5R4073という保存値で、empty parentは無料no-opではない。

保存logical差は小さく、strict native・全law期待価格・認証十分N・D halves/readout/full wrapperは未取得である。経験的weight二乗平均や観測maxから、新しいM2/range/Nを作らない。Fullの4096 local packet query、Closed P5の104,977群setup等は既存費用であり、cacheを認めても独立量子shotが一度で済むことにはならない。異なるprocess/cache/監視境界のCPU/wallから新speedupを計算しない。[C1LIMIT]

R1の324 exact-dyadic係数とbasis registryは保存されているが、物理DFとの結合は `MODEL_BINDING_NUMERICALLY_SUPPORTED` と `MODEL_BINDING_UNCLOSED` を区別する。新basis/係数の来歴を旧SOURCEと同一視しない。[C1R1]

**C1は `C1_PAUSED_AFTER_EXPLORATORY_RESOURCE_ASSESSMENT` を維持する。** 旧worker直接exit原因UNKNOWNを解消する作業や、全UNKNOWN解消のための再実行・native整備を今回の候補にしない。[C1STOP]

## 5. 入力・方式ごとの技術的適格性

次表は保存値の新ランキングではなく、既存結果と提案機構の重複・不足による分類である。全候補に§1の同task、signal、統計、full-wrapper条件を適用する。

| 検討対象 | 想定される改善機構・非自明return | 強いbaseline／反証材料 | 残る不足と検証価値 | 選別結果 |
|---|---|---|---|---|
| G7/G8既知P3・P5のFull | return係数集約、reduced parent、pre-quantum zero | P3はClosedが強い。P5の旧対照は後続G9 Closed P5で更新済み | on-demand/cacheの存在は既知。全support native取得・生成保証を埋めても独立機構が新たに示されるとは限らない | 既存機構・後続比較で説明済み |
| G9既知P5のFull/Closed | 全P5 returnを少数群へ集約、正規化とqueryを軽量化 | Closed P5そのものが強いRA baseline。同理想ensemble、CTSはCXで強い | P5の限定T成果は有効。ただしClosed後に残る追加full-wrapper効果の指定がない | 新しいR1候補として支持せず |
| G10 m7 Full | degree6/7を含む追加return、長いraw parentの短縮 | 同finite P7のClosed P5+tailがTで強い | 新境界構造が逆転させるという入力依存根拠は未取得。高degreeというだけの再探索は不適格 | この固定入力では支持材料不足 |
| C1固定H4 | 分子basisの相殺、return cache、zero回避 | ほぼempty parentでもchild4回、zero回避0、small logical差。Closed P5と同等最適化が必要 | native/N/D/readoutが不足する事実だけでは有望性を支持しない。凍結継続 | 再開候補として支持せず |
| D1〜D3の境界＋RA | event/step間basis逆対、actual adjoint | D2が既に全方式へ実施。D3の価格/保証で勝者が変わる | D halvesとの接続は未評価だが、実際に利用できる追加境界と効果が未特定 | 既存結果の再命名を避ける |
| Parent/S4＋RA／J1 | child配分、安い端点、非加法path価格との連携 | 既知IS/既存allocation、同情報Closed/P3/CTS、S2の元law比較 | 安価なjoint normalizer/price取得という追加性質が具体化されていない。S4の既存正結果は保持 | 独立した具体候補を支持せず |

係数集約でsupport・normalization・weightが変われば、同じ補正meanでもraw signal、M2/range、必要Nは変わり得る。既存G7/Parentには既にBernstein十分Nの会計があり、variance-aware化を新法と数えない。zeroを含む全planned classical trialsとaccepted quantum trialsを区別し、accepted-only価格を無条件価格として使わない。zero判定前に実行したD/prefixの実費と、全trialの古典生成費を残す。根拠SOURCEは[再利用対応](O2_R1_BASELINES_AND_REUSE.md#3-静的に確認した再利用接口)に示す。

## 6. S2負結果との関係と、最小検証案の扱い

[S2 v2][S2]はD2 Closed P5の固定r2, x=5/14の同じ有限pair表で、qB/Bが元law P0/BをT主指標で改善しなかったという限定結果である。RA表現を変更した比較ではない。本判定はS2のno-go化に依存せず、G9/G10/C1/D/Parentの別の証拠から行った。

**具体的な支持候補がないため、今回の指示§10に従い次の最小科学検証案は作成しない。** 新条件・別分子・新degree・precision・regularizerを補って候補を作らない。独立候補を並行検証する提案も現証拠からは残さない。

再検討材料になるのは、既存Closed/CTSに吸収されない具体的なreturn/basis/child構造、取得可能な価格・normalizer・生成法、共通finite target/phase/統計/full-wrapper条件を持つ入力を、別途指定できた場合である。これは新しい証明・探索・実行の計画でも許可でもない。ユーザー／GPTが継続方針を判断し、必要な追加実装・科学計算・重要レビューは各範囲の明示許可後に限る。公開・取得照合後mandatory STOP。

[PURPOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/研究目的・研究課題.md
[RPEMAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/END_TO_END_COST_ANALYSIS_MAP.md
[G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md
[G6ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md
[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[TEST6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/tests/tracks/algorithm_codesign/test_g6_return_aggregation.py
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[G9CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/resource_rows_display_v2.csv
[G10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[G10CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g10_degree_result/2026-10-10/v3/resource_rows_display_v3.csv
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[C1LIMIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
[C1R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b48bb3f4f7b080baca15d6fa45711619880d64d/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/handoff.md
[C1STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_PAUSE_AND_RESTART_CONDITIONS.md
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md
