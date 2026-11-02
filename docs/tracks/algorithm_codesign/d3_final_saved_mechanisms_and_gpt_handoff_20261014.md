# D3 v2保存結果：最終機構分析とGPT引き継ぎ

2026-10-14 JST。保存scalar・既存key incidenceへのexact rational演算による技術分析。
科学分類を変更せず、新しい科学取得・合成・shot予算評価・pair評価は0。**mandatory STOP**。

P1でP3+tailがT最小となった主因は、先頭3 eventのroot keyの値上がりがP5で大きかったこと。
G1_P1のr1 P3+tailのT切片はr1 P5より約59.9万低いが、P5を分母に約0.170%の差にとどまる。
保証変更はr1の統計残余幅を大きく回復し、r2より大きく十分shot数を減らすため、最良rを変える。
これは今回の保存会計の説明であり、一般的優位・新規性・研究採択の判断ではない。

## 固定入力・証拠と分析の境界

正本result commit **`f0df9a3b36bc3540f0c83941871fa0d3110ddbf7`**、science S2 **`dceccaa774229527a46a1386131583a4f618f8ad`**。
[原結果/GPT引き継ぎ](d3_v2_results_and_gpt_handoff_20261014.md)、
[既存exact48行](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/resource_rows_exact_v2.csv)、
[既存comparison projection](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/saved_comparison_projection_v2.json)、
[既存factorial effects](../../../artifacts/track_b_d3_result/2026-10-14/v2/factorial_effects.json)を正本とする。
44入力をGitHubの上記固定commitから取得し、local bytes・fixed git blob・既存manifest（記載対象）と照合した。
[入力identity記録](../../../artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1/input_remote_identity_v1.json)に各URL・bytes・SHA256を保存。
resource_mapは531,617 bytes、SHA256
`a4bc655cdeda9a5a438f3034a70dce0d0b7bf647029c115b4d3696292f60a43a`であり、原2 chunksの結合一致も確認した。

対象は既知developmentの3-system-qubit synthetic involution、p=(1/5,3/10,1/2)、tau=5/7、sigma=+1、state=000。
Q0=Z0、V1=R_XX01(pi/4)、V2=R_XX12(pi/4)R_ZZ01(pi/4)、Qi=Vi†ZiVi。
finite P5(-i tau R/r)^r、r=1,2、同I1六方式。分子geometry、軌道basis、DF rank、split L_Dは適用外。
`closed_P3_tail`も同じfinite P5 targetを実現する表現で、Taylor targetを3次へ変更した比較ではない。
旧2579監査・36合成検証・D1/D2 event監査・全ランキング・全交点は再実行していない。
旧source、contract、authorization、消費済みmarker、STOP、原結果は変更しない。

## 価格寄与の接続

D2の保存positive-step条件付きproposal pi=q/Zから得た係数muをそのまま使う。
条件付き単価d、全trial単価C、全二axis切片Aは

\[
d_{i,T}=\ell_i+\sum_k \mu_{ik}t_k,\quad
C_{i,r,T}=r Z_i^r d_{i,T}-S_{i,r,T},\quad
A_{i,r,T}=2N_{i,r} C_{i,r,T}.
\]

ここでSは保存済みliteral-only境界節約額（r1は0）。D3計画のno-R-cancellation静的認証を参照し、
219,303 pairを再評価しない。unconditional step費用はZ*dなので、単純なr倍やacceptanceの欠落は避ける。
全trialのkey期待出現回数はnu=r Z^r mu、二axis全体は2N nu。
同Gの価格変化は `Delta A=2N sum nu Delta t`。N・law・moment・range・biasはP0/P1で固定。
native1Qも同じnuと保存1Q価格を用いる。Tのellは既存独立係数と一致し、1Qのliteral部分は
旧保存step scalarから既存incidence*旧価格を引いた残差として明示する。
native CXのprice-only差は0。Tはnative1Qに含まれ、重複加算しない。

[36 key価格・用途・event index](../../../artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1/key_prices_and_incidence_v1.json)、
[小さい価格CSV](../../../artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1/key_prices_display_v1.csv)、
`rows/row_00.json`〜`row_11.json`のexact寄与表で、角度ratio、old/new identity、価格差、mu/nu、
各方式/rのT・1Q増価を結ぶ。signed逆回転は保存sequenceのadjointで同価格。
RZ角度はsigned atan(ratio)、scale1のラジアン。Wはnative1Q数に含めず、controlled relative phaseの既存strict guardは保持。
全12構成×2保証×T/1Qのkey寄与合計は、既存factorialのprice-only effectとexact一致した。

## G1_P1 r1：P3+tailとP5の小さい差

| 保存量 | P3+tail | P5 |
| --- | --- | --- |
| 十分N/axis | 1103489 | 1083653 |
| 全trial T単価 P0 | 143.277318797 | 139.489993570 |
| 全trial T単価 P1 | 159.700056809 | 162.899691745 |
| P0 T切片 | 316,209,890.483 | 302,317,500.005 |
| P1 T切片 | 352,454,511.977 | 353,053,479.316 |
| P1−P0 T切片 | 36,244,621.494 | 50,735,979.312 |
| 保存M2 upper | 1.690512970 | 1.660089378 |
| 保存range upper | 1.300197281 | 1.288444558 |

両方式のacceptanceは1、同rなのでbiasとremainingは同じ。P3+tailは保存moment/rangeが大きく、Nが19,836多い。
P0ではP3+tailの単価も高い。P1ではP3+tailの単価が約3.200 T低くなり、shot面の不利を小さく上回る。
保存Nは実測量子shotsでも必要最小shotsでもなく、固定common Bernstein会計の十分forecastである。


符号をP5−P3+tailとして、旧価格の差は-13,892,390.478 T、
価格差によるmargin移動は+14,491,357.818 T、
新価格の差は+598,967.339 T（P5分母で0.169653%）。
対称恒等式による記述的配分では、shot差項-6,399,088.612 T、単価差項+6,998,055.952 T。
二項の和が新価格marginへexact一致する。一意の因果割合とは扱わない。

以下のkey寄与は「P5の増価−P3+tailの増価」。N変更の寄与を混ぜない。
| key alias / exact ratio | old→new T | marginへの価格差寄与T |
| --- | --- | --- |
| K32 / 2189372/2921709 | 68→80 | 45,675,681.914 |
| K33 / 310/413 | 70→78 | -30,666,353.855 |
| K19 / 1/6 | 70→80 | -1,843,748.099 |
| K15 / 5/42 | 70→78 | -1,744,289.452 |
| K22 / 4/21 | 68→80 | -1,692,671.622 |
| K20 / 3821/22720 | 70→82 | 1,513,207.320 |
| K23 / 15264/79345 | 68→82 | 1,179,320.244 |
| K24 / 3040/15757 | 66→80 | 702,670.558 |


K32=2189372/2921709はP5/local-fullの先頭3 event（index0,1,2）に使い、条件付き出現回数約1.75624。
K33=310/413はP3+tailの同じindex0,1,2の3 event、約1.73690。K32は68→80、K33は70→78で、
この二keyだけのmargin移動は約+1,500.933万T。他keyのnet移動約−51.797万Tを合わせ、全移動約+1,449.136万Tとなる。
多数の既存non-root eventを対応のない一対一因果比較にはしていない。

### 記号的価格感度

他価格・N・lawをP1のまま固定すると、margin Mは `M + a_k (t_k−t_k,P1)`。
K32の1T単位slopeは約+3,806,306.826 T、tie価格79.842638188。
K33のslopeは約−3,833,294.232 T、tie価格78.156253943。
すなわち連続記号変数としては約0.157T程度の一key変位でtieとなる。
これらは整数・parity・精度・strict phase条件を満たす合成sequenceの取得可能性を示さない。
全keyのexact tieと符号は[focused対照JSON](../../../artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1/P3_vs_P5_price_mechanisms_v1.json)に保存。
新精度・seed・合成器への頑健性は追加取得なしには判定できない。

## r別のP3+tail/P5 marginと保証によるr選択

| G | r | P0: P5−P3 T | P1: P5−P3 T | P1差/P5 |
| --- | --- | --- | --- | --- |
| G0 | 1 | -19,703,072.606 | 847,628.355 | 0.169295% |
| G0 | 2 | -13,531,909.403 | 10,046,003.364 | 2.002933% |
| G1 | 1 | -13,892,390.478 | 598,967.339 | 0.169653% |
| G1 | 2 | -13,404,778.483 | 9,953,287.851 | 2.003134% |

P1/r2でもroot key K27=18205486/50387421（68→80）とK28=43/119（70→78）の差が主要因。
P3+tailとP5の価格機構はr1/r2で対応するが、総費用には各rのNと反復単価を戻す。


| 保証 | r | Taylor積上界 | native実装積上界 | axis bias上界 | 統計残余幅 |
| --- | --- | --- | --- | --- | --- |
| G0 | 1 | 0.000553376286 | 0.000016000008 | 0.001138752588 | 0.003861247412 |
| G0 | 2 | 0.000017293084 | 0.000032000549 | 0.000098587265 | 0.004901412735 |
| G1 | 1 | 0.000184458762 | 0.000016000008 | 0.000400917540 | 0.004599082460 |
| G1 | 2 | 0.000005764345 | 0.000032000364 | 0.000075529418 | 0.004924470582 |

同rのこれらの予算は六方式共通。native primitive誤差会計は全4ケース1e-6固定。
G1でnative積上界が微小に変わるのはTaylor上界との積の伝播によるもので、新1e-7精度へのcreditではない。
G1はr1のremainingを0.003861→0.004599へ回復する一方、r2は0.004901→0.004924の小変化。
Nの主項は保存moment/remaining²、range補助項はrange/remainingであることを固定sourceの静的式から確認した。
今回この式を使ったNの再評価・最適化は行わず、以下は保存整数の表示である。
| 方式 | r1 N G0→G1 | r2 N G0→G1 |
| --- | --- | --- |
| ordinary | 2,088,098→1,472,332 | 926,574→917,928 |
| partial | 1,590,438→1,121,482 | 778,141→770,881 |
| P3+tail | 1,564,916→1,103,489 | 776,048→768,807 |
| local full | 1,803,142→1,271,473 | 847,756→839,847 |
| P5 | 1,536,780→1,083,653 | 773,966→766,746 |
| CTS | 2,675,065→1,886,134 | 1,185,751→1,174,685 |


r1は約29.5%減、r2は約0.93%減。同GのNはP0/P1で同じ。
`Delta_G A = 2 Delta N * C(P)`なので、同価格の保証効果に回路単価の変化を含めない。
factorial interactionは `2 (N_G1−N_G0) Delta_P C`で、Nへの価格creditではない。
[保証/r分析](../../../artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1/guarantee_and_repeat_mechanisms_v1.json)は各方式/rのsaved bias、remaining、N比、T差、factorial接続を保存。
| 方式 | P0: r2−r1 T G0 / G1 | P1: r2−r1 T G0 / G1 |
| --- | --- | --- |
| ordinary | -71,482,411.310 / 96,079,499.513 | -64,429,422.755 / 129,953,038.715 |
| partial | 20,712,916.433 / 144,374,154.607 | -2,465,836.689 / 142,247,313.443 |
| P3+tail | -7,031,335.539 / 121,074,156.391 | -8,315,606.255 / 134,478,061.062 |
| local full | -3,208,926.284 / 120,023,657.039 | -1,858,346.132 / 142,037,935.793 |
| P5 | -860,172.335 / 121,561,768.387 | 882,768.754 / 143,832,381.573 |
| CTS | 32,553,354.688 / 206,612,524.194 | 32,179,932.452 / 226,755,877.867 |

符号負はr2が低費用。G0/P0のP5ではr2の利益は約86.0万Tにすぎず、P1ではr2の追加増価がr1より大きく約88.3万Tの不利へ反転する。
G1ではr1の大きなshot削減がr2の反復費用節約を上回り、全六方式の各価格でr1が低いh=0 Tとなる。
保証と価格を同時変更した一つの効果として扱わず、既存factorialを参照する。

## primary partial/P5と全六方式

h=0 Tについて、primaryは両方式へ同じr=1,2自由度を与える。

| ケース | partialの選択 | P5の選択 | P5−partial T | partial分母のP5削減 |
| --- | --- | --- | --- | --- |
| G0_P0 | r1 partial | r2 P5 | -5,900,214.070 | 1.360214% |
| G1_P0 | r1 partial | r1 P5 | -3,551,866.560 | 1.161236% |
| G0_P1 | r2 partial | r1 P5 | -3,595,025.363 | 0.712907% |
| G1_P1 | r1 partial | r1 P5 | -4,271,340.721 | 1.195366% |


全12構成の勝者・次点は既存rankingから直接読む。次点が同方式の別rの場合も明記する。

| ケース / 指標 | 最小 | 次点 | 次点−最小 | 差/次点 |
| --- | --- | --- | --- | --- |
| G0_P0 / T | r2 P5 | r2 local full | 411,515.741 | 0.096085% |
| G0_P0 / CX | r2 CTS | r1 CTS | 1,586,455.445 | 9.692741% |
| G0_P0 / 1Q | r2 P3+tail | r1 P5 | 8,589,233.813 | 0.792293% |
| G1_P0 / T | r1 P5 | r1 local full | 1,945,462.667 | 0.639402% |
| G1_P0 / CX | r1 CTS | r2 CTS | 3,102,697.422 | 21.188858% |
| G1_P0 / 1Q | r1 P5 | r1 local full | 4,919,341.335 | 0.639402% |
| G0_P1 / T | r2 P3+tail | r1 P3+tail | 8,315,606.255 | 1.663672% |
| G0_P1 / CX | r2 CTS | r1 CTS | 1,586,455.445 | 9.692741% |
| G0_P1 / 1Q | r2 P3+tail | r1 partial | 34,734,793.343 | 2.705160% |
| G1_P1 / T | r1 P3+tail | r1 P5 | 598,967.339 | 0.169653% |
| G1_P1 / CX | r1 CTS | r2 CTS | 3,102,697.422 | 21.188858% |
| G1_P1 / 1Q | r1 partial | r1 P3+tail | 14,442,452.657 | 1.570076% |


G0_P1のT次点は同じP3+tailのr1。異なる方式で最も近い対照はr1 P5、差約916.323万T。
G1_P1のT次点はr1 P5で前述約59.897万T。
primaryのP5 T優位と、全六方式でP3+tailがT最小になることは矛盾しない。
G1_P1のnative1Q最小はpartial、CX最小はCTSで、T最小方式に全指標優位を付与しない。
各方式のr選択を許した保存順序、primary各指標のbest-r対照、exact winner/runner-up差は
[選択した比較のexact記録](../../../artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1/selected_saved_comparisons_v1.json)。新しい48行rankingやParetoは生成しない。

## 準備・読出し費用h>=0の領域

保存affineは各指標別 `G(h)=A+Bh`、B=2N Z^r。
hはaccepted whole quantum call当たりの準備・読出し費用、三指標を独立表示するformal domain。
physical条件h_1Q>=h_Tは保持し、未知classical online費用はNOT_ACQUIREDで0としない。
h=0はmodelの断面で、実装の準備・読出しが無料と測定された意味ではない。
以下の領域は既存comparisonのminimum_open_regionsを直接表示した。端点のtieは原JSONにある。


| ケース | 候補集合 | 指標 | 保存open領域: 最小構成 |
| --- | --- | --- | --- |
| G0_P0 | all12 | T | (0.000000, ∞): r2 P5 |
| G0_P0 | all12 | CX | (0.000000, 6.332293): r2 CTS; (6.332293, 6.637185): r2 partial; (6.637185, ∞): r2 P5 |
| G0_P0 | all12 | 1Q | (0.000000, 6,320.486372): r2 P3+tail; (6,320.486372, ∞): r2 P5 |
| G0_P0 | primary_equal_r_choice | T | (0.000000, ∞): r2 P5 |
| G0_P0 | primary_equal_r_choice | CX | (0.000000, 6.637185): r2 partial; (6.637185, ∞): r2 P5 |
| G0_P0 | primary_equal_r_choice | 1Q | (0.000000, 11.620966): r1 P5; (11.620966, ∞): r2 P5 |
| G1_P0 | all12 | T | (0.000000, 191.794073): r1 P5; (191.794073, ∞): r2 P5 |
| G1_P0 | all12 | CX | (0.000000, 2.180548): r1 CTS; (2.180548, 6.332387): r2 CTS; (6.332387, 6.642139): r2 partial; (6.642139, ∞): r2 P5 |
| G1_P0 | all12 | 1Q | (0.000000, 478.054875): r1 P5; (478.054875, 6,325.855868): r2 P3+tail; (6,325.855868, ∞): r2 P5 |
| G1_P0 | primary_equal_r_choice | T | (0.000000, 191.794073): r1 P5; (191.794073, ∞): r2 P5 |
| G1_P0 | primary_equal_r_choice | CX | (0.000000, 2.027737): r1 P5; (2.027737, 6.642139): r2 partial; (6.642139, ∞): r2 P5 |
| G1_P0 | primary_equal_r_choice | 1Q | (0.000000, 516.085962): r1 P5; (516.085962, ∞): r2 P5 |
| G0_P1 | all12 | T | (0.000000, 2,412.584862): r2 P3+tail; (2,412.584862, ∞): r2 P5 |
| G0_P1 | all12 | CX | (0.000000, 6.332293): r2 CTS; (6.332293, 6.637185): r2 partial; (6.637185, ∞): r2 P5 |
| G0_P1 | all12 | 1Q | (0.000000, 14,642.825178): r2 P3+tail; (14,642.825178, ∞): r2 P5 |
| G0_P1 | primary_equal_r_choice | T | (0.000000, 0.578626): r1 P5; (0.578626, ∞): r2 P5 |
| G0_P1 | primary_equal_r_choice | CX | (0.000000, 6.637185): r2 partial; (6.637185, ∞): r2 P5 |
| G0_P1 | primary_equal_r_choice | 1Q | (0.000000, 16.067869): r1 partial; (16.067869, ∞): r2 P5 |
| G1_P1 | all12 | T | (0.000000, 15.097987): r1 P3+tail; (15.097987, 212.610441): r1 P5; (212.610441, 2,414.674394): r2 P3+tail; (2,414.674394, ∞): r2 P5 |
| G1_P1 | all12 | CX | (0.000000, 2.180548): r1 CTS; (2.180548, 6.332387): r2 CTS; (6.332387, 6.642139): r2 partial; (6.642139, ∞): r2 P5 |
| G1_P1 | all12 | 1Q | (0.000000, 223.985033): r1 partial; (223.985033, 500.670225): r1 P5; (500.670225, 14,654.651995): r2 P3+tail; (14,654.651995, ∞): r2 P5 |
| G1_P1 | primary_equal_r_choice | T | (0.000000, 226.931531): r1 P5; (226.931531, ∞): r2 P5 |
| G1_P1 | primary_equal_r_choice | CX | (0.000000, 2.027737): r1 P5; (2.027737, 6.642139): r2 partial; (6.642139, ∞): r2 P5 |
| G1_P1 | primary_equal_r_choice | 1Q | (0.000000, 223.985033): r1 partial; (223.985033, 592.720437): r1 P5; (592.720437, ∞): r2 P5 |


全hの既存[projection](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/saved_comparison_projection_v2.json)と
`comparisons/<ledger>.json`のexact endpointを参照。丸め境界で順位判定しない。
G1_P1のP3+tail/r1利益はTのhが約15.098未満の領域で成立する。
primaryだけを見る場合はr1 P5が約226.932まで最小だが、全方式ではP3+tail/r2が先に登場するためprimaryの領域を全方式へ移転しない。

fixed-rのP5−partialのaffineは選択記録へ保存。Tでは全ケース/rでP5が低く、slope差も負なのでh>=0で逆転しない。
CX/r1はP5が低いが、CX/r2はh=0でpartialが低く、P5へ変わる交点はG0で約6.637185、G1で約6.642139。
native1Q/P1/r1はh=0でpartialが低く、P5へ変わる交点はG0で約223.857638、G1で約223.985033。
これら二方式の交点は全六方式の最小領域と同義ではない。

## 何を説明でき、何を取得していないか

保存価格差から全方式/rのT・1Q増価、P3/P5逆転の主要root key、N変化から保証効果とinteractionをexactに説明した。
全法のh領域は既存資料を再利用。脆弱な比較は、G0_P0 T勝者/次点の約0.096%、G1_P1 P3/P5の約0.170%、
P1/r1 key tieの小変位、P0→P1で反転するP5/partial native1Q、P5自身のG0/P0からG0/P1へのr選択である。
固定値の数学的誤差と、合成価格を別取得した場合の不確かさを混同しない。

未取得は、別seed/provider/precisionでのsequence価格、実装可能なtie価格、whole-circuit最適化、
古典online生成費用、実量子shot数、独立validation、DF/分子/一般Hamiltonianへの成立範囲。
追加取得は今回行わない。各ledgerのconfidenceは個別12行familywise0.05で、四ledgerを一つの96axis同時保証と呼ばない。

## 検証・公開とGPTへ返す論点

追加の保存算術同値checks 369件PASS、FAIL0。
9人工unit tests（acceptance込み出現回数、signed価格寄与、shot/単価恒等式、affine/tie、bytes改ざん、未宣言入力、固定commit拒否、有理数保存）がPASS。
これはlocal技術検証で、旧監査の再実行、CI/外部独立再現、新しい科学結果取得ではない。
[検証・scope記録](../../../artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1/validation_and_scope_v1.json)、[証拠manifest](../../../artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1/evidence_manifest_v1.json)。
新しい分析sourceはstdlib保存値専用でscience libraryをimportせず、登録solver/shot budget/native価格kernel/runnerを呼ばない。
公開後に新報告・evidenceのremote bytes/SHA、branch HEAD、worktree cleanと親既存変更保護を照合する。

GPTには、今回の価格依存と小marginからどこまで一般化できるか、誤差保証と表現選択の成立条件をどう主張するか、
追加の独立検証に研究上の情報価値があるか、論文の着地点を判断する材料として返す。
Codexは科学的GO/STOP、アルゴリズム採択、新規性、論文化の十分性を決めない。
GPTの新しい詳細科学レビューはユーザーの明示開始承認後のみ。
**技術分析完了後mandatory STOP。追加の合成・再会計・r/provider探索・compile・pair評価・SP/DF/分子へ進まない。**
