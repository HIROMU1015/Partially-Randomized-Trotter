# O2-SCV：2件の固定pilotから確認できること

2026-11-07 JST。**既報値の引用・静的照合。新しい科学計算、保存bit/rawからの数値再解析、controlの再ランキングなし。**

## 対象・一次資料

両件とも保存linear H4／1.00 Å／STO-3G／legacy DF12／8 system qubits／全4電子sector／L_D=3／m0,q1,δ=.02,r1,K2／coefficient_atol=0.0、X/Y、理想gate。N1 actual finite-grid Q・全5正層・selected full policy、N2保証ground、認証済みexact Fraction engineを使用する。同じ有限複素signalが一次targetで、未近似の分子Hamiltonian、RPE energy精度、全round総費用、実機noiseの評価ではない。

理論基準は元の保存DF Hの厳密ground。実行には保存数値stateと認証engineを使い、理想finite signalとの座標差は既存の厳密な保存上界（表示上界7.8079008e-7）の範囲で区別する。この誤差を0と置いた不偏性保証や、pilotのε/α保証を主張しない。

| 一次資料 | 結果commit | 科学SOURCE | 本書で確認した固定ファイル |
|---|---|---|---|
| Fragment群indicator 224 | `ba483ab3038693048e58830e39a676493a892053` | `996e434e63ce9b1d14c6ae191649c9abe213cf47` | [REPORT][r224]、[RESULT_SUMMARY][j224]、[EXECUTION][e224]、[FINAL_METADATA_AUDIT][a224]、SOURCE・manifest・STOP |
| Signed Support 448 | `a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e` | `f70e555fbfe08676f530d989192aade75a57ab1a` | [REPORT][r448]、[RESULT_SUMMARY][j448]、[EXECUTION][e448]、[SAVED_RECORD_INTEGRITY][a448]、[CLASSICAL_RESOURCE_SUMMARY][c448]、SOURCE・manifest・STOP |
| Signed metadata-only | `b4512675ae1d650858256547d359210361d77f68` | `71778dae6954ed9a7a6821062a2a6e6be17672c4` | [REPORTとexact moment表][meta]、保存manifest/STOP。再実行なし |

数値は上記の固定REPORT/RESULT_SUMMARY等の転記。表示桁を超えるexact Fractionは元JSONを正本とする。2件の学習量・評価quota・control・seed・record・係数は異なる。**pilot間の絶対V̂やscoreの大小から、controlの性能順位を決めない。** 各pilot内のB1-OL/SCVは同じfresh評価recordのpaired比較、pilot間は別campaignの有限標本である。PCG64のstream分離は独立uniform 53-bitモデル下の設計で、数学的なPRNG独立性の証明ではない。

## Controlと取得設計

適用順はproducts→rotation。n=0ではL0、n=2ではL2がrotation。p_h=Q(h)。

| 比較項目 | Fragment群indicator 224 | Signed Support 448 |
|---|---|---|
| 定義 | u=1{F(L_n)∈S}, S={df-fragment-3} | u=s_(L_n)×|diagonal support_(L_n)|/2 |
| 活動層 | h1のみ。h0/2/3/4のλ=0を取得前固定 | 同左。h2–4のmetadata varianceが正でも、このcampaignでは非活動 |
| ν1 exact | `7212243195074438/8558339509381393` | `-1878702282177856/8558339509381393` |
| Σ1 exact | `9708373982775118057365606116290/73245175157838542641857534620449` | `69532480438749347841126879238803/73245175157838542641857534620449` |
| 平均・varianceの扱い | actual Qによる全5層exact表をtrain前固定。u²=u | 保存metadata結果のν/m₂/Σをtyped復元。momentの再計算なし。u²=uとは置かない |
| 学習 | h1 32/軸、計64 | h1 64/軸、計128 |
| fresh評価quota/軸 | (4,64,4,4,4)、80/軸、計160 | (8,128,8,8,8)、160/軸、計320 |
| physical shots | 224 bit/224価格を完備 | 448 bit/448価格を完備 |
| 係数の凍結 | 軸別intercept OLSを各1回、binary64 slopeをFraction化。追加fit/clipなし | 同左。両軸のfit/guard後に評価開始 |
| Rank・range | 両軸rank2、D=2+abs(λ)≤4にPASS | 両軸rank2、D=2+2abs(λ)≤6にPASS |
| Paired診断 | 全5層full評価、2軸。層別variance/covariance/feature count保存 | fullとtrial-ID固定half A/B、各軸3、計6回。half quota=(4,64,4,4,4) |
| Seed | 20261109 | 20261110 |
| 終了 | COMPLETE、mandatory STOP | COMPLETE、mandatory STOP |

Σはmetadataの母分散、下表のV̂はbit/residual標本から既存runnerが報告した**推定平均の分散点指標**で、異なる量である。Σ>0は非定数性を示し、Cov(B,u)≠0や利益を示さない。[metadata結果][meta]

## 軸別係数と点指標の転記

| 項目 | 224 X | 224 Y | 448 X | 448 Y |
|---|---:|---:|---:|---:|
| λ1（報告の表示近似） | 1.01641576466e-16 | 0.0666666666667 | -8.56211664316e-17 | -0.0263157894737 |
| D（報告の表示近似） | 2 | 2.06666666667 | 2 | 2.05263157895 |
| B1-OL平均 | 0.99993027462 | 0.0593507420493 | 0.999946435558 | -0.0766145415653 |
| SCV平均 | 0.99993027462 | 0.0533470637 | 0.999946435558 | -0.0746420262728 |
| V̂_OL | 1.39572692233e-09 | 0.0151001791303 | 5.60860020945e-10 | 0.00737790271621 |
| V̂_SCV | 表示精度で同じ | 0.0150964477577 | 表示精度で同じ | 0.00740702201167 |
| 追加改善の既報解釈 | 実質的な点改善なし | 約0.024711%の微小点低下 | δ=-5.01919613261e-35、点改善なし | δ=-2.91192954577e-05、点指標増加 |
| Paired情報 | Covhat=1.39572692233e-09、V̂(平均差)=1.25765614978e-35 | Covhat=0.0150964477577、V̂(平均差)=3.7313725623e-06 | active h1退化/疎flag。full/half A/Bのδは負 | full/half A/Bのδは負。flagなし、有意性は未評価 |

出典：[224 REPORT][r224]/[summary][j224]、[448 REPORT][r448]/[summary][j448]。δ=V̂_OL−V̂_SCV。halfのscaled δ/spreadは元448 JSONに保存されている。本書で再計算・検定していない。

Exact λは224 X=`8246144346576529/81129638414606681695789005144064`、Y=`2401919801264255/36028797018963968`。448 X=`-6946414273234819/81129638414606681695789005144064`、Y=`-1896252474682317/72057594037927936`。微小なX係数は非零のまま保存され、actual幅も厳密には2より大きい。表示の2をexact幅と読まない。

224のtrain feature出現はX u=0/1が4/28、Yが2/30。224 Y評価のu=0は4件。両pilotのXでraw sample variance0だったのは**h1**であり、全targetのV̂は上表の非零値。母分散0とは結論しない。rank2は回帰が可能という検査で、係数の精度保証ではない。

Paired covarianceとV̂(平均差)は、variance削減量そのもののCIではない。448のhalf spreadもCI/SEではなく、Yのflagが空でも有意差を示さない。448の`full.range_guard=NOT_ADOPTED`は元汎用serializerの項目で、campaignでguard未検査という意味ではない。実際のsigned_guard/固定仕様は幅6で、両軸PASSを保存している。[448 SOURCE][driver]

## 学習費込みの実現compiled RZ

価格profileは共通：Qiskit1.3.0、basis=(rz,sx,x,cx)、optimization level2、seed_transpiler17、selected full no-prep。**状態準備だけを除外**。cache hitでも毎shotの量子RZを全額計上する。

| pilot・軸 | train RZ | 共通eval＝B1-OL参照 RZ | train+eval＝SCV参照／physical RZ |
|---|---:|---:|---:|
| 224 X | 153,545 | 386,274 | 539,819 |
| 224 Y | 153,445 | 385,977 | 539,422 |
| 224 合計 | 306,990 | 772,251 | **1,079,241** |
| 448 X | 306,561 | 771,512 | 1,078,073 |
| 448 Y | 306,789 | 771,943 | 1,078,732 |
| 448 合計 | 613,350 | 1,543,455 | **2,156,805** |

| pilot・軸 | B1-OL：V̂×C_eval | SCV：V̂×(C_train+C_eval) |
|---|---:|---:|
| 224 X | 0.000539133021195 | 0.000753439911483 |
| 224 Y | 5828.32184018 | 8143.35604237 |
| 448 X | 0.000432710236479 | 0.00060464804536 |
| 448 Y | 5695.32035646 | 7990.1916687 |

上記は元REPORTの実現費・探索的point score。参照2費用は共通評価を含むため足して物理支出にしない。両件とも学習費を含めたscoreは両軸で不利だった。**期待RZ改善、保証付き必要shot数、最適配分baselineへの優劣を示す指標ではない。**

古典負荷の既報：224は親wall1455.440秒（約24.26分）、127 workers、親peak RSS510,152,704 bytes。448は親wall3965.35秒（約66.09分）、63 workers、親peak RSS1,301,532,672 bytes。[224 REPORT][r224]と[448資源記録][c448]に基づく。環境・並列度・記録処理も異なるため、control自体の速度比と解釈しない。nested timerや各worker RSSの加算をcampaign wall/同時物理RAMへ置き換えない。

## 言えること・言えないこと

| 区分 | 現在の主張 |
|---|---|
| CONFIRMED | 2件ともexact中心化、独立train/freeze/fresh paired評価、全5層・rank/range・bit/価格・全費用・STOPを接続し完了した |
| SUPPORTED_BUT_LIMITED | この具体的control・凍結係数・quotaで、学習費込み改善を支持する明確な点材料は得られなかった |
| HYPOTHESIS | 弱いmetadata相関、Bornノイズの残存、有限学習誤差が利益を小さくした可能性。個別原因の寄与は未同定 |
| UNKNOWN | 真のCov・oracle係数・母分散差・効果のCI、学習policyの期待性能、十分N、期待単価/全RPE費用、他群・他層・他roundの効果 |
| NOT_ADOPTED | 現2候補の一区切り・SCV探索一時停止・Product一致への移行。今回の提案は正式方針ではない |

UNKNOWNの一部は将来の独立検証で問えるが、同じ係数のfresh評価、別の独立train、保証付きcal/productionは異なる問いである。[要因整理](SCV_NONIMPROVEMENT_HYPOTHESES.md)と[方向比較](SCV_RESEARCH_DIRECTION_OPTIONS.md)を参照する。今回、新しい数値を取得せず、旧STOPを維持する。

[r224]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/REPORT.md
[j224]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/RESULT_SUMMARY.json
[e224]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/EXECUTION.md
[a224]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/FINAL_METADATA_AUDIT.json
[r448]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e/docs/research/scv_signed_support_448_science_20261107/REPORT.md
[j448]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e/docs/research/scv_signed_support_448_science_20261107/RESULT_SUMMARY.json
[e448]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e/docs/research/scv_signed_support_448_science_20261107/EXECUTION.md
[a448]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e/docs/research/scv_signed_support_448_science_20261107/SAVED_RECORD_INTEGRITY.json
[c448]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a7b70ffb6a39a3af4dbbf438f35f959e9cafe22e/docs/research/scv_signed_support_448_science_20261107/CLASSICAL_RESOURCE_SUMMARY.json
[meta]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4512675ae1d650858256547d359210361d77f68/docs/research/scv_signed_support_h4_metadata_science_20261107/REPORT.md
[driver]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f70e555fbfe08676f530d989192aade75a57ab1a/src/trottertracks/algorithm_codesign/scv_signed_pilot.py
