# N3保存帰属v1：保存データ事後評価・既存3証拠再認証結果

2026-10-13 JST。一回実行完了、retry0。**新しい科学判断待ち・Mandatory STOP**。
[明示認可](n3_saved_attribution_v1_execution_authorization_20261013.md)、[凍結契約](n3_saved_attribution_v1_posthoc_contract_20261013.md)、[plan](n3_saved_attribution_v1_execution_plan_20261013.json)に従った。
[一次事後結果](../../artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1/result.json)、[独立scalar/timing監査](../../artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1_audit/independent_saved_scalar_timing_audit.json)、[実行provenance・出力SHA](../../artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1_audit/execution_provenance.json)を保存した。

## 対象と固定条件

保存済み6H：banded/dense_density/two_external_stress × full5/6、active4/5、ν2/3、実一体＋非負density。
一候補一外部mode、候補modeは0とfull−1の2点/H、元ε_proj=.01。化学geometry、basis、DF rank/policy、split L_D、PF delta窓は対象外。
既存run1のkernel/cold各3反復を解析した事後developmentで、確認的な新移送実験ではない。反復を独立入力数へ加算しない。

準備 `239402c8e0825e6751e1ee1c2be313a8606d4087`、数値事後source `a7a9f42d53d5efaca8715bee26edd9f1a0bad999`、認可commit `0974fa114c319599e70988cc802cafdfedff3ea5`。
原科学結果payload `6d8f769b857e50c8d262e7a4e94f8277a8967597`、原generator source `3eba0814e690e95993db60809c4c3c96784b07c0`を維持した。
新しい入力/ε/κ探索/solver/certificate生成/eigensolver/元batch再実行0。216保存mode-path、432保存solver event、696保存候補を解析した。
独立auditは別実装のstdlib有理数算術でコピー値・q/g/box・τ/Z・ℓ/e・保存幅区間・phase残差を照合し、9 source files・40登録filesのSHA維持を確認した。auditで追加PSD再認証0。

## 幅差の算術的な説明と限界

全696保存候補（独立6H、kernel/cold×3反復を含む。kernel反復0の候補116点）で活性項はROOTだった。
\(g=c-U>0,\ q=\kappa U-s,\ b=\max(0,\kappa-g)\) とすると、保存候補は正のrootがboxと0を上回り、幅は \(\delta(g+\delta)=q\) のroot側で決まる。
box項がOCC/同値SDPの保存幅差を支配する例はこのpoolにはない。

kernel反復0の全12 same-mode比較で、U/cは一致し、選択OCCの最終qと保存幅区間は同値SDPより厳密に小さかった。
最終κは一般に異なる。これは最終保存証拠の差を記述し、同じ理想条件族の保証能力が異なることを示さない。
raw proposal、修復前行列/objective、repair shift rは保存されていないため、κ選択・数値品質・有理化・PSD修復への原因別帰属は未判定。
Zの最終traceは照合できるが、rや\(m r\)を一意に逆算しない。理想最適幅や新しいcertificateを計算していない。

SOSの保存式 \(s=\ell-e\)、\(q=(\kappa U-\ell)+e\) は全候補でexactに一致した。ℓは保存された丸め値で、未保存のraw optimumではない。
18 inaccurate eventが集中する `V2_banded_6 / external5` のkernel反復0では、3tierのe/qは約98.9117–98.9165%。
選択されたδ=.005 tierの保存候補は、κ≈.00374131644、κU−ℓ≈.000116161496、e≈.0106049951、最終q≈.0107211566、保存幅上端≈.00637886085。
この最終qでは残差補正が大部分を占める。eには数値proposal・丸め・修復等の影響が混在し、inaccurateというstatusだけに原因を限定しない。
残差を削除した未認証の幅は出していない。同値SDPの保存solver eventにinaccurateは0であり、18件を同値SDP/OCC幅差の原因として使わない。

## 保存タイマと出力義務

以下は元runのcold path（三反復median、U/c取得＋候補生成＋selector＋メモリ内serialization。shared disk IO/worker依存importは含めない）の保存値で、今回のposthoc実行時間ではない。

| 入力 | OCC cold ms | 同値SDP cold ms | SOS cold ms |
|---|---:|---:|---:|
| V2_banded_5 | 33.299 | 69.440 | 1195.192 |
| V2_banded_6 | 57.320 | 111.341 | 5776.770 |
| V2_dense_density_5 | 41.790 | 70.162 | 1603.426 |
| V2_dense_density_6 | 71.176 | 168.523 | 6500.116 |
| V2_two_external_stress_5 | 42.275 | 67.851 | 1211.030 |
| V2_two_external_stress_6 | 71.104 | 124.597 | 4097.345 |

次表は各H cold pathのphase割合medianの範囲。Engineのexclusive phaseを両候補modeで合計し、各反復のpath totalで割った値をmedianにしている。各phaseのmedian合計はpath medianと必ずしも一致しない。

| 保存path | numerical proposal | rationalization/repair | 正証拠checker |
|---|---:|---:|---:|
| OCC | 4.76–7.04% | 24.42–29.90% | 44.36–48.82% |
| SDP_EQUIVALENT | 24.16–51.84% | 14.33–23.70% | 15.65–23.82% |
| DENSITY_SOS | 41.96–65.87% | 18.58–35.63% | 13.44–21.30% |

OCCでは正証拠checkerとrepairが主な保存phase。同値SDPではnumerical proposalにCVXPY solve全体が入り、SOSではbackend solveに加えて大きな修復後Gramとsector checkerを扱う。
これは固定prototypeの測定範囲の説明であり、一般的な算法の速度比較や新しいspeedup倍率ではない。

OCCの出力は正証拠中心、同値SDPは3tierの正証拠に加えて可能なstrict dual反証を作り、SOSは別の強い証拠族と大きなGramを扱う。
今回の同値SDPの各反復H pathには、banded5/6で反証0、dense5/6とtwo_external5で各1、two_external6で3の保存recordがある。
反証取得・混合・検証の専用timerはない。numerical proposal−reported backend solve/setupとrow total−exclusive sumは混合未割当として保存した。今回、これらの負の残差は0件。
「反証を外した費用」「3で割った一回費用」「同じ最終品質での費用」は計算できない。原因を完全に識別した同機能・同品質benchmarkとしては扱わない。

## 保存3証拠の再認証

[凍結3証拠bundle](../../artifacts/n3_saved_attribution_v1/2026-10-13/preparation/SOS_OCC_existing_proof_connection.json)を丸め直さず、[再認証結果](../../artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1/selected_existing_proof_recheck.json)を得た。
`V2_two_external_stress_5`、full5→active4、ν2、一外部mode4、同じU/c、δ=1/400で、SOS正証拠PASSと理想OCC条件へのstrict dual反証PASS。
SOS保存幅上端≈.002389289446 < .0025。他mode0のOCC正証拠もPASS、保存幅上端≈.001988397586 < .0025。
これは同modeでの証拠族分離であり、このHにOCCが追加縮約できないという結果ではない。物理的不能、一般SOSへのOCC優越、追加H縮約は主張しない。
原ε=.01でNORM/OCC/同値SDP/SOSが6/6H縮約、追加H縮約0という旧結果は変わらない。

## 実行費用・証拠状態

Python3.12.3、CPU2 spawn worker・各1 thread、AS1GiB/worker、sampled aggregate RSS6GiB上限。JSONをcaseごと一回decodeし、stdlib Fractionを使用した。
一回のposthoc wall `0.205411824` 秒、sampled aggregate peak RSS `105836544` bytes（約 `100.9` MiB）、samples5・最大sample間隔約.0671秒。
sample間の瞬間peakは未測定。wall/CPU科学capなし。予算違反0、retry0、失敗0。
一次出力 `4034518` bytes、最大file `696155` bytes、run.log0 bytes、file4MiB・総32MiBの範囲内。
選択positive checkerの最大有理bit長はSOS802、他mode OCC647。sourceのbit/operation上限内。旧batchや全1224証拠の監査は再実行しない。

個別audit PASSはlocal evidence。immutable CIや独立外部環境でのscience再現ではない。
全体manifest checkerは過去からの `manifest.relevant_commits[7]` のdescription欠落でFAILのまま。今回scopeの健全性PASSと区別する。

## 判断境界

幅差とtimingに関する観測可能な最終会計を得たが、原因の完全分離・同品質費用・新用途の価値は保存データだけでは未確定。
[claim/観測可能性資料](n3_saved_attribution_v1_claim_and_observability_20261013.md)と[GPT科学判断引継ぎ](n3_saved_attribution_v1_results_review_handoff_20261013.md)にまとめた。
文献・既知理論の確認限界は前準備のまま。新しい全文確認・先行性網羅・世界初の主張はない。
認可消費済み、N1停止/N2保留、旧primary/gate/STOP保持、中心テーマnull、次段未認可。
研究を継続・縮小・技術記録化する判断、同機能benchmark、実分子や多外部modeへの拡張はCodex内で採択しない。次は新結果をまとめたGPT科学判断の開始認可。
