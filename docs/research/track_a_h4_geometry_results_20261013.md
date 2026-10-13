# H4 geometry signal/compile map 結果要約（2026-10-13 JST）

全6距離・各218候補、計1,308候補と74,784 logical wrappersが `MAP_COMPLETE_STOP` で完了した。driver・observer・4 workersの全PIDは `/proc` に存在せず、次stageへは進んでいない。

研究上の主な観測は、保存済み302精度点（ε=0.005〜0.1、ε=0.05を含む）と6距離の全1,812比較点で、部分ランダム化B2が測定込みRZコストの点推定最小だったことである。比較した登録候補集合・保存された有限標本・今回のcompiler/circuit構成・状態準備cost P=0に限る。

条件はH4 linear、STO-3G、DF fragment count12、8 system qubits＋ancilla1、T=0.8、二次DF-prefix PF / canonical finite RTE。距離0.70/0.80/0.90/1.10/1.40/1.60 Å、L_D=0/3/4/5/6/9/12、q=1/2/4/8（δ=T/q=0.8/0.4/0.2/0.1）。B2/B3は32 paired trajectories、deterministic B0/B1は1 paired circuit。B0は残差discard、B1はDF12の完全決定論、B2は決定論prefix＋finite RTE、B3はrandom-dominant endpoint。

εはcomplex-signalの誤差要求であり、エネルギー誤差Haではない。cosine/sine各軸α=0.025の解析的shot数と、状態準備を除いたfull measured Hadamard wrapperのcompiled RZ期待値を掛けて足した値を比較する。量子backendのshot実行は行っていない。

ε=0.05では下記のとおり。表のB2は全てL_D=3、決定論比較の最小は全てB0 L_D=5・q=1（PM1追加候補）だった。コスト単位は百万RZ×shot。

| 距離 Å | B2点最小 q/r/K | B2コスト | 決定論最小コスト | 決定論比の削減 |
|---:|---|---:|---:|---:|
| 0.70 | q2-r4-K2 | 206.51 | 286.25 | 27.9% |
| 0.80 | q1-r4-K4 | 171.34 | 260.06 | 34.1% |
| 0.90 | q1-r4-K4 | 134.06 | 231.98 | 42.2% |
| 1.10 | q1-r4-K2 | 112.91 | 197.00 | 42.7% |
| 1.40 | q1-r8-K4 | 102.93 | 159.90 | 35.6% |
| 1.60 | q1-r8-K4 | 97.28 | 143.23 | 32.1% |

ε=0.05でのB2の削減は27.9〜42.7%。0.70 Åではq=2（δ=0.4）、他の5距離ではq=1（δ=0.8）が点最小。1.40/1.60 Åではr=8が選ばれ、それより短い距離ではr=4だった。ε=0.005ではq=2/4のB2が最小で、決定論最小比58.1〜60.4%低かった。許す誤差によってq/r/Kの最小点が変わる。

近いB2候補同士にはpaired MC costの±2SE engineering intervalが重なる。ε=0.05の各距離で点最小と区間の重なる競合は2〜4件あり、特定のr/Kを厳密winnerと認定しない。このintervalはfinite sampleのcost平均の診断であり、厳密な同時信頼保証やcompiler/scientific modelの全不確かさを覆うものではない。

予算上の消費・予約compile呼出は73,023 / 74,805件。保存された固有compile結果は73,018件、candidate内の合法なcache reuseは1,766件。過去の未完了予約5件の費用を返却せず、logical wrappers全件のCOMPLETEと両立している。resume02へ29,388完成wrapperを引き継ぎ、残り45,396 logical wrappersを追加した。

記録された累積wallは74.53時間（約74時間32分）、resume02分は45.40時間（約45時間24分）。これには契約上のcarryが含まれ、CPU時間の総和ではない。利用者承認による72h STOP解除後の完了である。

累積の保守的chargeは13,615,088,338 bytes＝12.68 GiB / 17 GiB。resume02のoutput directoryは内容540.66 MiB、directoryを含む実割当1.13 GiB、195,174 files。過去runの別directory、入力NPZ、worktreesはこの実容量に含めない。保守的chargeと実容量は別量である。

今回の事後照合はsaved JSONとfile metadataだけを読む単一processで実施。全1,308 signalのgeometry/template coverage、全74,784 recordのCOMPLETE・識別field・trajectory/axis・sample weight・paired compiled metricsを照合し、cache-ownerのfingerprint/metrics等を照合した。ε=0.005/0.05/0.1の3,924 candidate pointsではshot・work・SEを独立したstdlib計算で再現した。凍結科学arrayの読込、新signal/trajectory/build/compile/transpile、GPU、共有環境・venv・他job変更は0。

これは保存値からの事後要約であり、独立最終scientific review、外部再現、immutable CIではない。新しいHamiltonian、compiler、gate合成、ノイズ、状態準備P>0、エネルギー/RPE全体やH6へ一般化しない。structured Gaussian Givens cost系列の結果で、旧dense Gaussianや旧compilerのcost数値と直接混合しない。旧compilerとのoutput完全同一性は未検証のまま保持する。研究判断は自動更新せず、next_stage_authorized=false。

実行SOURCE：`d0726e946f89f8c30a08d383fb8ff48b363629ec`。dataset/seed identity SOURCE：`5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`。固定REVIEW artifact：`81c78b366d2cf2c699bb31977f72ae9deab4f1d4`。

公開資料は[結果bundle](../../artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/README.md)、[summary.json](../../artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/summary.json)、[18比較行CSV](../../artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/precision_comparison.csv)。全1,308候補のbias・normalization・paired RZ cost標本と6指標の軸別平均を、[軽量保存値projection](../../artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/saved_candidates_v1.json)へ切り出した。公開用[集計照合script](../../artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/verify_saved_result_projection.py)はstdlib-onlyで、公開JSONだけから395,016候補精度点の費用と1,812点最小、18比較行を照合する。新しい物理計算の再現ではない。実行時の固定worktree・SOURCE・runtime原本は保持し、別の結果公開branchへ軽量資料のみを収録した。
