# Track D × PR＋RPE：r16期待回路価格の限定校正

**q=2・4・8で各4 pathの有限paired価格校正を完了。** 保存q4の1本・4 wrapperを再利用し、新規11 path・52 occurrence・832 event、44 P0 compile/postcheckを取得した。全raw/post/paired χはPASS。これは短qの探索的有限標本で、母平均・長q価格・全13round総資源の保証ではない。

## 条件・SOURCE・事前固定

Linear H4／1.00Å／STO-3G／Nα=Nβ2・元H ground仮定／準備費除外、number-CDF8／元frame順LD2／S2/s1/r16/K0／h=.1。中央complete-step cutはq2/4/8で1/2/4、whole-path一gadget、Strongの既存Gaussian/directional/head relative/tail共有/step境界融合とReference A/B内部最適化を維持。Referenceの16 CSWAP・χ・追加register・cut損失を含む。C1追加融合OFF。

実行SOURCE `890bd13337066c2f263a6a14dd8107df63ca768a`、基点は前回r16結果`e6c6edac084943843b66c87c7899afee327c51c1`。同じr16研究branchを再利用。新sampling／価格観測の前に [CALIBRATION_PLAN](CALIBRATION_PLAN.md) と全seedの [sampling_plan.json](sampling_plan.json) をcommit固定した。旧r16/r32 SOURCE・原本・結果・STOPは無変更。

2水準だけではaffineの残差が識別できないためq2/4にq8を加えた。n4は分散自由度3の初期pilotとして選び、母平均精度の最適nとは扱わない。保存r32のpaired変動は計画の参考とし、r16へ分布を転用しなかった。q16/32は新規実行していない。

## 結果（X、Yは同一価格と確認した場合に限り共通）

|q|資源|Strong平均|Reference平均|paired差平均|paired SD|
|---:|---|---:|---:|---:|---:|
|2|rz_count|6,377.5|6,750|372.5|0.577|
|2|cx_count|2,662.5|2,486.5|-176|0|
|2|rz_depth|1,509.25|859.75|-649.5|87.065|
|2|cx_depth|1,348.25|591|-757.25|46.557|
|2|total_depth|3,639.25|1,928.5|-1,710.75|189.467|
|2|circuit_size|13,088.5|13,477|388.5|0.577|
|2|qubit_count|9|17|8|0|
|4|rz_count|12,920.25|13,208.5|288.25|2.5|
|4|cx_count|5,388.5|4,828.5|-560|0|
|4|rz_depth|3,103|1,586.25|-1,516.75|98.885|
|4|cx_depth|2,740.75|1,066.5|-1,674.25|55.066|
|4|total_depth|7,468.25|3,553.5|-3,914.75|212.813|
|4|circuit_size|26,598.25|26,518.5|-79.75|2.5|
|4|qubit_count|9|17|8|0|
|8|rz_count|25,623.75|25,745.5|121.75|2.63|
|8|cx_count|10,712|9,384|-1,328|0|
|8|rz_depth|6,141.25|3,026.75|-3,114.5|256.526|
|8|cx_depth|5,450.75|2,005.5|-3,445.25|139.242|
|8|total_depth|14,800.5|6,767.25|-8,033.25|563.26|
|8|circuit_size|52,779.25|51,765|-1,014.25|2.63|
|8|qubit_count|9|17|8|0|

全軸・7資源の平均/SDとr32参考比較は [PAIRED_PRICE_RESULTS](PAIRED_PRICE_RESULTS.md)。28変量と14変量の共分散・相関をJSONに保持し、X/Yを標本数に二重計上していない。変動0資源も全supportの母分散0とはしない。

機械可読：[全48価格](../../../artifacts/track_d_r16_price_calibration/2026-11-10/run01/prices.json)、[有限統計・共分散](../../../artifacts/track_d_r16_price_calibration/2026-11-10/run01/paired_statistics.json)、[SOURCE](../../../artifacts/track_d_r16_price_calibration/2026-11-10/run01/SOURCE.json)、[完了結果](../../../artifacts/track_d_r16_price_calibration/2026-11-10/run01/result.json)、[原本hash索引](../../../artifacts/track_d_r16_price_calibration/2026-11-10/run01/payload_index.json)、[STOP](../../../artifacts/track_d_r16_price_calibration/2026-11-10/run01/STOP.json)。

## 再利用・正確性・実行状態

既存canonical sampler、r16 law、C3 prepare_steps/StrongBuilder、C6a/B2 CAR、r16 P0 compile/postcheckとpaired χ関数、既存batched Aerを再利用した。C6a/C6bの固定r32 cohort guardは変更せず、新集合のbinding/統計adapterを追加した。describe/covarianceの算術は既存と同じ。既存SOURCEへの変更はない。

P0=Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free、既存SymEngine MPFR160宣言global phase保持。raw全256入力、AB=V、actual adjoint、vacuum ray/χ、signed event/scalar/identity、complex coherence/X/Y observable、compiled対CAR/rawとpairedを原precision（component1e−10、observable Frobenius2e−10、rtol0、phase fitなし）で検査。科学retry・resampling・threshold緩和0。

実測wall timeは897.06秒。実空きCPU/RAMに従いraw11並列、qに応じたstate-action threadsとCPU重み付きcompile/post dispatchを使用した。共有環境設定や他ユーザーprocessには変更を加えていない。

人工回帰15件PASS。sampling原本・raw/compiled QPY・参照NPZ・内部logはlocal限定で、公開は科学結果・SOURCE・hash索引・STOP。前後SOURCE/inputと旧r16原本を照合済み。

## N・長q・総費用の限界

旧r16 S1は全13roundで条件付き十分条件PASSのまま保存・再利用した。Nはq8で513/軸（r32保存512）、q4096で1359/軸（910）。新しいPF/RTE誤差やNは計算していない。r変更による単価差を同一r内のReference効果と分け、短qでの単価低下だけで総費用優位としない。

今回のq8までの有限平均からq1024〜4096の価格や13round共同誤差を保証できない。r32 A/F/S係数・固定総値を変更せず、新しい長qモデルfit/登録・全round価格表・総資源集計も行っていない。新A Q、Closed P3、α最適化、ground solve、DF fit、詳細科学レビューは0。

## q依存とr32との共通点・相違点

各qの保存4 pathすべてでCX・三depthはReferenceが低かった。paired CX平均はq2/4/8で−176／−560／−1328、標本SDは0。今回の12本では`208−192q`に一致する。これは観測後の有限値の再記述で、r16全lawの式や新しい長qモデル登録ではない。r32の保存88本で観測された`208−224q`と傾きが異なり、r32式をr16へ転用する根拠にはならない。

total depthのpaired平均は−1710.75／−3914.75／−8033.25で、削減幅はqとともに大きくなった。一方、SDは189.47／212.81／563.26でpath依存が残る。r32と同様、countの単純な傾向とdepthのlane/basis変動を分けて扱う必要がある。3水準・n4から非線形項や長qの係数を識別したとはしない。

RZは全3qで増加し、paired平均は+372.5／+288.25／+121.75。追加幅は常に+8（9→17）。sizeはq2で+388.5と不利、q4で−79.75、q8で−1014.25であった。したがって全資源で一様に優れるとの結論にはならない。特にr32 q8のRZ差保存平均−16.90625に対し、r16 q8はまだ正で、同じqでもトレードオフが異なる。

同一pathの両方式は強く共変動する。例えばq4の絶対RZ SDはStrong526.58、Reference525.62に対し、paired RZ SDは2.5である。paired設計の有用性を示す有限観測だが、n4共分散のrank≤3から母集団相関を精密推定したとはしない。

r16固有の価格を検討する技術的価値はある。短qのCX/depth削減と、r32保存群より低いr16の有限平均単価が観測され、N増加を踏まえた全体比較の候補になったためである。ただしRZ・size・幅の不利も残り、長qの期待単価が未検証である。次の最小候補はq16の独立4 pathで、候補式と評価条件をその価格観測前に固定する。これを正式GOや全RPE優位の採択とはしない。

`prices.json`のreused行のsourceは結果carrier `e6c6edac084943843b66c87c7899afee327c51c1`を表し、旧compile実行SOURCEは`08f15308ef16429da7e375b1f82e7799510d2578`である。新規44 compileの実行SOURCEは`890bd13337066c2f263a6a14dd8107df63ca768a`。旧SOURCEと原本hashの対応は`SOURCE.json`の保存bindingから追跡できる。

技術的な傾向と次段階の判断材料は [NEXT_EXECUTION_PROPOSAL](NEXT_EXECUTION_PROPOSAL.md)。正式な研究GOやモデル採択はユーザー／GPTへ戻す。

成果を既存r16専用branchへcommit/pushし、remote SHA/tree/公開bytesを照合する。最終SHAと照合記録はチャットとlocal publication記録に残す。公開照合後mandatory STOP。
