# RPE m17：行別丸め・signed energy・全prefixの人工adapter

2026-11-04 JST。**宣言された有限人工領域について、全supportの行別残差・signed energy・
全prefix・最終signal・guardを接続した。新18＋再利用28、計46 tests PASS。**
現行kernelを変更せず、人工系でguardの証明改善を実現した。
実H4の到達集合／不変領域は未構成であり、H4用科学SOURCEは準備していない。

基点は`e16c56d13f21b9b67f0b6ad27264139346ada31a`の
[調査](../rpe_m17_guard_proof_refinement_preflight_20261104/TECHNICAL_FINDINGS.md)・
[計画](../rpe_m17_guard_proof_refinement_preflight_20261104/MINIMAL_VALIDATION_PLAN.md)。
固定研究対象はlinear H4 / 隣接1.00 Å / STO-3G / 保存DF12 / B raw physical state /
4電子sector70、Partial L_D3 / δ=.02 / m17 q131072 r32 K2。
保存N/α/β/ε・extract_identity_phase・coefficient_atol=1e-12・kernel/FP条件・guard閾値を保全する。
以下の人工2次元行列・state・trig値は技術fixtureであり、H4の科学条件を変更した実験ではない。

## 実装と再利用

[row_energy_adapter.py](row_energy_adapter.py)を追加した。file/data loader、RNG、科学CLI、出力writerはない。
旧SOURCEは変更せず、[energy_prototype](../rpe_m17_guard_proof_refinement_preflight_20261104/energy_prototype.py)の
real_row_error/energy_step、KernelBounds/path_bounds、ball_adapterの上向きsqrt、
array検査、finite envelope、complex_abs_error、guard_completion、classify_axesを再利用する。

| 接続部分 | 内容 |
|---|---|
| GEMVの実row | exact dyadicな積和と実出力残差を行ごとに保持し、導出済みforestの各座標上界へ照合 |
| Exact構造 | zero row、単一±1/±i係数のsign/swapを区別。推測で残差を0にしない |
| Energy | H=Iでdrift、signed radial、二次誤差をexact rationalとして分離 |
| Rotation | P@v、c*v、(i*s)*(P@v)、subtractを元の順序で評価し、各丸めと内部prefixを保持 |
| 全support | 宣言された有限state集合で全primitiveの閉性を確認。正のevent massを全て含める |
| 長列 | endpointと全prefix最大energyを保持するrelationを合成し、r/qをbinary powerで扱う |
| 測定 | 全到達endpointの実vdotとrow残差を検査し、signal magnitude・abs誤差・座標guardへ接続 |
| 平均 | 旧作用誤差を独立に受け取る。欠落ならUをnullに保ち、norm証明で置換しない |

既存のNormAffineは任意のnorm envelope用の人工prototype。
今回はendpointの相関を失わないよう、同じ合成順序を有限relationへ拡張した。
これはq/r trajectoryのsamplingや列挙ではない。有限domainの各入力についてprimitiveの全遷移を検査し、
正のmassを持つ全event列をrelationの和集合と合成で覆う。

## 行別誤差とsigned energy

保存binary64を厳密dyadicとして解釈し、各実rowについて

    |e_re| <= γ_D Σ(|ar*xr|+|ai*xi|) + Jη/(1−Du)
    |e_im| <= γ_D Σ(|ar*xi|+|ai*xr|) + Jη/(1−Du)

を用いる。旧係数の`mv_rel/2`と`dot_abs/2`を再利用し、underflow項はFractionで保持する。
Fractionは残差の証明計算だけに用い、state-action/rotation/vdotは元と同じNumPyのbinary64/complex128演算である。
Frobenius normはfinite screenだけに使用し、row誤差を最初から一律gainへ潰さない。
係数を実測誤差からfitしたり、ηをfloat化して0にしたりしない。

各**宣言された入力**で、実際のFP出力y、exactな数値作用Avからe=y−Avを求め、

    ||y||² = ||v||² + (||Av||²−||v||²)
             + 2 Re((Av)†e) + ||e||²

をexact rationalで照合する。radialの符号はこの差から計算し、独立性・無偏性・相殺を仮定しない。
これは有限domainにおける全数検査であり、任意の未知stateへの経験的な誤差外挿ではない。
ゼロ以外のroundoffが生じる一般の人工行列と、最小subnormalの半分へのunderflowも検査した。
後者では非ゼロの加法誤差と負のradialを保持して、出力energy0を得る。

Rotationのexact数値作用T=cI−isPについて、H=Iのdriftを

    (c²+s²−1)||v||² + s²(||Pv||²−||v||²) + 2cs Im(v†Pv)

へ分け、最後の非Hermitian項を含むことを確認する。
runtime丸めはさらにmatvec・二つのscale・subtractの残差を含める。
理想θからのtrig差や位相誤差を平均側から消す処理はない。
正定値metric Hへの拡張は、今回の人工系には不要なので実装していない。

## 閉性、作用順序、prefix、observable

state membershipはcomplex128のraw bytesで判定する。近傍stateのtolerance merge、
signed zeroの同一視、stateのnormalizationは行わない。
人工fixtureには必要なsigned-zeroの全表現を明示的に含める。
閉じない出力を発見したらUNKNOWN_CLOSUREでSTOPし、探索を打ち切って証明済みと扱わない。

eventは`product_idsの順序→rotation→phase`、全列は`F→r events→R`のq反復。
relationのedgeには、endpointに加え、全vector prefixとrotation内部の最大norm²を格納する。
before/afterの合成では各経路のpeakのmaxを取り、同じendpointへの別経路もmaxで合流する。
これにより途中増幅と最終収縮を区別できる。正のrare massも除外しない。
小さい人工列のbrute-force参照とrelation powerを照合した。

主fixtureはF=R=[[0,2],[1/2,0]]とZを使う、非unitary・非Hermitian・非可換な人工系。
64個の宣言stateで閉性を確認し、r32/q131072の全event列を合成した。
条件付きのprefix energy上端は4、最終signal magnitude上端は1となり、既存guardを通せる。
別fixtureでは、同じ中間増幅の後にzero operatorで最終energyを0へ収縮させても、prefix上端4を保持する。
これらは人工値であり、H4の新しい測定・norm・guard評価ではない。

最終signalには実vdotを使い、そのexact残差を同じforestへ照合する。
endpointとobservableの相関を保つため、途中のnorm上端だけからsignalを覆う必要はない。
旧complex abs誤差と両座標guard、finite screenの条件を維持する。

## Runtime条件と保証の限界

数学的な前提は、既存binary/dispatch/profile、全参加threadの安定したFP mode、
および**同じ入力に対するdispatch/partition/reduction graphの再現性**である。
有限transition tableの再利用には、誤差normの上界だけでなく、この決定性が必要。
異なるgraphも許すなら、その全出力をrelationへ包含し直す必要がある。

人工テストはこれらを明示的な仮定として渡す。実runtimeを観察した結果ではない。
`runtime_live_observed=false`で、fixtureのguard結果は常にCONDITIONAL_ARTIFICIALのscope。
仮定欠落、profile/FP不一致、closure/finite不足ではUNKNOWN_STOP。
この追加の証明前提が実H4の固定runtimeで満たされるとは認定していない。
旧科学契約のruntime条件を緩和・変更するものでもない。

finite envelopeは、宣言domain全体のnorm・全行列norm・scalarから旧SOURCEのno-overflow条件を確認する。
実kernelの内部accumulatorについては旧graphのO(B²)/O(B³)導出を使う。
phase/scale/subtractの一時vectorもprefix記録へ含める。
complex absの範囲条件が不成立の場合も未判定に保つ。

平均のstrict判定には旧作用誤差、保存logical interval・mean・s・εが独立に必要。
人工callbackで接続を検証したが、これをH4の十分条件と報告しない。
runtime前提を渡しても、トップレベルの`U_virtual_IID_upper`/`U_exec_upper`はnull、
`scientific_status=UNDETERMINED`、`RPE_tolerance_certified=false`を常に保持する。
固定PCG64の数学的IID性や全18round、compiled回路、費用の保証は得られない。

## 科学SOURCEについての判断

人工adapterとしての接続は完了したが、H4の全到達stateを覆う有限domainや不変領域は未構成。
未知stateでsigned radialを負や0とする根拠もない。
H4のsupportやdomainを未検証のまま、この全数検査を科学runnerとして固定することはしない。
**今回、実H4向け科学SOURCE・実行計画は準備しない。**

保存raw stateには既存96 bit norm²区間を再利用できるという前段の確認を維持し、再計算していない。
旧SOURCE/raw/grant/ledger/marker/STOP・他branch・SERVER2 H8・共有環境も変更していない。
新H4数値計算・保存値再解析・sampling・compileは0。

[次段判断と代替方針](NEXT_STAGE_DECISION.md)、[人工検証記録](VALIDATION.txt)を参照。
専用branchへの公開・remote照合後にmandatory STOPする。
