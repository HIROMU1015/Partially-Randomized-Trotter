# 検証済み範囲と全18roundへの不足

2026-11-04 JST。全体 `RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`。
今回の人工PASSをH4科学検証、独立詳細科学レビュー、全18roundの保証に昇格しない。

## 再利用する証拠

| 証拠 | 確認できること | 接続してはいけないこと |
|---|---|---|
| [候補B e22d7718](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e22d7718c47358ff650dbc04e8e72f6ab1c8bf41/docs/research/rpe_fixed_request_measurement_b_20261103/REPORT.md) | H4 linear1Å/STO-3G/legacy DF12、保存physical fingerprint、sector70、δ=.02。Partial L_D3/r1/K2のq1,8×all0/first2とFull L_D12/r0/K0のq1,8。6 request・12軸がatol1e-10内 | 他event、全成分、複数order2、r≥2、q>8、law平均、compiled correctness |
| [V2 78c49d0b](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md) | 同H4/δ=.02、Partial3/Full12の保存18round mean/N/α/β/εの条件付きbranch・成功モデル。両候補CONDITIONAL_SUCCESS | 保存meanが実回路marginalに等しいという証明、実IID、実機、厳密joint成功確率 |
| [m17 34840ce9](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34840ce9bf81d985b32764d9bc9d77645ed52e59/docs/research/rpe_m17_rigorous_mean_enclosure_execution_20261103/REPORT.md) | 同H4/LD3/δ=.02/r32/K2/q131072、normalized saved-state ray、指定SOURCE replay、IID uniform53、ideal exact-parameter gateで両軸categorical十分条件。π64転送を含む | actual controlled circuit内部合成、PCG64固定列、float virtual測定完了、noise/compile/state prep。これらbridgeはnull |
| [全18準備 f4905fff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f4905fff592351d33ba07b927892fb9532c99711/docs/research/rpe_18round_ideal_enclosure_preparation_20261104/ROUND_INPUT_AVAILABILITY.md) | 18保存tuple・input可用性とm0..16用runner、m17再利用 | 実行済み全18結果ではない。進行中の別Codex結果を予測しない |
| [今回人工テスト](ARTIFICIAL_TEST_RESULTS.md) | 2-mode人工行列上のr/K、非可換順序、符号、control両branch、wrapper、分布平均恒等式 | 新H4の確認、全supportの証明、RPE成功認定 |

Bの科学実行SOURCEは `4d7c455db3ed7d3e4bc9e2e1fd4dbeba9f46b272`、
e22d7718は結果commit。Bの強制profileはRTE lawからのsampleではない。
異なるbasis_idを選んだだけで実H4 componentの非可換性を証明したとはしない。
FullのB記録で元Full18round診断の欠測fingerprintを遡及補完しない。

## 静的に確認した重要な対応差

- B runnerの`prepare_df_partial_s2(...)`は`identity_policy`だけを明示し、
  `coefficient_atol`は既定0。今回の1e-12/218component/preparation hashと
  同じproduction preparationを検査したとは言えない。
  [候補A run01の既存停止記録](../rpe_fresh_iid_measurement_a_execution_20261103/REPORT.md)にも
  同じ既定値と保存条件の不整合がある。Bの条件付き回路PASSそのものを取り消す理由ではない。
- 保存signal τ64とmodel τ64は別literal。
  r1では `0.011665408492679006` と `0.011665408492679972`、
  r32では `0.0003645440153962189` と `0.0003645440153962491`。
  いずれも保存値の引用で、新差分・影響評価はしていない。
  `_finite_inputs(preparation,...)`の再利用だけでsignal側のθ/lawと同じとは認定しない。
- 他5r群のorder/CDF/θの過去payloadは固定入力集合に未保存。
  f490 runnerの新SOURCE replayは過去samplerの復元ではない。
  Bを再実行したり浮動小数のclose判定だけでliteralを置換したりしない。
- 本H4 scopeの小さい比較基準1e-10と、長qの保証用包含/π64 boundは別の基準。
  長q数値診断の旧INCONSISTENTやu_num=nullを、今回人工PASSで変更しない。
- boundary最適化のscalar評価 `-δ*q*(c_const+c_I)` と、保存Fのstep phase2個をq回掛ける評価は
  binary64の式が異なる。既存`builder_phase_bound(q,r)`はevent/productのπ64加算用であり、
  この再結合やblock融合・Qiskit gate合成の全誤差の保証ではない。
  差の大きさは今回未計算。長qで当該回路policyへ接続するなら、この転送も固定して確認する必要がある。

## r/q適用の静的判定

`DFPartialS2RepeatedRequest`は正整数qと各stepのr eventを受け取れる。
通常Hadamard builderのq∈{1,2,4}制限にはbenchmark builderが必要。
benchmarkのmaximum=131072というmetadata受理と、回路を実際にbuildして照合済みという事実を分離する。

| r | 保存mとq | Bの条件付き実H4証拠 | SOURCEの利用可能性／不足 |
|---:|---|---|---|
| 1 | m0..7：1,2,4,8,16,32,64,128 | q1/8の指定2profileのみ。cutoff対応差あり | 構造上利用可能。未使用q・複数order2・保存preparationへの接続未確認 |
| 2 | m8..9：256,512 | なし | r event列はAPI適合。signal側θ/lawとの結合・複数event・boundary未確認 |
| 4 | m10：1024 | なし | 同上 |
| 8 | m11..12：2048,4096 | なし | 同上 |
| 16 | m13..14：8192,16384 | なし | 同上 |
| 32 | m15..17：32768,65536,131072 | なし | m17の非列挙平均包含はあるが、controlled合成の実証を代替しない |

原N/α/β/εは[f490固定18行](../rpe_18round_ideal_enclosure_preparation_20261104/fixed_inputs.json)
とV2 CSVを維持し、比較のための再配分や新schedule選択はしない。

## 全18roundの接続に必要なもの

1. 固定1e-12のpreparation・成分・basis・signal側θを、回路へ渡す入力に結び付ける。
   同一固定stateのsector/endianとnormalized-ray規約も必要。
2. r≥2・複数order2・noncommuting順序・negative sign・scalar位相を含む
   選定requestで、回路に独立な参照とcontrol0/1を照合する。
3. 一般qへの合成則を局所同値性から示す。有限のprobe PASSを全trajectoryの証明とはしない。
   `support_run_le_1`の任意run、全component、長qのphase丸めは残る。
4. 各rのcategorical payload、平均演算子、π64転送を別Codexの固定公開結果と結び付ける。
   その全18roundの局所判定が揃っても、回路内部の全support保証が自動で付くわけではない。
5. 実行対象ν_execを保証する場合は、PCG64/IID、float arithmetic/guard、sampling、
   compiled circuitまたは実機の対応bridgeが別途必要。今回のscope外。

V1のFull半径7点不一致、同δのPF control未発見、V2の条件付き判定、
V3長q価格外挿、R1の約4.858%点選好と総合UNDETERMINEDを保持する。
総費用再解析・noise・状態準備へ拡張しない。
