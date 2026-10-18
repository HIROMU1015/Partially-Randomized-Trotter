# 静的構造と人工測定の対応

固定F01 driverは各pair fresh process、保持stateをX/Yで共有。Fullはraw-validならcoin reject前にpacketを払い、子lawの構築/sort・有理root・weightを処理する。canonicalもconditional lawを再構築する。Bでは毎acceptedにF3/canonical audit・source/state fingerprint・元event再query・数学的angleを追加する。これらは静的な呼出順の確認であり、実費の順位ではない。

静的参照：`g7_generator.py` のDeterministicBits/_law/FullReturnGenerator.sample/packet、`return_aggregation.py` のdigital_parent/dyadic_distribution、`g6_f3_generator.py` の_guard/angle_for_event、`g6_f3_precision.py:145` の_atan_endpoint、F01 driverのMeter/FunctionObserver/Stream/run_pair。いずれもimport・実行せず読んだ。

## mixed 人工stage CPU中央値

| Stage | CPU s |
|---|---|
| online.EVENT.angle_and_requery | 0.640362 |
| online.EVENT.evidence | 0.489664 |
| online.EVENT.generator_and_audit | 0.430018 |
| save.serialize | 0.180492 |
| save.write | 0.146952 |
| online.ZERO_R.generator_and_audit | 0.090206 |
| online.ZERO_C.generator_and_audit | 0.085641 |
| save.hash | 0.084983 |
| online.ZERO_R.evidence | 0.075468 |
| online.ZERO_C.evidence | 0.071427 |

## heavy 人工stage CPU中央値

| Stage | CPU s |
|---|---|
| online.EVENT.angle_and_requery | 1.944263 |
| online.EVENT.evidence | 0.088797 |
| online.EVENT.generator_and_audit | 0.088437 |
| save.serialize | 0.043652 |
| save.write | 0.022769 |
| online.ZERO_R.generator_and_audit | 0.021523 |
| online.ZERO_C.generator_and_audit | 0.018212 |
| save.hash | 0.018031 |
| online.ZERO_R.evidence | 0.009746 |
| online.ZERO_C.evidence | 0.007943 |

heavyの人工角度級数が支配的で、GMP/漸化式の効果を確認できた。mixedではevidence/canonical化やgenerator処理も大きく、angleだけを改善しても全体利益は小さい。serializationのmicro改善がend-to-endで希釈されることも確認した。実G7の最大ボトルネックや4方式の科学的な優劣は決めていない。

新prototypeはguardやevent再queryを省いていない元run_pairの骨格を使うが、人工state hashは実F3の全SOURCEファイル照合ではない。人工packet/lawもReturnKernelと同じ計算ではなく、zero率・operand幅は設計値。実SOURCEのCPU profilerや数値再解析を実施していない。
