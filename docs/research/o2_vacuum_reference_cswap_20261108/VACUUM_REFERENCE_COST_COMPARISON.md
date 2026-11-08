# 真空参照CSWAPの人工full-wrapper資源比較

2026-11-08。数値は人工1–3 system qubitsの発明したgate列だけで、分子・geometry・basis・DF rank・L_D・deltaを持つ科学入力ではない。実H4・全RPEの価格や優位性を示さない。

## 公平な比較条件

既存 _build_from_operations の diagonal_only を使い、basis gateを制御せず中央RZ/RZZとscalar phaseだけを制御する。changing-basis 7 pathでは固定gate順序を保持し、全方式を同じ最適化へ渡す。弱い全gate-controlled対照は使わない。隣接inverse/境界最適化はcompilerに許し、共通basisを持つ追加人工例では既存 build_conjugated_pauli_sequence の明示fusionに加え、同じsupportの可換回転角を有理数で合計した強いcontrolled対照を採用した。SE側にも同じ群化を認めた。

既存 CompilerProfile / compiled_wrapper / compile_cache_key / matrix_gate を再利用。P0はQiskit1.3.0、basis=(rz,sx,x,cx)、optimization level1、seed17、approximation_degree1、qubits_initially_zero=False。古いA1科学profileのlevel2とは異なる、全方式共通の人工比較profile。最適価格の証明ではない。既存のphaseを保つdefinition展開も全方式に同一適用し、shared equivalence libraryは変更しない。

全結合に加えて、同じnative circuitへ同じlevel1/seed17のline routingを適用。identity initial layout、trivial layout、sabre routingで、最終permutationを追跡してcorrectnessを検査した。qubit再配置の有利な探索はしない。読み出しはsignal 1 bit。候補・SEはreference n個のreset、cat版はさらにn−1個のresetを明示し、depth/sizeへ含める。共通target state準備だけを除外する。

基本SEと候補のfull IR hashが各cellで一致することをassertし、同じcacheを使用した。cat版の追加fanoutも既知SEの最適化として比較した。新規性だけを理由に価格改善経路を棄却しない一方、候補がSEに対して独立した増分利益を持つとはしない。

## 人工結果

以下はX軸の「controlled / 非fanout参照候補」。Y軸、RZ/CX depth、全gate count、reset/measure、cat版の全結果は機械可読artifactにある。候補と既知SEは同じ数値。

| system n | block数 | 接続 | RZ count | CX count | total depth |
|---|---|---|---|---|---|
| 1 | 2 | 全結合 | 9 / 26 | 4 / 16 | 14 / 35 |
| 1 | 2 | 固定line | 9 / 26 | 4 / 25 | 14 / 50 |
| 2 | 2 | 全結合 | 54 / 80 | 24 / 42 | 68 / 75 |
| 2 | 2 | 固定line | 54 / 80 | 39 / 102 | 88 / 148 |
| 2 | 12 | 全結合 | 283 / 228 | 144 / 92 | 387 / 173 |
| 2 | 12 | 固定line | 283 / 228 | 219 / 152 | 492 / 236 |
| 2 | 36 | 全結合 | 839 / 588 | 432 / 212 | 1159 / 421 |
| 2 | 36 | 固定line | 839 / 588 | 651 / 272 | 1468 / 484 |
| 3 | 2 | 全結合 | 80 / 130 | 32 / 66 | 104 / 101 |
| 3 | 2 | 固定line | 80 / 130 | 56 / 151 | 126 / 199 |
| 3 | 12 | 全結合 | 385 / 354 | 176 / 140 | 535 / 249 |
| 3 | 12 | 固定line | 385 / 354 | 323 / 237 | 661 / 356 |
| 3 | 36 | 全結合 | 1137 / 910 | 528 / 324 | 1595 / 625 |
| 3 | 36 | 固定line | 1137 / 910 | 963 / 456 | 1973 / 739 |

長くchanging-basisを繰り返す人工例では、制御中央回転の削減がCSWAPを上回り、二laneでdepthも減った。短いpathではgate数が増えた。n3/2blockでは全結合depth104→101でも、CX32→66・RZ80→130と増加するため、一つの資源だけで総合優位とはしない。lineではdepthも126→199と悪化する。

明示的共通basis融合の追加2-qubit/12block例は、中央rotationを3個へまとめた。X/Yとも全結合でcontrolled→参照はRZ43→86、CX22→44、depth54→75、固定lineではRZ43→86、CX40→104、depth79→138。長いという理由だけでは有利にならず、強いfusionが効くpathでは参照が負ける。

cat版は2n+1から3n qubitsへ増え、fanout prepare/unprepareは2(n−1)CXを必要とする。例えばn3/36blockの全結合depth625→605でもCX324→328、lineではdepth739→1260・CX456→473と悪化した。routingを含めるとfanoutが常に得ではない。どの最適化も既知SEへ同等に認めている。

独立FredkinのP0分解は8 CX、10 RZ、2 SX、CX depth8、RZ depth7、total depth15。全wrapperでは周辺との融合があり、2n倍を単純加算した値を価格とはしない。T countはUNKNOWN：Clifford+T合成とrotation誤差配分を定義していない。RZ countをT countへ読み替えない。

## Correctness・実行負荷

人工12件＋関連A2 2件=14 PASS。任意targetに対する許容入力isometryの全basis column、wire permutation、真空/非ゼロ位相、noncommuting L/R、scalar/event phase、X/Y符号、非自明なdyadic Q、既知SE同一性、coherent QPE反例と共通固有基底対照を含む。全行のcompiled isometry監査はphase fittingを使わず、routing permutationを反映した。

最大compiled entry差は 1.215e-12、許容差2e−10。新gateの全support一様数値認証ではない。correctnessの許容差8e−12も変更しなかった。

主benchmark112行は約30.6秒、fusion追加16行は約0.8秒、process peak RSSは約594 MiB。successful benchmarkでnative47回（Fredkin1回含む）＋routing46回=93 transpile calls。known SEとの同一IR等をcacheしてnative/routing各18 cache hits。cold/warm時間は各rowへ記録し、warmの小さい取得時間をcold compile速度の優位としない。全実行には失敗人工attemptと診断もあり、計99 transpile callsをEXECUTION_HISTORYに分離した。RNG sampling/shot/実H4はすべて0。

初期fixtureのAPI誤り、対称位相反例の読出し相殺、監査側Statevector共有配列のglobal phase変化を修正し、失敗SOURCE・RUN・STOPを保全。ライブラリの科学engineや既存compilerは変更していない。run01/02/03/04のPASSではなく、run05とfusion01を最終人工証拠とする。

総elapsedには回路生成・phase取得・監査を含むが、それぞれのオンライン費用は独立測定していない。古典path生成・phase ledgerの実PR費用、物理reset/readout latency、制御/校正error、参照qubit寿命はUNKNOWN。1–3 qubitのbinary64検査をA100 GPUの性能検証や旧exact Fraction engine高速化としない。

## 改善条件と全RPEへの限界

| 資源 | 必要な条件 |
|---|---|
| RZ/CX count | 制御除去による削減が、二層CSWAP・相対P・失う境界fusion・routing等の追加を上回る |
| depth | L†/Rの並列化利益がCSWAP/fanout/readout/reset/routingのcritical pathを上回る |
| qubits | 原法n+1に対し基本2n+1、cat3nを使用可能で、そのmemory費を許容できる |
| shot数 | 同じQ・fresh state・理想binary分布なら同じ。新誤差/失敗を入れるなら再評価が必要 |
| 期待full費用 | 同じtarget/精度/失敗率の下で E_Q[C_ref] < E_Q[C_ctrl]。Q平均価格は今回未取得 |

C_refの項目はlane evolution、CSWAP、phase、routing、reference reset、readout、必要な校正/誤差抑制を含める。compiled countは相互融合するため項別単純和と仮定しない。state準備は一shotで一個消費し、制御付き準備/逆準備を要求しないが、targetの反復再利用は保証しない。最終referenceは理想ではvacuumでも、次shotで初期化確認を無償にしない。

全RPEへ進むには実actual Qのfull価格、round別path長・split/fusion、共通誤差とreference初期化、hardware/readout/校正、target state消費を接続する必要がある。現在の結果はそれらの評価完了や期待費用優位の証明ではない。
