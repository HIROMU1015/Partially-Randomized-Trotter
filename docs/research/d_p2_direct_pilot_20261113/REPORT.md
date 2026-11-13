# P2直接paired残差pilot：完了結果

2026-11-13 JST。**`P2_DIRECT_PILOT_COMPLETE`。replay2 pathの全照合PASS後、fresh8 pathを取得した。
科学compileは20/20件、cache hit 0、retry 0。** 実行wallは1,774.36秒（29分34秒）。
全law E[C]・2%/95%達成・他round移送は未資格で、追加科学実行へ進まない。

[RESULT](run01/RESULT.json)／[REPLAY](run01/REPLAY.json)／[FRESH](run01/FRESH.json)／
[20軸CSV](PATHS.csv)／[CHECKS](CHECKS.json)／[HANDOFF](HANDOFF.md)。

## 1. 固定条件と実行

| 項目 | 条件・出典 |
|---|---|
| 基点 | `5a67cea7705e80ff295306606611d774724ae2e2`、既存D branchを継続 |
| 実行SOURCE | `ea0706ab05313423f336614912758a93e9e8285c`（runner・人工test・PLANをcommit/push、remote一致後に1回実行） |
| 計画 | [PLAN.json](PLAN.json)、seedを含め実行前に固定。hashはPREFLIGHT/RESULT/CHECKSに保存 |
| 物理条件 | Linear H4/1.00 Å/STO-3G、保存tol-only DF10、S2/L_D1/K2/h=.1/M12/P0、元exactκ・H/state/snapshot・fragment/basis/qubit順・scalar一回 |
| 回路 | symmetric-directional controlled full Hadamard X/Y、`CONTROLLED_ZZ_CX_CRZ_CX_V1` |
| compiler | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` |
| runtime | NumPy1.26.4、Python3.11.0rc1、既存専用venv（[実行記録](EXECUTION.txt)、[資源記録](EXECUTION_RESOURCES.txt)） |
| 再利用 | pinned C1 Backend/context `1a87e533`、carrier `155529e2`、provider `c31ef424`、compiled_price_fast、paired_residuals/cohort_summary |

canonical configurationは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。
C1 contextによる保存input・preparation・SOURCEの照合が通過してからworkerを起動した。
高速化本体・新ZZ・fast fingerprint fileは基点5a67cea7とbyte一致のまま。

replayは保存component列から元`_make_event`で復元し、各eventのJSON表現・seed・law・digestを照合した。
乱数による再samplingは0。freshは`D_P2_DIRECT_PILOT_V1`、m0/m6各index0..3の固定seedで
元のnormal samplerを1回ずつ使用した。forced rareは追加していない。
replay2 process→全一致gate→fresh8 processの順にpath単位で並列化し、X/Yは各path内で個別に取得。
CPU/RAM/affinityの人工上限や共有設定の変更、他processの操作はしていない。

## 2. replayの同値性と実H4性能

両pathでevent digest、全seed、C、S、X/Y wrapper fingerprint、compiler hash、全6 compiled指標が一致。
原event fileのSHA-256も前後で同じだった。fingerprintの範囲は**full wrapperのpre-transpile回路**である。
旧回路を新たにcompileして比較した結果ではなく、P1 run02の固定recordとの照合である。

| replay | C（X=Y） | S | D | P1 phase-B秒 | 新price call秒 | 新path秒（復元込み） | 履歴/新path比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| m0/index0 | 1,906 | 1,845 | 61 | 4.037 | 2.558 | 2.566 | 1.57 |
| m6/index0 | 1,826,433 | 1,747,585 | 78,848 | 3,139.163 | 856.896 | 881.442 | **3.56** |

P1のphase-B時間は再sampling等を含み、新pathは保存eventの読込み・復元等を含む。
新price callだけを分母にしたm6比は3.66だが、同じ計時境界ではない。
上表は同一pathの**履歴対今回の単回観測比**で、実行時点・並列負荷の差を含む。
P1はm6が2 worker、今回replayはm6が1 workerである。一般の倍率や統計的な性能保証とはしない。

worker peak RSS（MB=10^6 bytes）：m0は旧280.92→新277.04、m6は旧2,998.73→新2,794.41。
旧`rss_mb_peak`はP1実装上MiBなので換算した。これは各processのhigh-waterで、同時processの合計peakではない。

m6 replayの新pipeline内訳：

| 段階 | 秒 |
|---|---:|
| event dict / JSON write / digest | 8.70 / 9.30 / 1.64 |
| 構造S（X/Y共通本体＋個別suffix） | 94.84 |
| compose（部分/全evolution fingerprintを含む） | 107.00 |
| X wrapper / fingerprint / mutation check | 27.82 / 24.00 / 23.98 |
| X compile＋6指標取得 | 241.75 |
| Y wrapper / fingerprint / mutation check | 27.80 / 23.98 / 24.43 |
| Y compile＋6指標取得 | 241.55 |

P1のm6 compileは241.3/241.8秒で、今回もほぼ同じ。価格を変えずPython側の処理時間が減り、
現在はcompile＋指標取得がprice callの約56%を占める。compose・S・繰返しhashも残るが、
今回の固定SOURCEを変更して追加高速化は行っていない。

## 3. fresh cohortの結果

下表は**freshだけ**。X/YのCとSは今回の全pathで同値だった。nは各roundで4 pathであり8ではない。
全pathのorder2=0、order0はm0が2、m6が8,192、product/identity applicationも0。
各axisのCX数・全depth・size・fingerprint・compile時間は[PATHS.csv](PATHS.csv)に記録した。

| round/index | C | S | D | N_ZZ | N_Z | price call秒 | path wall秒 | peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m0/0 | 1,750 | 1,535 | 215 | 2 | 0 | 2.074 | 2.082 | 273.37 |
| m0/1 | 2,100 | 1,849 | 251 | 2 | 0 | 2.498 | 2.506 | 278.09 |
| m0/2 | 1,912 | 1,843 | 69 | 2 | 0 | 2.633 | 2.642 | 278.61 |
| m0/3 | 1,748 | 1,533 | 215 | 2 | 0 | 2.075 | 2.084 | 273.90 |
| m6/0 | 1,844,393 | 1,769,572 | 74,821 | 7,502 | 690 | 878.254 | 889.917 | 2,813.45 |
| m6/1 | 1,831,377 | 1,760,889 | 70,488 | 7,519 | 673 | 834.759 | 846.211 | 2,795.81 |
| m6/2 | 1,853,568 | 1,774,969 | 78,599 | 7,535 | 657 | 857.119 | 868.730 | 2,811.34 |
| m6/3 | 1,836,368 | 1,762,721 | 73,647 | 7,495 | 697 | 871.845 | 883.280 | 2,752.20 |

元adapterを使ったexact rationalの記述統計（以下の両軸は同値、分散/共分散はddof1）：

| round | D平均 | D標本分散 | DのX/Y共分散 | D標本SD | C平均 | S平均 |
|---|---:|---:|---:|---:|---:|---:|
| m0 | 187.5 | 6,529 | 6,529 | 80.802 | 1,877.5 | 1,690 |
| m6 | 74,388.75 | 134,716,355/12 ≈ 11,226,362.917 | 同左 | 3,350.577 | 1,841,426.5 | 1,767,037.75 |

| round | C標本分散・X/Y共分散 | S標本分散・X/Y共分散 |
|---|---:|---:|
| m0 | 83,723/3 ≈ 27,907.667 | 97,364/3 ≈ 32,454.667 |
| m6 | 282,797,017/3 ≈ 94,265,672.333 | 503,074,427/12 ≈ 41,922,868.917 |

このcohortではVar(D)がVar(C)より小さく、E[S]+平均Dの直接法を続ける実用的な理由がある。
母分散の関係や全roundへの一般化は認定しない。m0は全4 pathでN_ZZ=2でもDは69〜251と異なった。
全8 fresh pathでD>0だが、符号の一般保証・N_ZZ model・fitの採用根拠にはしない。
P1 developmentとreplayの値は上の統計に混合していない。

## 4. 次の標本数・round・負荷の見積り

[RESULT.future_design_only](run01/RESULT.json)は、旧P2提案の候補条件を仮に使った**設計値**である。
95%同時confidenceの26軸Bonferroniを仮定しz=3.1018618、MC絶対予算を
1.2%×保存E[S]（Cの9d95d3eaからm0/m6の既存値をcopy、数値enclosure半幅を控除）とした。
`ceil(z² s_D²/b_MC²)`でpilot分散を代入したもので、2%/95%の達成判定ではない。

| round | MC予算RZ/axis | plug-in n | 独立確認cohortの候補n | compile数 | 今回の平均path時間からの仕事量 |
|---|---:|---:|---:|---:|---:|
| m0 | 20.9240 | 144 | 144 | 288 | 約335秒 |
| m6 | 21,175.9113 | 1 | 4 | 8 | 約3,488秒 |
| 合計 | — | — | 148 path | **296** | 約3,823秒＝63.7分のpath処理時間和 |

m6のn=1は分散推定やtail評価には不足するので、候補には少なくとも4 pathという設計上のfloorを置いた。
CPU/RAM上限ではない。このfloorにも統計的な精度保証はない。n=4のs自体が不確かで、
rare未観測・実sampler law・分布の裾の条件はこの式に解決されていない。
現行精度を正式採択し、これらを扱う区間・最終固定nを決めるまでは候補296 compileを実行しない。

今回fresh m6の4 processは約14〜15分/path、peak約2.75〜2.82 GB/processだった。
同条件の4 long pathを並列化し、小さいm0を重ねられるなら候補148 pathもwallは約15分＋起動/I/Oの
見積りになるが、これは仕事量からの目安で、資源競合や実行時間の上界ではない。
同時RSSの全job peakは未測定。任意worker数での強スケーリングを測った結果でもない。

**全13roundに必要な残りはm1〜m5、m7〜m12の11round。** 今回からnやD平均を移植しない。
全26 E[S]は再利用できるが、それだけで未測定Dやcompiled平均は埋まらない。
長roundを直接測る実行可能性を次に問うなら、最初の隣接点m7（q128/R255）を
fresh2 path、両軸最大4 compileの負荷診断として提案できる。これは平均価格認定用nではない。
event位置は32,640とm6の約4倍だが、wall/RSSの線形増加やm8..12の実行可能性は未測定のまま。
この提案も未実行で、別SOURCE・seed・有限予算の明示許可が必要。

推奨する次の判断は、(a)必要価格domainと精度、(b)ideal/実装lawのtargetとPRNGモデル、
(c)rare/tailを含む不確かさの扱い、(d)既観測2 roundの独立確認かm7負荷診断か、である。
価格modelや局所機構の完全分解は、直接法が予算・実行可能性を満たせない場合だけ検討する。

## 5. 資格・保全・STOP

今回のC/S平均・D平均/分散はfinite pilotの観測で、all-law E[C]はnull。
ideal E[S]と実PCG64/CDF samplerのlaw差、rare寄与、長q/R系統差、total CIもnull。
order2未観測を寄与0と扱わず、n増加でlaw biasが消えるとも扱わない。
P2計画の2%・95%は今後の候補のまま。十分N・RMSE・Gは未評価。

[人工テスト](TESTS.txt)：新7＋既存pipeline9＝16 PASS。
[CHECKS](CHECKS.json)：20 compile、固定seed、exact fresh moments、原replay event SHA不変、
実行中SOURCE不変、基点高速化本体のbyte一致、private70 fileのbytes/SHAを確認。
作者自身の技術確認であり、別reviewerによる独立科学レビューではない。

旧SOURCE/input/raw/結果/KEEP/grant/ledger/marker/STOP、P1 run01/run02とPLAN/RUN02_PLANは不変。
新privateは`/home/AbeHiromu/projects/d-p2-direct-pilot-20261113-private/run01/`。
追加round/path・retry・forced rare・fit/外挿・全13round平均C認定・N/RMSE/G・8次PFは0。
結果をcommit/pushしremote SHA・tree・主要成果物を照合して**mandatory STOP**する。
