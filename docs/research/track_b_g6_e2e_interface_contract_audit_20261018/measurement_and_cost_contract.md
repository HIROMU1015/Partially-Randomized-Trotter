# 全trial estimator、測定、費用の必要契約

以下は既存M7〜M10、G7/G8会計と現行Hadamard SOURCEの対応整理で、backend実行・sampling・shot forecastではない。新しい測定方式、confidence値、実験契約を採択しない。

## 1. 同じmeasurementという名前でも完成段階が違う

S14はalready-controlled trajectoryへX/cosine又はY/sineのHadamard wrapperを付ける。sineのS†→H、bit b→(−1)^b、相対位相を保持する既存testのSOURCEがある。resultはstate preparation/backend executionを含まず、quantum shots=0、fresh-IID verified=False。S16の測定命令込みcompile費用も量子測定を実行した意味ではない。

S17はprequantum zeroのときcache/synthesis/provider/readoutを避け、attempt/accepted/zeroを数える。ただしbit測定結果も加重accumulatorも持たない。監査対象で不足するのは、**G6 event／zeroからbackend outcomeと正確なWを結び、全trialで集計する接続API**である。

## 2. 変えてはいけないmeanと分母

ideal一試行は、拒否ならX=0、受理時はHadamard Y=±1にX=B_o Yを対応させる。全trial Nで平均し、既存M7のoperator meanを得る。accepted-onlyの分母へ変えると異なるmeanになる。

digitalでは実πとc̃からW̃=c̃/πを使い、拒否0／受理W̃Yとする既存構成を接続する。targetは一般にM̃で、元Mへの係数biasは別に課金する。単にordinaryの定数B_oを全digital eventへ掛ける処理ではない。πやWを受理集合内で再正規化しない。

将来のtrial recordには、`trial_id, axis, source/input/policy identity, signed time, outcome_kind, event-or-zero, actual_proposal, target_coefficient, weight, quantum_execution_count, bit/outcome, failure, budget/certificate IDs`に相当する情報が必要。これらは必要情報の仕様で、採用済みclass名やJSON schemaではない。

カウンタは少なくとも、attempt、事前zero、accepted、測定完了、backend failure、例外を区別する。固定N全trialの完了を判定し、accepted数やclassical cost sample数をNの代用にしない。失敗・dropout・precision例外を確率的zeroとして扱わず、成功まで再sampleして分母を変えない。途中失敗は記録を保全してSTOPし、不完全prefixを完全confidence結果として報告しない。

## 3. X/Y、IID、confidence

X/Yのmeasurement bit conventionと、σ±、label sign、ancilla-relative phaseを一緒に検証する。回路構築testの期待値一致と、fresh-IIDな量子測定の実行は別の証拠である。G7 `DeterministicBits`やPCG64のseed再現性を、fresh uniform bitsの数学的認証とは呼ばない。

S02の十分Nは保守的moment/range/bias予算を使い、αは1/320固定、16 axesのfamilyを想定する。S18はestimation failureとresource failureを分離する。S15のordinary RPE Hoeffdingは±1 outcomeを想定する経路で、event依存Wとzeroを含むG6へそのまま適用しない。新taskのaxis/family/confidence、RNG/IID policy、weighted range、bias残余幅は別途固定が必要で、今回数値を選ばない。

## 4. Multi-stepとwrapperの境界

現行ordinaryはRTE microsteps、partial-S2、repetition、one final Hadamard measurementの階層を持ち、共通normalizationのattenuationを会計する。G6の補正後全trial meanを同じ未補正信号と扱い、ordinary B_oの冪を重ねて掛けない。

複数stepでは、独立なpath draw、各eventの実確率とweight、ordered product、zero発生時の扱い、実行前に全pathを選別するかを明示する必要がある。既存D系列には固定providerの独立path費用会計があるが、一般DF／PFへ接続した測定契約ではない。各stepで測定する方式へ黙って置換しない。

全pathを量子前に選別する場合と、途中まで量子実行後に棄却する場合では費用が異なる。前者の期待accepted-path式を後者に移さない。既存の多step保証を超える条件は未採択の科学判断として残す。

## 5. N、Z、量子呼出し、cap

idealでは全試行受理率Z=B_RA/B_o、E[X²]=B_o B_RA、range≤B_o。valid parent条件付き受理A_uとZは別である。Nは統計的全trial、期待量子呼出しNZはhard capではない。digitalの実受理率や保存traceの比率を、未知Zの認証値として予算に代入しない。既存上界U、range/moment、S18の認証resource-tailを使う場合の条件を保持する。

拒否に量子回路・state preparation・測定を行わない契約なら、これらの費用はaccepted側に課す。古典draw・reduced判定・local query・coinまでの費用は拒否にも残る。backend失敗が起これば実際に消費した回路・準備・測定資源も記録し、科学出力が失敗したから費用0とはしない。

## 6. 費用vectorと既存実装

| 層 | 既存SOURCEで再利用できるもの | 未取得・必要契約 |
|---|---|---|
| Input/setup | DF hash/partition、basis registry、G6 series、U | 分子入力取得、対角化、正規化、有理変換と証拠取得の費用。無料oracleにしない |
| Local/bit/reject | G7 sample、G8 attempt/zero/cold/warm counters | selector線形走査・sort・任意精度演算・実bit sourceを含むonline費用 |
| Accepted circuit | DF controlled primitive、G7 conditional IR | どちらのloweringを使うか、strict error/provider実装、eventごとの費用 |
| Acquisition/cache | G8のon-demand memo、hit/miss/evictionの区別 | σ±・precision・tool identity・phase conventionを含むkey binding。旧capacityを新実験へ自動適用しない |
| Preparation | wrapperは明示的に除外 | state/sector/basis ordering、成功率、cleanup、acceptedごとの準備費用 |
| Measurement | X/Y wrapper、measurement bit map | backend shots/readout、failure、weighted全trial accumulator |
| Full wrapper | ordinary partial-S2/repeated/costの構造 | G6 lawとD/R・phase・全誤差・confidenceのbinding。直接costの転用不可 |

CPU秒、memory、bit処理、T/CX/1Q/depthを分け、異なる単位を無根拠に足したscalar scoreにしない。古典cost sample数はquantum shotsでない。二次momentだけから総量子費用を認定しない。未知B_RA、provider誤差、prep/native費用を0として扱わない。同じ入力・精度・state・compiler/cache・合法なbasis共有をordinary対照にも認める。

新しい費用値、shots、break-even、順位は作成していない。既存contractのcaps/STOPも変更していない。
