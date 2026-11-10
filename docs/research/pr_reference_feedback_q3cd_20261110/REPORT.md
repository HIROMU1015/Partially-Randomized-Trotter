# 新A Q3-C〜D：Native移送・340 shots・18round native費用

2026-11-10 JST／SERVER 2。**固定モデル内でnative全contextへの移送とm17の340-shot十分保証が接続し、保存X/Y方式に対する全18roundのnative費用削減が成立した。** Referenceの総RZ削減率下限は **50.381058%**。同じQ方式を与えた強B′に対してもReference総RZ上界はB′下界を下回り、削減率下限は **0.287865%**。

これは指定H4・SOURCE・理想native gate・binary64算術・192-bit区間・fresh independent finite-bitモデル内の結果である。著者Codexの実装・技術／数値検証であり、独立レビュー、新しい科学保証の正式採択、実機成功率、総T削減、強Full/Truncated PFへの優越は認定しない。

## 固定条件・成果への入口

基準Q3-B `1795d72b704cfd89aeac7c0075ebb576406ba624`。既存branch `codex/algo-improve-new-a-total-cost-20261109`を継続した。Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2／LD3／h=.02／K2、元科学SOURCE `f2638a7845685a119619382b850441f5d8d33713`。m17はq131072/r32/bits88。m0–16のq/r/bits/N・X/Y、energy精度1.5936e−4 Ha、全failure .05、phase cap .4、Compiler A/Qiskit1.3.0を維持した。

- [NATIVE_GUARANTEE](NATIVE_GUARANTEE.md)：実行前scope、全context閉包、一般native誤差のvirtual-state補題、consumer／reset条件。
- [RPE_AND_COST_RESULT](RPE_AND_COST_RESULT.md)：成功保証の接続、資源別境界、費用式、残る判断。
- [SOURCE](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/SOURCE.json)／[C0](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/C0.json)／[D](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/D.json)／[COST](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/COST.json)／[CHECKS](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/CHECKS.json)／[STOP](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/STOP.json)。全価格／差／判定はexact Fraction。JSONのdisplayは概数であり外向き端点ではない。

## C0／C1：何が閉じたか

保存B′4 contextとReference4 contextを照合した。Y有無の共通recordは合計2,640組すべてloweringが同じ。P/H/Sdg template・D/frame fenceと、Q2保存16 pass traceのpost-unroll集合も一致した。新Qは旧Xと同じsource gate集合を持ち、追加Pは既に常在するfamilyである。BasisTranslatorの全体gate集合依存を回避せず、その集合と共通templateの一致を確認した。

P(−φbin)は任意の有限binary64 parameterに対してRZ(−φbin)となる。角度を丸め直さず、PとRZの差はjoint全体のcommon global phase。core/Hのloweringと旧native誤差chargeを変えない。旧YとQのnative gate名・wire列は同一で、ancilla上の1 RZのparameterだけが異なる。初段／最終passでこのRZを消す規則はない。

一般のcore/H誤差は単一のrad誤差と見なさない。readoutの不完全なHをstateへ移した正規化virtual stateを用いることで、各固定履歴・角度に対しphysical meanを、旧raw disk内の複素信号の射影として表せる。virtual coherenceの絶対値≤1も保つ。旧2H＋Sdgの安全側chargeはそのまま保持し、PF／finite-bit／SOURCE／native／B1 vacuum diskを再加算しない。

これでsignal/coreの移送は構造的に閉じたため、**C1の新しい科学fixture compile／operator計算は不要、実行0件**。補助的に、全4 Q gate集合を自由symbolのままlowerする人工技術testを4 compile実施し、最終消去前からnative Xがないことを確認した。これはH4の新角度・回路の科学compileではなく、sizeのSX会計に必要なmap構造の確認である。[SYMBOLIC_CONTEXT](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/SYMBOLIC_CONTEXT.json)。

## Consumer・人工検証

新[AdaptiveReadout](../../../src/trottertracks/pr_reference_feedback_q3cd.py)は、Q3-B prefix→Q1 feedbackを一度bindし、各physical-shot要求で同じfeedback objectを使う。既存canonical producerを毎回呼ぶため、trajectoryを使い回さない。B′は既存tail builder／rewrite_rte／symmetric_fused、Referenceは既存V6-Q caller／exact-binary B1／F.inverse()／coreを使う。Q wrapperそのものをcompileし、旧X/Y回路のaxis名を変更していない。Reference reset8個は旧consumer関数を再利用し、compile後prefix・物理順序は最初を維持した。

人工3 testで両armの共通角、fresh producer要求、実Q wrapper、mock compile/reset順、decoder binding、virtual-state恒等式を確認した。さらに上記symbolic context test1件がPASS。合計4 test、symbolic compile4件。H4 cache読込、trajectory生成、実m17 whole build/compile/simulation、backend measurementは0件。[人工test](../../../tests/test_pr_reference_feedback_q3cd.py)／[symbolic test](../../../tests/test_pr_reference_feedback_q3cd_context.py)／[TESTS](../../../artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/TESTS.json)。本番backend shot loopを実行したという意味ではない。

## D：340-shot十分保証

判定は両armとも **`GUARANTEE_CLOSED_IN_FIXED_MODEL`**。Q0／Q3-A／Q3-Bの固定成功event・interval branch・alias・全count幅証拠を再利用し、Q0 count走査・二分探索やQ3-Bの587,815 count位置監査は再実行しなかった。

δarith≤約2.22034e−16 rad、追加Pの純粋角度差0という解析結果を、保存cap `.000748774257 rad`に接続した。一般native誤差を0にしたのではなく、旧raw diskへ残している。B2の丸め余裕はReferenceでも約8.61443e−13 radが正であり、中心・hullの丸めcharge後に残る。

prefixは34座標、各failure≤1/720。任意の成功履歴を固定すれば、feedbackは過去データだけの関数で、340 fresh trialに同じbinary角を使う。Hoeffdingの条件付きfailure≤1/360を積分し、union boundは `34/720+1/360=.05`。成功event上の新たな受理失敗を除外せず、Q3-Bの一様受理を用いる。最終energy半幅上界は約1.52587890625e−4 Haで、指定精度以下。総physical shot数は両armとも8,598→7,686（prefix7,346＋末尾340）。

## m17・全18roundの費用

判定は資源ごとに **`TOTAL_COST_REDUCTION_ESTABLISHED_IN_FIXED_MODEL`**。比較元はB′／**Reference-B1**の626 shots/axisであり、旧naive Referenceの630/axisとは区別する。Q方式を両armへ同じ条件で与えた。

| arm | 従来全18round総RZ（概数） | Q方式総RZ（概数） | 削減率の保証下限 |
|---|---:|---:|---:|
| B′ | 3.571960961e12 | 1.772330243e12 | 50.382149% |
| Reference-B1 | [3.561600143, 3.561600146]e12 | [1.767228305, 1.767228307]e12 | 50.381058% |

| 指標 | B′の従来X/Y比・全18round削減率下限 | Reference-B1の従来X/Y比・全18round削減率下限 | Q方式同士でReference上界がB′下界より低い率 |
|---|---:|---:|---:|
| RZ | 50.382149% | 50.381058% | 0.287865%以上 |
| CX | 48.767898% | 48.656658% | 5.182861%以上 |
| size | 50.225113% | 50.225178% | 0.698523%以上 |
| shot加重DAG depth | 40.233483% | 38.719184% | 41.351581%以上 |

下限表示はexact判定から小数点以下6桁へ切捨て。境界は最適実装や実行可能な端点policyではない。m17のRZは約2.47055e12→6.70915e11（B′）、約2.46333e12→6.68954e11（Reference）。そのroundの削減率は両armとも約72.843450%。m0–16の費用は同じものを保つため差では厳密に相殺する。

## 費用保証の根拠・追加費用

Qは旧Yと同じnative価格なので、資源jごとのm17差は `626 μX,j+286 μY,j`。Pの1 RZ／1 gateはQ単価へ含めた。新readout用のancilla/registerや別quantum feedback gateは使わず、interrogation外の追加native量AはこのSOURCEでは0。古典feedback／校正／回路生成を0価格として量子費用へ足した結果ではない。

RZ/CXは保存TOTALSの安全側価格、depthは保存V5とinverse-D／非X鎖の既存補強を継承。sizeはRZ＋CX＋保存SX多重度＋measurement/resetで囲む。全4 symbolic mapが最終cancel前からX-freeであるため、未知X価格を0補完していない。F†についても、保存20 inverse-D recordのnative RZ/CX/SX多重度が正方向と同じであることを照合した。Reference CX下界はfenced D/frameのCXだけを残し、CSWAP／rotation CXを安全側に捨てた数学的下界である。

canonical lawの保存basis確率／order2確率／support期待値と旧basis-run式を使い、標本平均や平均event回路を作らなかった。m17で同じ価格端点を新旧費用に対応させ、負のshot差に対する独立区間減算を避けた。V6-P保存RZ総計をexactに再現。144 arm/round/metricセル、全和・差・SOURCE/input hashをpost-run checkerで検算した。検算PASSは著者自身の検査であり独立レビューではない。

## 実行・保全・残余・次の判断

[新runner](../../../scripts/run_pr_reference_feedback_q3cd.py)の保存値campaignは1回、約1.47秒。小さい有理数表の依存処理なので不要なworkerを作らず、CPU/RAMの恒常上限は設けていない。共有環境のCPU/memory観測はSOURCEに保存し、共有設定／他者process・fileを変更していない。旧SOURCE／raw／結果／grant／ledger／marker／STOP／KEEPは全て不変更。今回の全追加ファイルだけをcommit/push・remote照合する。

残る未資格は実RNGと実機ノイズ／runtime、backend loopと実機成功率、古典feedback・長round回路生成の時間／メモリ価格、全18round総T／T-depth・FT資源、強Full／Truncatedへの公平なQ適用比較である。peak qubitsはB′9／Reference17で不変、Referenceの追加8 qubitをgate削減へ換算していない。旧CTのprecision/domain/word/emitter等の負結果とSTOPも維持する。

**次担当はGPT／ユーザー：** 固定モデルのnative移送・340-shot保証と資源別費用の研究上の正式採否、及び別途必要な実機／CT／Full・Truncated比較の優先度と承認範囲を判断する。**Codex：公開・照合後mandatory STOP。** 自動的な追加探索・実測・合成・別Trackへの展開は行わない。
