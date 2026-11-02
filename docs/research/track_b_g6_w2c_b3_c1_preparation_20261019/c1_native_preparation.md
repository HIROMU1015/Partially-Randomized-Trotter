# C1：native provider接続・価格・cacheの準備

[入口](README.md) / [証拠対応](cost_evidence_matrix.md) / [契約候補](execution_contract_candidate.md)。今回はrecord/interface/guardの人工検査であり、実IR・回路build/compile/synthesis・strict数値誤差の再評価は行っていない。

## 1. 固定P5ケースだけでは物理CQが決まらない

G7の登録P5はp/x/mと抽象Hermitian involution Q_iを用いる。既存runnerはexact controlled-Q query-modelを採用し、一般Q_iの行列/encoding/gate実装を固定しない。従ってlabelごとのCQ1回をnative T/CX/depthへ変換できない。**新しいQを便宜的にPauliや既存2-qubit toyへ置き換えることはしない。** 物理Q/encodingの選定はユーザー判断が必要であり、「同じP5をcompileするだけ」の技術的作業には還元できない。

旧G7のprovider誤差0はconditional assumptionである。非zero physical CQ誤差を導入すると既存bias契約との互換性確認が必要。price guardは指定されたerror allowanceを超えるrowを拒否する。既定のCQ allowanceを0のまま保ち、物理誤差を黙ってRZやρへ吸収しない。F3 angle enclosureもphysical certificateではない。

## 2. 再利用可能なSOURCE・保存cache

| 既存物 | 再利用できる部分 | 不足・差分 |
|---|---|---|
| G7 `controlled_event_ir` / `bind_synthesized_provider_ir` | wordの逆時間順、child forward/actual-adjoint、H4/RZ2/skeleton CX2、outer Z、σ±とW scalarを保持 | CQは抽象。実system回路、provider ancilla、全回路depth未提供 |
| G7保存`result_v1.json.synthesis_cache` | 取得済みpositive RZ sequence/key/ε/SHA/T/1Q/strict error record。negativeは保存positiveのactual adjoint、再synthesis不要 | 有限registered keyのみ。環境/precision/phaseが変われば自動流用しない |
| G8 `AcquisitionCache/NativePipeline` | cold取得memoとruntime LRU、hit/miss/eviction、失敗STOPという分離設計 | ratioだけのkeyや旧固定capacityを新契約の完成key/policyとしない。G8の別budget・failure policyは継承しない |
| `rte_reallocation/native.py` | 既存のphase保持adjoint・制御・scalar処理の参考 | 特定2-system-qubit context/Pauli等のAPI。G7 P5の未知Qへそのまま適用不可 |
| `rte_reallocation/numeric.py` | strict phaseのRZ合成/保存guardという既存手順、`up_to_phase=false`、W scalar込み | mpmath/pygridsynthを今回import/実行しない。physical CQや全回路費用の保証にはならない |

G7/G8の保存cacheとSOURCEはhash/identity・field構造を読み、同じ有限RZ取得があることを確認した。helper CX2はskeleton-onlyであり、総native CX2ではない。実matrix生成や誤差bound再計算は行っていない。同backend/seedの一致を独立再現としない。新しいcacheを生成して既存の科学証拠として保存していない。

## 3. 新しい価格interface

`price_key`はoperator/label、outer/helper control役割、forward/actual_adjoint、compiler、gate set、precision、strict phase、optimization policyをidentityに含める。arm名をkeyへ入れず、**同じ合法実装を4方式が共有可能**にする。異なるcontext/precision/phaseのrowは拒否する。whole-event最適化を選ぶ場合はword順・child・ratio・σ・phaseを含むevent digestとmappingがさらに必要で、primitive加算の価格だけで全event最適化結果を表現しない。

価格rowにはT/CX/1Q、serial event depth、ancilla peakの個別区間又は`null`を持つ。status=UNKNOWNは全座標nullとし、0との置換を拒否する。KNOWNでも欠測座標はPARTIALで、未知depthを既知Tへ吸収しない。artifact/certificate SHA256、phase保持、strict errorとactual-adjointのforward artifact/照合証拠を要求する。これは**提出された証拠のmetadata・予算照合**であり、合成の正しさを独立に数値証明するものではない。

`rz_cache_witness`はsaved key・ε・sequence hash/count・strict error record・W保持を確認する。人工sequenceのみでテストした。実cacheの数値誤差再検証は将来の承認対象とし、既存certificate引用と独立数値検証を混同しない。

## 4. 価格から費用へ渡す条件

W2c-Aの`N_F Σπ_F c_F,d ≤ N_a Σπ_a c_a,d`へ、保存N/πの来歴と新price artifactを結ぶ。現G7保存summaryにはlabel別**合計**CQ係数があるが、control役割/forward/adjoint別価格やword相関・depthを全て復元できるとは限らない。新価格が既存係数の粒度を超える場合、B2の追加role/angle/event contractionが必要である。これは新しい数値再解析なので**別途承認**を要し、今回行わない。全event tableが保存済みと仮定しない。

同じprimitive価格・literal additive modelなら旧affine係数を再利用できる。whole-circuit最適化・非加算depthなら新event mapping/aggregationが必要。native gate countとserial depth/parallel makespan、ancilla peakは別座標。全Nでpeakやdepthを機械的に掛けず、どのschedule量かを明示する。

## 5. Cache・最適化・次に必要な選択

cold取得費用、validated cacheのload/lookup、hit/miss、eviction、actual adjoint、readout/prepを別項とする。共通key poolの取得費を共有シナリオ/arm単独coldシナリオで明記し、片側だけ無料にしない。合法compiler最適化やbasis共有の機会は同じにする。新しいlaw/importance sampling/CTSへ変える実験はこの費用比較に追加しない。

残る選択は、(i) 物理Q/encoding/sector/basis、(ii) gate set/compiler/ancilla/最適化、(iii) CQ physical誤差と既存budgetの互換性、(iv) literal primitive価格かwhole-event価格か、(v) C2を除く限定native比較かprep/readout込みか、である。未取得のprep/readout/backendはnull、価格/科学判断をCodexが0で補わない。

**C1のinterface準備は完了、native取得の実行準備はこれらの選択待ち。** 選択・契約採択・別途実行承認後だけ、必要なprovider/鍵のcompileを行える。新native価格取得やDF/PR/RTE回路の転用を自動開始せず、mandatory STOP。
