# 新A CT-4A：大きいqの有限Clifford+T比較・静的preflight

2026-11-07 JST、SERVER 2。固定参照 **`c21a43f444f3b7936de36acd0fbd38556ae594d0`**。独立branch `codex/pr-reference-ct4a-20261107`。**CT-4Aのみ完了、CT-4Bは未承認・未開始。** [承認対象の提案](SCOPE_PROPOSAL.md)。

## 結論

推奨する有限比較は、保存scheduleの **m=7、q=128、r=1、rare/nonrare各j=0**。同じevent列をB′/ReferenceのX/Yへ与える8構成だけとし、一様strict precisionは **epsilon=10^-11（e=-11）** を提案する。実行・正式採択は次のユーザー承認を待つ。

両stratumのevent列、seed、Reference vacuum phase、両方式・X/Yのnative価格が保存されている。一方、q128のwhole native QPYと、その有限8構成の完全なliteral key一覧は未保存。**whole native再compileは8構成必要**であり、exact missing key数と新T価格は現時点では未確定。保存SOURCEからの再構成と、native実key表の固定を、承認対象に明示してから合成する必要がある。

CT-3Aの全support policy未資格は不変更。この有限比較の実施は、SK fallback、全support T保証、全18round期待Tへ進む承認にはならない。

## 1. 固定条件と資料

Linear H4・1.00 Å・STO-3G・DF12、元Hの厳密groundを各shotで供給し状態準備費を除外。Partial S2・LD3・h=.02・finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、保存18round q/r。エネルギー精度1.5936e-4 Ha、全失敗率.05、位相予算.4 rad。Qiskit1.3.0・compiler A・rz/sx/x/cx・全結合。B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum補正・2 register・8-CX CSWAP、D融合・basis共有・scalar・event符号・X/Y wrapperは変更しない。

対象はcompiler後の実binary64 RZ。strict matrix、up_to_phase=False、W保持、負角はpositive wordのactual adjoint、written product順を保持する。phase fit、理想pi snap、2pi key縮約は行わない。

| 固定資料 | 今回再利用した内容 |
|---|---|
| [CT-1：6ac8e1a](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/6ac8e1aa3104d8563188612410c1a7533ec47072/docs/research/pr_reference_ct1_20261107) | 有限q1/r1・e=-9のword/guard/emitter、local/whole correctness、会計規約とSTOP |
| [CT-0：f5f16fa](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/f5f16fad64748236b0160284dc6c3017974b0132/docs/research/pr_reference_ct0_20261107) | 保存round別worst RZ/headroom、precision候補、record coverage、全support位相の定義 |
| [CT-2：6b0b666](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/6b0b6669bbf58ef22e789bfef134a1a6da9f2343/docs/research/pr_reference_ct2_20261107)、[CT-3A](../pr_reference_ct3a_20261107/REPORT.md) | 固定backendの全入力停止/T上界未資格、strict/phase/価格の境界。再監査やfallback実装はしない |
| [V3：224440f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/224440f740051ed5794e8db87a57908fc02dd02f/docs/research/pr_reference_rotation_v3_20261107) | 保存local inventory、numeric/symbolic key、signed融合差分、cohort/QPY検査 |
| [V5補強：a095a22](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/a095a225ceae383027b300e1f31ec5ccbb1522fc/docs/research/pr_reference_depth_v5_reinforcement_20261107) | actual inverse Dの資格と既存native depth保証。T-depthへの移送はしない |
| [B′：43bfc59](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/43bfc5918703e8dcd020be4fcf96bb8dfd99f909/docs/research/pr_reference_bprime_20261107) | 保存paired価格、4-CX中央回転、native後のliteral RZ、誤差/RPE条件 |

親/worktree AGENTS.md、PROJECT_MAP.md、研究概要、正式共通運用指示2026-10-25を確認した。保存された過去のCPU/worker設定は今回の恒常上限にしない。旧SOURCE/raw/result/grant/ledger/marker/STOPを変更しない。

## 2. 比較条件の選定

| 保存条件 | 静的に確認した利点/負担 | 選定 |
|---|---|---|
| q1/r1 | CT-1完了済み。新しい大きいqの証拠にはならない | 再実行しない |
| q2/r1 | 保存短round。ただしCT-4の成長側の比較点としては小さい | 追加しない |
| q4/r1 | 両方式・両stratumのX QPYあり。Y4構成のみ再compileで済む | 低負荷候補だが採択しない |
| q8/r1 | Reference X QPYあり、B′ whole QPYなし | 追加しない |
| q32/r1 | 保存growth holdout、paired event/価格あり | q128との多点探索にはしない |
| **q128/r1** | 保存scheduleの最後のr1。q1とrを揃えたまま反復を増やせる。paired event/両軸価格あり。whole QPYなし | **1条件だけ推奨** |
| q256/r2 | 保存paired価格はあるがrも変わり、処理量が増す | 対象外 |

q128を、添付例という理由だけで採用していない。保存paired情報・same-r比較・長い反復・1条件だけという診断範囲を理由に選ぶ。q4の保存QPY利用は安価だが、今回はr1区間の終端を1点だけ確認する案とする。これはT価格を見る前の実験設計上の提案で、q128で利益が出るという予測ではない。負荷不足や失敗時にq4等へ自動変更しない。

q1とq128では要求precisionが異なるため、旧CT-1との差をqだけの因果効果とは呼ばない。主判定は**q128内で同一precisionを使うB′対Reference**。q1をe=-11で再価格する追加作業は今回提案しない。

### 保存trajectoryのidentity

入力archive：[transfer/raw_prices.tar.xz](../../../artifacts/pr_reference_transfer_qualification/2026-11-06/raw_prices.tar.xz)。以下は保存値/metadataの引用・照合で、新samplingではない。

| stratum | archive member | seed | event数 | 保存vacuum phase hex |
|---|---|---:|---:|---|
| nonrare（rare=False） | raw/q128_r1_rare0_0.json | 14787950215150025216 | 128 | -0x1.797016069e704p+2 |
| rare（rare=True） | raw/q128_r1_rare1_0.json | 9772911353196937292 | 128 | -0x1.40e61dfde8f76p+1 |

event identityは保存event配列の `json.dumps(events,sort_keys=True)` のSHA256で、保存B′ recordのevents_sha256とも一致する。

- nonrare：`6bd2c4a9f5e861db301fd2df1cdc86dd646dd06fbd883c097191835b14c0a10e`
- rare：`688f3c52622170692c0f4846f1d07c3c54c8b83aae5e341e03057f5271abbcf8`

対応する[B′ paired archive](../../../artifacts/pr_reference_bprime/2026-11-07/paired_Bprime_prices.tar.xz)はraw/path48.json、raw/path50.json。B′/ReferenceとX/Yでevent配列を共有し、保存seedから再抽選しない。seedはprovenanceであり、eventそのものを入力とする。rare/nonrareを無加重平均して期待Tとしない。

### 保存native価格（再集計なし）

metric順は既存 `pilot.metric` のRZ/CX/native DAG depth/size。以下は各保存rowの値をそのまま引用した。T価格は未取得。

| stratum | 方式 | X:RZ | Y:RZ | CX（両軸） | X depth | Y depth | qubit |
|---|---|---:|---:|---:|---:|---:|---:|
| nonrare | B′ | 917709 | 917710 | 153846 | 311616 | 311616 | 9 |
| nonrare | Reference | 916901 | 916902 | 140222 | 140631 | 140632 | 17 |
| rare | B′ | 916806 | 916807 | 153746 | 311892 | 311892 | 9 |
| rare | Reference | 915985 | 915986 | 140118 | 140896 | 140897 | 17 |

V3の[QPY_CHECKS](../../../artifacts/pr_reference_rotation_v3/2026-11-07/QPY_CHECKS.json)と保存SOURCEのQPY保存条件を確認した。既存whole QPYはq1/4/8のX代表とCT-1のq1 X/Yで、確認対象archiveにq128 whole QPYはない。旧native価格だけから新CT価格を補間しない。

## 3. Precisionの根拠

以下はCT-0 [PRECISION_FEASIBILITY](../../../artifacts/pr_reference_ct0/2026-11-07/PRECISION_FEASIBILITY.json) m7/q128/r1の保存値。新しいheadroom計算やshot再解析はしていない。

| arm/axis | worst-case RZ上界L | 保存N/axis | 一様gate errorの保存十分上端 |
|---|---:|---:|---:|
| B′ X | 1187357 | 199 | 7.916272634734305e-11 |
| B′ Y | 1187358 | 199 | 7.916265967602206e-11 |
| Reference X | 1184885 | 199 | 7.932780474995138e-11 |
| Reference Y | 1184886 | 199 | 7.93277378002155e-11 |

保存された要求仕様e=-13,…,-6のうち、10^-11は全4十分上端より小さく、次に緩い10^-10は条件を満たさない。よって既存仕様内の最も緩い十分候補としてe=-11を提案する。これは非一様配分や価格最適化ではない。保存の最小上端は丸め近傍ではなく、採用可否を新しい境界探索で決める必要はない。

既存Frobenius strict guard上界はspectral上界として使用できる。precisionは両arm/軸/stratumで一様。CT-1 e=-9を移送せず、全18round共通の6.902168833913054e-16もこの1条件へ過剰に採用しない。alpha=1/720、phase cap .4、m7の保存N=199/axisを維持する。全18roundの新合成精度採択ではない。

## 4. Coverageと未確定事項

既存[CT-0 coverage](../pr_reference_ct0_20261107/KEY_COVERAGE.md)を再利用する。保存C2-B 7,603 recordのうち、e=-11は954 record。V3の全schedule numeric域では、B′ positive400 key中153、Reference384 key中154が同一eでstrict再利用候補だった。これは**q128有限8回路の新coverage件数ではない**。既確認の7,603件を再検査・再集計していない。

| 区分 | CT-4Aで確認できること | CT-4Bで必要なこと |
|---|---|---|
| 同一binary角・e=-11 | C2-Bの保存spec/word/guardを再利用できる候補がある | 実8回路のkeyと照合して必要recordだけ採用 |
| CT-1 e=-9の268 word | correctness/emitterの資産。要求precisionは異なる | 本提案ではe=-11へ自動再ラベルしない。cross-precision価格選択は行わない |
| legacy record | guard_method未記載4件は保存e=-13。今回候補e=-11とは別 | 旧recordへmetadata追記・再資格をしない |
| 未取得numeric key | V3全域には欠損がある。q128のexact件数は未確定 | native実keyを固定後、missingのみ取得 |
| B′ event phase | r1の逐次pi加算supersetはCT-0に保存済み | 実event/sourceからliteralな使用keyを確定。parityやk*piで置換しない |
| Reference vacuum | 上記2保存binary角を入力として固定できる | SOURCE再構成/compileと一致を確認し、実targetのe=-11 recordを照合 |
| inverse/boundary/wrapper | V3/CT-0/local catalogueに参照資産あり | 実nativeで出現する符号・最後のulpを保持し、分類名だけで価格を貼らない |

V3のANGLE_KEY_MULTIPLICITIESはround/role別の期待台帳で、q128の8 whole literal histogramそのものではない。COHORT_CHECKSには各行の総数があるが、完全なliteral histogramは保存されていない。したがって同台帳を無条件に実key whitelistに変換せず、正確な新規key数を0や推定値で埋めない。

必要key数・処理量は、承認後の固定native8回路から `(signed float.hex, role/position)` を読み、positive lookupへ対応させ、同一eの保存recordとの集合差で得る。T/価格の集計はこの固定の後だけ行う。今回この処理・新key catalogue生成・QPY loadは実行していない。

## 5. 再利用と負荷

主な再利用は[CT-1](../../../src/trottertracks/pr_reference_ct1.py)のinstructions/record照合/word検証/会計、[B′](../../../src/trottertracks/pr_reference_bprime.py)のreconstructed/build_saved、[transfer](../../../src/trottertracks/pr_reference_transfer_qualification.py)のassemble、[pilot](../../../src/trottertracks/pr_reference_interference_pilot.py)のreference_core/wrapper、[compiler A](../../../src/trottertracks/track_c_compiler_a.py)、[strict engine](../../../src/trottertracks/track_c_h4_synthesis.py)、[lower_native](../../../src/trottertracks/track_c2_b_verify.py)。[V3](../../../src/trottertracks/pr_reference_rotation_v3.py)のsaved_path_partsとlocal catalogueはrole照合の候補。[B′会計](../../../src/trottertracks/pr_reference_bprime_accounting.py)、[transfer誤差](../../../src/trottertracks/pr_reference_transfer_error.py)、[C2-B強baseline](../../../src/trottertracks/track_c2_b_strong_baseline.py)の規約を確認したが、18round会計・optimizer・旧runnerは起動しない。

CT-1はq1、出力先、PREDECLARED endpointが固定されているため、旧stage_a/b/cをそのまま再開できない。別namespaceでq128入力とm7 endpointを渡す最小adapter、literal histogram/T会計、必要局所検証だけが追加候補。合成アルゴリズムや回路構成は新設しない。

保存nativeは各構成約91万RZ・約135万sizeで、CT wordを全物理gateとして展開すると出力・memoryが膨らむ。T会計はliteral histogramと実wordの積算で行い、各出現を課金する。全expanded CT QPYや17-qubit dense matrixを価格取得の必須条件にしない。局所operator/emitter検証とtelescopingを再利用し、既存guardの証明を巨大state simulationで代用しない。

SERVER 2観測時のhostはGPUA100-gleap02、CPU affinity64、MemAvailable485264496 kB、worktree filesystem空き217442889728 byte。loadavgは32.35/21.49/18.55。これは観測時点の利用状況で、将来の割当や実行時間の保証ではない。実行時に空き資源/適用制御を再確認する。CPU/RAM/workerに恣意的な恒常上限を置かず、8 independent compileやkey単位合成を利用可能資源に合わせ並列化し、入力/word行列を再利用する。共有設定・他ユーザーのprocessは変更しない。

正確な新規合成数、全word長、再compile/QPYのRAM・保存量、壁時計時間は未取得。過去の有限成功から取得完了を保証せず、20,000文字の事後拒否を時間上限や全域T価格にしない。新fallback/恣意的timeoutは今回採用しない。

## 6. 終了

実施したのは既存資料/SOURCEの読取、保存row/metadata照合、入力identityの確認、precision十分条件の記号的比較、資源観測、2文書と最小索引の作成。**新規合成・strict guard実行・回路build/compile・simulation・sampling・科学数値再解析・新T価格集計・実装/testは行っていない。**

CT-4A完了で停止する。CT-4Bは、[SCOPE_PROPOSAL](SCOPE_PROPOSAL.md)の「固定8構成のnative準備→実key固定→missing合成」という順序も含めた別途明示承認を要する。旧STOPを保全し、CT-3B/全support/全18round/T-depthへ自動進行しない。次担当は**ユーザー**によるscope/precisionと実行承認の判断。その後の科学的評価は**GPT／ユーザー**。
