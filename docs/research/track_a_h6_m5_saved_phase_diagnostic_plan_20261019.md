# H6 full m5_best：保存証拠だけによるphase追加診断計画

2026-10-19。**計画作成のみ完了。実データ解析・QPY復元・statevector再生は未実行、runnerも未実装。** ユーザーが承認したのは追加診断の計画方針であり、実解析の承認へ読み替えない。新compileは提案範囲にも含めない。元の2つのSTOP、raw基準1e-9未満、成功済み58軸を保持してmandatory STOPする。

起点結果commit：`282cba7d6047ea6613a9e69c818e467cb962dd65`。調査対象SOURCE：`f743a88225f7cb2c8d2c91d7e19813a5780f803b`。branch：`track-a-h6-m5-saved-phase-plan-20261019`。[元結果](track_a_h6_m5_full_recovery_run01_results_20261019.md)、[未承認plan](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_plan/2026-10-19/plan_DRAFT.json)、[入力metadata](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_plan/2026-10-19/input_manifest_metadata.json)、[並列DAG](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_plan/2026-10-19/parallel_dag.json)、[判断表](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_plan/2026-10-19/decision_table.csv)。

## 1. 問いと固定対象

保存されたH6 full m5_best/p19/q2 cosineでは、raw差1.504043707445665e-9、共通位相1.5028193689500676e-9 rad、phase-aligned差6.067672593889436e-11が記録された。これらは前回結果の転記で、今回再計算していない。

次に知りたいのは、差が通常精度の回路作用・statevector演算に敏感なのか、保存gate列をより広い精度で作用させても残るのかである。共通位相の主成分を確認しただけでcompiler bugや数値丸めを確定しない。

対象はH6直鎖1 Å/STO-3G、保存tol-only1e-8 DF19、full PF/prefix19/q2、T=.8、P=0、固定binary64版 `PF_M5_BEST_20D64C2_BINARY64_V1`。symmetric-directional/full Hadamard cosineの既存1点・既存初期low状態だけ。HamiltonianやDF/stateを再生成しない。

## 2. 入力は12役、数値配列としては未読

今回行ったのは既存JSON metadataの確認とbyte SHA照合のみ。QPYをloadせず、NPZもNumPy配列として開いていない。役割とhashは公開manifestへ、実path対応は専用private準備領域へ保存した。

|役割|内容|
|---|---|
|wrapper_qpy / compiled_qpy|保存された2回路。新しい回路合成・transpileの入力にはしない|
|initial_npz / vectors_npz|同じ初期low、before/after、独立expected作用|
|3 receipt|QPY/NPZのSHA・global phase・compiler・dtype/shape|
|saved_diagnostics / saved_STOP|再現すべき診断値と元のFAIL|
|saved_completion / saved_exit|原run終了・reap・追加Phaseなし|
|recovery_config|固定SOURCE/runtime/input/controlの来歴のみ。旧grantを再使用しない|

QPY/NPZ本体とprivate traceback/configは公開しない。元SOURCE/runtimeの固定条件、Qiskit1.3.0・NumPy1.26.4を使う案。今後必要な新analysis SOURCE・runner・runtime closure・output pathは未sealであり、実行commandはnull、有効grantは0。[無効grant](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_plan/2026-10-19/grant_INVALID.json)はplan digestへ結合している。

## 3. 最小の1 Phaseと内部手順

将来承認を受ける単位は `SAVED_PHASE_DIAGNOSTIC` の一回実行。以下はその中の依存手順であり、結果を見てscopeを増やす別Phaseではない。Phase終了時には必ずSTOPする。

### GATE：保存証拠とfresh gate

新SOURCE/input/runtime/planと新grant・ledger・marker、未使用output、元STOP/reapを確認する。拒否gateは実際に使用するfileに限り、無関係worktreeを凍結しない。旧58軸を解析対象やfit用データへ拡張しない。

### SAVED_METRICS：保存値の再現

保存NPZのdtype/shape/finiteと、initial.npzのlowがvectors.npzのlowに一致することを確認する。保存before/afterからraw差・norm・内積・共通位相・phase-aligned残差・枝別差・Hadamard値を再現する。この実配列解析は**今回未実施、別途承認が必要**。

再現用の数値allowanceは絶対1e-12を技術案として提案するが、未採択。これは旧1e-9基準の緩和・置換ではない。実行前に新契約で固定し、再現しなければSTOPする。

### QPY_SCHEMA：保存2回路の構造確認

既存QPYを復元し、13 qubit・ancilla位置/順序・layout None・数値parameter・最後のmeasurementを確認する。除くのは最後のmeasurementだけで、中間measurementや未知gateを黙って落とさない。SOURCE由来のgate whitelist案をplanに明記した。

compiled側はrz/sx/x/cx、wrapper側はh/p/rz/rzz/cx/xx_plus_yy等の既存操作を対象にする。未知gate・symbolic parameter・想定外layoutではSTOP。transpile・decompose・一般unitary合成でwhitelistへ自動変換しない。

### 4 REPLAY：2回路×2精度、同じ初期状態

|job|作用するもの|精度・基線|
|---|---|---|
|WRAPPER_ORDINARY|保存wrapper QPY|元のQiskit Statevectorとphase適用規則|
|COMPILED_ORDINARY|保存compiled QPY|同上。保存afterの再現を確認|
|WRAPPER_EXTENDED|同じwrapperの保存gate列|独立した小gate解析式とlongdouble/clongdouble|
|COMPILED_EXTENDED|同じcompiledの保存gate列|同上|

最大4回のstatevector作用だけで、4種類の初期状態を新しく作るわけではない。通常精度の2本は、QPYの復元を含む経路が保存ベクトルを再現するかの対照である。これを省くと、拡張精度との差をQPY復元の差と区別しづらくなる。

4 jobはSAVED_METRICS/QPY_SCHEMA後に独立して並行できる。通常精度再現の合格が確認されるまで、全出力を診断用quarantineとし、拡張精度の結果だけで原因を選ばない。通常精度が再現しなければ新dispatchを止め、実行中のown jobの証拠は未受理として回収する。自動retryはしない。

### DIAGNOSTIC_REPORT：保存・比較・終了

raw差、各norm、complex overlap、共通位相、phase-aligned残差、ancilla枝別差/位相、Hadamard値、通常/拡張精度間の差を同じ表にする。旧raw FAIL・未受理費用は変更しない。exit/reap、SHA、使用した精度・gate実装を保存してmandatory STOP。

## 4. 拡張精度の条件と限界

独立engineは1/2 qubitの小matrix・bit-indexによるstatevector更新だけを行い、13 qubit全体のdense unitaryを構築しない。gate順序、qubit順序、保存されたbinary64の角度値、初期状態を保つ。浮動小数値をlongdoubleへ正確に移すのであり、元係数の桁を作り足したり、保存角度を最適化し直したりしない。

H・SX・X・CX・RZ/RZZ・P・XXPlusYY等を解析式で評価し、全体の保存global phaseも一回適用する。未知gateに対してbinary64の`to_matrix`へfallbackしない。phase fitting・再正規化・full-H referenceの位相合わせを受入のために行わない。

NumPyのlongdoubleは環境依存なので、実mantissaがbinary64より広いことを人工検証・起動前gateで確認する。同じ精度ならSTOPし、別backendやdtypeへ黙って切り替えない。拡張精度は無限精度ではなく、厳密な誤差上界や全状態unitary等価性のcertificateにもならない。

2回路の`global_phase`値だけを直接引いて原因とはしない。basis変換ではgate本体とphase metadataの間へ位相が移るため、gate列の作用と一緒に評価する必要がある。

## 5. 結果の読み方とSTOP

[判断表](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_plan/2026-10-19/decision_table.csv)の要点は次のとおり。数値条件は実行前に固定し、結果を見て有利な精度・測定規則を選ばない。

|観測|許される解釈|そのままは言えないこと|
|---|---|---|
|通常精度で保存値を再現しない|再生・QPY復元・runtimeの問題が未解決|compilerの正誤や拡張精度の優越|
|通常は再現し、拡張精度でraw差が小さくなる|演算/精度依存が示唆される|丸めだけが唯一の原因、旧FAILの解消|
|拡張精度でもraw位相差が残り、位相を外した差は小さい|保存gate列/phase帳簿に共通位相成分の差が残る可能性|全状態で等価、compiler bug確定、位相補正した回路を採用してよい|
|拡張精度でも非共通位相の差が残る|gate/parameter/order等の調査が必要|全てをglobal phaseとして無視できる|
|未知gate・精度不足・nonfinite|解析実装の対象範囲不足としてSTOP|黙った近似・backend置換・追加scope|

これらは連続指標に基づく限定的な説明で、原因の自動認定規則ではない。1つの保存初期状態での比較から、全入力状態へのunitary保証を主張しない。

hash/来歴不一致、想定外qubit/layout/measure、nonfinite、保存値再現不合格、拡張精度不足、未知gate、own process異常・実OOM、output/ledger/marker衝突はSTOP。host/上位cgroupの一時PSI/CPU競合だけではkillしない。元のSTOP・grant・ledgerを変更せず、新解析の成否と分離する。

## 6. 並列化と計算量の限定

解析対象は2回路・1初期状態・通常/拡張各2作用に固定する。4 workerは過去の資源capではなく、計画に4独立jobしかないための上限である。fresh affinity・idle CPU・RAM/diskとworkload proxyから実並列度を決める。

state更新をベクトル化し、同一gate/角度・bit-indexを正確な値のkeyでcacheできる。QPY/入力を既存byteから再利用し、sampling・新回路構築・新compile・parameter格子探索・巨大な全回路matrixを避ける。共有package/OS/cgroupを変更しない。今回、実workloadの時間測定や資源推定の数値fitはしていない。

## 7. 実装・実行準備で残ること

**runnerはまだ未実装で、今回の文書を実行準備完了とは呼ばない。** Codexの次の技術作業は、限定gate engine・保存入力adapter・one-shot認可・4job scheduler・出力schemaを実装し、次を人工データで確認することである。

- 各gate・逆操作・little-endian位置・global phase一回適用と相対phaseを含む人工回路の一致。
- QPY人工往復と保存NPZ再現、phase-only/relative-phase/norm-errorの切り分け。
- 実mantissa確認、未知gate/非finite/再現失敗のSTOP、precision fallbackなし。
- 新compile・sampling・候補追加・状態追加・位相受入・旧grant再使用の拒否。
- worker回収、未使用output/marker、入力/SOURCE/runtime封印、旧STOP保全。

人工検証・新SOURCE・入力hash・runtime・無効grant・fresh resource契約まで用意し、ユーザーに具体的な一回実行範囲を示す。これは今回の計画に続く実装段階であり、今回実施済みのtest数へ数えない。

**ユーザー（実行前の担当）**：完成した固定契約について、保存値数値解析と最大4 replayを一回実行するか承認する。n=48等の標本数や新seedはこの解析に不要。承認後にのみCodexがone-shotで実行する。

m5主比較採用AD03、raw基準の変更、費用再会計、sine compile、追加のm5探索、後続Coreは今回の計画にも自動的に含まれない。

## 8. 今回の成果と保全

計画JSON、並列DAG、12役の入力metadata、判断CSV、無効grant、構造検証・保全receiptを作成した。実QPY/NPZはbyte hashを確認しただけで、復元・数値解析・新simulation・compileは0。元2回のrunと元SOURCE/入力/runtime等を684 fileの保全一覧に結合し、科学SOURCE20 fileと旧science16 fileは不変。

資料は独立branchへlocal commitしてmandatory STOP。公開pushは行わない。今回の構造確認は作者自身の確認で、独立した科学レビューではない。
