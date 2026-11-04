# Backend比較・残存gap・次の最小計画

2026-11-05 JST。今回は技術準備だけで、backend採択・新H4科学SOURCE/runnerの起動準備は行わない。
固定条件と保証対象は[対応表](INTERFACE_SOURCE_MAP.md)、意味論は[callback契約](CALLBACK_CONTRACT.md)。

## 1. 既存backend候補の適合性

| 候補SOURCE | 実装済みの範囲 | 今回の条件との距離・最小変更 |
|---|---|---|
| repeated logical Qiskit builder＋Hadamard wrapper | 与えたDFPartialS2RepeatedRequestからcontrolled evolution、Re/+Im wrapper、1測定clbitを構築。shot/reset/stateは外部 | 最も直接に回路意味論へ対応する接続候補。standard wrapperのq1/2/4制限、run_le_1の明示適用、state/bit schema/shot providerが必要。backend未採択 |
| benchmark wrapper | `QiskitRPEHadamardBenchmarkCircuitBuilder(maximum_repetition_count=...)`は指定上限内のdyadic qを受理し同じ`_build_validated`を使う | **validation-only** API。q131072を受理させることは長qのbuild/compile/実行可能性やcorrectnessの証拠ではない。巨大回路を既定路線にしない |
| 既存local controlled probe | `probe_adapter.evaluate_group`はStatevectorと独立参照で固定r・q_probe1/2のcontrolled branch/wrapper平均を検査。policy・scalar位相を明示 | 保存12 probeの局所証拠は再利用可能。量子shot backendでも全18round callerでもない。新callbackの対照方法には適するが、そのままproduction measure_oneではない |
| sector state-action diagnostic | `rpe_hadamard_failure_validation._trajectory_signal`はF→events→Rのq*r順序でcomplex信号を返す | controlled wrapperの実測定ではない。`_explicit_fresh_iid_validation`全体はH/state再生成、既定cutoff、独自seed/float probability/clip、full component actionを含み、保存ray/run_le_1のdrop-inではない。全runnerは今回起動しない |
| fresh-IID仮想測定診断A | `fresh_iid_adapter.one_shot`はflat q*r sampler→signal_fn→条件付きBorn確率→PCG64 outcomeを返す。組込みint outcomeは取り出せる | q1/q8・δ.02・r1の旧診断を全18へ移送しない。callback分離、選択basis/circuit、浮動小数点signalとclip/abort、seed streamが別契約。停止済みvirtual float保証を再開しない |
| 実機/compiled/noisy | 上の回路を受理する外部serviceの可能性と、このrepoで証拠のあるproduction測定を区別 | 今回採択/接続/精度認定しない。実状態準備・noise・compiler誤差・費用は別研究範囲 |

## 2. 実装保証へのgap分類

| 接続 | 現在の証拠 | 必要な追加作業 | 分類 |
|---|---|---|---|
| Event law | 保存CDFに対する独立uniform53 law、NumPy choiceの同じbin規則 | 固定component/order入力とversion/CDF bindingを維持。PCG64独立性は未証明 | SOURCE接続＋数学的仮定。PCG64の新研究は範囲外 |
| Trajectory | q/r/K/δ metadataは全18で一致、event/control/source実装あり | fixed context・signal τ・per-step run_le_1・construction policyをfactoryに明示接続 | 既存SOURCE接続＋最小callback glue |
| Fresh-per-shot | interfaceは各axis/shotにfactoryを呼ぶ | providerのdraw/reuse/cache条件とmeasurement resetを実装に束縛 | 技術実装＋backend証拠。call数だけではIID証明にならない |
| Hadamard意味論 | logical Re/+Im・scalar/control符号、旧局所/平均証拠 | 選択backendのstate/endianness/clbit/action・数値近似の接続 | callback glue＋必要なら限定科学検証 |
| 平均生成 | built-in int契約ならexact Sと単一binary64除算の条件付き旧補題 | provider出力型の強制、RN等の前提。異型値の暗黙受理を避ける | 最小型検査＋既存丸め条件のbinding |
| Phase入力 | cmath.phaseを呼ぶSOURCEが存在 | 成功域全体のmean→cmath Arg→wrap誤差a_m | 新しい数学的証拠が必要。今回kernel研究はSTOP維持 |
| Decoder | 数学的nearest-branchの17 strict条件、同じscalar SOURCE | 全root対応/c_m、全computed distanceのd_m、全候補順序・有限性。近傍2候補だけの比較へ短絡しない | 数学的証拠が必要。H4科学SOURCE未準備 |
| Energyと完了 | 数学的decoder＋alias条件→保存DF sector ground上界 | Float64 decoder、diagnosticのG全域での完了、最終float除算、alias/liftへの誤差接続 | 証明後にのみ限定科学検証を計画 |
| 実物理系・費用 | 今回の保存DF sectorと異なる対象 | upstream/model/DF誤差、他sector、実状態準備、compiled/noise/総費用 | 現行範囲外 |

SOURCEの接続だけで、数学的a/c/dや固定PCG64のIIDが取得されたとは認定しない。
既存の95%以上の条件付き成功保証を実装energy成功保証へ移送するには、G上の
**全域での決定論的数値接続とbackend条件**が必要である。

## 3. 次の最小実装計画（未採択）

次は**まずbackendと具体callback bindingの選択**。logical Qiskit＋既存local Statevector評価を
意味論対照の第一候補として提示するが、正式採択はしない。
現task内で新しいproduction backendを作る必要はない。

### 技術stage T

使用する既存SOURCEはload/collect/decode、make_rte_configまたは既存signal_configの照合、
make_df_partial_s2_repeated_request、apply_run_threshold_policy、controlled builder/wrapper。
全版は[SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)と同じに固定する。

新しい関数群やsamplerを作る代わりに、必要な差分は以下に限る。

1. provider closureへH/state/law/export/preparationと固定parameter contextを束縛し、
   requestを既存repeated requestへ接続するfactory。controlled=True/ancilla8とper-step planを明示する。
2. 選択backendの1-shot result schemaを既存`measure_one`へ接続し、
   生bitの位置/値検査と組込みint±1の型境界を設ける。decoder/collectの集約式は変えない。
3. 設定18件を順にcollectし、18個のMeasuredMeansだけを同順のdecodeへ渡す薄いcaller。
   専用実行entryを公開する前に、SOURCE/input/backend/実行条件とSTOPを固定する。

factory/measurementのI/O・例外・freshness/cache契約はCALLBACK_CONTRACT通り。
型の具体bindingは選択backendで固定し、その後はimplicit別型へ切り替えない。
Input identityは旧18round表、V2/model、H/state NPZ、218成分export、保存law、
support training fingerprintとconstruction policyを既存証拠へ結び付ける。新H/stateを生成して補わない。

想定範囲は**人工fixtureだけ**。全18件のmetadata伝播、per-step policy、callback型/bit schema、
各shot call、例外/欠測/順序違反STOP、同じscalar decoderへの入力shapeを検証する。
数式の成功率・energy、全域kernel誤差はこのstageの判定に含めない。

### 新しい科学検証が必要になった場合だけのstage S案

新callbackの浮動小数点state-action/測定平均を既存ideal意味論へ接続するなら別途科学評価が必要。
その前にcallback SOURCEと参照・精度・科学call数を固定・公開し、ユーザーの別途明示承認を得る。
**本書はその科学実行契約でも、起動可能なrunnerでもない。未取得のprecision/call範囲を架空に確定しない。**

最初から18roundの新trajectory samplingや巨大qの回路build/compileを既定にしない。
既存12局所probeの入力・原参照値を再利用し、新callbackの接続差だけを見る案を優先する。
最小候補はq_probe1/r1とq_probe2/r32の保存profile各1件、両axisの条件付き平均（計2request/4軸）。
後者は親m17のr32を使う**q_probe2**であり、実際のm17 q131072の実行として報告しない。
order2・sign・非可換順序・step境界policyが新callbackにも同じに渡るかを対象とし、
採用profileが必要coverageを持つことをSOURCE固定時に確認する。
これは候補であり、今回新requestを生成/評価したという意味ではない。
実shot sampling、成功率の再評価、既存mean/certificate/幾何/referenceの再計算は提案に含めない。
この小さい数値対照が通っても、全18のFloat64/IID/backend保証へ自動拡張しない。

## 4. 成功・失敗・STOP

技術的な接続成功は、固定setting/contextが保全され、必要callbackが毎shot呼ばれ、
source policy/型/bit意味論が一致し、人工異常が例外または既存statusで停止すること。
科学的95%成功認定とは別である。

以下でSTOPする：SOURCE/input/shape/sign/policy/dtype不一致、欠測、callback失敗、
zero/tie/alias診断、採択していないbackend/domainへの拡張が必要な場合。
科学計算・kernel証明・PCG64 IID研究・compiled/noisy/state-preparation/費用へ拡張が必要なら、
不足点を示して別範囲の明示許可を待つ。仮の誤差0や任意allowanceで通さない。
今回の公開照合後もmandatory STOPし、自動retry/resume・次stage・詳細科学レビューを開始しない。
