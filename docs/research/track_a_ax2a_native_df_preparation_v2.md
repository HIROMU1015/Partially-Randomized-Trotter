# Track A AX-2A v2：native DF接続とAX-2B実行前仕様

公開時のリンク補修（2026-10-10）：未公開sourceへの参照は[公開依存関係](track_a_ax2b_gpt_review_index_v1.md)。原文と補修一覧を保存し、科学的主張・数値は変更していない。

2026-10-09 JST。`AX2A_NATIVE_PREPARED_AX2B_NOT_AUTHORIZED`。
利用者の「次の作業に進んで」を受け、v1に残したnative接続とpilot条件の具体化を行った。
研究方針は[AX-2A追補案v1](track_a_ax2a_research_amendment_v1.md)から変更しない。
本書を技術準備の最新入口とし、[技術準備v1](track_a_ax2a_technical_preparation_v1.md)、
旧57-test証拠・source・artifactはその時点の履歴としてbyte不変で保持する。
分子科学計算・trajectory sampling・compile・GPUは未実行。

## 1. native実装の接続

[ax2a_native_df.py](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)は、
既存`DFDeterministicOneBodySpec` / `DFDeterministicFragmentSpec`を入力とする。
既存Gaussian basisのappend、one-body/DF-squaredのdiagonal primitive生成、phase-aware
`simulate_statevector`、numeric circuit fingerprintを再利用し、共有科学sourceは変更しない。
Hamiltonian/integral/stateを作る機能、sampler、transpiler、科学runnerは含めない。

| 接続 | 今回できること | 未検証の範囲 |
|---|---|---|
| global決定論PF | 標準二次/四次、q反復、負時間、ordinary/directional制御 | H4分子での作用・資源、精度に応じたq選択 |
| partial S2 | B0とB2/B3 backbone、既に制御されたtailを順序通りに挿入 | H4/H6のrandom cost統計とfinite-signal正しさ |
| legacy request replay | `DFPartialS2StepRequest`から既存RTE event builderへ接続、phase/basis plan保持 | 分子の既存trajectoryとの照合 |
| state action | full Qiskit-order vectorへの単一block作用、非unit norm保持 | sector/JW順のbridgeと分子primitive証明 |
| measured wrapper | H/U/H、またはH/U/Sdg/H、最後にancilla測定 | 実compile費用・sampling統計 |

入力tailは既に`diag(I,U_event_sequence)`として正しいことをcaller/旧RTE builderが保証する。
低level APIのGate構造検査だけではordinary-controlの意味論を証明しない。
replay経路は新しい`make_*_request`やsamplerを呼ばず、明示したeventだけを旧builderへ渡す。
finite corrected meanの非unitary numeratorを量子回路として挿入しない。
scalarは`constant + extracted identity`を一度だけordinary-controlled phaseで加える。

global四次は対称二次pieceをYoshidaの`w1,w0,w1`でcomposeする。w0は負。
両control実装とも同じ未mergeのpieceを用いるので、同じ近似作用を比較できる。
source-v1のmerged state-action iteratorとは数学的に同じ作用である。
全体partialの四次化は追加していない。標準四次はglobal決定論対照である。

## 2. directional diagonalの位相

diagonalを `D(t)=exp(iφ) Π RZ(θ) Π RZZ(η)` と表す。
directionalは `diag(D(−t),D(t))` を実装する。
RZはancillaをparityに含めた`CX–RZ(−θ)–CX`、
RZZはsystem parityとancilla parityを作り、`RZ(−η)`後にundoする。
diagonal global phase φはancillaの`RZ(2φ)`であり、除去しない。

forward halfは無制御、reverse halfはdirectional。
control=0では各halfが逆順で相殺し、control=1では従来S2と同じ作用になる。
中央RTE eventと独立scalarのordinary controlは保つ。
これは[Simon–Loveの対称controlled構成](https://arxiv.org/html/2511.13855v1)をDF primitiveへ接続したもの。
任意角rotation数の理論と、compiler後full-wrapper RZ-workの比率は別である。
今回の実装一致から費用半減・PR利益・baseline順位は主張しない。

## 3. wrapperとcompilerの対応

既存Hadamard builderのcircuit factoryを使い、既存と同じgate順と`rpe_measure`を保った。
system state準備、二重制御、量子shot、backend実行は追加しない。
新しいresultを旧`DFPartialS2RepeatedCircuitResult`に偽装せず、別型`NativeEvolution`として扱う。
wrapper作成時にはnumeric circuit fingerprintを再照合し、作成後の改変を拒否する。

将来のcompile入口は既存`rte_compiled_cost.transpile_and_measure_cost(wrapper,compiler,...)`。
その入力は完全な測定付きwrapperと実fingerprint、compiler identityである。
古いtyped cost providerへ新resultをそのまま投入しない。
新しい科学runnerには明示したschema・task identity・cap enforcement・出力照合が必要で、まだ実装していない。
今回transpilerを一度も呼んでいない。

## 4. synthetic証拠

[native tests](track_a_ax2b_gpt_review_index_v1.md#unpublished-source) 53件、
[preflight tests](track_a_ax2b_gpt_review_index_v1.md#unpublished-source) 7件、
旧57件を合わせて **117 passed**、local・未commit、immutable CIではない。

固定した2 orbital Fock空間に対し、Qiskit bit順のoccupation energyから独立に参照matrixを作った。
複素Gaussian basis、正負時間、one-bodyとDF-squared、全basis column、両control branchを確認した。
global二次/四次は非可換な3 blockでcompositionを照合。
partialは異なる中央event・event phaseの順序、空tail、scalar-only、B0を照合した。
既存request replayはprefix 0/1/2のtoyで、samplingせず、列挙したK0/K2 eventを使用した。
native callback→finite K2/K6 corrected/raw接続と非unit norm保持も確認した。
wrapperの実部/虚部の符号・測定qubit、作成前instruction cap、改変検出を確認した。

testsでは科学artifact/NPZ/runtimeのopen、`np.load`、sampler、transpilerを禁止した。
toy circuitのOperator/Statevectorによる数学的検査は実施したが、科学回路compileや量子shotではない。
[準備v2 artifact](../../artifacts/resource_applicability/track_a_ax2a_native_preparation/2026-10-09/)
にJUnit、source hash、preflight、保護照合を収録する。

## 5. AX-2Bの具体的task草案

[preflight module](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)はstdlibのみ。
[metadata writer](track_a_ax2b_gpt_review_index_v1.md#unpublished-source)は新規outputへのexclusive createで、
launch optionを持たない。
草案は結果・モデル順位を入力にせず、H4 correctnessを先に通し、H6は6構造cellに限る。

H4 correctnessは1.00 Å/STO-3G/legacy rank12/T=0.8の同じ保存stateを提案する。
B1二次q1/q4、B0 prefix6/q4、B2 prefix6/q4/R8/K2およびK4、
B3 prefix0/q4/R8/K6、B1標準四次q1/q4の8 cell。
H4 primitive/basis/phase、B0 exact truncated signalとsigned分解、finite meanの一致を確認する。
旧rank12結果と共通DF政策は混同しない。

H6はv1のB1二次/四次×q1/q8、B2 fraction1/2・q=R=1・K2、B3 fraction0・q8/R64・K6の6 cell。
1.00 Å/STO-3G/T=0.8、共通positive DF tol `1e-8`を**提案**する。
tolはまだ採用していない。実rank・target/state・DF pipeline保証は未確定。
H6 snapshot生成が必要なら、その作成自体を別の明示認可範囲に含める必要がある。
H8 taskは0件。

| full-wrapper compileの草案 | deterministic | random | 計 |
|---|---|---|---|
| H4 | 四次q1/q4＋B0の3 cell ×2 control ×2 axis =12 | B2/B3 ×2 trajectory ×2 control ×2 axis =16 | 28 |
| H6 | B1の4 cell ×2 control ×2 axis =16 | B2/B3 ×2 trajectory ×2 control ×2 axis =16 | 32 |
| 合計 | 28 | 32 | **60 ≤ 上限案64** |

random cellの2 trajectoryは負荷・実装を見るための標本で、費用順位を確定する量ではない。
同じ明示trajectoryを両axis・両control実装へ使い、比較のための無関係な再samplingを避ける。
seedはcell ID/replicaのhashから決め、8つのrandom trajectoryを草案に列挙し、衝突を拒否する。
wrapper重複をcacheで省略できても、実build/compile callを別に数える。

## 6. 上限案とpreflight

v1のCPU1/process1/各BLAS1、GPUなし、address-space 8 GiB、H4 bundle900s・H6 cell1800s、
全wall14,400s、output512 MiB、compile64、random cost標本2を維持する。
追加案はwrapperごとuntranspiled Qiskit instruction 100万、transpiled instruction 500万、
signalごとdeterministic action 10万、ground solver/system 2万matvec、reference/action 2万matvec。
finite tailは一経路448、corrected+raw896を維持する。
これらは科学的十分性を保証する値ではなく、限定pilotのengineering cap案である。

native builderはstage・basis・diagonalの構造上界を**schedule/circuit allocation前**に検査し、
作成後にも実instruction数を照合する。空blockの巨大qはscalar-onlyとして処理する。
この上界はQiskit instruction数で、Gaussian gate内部のnative expansionを数えた費用ではない。
compiler中のメモリ/時間は将来runnerの子process/watchdogで制限する必要がある。
transpiled-size cap、matvec cap、総compile/disk/wallの実行時enforcementはrunner実装に残る。

preflightはidentityの**存在**だけを検査する。placeholderが埋まっても科学的証明にしない。
全fieldが埋まっても`launch_allowed=false`、`science_authorized=false`、`mandatory_stop=true`を返す。
不足はH4/H6 snapshot/state、採用DF政策、primitive/basis証明、reference allowance、
compiler、実CPU/RAM割当、科学source/task freeze、runner/cap enforcement、別の明示実行認可。
hostの検出CPU/RAMを実割当や許可と呼ばない。

## 7. GO/STOPと次の一段階

今回の技術GOはnative loweringのtoy作用一致とmodel-independent task/cap草案まで。
H4/H6 molecular correctness、数値headroom、compile費用、最悪回路長・wall/RSSは未検証。
H6 profile値は全てdevelopmentで、H8独立性は保持する。
PR利益、費用モデルの改善、H6/H8移送性能について新しい結論は出さない。

次は bounded科学runnerとcap enforcement、H4 snapshot/sector bridge/referenceの実行前固定。
科学計算を開始する段階ではtask identity・対象・実割当・STOP条件を埋め、明示認可を別途受ける。
H4一致が通る前にH6へ進めず、失敗/上限/数値未確定ならpartial evidenceを残してSTOPする。
自動retry/resume、予算拡大、H8利用、旧結果/source/manifest変更は行わない。
