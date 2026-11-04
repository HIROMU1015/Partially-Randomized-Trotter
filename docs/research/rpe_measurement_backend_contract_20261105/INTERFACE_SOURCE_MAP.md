# H4 RPE 18round：既存測定interfaceの実経路

2026-11-05 JST / LOCAL。**静的調査・metadata照合・人工テストのみ、新H4科学実行0。**
baseは`d4c016960ff4b1ca16cc73770147cde326989427`。
正式共通運用指示2026-10-25版§8.1/§8.4/§10を確認し適用した。
既存interface・scalar decoder・sampler・backendを変更/置換/正式採択しない。

固定対象はlinear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D3、
binary64 δ=.02、K2、8 system qubit／4電子sector70、保存physical normalized ray、
m0..17・q=2^m・保存r/N/α/β/ε/μ₀/E_ref。
`extract_identity_phase`、cutoff1e-12、保存component/basis/sign、run_le_1、uniform53 lawを維持する。
初期状態準備コストは対象外。

## 版と根拠

| 根拠 | 固定版・使用範囲 |
|---|---|
| 測定interface | [`8722b5a`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8722b5a461c617b8846c2754275895f152df78f3/docs/research/rpe_18round_measurement_connection_20261103/measurement_adapter.py)：load/collect/decode/AST loader。現treeとbyte同一 |
| 既存接続報告 | [同commitの報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8722b5a461c617b8846c2754275895f152df78f3/docs/research/rpe_18round_measurement_connection_20261103/IMPLEMENTATION_CONNECTION_REPORT.md)。旧計画と当時の未保証を時点記録として扱う |
| sampler/builder/wrapper/decoder | carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`。対象10 libraryとdecoderは現treeでbyte同一 |
| 現在の保証 | [`d4c0169`報告](../rpe_controlled_iid_success_energy_execution_20261105/REPORT.md)、SOURCE `ffa1e22b3001ad1c4eb275ff1121ed17111452ec`。指定ideal IID＋数学的decoder＋alias条件のみ |
| 平均と回路 | 平均結果`1789a8ca731a1de631e80a34d2b8c90657d24458`／SOURCE `7a39a78eccdbe457786978e14d36b2f0d7abc290`、旧一様結果`b0c354ce071af71a6ce857b11f8acc0c129a05b8`。旧18/36判定を保持 |
| Float64調査 | [`ccbd7aa`追補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccbd7aa974eda1e017138066a70dc8ac35221907/docs/research/rpe_three_gaps_preparation_20261104/INPUT_KERNEL_ADDENDUM_20261105.md)。cmath経路・整数和条件を引用。kernel研究は再開しない |
| Dの数学的含意 | [`89cb94f`結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f/docs/research/rpe_branch_energy_execution_20261104/REPORT.md)。同じG→branch/alias/energyを再利用する既存証拠 |

固定commit/path/blob/SHA256とcurrent照合対象は[SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)。
必須のGitHub固定SOURCE・結果をconnectorで実際に取得し、local immutable blobとも照合した。
旧kernel追補の内容をこのtreeへmergeしていない。

## 実際の呼出経路

```text
load_fixed_rounds(3) -> tuple[RoundSettings, ...]（18設定）
  [外部callerが m=0..17 順に呼ぶ。専用production callerは未接続]
  collect_virtual_round(settings, trajectory_factory, measure_one)
    axis=cosine,sine / shot=0..N_axis-1
      MeasurementRequest(settings, axis, shot, (ld,m,axis,shot))
      trajectory_factory(request) -> opaque trajectory
      measure_one(request, trajectory) -> outcome
      total=0; total+=outcome; total/N -> MeasuredMeans
  decode_measurements(rounds, measured_means)
    complex(cosine,sine) -> cmath.phase -> list[float]
    original_decoder() -> unchanged scalar reconstruct_rpe_phase
    diagnostic status -> energy=-final_phase/delta or None
```

controlled evolutionとHadamard測定は**既存collectの内部実装ではなく、外部callbackの責務**。
下表のcallback欄は接続候補SOURCEであり、productionへの接続済みという意味ではない。

| 段階 | SOURCE・関数と型/shape | 実装・条件・不足 | 最小の接続作業 |
|---|---|---|---|
| 固定設定 | `measurement_adapter.load_fixed_rounds(3)` → immutable tuple長18、各RoundSettings | V2 CSVと元opt2 modelのq/r/K/N/αを照合。今回、最新固定round表の全β/ε/δ/stateも一致を確認 | 既存loaderをそのまま使用 |
| 全round caller | `collect_virtual_round`を18回、その結果を`decode_measurements`へ渡すcaller | interfaceは接続可能。productionの全round orchestrationは未接続 | backend選択後、順序・結果数を保つ薄いcallerだけを接続 |
| shot request | MeasurementRequest、axis文字列、shot組込みint、key 4-tuple、event数=q*r | 1 axisにつきN callback。q/r/K/δはsettingsを介して渡る。具体trajectory内容は検査しない | callback closureに固定SOURCE/input contextを束縛 |
| trajectory生成 | 候補`make_df_partial_s2_repeated_request` → DFPartialS2RepeatedRequest、q個のoccurrence×r events | 既存samplerはある。seed規則・support plan・構成policyは自動で最新保証に一致しない | [callback契約](CALLBACK_CONTRACT.md)の検査/接続 |
| controlled evolution | `QiskitDFPartialS2RepeatedCircuitBuilder.build` → result.circuitとmetadata | 既存実装あり。controlled=True/ancilla8、固定phase/basis/policyが必要。shotを実行しない | backend側で適合するrequestを渡す。長qの直接buildを既定にしない |
| Hadamard wrapper | `QiskitRPEHadamardInterrogationBuilder.build` → 9-qubit circuit＋1 clbit | 標準公開domainはq=1,2,4。H/control/Sdg/Hの意味は既存人工検証あり。状態準備・shotsなし | backend domainを明示。benchmark拡張APIをproduction認定へ読み替えない |
| 1 outcome | `measure_one(request,T)` → **組込みintの±1** | 既存annotationはintだがbodyはfloat±1や一部NumPy型も受理。測定意味論/IID/resetは外部条件 | backend境界でstrict型・bit schemaを検査する必要 |
| 平均 | Python整数total、`total/N` → Python float 2個 | 組込みint契約ならexact sum。異dtypeでは同じ証明を自動適用できない | 既存集約式を維持し型条件を確定 |
| phase | `MeasuredMeans.signal` → Python complex、`cmath.phase` → float list | math Argとは別。実装a_mは未保証。ゼロは先に停止 | 既存cmath経路を記録し、np.angleの証拠を代用しない |
| decoder | `original_decoder`がcarrierの3 FunctionDefをAST無変更で読込み。scalar callerから内部に全q候補arrayを作る | root/wrap/distance/argminの全域誤差は未保証 | SOURCEを変更せず旧STOPを維持 |
| energy | `-final/delta` → float またはNone | 符号は正しい規約。aliasは外部条件、float除算とdiagnostic完了も保証が必要 | 数学的energy上端をそのまま実装energyへ移送しない |

RoundSettingsはμ₀/E_ref、Hamiltonian hash、cutoff、law、support/basis plan、controlled指定、
backend、状態resetを保持しない。これらは固定context側の契約で補う必要がある。
decoderにE_refを渡してbranchを選ばない。Partialだけが今回の対象で、旧Full経路は変更しない。

## 保証との距離

既存条件付き失敗上界≤0.0496929886343339467、成功下界≥0.950307011365666053、
reference込みenergy上端≤0.000138689200360119854 Haは**既存結果の引用**。
今回はt/確率/集約/energy/平均/certificateを再評価していない。
これらはcallbackが存在するだけでは実装へ移らない。
詳細な数値経路と不足は[CALLBACK_CONTRACT.md](CALLBACK_CONTRACT.md)、
[次の最小計画](VALIDATION_GAPS_AND_NEXT_PLAN.md)を参照。
