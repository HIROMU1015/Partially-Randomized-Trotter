# Controlled RTE・Hadamard局所probe v2：selector修正と1回の再実行

2026-11-04 JST。今回の修正・再実行指示の範囲で、新SOURCEをcommitへ固定してからrun02を1回実行する。
旧SOURCE `76d25ac69c22da6efeec673dc36c5fe04b71533f`、
旧停止結果 `a6d1a2945e0afd5b4ae5200c4572395c29fb4850`とrun01を変更しない。
このv2の後に別profile・precision・namespaceへ自動的に進まない。

## 修正点と固定selector

run01では旧B由来ID2個が保存cutoff1e-12 payloadに存在せず、回路を構築する前に停止した。
今回は保存m17 exportのID/sign/basisだけから、次の規則で2個を結果前に固定した。

1. component IDで昇順に並べ、最初のsign−1をc0とする。
2. c0と異なるbasis_idを持つsign＋1のうち、最初のIDをc1とする。
3. その完全metadataを[selector.json](selector.json)へ固定し、将来の実行時に他候補へ選び直さない。

| 役割 | 固定ID | sign | support |
|---|---|---:|---|
| c0 | df-fragment-3:Z0Z2 | −1 | [0,2] |
| c1 | df-fragment-4:Z0Z1 | ＋1 | [0,1] |

係数・確率・basis IDもselector.jsonに保存した。出典exportのSHA256は
`ad7fd3617a524207fbc86a77de471ed6222b010046822a11bd9675edcace1f08`。
signal/operatorを計算して有利なpairを選んだものではない。
異なるbasisだけで実H4 pairの非可換性を証明したとはしない。

defaultの静的検査は、実保存218 componentにこのpairが存在し、sign/basis/support/係数/確率が固定metadataと
一致することを確認する。実H4 load/preparationより前にこの検査を完了する。
runtime preparationにも同じmetadataとbasis hashを要求する。

## 科学条件・12 profile・参照

以下は[旧固定範囲](../rpe_controlled_hadamard_runner_preparation_20261104/EXECUTION_SCOPE.md)から維持する。

- linear H4／隣接1.00Å／STO-3G／保存legacy DF12、8 system qubit＋ancilla8、
  B physical-state normalized ray・4電子sector70、Partial L_D3、δ=.02、K2、
  extract_identity_phase、coefficient_atol=1e-12。
- 親m0,8,10,11,13,17、r1,2,4,8,16,32。親q/N/α/β/εは保存literalのmetadataとして保持する。
  実際のprobeは各rのq_probe1/all0とq_probe2/stress、計12本。親の大きいqでは回路を構築しない。
- all0はrotation c0/c1のorder0 eventを交互にする。
  stress/r1はT−,T+、stress/r≥2は2rの交互列のstep0先頭をT−、step0末尾とstep1末尾をT+へ置換する。
  T−はrotation c0/products c1,c0、T+はrotation c1/products c1,c1、両方order2・event.phase=−1。
  旧コードのZ0/Z1というローカル変数はprofile labelであり、今回のPauli supportがsingleton Zであるという意味ではない。
- 保存signal側τをhex完全一致で結び、θ=atan(τ/(n+1))をparameter replayで生成する。
  6群のτ/θ hexをparameters.jsonへ保存してから参照行列・回路を構成する。
- 96-bit Acbのanalytic basis＋occupation energyによるF/Rとsigned component参照を使用する。
  production diagonal primitive generatorやcontrolled出力から理論signalを作らない。
- uncontrolled raw、controlled raw、controlled boundary_optimized（support_run_le_1）を照合する。
  control両branchとcosine=Re／sine=+Imを、位相alignmentなし・参照不確かさ込み上端≤1e-10で比較する。
  Statevectorには各actionで配列copyを渡す。

入力/H/state/preparation/tail hash、保存basis descriptors、sector/endian、training fingerprintは旧範囲から変更しない。
新SOURCEはこのv2ディレクトリに分離し、旧production module・旧runner・rawを編集しない。

## 実行call・資源・出力

| top-level操作 | 今回の上限 |
|---|---:|
| H load／state load／DF preparation／support登録 | 各1 |
| finite config/distribution parameter replay | 6群 |
| 参照F/R pair／選定signed component | 1組／2個 |
| 参照trajectory vector＋signal | 12 |
| uncontrolled raw build／action | 各12 |
| controlled raw build／boundary_optimized build | 各12 |
| controlled branch action | 48 |
| measured wrapper build／ideal mean action | 各24 |
| CDF／sampling／compile／quantum shot／q_probe≥4回路 | 0 |
| H/state solve／SCF／DF再因子化／全18平均・成功率・費用再解析／noise・状態準備 | 0 |

Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、python-flint0.9.0/FLINT3.6.0、Python3.11、96 bit。
6r群を利用可能CPUと群数に合わせて並列化し、worker BLAS1 thread、参照の小行列はFLINT1 thread。
CPU/RAM/worker/phase/total wallに人工上限を設けず、実際の空き資源・quota・共有環境への非干渉を優先する。
CPU負荷が高ければ自jobの優先度を下げる。共有設定・他job・SERVER2 H8を操作しない。

```bash
<research-python> docs/research/rpe_controlled_hadamard_runner_v2_20261104/run_local_probes.py \
  --run --output-dir artifacts/rpe_controlled_hadamard_local_probes/2026-11-04/run02
```

fresh run02にparameters.json、RESULT.json、r別vectors.npz最大6個、計最大8ファイルを保存する。
実行前に新SOURCE commitを固定し、原run01と旧SOURCEのhashを保全する。
今回の修正・再実行許可の範囲でこの1回を行い、許可の証明・新grant・承認文保存は要求しない。

## STOPと解釈

固定SOURCE/input/hash/tuple/selector/τ不一致、nonfinite、参照半径不足、1e-10不合格でSTOP。
failed probeのraw/residualと開始call数を保全し、進行中の他workerの結果を回収する。
失敗後にselector/条件/precisionを変えて自動retryしない。今回の1batch完了後もmandatory STOP。

PASSでも固定12条件の局所整合性に限り、全component・任意trajectory・保存全18q、
長qのscalar/boundary phase転送、PCG64/IID/guard、compiled/noise/実状態準備は保証しない。
旧V1/V2/V3/R1/B/m17の条件付き判定を変更しない。
全体RPE_tolerance_certified=false、scientific_status=UNDETERMINED、U_exec=nullを維持する。
