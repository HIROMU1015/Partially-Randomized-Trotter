# T0 interface・仕様

2026-11-09 JST。本体：[track_d_native_resources.py](../../../src/trottertracks/track_d_native_resources.py)。既存C7-2のrecurrenceと固定設計mockのmax-plus row/productを適応した。旧科学コードは変更していない。

## Native wordと計数

```python
from trottertracks.track_d_native_resources import (
    NativeInstruction, NativeCounter, NativeTemplate, count_native,
    remap_native, iter_qiskit_native,
)

def word():
    yield NativeInstruction("rz", (0,), params=(0.17,))
    yield NativeInstruction("cx", (0, 1))
    yield NativeInstruction("measure", (1,), (0,))

result = count_native(word(), num_qubits=2, num_clbits=1, global_phase=0.37)
assert result["qualification"] == "NATIVE_EXACT"
assert result["p0_qualification"] == "P0_UNQUALIFIED"

# 継続して届くchunkを、一つのcounterへ渡せる。
counter = NativeCounter(2, 1)
counter.extend(word())
snapshot = counter.result()
```

`NativeInstruction(name, qubits, clbits=(), params=(), condition=None)`はimmutable。indexはqubit/classical各register内の非負整数。rzは1q・1有限実parameter、sx/xは1q、cxは異なる2q、measureは1q/1cである。RZ(0)や連続CX対も入力wordにあるまま数え、最適化しない。`id`は今回のP0 basisに含めず、identity templateは空wordで表す。

`NativeCounter.feed`は一命令、`extend`は一回限りのiteratorを逐次消費する。全wordのlist化や二度読みはない。unsupported opcode、arity、重複wire、範囲外index、条件、symbolic/非有限parameterは例外となる。途中で入力が失敗したcounterには既に処理したprefixだけが残る。`result`はそのprefixのsnapshotで、元の全streamを完了したという資格ではない。`count_native`は正常に読み終えた場合だけ結果を返す。

出力は`metrics`（既存metricsと同じ9項目）、`logical_width`、`classical_width`、gate別`counts`、指標別`wire_arrivals`、二つの資格表示である。logical widthは宣言したqubit数であり、未使用wireを除去しない。global phaseは別metadataでsize/countに加えない。

## Arrival・transfer

arrival dictのkeyは`rz_depth`, `cx_depth`, `total_depth`の三つ。各vectorは長さW=nq+ncの非負整数で、qubitsの後にclbitsを並べる。省略時は零入力。出力depthはmax(exit)で、非一様入力では入力由来の待ち時間を含む累積dependency levelである。

各命令のwire最大値＋重みを全対象wireへ伝播する。RZ/CX filtered depthでは他gateの重み0だが同期は残す。measurementの量子／古典wireも同期する。totalでは全対応命令の重み1。gate duration・routing・noiseは扱わない。

```python
start = {key: (7, 0, 2) for key in ("rz_depth", "cx_depth", "total_depth")}
t = NativeTemplate.from_stream(word(), 2, 1, global_phase=0.37)
assert t.result(start) == count_native(word(), 2, 1, arrivals=start, global_phase=0.37)

# 一度に計数しながらtransferも必要な場合。
c = NativeCounter(2, 1, arrivals=start, capture_transfer=True)
c.extend(word())
assert c.template().result(start) == c.result()
```

`NativeTemplate`は三種類のimmutable matrix、count tuple、register幅、global phaseを保持する。`None`はmax-plusの−∞で、整数0と区別する。`apply(arrivals)`は全wire exit、`result(arrivals)`はcountとdepthを返す。通常counterは行列を持たず、`capture_transfer=True`を明示した場合だけ保持する。captureされたtransfer自体は初期arrivalに依存しない。

`first.then(following)`は時間順first→following、すなわちM_following⊗M_firstを計算する。両方を同じ宣言registerへembedしてから合成する。gate countは別に加算し、P0の境界簡約はしない。empty templateは恒等行列。phaseの合算はbinary64 metadataの和で、厳密な量子位相保証やchunk間のbitwise位相加算保証ではない。**整数count/depthのchunk不変性とは分ける。**

## Lane・wire mapping・actual inverse

```python
a = NativeTemplate.from_stream((NativeInstruction("x", (0,)),), 1)
b = NativeTemplate.from_stream((NativeInstruction("rz", (0,), params=(0.2,)),), 1)
left = a.embed(3, qubit_map=(0,))
right = b.embed(3, qubit_map=(1,))
shell = NativeTemplate.from_stream((NativeInstruction("cx", (2, 0)),
                                    NativeInstruction("cx", (2, 1))), 3)
combined = left.then(right).then(shell)
```

`embed(nq,nc=0,qubit_map=...,clbit_map=())`は全local wireをinjectiveに写し、他wireはidentityで保つ。異なるregisterを使うlaneは独立のまま、後続gateがancillaを共有すればその依存で同期する。qubitとclassicalの対応を混ぜない。

`remap_native(iterator,qubit_map=...,clbit_map=...)`は同じmappingを命令streamへ逐次適用する。target registerの上限は受け手counterが検査する。template連結でqubitsを足し続けず、embed先の同時使用幅を返す。

A†には`NativeTemplate.from_stream(actual_inverse_native_word,...)`を使う。transferの`inverse()`は提供しない。テストでは逆順RZ(−θ)、CX、callerが明示したSX逆の3SX展開を使ったが、この評価器が逆回路やGaussianを生成したものではない。

## 既存emitterへ接続する最小境界

`iter_qiskit_native(circuit)`はcaller所有の**既存native circuit**の`.data`を逐次読み、Qiskit標準gateのbase_class／parameter／条件を検査して`NativeInstruction`を返す。Qiskit importはこの関数内だけ。copy／compile／lowering／simulationはしない。既存circuitを使うときは幅と`circuit.global_phase`もcallerから明示する。native instructionのgeneratorを出せるemitterなら、このreaderなしでcounterへ直接接続できる。

| 既存部位 | 今後の接続点・今回の境界 |
|---|---|
| C3 `mixed_head`／StrongBuilder.sequence | 既存のgroup・control mode・phase規約を保持したnative loweringの出口。C3全体を呼んでdense targetを生成しない |
| StrongBuilder.transition | 実際のrelative Gaussianを再利用。未展開Gaussian/XXPlusYYはreaderで拒否し、新しい合成器を作らない |
| `build_event_sequence` | signed rotation・basis-run共有を保ったnative出口。同じeventから別samplerを作らない |
| step融合 | builderが決めた合法boundary命令をそのまま渡す。counter側で融合・最適化を追加しない |
| Reference A†/B | 供給されたactual inverse/physical native wordを各laneへembedする。既存gadgetは再実装しない |
| CSWAP、scalar/χ、X/Y readout | 既に確定したnative展開だけ受け付ける。未展開CSWAP/H/P等をnative一命令と数えない。global phaseと明示RZを区別 |

C7-4 `event_features`は価格の完全なIRではなく、basis/lane-runの定義も変更していない。本T0でfull H4 stream emitterやpre-P0→P0変換は完成させていない。既存 `_checked_price` 等へ登録する変更もない。

## 計算量

有界arity、L命令、W=nq+ncに対して、通常counterはO(L)時間／O(W)作業memory。transfer captureはO(LW)／O(W²)、applyはO(W²)、thenは一般にO(W³)、embedはO(W_target²)。constant種類のgate countとphase metadataは別に保持する。テンプレートは再利用できるが、一般のPR列を同一templateのq乗とは扱わず、自動cache・新optimizer・管理frameworkは追加していない。
