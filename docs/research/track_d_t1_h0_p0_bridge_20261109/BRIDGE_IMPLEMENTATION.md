# 最小CX frontier bridge

2026-11-09 JST。[実装](../../../src/trottertracks/track_d_p0_cx_bridge.py)／[runner](../../../scripts/run_track_d_t1_h0.py)／[tests](../../../tests/check_track_d_p0_cx_bridge.py)。実行SOURCE2712a73f、T0 c583abbaを無変更で再利用した。

## 規則とinterface

```python
from trottertracks.track_d_native_resources import NativeInstruction
from trottertracks.track_d_p0_cx_bridge import CXBoundaryBridge

a = NativeInstruction("cx", (0, 1))
b = NativeInstruction("cx", (1, 2))
bridge = CXBoundaryBridge(3, global_phase=0.)
bridge.extend(iter([a, b]))
bridge.extend(iter([b, a]))
result = bridge.result()
assert result["removed_cx"] == 4
assert result["output"]["metrics"]["cx_count"] == 0
assert result["p0_qualification"] == "P0_UNQUALIFIED"
```

各wireに過去の未除去nodeのstackを持つ。CX(a,b)が届いたとき、a/b双方のtopが同じ過去nodeで、同じ順序のCX(a,b)なら、そのnodeと新CXを消してstackをpopする。それ以外はnodeを保存して全対象wireへpushする。other native gateもstack上のblockerとなる。古典wireはT0と同じく量子wireの後へmappingする。

disjoint wireの命令が間にあっても両対象wireのtopは変わらず、同じCX対を消せる。ただしRZがcontrol/targetのいずれかにあればtopが変わるので消さない。逆方向CX、RZの可換移動、1Q run合成、RZ(0)の除去、parity展開、Gaussian／CSWAP loweringは実装しない。gateのparameter・sign・control/targetを変更しない。消すCX対は位相変化0であり、宣言global phaseはそのまま渡す。

`feed`／`extend`は同じstateへ逐次入力する。`operations()`は入力を止めた状態で生存native列を返す。`result()`はT0によりinput／outputを計数し、削除CX数とrule scopeを返す。`template()`は**最終出力列**のT0 transferを返す。途中templateへ確定してopcodeを捨てた後、max-plus合成だけでCXを相殺する機能ではない。

schema／wire範囲／条件の検査とinput計数は既存NativeCounter、output計数はcount_native、出力transferはNativeTemplateを再利用する。ASTによる旧SOURCE抽出や新しいQiskit compilerは本体で使用しない。

## Chunkとfrontier

同じbridgeを維持すれば、命令列のどこでextendを分けても同じ入力順・stack操作となる。長さ0〜4の4種alphabet、341 wordの全1593 cutで、出力列・count/depth・templateを一括処理と比較した。新8testsと旧29回帰は全37 PASS。これは定義したreducerの不変性で、全P0との同値性ではない。

人工の32 CX prefixを逆順suffixで消す試験では、prefixのlive frontierが32 nodeへ増え、suffix到着後全て消えた。任意に長い同種の依存列が考えられるため、固定個数suffixだけを保持して常に正確と主張できない。実装は可変長履歴を保持する。独立chunkの価格やtransferだけを渡すと、指定3 P0の反例に必要なgate情報を失う。

## 計算量・memory

有界arityでingestion／push/popはO(L)、nodeとwire stackは最悪O(L+W)memory。消えたnodeはtombstoneとして順序配列に残るので、全て消去できた結果でもstorageがO(W)へ戻る保証はない。最終resultの走査はO(L)、template化はO(LW)/O(W²)。resultを各命令後に反復呼出しするとO(L²)となり得るため、runnerは最後だけに適用する。

このbridgeはfull QuantumCircuit／DAG／unitaryを持たないが、任意長qで一定memoryになる評価器ではない。T0だけの通常計数は引き続きO(L)/O(W)。q1024〜4096の新benchmark、cache効果、speedupは測定していない。

## 適用範囲

- **厳密**：入力されたnative列のT0計数、指定CX ruleによる変換結果とその資源、同じstateを保ったchunk不変性。
- **有限P0一致**：保存native raw16列中10列は全7資源、15列はCX count/depthで一致。指定追加3 P0も一致。これらのcaseとSOURCE／profileへ限定した証拠。
- **未資格**：任意native列／全PR path／P0全体の閉包、Gaussian・1Q・zero/special-angle・CSWAP界面、H4 full pre-P0価格、全law期待値。

純CX列では該当inverse-cancellationの固定点との構造対応を考えられるが、本APIはprofileを検証する全P0互換checkerではない。`P0_UNQUALIFIED`を自動で適格P0へ昇格させず、既存のC6a〜C8-1価格・A/F/Sへ登録しない。

## H4用の保守的な接続

保存compiled QPYは既存`iter_qiskit_native`で厳格に読み、保存絶対価格と比較する。rawは同じreaderを一命令viewへ適用し、未展開・不適格命令では全frontierをflushする。適格native連続segment内だけでruleを使うため、unknown gateを飛び越す相殺はしない。これは既存emitterの規則変更や新loweringではない。

局所segmentでの削除数はfull P0価格への加算／減算項ではない。未対応領域を跨ぐ後続P0最適化は残り、全pre-P0予測欄をnullとする。原本はread-only、旧科学builder・T0・event_featuresを変更しない。
