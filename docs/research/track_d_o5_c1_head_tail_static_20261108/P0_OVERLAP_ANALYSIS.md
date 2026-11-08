# P0との重複：確認済みとUNKNOWN

新compile・保存QPYロード・価格再解析は0。以下は既存compiler SOURCE、Qiskit公式SOURCE、固定C7-2報告の引用である。

## 実際のpipeline

P0はQiskit1.3.0、optimization_level=1、seed17、basis rz/sx/x/cx、approximation_degree=1.0、qubits_initially_zero=False、topology-free。`c1_h6_compiler_precision.compiled_wrapper`がdeclared global phaseをMPFR160のsymbolic offsetで追跡し、最後に一回bindする。state/operatorからのphase fittingではない。

その内側の `pr_dictionary_validation.gates.compiled_wrapper` はまず空のlocal EquivalenceLibraryを使うUnrollCustomDefinitionsでnative basis＋uへlowerし、その後transpileする。共有SessionEquivalenceLibraryや環境を変更しない。head/tailごとの個別compileではなくfull wrapperを渡す。`qubits_initially_zero=False`なので、reference初期真空を根拠とする任意のgate削除を期待しない。

[公式Qiskit 1.3.0 builtin_plugins.py](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L109)のlevel1 initにはInverseCancellation、[optimization stage](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L565)にはOptimize1qGatesDecompositionと指定gate集合のInverseCancellationがある。level2/3のCommutativeCancellationと同じとは扱わない。ローカルにあるこのplugin fileのbytesは公式1.3.0公開本文と一致を確認した。既存distributionのentry_pointsでもdefault init/optimizationがこのpluginへ対応することを確認した。パッケージのimport・PassManager生成・実行はしていない。

この構成には、PRのhead/tail、NativeModelのrotation/gauge、保存eventの意味を読み取ってG_T†G_Hを再合成する専用passはない。ただしlow-levelのunroll・単qubit簡約・CX相殺だけでも同じ利益を回収する可能性があり、その不在を証明したことにはならない。

## 対象別判定

| 対象 | Builder段階 | P0重複の証拠 | 分類と限界 |
|---|---|---|---|
| head内部relative / 同rotation group | 実装済み | P0前から削減 | ALREADY_HANDLED |
| tail同basis runの一組のB†/B共有 | 実装済み。中心は順番に残す | P0前から共有 | ALREADY_HANDLED |
| stepのhalf0→full0・scalar集約 | 実装済み | C3/C5構造に含む | ALREADY_HANDLED |
| 隣接するCX二個、単wire回転の局所合成 | 専用head-tail処理とは別 | C7-2のadjacent CXが2→0、physical同Z列の一回転化。公式passとも整合 | P0_OVERLAP。ただし**この局所pattern**への分類 |
| 実Aとactual A†の人工連結 | builderは別回路を連結 | C7-2の人工A→A†がP0 size0 | P0_OVERLAPは固定人工例に限定。任意Gaussian対やH4接口の証明ではない |
| 同じfull-frame Gaussianのhead-tail相殺 | 横断処理なし | 上の事例は示唆。現在のH4全接口で同じnative列が現れ全て消える証拠はない | UNKNOWN。rawの冗長性だけで利益認定しない |
| 異frameのhead出口→tail入口のrelative再合成 | head内に実装、横断入口には未接続 | 新しいrelative circuitと現行full P0の比較は未取得 | POTENTIAL_GAIN、P0との増分重複はUNKNOWN |
| C7-2のCX–RZ(control)–CX | 未手動簡約 | 同じunitaryを持つRZだけへP0が縮約せずCX二個が残る既存反例 | 「合法書換えは全てP0が実施」の反例。head-tailの利益量は導かない |

## 加法性・他境界への影響

C7-2は、単体CSWAPの局所価格和とcascade全体が一致しないこと、A/A†連結の大幅な相殺、単純shell＋max(lane) depth式の反例を保存している。これらを再計算せず、局所節約の加算を禁じる根拠として用いる。

relative Gaussianの再合成は末端gateの種類・角度・wire順を変える。head内部transition、tailのparity、readout/CSWAPとの隣接簡約を失い、countやdepthが増える可能性もある。raw gate数の減少・単体relativeの価格差・接口数の積はfull-wrapper利益の証拠にならない。

**必要な未取得証拠：** 同じ人工fixtureのS₀/S₁/R₀/R₁ full-wrapperを同P0で比較し、局所ではなく実compiled全7資源を記録すること。同じ合法橋をStrongへも与え、Referenceだけの見かけの相対改善を排除すること。具体案は[NEXT_EXECUTION_PROPOSAL](NEXT_EXECUTION_PROPOSAL.md)。今回は実施しない。
