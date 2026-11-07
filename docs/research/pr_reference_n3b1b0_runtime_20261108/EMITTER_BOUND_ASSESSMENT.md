# N3-B1b-0：Emitter上界の実経路への対応

2026-11-08 JST。[REPORT](REPORT.md)、[runtime条件JSON](../../../artifacts/pr_reference_n3b1b0_runtime/2026-11-08/RUNTIME_COMPATIBILITY.json)。判定は**RUNTIME_EMITTER_COMPATIBLE**。対象を以下の固定runtime・経路へ限定する。

## 1. 何が保証を支えるか

| 層 | 根拠 | この層だけでは言えないこと |
|---|---|---|
| 論理word | H/S/T/Tdag/Xとexact W、固定written product、全文字列のactual adjoint | 基本RZ近似の成功／guard |
| 位相算術 | N3-B1aのbinary64 lattice閉包、周期差、compose／inverseの記号上界 | 非IEEE演算／別emitter／無制限後処理 |
| 実runtime | 定数hex、compiled setter、丸め状態、SOURCE byteとwheel fingerprint、固定局所例 | 全機械語の形式的同値性 |
| 数値Operator | 正負・順序・局所埋込みの有限診断 | strict残余のcertificate |
| 未取得word | 今回は証拠なし | 43 key成功、完成辞書、全support RPE／T |

Qiskit公式commit `896d1c8b72888a52a46081a6d1f4de8c56db8aea`の[SOURCE対応](../pr_reference_n3b1a_emitter_static_20261108/EMITTER_SOURCE_AUDIT.md)を引き継ぎ、実際に使用したPython10ファイルを同commitのblobへ対応させた。後の2ファイルは[Operator](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/quantum_info/operators/operator.py)と[circuit_to_instruction](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/converters/circuit_to_instruction.py)。

通常IEEE意味論に対するSOURCEと実runtimeの対応は確認したが、Rust wheelの全build flagsやlibmを形式証明したという主張はしない。有限テストだけから一般定理を新たに推論していない。

## 2. W閉包・正負・zero

保存Pi0／Pのhexは一致した。Pの50-bit significandに基づく8状態閉包はN3-B1aの代数を用いる。実runtimeは全residueと複数wrapの24固定例で対応し、W個数を推測していない。

数学的exact wordの負側は**連結全体**のadjoint。既存lower_nativeは逆順emit後にlocal.inverseを使い、chunk順と文字順を両方反転する。S→Sdg、T↔Tdag、Wの負位相を保持する。float位相のpositive／negativeが厳密な複素adjointと全bit一致すると仮定せず、binary周期と真の2piの差を上界へ含める。

zeroは空の人工wordと±0で局所検査した。実辞書のzero分岐を実装したものではない。固定RZ targetをpiへsnapしたり、角度keyを2piで同一視したりしていない。

## 3. 上界を適用する経路

推奨される単一word経路は、基本chunkを固定written順に連結して既存lower_nativeへ1回渡す形。今回も人工compositeでその機構を検査した。

比較用の43-carrier経路では、written productを一致させるためchunkを逆順に時系列composeする。各positive carrierは0から既存emitterで作り、43非自明joinはinplace=True。negativeの場合だけ全体inverseを1回行い、最終joint fixtureへinplace=Trueでjoinする。

| 操作 | 誤差会計 |
|---|---|
| local W loop／named append | local8状態証明。W数へ比例する推測値を使わない |
| lower_native内部の0-phase outへのjoin | local latticeをbit保存し、追加の非自明丸めなし |
| 43個のcarrier join | 各local W定数差とjoinを合わせ64u未満で覆う |
| 最終join | 64u未満 |
| 全体inverse最大1 | 32u未満 |
| 追加phase設定／再読込最大2 | 各16u未満。今回の非zero初期設定は1、QPYは0 |

u=2^-53。0からの初期化や、誤差を加えないlocal lattice上の0-phase joinを、非自明joinと同じ費用で二重計上しない。独立したruntime試験全件を連結した巨大回路にこの回数表を適用しているのではない。

この固定経路なら、保存係数による上界は

\[
e_{\rm emit}<45/140737488355328<2^{-41}<R_{\rm emit}^{lo}
\]

となる。N3-B0の残余Fractionは変更せず、最後の順序比較だけを今回検算した。元の量子化・基本word誤差へemitterを一度だけ加える。

## 4. CopyとOperator診断の落とし穴

Qiskitの[quantumcircuit.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/quantumcircuit.py)では、composeのinplace=Falseはcopyを呼ぶ。copy_empty_likeの一時dataと、その後に採用されるCircuitData.copyは、[CircuitData.new](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/circuit/src/circuit_data.rs)を通じてsetterへ到達する。従ってcopyにも位相再正規化の対応が必要となる。

N3-B1aの一般的なcompose説明だけから、copyを含む任意経路の再設定数が2以下と推定しない。今回の上界適用経路はinplace=Trueへ明示し、旧lower_nativeをそのまま使える。default copying compose、to_instruction等を後続で使うなら、その保持される位相処理と追加誤差を別途会計する。旧静的SOURCE／記録を書き換えたのではなく、実経路の適用条件を具体化した。

phase=tauからcopyで0への変化は、binary周期では差0だがexact exp(i*tau)と1の同一性ではない。さらにOperator(circuit)がto_instructionへ変換すると、converterも数値global_phaseをsetterへ再設定する。この境界でOperator差が0と観測されたことはstrict scalar誤差0の証明にならない。直接のscalar exp診断と区別して保存した。いずれも浮動診断で、周期差の保証は記号的なd_tau上界に依る。

## 5. 未資格と最小差分

今回の経路に対して新emitter／integer-W adapterは必要と判定しなかった。検証helperの人工recordは、RZ近似・新strict record・本番providerではない。

実43基本recordの取得、各guard、seq/hash/W/T/Tdag metadata、固定bit分解／指数順／actual adjointを持つprovider、実wordの最小局所適合は未実装・未検証。既存C0側のguard＋emitter余裕、全supportでのcompiler／synthesis diskとRPE会計も別の条件である。

QPYは今回不要・未実行であり、その追加後までRUNTIME_EMITTER_COMPATIBLEを拡張しない。仮に必要となる場合、再設定／copy／読込を含む全経路が保存係数の回数内に入るか判断する。収まらなければMINIMAL_ADAPTER_REQUIRED／CONDITIONAL_COMPATIBILITY／UNQUALIFIEDへ戻し、b／precision／keyを勝手に変更しない。

本限定技術検証は完了。**mandatory STOP**。後続は[NEXT_SCOPE](NEXT_SCOPE.md)のとおりGPT／ユーザーの判断に戻す。
