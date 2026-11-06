# V3角度分類：規約・未確定・保証の境界

## 二つの記録

`literal_native` を主表とする。実際にnative RZへ渡されたsigned binary64値を対象に、確認できた型だけを数え、残りは未確定に残す。`analytic_provenance` はSOURCEの解析的な角度由来を示す補助表である。両者を同じ「exact」の意味で使わない。

- ゼロは実native角として厳密なClifford。compiler AがRZ(0)を残すので、指標上のRZ数には含める。新たに削除しない。
- 固定gateの定義から由来が明確なkπ/4は、補助表でk偶数をClifford型、k奇数をT型とする。例えばCRZZのCCX由来、CSWAP、H/Sdg/CZ/Z、XX+YYの固定pi slotである。実native角が規則通りのbinary encodingと一致することも確認する。角度の近さだけを根拠にしない。
- 非零binary64 pi定数は数学的なpiの厳密値ではない。補助表では解析的由来と丸め残差上界を保存するが、主表ではexact Clifford/Tとして無料計上せず未確定とする。この分類は演算子や既存誤差diskを変更しない。
- Gaussian SOURCEの `np.angle` 由来phase/betaは、入力が実行列で観測値が0/πや±π/2でも、生成時のsymbolic branchがsnapshotに残っていない。数値一致だけから補助表のCliffordへ昇格させない。非零のcanonical encodingは未確定、ゼロはCliffordとする。
- 数値角について、最近傍と両隣のπ/4格子点との差を192-bit外向き区間で評価し、交差しないものは任意角と認定する。対象範囲の数値角を保持し、単に近いからT型へsnapしない。canonical pi encodingだけが与えられ由来不明の場合は、未確定に残す保守的規約を優先する。
- unbound symbolic angle、全supportを列挙していないtrajectory vacuum補正は未確定。補正は1 RZ/shotとして残し、0埋めしない。

この結果、未確定数は「実際の任意角数」でも「T型数」でもない。任意角の既知数Aと未確定数Uから、総任意角数に対して安全な区間 `[A_lower, A_upper+U_upper]` を作る。区間が重なる場合、全任意角数の削減を保証しない。確定部分だけの減少と明確に区別する。

## SOURCEの根拠

Qiskit1.3.0の[XXPlusYY定義](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/library/standard_gates/xx_plus_yy.py)、[equivalence library](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/library/standard_gates/equivalence_library.py)、[CCX定義](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/library/standard_gates/x.py)、[CSWAP定義](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/library/standard_gates/swap.py)を確認した。

保存XX+YY loweringの18 RZは、固定pi/zero slot 14、numeric beta由来2、`pi-source_theta/2`由来2である。後二者の分類は実入力によって変わり得るので、全てを任意角とは数えない。固定slotは補助表にのみ解析的Clifford由来として認定する。

登録CRZZの24 RZはCCX/H由来Clifford slot8、T型slot14、実theta/2のslot2。B′は最後の2 RZだけを保ち、referenceの中央回転は1 RZ。CSWAPの11 RZはClifford slot4、T型slot7。このSOURCE内訳は、whole回路の最小T数ではない。

## 周期・符号・control・厳密合成

型の解析ではkπ/4の整数parityを使用するが、実角keyを2πで正規化しない。RZ(theta+2π)はRZ(theta)と符号が異なり、controlled SOURCEでその差を無条件に捨てることはできない。signed `float.hex()`、nativeの位置、親gate族、frame/inverse、round、roleを保持する。

今回対象は既にcontrolled joint回路をlowerした後のnative RZである。P→RZの共通joint global phaseと、source scalar・ancilla相対位相を混同しない。reference F†の実角符号、Gaussian actual inverse、RTE component符号は保存SOURCEの通り扱う。新たなcontrol追加は未資格のまま。

理想的なT型RZは、Cliffordと必要なglobal factorを伴う標準T/T†一つで表せる。しかしそのslot数は回路の最小T数ではなく、binary64角の厳密な実装費でもない。任意角合成費、誤差配分、phase保証を別途接続する必要がある。

既存strict合成はexact-binary RZ、`up_to_phase=False`、global W保持、角度・precision・seed別key、strict operator guardを使用する。Clifford fast pathのPASSも所定epsilon内の近似保証であり、角度のexact Clifford性を証明するものではない。今回はprecisionを選ばず、保存7603 recordを読取確認するだけで、T価格やwordを新Aへ移植しない。

## 全supportへの個数移送

旧V1/V2の固定compiler contextと「最終passはCX/Xだけを除去する」性質を再利用する。SOURCEでの非加法なbasis共有は、既存の厳密basis-run期待値により評価する。referenceのproduct-Z消去は全productを残す上界／全て捨てる下界で扱う。未確定を勝手にClifford、T、任意角のいずれかへ配らない。

Dの非融合寄与と境界融合の差分を別に保存する。融合差分はsignedな会計補正であり、負のgateを実装する意味ではない。各分類・角度keyに同じ既存有理確率・multiplicityを適用する。角度keyの再利用は、回路内gateの削減・合成word実行の共有を意味しない。
