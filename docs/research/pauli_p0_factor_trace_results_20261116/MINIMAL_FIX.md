# 原因に対応する最小修正案（未採用）

原因診断の科学SOURCEは655e74212。元のP0とdictionaryは変更しない。

## 修正対象

`df_head_pauli_tail_p0.square_factor_dictionary`が使う記号演算子構築の中だけを対象にする。

1. nested one_bodyのFermion項集積：非zero係数を途中で除去しない局所helper。
2. weighted squareのFermion residual集積：同じ入力・順序で直接係数を加算し、近似ゼロ削除をしない。
3. JWのPauli部分和集積：全寄与を保持してから既存collectorへ渡す。

今回の局所subclass/JW globalsの診断経路は、この変更をライブラリ全体へ波及させず実現できる証拠。
production化時もOpenFermionのglobal EQ_TOLERANCE・クラスmethod・他studyの演算を変更しない。
元のPF/PR lawやscalar規則、snapshot/completion、fragment順、signed weightsを変える修正ではない。
元codeを上書きせず、新SOURCEのbuilderとして区別する。

## なぜJWだけの修正では足りないか

LD7のbridge（default Fermion + no-filter JW）は残差1.0680e-16で元gate内。
LD8/9のbridge残差はそれぞれ6.3918e-7で元gate外。従って3分割に対する対策にはFermion集積も必要。
one_bodyの最初の削除は今回のgate超過の主因ではないが、経路全体を無除去にするなら同じ局所helperで
扱える。積・weight乗算・行列builderの作り直しやglobal tolerance低下は必要ない。

## 採用前に別途必要なこと

- 修正方針と新SOURCEでのP0再資格について実行範囲を承認する。今回の診断をP0 PASSへ再ラベルしない。
- 既存のoperator同一性・scalar・sector・数値gateを全て保持して新辞書を評価する。
  今回はcause traceとfull256 tail一致までで、修正辞書の全P0資格を認定していない。
- 途中除去なしでは微小なroundoff由来の項も残り得る。最終係数の扱いに別のcutoffを提案する場合は、
  元Hamiltonianとの誤差を評価できる明示的な科学条件として決める。ここでは閾値を新設しない。
- 新しい辞書を採用すれば辞書identity、λ_P、B canonical configuration/scheduleのSOURCEが変わり得る。
  旧DF/Pauli ID・旧signal・旧compiled価格をそのまま流用しない。B2に変更を渡す。

本番辞書への採用、signal、価格較正、Q3は今回の後続作業。自動実行しない。
