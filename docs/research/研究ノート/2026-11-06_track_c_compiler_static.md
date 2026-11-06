# 2026-11-06 Track C compiler静的調査

基準a78054dcの保存trace・SOURCE・QASMと、Qiskit1.3.0のPython/Rust実装を静的に照合した。
CRZ半角をidentity同様に扱う交換判定が、CXを越えた誤相殺を許す経路を特定した。合算角cutoffの変更だけでは直らない。
精度を優先するgate定義ベースの参照構成を先に検証し、誤差を確認できる最適化を後から戻す案を推奨する。
H4誤差全量の原因帰属や新profileの性能を確定したものではない。新規compile・科学計算・テストは行わず、run01のSTOPを保全した。
[報告・固定出典・最小再検証範囲](../track_c_compiler_static_investigation_20261106/REPORT.md)を参照。次はユーザーが構成案と最小実行範囲を判断する。
