# 決定論PF：保存compiled QPYの閉包監査

計画（`dd435d07e` §6）どおり、0-compileで監査した。

**規則の固定と読込み**
- 判定規則（A1〜A5）、wire tokenの定義、共通のH・B・Eと整数cの抽出手順、閾値（関係の残差1e−9、終端角の余裕1e−6）を、読む前に `d3338c7ef` で固定した。
- 読んだのは、compiled QPY 44件とm9のwrapper QPY 12件。

**結果**
- 12 group×軸でA1〜A5がすべてPASSした。
- compiled回路は全wireで `H+B×(q−1)+E` になり、tokenが完全に一致した。
- 本体のRZ数は凍結傾きbに、端と最後のrunのRZ数は `a+b` に一致した。
- 障壁 `cx(anc,k)·rz·cx(anc,k)` は、全system wireに残っていた。
- 最後のrunは `rz·sx·rz` で、角度は傾き −scalar の関係に残差1e−14で整合した。
- 未観測60軸では、最後のrunの余裕がすべて1e−6以上だった（最小0.0225 rad）。

**監査qの外への延長**
- 次の論証による：入力のbit同一性、piece境界での切断、CXが新しく作られないこと、書換えの局所性、qによらない固定点。
- 残る仮定：Rust binaryがsourceと同じ挙動をすること、規模依存の分岐がないこと、未観測の回路は見ていないこと。
- 値は算出しておらず、96/156にも加算していない。

**その他**
- 資料：[REPORT](../d_pf_qpy_closure_audit_20261115/REPORT.md)、[HANDOFF](../d_pf_qpy_closure_audit_20261115/HANDOFF.md)。
- mandatory STOP。
