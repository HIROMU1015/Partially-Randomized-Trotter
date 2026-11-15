# 決定論PF：未観測60軸のcompiled RZ閉包（静的調査）

**対象**：Full PF・B0I p7 × S2・Y4・M5の未観測round（m7・m8・m10〜m12）の60軸。

**方法**
- 実装（compose、control plan、native lowering、wrapper）を読んだ。
- Qiskit 1.3.0のopt1経路を読んだ。Pythonはinstalled、Rustはtag 1.3.0のsource。
- 判断に使ったのは保存済みのreceiptと凍結係数だけである。

**分かったこと**
- 入力のpiece列は、m4〜m12で観測roundとbit単位で同一である（metadata 540 check、PASS）。
- roundによって変わる角度は、最後のancilla runに入るscalar位相だけである。
- 経路上のpassは局所的なものだけで、各pieceのone-body方向制御が全wireで障壁になる。
- 以上から、固定幅のcompiled RZ数は `A + B·q + δ`（δは各軸で0または−1）と実装から説明できる。
- m6・m9の完全一致、共通の切片307、X=Yはこれと整合する。
- Fullでは、piece価格がsigned timeに依存する（Y4 = 3·S2 + 162、M5 = 11·S2 − 48）。
  - 幅や公式の間で価格を移すことはできない。
  - 固定幅のq閉包には影響しない。

**まだ確かめていないこと**
- 最後のrunの角度の余裕。
- 障壁の角度が数値的に十分か。
- 実際のcompiled回路が周期的か。

**次の1作業（推奨）**
- 保存compiled QPYを0-compileで閉包監査する（最大56件の読取）。判定規則は観測前に固定する。
- m12の12 compileは既定の次作業にしない。

**その他**
- compile、保存回路の読込み、価格予測、fit、N、Gは行っていない。
- 資料：[REPORT](../d_pf_unobserved_round_closure_20261115/REPORT.md)、[HANDOFF](../d_pf_unobserved_round_closure_20261115/HANDOFF.md)。
- mandatory STOP。
