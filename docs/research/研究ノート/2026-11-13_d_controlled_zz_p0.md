# 2026-11-13：P0 controlled ZZ lowering（Codex D）

## 確認済み（承認範囲、小回路8 compile・sampling 0）

- 計画・実装を先に固定・公開した（`a287b8e6`）。そのSOURCEで、既存の`transpile_and_measure_cost`を使い8件をcompileした。
- compiler条件：Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17。
- 汎用形（`RZZGate.control(1)`、CCX·CRZ·CCX）と解析形（CX·CRZ·CX）のunitaryは、global phase・制御位相を含めて同一だった。
- compile後の差（ZZ回転1回当たり）：
  - 単独：RZ +19・CX +10
  - 保存basis対の内側：RZ +17・CX +10
  - 回転角には依存しない。
- 保存basis対の内側では、compilerの相殺・再合成によりC−Sが−140になった。

## 判断

- lowering差は成立した。
- 残差の扱いは案B（Sを維持し、δ_ZZ·N_ZZを残差の一項にする）を推奨する。
- 回路を安価な形へ変えるかどうか（価格定義）は、ユーザーが判断する。
- P1には進まない。S′は実装せず、fitもしない。mandatory STOP。

[REPORT](../d_controlled_zz_p0_20261113/REPORT.md)
