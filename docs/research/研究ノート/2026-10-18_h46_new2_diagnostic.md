# 2026-10-18：H4/H6 new_2の限定検証

利用者から4th(new_2)をH4/H6でできる範囲から並行検証する明示指示を受領。独立cloneと固定済み保存入力を使用し、SOURCE `7f4c9a171a1d4caeaeaee6c443217edec761f4c9` でexact-tail48・有限RTE96条件を完了した。

[結果・範囲](../track_a_h46_new2_diagnostic_20261018.md)。2次PRだけでは高次PRの可能性を評価し切れないという検証動機に基づく技術診断。新しい科学的最終採択はしていない。同じqでのsignal改善を、そのまま同精度の総費用優位へ読み替えない。

初回の回路basis依存STOPと両worker exit1を保持し、環境installなしのmatrix-only修正後に新run02を実行。両exit0・全回収。旧source・入力・grant・ledger・marker・STOP、管理rootの958 staged削除を保持する。

次の作業案：Codexがnew_2のnative adapterと、代表条件での同精度compiled費用比較を準備する。今回の診断完了後はmandatory STOP。追加の本計算は自動開始しない。

## 追加検証完了

利用者の不足検証追加指示を受け、[native・同精度代表比較](../track_a_h46_new2_native_validation_20261018.md)を実施。H4 SOURCE687679a4…/H6 executor f0dd2a9a…、同じPR分割で各n8・採用96軸。人工18件/実数値222件PASS。8桁係数は厳密4次の立方和条件を満たさない一方、代表ε=.001でnew_2のshot重み付きRZ標本平均はS2/Yoshidaより小さかった。係数の修正や全研究の採否は行わない。

generic unitaryのcompile不一致を検出して旧STOPを残し、閾値不変のexplicit XYへ修正。H4完了を再利用してH6だけ24軌道並列へ引継ぎ、重複13軸を一致確認。旧SOURCE/原本/958 staged削除は保全。実entry累計121を記録し返却なし。共有環境・venv・他jobは変更0。次の範囲拡張は自動開始しない。
