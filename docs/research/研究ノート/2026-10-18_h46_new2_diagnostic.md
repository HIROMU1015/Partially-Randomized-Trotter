# 2026-10-18：H4/H6 new_2の限定検証

利用者から4th(new_2)をH4/H6でできる範囲から並行検証する明示指示を受領。独立cloneと固定済み保存入力を使用し、SOURCE `7f4c9a171a1d4caeaeaee6c443217edec761f4c9` でexact-tail48・有限RTE96条件を完了した。

[結果・範囲](../track_a_h46_new2_diagnostic_20261018.md)。2次PRだけでは高次PRの可能性を評価し切れないという検証動機に基づく技術診断。新しい科学的最終採択はしていない。同じqでのsignal改善を、そのまま同精度の総費用優位へ読み替えない。

初回の回路basis依存STOPと両worker exit1を保持し、環境installなしのmatrix-only修正後に新run02を実行。両exit0・全回収。旧source・入力・grant・ledger・marker・STOP、管理rootの958 staged削除を保持する。

次の作業案：Codexがnew_2のnative adapterと、代表条件での同精度compiled費用比較を準備する。今回の診断完了後はmandatory STOP。追加の本計算は自動開始しない。
