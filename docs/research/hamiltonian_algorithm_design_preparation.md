# N1/N2/N3限定feasibility：source固定前の準備記録

2026-10-11 JST。利用者が提供設計の進行を指示したため、基点f98050eの独立worktreeで
[結果前scope](hamiltonian_algorithm_design_scope.md)と有限constructor・実native比較・保存verifierを準備した。
旧A-core/B′探索の自動再開ではなく、新設計の問いを固定して確認する作業である。

利用可能CPU32、MemAvailable 54,576,204 KiBを確認。GPU driverへのnvidia-smiはexit9。
独立contextをspawn 2 workers、thread1へ配分し、worker AS4GiB・合計8GiB budgetとする。
今回の新規計算にはwall/CPU-time capを設定しない。科学条件・task seedは固定し、
toolのpolling timeoutを科学実行上限へ転用しない。

source固定前の局所検査は83 passed、176 warnings、2.13s。
warningsは主にQiskit 1.3 MCXの将来廃止予告と既存テスト経路の警告である。
最初は81 passed / 2 failed、修正後は82 passed / 1 failedとなったログも保持した。
失敗原因はfloat frameへのcomplex位相乗算、既存Gaussian helperのvacuum global phase規約、
旧JW参照の3-mode上限だった。今回のmodule内でcomplex frame、vacuum actionによる絶対Γ(U)規約、
6-modeまでの直接JW状態作用を実装した。旧helper/sourceは変更していない。
これらはsource固定前の技術修正で、科学結果を見た候補・予算の変更ではない。

checksはsigned sector平方bound、複素縮退frame、絶対Gaussian位相、近縮退のmodel変更、
正負modular加算の全register値、全system/ancilla入力上のsigned phaseとworkspace消去、
J-only生成、係数のみの下界と保証不能、旧representation回帰を含む。
source固定後に一回の限定batchを実行し、保存された全native IRをNumPy/SciPy verifierで照合する。
付録algebra replayは別に記録する。immutable CIや外部科学再現とは呼ばない。

関連実装・scope・input・test logをsource commitへ保存する。そのhashをrun auditの正本とする。
完成後は結果・独立Git取得の確認を公開し、GPT判断へ戻す。中心テーマの採択と次batchは未承認。
