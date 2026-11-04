# guard証明改善の次段案と実行境界

2026-11-04 JST。**静的検討と人工検証を完了。新科学実行SOURCEは未完成で、実H4計算は開始しない。**

[静的検討](GUARD_BOUND_ASSESSMENT.md)により、作用誤差とノルム増幅を分ける改善候補を整理した。
現行binary・BLAS dispatch・complex128 kernel・FP条件を維持できる数式上の改善である。
実データでの改善量、kernel丸め項が残ったときのguard成立性はUNKNOWN。
今回、科学runner・KernelBounds・判定規則を変更していない。

## 担当と具体的な次の作業

**提案する次段：Codexの技術準備。** ユーザーがこの次段を依頼した後、
まず現行係数の楽観的な証明限界をSOURCEと解析式から詰める。
有効な余地があれば、独立のnorm-only伝播を最小adapterとして追加し、
旧作用誤差との取り違え、rotationの非Hermitian項、underflow、全support guardを人工検証する。
現行の成熟したload/runtime/guard/strict判定処理を再利用し、旧SOURCEを修正しない。
この技術準備は新科学計算の実行を含めない。

**その後のユーザー担当：** 公開された新SOURCE・固定入力・限定科学call計画を確認し、
必要な保存context診断を新しい科学実行として別途許可するか判断する。
この文書や「技術準備を続けてよい」だけで実データ評価を始めない。
承認証明書・grant発行・承認原文保存は新設しない。

## 後で固定すべき最小の科学評価範囲（未実装・未許可）

- 入力候補は既存の保存context `artifacts/rpe_m17_virtual_measurement_bridge/2026-11-04/run01/numeric_context.npz`、
  同RESULTのcomponent/trig対応、既存のnorm²/logical interval/laws/固定literal原本。
  科学SOURCE c3a5cc387および結果81fa68131とのidentityを確認する。
  contextのarrayロードや新しいnorm/defect/予算評価はその科学callに含める。
- 条件は前書のlinear H4/1.00 Å/STO-3G/保存DF12/B raw state/sector70、
  L_D3/δ=.02/m17 q131072/r32 K2、保存N/α/β/εをそのまま固定する。
- 既存contextを再利用し、P再構成・新cos/sin・ideal mean再計算・sampling・trajectory・compileは含めない。
  楽観的な上端の確認で明らかに不足する場合、不要な全行列包含を始めない計画とする。
- 必要な場合だけ欠損とノルム伝播を評価し、各項を分離して出力する。
  新しいprecision/pass回数、必要な入力path/hash、call範囲、出力先、runtime観測をSOURCE作成時に固定する。
  本書は再実行可能な科学計画や実行許可の代替ではない。
- 旧Runtime条件、有限性、complex abs、両座標のguard、平均作用誤差、strictなRPE十分条件をすべて維持する。
  条件不明・不一致・guard不足は未判定/nullでSTOP。旧結果を上書きしない。
- 将来重い欠損評価を行う場合、独立行列を並列化し、immutableなcontextを共有し、
  BLAS/FLINT workerのoversubscriptionを避ける。利用可能資源から構成を選び、恣意的なCPU/RAM/worker上限は置かない。
  実測なしに速度改善を主張しない。旧実行の固定pass/STOPを解除しない。

## 現在の状態と保全

`scientific_execution_ready=false`。guard改善後の科学保証は未評価。
両軸`U_virtual_IID_upper=null`、固定PCG64の`U_exec_upper=null`、
`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
旧`numeric_diagnostic_status=INCONSISTENT`、旧`u_num_bound=null`を保持する。
ideal categorical包含の既存局所十分条件確認も変更しない。
重要な独立科学レビューは今回未実施で、その必要性が生じれば科学実行と別の許可範囲になる。

今回の人工テストは実データをロードせず、実H4の新しい作用・norm・平均・包含・予算・sampling・build/compileは0。
旧SOURCE/raw/grant/ledger/marker/STOP、元dirty worktree、他branch・SERVER2 H8・共有環境を保全する。
branchは`codex/rpe-m17-guard-bound-static-assessment-20261104`。
commit/remote SHAと公開fileの照合結果は最終チャットへ記載する。
**公開照合後にmandatory STOP。次段の技術準備・数値評価を自動開始しない。**
