# H6共通PF：残22候補を新grantで取得開始

2026-10-22 JST。**利用者の「計算を再開して」を、直前に提示したfresh22候補の追加内部予算案へ結合し、新SOURCE・新pack・新COST grantを起動した。** 本資料は起動記録であり、取得完了・費用順位を認定するものではない。

科学SOURCE `bdf6772aade7063c65d69f24dfdc5e303a2819a2`、branch `pf-common-h6-metadata-execution-20261022`。旧中断SOURCE `b7662487185086aeaa1b298f82089e194b7270db` とそのgrant/ledger/marker/STOPは不変。metadata SOURCE `cba2a82021b17cba0eec31b03fb19d80f35225f6` の取得成功も別記録として保持する。

## 対象と承認範囲

linear H6、隣接1 Å、STO-3G、tol-only DF threshold1e-8、原保存rank19。T=.8、P=0、δ=.8/.4/.2/.1、S2/Yoshida4/m5_bestの元係数とsigned schedule、B0/B0I L_D5..15・Full19、q1/2/4/8の固定276候補を継承する。wrapper/control/scalar/compilerは元のfull cosine/sine Hadamard、symmetric-directional、scalar一回、Qiskit1.3.0/rz/sx/x/cx/opt1/seed17のまま。

新取得は[前回の不足22候補](../../artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22/unresolved_candidates_22.json)だけ。元worker `pf_common_run_native.cost_one`を変更せず使い、保存価格を持つ187完成pairと同native30適格pairは参照する。63 metadata、新102 signal、1380適格性/shot endpointは再計算しない。旧partial軸は価格として転用しない。

新COST grantのcapsはnative evolution/独立occupation/low/high各22、signed piece1520、wrapper/build/compile/各vector作用44、snapshot/reference/oracle/full preparation各1、prefix preparation7。旧非返却消費を含む累積案はnative各231（旧224から各7追加）、piece4854（旧3625から1229追加）、wrapper build452（旧448から4追加）、compile438/uncompiled441/compiled434（各448以内）。候補集合や数値政策の追加変更は0。

旧grantをresumeせず、新packでCOST→ACCOUNTへ進む。新COSTが正常終了した場合だけ、別ACCOUNT grantで保存1380 endpointと保存PR literal値の固定会計20比較を行う。H2/H4は原結果を参照結合する。新PR sampling、旧cohort/確認標本の混合、q16、候補拡張、pushは0。

## 実行と確認

新しいselectorと固定整数会計を人工5件で検査した。対象22の厳密一致、完成価格の再取得拒否、旧SIGNAL/QUALIFY成功と旧COST STOPの維持、保存PR統計のliteral不変、必要価格欠測時の拒否、不適格候補にcompile不要という条件がPASS。

自分専用の一時user serviceで、チャット側の起動元processから実行を分離した。自動再起動なし、旧grant再使用なし。起動時affinity128・idle128・空きRAM約1TB・disk約124GBから22 workerを割り当て、全未完候補を並列化した。内側BLAS/compilerは各1、元のshared immutable preparation/contextを利用する。固定の低い資源上限や共有設定の変更は0。

科学SOURCEに含まれるpipelineは、新COSTとACCOUNTを一回ずつ起動し、各終了receiptのPASS後だけ次へ進む。終了監査・結果export・H2/H4参照結合は別のread-only SOURCEへ固定して接続する。旧187pairの受入監査と、新22 workerの正常exit/reapは別に記録し、旧parentの不明なexit/reapを成功と書き換えない。

科学call消費後の失敗では当該stageをSTOPし、同grantのretry/resumeは行わない。元のCOST中断原因は未特定で、今回の実行分離方式が実分子計算の完走を保証するとは主張しない。

## 起動証跡と終了条件

[起動snapshot](../../artifacts/resource_applicability/track_a_h6_pf_recovery_execution_start/2026-10-22/start_receipt.json)、[承認receipt](../../artifacts/resource_applicability/track_a_h6_pf_recovery_execution_start/2026-10-22/approval_receipt.json)、[人工検証](../../artifacts/resource_applicability/track_a_h6_pf_recovery_execution_start/2026-10-22/artificial_validation.json)。最終取得状態はprivateの新grant/ledger/completion/STOPを正とし、本起動記録から推測しない。

必要な取得が揃えば、同一公式内と全公式PF最良に対する保存PR比較、588候補の取得状況、5精度の費用・改善率・保存分散/SE、旧比較からの変化、累積科学call、数値照合、結果保全まで実施する。経験的allowance・有限探索・未観測rare event・母平均保証なしという限界を維持する。固定比較と監査を完了後、専用branchへローカルcommitしてmandatory STOP。push、production、詳細科学レビュー、論文本文作成は行わない。
