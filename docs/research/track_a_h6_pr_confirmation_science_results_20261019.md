# H6新A候補の独立確認・科学実行結果（2026-10-19）

事前固定した「ε=.05ではPR不利」「ε=.0005ではPR有利」の両方向を、新しい独立標本の点費用で再現した（両方REPRODUCED）。固定152 trajectory・304 compiled軸・3,712 RTE eventを完了し、両scopeともnative exit0・全worker回収・終了監査PASS、scopeごとのmandatory STOPと最終STOPを保存した。母平均の保証ではない。公開push・新しい科学レビュー・H4/H8最終統合へ進まず、local commitで停止する。

## 承認・条件・判定

利用者の「H6独立確認の科学実行」による2条件の明示承認を記録した。最終sealは`f79366f68caf8585cc36d2f2b0fefcb1a4bd0d1f`、科学SOURCEは`bf0c26dae4477de5fe7264a7c91e077661ef9b38`。SOURCE専用のdetached worktreeはclean、不変。

linear H6、隣接1 Å、STO-3G、tol-only 1e-8/rank19、prefix4、T=.8、P0。既存二次PR・元の無条件finite-RTE・K2/r4・通常signed bias・allowance倍率1。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、既存symmetric-directional full Hadamard wrapperを維持した。

- ε=.05：`B2_p4_q2_2nd_r4_K2`、固定n48、直接実測済みPF `B0I_p7_q1_2nd`、期待はPR不利。
- ε=.0005：`B2_p4_q8_2nd_r4_K2`、固定n104、直接実測済みPF `B0I_p11_q4_4th`、期待はPR有利。

費用Gは各trajectoryの保存shot×compiled RZのcosine/sine合計。判定は`sign(PF固定G−PR paired平均)`と期待方向−1/+1の一致。`REPRODUCED`と`PR_CHEAPER`は別項目、厳密な同費用は`INCONCLUSIVE`。結果の方向にかかわらず保存する固定仕様を変えていない。旧探索n8・union n32・条件付き標本をpoolしていない。

## 費用と不確かさ

| ε | n | PR平均G | paired SE | 固定PF G | PR削減率（対PF） | 方向再現 | PR_CHEAPER |
|---|---:|---:|---:|---:|---:|---|---|
| 0.05 | 48 | 634050918 | 4054009.11 | 356097424 | -78.055463% | REPRODUCED | False |
| 0.0005 | 104 | 3.28756265e+13 | 6.553932e+10 | 36384720355183 | 9.644416% | REPRODUCED | True |

表は実行kernelが保存した値の表示。削減率が負ならPRの方が高費用。平均の厳密な有理数表現、sample SD、paired SE、固定PFと符号は[result.json](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json)に保存した。公開用処理で統計を再計算していない。

| ε | 保存された0次event数 | 保存された2次event数 |
|---|---:|---:|
| 0.05 | 374 | 10 |
| 0.0005 | 3327 | 1 |

ε=.0005で観測した2次eventは1回であり、この標本のpaired SEを希少support全体の分散精度保証と扱わない。上表もkernel保存countの転記で、追加のstratum解析やvisibility計算は行っていない。

paired SEは同一trajectoryのcosine/sineをまとめた費用平均のMCSE。厳密CI・実power・母平均費用順位・全候補最適性の保証ではない。保存bias/allowanceの経験的性格や未観測の希少component費用まで解決したと扱わない。新CI/p値・power/visibility解析・fit・候補再選択を追加していない。K2のsupportは0/2次であり、K4のsupportを流用せず、未観測費用の0補完もしていない。

## quota・seed・grant・予算

各scopeの直前にSOURCE34 file、入力114行、provider877 file、runtime34,273 leaf/Python/compilerを再照合し、終了後もPASS。既知campaignのseed原本を再読取し、封印済み1,080 master/step整数との衝突0を確認した。固定registry SHAは前段と同一。新しいseed set・候補・signal・shot再計算・PF compile・追加samplingは0。

quota CLIの不在から無制限と推定していない。出力先`/home`のext4 filesystemに対し、読取専用`quotactl(Q_GETFMT)`でUSER/GROUP/PROJECTの3種が全てESRCH（quota無効）であることをscopeごとに確認し、既存gateの`NO_QUOTA_CONFIRMED`を適用した。確認不能の許容政策は使用せず、gate・quota・共有設定は変更していない。照会結果とgrant消費SHAは[grant receipt](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/grant_consumption_receipts.json)に保存した。

scope別に新しい有効one-shot grantを発行し、各1回消費。q2完了・監査・scope STOPを保存してから、同じ利用者指示で明示承認済みのq8を主agentが別途dispatchした。launcher間の自動chainはなく、[scope境界](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/scope_boundary.json)を保存した。両scopeとも自動retry・追加実行・次stageへの移行は0。

| call | q2/n48 | q8/n104 | 合計 |
|---|---:|---:|---:|
| trajectory / native evolution | 48 | 104 | 152 |
| occurrence | 96 | 832 | 928 |
| RTE event | 384 | 3,328 | 3,712 |
| cosine/sine compile | 96 | 208 | 304 |
| control probe | 4 | 4 | 8 |
| independent event oracle | 1 | 1 | 2 |

非払い戻しledgerの親/子予約を固定予算と照合した。[ledger監査](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/ledger_audit.json)に2つの親ledger・152 task ledgerの合計とfile SHAを保存した。新cohortのpaired費用集計は各scope1回、計2回。科学kernelはSOURCEから変更していない。

## 並列実行と終了監査

各scopeの代表control PASS後に残りをtrajectory並列で投入し、paired軸を2 CPU groupで並列compileした。worker数はfresh affinity・idle capacity・RAM・同時jobのCPU割当から選び、過去のworker数を上限にしていない。driverは別CPU、BLAS/transpiler内側threadは自分のprocess内で1。準備をfork共有し、共有設定・他job・Codex Bの作業を変更していない。

| scope | worker用CPU | 最大trajectory並列 | startup→native receipt秒 | wait4 CPU秒 | 単一process最大RSS byte |
|---|---:|---:|---:|---:|---:|
| H6_CONFIRM_P4_Q2 | 48 | 24 | 106.246 | 2448.467 | 516648960 |
| H6_CONFIRM_P4_Q8 | 49 | 25 | 712.384 | 19290.163 | 1180647424 |

wall値は保存timestampによる技術telemetryで、最大RSSはpool合計ではない。CPU競合・一時的PSIだけでworkerを停止していない。driver、trajectory親、両軸・control/oracle子のnative正常終了と回収、完全なtask集合、raw SHA、compiler/fingerprint、cohort/law、ledger、終了後SOURCE/input/runtimeを監査した。追加の科学統計再計算は行わず、作者自身による終了監査として記録した。別担当の独立レビューではない。

[公開bundle](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/README.md)には152 trajectory receipt、304軸CSV、保存primary結果・event counts、control/native完了、grant消費SHA、SOURCE保全とSTOPを収録した。raw event/seed値・NPZ・private入力契約・有効grant・生ledger/logはprivateに保全した。

**未解決事項**は、希少support全体を含む母平均の厳密な推論と経験的bias/allowanceの保証である。今回の固定標本による点順位再現と区別する。新しい科学レビューやH4/H8最終統合には進んでいない。

**次の作業：ユーザー**が結果の採用・公開pushの要否を判断する。**Codex**は最終mandatory STOPを維持する。追加科学作業なし。
