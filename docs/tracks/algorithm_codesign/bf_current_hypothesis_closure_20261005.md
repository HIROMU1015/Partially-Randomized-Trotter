# B-F現行仮説のclosure記録

2026-10-05 JST。GPTの返却reviewを受領し、**今回固定したB-F主仮説を限定negative resultとして閉じる**判断を記録する。
この記録は新しい科学結果、原resultの再分類、追加実行のauthorizationではない。
研究方針の根拠は[返却計画書](inputs/track_b_post_bfa_research_redesign_20261005.md) §1–3、§16–18。
Codexはその判断と証拠境界を文書へ反映する。

## 1. 閉じる仮説と証拠の範囲

仮説は「同じ5-stage symmetric fourth-order family、同じ情報・32 coefficient evaluations/armで、
finite-task objective Fに、ordinary O／既知leading-tail Lを超えるdecision-relevantな追加価値があるか」。
対象はknown/developmentのH4 linear 1.00 Å、STO-3G、8 system qubits、DF rank12、generation-prefix
`L_D=3`（one-body込みdeterministic generators 4）、`T=0.8`、primary `epsilon=.01, alpha=.05`。
`q={1,2,4,8}`、`R_bud={5,10,20,40,80}`、固定allocation、K2／事前triggerのみのK4。
primary materialityはaction-proxy ratio `<=.95`。結果前条件をここで変更しない。

| 証拠層 | 固定identity | 保持する状態 |
|---|---|---|
| 元science source | `e59344a564e70d64dc3ea39d640581c72676df31` | 元の結果前契約 |
| 元authorization-only child | `cc971e4a2bff9b0c5708003fde7b9519eed27241` | 一回実行のauthorization、consumed |
| 中断result | `09c9e89555032213b52a6a60b38f55563d07a34d` | `BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY`、`INCONCLUSIVE` |
| R0 source | `f226f8c81b5ba07b6d0c4b248c0eb15bf080a622` | 保存済みcellだけのfailure後replay |
| R0 result | `6d2645a09440f50e5b869ef42a1b73a1b625a1af` | `BF1_READ_ONLY_RECOVERY_COMPLETE`、事後復元primary BF-A |

根拠は固定commitの[R0結果照合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d2645a09440f50e5b869ef42a1b73a1b625a1af/docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md) §1–3。
今回、その文書の保存値を引用した。raw resultの再集計・再採点は行っていない。

## 2. 観測された限定negative

| 共通finite評価 | 最良構成 | Finite action proxy |
|---|---|---:|
| F | Suzuki5、q1/R10/K2 | 20,709,936.722970817 |
| Lの評価集合 | 同じSuzuki5、q1/R10/K2 | 20,709,936.722970817 |
| O∪L∪fixed refs | native S2、q2/R5/K2 | 12,924,223.160897588 |

F/L比は1.0、F/共通参照比は1.6024124982。BF-Aは元のoperational ruleによる分類で、
Fが共通参照とほぼ同じという意味ではない。F winnerはLの集合にも含まれる。
cross-scoreの`SEARCH_REACHABILITY_OR_BUDGET_EXPLANATION_NOT_EXCLUDED`は、
今回search到達差が実在したとの認定ではない。

閉じるclaim：**この固定development条件・family・budgetでは、finite-RTE詳細を係数設計へ戻した
F固有のdecision-relevantな追加価値は得られなかった。** 指標はshotsとnative actionのproxy。
compiled RZ、最終RPE総cost、全PF familyについての限界、immutable CI、外部再現を主張しない。
bridge `epsilon=.05`の不足cellは未補完のまま。このclosureのための補完は要求しない。

## 3. 継続しない作業と、残る研究候補

- B-Fのretry、bridge穴埋め、別geometry・precision・split・7-stage追加によるpositive result探索を行わない。
- P-Dの公平再最適化後の選択一致、R3の`STOP_R3_NO_METHOD_DELTA`、FRの既存STOPを解除しない。
- B-Sの有限標本diagnosticを全samplerのheadroom上界と扱わない。B-Fの失敗からB-Mの成功を推論しない。
- B-Mは実行頻度を変える**native列とDF依存評価の構成候補**。採択済みnew method・改善結果ではない。

P-D/B-Fは固定構成内の誤差model・係数設計のdecision valueを調べた。
B-Mは実行列を変える候補だが、一般optimizerへ既知特徴量を渡すだけならR3との差は残らない。
新しいRQ・論文着地点・追加検証の採否はGPT側。[BM-0 packet](bm0_review_packet_20261005.md)で
構成・既知対照・未決事項を示す。BM-1/BM-2/BM-3は未認可、mandatory STOP。
