# 研究C：PR対weighted entrywise L1・有限手順比較IIの限定実行

2026-11-03 JST。**645/1,629 logical units COMPLETE、correctness FAIL 0、retry/resume 0。両arm16 U更新で登録上限停止。**

同じ保存H4初期値と有限アルゴリズム政策の下で、PR endpointのexact tail λはweighted L1 endpointより小さかった。
これは比較IIの有限recipeの観測であり、目的の一般的な最適能力、PR量子資源・保存XDF8への優位、厳密CAを認定しない。

## 固定条件・来歴

- 科学SOURCE（契約を含む公開commit）：`70a98adfb701a209db2c65ea60fb789ff62b1fe7`。実装SOURCE：`9e9fc8391590aeb7f9d36b76d90b5a25bbe2fe5e`。
- [固定契約](research_c_near_objective_practical_comparison_contract_unapproved_20261103.json)、SHA256 `4a40d894a21710a3fcc65b4fef2f94ff5c4271711081aa9b6c3c7856cd7be1f0`。歴史的unapproved名/metadataは変更していない。
- [固定SOURCE](../../artifacts/research_c_near_practical/2026-11-03/run01/SOURCE.json)、SHA256 `6442b2061e0be7533344d96a13e5059d738af343ffa98122841a405e0457e8fb`、ローカルclosure108 files。
- 保存直鎖H4、隣接1.00Å、STO-3G、電荷0/singlet、Nα=Nβ=2/sector36。通常number CDF8、一般signed J、実直交U、元frame順/LD2。
- PR追加最適化前のtensor-only CDF8から修正codecで保存した共通J/U/x0。元g補正κ、核scalar一回、同じtensor loss/物理予算CA/8=.00019875Haを保持。
- γ=.1、β=.1は各自己初期massで正規化した初期penalty寄与一致の一設定。ε=1e-8、ENTRYのη=ε(2n+1)/(1.5n−1/4)、ordered diag5/4・offdiag3/2。
- 各arm最大16受理U更新/128 total trial、各方向32 halving、各新U点にNewton-CG＋2 refinement、maxiter1000/xtol1e-12。warmstart・強度追加探索・再始動なし。
- 新ffsim full-fitter呼出し0。今回の新規最適化は下表のNewton-CG J solveとrecentered U更新であり、保存値閲覧のみではない。

[実行preflight](../../artifacts/research_c_near_practical/2026-11-03/execution_preflight.json)でSOURCE/input/両環境/資源/fresh outputを再照合した。
チャットで明示許可された固定範囲を一度実行し、承認証明・承認原文保存・新grantは作成していない。

## 終端候補と共通物理判定

共通の未再solve初期J/Uは、λ=2.2093719370656357Ha、sector DF誤差7.666027761495033e-5Ha、guard3.278760734406151e-9Haで予算PASS。
各armのrefined初期点は既に別のJ solve後の点である。終端はbest-of-trialsではなく、最後の独立・物理適格受理点。

| 指標 | PR | weighted L1（ENTRY） |
|---|---:|---:|
| Exact tail λ / Ha | 0.500044243591 | 0.941312847393 |
| Exact entry envelope B0 / Ha | 1.5591348809 | 1.23977058606 |
| Row-cancellation gap / Ha | 1.05909063731 | 0.298457738667 |
| Sector operator DF誤差 / Ha | 4.84367101908e-09 | 1.11037670289e-08 |
| Engineering guard / Ha | 3.2165809758e-09 | 3.31972010087e-09 |
| Tensor Frobenius residual | 2.87199858012e-09 | 1.40916473855e-08 |
| DF予算CA/8 | PASS | PASS |
| J∞ / 一次判定 | 7.72744327658e-08 / MET | 1.06094643043e-07 / MET |
| 物理U接空間L2 / 一次判定 | 0.478929473236 / NOT_MET | 3.75640364765 / NOT_MET |
| General-K/raw CAR sector max-entry差 | 4.66293670343e-15 | 3.99680288865e-15 |
| 独立終端/endpoint eligible | PASS / true | PASS / true |
| 停止 | MAXIMUM_ACCEPTED_UPDATES | MAXIMUM_ACCEPTED_UPDATES |

両armともJ一次・rank/nullspace・数値stability・独立終端・物理予算に適格な16更新を得たが、U一次は未達。
有限手順としての比較適格性と、J/U同時一次収束は別である。達成DF誤差は一致させておらず、双方が共通の物理予算を通る条件で比較した。
Guardは`sector_operation_scale_guard8_proposal_v1`の工程的尺度であり、厳密丸めenclosureではない。
SectorのGeneral-K/CAR確認をfull256/native回路の保証へ拡張しない。

自己Fの変化（各arm内のみ）：
- PR：refined初期F 0.0122676444298 → 終端F 0.00512247549251。refined初期λ 0.773835579843Ha。
- ENTRY：refined初期F 0.0356354131782 → 終端F 0.0109849099959。refined初期λ 1.69621245571Ha。

自己Fをarm間で比較して勝者にはしていない。ENTRYではB0が小さく、PRではλが小さいという別のendpointを得た。
PR側の大きいrow-cancellation gapは写像後massとentry envelopeの差を示す。ただし、最適化軌跡・curvature/metric・未収束の影響を除いた因果効果とは認定しない。
直接のmass smoothing上限は各2.16e-6Haだが、smoothingを変更した再最適化や感度実験は今回行っていない。

## 到達度・負担・未資格trial

| 実績 | PR | ENTRY |
|---|---:|---:|
| 受理U更新 | 16 | 16 |
| 新U点（初期含む） | 46 | 48 |
| Trial数 | 45 | 47 |
| Inner Newton-CG | 138 | 144 |
| うちrefinement | 92 | 96 |
| 数値admitted点 | 44 | 47 |
| Stability未資格trial | 2 | 1 |
| Armijo/値尺度棄却trial | 27 | 30 |
| SciPy nfev / njev | 40391 / 40370 | 41671 / 41629 |
| Newton iteration合計 | 20009 | 20906 |
| VG呼出し（checks込み） | 40805 | 42103 |
| Hessian呼出し | 20037 | 20924 |
| Raw status0 / status2 / status1 | 109 / 28 / 1 | 125 / 18 / 1 |
| arm wall / s | 90.872153651 | 52.815973171 |

数値未資格はPR2点・ENTRY1点で、全てlatest-pair stability >1e-6。各点のJ一次・rank/nullspace等は通ったが、未資格の値/勾配を受理方向へ使わず棄却した。
PRのshiftは7.31089427124e-6と4.93620064667e-4、ENTRYは1.36933770599e-6。元閾値は変更していない。
Raw precision loss合計46回とiteration-limit2回を保持し、raw successを独立J/数値資格/DF予算の代わりにしていない。SciPy line-search warningも[stdout](../../artifacts/research_c_near_practical/2026-11-03/execution_stdout.txt)に保存した。
実VG/Hessian回数は近いが、wallの違いをアルゴリズムの一般的速度差にはしない。JAX初期compile、共有shape cache、並列重なりがある。

## 科学call・正しさ・資源

| call種別 | 実績 | 契約上限 |
|---|---:|---:|
| saved_load | 1 | 1 |
| initial_identity | 1 | 1 |
| physical_context | 1 | 1 |
| common_physical_assessment | 1 | 1 |
| objective_setup | 1 | 1 |
| comparison | 1 | 1 |
| export | 1 | 1 |
| optimization_driver | 2 | 2 |
| inner_J_solve | 282 | 774 |
| precision_state | 188 | 516 |
| reduced_admission | 94 | 258 |
| independent_terminal_assessment | 34 | 34 |
| physical_assessment | 34 | 34 |
| endpoint_diagnostics | 2 | 2 |
| arm_export | 2 | 2 |
| **合計** | **645** | **1,629** |

Journalは645 attemptと645 COMPLETE、FAIL0。PR313・ENTRY325・共通7 units。未使用984 unitsを次段階へ転用しない。
Physical assessmentは共通1＋各arm17＝35件で、全てDF予算/sector General-K–CAR PASS。各受理前と終端の独立assessmentは計34件。
Correctnessと工程的数値資格を区別し、未資格trial3件をcorrectness FAILと混同しない。

Preflight：affinity32 CPU、空きRAM約56GB、既存resource_planの2 worker。独立armを並列化し、CAR pipeはlock、各armのtrial/refinementは依存順。
全体wall 92.7661544336s、親CPU 134.805493673s、親peak RSS 477944KiB、CAR child最大RSS 282432KiB。
CPU/RAM/worker/wallの人工cap・共有設定変更・既存venv更新・他job操作なし。

## 証拠・結論の範囲・次の判断

- [result.json](../../artifacts/research_c_near_practical/2026-11-03/run01/result.json)、[STOP](../../artifacts/research_c_near_practical/2026-11-03/run01/STOP.json)、[science journal](../../artifacts/research_c_near_practical/2026-11-03/run01/science_calls.jsonl)、[再照合](../../artifacts/research_c_near_practical/2026-11-03/execution_verification.json)。
- [PR endpoint](../../artifacts/research_c_near_practical/2026-11-03/run01/PR/endpoint.npz) SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
- [ENTRY endpoint](../../artifacts/research_c_near_practical/2026-11-03/run01/ENTRY/endpoint.npz) SHA256 `345dd11a14a281bf716f42009565b2d1875fdcf9845b928a15c102164208cb3f`。
- Run内844 files/43,228,247 bytesを含め、raw配列・全trial・raw solver情報・physical referencesを保存。旧SOURCE/raw/契約/grant/ledger/marker/STOPは変更していない。

今回支持されるのは、登録した共通初期化/強度/有限予算の手順で、PRが共通DF予算内の小さいtail λ endpointを得たという限定的な観測。
比較Iの目的能力、global optimum、他初期値/強度/分子への一般性、保存XDF8やUWC全体への優位は未判定。
PF/RTE、ground/alias、attenuation、RPE shots/W、native kernel価格、強D分母、総RZは今回新規評価していない。
全law synthesis/implemented-law signal/actual CA/保証付き資源優位はnullのまま。旧update064の下流結果を新endpointへ転用しない。

次の最小候補は、両endpointに同じs1/r32/K0/δ=h=.1/m0..12政策を与えるground/alias→PF/RTEの技術準備。
その結果と共通精度政策を確認した上で、候補固定後の独立native価格校正と同じJ/U強Dを設計する。
担当：ユーザー／研究Cが進行方針を判断し、Codexが別の具体的契約を準備する。追加科学計算・新fit/U更新・正式科学レビューは自動開始しない。

公開・remote SHA/tree/成果物照合後にmandatory STOP。
