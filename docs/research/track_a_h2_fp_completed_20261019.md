# H2 FP one-shot完了（2026-10-19）

利用者が「FPのみのone-shot実行可否」を選択して「次の作業に進んで大丈夫」と承認。
[固定FP契約](track_a_h2_fp_d_a_preparation_20261019.md)を変更せず1回実行し、
native exit0、同一親wait4回収、runner終了監査と保存bytes監査がPASSした。
**COMPLETE_MANDATORY_STOP**。D/A/S_STATS/Pは実行していない。
今回の承認はpush承認を含めず、成果はH2専用branchにlocal commitで保存する。

## SOURCE・入力・実行範囲

- 科学SOURCE：`de72e5d4e48177734c0ddd79dffac14222410658`（23 files）。
- 保存F：受入済み完了commit `7ea6df2dbe34e64a34f468c343bb12922d3e2829` の入力を再利用。
- linear H2、1.0 Å、STO-3G、tol-only DF1e-8の自動決定rank3、4 qubits。
- spin sectorはNalpha=Nbeta=1・次元4。保存状態を再生成・再正規化していない。
- T=.8、P=0、ε=.05/.01/.005/.001/.0005、通常signed bias、allowance倍率1。
- Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、full Hadamard wrapper、symmetric directional。
- original scalarとtail identityは1回だけ適用し、full-H referenceは回転しない。

snapshot SHA256は `62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa`。
Hamiltonian `d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80`、
state `080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf`、
compiler profile `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` を確認した。
旧H2のrank指定DFやsector6は混用していない。

## one-shotの結果と予算照合

| category | 実行・保存済み | 固定上限 |
| --- | ---: | ---: |
| full-H reference pair | 1 | 1 |
| full-H primary / independent columns | 4 / 4 | 4 / 4 |
| truncated reference pair | 1 | 1 |
| preparation再構築 | 4 | 4 |
| primitive probe / sector spectrum | 111 / 4 | 111 / 4 |
| 独立tail matrix | 2 | 2 |
| base signal / independent forward | 6 / 6 | 6 / 6 |
| B0I analytic overlay | 2 | 2 |
| signed shot endpoint | 40 | 40 |
| PR trajectory / occurrence / RTE event | 2 / 3 / 10 | 2 / 3 / 10 |
| native evolution / compiled wrapper | 8 / 16 | 8 / 16 |
| control probe / independent event oracle | 32 / 2 | 32 / 2 |
| endpoint統計会計pass | 0 | 0 |

SCF・積分生成・DF・保存状態solver・実quantum shotsは全て0。
予約台帳は非返還で、全予約がnative完了receiptと一致した。
必要shotはFP内に保存した資源会計用の値であり、測定shotを生成していない。
native compilerの測定値は以下のとおり。FPはnative診断であり、ここでD/Aの最良費用を選択しない。

| 登録候補 | cosine RZ | sine RZ | PR cohort |
| --- | ---: | ---: | --- |
| B0_p1_q1_2nd | 133 | 133 | 決定論 |
| B0_p1_q1_4th | 329 | 329 | 決定論 |
| B0I_p1_q1_2nd | 133 | 133 | 決定論 |
| B0I_p1_q1_4th | 329 | 329 | 決定論 |
| B1_p3_q1_2nd | 368 | 368 | 決定論 |
| B1_p3_q1_4th | 1,034 | 1,034 | 決定論 |
| B2_p1_q1_2nd_r2_K2 | 297 | 297 | native診断n1・S0_ONLY |
| B2_p2_q2_2nd_r4_K4 | 784 | 784 | native診断n1・S0_ONLY |

PRは元の無条件lawで各1軌道を生成し、同じ軌道を両axisに使用した。
cohortは `FP_NATIVE_DIAGNOSTIC_N1`。条件付き・強制希少eventを実FPへ入れていない。
空tailによる退化・候補置換は発生していない。
両PRはS0のみであり、希少S2/S4の実出現・費用分布・母平均・優位性は認定しない。
Aではこのn1費用を混合せず、適格な66登録候補のそれぞれに新n8を用いる契約を維持する。

保存されたreferenceのexpm/eigh state差は `2.0973428995247362e-16`、
独立occupation matrix差は `6.972611193684198e-32`。
primitiveの37実時間×3状態は全件検証し、負Yoshida時間を含む。
native/expm最大誤差 `8.749742427562724e-16`、spectral/expm最大誤差 `7.265198041559605e-16`。
8 native構成のcontrol最大誤差は `8.195371285584486e-15`。
これらは局所検査の保存値であり、厳密bias証明ではない。

## 実行・終了の証拠

実行attemptはH2所有 `attempts/FP_run01`。SOURCE/input/runtimeと元pending契約を保全し、
H2所有 `FP_activation01` に利用者承認・有効grant・fresh resource/seed receiptを結合した。
SOURCE/runtime入力は起動前後に照合し、終了監査は科学計算を再実行していない。

CPU affinityは64、観測時62論理CPUが低負荷で、空きメモリは約469 GiB。
候補CPU62から既存owned poolが最大8 taskを候補間並列化し、内部thread1で多重並列を避けた。
cgroup階層にCPU/RAM quotaの有限制限はなく、OS既存制御を変更・回避していない。
ホスト・上位cgroupの履歴や一時的な観測stalenessを計算停止条件に追加していない。
CPU/RAM/worker/wallの恣意的上限は設けず、共有環境・他jobは変更していない。

- native wall：`19.184018643572927` 秒、native peak RSS：392,355,840 bytes。
- signal worker6とcompile worker8、計14 workerは全てexit0・回収済み、result hash一致。
- native parentも同一driverがwait4で回収。native exit0、terminal/runner auditともCOMPLETE。
- retry0、中断0、driver/worker STOPなし。次stage自動起動なし。
- seedは事前固定2 master/4 slotを維持し、redrawしていない。
- 既知72,080値とH2 A予約3,888値、計75,968 seed整数に衝突0。
- geometry登録hash不変を確認し、既存7,737,216予約slotとの衝突0の証拠を再利用。再samplingなし。
- F関連295レコード・元準備seal・activationを保全し、新FP完了204 filesもprivate manifestで保存した。

公開用保存先は
[`FP_completed_run01`](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/FP_completed_run01/)。
result・signals・preparations・primitive・control・native exit・worker receipts・grant binding・台帳・
終了監査・保全receiptの20 filesを保存した。
`result.json` SHA256：`0b43326c03b1e6786841920d0f38ad5db01fe5c22d9e1172e2354b87987bf80f`。
有効grant本体とraw logs/worker出力はprivateに保全し、pushは行っていない。

保存bytes監査は `scripts/audit_h2_fp_completed.py`。
科学packageを無効にしたPython `-S` で、SOURCE・grant binding・seed・worker identity/reap・
fingerprint・call台帳・F保全・後段未実行を検査した。
この検査は主担当Codexによるもので、別reviewerの独立科学レビューではない。
前段の人工・回帰54件の証拠は元preparationに保全し、無変更の科学SOURCEで実FPを1回だけ実行した。

## 次の作業と残る制約

| 担当 | 作業・判断 |
| --- | --- |
| ユーザー | FP完了結果を正式に受け入れるか判断し、Dのone-shot実行を別途明示承認する。公開する場合はpush対象も指定する。 |
| Codex | FP正式受入後にDの依存result/audit/acceptance SHAを結合する。D個別承認後だけfresh resource・grantを用意し、Dだけを1回実行・監査してSTOPする。 |

D70/A66の探索結果は未取得で、PF対照・PRの費用優劣・適格候補の有無は未判定。
H4/H6/H8と会計・compiler・native controlを共通化したが、分子・rank・sector・探索集合はH2固有である。
FPのn1はH4 FP n32相当の統計的証拠を与えない。
S_STATS・P最終seal・P科学実行・詳細科学レビューには進まない。
