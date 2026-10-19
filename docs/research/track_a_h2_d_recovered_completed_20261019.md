# H2 D：保存signalを用いたD_run02完了（2026-10-19）

**D70の5精度PF対照を保存し、COMPLETE_MANDATORY_STOP。**
native exit0・同一親wait4・28 worker全回収・固定予算照合・保存bytes監査がPASSした。
科学実行はD_run02の1回だけで、A/S_STATS/P・詳細科学レビューには進んでいない。
成果はH2専用branchへのlocal commitで保存し、新規pushは行わない。

## 承認と旧STOPの扱い

利用者は先の中断について「これは私が間違って切断しただけ何度絵気にしなくてよい」と説明し、
続いて「その方針で進めて。元もと行う予定だったところまで進めてよい」と指示した。
これを、保存signalの入力受入と、固定した追加予算によるD完了・監査までの承認に結合した。
切断耐性のある新launcherの開発や、再度の承認要求は追加していない。
既存のsupervised driver・同一親wait4・owned worker poolを用いた。

[D_run01のSTOP](track_a_h2_d_interrupted_20261019.md)はそのまま保全した。
利用者の説明と、元runにnative exit・terminal・最終resultが残っていない事実を区別し、
元runを後からCOMPLETEへ変更したり、存在しない回収receiptを作ったりしていない。
元marker・grantは再使用せず、新SOURCE・grant・未使用のD_run02へ結合した。
旧raw compile 12組は採用せず、旧56軸の予約返還は0。
今回の追加56軸と合わせ、Dのcompile予約は累計112軸である。
これはオフライン検証の予約台帳であり、下表のアルゴリズム資源費用とは別の量である。

## SOURCE・入力・候補集合

- 再開SOURCE：`cf99e4452df96291d7e9d37d76d2e91be5cbc413`。
- 元FP/D/A科学SOURCE：`de72e5d4e48177734c0ddd79dffac14222410658`。
- 再開のimport closure 25 filesと元closure 23 filesを検査。和集合29 files、元23 filesは無変更。
- F受入commit：`7ea6df2dbe34e64a34f468c343bb12922d3e2829`。
- FP完了commit：`af724928689be3592b391fab3a14538bf4752b76`。
- linear H2、距離1.0 Å、STO-3G、tol-only DF1e-8・rank自動決定3、4 qubits。
- Nalpha=Nbeta=1のspin sector次元4。保存状態の再生成・再正規化なし。
- T=.8、P=0、ε=.05/.01/.005/.001/.0005、通常signed bias、allowance倍率1。
- full Hadamard wrapperのcosine/sine各compiled RZと、保存済み各必要shotの積を合計。
- Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routingは未指定。
- symmetric directional、original scalar＋tail identityを1回だけ適用、full-H referenceは無回転。

snapshot SHA256は `62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa`。
Hamiltonian `d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80`、
state `080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf`、
compiler profile `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` を維持した。
provider262 files・runtime12,202 filesも起動前後に照合した。旧H2のrank指定DF・sector6は混用しない。

元のD70集合を変更していない。B0/B0Iはsplit prefix L_D=0/1/2、full PFはprefix3、
各q=1/2/4/8/16、S2/S4を登録。delta=T/q=.8/.4/.2/.1/.05の範囲である。
full-rank B0/B0IはB1のaliasとして重複登録せず、prefix0のideal unitary同値でもnative分解は区別する。
適格性は旧保存endpointをそのまま使用し、不適格の費用を0に置き換えない。

再開入力9件は固定SHAに結合した。保存70 signalは元Dの36 base＋28 overlay＋FP6に由来し、
元signal poolの36 workerはexit0・回収・result hash一致のreceiptを持つ。
FP6 signalと決定論的compile pair6組は元保存値と一致する。
今回の新compileはB0/B0I prefix2の20条件とfull PF prefix3・q>1の8条件、計28条件。
新native preparation 2件のhashも元D保存preparationと一致した。

## 固定追加予算と実行結果

| category | 固定予算 | 完了・予約照合 |
| --- | ---: | ---: |
| 保存入力load | 1 | 1 |
| native preparation | 2 | 2 |
| native evolution | 28 | 28 |
| Hadamard wrapper compile軸 | 56 | 56 |
| native control probe | 112 | 112 |
| endpoint会計pass | 1 | 1 |
| SCF / 積分 / DF / 保存状態solver | 各0 | 各0 |
| full / truncated reference生成 | 各0 | 各0 |
| primitive probe / spectrum | 各0 | 各0 |
| base signal / independent forward / B0I overlay | 各0 | 各0 |
| 新signed shot endpoint | 0 | 0 |
| trajectory / occurrence / RTE event / 実quantum shots | 各0 | 各0 |

保存shotは資源会計上の必要数であり、測定shotを生成していない。
追加予算は起動前にSOURCE・入力・承認receipt・one-shot grantへ結合した。
全waveの予約は非返還。自動retry・旧attempt再起動・次stage自動移行は0。

## 保存済み最良PF

会計は1 passだけ実行し、登録70条件×5精度の350行と5件の最良PFを保存した。
以下は保存resultからの転記であり、報告・監査時の再会計や新順位計算ではない。
全winnerはfull PF、L_D=3、S2である。

| ε | 保存winner | cosine必要shot | sine必要shot | 総compiled RZ×shot |
| --- | --- | ---: | ---: | ---: |
| .05 | B1_p3_q1_2nd | 7,565 | 7,473 | 5,533,984 |
| .01 | B1_p3_q1_2nd | 264,714 | 246,580 | 188,156,192 |
| .005 | B1_p3_q2_2nd | 849,095 | 820,712 | 1,170,534,707 |
| .001 | B1_p3_q4_2nd | 22,307,132 | 21,345,852 | 59,673,629,128 |
| .0005 | B1_p3_q4_2nd | 117,378,315 | 106,246,467 | 305,695,076,994 |

これは登録した集合と保存済みsigned endpoint規則における対照である。
全q・全分解の大域最適性や厳密bias上界を認定しない。
PR探索は未実行であり、PR優位・不利・適格候補の有無はまだ結論しない。

## 検証・資源・証拠

新しい人工・回帰テスト27件がPASS。
無効grantのdriver/worker/adapter拒否、入力・SOURCE・予算改変拒否、
旧partial費用の非混合、人工4-qubitの独立occupation/expmとnative wrapperを検査した。
最初の人工実行ではfixtureのstaticmethod指定に1件誤りがあり、fixture修正後に27件全てPASS。
実H2 arraysは人工テストで読み込んでいない。元SOURCEの人工54件の証拠も保持し、再実行していない。

実行前のaffinityは64 CPU、低負荷候補61 CPU、MemAvailableは491,518,052 kB。
観測したcgroup階層にCPU/RAMの有限quotaはなかった。28独立taskをowned poolで並列化し、
内部thread1により多重並列を避けた。CPU/RAM/worker/wallへの恣意的上限、
共有設定の変更、他jobの停止・変更は行わない。無関係なworktreeや一時的PSIを停止条件にしない。

- native wall：62.838754726573825秒、native単一processのpeak RSS：477,265,920 bytes。
- 28 compile workerは全てexit0・同一親回収・result hash一致。nativeも同一driverがwait4回収。
- native control最大誤差：`8.003714750568779e-14`、登録閾値1e-9未満。
- runner completion、terminal、stdlibのみの保存bytes監査がCOMPLETE_MANDATORY_STOP。
- F295レコード、FP204 files、元D541 filesのhashを照合。新D完了314 private filesもmanifestで保全。

保存bytes監査は `scripts/audit_h2_d_recovery_completed.py`。
NPZ decode・signal/compile・会計を再実行せず、SOURCE/入力/台帳/worker identity/reap・保存350行を検査する。
これは主担当Codexの監査であり、別reviewerによる独立科学レビューではない。

公開用保存先：
[`D_recovery_preparation01`](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/D_recovery_preparation01/)（7 files）、
[`D_completed_run02`](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/D_completed_run02/)（21 files）。
有効grant本体・F snapshot・worker raw/logsはprivateに保持する。
`result.json` SHA256：`a7d75179415b14f21ec7e77ddc3e95b4dc5c7f406d664fddbf0755431f19454c`。
旧public artifact、SOURCE・grant・ledger・marker・STOPは上書きしていない。

## 比較可能性と次の作業

H4/H6/H8とtol-only、signed/factor1、full wrapper会計、native controlの規則を共通化した。
分子、rank、sector、探索集合、状態solverの来歴はH2固有である。
H2はDまでであり、H4等のA/S_STATS/独立Pと同じ証拠段階には達していない。
FPのPR n1はnative診断に限り、後続Aの新n8へ混合しない。

| 担当 | 次の作業・判断 |
| --- | --- |
| ユーザー | 本D完了結果を正式受入し、次にAのみのone-shotを承認するか判断する。公開する場合はpush対象も指定する。 |
| Codex | D正式受入後、A66の固定SOURCE・依存result/audit/acceptance SHA・新n8 seed・予算を結合する。Aはその個別明示承認後だけ実行し、再びSTOPする。 |

現在はmandatory STOP。A/S_STATS/P科学実行・P seal・詳細科学レビュー・論文最終evidence freezeには進まない。
