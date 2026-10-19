# H2 FP・D・Aの技術準備（2026-10-19）

F完了commit `7ea6df2dbe34e64a34f468c343bb12922d3e2829` を利用者指示により正式受入。
FP/D/A adapter・候補・SOURCE・保存F入力・compiler・科学call上限を固定した。
実H2のFP/D/A signal・sampling・compile・数値解析は0。有効grantと実行markerは作成していない。
**TECHNICAL_PREPARATION_MANDATORY_STOP**。S_STATS/P、詳細科学レビュー、論文evidence freezeへ進まない。

科学SOURCEは `de72e5d4e48177734c0ddd79dffac14222410658`。
最初の実装commit `02174917` に続き、metadata preparerが既存seed inventoryの
`(rows, workers)` 戻り値を受け取る修正をこのSOURCEに含めた。
公開承認の対象を特定できる新たな指示はないため、今回のpushは行っていない。

## 入力・共通会計・再利用

linear H2、1.0 Å、STO-3G、tol-only DF `1e-8`、rank自動決定の実rank3。
4 qubits、`Nalpha=Nbeta=1` のspin sector次元4。Fの保存状態をそのまま用い、再生成・再正規化しない。
旧H2のrank指定DF・total-number sector6のデータは混用しない。

| 固定対象 | SHA256 |
| --- | --- |
| F snapshot | `62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa` |
| Hamiltonian | `d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80` |
| 保存状態 | `080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf` |
| F result | `63acdc965e57670531fa689824c8efddd489374e95a03c05ca723a038d6ed2c5` |
| compiler profile | `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` |

T=0.8、P=0、ε=0.05/0.01/0.005/0.001/0.0005、通常signed bias、allowance倍率1。
既存のaxis精度分割とaxis α=0.025、empirical allowanceを用いる。
full Hadamard wrapperのcompiled RZ数に、各axisの必要shotを掛けて合計する。
必要shotは資源見積りの重みであり、量子shotを実際に生成しない。
厳密bias証明・母平均保証・登録外の大域最適性は認定しない。

compilerはQiskit 1.3.0、basis `[rz,sx,x,cx]`、optimization level1、transpiler seed17。
backend/coupling/layout/routingはNone。controlは既存のsymmetric directional、low branch I/high branch U。
original scalarと抽出tail identityを1回だけ適用し、full-H referenceは回転しない。
compiler hash自体の再計算・照合は各stageの承認後に行う。

既存F provider `217c971dd537b1cc7fbc13742bc0f6c263ff7f45` のH2所有snapshot 262 filesと、
A kernel `31ef62a12407010e9c1a07f9b37facedb8dff5fe` の必要closure 4 filesを読取再利用する。
前者のMatchedPort、独立occupation oracle、native DF・wrapper・compiler、
後者の無条件RTE stepsとcontrolを利用する。費用統計・共分散とowned worker poolも既存実装を再利用する。
H2側SOURCE closureは23 files、runtime manifestは12,202 files。FのSOURCEを編集していない。

新規コードは `scripts/h2_panels_contract.py`、`h2_panels_science.py`、
`run_h2_panels.py`、`prepare_h2_panels.py`。
実行入口は承認検査後にscienceをimportする。F保存入力のloaderもその後だけ呼ぶ。
metadata preparerはNPZをdecodeせず、保存bytes・Git object・seed整数だけを処理する。

## 事前登録した候補

| stage | 集合 | 用途 |
| --- | --- | --- |
| FP | B0/B0I prefix1とfull PF prefix3、各q1・S2/S4の6条件。PRは `B2_p1_q1_2nd_r2_K2` と `B2_p2_q2_2nd_r4_K4` | 計8条件。決定論6条件と非退化PR各1軌道のnative診断 |
| D | B0/B0I prefix0/1/2、q=1/2/4/8/16、S2/S4。full PF B1 prefix3も同q・両formula | 計70条件。5精度の登録内最良PFを選択 |
| A | B2 prefix0/1/2、全q・r4・K2/K4。追加で各prefixのq1/4/16にr2/r8・K2/K4 | 計66条件。適格候補ごとに新規n8 |

prefix0は一体項だけを決定論側に保持する意味のある候補として含める。
full-rank B0/B0IはB1と重複するので登録せず、prefix3のPRは空tailなので登録しない。
prefix0のS2/S4は理想unitaryが一致し得るが、native分解・compiler出力を勝手に同一視しない。
それ以外のprefix間の数値的な等価性は、この準備では実H2入力を解析して判定していない。
候補IDの構造重複は拒否する。

tail normalizationが厳密に0なら `DEGENERATE_EMPTY_TAIL` と明示し、全endpointを不適格にして
sampling/compileを行わない。負値・非有限値・eligibility未解決はSTOP。
丸めで正の小tailを0扱いせず、別候補への置換や自動探索拡大はしない。
q16、r2/r8、prefix0/2のwinnerは登録境界の結果として扱う。
PR不利・適格PRなし・適格PFなしを許容し、未取得費用を0で埋めない。
合法なRZ数0を保持し、PF費用0なら比率はnullと理由を記録する。

## 科学call予算（すべて承認後の上限）

| category | FP | D | A |
| --- | ---: | ---: | ---: |
| full-H reference pair（expm/eigh agreement） | 1 | 0 | 0 |
| full-H primary / independent columns | 4 / 4 | 0 / 0 | 0 / 0 |
| 新規truncated reference pair | 1 | 2 | 0 |
| native preparation再構築 | 4 | 4 | 3 |
| primitive probe | 111 | 411 | 90 |
| 独立primitive sector spectrum | 4 | 4 | 4 |
| 独立tail matrix | 2 | 0 | 3 |
| 新規base signal / 独立forward | 6 / 6 | 36 / 36 | 64 / 64 |
| analytic B0I overlay | 2 | 28 | 0 |
| 新規signed shot endpoint | 40 | 320 | 320 |
| trajectory | 2 | 0 | 528 |
| occurrence | 3 | 0 | 3,504 |
| RTE event | 10 | 0 | 16,032 |
| native evolution | 8 | 64 | 528 |
| wrapper compile | 16 | 128 | 1,056 |
| control probe | 32 | 256 | 264 |
| 独立event oracle | 2 | 0 | 66 |
| endpoint会計pass | 0 | 1 | 1 |

SCF・積分生成・DF・保存状態solver・実quantum shotsは全stage 0。
primitive spectrumやreferenceのeighは、保存状態を作るsolverとは別に上表で予算化している。
signal内のTaylor matvec・決定論actionは固定q/r/K/formulaに従う既存ActionBudgetで検査する。
primitiveは実fragment/timeの重複を除き、各3状態を検査し、S4の負Yoshida時間も含む。
native instruction guard 1,000,000は既存の構築検査として保持する。

DはFPの決定論signal6条件（raw4＋overlay2）・compiled pair6条件を再利用し、
truncated reference prefix1とfull-H referenceも再利用する。
Dの会計は全350 endpoint、うち新規shot計算320。
AはFPのPR ensemble signal2条件だけを再利用し、全330 endpointを会計する。
FPのPR n1費用は `FP_NATIVE_DIAGNOSTIC_N1`。
Aの66条件にはそれぞれ新規n8を確保し、FP費用を混合しない。
両axisは同一軌道を共有し、候補内5精度のsample covarianceを保存する。
最大値は適格性が成立した場合の範囲であり、候補追加・n増加・再試行の許可ではない。

## 固定契約と、今後必要な結合

公開用機械資料は
[`fp_d_a_preparation01`](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/fp_d_a_preparation01/)。
各stageのpreregistrationにSOURCE・全候補・compiler・call予算・入力hash・private契約hashを記載した。
private契約はH2所有 `fp_d_a_preparation01/{FP,D,A}_spec_PENDING.json` に保存する。
全契約は `SEALED_PENDING_APPROVAL`、公開grantは明示的にINVALIDである。

FPの科学入力bytesは固定済み。D/AはSOURCE・F入力・予算・依存結果の受入規則を固定したが、
**まだ存在しないFP/D結果の最終SHA結合は未完了**であり、実行可能な最終入力sealとは区別する。
今後のactivationは元pending契約を保存し、SOURCE・plan・F入力・compiler・call予算を変更せず、
次の項目を結合する。変更が必要なら元のsealを残して新しい来歴と承認を用意する。

1. stage単位の明示的な利用者承認、発言receiptとそのhash、有効grant。
2. fresh CPU affinity・空きメモリ・既知の並行割当との非干渉を確認したH2所有resource receiptとworker CPU集合。
3. FP/Aは、固定seedを変更しないactivation時の既知campaign追加照合receipt。
4. Dは正式受入したFPのresult・completion audit・user acceptance各bytesとSHA。
5. Aは同じFPに加え、正式受入したDのresult・completion audit・user acceptance各bytesとSHA。

依存SHA・承認・資源receiptのどれかが未結合なら起動を拒否する。
承認後の起動は `run_h2_panels.py --spec <そのstageのexecution契約> --grant <そのstageの承認grant>`。
専用attemptは `FP_run01`、`D_run01`、`A_run01`。atomic claim後は再使用不可。
call予約は非返還、異常時は証拠とSTOPを保存し、自動retryしない。
親はnative childをwait4で回収し、各workerの終了・result hashを検査する。
正常終了でもmandatory STOPし、次stageを起動しない。

## seed・保全・検証証跡

FPは2 master/4 seed slot、Aは528 master/3,888 slot。
H2独自namespace・SOURCE・plan・input binding・candidate・replicaからSHA256 first64bitでmasterを固定し、
元のdirect-master step hierarchyを用いる。FP/A相互・既知72,080整数・
6 geometry予約7,737,216 slotとの衝突0。RTE event samplingは行っていない。
参照した既存inventoryのcutoffは `2026-10-19T05:40:56.655233+00:00`。
将来の未登録campaign全体の網羅や統計的独立性の証明を主張せず、activation時に新登録範囲を再照合する。
照合は既知の封印済みmetadataをH2内へコピーして行い、他worktreeのHEAD/statusをfreezeしない。

人工・回帰54件PASS（新規38＋既存F16）。人工4-qubit係数・基底状態を使い、
S2/S4・prefix0・B0I・full wrapper・実Qiskit compiler・PR K2/K4・強制S2/S4 event controlを検査した。
強制eventは人工fixtureだけの条件付き診断であり、実FP/Aの無条件samplingに使わない。
新規grant・SOURCE・runtime・入力・provider・kernel・予算・依存結果・seed refreshの拒否も検査した。
人工希少fixtureの旧API名を修正した履歴と途中ログはprivateに保持し、最終54件は全PASS。
SOURCE固定後のmetadata preparer完走も別途PASSした。
現在の実pending契約で、driver/worker/direct adapterの計9入口が無効grantを科学import前に拒否した。

F保存SOURCE/provider等266レコードとF完了29レコード、計295のhash一致を準備前後に確認。
Fのgrant・ledger・marker・STOPと既存raw bytesを保全し、新FP/D/A attemptは未作成。
H4/H6/H8のSOURCE・契約・結果・jobを変更していない。
検査は主担当Codexによるもので、別AI reviewerの独立科学レビューではない。

CPU/RAM/worker/wallの恣意的上限を導入しない。重い独立taskは既存owned poolで並列化し、
worker内thread1は外側並列との多重化回避に用いる。
一時的PSI上昇や他worktreeの資料更新を強制停止条件にしない。
共有OS・環境・他利用者processへの変更は行わない。

## 次の作業

| 担当 | 作業・必要な判断 |
| --- | --- |
| ユーザー | 上記SOURCEとFPの範囲・予算を確認し、FPだけのone-shotを別途明示承認する。公開する場合は新成果物の対象範囲を指定する。 |
| Codex | FP承認後のみfresh resource/seed照合とgrant結合を行い、FPだけを実行・終了監査してSTOPする。 |
| ユーザー／Codex | その後、FPの正式受入・D依存SHA結合・D個別承認、D正式受入・A依存SHA結合・A個別承認の順で進める。今回は実行しない。 |

H4/H6/H8との会計条件は共通だが、分子・実rank・sector次元・探索集合はH2固有。
H2 FP n1はnative診断であり、H4 FP n32の精度の統計的再現を主張しない。
H2の費用優劣・適格候補の有無・S_STATS標本数・P最終sealは未取得である。
