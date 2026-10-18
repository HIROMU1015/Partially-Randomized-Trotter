# Track A：新A_COSTの保存値総費用比較・実行準備

2026-10-18。利用者の「その方針で進めて」に基づく**実装・人工検証・入力bindingの準備**。
**実データの再会計・SE計算・ランキングは未実行。新しい実行承認は受領していない。**
SOURCE：`8601b921342fc1cc83780a830e1f65bf64fe9e93`。
起点は[A/B/D完了報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc6bd039dc7171d5674ac04f0b3ad427f1b6495b/docs/research/track_a_pr_cost_batch_completed_results_20261018.md)。
[機械可読契約・検証・無効grant](../../artifacts/resource_applicability/track_a_pr_new_a_total_cost/2026-10-18/README.md)を併用する。

## 固定する評価

Linear H-chain、1.0 Å、STO-3G、T=0.8、P=0、既存S2 PR・元finite-RTE law、
symmetric-directional control。Hamiltonian、状態、DF順序、normalization、科学条件、compiler optionsを変えない。

| 系 | A入力系列・prefix | 新PR | 保存済み旧PR | 固定する決定論集合 | 保存shotの範囲 |
|---|---|---|---|---|---|
| H4 | legacy固定DF12、p2/3/4 | 16候補×探索n8 | 8候補×n32、fixed_anchor_panel。選択・推定に用いた標本として保持 | old_fixed_deterministic 8 / old_plus_direct_subset 16 | 5精度・通常/非相殺・倍率1/10 |
| H6 | tol-only 1e-8 DF19、p3/4/5 | 16候補×探索n8 | 確認8候補×n32 | old32 32 / strengthened_S2_S4 136 | 5精度・通常/非相殺・倍率1のみ |
| H8 | tol-only 1e-8 DF28、p6/7/8 | 16候補×探索n8 | 独立確認union6×n32 | old32 32 / strengthened184 184 | 5精度・通常/非相殺・倍率1のみ |

精度はε=.05/.01/.005/.001/.0005。新A候補48件と384 paired trajectory/768軸の登録は
[公開済み登録表と測定表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc6bd039dc7171d5674ac04f0b3ad427f1b6495b/artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/README.md)をそのまま使用する。
旧PR候補・決定論集合は今回のD完了payload/resultに固定し、別branchの新しい結果を自動採用しない。
H4の別rank10やgeometry系列、高次PF研究は対象外。

**H6/H8の新Aには倍率10の保存shotがない。各160 endpoint、計320 endpointが欠測。**
契約は20 endpoint/候補の出力枠を用意するが、欠測は`ACCOUNTING_MISSING`/nullにする。
通常biasからの代用、倍率1の流用、shot再計算、他候補の補完は行わない。
欠測を含むdomainでは「全登録新A最良」や削減率を確定しない。倍率1とH4の正常な評価は独立に進められる。
保存ありは640/960新候補endpointであり、そのすべてが精度適格という意味ではない。

## 算術・比較・不確かさ

各trajectoryの費用は `G_i = N_real C_i,cosine + N_imag C_i,sine`。
保存された正整数shotとN_totalの一致を確認し、既存`paired_statistics`で平均と標本SEを計算する。
同一trajectory内の両軸の共分散を保ち、軸別SEを独立として合成しない。
同じn8の回路費用を各endpointで再重み付けし、新しい乱択標本とは数えない。

比較表はendpointと決定論集合ごとに、次を分ける。

1. 新A探索n8内の取得済み適格な有限標本点費用。
2. 保存Dにある旧PRの費用・SE・cohort。旧PR平均やSEを再計算・poolingしない。
3. 保存Dの決定論点費用と原`cost_status`/軸証拠。直接実測、model推定、旧profile不明等の注記を保持。

各比較は新A対PF、旧PR対PF、新A対旧PRを別欄にする。exact fractionで平均費用・削減率を保存し、
最小値の同値時はID昇順。欠測・数値未解決をineligible/0費用へ変換しない。
利用可能subsetの最小値は表示しても、欠測domain全体の最良や削減率と同一視しない。
新Aと旧PRの差の検定・信頼区間、モデル誤差の確率分布は作らない。

新n8は探索結果であり、候補選択効果を補正した推定ではない。
H4の旧n32を新たな独立確認と表記しない。H6/H8も有限標本点比較という制約を保持する。
SEは母平均保証・厳密95%区間・最適候補保証ではない。
通常signed bias、経験的非相殺、厳密bias保証を分け、後者は認定しない。
B_ANALYSISの条件付きpilotは一切混ぜず、無条件期待費用の未確定はそのまま保持する。

## 入力とSOURCEの結合

3系ともA/DのHamiltonian/state/snapshot/compiler/profile/control bindingが一致した。
元signal record hash、固定128 taskのcell/replica/seed、両軸event digest、完了result/terminal SHA、
元parent native exit0/reap receipt、旧比較集合と保存rowの完全性をmetadataとして確認した。
384 taskの重い監査や過去の人工40件を再実行していない。
入力bytes/SHA、Python実行file、scalar解析のstdlib/ロード済みruntime葉とSOURCE closureをsealした。
旧compile runtime監査receiptは来歴として結合し、venvのinstall/upgrade/設定変更は0。

新A取得SOURCEはH4/H6が31ef62a1…、H8の残112件が87f1f0d5…。
H8の元16代表と全128件のseed導出SOURCEは31ef62a1…のまま。
H8のDは修正後8ff002dd…、旧STOPや未完了予約を変更・返却しない。
新会計SOURCEへの変更は新しいsamplingやseed導出を意味しない。

compilerは既存rz/sx/x/cx、optimization1、seed17。
H4/H6 profileは既公開fileのSHA照合済み。H8の622ce4a8…は**元native契約の識別子**であり、
今回確認できたprofile原本fileのbyte SHAとは主張しない。旧compilerとの全output一致は未検証のまま。

private原本・sealed契約は `/home/AbeHiromu/projects/pr-new-a-total-cost-preparation/20261018/sealed03/`。
public契約にはtechnical digestと件数のみを掲載し、private payload/shot表/運用原本は追加公開しない。

| 独立scope | technical digest |
|---|---|
| H4_NEW_A_TOTAL_COST | `54550fb2b84846441abb5fbe98315d98a46d04ad94716493e30c62fa6e970841` |
| H6_NEW_A_TOTAL_COST | `5782d343ed8ab2c2a7eee52823d7745cce2e1a287dbfaa2a8f851fbce67acd78` |
| H8_NEW_A_TOTAL_COST | `1b89eac6320369cc8348a36b93333c5e1d166c9835f843c79acbd236c9dc0fc8` |

## 実装・人工検査

[会計adapter](../../scripts/pr_new_a_accounting.py)、[実行gate](../../scripts/run_pr_new_a_accounting.py)、
[metadata準備](../../scripts/prepare_pr_new_a_accounting.py)、[人工tests](../../tests/test_pr_new_a_accounting.py)。
既存`pr_cost_saved.signal_accounting`、`pr_cost_integration.account_rows/paired_statistics/compare`、
保存shot検査を再利用する。新しいprecision solver、native adapter、sampler、compilerは作らない。

人工25件PASS。paired費用/SE、cohort分離、固定seed/cell/profile、欠測倍率10・非相殺の非代用、
不適格と未解決の区別、入力不変、保存shot sum、重複/不足row、旧grant拒否、承認receipt/digest、
CLIで入力読込前に拒否、one-shot markerと最初のSTOP保持、H8 profile識別子の区別を確認した。
構造auditから統計・shot・ranking関数を呼べないこともmockで検証した。
作者の限定技術検査であり、他担当の独立reviewや実データでの結果再現とは呼ばない。

初回metadata準備はH8の旧schemaに`profile_reference`がないため停止した。
原記録とsealed01/02を保持し、既存識別子を明示的に継承する修正を行った。
公開SOURCEのactual SHAへ結合したsealed03が現契約。科学attempt・grant消費は発生していない。
旧結果・SOURCE・grant・ledger・marker、他worktree/jobは変更していない。

## 将来の起動・資源・STOP

一括承認を受けたscopeを**各1process、最大3系同時**に処理する。
3は登録system数で、計算資源に対する恒常的なCPU/worker上限ではない。
小さい保存scalar処理のため、candidate/precisionごとのprocess増殖や入力の重複読込を避ける。
std-libraryのみで既存private Pythonを使用し、`-P -B`、全一時fileはhome内、GPU/provider importなし。
新旧SOURCE・他jobのCPU割当は変えず、無関係なworktree更新や一時的PSIをSTOP条件にしない。

admission見積りは1系memory512MiB・output64MiB・32 inode、3系で1.5GiB・192MiB・96 inode。
これは予約量や強制上限ではない。phase/total wall上限、AS/RSS cap、tasksetは新設しない。
各系の入力role合計は約75/86/81MB（重複roleを含む）で、科学array/NPZを読まない。
起動直前に空きmemory・filesystem block/inode・quotaの読取確認を行う。
quota非公開なら明示されたENOSPC fail-closed方針の承認を必要とし、無制限と推測しない。

SOURCE/blob・input bytes/SHA・compiler/profile来歴・signal hash・seed/両軸・保存shot・候補集合の
不一致、重複、未完了/不正なresult、非finite数値、I/O失敗は当該scopeのfail-closed STOP。
未保存endpointは事前登録の欠測としてnullにする。正常な他scopeは独立して継続できる。
output/lockは新規未使用域、before-call予約はsaved_scalar_pass=1/系、成功・失敗後ともmarkerを削除しない。
retryなし。出力と入力の最終byte gate後に完了terminalを保存する。

新signal/trajectory/sampling/native build/compile/SCF/DF/state/fit/preparation/量子shot実行は各0。
新しい独立確認候補の採択・標本数決定・samplingには進まない。結果保存・公開後mandatory STOP。

## 次の承認案

ユーザーが対象scopeを個別指定した一度の承認を出せる。

> SOURCE `8601b921342fc1cc83780a830e1f65bf64fe9e93` と上表のtechnical digestに固定した
> H4/H6/H8_NEW_A_TOTAL_COSTの3契約を承認する。既存保存shotとn8費用による
> 総費用・paired SE・通常/経験的非相殺の限定ランキングを各1回、独立並列実行する。
> H6/H8の新A倍率10は欠測のままとし、shot再計算で補わない。
> 新sampling/compile/fit、旧標本とのpooling、独立確認への自動進行、retryは禁止する。
> quotaを確認できない場合はその事実を記録し、ENOSPC等ではfail-closed STOPする。
> 結果を保存・公開してmandatory STOPする。

現在は`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、
`absolute_launch_command=null`。新科学実行0、新有効grant0。
**次の担当はユーザー：対象scopeと上記範囲の実行承認。承認後の担当はCodex：fresh gateと限定再会計・公開。**

公開範囲：大きな既存索引全文の転送は自動承認審査で拒否されたため、旧索引を元bytesで保持し、
本資料・専用bundle・日付付きノートだけを軽量な追補として公開する。原科学結果やprivate運用原本は追加公開しない。
