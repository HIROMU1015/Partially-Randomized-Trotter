# H2 P：固定92軌道の独立確認・終了監査完了（2026-10-19）

**COMPLETE_MANDATORY_STOP。n80/12の独立92 trajectory・184 compiled軸を1回実行し、2 endpointの費用差方向を再現した。**
native exit0・同一親wait4・92 worker全回収・固定call台帳・保存bytes監査PASS。
有限標本の点比較であり、母平均優位・実power・希少tailの保証ではない。
追加sampling・retry・候補再選択・詳細科学レビュー・論文最終evidence freezeは0。新規pushなし。

## 承認・固定SOURCE・入力

利用者が「P科学実行」を選択し、「承認する」と明示承認した。
[P最終seal](track_a_h2_p_final_seal_20261019.md)の科学SOURCE・n・seed・入力・PF/shot・予算を維持し、
新しい`P_activation01`の実行spec・承認receipt・有効grantへ結合した。
旧seal承認を科学実行承認に使い回さず、今回の明示承認を記録した。

- 科学SOURCE：`ce1f511c4f7429190adcee58cde86be8a87a2fb1`（24 files、変更なし）。
- 最終seal commit：`6eacd7f82afb1c6d3fa6b66987df8dcfa1ed8bcb`。
- 消費前private spec digest：`17356c8c3fa9de3a3f779635728815529a03285db9cb4d91b2397a73b3694c59`。
- 実行spec digest：`43991ca79559bd1957fd8588e63c0b78bcf6b8f8faa9ef02519c6187279bc168`。
- seed plan digest：`e6a19d26f531d4a16ec8e5dcab0374fcc306d609f5399046dbca7c80a444ad77`。
- private seed registry SHA256：`667d4310f20e270b121a60396b8333f6fb62c10eaf71e1727487dc7c1dc71ace`。
- 科学result SHA256：`e5ad454f1588eeca626f14e099985897e56ba732d9301c3e82650fd1972392a1`。
- 終了監査metadata SOURCE：`3252a1202951c04626a6545463889a8642e2718c`。

SOURCE24・入力59・provider262・RTE kernel4・runtime12,202 filesとPython executableのbytesを起動前後に照合した。
F/FP/D/A/S_STATSの完了commit・科学SOURCE・result SHA・native終了証拠を維持した。
依存commitはF `7ea6df2d…`、FP `af724928…`、D `7b202a1a…`、A `1d2f67ec…`、S_STATS `829d697a…`である。
未確認の旧D_run01部分費用は混合せず、元STOP・旧予約を保持した。

## 科学条件と実行した2 endpoint

linear H2、結合距離1.0 Å、STO-3G、tol-only DF1e-8の自動rank3、4 system qubits＋ancilla。
Nalpha=Nbeta=1のspin sector次元4、T=.8、状態準備費用P=0、通常signed bias、allowance倍率1。
full Hadamard wrapperのcosine/sine各compiled RZ×各保存必要shotを合計する。
Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routingは未指定。
native symmetric directional、original scalar＋tail identityは1回だけ、full-H referenceは無回転。
受入FのHamiltonian・状態・DF分解を再生成・再正規化せず、旧H2のrank指定DF・sector6は混用しない。

| ε | PR候補 | L_D / q / r / K / delta | 固定n | 保存PF対照（S2） |
| --- | --- | --- | ---: | --- |
| .05 | B2_p1_q1_2nd_r2_K4 | 1 / 1 / 2 / 4 / .8 | 80 | full PF B1_p3_q1_2nd |
| .0005 | B2_p1_q4_2nd_r2_K4 | 1 / 4 / 2 / 4 / .2 | 12 | full PF B1_p3_q4_2nd |

signal・必要shotはAの保存値を使い、PFはDの保存対照を使った。
必要shotのcosine/sineはそれぞれ8,037/7,940、119,134,013/107,845,386。
Pはこの2 endpointのみを対象とし、Aの5精度を再探索しない。
新cohortは`INDEPENDENT_H2_CONFIRMATION`。探索A n8・FP診断n1・他H系列の費用標本は混合しない。
候補間CRNなし、同一軌道のcosine/sineだけをpairedにした。

## 保存された独立確認の点比較

統計会計は承認されたP内で1 pass、2比較だけを実行した。
以下は保存結果の転記であり、報告・終了監査で平均・SE・費用差を再計算していない。
基準方向はS_STATSに保存されたPF−PR>0であり、独立Pの保存方向が一致した。

| ε | PのPR平均compiled RZ×shot（exact） | 保存paired SE | 保存PF費用 | Pの保存status |
| --- | ---: | ---: | ---: | --- |
| .05 | 384135011/80 | 15992.018778605378 | 5,533,984 | REPRODUCED |
| .0005 | 998482376201/4 | 276760628.92856276 | 305,695,076,994 | REPRODUCED |

両endpointでPRの点推定が固定PF対照を下回る方向を再現した。
これは固定候補・固定n・通常signed/factor1の有限標本の確認であり、PR一般の優位性や母平均の優位性の認定ではない。
追加のCI/p値、bootstrap、選択後coverage、n/power感度、相対削減率の新解析は行わない。

保存eventのstratumはq1が`S0_ONLY:79, S2_SINGLE:1`、q4が`S0_ONLY:12`。
これらは実行証跡の件数であり、法則確率の推定や希少stratumの母平均・分散の認定には使わない。
法則確率、未観測event費用・分散はnull/unknownのまま保持する。
S_STATSのnは正規近似の計画値であり、実際のpower=.8や希少tailのcoverageを保証しない。

## 固定予算と実行照合

| category | 固定予算 | 保存された非返還予約・実行 |
| --- | ---: | ---: |
| 保存入力load / prefix1 native preparation | 1 / 1 | 1 / 1 |
| unconditional trajectory / native evolution | 92 / 92 | 92 / 92 |
| full Hadamard wrapper compile軸 | 184 | 184 |
| occurrence / RTE event | 128 / 256 | 128 / 256 |
| control probe / independent event oracle | 8 / 2 | 8 / 2 |
| endpoint統計会計pass / 比較 | 1 / 2 | 1 / 2 |
| SCF / 積分 / DF / 状態solver / 実quantum shots | 各0 | 各0 |
| full-H・truncated reference / primitive probe・spectrum | 各0 | 各0 |
| 新signal / forward / B0I overlay / signed shot設計 | 各0 | 各0 |
| PF compile / bootstrap / shot再計算 / 再選択 / 追加sampling | 各0 | 各0 |

同じprefix1 native preparationを親processで1回構築し、2候補とworkerで共用した。
preparation hashは保存値`331071a2af57eb13ac0b04843f3c1ec70b035ceb741b0a5605943eeca7345121`と一致した。
先頭replica0の2 workerをcontrol gateで受理した後、残り90 workerを実行した。
全92 task・両軸・seed/cohort/preparation・event digest・compiler/fingerprint・台帳を照合した。

native exit0、同一親wait4、wall`19.998608820140362`秒。92 workerは全てexit0・回収済み。
`wait4`のreceiptに保存したpeak RSSは393,846,784 bytesであり、worker全体のRAM合計としては扱わない。
control2件PASS、最大誤差`7.771561172376096e-16`、閾値1e-9未満。
自動retry・科学run中断・再起動・追加samplingは0。one-shot markerは消費済み。

## fresh資源・seed gate

起動前に許可CPU64とCPU使用状況を観測し、低負荷の62 CPUを選んだ。
MemAvailableは503,548,506,112 bytes。既知PIDの観測・cgroup制御の読取も記録し、他processは変更しない。
worker数は現在の許可・空きCPUとwaveのtask数から決定し、過去のworker数を恒常的上限にしない。
worker内thread1で多重BLASを避け、OS/cgroup/scheduler・共有venv・他jobを変更しない。
PSIは診断情報であり、一時的競合・無関係なworktreeのHEAD/statusをSTOP条件にしない。

新92 master・140 seed slotはseal時から不変、redrawなし。
既知30 seed originと除外78,174値の実bytesを再確認し、衝突0を確認した。
新seed集合とgeometry登録が不変のため、7,737,216予約slotへの封印済みmembership proofを再利用した。
大きなgeometry streamを重複計算せず、法則・Taylor order・componentの再samplingも行わない。
衝突0は固定された既知inventoryとの照合であり、確率的独立性や未知の将来campaignの網羅の証明ではない。

## 終了監査・保全・公開用証拠

`audit_h2_p_completed.py`はstdlibのmetadata検査だけを行う。
人工9件PASSでは保存2比較の対応、逆方向・同値の受入、不正n/PF/shot・母平均主張・unknown埋めの拒否、
public projectionからのseed除去、統計関数非dispatchを検査した。
元SOURCEの人工・回帰82件PASSも保持した。
終了監査は統計関数・標本数関数・科学module importを拒否するguard下でPASSした。
NPZ decode、sampling、compile、平均/SE/分散・費用統計会計の再実行は行わない。
主担当Codex自身の終了監査であり、別reviewerによる独立科学レビューではない。

F/FP/D/A/S_STATS・旧metadata STOP・原D STOP・grant/ledger/markerを含む旧7,272レコード、
最終seal63レコード、今回のattempt/activation等1,082 private filesをhash manifestへ保全した。
有効grant本体・new seed一覧・raw event・F snapshotはprivateに保持する。
公開用[完了projectionとnative証拠](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/)では、
保存比較とcompiled費用を転記し、master/step seedとraw eventを除いた。
[人工テスト・guard記録](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_checks01/)も保存した。
この公開用directoryへの保存はremote公開ではなく、今回はlocal commitのみでpushしない。

H4/H6/H8と会計・compiler・native control・独立確認の規則を共有し、H2固有のrank3・sector4・候補・nを使った。
既存H4/H6/H8のSOURCE・結果・cohort・契約・job・worktreeを変更・混合しない。
現在は独立Pの有限標本確認まで完了したが、母平均保証・厳密bias上界・大域最適性・論文最終evidence freezeは未取得である。

## 次の担当と停止境界

| 担当 | 次の作業・判断 |
| --- | --- |
| ユーザー | P完了結果と希少tailの制約を確認し、結果の受入・公開範囲を判断する。 |
| Codex | 公開指示がある場合だけ、その範囲をpushしてremote SHAを確認する。追加科学実行・詳細科学レビュー・最終freezeは進めない。 |

**mandatory STOP。追加の科学作業の予定なし。**
