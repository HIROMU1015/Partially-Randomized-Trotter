追記（2026-10-19）：利用者の別途明示承認により固定92軌道をone-shot実行し、[P独立確認・終了監査を完了](track_a_h2_p_completed_20261019.md)した。2/2の点比較方向を再現しmandatory STOP。以下は最終seal時点の記録である。

# H2 P：最終SOURCE・独立seed・予算・one-shot契約の技術seal完了

2026-10-19（Asia/Tokyo）。**FINAL_SEAL_MANDATORY_STOP。Pの科学実行は未承認・未実施。**
利用者はS_STATS完了`829d697a`を正式採用し、ε=.05のn80、ε=.0005のn12を指定した。
固定92 trajectory・184 compiled軸・256 RTE eventを実行可能なH2専用adapterと契約へ結合した。
人工・回帰82件、無効grant拒否3入口、SOURCE/input/runtime・保全・seed監査PASS。
Pの有効grant、activation、科学attempt、one-shot marker、sampling、compileは作成・実行していない。

## 固定科学条件と対象

linear H2、結合距離1.0 Å、STO-3G、tol-only DF1e-8の自動rank3、4 system qubits＋ancilla。
Nalpha=Nbeta=1のspin sector次元4、T=.8、状態準備費用P=0。
通常signed bias、allowance倍率1、full Hadamard wrapperのcosine/sine各compiled RZ×各保存必要shotを合計する。
native symmetric directional、original scalar＋tail identityは1回だけ、full-H referenceは無回転。
Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling/layout/routingは未指定。
受入FのHamiltonian・状態・DF分解を再生成・再正規化しない。旧rank指定DF・sector6は混用しない。

| ε | 固定PR候補 | L_D / q / r / K / delta | 固定n | 保存PF対照（S2） | 保存shot cosine / sine |
| --- | --- | --- | ---: | --- | ---: |
| .05 | B2_p1_q1_2nd_r2_K4 | 1 / 1 / 2 / 4 / .8 | 80 | full PF B1_p3_q1_2nd、費用5,533,984 | 8,037 / 7,940 |
| .0005 | B2_p1_q4_2nd_r2_K4 | 1 / 4 / 2 / 4 / .2 | 12 | full PF B1_p3_q4_2nd、費用305,695,076,994 | 119,134,013 / 107,845,386 |

nはS_STATSの事前登録基準SD倍率4・検出差はPF費用の2%の保存提案をそのまま採用した。
今回はn/SD/quantile/power・signal・shot・PF費用の再計算、候補再選択を行わない。
Pの比較対象はこの2 endpointのみ。Aの5精度全てを再評価する契約ではない。
比較方向はS_STATSの保存方向+1を固定し、独立Pの結果が逆向き・同値でも除外や追加samplingをしない。

## 固定SOURCEと依存関係

P科学SOURCE：**`ce1f511c4f7429190adcee58cde86be8a87a2fb1`**。
新しいH2 Pのcontract・science adapter・driver・metadata seal・人工testを固定した。
既存`h2_panels_science.compile_one`の関数codeを、P専用の契約namespaceに結合して再利用する。
元H2/H4のfile bytes・module globals・法則・compilerを変更しない。
H2に必要な保存入力のloadとprefix1 native preparationを1回行い、2候補とworker間で共用する構成である。
保存preparation hash `331071a2af57eb13ac0b04843f3c1ec70b035ceb741b0a5605943eeca7345121`を実行時に検査する。
既存のnative control/event oracle、fingerprint再利用、owned並列pool、同一親wait4を使う。

| 依存stage | 完了commit | 科学SOURCE |
| --- | --- | --- |
| F | `7ea6df2dbe34e64a34f468c343bb12922d3e2829` | `37b1cf72990885962d8449d8bd5ec7e303329350` |
| FP | `af724928689be3592b391fab3a14538bf4752b76` | `de72e5d4e48177734c0ddd79dffac14222410658` |
| D | `7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb` | `cf99e4452df96291d7e9d37d76d2e91be5cbc413` |
| A | `1d2f67ec285956ef4090ea26520d64b11e925f9c` | `de72e5d4e48177734c0ddd79dffac14222410658` |
| S_STATS | `829d697ae7905b3ea90a61da9d2d4da15d8b757a` | `e0b959e8a617f7934d05c3e249832ccc1c43909f` |

各stageのresult・native終了監査・native exit・execution binding・公開manifestをGit固定のH2所有snapshotへ取り込んだ。
完了result SHA、exit0・同一親回収・native wait4、SOURCE bytesと保存入力の同一性を照合した。
Fの旧監査schemaにはwait4 fieldがないため、Fのnative exit receiptのwait4証拠を用いる。
F/FP/D/A/S_STATSの古い終了監査を次stage未実行条件付きのまま再dispatchせず、保存receiptとbytesを照合する。

P closureはSOURCE24 files、入力59件、凍結provider262 files・RTE kernel4 files、runtime12,202 files。
元Aの専用Python・compiler/runtime manifestと保存F snapshot/receiptをそのまま固定した。
科学SOURCEと技術sealを含む結果commitは別であり、以下のdigestはP科学SOURCEに結合した契約の値である。

- seed plan digest：`e6a19d26f531d4a16ec8e5dcab0374fcc306d609f5399046dbca7c80a444ad77`
- private seed registry SHA256：`667d4310f20e270b121a60396b8333f6fb62c10eaf71e1727487dc7c1dc71ace`
- private final spec digest：`17356c8c3fa9de3a3f779635728815529a03285db9cb4d91b2397a73b3694c59`

## 独立seed・衝突検査

H2専用namespace `H2_P_INDEPENDENT_CONFIRMATION_V1_20261019`、SOURCE、seed plan digest、candidate、replicaの
SHA256先頭64bitから92 masterを固定した。seed planはn・科学条件・保存PF/shot/方向・固定依存commitを含み、seed一覧自身は含めない。
q1はmasterをそのまま使用し、q4は既存`direct_master_to_step_seeds_v1`で4個のstep seedを派生する。
候補間CRNなし、cosine/sineだけ同一軌道をpairedにする。A/FPの費用標本は混合しない。

新一覧は**92 master・140 seed slot**。内部重複0、旧除外集合との衝突0、geometry streamとの衝突0。
旧集合は既知H4/H6/H8 campaignとH2 FP/A予約を含む継承台帳を保持し、既知27 originを読取確認した。
さらに明示metadata内の整数と通常taskの派生seedを保守的に加え、除外集合は78,174値となった。
この件数にはseed以外のmetadata整数も含み、全てが実際に使われたseedという意味ではない。

geometryは6距離×194 stochastic候補×32 replicaについて、未使用component slotを含む全予約領域を照合した。
各距離1,289,536、合計**7,737,216 slot**。元の整数encoder・streamを再利用し、6 metadata processでwall13.069秒。
Taylor order・component・RTE eventは生成していない。巨大な全seed一覧も複製しない。
衝突時のredrawは禁止。衝突0は固定した既知inventoryとの照合結果であり、確率的独立性の証明や未知の将来runの網羅ではない。
新一覧はprivateに保持し、公開はhash・件数・検査receiptとする。

## 最終固定call予算

次表は**将来のP科学実行1回の予算**であり、今回の科学実行件数は全て0である。

| category | 固定予算 |
| --- | ---: |
| 保存入力load / prefix1 native preparation | 1 / 1 |
| unconditional trajectory / native evolution | 92 / 92 |
| full Hadamard wrapper compile軸 | 184 |
| occurrence / RTE event | 128 / 256 |
| first-replica control probe / independent event oracle | 8 / 2 |
| endpoint統計会計pass / endpoint比較 | 1 / 2 |
| SCF / 積分 / DF / 状態solver / 実quantum shots | 各0 |
| full-H・truncated reference / primitive probe・spectrum | 各0 |
| 新signal / forward signal / B0I overlay / 新signed shot設計 | 各0 |
| PF compile / bootstrap / shot再計算 / 候補再選択 / 追加sampling | 各0 |

replica0の2候補のcontrolを受理した後だけ、残り90軌道を並列実行する。
全92軌道・両軸・固定seed・cohort・preparation・fingerprint・非返還台帳・native exit/reapを受入条件とする。
途中平均によるn変更、悪い結果の除外、未完了をPASSとする処理、retry・追加samplingはない。
不一致や異常は証拠を保存してSTOPする。成功時もmandatory STOPし、詳細科学レビューや最終evidence freezeへ移行しない。

## 人工テスト・拒否・保全

新人工38件＋既存H2回帰38件＋既存geometry整数回帰6件＝**82件PASS**。
人工DFのP q1/q4でnative context・同一prefix共用・元RTE・両軸wrapper・controlを検査した。
metadata fixtureでは固定n・PF/shot・SOURCE/input/runtime・seed・独立cohort・標本欠落/重複・再実行拒否・STOP保全を検査した。
fresh seed gateはreceiptのstatusだけでなく、除外集合の実bytes・既存除外の保持・新P seedとの非交差も要求する。
実H2配列・保存費用標本を人工テストへ読み込んでいない。
初回の混合テストではH4 geometryのimportがproviderと競合したため、最終回帰は別Python processに分離した。最終2 processは76件と6件で全PASS。

PのINVALID grantをdriver・worker・直接adapterの3入口で拒否した。
driver/workerは科学adapter importより前、adapterは科学package importより前に拒否し、attempt/output/markerを作成しない。
metadata sealは数値関数・科学module importを拒否するguard下で完了した。
n/SD/power再計算、real signal/sampling/compile・統計会計、有効grantは0。
これらは主担当Codex自身の技術検査であり、別reviewerによる独立科学レビューではない。

初回metadata準備`P_seal01`はF/FPの旧公開manifest形式を扱えず、Fの5 JSON snapshotを取り込んだ時点で停止した。
この時点ではseed生成・科学attempt・P科学callは0。SOURCE `ae8e0da9…`、technical STOP、途中ファイル・log10件を保持した。
旧SOURCEや証拠を上書きせず、互換adapterの人工テストを追加し、最終SOURCE `ce1f511c…`と新`P_seal02`へ固定した。
これはmetadata準備の技術修正であり、P科学実行のretryではない。

F/FP/D/A/S_STATSと旧STOP・台帳・予約を含む7,272固有レコード、および新seal63 recordsをhash照合した。
科学出力の存在を予約する空directoryも作成せず、`attempts/P_run01`と`P_activation01`は未作成である。
既存H4/H6/H8のSOURCE・契約・cohort・結果・grant・ledger・marker・STOP・job・worktreeを変更しない。
公開準備用artifactは[最終契約](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_seal02/)と
[人工テスト・guard・保全証拠](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_seal02_checks01/)を参照する。
有効grant・新seed一覧・F snapshot・元rawは公開しない。今回はlocal commitのみでpushしない。

## 実行準備状態・制約・次の担当

**技術seal完了、P科学実行は別途承認待ち。**
承認後は固定SOURCE・n・seed・候補・PF/shot・科学予算を維持したまま、fresh resource・seed inventoryを確認し、
新しいユーザー承認receipt・実行spec・有効grantを結合する。今回のseal承認をP科学実行承認に使い回さない。
worker数・承認status・fresh receipt等の実行時fieldのみを設定でき、科学条件の変更は拒否する。

metadata準備時はallowed CPU64、観測MemAvailable約503 GB、seed検査は自然な独立6単位を並列化した。
この観測値を将来の上限や保証に使わない。Pでは実空き資源・許可CPU・非干渉を再確認してtask数に応じたworker数を選ぶ。
worker内thread1で多重BLASを避け、OS・cgroup・scheduler・共有venv・他プロセスを変更しない。
一時的PSI・無関係なworktreeのHEAD/status・監視遅延だけを停止理由にせず、実OOM・own異常・入力破損・科学的不一致の保護を維持する。

S_STATSに用いた探索n8は両候補ともS0_ONLY8/8。希少eventの法則確率・未観測費用・未観測分散はunknownのままである。
固定n80/12は正規近似の計画値であり、実power・母平均優位・選択後coverage・厳密bias上界の保証ではない。
H4/H6/H8と会計・compiler・native control・独立確認の規則を共有するが、H2固有のrank3・sector4・候補・nである。
P最終sealまでという元の到達目標を達成し、P独立確認の実測結果はまだない。

| 担当 | 次の作業・判断 |
| --- | --- |
| ユーザー | 固定P SOURCE・seed registry hash・92軌道/184軸/256 eventと希少層制約を確認し、P科学実行を別途明示承認するか判断する。公開する場合はpush対象を指定する。 |
| Codex | P科学実行の明示承認後だけ、fresh resource・seed・入力/runtimeを再照合し、固定92軌道をone-shotで実行・終了監査してmandatory STOPする。 |

**現在はmandatory STOP。P sampling/compile・詳細科学レビュー・論文最終evidence freezeは実施しない。**
