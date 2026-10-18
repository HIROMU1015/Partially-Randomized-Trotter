# Track A H4 P：SOURCE・seed一覧・実行契約の技術seal

状態：技術準備を承認受領。人工新12件＋既存39件PASS。SOURCE固定後、整数seedの最終照合を行う。
P科学実行は未承認。sampling/compile/新科学統計は0、grantはINVALIDのままにする。

## 固定する科学範囲

H4 linear/隣接1.00 Å/STO-3G/legacy固定DF12/8 system qubits＋ancilla、T=.8、P=0。
保存Hamiltonian/state/reference/fragment順序/scalarと元finite-RTE law、二次partial PFを維持する。
native symmetric-directional、Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling指定なし。

D結果commitは4e8ad1936363266d1b55bb689c36846e6c3dfcdb、
S_STATS結果commitは5e18773db3ad18c591b3c0fdc83cc1b5f1a29cfc。
新しいn計算・候補再選択・探索meanの再集計を行わず、S_STATSのf4/η.02推奨と14方向をそのまま固定する。

|候補（B2・2nd共通）|固定n|役割|
|---|---:|---|
|p2/q4/r4/K2|901|主|
|p2/q8/r4/K2|115|主|
|p2/q8/r4/K4|115|診断|
|p3/q1/r2/K2|277|主|
|p3/q1/r4/K2|2119|主|
|p3/q2/r4/K2|459|主|
|p3/q4/r4/K4|145|主|
|p3/q8/r2/K2|26|主|
|p3/q8/r2/K4|28|診断|

新無条件trajectory/native evolution各4185、full wrapper compile8370、occurrence9770、
RTE event37662、control最大36、独立event oracle最大9、endpoint会計1 pass/主10＋診断4。
新PF compile、SCF/DF/state/reference/signal/shot再算出/fitは0。
同一候補の複数εには同じn軌道を共用し、cos/sinにも同じtrajectoryを使う。
途中平均に応じたn変更や悪い軌道の除外を行わない。
14方向は保存探索PR平均がPFより小さい方向で、母平均優位の保証ではない。

## 技術修正とSOURCE

旧P templateのimport先には元Aの`pr_cost_native.py`本体が含まれていなかった。
元A SOURCE31ef62a12407010e9c1a07f9b37facedb8dff5feの契約source_hashesに従い、
15 fileをGit blob SHAどおりの専用private snapshotへ追加した。
既存stage/providerを変更せず、lazy moduleのimportだけを実際に確認した。NumPy/Qiskit/SciPyのimportはない。
元Aのload_context/steps/controlを呼び替えたり、lawやscalarの科学実装を変更したりしない。

今回のP SOURCEには、次の技術修正を含めて固定する。

- S_STATSのn・14方向・PF/shot・科学call予算を実行前gateで照合する。
- seed plan digestと、旧geometry/明示seed集合との照合receiptをP taskに束縛する。
- trajectory保存schemaにmaster_seedとstep_seedsを明記する。

元のSOURCE候補4e8ad193…から、これらの修正を含む新SOURCEへ固定し直す。
旧D/S_STATSのSOURCE・結果・grant・ledger・STOPは変更しない。

## seed inventoryと新一覧

新namespaceはH4_PR_INDEPENDENT_CONFIRMATION_V1、masterは
SHA256([namespace,SOURCE,seed_plan_digest,candidate_id,replica])先頭64bit。
planは科学条件・n・shot・PF・方向・D/S_STATS結果SHAを含み、新seed一覧を含めない。
q=1はmasterを直接使い、q>1は元direct_master_to_step_seeds_v1のq個の派生step seedを使う。
新4185 master＋step seedは11559 slot。候補間CRNなし、cos/sinだけをpairedにする。
新一覧・台帳はprivateに保存し、公開はSHA・件数・衝突件数だけにする。

旧系列は明示除外台帳、旧固定n32/記録済み探索、A探索128軌道、条件付きB32、
T=.4/1.6各32、DF政策FP32、geometry六距離を出典付きで照合する。
geometryは登録1308候補のうち1164 stochastic候補×32＝37248 masterを含む。
固定dataset identity/SOURCEと整数seed規則を用い、各距離の保存wrapper seed欄をanchorとして照合した。
全候補のidentityは公開保存projectionと固定planを照合する。

geometryの派生seedは、orderと最大Kまでの全component draw slotを保守的に除外する。
実際には使われないslotも含むsupersetであり、RTE event・Taylor order・componentを再samplingしない。
そのstreamは新11559 seedとのmembership照合とSHAだけを計算し、大きな全seed一覧は複製しない。
六距離という独立単位を計算processで並列化する。AI subagentは使用しない。
39908の明示旧seed値（geometry masterを含む）とgeometry派生streamの双方を照合する。
衝突0は数学的独立性の証明ではなく、固定した旧inventoryに対するmetadata照合結果である。
未知の将来runのseedを網羅したとは主張しない。

## 受入・承認境界

新12件はgeometryのexact encoderとの一致、未使用slotを含む除外、衝突検出、
条件付きstream、固定n/方向/予算とplanの変更拒否、mock P kernelのseed/paired/cohort schemaを検査した。
既存39件にもPASSし、うち2件は人工DFを使う。実H4 preparation/trajectory/control/compileは0。
作者による確認であり、独立reviewとは呼ばない。

技術seal後もstatusはSEALED_TECHNICAL_PENDING_USER_APPROVAL、grantはINVALID。
runner/witnessがこの状態を科学import・output/lock作成前に拒否することを確認する。
旧377保全記録とcopyを照合し、今回の入力・SOURCE・seed一覧・照合receiptも保全する。
無関係なworktreeのHEAD/status/資料更新はgateにしない。

Pの実行承認後だけfreshなCPU/RAM/runtime/actual inputを再照合し、実効資源に応じたworker数を選ぶ。
旧worker55、memory/wall上限は移送しない。own worker内は1 thread、prefix2/3の準備cache、
cos/sinのevolution共有、fingerprint再利用を維持する。準備再構成上限は各workerで2 prefix。
cache/tempはP専用output内。OS・cgroup・affinity・共有venv・他jobへ変更しない。
一時的PSIやCPU競合だけで科学workerを中断せず、実OOM/own異常/入力破損/科学的不一致ではSTOPする。
各候補replica0のcontrolを通してから残りを取得する。one-shot・retry=false・次stage不許可。
全固定標本・両軸・call ledger・native exit0・全回収・final closure受入後に保存監査しmandatory STOP。

次の担当：ユーザーが固定n・新SOURCE・private seed manifest SHAと衝突receipt・予算を確認して、
Pの科学実行を別途判断する。Codexは承認前にsampling/compileを開始しない。
