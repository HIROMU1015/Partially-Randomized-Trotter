# Track A H4：登録内の決定論的PF対照完成

2026-10-19 JST。`COMPLETE_MANDATORY_STOP`。主担当Codexによる実行・終了監査。
AI subagentは使用していない。外部reviewや独立科学再現とは呼ばない。

## 実行条件・承認

linear H4、隣接1.00 Å、STO-3G、legacy固定DF rank12、8 system qubits＋ancilla、
保存Hamiltonian/state/reference、元fragment順序とscalar、T=.8、P=0。
native symmetric-directional full Hadamard wrapper、Qiskit1.3.0、rz/sx/x/cx、opt1、
seed_transpiler17、backend/coupling/layout/routing指定なし。
主評価は通常signed、副評価は保存済み経験的非相殺、allowance倍率1、
ε=.05/.01/.005/.001/.0005。bias・適格性・shotを保存値から変更していない。

科学code SOURCE：`8a82b9272ee24896e7d7cbb7400d9b8c12d8f804`。
固定providerは`217c971dd537b1cc7fbc13742bc0f6c263ff7f45`、compiler hashは
`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
実行中も前後もSOURCE closure 7 file・input 10 file・provider 236 file・runtime 34,273 fileを照合した。
無関係なworktreeのHEAD/statusは実行gateにしていない。

ユーザーは修正版Dを承認し、「中断された場合は修正して再計算して」と指示した。
これを同じD科学範囲内の技術修正・再計算の承認としてprivate receiptに記録した。
旧契約のworkflow全体での自動技術retry禁止は、この新しい利用者指示で更新された。
各attemptのone-shot、SOURCE/spec/grant/output/lockの分離と旧記録の保全は維持する。
成功したDの再試行、P、S_STATS、raw統計、追加候補や科学条件変更はこの承認に含めない。

旧run01のpreparation guardによるTECHNICAL_STOPはそのまま保全した。
今回のrun02は新しいgrant/output/lockで実行し、追加の中断・技術retryは0回だった。
[停止と修正の履歴](track_a_h4_d_binding_fix_20261019.md)に原因と人工39件PASSを記録している。

## 並列実行と終了受入

直前観測は64物理core中約61.08 core相当idle、実効空きmemory約469.74 GiB。
有限55候補に対して55 spawn workerを使い、BLAS/OMP/Numba等を各own workerの1 threadにした。
実観測でも55 workerのthread合計55を確認した。旧55を恒常的上限にはしない。
prefix preparation cache、cos/sin間のevolution共有、fingerprint再利用、保存16候補の再利用を維持した。
spawn時の不要manifest複製も削除済み。OS/cgroup/affinity/scheduler/共有venv/他jobは変更していない。
PSIや一時的CPU競合だけでworkerを中断していない。

native driver wallは536.0692663248628秒（約8分56秒）、親witnessがexit0を回収した。
runnerは全55 workerのexit0を検査してから結果を保存した。監査時には既知worker PIDは全て不在。
直列比較benchmarkはないため、速度向上倍率は未認定である。

|項目|完了実績|
|---|---:|
|新規native deterministic evolution|55|
|測定full wrapper compile|110（cosine55＋sine55）|
|native control probe|220、全55候補PASS|
|control最大誤差|1.0351018364643335e-13（閾値1e-9未満）|
|決定論費用会計|1 pass、104候補×10 endpoint＝1040行|
|再利用した実測候補|16|
|全10 endpointの保存不適格で除外|33|
|新signal/SCF/DF/state/reference/fit/shot再計算/PR sampling・compile|0|

call ledgerと両軸の整数metrics/fingerprint/compiler hash、control、全task回収、
final execution closure、native exitとresult SHAを受け入れた。
結果SHA：`9b719010119e2a6a0370bd582c6ab6ab5f3b7c05bb64041ecf40dfb8340a8a8c`。
終了監査はmetadataを照合して結果を転記しただけで、G・shot・winnerの二回目の会計は行っていない。

## 保存会計の最小費用候補

`G=N_cos*C_cos+N_sin*C_sin`。保存shotと直接実測RZを使った整数値である。
同費用の代表はcandidate ID辞書順とし、tiesも結果に保存した。

|bias|ε|代表候補|G exact|
|---|---:|---|---:|
|通常signed|.05|B0I_p4_q1_2nd|86045388|
|通常signed|.01|B0_p9_q2_2nd|6523645968|
|通常signed|.005|B0_p9_q4_2nd|42602298697|
|通常signed|.001|B0I_p9_q2_4th|1835103650289|
|通常signed|.0005|B0I_p9_q4_4th|11241349628381|
|経験的非相殺|.05|B0I_p9_q1_2nd|124439127|
|経験的非相殺|.01|B0I_p9_q2_2nd|6523790959|
|経験的非相殺|.005|B0I_p9_q4_2nd|42603784630|
|経験的非相殺|.001|B0I_p9_q2_4th|1835103650289|
|経験的非相殺|.0005|B0I_p9_q4_4th|11241349628381|

旧16候補subsetの代表winnerから10/10条件で不変だった。
経験的非相殺ε=.05ではB0_p9_q1_2ndも同費用。代表と同率集合を区別する。
prefixは全代表が登録sliceの内部、ε=.05のq=1は物理的下端、他はq sliceの内部。
代表の次数選択肢とfull-rank属性もmachine resultに残した。

登録はtruncated96＋full8の104候補。55新規＋16保存の71候補に費用があり、
33候補は全10条件の保存経験的適格性が不適格なのでcompileしなかった。
欠測は0補完していない。全104候補を直接compileしたとも、登録外の大域最良とも主張しない。
保存経験的bias/非相殺は厳密なbias証明ではない。
新PR順位・PR母平均優位・energy/QPE/RPE/FT資源について結論を追加していない。

## Pに引き渡した事項と残る承認

事前規則どおり、通常signedの各εで上表のPF代表・保存shot・exact Gを固定した。
事前登録9候補・14比較（主10＋診断4）に同じεのPFを共通参照させた。
候補の再選択、新seed生成、旧PR標本の再集計や探索meanと新PFの数値的符号比較はしていない。
nは全9候補でnull、比較方向もnullのままである。

次にユーザーがS_STATSを別途承認する場合、Codexが転記済みmean/SE/nと完成D費用だけを使い、
事前設計の最大84 scenarioと候補別n、14固定pairの基準方向を1 passで数値設計する。
raw trajectoryの再集計/bootstrap・実H4 sampling/compileはその承認に含めない。
その後、具体的n/seed/SOURCE/予算を固定してPを別途承認する。現在は両方未実行でmandatory STOP。

## 成果物と保全

- [終了監査](../../artifacts/resource_applicability/track_a_h4_d_completed/2026-10-19/completion_audit.json)
- [1040行の決定論会計と10条件のwinner](../../artifacts/resource_applicability/track_a_h4_d_completed/2026-10-19/deterministic_result.json)
- [新規55候補の両軸実測](../../artifacts/resource_applicability/track_a_h4_d_completed/2026-10-19/new_native_measurements.json)
- [P：D後のPF固定・n未定契約](../../artifacts/resource_applicability/track_a_h4_d_completed/2026-10-19/P_after_D_contract_DRAFT.json)
- [native exit](../../artifacts/resource_applicability/track_a_h4_d_completed/2026-10-19/native_exit_projection.json)
- [並列実行の観測](../../artifacts/resource_applicability/track_a_h4_d_completed/2026-10-19/parallel_execution_observations.json)

終了監査scriptは`scripts/audit_h4_d_completed.py`で、科学providerをimportせずmetadataだけを扱う。
旧57記録とそのcopyのSHA一致を確認し、新runのSOURCE・grant・receipt・ledger・結果等297記録も
private SHA storeへ保全した。private絶対path・PID・grant/receipt原本・seed一覧は公開projectionに含めない。
local commitまでを実施し、remote pushは行っていない。

次の担当：ユーザーがS_STATSの実施を判断する。承認後にCodexがnと基準方向を設計し、
具体的なP契約を準備する。Pの科学実行は別承認。
