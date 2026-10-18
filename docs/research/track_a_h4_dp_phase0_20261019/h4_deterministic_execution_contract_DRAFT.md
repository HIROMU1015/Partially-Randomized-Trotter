# H4 決定論的PF費用取得契約 DRAFT

状態：`DRAFT_UNAUTHORIZED`。技術準備だけが承認済み。D科学実行・再会計の承認ではない。
本書と機械可読D契約・SOURCE closure・入力SHAを一組として審査する。

## 科学条件と取得対象

linear H4、隣接1.00 Å、STO-3G、legacy固定DF rank12、8 system qubits＋ancilla、
保存Hamiltonian/state/reference、既存fragment順序/scalar、T=.8、P=0。
native symmetric-directional full Hadamard wrapper、Qiskit1.3.0、rz/sx/x/cx、opt1、
seed_transpiler17、backend/coupling/layout/routing指定なし。

主評価：通常signed bias、保存allowance倍率1、ε=.05/.01/.005/.001/.0005。
副評価：保存済み経験的非相殺、同じ5精度・倍率1。倍率10は今回は対象外。
両biasのshot・適格性を原本からコピーし、再計算・refit・誤差定義の変更をしない。

登録104候補を保持。D_tasks.jsonの55候補だけを新規compileし、取得済み16候補を再利用する。
残33候補は全10 endpointの保存済み明示不適格を理由として除外する。詳細はcoverage文書。
費用が悪いこと、proxy順位、2%モデル誤差を除外根拠にしない。

## 科学call予算と会計

|操作|固定範囲|
|---|---:|
|新native deterministic evolution|55|
|測定用full wrapper compile|110（各候補cosine/sine各1）|
|native control probe|最大220（候補ごとlow/high/X/Yの4）|
|control用・非測定wrapper生成|110（compileしない）|
|保存値の決定論費用会計|1 pass、104候補×10 endpoint|
|新signal recurrence/reference/SCF/DF/state生成|0|
|新PR trajectory/PR compile/PR順位再選択|0|
|fit/shot再計算/追加ε/追加candidate/retry|0|

保存snapshotから元builderの打切りpreparationをprefix4/6/9/11で再構成する。
これは新SCF/DF/state生成ではない。各worker内でprefix cacheを共有し、両軸で同じevolutionを使う。
spawn workerごとの初期化があるため、preparation構築上限は4×起動worker数。
科学的なlaw/回路構成を変えた高速化は含めない。

各候補は保存corrected signalとのnative control/scalar/X/Y照合（1e-9未満）後にcosine、sineを取得。
compile予約は呼出し前に専用ledgerへ書き、実測整数metrics・回路fingerprint・compiler hashを保存する。
一方の軸失敗時に欠測を0補完せず、そのscopeをTECHNICAL_STOPにする。自動retryしない。

全適格候補の両軸が揃った後だけ、`G=N_cos*C_cos+N_sin*C_sin`を整数算術で1回評価する。
各endpointの最小はG、candidate IDの辞書順で決定。同費用の全候補も保存する。
登録内のboundary属性はprefix、q、次数ごとの登録sliceとfull-rank属性を別記録する。
旧決定論16候補subsetの保存winnerと新winnerを対応付け、変化の方向によらず保存する。
本Dの会計はPR raw統計、PR新winner選択、n計算を含まない。

## SOURCE・入力・compiler

科学bindingはscientific_binding.json、保存input/結果のSHAはsource_inventory.json。
新実装はh4_dp_contract.py、h4_dp_science.py、run_h4_dp_verification.py。
旧provider SOURCE217c971d…、D原実行SOURCE79758d28…、compiler hash
1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b。
原profileは87b6b9f4…のenvironment_compiler_profile_v2.json、
SHA4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75。
Hamiltonian/state/snapshot/profileのSHAは機械可読bindingを正本とする。

private技術sealは新コードSOURCEのGit blobと実file SHA、実際に使うprovider/runtime/inputだけを固定する。
新しいユーザー承認receiptのSHA・scope・spec digestを持つone-shot grantが必要。
旧grantや旧one-shotを流用しない。DRAFT/INVALID grantは科学importや出力作成の前に拒否する。
実行時Python/全runtime closureと新コードをfreshに照合する。未確認をPASSにしない。

## 並列化・保全・停止

候補間をspawn processで並列化。worker数は直前の利用可能CPU・実効memoryと予想作業量に応じて設定し、
古いCPU数・memory/wall capを移送しない。内部BLAS等のthread数は各own workerで1にし、入れ子競合を避ける。
cache/tempは今回のexclusive output内。CPU affinity、scheduler、cgroup、quota、共有venv/cacheを変更しない。
旧SOURCE/結果/grant/ledger/marker/STOPや他のjobには書き込まない。
一時的PSI・CPU競合・無関係worktree更新をSTOP条件にしない。
実OOM、継続不能、own native異常、入力/SOURCE/runtime破損、非finite、科学的不一致はSTOP対象。
観測の欠測は未確認として記録し、単なる監視遅延で正常な科学workerを強制終了しない。

完了受入：固定55候補・110軸、call ledger、両軸fingerprint、control、全native worker回収、
driverの親によるnative exit0記録、final closure照合、結果SHA、全104候補のaccounting coverage。
33件を含む全行を保持し、「全候補直接実測」や登録外大域最良を認定しない。

終了監査→local commit→許可された公開範囲の手続き→公開する場合だけremote SHA/内容照合→mandatory STOP。
公開範囲未承認ならlocal保存までとし、公開を成功条件に偽装しない。Pは自動起動しない。

次の担当：ユーザーがDの55候補/110 compile・最大220 control・限定会計1 passを別途明示承認する。
