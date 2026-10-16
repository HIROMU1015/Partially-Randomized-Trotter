# V1：B0+I p7/q1 S2の2軸実compiled検証結果

**登録したcosine/sineの両軸が14,492 RZ、予測14,524.2 RZとの差は各0.222191554%でPASS。実行は正常終了し、mandatory STOPした。**
実測を用いた総費用再会計・PR費用削減率の更新は実施していない。次の数値再解析は別途の明示承認を待つ。

利用者の「進めてください」を、公開済み[実行契約](track_a_v1_p7_compile_readiness_20261016.md)の最大2軸・一回限りcompileへの承認として記録し、fresh gate通過後に一度だけ起動した。追加準備・再fit・retry・候補探索は行っていない。実行SOURCEは`84b9ea22bf78c7193f2137ce42a04400292b8e68`、providerは`87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`。

[実測・終了記録・監査](../../artifacts/resource_applicability/track_a_v1_p7_compile_run01/2026-10-16/README.md)。旧[Phase A signal](track_a_v1_phase_a_run02_results_20261016.md)、[Phase B検証](track_a_v1_phase_b_run01_results_20261016.md)、[モデル費用による再会計](track_a_v1_reaccount_results_20261016.md)は保存したまま。

## 対象・維持した条件

linear H6、間隔1.00Å、STO-3G、保存tol-only DF rank19、12 modes、sector400、保存Hamiltonian/state、T=.8、δ=.8。候補は`B0I_p7_q1_2nd`のcosine/sineだけ。通常signed signal biasと経験的allowanceの有限時間complex signal benchmarkを維持する。

archiveの`faithful:p19`先頭8 blockからp7 S2 evolutionを一度構築し、saved full-Fock tail identity C=`0.01946666126068529`とfull preparation constantをancilla relative phaseへ一度だけ適用した。symmetric-directional control、measurement込みfull-wrapperを使用。保存full-H reference・signal・allowance・shotは置換も再計算もしていない。

compilerはQiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、topologyなし。hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
固定費用model SHAは`406c6f36f93dac317f9709e68b042c4563edf5ec4b06219e1b60de660b695211`。cosine/sine独立stratumの係数a=797、b=269、d=9613/5を変更していない。

## 各軸の独立判定

|軸|固定モデル予測RZ|実測RZ|相対誤差|判定|compile秒|
|---|---:|---:|---:|---|---:|
|cosine|14,524.2|14,492|0.222191554%|PASS|11.553464|
|sine|14,524.2|14,492|0.222191554%|PASS|10.420169|

相対誤差は各軸`abs(72621/5 - measured)/measured = 161/72460`。事前固定した`>=1/50`のSTOP条件を有理数で各軸別に適用し、cosine PASS後にsineへ進んだ。平均での判定は行っていない。実測CXは各6,544。入力wrapper命令数はcosine5,524、sine5,525。

compiler/input fingerprintは各軸で一致。最大4回のnative control probesは保存signalと回路の同一性検査だけに用い、最大discrepancyは`3.2803286093439427e-15`で、固定technical guard内だった。元signal評価pipeline・reference solver・PR samplingは実行していない。

## 終了・資源・保全

wallは73.241100秒、native exit0、observerのfirst_stop=null、cleanup_errors=[]、全回収、driver残存なし。最大monitor gapは1.003944秒、監視された総RSS peakは745,910,272 bytes（711.355469 MiB）。driver1＋observer1、各thread1、driver CPU11・observer CPU16をfresh idle検査で割り当てた。

予約はsnapshot1、archive decode1、native evolution1、wrapper2、control probe4、compile2。登録外compile・retry・refit・新signal/reference/scalar/SCF/DF/state/primitive/sampling・費用再会計・V2は0。旧draftは未承認のまま保全し、今回の承認は新しいgrantにのみ記録した。one-shot lockと取得済み予約は保全し、再利用しない。

起動前と終了後にSOURCE21/provider230/runtime34,273ファイル、入力15receiptをbytes/SHAで照合。H5/H7を含む旧grant/ledger/carry/marker等660ファイルは全て不変。元V1結果branch・他worktree・V2・H5/H7へ変更を加えていない。

既存の90人工testsは準備時の検証結果として保持し、今回は変更のない科学実装について再実行を追加していない。終了後の検証は保存receipt・予算・軸判定・cleanup・bytes/SHAの照合であり、primary agent自身によるローカル監査である。

## 科学的な到達範囲とSTOP

この候補の両軸費用はモデル外挿から実測へ進んだ。ただし全gridの2%誤差保証、厳密bias certificate、旧binary bit同値、PR母平均費用保証は認定しない。旧B0I12候補の非相殺bias分解欠測も未解決のまま。

ε=.05の総費用・費用削減率はまだ更新していない。保存shotと実測を接続する比較は[既存の計画](../../artifacts/resource_applicability/track_a_v1_p7_readiness/2026-10-16/future_accounting_plan.json)に従い、**別途承認後**に行う。新しい勝敗・削減率は今回主張しない。

**2軸検証完了・mandatory STOP。次Phaseの自動起動はしない。**
