# H6 full m5_best：保存回路の通常・拡張精度診断の実行準備

2026-10-19。**実装・人工検証・固定契約の準備が完了し、mandatory STOP。実H6の保存値解析は未承認・未実行。** 新compileも行っていない。

SOURCE：`99e8723d402245a9ad923e7dbc0800f47bc575b6`。独立branch：`track-a-h6-m5-saved-phase-readiness-20261019`。

固定private実行plan SHA256：`b2b2a818dd318a9a981cacae3eeb59617501992e9d6e60ff45d1c151dc288254`。公開planはprivate pathを除いた投影であり、このprivate実行planとは別のfileである。

[直前の計画](track_a_h6_m5_saved_phase_diagnostic_plan_20261019.md)を実装した。[再計算STOPの結果](track_a_h6_m5_full_recovery_run01_results_20261019.md)と旧STOP・成功58軸・未検証費用を変更していない。主比較への追加PF採用AD03、後続Core、基準変更は保留を維持する。

## 1. 今回完成したもの

|file|役割|
|---|---|
|[h6_m5_phase_contract.py](../../scripts/h6_m5_phase_contract.py)|stdlib認可、固定4 job、旧STOP/receipt/source/input/runtime照合、新grant・ledger・marker|
|[h6_m5_phase_engine.py](../../scripts/h6_m5_phase_engine.py)|通常精度の元Qiskit手順、独立した拡張精度gate作用と診断指標|
|[h6_m5_phase_adapter.py](../../scripts/h6_m5_phase_adapter.py)|保存NPZ/QPY検査、保存値再現、worker、quarantine保存、4結果の比較|
|[run_h6_m5_saved_phase.py](../../scripts/run_h6_m5_saved_phase.py)|fresh resource gate、spawn scheduler、STOP/回収、外部wait4|
|[prepare_h6_m5_saved_phase.py](../../scripts/prepare_h6_m5_saved_phase.py)|metadata/byte SHAだけで封印する未承認plan・無効grant作成器|
|[test_h6_m5_saved_phase.py](../../tests/test_h6_m5_saved_phase.py)|人工24テスト。実H6配列の復元なし|

新しいscientific SOURCEとして上記5 scriptと既存固定plan/input manifestの計7 fileをGit objectへ結合した。原runの科学SOURCE20 file・さらにその元の16 fileは変更していない。

## 2. 固定された一回実行案

対象はH6直鎖・隣接1 Å/STO-3G、保存tol-only1e-8 DF19、full PF/prefix19/q2、T=.8、P=0、`PF_M5_BEST_20D64C2_BINARY64_V1`、symmetric-directional/full Hadamard **cosineのみ**。原run SOURCEは`f743a88225f7cb2c8d2c91d7e19813a5780f803b`、結果commitは`282cba7d6047ea6613a9e69c818e467cb962dd65`。

将来の承認単位は`H6_M5_SAVED_PHASE_DIAGNOSTIC_V1`の一回だけ。内部手順ごとに後続科学Phaseを起動する契約ではない。

1. 新SOURCE・12入力role・runtime・旧STOP/exit/reap・新未使用出力・fresh resourceを照合する。
2. 保存NPZのdtype/shape/finite/initial low重複一致を確認し、保存診断を再現する。
3. 保存2 QPYの13 qubit・ancilla12・layout None・数値parameter・標準gate class・最後の測定を確認する。
4. 同じ保存lowに対して、wrapper/compiledそれぞれ通常精度と拡張精度の4再生を行う。
5. 通常再現を通った場合だけ比較報告をまとめ、結果SHA・全worker exit/reap・launcherの外部wait4を保存してSTOPする。

|job|回路|作用の精度|
|---|---|---|
|WRAPPER_ORDINARY|保存wrapper|元のQiskit Statevector、global phaseを元の規則で一回適用|
|COMPILED_ORDINARY|保存compiled|同上|
|WRAPPER_EXTENDED|同じ保存wrapper|longdouble/clongdoubleの解析式|
|COMPILED_EXTENDED|同じ保存compiled|同上|

保存QPYは2つの**異なるfile**に限定。preflightと各spawn workerは同じfileを再loadするため、deserialize呼出数を2と主張していない。4再生とも初期状態は同じ保存lowのみ。新しい状態・probe、SCF/DF/full-H reference、回路合成、transpile、sampling、signal探索、shot/費用再会計、fit、ranking、sine dispatchは0に固定する。

## 3. 精度・再現条件・解釈

通常精度は原SOURCEの`Statevector(low).evolve`と、保存global phaseを別に一回掛ける規則を再利用する。保存measurement以外を削除せず、layoutやgateを自動修復しない。

拡張精度はH/P/RZ/RZZ/CX/XXPlusYY/SX/Xの解析式を使い、little-endianのbit-indexでstatevectorを更新する。保存binary64 parameterを正確にlongdoubleへ移し、三角関数も拡張精度で評価する。元の係数・角度の桁を推測したり、最適化し直したりしない。binary64の`to_matrix`へfallbackせず、回路全体のdense unitaryも作らない。

実mantissaがbinary64より広く、1付近で実際に追加精度が得られることを起動前に確認する。同じ精度ならSTOP。未知gate、customな同名gate、symbolic parameter、中間measurement、予想外のqubit/classical配置・layoutでもSTOPする。

**通常再現の絶対allowanceは1e-12を実行契約案に固定した。採択・実行は別承認待ち。** 保存metricの再現と、通常2再生が保存before/afterへ一致することに適用する。原runのfinite/raw差**1e-9未満**という受入基準は変更しない。

raw差、norm、overlap、common phase、phase-aligned残差、枝別差・位相、Hadamard、通常/拡張間の差を出す。phase fittingによる補正、再正規化、referenceの位相合わせは行わない。phase-aligned値は診断だけであり、旧FAILをPASSに置換しない。

通常再現に失敗したら新規dispatchを止める。既に走るown jobは未受理証拠として回収し、有利な拡張精度結果だけを使わない。通常再現が通っても、拡張精度で差が減ることは精度依存の示唆に留まる。丸めが唯一の原因、compiler bug、全状態unitary等価性、旧費用の受理を自動認定しない。[判断表](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_readiness/2026-10-19/decision_table.csv)を維持する。

## 4. 並列化・資源・停止

4再生は独立なので、fresh affinity・idle CPU・実効空きRAM・diskから1〜4 workerを選ぶ。4は4 jobしかないための数で、過去runの固定上限を引き継いだものではない。軽いpreflightは順に行い、大きなQPY objectをparentへ保持し続けない。

- statevector更新をベクトル化し、bit-indexと同一のgate/parameterをcacheする。
- worker内部は元通常環境と同じ1 thread、QISKIT_PARALLEL=FALSEをown process内だけに設定して戻す。
- RAM admission proxyは前回同一点の観測peakの2倍＝6,052,372,480 bytes/worker。disk proxyは保存入力byte総数の16倍＝2,071,284,480 bytes。どちらも計画見積りであり、実行中の資源capや必要量の保証ではない。
- affinityと`MemAvailable`に加え、既存cgroupのmemory.max/currentによる実効空きを読む。制御・権限は変更しない。
- host/上位cgroupの一時PSIやCPU競合は観測のみ。正常workerをその理由でkillしない。無関係worktreeのHEAD/statusはgateに含めない。
- 実際の資源不足で次jobを投入できない、own worker異常、nonfinite、入力/SOURCE/runtime不一致、再現失敗、出力衝突ならSTOP。自動retry・追加jobはない。
- own SIGINT/SIGTERMはcooperativeに伝え、全own workerを回収する。外部supervisorはown launcherだけをwait4し、native exitとreap・RSSを記録する。

新しいruntime capや一律wall timeoutを設定していない。現時点で実H6の再生時間を計測したとは扱わない。

## 5. 検証と封印

[人工検証記録](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_readiness/2026-10-19/artificial_validation.json)：固定Python3.12/Qiskit1.3.0/NumPy1.26.4で**24 test method PASS、2.046秒**。

- 全対応gate・1/2 qubit配置・逆順配置・inverse・global phase一回適用をQiskitの人工作用と照合。
- 拡張精度の実mantissa、binary64 matrix fallbackなし、phase-only/relative phase/norm差の区別、unknown/非finite/非binary64 parameter拒否。
- 人工QPY/NPZ往復・保存metric改変/low不一致の拒否・出力のexclusive保存。
- 人工13 qubitで、**実装したadapter・4 spawn worker・比較report**まで通過。人工的にraw FAILする回路でも、診断の完了と旧FAIL保持を分離。
- worker異常・通常非再現で後続を止め、全workerをreap。marker/ledger/一回実行の再使用拒否。
- 外部wait4で人工childのnative exit7とreapを保存。compile等の呼出経路が無いこともASTで検査。

最初の人工inverse検査1件は、参照式側のNumPy scalar promotionがcomplex128へ戻っていたため失敗した。参照lowを明示的にclongdoubleへ移し、最終24件でPASSした。実engineの受入基準や実H6保存値を都合よく変更したものではない。

[metadata改変検査12種](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_readiness/2026-10-19/metadata_negative_checks.json)も全て拒否。scopeのcompile/状態追加・基準緩和・plan/SOURCE結合・closure/input/runtime・旧grant scope・replay追加・出力alias・resource案変更を検査した。人工grantはテスト用のメモリ上だけで、有効な実grantは発行していない。

[DRAFT拒否](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_readiness/2026-10-19/draft_refusals.json)：`--execute`と`--supervise`の両方で、実入力を読む前にexit2・`EXPLICIT_SAVED_NUMERICAL_ANALYSIS_APPROVAL_REQUIRED`。run root、supervisor marker/log/exitは未作成。

12入力roleとSOURCE/runtime **1,831 file、383,526,197 bytes**を封印。Qiskit/NumPyの実装・native libraryはinstalled RECORDと照合し、importされたruntime/libraryと原closureを結合した。[公開closure metadata](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_readiness/2026-10-19/source_runtime_closure_metadata.json)。実path・固定private config・実QPY/NPZはprivateに保持する。

これらは作者自身の技術確認であり、別レビュアーによる独立レビューではない。AI subagentは0。

## 6. 成果物と保全

公開bundle：[入口](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_readiness/2026-10-19/summary.json)。plan、入力metadata、closure、無効grant、DAG、判断表、人工/metadata/構造検証、保全、inventoryを含む。公開plan/grantはprivate pathを除いた投影なので、そのまま実行するconfigではない。

専用private準備領域は`h6-m5-saved-phase-diagnostic/20261019/private/readiness01/`。無効config/grant、exact private plan、人工log、metadata監査tool、byte保全receiptを置いた。ここにも有効実grant・run outputは存在しない。

[保全receipt](../../artifacts/resource_applicability/track_a_h6_m5_saved_phase_readiness/2026-10-19/preservation.json)：既存684 fileは全て不変。旧SOURCE/input/raw/grant/ledger/marker/STOP、旧58受理軸、他worktreeと共有環境は変更していない。過去の結果や証拠entryを上書きせず、索引は新しい準備状態を追加した。

今回の実H6 QPY復元、NPZ数値配列restore、statevector作用、科学数値再解析、compile、sampling、有効grant発行は全て**0**。local commit後mandatory STOP、pushは行わない。

## 7. 次の担当と承認境界

**ユーザー**：SOURCE `99e8723d…`・private plan `b2b2a818…`と、通常再現allowance1e-12案を確認し、保存値診断＋固定4再生を一回実行するか判断する。今回はその実行承認を受けたとは扱わない。

**Codex（別承認後のみ）**：fresh gateを再確認し、別の承認receiptと有効grantを新規に作成する。元の無効grant/configを上書きせず、同じplanへ結合する。固定環境で`run_h6_m5_saved_phase.py --supervise --config <新承認config>`を一回だけ起動する。終了後はnative exit/全worker回収/結果SHA/保全を監査し、結果を報告してmandatory STOPする。

この解析を承認しても、新compile、sine、基準変更、費用再会計、ランキング、主比較へのm5採用、後続Coreを承認したことにはならない。
