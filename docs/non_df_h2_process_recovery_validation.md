# H2非DF pilot：Qiskit process分離と未完了2caseの取得

2026-10-23 JST。base `bd2f3ce9b2035d9caefac75785e6fe70a3dcbca2`。
専用branch `codex/non-df-h2-process-recovery-20261023`。GitHub対象branchとローカルHEADの一致を作業前に確認した。
旧run01・科学SOURCE `b5ed2c6e2399b55a0c24eb8437f842892a6703ef`・消費済みgrant・失敗STOPを永久保全する。
今回の利用者承認は、同じ保存入力を用いるPauli PR / DF PF各1回の新科学実行である。
Pauli PF / DF PRは受理済み保存結果をhash照合して再利用し、再実行しない。

## 原因調査と技術修正

旧tracebackのPauli PRはHLSのequivalence library `keys()`、DF PFはcontrolled RZZからBasisTranslatorへ至る。
Qiskit1.3.0の `circuit/equivalence_library.py` はprocess-global SessionEquivalenceLibraryを定義し、
`circuit/add_control.py` と `transpiler/preset_passmanagers/common.py` は同じsession objectをimportする。
実行時にもadd_controlの `sel is SessionEquivalenceLibrary` を確認した。

小さい人工4caseのthread試験は全PASSしたが、PauliEvolutionGateのtranspileとcontrolled RZZを競合させる
4 thread・各最大64反復の固定人工負荷で、`Already mutably borrowed` と `Already borrowed` を再現した。
worker0は3反復後HLS keys、worker2は初回BasisTranslator、worker3は15反復後controlled RZZのBasisTranslatorで失敗。
worker1は64反復完了。同じ人工負荷を独立4processにすると各64反復完了した。
科学入力を使わず同じborrow例外・経路を再現したことと静的共有object照合から、thread間の共有mutable session競合を
原因として支持する。Rust内部の個々のborrow区間は計装していない。Qiskitの共有installationは変更しない。

`isolated_qiskit_execution.py` はcaseごとに新Python interpreterをPopenし、ready/release barrier、
別出力、stdout/stderr、PID、終了コード、必ずwaitする回収を実装する。caseの再投入・pool再利用・fork済みQiskit状態の共有はない。
科学runnerの子processは、旧 `evaluate_case` / `validate_case` をbyte不変で使用する。
旧 `ThreadPoolExecutor` runner自体を修正・再実行しない。

## 人工preflightと安全な回帰

Pauli PF/PR・DF PF/PRに対応する人工4caseを実際に同時実行し、4個の異なるPIDと実処理時間の共通重なりを検査した。
同一入力の逐次process実行に対し、全標本のseed/RZ数とexplicit controlled Pauli/RZZ probe数が一致、
演算子・信号・normalization差は1e-12以内。各caseで両軸full Hadamard、transpile、compiled probability、
finite-RTE event phase、DF ordinary controlを確認する。人工PASSを実分子の検証完了とは扱わない。

安全な既存回帰223件、従来人工87件、最終の新人工・保存値・運用gate18件、計328 unique PASS。
最初の合同実行325 PASSと、新テスト追加後の18 PASS（うち15は重複）を区別して保存する。
分子生成を内部で伴う5件は明示除外、Python3.11.0rc1固定fixture1件は現3.12.3との差でskip。
除外nodeid・理由、JUnit、全ログは `technical/regression.json` を参照。SCF/積分/DF生成禁止のopt-in pytest guardを使用した。

## 新科学契約・保全

[execution contract](../artifacts/non_df_h2_process_recovery/2026-10-23/execution_contract.json) は旧契約の
system/DF/Pauli/evaluation/compiler/numerical/environmentをそのまま参照する。H2 1.00Å STO-3G、charge0/spin0、
canonical real RHFの保存2 spatial/4 spin orbital、active/frozenなし。DFはtol-only1e-8 rank3、Pauliは非identity14項。
保存npzを直接復元し、新SCF/FCI/DF分解/state eigensolveは0。旧raw23ファイル、旧SOURCE163ファイル、
入力snapshot/receipt、受理済みresult hashとSOURCEを照合する。

T=.8/q4/δ=.2/S2、Pauli PRはabs係数prefix L_D7/r2/K4、DF PFは全3fragment L_D3。
旧seed規則・Pauli PRの64 paired trajectories・全位相・回路構造を保持する。
各ε=.05/.01/.005/.001/.0005、両軸ε/√2、α各.025、経験的numerical allowance、P=0、full ordinary-control Hadamard RZ。
Qiskit1.3.0/Nature.7.2、rz/sx/x/cx・opt1・seed17・topologyなし。許容差を変更しない。
DFとPauliの分割単位は異なり、L_Dを演算子分割の同一性とは扱わない。

新clean commitをSOURCEにbindして新grantを作成し、別namespaceのrun02排他的mkdirでone-shot消費する。
各caseにも排他的worker consumption、SOURCE/input binding、marker、ログ、numerical evidence、statusを保存する。
親ledgerだけが子processの回収を追記する。失敗時は新runをFAILED_NO_RETRY_MANDATORY_STOPにし、自動retryしない。
共通指示§8.4の保存正本を読み、128 CPU affinity・約981GiB空きRAM・約113GiB空きdisk・低loadを確認した。
資源上限を追加せず、承認された独立2caseを並行実行する。共有venv/cache/他jobは操作しない。

## 成果物と適用範囲

[成果物](../artifacts/non_df_h2_process_recovery/2026-10-23/)、
[ライブラリ](../src/trotterlib/h2_non_df_process_recovery.py)、
[process基盤](../src/trotterlib/isolated_qiskit_execution.py)、
[保存値統合](../src/trotterlib/h2_non_df_process_integration.py)、
[runner](../scripts/run_h2_non_df_process_recovery.py)、[test](../tests/test_h2_non_df_process_recovery.py)。

C/N/Gは固定時間複素信号の条件付き評価。経験的allowanceをcertificateに格上げしない。
PRのC/Gは有限64標本平均で、paired covarianceを含めたSEと名目Student-t95%を示す。
未観測Taylor orderの母平均費用は0とせず未知として保存する。全RPE最終費用・一般的優位性・サイズ依存を結論しない。
詳細科学レビュー・新モデル・H4以降・独立確認・候補探索・pushは今回実施しない。

実行前SOURCE commit時点では科学取得未実行。結果は以下に追記し、旧報告・旧STOPは変更しない。
