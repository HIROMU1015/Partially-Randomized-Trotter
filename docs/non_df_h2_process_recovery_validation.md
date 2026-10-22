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

## 取得結果（2026-10-23 JST）

**新規2caseともCOMPLETE_FIXED_CASE・実施gate PASS。旧成功2caseと結合し、4方式×5精度の比較を完了した。**
科学SOURCE `7a6c4263d96153ecbe721dcfb6941d0d2684b3b0`、新grant `102d235ec47f37365370b7949e356302e6d1321b6f24f99b24f61d2fdd5726be`、
run02は `COMPLETE_MANDATORY_STOP`。親/子2processすべてexit0、全子回収、全体wall71.516秒。
Pauli PRの実処理68.416秒、DF PF6.784秒、実処理の共通重なり6.777秒。
`process.json` のelapsedは親が回収した時点までの時間で、個別処理時間は各statusのstarted/endedで区別する。
新SCF/FCI/DF分解/state solveはすべて0。成功2caseの科学再実行0。

旧科学kernel・旧契約の科学条件・数値閾値・compiler・乱数生成規則はbyte/hash照合で不変。
変更はcase間の実行状態のprocess分離と別契約/記録であり、旧run01の失敗とSTOPを消していない。
元Pauli PF/DF PRのresult SHAはそれぞれ `c09e0d5437a93065a406468d02459823ac3b775f23c14e1d7c5870aec5f21129`、
`ea68f7293b4d55a8eed926fa7476e5df9f28d924333c032ed2c43d1ed4d751df` のまま。

### 入力・参照と対象条件

入力snapshot SHA256 `fb5ca11d7f0be73abd063ae7b67c6b72399c5a2336635f3764af1a7316e81175`。
全caseの状態hash `4dc87df6c0466c1e5559ffcf3603b3a45ca7d1453b0657603188c120fe297b11`。
Pauli Hamiltonian hash `4eca732f1835e5d52ba2672277edbe6ee911114ef9b336d7e820c23ed3abc840`、
DF Hamiltonian hash `fe05aa6b88aeab032cf9232c59309695375a83d26f1ed59aea031aeaa77647fd`。
別表現のhashが違うことは演算子差を意味しない。同一MO積分に対する旧receiptのoperator差8.88e-16と参照信号差0を
hash付きで再利用した。今回のdense/sparse参照差も1.57e-16以下、許容1e-10をPASSした。
参照信号は約0.6364415911143844+0.771324899831317i。
以下はH2/1.00Å/STO-3G、DF tol-only1e-8/rank3、T=.8/q4/δ=.2単一点の値に限る。

### primary精度ε_sig=.05

Cはcosine/sine共通値、Nはcos+sin、GはRZ。

|方式|固定分割|C/軸|N_cos+N_sin|G|G SE|
|---|---|---:|---:|---:|---:|
|pauli_jw_PF|全14 Pauli項|3850.0|7044+7038|54215700.0|0|
|pauli_jw_PR|7/14 Pauli項・r2/K4|1005.96875|7365+7361|14813895.8125|228998|
|df_native_PF|全3 DF fragment|4047.0|7044+7038|56989854.0|0|
|df_native_PR|1/3 DF fragment＋one-body・r2/K4|2030.921875|7152+7146|29038120.96875|14776.9|

5精度の各軸C/N・exact G・SE・適格性は[20行の比較表](../artifacts/non_df_h2_process_recovery/2026-10-23/comparison.md)、
全signal/bias/normalization/hash/SOURCE/検証scopeは[comparison.json](../artifacts/non_df_h2_process_recovery/2026-10-23/comparison.json)。
全20行は固定経験的allowance下で条件付き適格。名目Nは量子shotの実測値ではない。
保存値だけの[補助図](../artifacts/non_df_h2_process_recovery/2026-10-23/figure/H2_four_fixed_G.png)もPNG/SVG/PDFで保存した。

|方式|corrected signal (real,imag)|signed bias (real,imag)|B|各軸allowance u|
|---|---|---|---:|---:|
|pauli_jw_PF|(0.636521723267535, 0.77125869578789)|(8.01321532e-05, -6.62040434e-05)|1|9.37144762e-08|
|pauli_jw_PR|(0.636487457878171, 0.771286988488513)|(4.58667638e-05, -3.79113428e-05)|1.02352940950889|6.11181861e-08|
|df_native_PF|(0.636521895502547, 0.771258553606606)|(8.03043882e-05, -6.63462247e-05)|1|4.71483474e-08|
|df_native_PR|(0.636521838048939, 0.771258601039023)|(8.02469346e-05, -6.62987923e-05)|1.00766277822851|3.78351217e-08|

### 実施した数値gateと限界

新Pauli PRはtail再構成差6.70e-17、独立法則平均差2.88e-15、raw/補正closure差2.94e-15以下。
強制event order0/2/4の最大演算子差1.24e-16、先頭3trajectory最大2.97e-15（phase alignmentなし）。
128 sampled compiled wrapperの最大確率差2.07e-14（許容1e-9）。
DF PFはfull-PF演算子差3.22e-15、ordinary controlled演算子差2.18e-15、compiled wrapper最大確率差4.60e-14。
新2caseの数値閾値はすべて実行前固定のままPASS。全4caseのcompiled最大誤差は4.60e-14以下。
20行のHoeffding会計・有理数G恒等式・paired covarianceによるG SEを保存値から再検査しPASSした。

Pauli PRは512 event位置でorder0を510件・order2を2件・order4を0件観測した。
order4確率3.5227e-7、期待0.0001804件、未観測確率約0.99982。その母平均費用寄与は未同定であり0ではない。
旧DF PRはorder2/4未観測をそのまま継承する。強制order回路検査のPASSはMC母平均費用の実測ではない。
Pauli PR CのSE15.5506、G(.05)の名目95%半幅457,615.559 RZ。旧DF PR CのSE1.03349、同G半幅29,529.240 RZ。
PRの名目95%区間は希少事象や母平均の厳密保証ではない。全軌道・compiled法則平均の独立認証は行っていない。

新しい科学モデル・最良候補探索・一般的PR/PF優位性・サイズスケーリング・全RPE総費用・詳細科学レビューを追加しない。
H4以降0、追加独立確認0、科学retry0、元成功caseの新科学実行0、共有環境変更0、push0。
ローカル成果物commit後mandatory STOP。

最終保全監査：旧tracked 6,909件（artifact 4,729件）、旧manifest 153 entryを不変で照合。新SOURCE 266ファイルとrun02 STOP内の全result hashも一致。元worktree clean、旧rawは元/新worktree両方で一致。最終監査は結果commitからも保存値だけで再実行できる（`verify_saved_evidence.py`）。初回監査とその実行sourceは `postprocess_audits/` に保全した。
