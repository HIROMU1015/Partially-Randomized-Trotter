# C1 Stage 3：8候補cost-only実行の技術的STOP

2026-10-25 JST。固定公開SOURCE `edb9f6ef99d00e656265900af8447223029fae41`。
Branch `codex/research-c-c1-stage3-cost-science-20261025`。

**Qiskitの並列compile中に`RuntimeError: Already mutably borrowed`が発生し、契約どおりSTOPした。106/最大936 logical units、新fit／optimizer／retry／resume0。8候補の総費用・12比較には未到達。部分証拠を保全し、mandatory STOP。**

[実行要約](../../artifacts/research_c_c1_stage3_cost_science/2026-10-25/execution_summary.json)／[原本・traceback・QPY・STOP](../../artifacts/research_c_c1_stage3_cost_science/2026-10-25/run01/)／[候補状態CSV](../../artifacts/research_c_c1_stage3_cost_science/2026-10-25/candidate_status.csv)／[34行の部分費用CSV](../../artifacts/research_c_c1_stage3_cost_science/2026-10-25/partial_round_costs.csv)／[call表](../../artifacts/research_c_c1_stage3_cost_science/2026-10-25/calls_summary.csv)。

## 固定条件と起きたこと

ユーザーが直前に提示した[936-call cost-only契約](research_c_c1_stage3_cost_contract_unapproved_20261025.json)の限定科学実行を明示許可した。独立worktree `/tmp/prt-c1-stage3-cost-science-20261025`、fresh output `artifacts/research_c_c1_stage3_cost_science/2026-10-25/run01`で、SOURCE・並列度・許容差を変更せず1回起動した。承認証明書・承認記録・新grantは作成していない。準備時点のunapproved契約名・metadataは履歴として保持する。

対象は保存P-H4（直鎖1.00 Å・STO-3G、Nα=Nβ=2）、CDF保存rank8 number・元g補正κ／保存XDF rank10。CA0.00159 Ha、DF reserve CA/8、δ0.1、m0..12、各DF ground-state oracle・状態準備除外。各armで決定論ordered S2 s1／s4とPR L_D2・s4/r16/K0／K2の8候補。P0 Qiskit1.3.0・opt1・seed17・rz/sx/x/cx・approximation_degree1.0・qubits_initially_zero=false、full-frame basis共有・角度結合、serial fence、保存楽観shotsを維持する。元20候補の結果と対象外12候補の状態は変更しない。

最初の失敗は`compile`、label `CDF_PR_s4_r16_K0_0`。Qiskit1.3の`HighLevelSynthesis._definitely_skip_node()`内の`self._equiv_lib.keys()`で例外が発生した。同時に走っていた`XDF_PR_s4_r16_K0_0`も同じ例外で終了した。これらは別候補の既に開始していたcallであり、retryではない。

```text
qiskit/transpiler/passes/synthesis/high_level_synthesis.py
  _definitely_skip_node(...)
    in self._equiv_lib.keys()
RuntimeError: Already mutably borrowed
```

並列compile時のequivalence-library参照で発生したことはtracebackで確認できる。共有objectの同時アクセスとの因果関係や、どの隔離・同期方法で解消するかは未再現・未検証であり、ここでは根本原因を断定しない。人工回帰13件は実行前にPASSしたが、この実H4並列経路での実行時例外を防げなかった。

これはcompilerの技術的STOPであり、PF/RTE・adapter・物理精度の数値FAILを示す結果ではない。SOURCE修正・並列条件変更・再compile・retry／resumeは行っていない。

## 到達範囲と未取得

| 候補 | 実行状態 | 保存round／axis行 | 総費用・比較 |
|---|---|---:|---|
| `CDF_D_s1_r0_K0` | STOPPED_DEPENDENCY | 21 | 未取得 |
| `CDF_D_s4_r0_K0` | STOPPED_DEPENDENCY | 0 | 未取得 |
| `CDF_PR_s4_r16_K0` | FAIL | 0 | 未取得 |
| `CDF_PR_s4_r16_K2` | STOPPED_DEPENDENCY | 0 | 未取得 |
| `XDF_D_s1_r0_K0` | STOPPED_DEPENDENCY | 13 | 未取得 |
| `XDF_D_s4_r0_K0` | STOPPED_DEPENDENCY | 0 | 未取得 |
| `XDF_PR_s4_r16_K0` | FAIL | 0 | 未取得 |
| `XDF_PR_s4_r16_K2` | STOPPED_DEPENDENCY | 0 | 未取得 |

CDF／XDFのD(s1)はStage2保存kernel費用の再利用とreadout shell費用の接続まで完了した。round／axis費用行は各21行／13行だけが保存され、いずれも全26行や候補合計へ到達していない。部分和を総費用として補完しない。

新segment build9、compile9 attemptのうち7 COMPLETE・2 FAIL。raw／compiled・unitarity照合とmetricsが完了した3件は共通readoutのhead／X／Yだけで、新しいD(s4)／PR kernelの同値性照合は未実行。compileが完了した一部kernelも、後続照合前に依存STOPしたため未資格として保全する。raw／compiled QPYは計16個。

今回の全候補総費用、条件付き総RZ、CDF/XDF費用比較、PRのshots負担を回路削減が補えるかという問いには答えが得られていない。候補の科学的NO-GO、CA達成、保証付きshotsや総資源優位を判定しない。費用support範囲と合成／実装law誤差も未取得・未資格のままであり、未知値を0へ置き換えない。

## call・資源・保全

| 項目 | 実績 |
|---|---:|
| SOURCE19・保存入力20点・元input・依存5版preflight | PASS |
| 人工回帰 | 13 PASS、9.66秒 |
| 科学call | 106／最大936 |
| call終端event | COMPLETE98／FAIL8 |
| compile | attempt9／COMPLETE7／FAIL2 |
| compile同値性照合・metrics | 各3 COMPLETE、readout shellsのみ |
| round／axis費用展開 | 34／最大208 |
| 候補合計・比較 | 0／8、0／12 |
| 最初のFAIL後の新規call | 0 |
| retry／resume／新fit／optimizer | すべて0 |
| PF/RTE・attenuation・shots式の再計算、実量子shots | すべて0 |

FAIL event8件には、2件のcompile例外と、6件のnative calibration wrapperの例外伝播・依存停止を含む。8回の独立した数値FAILを意味しない。候補ファイルの状態は例外FAIL2／依存STOP6。数値同値性FAILは観測していないが、未到達kernelをPASSとは扱わない。

契約SHA256は`d563c4dee6723dda5b2e965aa5fc4e9c0fa258fb4d59daf00c1a2366858d6477`、SOURCE manifestは`53ea2dba6cab1d807df1c64a97bd80b0355e3a25313bdc6747343e2e8334e81c`。元P-H4 input SHA256は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。保存shotsは公開commit `f85ac4c131b939bf4b8146a98997e61565345d45`の値をそのまま使用した。

実行開始時のCPU idle snapshot99.3%、affinity32 CPU、空きRAM約52 GiB、既存resource planの8 candidate worker・process内BLAS1 threadを使用。wall14.318秒、process CPU22.678秒、peak RSS553,164 KiB。人工的なwall／CPU／RAM capは加えていない。既存venv・共有設定・他ユーザーのjobに変更を加えていない。

SOURCE19と保存input20点の実行後byte照合もPASS。旧SOURCE・結果・grant・ledger・marker・STOPを保全し、run01のfailure／traceback／progress／call／raw・compiled QPY／resources／SOURCE／契約／runtime／STOPを保存した。停止後は保存JSON・journal・byteの整合確認とCSV整形だけを行い、追加Operator評価・compile・費用合計を実施していない。これは固定SOURCEでのlocal execution evidenceで、外部独立再現ではない。

## 残るblockerと次の作業

blockerはQiskit並列compileの実行時例外。次の担当はユーザー／研究Cで、Codexへ人工入力による同時compileの再現・task内の隔離／同期修正・回帰テストを依頼するか判断する。Codexはその技術依頼後に別SOURCEと新契約を準備する。現行run01はretry／resume0で消費済みのため再起動せず、新SOURCEでの科学実行には別途明示許可が必要。

今回は証拠をcommit・push・remote照合して**mandatory STOP**。追加H4評価・compile・総費用再解析へ自動進行しない。
