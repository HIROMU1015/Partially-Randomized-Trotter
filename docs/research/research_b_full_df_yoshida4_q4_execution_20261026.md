# 研究B Full DF-Yoshida4 q4：限定科学実行結果

2026-10-26 JST。**COMPLETE（固定2 caseの記述的pilot）／MANDATORY STOP**。
科学SOURCE：[620448b14235d2d9e0e76d5d0ba7273ce887a628](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/620448b14235d2d9e0e76d5d0ba7273ce887a628)。FD5／FD6はともに全correctness PASS、観測bias予算内。
Runtimeは`COMPLETED_DESCRIPTIVE_MANDATORY_STOP`、scientific_statusは`DESCRIPTIVE_ONLY`。科学attempt1、SOURCE変更・科学的中断・retry／resume0。

## 1. 固定条件・既存実装の再利用

Synthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation。同じphysical Hamiltonian／state、全4 block順one-body/DF0/DF1/DF2、元c0=0.17を一度だけ適用。分子geometry／AO basis／分子DF rank探索は非該当。
FD5=Full Yoshida4 q4 T+0.18、FD6=同T−0.18。Macro時間T/4、既存Yoshida4の3-piece係数・signed時間を変更しない。Full PFにはD/R splitting、内部n_D、RTE r/K・normalizationを導入しない。共通入力は旧LD2 snapshotへ結合する。
Ordinary control、mode0状態準備・readout・measurement込みX/Yの4 wrapper。Complex128、operator spectral2-norm atol1e-12／rtol0、phase fitなし。Native-v5、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、approx1、qubits_initially_zero=False、free topology。

既存`full_df.py`のq引数付きschedule／inventory／system／controlled／CAR数学参照／実行ループ／preflight／CLIを少数の引数で再利用。既定q2 whitelistとscopeは旧bindingに一致することを検査した。新`full_df_q4.py`は2 case whitelist・scope・観測bias分類・保存比較だけを担当する薄いadapterで、PF・回路builder・compilerを再実装しない。
`_full_df_native.py`／`_full_df_schedule.py`の旧AX-2A/AX-2B定義、`df_trotter.circuit`、B0 wrapper／native-v5／Evidence、Exact-R signal指標を変更せず使う。旧SOURCE・結果・契約・STOPを保全する。

## 2. 実行前に固定したscopeと検査順

入口は`scripts/run_b2d_full_df_y4_q4.py`。`--inspect`／`--preflight`はmetadataのみ、科学処理は`--execute`のみ。
Scheduleから導出した登録数：2 case、4 wrapper、各case96 signed block append、32 unique block probe、22 exponential key、44 operator gate、8 expectation、4 compile要求、4費用要求、**138 call**（events276、max308）。完全なidentity一覧は`--inspect`およびrun bindingに保存する。

| 境界 | 登録数 |
|---|---:|
| fixture/split/prepare/full_inputs/input_gate/full_prepare/definition_gate/saved_comparisons | 各1 |
| block_build/block_gate | 各32 |
| exponential | 22 |
| system_build/controlled_build/reference/system_gate/controlled_gate/signals | 各2 |
| wrapper_build/wrapper_reference/raw_wrapper_gate/raw_expectation/compile/compiled_gate/compiled_expectation/measure | 各4 |

固定fixtureの入力・sector/state・非可換性、全H/scalar/DF identity、全block signed/control、system/controlled、raw/compiled wrapperとexpectation、保存physical入力整合を**全件先行**させる。その後だけ2 signalと4費用を保存する。Global phase fitting／閾値緩和なし。Gaussian actual inverse、control0 identityとcontrol1相対phaseを維持する。

上記scopeは科学SOURCE commitで事前公開した。旧q2の224 call／72 operatorを流用せず、q4のscheduleとcache keyから導出した。実行実績は全138 callが予定＝試行＝完了、failed={}、events276。全gate／expectation／保存整合の最後のcompleteがevent位置263、最初のsignal解放が264で、後続4費用もその後だった。

## 3. 人工検証・preflight・環境

人工・回帰検証は**29件・33 subtest PASS**（9.04秒）。旧q2の6管理＋14独立toy、新q4の7管理＋2独立toy。Mockで固定whitelist、138 callの一致、SOURCE/snapshot/条件比較、全correctness先行、fresh output、失敗partial／STOP、retry／resume禁止を確認した。旧q2 scopeは旧公開bindingと一致。独立toyでq4の4 macro数学積、全block順・負stage、scalar-once、control0/1・相対phase、prepared X/Yを確認し、既存toyの全4 block・正負時間・actual inverseも回帰した。

```bash
PYTHONPATH=src /tmp/pr-phase2df1pre-private-20261023/venv/bin/python -B -m pytest -q \
  tests/management/test_b2d_full_df.py tests/management/test_b2d_full_df_q4.py \
  tests/test_b2d_full_df_toy.py tests/test_b2d_full_df_q4_toy.py
```

この技術段階で固定研究fixtureのoperator／signal／native build/compileは実行していない。人工PASSは科学性能の資格ではない。準備中に保存compilerのtuple／JSON list差を読み取り側で正規化し、B1から既存measurement_countを除く6費用指標を引用する接続を検証した。係数・物理条件・compilerの変更なし。

新SOURCEをcommit・push、remote SHA／adapter blobを照合し、clean detached worktreeを作成した。科学パッケージのimportを禁止したread-only preflightがPASS：固定SOURCE、fixture AST、旧22行・Exact-R・q2・B1の必要な公開blob、H／sector／state／array identity、同一compiler、fresh outputを確認。[Preflight記録](../../artifacts/research_b_outer_full_df_y4_q4/20261026/execution_logs/preflight.json)。補助ログ保存でPath型のJSON変換を一度修正したが、preflightの科学dispatchは0でSOURCE修正もない。

CPython3.11.9、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0。Native-v5条件は§1と旧q2／B1に一致。共有venv・compiler・他jobの設定を変更していない。
起動前32 CPU、MemAvailable55,101,644 kB、load約1.17。GPU driverは利用不可で、この小さい2 caseは既存CPU cache経路を採用した。22 exponential／32 probeの共有と既存compile cacheを利用し、最初の失敗で後続compileを起動しないcase順。不要なworker基盤・人工CPU／RAM／wall-time capは追加しない。

## 4. 一回の実行・原本・provenance

以下は**完了済みの実行commandで、再起動しない**。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-b-full-df-y4-q4-science-20261026/scripts/run_b2d_full_df_y4_q4.py \
  --execute \
  --source-commit 620448b14235d2d9e0e76d5d0ba7273ce887a628 \
  --output /tmp/prt-b-full-df-y4-q4-science-20261026/artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once
```

計時は`/usr/bin/time -v`、job専用`MPLCONFIGDIR`を/tmpへ指定した。Wall **10.29秒**、user15.90秒、system3.10秒、CPU184%、peak RSS476,784 KiB、exit0、stderr空。Compile要求4／actual transpile4。実行中の中断・修正・再実行は0。速度倍率は測定しない。

[原本7ファイル](../../artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/)は**486,126 bytes**。科学worktreeから公開branchへbyte一致で複製し、statusのscope一致・保存失敗0・STOPを確認した。[Result](../../artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/result.json)、[binding](../../artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/binding.json)、[inputs](../../artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/inputs.json)、[status](../../artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/status.json)、[STOP](../../artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/STOP.json)。Process4記録は[execution_logs](../../artifacts/research_b_outer_full_df_y4_q4/20261026/execution_logs/)。新しい承認証明用artifactはない。
入力data SHA256は`54a9233d59f670c7b8d38c186c9543925b6ca70bb09357c0176f1daf98fa819a`。H／sector／state IDはinputs、SOURCE／scope／依存version／資源はbinding、Gaussian basis hash・block順・c0・compiler・Python情報はresult.provenanceへ保存した。

## 5. 独立correctness

Full-Fock CAR／expm、native列に依存しないcanonical S2と既知Yoshida4の数学積、明示occupation／H／Sdg wrapperを参照とする。Gaussian actual inverse、全blockの負時間、control0 identityとcontrol1相対phase、元c0だけを一度付けることを維持。Phase fittingも許容差緩和も行っていない。

| 検査 | 完了数 | 最大誤差／結果 |
|---|---:|---|
| 入力H／state整合 | 1 closure | H 1.1102230246251565e-16、state 0 |
| Full4 block/scalar identity | 1 closure | 5.551115123125783e-17、c0=0.17 |
| Signed block・control有無 | 32 operator | 6.596549888815988e-16 |
| Full system／controlled | 4 operator | 5.334921997382396e-15 |
| Raw prepared X/Y wrapper | 4 operator | 2.6789522848988178e-15 |
| Compiled prepared X/Y wrapper | 4 operator | 4.4547920656664206e-14 |
| Raw expectation | 4 | 8.881784197001252e-16 |
| Compiled expectation | 4 | 3.297362383136715e-14 |
| 保存physical signal整合 | 1 closure | PASS、個別残差はclosure出力へ保存しない |

全44 operator／8 expectationはcomplex128・spectral2-norm atol1e-12／rtol0を満たした。Compiled operatorのcorrectness残差と次節の数学的PF biasは別の量である。

## 6. Signal・観測bias予算

主指標bは独立数学参照PFのsignalと`<mode0|exp(−iH_phys T)|mode0>`の直接複素絶対差。Compiled X/Yは別のcorrectness比較を通過しており、compiled値から新しいbやγを推定していない。

| Case | T | b | Full-Fock operator誤差 | N1 sector誤差 | 観測bias判定 |
|---|---:|---:|---:|---:|---|
| FD5 | 9/50 | 1.21606169604673e-10 | 3.159932657351753e-10 | 3.159932657351752e-10 | WITHIN_OBSERVED_BIAS_BUDGET |
| FD6 | -9/50 | 1.21606169604673e-10 | 3.159932657351753e-10 | 3.159932657351752e-10 | WITHIN_OBSERVED_BIAS_BUDGET |

| Case | X | Y | 半径 | X直接誤差 | Y直接誤差 | 半径直接誤差 |
|---|---:|---:|---:|---:|---:|---:|
| FD5 | 0.9704563130034695 | -0.22941820413712516 | 0.9972051784050219 | 6.256772877577532e-12 | 1.2144510397327224e-10 | 2.185085445916002e-11 |
| FD6 | 0.9704563130034695 | 0.22941820413712516 | 0.9972051784050219 | 6.256772877577532e-12 | 1.2144510397327224e-10 | 2.185085445916002e-11 |

ε=1e-9の参考点予算b≤5e-10について、**両caseともWITHIN_OBSERVED_BIAS_BUDGET**。Normalizationは1。観測bが予算内でも、b+u_numの包含・end-to-end精度・implementation資格は未認定。**u_num／γはnull**で、atolをその保証へ転用しない。未確定／非finiteの値はINCONCLUSIVEへ分類する実装だが、本runは有限の点観測として予算比較を完了した。

[SIGNALS.csv](../../artifacts/research_b_outer_full_df_y4_q4/20261026/SIGNALS.csv)は2行の直接列抽出で、新しい科学計算や統計再集計ではない。

## 7. Native full-wrapper単価

| Case | Axis | RZ count | CX count | RZ depth | CX depth | Total depth | Circuit size |
|---|---|---:|---:|---:|---:|---:|---:|
| FD5 | X | 2356 | 1636 | 1544 | 1636 | 3609 | 4996 |
| FD5 | Y | 2356 | 1636 | 1544 | 1636 | 3609 | 4996 |
| FD6 | X | 2356 | 1636 | 1544 | 1636 | 3609 | 4996 |
| FD6 | Y | 2356 | 1636 | 1544 | 1636 | 3609 | 4996 |

全4件でmode0準備・ordinary control・readout・measurement1を含む。同じnative-v5の決定論的単価で、shot加重費用ではない。[WRAPPERS.csv](../../artifacts/research_b_outer_full_df_y4_q4/20261026/WRAPPERS.csv)へ全metric・correctness・expectation誤差を直接抽出した。

## 8. 保存q2・高精度PR候補との限定比較

q2は[commit3244f2af9][Q2]、PR biasは[固定22行commit33715848b][R22]、PR費用は[B1 commitbef63edbd][B1]の保存セルを引用した。H／state／compilerの対応を確認し、旧科学計算・MC・統計集計を再実行しない。

| 方法 | 条件 | 保存／観測b（両T） | RZ／CX（各prepared axis） | 費用の意味 |
|---|---|---:|---|---|
| Full Yoshida4 q2、FD3/4 | 全4 block | 1.947702126906798e-9 | 1186／820 | 決定論的単価 |
| Full Yoshida4 q4、FD5/6 | 全4 block | 1.21606169604673e-10 | 2356／1636 | 今回の決定論的単価 |
| PR D1 | outer-Yoshida4/A3、q2,nD1,r2K2 | 4.526927441031103e-10 | 下表 | B1の有限S0標本推定 |
| PR R2 | 同r4K2 | 8.337749292598329e-11 | 下表 | 同上 |
| PR R4 | 同r2K4 | 7.003598416351649e-11 | 下表 | 同上 |

PRはLD2 weight-ranked、D=one-body+DF0/1、R=DF2 identity分離後、T±.18。Full PFにはこの分割を導入しない。PRのbはnormalization補正後physical複素signal誤差。上のq2は予算外、q4とPR3候補は観測bだけなら予算内である。

B1の以下は**各axis n=256、mean ± SEの保存セル**。XとYが同じ保存値であることを確認して各Tを一行に表示する。SEはrare tailの保証でも全候補同時信頼区間でもない。

| 候補 | T | RZ mean | RZ SE | CX mean | CX SE |
|---|---:|---:|---:|---:|---:|
| D1 | 9/50 | 27423.63281249999 | 1.65512926553982 | 18772.046875000004 | 1.2939304808042782 |
| D1 | -9/50 | 27423.23828125001 | 1.675764681519188 | 18771.531250000004 | 1.3164200912389705 |
| R2 | 9/50 | 27548.050781249996 | 2.3266371655368503 | 18868.609374999985 | 1.8070571484276778 |
| R2 | -9/50 | 27549.84765625 | 2.3831614754699606 | 18870.203124999993 | 1.8433332343024578 |
| R4 | 9/50 | 27424.371093749996 | 1.707662527135671 | 18772.656250000015 | 1.3492108600411123 |
| R4 | -9/50 | 27423.453124999993 | 1.4478127387232453 | 18771.62499999999 | 1.1578536176909597 |

[PRの保存72統計セル](../../artifacts/research_b_outer_full_df_y4_q4/20261026/SAVED_PR_B1_STATISTICS.csv)はRZ/CX count・depth・total depth・sizeの全両T／両axisをそのまま抽出した。B1の全標本がorder0だった事実を維持し、rare-event条件付き費用は未取得、無条件母平均は未認定。Full PFの2356／1636はこれらの保存S0平均より小さい点だが、保証付き同精度総資源順位やPR一般の劣位は認定しない。

文脈として既存Exact-R E1/2はb=5.926752262417758e-10、RZ/CX=4114/2800、E3/4はb=2.9044769538026594e-9、2100/1424。保存finite-RTE B3は6.756873465875987e-10、B4は2.905057732889878e-9。これらも原本セルの引用のみで、新しい探索ではない。

## 9. 確認できたこと・残る判断・STOP

固定synthetic2mode／N1で、既存Full DF-Yoshida4のq4はq2より小さいphysical signal biasを観測し、ε=1e-9のbだけの参考予算内に入った。同じfull prepared wrapper／native-v5で、既存高精度PR候補の有限S0推定より小さい決定論的回路単価を取得した。今後PRを評価する際に、この強い既知baselineを対照へ含める判断材料になる。

未確定なのはu_num／γ、PRの無条件期待費用、shot加重G、方式別の同精度最適化、他qでの収束・最適反復数、多電子・実分子・全RPEへの一般化、研究Bの独自性・正式継続方針。今回の小さい2-mode／1電子だけから総資源優位を結論しない。新q/PF、MC、rare費用、shot/G、A1、γ、分子、R2-CS再開、詳細科学レビューを開始していない。

結果branch：`codex/research-b-full-df-y4-q4-20261026`。科学SOURCEは上記620448b14235d2d9e0e76d5d0ba7273ce887a628、結果公開commitはbranch履歴で区別する。旧SOURCE／raw／grant／ledger／marker／STOPと固定契約、共通運用指示、元dirty workspace、他研究・共有環境を変更していない。新raw7・process4・直接抽出CSV3・本報告・必要な研究記録をcommit・pushし、remote SOURCE／成果物を照合して**mandatory STOP**。

次の担当は**ユーザー（研究B）**。q4を今後のbaselineに含めるか、追加保証・比較に進む必要があるか、既存成果で一区切りにするかを判断する。**Codex**は今回ここで停止し、新たな指示と必要な科学実行許可があるまで追加計算へ進まない。

[Q2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3244f2af9c4017c57c180b9057b22e3eb19c62b1/docs/research/research_b_full_deterministic_df_pf_execution_20261026.md
[R22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_rte_22row_execution_20261025.md
[B1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bef63edbd8cc564444b44e4e8cf23baa90f05fe5/docs/research/research_b_outer_b1_expected_native_cost_20261025.md
