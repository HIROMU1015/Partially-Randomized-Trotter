# 研究B：B0 Native Pilotのcompiled correctness停止

2026-10-25 JST。**B0は最初のcompiled wrapper operator検査でFAILし、FAILED_MANDATORY_STOP。限定pilotは未完了、費用計測0件。** 原因は未確定。固定SOURCEを一回実行し、retry／resume・許容差変更・追加条件による再実行は行っていない。A0の既存結果は保全した。

## 1. 固定SOURCE・契約・原本

| 項目 | 固定値／保存先 |
|---|---|
| 科学SOURCE | `8d0cb96df9d1f6229f207550634c7e11aff2851c`、変更なし |
| 実行契約 | [3c1e10f1cのB0仕様・scope・command](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3c1e10f1c061b9b4d15e2690403fd26b841cbc06/docs/research/research_b_outer_a0_b0_execution_preparation_20261025.md) |
| 保存入力identity | 結果公開commit `33715848bfb83d49ec4e54d6eef481012a7441e4` の[22行snapshot](../../artifacts/research_b_outer_rte22/20261025/analytic-once/)5ファイルのblobと一致 |
| Raw7ファイル | [native-pilot-once/](../../artifacts/research_b_outer_b0/20261025/native-pilot-once/)：binding／inputs／graph／events／partial_results／status／STOP |
| 境界別実績 | [CALL_COUNTS.csv](../../artifacts/research_b_outer_b0/20261025/CALL_COUNTS.csv)：登録27境界の予定・attempted・completed・failed・未試行数 |
| Process記録 | [execution_logs/attempt-1/](../../artifacts/research_b_outer_b0/20261025/execution_logs/attempt-1/)：preflight／stdout／stderr／time |

公開branchは `codex/research-b-outer-b0-results-20261025`。A0結果公開commit `31662b49157b15f2d1c03eef1c2faa07e2f905e7` を起点とする独立結果worktreeに保存した。科学checkoutは固定8d0cb96dfのままで、7 raw＋4 process記録をbyte一致でコピーした。旧36／16／22行・A0のSOURCE、原本、契約、grant／ledger／marker／STOPを変更していない。CSVは管理call数の列抽出であり、科学値の再評価ではない。

## 2. 固定条件と実行前確認

既存synthetic2-mode one-body＋3 DF因子、N=1／mode0、L_D=2 weight-ranked prefix、n_D=1、T=±0.18。分子geometry／AO basis／分子DF rank探索は非該当。
B3=Yoshida4/A2/r2K2、D1=Yoshida4/A3/r2K2、R2=Yoshida4/A3/r4K2、R4=Yoshida4/A3/r2K4、B4=SABA2/q2/A2/r2K2、F1=refined-S2/q2/f8/A2/r2K2。Yoshida4のmacro q=2。計12 case、4種類の決定論的代表で48 trajectory、状態準備を含むX/Y wrapper96要求を予定した。IID標本や期待費用MCではない。

Correctnessはcomplex128、system4×4／controlled・wrapper8×8のoperator 2-norm、absolute atol1e-12、rtol0、phase fitなし。Qiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、approximation_degree1、qubits_initially_zero=False、backend／target／coupling map・layout／routing指定なし、既存official-definition phase-preserving loweringを固定した。

実行前のread-only preflightはPASSした。HEAD／SOURCE、clean checkout、保存入力5ファイル、candidate／scope、依存version／origin、compiler profile、fresh output非存在を確認し、preflightで科学moduleをloadしていない。その後、契約の通常commandを一度だけ起動した。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-b0-science-20261025/scripts/run_b2d_outer_b0.py \
  --execute \
  --source-commit 8d0cb96df9d1f6229f207550634c7e11aff2851c \
  --output /tmp/prt-research-b-outer-b0-science-20261025/artifacts/research_b_outer_b0/20261025/native-pilot-once
```

このoutputは使用済み。上書き・再起動・resumeしない。

## 3. 停止箇所と実行実績

停止identityは **`B3:9/50:0:cosine`**：B3、T=+0.18、代表0（全short-step order0、最初のcomponent）、X wrapper。Compile要求は正常に返り、続く `b0.compiled_gate` で `Stage1Failure("B0_CORRECTNESS_FAILURE")` が発生した。最終statusとSTOPに同じ理由が保存されている。

| 登録境界 | 予定 | 完了 | FAIL | 未試行 |
|---|---:|---:|---:|---:|
| Fixture／split／DF prepare／full inputs／input gate | 各1 | 各1 | 0 | 0 |
| RTE config | 76 | 76 | 0 | 0 |
| Event descriptor／独立event operator | 各1740 | 各1740 | 0 | 0 |
| Single short-step mean | 24 logical、event項1692 | 24 | 0 | 0 |
| Full-space analytic mean | 12 | 1 | 0 | 11 |
| System／controlled build | 各48 | 各1 | 0 | 各47 |
| System／controlled operator gate | 各48 | 各1 | 0 | 各47 |
| X/Y wrapper build／independent wrapper reference | 各96 | 各1 | 0 | 各95 |
| Raw wrapper operator gate | 96 | 1 | 0 | 95 |
| Raw X/Y expectation | 96 | 1 | 0 | 95 |
| Wrapper compile要求 | 96 | 1 | 0 | 95 |
| Compiled wrapper operator gate | 96 | 0 | 1 | 95 |
| Compiled X/Y expectation | 96 | 0 | 0 | 96 |
| 費用measurement | 96 | **0** | 0 | 96 |

共有native D blockは18/372、RTE sequenceは12/492、独立exponentialは14/206、finite momentsは2/16まで完了した。Full operator gateは予定288のうち3 PASS、1 FAIL、284未試行。Expectation診断は予定192のうちraw1件だけ完了し、191未試行だった。Full-space meanの1件はB3、T=+0.18で、5対象signalの保存sector値との照合を含む。全24 logical single-step meanを通過したことを全48 trajectory・全12 caseのcorrectnessへ拡張しない。

Event記録は7,286（start3,643／complete3,642／failed1）で上限11,414内。全callは登録identity内で、scope超過ではない。Costは全correctness後にしか計測しない構造のため0件であり、compiled expectationにも到達していない。Compile要求1完了を96要求の完了・期待compile費用と呼ばない。最終cache統計は未保存である。

## 4. 保存された根拠と欠落した診断

記録から確定できるのは、入力・24 single-step mean・最初のfull mean・system／controlled／raw wrapperとraw expectationが正常に戻り、最初のcompiled operator検査がFAILしたことである。Compiler APIのimport例外やtranspile例外ではなく、compileが返った後の数値correctness guardで停止した。

現行 `Native.check` は `norm(a-b,ord=2)` が非有限または1e-12超の場合に同じfailure codeを投げ、数値差を例外へ添付しない。`ScopeLedger`の既定event summaryも成功値を数値保存しない。さらにrowのappendはcompiled expectationの後にあるため、今回 `partial_results.json.rows=[]` であり、`result.json` は存在しない。

そのため、FAIL時の誤差量、raw／compiled circuit、比較operator、gateごとの数値誤差、最初のsignal値、mean誤差、component／trajectoryの最終provenanceは今回のartifactから復元できない。入力定義・snapshot identity・metadata graph・予定代表列はSOURCE／binding／graphに残るが、実数値診断の代わりにはならない。失敗記録・call実績・STOPは保存され、`evidence_write_failures=[]`。これはEvidenceの書込失敗とは別の、現行bodyが数値を返す前に停止した際の診断不足である。

**原因は未確定**。Phase差、compile後の累積数値誤差、gate定義変換、reference／mappingの不整合、非有限値などを識別する数値根拠は保存されていない。既存compilerの静的確認ではprofile一致、official-definition lowering、global phaseを含むcache identity、layout検査を確認したが、実compiled circuitの正当性を補完する証拠にはしない。既存F2の異なる条件・許容差でのPASSも今回へ転用しない。

追加のoperator評価・回路build／compile・phase fit・許容差緩和で原因を補完せず、未解決のcompiled correctness問題として停止した。科学的手法の一般的な失敗、compiler bug、単なる丸め誤差のいずれかに都合よく分類していない。

## 5. 費用と資格

RZ／CX count、filtered depth、total depth、sizeの測定は0件で、候補間のnative費用比較表は得られていない。作成した `CALL_COUNTS.csv` は実行管理の実績であり回路費用表ではない。代表trajectoryの順位、expected cost、shot加重G、ε=1e-8／1e-9の保証付き総資源比較は評価していない。

Gamma／u_num／shotsは今回資格を取得しておらず、0で補完しない。A0のEMPIRICAL_CONSISTENCYとu_num=nullは[既存結果](research_b_outer_a0_execution_20261025.md)のままで、B0のcompiled gateを保証するものではない。全RPE、別fixture／sector、別PF・係数・grid、A1包含、B1 MCへ進んでいない。

## 6. 実行環境・保全確認

既存venv CPython3.11.9、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Qiskit Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0。SOURCE・依存version・module origin・snapshot blobを記録した。Install・共有設定の変更はない。

実行直前は32 CPU、空きRAM約52.7 GiB、/tmp空き約330 GiB。GPU driverは通信不能だった。固定2-mode／full-FockのCPU経路、共有block・RTE・exponential・momentと実QC compile cacheを使用し、CPU／RAM／worker／wall-timeに人工的な低い上限は設定していない。Whole-process wall **12.41秒**、user CPU18.12秒、system CPU3.00秒、最大RSS396,888 KiB、exit2。

StderrはMatplotlibが書込不可の既定cacheから一時cacheへ移したwarningのみ。停止は別途statusに記録されたcompiled correctness FAILであり、このwarningを原因としない。修正・retry／resumeは0回。準備段階の人工138件PASSを今回の科学的PASSと呼ばず、SOURCE変更がないため同suiteを再実行していない。

保存後にraw7・process4のcopy一致、status／partialのcall一致、イベント数と唯一のfailed event、費用・compiled expectationの未起動、scope内実績、SOURCE・旧証拠の不変を確認した。科学処理の再実行ではない。

## 7. 最終状態・次の判断

最終状態は **FAILED_MANDATORY_STOP／B0_CORRECTNESS_FAILURE、B0 pilot未完了**。科学SOURCEも過去の結果も変更していない。全gateが揃わず費用計測へ進まなかったことが今回の実績である。完了済み境界のcall記録を数値全件保存や費用比較の成功に読み替えない。

Codexは失敗原本と報告を公開し、remote SOURCE／成果物を照合してmandatory STOPする。次の担当は研究Bのユーザーで、最初の失敗caseに限定した新しい診断attemptを許可するか判断する。

提案する次作業は、Codexが将来のSOURCEにFAIL時の数値差とraw／compiled circuit保存を最小限追加し、mockで保存を確認したうえで、**別途許可された場合だけ**B3／T=+0.18／代表0／Xの同条件をfresh outputで診断することである。旧attemptの上書き・resume、許容差変更、原因未確定の全48 trajectory再実行は提案に含めない。今回はその実装・診断実行も開始しない。
