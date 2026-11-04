# 研究C：保存native pair・7-call batch certificate pilot

契約日2026-11-04、実行日2026-11-04〜05 JST。ユーザーの明示許可に基づく固定7-call限定実行。
Arb192・full512の同じ数学的native pairを新しいbatch経路で包囲し、保存された旧包囲と照合する。
新しいword抽出・回路build/compile・literal reference再計算・期待bias校正・Stage Cは対象外。

## 固定SOURCEと範囲

- 科学SOURCE：`ce3185513578a8329efe364c748b5ce2aacdda45`、tree `ba132cca7ea72019583baa5ebbe1be656fc08a4f`。
- [固定7-call契約](research_c_fixed_pair_batch_pilot_contract_unapproved_20261104.json)。歴史的unapprovedファイル名・metadataは保持した。
- [技術準備](research_c_stage_p_totality_calibration_feasibility_20261104.md)の172 SOURCEファイル・74保存inputを照合。新SOURCEや科学契約を作り直していない。
- Linear H4・1.00 Å/STO-3G、number CDF8/LD2、S2/s1/r32/K0、h=δ=.1、controlled512。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
- 固定wordは `CDF_proposed:F6:0` と `CDF_proposed:F7:6,7` を16回交互に並べた32-event列。同じ保存compiled/raw/actual inverse recordだけを用いた。
- P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、MPFR160 declared phase、既存Gaussian/relative fusion、own-DF oracle・状態準備費用除外、IID53-bitモデル、CA/RPE政策を変更しない。
- 比較1：保存compiled対保存rawの数学的gate作用。比較2：保存compiled actual inverse対同じcompiled recordの数学的adjoint。
- 新評価器はArb192、batch16、全512残差基底列・全出力、丸め包囲、両global phaseを保持する。共通unitary因子削減後の列は元の同名計算基底列とは区別する。

Fresh出力：`artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot01`。
実行entryは無変更の `scripts/prepare_research_c_totality_calibration.py --execute-fixed-pair-pilot`。
独立branch：`codex/research-c-fixed-pair-batch-pilot-execution-20261104`。

## 起動・資源

PreflightでSOURCE/input hash、元word/identity、依存version、fresh出力先を確認した。CPU affinity32、available RAM約56GB、load約1.65、ディスク空き約41GBだった。
既存resource planを使い、32個の独立batchを32 processへ割り当てる。自job内のBLAS/OMP等は各1threadで過剰並列を防ぎ、CPU/RAM/worker/wall capは追加しない。実行中の読み取り観測でも32 workerがCPU計算を行い、自job全体のRSS合計は約1.5GiBだった（process間の共有pageは重複計上し得る）。共有環境・他jobを変更していない。

最初のdetached operational launcherは実行環境の短いcommand lifetime内で子processを保持できず、科学runner・科学出力・科学journalが作成される前に終了した。
この0-call観測を保存し、既存supervisorを保持されたforeground exec sessionから起動した。科学SOURCE・契約・数値runnerは変更していない。
科学runnerを開始した実行は1回だけであり、科学retry/resumeは0。外側wrapperは既存supervisorを再利用する運用記録で、数値評価経路を変更しない。timeout・restartはない。

## 結果

以下は固定result・call journal・STOPの取得結果。

7/7 logical calls COMPLETE、FAIL/retry/resume 0。Supervisor returncode 0、全wall 1175.262814秒（約19.6分）。

| 数学的native pair | 新operator上界 | 新operator下界 | 新wall秒 | 保存旧単発wall秒 | 新/旧wall |
|---|---:|---:|---:|---:|---:|
| compiled_raw | 1.585733058571e-11 | 1.059987230566e-12 | 1173.086006 | 1000.435076 | 1.172576 |
| compiled_inverse | 5.542118759907e-15 | 2.449293598295e-16 | 1.734899 | 1667.074093 | 0.001041 |

両pairとも全512列・全出力、Arb192、phase fitなし、rounding enclosed。Exact rationalと列別半径は保存JSONを正本とし、表は表示近似である。新旧包囲は同じtargetで非矛盾だった。

Compiled→rawは共通prefix/suffixが0で、17.26%長いwallを観測した。したがってこの比較ではbatch経路の高速化を確認できない。
Inverseでは共通prefix 11,386 gatesを削減でき、残差の全512列評価は小さいworkのため1 workerを使用した。両global phaseは保持した。保存旧wallとの比では約960.9倍短いが、これはこの保存pairの構造による効果であり、未知wordへの速度保証ではない。

旧2比較の保存wall合計 2667.509169秒に対し、新2比較の合計は 1174.820905秒だった。旧実行は異なる時点の各1回の観測であり、今回freshな対照実行・反復benchmarkを追加していない。負荷・thread構成等を統制した統計的speedupとは主張しない。

| call種別 | COMPLETE |
|---|---:|
| evidence_metadata | 1 |
| records_load | 1 |
| identity_gate | 1 |
| accelerated_certificate | 2 |
| compatibility_report | 1 |
| summary | 1 |

準備commitで人工40件・関連108件PASS。今回そのSOURCEを維持し、消費済みfresh出力を要求する旧preflightテストや数値評価を再実行していない。実行後はSOURCE/input hash、全列coverage、call/STOP、保存JSON構造と旧成果不変更を独立に確認した。

[result](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot01/result.json)／[compiled_raw](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot01/compiled_raw.json)／[compiled_inverse](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot01/compiled_inverse.json)／[journal](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot01/science_calls.jsonl)／[STOP](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot01/STOP.json)／[preflight](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot-preflight.json)／[照合](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot-verification.json)／[終了状態](../../artifacts/research_c_stage_p_totality_calibration/2026-11-04/fixed-pair-pilot-launch01/exit_status.json)。


## 資格と次の判断

新旧区間の非矛盾は整合検査であり、区間の重なり自体が正しさの証明ではない。元record/phaseのbinding、共通unitary因子によるnorm不変性、全基底評価、Arb丸め包囲が根拠である。
上界のtightnessやwallは変わり得る。同一固定pairの旧1回実績との比較から、未知word・全sampling lawでの速度や平均誤差を認定しない。
本pilotでは非unitary literal Dを新評価器へ渡していない。保存済みC→D、raw signal、head/referenceの旧上界を今回取得し直したとは扱わない。

Compiler全word成功、全word synthesis、期待bias・tail・分散、IID-grid RPE十分条件、actual CAは依然未資格。
期待bias契約の方法/N/δ/abort/cache/非停止政策・call上限は未採択のまま。RPE測定α=.01の配分も変更しない。
次の担当はユーザー／研究C。本pilotの費用と資格を受けて、failure-aware scopeと校正法を選定する。Codexは別途固定・許可された科学実行以外へ進まない。

## 保全・公開

旧SOURCE/input、旧41-call部分STOP・独立15-call完了・Stage R・Head bridge・B2・Stage Aの契約/result/STOPを保持する。
本実行のSOURCE/契約コピー、exact rationalの列別包囲、call journal、STOP、preflightとsupervisor終了記録を保存する。
公開commitは結果の保存用であり、科学SOURCE commitを付け替えない。公開後にremote SHA/tree・報告本文・主要blobを照合し、mandatory STOPする。
