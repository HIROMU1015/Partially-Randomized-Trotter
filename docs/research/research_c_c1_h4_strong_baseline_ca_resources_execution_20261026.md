# H4強baseline：共通CA・RPE shots・条件付き総資源の86-call実行

2026-10-26 JST。**固定SOURCE・修正契約v2を変更せず一度実行し、86/86 COMPLETE、correctness FAIL／inflight／retry／resumeは0。B2 CDF8／B3 XDF8とも共通CAの数値十分条件を満たし、条件付き総RZはB3 XDFがB2 CDFより1.7529746%少なかった。** これはown-DF ground oracle・ideal IID Born・状態準備除外・固定serialモデルとlevel B誤差証拠に条件付けた数値pilotである。

- 科学SOURCE：`1aeedd96178a3f50a15bc2701c3c332f4f391b64`、tree `529404ba5716c177c0a004d0027e382d684f5b28`。
- 数値実装SOURCE：`969c17d623d56eb645417b281deadb1be4757699`。今回の実行中にSOURCE／runner／条件を変更していない。
- Branch：`codex/research-c-c1-h4-strong-ca-execution-20261026`、独立worktree：`/tmp/prt-c1-h4-strong-ca-execution-20261026`。
- [固定契約v2](research_c_c1_h4_strong_baseline_ca_resources_contract_binding_v2_20261026.json)、SHA256 `1bf82b39693f07503eec2879c93cb17a734ccdfdcdca03b50544a15444a56cd2`。
- [SOURCE manifest](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/technical/SOURCE.json)、SHA256 `881694bde50cfa1ad50a67fef86c0bffc42c165559a6f8be56e148573e7cad8c`。
- [raw13件・SOURCE／契約／journal／STOP](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/run01/)／[機械可読要約](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/execution_summary.json)。
- [全26 round](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/rounds.csv)／[52軸shots・serial費用](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/axes.csv)／[候補別結果](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/candidates.csv)／[6指標比較](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/comparison.csv)。

## 固定条件・再利用・実行順序

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36。通常number CDF rank8の保存`B2_CDF_relative`と、保存OpenFermion1.6.1 eigh元順prefix8 XDFの`B3_XDF_spin_relative`を比較した。元g補正κ、signed λ/G、核scalar一回、frame0〜7の順、元Hamiltonian／DF近似は不変。全frame ordered symmetric S2、**s=1のみ**、δ=.1、m0..12、n=2^m、PRは非適用。

共通CA=.00159 Ha、DF reserveCA/8=.00019875 Ha、β_total=π/6、β_stat=π/12、決定論phase room=π/12。failureは各候補.01、各round/axis .01/26。各DF自身のground-state oracle、ideal IID Born、状態準備費用除外。Gaussian `nature_jw_exact_zero_lapack_atan2_v1`／P0 `qiskit_P0_mpfr160_declared_phase_v1`、Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=falseと既存layout・all-wire fences・shell分離政策を保持した。

[準備報告](research_c_c1_h4_strong_baseline_ca_resources_preparation_20261026.md)の既存接続を使用した。保存B2／B3のfull512 Frobenius・raw/compiled/reference別norm guard、共通head/X/Y shell、元の候補別PF13、DF norm＋guard、own-ground／aliasの保存identityを再利用した。既存`propagate` → `assess_saved_round`／`assess_round`／Hoeffding／`assess_fixed_round` → `assess_energy` → `serial_round_cost`／`compose_serial_cost` → `aggregate_candidate`／`aggregate_totals`／`compare_totals`をそのまま呼んだ。

保存B2の科学SOURCEは`880006396326a4d46f4a4030564279e92fadb9cf`、B3は`b40fce5c62c9fff97f68073cbc02182e4bbf2b9b`。元公平nativeと旧CAのSOURCE・結果も来歴として保持した。旧shots／総資源を新routeの結果として流用せず、今回許可された26 roundを新しく伝播・評価した。回路・Operator・PF・DF・groundの再生成は0。

前回、同一bytesの保存契約を指すdocs／artifact pathの文字列差を科学call前に検出した。[修正報告](research_c_c1_h4_strong_ca_binding_repair_20261026.md)の契約v2への明示許可後に、今回のfresh run01を初めて開始した。旧契約の歴史的draft metadataは変更していない。承認証明書・承認原文保存・新grantは作成していない。

## Gateと合成誤差・CA

| gate | B2 CDF8 | B3 XDF8 |
|---|---|---|
| SOURCE／保存入力／環境／fresh output | PASS | PASS |
| 保存full512 coverage・correctness・norm guard／共通shell | PASS | PASS |
| DF予算／own-ground／alias／元PF identity | PASS（保存証拠再利用） | PASS（保存証拠再利用） |
| 13 round合成＋PF位相条件 | 13/13数値PASS | 13/13数値PASS |
| RPE／整数shots／fixed-N margin | 13/13 PASS | 13/13 PASS |
| 共通CA数値十分条件 | `NUMERICAL_PILOT_CA_PASS` | `NUMERICAL_PILOT_CA_PASS` |
| serial費用・集計 | AVAILABLE | AVAILABLE |

| 数値pilot量 | B2 CDF8 | B3 XDF8 |
|---|---:|---:|
| 保存DF norm＋guard［Ha］ | 7.666215916301914e-5 | 9.533901455091894e-5 |
| 保存compiled/reference Frobenius | 3.118100926800043e-11 | 4.738879866495084e-11 |
| 保存kernel norm上限pilot | 1.000000000004071 | 1.0000000000039608 |
| 最大合成complex signal誤差（m12） | 3.61239425568958e-7 | 5.490105208994203e-7 |
| 最大PF＋合成complex誤差（m12） | .06913729194855545 | .06842849973806361 |
| 最大位相bias pilot（m12） | .06919248973495769 | .06848201485765056 |
| 最小fixed-N statistical margin | 2.166962701555253e-6 | 2.1852248319187417e-6 |
| Energy誤差上界pilot［Ha］ | .0013549794824010532 | .001373656337788953 |
| 候補別failure上限 | .01 | .01 |

PFと合成signal誤差は別fieldから各一度だけ入力した。DF誤差をdimensionless signalへ混ぜず、`assess_energy`の既存式でDF［Ha］を一度だけ最終RPE energy intervalへ接続した。上表のenergy量は実測energy誤差ではない。13 roundすべてが必要な位相・shots条件を満たしたため、両候補を費用評価へ進めた。

## RPE shots・条件付き総資源

下表のshotsは**各roundの各X/Y軸あたり**。既存Hoeffding solverの整数結果を固定N marginで検証し、事後のN増加・clip・別solverへの変更はしていない。実際の量子outcome抽選は0。

| m | n | B2 CDF shots/axis | B3 XDF shots/axis |
|---:|---:|---:|---:|
| 0 | 1 | 511 | 511 |
| 1 | 2 | 512 | 512 |
| 2 | 4 | 512 | 512 |
| 3 | 8 | 512 | 512 |
| 4 | 16 | 512 | 512 |
| 5 | 32 | 512 | 512 |
| 6 | 64 | 513 | 513 |
| 7 | 128 | 514 | 514 |
| 8 | 256 | 516 | 516 |
| 9 | 512 | 520 | 520 |
| 10 | 1,024 | 530 | 529 |
| 11 | 2,048 | 549 | 548 |
| 12 | 4,096 | 590 | 589 |

| 全13round・X/Y shots加重指標 | B2 CDF8 | B3 XDF8 |
|---|---:|---:|
| **条件付き総RZ count** | **102,056,444,569** | **100,267,421,030** |
| 総RZ depth | 51,452,635,379 | 50,884,473,032 |
| 総CX count | 64,654,784,832 | 63,485,782,056 |
| 総CX depth | 57,938,363,120 | 57,037,719,456 |
| 総total depth | 111,882,019,897 | 110,289,593,414 |
| 総circuit size | 193,844,506,833 | 189,812,628,308 |
| 総shots／executions／measurements | 13,606 | 13,600 |

登録した比較のverdictは `XDF_LOWER_CONDITIONAL_INTERROGATION_COST`。B3 XDFの総RZはB2 CDFより **1,789,023,539（1.7529745883%）少ない**。単一kernelのRZ10,885対11,062に加え、今回の登録shots処方ではm10〜12のXDFが各軸1 shot少ない。保存PF／合成誤差・radius・shotsの全値はround CSVに残し、DF誤差が小さいことだけで資源順位を説明しない。新たな因果分解実験はしていない。

費用は各axisの `n × kernel_metrics + head_metrics + readout_axis_metrics` をそのaxisのshotsで重み付けした既存serialモデルである。加重depthはscheduled serial sumで、全roundを一体compileした最小depthではない。deterministic保存gate metricsから得た費用intervalはpointと一致するが、誤差証拠の厳密enclosureを意味しない。

**今回固定した強baseline比較では、旧H4でのCDF条件付き総RZ優位は維持されなかった。** 旧約11.85%のB0比較は、その元の実装・scope・qualificationにおける保存結果として保持する。単一kernel順位を総資源へ推定したのではなく、許可された今回の13round伝播・shots・CA・費用計算の結果として新比較を報告した。一般分子／rank／compilerでのCDF/XDF優劣、独立したアルゴリズム新規性、C1継続／終了は認定しない。

## 科学call・資源・保全

| call kind | 実績／上限 |
|---|---:|
| saved_bundle_load | 1／1 |
| evidence_attachment | 2／2 |
| round_synthesis | 26／26 |
| round_accuracy（X/Y pair） | 26／26 |
| energy_assessment | 2／2 |
| round_serial_cost（X/Y pair） | 26／26 |
| cost_aggregation | 2／2 |
| cost_comparison | 1／1 |
| **合計** | **86／86** |

86 attempt／86 COMPLETE／FAIL0／inflight0、retry0／resume0。新fit／optimizer／rank探索／DF・PF再計算／Operator再生成／NPZ-QPY数値decode／state-action／回路build-compile／量子outcome抽選／s4／PRはすべて0。正常終了後に科学runnerを再起動していない。

[preflight](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/execution_preflight.json)／[postflight](../../artifacts/research_c_c1_h4_strong_ca_resources/2026-10-26/execution_postflight.json)でSOURCE31、直接保存input22、継承分を含むprotected65件と旧B2/B3 raw84件、契約・manifestが不変であることを確認した。元P-H4 bytes SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` も確認し、分子配列はdecodeしていない。新raw13件とそのbytes hash、全26round・52軸、journal、partial progress、STOPを保存した。

preflightはaffinity32 CPU／利用可能memory56,462,344,192 bytes、CPU／memory pressure avg10=0、I/O avg10=.04。cgroup cpu.max／memory.maxは参照不能であり、quota無制限とは主張しない。既存resource_planから2独立候補のscalar chainにworker2を選び、共有thread設定を変更していない。新たなCPU／RAM／worker／wall上限を追加していない。

既存read-only venvのPython3.11.1実体を直接起動し、依存lockと全指定versionを照合した。cache／tempはtask専用。runner wall **1.481289278715849秒**、CPU **9.774233627000001秒**、peak RSS **345,732 KiB**。speedup倍率は測定していない。同一SOURCE／環境の人工・回帰78 PASSを再利用し、不要な再試験は行わなかった。

## 未資格事項・次の担当・mandatory STOP

full512 Frobenius・norm guardsとその伝播は**level B数値pilot**であり、reference／expm／Operator／norm／solver全体の厳密丸めenclosureではない。`CA_achieved`／`rigorous_enclosure`／`guaranteed_actual_shots`はnull、`guaranteed_total_resource_superiority`はfalseを維持する。own-DF ground oracle、ideal IID Born、状態準備費用除外という条件は外せない。候補別failure.01は2候補同時・事後選択後の保証ではない。

旧H2/H4/H6・B0/B1/B2/B3・XDF9のSOURCE／raw／grant／ledger／marker／STOP、C4／singletonの終了状態は保全した。重要な科学レビュー、追加rank・PF探索、別分子、厳密enclosure補強、実機・状態準備込み評価は行っていない。

**次の担当はユーザー／研究C／統括**。今回の限定比較を確認し、研究C・C1の継続／縮小／終了や追加検証の必要性を判断する。**Codex**は具体的な依頼と必要な別の明示許可を受けた範囲だけを担当する。今回scope内の未完了科学計算はない。

**結果・SOURCE・call・STOPをcommit／pushし、remote SHAと成果物を照合した後mandatory STOP。消費済みrun01のretry／resumeや次の科学計算・重要レビューへ自動進行しない。**
