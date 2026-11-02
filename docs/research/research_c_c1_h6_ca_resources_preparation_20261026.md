# C1：H6共通CA・RPE shots・条件付き総資源の技術準備（2026-10-26）

**既存H2/H4のRPE・shots・serial費用処理を変更せず、保存H6の資格B結果へ接続する評価専用経路を準備した。** 人工・回帰は異なる102件PASS。未承認の次回科学契約は最大119 logical unitsである。

今回の実H6に対する新規数値join、shots solving、energy assessment、cost aggregationは**すべて0**。保存値から新しい順位・CA適格性・総費用を計算していない。準備完了はH6の科学的PASSや資源優位を意味しない。

## 1. 固定する比較条件と来歴

起点`28549bc631ddf39131bf0455dc6345b9da8e19ca`。直鎖H6・1.00 Å・STO-3G、Nα=Nβ=3／sector400、修正codecの保存fit最終CDF rank14／保存selected XDF rank14。通常number表現、元g補正κ・scalar・frame順、全frame ordered symmetric S2、s=1／4、δ=0.1、m=0..12を保持する。PR splitは非適用。

共通CA=0.00159 Ha、DF reserve=CA/8=0.00019875 Ha、各DF ground-state oracle、system状態準備費用除外、ideal IID Born RPEを維持する。P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation degree1／qubits initially zero=falseの保存metricsを使う。

登録候補順はCDF s1、CDF s4、XDF s1、XDF s4。s1を主対象、s4を対照とする。literal/sharedは前段の同target・ordered gate／parameter／qubit／declared phaseを含むexact IR同一性を再利用して、科学評価は各候補1回だけ行う。両routeの入力・alias来歴は結果へ保持する。

- [全基底合成上界結果](research_c_c1_h6_synthesis_bound_execution_20261026.md)：科学SOURCE `e36e0560624c66df0aedf619fc5c8c33ecb4f837`、公開commitは今回の起点。資格Bのkernel／104 roundとIR identityを再利用。
- [native結果](research_c_c1_h6_native_stable_phase_execution_20261025.md)：SOURCE `3d7b56b44832a2a90a8d3e500bf55e856f36da76`、P0 kernel8件とhead／X／Y shell3件の保存metricsを再利用。
- [保存PF結果](research_c_c1_h6_saved_pf_execution_20261025.md)：SOURCE `87ea924ae28353707bd2eb3adf94231042b67361`、保存DF誤差＋guard、PF52行、ground／aliasを再利用。

元分子input、保存J/U、ground NPZ、QPY、PF kernel NPZはhash／来歴の固定に留め、次の経路でも数値decodeしない。必要なのは保存JSONのscalar evidenceとmetricsだけである。

## 2. 再利用SOURCEと最小adapter

| 既存処理 | 今回の使用 |
|---|---|
| `c1_stage3_accuracy.AccuracyTarget`／`RoundEvidence`／`assess_round` | 同じsignal geometry、統計phase予算、候補target、整数shots算定 |
| 既存`rpe_resource_accounting._hoeffding_shots` | 既存のbinary64 log／ceil十分式。新solverは作らない |
| `c1_stage3_deterministic_accuracy.assess_fixed_round` | 新しく算定した整数Nの統計marginを既存経路で再確認。H2/H4の保存Nは転用しない |
| `AccuracyTarget.horizon`／`assess_energy` | 既存horizon、WeylによるDF energy shift、最終phase幅とfailure集計 |
| `compose_serial_cost` | kernel n回＋head／axis tailのserial fenceモデル |
| `aggregate_totals`／`compare_totals` | shots加重6指標・実行数と同sのCDF/XDF比較 |
| `Calls`／`progress`／`resource_plan` | 既存call上限、first-FAIL停止、完了済み値のpartial保存、資源に応じた独立候補処理 |

新規SOURCEは[`c1_h6_ca_resources.py`](../../src/trottertracks/c1_h6_ca_resources.py)、[専用runner](../../scripts/run_research_c_c1_h6_ca_resources.py)、[人工test](../../tests/management/test_research_c_c1_h6_ca_resources.py)。H2/H4のlibrary、fitter、simulator、CAR、PF、compiler、shots solver、energy evaluator、cost aggregatorは変更していない。

旧H2の`native_H2`は回路生成を含み、H4のfixed-shots runnerは保存Nを評価する経路なので、そのままH6へ再起動するのは不適合である。新adapterはJSON schema・candidate対応・資格と整数／数値容量の確認だけを追加し、適合する内部関数へ渡す。

## 3. 誤差・CA・failure配分

H2/H4の固定cost互換契約の`accuracy_target`と、H6保存PF契約のphase contextを静的に照合した。以下はすべて一致し、未決定の新配分はない。

| 項目 | 登録値 |
|---|---|
| β_total | π/6 |
| β_stat | π/12 |
| 決定論phase room | π/12 |
| 候補ごとのfailure上限 | 0.01 |
| 各round・X/Yのα_axis | 0.01/(13×2) |
| horizon | 既存関数でM=12を確認し、m0..12を要求 |
| 決定論law | attenuation=1、corrected-RTE discrepancy=0、deterministic-law discrepancy=0 |

保存済みPF誤差と保存済み合成complex上界を**それぞれ1回だけ**`RoundEvidence`へ渡す。既存solverの戻す`raw_signal_error`が保存`combined_complex_error_pilot`と一致することをgateにする。保存combined値をPF欄へ渡してさらに合成誤差を足す、という二重計上はしない。保存combined／phaseの整合性も確認し、新しいPFや合成上界を作り直さない。

この決定論モデルでRTE／law discrepancyを0とすることと、未知の誤差を0で埋めることは異なる。合成・PF・combined値や資格が欠ける場合は`UNRESOLVED`であり、shotsやCAを認定しない。資格Aのprobeを資格Bの代用にしない。

DF誤差には保存`norm_plus_guard_Ha`を使用し、保存sector誤差＋元guardとのidentityを確認する。`assess_energy`にこのHa値を1回だけ渡し、既存の

\[
\epsilon_{energy,pilot}=d_{DF+guard}+\frac{\beta_{total}}{\delta\,2^M}
\]

で評価する。DF誤差をdimensionless signalへ混ぜたり、guardを再加算しない。各armの実DF誤差が異なることは維持する。これは共通CAに対する数値pilot十分条件であり、実測energy誤差や厳密enclosureではない。

failure上限は**事前登録した候補1件ごと**である。4候補同時保証や事後に選んだwinnerの追加保証は付けない。比較結果に`guaranteed_total_resource_superiority=False`、厳密same-accuracyにnullを保持する。

## 4. shots算定と数値範囲

既存`assess_round`は、保存誤差の共通complex値d、半径下界ρ=1−dに対し、既存の統計coordinate tolerance ρ sin(β_stat)/√2とα_axisを使い、`_hoeffding_shots`を呼ぶ。各候補×13roundでX/Yに同じ登録Nを与え、両軸の整数decimal値を保存する。H2/H4の保存shotsは読まない。

さらに既存`assess_fixed_round`で、算定したNを固定して統計margin≥0を確認する。その引数containerの歴史的な`optimistic_shots_decimal`名は旧関数との接口だけであり、新Nは今回の保存PF＋資格B合成誤差を含めた算定値である。

- Nは正のPython整数であることを確認する。
- log／coordinate-squareのoverflow／underflowや、既存binary64処理で整数精度を資格付けできないN≥2^53は`UNRESOLVED_SHOT_NUMERIC_RANGE`。取得できた巨大整数は未資格decimal文字列として保持し、clipしない。
- 既存fixed-N marginがPASSしなければ`UNRESOLVED_INTEGER_SHOT_MARGIN`。結果を見てNを増やしたり、solver条件を変えたりしない。
- 上記は既存binary64数値経路の容量確認であり、CPU/RAM制限や新しい科学誤差予算ではない。厳密ceil certificateでもない。

実際の量子samplingは0であり、算定shots／予定実行数と、実行済み測定shotsを区別する。

## 5. native費用と条件付き総資源

保存metricsから、各round・軸について既存`compose_serial_cost`の

\[
C_{m,a}=s2^m C_{kernel}+C_{head}+C_{tail,a}
\]

を使う。対象6指標はRZ count、RZ depth、CX count、CX depth、total depth、circuit size。systemのground状態準備費用は除外する一方、Hadamard headはinterrogationの一部として数える。

保存kernel／headのmeasurement countは0、X/Y tailは各1を要求する。tailのtotal depth／sizeにはその測定がすでに含まれるため、測定depthを追加して二重計上しない。予定measurement数は算定shots＝予定executionsに対応する。

serial fenceと別compile済みsegment境界を保ち、一般相対Gaussian融合・inter-kernel融合・global再compileを仮定しない。depthは**scheduled serial stageの和**であり、巨大な回路をglobal compileした最小depthではない。

各候補が共通CAに数値的に適格だった場合だけ、13 roundのX/Y costを構成して既存`aggregate_totals`へ渡す。N_X C_X＋N_Y C_Yを6指標で合計し、shots／executions／予定measurementsも記録する。精度未資格なら費用compositionを省き、既存aggregator gateによって総費用をnullにする。

固定決定論kernelの費用supportはsingletonなので、serialモデル内でlower=mean=upperを使える。これは費用値の登録モデル内の資格であり、RPE成功確率や数値誤差を厳密保証へ昇格させない。既存aggregatorの出力が非有限、整数精度範囲外（≥2^53）、または費用coverage未資格なら総費用は`UNRESOLVED`とする。代替aggregatorや黙示的clipは追加していない。

比較は同sのCDF対XDFの2件。保存kernel RZ（CDF s1=52056／s4=52048、XDF=51046）は既存公表値の引用であり、今回再計測しない。この差や合成上界の大小から、新shotsや総資源順位を予告しない。

## 6. 人工・回帰検証と実行制限

**新規30件＋既存回帰72件＝異なる102件PASS、最終FAIL／skip 0**。回帰はH2/H4のaccuracy／shots／cost、決定論fixed-N、H2/H6拡張、H6 saved-PFを対象とした。mockを使うStage3回帰は独立processで実行した。

人工JSONによるend-to-endで、正常4候補、登録不適格、未解決、route identity／alias／非finite不一致、combined値の二重計上不一致を検証した。固定SOURCE・input hash・budget・call変更のpassive拒否、fresh output保護、最初のFAIL後の新規依存call0、retry/resume0も確認した。

追加では、既存solverへ渡すPF／合成の一度ずつの対応、DF＋guardのHa加算1回、同一failure配分、s4/m12でn=16384、head／tail込みserial depth、予定測定数、未資格accuracyや未資格costへの総費用null、巨大整数／overflow／underflow／不正N／負の統計marginを確認した。途中m3でFAILさせてもm0..2の完了値が既存`progress`経路へ保存される。

証拠：[technical](../../artifacts/research_c_c1_h6_ca_resources/2026-10-26/technical/)。初回の人工fixture書出し／exclusive-save使用上の不具合は技術的に修正し、最終runでPASSを確認した。実H6の科学結果に対する修正・再実行ではない。

保存H6について行ったのはsource確認、既存値の引用、schema／来歴／hash／環境の確認だけである。47 input pins、SOURCE54件のbyte preflight PASS、旧protected証拠872件のhash不変を確認。既存共有環境・H2/H4/H6 SOURCE・旧契約・grant／ledger／marker／STOPは変更していない。今回のH6数値join／shots／energy／総費用／新科学callは0で、予定fresh run01も存在しない。

## 7. 未承認科学実行契約

[固定契約](research_c_c1_h6_ca_resources_contract_unapproved_20261026.json)：scope `C1_H6_SAVED_CA_RPE_SERIAL_RESOURCES_V1`。

| 科学call | 最大logical units |
|---|---:|
| 保存JSON bundle・既存alias／horizon確認 | 1 |
| 4候補のevidence attachment | 4 |
| round精度・shots solving＋整数margin確認 | 52 |
| energy assessment | 4 |
| round serial cost（1 unitがX/Yの2レコードを構成） | 52 |
| cost aggregation＋候補分類 | 4 |
| 同s比較 | 2 |
| **合計** | **119** |

最大出力規模は4候補、52 round accuracy行、104 axis別per-shot cost行、4 energy／総費用、2比較。精度不適格・未解決なら該当候補の13 cost callを省き、未使用枠を追加計算へ流用しない。前段3335／上限6535枠とは独立である。

順序はpassive preflight→保存JSON／ground-alias／horizon→4 packet→独立候補ごとの13round shots／energy→適格候補だけcost→2比較→全結果・partial・SOURCE・call・資源・STOP保存→commit／push／remote照合→mandatory STOP。fresh outputは`artifacts/research_c_c1_h6_ca_resources/2026-10-26/run01`、retry／resume 0。

candidate状態は`NUMERICAL_PILOT_CA_PASS`／`INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION`／`UNRESOLVED`を区別する。不適格は登録十分条件を満たさない意味で、物理的な不可能性ではない。不適格・未解決でも他の登録候補は継続する。SOURCE／input／alias／非finite／phase対応／measurement等のcorrectness FAILでは依存callを止め、開始済み処理を回収してpartial evidenceを保存する。条件変更・fallback・retryは行わない。

新規計算はscalar JSON処理のみ。4候補を既存resource planで並列化し、行列・回路・fitを再生成しない。CPU／RAM／worker／wallの恣意的上限を設けず、task-local cacheを使う。既存eval venv（Python3.11、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0等）と依存lockをread-only再利用する。想定は秒～数十秒だが、実H6 benchmarkは今回実施していない。

SOURCE manifest SHA256：`e9b6b4b106b3a8abd8d7973bae8df8c6320949e9045085802074faf100d64678`

契約 SHA256：`2a0af39e3a795eda74eb9ebac7b60b91a818f6f0c92135179a7b62e90a608504`

## 8. 残るblocker・資格・次の担当

既知の技術的blockerはない。条件・β／failure配分・費用定義に不一致は見つかっていない。ただし**実H6の新しい数値joinと評価は未実行**であり、科学結果はまだない。

次回取得できるのは、資格Bの誤差に条件付けた共通CAの数値pilot判定、候補ごと13round×X/Yの算定shots、energy評価、serialモデルのshots加重総RZ／depth／CX等、同sの条件付き比較である。厳密CA・device成功・状態準備込み資源・DF/PF/合成誤差の厳密enclosure・4候補同時保証・無条件資源優位は未資格のまま。

**ユーザー／研究C**が公開されたSOURCE・契約を確認し、最大119 unitsの新しい保存値限定科学実行を許可するか判断する。**Codex**は別途の明示許可後にのみfresh run01を1回実行する。承認証明・承認原文保存・新grantは不要。今回はcommit／push／remote照合後mandatory STOPし、実H6のshots・CA・総費用へ自動進行しない。
