# 研究B B0：技術復旧と新attempt

2026-10-25 JST、追補後の最新状態：**B0 COMPLETE。SOURCE b190fe28b、全288 operator／192 expectation検査PASS後に96 wrapperの費用を測定・保存しmandatory STOP。** 許容差・precision・物理条件を変更していない。Gamma・shots・expected costの資格は未評価。以下の初回復旧履歴と旧STOPは保全し、末尾に再開後の結果を追記する。

旧SOURCE `8d0cb96df9d1f6229f207550634c7e11aff2851c` と[旧失敗原本・STOP](../../artifacts/research_b_outer_b0/20261025/native-pilot-once/)は保全する。今回の修正・同条件再実行はユーザーが明示的に指示した範囲。承認証明・grantは作らない。

## 将来attemptの限定変更

科学fixture／sector／state、6候補・両T・4代表、全48 trajectory・96 compile要求、complex128、operator 2-normのabsolute atol1e-12／rtol0／phase fitなし、compiler profileは[既存契約](research_b_outer_a0_b0_execution_preparation_20261025.md)と同じ。旧attemptを上書き・resumeせず、各attemptはfixed clean SOURCEと排他的fresh outputで実行しSTOPする。今回許可された技術復旧に限り、原因と修正理由を確認して新attemptを開始できる。原因不明の反復・科学的条件の変更には拡張しない。

まず `--diagnose-first` で旧失敗identity B3／T=+0.18／代表0／Xだけを検査する。入力5境界、config6、event／event operator各6、RTE sequence12、D block18、独立exponential9、system／controlled／reference各1、operator gate4、expectation2、compile1。費用、全mean、他caseは起動しない。Metadata scopeに登録した全境界を照合する。

FAIL時には数値差・actual／reference matrix、raw／compiled回路をgzip+base64 QPYでpartial_results内に保存する。成功したgate・meanの数値をeventsへ保存し、wrapper rowをcompiled gateより前にpartialへ登録する。既存EvidenceのJSON／総出力枠は維持する。保存失敗もSTOP。共通認証・汎用frameworkを追加しない。

診断で明確な実装問題を確認した場合だけ最小修正・関連テストを行い、新科学SOURCEをcommit固定し、同じ科学条件の全B0をfresh outputで実行する。全gateを費用96要求に先行させ、数値許容差を緩和しない。技術的に解決できないcorrectness FAIL・識別力不足は保存してSTOPする。

## 実績

診断SOURCE `2721d69a98213a35facf4dc7fafa08ab1f0b40ea` で最初の失敗caseを再現し、spectral誤差1.791447313e-11、raw wrapper誤差1.155206482e-14とQPY／行列を保存した。保存回路だけのpass診断（script SOURCE e82e99558）では、最適化前のofficial-definition展開ですでにrawから1.792369935e-11ずれ、global phaseが2045.1768174869376 radに累積していた。

修正はB0からだけ指定する `compiled_wrapper(...,stable_phase=True)`。Gateの公式definitionをそのまま辿り、global phaseの項を `math.fsum` で一度集計し2πでreduceしてから既存transpileへ渡す。Operatorから位相を推定・fitしない。Native basis、optimization1、seed17、layoutその他profileは維持し、loweringの数値実装名だけ `official_definitions_fsum_phase_v1` と記録する。旧研究の既定compiler経路は変更しない。Cache identityにも新loweringを含める。

関連人工検証はB0 opaque11件／共通10件PASS、compiler人工5件PASS（旧3件＋多数phase項・controlled公式definition2件）。修正版は最初のcaseを同じtolで確認後に同条件全pilotを行う。旧契約や旧結果を遡及変更しない。A0再実行、A1包含、MC、shots、新PF・grid、全RPE評価は対象外。

V1 SOURCE `c7cde1868a6b38f9505ecc4f0858b98b49c506e8` の限定診断はcompiled誤差4.386058909e-13でPASSしたが、同SOURCE全体attempt corrected-1は16 wrapper通過後、D1／+T／代表0／Xで2.318114585e-12となりSTOP。保存D1回路の段階診断では、lowering直後1.585565770e-12、BasisTranslator後4.635958878e-12であり、U-to-native展開と位相の大きな和の剰余処理にも丸めが残っていた。

V2では定義由来binary64 phaseの和と2π剰余をFractionで正確に処理し、最後に一度float化する。Uの既知ZSX分解とP=phase(theta/2)RZ(theta)を明示し、scalar phaseを同じ和へ含める。これでBasisTranslatorへ大量のUを渡さず、native gate列へ同じoptimization1・seed17を適用する。Operator評価や位相fitをcompilerへ導入せず、許容差は固定。Lowering名は `official_definitions_exact_phase_native_v2`。追加の人工U/P identity検証もPASSした。

## 最終修正SOURCEと残存問題

最終の実装SOURCEは **`5c1fc5bc36bb3dad32efe7825ca6ac9058deeafb`**。B0専用lowering名は `official_definitions_exact_phase_native_v4`。公式の短いH/S/Sdg/T/Tdg/SXdg恒等式も使い、冗長なU経由の展開を避けた。さらにQiskitのlevel1 pass構成・gate選択・既存one-qubit optimizerを再利用し、各1q runを同じoptimizerで処理して親DAGへのscalar phaseだけを正確に集計する。変更は新しいローカルpass managerのinstanceに限り、共有optimizer class／SessionEquivalenceLibrary／他研究の既定compilerを変更していない。

これは許容差を満たすと認定済みのSOURCEではない。保存されたD1／+T／代表0／X回路の追加診断では、lowering後のrawとの差は約5.997e-13まで低下したが、最終compiledとの差は約1.387e-12で、固定1e-12をなお上回った。これをglobal phase fit・再正規化・tol緩和・高精度への変更でPASSにしなかった。定義展開に起因する初回誤差の一部は技術的に説明・改善できた一方、残る差を単なる評価丸めかcompilerのgate値の差かに確定できない。追加の全pilotを反復せず、未解決のcorrectness／数値識別力問題としてSTOPする。

## 各attempt・保存結果

系は既存synthetic2-mode one-body＋3DF、N1／mode0、LD2 weight-ranked、nD1、T±.18。候補B3/D1/R2/R4/B4/F1、r/K・q・refinementは既存契約どおり。分子geometry／AO basis／分子DF rank探索は非該当。以下は同条件の新attemptと保存回路だけの技術診断である。

| 新launcher attempt | SOURCE | 結果 | Compile要求完了／compiled gate PASS | 費用 |
|---|---|---|---:|---:|
| [diagnostic-1](../../artifacts/research_b_outer_b0/20261025/diagnostic-1/) | 2721d69a9 | B3/X FAIL、誤差1.791447313e-11を保存 | 1／0 | 0 |
| [diagnostic-2](../../artifacts/research_b_outer_b0/20261025/diagnostic-2/) | c7cde1868 | B3/X PASS、compiled誤差4.386058909e-13 | 1／1 | 0 |
| [corrected-1](../../artifacts/research_b_outer_b0/20261025/corrected-1/) | c7cde1868 | 全pilot再実行、D1/X FAIL、誤差2.318114585e-12 | 17／16 | 0 |

全pilot corrected-1では入力、config76、event／event operator各1740、single-step mean24、analytic mean3、system／controlled各9、raw wrapper17が通過した。Operator gateは51 PASS／1 FAIL／236未試行、expectationは33完了／159未試行。Compiled expectation16完了、費用96要求は全て未試行。3 caseのmeanはB3・B4・D1のT>0で、残る9 caseは未試行。Partial18行は完了wrapper16、途中wrapper1、FAIL診断1で、成功wrapper18件ではない。最終result.jsonはない。

旧attemptの空partialに対し、新FAIL partialには誤差量、actual／reference matrix、raw／compiled QPYが残り、完了したgate／mean値もeventsに保存された。各runのstatus／STOPを保全し、evidence_write_failuresは全て空。[境界別実績CSV](../../artifacts/research_b_outer_b0/20261025/RECOVERY_COUNTS.csv)は予定・試行・完了・失敗数の列抽出だけで、費用表ではない。

保存回路の追加compiler段階診断は各1 compile要求、費用0、新fixture／代表なし。以下の誤差は保存raw operatorからの差で、独立full-space参照からのlauncher gate誤差とは区別する。

| 技術診断 | SOURCE | Case | Lowering後 | 最終compiled |
|---|---|---|---:|---:|
| [1](../../artifacts/research_b_outer_b0/20261025/compiler-diagnosis/passes.json) | e82e9955826afeae8a897820f7477182e8446b38 | B3/X | 1.792369935e-11 | 1.791470382e-11 |
| [2](../../artifacts/research_b_outer_b0/20261025/compiler-diagnosis-2/passes.json) | e6e0e1b4894d4739269b8cbe79dbaf89fcea95ab | D1/X | 1.585565770e-12 | 2.318114770e-12 |
| [3](../../artifacts/research_b_outer_b0/20261025/compiler-diagnosis-3/passes.json) | b84f2653a74897c7227e9505a078f458d479b2dd | D1/X | 1.445601885e-12 | 1.551542939e-12 |
| [4](../../artifacts/research_b_outer_b0/20261025/compiler-diagnosis-4/passes.json) | 4c20ecd2baa66a8d9bb0676a6b030a0ddb96dfca | D1/X | 5.997003273e-13 | 1.651187457e-12 |
| [5](../../artifacts/research_b_outer_b0/20261025/compiler-diagnosis-5/passes.json) | 5c1fc5bc36bb3dad32efe7825ca6ac9058deeafb | D1/X | 5.997003273e-13 | 1.387341732e-12 |

診断3はU/P native化と正確な剰余、4は固定gateの短い恒等式、5は1q optimizer間の位相集計という具体的変更の効果確認である。原因不明の同一SOURCE反復ではなく、SOURCE・理由・結果を区別して残した。技術診断3～5は同じ保存raw／旧compiled QPYを入力とし、出力JSONには段階別数値・phase・gate数・SOURCE・operator評価数を保存している。試行生成した最終compiled QPY本体はこの技術scriptの出力には含まれず、新launcherのFAIL QPYとは区別する。

## 検証・環境・保全

Focused人工検証はB0 opaque11、共通IO10、compiler数値人工7、cache metadata2の**計30件PASS**。Compiler人工では固定gate/U/P恒等式、controlled相対位相、多数phase項、500個の1q run、共有optimizer class不変を確認した。[test logs](../../artifacts/research_b_outer_b0/20261025/recovery_tests/)を保存する。実fixture・保存科学arrayを人工テストに使っていない。今回許可された技術回帰では人工回路をbuild／compileしており、準備時の「回路生成0」の記述とは時点が異なる。テストPASSを全B0科学PASSとしない。

既存CPython3.11.9／NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Nature0.7.2／OpenFermion1.6.1／OpenFermion-PySCF0.5／PySCF2.7.0を使用し、依存install・共有設定変更なし。全pilot再実行直前に32 CPUと空きRAM約52.6 GiBを確認し、固定2-mode、既存cache、CPU backendを使った。CPU／RAM／worker／wall-timeへ人工的な低い制限は設けていない。

Launcherのwallはdiagnostic-1が8.05秒、diagnostic-2が16.85秒、corrected-1が37.38秒。対応する[process記録](../../artifacts/research_b_outer_b0/20261025/execution_logs/)と各technical diagnosisのtime／stdout／stderrを保全する。速度倍率は認定しない。各attemptのSTOPはそのまま残し、上書き・resumeはない。新原本とprocess記録のcopy一致、登録scope内実績、旧証拠とA0の不変を確認した。

## d6c519146時点の状態・次の担当（履歴）

**B0全体は未完了、corrected-1のraw状態はFAILED_MANDATORY_STOP／B0_CORRECTNESS_FAILURE。** 最終修正SOURCEについても保存D1回路の数値検査は固定tol未達で、全48 trajectoryの新しいPASSや費用表は得られていない。初回技術問題の改善と、全科学correctnessの成立を分ける。Gamma／shots／expected cost／u_numの新資格は取得していない。

公開branchは `codex/research-b-outer-b0-recovery-20261025`。Codexは修正・全attempt・診断・限界をcommit／pushし、remote SOURCE／成果物を照合してmandatory STOPする。次の担当は研究Bのユーザーで、長いnative列のoperator評価とcompiler phaseを独立に検証する方針を判断する。固定precision／toleranceやcompiler条件の変更、追加の科学診断、全pilot再実行は自動開始しない。A0／旧run／既存grant・STOPは維持する。

## 再開後の独立D1診断と角度保存修正（2026-10-25）

上記は結果公開commit d6c519146時点の履歴である。ユーザーの継続指示により、同じD1／T=+0.18／代表0／Xを再開した。物理条件、complex128、absolute operator 2-norm tol1e-12、rtol0、phase fitなしを変更していない。

独立経路は各primitive gate matrixを既存Qiskitから取得し、little-endianのbit actionで8×8へ埋め込み、append順を保つbalanced matrix productをcomplex128で計算する。既存sector helperはfull Operator(circuit)を呼ぶため、この蓄積誤差の切り分けには独立ではなかった。Precisionを上げず、射影・再正規化・phase fitも行わない。人工bit-order／非可換積／global phase／odd tree／terminal measurementの2テストで既存Operatorとの一致を確認した。

| D1保存回路の同条件比較 | 修正前：SOURCE edd2d6ffb | 角度保存修正後：SOURCE b190fe28b |
|---|---:|---:|
| Raw、従来Operator対独立数学参照 | 1.817887053e-14 | 同じ保存raw |
| Raw、balanced対独立数学参照 | 3.535669878e-14 | 同じ保存raw |
| Compiled、従来Operator対独立数学参照 | 1.387315027e-12 | 3.820996400e-13 |
| Compiled、balanced対独立数学参照 | 1.456333140e-12 | 5.210662172e-13 |

両経路で修正前のtol超過が残り、Operator(circuit)の蓄積誤差だけでは説明できなかった。[独立診断1](../../artifacts/research_b_outer_b0/20261025/independent-1/diagnosis.json)／[独立診断2](../../artifacts/research_b_outer_b0/20261025/independent-2/diagnosis.json)に数値・raw／compiled QPY・SOURCE・実績を保存した。各1 compile要求、他の科学caseは追加していない。

新修正SOURCEは **`b190fe28b98d2f94e1509e19cd3d36a63b2bb99e`**、lowering名は `official_definitions_angle_preserving_native_v5`。既存RY→U→ZSX列のtheta+piで微小角を丸めないよう、厳密な `RY(theta)=SX† RZ(theta) SX` と `SX†=X SX` を使用する。Uは `exp(i(phi+lambda)/2) RZ(phi) RY(theta) RZ(lambda)` に従って描画し、元のthetaを保持する。

さらにnative1q gateのdeterminant（RZは1、SXはi、Xは−1）から、1q再合成のscalar phaseがpi/4の整数倍となることを使う。既存optimizerが選んだ離散branchとdeterminant invariantを検査し、既知の位相項を整数として集計する。外部referenceやfull operatorから位相を推定して合わせない。Guard不一致ならSTOPする。科学的correctness tolを緩和する変更ではなく、compiler内の既知の恒等式と離散位相の数値実装修正である。

Focused人工検証はopaque11＋共通10＋compiler8＋cache2＋balanced2＝**33件PASS**。[新test logs](../../artifacts/research_b_outer_b0/20261025/independent_tests/)を保存する。追加した微小RY testは元角度がRZ引数へそのまま残ることも検査した。科学fixtureを人工テストに流用せず、科学的性能・数値保証を人工PASSで代替しない。

## v5による固定B0全体の完了

同じ修正SOURCEの独立clean checkout `/tmp/prt-research-b-outer-b0-v5-science-20261025` を作成し、HEAD／SOURCE、snapshot5ファイル、依存version、固定scope、fresh output非存在を確認してから一回実行した。新たな中断・SOURCE修正はなく、**COMPLETED_DESCRIPTIVE_MANDATORY_STOP** で正常終了した。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-b0-v5-science-20261025/scripts/run_b2d_outer_b0.py \
  --execute \
  --source-commit b190fe28b98d2f94e1509e19cd3d36a63b2bb99e \
  --output /tmp/prt-research-b-outer-b0-v5-science-20261025/artifacts/research_b_outer_b0/20261025/corrected-2
```

この出力は使用済みで再起動しない。[原本8ファイル](../../artifacts/research_b_outer_b0/20261025/corrected-2/)と[process記録4ファイル](../../artifacts/research_b_outer_b0/20261025/execution_logs/corrected-2/)を実行場所からbyte一致で保全した。[全96 wrapper CSV](../../artifacts/research_b_outer_b0/20261025/WRAPPERS_v5.csv)は今回resultの数値列抽出だけで、operator／signal／回路の再評価ではない。

| 登録境界／実績 | 完了 |
|---|---:|
| Candidate×T／代表trajectory | 12／48 |
| Input fixture／split／prepare／full inputs／input gate | 各1 |
| Config／event／independent event operator | 76／1740／1740 |
| Single short-step mean／full analytic mean | 24 logical（event項1692）／12 case |
| System／controlled buildとoperator gate | 各48 |
| Raw／compiled wrapper operator gate | 各96、operator gate総数288 |
| Raw／compiled X/Y expectation | 各96、総数192 |
| Wrapper build／compile要求／費用測定 | 各96 |
| 実transpile／cache hit | 96／0 |
| Shared D block／RTE sequence | 372／492 |
| Shared exponential／moments | 206／16 |
| Event記録 | 11,382（上限11,414内） |

全kindのattempted＝completed＝登録scope、failed={}、evidence_write_failures=[]。全入力・mean・operator・expectationのgate完了を、最初のcost measurement開始より前に確認した。Scopeの入れ子を合算した「独立job総数」は定義しない。

| 保存された数値診断 | 最大誤差 |
|---|---:|
| System operator | 1.490079946e-13 |
| Controlled operator | 2.270468419e-14 |
| Raw wrapper operator | 2.429614902e-14 |
| Compiled wrapper operator | **4.426725459e-13** |
| Raw X/Y expectation | 2.342570582e-14 |
| Compiled X/Y expectation | 2.504663144e-13 |
| Single-step event mean | 4.662937506e-15 |
| Full analytic signal対保存sector signal | 2.097888250e-15 |

Input gateではstate mapping差0、H identity差1.110223025e-16、R identity差5.785335647e-17だった。Control-zero／one、mode0準備、X/Y符号、signed RTE、scalar／event相対位相、compiled full operatorを、固定atol1e-12／rtol0・phase fitなしで確認した。これらはローカル経験的correctnessであり、γや数値包含のcertificateではない。

## Native費用の観測範囲

以下は各候補16 wrapper（T両符号×4代表×X/Y）での保存値のmin–max。既存 `circuit_metrics` のRZ／CX count・filtered depth・total depth・sizeを使い、mode0準備とancilla measurementを含むfull controlled wrapperを測定した。各wrapperのmeasurement_countは1。全値はCSV／raw JSONへ保存する。

| 候補 | RZ count | CX count | RZ depth | Total depth |
|---|---:|---:|---:|---:|
| B3：Yoshida4/A2/r2K2 | 3994–4070 | 2716–2764 | 2588–2636 | 6063–6165 |
| D1：Yoshida4/A3/r2K2 | 27322–27398 | 18700–18748 | 17708–17756 | 41487–41589 |
| R2：Yoshida4/A3/r4K2 | 27346–27494 | 18724–18820 | 17732–17824 | 41535–41733 |
| R4：Yoshida4/A3/r2K4 | 27322–27398 | 18700–18753 | 17708–17756 | 41487–41592 |
| B4：SABA2/q2/A2/r2K2 | 2020–2066 | 1368–1395 | 1308–1337 | 3065–3126 |
| F1：refined-S2/q2/f8/A2/r2K2 | 10624–10822 | 7236–7359 | 6888–7013 | 16133–16398 |

CX filtered depthは今回CX countと同じ範囲。Circuit sizeはB3 8446–8586、D1 57694–57834、R2 57742–58018、R4 57694–57839、B4 4276–4361、F1 22456–22821だった。これは6候補の決定論的代表の観測値であり、IID平均・確率重み付き期待費用・同精度達成の最小費用ではない。A3のD精度と今回のnative負担を分けて検討する材料で、B4等の正式採択・総資源優位を自動認定しない。

Resultの **expected_cost=null、shots=null、gamma=null** を維持する。Normalization／analytic signalとgate価格の接続を観測できたが、shot数・finite-bit／γ証明、A1包含、B1 MC、全RPE総資源を実施したことにはならない。

## 再開後の環境・最終状態・次の担当

CPython3.11.9、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0の既存環境を再利用した。起動直前32 CPU、空きRAM約52.6 GiBを確認し、依存install・共有設定変更なし。2-mode／shared cache経路を維持し、CPU／RAM／worker／wall-timeへ人工的な低い上限は設定していない。

Whole-process wall **11分13.74秒**、user CPU680.97秒、system CPU3.75秒、最大RSS1,509,036 KiB、exit0。既存launcherと1q optimizerの走査は逐次で、実CPU利用は約101%。並列speedupは測定・認定していない。将来の大規模MCではこの古典compile経路の高速化が残る課題だが、今回完了したfixed runを速度測定のために再実行しない。

新loweringは微小角と離散phaseの両方を修正しており、個別寄与の完全なablationはしていない。共同修正前後の同じD1で両operator経路を照合し、同じ科学条件の全gate通過を確認した範囲に結論を限定する。Balanced経路もprimitive gate matrixをQiskitと共有する経験的な独立蓄積確認で、rigorous u_num／γにはしない。

**最新状態はB0 COMPLETE、mandatory STOP。** 原本・全旧attempt・STOP・A0結果を保全した。公開branchは `codex/research-b-outer-b0-d1-independent-20261025`。CodexはSOURCE／結果／文書をcommit・pushしremote照合した後に追加科学計算を開始しない。次の担当は研究Bのユーザーで、この限定native費用とA0結果を踏まえ、期待費用B1／shot資格等の次段階を必要とするか判断する。いずれも今回から自動移行しない。
