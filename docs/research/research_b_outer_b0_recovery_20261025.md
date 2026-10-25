# 研究B B0：技術復旧と新attempt

2026-10-25 JST。**初回の大きなcompiler位相集計誤差と失敗保存不足を修正したが、B0全体は未完了。D1の残存compiled correctnessは未解決のままmandatory STOP。費用計測0件。** 許容差・precision・物理条件を変更してPASSにはしていない。

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

## 最終状態・次の担当

**B0全体は未完了、corrected-1のraw状態はFAILED_MANDATORY_STOP／B0_CORRECTNESS_FAILURE。** 最終修正SOURCEについても保存D1回路の数値検査は固定tol未達で、全48 trajectoryの新しいPASSや費用表は得られていない。初回技術問題の改善と、全科学correctnessの成立を分ける。Gamma／shots／expected cost／u_numの新資格は取得していない。

公開branchは `codex/research-b-outer-b0-recovery-20261025`。Codexは修正・全attempt・診断・限界をcommit／pushし、remote SOURCE／成果物を照合してmandatory STOPする。次の担当は研究Bのユーザーで、長いnative列のoperator評価とcompiler phaseを独立に検証する方針を判断する。固定precision／toleranceやcompiler条件の変更、追加の科学診断、全pilot再実行は自動開始しない。A0／旧run／既存grant・STOPは維持する。
