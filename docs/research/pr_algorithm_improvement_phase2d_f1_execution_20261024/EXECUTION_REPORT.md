# F1 one-shot実行結果

## 判定と固定範囲

状態は **F1_TECHNICAL_FAILURE_PARTIAL_RESULTS_MANDATORY_STOP**。実行済みの数値gateでは許容差超過を観測していないが、技術的エラーで打ち切られた。全F1のPASSでも、Pauli／frame混在アルゴリズム全体の正しさの証明でもない。

P0の4 mode、2粒子、occupied(0,1)、L_D=1、決定論one-body block1、残差density-square2fragmentの合成fixture。分子geometry／AO basis／分子DF rankは非該当。SS／SP／PS／PP、T=1/5、q2、delta=1/10の一点、r2、h1/20、K2。full16／controlled32、JW little-endian／ancilla-last、faithful_identity_in_tail、scalar／products→child順／paired phaseを保持した。catalogは保存4label、c=-1/31、T=h=1/20、q=r1、K2。

今回のユーザー指示で、literal価格の数値bindingを登録price case内へ移すことと、未資格intervalの既定UNPRICED_FALLBACKを正式採択した。他のP0条件、数値許容差、compiler profileは変更していない。

## SS／SP／PS／PPの比較

| arm | H-ID full／sector | LAW event record完了／予定 | component完了／予定 | finite mean |
|---|---|---:|---:|---|
| SS | PASS（保存max-entry 0） | 2,758/2,758 | 42/42 | one-step 1.11036e-16／全体 7.77226e-16：PASS |
| SP | PASS（保存max-entry 5.63536e-132） | 13,857/195,170 | 0/174 | 未実行 |
| PS | PASS（保存max-entry 5.63536e-132） | 17,602/17,602 | 12/78 | 未実行 |
| PP | PASS（保存max-entry 5.63536e-132） | 13,672/132,702 | 0/153 | 未実行 |

H-ID閾値は1e-11、4armすべてのsector残差は保存値0。各保存S/P fragmentと独立CAR fragmentの照合、HのHermiticityも通過した。H-IDのreportは128digits参照をbinary64へ変換した後の比較であり、極小の保存残差を百桁精度の保証とは解釈しない。

SSだけがCOMPONENT42件とMEAN1件を完了した。SSのmeanはactual component control1 blockから構成し、候補Rへ同じ多項式を適用しただけの比較ではない。SP／PS／PPのmeanは未実行であり、SSのPASSを外挿しない。価格・資源順位の比較は行えていない。

SSとPSは全一microstep eventの個別metadata検査を完了したが、LAW最終mass／集約L1 gateはpartition終端にあり、今回の停止時には実行・保存されていない。保存event値から後追いでこの科学gateを補完していない。従ってLAW全体のPASS表示は保留する。

## 登録familyごとの実績

| family | 登録 | attempt | completed | 状態 |
|---|---:|---:|---:|---|
| F1.CATALOG.EVENT | 136 | 60 | 60 | PARTIAL_PASS |
| F1.CATALOG.MEAN | 2 | 0 | 0 | NOT_EXECUTED |
| F1.COMPILE | 64 | 1 | 0 | TECHNICAL_ERROR |
| F1.COMPONENT | 447 | 54 | 54 | PARTIAL_PASS |
| F1.EXHAUSTIVE | 1 | 0 | 0 | NOT_EXECUTED |
| F1.H | 4 | 4 | 4 | PASS_REGISTERED_COMPLETE |
| F1.LAW | 348,232 | 47,889 | 47,889 | PARTIAL_EVENT_CHECKS_FINAL_MASS_GATE_UNEVALUATED |
| F1.MEAN | 4 | 1 | 1 | PARTIAL_PASS |
| F1.MOTIF | 96 | 3 | 3 | PARTIAL_PASS |
| F1.PRICE.ARMS | 4 | 0 | 0 | NOT_EXECUTED |
| F1.PRICE.COMPARE | 1 | 0 | 0 | NOT_EXECUTED |
| F1.PRICE.DP | 1 | 0 | 0 | NOT_EXECUTED |
| F1.PRICE.ENUM | 68 | 30 | 30 | DIRECT_COUNTS_ONLY_DP_COMPARISON_NOT_EXECUTED |
| F1.REFERENCE | 506 | 109 | 109 | PARTIAL_REFERENCE_EVALUATIONS_AND_PAIR_AGREEMENTS |
| F1.SELECTOR | 1 | 0 | 0 | NOT_EXECUTED |
| F1.WRAPPER | 64 | 1 | 1 | PARTIAL_PASS |

LAWはevent records、COMPONENT／MOTIF／CATALOG EVENTはcontrolled buildsと対応gate、WRAPPERはwrapper builds、COMPILEはcompile requests、REFERENCEはprecision evaluations、その他はlogical caseの単位。行を合算して科学call総数を作らない。

失敗attemptは `SS_motif_0_X_compile` の一件。他4partitionは共有STOP_REQUESTEDを確認し、次のcaseをattemptする前に協調停止した。これらのRuntimeErrorを別の数値的FAILや失敗buildとして数えない。parallel実行のため既にattempt済みの処理は完了recordを残して終了した。

REFERENCE109件は90digits55件／128digits54件。54組のagreementを確認した。catalog_30_90は計算済みだがpaired128は未実行であり、二精度整合済みには含めない。精度増加やretryはない。

## 保存された主な残差

| 保存診断 | 観測最大値 | 固定許容差／範囲 |
|---|---:|---|
| 各event probability差 | 2.77556e-17 | 1e-12、実行済み47,889 recordsのみ |
| 各event coefficient差 | 2.77556e-17 | 1e-12×max(1,reference)、同上 |
| controlled operator | 1.11027e-15 | 1e-10、component54／SS-M0の3variant／catalog60 |
| unitarity | 2.10942e-15 | 1e-10、同上 |
| component Hermiticity／involution | 2.44929e-16／7.09736e-16 | 1e-11、完了したproduct probesのみ |
| SS one-step／full finite mean | 1.11036e-16／7.77226e-16 | 2e-10、SS一構成のみ |
| raw X信号 | 6.66134e-16 | 2e-10、SS-M0-X一件のみ |
| paired reference operator | 5.16038e-91 | 1e-60、完了した54組の比較範囲のみ |

全gateはrtol=0、phase fittingなし。M0はSS一trajectoryのraw／optimized／actual inverseの3variantを確認しただけで、M1–M7やmixed armのfull motif検証は未実行。Y wrapper／compiled operator・signalは未検証。上表はjournal内の保存診断値の最大を引用したもので、新しい行列・信号・reference再計算はしていない。

## 技術的停止の証拠と原因

最初のcompile requestが、固定SOURCE `src/trottertracks/pr_dictionary_validation/gates.py:81` の `normalized.metadata=None` で失敗した。Qiskit1.3.0のmetadata setterはdict以外を拒否するため、TypeError `Only a dictionary is accepted for circuit metadata` となった。例外位置はcache用IRのnormalization中であり、後続の `transpile(...)` へ到達していない。compile request1、actual transpiler invocation0、completed compile0を区別する。

F1-PREではtranspile keyword signatureを確認したが、このmetadata設定の接続不具合は検出できていなかった。今回は追加benchmarkや再現用回路を作らず、保存tracebackと固定SOURCE・インストール済みsetterの読解で原因を特定した。科学的なoperator不一致による停止ではない。SOURCE修正、Noneをdictへ置換した確認run、grant再発行、retry／resumeは実施していない。

## 価格とinterval

catalogのliteral tableは最初の登録price caseで一度bindingし、直接count30件を保存した。68件全列挙、DP、直接／DP aggregate比較は未完了なのでPRICE_PASSを出さない。physical4armのliteral期待価格はまだ計算されていない。

interval providerは承認範囲どおり未資格。ただしARM_PRICE／SELECTOR／EXHAUSTIVEへ到達していないため、fallbackの実処理も今回は未検証。UNPRICED_FALLBACKを正しく処理できたというPASSはない。認証score、最適選択、shot数、新しい価格／shot資格、F2の資源性能はすべて未取得である。

## 実行費用と保存

2026-10-24 01:00:01.174816–01:00:11.776060 JST。wrapper実測wall 10.601283506秒、GNU time user CPU36.75秒／system CPU1.38秒、CPU utilization359%、max RSS533,596KiB。RSSはGNU timeの最大値であり、同時workerの合計RAMではない。

CPU affinity32、5独立datasetのspawn ProcessPool、BLAS/OpenMP/Numba/Qiskit子process各1、Rayon各6。wall／memory capなし。起動時RAM空き約52GiB、disk空き約346GiB、load約1。NVIDIA driverは利用不能だがfull32小系のCPU経路でありGPUを必要条件にしない。共有環境や他jobを変更していない。速度向上倍率は測定していない。

各完了caseのwall時間はmachine summaryのfamily別に集計した。失敗したcaseやphase境界のSOURCE照合・pickle・import・I/O等はそのcompleted合計に含まれず、並列case時間を全体wallへ足し合わせない。partition終端のCPU／RSS記録は途中停止により未出力なので、arm別CPU費用を推定しない。

rawは369file、29,981,263 bytes。journal、QPY、NPY、reference文字列表現、入力snapshot、resources、binding snapshot、STOP、output hash台帳を非公開で保全した。科学処理の再実行なしでproducer hash台帳を照合し、公開用には要約とrelative-path hash台帳だけを作った。ユーザー承認原文、grant／marker、内部absolute path、raw係数・回路・行列は公開していない。

## 再現性と次の段階

科学SOURCEはF1-PRE固定commit `31e8391a56fbb1a0e64f477f78e242655329de99` のまま。B固定結果 `32c76bc6392ba277ce9fdd8d4bc54ffb5339b767` の入力13file（B原本12＋lock）をbyte-copyした。SOURCE closure、input hash、env lock、compiler、閾値はmachine summaryとF1-PRE manifestで追える。新grant1件を一度消費し、旧BのSOURCE／raw／packet／marker／STOPと旧F1-PRE資料を保持した。

これは固定SOURCEに基づくローカルone-shot部分検証であり、immutable CIや外部の独立再現ではない。実行済みgateの数値PASSと、F1全体の未完了を区別する。新規性や一般的な資源優位性は認定しない。

次の担当はユーザー／GPT。取得済み結果を確認し、metadata処理の技術修正と新しいF1実行範囲を承認するか判断する。CodexによるSOURCE修正や再実行は今回のSTOP後に自動開始しない。F2にはF1の残る正しさ検証、価格／intervalの扱いと必要な資格・比較条件の確定、さらに別の明示実行承認が必要である。重要科学レビューも自動開始しない。
