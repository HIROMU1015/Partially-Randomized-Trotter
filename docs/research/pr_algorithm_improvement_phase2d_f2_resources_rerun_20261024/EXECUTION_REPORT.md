# 限定F2：位相保持修正後の全件結果

**F2_RESOURCE_NUMERICAL_PASS_MANDATORY_STOP**。
ユーザーの修正・新規実行承認により、修正版SOURCEと新grantで一回実行し、全登録検査・資源集計を正常完了した。
今回の実行中の中断・追加修正・別attemptは0件。旧失敗結果・grant・STOPは保全し、旧runをresumeしていない。
認証N・N加重費用・physical同精度の総資源優位は対象外のままである。

## 固定条件と技術修正

SOURCEは `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`、closureは
`cf027aae3e06d88154c782be8b8dc0f932377ef333c6751d3ef58a04d5992022`。
107 SOURCE、保存B入力7file、固定依存50件、最新承認原文・新one-shotのbindingをpreflightで確認した。

4 mode・2粒子・occupied(0,1)・L_D=1・残差density-square 2fragmentの同じ合成Hamiltonian。
geometry／AO basis／分子DF rankは非該当。T=1/5、q1、delta=1/5の一点、r2、h1/10、K0。
Hamiltonian・state・sector・scalar・D/R・faithful identity tailを維持した。
compilerはQiskit1.3.0、opt1、seed17、rz/sx/x/cx、topology-free、approximation_degree1、qubits_initially_zero=False。
独立reference90/128digits、rtol0、phase fitting禁止、全P0数値閾値を変更していない。

旧不一致はQPY読戻しではなく、固定環境のBasisTranslatorで入れ子equivalenceを変換する段階の位相欠落に局在化した。
公式UnrollCustomDefinitionsと独立した空EquivalenceLibraryを使い、公式gate定義をnative＋Uへ展開してから同一profileのtranspileへ渡す修正を行った。
位相fit、推測によるπ補正、共有環境やSessionEquivalenceLibraryの変更はない。
[修正証拠](fix_evidence.json)に、旧失敗3件＋対照1件の修正後一致、人工回帰5件PASSを記録した。
これらの修正診断と、本表の登録F2 callは別枠であり、異なる作業を科学call総数へ合算しない。

## 科学的正しさと実績

| 登録単位 | 実績 | 判定 |
|---|---:|---|
| arm law／probability mass | 各4 | PASS |
| ordered trajectory | 6,837 | 全件完了 |
| build worker job（共通診断込み） | 6,838 | 全件完了 |
| full-wrapper build要求 | 13,676 | 全件完了 |
| raw operator／signal | 各13,676 | 全件PASS |
| reference精度評価 | 27,356 | 全件完了・90/128一致 |
| logical compile要求 | 13,676 | 全件完了 |
| unique full-wrapper transpile | 13,006 | 全件完了、alias670 |
| compiled operator／signal | 各13,676 | 全件PASS、aliasも個別照合 |
| finite corrected mean | 4 | 全arm PASS |
| exact weighted resource集計 | 1 | 完了 |
| 認証N・selector／quantum shots | 0 | 対象外・未実行 |

13,676は物理X/Y13,674＋共通prep/readout-only診断2。
各trajectoryの同じevolutionをX/Yで共有し、full IRが同じ場合だけcompileを再利用した。
診断回路費用はcompiler fusionの非加法性を考慮し、主費用から差し引いていない。
数値不一致・技術エラー・未実行の登録対象は0件。これは固定許容差内のlocal数値検証で、厳密な有限ビットcertificateではない。

| 主要残差 | 最大値 | 固定許容差 |
|---|---:|---:|
| raw operator max-entry | 1.887403257234966e-15 | 1e-10 |
| raw X/Y signal | 2.886579864025407e-15 | 2e-10 |
| compiled operator max-entry | 2.999433138796659e-14 | 5e-10 |
| compiled X/Y signal | 7.029099524658022e-15 | 5e-10 |
| finite corrected mean max-entry | 5.1070836584997486e-15 | 2e-10 |

mean残差はSS 8.881788432164979e-16、SP 4.440925979689270e-15、
PS 2.7756356230215864e-15、PP 5.1070836584997486e-15。
meanは検査したraw wrapperのcontrol1 blockを全列挙平均して補正係数を掛け、共通finite targetと照合した。
compiled回路は全caseで独立referenceとのoperator・signalを検査した。phase fittingで救済していない。

## Actual 1-shot資源比較

以下は**両axisそれぞれのfull-wrapper 1 shot当たりの確率重み付き期待値**。
X/Yの全metric期待値はexact有理数として同じであり、axisを平均した表ではない。
各armの正規化済みfinite-bit lawのordered-pair確率を用い、全列挙から求めた。
state preparation X0X1、ancilla操作、controlled evolution、readout、terminal measureを含む。

| arm | 期待RZ | 期待CX | total depth | RZ depth | CX depth |
|---|---:|---:|---:|---:|---:|
| SS | 70.825335 | 32.696703 | 86.953886 | 39.474832 | 28.696200 |
| SP | 45.161578 | 26.354128 | 60.338726 | 27.748491 | 24.387560 |
| PS | 53.950984 | 27.813111 | 69.415121 | 32.228871 | 25.591713 |
| PP | 35.445140 | 23.528787 | 49.627289 | 23.427482 | 22.427796 |

全7 circuit metric（上表＋quantum-only depth・circuit size）で、小さい順は **PP < SP < PS < SS**。
PPはSS比で期待RZが49.9542%、期待CXが28.0393%小さい。
混在SP／PSはSSより安いが、この固定条件でPPより安い1-shot費用は得られなかった。
各caseの整数metricは保存しており、表は小数表示の丸め。正確な有理期待値は要約JSONに保持する。

terminal measureは全caseで1件。total depthはmeasure込み、quantum-only depthの期待値は上表のtotal depthより1小さい。
RZ／CX depthは既存filter定義に従い、RZ／CX／depth／秒を一つの恣意的scoreへ換算していない。

比較上のtradeoffとして、保存lawのlambdaとK0補正係数Btotは次のとおり。
これは今回の既存law生成で得た値で、追加のN評価ではない。

| arm | lambda | Btot |
|---|---:|---:|
| SS | 0.155739 | 1.000242547 |
| SP | 0.203588 | 1.000414480 |
| PS | 0.198515 | 1.000394081 |
| PP | 0.208101 | 1.000433059 |

SSはlambda／Btotが小さい一方で、actual 1-shot回路費用は大きい。
**γ・統計資格が未成立なので、このtradeoffから認証Nや同精度総費用の順位は導かない**。
今回の新しい知見は、このsynthetic fixtureと修正compiler接続でのactual資源順位と、全4armの数値的一致である。
一般的なアルゴリズム優位性、独立新規性、他fixture、分子、長時間、coupling/noise条件へ外挿しない。

## 古典計算費用

32 worker、内側BLAS等1thread、開始時MemAvailable 56,435,294,208 bytes。
CPU／RAM／wallの人工capや共有環境への変更はない。

| 処理 | wall秒 |
|---|---:|
| law／finite-bit setup | 0.004343 |
| common references | 0.851877 |
| build／wrapper references／raw gates | 163.056051 |
| unique compile／logical checks | 33.555846 |
| actual block mean4arm | 0.265436 |
| weighted resource集計 | 0.338819 |

科学pipeline wallは198.141125秒、GNU timeによる起動・保存までのwhole-job wallは205.04秒。
user CPU6147.88秒、system CPU30.72秒、合計6178.60秒、平均CPU3013%。
実transpile wallの全13,006call合計は269.363514秒。並列処理の個別時間合計であり、whole-job経過時間とは異なる。
主process CPU8.266433秒、子process CPU6163.440305秒はpipeline内の計測で、GNU timeの全job値と範囲が異なる。

GNU time最大RSSは461,560 KiB、完了attemptで観測したworker最大410,232 KiB。
いずれも個別processの最大値で、並列aggregate peak RSSではない。aggregate peakは未測定のままにした。
辞書生成は保存B再利用のため再計測しておらず、0費用や償却済みとしない。
これは4armの全件検証を行った古典費用であり、各arm独立のcold-start benchmarkや実機運用費の認定ではない。
比較runを追加していないためspeedup倍率を主張しない。

## 公開結果と保全

[RESULT_SUMMARY.json](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/RESULT_SUMMARY.json)に
各arm／axisのexact期待値、metric範囲、残差、counts、費用、SOURCE・input・compiler・環境版のprovenanceを保持した。
全13,676caseの[compiled資源CSV](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/CASE_RESOURCES.csv.gz)と
[raw資源CSV](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/RAW_CASE_RESOURCES.csv.gz)に整数metricと各caseの残差を保存した。
[F2_laws.json](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/F2_laws.json)に正規化確率、
[common_reference.json](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/common_reference.json)に二精度照合と登録済みphysical数値診断を保存した。
physical診断を厳密bias保証へ読み替えない。

専用raw runは `phase2df2_resource_r1_20261024T034829+0900`。
94,411file／409,179,569 bytesを非公開で保全し、全record hashを確認した。
raw hash台帳SHA256は `3ed8d042319e8021baf4833caaf8165b15444f3d13828ea3b1d4429e61b83ef7`。
承認原文・grant・marker・内部絶対path・認証情報は公開せず、新grantは1件だけ消費した。
旧F2 raw54,892file、旧F2 SOURCE106file、旧F1 SOURCE97file、保護202file、ユーザーworkspace2,110fileの内容が変わっていないことを確認した。
旧失敗結果 `06ce7e5` と旧grant・STOPを保持し、新結果へ上書きしていない。

## 未資格事項とmandatory STOP

η／α／γ／intervalの資格は未成立のまま。NとN加重費用はnull、shot資格・selector・量子測定は0件。
finite-targetの数値的一致を、厳密な有限ビット保証やphysical同精度の総資源優位として扱わない。
この実行に数値FAIL・技術的不具合・登録対象の未検証は残っていない。

**mandatory STOP。次の担当はユーザー／GPT**：4armのactual資源順位とlambda／Btotのtradeoffを評価し、
γ保証の追加研究や認証N加重比較へ進む価値を判断する。
Codexは追加実験、新fixture、physical同精度比較、重要科学レビューへ自動移行しない。
