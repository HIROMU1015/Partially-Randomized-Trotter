# Phase 2D-F2 限定資源計測：compile後operator不一致でSTOP

**F2_COMPILED_OPERATOR_FAIL_PARTIAL_RESULTS_MANDATORY_STOP**。
登録条件で一回実行し、raw wrapperは全件通過したが、compile後full-wrapper operatorの照合3件がFAILした。
最大成分誤差は1.9999996023050213で、固定許容差5e-10を大きく超えた。
4armの期待actual資源・finite corrected meanの集計は未完了で、性能順位は判定していない。
修正・retry／resume・追加probeを行わず、取得済み結果と消費済みgrant・STOPを保全した。

## 条件とSOURCE

実行SOURCEは `9b1e103102f918bb157dea90c7eaf3acb5d88b6c`。
独立したF2_RESOURCE契約で、共通finite targetに対するγの厳密保証を前提としない1-shot資源計測を承認した。
旧統計資格付き契約・f2_preflight・f2_statisticsは変更していない。
SOURCE106file、保存B入力7file、固定依存50件、承認原文と新grantのbindingをpreflightで確認した。
source closureは `3d31712f9c9d498809a10fc326c112d73a3985b22f7b6f6c414c8249ac9eda8c`。
管理／mock testは10件PASS。実行前の追加研究回路・科学probeは0件。

対象は4 mode・2粒子・occupied(0,1)・L_D=1・残差density-square 2fragmentの合成Hamiltonian。
geometry／AO basis／分子DF rankは非該当。T=1/5、q1、delta=1/5の一点、r2、h1/10、K0。
state、sector、scalar、D/Rとfaithful identity tail、独立reference90/128digitsを維持した。
compilerはQiskit1.3.0、opt1、seed17、rz/sx/x/cx、topology-free、approximation_degree1、
qubits_initially_zero=False。rtol0・phase fitting禁止とP0閾値を維持し、失敗後に緩和していない。

## 実績とPASS／FAIL／未完了

| 登録単位 | 計画 | 実績・状態 |
|---|---:|---|
| arm law／確率mass検査 | 各4 | 完了・PASS |
| ordered trajectory | 6,837 | 完了 |
| build worker job（共通診断込み） | 6,838 | 6,838完了 |
| full-wrapper build要求 | 13,676 | 全件完了 |
| raw operator／X/Y signal検査 | 各13,676 | 全件PASS |
| reference精度評価 | 27,356 | 全件完了、90/128一致gate通過 |
| logical compile要求 | 13,676 | 計画を登録、全件は未完了 |
| unique full-IR compile候補 | 実行時に確定 | 13,006、alias候補670 |
| 実際のunique transpile | — | 57実行、照合まで54完了、3FAIL |
| compiled logical operator検査 | 13,676 | 60PASS、3FAIL、13,613未評価 |
| compiled logical X/Y signal検査 | 13,676 | 60PASS、残り未評価 |
| finite corrected mean | 4 | compile段階の停止により0、未評価 |
| weighted resource集計 | 1 | 未実行 |
| 認証N／N加重集計・selector | 0 | 対象外、未実行 |
| quantum measurement draws | 0 | 0 |

compile後の3FAILは3unique groupの最初のlogical caseで発生した。
同一groupの未検査aliasを直接FAIL件数へ加算していない。
共通prep/readout-only診断X/Yの2件は13,676に含まれる。
異なる単位を科学call総数として合算しない。

| 主要残差 | 完了範囲の最大値 | 固定許容差／判定 |
|---|---:|---|
| raw operator max-entry | 1.887403257234966e-15 | 1e-10／PASS |
| raw X/Y signal | 2.886579864025407e-15 | 2e-10／PASS |
| raw unitarity max-entry | 3.885780586188048e-15 | 1e-10／PASS |
| compiled operator、60PASS件 | 8.034838406612828e-15 | 5e-10／完了部分のみPASS |
| compiled X/Y signal、60件 | 1.3114509478384662e-15 | 5e-10／完了部分のみPASS |
| compiled operator、3FAIL件 | 最大1.9999996023050213 | 5e-10／FAIL |

共通finite referenceの90/128差は約2.35e-91で一致gateを通過した。
これはactual corrected meanとfinite targetの一致を検証した結果ではない。mean4件は未実行である。

## 不一致と限定される解釈

| 直接FAILしたlogical case | max-entry | Frobenius | spectral |
|---|---:|---:|---:|
| SS_000_008_X | 1.9999996023050213 | 11.313708498984761 | 2.000000000000001 |
| SP_052_013_Y | 1.437966534951478 | 11.313708498984761 | 2.0000000000000004 |
| SP_055_040_Y | 1.4379665349514785 | 11.313708498984761 | 2.000000000000001 |

SSの該当IR groupにはSS_007_008_X、SS_008_000_X、SS_008_007_Xも含まれるが、それらは未照合である。
raw側は同じcaseも通過し、compile後の厳密な位相を保持するoperator gateで不一致を検出した。
保存済みnormだけから原因をglobal phase、compiler、QPY、アルゴリズムのいずれかへ断定しない。
位相fitで救済せず、失敗後の追加operator生成・数値probeも実行していない。
一般的な混在アルゴリズムの破綻や優位性の結論へ広げず、登録されたcompiled経路のFAILとして扱う。

## SS／SP／PS／PPのactual資源取得状況

| arm | L | 保存済みraw wrapper | compiled資源・照合PASSの保存件数 | 確率重み付き期待actual費用 |
|---|---:|---:|---:|---|
| SS | 14 | 392 | 1 | 未取得 |
| SP | 58 | 6,728 | 31 | 未取得 |
| PS | 26 | 1,352 | 6 | 未取得 |
| PP | 51 | 5,202 | 22 | 未取得 |

上表に共通raw診断2件を加えraw合計13,676。
各caseのraw RZ／CX／depthは全件保存済みで、compiled資源は60PASS件の整数metricsを公開する。
未評価caseの費用を0で補完せず、不完全なcase集合を再正規化して4armの順位を作っていない。
停止後にraw期待費用へ主比較を置き換えたり、未実行の集計を再開したりしていない。
したがってSS/SP/PS/PPの性能差・tradeoffの主比較は未完了である。

## 古典費用

32 worker、worker内BLAS等1threadで並列化した。起動時MemAvailableは56,498,376,704 bytes。
GPU driverは利用不可。CPU／RAM／wallの人工capや共有環境への変更は行っていない。

- 全job wall：**183.79秒**。GNU timeによるuser CPU5531.28秒、system CPU26.25秒、合計**5557.53秒**。
- 平均CPU使用率3023%。比較実行をしていないためspeedup倍率は主張しない。
- GNU timeの最大RSSは477,604 KiB（個別processの最大値、並列jobの合算peakではない）。完了attemptで観測したworker最大409,560 KiB。
- 全build完了attemptのCPU合計4798.119秒、wall合計4875.460秒。wall合計は並列job全体の経過時間ではない。
- 完了54unique compile attemptのCPU小計3.363秒、実transpile wall小計0.896秒。失敗3件のtranspile個別費用は未記録で、0扱いしない。
- 辞書生成は保存B再利用で再計測していない。生成費0・償却済みとは扱わない。

測定できなかった並列aggregate peak RSSを推測値で埋めていない。
これらは今回の失敗までの実費用であり、13,676 logical compileが完了する費用の見積りではない。

## 証拠・provenance・保全

公開結果は[RESULT_SUMMARY.json](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24/RESULT_SUMMARY.json)。
全raw caseの資源・残差は[RAW_CASE_RESOURCES.csv.gz](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24/RAW_CASE_RESOURCES.csv.gz)、
compiled60PASS件は[CASE_RESOURCES.csv.gz](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24/CASE_RESOURCES.csv.gz)。
arm・axis・i/j・資源metric・operator／signal残差をcase IDで追跡できる。
[F2_laws.json](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24/F2_laws.json)は保存した正規化law、
[common_reference.json](../../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24/common_reference.json)は二精度照合の記録。
共通physical biasは登録済み診断1件で、厳密保証・主評価には使わない。

raw原本は専用run `phase2df2_resource_20261024T032200+0900` 内の54,892file／319,938,150 bytesを非公開で保全した。
QPY、reference配列、per-case JSON、attempt journal、input snapshot、消費済みgrant・marker・STOPを保持し、全record hashを照合した。
raw hash台帳のSHA256は `6bef3ee6b28ae0127d800b4139befdec7cc040ad4d8950bbd880f0678ca4b040`。
公開した要約にはSOURCE・入力hash・compiler・閾値・環境版・公開CSV hashも保存した。
承認原文・grant本体・認証情報・内部絶対pathは公開していない。
旧B/F1/PRE/資格確認の保護176file、旧F1 raw2,299file、ユーザーworkspace2,110fileに内容変更がないことを確認した。
実行SOURCE106fileも固定hashのまま。新grantは1件消費、別attempt・retry／resumeは0件。

## 結論の制約と次の担当

今回得た新しい証拠は、F2/K0の全raw trajectory／Hadamard wrapperが独立referenceと数値的に一致すること、
その同じ条件のcompiled全operator経路には固定gateを満たさないcaseがあること、および保存できた部分資源・費用である。
同じsynthetic fixtureに限るlocal結果で、CI・外部再現・一般的なアルゴリズム優位・独立新規性の認定ではない。
η・α・γ・intervalの未資格はそのままで、認証N・N加重費用・同精度総資源優位は一切主張しない。

**mandatory STOP。次の担当はユーザー／GPT**：今回のoperator不一致を踏まえ、原因調査・修正と別途新規実行を承認するかを判断する。
γ保証の追加研究や認証N加重比較の価値も、その後の判断事項である。
Codexは原因を確定する新しい科学probe、SOURCE修正、再実行、追加実験・重要科学レビューへ自動移行しない。
