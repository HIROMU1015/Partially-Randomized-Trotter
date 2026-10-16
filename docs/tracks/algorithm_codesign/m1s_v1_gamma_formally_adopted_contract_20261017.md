# M1-S V1 γ依存検証：正式契約・採択記録

2026-10-17 JST。`FORMALLY_ADOPTED / SCIENTIFIC_EXECUTION_NOT_AUTHORIZED / MANDATORY_STOP`。

## 採択の根拠

ユーザーが候補commit `fcd1c464fae48743e829fbe765a38141304f879d`を明示して、直前の推奨条件を含め正式採択し、採択記録と契約Cの固定・公開を指示した。同じ発言で「科学計算は承認しません」と明示している。発言全文と採択の文脈を、`artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1/user_adoption_record.json`に保存した。

採択対象の[候補原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcd1c464fae48743e829fbe765a38141304f879d/docs/tracks/algorithm_codesign/m1s_v1_gamma_dependent_contract_candidate_20261017.md)、[技術根拠](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcd1c464fae48743e829fbe765a38141304f879d/docs/tracks/algorithm_codesign/m1s_v1_gamma_precision_resource_and_handoff_20261017.md)、受領GPT原本は保持する。過去の「未採択」表記を遡って変更せず、本記録を新しい正式状態とする。

正式共通運用指示は`docs/research/research_project_instructions_unified_2026-10-16.md`、SHA256 `8323be3f65856adf0f4d095d8e68546db3326822023c20e101ba5f596605ebfa`。今回追加のGPT詳細科学レビューは行っていない。

## 採択済み条件

|項目|正式条件|
|---|---|
|Task|Aを先行：保存raw-H理想信号の単一時刻・円周位相の事後的記述診断。Bの多round energy/RPEは将来目標であり本契約の実行対象外|
|γ|固定値／gridを選ばず`0<gamma<pi/2`全域の成立条件・境界を扱う|
|route|無補正raw直接route。モデル差を含む総raw-H biasへetaを二重加算しない|
|confidence|`PER_METHOD_ROW`。α=`0.05`を正確な有限小数文字列として固定。X/Y等配分。各方式行で95%の統計的十分条件であり72行同時保証ではない|
|統計policy|`FRESH_IID_HOEFFDING_EQUAL_AXES`。fresh-IIDのlabel・±1測定outcomeと理想信号に一致するmeanを仮定|
|accuracy|`HIGH_PROBABILITY_CIRCULAR_PHASE_ERROR`。energy精度・RMSE・principal実数差のunwrap保証ではない|
|coverage|`SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY`。保存数値半径の条件付きengineering coverageを採用。保存誤差を統計的αへ足し込まない|
|比較|`GAMMA_CONTINUOUS_SUFFICIENT_FACTOR_WITH_CEILING_CAVEAT`。同じ十分設計を比較し、最適shot数を主張しない|
|算術|共通p=128 bits。p+8のdyadic quantumで外向き区間算術。結果依存のprecision変更・retryなし|
|判定|`DESCRIPTIVE_ONLY_NO_SCIENTIFIC_GO`。自動GO/NO-GO、cellランキング、科学的優位性の認定をしない|
|CPU/wall|数値worker CPU soft cap160秒、hard cap161秒（cleanup猶予1秒）。親supervisorのworker wall cap320秒。完了時間の保証ではない|
|その他resource|数値worker1／GPUなし、metadata supervisor、aggregate virtual memory2 GiB、入力1 MiB、result/log各16 MiB、rational65536 bits、series4096項、bound step100万、BLAS等thread1|
|one-shot/STOP|将来別承認の一回解析。起動後の失敗も消費、retry・閾値調整なし。全行・対照のprefix／未評価placeholderと失敗情報を保全し、完了／失敗いずれもmandatory STOP|

α=0.05は採択された探索的な保証水準であり、研究上唯一の正解でも、観測済み結果から導かれた値でもない。上記のconfidenceは仮定された測定分布の統計保証であり、保存半径・platformを95%で認証したという意味ではない。

## 対象・SOURCE・入力の固定

Hamiltonianはlinear H4 1.00 Å／STO-3G／8 modes／保存DF rank12、G3/G6/G9。保存stateの元norm²商と`+iTH`を使用する。既存δ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1、shared_pm／ordinary／independent_pmを保持する。全24非零cellの72方式行、36保存zero control、計108行と48対照を報告する。新分割・新時刻・新γ点を追加しない。

SOURCE Sは`f2a5611ca9fc7461c64f2deb74a1f624ac8aa7c5`。`m1s_saved_signal_gamma_v1`の7 module、runner、結果schemaの9 fileは変更せず、採択machine契約へ同じSHA256 inventoryを引き継ぐ。

入力はcommit `025c5a90888b02b73d2815fbb10d6bd1d2823602`の`artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json`のみ。454422 bytes、SHA256 `ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`。今回の採択作業ではbytes/hashだけを確認し、科学数値をJSONとして開いて再解析しない。

新しい正式machine契約は`artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1/execution_contract_adopted.json`。元候補を保持し、status・採択flag・8科学field・CPU/wallの2 fieldだけを指定どおり変更した。JSONの`base_commit=be2da2c3...`はSOURCE設計基点のまま維持し、今回の候補採択基点fcd1c464は別recordに記録する。

## 数学的対象・比較・failure分類

採択した数式は候補と同じ`M_m(gamma)=A_m^- sin(gamma-b_m^+)`、`A_m^-=max(0,A_hat-r_A)`、`b_m^+=min(pi,abs(b_hat)+r_b)`。共通領域は`max(b_sh,b_cmp)<gamma<pi/2`かつ両振幅下界が正の場合だけ。

レビューのtrade-off仮定を検査し、X/Yの係数区間とarctanによる境界を扱う。逆向きはラベルを入れ替えた同一定理。同bias・dominanceでは内点境界式の適用外を明記する。領域なし、証拠不足、算術未解像、root存在と位置未解像、TECHNICAL_STOPを区別する。未解像を一致や成功へ読み替えない。

全方式の共通十分設計は`N_axis(gamma)=ceil(K/M_m(gamma)^2)`、`K=4 log(2/alpha_axis)`。今回のrow policyではα_axis=α/2という配分規則を採択するが、今回K・境界を数値評価しない。連続marginがstrictに大きくても天井関数による整数同数があり得る。本scopeではK区間・γ境界・成立条件と記号的な整数strict条件を扱い、**絶対shot数・整数削減率・native費用・RPE総資源は算出しない**。

旧G9の`UNRESOLVED_ABSOLUTE_CONTRAST`を保持し、相対因子や境界から物理的有意差へ格上げしない。保存oracleを認証済み厳密固有stateと扱わない。旧v5の未保存full-H/HD直接一致はUNDETERMINEDのまま。

## 採択と実施承認の分離

`adopted=true`と`coverage_model_adopted=true`は、ユーザーが**条件付きengineering評価を採用したこと**を表す。IEEE/BLAS/trig/underflow等の未検証仮定が解消されたことや、platform certificate取得を表さない。証拠区分は`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。

`contract_grants_execution=false`を保持する。実施承認、execution authorization、grant、markerは発行しない。SOURCEの既定pathは旧候補のままであり、正式契約を使う将来の呼出しには新しい`--contract` pathの指定が必要。今回`status`／契約metadata検査だけを行い、runは行わない。

契約payload Cの公開後、receiptだけの公開HEAD Hを追加する。将来はユーザーが契約bytesを含むCまたはHを実行基点として選び、その直接のauthority-only子Aを別途固定する必要がある。既存guardはAの親を契約固定commitとして検証し、SOURCE Sより後で同じ契約bytesを持つことを確認する。**今回Aを作成しない**。

旧SOURCE・候補・結果・authorization・ledger・消費済markerと親／他worktreeは保全する。既存の数値testsやbenchmarkを今回再実行しておらず、登録H4のV1 correctness・runtimeは未検証。

## 今回の完了範囲

採択記録、正式machine契約、SOURCE/input identity・許可差分・guard拒否の非数値検査、保全監査、GitHub公開bytes/SHA256確認まで。新科学計算・保存H4再解析・γ境界・shot forecast・数値correctness test・execution authorization/grant/markerは全て0。

**次はユーザーによる、固定SOURCEと正式契約に基づくV1 one-shot解析の別途明示的な実施承認。今回の契約固定完了後はmandatory STOP。**

## 採択監査・公開証拠

今回の非数値契約検査109 PASS／0 FAIL、保全監査8 PASS／0 FAIL。元候補との差分は採択flag/status、8つのscience field、CPU/wallの2 fieldだけ。SOURCE／入力／モデル・grid・式・classification・one-shotは不変。既存7,656非index file、親1284 tracked file・index・HEAD、全206既存worktree HEADを保全した。

`artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1/`の`static_validation.json`、`preservation_audit.json`、`evidence_index.json`、公開後の`publication_receipt.json`が証拠。静的検査PASSは登録H4の数値correctnessやruntime検証ではない。採択payloadを契約Cとして先に固定し、receiptは別commitで公開する。
