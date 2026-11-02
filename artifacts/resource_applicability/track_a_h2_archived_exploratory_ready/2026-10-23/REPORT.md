# H2 4方式：保存値による探索的総費用比較の実行準備

**Phase 1・2が完了し、Phase 3の科学実行承認待ち。** 元PR134軸＋共通Full4軸と、受入済みDIAGNOSTICSの4候補の保存bias/reference/log_B/allowance入力を結合し、既存数値処理を呼ぶ別SOURCE・別契約を封印した。実H2のu/N/G再会計・rankingはまだ0。旧strict COMPAREはBLOCKEDのまま。

準備開始2026-10-23、終了監査2026-10-24 JST、GPUA100-gleap02。campaign名・branch・artifactの20261023表記は維持し、最終private実行準備領域は20261024で分離。専用branch `codex/h2-archived-exploratory-ready-20261023`、基準 `041511aa567c288bcead32d0bf7579fd7f434e44`。Codex主担当自身の技術検査であり、独立科学レビューではない。

## 承認対象となる固定事項

- 契約：`H2_FOUR_METHOD_ARCHIVED_COST_EXPLORATORY_V1`。
- 実行SOURCE：`aca689060d302b4e1f37d32d7075c9266b637f56`。
- 契約canonical SHA256：`a71201e23738f5b7117213fb94dfe265d307ca4f194de2ef0defc55a950d6e23`。
- input manifest file SHA256：`e30c4de7ef9a48fd04696367bb2dda68ba8829cc4ba8026598af0d337775fffe`。
- [契約案](execution_contract_DRAFT.json)、[SOURCEとfile SHA](SOURCE.json)、[固定入力manifest](fixed_input_manifest.json)、[preflight](metadata_preflight.json)。

4候補すべてで保存された会計入力が揃い、schema・元入力binding・共通reference・prefix preparation・保存費用のSOURCEを検査できた。新しいallowance、headroom、shot、G、順位を実データへ適用して適格性を判定したわけではない。preflightは`READY_AWAITING_EXPLICIT_SCIENTIFIC_APPROVAL`、technical_ready=true、execution_authorized=false。

## 固定対象と保存入力

Linear H2/1.0 Å/STO-3G、tol-only DF1e-8/rank3、T=.8、ε_signal=.0005、α_axis=.025、P=0、system4/ancilla4、usual signed bias。元RTE law/signed schedule/scalar/control、empirical allowance・旧reference条件付き再利用・shot配分を維持する。

| 方法 | 固定候補・条件 | 費用入力 |
|---|---|---|
| Full S2 | B1_p3_q4_2nd、prefix3/q4 | 保存H2_FULL_3_4_2nd、cosine/sine各1655 compiled RZ |
| Full Yoshida4 | B1_p3_q2_4th、prefix3/q2 | 保存H2_FULL_3_2_4th、各2463 compiled RZ |
| PR S2 | H2O32_c7ec590a32a859bc、prefix1/q4/K4/outer32 | EXPLORATION33 trajectory/66軸 |
| PR Yoshida4 | H2O32_8a2ecdd0294ffb74、prefix1/q2/K4/outer32 | EXPLORATION34 trajectory/68軸 |

費用はB固定commit `ca29dd9d…` の既存2表を元byteのままprivate準備領域へ置き、固定Git全文とSHAで封印。B-2の138軸identity検証結果を継承し、QPY/archive/原PR eventの監査をやり直していない。旧Full1367/2033 RZ、別cohort、追加候補、未観測eventの0補完は拒否する。Full原本byteの確認者はGPU01であり、今回GPU02が原QPYを取得・検査したとはしない。

保存signal/bias等は元DIAGNOSTICS SOURCE `35444654…`、契約 `f71e52ce…`、completion `cc7ebfee…` に固定。既存`accepted_diagnostics_receipt`と`adaptive`で完了・ledger・result SHA・定義を確認し、必要なJSON fieldだけを読む。4つのNPZは既存completionに従うhash検査のみで、配列decodeやvector再評価は0。Hamiltonian/state/DF/時間/candidateの基本bindingは停止条件として維持する。

## 数値実装の再利用と必要最小限のadapter

[正確な関数・SOURCE・SHA・line](reuse_inventory.json)。既存COMPARE SOURCE `46a3b33e…` の`Backend.allowance/account/rank`を変更せず呼ぶ。元Backendのconstructor/setup/prepare/nativeは呼ばず、saved JSON入力と数学関数を結ぶ最小proxyだけを作る。元メソッドのreceiptは新run内`delegated_receipts/results`へ保存し、外側の探索的schemaと旧数値実装のseriesを区別する。

allowanceは`h6_matched_accounting_v1.empirical_allowance`、shotは`u_aware_shots → axis_headroom`、有限標本統計は同moduleの`resource_row/metric_summary`。名前にH6を含むが、使用関数は系固有のHamiltonian計算を含まず、固定axis整数費用とshot/allowance recordに作用する既存の汎用算術である。SOURCEと入出力を確認して利用した。数式・reference・normalization・headroom guard・統計規則を新規作成していない。

元COMPAREのGと演算順を主値に保つ。axis別Gは元`account`内の各項を表示するadapterであり、同じ`N_axis × sum(C_axis)/n`を用いる。`resource_row`は元どおり平均を先に求めるため、補助G_pointとのbinary64丸め差を同一bitの必須条件にしない。rankingには元COMPAREのGを渡す。

PRの費用標本をcosine/sineで独立に組み直さず、task identityで対にする。Cの標本平均・SD/SEと、対になったtrajectory総費用のG標準誤差を既存関数で算出する設計。n=33/34の有限標本であり、母平均・未観測tail費用・選択後coverageを保証しない。今回再利用した関数に名目区間の定義はないため、intervalはnullとして新しい正規近似等を加えない。Fullはdeterministicな1組で、標本SD/SEを0で捏造しない。

## exact call予算

| 外側call | 回数 | 内部call |
|---|---:|---|
| bootstrap | 1 | 純粋算術moduleとadapterの接続のみ |
| bundle_load | 1 | 保存JSON/cohortの読込、vector decodeなし |
| allowance | 4 | empirical_allowance4 |
| accounting | 4 | u_aware_shots4、axis_headroom8 |
| cost_statistics | 4 | resource_row4、metric_summary8 |
| ranking | 1 | 元のFull S2/Y4・PR S2/Y4比較、causal attributionなし |

合計は外側15＋内側28＝**43 ledger event**。[完全call表](call_graph.json)。不適格時はresource_rowがNoneを返し、adapterがC-onlyのmetric_summaryを2回呼ぶため、適格/不適格どちらでもmetric_summaryは計8回。人工fixtureで両分岐を検査済み。math/statistics内部の算術はそのkernel内の処理で、別の量子回路・simulation callを隠していない。

sampling/circuit_build/lowering/compile/transpile/simulate/signal/reference_recompute/prepare/setup/primitive_evaluationは**すべて0**。今回の実消費は上表も含め全0。保存値からの新u/N/G/費用統計・rankingも、別途明示承認前には実行しない。

## runtime・authority・出力・停止

[runtime封印](runtime_seal.json)：数値計算runtimeは元COMPAREと同じ選択Python3.11 binary `ebc2fae9…` とstdlib算術で、保存compiled費用の歴史runtimeはPython3.12と明記する。新compileの出力同等性を主張しない。stdlib/ELF944 file、SOURCE16 file、必要な保存入力等を含む996 fileを封印。isolated `-I -S -B`、固定import path・source byte検査・実行可能mapping検査を用い、Qiskit/NumPy等を起動しない。共有venv/OS/schedulerの変更0、恣意的な低いCPU/RAM上限は新設しない。

[authority template](authority_templates.json)はINVALIDで、有効grant発行0。別途ユーザー承認を契約digestと限界一覧に結び、初めて新namespaceのUNUSED one-shot grantを1件発行する。未来の独立run_rootは未作成。元grantのID/purpose/budget/契約の流用と既存root再使用は拒否し、開始時のCONSUMED markerとcall ledgerは返却・再試行しない。

[出力schema](output_schema_DRAFT.json)は4方式のC統計、analytic shot、u、axis/total G、PRのpaired G標準誤差、元ranking結果と限界を保存する。算術上不適格な候補はG=nullとC-onlyを残し、欠測を0で補完しない。元rankが4方式すべて適格でないと判断した場合は不完全比較のままとする。必須保存入力の欠測やSOURCE/cohort/費用取り違えは事前/実行中STOPであり、推測値による継続はない。

終了時は元Sessionのmarker/SOURCE/loaded-closure検査を再利用し、disk上の43 ledgerと予算・memory消費を照合。全method・delegated receipt・comparisonのSHA index、completion、mandatory STOPを保存する。[受入・停止条件](acceptance_and_STOP.json)。失敗時もpartial/marker/ledgerを保持し、科学retry/resume・予算追加・別候補拡張をしない。

## 探索の限界と旧契約の保全

今回のユーザー指示により、Full全parameter stream、完全historical load、provider差の完全動的非干渉、Python3.12/3.11出力同値性、A独立6 primitive評価は**この探索契約の開始条件から除外**した。B-2/Aの未解決状態は変更せず、PASSにしていない。保存single-state correctnessをall-state/operator保証へ昇格せず、経験的allowanceをcertificateにしない。

旧strict COMPARE契約は変更・実行しない。探索結果は別namespaceの条件付き比較で、一般的PR/PF優位性・完全共通native実証・母平均保証・正式採択結果として扱わない。詳細科学レビュー・最終evidence freezeも開始しない。

## 人工テスト・保全・次の担当

[封印runtimeの人工smoke](sealed_runtime_artificial_smoke.json)でも、実保存値を数値backendへ渡さず、実import seal・loaded closure・43 eventのdisk ledger・mapping確認がPASS。本番run_rootと有効grantは作成していない。

[人工テスト](test_results.json)は最終選択Python3.11で新32＋既存COMPARE40＋production36＝108件。人工値だけでu/N/Gを評価し、実H2保存入力は数値kernelへ渡していない。SOURCE/input/runtime SHA変更、物理条件/candidate/cohort/旧Full費用混入、予算超過、旧grant/root再利用、未承認科学入口、限界削除、C-onlyとpaired SE、終了ledger破損を検査した。最初の人工testで検出したdelegated receipt directory不足は修正済み。初期metadata sealは`attempts/initial_metadata_seal`と旧private準備領域に保存し、終了監査を補強した上記最終SOURCEで新しく封印した。科学失敗・科学再試行ではない。

[保全監査](preservation_audit.json)はB-2の既知保護対象索引を流用した前後SHA比較であり、QPY identityの再監査ではない。旧SOURCE/結果/grant/ledger/marker/STOPと既存manifest entryは不変。[最終照合](final_verification.json)。

**次：ユーザーが上記契約・SOURCE/input・43 eventのexact予算・限界を伴う探索的会計の一回実行を別途明示承認する。** 担当Codexはその承認後にだけ新しいUNUSED grantを作成し、同SOURCEでone-shot実行・終了監査・STOPする。現在は添付指示§8 Phase3の承認ゲートでmandatory STOP。追加identity調査・A6評価・compile・samplingへ戻らない。公開承認は未確認のためpushしない。
