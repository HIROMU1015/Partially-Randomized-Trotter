# H4 C修正版：固定40軸の数値証拠候補生成

2026-10-23 JST。利用者の「次の作業に進んで大丈夫」を、直前に提示した修正版SOURCEでの固定40軸実行への承認として受領した。承認原文と担当者のscope解釈は[approval.json](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/launch01/approval.json)で区別する。

実行branchは `codex/non-df-h4-c-reference-evidence-20261023`、基点 `450b0a787f376a7a72b5d944685d102aac911ae4`、科学実装SOURCEは **`e80d7510880f8db993b777c627bb8e95b05a070d`**。実装・runtimeは修正専用worktreeの封印pathに保持し、今回の結果用worktreeへ編集を分離した。封印済みコードの変更は0。

## 実行結果・終了監査

18:53:52 JSTに新grantを一回消費し、**19:15:15 JSTに固定40軸すべてを64 bitでTARGET_METとして保存し終了した**。所要時間は約21分24秒。96/128 bitへの昇格・科学retry・追加grantは0。runのSTOPは `C_CANDIDATE_EVIDENCE_COMPLETE_PENDING_SCIENTIFIC_ACCEPTANCE`。数値設計目標を満たす候補証拠であり、正式certificateの受理は0軸である。

| 方式 | 固定5精度×cosine/sine | precision | 保存結果 |
|---|---:|---:|---|
| pauli_jw_PF | 10軸 | 64 bit | TARGET_MET 10/10 |
| pauli_jw_PR | 10軸 | 64 bit | TARGET_MET 10/10 |
| df_native_PF | 10軸 | 64 bit | TARGET_MET 10/10 |
| df_native_PR | 10軸 | 64 bit | TARGET_MET 10/10 |

[全40軸CSV](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/axis_outcomes.csv)・[JSON](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/axis_outcomes.json)は、saved resultの精度・軸・target・bias distance・error上端・判定を転記した。新しい科学値・shot/Gを計算していない。[終了監査](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/completion_audit.json)。

今回の待ち時間の大部分はDF側で生じた。DFのnormalization call予約からimplemented PR mean予約まで約17分27秒あり、source上では324 componentの基底変換・rotation構成を含む区間に対応する。これは既存ledgerの時刻とsourceの対応であり、関数別の実測profilingではない。計算中のSOURCE・runtime・契約を変更せず、CPU競合や待ち時間を理由とする中断・再実行は行っていない。

## 保存normalization・残るcertificate gate

両表現のPFは保存結果でB=1/log_B=0の決定論的構造を記録した。PRはPauli/DFともに `saved_log_B_upper_encloses=true` を保存し、有限係数から得たB/log_Bの区間と保存上界を照合した。[保存normalization](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/normalization_saved.json)は原結果の転記である。

| 保存gate | Pauli PR | DF PR |
|---|---|---|
| saved log_B上界の包含 | true | true |
| exact ±saved B event範囲 | true | false |
| PCG64独立性の証明 | false | false |
| formal certificate受理 | false | false |

literal compiled wrapperへの転送も未certifyであり、数値目標達成だけで旧shot gateを通過したとはしない。旧20精度行N/G=null・40軸UNDETERMINEDは変更していない。

## 今回のnative終了とcall予算

| process | native exit | wait/reap・witness STOP |
|---|---:|---|
| Pauli preparation bridge | 0 | 保存・照合済み |
| DF preparation bridge | 0 | 保存・照合済み |
| Pauli worker | 0 | 保存・照合済み |
| DF worker | 0 | 保存・照合済み |
| supervisor | 0 | outer witnessで保存・照合済み |

[native監査](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/native_audit.json)は各claimのpidとworker identity、native wait receipt、exit、回収、STOPを対応させた。全witnessでall_owned_reaped=true、異常な子孫停止はなく、同一pid/start_ticksの残存0。これは今回の新runだけの終了証拠であり、元H4の監視欠落を補完しない。

| call区分 | 予約数 | 固定最大 |
|---|---:|---:|
| Operator/preparation | 2 | 2 |
| Probability mapping | 2 | 2 |
| Reference | 2 | 6 |
| Nominal mean | 4 | 12 |
| Implemented PR mean | 2 | 6 |
| PR normalization | 2 | 6 |
| Dense exponential | 13 | 39 |

予約はcall前の非返却記録であり、予約数を完成証拠件数へ読み替えない。今回の完成結果は別途40軸・最終tier/result・native receiptで照合した。新SCF/FCI/DF生成/状態最適化/sampling/RNG replay/compile/shot/G会計・科学retryは0。

## 固定条件と承認境界

linear H4/隣接1.00 Å/STO-3G、8 system qubit、保存共通状態Nα=Nβ=2。Pauli-JW184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body、S2、T=.8/q8/r4/K4、ε_sig=.05/.01/.005/.001/.0005のcosine/sine計40軸。DFの保存準備はPF11 block、PR2 block・tail324 component/9 basisである。Bの必要u上限/2を設計目標とし、候補・allowance・比較閾値を変更していない。

科学callの最大予算はop2、probability2、reference6、nominal12、implemented PR6、normalization6、dense exp39。precisionは64/96/128 bit、有限WIDTH_INSUFFICIENTだけを昇格させ、完成caseは再計算しない。SCF/FCI/DF生成/状態最適化/sampling/RNG replay/compile/shot/Gはすべて0の契約を維持した。

元H4科学SOURCEは `0c329b30a81accb35c212be8defa66fee0dd0f12`、B SOURCE/結果は `2bd212288e8530e519c4a8135257df37f7608662` / `b0f94185f93cd5614e8f183613a2a9313a5b9d4f`。実行準備時に修正SOURCEのcode19/private runtime1,050/旧source738/旧runtime6,999/input16ファイルを照合し、人工reference14件も再確認した。科学条件の差分は0。

## 新grant・監視構成

新grantは `6862c863f376be24c4ab56bc2d21abd2f8c2560b80250f95d4a1e6fd8b0cb4b4`。前回の失敗runのgrantを再利用せず、今回専用の未使用namespace `artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01` に発行した。`CONSUMED.json` と `LAUNCH_CLAIM.json` を一回だけ保存した。旧inactive契約・SOURCE・過去STOPは変更していない。

新しい[active契約](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/launch01/active_contract.json)は承認とnamespaceだけを変更し、[契約差分](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/launch01/contract_delta.json)で照合した。発行準備・起動wrapperのcommitは `3e4cb7b3f2c430c1326397a2a5e25f790743d776`。科学実装SOURCEとは別の来歴である。

Pauli/DFを独立workerで並行実行し、各worker・各旧runtime bridge・supervisorをnative witnessで監視する。witnessはnative wait/reapとSTOPをfsyncで保存する。CPU/RAM/workerへの新しい恣意的上限や共有設定変更は加えず、一時的なPSI上昇・他chatとのCPU競合で中断しない。

## 証拠の扱いと保全

科学設計目標のTARGET_METと正式certificate受理は分離する。保存した数値結果から新しいshot/Gを計算せず、経験的な旧biasやallowanceをCERTIFIEDへ格上げしない。DFのexact event範囲、literal compiled wrapperへの転送、IID uniform-word仮定/PCG64独立性、科学的受理の制約は別に残る。

元H4の監視消失原因UNKNOWN・native終了未証明、旧20精度行N/G=null・40軸UNDETERMINEDは保持する。旧SOURCE49c18ccc3の失敗C run・不受理Pauli20軸raw・DF最終20軸未取得・消費済みgrant・FAILED_NO_RETRY・execution holdも履歴として残す。今回の新結果とnative記録で旧runの証拠欠落を埋めない。

今回の保存結果・監査は主担当Codex自身の検査であり、別担当の独立科学レビュー、正式certificate受理、最終evidence freezeではない。公開pushとN/G再会計は別の承認対象。

[保全確認](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/launch01/preservation_after.json)では既知の2,239ファイルがSHA一致した。元H4/B/原因調査/旧C失敗runと修正技術packageを元の絶対pathで検証し、元commit6892e95のprecision table20行もbyte一致・N/G欄空値・accounting_usable=falseを確認した。CSV空値を0や適格判定へ読み替えていない。

## 次の作業・保存状態

**ユーザー**：数値候補の保存結果と残るDF event範囲・compiled wrapper転送・統計仮定を確認し、次の証拠検証・科学的受理の範囲を判断する。N/G再会計・公開push・詳細科学レビューは別承認。

**Codex**：その承認後に指定された証拠検証だけを行う。今回のgrantは消費・正常STOP済みであり、再発行・再利用・resume/retryを行わない。

branch `codex/non-df-h4-c-reference-evidence-20261023` に結果と終了監査をlocal commitする。科学SOURCE e80d75108と結果保存commitは区別する。公開pushなし。**mandatory STOP。新しい数値解析・科学計算・certificate受理・N/G会計へ自動移行しない。**
