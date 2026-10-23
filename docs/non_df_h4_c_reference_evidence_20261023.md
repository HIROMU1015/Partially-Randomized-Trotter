# H4 C修正版：固定40軸の数値証拠候補生成

2026-10-23 JST。利用者の「次の作業に進んで大丈夫」を、直前に提示した修正版SOURCEでの固定40軸実行への承認として受領した。承認原文と担当者のscope解釈は[approval.json](../artifacts/non_df_h4_c_reference_evidence/2026-10-23/launch01/approval.json)で区別する。

実行branchは `codex/non-df-h4-c-reference-evidence-20261023`、基点 `450b0a787f376a7a72b5d944685d102aac911ae4`、科学実装SOURCEは **`e80d7510880f8db993b777c627bb8e95b05a070d`**。実装・runtimeは修正専用worktreeの封印pathに保持し、今回の結果用worktreeへ編集を分離した。封印済みコードの変更は0。

## 実行状態（終了監査前）

18:53 JSTに新grantで一回起動した。Pauliの20軸は64 bitでTARGET_METを保存、worker native exit0。DFはPR implemented-lawの計算を継続中で、最終20軸・全体STOP・終了監査は未確定。この段落は最終保存結果の確認後に更新する。

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
