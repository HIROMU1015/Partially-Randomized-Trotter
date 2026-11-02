# F01とAdmission/STOPの技術契約

F01のcurrent attempt・last completed・candidate trace・committed prefixの意味は固定版のまま。
結果取得前はinflight_event=null、前trial eventを流用しない。emit正常return後だけtrial完了を確定する。
partial write/flush/file-hash failureをrandom ZEROや完了trialへ変換しない。保存不能なhash/costはnull/未観測のまま。
child failure logのcanonical trial_failure_provenanceを保持し、親hostの終了は外側admission contextへ必ず渡す。
childの強制exitでRAM上情報が失われたときは、保存済prefixと異常終了を記録し、未保存eventを復元したとはしない。

## 新schema v2

| 項目 | 旧v1 | 新v2 |
|---|---|---|
| source_roles | science/driver/guard/admission | 4種＋hostが必須 |
| host path | なし | 固定g6_w2c_b3_host.pyだけ |
| entrypoint | なし | path/完全SHA/api=execute_approved_v1を完全一致 |
| production port | v1を受理した旧版 | v2のみ。v1は拒否 |
| artificial v1 | 旧69件のfixture | private人工portだけで互換性維持。新hostはv2のみ |
| runtime import | driver/adapter root | host rootを追加、完全SOURCE bytesも照合 |

追加権限はschemaだけから発生しない。正式契約digest、全input/source/adoption、output、独立実承認、scope/claim/storeを引き続き結合する。
旧candidate編集、approved=true、CLI/envでの承認はない。未pin/別path/別API/別SHA hostは拒否。
[JSON schema](admission_host_v2.schema.json)は外側の必要構造を示し、固定R1値・既存candidate全項目・pin照合はruntime validatorがさらに検査する。

## Atomic reserveからSTOP

host内SOURCE/独立authority preflight → execute/Boundary → 全binding再確認 → atomic reserve → 再確認 → receipt → 既存driver処理。
reserve commit以後はRESERVED/ADMITTED/STOPPEDのすべてがspent。receipt前後、budget/module load相当、output前後、correctness前に失敗しても解除しない。
SQLiteの未commit transactionだけがrollbackされる。committed reservationの解放・自動retry/resumeはない。
通常return/例外の両方で外側contextがSTOPを試みる。STOP保存失敗はprimary例外へsecondary情報を付け、primaryを保つ。
primaryがなければSTOP保存失敗を技術例外として返し、正常完了と偽らない。どちらもrow一意性が再取得を拒否する。

correctnessは予定16cells×16=256、coverage不足で即STOP・追加trialなし・full-Nへ進まない。成功時だけbaseline→instrumentedへ進む既存順序を維持。
全N、現入力のlaw/precision/angle/audit、実費の成立は未検証。実correctnessは0。
保存flushをfsyncや電源断耐性と同一視せず、SQLiteの原子性と科学出力の耐久性を分ける。
