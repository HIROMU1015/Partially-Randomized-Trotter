# D7較正：計算完了と公開までの経緯（公開完了）

現在は結果pushと公開後bindingが完了した。下記は公開前の状態を保全した履歴で、最終の追記が現状である。

固定SOURCE `8cce540645b99e515d72888b817782a1b9f97334` の再開jobは正常終了した。
全5 cell・9 path・18軸の最終検査はPASS、FAIL/UNKNOWN/価格未取得は0。
18 unique実compileは元d323で16・887で2。今回の再開はそのうちG36223の4 compileを再利用し、
新sampling/compileは0。既存4 cellの14軸は元の公開検査結果を維持する。

G36223 path0の保存event/seed/順序・scalar/phase・parity・X/Y wrapper/fingerprint照合が完了。
元native index0の7数値gateとcompileは再実行せず、元PASS receiptを照合して再利用した。
path1の完了receiptもそのまま再利用した。X/Yの観測RZはpath0が各5,552,901、path1が各5,572,406。
全値は固定有限cohortでの観測であり、law E[C]やモデル資格を意味しない。

親processの終了codeは0。子processもreturncode 0、signalなし。
全job wall21:39.15、peakRSS9,530,444 KiB。子観測wall1173.696940秒。
段階log・終了記録・資源は [recovery01](recovery01/) に保存した。
元887/0dcのSOURCE・STOP・raw/QPYは保全した。

local公開準備とlossless復号検査は完了している。

- raw：2,336,436,443 bytes、SHA256 `b7efd5c3bbf4a97ba5eac79f208c185d47186d4873f7f0d072b6bf379c8b5eff`
- gzip：94,085,922 bytes、SHA256 `7ed098e79b72379a2eabe58c6ab767dce314bac252a9a19b82a4e190edd4b8c3`
- Git公開用の64MiB以下の2 partと元bytesの復号一致を確認。
- G36223のC_INPUTS/PRICES/IMPLEMENTATION_EVIDENCEと元4 compile JSONのbyte/hash対応をlocalで確認。
- 全5 cellのPRICE_CALIBRATION_RECORD_V1はlocalで10件揃ったが、公開済み固定locatorは従来4 cellの8件まで。

**未完了**：今回の結果のcommit/push、固定結果SHAへのREAD_RECEIPT.bind、全18軸HANDOFFの公開後照合、remote最終照合。
旧HANDOFF_C.jsonの4 cell locatorは従来の公開snapshotであり、現在の計算終了状態と区別する。
C担当の科学的受入はNOT_PERFORMED。canonical-round専用guardは呼んでいない。

上記公開準備の後、SOURCE読取commandの自動承認レビューが、Codex認証のrefresh token失効で失敗した。
そのcommandは未実行であり、安全性を理由とする拒否ではない。認証復旧後は既存結果から公開作業だけを継続する。
科学jobを再投入しない。元Git credential helperの失敗は観測されておらず、書き換えない。

公開作業の準備script：`/tmp/d7-publish-recovery-20261115.py`。
`archive` は実行完了済みのため繰り返さない。結果fileをcommit/pushしてSHAを固定した後に `bind <結果SHA>`。
その後のREPORT/HANDOFF・索引・manifest更新とremote照合を完了してmandatory STOP。
fit/係数凍結はCの後続作業。HOLDOUT6 compile・外挿・E[C]/N/Gは未実施であり進まない。

## 後続追記：公開完了

上記の認証エラーによる未完了状態は履歴。利用者の再試行指示後、結果 f98cb96384b775bec2c2b9e1e63fe01a6ccdf2b1 のpushと公開後bindingが成功した。全18軸・10較正recordの固定locatorをHANDOFF_C.jsonへ結合済み。rawのarchive/bindや科学jobは再実行しない。全較正引渡しでmandatory STOP。
