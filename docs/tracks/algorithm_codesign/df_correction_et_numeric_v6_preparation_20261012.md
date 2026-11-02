# E/T numerical observer v6：技術修正・継続実行の準備

旧v5はr6 controlled wholeのnumerical upper 2e-7を認証できず停止した。
combined upper／内訳／失敗compiled IRが保存されず、原因帰属が未完了だった。
旧結果、source、authorization、消費済みmarker、terminalと分類を保持する。

利用者の追加認可（原文）：

> 次の作業に進んでよい。
> 次も停止した場合についてもそのまま修正作業に入って、計算完了まで進めて

この指示を、既定E/T比較の技術修正・focused validation・新source固定・新versionの実行を
継続する承認として記録する。個々のrunはsource-bound、runs1/retry0、専用markerで停止し、
technical continuationでは新source・authorization・namespaceへ分ける。
versionを跨ぐ実行回数は別途正直に報告し、旧runのretry0を全campaign一回と表現しない。
誤差保証、比較条件、fixture、精度、資源上限の変更はこの技術修正scopeに含めない。

## v6の変更

- numerical guardの既存assignmentsとreturn upperはAST同一。計算モデル・式・2e-7判定を変更しない。
- 計算済みOperator residual upper、forward guard、reference radius、angle/global phase上界、
  combined upperと判定、candidate native row／ordered IRをguard分岐前に保存する。
- candidateは診断であり、guard失敗時にaccepted native結果へ追加しない。
  成功時は同じrowをresultへ追加し、immutable blobを共有する。
- B個別guard、B uniform boundも既存計算値を分岐前に記録する。
- 二回目のOperator／fingerprint／transpile／spectral actionは追加しない。
- storage、guard、acceptance、accounting、saved_dataはv5とbyte同一。
  worker／coordinatorはnamespace・contract path以外同一。

contract v6の変更はversion、base、status、observation amendment、利用者のtechnical continuation記録のみ。
既存scientific sectionsと全capsは同一。CDは旧6行参照だけ、Eは保存済みr3/r6・rank2×3・seed7101107・
sign(+1,-1,+1)・u1/128、uncontrolled/MSB-controlled。compiler、numerical allowance、T精度・選択を維持。
native22／T64、各2worker、threads1、output64 MiB、log4 MiB、review512 KiBとCPU/RSS/wall上限は維持する。
既存のdominant QSD/Operator・synthesisの並列設定とlossless共有codecを利用し、speedupは主張しない。

## focused validation

最終**134 PASS、1 deselected、3.20秒**。
新observer6、v6 integration53、既存storage23、v5 integration52。
登録外2mode controlとmockだけで検証し、registered E-r3/r6・T合成は呼ばない。
新旧guard値の完全一致、保存後の同じ失敗判定、保存失敗時の停止、large candidate/final blob共有、
旧consumed marker launch拒否を確認した。

初回124 PASS/10 FAIL、次125 PASS/9 FAILを保存する。
v6 launchに残った2箇所の旧contract pathを修正した。
旧v5のreal pending-state testはconsumed historyを持つworktreeには適用しないため、
そのtestを保持してdeselectし、新しいconsumed-state拒否testで置換した。
新v6 pending testは維持した。全attemptのstdout/JUnitを保存し、full suiteは実行していない。
local technical validationであり、immutable CIや科学的比較結果ではない。

## 来歴・次の実行

準備branch：`track-b-df-correction-et-numeric-v6-preparation-20261012`。
親は旧結果公開commit `062ed4125dbd6fa3c7f085f3962ae8128c168be1`。
新namespaceは`src/trottertracks/algorithm_codesign/df_correction_et_numeric_v6/`。
runnerは`scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v6.py`。
新resultは`artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v3/`。
source／contract／manifestを固定・pushした後、その直接子にauthorization JSONとreceiptだけを置く。
今回の継続認可を原文で記録し、remote SHA／clean／fresh namespace／固定runtimeを確認して一回実行する。

失敗した場合は原結果とmarkerを保持し、観測された技術的原因を調べる。
既定保証を保つ技術修正なら利用者の継続認可内で次の新versionを準備できる。
科学条件や誤差保証の変更が必要ならGPT／利用者へ戻す。
比較を完了したらmandatory STOPして、科学的優劣・新規性・研究方針の判断はGPTへ渡す。
G10／RA-D0、分子/NPZ、GPU、sampling、LP、広範な探索へ進まない。
