# 説明専用再解析契約案

状態：**READY_FOR_USER_AUTHORIZATION**。approved=false／runtime_authorization=false、有効grant0、実科学数値再解析0。

SOURCE：`b6716396acc5760fde7cf7cc6f2481b183a08f10`。[修正仕様](mechanism_explanation_fix_specification.md)、[64 side入力projection](explanation_input_binding_projection.json)。private companionに、正確なSOURCE file hash、input binding SHA、元core SHA、runtime profile SHAと未承認canonical契約を固定した。

対象は前回の32比較・64 method sideのnormalization／bias／headroom／log bound／丸め説明だけ。主要30とH8独立n48の2比較、通常signed・factor1、同じ固定候補・対照・cohortを維持する。H4はlegacy DF12・subset16・H4_PRE_D_P_VALIDATION、H6 DF19/PF136、H8 DF28/PF184、1 Å/STO-3G/T=.8/P=0。32 ID・64 side IDは入力projectionに列挙する。

元coreは2bf5d832…に結び付いたprivate results.jsonを固定SHAで継承する。G、回路単価平均、費用比、削減率、shot合計、SEを再計算しない。candidate再選択／ランキング／pooling／212補足／H4 D/P／新統計／sampling／signal／compileは0。

1 passで説明64 sideを検査する案。必要fieldがある63 sideは説明guardの完了が受入条件。A82_OLD_PR.PRはINPUT_NOT_RECOVEREDのまま欠測を許容し、31比較＋1欠測という将来の結果を事前にPASS認定しない。実数値不一致・未登録の追加欠測・identity mismatch・入力／SOURCE／runtime破損はfail-closed STOP。保存headroom/log boundとの許容abs差1e-12、保存upper経路のstrict log bindingを保持する。

未来の出力はexplanations.json、explanation_sides.csv、joined_core_reference.json、worker_result.json、native_wait.json、terminal.json。失敗時STOP.jsonと消費済みmarker/ledgerを保全する。新しい専用rootだけを使い、旧結果・入力・SOURCE・grant・ledger・STOPは出力先にできない。native exit0・wait/reap、入力／runtime／認可記録の前後hash、結果SHA、旧coreの値そのもののコピー一致、固定32 IDs／64 sideのschema/statusを受け入れた後にmandatory STOP。retry／返却／次stage／公開pushなし。

runtime SOURCE closureは旧core＋新説明module＋新launcherの3file、解析Pythonと68 runtime fileを固定。小さいscalar処理の1 workerは仕事の単位であり、CPU/RAM/時間の恒常capではない。共有OS・cgroup・venv・cache・他jobを変更しない。

## 次のユーザー承認案

> 新SOURCE `b6716396acc5760fde7cf7cc6f2481b183a08f10` とprivate説明専用契約・入力binding・元core結果の固定SHAに結合した32比較／64 sideについて、説明側だけの保存値検査を新規one-shotで1回承認する。63 sideのnormalization／bias／allowance／headroom／log bound／丸めを評価し、A82_OLD_PR.PRは欠測を保持する。既存32 coreは固定結果を継承し再計算しない。数値不一致はSTOP、retry・新統計・sampling・compile・212補足・H4 D/P・公開push・次stageは含めない。

この文案は有効grantではない。ユーザーの新しい明示承認receiptを受け取るまで、有効grant発行・消費・production適用をしない。未回収signalの所在が判明して入力を変更する場合は、先に別versionのbinding／契約を固定する。
