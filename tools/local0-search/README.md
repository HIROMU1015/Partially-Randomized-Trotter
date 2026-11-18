# ローカル0からRomuTailの通常入口へ探索範囲を渡す

## 開始・再開と公開結果の同期

StartとResumeは、RomuTailで確認後に公開結果を1回同期し、実行中だけ600秒ごとに確認する。
同じrunは同じWindows monitor taskを使い、完了・途中停止で最終同期して終了する。
公開が遅れた最終結果は「最終結果は公開待ち」と記録する。
「結果を同期して」は`tools/local0-result-sync/Invoke-H4ResultSync.ps1 -Action Sync`。
plan-only・dry-check・Windowsログオンからmonitorは開始しない。
固定の閲覧入口と詳細は[公開結果同期README](../local0-result-sync/README.md)を参照。

## 絶対κ区間を通常入口へ渡すPlanOnly

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action PlanOnly -RequestFile .\tools\local0-search\kappa-scope\REQUEST.json
```

現在の固定serviceは、PR-Pauliの絶対κ区間を既存の物理schedule探索へ自動選択する。
REQUEST.jsonはL_D=4〜7、絶対κ=2〜64、shot配分auto。両端込みの範囲を保存し、
`run_rpe_schedule_search.py --plan-only`が実際に読む同じROUNDにconcurrent/autoも設定する。
物理scheduleの選択にはSCHEDULE_WIDE_V1を使用し、範囲外候補は実行候補から除外する。
省略した幅・公式・shot上限は採択済み設定を保持する。倍率への変換や範囲縮小は行わない。

同じ指示の再送は同じrunをhash照合して再利用する。上限変更の検収入力は次のとおり。

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action PlanOnly -RequestFile .\tools\local0-search\kappa-scope\REQUEST.upper128.json
```

科学承認false・Q3予算0のROUND.plan-only.json、受信原文、native固定入力、物理schedule案を保存する。
原文の言い換えや省略も別受信記録として保持する。範囲変更は別runになり、旧設定を上書きしない。
この経路は--prepare-readinessを付けず、返送済み12件のdry-checkを再実行しない。
CURRENT_TARGETは変更せず、実Q3・controller/relayも起動しない。

検収結果は[絶対κ区間REPORT](../../docs/research/local0_absolute_kappa_20261117/REPORT.md)を参照。
以前の未対応停止と別roundでの分散設定確認は[配置REPORT](../../docs/research/local0_integrated_349_20261117/REPORT.md)に履歴を保持する。

## 通常の開始・再開

Windows PowerShellから使用する。利用者の明確な開始指示を実行承認として記録し、同じ承認を取り直さない。
この入口自身は候補選択・Q3計算を実装しない。通常optimizerの設定、Executor、prior_records/ingestを使用する。

```powershell
# Previewで確定したrunを明示して開始・再開する
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -RunId '<run ID>'
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start -RunId '<run ID>' -Instruction '探索を開始して'
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId '<run ID>' -Instruction '探索を再開して'

# 中央の対象だけを確定する（実行承認・科学起動は0）
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Prepare -RequestFile '<確定する範囲.json>'

# 範囲を受け取って開始する
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start -RequestFile '<指定範囲.json>' -Instruction '<利用者の範囲・開始指示>'

# 実行せずに適用内容と未対応項目を確認する
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -RequestFile '<指定範囲.json>'
```

自然言語の解釈はローカル0担当が行い、指定された全項目をJSONへ写す。曖昧な値を自動補完するNLPや外部LLMは追加しない。
例示文だけを実行承認にしない。対象と範囲が明確な実行指示では、その原文をInstructionに渡してStartする。
省略した項目は中央の採択済み設定を使う。Start・Resumeにはrun IDか範囲を一つ指定する。
信号・価格の不足判定と準備タスクの生成・分配は中央が行う。入力JSONは元のバイト列で転送する。
`scope.preparation_budgets`の省略は中央の採択値を使い、`0`は明示した工程の新規枠0として送る。
詳細な状態表示と停止runの引継ぎは[中央との応答契約](PROGRESS_CONTRACT.md)を参照。
既存runが既定SERVICEと別のrun台帳を使う場合は、そのrunの `-Config` と `-RunId` を指定する。releaseの明示が必要な場合は、中央の記録にあるIDを `-Release` で渡す。中央が返した対象不明などの理由はそのまま表示する。

## 入力形式

```json
{
  "scope": {
    "methods": ["PR_PAULI"],
    "per_method": {"PR_PAULI": {"L_D": {"min": 5, "max": 7}}},
    "shots": "auto",
    "stop": {"max_new_q3_per_method": 128}
  }
}
```

方式IDはFULL_PF、TRUNCATED_PF（B0I）、PR_PAULI、PR_DF。
L_D/pは包含min/maxまたは整数配列。PF公式はformulas配列、PF幅比はdelta_over_hの厳密な上下限文字列。
shotsはauto、またはmin=1とmax_per_round（整数または各roundの上限配列）。停止条件はmax_new_q3_per_method、max_steps_per_method。
batch/max_cyclesはその値を実際に消費するschedule/neighbourhood入口だけが受け付ける。

絶対幅delta_pfは`{"kind":"exact","value":"1/5"}`と`{"kind":"binary64","value":"0.2"}`を区別する。
κは`{"mode":"absolute","min":"2","max":"64"}`とmode=multiplierを区別し、相互変換しない。
元のJSONと原文を保存し、binary64を十進分数へ丸めない。

## 現行入口の対応範囲

既定のmethod入口は、方式・L_D/p・PF公式・PF幅比・自動shot/round別上限・Q3数/step停止を適用する。
絶対Δ_PFは、中央が採択したrequested-width schedule設定を登録したときに使用できる。
PR-Pauliの絶対κ上下限は、現在のserviceのabsolute_kappa_routeにより、既存schedule探索の両端込みkappa_boundsへ渡す。
κ区間とL_Dの実行候補filterも同時に設定し、既存の物理schedule単位の選択・shot探索を使う。
κ倍率だけで保存候補全体を制限する機能は現行入口にないため、generation.kappa_multipliersを範囲制限と偽って使用しない。
PRの公式限定、shot下限>1、未対応停止条件、未知のキーも、具体的なfield/reasonを返して開始しない。
absolute_kappa_routeを持たない旧serviceや未対応方式では、具体的なfield/reasonを返して停止する。
範囲を別のκ・少ないL_D・保存済み候補だけへ黙って変更しない。

## 保存と再利用

中央の各runにREQUEST.json、APPLIED_SCOPE.json、PROBLEM.json、ROUND.requested.jsonを固定保存する。
Prepareは次回の実行設定とCURRENT_TARGETを固定するだけで、AUTHORIZATIONやLAUNCHを作らない。
実行設定のscience_approvedは次回Startのための意図であり、Gatewayは明示Startのscope-bound AUTHORIZATIONなしには通常入口を呼ばない。
範囲・停止条件・評価契約・固定codeが同じなら同じrun IDになる。範囲変更は新runとし、旧ROUND/receiptを上書きしない。
同じ評価契約の既存runと採択済みprior_run_dirsを、通常Executorの完全照合へ渡す。
開始前の§10.5照合と、通常optimizer開始時のprior_records/ingestの双方で保存Q3を照合する。
入力・route・binding・configuration・shot・評価条件が一致した記録だけを取り込む。
同じ配置でも評価条件が変わった記録は採用しない。参照先不明・壊れた同一Q3 receipt・未解決の既存launchはHOLDする。
未評価点は通常の候補集合へ残し、現行optimizerが未準備・予算・停止条件を明示する。

開始時にrun ID、方式、適用範囲、shot上限、固定SOURCE/RPE条件を表示し、全条件はresponsesのJSONから読める。
計画templateのscience_approved=false/予算0という準備gateは、明示Start時に実行grantへ変更する。
その場合も採択済みstep上限を維持し、範囲内Q3予算をUNLIMITEDと記録する。利用者が明示した予算0は保持する。
この変換はSERVICEのplan_template=trueで明示する。通常の採択済み実行予算は変更しない。

## 運用境界

CLIENT.jsonはRomuTailの通常入口と固定SERVICE.jsonを参照する。controllerの起動と継続は中央が担当する。
Windowsクライアントからの通信は既存のSSH経路を使う。Windows relay、各ホストのprofileとWSLの参加flagはそれぞれの運用入口が管理する。
SERVICE/採択済みtemplate変更は新しい固定版として扱う。確定targetはCURRENT_TARGET.jsonに保持する。
旧runのUNKNOWN/STOPや未解決launchを自動retryしない。通常入口の既存停止条件を維持する。

## 人工受入

旧L_D=4〜6、新L_D=5〜7。旧L_D=5の完全一致Q3を開始時に取り込み、同Q3再起動0。
新L_D=6を人工adapterで1件だけ評価し、L_D=7を未評価候補として残す。
Windowsの本コマンド→実SSH→サーバー2 Gateway→通常round_mode→実Executor/ingestで確認した。
人工adapterだけをmockし、実H4・信号・回路compile・本番queueは起動しない。詳細は固定REPORTを参照。
