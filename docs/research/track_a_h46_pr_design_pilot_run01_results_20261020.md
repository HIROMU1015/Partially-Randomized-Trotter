# H4/H6 PR DESIGN_PILOT run01：取得・技術終了監査

科学SOURCE `de5485d4b7c6de25dd9b6137c9a12bce697242c6`。各群33本の承認済み取得だけを実行した。

| 系 | 状態 | 完了trajectory | 受理compile軸 | native exit/reap |
| --- | --- | ---: | ---: | --- |
| H4 | ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP | 396/396 | 792/792 | 0/True |
| H6 | STOP | 74/396 | 148/792 | 1/True |

byte完全復元と科学native受入を分離した。非完了・STOPをPASSに読み替えず、未取得値を0補完しない。

保存容量・native gate・SOURCE/結果SHA・wall/RAM/scratch・終了状態は[公開技術summary](../../artifacts/resource_applicability/track_a_h46_pr_design_pilot_run01/2026-10-20/run_summary.json)を参照する。RAM/scratch peakは定期観測値であり厳密な上界ではない。private原本・seed一覧・private絶対pathは公開しない。

分散・SE/MCSE・n_E・shot・総費用・順位を計算していない。母平均費用・rare tail保証、AD03採用も行っていない。既存pilot/探索/独立確認との標本混合、retry、top-up、条件変更は0。

**mandatory STOP。** 次の担当はユーザー：取得・終了監査結果を確認し、pilot限定数値評価を別途承認するか判断する。Codexは新たな承認まで科学統計・本探索へ進まない。

## 追記：H6停止原因と部分byte監査

H6の最初の科学STOPはm5 sampling factoryの `finite Taylor cutoff does not meet truncation_tolerance.` である。native exit1・全worker回収、元runのSTOP・係数・law・seedを保持し、retry/top-upは実施していない。74 pairは完了済みだが396本の契約全体は未完了である。

初版summaryのH6 archive数0/容量0は、全396本を要求する終了監査がbyte監査へ到達しなかった際の表示であり、実ファイル不存在を示さない。初版JSONのbytesを保持し、[補足summary v2](../../artifacts/resource_applicability/track_a_h46_pr_design_pilot_run01/2026-10-20/run_summary_v2.json)では全stage値をnull・NOT_EVALUATEDとした。

[別の部分byte監査](../../artifacts/resource_applicability/track_a_h46_pr_design_pilot_run01/2026-10-20/H6_partial_byte_audit.json)で、74完了pairの444 archiveと、途中保存を含む存在する510 archiveの元byte SHA復元を確認した。全510の元byte 19,101,433,587、圧縮byte 1,172,447,462。部分監査は全stage科学PASSや再利用承認ではない。

今回の追記は既存receipt・保存byteの技術監査のみで、分散/MCSE・shot・会計・順位の解析は0。次の担当：ユーザーがH4のpilot数値評価と、H6のcutoff受入条件調査・修正後の別実行を承認するか個別判断する。CodexはH6を自動再実行しない。
