# H2高次PF outer32：取得終了記録

状態：`H2_OUTER32_THROUGH_EXPLORATION_ACQUISITION_COMPLETE`。
科学SOURCE：`4ec26921c6872ed6e8667f167074e1e405cf1a3f`。

H2直鎖1 Å、STO-3G、tol-only `1e-8`・rank3、保存Hamiltonian/状態、T=0.8、P=0、prefix1、S2/Yoshida4/m5_best、q=1/2/4/8、K4、outer32。旧H2のr系列・旧標本とは別cohortであり、混合していない。

| 段階 | 保存済み取得範囲 | 終了確認 |
| --- | --- | --- |
| SIGNAL | 12候補・120適格性label、full-H reference1件 | PASS、native exit0・reap済み |
| DESIGN_PILOT | 396/396 trajectory・792/792軸 | native/compiled・原byte SHA復元・終了監査PASS |
| PILOT_DESIGN | H2自身のpilotを使う2%/分散倍率4/群別下限33設計 | PASS、探索とは非混合 |
| PR_EXPLORATION | 525/525 trajectory・1,050/1,050軸 | native/compiled・原byte SHA復元・終了監査PASS |

pilot・本探索とも全25 workerがexit0で回収済み。H2専用の既存compiler位相guard監査もそれぞれ792軸・1,050軸でPASSした。変更前後のgate/angle/wire stream同一性を条件とする既存guardであり、新しいcompiler政策や閾値を採択したものではない。

[技術summary](../../artifacts/resource_applicability/track_a_h2_outer32_run01/2026-10-21/summary.json)には、各契約のplan SHA、終了監査SHA、native exit、保存量・復元検証を転記した。[文書構造検証](../../artifacts/resource_applicability/track_a_h2_outer32_run01/2026-10-21/structure_checks.json)は、保存済み監査記録との一致を確認する技術検査であり、追加の科学解析ではない。SIGNAL/PILOT_DESIGNのtrajectory欄は対象外のためnullであり、欠測を0補完していない。

## 文書保存処理の復旧

科学計算と全終了監査の完了後、資料commit用の公開検査が、更新した索引全体に元から存在する内部絶対パスを検出して停止した。`PUBLIC_METADATA_ONLY`の元停止receiptは保持した。新しい成果物全体と既存文書の追加行を検査し、公開資料・索引・manifestだけを復旧した。封印済みSOURCE、private原本、grant/ledger/marker/STOP、計算結果、H4/H6 jobは変更せず、科学計算の再実行は0件。

## 解釈と次の作業

本探索の取得終了は、費用順位や母平均優位の確認を意味しない。未観測希少層の費用・分散はunknownのまま。通常/非相殺のshot会計、総費用、再ランキング、独立確認P、AD03採用は未実施である。旧pilotだけの標本設計と、新探索の取得結果を区別する。

**mandatory STOP。** 次の担当：ユーザーが本探索結果の限定統計評価の採否・範囲を判断する。Codexは別途承認まで後続解析を開始しない。
