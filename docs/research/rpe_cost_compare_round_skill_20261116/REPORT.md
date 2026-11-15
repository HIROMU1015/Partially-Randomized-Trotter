# rpe-cost-compare：B2探索round実行・再開への接続

Skillを新しい`--round` / `--run-dir`入口へ更新した。**新規Q3予算0で、Skill手順・直接CLI・同一round再開の全4方式の結果が一致**。p7の372,035,410 RZが反映された。新規Q3・signal・sampling・compile・fitは各0件。

## 固定SOURCEと実際の入口

- B2公開成果/HANDOFF：`a4b4d0e1f2110e63ae9a1dc6a89e1699f95dc0eb`。
- B2実行SOURCE：`a2e2ef50a108d6f34089a99bd00f11a7085dd9c9`。公開成果とrunner/optimizerのbytes一致を確認。
- 旧Skill：`eb14ca70e4de61e0fe1a8556f75571ee5e56fa18`。
- 今回検証したSkill手順SOURCE：`3262c0b4a2610a60c6b0bfb2c4ca010beda2d271`。
- 実行はB2の作業worktreeへ書かず、自分の独立checkoutを実行SOURCEで固定。Skillの既存専用branchを継続した。

実引数は`--problem <problem.json> --round <round.json> --run-dir <run directory>`。同一round再開は同じコマンドを再実行する。Skill内にexecutor・候補選択・予算台帳・再開ロジックを追加せず、B2を呼ぶ説明と入出力対応を更新した。

旧版の「executor未実装」「checkpoint再開未実装」を現在の状態から削除し、既存Q3 routeでの起動→検証/取込→次候補、および同じroundのjob/予算再開に対応することを記載した。今回の実証は予算0の接続/再開であり、正予算の実科学Q3をBが新たに試したものではない。B2の人工executor受入は元HANDOFFを参照する。

## 保存H4の一致

H4/1.00 Å/STO-3G/DF10、保存sector(2,2)、元H/state/snapshot、保存solver proxy、Z1+T1 total、MODEL_MSE_FILTER_ONLY、要求εE=0.00159360010199040 Ha、P=0を維持。公開H4 problemを変更せず、各方式予算0・initial=allの同一round fileを使った。

| 方式 | 確認済み最安構成 | 条件付き費用 RZ | 次候補 | 停止理由 |
|---|---|---:|---|---|
| FULL_PF | FULL_PF:formula=2nd,delta_over_h=16/7:11cf8ca6 | 559,457,386.00 | FULL_PF:formula=4th(m5_best),delta_over_h=16:70f7b317 | NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET |
| TRUNCATED_PF | TRUNCATED_PF:formula=4th,delta_over_h=32/5:dd8bfc24 | 372,035,410.00 | TRUNCATED_PF:formula=4th,delta_over_h=32/5:dd8bfc24 | NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET |
| PR_PAULI | PR_PAULI:L_D=5:4d6e2c9c | 541,743,905.57 | PR_PAULI:L_D=3:dd5b881d | NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET |
| PR_DF | PR_DF:L_D=7:ce9c7f08 | 1,235,340,731.91 | PR_DF:L_D=7:ce9c7f08 | NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET |

構成ID・全round X/Y整数列・exact費用・MSE上下界・pending_request・停止理由・予算状態を含む`methods`全体が3結果で一致。PER_METHOD.csvはbytes一致、NEEDS.jsonも一致。直接CLIと初回Skillは別run directory、再開はSkillの同一directory/round_idで実施した。ROUND.jsonは再開前後でbytes不変、INVOCATIONSは1→2、各launched/new_calls/spent_beforeは0、LAUNCH/OUTCOMEファイルは0件。

[VERIFICATION.json](VERIFICATION.json)にSOURCE・コマンド・全方式の最安/次候補・停止理由・検査結果を保存。[ROUND_BUDGET0.json](ROUND_BUDGET0.json)は実行した入力。初回の状態は`verification/skill_initial`へ保存し、再開後は`verification/skill`、対照は`verification/direct`。原の科学結果・旧Skill検証・旧STOPは書き換えていない。

## 機能の説明と制約

- round_id・対象方式・範囲・initial等は保持して再開する。予算増額は明示許可された方式別総上限だけを同じroundへ反映し、起動済み件数はprogramに数えさせる。
- JOB_IN_PROGRESS / STALE_JOB_NEEDS_REVIEW / UNKNOWN_RESULT / モデル不適合を区別し、Skillから無断retryしない。
- signal/価格/configurationの新取得、未対応configurationのroute、route viewの用意は別の不足。4方式menuをFull-H込みの5方式完了とは呼ばない。
- 費用は保存価格条件・モデル参照/条件付き閉包等の資格を引き継ぐ。Q3モデル適合を物理RMSE/十分N/E[C]/正式Gの保証へ昇格しない。次候補が残るため、出力は既評価範囲の確認済み最安。
- `approved_by`任意文字列を承認の証明・必須項目にしない。利用者の指示を許可とし、追加証明書/grant発行を要求しない。

Skill構造検査とlocal link検査を実施。上記の作成者自身の接続検査に加え、別agentがSkill本文から予算0の新roundと同じコマンドによる再開を実施した。両方exit 0、全方式のmethodsは直接CLI結果とも一致、予算/新規call/起動数は全て0、job directoryなし。確認対象は別のround_idを用いた手順の技術検査で、科学的レビューではない。[forward観測](verification/forward/OBSERVATIONS.json)と[方式別結果](verification/forward/METHOD_OBSERVATIONS.json)を保存した。

programはsignal/sampling/compile/fitの個別counterを出していない。これらを起動していないことは、確認したround制御経路と新規job起動なしで判断し、存在しないcounterを報告していない。Aによる残る受入テストの代行完了とは表示しない。起動・再開方法とSOURCEの引渡しは[HANDOFF](HANDOFF.md)。ロードマップ本体は編集せず、このSkill更新と未確認範囲を統合担当へ渡す。
