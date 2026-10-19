# PF直接compile：現行722軸とlegacy66軸の独立契約案

2026-10-20。788軸を計画対象に維持し、現行4系列とH4 legacy DF12を独立した実行契約案へ分ける方針を記録した。[固定計画](track_a_pf_direct_compile_expansion_plan_20261019.md)と[技術準備](track_a_pf_direct_compile_technical_preparation_20261020.md)は変更していない。

| 対象 | 新規取得予定軸 | 契約案 |
| --- | ---: | --- |
| H2 DF3 | 72 | 現行4系列 |
| H4 DF10 | 122 | 現行4系列 |
| H6 DF19 | 214 | 現行4系列 |
| H8 DF28 | 314 | 現行4系列 |
| 現行4系列合計 | 722 | current_4series_722 |
| H4 legacy DF12 | 66 | legacy_h4_df12_66 |
| 全体 | 788 | 2件の独立契約案 |

[packet](../../artifacts/resource_applicability/track_a_pf_direct_compile_contract_split/2026-10-20/README.md)には、各案の完全なtask ID集合・転記CSV・候補行digest・予定call数を保存した。cosine/sineの組を保ち、2集合は重複なしで元の788軸を過不足なく覆う。qによる絞り込みや候補の再選択はしていない。旧profile等の再照合が必要な別48軸は含めない。H6 ε=0.005の`B0_p11_q4_2nd`の両軸も現行契約案に保持する。

## 独立させるもの

各契約に別の承認receipt・one-shot grant・出力root・ledger・marker・STOPを結合する。1契約への承認を他方へ流用せず、完了しても他方を自動起動しない。読み取り専用のSOURCEや入力が共通になる場合も、各契約で入力・回路・control・compiler・profile等の一致とbyte hashを検証する。書き込み可能な結果・消費状態は共有しない。

これは将来の契約設計であり、2つの本番実行領域やgrantを作成したことを意味しない。旧の一体型DRAFTも履歴として保全した。

## 承認と実装の状態

今回の発言は契約分割の方針として採用した。取得範囲または科学実行の明示承認には読み替えていない。両案とも`selected_task_ids=null`、`acquisition_scope_approved=false`、`execution_authorized=false`、`grant_issued=false`、有効call数0。`proposed_task_ids`と`proposed_call_caps`は予定範囲を表すだけである。

本番loader・独立control oracle・owned worker/wait4・直前gate・SOURCE/input/provider/runtime/profile sealは未完了。既存adapterのmetadata受付は分割案を実行する入口ではなく、両案を未承認として拒否する。最終結合時は、選択task IDと承認digestが当該契約の固定範囲に一致し、他契約のtaskを受け付けないことまで検証する必要がある。

CPU/RAM/worker/qに恣意的な上限は加えていない。将来は許可CPU・実効空き資源・task footprintから投入し、一時的なPSI上昇や無関係worktreeの更新を正常workerの停止条件にしない。共有環境・他ユーザーへ変更を波及させない。

## 今回の検証と限界

[検証記録](../../artifacts/resource_applicability/track_a_pf_direct_compile_contract_split/2026-10-20/structural_validation.json)は、系列別件数・完全pair・2集合の非重複と788軸全体の一致・CSV行の同一性・両契約の実行拒否・旧packetと各時点manifestのhash保全を確認する。主担当自身のmetadata検査であり、実分子の動作検証や独立科学レビューではない。

新規入力decode・回路生成・control数値検証・compile・sampling・fit・費用再解析は0。旧SOURCE・raw結果・grant・ledger・marker・STOP・Figure 2は不変。実測数やモデル精度について新しい科学的結果はない。詳細科学レビュー・最終evidence freezeは未開始。

次の担当は**ユーザー**：722軸と66軸それぞれの実取得範囲を判断する。続いて**Codex**：承認された範囲の本番結合・SOURCE等のsealと構造検証を行い、科学実行の別承認まで起動しない。今回の新規変更は旧28ファイルの公開承認に含めず、公開確認後のみpushする。

**mandatory STOP。**
