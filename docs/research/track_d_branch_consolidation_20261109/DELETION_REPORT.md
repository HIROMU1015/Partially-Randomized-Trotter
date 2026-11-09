# Remote branch削除結果

2026-11-09 JST。**実削除0本。認証上の障害により削除段階は未完了。** 成果統合と履歴保全は実施した。

| 項目 | 実績 |
|---|---:|
| 開始時remote総数 | 405 |
| 開始時Track D対象 | 43 |
| O2移管対象 | 2 |
| 統合履歴から到達する元Track D branch | 43 |
| 追加収録した元branch（C7-4＋B3〜B5） | 5 |
| 通常merge回数 | 2 |
| 新統合branch | 1 |
| 削除したremote branch | **0** |
| 元対象の残存 | **45** |
| KEEP_ACTIVE | 5 |
| KEEP_INDEPENDENT | 1 |
| HOLD | 39 |
| 新archive tag | 0 |
| 公開前再照会時remote総数 | 407 |

405＋当作業の統合branch1＋別チャットの対象外 `codex/research-c-branch-cleanup-20261109` 新設1＝407。対象外branchの作成・削除・更新はしていない。総数は同時作業で変わり得るため、最終チャットの実照会値と区別する。元45本のHEADは再照会時すべて開始時と一致。

## 削除を妨げた実際の接続制約

- HTTPS `git push --dry-run` は `could not read Username` で失敗。credential helper/共有設定/認証情報を変更していない。
- 既存のstrict SSH接続確認はGitHub ED25519 host key未登録で失敗。known_hostsへ追加せず、検証を無効化していない。SSH鍵や新しいtokenは作成していない。
- 接続済みGitHub pluginはtree/commit/branch作成とlease付きbranch更新が可能。公開は成功した。一方、提供toolにはtag作成・branch/ref削除がなく、GET fetchで変更操作は行えない。別用途toolの不正利用で迂回していない。
- 削除API/認証が確立していないため実DELETEは送っていない。失敗を成功・削除済みと扱わない。

## 保持・HOLD一覧

| branch | 分類 | 具体理由 |
|---|---|---|
| `codex/o2-vacuum-cswap-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/o2-vacuum-h4-preparation-20261108` | KEEP_INDEPENDENT | O2/A1依存と固定契約を保つ独立の正式コードcontext。 |
| `codex/track-d-b1-c1-validation-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-b6b-bc-integration-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-bc-static-preparation-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-bc-total-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c2-q2-compile-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c3-q4-q8-validation-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c4-price-model-preparation-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c5-whole-path-q2-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c6a-paired-price-pilot-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c6b-accelerated-completion-20261107` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c6b-q8-exploratory-20261107` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c6b-state-action-acceleration-20261107` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c6b0-paired-model-preparation-20261107` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c7-0-structural-preparation-20261107` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c7-1-q16-exploratory-20261107` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c7-2-cx-structure-20261107` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c7-3-q32-accelerated-20261107` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c7-3-q32-exploratory-20261107` | HOLD | 元runと収集script2本が未追跡。今後のpush用途未確認、加えて削除認証なし。 |
| `codex/track-d-c7-4-model-stability-20261107` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-c8-1-q64-20261108` | KEEP_ACTIVE | q64特別保護。元/修正/限定完了SOURCEと未公開closeoutに必要。 |
| `codex/track-d-c8-1-q64-accelerated-20261108` | KEEP_ACTIVE | q64特別保護。元/修正/限定完了SOURCEと未公開closeoutに必要。 |
| `codex/track-d-c8-1-q64-completion-20261108` | KEEP_ACTIVE | q64特別保護。元/修正/限定完了SOURCEと未公開closeoutに必要。 |
| `codex/track-d-c8-1-q64-nofusion-repair-20261108` | KEEP_ACTIVE | q64特別保護。元/修正/限定完了SOURCEと未公開closeoutに必要。 |
| `codex/track-d-c8-1-q64-numerical-repair-20261108` | KEEP_ACTIVE | q64特別保護。元/修正/限定完了SOURCEと未公開closeoutに必要。 |
| `codex/track-d-cut-artificial-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-cut-h4-q4-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-cut-h4-q4-completion-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-h4-paired32-reference-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-h4-reference-validation-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-minimal-reference-validation-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-o5-c1-h4-q4-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-o5-c1-head-tail-artificial-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-o5-c1-head-tail-static-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-o5-design-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-o5-q4-cohort-metadata-20261108` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-pr-rpe-static-survey-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-real-pr-transfer-static-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-reference-control-feasibility-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `codex/track-d-vacuum-path-static-check-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `track-d-b3-rpe-statistics-preparation-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `track-d-b4-1-rpe-statistics-implementation-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `track-d-b4-2-rpe-shots-science-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |
| `track-d-b5-actual-rpe-static-survey-20261106` | HOLD | 保全後の候補だが削除認証/APIなし。削除直前の使用状況等は後日再確認が必要。 |

安全確認を完了できない枝をDELETE_CANDIDATEと確定せずHOLDにした。[後日再確認用の38候補](DELETION_CANDIDATES.json) と [条件付き手動コマンド](DELETION_COMMANDS.md) を残したが、今回の実削除実績には含めない。
