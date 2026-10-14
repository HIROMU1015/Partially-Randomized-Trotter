# Track A サーバー・ローカル差分と競合報告

2026-10-14 JST。サーバーbranch `track-a-cross-environment-sync-server-20261014` / `2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c` と、ローカル固定metadataの照合結果。
判定は資料の所在・bytes・実行identity・比較scopeに限定する。科学的な不一致の最終解決や実験認可は行わない。
[ローカルinventory](track_a_local_inventory.md)、[統合候補](track_a_shared_research_status_candidate.md)、[固定参照一覧](track_a_local_sync_reference_index.md)を参照。

## 7分類と確認結果

|分類|この調査での内容|扱い|
|---|---|---|
|SAME|選定265実体の217件が確認済み公開Git blobと一致。サーバー主要19証拠も取得bytes/SHA一致|同じ原本を参照。再commit・再計算しない|
|SERVER_ONLY|ローカル原worktreeにない最新H4 B0+I全map/v0.4、H6条件付きs2、H8 resume02/snapshot、server inventory|GitHub固定commitへのリンクを追加。巨大raw/sourceは複製しない|
|LOCAL_ONLY|44実体・16種類が調査対象公開版に未一致。再配置されたplan、dirty dated note、旧export terminal等|原本保持。摘要/hash公開、terminalだけexact copy。全server不在の証明ではない|
|DIFFERENT_VERSION|rootのPROJECT_MAP、研究概要、研究目的、VALIDATION_STATUSの4実体|dirtyを保持。時点・系統の差を説明し、rootを新server版で上書きしない|
|SCIENTIFIC_CONFLICT|同一identityの数値矛盾は未確認。比較集合・評価階層の混同でclaim競合が起き得るC01–C03|両根拠と限定を保持。新しい科学判断をCodexで確定しない|
|ENVIRONMENT_SPECIFIC|原本array、runtime/cache/venv、環境設定、ローカルZIP/診断bundle、物理path|同期対象外。metadataのidentityだけ共有|
|UNVERIFIED|H8現時点/最終result、H8モデル予測契約、旧export450 mismatchの実event、server run03公開到達性等|未確認と明示。科学計算で埋めない|

ファイル件数は重複worktreeの実体数。265実体は93種類の内容であり、265実験/独立証拠ではない。
SERVER_ONLYは**調査した原ローカルworktreeに未配置**という意味で、取得後もサーバーexclusiveと主張しない。
LOCAL_ONLYは照合範囲を限定した分類。GitHub全履歴/サーバーfilesystemの不存在判定ではない。
機械可読の論点・未解決項目は[registry](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/difference_registry_v1.json)。

## 一致した研究情報

有限時間complex signal精度、bias/normalization/shot/full measured wrapperを接続する資源研究という目的は一致する。
PRの必勝、新algorithm、厳密population winner、chemical-accuracy energy/QPE/RPE総資源は達成済みとしない。
M1/M2/PM・AX-0/AX-1・v0.1の原本は公開copyと同一。サーバー共有入口で記述が短いことと未公開であることを区別する。
H6 scientific source5a903…と後続のserver実行sourceを分ける方針、旧STOP/grant保全、新しい科学実行の明示承認制も一致する。
rootの同名remote未push2 commitは、サーバー2298祖先として公開済みでありsource欠落ではない。

## claim・scopeを混同しないための競合整理

|論点|ローカル側の根拠|サーバー側の根拠|統合候補での扱い|
|---|---|---|---|
|C01 H4「B2が全条件で最小」|PM-2のdevelopment218/固定302点、後のgeometry元4方式で1812 B2点最小|B0+I追加後397 B0+I /1415 B2|元集合は保全し、新集合のclaimを別記。主張の科学採否はGPT判断待ち|
|C02 geometryとnative|旧H4 rank12/ordinary/structured|1Å native anchor16panel・B0不在、H6/H8 native/full S2/S4|同一系サイズ比較へ合算しない。DF/control/状態/探索自由度を条件表で保持|
|C03 H6未完了と完了|旧v2 MemoryError449/592、v3親directory STOP、launch_v2 Phase A abort/newcomplete0|別実行source78a…探索592/592、87b…確認256group|別execution identityの履歴。旧STOPを成功へ書換えずgrant流用とも表示しない|
|探索 vs 独立確認|探索92候補、random8標本|事前選抜union8構成、fresh32標本|poolしない。確認集合内point comparisonと母平均winnerを分ける|
|H8準備 vs 実行|旧reviewの「H8未開始」/提案・認可待ち|後のresume02起動と203/592 snapshot|当時のreviewを改変せず後続の別認可・実行recordへリンク。最終結果ではない|

上表は同一条件の数値矛盾の発見ではない。C01–C03の限定なしの主張を共有正本へ採用すれば競合するため、双方を保存した。
AX-1bの「凍結モデルのH8独立評価」提案に対し、今回取得したH8 core metadataだけでは未観測予測/モデルの凍結・評価契約は確認できない（C04）。H8 resource-mapの完成をそのままRQ-P検証完了としない。

## 追加すべき情報と保留すべき資料

共通ノートへ安全に追加できる候補はM1/M2/PMのscope付き履歴、AX-1b後のRQ優先順位、原稿v0.1の保全リンク、旧H6失敗台帳、export terminalの未解決事項である。
公開済みsame bytesはリンクだけ追加する。ローカルplan・未commit研究ノートの採択、H4 B0+I後の主要claim/新規性、population winner認定、H8モデル独立性は保留する。
rootの横断progressは2026-10-13時点でH4 production/H6 reviewの公開を未確認と記す部分がある。新しい固定公開証拠で所在が確認できたことを併記するが、原ノートを事後書換えしない。Track B部分は統合しない。
旧exportの診断v2案は未実装・未認可。復旧を本taskから自動実行しない。

## 未取得・未確認

- H8 live状態、最終探索/確認mapと結果commit：今回server操作を禁止。公開snapshot時点に限定した。
- server旧H4 run03 `a3248055…` の公開到達性：server側の保留事項を継承。rawを探索的にコピーしない。
- old-export partial450の再生成raw event/digest：旧STOPに未保存。source/terminalは取得済みで、数値原因はUNDETERMINED。
- ローカルonly plan/dirty noteの採択状態：科学結果の固定原本とは区別し、hash保全まで。
- 厳密母平均/同時winner、経験的uのcertificate、旧compiler bit同一性、最終energy資源：既存の未解決事項。今回解決していない。
- 他repositoryとの科学的対応：identity確認のみで未確定、独立研究を統合しない。

## 推奨統合手順

1. 両Codexはserver2298と本branchの[統合候補](track_a_shared_research_status_candidate.md)・[参照一覧](track_a_local_sync_reference_index.md)を同じ固定SHAで読む。
2. 本報告にないlocal/server資料があれば固定commit/blobで追加差分を返す。更新日だけで正本を選ばない。
3. user/server側が統合候補の採択対象を決める。科学的判断待ちはそのまま残し、共通正本の正式変更は別作業として行う。
4. source統合、大きい入力同期、再計算・H8操作は個別認可/identity契約で扱う。本taskの資料branchを科学実行branchへmergeしない。

original server入口・inventory・policyは変更しない。本候補は `INTEGRATION_CANDIDATE_NOT_CANONICAL`、`DRAFT_NOT_AUTHORIZATION`、`next_stage_authorized=false`。公開後mandatory STOP。
