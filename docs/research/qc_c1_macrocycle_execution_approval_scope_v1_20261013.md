# C1 v1最終source・実行承認の対象

2026-10-13。Codex技術準備完了、**科学実行未認可・Mandatory STOP**。
固定source `ddaa05d2b51d575c3cfea1194bbfdffa182f9636`、branch `qc-c1-macrocycle-preparation-20261013`、origin `HIROMU1015/Partially-Randomized-Trotter`。

[技術設計](qc_c1_macrocycle_technical_design_v1_20261013.md)、[結果前契約](qc_c1_macrocycle_preregistration_v1_20261013.md)、
[source/input/env lock](../../artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/source_manifest.json)、
[固定execution plan](../../artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/execution_plan.json)、
[preflight](../../artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/launch_readiness.json)、
[未発行approvalの対象](../../artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/approval_request.json)。
approval_requestは実行認可ではなく、user approval referenceもnullのまま。

固定20filesのsource/入力/数学契約/runtimeをcommitとSHAで照合した。
CPython3.12.3/Linux x86_64とNumPy1.26.4の792配布filesのbyte lock、CPU/memory前提もmetadataだけで確認した。
NumPy import0、登録物理入力へのD代入/selector/PF/eig/compile0。
登録外18tests PASSはBCH/回路word/schema/認可gateの技術確認であり、science backendの有限h性能は未実行。
public runnerだけでなくpipeline入口も消費済み認可receiptを要求する。

次の承認対象は、4固定2qubit toy/39unitary方法/6h、H reference4＋PF eig936＝940calls、C2導出72record、
保存のみ独立監査、一次結果/失敗/GitHub公開、STOPまでの**一回**。
attempt1/retry0、2workers/1thread、memory/output/log/calls制限、wall-time上限なし。
H2/H4等分子計算、FCI、数値パラメータ最適化、sampling、compile、C4起動は含まない。

source-manifest SHA `d1ba6ab690f2b4fb2e5ebb7ca4cfcf0991afb45b137ea9d9965718fb106dc4c6`。
plan SHA `43d8d99a3d742d49b489b0ba7f49c48d9e21d624a91adfbaf84a7a06ed959a6e`。
runtime SHA `b5400afece5e555164cf4ed9dbed9afaf87e17563e153e5156e7fffd055aeae1`。
認可はこれらのidentity、scope、output、会話の明示approval referenceを持つ別JSONとして、承認後に固定する。
契約修正や別入力を同じone-shotで実行しない。

本selectorはknown compositionでも同じcatalog/情報から生成できる。独立新規算法として認定した数はzero。
今回の最小判別は、既知対照を含むsigned/finite-h/information費用の条件付き機構を確認するためのもの。
全QPE/RPE総費用UNKNOWN、運用certificate hと事後oracle frontierを分ける。
性能や新規性の科学判断は結果後の別GPT開始承認で扱い、現在同じ選定レビューを繰り返さない。

## GitHub公開・取得確認

準備payload `75fde54db905bd4ce5aca56b9a57569c57ce66b5` をGitHubから独立cacheへ再取得し、37変更blobs/1,485,738bytes、固定20files、旧229files、旧130manifest entries/non-result fields、追加42relative linksを照合した。
[取得receipt](../../artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/publication/receipt.json)。root既存130filesも保持した。
これは資料byte identityの確認で、外部科学再現ではない。source/input/plan/runtime不変、科学実行未認可、現在の準備作業は完了しMandatory STOP。
