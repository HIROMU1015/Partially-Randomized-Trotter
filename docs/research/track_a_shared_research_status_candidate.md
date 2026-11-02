# Track A 共通研究ノートの統合候補

2026-10-14 JST。`INTEGRATION_CANDIDATE_NOT_CANONICAL` / `DRAFT_NOT_AUTHORIZATION`。
サーバー固定commit `2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c` を基礎にローカル照合で不足した履歴・所在・留保を追加した**候補**。正式共通正本へ未採択、科学認可なし。
元の[サーバー入口](track_a_shared_research_status.md)、inventory、同期policyは原bytesのまま保全する。
[差分・競合](track_a_environment_sync_differences.md)、[local inventory](track_a_local_inventory.md)、[source/result参照一覧](track_a_local_sync_reference_index.md)、[照合metadata](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/README.md)が根拠。

## 読み方と研究方針

確認済みの保存結果、GPT reviewの科学判断、技術的完了、未検証仮説、将来候補、実行認可待ちを別ラベルで扱う。
2026-10-09の[AX-1b後GPT判断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_ax1b_post_scientific_review_2026-10-09.md)では、RQ-R（精度をそろえた測定込み実コンパイル資源の競争力と適用範囲）が主、RQ-P1（one-shot費用予測）が補助、RQ-P2（truth-free shot/資源/選抜）は未達成で必須完成条件外。
これは既存判断の記録であり、新しい採択ではない。古いAX-0/拡張planのRQ-R/RQ-Pの広い案を現行科学実行へ自動適用しない。
原稿v0.1と後のv0.4を保全し、今回原稿化は進めない。

## ローカル照合から加える研究履歴

|段階|確認済みの証拠・scope|区分と現行の扱い|
|---|---|---|
|M1/M2|H4 linear/STO-3G/DF12、1.00Å development210、1.30Åの結果前固定5構成transfer、T=.8、S2、δ=.1〜.8、canonical finite-RTE、one-body/scalar保持|保存実行結果。現在のH4 geometry/nativeに同一条件として合算しない|
|PM-0|既存same-R/帰属解析|POSTHOC、追加独立標本ではない|
|PM-1|developmentに近接B0 prefix4/5×q1/2/4/8の8構成、計218|保存実行結果。全prefix/高次PF最適化ではない|
|PM-2|development218と元M2 5構成、302精度 .005〜.1を保存bias/費用で再会計|POSTHOC感度。M2の未観測精度transferにしない|
|原稿v0.1|既知残差処理のfinite-signal資源台帳とclaim限定|保存draft。科学的最終承認/新algorithm claimではない|
|AX-0/AX-1|比較契約、保存値model比較と条件付きoracle解析|技術的完了・GPTレビュー済みの研究優先順位。未知分子operational selectorの検証ではない|

この旧系列はP=0、full Hadamard measured wrapper、prefixとDF rankを区別する。必要shotは解析式であり実機shot実行ではない。
元M1/PM familyでのB2点最小を、後のB0+I-inclusive familyの全点優位へ読み替えない。
一次artifact/source/result commitは[履歴索引](track_a_local_sync_reference_index.md)に固定した。公開済み原本を重複commitしない。

## 旧H6失敗とexportの履歴（完了結果とは別identity）

|記録|保存状況|統合候補での扱い|
|---|---|---|
|旧v2 MemoryError|result8bd903…、449/592探索group、902wrapper|STOPを保全、後続server完了とは別run|
|v3 launch_v1|出力親directory欠落で科学Phase A未開始、新結果0|consumed/retired grantを再利用しない|
|launch_v2|Phase A B1 full19/S4/q8、cosine transpileでallocation abort、exit−6、新complete0/不足143、worker terminal欠測|result3dc947…/final054bb7…を原まま保全、Memory対策成功とは表示しない|
|old-environment exporter 2026-10-12|source packetb53271f…、snapshot054bb7…、EXPORT_STOP/FROZEN_PARTIAL_450_REPRODUCTION_MISMATCH|[terminal exact bytes](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/old_environment_export_terminal_original.json)。rawevent/digest未保存、原因UNDETERMINED、実呼出し数unknown|

後のサーバー探索592/592と独立確認は別execution source78a390…/87b6b9…の公開結果。失敗runの記録を成功に直さず、同じgrantのretryと推定しない。
旧export sourceは既公開。local ZIP/venv/runtimeは同期せず、terminalのみpublic metadataへexact copy。
旧報告にある7準備/codec gate通過は制御順に基づく推論で、raw保存による独立再現ではない。
export診断v2案（再生成eventをgate前保存してdirect/codecを分離）は将来候補、未実装・未認可。

## 留保・未検証仮説・今後の候補

H4比較familyによるwinnerの変化、ordinary/structuredとnativeのscope差、H6探索8とfresh32確認、H8snapshotと最終結果を区別する。
同一identityの数値矛盾は今回確認されていない。新規性/母平均winner/追加検証の科学的採否はGPTが判断する。
H8を凍結モデルの独立予測検証と呼ぶための未観測予測artifact・契約は今回取得範囲では未確認。
H8 current live/final status、old partial450 raw、server run03a324公開到達性、local working note採択はUNVERIFIED。
今後の基本候補H4〜H8・1Å、H5/H7追加、H9以上保留、ε=.05/.01/.005/.001と必要なら厳格化、T=.4/.8/1.6、B0+I/full S2/S4、RZ/CX/depth/shot/P感度はuser方針とserver初版で一致。
奇数鎖開殻・rank/state/候補集合・時間/精度拡張の科学契約は後段で固定する。この一致は計算認可ではない。
今回の新科学計算・sampling・compile・H8変更・grant発行は0。`next_stage_authorized=false`、資料公開後mandatory STOP。

---

## 基礎となるサーバー初版（固定commitの本文をそのまま収録）

以下は `2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c` の `docs/research/track_a_shared_research_status.md` 本文。内包された「ローカル未確認」は**server初版の時点**、H8状態は**snapshot時点**の記述である。
この資料内では[ローカル差分報告](track_a_environment_sync_differences.md)を併読する。原本を正式採択済みと扱わず、観測時点を現在へ更新しない。

# Track A 共通研究情報の入口

2026-10-14 JST、**サーバー観測に基づく暫定初版**。ローカル側との照合は未完了。GitHubを共有参照先とするが、本書は固定科学SOURCE・原本結果・provenance・承認済みレビューの索引であり、それらを置換しない。
原稿・論文化を進める作業ではなく、研究情報の棚卸しと引継ぎである。

最初に本書、[サーバー棚卸し](track_a_server_inventory.md)、[同期ルール](track_a_environment_sync_policy.md)、[ローカルCodex handoff](track_a_local_codex_sync_handoff.md)を読む。
背景は[研究概要・現状](研究概要・現状.md)、[目的・課題](研究目的・研究課題.md)、[validation status](../../VALIDATION_STATUS.md)を参照。ただし各branchの概要にある「未実行」「起動済み」は当時の記録であり、以下の固定結果・観測時点と照合する。

## 研究目的と現在の方針

既知のpartially randomized time evolutionについて、固定DF表現、PF/control/Gaussian実装、有限時間complex signalの精度、normalization、必要shot、compiled費用を接続し、低資源となる条件と適用範囲を調べる。
PRが必ず勝つこと、新algorithmの発明、厳密母平均certificateを現在の達成条件にしない。εはsignal精度、主指標は状態準備P=0のshot×compiled RZであり、最終energy/QPE/RPE総資源の評価とは区別する。
H4 geometry/ordinary/structured系列と、H4 anchor・H6/H8 native/symmetric-directional系列を同一視しない。[条件対応](track_a_resource_comparison_conditions_20261014_v0_4.md)でDF・PF・reference/u・標本・選抜設計を分ける。

## 段階を分けた現在地

|研究単位|計画・準備|科学実行|結果整理|GPT科学判断|次の担当・STOP|
|---|---|---|---|---|---|
|H4 geometry 元B0/B1/B2/B3|固定済み|全74,784 logical wrappers、1,308候補完了|302精度×6距離の保存値地図公開|scalar保持対照を追加評価する既存判断を継承|追加科学計算STOP。元集合の証拠を保持|
|H4 B0+I全面地図|追加対照として採択・別利用者実行承認済み|全240 wrappers完了、234新compile＋6照合再利用|397 B0+I/1,415 B2の点最小、v0.4図表・handoff公開|対照採択レビュー済み。全面結果後の主要claim・新規性・論文化は判断待ち|GPTが結果判断。追加launchなし、mandatory STOP|
|H4 1Å native anchor|16固定panel|PR/full S2/S4の532 compile完了|固定panel結果公開|H6/H8との条件差を明示して使う|新grantなし。全deterministic最適化と呼ばない|
|H6探索|92候補・592 group固定|592 group完了、1,188 wrappers|探索8本のmap公開|独立確認と希少次数診断へ進む既存判断|探索は完了STOP、確認結果と別標本|
|H6独立確認|探索上位union8構成を事前固定|別seed32本×8、256 group/512 wrappers完了|選択集合内の費用と536行算術照合公開|留保付き資源benchmarkとして研究継続、母平均winner保留|H6追加samplingは後続pilotレビューで一旦停止|
|H6条件付きs2 pilot|q8/r2/K2の非zero order2層だけ別認可|8 trajectory/16 compile完了|局所平均補正+0.00734086%公開|追加H6 samplingを必須工程にせず既存証拠を統合する判断|科学実行STOP。補正をH4/H8へ移送しない|
|H8 input/state・feasibility|rank28/sector4900入力・state準備完了|8 cases/20 wrappersのfeasibility完了|固定結果公開|feasibilityはmatched-accuracy勝敗ではない|既存coreとは別scope・証拠|
|H8 core resume02|既存core grantとcomplete-only継承固定|**読取snapshotで探索継続**。92 signal、203/592探索group、8 active workers|最終確認・総結果は未取得|core結果の判断は未実施|既存監視下のH8を変更せず継続。本taskから停止/再起動なし|
|サーバー/ローカル情報統一|今回認可された資料作業|科学計算0|本暫定初版|科学レビューを代行しない|ローカルCodexが照合。資料公開後STOP|

H8数値は2026-10-14の[read-only snapshot](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/h8_readonly_snapshot.json)に固定した一時点。
祖先126 COMPLETE＋resume02新77 COMPLETE=203、残389探索group。92 signalは4受理済みsignal groupに含まれる。探索後の独立32確認はこの観測ではまだ確認されていない。
coordinator PID/start identity一致、8 own-user worker生存、保存first STOP理由null、265 source hashes一致を確認した。これはnon-atomic metadata読取で、完了・母平均・将来の稼働状態の認定ではない。
既知のproduction設計 `eaeeefc3…` を最新runのSOURCEと誤認せず、resume02を参照する。

## 主要成果と固定証拠

|単位|実行SOURCE / 科学identity|結果・資料commit|短い結果と証拠|
|---|---|---|---|
|H4 geometry|実行 `d0726e946f89f8c30a08d383fb8ff48b363629ec`、data `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`|`7c4483378047b57f514c47dad98905e3f2a9a756`|[元結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md)：B0/B1/B2/B3のみなら全1,812点B2点最小|
|H4 B0+I|`924b779c2bc6f6e844d6283c355f91c73aed6445`、dataは上記5aa|結果 `98cdb35cd08469714f20d6cc3b982d9c80908f46`、統合 `47dc17a83e2fc9216e22b59317969952ef79028a`|[実測結果](track_a_h4_b0i_map_results_20261014.md)、[統合](track_a_h4_b0i_manuscript_integration_20261014.md)：全240軸の旧B0 metric差0。追加後397/1415、374点は記述的B2−2SEも下回る|
|H4 anchor|`217c971dd537b1cc7fbc13742bc0f6c263ff7f45`|`0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`|[固定result summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/result_summary.json)：B0不在の16固定panel、全H4最適化ではない|
|H6探索|実行 `78a39051d8b6d41c33d592ecfe172f8824adfaeb`、科学 `5a903aea2492c2d211c1e15e7737df0d50694c12`|`df99d6ee28942df8d12aac3c4db7cd202df55d77`|[探索報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df99d6ee28942df8d12aac3c4db7cd202df55d77/docs/research/track_a_h6_exploration_result_v1.md)：各PR8本、92候補、独立確認前の順位|
|H6確認|実行 `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、科学5a903|`321cd8c3066194fe98741cbebbcdba1515d1deaa`|[確認報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_independent_confirmation_result_v1.md)：確認集合内PR/full S2/S4点費用比約0.37〜0.46、母平均未認定|
|H6 s2 pilot|`843265b624eac8fbb9acea47ef21cab1207e1880`|`e0e6473d7047158a7f0f788d026a7a89a01898b8`|[pilot報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0e6473d7047158a7f0f788d026a7a89a01898b8/docs/research/track_a_h6_conditional_s2_pilot_result_v1.md)：ε=.001、対象構成PR/PF約0.387034。全support保証ではない|
|H8 feasibility|`9bfea8bde44cbb2ea9187476ddd496066a8853b3`|`a55da5c2b012912c485184845aa73ca053ac00b9`|[固定feasibility](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a55da5c2b012912c485184845aa73ca053ac00b9/docs/research/track_a_h8_fast_feasibility_result_v1.md)：8 cases/20 compile。正式shot mapではない|
|H8 core resume02|実行 `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、seed `cb8a56c6b2a772827275f9200b02638030f33f72`|起動資料 `38d386c8732295e6fab5d6fcbd47fbe586dc207d`、最終result commit未確認|[最新run資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38d386c8732295e6fab5d6fcbd47fbe586dc207d/docs/research/track_a_h8_resource_map_resume02_v1.md)：snapshot時点で探索継続|

H4数値のscope：linear/STO-3G、DF12、6距離、T=.8、prefix3/4/5/6/9、δ=.1〜.8、S2、ordinary、structured、P=0。
H6はlinear1Å/STO-3G、DF19/tol1e-8、prefix5/10/15/full19、δ=.1〜.8、partial S2対full S2/global Yoshida S4、symmetric-directional/native、経験的u/B_upper、P=0。
H8はlinear1Å/STO-3G、DF28/tol1e-8、prefix7/14/21/full28、同δ窓とnative系列。異なる条件を同期のために同じ結果へまとめない。
input H/DF/state/compiler identity、file単位commit/Git blob/bytes/SHAは[科学identity projection](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/science_identity_projection.json)と[固定証拠索引](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/fixed_evidence_index.json)を参照。

## 既にある科学判断と判断待ち

- H4 B0+Iを追加評価対象に採択し、元B0・元集合を保全する判断：[直前レビュー](../../artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/review_context/previous_scientific_review_20261013.md)。実測後の主要claimは[v0.4 handoff](../manuscripts/track_a_h4_b0i_v0_4_scientific_review_handoff.md)で判断待ち。
- H6確認後の研究継続・母平均winner保留：[固定レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_scientific_review_20261013.md)。
- H6 pilot後は追加samplingを一旦停止し留保付きbenchmarkとして統合：[後続レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38d386c8732295e6fab5d6fcbd47fbe586dc207d/docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md)。レビュー当時のH8「未開始」は後の別実行承認で更新された。レビュー原文は上書きしない。
- 共通の留保：H4の同じ32本での選抜/推定、event履歴NOT_SAVED・172丸め候補未解決、H6/H8の希少母平均・経験的u、旧compiler bit同一性、energy/RPE総費用。共有ノートでこれらを解消したことにしない。

## 将来の設計候補（今回の利用者指示、科学実行未認可）

距離は原則1.0 Å、H4〜H8を基本対象。H5/H7奇数鎖を検討し、H9/H10以上は当面保留。
基本ε候補=.05/.01/.005/.001、より厳しい精度の必要性、T=.4/.8/1.6依存性を検討する。
B0+I/full S2/S4等の対照、RZ/CX/depth/shot/P感度、H4/H6/H8の条件差を解消または明示する案を整理する。
奇数鎖の開殻処理・新しい比較条件・追加実行scopeは決めていない。既存H8 core grantを拡張せず、実験済み・科学レビュー承認済みとは表示しない。

## 後続Codexの照合先

GitHub branch/commitとfile blobの一覧は[棚卸しbundle](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/README.md)。[研究ノート索引](研究ノート/README.md)は本branchの履歴入口で、H6/H8の最新ノートは固定各branchの同pathを参照する。
ローカル側の未push commit・dirty・review・新計画は今回未知。差分は[handoff](track_a_local_codex_sync_handoff.md)に沿って別資料として返し、両側照合が済むまで本書の暫定ラベルを外さない。
