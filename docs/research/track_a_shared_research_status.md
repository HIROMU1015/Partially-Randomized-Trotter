# Track A 正式共通研究ノート

2026-10-15 JST追補。正式共通版 `4675730e7e40d25ba3618334a4a0e28b4bd2f145` を基準に、指定公開資料とサーバー保存metadataを照合した。[今回の確認結果・未解決事項](track_a_server_information_update_20261015.md)、[限定受領監査](../../artifacts/resource_applicability/track_a_shared_status_server_update/2026-10-15/README.md)。
2026-10-14のv1は、server固定初版 `2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c` とlocal照合版 `8e84398995bb162bc74018c1b66fbb14448c7d27` を利用者指示で統合したもの。旧inventory・候補・原reviewは保全する。
**情報ノートとして正式採択**。原科学SOURCE・結果・review・grantの正本を置換せず、未検証事項や将来案を科学的に採択したことにはしない。
初読は本書 → [科学レビュー判断](track_a_scientific_review_decisions.md) → [条件対応](track_a_resource_comparison_conditions_20261014_v0_4.md) → [source/result索引](track_a_local_sync_reference_index.md) → [同期ルール](track_a_environment_sync_policy.md)。
[server初版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c/docs/research/track_a_shared_research_status.md)、[server inventory](track_a_server_inventory.md)、[local inventory](track_a_local_inventory.md)、[差分](track_a_environment_sync_differences.md)、[旧統合候補](track_a_shared_research_status_candidate.md)は履歴として保全した。

## 研究目的・評価対象

既知のPartially Randomized Trotterを量子化学DF Hamiltonianへ実装し、**同じ有限時間complex-signal精度での測定込み回路資源と適用範囲**を調べる。
対象は保存されたDF Hamiltonian・正規化指定stateの `z(T)=<ψ|exp(-i H_DF T)|ψ>`。DF近似と元化学Hamiltonianの差、基底状態certificate、最終energy推定精度は別問題である。
候補のbias、finite-RTE normalization、各軸の既定shot規則、実compileしたHadamard full-wrapper費用を接続する。
主指標はP=0の `G_RZ=Σ_a N_a C̄_RZ,a`。Nはbias/headroom/normalizationに基づく解析的な十分shot数、C̄は固定標本のcompiled one-shot費用。
CX/depth等と仮想状態準備費用P感度は補助。RZ countはT/Toffoli数・物理runtimeと同義ではなく、実機でN shotsを実行した総費用ではない。
εはfinite-time signal精度、Haではない。最終energy/QPE/RPE総資源や未知分子でのtruth-free selectorを評価済みとしない。
主RQはRQ-R、RQ-P1は補助、RQ-P2は未達成・完成必須条件外（R01）。PR必勝、新PR algorithm、厳密母平均winnerを達成条件にしない。

## 比較scopeと固定identity

|系列|Hamiltonian・geometry・DF・split・時間/δ窓・実装・検証scope|
|---|---|
|M1/PM旧H4|linear/STO-3G/legacy DF12、development1Å218、transfer1.3Å固定5。prefix0/3/6/9/12（PM-1で4/5追加）、T=.8、q1/2/4/8・δ=.1〜.8、S2/canonical finite-RTE。保存32 paired費用・302精度.005〜.1、P=0|
|H4 geometry/B0+I|linear/STO-3G/legacy DF12、.70/.80/.90/1.10/1.40/1.60Å、prefix3/4/5/6/9（full12・endpoint0別）。T=.8、δ=.1〜.8、S2、ordinary/structured Gaussian。32 paired標本、302精度.005〜.1、明示u差引なし、P=0|
|H4 native anchor|linear1Å/STO-3G/DF12、PR prefix3/full PF12、T=.8・δ=.1〜.8、S2/S4を含む16固定panel。native/symmetric-directional、P=0、B0不在。geometry全domainに合算しない|
|H6|linear1Å/STO-3G/12modes/α3β3/sector400、tol1e-8・rank19・cutoff0・returned order・weighted Hermitian projection、prefix5/10/15/full19。T=.8、δ=.1〜.8、partial S2 vs full S2/global Yoshida S4、native/symmetric-directional、経験的u/B_upper、P=0。探索8と確認32を分離|
|H8|linear1Å/STO-3G/rank28 tol1e-8/sector4900、prefix7/14/21/full28、T=.8・q1/2/4/8・δ窓.1〜.8、native系列・経験的u/B_upper・P=0。core探索92/確認union6は完了。D0の5精度・6指標・P解析は後続の別保存値解析|

H4 geometryとnative H4/H6/H8は別の証拠群であり、同一統制size-scaling curveへpoolしない。
科学/input/compiler/state identityは[server projection](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c/artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/science_identity_projection.json)、file単位commit/blob/bytes/SHAは[固定参照一覧](track_a_local_sync_reference_index.md)から辿る。arrayを今回独立再検証したものではない。
H4 geometry source `d0726e946f89f8c30a08d383fb8ff48b363629ec`、data `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`、B0+I source `924b779c2bc6f6e844d6283c355f91c73aed6445`。
H6 science `5a903aea2492c2d211c1e15e7737df0d50694c12`、探索execution `78a39051d8b6d41c33d592ecfe172f8824adfaeb`、確認execution `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`。
H8 resume02 execution `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、seed `cb8a56c6b2a772827275f9200b02638030f33f72`、起動資料 `38d386c8732295e6fab5d6fcbd47fbe586dc207d`。旧production eaeeefc…を現execution sourceにしない。

## 研究履歴・到達点

状態タグは別軸で併記する。COMPLETEは登録実行完了、RESULT_DOCUMENTEDは保存結果公開、SCIENTIFIC_REVIEWEDは明示scopeのGPT判断済み。
PREPARATION_COMPLETEは実行完了ではない。REVIEW_REQUIREDは新結果/未確認事項の判断待ち。MANDATORY_STOPは当該終了run/今回資料作業の状態で、既存H8を停止する指示ではない。

|研究単位|段階|主要内容・限定|固定根拠|
|---|---|---|---|
| M1/M2 | COMPLETE / RESULT_DOCUMENTED | H4 rank12、1Å development210と1.3Åの事前固定5 transfer、T=.8/S2/canonical finite-RTE。 | [M1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_matched_accuracy_m1_b1_result_validation.md) / [M2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_matched_accuracy_m2_transfer_result_validation.md) |
| PM-0/1/2 | COMPLETE / RESULT_DOCUMENTED | PM-0 same-R事後解析、PM-1近接B0を8構成追加して218、PM-2保存値302精度再会計。独立試験に合算しない。 | [PM-0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/pr2_post_m2_evidence_attribution.md) / [PM-1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_pm1_discard_result_validation.md) / [PM-2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_pm2_precision_resource_result_validation.md) |
| AX-0/AX-1/AX-1b | COMPLETE / RESULT_DOCUMENTED / SCIENTIFIC_REVIEWED | 契約と保存値モデル照合。RQ-R主、RQ-P1補助、RQ-P2未達。新signal/compileではない。 | [AX-0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_ax0_research_contract.md) / [AX-1b判断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_ax1b_post_scientific_review_2026-10-09.md) |
| H4 geometry元4方式 | COMPLETE / RESULT_DOCUMENTED / SCIENTIFIC_REVIEWED / MANDATORY_STOP | 6距離×302点。74,784 logical wrappers、1,308候補。元B0/B1/B2/B3集合で全1,812点B2点最小。 | [result7c448337](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md) |
| H4 B0+I追加 | COMPLETE / RESULT_DOCUMENTED / SCIENTIFIC_REVIEWED / MANDATORY_STOP | 240軸＝234新compile＋6照合再利用。拡張family点最小397 B0+I /1,415 B2。R09採択は限定benchmark。 | [result98cdb35c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md) / [10/14 review](track_a_h4_b0i_v04_scientific_review_20261014.md) |
| H4 native anchor | COMPLETE / RESULT_DOCUMENTED / MANDATORY_STOP | 1Å固定16panel、532 compile。B0不在、全deterministic最適化ではない。 | [result0f96a2e4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/result_summary.json) |
| H6探索 | COMPLETE / RESULT_DOCUMENTED / MANDATORY_STOP | 92候補、592 group、1,188 wrapper。random探索8本。 | [resultdf99d6ee](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df99d6ee28942df8d12aac3c4db7cd202df55d77/docs/research/track_a_h6_exploration_result_v1.md) |
| H6独立確認 | COMPLETE / RESULT_DOCUMENTED / SCIENTIFIC_REVIEWED / MANDATORY_STOP | 選抜union8×独立32、256 group/512 wrapper。探索8とpoolせず、確認集合内point comparison。 | [result321cd8c3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_independent_confirmation_result_v1.md) |
| H6希少次数・s2 pilot | COMPLETE / RESULT_DOCUMENTED / SCIENTIFIC_REVIEWED / MANDATORY_STOP | 保存確率解析とq8/r2/K2非zero order2条件付き8trajectory/16compile。補正は局所scopeのみ。 | [rare-order](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md) / [pilote0e6473d](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0e6473d7047158a7f0f788d026a7a89a01898b8/docs/research/track_a_h6_conditional_s2_pilot_result_v1.md) |
| H8 input/state・feasibility | PREPARATION_COMPLETE / COMPLETE / RESULT_DOCUMENTED | rank28/sector4900、8case20wrapper。core最終accuracy-matched mapではない。 | [resulta55da5c2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a55da5c2b012912c485184845aa73ca053ac00b9/docs/research/track_a_h8_fast_feasibility_result_v1.md) |
| H8 core resume02 | COMPLETE / RESULT_DOCUMENTED / SCIENTIFIC_REVIEWED / MANDATORY_STOP | signal92、探索592group、確認union6×独立32＝192group。探索8とpoolしない。旧203/592 snapshot、受領時556/592 snapshotを経た後続完了記録。 | [結果681ec23c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md) / [サーバー受領2ef5ecbd](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ef5ecbdbdc222d73d9106618e20dc27b5466ea2/docs/research/track_a_cross_environment_server_acceptance_20261014.md) |
| H4/H6 D0保存値解析 | COMPLETE / RESULT_DOCUMENTED / SCIENTIFIC_REVIEWED / MANDATORY_STOP | H4 native fixed16、H6探索92/確認union8。旧precision1088・cost2659一致後、ε=.0005を含む5精度・6指標・P解析。新signal/compileではない。 | [結果8dbdb682](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dbdb68293c988a999645e21a82b53966f9ef6cd/docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md) |
| H8 D0保存値解析 | COMPLETE / RESULT_DOCUMENTED / MANDATORY_STOP | 別承認・SOURCE20f89bc3。旧precision944・cost3576一致後に5精度/6指標/Pを再会計。H4/H6 D0再実行0。R10原reviewの「H8 .0005未知」は実行前の履歴。 | [結果3d910ee1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_saved_value_result_20261015.md) |
| H6 native B0+I pilot | SCIENTIFIC_REVIEWED（方針） / AUTHORIZED / SEALED / COMPLETE_METADATA_RECEIVED | 準備Fの未認可状態から後続grantへ。run04-v3、SOURCE2e733d8a、3構成6軸。16:52 JST頃のpartial4/6後、17:03 JST以降に6軸結果・COMPLETE/driver wait0/全worker reapを受領。費用の新科学採択は未実施。 | [起動・修正d1fa0974](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1fa0974064ae4feeb1c8168a88b43f89fc58340/docs/research/track_a_h6_proc_exit_fix_20261015.md) / [認可・4h方針差の照合](track_a_server_information_update_20261015.md#h6の認可資源実行記録) |
| 追加研究候補 | SCIENTIFICALLY_ADOPTED / EXECUTION_SEPARATE | R10がH5先行/H7後続、H6固定δのT比較、native対照拡張を採択。H6現pilotを超えるlaunch認可の根拠にはしない。 | 下記の採択と認可の区別 |

旧H4/M1/PMの原稿v0.1とH4 B0+I/v0.4は保存draftとして保全。本作業では原稿・図表を改訂しない。
M1 source33f436bb…/result8e0814e7…、PM-2 source324435d7…/result5a1adffa…、AX-1b sourcefc297cd9…/resultb2e1bf65…は[履歴索引](track_a_local_sync_reference_index.md)にfull SHAと一次artifactを保存した。

## 科学レビュー済みの現在方針

日付・scope・採択/不採択/保留・原本・追補関係は[review判断索引](track_a_scientific_review_decisions.md)のR01–R10に記録した。

- **10/9 AX-1b（R01）**：RQ-R主、RQ-P1補助。追加fitだけの拡張を優先しない。operational selector未達成を別扱い。
- **10/10–11 H4/H6技術・DF・pilot（R02–R04）**：登録scopeの部分証拠を採用、weighted DF政策と同信号精度S2/S4比較を具体化。旧STOPを成功へ変更せず、実行認可は別。
- **10/13 H6確認→s2 pilot（R07→R08）**：fresh32確認と条件付き費用の証拠を統合。追加H6 samplingを現時点の必須工程にせず留保付きbenchmarkを採用。厳密母平均・希少component上界は未保証。
- **10/13 H4 geometry→B0+I（R05→R06）**：既知scalar保持の追加対照を採用し、元B0と旧集合を保全。実測domainを固定し、計算grantと区別。
- **10/14 H4全面実測後（R09）**：[取得したGPT原review](track_a_h4_b0i_v04_scientific_review_20261014.md)に従い、対照を強化したPRのgeometry/精度依存資源領域（claim B）を主、397/1415件数（A）を補助、別条件H6/H8事例（C）を構成として採用。一般的PR全勝、新algorithm/scalar初提案は採用しない。
- **10/15 H8・D0統合（R10）**：[固定原review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)がH8/H4-H6 D0を条件付きbenchmarkとして採用。H8 D0を先行、その後H6 native B0+I 6軸pilot、各系24軸対照、H5→H7、H6固定δ=.2/T=.4/.8/1.6を科学的に採択。ε=.0005、6指標/P感度を採用し、高精度ほどPR有利という一般則・q16/ε=.0001自動追加・全support保証の必須化は採用しない。原稿化は後回し。原reviewは実行認可を含まない。

R09は追加H4独立32、全support保証、172候補復元、高次PF全domain取得を現時点の必須工程にしない。H4の科学編集開始にH8完了を必須としないが、全体結論へH8を含める際は結果reviewが必要。
「最良決定論対PR」の分母は元B0/B1/B0+Iの最小であり、B0+I単独比と分ける。matched B0→I効果とfamily内再選択も分ける。既存shot/費用・主要指標を今回変更しない。
これらは**既存GPT判断の記録**で、Codexが新科学方針を決めたものではない。レビュー完了はCPU使用/launch/grantの認可を含まない。

## 失敗・STOP・未公開証拠の保全

旧H4 v3/v4 STOP、H6 DF無重みgate STOP、技術pilot32/36 STOP、旧H6 matched v2 MemoryError449/592・902wrapper、v3親directory起動STOP、launch_v2 Phase A allocation abort/newcomplete0を別runとして保全。
後続server探索592/592と確認は別execution identity。旧grant流用、未完了taskの成功書換えとは表示しない。[旧記録索引](track_a_local_sync_reference_index.md)。
旧exportはsourcepacketb53271f…/snapshot054bb7…、EXPORT_STOP/FROZEN_PARTIAL_450_REPRODUCTION_MISMATCH。[原terminal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8e84398995bb162bc74018c1b66fbb14448c7d27/artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/old_environment_export_terminal_original.json)は330 bytesをexact公開したが、再生成event/digest未保存、原因UNDETERMINED、actual call数unknown。診断v2は未実装・未認可。
LOCAL_ONLY44実体/16内容のplan・dirty noteはhash索引で保全し、未公開原本を実行sourceや採択計画へ昇格しない。private array/runtime/venvとserver run03a324…到達性は別の未確認/環境固有事項。

## 未解決事項・取得待ち

- H4同じ32標本での選抜/推定、NOT_SAVED Taylor/component履歴、B=1丸め172候補。H6局所補正をH4/H8に移送しない。
- H6/H8経験的u/B_upper、希少order/component、有限標本期待費用・population winner。技術的PASSをcertificate/同時保証へ昇格しない。
- H8 core最終結果681ec23cとR10、後続H8 D0結果3d910ee1は取得済み。旧203/592・8workerは2026-10-14の非atomic観測であり、完了結果で置き換えず時点履歴として保持する。
- H6 pilotは認可・途中snapshotの後、全6軸の終端metadataを受領。詳細費用の公開・科学的解釈は未実施。H6/H8各24軸全面native B0+I比較の完了を意味しない。
- 共通指示：server原本50,268 bytesは確認・exact copy追加。local申告48,223 bytes版は実体未受領で直接差分未確認。H6の既存sealed契約stage wall4hと将来wallなし既定方針には明示例外の確認待ちがある。[来歴・方針差](track_a_server_information_update_20261015.md)。
- AX-1bが提案した凍結モデル・未観測予測によるH8検証artifactは未取得。資源map完成をRQ-Pの独立検証完了と自動認定しない。
- H4/H6/H8のDF/control/state/基線探索/数値政策の条件差、旧export不一致、旧compiler bit再現性、最終energy/QPE/RPE費用は残る。
- R09の補助算術ZIP/scriptは今回未取得。review原本は取得・exact公開済みで、補助物未取得をreview本文欠落と混同しない。

## R10による採択と実行認可の区別

|項目|方針候補|
|---|---|
|geometry・対象|1.0Å/H4〜H8。中性ROHF共通空間軌道のH5を先行しH7はH5 correctness/実資源確認後。科学採択であり両系のlaunch認可は本追補で取得していない。H9以上/追加geometry対象外|
|要求精度|ε=.05/.01/.005/.001/.0005を採択。保存値D0完了と、新signal/q16/ε=.0001の未認可を区別|
|時間|H6 δ=.2固定、T=.4/.8/1.6・q2/4/8を主比較として採択。旧q2固定案は後順位。新48compile/PR16trajectory/320occurrence案は別予算・承認が必要|
|対照・指標|native B0+Iを強化対照として採択。H6 6軸pilotに既存認可あり、H6/H8各24軸の全面比較は別認可。RZ主・6指標/P補助、matched旧B0自動追加なし|
|H8|coreとD0の完了資料を取得。今回停止/再起動・SOURCE/grant/環境変更なし|

H5/H7のtarget/orbital/DF政策はR10で科学的に採択済み。実input・固定SOURCE/予算/CPU/新grantを得たとの意味ではない。H6実行契約も本追補では変更しない。
情報統一後の詳細研究設計と、原稿科学編集は**別作業**。本taskから自動開始しない。

## 両環境への引継ぎ

server Codexは本branchの固定commitと[検証bundle](../../artifacts/resource_applicability/track_a_cross_environment_canonical_status/2026-10-14/README.md)をread-only取得し、入口・sync policy・R09原reviewのbytes/hashを照合する。
H8最終resultとD0、H6後続認可の受領は[10/15追補](track_a_server_information_update_20261015.md)で解決。run03a324…公開到達性、local指示原本実体、H6 wall方針差と結果の科学的解釈は別の未確認として残す。sourceや稼働jobは変更しない。
研究ノートの更新は[同期ルール](track_a_environment_sync_policy.md)に従い、原review/結果を保存して日付付き追補と固定証拠で行う。
今回の新SCF/DF/state/signal/trajectory/compile、H8操作、新grant発行、原稿改訂はすべて0。`next_stage_authorized=false`、公開後MANDATORY_STOP。
