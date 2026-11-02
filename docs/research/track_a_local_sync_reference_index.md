# Track A ローカル照合 source・result・資料参照一覧

2026-10-14 JST。固定SHAへの索引。資料の再実行や科学的な正しさの追加認定ではない。
共有入口：[server初版](track_a_shared_research_status.md) / [統合候補](track_a_shared_research_status_candidate.md) / [差分](track_a_environment_sync_differences.md)。
server branch `track-a-cross-environment-sync-server-20261014` / `2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c`、local資料branch `track-a-cross-environment-sync-local-20261014`。
[source・input・result identityのserver projection](../../artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14/science_identity_projection.json)は保存metadataとして取得した。arrayを独立にdecode/rehashしたものではない。

## サーバー主要19証拠

全19件のGitHub取得bytesと宣言Git blob/bytes/SHA-256が一致。[照合JSON](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/github_reference_verification_v1.json)は19件＋入口等、計37件。
科学的数値は各ファイルのHamiltonian・geometry・DF policy・prefix・δ窓・sampling・compiler scopeに従う。

|固定文書・一次artifact|保存commit|Git blob|
|---|---|---|
| [docs/research/track_a_h4_geometry_results_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md) | `7c4483378047b57f514c47dad98905e3f2a9a756` | `0ed30754a816dad2825f4461dc2c852e016f45a7` |
| [docs/research/track_a_h4_b0i_map_results_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md) | `98cdb35cd08469714f20d6cc3b982d9c80908f46` | `ffc79f3725000f2cee8035fdde842c658fc26b1f` |
| [artifacts/resource_applicability/track_a_h4_b0i_map_results/2026-10-14/summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/artifacts/resource_applicability/track_a_h4_b0i_map_results/2026-10-14/summary.json) | `98cdb35cd08469714f20d6cc3b982d9c80908f46` | `677f55396d3592a57e49e625677c4f176ab8b39f` |
| [docs/manuscripts/track_a_resource_study_geometry_native_v0_4.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47dc17a83e2fc9216e22b59317969952ef79028a/docs/manuscripts/track_a_resource_study_geometry_native_v0_4.md) | `47dc17a83e2fc9216e22b59317969952ef79028a` | `1cf7943dc31b59fb4b08916e09b99e401e7a1c22` |
| [docs/manuscripts/track_a_h4_b0i_v0_4_scientific_review_handoff.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47dc17a83e2fc9216e22b59317969952ef79028a/docs/manuscripts/track_a_h4_b0i_v0_4_scientific_review_handoff.md) | `47dc17a83e2fc9216e22b59317969952ef79028a` | `573db93abb306d8bb0286a2348d4603746a1e76f` |
| [artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/review_context/previous_scientific_review_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47dc17a83e2fc9216e22b59317969952ef79028a/artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/review_context/previous_scientific_review_20261013.md) | `47dc17a83e2fc9216e22b59317969952ef79028a` | `8df32b3ae65cf1fcc8c13f70449146a2df9b601d` |
| [docs/research/track_a_h6_exploration_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df99d6ee28942df8d12aac3c4db7cd202df55d77/docs/research/track_a_h6_exploration_result_v1.md) | `df99d6ee28942df8d12aac3c4db7cd202df55d77` | `4eb88d92c8927d962a0d406191a6cc8b158232e7` |
| [docs/research/track_a_h6_independent_confirmation_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_independent_confirmation_result_v1.md) | `321cd8c3066194fe98741cbebbcdba1515d1deaa` | `a467dd6b057f42ad8e9ec6316fdce69796dbdb32` |
| [docs/research/track_a_h6_rare_order_cost_reliability_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md) | `eee1343f6e99aa7661892ddfbe1067e974e2b74a` | `48a6f9df95a4eaa84b04189dd8481ad2e8c52331` |
| [docs/research/track_a_h6_conditional_s2_pilot_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e0e6473d7047158a7f0f788d026a7a89a01898b8/docs/research/track_a_h6_conditional_s2_pilot_result_v1.md) | `e0e6473d7047158a7f0f788d026a7a89a01898b8` | `1d72c87737aed82ebeefa9d73983b69179b0156b` |
| [docs/research/track_a_h6_independent_confirmation_scientific_review_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_scientific_review_20261013.md) | `0dd5c3d204dab9b731268a1aa2add621080f3d20` | `3615f405fd925020b8cbae241e3f80e10abb0cc4` |
| [docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38d386c8732295e6fab5d6fcbd47fbe586dc207d/docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md) | `38d386c8732295e6fab5d6fcbd47fbe586dc207d` | `5cf111199e336053479a86dd9e17751cb1928fa2` |
| [artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/result_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/result_summary.json) | `0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41` | `225bc99c18a24d4b7b7ddd411088496a9fe01fc0` |
| [docs/research/track_a_h8_fast_feasibility_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a55da5c2b012912c485184845aa73ca053ac00b9/docs/research/track_a_h8_fast_feasibility_result_v1.md) | `a55da5c2b012912c485184845aa73ca053ac00b9` | `375392c086cd608ece665b40596dc7d5ce8ab89a` |
| [docs/research/track_a_h8_resource_map_production_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eaeeefc38be454375d141fabfb8ae229498175b9/docs/research/track_a_h8_resource_map_production_v1.md) | `eaeeefc38be454375d141fabfb8ae229498175b9` | `a1a10aa3af1e9d759a4a3c4c26912d160a8e3879` |
| [docs/research/track_a_h8_resource_map_resume01_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96b9560e6bc02a71a5917db5d0cf911a69aec3de/docs/research/track_a_h8_resource_map_resume01_v1.md) | `96b9560e6bc02a71a5917db5d0cf911a69aec3de` | `3ddcd2a7e7603273502e71716c7b4e26daca0f04` |
| [docs/research/track_a_h8_resource_map_resume02_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38d386c8732295e6fab5d6fcbd47fbe586dc207d/docs/research/track_a_h8_resource_map_resume02_v1.md) | `38d386c8732295e6fab5d6fcbd47fbe586dc207d` | `062e50eff7ec4226b998e644d8b3ee9501573629` |
| [artifacts/resource_applicability/track_a_h8_resource_map_resume02_v1/2026-10-14/launch_status.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38d386c8732295e6fab5d6fcbd47fbe586dc207d/artifacts/resource_applicability/track_a_h8_resource_map_resume02_v1/2026-10-14/launch_status.json) | `38d386c8732295e6fab5d6fcbd47fbe586dc207d` | `5636fc69eb9567e8949cce51ada2bf424c7ef31f` |
| [docs/research/track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md) | `054bb7b2b2ca788b47ebdc5c14c115f24c8b6379` | `c90761202859686b00899d53d48981985e88c950` |

## 科学sourceと結果commitの対応

|系列|science / execution source|結果・scope|
|---|---|---|
|H4 geometry|`d0726e946f89f8c30a08d383fb8ff48b363629ec`、input `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`|7c448337…、linear/STO-3G/DF12・6geometry・T=.8・prefix3/4/5/6/9・δ=.1〜.8・S2 ordinary/structured・P=0|
|H4 B0+I|`924b779c2bc6f6e844d6283c355f91c73aed6445`、同data|98cdb35c…、上記familyのscalar保持対照、結果採否レビュー待ち|
|H4 native anchor|`217c971dd537b1cc7fbc13742bc0f6c263ff7f45`|0f96a2e4…、1Å固定16panel・532 compile・B0不在|
|H6 exploration|science `5a903aea2492c2d211c1e15e7737df0d50694c12`、exec `78a39051d8b6d41c33d592ecfe172f8824adfaeb`|df99d6ee…、linear1Å/STO-3G/DF19 tol1e-8、prefix5/10/15/full19、δ=.1〜.8、T=.8、native/symmetric-directional、empirical u、P=0|
|H6 confirmation|同science、exec `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`|321cd8c3…、選択union8×独立32、探索8とpoolしない|
|H6 conditional s2|`843265b624eac8fbb9acea47ef21cab1207e1880`|e0e6473d…、q8/r2/K2非zero order2層だけ、8trajectory/16compile|
|H8 feasibility|`9bfea8bde44cbb2ea9187476ddd496066a8853b3`|a55da5c2…、linear1Å/STO-3G/DF28 sector4900、8case20wrapper、正式matched mapではない|
|H8 resume02|exec `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、seed `cb8a56c6b2a772827275f9200b02638030f33f72`|起動38d386c8…、prefix7/14/21/full28、native、snapshot203/592。最終result未取得|

H6 input receipt commit `f52787a22542b31bd39fd004a8d3d71325bc56b0`、H8 input source `2ff3fc9f6e1091cbd9915507acfa74112c7d3055`。
full H/DF/state/environment/compiler/selection hashはserver projectionを参照。科学sourceはdirty localrootのHEADに置換しない。
旧production `eaeeefc3…` を現在resume02のSOURCEとしない。H8snapshotをlive/final statusに読み替えない。

## ローカルが保全している公開履歴

以下はarchive commit `054bb7b2b2ca788b47ebdc5c14c115f24c8b6379` の同一Git blobへリンクし、ローカル実体のbytes/SHAも保存した。14件はGitHubからも実取得して同一bytes/SHAを照合した。[履歴registry](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/historical_reference_registry_v1.json)。
文書から元実行result/source/rawへ遡れる。旧記録は現行成果の成功へ書き換えない。

|資料|Git blob|ローカル実体SHA-256|
|---|---|---|
| [M1-B1 validation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_matched_accuracy_m1_b1_result_validation.md) | `538bd83427d2a43dd762059bf759486b6c0ede10` | `4b619186a35e0c46fa7574757686ac2cbd47b8aa1ddd43ba76d2a6db5aa23b71` |
| [M2 fixed transfer validation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_matched_accuracy_m2_transfer_result_validation.md) | `7e79a4e58eafc2443e3b7767d29c615a9e9d4089` | `59d4e4ffc2c7b348ef592a21af6da7c51482e892166576ce2feafe92165ab42d` |
| [PM-0 same-R/attribution](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/pr2_post_m2_evidence_attribution.md) | `57c831a1d658515adf3af04e4af29318d5200766` | `27dd80a20b101b61ae1bcb91519df65b09d83cac58d97f1a7b5dedb49ae7871b` |
| [PM-1 discard validation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_pm1_discard_result_validation.md) | `23df022eaf980d7c6a75f3917d4eaeed48f63a01` | `cf369a4250bd231f821c47fb5713ff3e72739db3fa2f468bbd2cd1bff0c5a20b` |
| [PM-2 precision validation](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_pm2_precision_resource_result_validation.md) | `fcbf99a44d5122c16e42ee955ed9ded04e4d0497` | `be23900ca1c89150592de5c415b210eb1ee8d1d1f77ba817e12410cab37c9bf4` |
| [PM-2 raw summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/summary.json) | `f1870186b993ac487b73dca927747a5c52545bf5` | `7b026d4cc657cf43ad23fd7d6e10d5aa31cd58af555649b7a8e858ea12155845` |
| [v0.1 claim/evidence map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_post_pm2_claim_evidence_map.md) | `3c245b108b0c17503092af861bee00e6626f2beb` | `12dc142a8018e33b1014d47edbdb29b59aa9740fc6b2c00a525e88d7ddc4b4b4` |
| [original manuscript v0.1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/manuscripts/track_a_resource_study_v0_1.md) | `2da6c31b9db77f8fac3a58d1fd60fef10f2b3025` | `6778cdd99867c156430d2f7bd01c0beec1e3e7c12859eb284f0810f4d781f0a4` |
| [AX-0 contract](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_ax0_research_contract.md) | `67a11697acacedfad928978cc5dd42104a573df4` | `2d194af4f00b4012a704fea1a7d685671f4d9076e8b3a78b1eed90b09541a665` |
| [AX-1b GPT priority decision](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_ax1b_post_scientific_review_2026-10-09.md) | `556e19d53e1aa8659415a81043f594ce9d6b448a` | `0ca70130c1fe1901ae928e136ea85b89962674321799b89b661f3ecb740e7b4d` |
| [old H6 v2 MemoryError](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_h6_matched_accuracy_result_v2.md) | `79b0248d247f557894734d400b4afc1e9c33be26` | `b5b46708b19cd9f92ecbffd065464dca094396f9468ff6241d4e9c28245310bd` |
| [old H6 startup STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_h6_matched_supplement_launch_stop_v3.md) | `039234baca4c4126dee398a56d5373fc755692e2` | `e47c1aa0c5038f2cc7763e77c0fd985a7e12bfc6594c26483be8e0b8abb7874d` |
| [old H6 launch_v2 STOP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/research/track_a_h6_matched_supplement_result_launch_v2.md) | `c6971f53b02e6eb2f4905b0cdb333b6750c62bbe` | `39accc31c6860eda02df8d63e2ee408292b7daa9452f322e9892d3bef0657618` |
| [old launch_v2 missing registry](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/artifacts/resource_applicability/track_a_h6_matched_execution_launch_v2/2026-10-11/missing_coverage_launch_v2.json) | `d2e7386923ab55b94402e316f1face35d5ceda26` | `28c10ab44ccd485a28bcbf1d0c170c40ca57d1ddeb33c59c9cc508500f9632cf` |

旧M1 source33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62 / result8e0814e70c14ecf526444fac8a2142799610dc96、M2 source2978e2fea672b7a1ff20cac74269ec9a610159dc、PM-1 sourcefd7552edc0334ccf57ecf501a128c85c8d22822a、PM-2 source324435d77b6642dbd44e8d1f178420daf62e77ed / result5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc。
scopeはH4 linear/STO-3G/DF12、development1Å218、transfer1.3Å固定5、T=.8/S2、δ=.1〜.8、canonical finite-RTE、P=0、保存32 paired trajectory、302精度.005〜.1。元のdocsから詳細prefix/r/K・compiler条件を読む。
AX-1b sourcefc297cd9ab840018c4f35b2764d0b8be07c57285 / resultb2e1bf65e21893b6c617223b42313623d3186f12。保存値解析であり新しいsignal/compileではない。

## ローカル未公開資料・未解決事項

[local-only hash索引](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/local_only_materials_v1.json)、[差分registry](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/difference_registry_v1.json)、[old-export terminal](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/old_environment_export_terminal_original.json)、[provenance](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/old_environment_export_provenance_v1.json)。
originalZIP/raw/venv/絶対pathは公開しない。terminalのSHA88e07438…のみ今回exact copyとして追加し、旧sourcepacketb53271f…は再commitしない。
H8最終結果、serverrun03a324公開到達性、partial450実event、localworkingnote採択、厳密winnerは未確認。
共通正本採択・source統合・計算実行は未認可。mandatory STOP。
