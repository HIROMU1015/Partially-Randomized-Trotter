# GAP-008 旧DF screeningの探索結果

2026年10月23日 JST。公開用に内部パス・host/user識別情報・private cache情報を除いた報告。環境IDは `ENV-LOCAL-01`。

**GAP全体の判定 `FOUND_UNVERIFIED`。部分発見であり未解消。** 削除された旧無効化入力と旧screening出力のhistorical identityは確認できた。一方、修正後の正しいcanonical入力と、元科学runの入力・SOURCE・環境を閉じる来歴は見つからなかった。

| 対象 | 判定 | 確認範囲 |
|---|---|---|
| 削除前集約Cgs表、index、35 split表 | `FOUND_MATCHED` | 固定監査が指す削除commitの親treeにある旧bytesと一致 |
| 旧screening JSON | `FOUND_MATCHED` | 固定監査commit内の旧結果と、現存248コピーのbytesが一致 |
| 修正後canonical集約表・split・index | `NOT_FOUND_IN_SEARCH_SCOPE` | 探索ファイル名、既知保存先probe、対象pathの既存Git履歴で未発見 |
| 元anchor rawと生成log、実行command/runtime | `NOT_FOUND_IN_SEARCH_SCOPE` | SOURCE参照先と対象履歴に未発見 |
| 小さなground-state cache候補 | `FOUND_UNVERIFIED` | 3278 bytesのNPZは存在するが旧runとのidentity根拠がない |

## 固定identityと無効化の来歴

[固定監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/VALIDATION_STATUS.md)が指す削除commitは `98f960c2dd09fc1ae6b8b5c802dc5ce84fc61604`、親は `57e49cf27d19bfb35cb71b7e9d90a5a30e371e5b`。削除理由は「基底状態ずれてたので削除」。旧集約表・index・35splitは、この環境のローカルGit objectに残っていた。原本のcheckout/exportはしていない。

37件すべてが追加commit `c580aec790c46dd9aaa2f2982bc9270210523a0d` のblobとも一致した。集約35行、index35行、35split間の保存entryは全件対応していた。確認したのは既存bytesと表の対応であり、新fit・screening・best/rank選択・性能評価はしていない。旧`DO NOT USE / stale`を保持する。

| 旧資料 | bytes | SHA256 |
|---|---:|---|
| `df_cgs_cost_table.json` | 31452 | `73ec98033b3876ac254ae78ca0d37d8a170cca14eee39bee073ba5ebf834e61c` |
| `index.json` | 8624 | `dd87cad339dc2a1a441d87a2387ed09098e3fc148164a60510247198a55b6440` |
| `df_screening_cost_minimization_eps_1.000e-04.json` | 560401 | `0c6c666ad949e500cb4faf359d45e774438249e4e3c9f45c1f6cd3bee67d9c6c` |

集約表のGit blob IDは `37a449d4648698ad64e12e751cce220f369db9b8`、indexは `8d50ab51917b4a0feb554b3af30c2f272c9711cd`、旧screeningは `c7e8c015c87bd605defd8664ad04d30ef78dba19`。全37件のGit IDと固定参照は[GAP-008_provenance.json](GAP-008_provenance.json)に収録。

旧screeningは`epsilon_total=1e-4`、635候補、`df_reduced_screening_cost_minimization_v1`、qdrift、anchor再利用規則を保存していた。元入力pathの旧rootはこの環境に存在しない。現存248コピーは全件SHA256が同じ旧結果で、修正後結果ではなかった。

## 保存された条件と未確認の来歴

旧表は`schema_version=1`、`table_type=df_cgs_cost_input`。PF/rank/anchorの保存値は次のとおり。

| 分子 | 保存DF rank | 保存anchor L_D | PF label |
|---|---|---|
| H3 | 5 | 2 | 2nd, 4th, 8th(Morales) |
| H4 | 7 | 3 | 2nd, 4th, 8th(Morales) |
| H5 | 9 | 4 | 2nd, 4th, 8th(Morales) |
| H6 | 11 | 5 | 2nd, 4th, 8th(Morales) |
| H7 | 13 | 6 | 2nd, 4th, 8th(Morales) |
| H8 | 15 | 7 | 2nd, 4th, 8th(Morales) |
| H9 | 17 | 8 | 2nd, 4th, 8th(Morales) |
| H10 | 25 | 12 | 2nd, 4th, 8th(Morales) |
| H11 | 21 | 10 | 2nd, 4th, 8th(Morales) |
| H12 | 28 | 14 | 2nd, 4th, 8th(Morales) |
| H13 | 25 | 12 | 2nd, 4th, 8th(Morales) |
| H14 | 36 | 18 | 2nd, 4th |

各行のtime grid、`source_kind`、`source_path`、cost定義は[GAP-008_provenance.json](GAP-008_provenance.json)に保存した。元raw参照は `artifacts/partial_randomized_pf/H3_H14_df_screening_anchor_cgs.partial.json`。このrawとfinal版はファイル名検索・既知保存先probeで見つからず、参照raw pathの既存Git履歴も0件だった。

basis/geometryやHamiltonian生成規則のSOURCEはあるが、実runにbindされたH/state content hash、実際に使用したPF係数のidentity、実行commit、command、依存版、生成logは未確認。宣言SOURCEのdefaultを実行条件確定とはしない。

既知のDF ground-state cache保存先にNPZ候補1件があったが、旧screeningとの独立したidentity根拠はなかった。保存header上のfull state_vectorのshapeは4で、旧H3–H14の原入力として採用していない。数値配列をloadせず、member/headerと文字列metadataだけを確認した。private cacheの所在・fingerprint・内容は公開版には収録しない。

別の`df_cgs_fit_cache.json`2件と`phase_bias_table.json`2件は`technical/final_workers/test_*`内の小さなsynthetic fixtureで、旧科学原本の候補から除外した。

## 発見した関連SOURCE

以下は旧workflow追加commitで特定したrepository SOURCEである。実runのSOURCEを確定したものではない。bytes/SHA256/Git IDを記録し、import・実行はしていない。

| SOURCE path | 保存commit | bytes | SHA256 |
|---|---|---:|---|
| `scripts/export_df_cgs_cost_table.py` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 8366 | `e4d35aee95071e9e3a7d04826edf8895fbdeb93fe92acb7ef8dfb51a912b0294` |
| `scripts/run_df_screening_anchor_cgs_h3_h14.py` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 7138 | `a719a0f494e593459b69cc04280ec68917fbbb118efd7b718d542eb809ff3d89` |
| `scripts/run_df_screening_cost_minimization.py` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 2913 | `299c5f58513ddc031d84d6d0d0511f6bd9ffbf13b31f50615f2df0b4026ba1ed` |
| `src/trotterlib/df_screening_cost.py` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 10886 | `e862319653377daf2c203e84f7719dd205c397d1fd298cc75ac6a2bb1f1f52d6` |
| `src/trotterlib/df_partial_randomized_pf.py` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 46245 | `a1ceb76f7cab3ff0deeb21f0dd4dfa4e427f86c0a1bdbeb92c3562137015ba31` |
| `src/trotterlib/df_hamiltonian.py` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 38938 | `c32cf6a36f3e4ec08b528a3660f222f30d2d76e300422ce01f6489cf93c04690` |
| `src/trotterlib/product_formula.py` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 4572 | `39c4e03748fcab03ea7e64936a19372cb5f1bd1bc9f72a4f09cef58c3c202eca` |
| `src/trotterlib/config.py` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 12783 | `15e12517f82ed3e9d56eac8a272693ced87226a8ae5e97351e1c2c9b4e3415ca` |
| `requirements.txt` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 297 | `d9f2e0af70bf58be27c6e46b3c28bfece031591de9988420f73d08cecfa2c111` |
| `requirements-gpu.txt` | `c580aec790c46dd9aaa2f2982bc9270210523a0d` | 43 | `cbc27b46c5766d24e10de2936a27932bd222a1c64acca52b7967c5ff35ae676a` |

## 削除前37ファイルのidentity一覧

この表のpathはすべてcommit `57e49cf27d19bfb35cb71b7e9d90a5a30e371e5b` 内の相対path。物理的な回収・再配置はしていない。

| 削除前の相対path | bytes | SHA256 |
|---|---:|---|
| `artifacts/partial_randomized_pf/df_cgs_cost_table.json` | 31452 | `73ec98033b3876ac254ae78ca0d37d8a170cca14eee39bee073ba5ebf834e61c` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H10/2nd.json` | 1100 | `56a5fcb89ca92bfcb60bd16189ac48f7e2f94bd98fe73d5a5910cb3ea272d6ae` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H10/4th.json` | 1101 | `a8535e1c556f911a4abee4b64c0104bf581d1d30287c9a9213c8b9fdf689f572` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H10/8th_Morales.json` | 1126 | `e795affde910a5a44142b0d3e9b20ddc4000182426a253f13d815b5dd9db10a9` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H11/2nd.json` | 1100 | `72de65b2f065840e06e3215e131d1c3add0d64e079ae8050d4df1b2aa04c3761` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H11/4th.json` | 1098 | `fa4735c227de3658e43235d5ea27bd218158dc6acbb060631c61517699ea889a` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H11/8th_Morales.json` | 1121 | `08e135fa242e3f005da57b2a370777295a51f3363d43d4544919b90d6d0e5f81` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H12/2nd.json` | 1083 | `3698973f52a8101ddc0097b906a6976c6fc4215066ef1dafe5899e975e90f351` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H12/4th.json` | 1084 | `8e82b3ff5738975ec120f1493c03e75a19ef63bfccbdd352e7bcdbfc12f8ecaf` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H12/8th_Morales.json` | 1110 | `156e4a60fd11bb7a6456250528fb86d28c407c2455185b47b516d05be03760cd` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H13/2nd.json` | 1081 | `f0d3530a221de8e6e41d58bf69312974e754efd6197f56b909ab70ed63582097` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H13/4th.json` | 1085 | `82aa5c80f7e279a256266d66af8ee9255c1610036e66c20902df5ef367f72f80` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H13/8th_Morales.json` | 1110 | `321e10f0bbea72b2b1e7c68dfa01f3458d0909f9ac860997cf39f8b8b360fdc7` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H14/2nd.json` | 1081 | `c93d2d1735248cd7eaff0c2d5ce25fd1210255a47a295ff5540525877c4cc373` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H14/4th.json` | 1085 | `a83b360b31b5085f40026c69f783b17d21e86b336b53a2a0d61e929765447acd` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H3/2nd.json` | 1086 | `229ac026826960688922193d38c28a15effbf1f5f3825414dd02d04e17c2215b` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H3/4th.json` | 1089 | `23f33ab73da671be22baeb67de71142b6e055028f6d29f2f6d34eb14deb8de76` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H3/8th_Morales.json` | 1111 | `017c825dc617bb0939d60f8e564e0953dd1cb7276a897939e803ff19ae74fb01` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H4/2nd.json` | 1083 | `f585f5f673d5d32cb98bd8299830ffe0b32775fe3ed359c960022a8094605b22` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H4/4th.json` | 1087 | `41250b5b0760b7801b6d1254b04cddd763e91eebb6fb9a789d9aa36beec7d3aa` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H4/8th_Morales.json` | 1110 | `1d5ac9bddd9d6cce8857a922e8e3a4c2f858779c5c655afe39949f6f81ed475f` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H5/2nd.json` | 1087 | `3e4f5672fb9b303c468586d530bfa83d272241d305cb4d394dbb197b9dcd4d3f` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H5/4th.json` | 1087 | `eb35acddf11240560a28853babc49e13d15f87cea6833af97436fd3d461271f0` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H5/8th_Morales.json` | 1114 | `ce2ea8de08a607cd30e8b41ca007b5bd4893615b074f0097af7485f24d3ed7a0` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H6/2nd.json` | 1086 | `19663f987f22b656ee518695ac5d065e1e8ea77cb1f51a5df1a2b05097fc3908` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H6/4th.json` | 1091 | `ae3a3ab0b78a8a809f5855d316672f3f3e2d971defd177d5f57d28375e15e6f7` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H6/8th_Morales.json` | 1116 | `1546aeb09a99701b3dd303d126df219c89b3d612391b8c9d1f1ab8128bdde330` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H7/2nd.json` | 1085 | `5c257f5bfc3a9f4d6d823c65697ef175d73f9c99e5c69a80c57110a1c2be0440` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H7/4th.json` | 1091 | `3b523ad99aea0c848ccddb2238c58ef57557cb53a17e1bfbf30010a2e6eb8dac` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H7/8th_Morales.json` | 1117 | `41eee9524ff0b8743370c1855585375cf36f0798ebc862ba96b7bea6b77d13bc` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H8/2nd.json` | 1090 | `484ba29916df84eba83b55fff20e93de81dc469203eb71cd07398220fcf2b02c` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H8/4th.json` | 1093 | `0bfa937c58904ba7f8a19a0e8466297899916f4e773f3b6b2b5dd81484deaaaa` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H8/8th_Morales.json` | 1119 | `9e88648d72b275defc1125b7ff9b5722c0c4d9d084bcfb537899bf08dc751614` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H9/2nd.json` | 1093 | `837d03eddd2cb2ecf514d921372c166ff9ed65d8317b5782671c895cd99d60e5` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H9/4th.json` | 1093 | `d2f970394bd112ebdca2d644b4e4bf9d0bb36d11c99861d3212b0bdf3145393d` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H9/8th_Morales.json` | 1119 | `1e6ceffbd97d675c329b2e10ad608b2a364d197679308d1f0953baf00ee4598a` |
| `artifacts/partial_randomized_pf/df_cgs_cost_tables/index.json` | 8624 | `dd87cad339dc2a1a441d87a2387ed09098e3fc148164a60510247198a55b6440` |

## 引継ぎと停止

探索範囲と未調査領域は[探索総報告](SEARCH_REPORT.md)を参照。旧37ファイルが見つかったことだけではGAP-008は閉じない。

引継ぎGAP-IDは **GAP-008**。次の環境では修正後canonical表・元anchor raw・生成log・runにbindされたHamiltonian/state/PF/SOURCE/runtime identityを探す。ユーザーが別途環境と作業を指定する。今回のCodexは報告後mandatory STOP。旧stale status、削除の履歴、既存STOPを変更せず、科学実行や回収・転送・公開を開始しない。
