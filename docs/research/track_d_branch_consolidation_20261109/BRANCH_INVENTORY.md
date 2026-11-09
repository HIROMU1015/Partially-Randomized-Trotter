# Track D branch inventory

更新：固定38本は削除完了、元対象7本＋統合branch1本を保持。[最終分類](deletion_completion_20261109/INVENTORY_FINAL.json)を参照。以下の表は初回HOLD時点のsnapshotとして保全する。

2026-11-09 JST。観測時刻 `2026-11-09T16:45:09.470993+09:00`。対象は開始時の45本（Track D 43＋O2移管2）。新しい統合branchは対象削除リストへ追加しない。

全remote開始405本。元45 HEADは再照会でも一致。現行branch protectionフラグは45本ともfalse、defaultはmain。Open PRは取得時0件。所有者は利用者のrepository。

履歴到達と作業treeのbytes一致は別判定。完全な親子関係・固有commit・変更数・SOURCE/STOP path・旧版blob・worktree観測は [JSON](BRANCH_INVENTORY.json) に記録する。

| 元branch | 元HEAD（完全SHA） | 固有commit数 | 最終分類 | 保存先 |
|---|---|---:|---|---|
| `codex/o2-vacuum-cswap-20261108` | `0edc0fa5ad4656fdd492ca6c96ade4f77cadf412` | 0 | HOLD | `codex/o2-vacuum-h4-preparation-20261108` |
| `codex/o2-vacuum-h4-preparation-20261108` | `781e155d0907a82c962e41ab8e89f985735bc49b` | 1 | KEEP_INDEPENDENT | `codex/o2-vacuum-h4-preparation-20261108` |
| `codex/track-d-b1-c1-validation-20261106` | `d3913866e6ade4ab97ac02645fed91b2d5fcfa75` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-b6b-bc-integration-20261108` | `7ffe8b281b76cef3a6fd06f43413904d6766d0ec` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-bc-static-preparation-20261106` | `5f94972adf5a10e96e9400d3919c00d14e7a718a` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-bc-total-20261108` | `8828b298975c88a2c0d1fd57ff6ed905233840f5` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c2-q2-compile-20261106` | `c8fe5091c42ec786688b6931105ad64d10c87d18` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c3-q4-q8-validation-20261106` | `cc554cf6daea08df6c20ea8e1e92405259ce49b5` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c4-price-model-preparation-20261106` | `a5784c4bd3e3fb9779849a1e1fc1759559e2caea` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c5-whole-path-q2-20261106` | `9a6920b8355643701ab7436a4fd3b34862e9e3bd` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6a-paired-price-pilot-20261106` | `e9d4e2f602d5f9e014507ca7e391719bab017b88` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6b-accelerated-completion-20261107` | `fed5f458be2f625de4e1b8c98b8cc5da6e551b14` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6b-q8-exploratory-20261107` | `23725fc6f6d6d0c2bec93e31e32074711804e250` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6b-state-action-acceleration-20261107` | `75f43855b02a49e710d7a2ba3ee82eda3de1631e` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6b0-paired-model-preparation-20261107` | `c93f3c2373b158f8eff3800a62fe1a94cce5c3b0` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-0-structural-preparation-20261107` | `8b422c28c22666a32103fdae33a3c9b9512ef248` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-1-q16-exploratory-20261107` | `86c5c9d48dbddc085f388b6521be8cd92ee49d0d` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-2-cx-structure-20261107` | `709d6ed9c234bf6d5a8848d3a30be69d88fd6476` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-3-q32-accelerated-20261107` | `acea43987edae0d5eb0c186a89aa8cd7139fbd7b` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-3-q32-exploratory-20261107` | `35bfd0423d08114b0a24107501ce545cdfb28f32` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-4-model-stability-20261107` | `64072ba81bb70db1df1b34e94adb08e78926f24b` | 3 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c8-1-q64-20261108` | `6b1d613ded963b7851afa1c07ba8c646124dd44f` | 0 | KEEP_ACTIVE | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c8-1-q64-accelerated-20261108` | `1e88074f1fd64e6efe5e458c0b189069801b727b` | 0 | KEEP_ACTIVE | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c8-1-q64-completion-20261108` | `b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7` | 0 | KEEP_ACTIVE | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c8-1-q64-nofusion-repair-20261108` | `a708a13077b0616a6047bbca2ba36f21ae58755b` | 0 | KEEP_ACTIVE | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c8-1-q64-numerical-repair-20261108` | `46ce57e18b7d8cb606c0446e562ca98be63b4f5d` | 0 | KEEP_ACTIVE | `codex/track-d-consolidated-20261109` |
| `codex/track-d-cut-artificial-20261108` | `a54d04059bfc61adef43aa2d40fb9c001b62378f` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-cut-h4-q4-20261108` | `848ca0deecc08fd09bbb9027fb372905edd4414d` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-cut-h4-q4-completion-20261108` | `2f316fb229149117177367720a6eaa5aa8a94b76` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-h4-paired32-reference-20261106` | `3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-h4-reference-validation-20261106` | `7d0dc55199c49a3f49a9ba93323843d8a336cce1` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-minimal-reference-validation-20261106` | `102e3c2f29999fb9fbdfa6f85020370df228a7f3` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-c1-h4-q4-20261108` | `0ef53fa9a65a70a251a7d6e5dbeb91141a1a67e9` | 3 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-c1-head-tail-artificial-20261108` | `14276fe0e2dadef54ef105debe6d6d99b43b6ecb` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-c1-head-tail-static-20261108` | `62c7b578b8c0bb3d0079434daf637c587f53c24c` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-design-20261108` | `2131917656446bd22f294946d7c1064a47332e4c` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-q4-cohort-metadata-20261108` | `0d596e7a51a79013bd93cad133b6b14c5b1e2573` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-pr-rpe-static-survey-20261106` | `523efe89fde52a14a8e0a4eba4ea6b70e3f512e1` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-real-pr-transfer-static-20261106` | `a9a74b1386ba713ed3648658dd94be4d992a492e` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-reference-control-feasibility-20261106` | `c3cd64755944be4aad2e032affd1ed0308cf5743` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `codex/track-d-vacuum-path-static-check-20261106` | `71ac8b96c833c09d17ae6a02118471acedd781df` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `track-d-b3-rpe-statistics-preparation-20261106` | `89a13fe6841c1017f0feba724ed71dfc79f9e81b` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `track-d-b4-1-rpe-statistics-implementation-20261106` | `31cd67c91abda3c9c5d4b643c434c040b8ceaeca` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `track-d-b4-2-rpe-shots-science-20261106` | `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` | 0 | HOLD | `codex/track-d-consolidated-20261109` |
| `track-d-b5-actual-rpe-static-survey-20261106` | `0a8b953aea1be20b110784960763d7f36ebccac8` | 1 | HOLD | `codex/track-d-consolidated-20261109` |

## 判定の読み方

- KEEP_ACTIVE 5本：q64の特別保護。controllerは棚卸し途中で消失を観測したが、科学完了状態をこの整理作業では判定していない。未公開closeoutと元/修正SOURCEの必要refを保持する。
- KEEP_INDEPENDENT 1本：O2 H4準備。A1環境・state-free adapter・phase/POVM検証を元の互換contextに保持する。
- HOLD 39本：38本は保全・統合後の削除候補だが削除認証経路なし。q32元実行枝1本は未追跡runと収集script2本も残る。
- 登録worktreeの存在だけで実行中とはしない。clean・保存STOP・本人process関係を区別した。既存worktree/local branch/dirty/ignored rawは一切削除・checkoutしていない。
- 他サーバーのlive worktree・将来pushは未確認であり、後日削除する場合の再確認事項。

技術的な保全・削除区分であり、研究結果のPASS/FAILや採択を変更しない。
