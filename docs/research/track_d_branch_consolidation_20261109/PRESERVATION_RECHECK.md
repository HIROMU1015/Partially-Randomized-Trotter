# Track D branch整理：保全再確認

記録：`2026-11-09T17:11:30.842620+09:00`（JST）。削除実行0本。固定資料commit `9d12d7c6cf3da8f739e0238f7abbbc7018bb6d53` の既存棚卸しを再利用し、科学的評価は行わなかった。

## 統合先とO2

統合先 `codex/track-d-consolidated-20261109` は `9d12d7c6cf3da8f739e0238f7abbbc7018bb6d53` のまま存在する。固定公開commitと元Track D 43 tipへの祖先検査は全てPASS。O2保持先 `codex/o2-vacuum-h4-preparation-20261108` は `781e155d0907a82c962e41ab8e89f985735bc49b` のまま存在し、元CSWAP tip `0edc0fa5ad4656fdd492ca6c96ade4f77cadf412` と保持先の元tipへの祖先検査もPASS。

| 元branch | 固定tip | 現在の保持refへの到達 |
|---|---|---|
| `codex/o2-vacuum-cswap-20261108` | `0edc0fa5ad4656fdd492ca6c96ade4f77cadf412` | PASS |
| `codex/o2-vacuum-h4-preparation-20261108` | `781e155d0907a82c962e41ab8e89f985735bc49b` | PASS |
| `codex/track-d-b1-c1-validation-20261106` | `d3913866e6ade4ab97ac02645fed91b2d5fcfa75` | PASS |
| `codex/track-d-b6b-bc-integration-20261108` | `7ffe8b281b76cef3a6fd06f43413904d6766d0ec` | PASS |
| `codex/track-d-bc-static-preparation-20261106` | `5f94972adf5a10e96e9400d3919c00d14e7a718a` | PASS |
| `codex/track-d-bc-total-20261108` | `8828b298975c88a2c0d1fd57ff6ed905233840f5` | PASS |
| `codex/track-d-c2-q2-compile-20261106` | `c8fe5091c42ec786688b6931105ad64d10c87d18` | PASS |
| `codex/track-d-c3-q4-q8-validation-20261106` | `cc554cf6daea08df6c20ea8e1e92405259ce49b5` | PASS |
| `codex/track-d-c4-price-model-preparation-20261106` | `a5784c4bd3e3fb9779849a1e1fc1759559e2caea` | PASS |
| `codex/track-d-c5-whole-path-q2-20261106` | `9a6920b8355643701ab7436a4fd3b34862e9e3bd` | PASS |
| `codex/track-d-c6a-paired-price-pilot-20261106` | `e9d4e2f602d5f9e014507ca7e391719bab017b88` | PASS |
| `codex/track-d-c6b-accelerated-completion-20261107` | `fed5f458be2f625de4e1b8c98b8cc5da6e551b14` | PASS |
| `codex/track-d-c6b-q8-exploratory-20261107` | `23725fc6f6d6d0c2bec93e31e32074711804e250` | PASS |
| `codex/track-d-c6b-state-action-acceleration-20261107` | `75f43855b02a49e710d7a2ba3ee82eda3de1631e` | PASS |
| `codex/track-d-c6b0-paired-model-preparation-20261107` | `c93f3c2373b158f8eff3800a62fe1a94cce5c3b0` | PASS |
| `codex/track-d-c7-0-structural-preparation-20261107` | `8b422c28c22666a32103fdae33a3c9b9512ef248` | PASS |
| `codex/track-d-c7-1-q16-exploratory-20261107` | `86c5c9d48dbddc085f388b6521be8cd92ee49d0d` | PASS |
| `codex/track-d-c7-2-cx-structure-20261107` | `709d6ed9c234bf6d5a8848d3a30be69d88fd6476` | PASS |
| `codex/track-d-c7-3-q32-accelerated-20261107` | `acea43987edae0d5eb0c186a89aa8cd7139fbd7b` | PASS |
| `codex/track-d-c7-3-q32-exploratory-20261107` | `35bfd0423d08114b0a24107501ce545cdfb28f32` | PASS |
| `codex/track-d-c7-4-model-stability-20261107` | `64072ba81bb70db1df1b34e94adb08e78926f24b` | PASS |
| `codex/track-d-c8-1-q64-20261108` | `6b1d613ded963b7851afa1c07ba8c646124dd44f` | PASS |
| `codex/track-d-c8-1-q64-accelerated-20261108` | `1e88074f1fd64e6efe5e458c0b189069801b727b` | PASS |
| `codex/track-d-c8-1-q64-completion-20261108` | `b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7` | PASS |
| `codex/track-d-c8-1-q64-nofusion-repair-20261108` | `a708a13077b0616a6047bbca2ba36f21ae58755b` | PASS |
| `codex/track-d-c8-1-q64-numerical-repair-20261108` | `46ce57e18b7d8cb606c0446e562ca98be63b4f5d` | PASS |
| `codex/track-d-cut-artificial-20261108` | `a54d04059bfc61adef43aa2d40fb9c001b62378f` | PASS |
| `codex/track-d-cut-h4-q4-20261108` | `848ca0deecc08fd09bbb9027fb372905edd4414d` | PASS |
| `codex/track-d-cut-h4-q4-completion-20261108` | `2f316fb229149117177367720a6eaa5aa8a94b76` | PASS |
| `codex/track-d-h4-paired32-reference-20261106` | `3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0` | PASS |
| `codex/track-d-h4-reference-validation-20261106` | `7d0dc55199c49a3f49a9ba93323843d8a336cce1` | PASS |
| `codex/track-d-minimal-reference-validation-20261106` | `102e3c2f29999fb9fbdfa6f85020370df228a7f3` | PASS |
| `codex/track-d-o5-c1-h4-q4-20261108` | `0ef53fa9a65a70a251a7d6e5dbeb91141a1a67e9` | PASS |
| `codex/track-d-o5-c1-head-tail-artificial-20261108` | `14276fe0e2dadef54ef105debe6d6d99b43b6ecb` | PASS |
| `codex/track-d-o5-c1-head-tail-static-20261108` | `62c7b578b8c0bb3d0079434daf637c587f53c24c` | PASS |
| `codex/track-d-o5-design-20261108` | `2131917656446bd22f294946d7c1064a47332e4c` | PASS |
| `codex/track-d-o5-q4-cohort-metadata-20261108` | `0d596e7a51a79013bd93cad133b6b14c5b1e2573` | PASS |
| `codex/track-d-pr-rpe-static-survey-20261106` | `523efe89fde52a14a8e0a4eba4ea6b70e3f512e1` | PASS |
| `codex/track-d-real-pr-transfer-static-20261106` | `a9a74b1386ba713ed3648658dd94be4d992a492e` | PASS |
| `codex/track-d-reference-control-feasibility-20261106` | `c3cd64755944be4aad2e032affd1ed0308cf5743` | PASS |
| `codex/track-d-vacuum-path-static-check-20261106` | `71ac8b96c833c09d17ae6a02118471acedd781df` | PASS |
| `track-d-b3-rpe-statistics-preparation-20261106` | `89a13fe6841c1017f0feba724ed71dfc79f9e81b` | PASS |
| `track-d-b4-1-rpe-statistics-implementation-20261106` | `31cd67c91abda3c9c5d4b643c434c040b8ceaeca` | PASS |
| `track-d-b4-2-rpe-shots-science-20261106` | `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` | PASS |
| `track-d-b5-actual-rpe-static-survey-20261106` | `0a8b953aea1be20b110784960763d7f36ebccac8` | PASS |

## q64・q32・対象外ref

| 保護対象 | 最終確認SHA |
|---|---|
| `codex/track-d-c8-1-q64-20261108` | `6b1d613ded963b7851afa1c07ba8c646124dd44f` |
| `codex/track-d-c8-1-q64-accelerated-20261108` | `1e88074f1fd64e6efe5e458c0b189069801b727b` |
| `codex/track-d-c8-1-q64-completion-20261108` | `b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7` |
| `codex/track-d-c8-1-q64-nofusion-repair-20261108` | `a708a13077b0616a6047bbca2ba36f21ae58755b` |
| `codex/track-d-c8-1-q64-numerical-repair-20261108` | `46ce57e18b7d8cb606c0446e562ca98be63b4f5d` |
| `codex/track-d-c7-3-q32-exploratory-20261107` | `35bfd0423d08114b0a24107501ce545cdfb28f32` |

上記6本は開始時SHAから変更なし。q64の完了・closeout・科学PASSは判定せず、process・SOURCE・output・STOPへ介入していない。q32元枝の非公開原本・収集scriptの未解決状態も変更せず、固定38候補へ追加していない。

開始時408本のremote refは全て同じSHAで残存し、対象外branchの削除・更新は0。本作業の変更は管理記録branchの新設だけ。並行作業の対象外新設1本を別に記録した。

## 固定SOURCE・結果・STOPと作業tree

履歴保全は元commitへの到達として確認した。元45 tipのSOURCE/STOP参照108件（異なるblobは85件）は読み出せる。元commitのtreeを参照するため、過去の結果・旧コード版・STOPも同じGit objectとして残る。統合先の現在treeへ全旧版を上書きしたという意味ではない。既存のPUBLICATION_CHECKや科学実行SOURCEは変更していない。

削除処理による作業tree変更は0。既存のtracked編集・untracked状態・登録worktree・local branchを維持し、今回の管理記録3文書だけを追加する。Git gc、reflog削除、worktree削除、履歴書換えは実施していない。

非公開raw・QPY・NPZを移動・削除・新規公開していない。他サーバーへ接続していないため、それらのbytesや依存worktree・今後のpush用途は独立再確認できていない。この限界が全38候補のHOLD理由であり、全環境の保全を直接検証したとは扱わない。

## 確認方法と境界

`git ls-remote --heads` の開始・可否確認後・管理branch作成後snapshot、`git merge-base --is-ancestor`、元commitの `git cat-file --batch-check`、既存候補別leaseコマンドの完全一致、GitHub default/protection/ruleset照会を用いた。新しい科学計算・数値再解析・compile・研究レビュー・統合mergeは全て0。

公開後にremote SHA・commit tree・3文書のbytesを照合してmandatory STOP。削除段階の完了、他サーバー非使用、q64の科学的完了は認定しない。
