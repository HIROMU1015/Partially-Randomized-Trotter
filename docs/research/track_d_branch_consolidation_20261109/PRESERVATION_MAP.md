# 固定履歴・成果の保存対応

2026-11-09 JST。Track D 43 tipは通常merge後の統合commit `a7557032854e8e7473284c02ee610616e7e912d8` の祖先。O2 2 tipは保持O2 H4 tipの祖先。archive tagは新設不要・実作成0。元remote branchも削除0で全て残る。

全pathの旧blobと現在blobの区別は [BRANCH_INVENTORY.json](BRANCH_INVENTORY.json) の `noncurrent_versions` に記録。`FIXED_HISTORY_VERSION` は現treeと異なる版であり、現ファイルに統合済みとは扱わない。該当commitから元bytesを読める。

| 元branch | 元HEAD | 保持ref | 履歴到達 |
|---|---|---|---|
| `codex/o2-vacuum-cswap-20261108` | `0edc0fa5ad4656fdd492ca6c96ade4f77cadf412` | `codex/o2-vacuum-h4-preparation-20261108` | ancestor確認済み |
| `codex/o2-vacuum-h4-preparation-20261108` | `781e155d0907a82c962e41ab8e89f985735bc49b` | `codex/o2-vacuum-h4-preparation-20261108` | ancestor確認済み |
| `codex/track-d-b1-c1-validation-20261106` | `d3913866e6ade4ab97ac02645fed91b2d5fcfa75` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-b6b-bc-integration-20261108` | `7ffe8b281b76cef3a6fd06f43413904d6766d0ec` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-bc-static-preparation-20261106` | `5f94972adf5a10e96e9400d3919c00d14e7a718a` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-bc-total-20261108` | `8828b298975c88a2c0d1fd57ff6ed905233840f5` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c2-q2-compile-20261106` | `c8fe5091c42ec786688b6931105ad64d10c87d18` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c3-q4-q8-validation-20261106` | `cc554cf6daea08df6c20ea8e1e92405259ce49b5` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c4-price-model-preparation-20261106` | `a5784c4bd3e3fb9779849a1e1fc1759559e2caea` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c5-whole-path-q2-20261106` | `9a6920b8355643701ab7436a4fd3b34862e9e3bd` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c6a-paired-price-pilot-20261106` | `e9d4e2f602d5f9e014507ca7e391719bab017b88` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c6b-accelerated-completion-20261107` | `fed5f458be2f625de4e1b8c98b8cc5da6e551b14` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c6b-q8-exploratory-20261107` | `23725fc6f6d6d0c2bec93e31e32074711804e250` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c6b-state-action-acceleration-20261107` | `75f43855b02a49e710d7a2ba3ee82eda3de1631e` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c6b0-paired-model-preparation-20261107` | `c93f3c2373b158f8eff3800a62fe1a94cce5c3b0` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c7-0-structural-preparation-20261107` | `8b422c28c22666a32103fdae33a3c9b9512ef248` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c7-1-q16-exploratory-20261107` | `86c5c9d48dbddc085f388b6521be8cd92ee49d0d` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c7-2-cx-structure-20261107` | `709d6ed9c234bf6d5a8848d3a30be69d88fd6476` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c7-3-q32-accelerated-20261107` | `acea43987edae0d5eb0c186a89aa8cd7139fbd7b` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c7-3-q32-exploratory-20261107` | `35bfd0423d08114b0a24107501ce545cdfb28f32` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c7-4-model-stability-20261107` | `64072ba81bb70db1df1b34e94adb08e78926f24b` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c8-1-q64-20261108` | `6b1d613ded963b7851afa1c07ba8c646124dd44f` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c8-1-q64-accelerated-20261108` | `1e88074f1fd64e6efe5e458c0b189069801b727b` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c8-1-q64-completion-20261108` | `b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c8-1-q64-nofusion-repair-20261108` | `a708a13077b0616a6047bbca2ba36f21ae58755b` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-c8-1-q64-numerical-repair-20261108` | `46ce57e18b7d8cb606c0446e562ca98be63b4f5d` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-cut-artificial-20261108` | `a54d04059bfc61adef43aa2d40fb9c001b62378f` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-cut-h4-q4-20261108` | `848ca0deecc08fd09bbb9027fb372905edd4414d` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-cut-h4-q4-completion-20261108` | `2f316fb229149117177367720a6eaa5aa8a94b76` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-h4-paired32-reference-20261106` | `3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-h4-reference-validation-20261106` | `7d0dc55199c49a3f49a9ba93323843d8a336cce1` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-minimal-reference-validation-20261106` | `102e3c2f29999fb9fbdfa6f85020370df228a7f3` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-o5-c1-h4-q4-20261108` | `0ef53fa9a65a70a251a7d6e5dbeb91141a1a67e9` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-o5-c1-head-tail-artificial-20261108` | `14276fe0e2dadef54ef105debe6d6d99b43b6ecb` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-o5-c1-head-tail-static-20261108` | `62c7b578b8c0bb3d0079434daf637c587f53c24c` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-o5-design-20261108` | `2131917656446bd22f294946d7c1064a47332e4c` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-o5-q4-cohort-metadata-20261108` | `0d596e7a51a79013bd93cad133b6b14c5b1e2573` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-pr-rpe-static-survey-20261106` | `523efe89fde52a14a8e0a4eba4ea6b70e3f512e1` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-real-pr-transfer-static-20261106` | `a9a74b1386ba713ed3648658dd94be4d992a492e` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-reference-control-feasibility-20261106` | `c3cd64755944be4aad2e032affd1ed0308cf5743` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `codex/track-d-vacuum-path-static-check-20261106` | `71ac8b96c833c09d17ae6a02118471acedd781df` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `track-d-b3-rpe-statistics-preparation-20261106` | `89a13fe6841c1017f0feba724ed71dfc79f9e81b` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `track-d-b4-1-rpe-statistics-implementation-20261106` | `31cd67c91abda3c9c5d4b643c434c040b8ceaeca` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `track-d-b4-2-rpe-shots-science-20261106` | `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` | `codex/track-d-consolidated-20261109` | ancestor確認済み |
| `track-d-b5-actual-rpe-static-survey-20261106` | `0a8b953aea1be20b110784960763d7f36ebccac8` | `codex/track-d-consolidated-20261109` | ancestor確認済み |

## 固定科学証拠

Track Dの確認対象では、既存SOURCE・結果・STOPの別版が現在treeから欠ける例は0件。旧コード版・索引版などは現在版と区別して固定履歴に保持した。O2旧CSWAP版の後続変更も元commitから読む。入力/law/compilerのprovenanceは各SOURCEとreportに残り、grant/ledger/markerの変更はない。

読み出し例：

```bash
git fetch origin codex/track-d-consolidated-20261109
git show 6b6667026ec8b670f610e41141d195883a9eb513:src/trottertracks/track_d_c6a_pilot.py
git show 0edc0fa5ad4656fdd492ca6c96ade4f77cadf412:docs/research/o2_vacuum_reference_cswap_20261108/REPORT.md
```

科学実行SOURCEと結果公開commit、整理commitは区別する。raw/QPY/NPZ等の非公開原本は各元保存場所のままであり、Git保全のための新規公開・移動・削除は0。
