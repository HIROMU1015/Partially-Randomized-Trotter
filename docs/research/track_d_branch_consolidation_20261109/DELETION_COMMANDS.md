# 後日再確認用の手動削除コマンド案

**未実施。38件とも現在はHOLD。** 通常Git認証が使える本人の環境で、[削除報告](DELETION_REPORT.md)と候補JSONの全確認条件を満たしてから対象を選ぶ。既存worktreeを消す手順ではない。q64 5本、O2保持先、q32未追跡元枝、main、対象外の研究枝を含まない。

事前に統合refとO2保持refを取得し、各旧HEADが当該refの祖先であること、現在の使用/保護/他研究依存がないことを確認する。remote SHAが変わった候補は実行しない。以下は個別branch名と期待SHAを固定したlease付き削除で、wildcardやforce overwriteを使用しない。

```bash
git fetch origin codex/track-d-consolidated-20261109 codex/o2-vacuum-h4-preparation-20261108
```

```bash
git merge-base --is-ancestor 0edc0fa5ad4656fdd492ca6c96ade4f77cadf412 origin/codex/o2-vacuum-h4-preparation-20261108
git ls-remote --heads origin refs/heads/codex/o2-vacuum-cswap-20261108
git push --force-with-lease=refs/heads/codex/o2-vacuum-cswap-20261108:0edc0fa5ad4656fdd492ca6c96ade4f77cadf412 origin :refs/heads/codex/o2-vacuum-cswap-20261108
```

```bash
git merge-base --is-ancestor d3913866e6ade4ab97ac02645fed91b2d5fcfa75 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-b1-c1-validation-20261106
git push --force-with-lease=refs/heads/codex/track-d-b1-c1-validation-20261106:d3913866e6ade4ab97ac02645fed91b2d5fcfa75 origin :refs/heads/codex/track-d-b1-c1-validation-20261106
```

```bash
git merge-base --is-ancestor 7ffe8b281b76cef3a6fd06f43413904d6766d0ec origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-b6b-bc-integration-20261108
git push --force-with-lease=refs/heads/codex/track-d-b6b-bc-integration-20261108:7ffe8b281b76cef3a6fd06f43413904d6766d0ec origin :refs/heads/codex/track-d-b6b-bc-integration-20261108
```

```bash
git merge-base --is-ancestor 5f94972adf5a10e96e9400d3919c00d14e7a718a origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-bc-static-preparation-20261106
git push --force-with-lease=refs/heads/codex/track-d-bc-static-preparation-20261106:5f94972adf5a10e96e9400d3919c00d14e7a718a origin :refs/heads/codex/track-d-bc-static-preparation-20261106
```

```bash
git merge-base --is-ancestor 8828b298975c88a2c0d1fd57ff6ed905233840f5 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-bc-total-20261108
git push --force-with-lease=refs/heads/codex/track-d-bc-total-20261108:8828b298975c88a2c0d1fd57ff6ed905233840f5 origin :refs/heads/codex/track-d-bc-total-20261108
```

```bash
git merge-base --is-ancestor c8fe5091c42ec786688b6931105ad64d10c87d18 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c2-q2-compile-20261106
git push --force-with-lease=refs/heads/codex/track-d-c2-q2-compile-20261106:c8fe5091c42ec786688b6931105ad64d10c87d18 origin :refs/heads/codex/track-d-c2-q2-compile-20261106
```

```bash
git merge-base --is-ancestor cc554cf6daea08df6c20ea8e1e92405259ce49b5 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c3-q4-q8-validation-20261106
git push --force-with-lease=refs/heads/codex/track-d-c3-q4-q8-validation-20261106:cc554cf6daea08df6c20ea8e1e92405259ce49b5 origin :refs/heads/codex/track-d-c3-q4-q8-validation-20261106
```

```bash
git merge-base --is-ancestor a5784c4bd3e3fb9779849a1e1fc1759559e2caea origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c4-price-model-preparation-20261106
git push --force-with-lease=refs/heads/codex/track-d-c4-price-model-preparation-20261106:a5784c4bd3e3fb9779849a1e1fc1759559e2caea origin :refs/heads/codex/track-d-c4-price-model-preparation-20261106
```

```bash
git merge-base --is-ancestor 9a6920b8355643701ab7436a4fd3b34862e9e3bd origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c5-whole-path-q2-20261106
git push --force-with-lease=refs/heads/codex/track-d-c5-whole-path-q2-20261106:9a6920b8355643701ab7436a4fd3b34862e9e3bd origin :refs/heads/codex/track-d-c5-whole-path-q2-20261106
```

```bash
git merge-base --is-ancestor e9d4e2f602d5f9e014507ca7e391719bab017b88 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c6a-paired-price-pilot-20261106
git push --force-with-lease=refs/heads/codex/track-d-c6a-paired-price-pilot-20261106:e9d4e2f602d5f9e014507ca7e391719bab017b88 origin :refs/heads/codex/track-d-c6a-paired-price-pilot-20261106
```

```bash
git merge-base --is-ancestor fed5f458be2f625de4e1b8c98b8cc5da6e551b14 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c6b-accelerated-completion-20261107
git push --force-with-lease=refs/heads/codex/track-d-c6b-accelerated-completion-20261107:fed5f458be2f625de4e1b8c98b8cc5da6e551b14 origin :refs/heads/codex/track-d-c6b-accelerated-completion-20261107
```

```bash
git merge-base --is-ancestor 23725fc6f6d6d0c2bec93e31e32074711804e250 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c6b-q8-exploratory-20261107
git push --force-with-lease=refs/heads/codex/track-d-c6b-q8-exploratory-20261107:23725fc6f6d6d0c2bec93e31e32074711804e250 origin :refs/heads/codex/track-d-c6b-q8-exploratory-20261107
```

```bash
git merge-base --is-ancestor 75f43855b02a49e710d7a2ba3ee82eda3de1631e origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c6b-state-action-acceleration-20261107
git push --force-with-lease=refs/heads/codex/track-d-c6b-state-action-acceleration-20261107:75f43855b02a49e710d7a2ba3ee82eda3de1631e origin :refs/heads/codex/track-d-c6b-state-action-acceleration-20261107
```

```bash
git merge-base --is-ancestor c93f3c2373b158f8eff3800a62fe1a94cce5c3b0 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c6b0-paired-model-preparation-20261107
git push --force-with-lease=refs/heads/codex/track-d-c6b0-paired-model-preparation-20261107:c93f3c2373b158f8eff3800a62fe1a94cce5c3b0 origin :refs/heads/codex/track-d-c6b0-paired-model-preparation-20261107
```

```bash
git merge-base --is-ancestor 8b422c28c22666a32103fdae33a3c9b9512ef248 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c7-0-structural-preparation-20261107
git push --force-with-lease=refs/heads/codex/track-d-c7-0-structural-preparation-20261107:8b422c28c22666a32103fdae33a3c9b9512ef248 origin :refs/heads/codex/track-d-c7-0-structural-preparation-20261107
```

```bash
git merge-base --is-ancestor 86c5c9d48dbddc085f388b6521be8cd92ee49d0d origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c7-1-q16-exploratory-20261107
git push --force-with-lease=refs/heads/codex/track-d-c7-1-q16-exploratory-20261107:86c5c9d48dbddc085f388b6521be8cd92ee49d0d origin :refs/heads/codex/track-d-c7-1-q16-exploratory-20261107
```

```bash
git merge-base --is-ancestor 709d6ed9c234bf6d5a8848d3a30be69d88fd6476 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c7-2-cx-structure-20261107
git push --force-with-lease=refs/heads/codex/track-d-c7-2-cx-structure-20261107:709d6ed9c234bf6d5a8848d3a30be69d88fd6476 origin :refs/heads/codex/track-d-c7-2-cx-structure-20261107
```

```bash
git merge-base --is-ancestor acea43987edae0d5eb0c186a89aa8cd7139fbd7b origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c7-3-q32-accelerated-20261107
git push --force-with-lease=refs/heads/codex/track-d-c7-3-q32-accelerated-20261107:acea43987edae0d5eb0c186a89aa8cd7139fbd7b origin :refs/heads/codex/track-d-c7-3-q32-accelerated-20261107
```

```bash
git merge-base --is-ancestor 64072ba81bb70db1df1b34e94adb08e78926f24b origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-c7-4-model-stability-20261107
git push --force-with-lease=refs/heads/codex/track-d-c7-4-model-stability-20261107:64072ba81bb70db1df1b34e94adb08e78926f24b origin :refs/heads/codex/track-d-c7-4-model-stability-20261107
```

```bash
git merge-base --is-ancestor a54d04059bfc61adef43aa2d40fb9c001b62378f origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-cut-artificial-20261108
git push --force-with-lease=refs/heads/codex/track-d-cut-artificial-20261108:a54d04059bfc61adef43aa2d40fb9c001b62378f origin :refs/heads/codex/track-d-cut-artificial-20261108
```

```bash
git merge-base --is-ancestor 848ca0deecc08fd09bbb9027fb372905edd4414d origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-cut-h4-q4-20261108
git push --force-with-lease=refs/heads/codex/track-d-cut-h4-q4-20261108:848ca0deecc08fd09bbb9027fb372905edd4414d origin :refs/heads/codex/track-d-cut-h4-q4-20261108
```

```bash
git merge-base --is-ancestor 2f316fb229149117177367720a6eaa5aa8a94b76 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-cut-h4-q4-completion-20261108
git push --force-with-lease=refs/heads/codex/track-d-cut-h4-q4-completion-20261108:2f316fb229149117177367720a6eaa5aa8a94b76 origin :refs/heads/codex/track-d-cut-h4-q4-completion-20261108
```

```bash
git merge-base --is-ancestor 3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-h4-paired32-reference-20261106
git push --force-with-lease=refs/heads/codex/track-d-h4-paired32-reference-20261106:3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0 origin :refs/heads/codex/track-d-h4-paired32-reference-20261106
```

```bash
git merge-base --is-ancestor 7d0dc55199c49a3f49a9ba93323843d8a336cce1 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-h4-reference-validation-20261106
git push --force-with-lease=refs/heads/codex/track-d-h4-reference-validation-20261106:7d0dc55199c49a3f49a9ba93323843d8a336cce1 origin :refs/heads/codex/track-d-h4-reference-validation-20261106
```

```bash
git merge-base --is-ancestor 102e3c2f29999fb9fbdfa6f85020370df228a7f3 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-minimal-reference-validation-20261106
git push --force-with-lease=refs/heads/codex/track-d-minimal-reference-validation-20261106:102e3c2f29999fb9fbdfa6f85020370df228a7f3 origin :refs/heads/codex/track-d-minimal-reference-validation-20261106
```

```bash
git merge-base --is-ancestor 0ef53fa9a65a70a251a7d6e5dbeb91141a1a67e9 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-o5-c1-h4-q4-20261108
git push --force-with-lease=refs/heads/codex/track-d-o5-c1-h4-q4-20261108:0ef53fa9a65a70a251a7d6e5dbeb91141a1a67e9 origin :refs/heads/codex/track-d-o5-c1-h4-q4-20261108
```

```bash
git merge-base --is-ancestor 14276fe0e2dadef54ef105debe6d6d99b43b6ecb origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-o5-c1-head-tail-artificial-20261108
git push --force-with-lease=refs/heads/codex/track-d-o5-c1-head-tail-artificial-20261108:14276fe0e2dadef54ef105debe6d6d99b43b6ecb origin :refs/heads/codex/track-d-o5-c1-head-tail-artificial-20261108
```

```bash
git merge-base --is-ancestor 62c7b578b8c0bb3d0079434daf637c587f53c24c origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-o5-c1-head-tail-static-20261108
git push --force-with-lease=refs/heads/codex/track-d-o5-c1-head-tail-static-20261108:62c7b578b8c0bb3d0079434daf637c587f53c24c origin :refs/heads/codex/track-d-o5-c1-head-tail-static-20261108
```

```bash
git merge-base --is-ancestor 2131917656446bd22f294946d7c1064a47332e4c origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-o5-design-20261108
git push --force-with-lease=refs/heads/codex/track-d-o5-design-20261108:2131917656446bd22f294946d7c1064a47332e4c origin :refs/heads/codex/track-d-o5-design-20261108
```

```bash
git merge-base --is-ancestor 0d596e7a51a79013bd93cad133b6b14c5b1e2573 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-o5-q4-cohort-metadata-20261108
git push --force-with-lease=refs/heads/codex/track-d-o5-q4-cohort-metadata-20261108:0d596e7a51a79013bd93cad133b6b14c5b1e2573 origin :refs/heads/codex/track-d-o5-q4-cohort-metadata-20261108
```

```bash
git merge-base --is-ancestor 523efe89fde52a14a8e0a4eba4ea6b70e3f512e1 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-pr-rpe-static-survey-20261106
git push --force-with-lease=refs/heads/codex/track-d-pr-rpe-static-survey-20261106:523efe89fde52a14a8e0a4eba4ea6b70e3f512e1 origin :refs/heads/codex/track-d-pr-rpe-static-survey-20261106
```

```bash
git merge-base --is-ancestor a9a74b1386ba713ed3648658dd94be4d992a492e origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-real-pr-transfer-static-20261106
git push --force-with-lease=refs/heads/codex/track-d-real-pr-transfer-static-20261106:a9a74b1386ba713ed3648658dd94be4d992a492e origin :refs/heads/codex/track-d-real-pr-transfer-static-20261106
```

```bash
git merge-base --is-ancestor c3cd64755944be4aad2e032affd1ed0308cf5743 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-reference-control-feasibility-20261106
git push --force-with-lease=refs/heads/codex/track-d-reference-control-feasibility-20261106:c3cd64755944be4aad2e032affd1ed0308cf5743 origin :refs/heads/codex/track-d-reference-control-feasibility-20261106
```

```bash
git merge-base --is-ancestor 71ac8b96c833c09d17ae6a02118471acedd781df origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/codex/track-d-vacuum-path-static-check-20261106
git push --force-with-lease=refs/heads/codex/track-d-vacuum-path-static-check-20261106:71ac8b96c833c09d17ae6a02118471acedd781df origin :refs/heads/codex/track-d-vacuum-path-static-check-20261106
```

```bash
git merge-base --is-ancestor 89a13fe6841c1017f0feba724ed71dfc79f9e81b origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/track-d-b3-rpe-statistics-preparation-20261106
git push --force-with-lease=refs/heads/track-d-b3-rpe-statistics-preparation-20261106:89a13fe6841c1017f0feba724ed71dfc79f9e81b origin :refs/heads/track-d-b3-rpe-statistics-preparation-20261106
```

```bash
git merge-base --is-ancestor 31cd67c91abda3c9c5d4b643c434c040b8ceaeca origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/track-d-b4-1-rpe-statistics-implementation-20261106
git push --force-with-lease=refs/heads/track-d-b4-1-rpe-statistics-implementation-20261106:31cd67c91abda3c9c5d4b643c434c040b8ceaeca origin :refs/heads/track-d-b4-1-rpe-statistics-implementation-20261106
```

```bash
git merge-base --is-ancestor 07d18bfe1d5145d6f8de026c0d994d5cb1c783c8 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/track-d-b4-2-rpe-shots-science-20261106
git push --force-with-lease=refs/heads/track-d-b4-2-rpe-shots-science-20261106:07d18bfe1d5145d6f8de026c0d994d5cb1c783c8 origin :refs/heads/track-d-b4-2-rpe-shots-science-20261106
```

```bash
git merge-base --is-ancestor 0a8b953aea1be20b110784960763d7f36ebccac8 origin/codex/track-d-consolidated-20261109
git ls-remote --heads origin refs/heads/track-d-b5-actual-rpe-static-survey-20261106
git push --force-with-lease=refs/heads/track-d-b5-actual-rpe-static-survey-20261106:0a8b953aea1be20b110784960763d7f36ebccac8 origin :refs/heads/track-d-b5-actual-rpe-static-survey-20261106
```

各削除後に当該branchの不在と保持refの残存を確認する。認証・保護・使用状況・SHAが不明ならHOLDを維持する。新しい秘密情報の作成や共有設定変更はこの手順に含まれない。
