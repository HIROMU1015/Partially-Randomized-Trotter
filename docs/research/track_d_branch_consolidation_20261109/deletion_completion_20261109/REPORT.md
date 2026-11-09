# Track D remote branch削除完了

2026-11-09 JST。**38本を実際に削除し、全件で不在と履歴保全を確認した。** 認証復旧後の実行は `2026-11-09T17:31:34.737111+09:00`〜`2026-11-09T17:33:14.120537+09:00`。

初回は通常Git認証がなくHOLDとして公開した。ユーザー提示の既存credential cacheをコマンド単位で利用することで認証が成立した。科学的条件や既存STOPを変更せず、同じ38候補だけを処理した。

| 項目 | 実績 |
|---|---:|
| 初回棚卸し時の全remote | 405 |
| 元Track D対象 | 43 |
| O2移管対象 | 2 |
| 認証復旧後の削除開始時remote | 354 |
| 当作業による実削除 | **38** |
| 削除後remote | **316** |
| 元対象の保持 | **7** |
| 統合branchを含むTrack D/O2保持ref | **8** |
| 削除処理の失敗/未確認 | **0** |
| 新規archive tag | 0 |

初回405本からの変化には他チャットの整理・新設を含む。当作業の削除実績は今回の354→316の38本であり、差額全体を当作業に帰属しない。この削除処理中、対象外refの消失は0件。

## 実施した確認

- 元38候補の全HEAD、保全先remote refと祖先関係、最新branch protectionフラグ、default main、open PR0件を再確認。
- 対応する本人worktreeはclean、開始時HEADから変化なし、候補worktreeを使う本人の実行中processは0。登録worktreeを削除していない。
- 各branchを個別の `--force-with-lease=refs/heads/<name>:<original SHA>` で削除。各操作直前にHEADと保持refを照会し、直後に不在・保持refの残存・祖先関係を確認した。
- 38件全て `DELETED_VERIFIED`。wildcard削除、force overwrite、rebase、main merge、無条件prune、gcは0。
- 元45 tipの履歴は統合branchまたはO2保持refから全て到達可能。元31 worktreeのHEADは全て不変。private raw/QPY/NPZや既存SOURCE/result/STOPを移動・削除・編集していない。
- 既存64人工・技術回帰PASSを継承。今回はdocs/refだけの操作であり、回帰の不要な再実行・科学計算・sampling・compile・価格再解析・fit・N/RPE再算定は0。

## 削除した38本

| branch | 削除直前の元HEAD | 保持ref |
|---|---|---|
| `codex/o2-vacuum-cswap-20261108` | `0edc0fa5ad4656fdd492ca6c96ade4f77cadf412` | `codex/o2-vacuum-h4-preparation-20261108` |
| `codex/track-d-b1-c1-validation-20261106` | `d3913866e6ade4ab97ac02645fed91b2d5fcfa75` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-b6b-bc-integration-20261108` | `7ffe8b281b76cef3a6fd06f43413904d6766d0ec` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-bc-static-preparation-20261106` | `5f94972adf5a10e96e9400d3919c00d14e7a718a` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-bc-total-20261108` | `8828b298975c88a2c0d1fd57ff6ed905233840f5` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c2-q2-compile-20261106` | `c8fe5091c42ec786688b6931105ad64d10c87d18` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c3-q4-q8-validation-20261106` | `cc554cf6daea08df6c20ea8e1e92405259ce49b5` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c4-price-model-preparation-20261106` | `a5784c4bd3e3fb9779849a1e1fc1759559e2caea` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c5-whole-path-q2-20261106` | `9a6920b8355643701ab7436a4fd3b34862e9e3bd` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6a-paired-price-pilot-20261106` | `e9d4e2f602d5f9e014507ca7e391719bab017b88` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6b-accelerated-completion-20261107` | `fed5f458be2f625de4e1b8c98b8cc5da6e551b14` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6b-q8-exploratory-20261107` | `23725fc6f6d6d0c2bec93e31e32074711804e250` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6b-state-action-acceleration-20261107` | `75f43855b02a49e710d7a2ba3ee82eda3de1631e` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c6b0-paired-model-preparation-20261107` | `c93f3c2373b158f8eff3800a62fe1a94cce5c3b0` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-0-structural-preparation-20261107` | `8b422c28c22666a32103fdae33a3c9b9512ef248` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-1-q16-exploratory-20261107` | `86c5c9d48dbddc085f388b6521be8cd92ee49d0d` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-2-cx-structure-20261107` | `709d6ed9c234bf6d5a8848d3a30be69d88fd6476` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-3-q32-accelerated-20261107` | `acea43987edae0d5eb0c186a89aa8cd7139fbd7b` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-c7-4-model-stability-20261107` | `64072ba81bb70db1df1b34e94adb08e78926f24b` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-cut-artificial-20261108` | `a54d04059bfc61adef43aa2d40fb9c001b62378f` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-cut-h4-q4-20261108` | `848ca0deecc08fd09bbb9027fb372905edd4414d` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-cut-h4-q4-completion-20261108` | `2f316fb229149117177367720a6eaa5aa8a94b76` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-h4-paired32-reference-20261106` | `3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-h4-reference-validation-20261106` | `7d0dc55199c49a3f49a9ba93323843d8a336cce1` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-minimal-reference-validation-20261106` | `102e3c2f29999fb9fbdfa6f85020370df228a7f3` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-c1-h4-q4-20261108` | `0ef53fa9a65a70a251a7d6e5dbeb91141a1a67e9` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-c1-head-tail-artificial-20261108` | `14276fe0e2dadef54ef105debe6d6d99b43b6ecb` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-c1-head-tail-static-20261108` | `62c7b578b8c0bb3d0079434daf637c587f53c24c` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-design-20261108` | `2131917656446bd22f294946d7c1064a47332e4c` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-o5-q4-cohort-metadata-20261108` | `0d596e7a51a79013bd93cad133b6b14c5b1e2573` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-pr-rpe-static-survey-20261106` | `523efe89fde52a14a8e0a4eba4ea6b70e3f512e1` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-real-pr-transfer-static-20261106` | `a9a74b1386ba713ed3648658dd94be4d992a492e` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-reference-control-feasibility-20261106` | `c3cd64755944be4aad2e032affd1ed0308cf5743` | `codex/track-d-consolidated-20261109` |
| `codex/track-d-vacuum-path-static-check-20261106` | `71ac8b96c833c09d17ae6a02118471acedd781df` | `codex/track-d-consolidated-20261109` |
| `track-d-b3-rpe-statistics-preparation-20261106` | `89a13fe6841c1017f0feba724ed71dfc79f9e81b` | `codex/track-d-consolidated-20261109` |
| `track-d-b4-1-rpe-statistics-implementation-20261106` | `31cd67c91abda3c9c5d4b643c434c040b8ceaeca` | `codex/track-d-consolidated-20261109` |
| `track-d-b4-2-rpe-shots-science-20261106` | `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` | `codex/track-d-consolidated-20261109` |
| `track-d-b5-actual-rpe-static-survey-20261106` | `0a8b953aea1be20b110784960763d7f36ebccac8` | `codex/track-d-consolidated-20261109` |

## 保持する元7本

| branch | 分類 | 理由 |
|---|---|---|
| `codex/o2-vacuum-h4-preparation-20261108` | KEEP_INDEPENDENT | O2のA1依存・state-free/phase/POVM実装の正式contextを維持。 |
| `codex/track-d-c7-3-q32-exploratory-20261107` | HOLD | 元runと収集script2本が未追跡。今後の用途確認が必要。 |
| `codex/track-d-c8-1-q64-20261108` | KEEP_ACTIVE | q64特別保護。固定SOURCEと未公開closeoutの参照を維持。 |
| `codex/track-d-c8-1-q64-accelerated-20261108` | KEEP_ACTIVE | q64特別保護。固定SOURCEと未公開closeoutの参照を維持。 |
| `codex/track-d-c8-1-q64-completion-20261108` | KEEP_ACTIVE | q64特別保護。固定SOURCEと未公開closeoutの参照を維持。 |
| `codex/track-d-c8-1-q64-nofusion-repair-20261108` | KEEP_ACTIVE | q64特別保護。固定SOURCEと未公開closeoutの参照を維持。 |
| `codex/track-d-c8-1-q64-numerical-repair-20261108` | KEEP_ACTIVE | q64特別保護。固定SOURCEと未公開closeoutの参照を維持。 |

統合先は `codex/track-d-consolidated-20261109`。そのほかに上記7本を保持する。O2元CSWAP tipはO2 H4保持tipの祖先、43 Track D tipは統合履歴の祖先であり、新規archive tagなしでも全履歴がremoteから到達可能である。

機械可読の[実行記録](EXECUTION.json)、[最終分類](INVENTORY_FINAL.json)、[原本保全確認](PRESERVATION.json)を参照する。[初回HOLD報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d12d7c6cf3da8f739e0238f7abbbc7018bb6d53/docs/research/track_d_branch_consolidation_20261109/FINAL_REPORT.md)と初回STOPはそのまま保持する。

## 次の担当とSTOP

- ユーザー／GPT：q32未追跡枝の今後の用途、q64の既存完了・公開作業、O1/O2/O3共同最適化の研究範囲を判断する。
- Codex：本完了追補のremote SHA/tree/公開bytesと残存refを照合して **mandatory STOP**。追加branch削除、科学実行、詳細科学レビューへ自動進行しない。
