# 保留7枝の成果・再現用ファイルの追加収録

2026-11-04。比較基準mainは`180292ae96d6797f59b56896072337fd1f652ea4`。
前回、別の公開履歴保持先が未確認だった7枝について、公開snapshotのpath・blob・modeをmainと照合した。
新しい科学計算・結果の再解析・科学的採否の判断は行っていない。

## 追加したもの

既存の報告・Python SOURCEの多くは、同じpath、別pathまたは保存原版へ既に収録されていた。
今回は53項目、5,590,070 bytesを元のblob・modeで追加した。異なるpath間の同一blobを含むため、内容は49種類である。
[出典と配置の対応表](priority7_collection_20261104_files.tsv)で各原path・公開snapshot・blob・mode・配置先を確認できる。

- 元のpathで不足していた23ファイル：Qiskit参照SOURCE 6件、依存lock 12件、保存patch 5件。
  既存の報告・再現記録が参照するpathを補った。patchは未適用、lockの環境は未構築である。
- 同じpathでmainと異なる30版：研究概要・索引・validation manifest等29版と、共有`src/trotterlib/rte.py`の1版。
  `docs/archive/priority7_collection_20261104/<blob>/<原path>.original`へ保存し、mainの採用版は維持した。

Qiskit SOURCEの著作権・Apache Licenseのヘッダを元のまま保持し、同じ参照ディレクトリに
[既存LICENSE.txt](../../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/upstream/LICENSE.txt)があることを確認した。
原SOURCEの出所と抜粋／全文の区別は[元の報告](../../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/README.md)を参照する。

保存した概要・manifest・共有SOURCEの旧版は、当時の研究contextを確認するための資料である。
旧版に記載された「現在」、次段階、実行許可、statusをmain全体の現行仕様へ読み替えない。
閲覧用の公開snapshotと、各実験で固定された科学SOURCE・入力identity・runtimeを区別する。

## 照合した固定snapshot

下表の数は、mainとの共通祖先から各snapshotまでの変更pathを対象にした照合件数である。
共通祖先から継承した別系列の成果や原データも含み、各枝で新たに実施した実験の件数ではない。
raw・大きなCSV・QPY/NPZ・運用記録等の不足pathを、未収録の研究報告と同一視しない。

| 当時の枝 | 固定公開snapshot | 変更path数 | 同一内容をmain内に確認したpath数（追加前） | 今回の関係 |
|---|---|---:|---:|---|
| `codex/pauli-pure-rte-paper-evidence-addendum-20261102` | [f6eb34dad982](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/f6eb34dad982f8f753a618c609b528fb3c20d142) | 20887 | 19146 | 出典対応表の17項目に同じ原版がある |
| `codex/pr-coverage-inventory-local-20261102` | [17aacf67ee85](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/17aacf67ee854657652ade8035c5fdfad1eb077a) | 20657 | 18946 | 出典対応表の17項目に同じ原版がある |
| `codex/pr-coverage-inventory-server1-20261102` | [2de84ed7af00](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2de84ed7af00e73339886429aabab0bf8180e9f0) | 20686 | 18946 | 出典対応表の17項目に同じ原版がある |
| `codex/pr-coverage-inventory-server2-20261102` | [8834de0cf938](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/8834de0cf938b13490c08e05006e3c566fed7c7a) | 20658 | 18946 | 出典対応表の17項目に同じ原版がある |
| `codex/research-c-contribution-positioning-20261102` | [88afb1acc78d](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/88afb1acc78dbc9b4af2bb5e149cd873a914b03b) | 11830 | 6886 | 出典対応表の27項目に同じ原版がある |
| `codex/research-c-r32-price-statistics-execution-20261103` | [459c8472d666](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/459c8472d66689bf149a60fd6cc7311ff5237c65) | 20113 | 11460 | 出典対応表の27項目に同じ原版がある |
| `codex/research-c-u-step-control-execution-20261103` | [ea54b9b8ba42](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6) | 13244 | 6980 | 出典対応表の27項目に同じ原版がある |

各枝に関係する追加項目数は重複を含む。同一内容の対応は出典TSVで共有する。
7本とも、この照合時点では先端を保持する別の公開ブランチ・タグが見つからなかった。
内容の選択コピーと、枝全体のGit履歴のmerge・保持先の新設は別の処理である。
今回の追加収録だけを理由に、これらの枝を削除しない。

## 元の場所に残すもの

大きな原データ、raw/QPY/NPZ、bundleや分割part、巨大なcoverage inventory CSV、
実行時のclaim/lock・cache・一時記録等は今回の追加対象外である。
これらの存在とsizeはGit treeのmetadataで確認し、decode、数値再解析、全hash、コピーは行っていない。
元の公開snapshot、既存のサーバー保全記録・保存先を維持する。
非公開原本・SOURCE withheldの範囲を拡張して公開する作業は含まない。

## 確認範囲と次の処理

53項目の出典path/blob/modeと配置先bytesを照合した。
PythonはASTのみ、JSONは構造のみを確認した。Rustのbuild、研究test、元runner、patchの適用、
lockのinstall・wheel hash検証、科学結果の再評価、SOURCEの実行互換性検証は行っていない。
既存の科学SOURCE・入力・原結果・契約・grant・ledger・marker・STOPを維持する。
新しい科学実行は0件である。

前回の判断表に載っていなかった34枝も公開履歴とtree metadataで一次分類した。
28枝は別の公開履歴保持先が見つかり、5枝は保持先未確認、1枝は以前に復元した固定検証用の枝だった。
この分類は削除許可・利用終了・mainへの完全収録を意味しない。原データを含む全変更pathには内容差が残る。
サーバー側の未commit/未push成果・可視job・現在の用途は、この一次分類では新たに確認していない。

今回、リモート枝・タグの追加削除は行わない。両サーバーの差分確認が返ったら、ローカル側で
成果の収録・現在の用途・履歴保持先を統合し、その後の整理対象を具体化する。
