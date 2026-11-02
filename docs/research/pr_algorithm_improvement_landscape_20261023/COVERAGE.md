# 調査coverage・方法・限界

## 発見方法

既知名の検索に加え、GitHub branch全ページ、各固定HEADのtree全path、Markdownの全異なるblob、commit subject indexを取得した。root/README/研究索引/closure/handoff/STOP/PAUSE等の文書を横断し、未知だった関連repository・研究名から一次資料へ展開した。タイトル/状態の機械的走査と、意味を読んだ主要資料の選択を区別する。

| Repository | 公開branch | 異なるpath | Markdown blob | commit subject記録 |
|---|---:|---:|---:|---:|
| HIROMU1015/Partially-Randomized-Trotter | 559 | 43563 | 5942 | 1774 |
| HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae | 168 | 5222 | 593 | 496 |
| HIROMU1015/Evaluation_numGate_highorder_DF | 4 | 1117 | 12 | 54 |

branch総数731は観測されたrefの数で、研究数・独立実験数ではない。Markdownのblob数はrepositoryごとで、同一内容のcopyを独立証拠として増やさない。主要ファイル211件は[SOURCE索引](SOURCE_AND_PROVENANCE_INDEX.json)で個別identityを登録。うち16件をconnectorで固定commitから再取得し、取得本文からGit blob hashとSHA256を照合した。その他はAPIで観測したHEAD/固定参照とローカルGit blobの照合で、個別HTTP取得とは区別する。

## 入口・source priority

PROJECT_MAP、研究概要・現状を最初に読み、規範文書、VALIDATION_STATUS、manifest、validation固有文書・resultを接続した。Track B README、QC coverage/foundation/4-track handoff、高次PF完結性監査・portfolio・retrospectiveを追加入口とした。概要だけでは古いPR-2停止位置が残るため、後続M1/M2/PMを明示的に追跡した。単一mainやactive IDE tabを対象決定に使っていない。

## 採否・除外

- 成功/負の結果/未完了/技術停止/理論/草案をすべて含める。純粋な資源計測・runtime修正はfoundation/evaluationとして分離する。
- SOURCEの静的構造とGit/manifest identityは確認するが、旧scientific validator/tests/runnerを実行しない。原rawを使う新しいfit、係数、shot再会計、順位は生成しない。
- 発表構成案、旧protocol/notebookは背景扱い。IDE active/open tabは指示・scopeとしない。
- private review原本・grant、内部絶対path、binary matrix/state/pickleを新資料へ転載しない。既に公開された旧成果を不要に再pushしない。
- 同じ研究のversion/retryは統合する一方、C1 macrocycleとTrack B C1、S4 PFとParent allocation S4等の同名を別IDにする。
- 詳細な新規先行研究レビューは行わず、既存の文献対応・既知還元を帰属付きで利用する。

## 網羅性の限界

全公開branchのpath一覧取得は、全branchの全SOURCE/rawを精読・独立監査したことを意味しない。70研究単位には複数の密接なstage/候補をまとめたfamilyもあり、全run一覧ではない。条件の完全な転記がないカードは元契約への固定参照を残す。原文書にある過去の「次」「未承認」を現在の再開指示へ移さない。

削除済みremote branch、reflog/未push commit、Git管理外の全cache、別host/他ユーザー、未公開原本、関連しない別repository、論文全世界の網羅性は保証しない。公開branchのページは逐次取得なので単一瞬間のatomic snapshotではない。参照した個々のcommitは固定した。

## 資源と保全

主負荷はGit tree/blobと文書I/O。初期観測はCPU32、RAM約62GiB中available約51GiB、disk空き約350GiB。独立treeは利用可能CPUに合わせたthread pool、blobはhashで重複除去・cacheした。GPU科学計算・ライブラリimport・共有設定変更なし。実測比較をしていないためspeedup倍率を主張しない。

元rootはdirtyで211登録worktreeがある。既存worktree/branchの切替・編集・stageを避け、/tmpの独立cloneでmain固定commitから新資料のみ作成した。元status/refs/worktree登録の前後照合はVERIFICATIONへ記録する。ローカルGit clone/cache操作の一回で元.git/FETCH_HEADへの書込がread-only拒否となったが、元refs/status不変を確認し、以後すべて隔離cloneを明示した。科学SOURCE/rawへの書込はない。

資料十分性A/B/Cと成績区分の件数は、カタログ上の文書分類である。科学的成功率・実験統計ではない。
