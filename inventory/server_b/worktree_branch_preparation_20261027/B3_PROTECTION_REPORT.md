# DF-native B3保護・非干渉確認

2026-10-27 JST。対象はlinear H4/H6・1Å・STO-3G・tol-only DF10/19、one-bodyを保持するL_D=0 B3。旧Track B/G6/W2Cの同名B3とは別研究。

**今回実際に確認したB3状態：取得未完了・停止観測。** 保存statusはSCIENTIFIC_ACQUISITION_RUNNINGのままだが、本人UIDの既知controller/workerは不在、2 user serviceはnot-found/inactive。既知journalには2026-10-27 09:36 JSTの停止処理とexit1があり、stderr末尾にも終了記録がある。最後の保存counterは探索1691/1728、active17/pending20のまま凍結している。このactive値を現在のprocess数として使わず、停止前の未反映結果の有無も本調査では数え直していない。

停止要求の主体や理由を本調査で断定していない。serviceがnot-foundのため表示するResult=successやstatus0を科学正常終了として採用せず、stages完了の確認・科学受入れ・再実行可の認定はしていない。今回の作業は停止・signal・再開・retryを一切実行していない。

保護する登録IDはSERVER_B-WT-0003（主共通Git owner）、SERVER_B-WT-0123（B3 SOURCE/実行checkout）、SERVER_B-WT-0124（保存基準研究checkout）。分類ラベルKEEP_ACTIVE_B3はこの保護意図であり、今も計算中という主張ではない。外部の保存provider/input/state/reference/private raw/結果/checkpoint、Python runtime/venv/cache、grant/ledger/marker/STOP、復旧trace・固定seed/1回復旧条件、共有Git/object storeと参照を保持する。

既存13 scriptのhash一致は前回観測の証拠として再利用し、今回全SOURCEを再hashしていない。原SOURCE/数値入力/結果を変更せず、状態配列やcacheの再帰走査、科学module/runner/verifier、compile/sampling、数値再解析は0。B3関連treeではgit status走査も保留した。

今回の整理調査によるB3 process/service/scheduler/cgroup/共有環境変更は0。書込みは独立した本人専用調査領域と公開cloneだけ。停止しているから物理資産が不要とは扱わず、整理対象から除外した。

次：ユーザー/研究担当がB3停止の既存事情と今後の復旧扱いを確認し、必要なら別の明示指示で扱う。本整理準備は再開や復旧条件変更を実施せずmandatory STOPする。
