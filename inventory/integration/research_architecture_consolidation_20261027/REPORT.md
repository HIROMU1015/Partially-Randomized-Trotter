# Research architecture consolidation report

LOCAL / 2026-10-27T17:02:00.871221+09:00。最新remote集合 **487本**を版固定して分類した。前回484本を固定値とせず、今回着手485本から追加されたPauli readiness/executionもmetadataとして取り込んだ。基準公開資料は `2f55ad4ec4a3f006b5b97c6f7855629547237ed9`。

## 全体設計

**5主科学研究系列＋1歴史研究portfolio**を分け、成果/原稿・共通技術基盤・環境運用の3支援層を置く。計9役割の分類である。Research CのC1/C4/PR-aware、Track BのG6/M1-S γ/π3、DFの分子/高次/長時間等は独立条件を残す。これは実装/研究の整理単位であり、全独立仮説・実験数の確定ではない。2026-10-23既存catalogの当該repo55単位は歴史索引として再利用し、現行project数へ転記していない。

|ID|研究系列/支援層|branch数|実装/成果代表候補|
|---|---|---:|---|
|RS-DF|DF / PR / PF / finite RTE / RPE|110|`all-r-coherent-opt2-reoptimization`|
|RS-PAULI|Pauli-JW fixed-signal / Hybrid / pure RTE|11|`codex/pauli-jw-research-summary-20261026`|
|RS-TB|Track B algorithm co-design|159|`track-b-g6-v1-2-draft-review-integration-20261022` / `track-b-m1s-v1-gamma-formal-adoption-20261017` / `track-b-m1s-gamma-pi3-formal-adoption-20261017`|
|RS-B2D|Research B / B2-D / outer|23|`codex/research-b-final-pause-handoff-20261026`|
|RS-C|Research C C1 / C4 / PR-aware DF|84|`codex/research-c-c1-h4-native-fair-execution-20261026` / `codex/research-c-c4-closeout-c1-survey-20261024` / `codex/research-c-pr-aware-df-zsxx-uniform-preparation-20261027`|
|RS-HIST|Legacy research portfolio / N1-N3 / QC / closed selectors|78|固定歴史catalog/closure。開発自動再開なし|
|RS-EVID|研究成果 / CEM / 原稿 / 図表|19|`codex/pr-boundary-evidence-integration-20261027`|
|PL-CORE|共有計算・実行基盤|2|`all-r-coherent-opt2-reoptimization`|
|OP-INV|環境棚卸し・運用資料|1|`codex/pr-environment-inventory-20261026`|

RS-DF/PL-COREは同じ既存all-rを開発候補として共用。科学的正本は分子別SOURCE/resultのまま。Research Cの代表は現行入口の維持候補であり統合実装を保留する。最新作業branch・科学的正本・実装代表を同じものとみなさない。

## 推奨アーキテクチャと代表入口

既存trotterlibを数値/回路/会計/実行の共有基盤、trottertracksを研究adapter/composer、scriptsを明示CLI、既存CEM/manifest/docsを成果導線とする。既存pathの一括移動はせず、新規利用に必要な薄いcomposerだけを提案する。共通基盤へ全研究logicを押し込まず、研究側からcoreへ依存する。

**推奨入口は既存branch計10（code 9＋成果1）**。DF/core1、Pauli1、Track B3、B2-D1、Research C3、成果1。新設branch数0、現在稼働中job数を意味しない。inventory branchは運用先として別扱い。各入口の完全tip/機能移行元/科学正本/現行checkout観測はCSVに分けて記録した。

別branchのmoduleは自動importできない。将来の共通core差分は、後続明示許可下でconsumer代表branchの同じcheckoutへ選択取り込みし、その共通commitを新SOURCEへ記録する。別worktree実path・共有venvに依存する設計やwhole branch mergeを採用しない。mainは変更しない。

## 共通基盤・分散・独立維持

12パイプラインcapabilityについて入力/出力/責務/依存/互換条件を具体化。重要な共有領域はHamiltonian型、PF/finite RTE/event、回路/backendとcompiler context、estimand別会計、process/output/checkpoint機構、証拠locator。個別utilityではなく、DF/Pauliの端点間接続と研究adapterの境界を先に整える。

- DFの共通library11 moduleは既継承だが、H4 geometry input/circuit/signal等の後続機能は旧baseに全てはない。入力/contractをcoreへ混入せず新composerから接続する。

- Pauliの分子別runnerはsnapshot namespace・dispatch・target/price/報告が分散。共通kernelを再利用し、現行固定runnerを変えず新規配線へ切り出す案。H6公平性調査はPauli-JWであり前回のDF仮分類を修正した。

- B2-Dはfinal-pauseへevaluator/backend/outer/Exact-R/Full-DF/q4を既継承。新たなmergeよりstage別入口と責務/資格の整理を優先する。

- Track BはG6後続にF3 generator追加を確認。一方M1-S v2はG6代表に不在、M1-S γとπ3も祖先でない。3入口と独立数理を維持し、descriptor/storageなど適合点だけを共有する。

- Research C現行研究とB3、N1/N3旧SOURCE欠落、閉じたR2/QC等は独立維持。code再利用が研究再開・科学採択の許可を意味しない。

## 重点5系列と実装順序

|順|フェーズ|大きな移行単位|
|---|---|---|
|準備|P0|既存研究入口と版固定したSOURCE/result/CEMの導線|
|1|P1|DF端点間pipelineと共有計算/実行基盤の責務分離|
|2|P2|Pauli-JWの分子横断pipeline・固定snapshot共存|
|3|P3|Research B/B2-Dのstage/evaluator/backend入口集約|
|4|P4|Track BのG6/M1-Sと合成/生成の独立入口接続|
|5|P5|成果・図表・論文を元SOURCE/resultへ版別接続|
|後段|P6|前回INT-01/02/03/05の小helperを実需要に応じて整理|

4科学系列＋成果1について、branch完全tip・実module/API・移行差分・保持条件・既存テスト/不足検証・リスク・受入条件・順序をINTEGRATION_ROADMAP.mdに記載した。最初の大規模実装候補はP1であり、今は未着手。

## 調査根拠と限界

正式運用指示2026-10-25の実ファイル（§8.1/8.4/10）を確認。前回7公開資料と非公開対応/静的比較を再利用し、PROJECT_MAP/現状/VALIDATION_STATUS/manifestの案内はその版とscopeに限定した。古いmain案内だけで最新科学判断をしない。23選定branch treeのうち18を再利用しM1-S π3のtreeを追加、759選定path存在/同blob関係、35 pipeline資料（10再利用）と4追加Track B資料、3つの公開SOURCE/result入口、既存70行study catalog中当該repo55単位を静的確認した。関連外repoの追加調査、全branch全ファイル、raw全件走査は行っていない。

既存testは前回静的読解または今回tree上の所在確認で区別し、今回はテスト未実行。API互換/mock/技術regressionと、数式・既存成果物の適用域・数値再検証・SOURCE/cohort同一性を後続計画で分けた。sampling/compile/fit/operator数値検証は別の明示科学許可を要する。性能改善や一般科学的優劣を新たに判定していない。

## 保護・公開・次の判断

B3の5worktreeと公開保護9 refは内容アクセス/操作0。Research Cの現行環境や他jobを調査/変更せず、並行更新を戻さない。N1/N3旧SOURCE11件/3版55宣言、独立cohort/失敗/STOP、grant/ledger/marker、one-shot/retry/resume、private原本を保持した。既存原本/archive/bundleは変更しない。

今回公開はこの新規directoryの8資料のみ。内部絶対path/host/IP/private配置/非公開branch実名/秘密復旧情報/credentialを除外し、正確な対応は本人専用の非公開原本へ保存。既存inventory branchの最新履歴へ通常fast-forward commit/pushし、remote SHAと公開内容を照合する。研究branch/main/コード/test/SOURCEの変更、merge/rebase/cherry-pick、branch/worktree削除、統合branch作成、科学計算/数値再解析、テスト実行は全て0。

ユーザーが次に決める事項: (1) 5主系列＋履歴/支援層の構造と10入口候補、(2) all-rをDF/coreの候補にするか、(3) 最初のP1で接続するH4非B3 adapterとAPI/context境界、(4) 新規非凍結経路だけを対象にする互換方針。

Codexは次回の明示実装指示後、選ばれたP1等の系列だけを実装し、許可範囲のAPI/mock検証を行う。科学実行・Research C統合・B3調査・研究再開は別判断のまま。今回の設計・計画・匿名化公開後はmandatory STOP。

公開前の並行更新: `codex/h6-hybrid-fairness-extension-execution-20261027` @ `887b69e449cb3c378286f09d0d7626599850fb67` を追加した。readiness/executionというbranch名からjobの進行/完了や新結果を推定しない。新branchのcode/SOURCE/result/実行環境は読まず変更せず、固定FAIR audit時点の提案状態と区別する。
