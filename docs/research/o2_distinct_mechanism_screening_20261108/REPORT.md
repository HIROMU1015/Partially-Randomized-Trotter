# O2：非同値性を先に確認した探索の報告

2026-11-08 JST／SERVER1 GPUA100-gleap-01。base f085b56566d60d9cb5df221f352a2b52ac51ea88、専用branch codex/o2-distinct-screening-20261108、独立worktree /tmp/o2-distinct-screening-20261108。

**既存法へ還元されず、改善条件も支持できる候補は今回0件。新しいinstrumentの実装へ進まなかった。** これは今回の選別結果で、O2の継続方針や旧結果・STOPを変更する判断ではない。

## 成果

- [候補の操作・式・選別](DISTINCT_MECHANISM_CANDIDATES.md)：normalizer二段取得、finite-state sampling/return、座標observable測定を具体化して、還元と未解決アクセスを実装前に区別した。
- [強いbaselineとの照合](EQUIVALENCE_AND_BASELINE_CHECK.md)：同じ情報・oracle・fusion/early abortを両側へ認め、SOURCEの提供範囲と不足を示した。
- [次の提案](NEXT_RESEARCH_PROPOSAL.md)：実装・科学実験を推薦する候補なし。安価なsummary又は測定アクセスの具体構成が出た場合に限る判別条件を残す。

| 方向 | 主分類 | 次段階 |
|---|---|---|
| normalizer表を使わない二段推定 | REDUCES_TO_EXISTING_METHOD | 既知rejection/ISで同law/weight/費用を再現、実装しない |
| finite-state joint sampler/return | DISTINCTNESS_UNRESOLVED | 再帰は既知法。新しい安価なsummary取得を未特定、実装しない |
| 座標observableのoutcome変更 | KNOWN_METHOD_APPLICATION | 同じアクセスなら既知direct測定。新しい低価格構成なし、実装しない |

二段推定の不偏性/normalizer分散項、backward score再帰のtelescoping、全状態不偏POVMのM2≥H²を記号的に整理した。一般の新定理・新規性・科学的GOの認定ではない。spectral測定費を無料にして候補を否定せず、抽象的なPOVM/Returnへの包含と費用付き操作的同値を区別した。

## 再利用と調査範囲

O2統合d28164b5、M metadata結果e98589ae、E設計08195940/人工結果f085b565の必要箇所を再利用。Mの232単独群やEの16 PASS/同early LCU差0は元REPORTの引用で、旧raw・bit・新しい集計を行わない。39 stageを全面再監査しない。

ReturnKernel/Parent外部価格・actual finite law/conditional取得・A1/CV分離・A2/E有料制御・M header/測定規約・RTE/Hadamard・D2/full費用の既存SOURCEを静的参照した。新しい関数・library・runner/test・監査frameworkは作成しない。一次文献はBernoulli race、weighted automata、unbiased observable estimationの必要箇所に絞り、論文の異なるtarget/費用モデルをPRへ移植しない。

正式共通指示2026-10-25（SHA256 e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720）は前作業で読了した同一本文と今回の適用条項を確認。適用AGENTS/PROJECT_MAP、既読の研究概要・目的/方法と最新追記を確認した。§8.1/8.2/8.4/10/10.1/10.3と実装前の非同値性確認という最新ユーザー指示を適用した。

今回は式とSOURCE比較で十分だったため、新code・人工代数test・旧test実行・量子simulation・Born/RNG/shot・回路build/compile/native synthesis・実H4科学計算・保存科学値数値再解析・重要科学レビューは全て0。文書/byte照合は科学計算や人工correctnessに数えない。

## 保全・公開・STOP

元main・dirty/untracked、旧O2/M/D/E・R0–G10/Parent/C1凍結・S2/A1/SCV/A2/J1・O1/O3/C-Sの固定SOURCE/raw/結果/STOP/grant/ledger/marker、他worktree/process/cacheを保全。既存索引・概要・当日ノートは追記のみ、manifestは今回の文書選別項目だけ追加して旧科学的statusを維持する。M/D/E追加研究・旧STOPを再開しない。

共有OS/scheduler/cgroup/venv/cache・他ユーザーを変更せず、文書作業に新しいCPU/RAM/worker上限・科学managerを設けない。

専用branchでcommit/公開し、remote SHA・公開本文・blob/tree・固定資料参照・旧manifest/元main保全を照合する。最終SHAは最終チャットと作業用照合記録に示し、自己SHA追記用の追加commitは作らない。

**公開照合後mandatory STOP。新科学・新instrument・旧研究再開・重要科学レビューへ自動進行しない。**

公開前の文書技術確認PASS：新4文書の22参照、固定Git3資料、旧manifest24項目不変、既存Markdown5点の追記のみ、元main HEAD/statusと既存2,136ファイルbyte保全を確認した。代数/量子/科学の追加検証ではない。
