# 段階Iの静的検査・公開照合

管理用verifierは標準libraryでSOURCEをテキスト／ASTとして読むだけである。task package、runner、test、科学dependencyをimport・実行しない。py_compile／pytest／collection／numerical mock／scientific preflightも実施しない。

## 実施した静的検査

- 新Python14fileのAST parse、関数／class／signature、module-level科学処理の不在、禁止APIの呼出し探索。
- 科学module importより前にstdlib authorization gateがあること、SOURCE/input/env/case/one-shotの拒否経路、実行用grant発行APIの不在をSOURCE読解。
- referenceがcandidate algebra/router/canonicalizer/momentをimportしないこと。独立のCAR/exterior-minor経路であることを静的に確認。
- 内部SOURCE80fileへ到達する保守的import graph。親package __init__／conditional importsも含む。external runtimeの解決は未確認。
- test関数20件の定義、TESTS_NOT_EXECUTED表示、P0 F1/F2条件・閾値・STOPの対応。
- 固定資料33recordのbytes／SHA256／Git blob。うちP0必須13とPhase2C4／R1 3／support helper1の21件は固定GitHubから取得したidentityとも照合。
- 新SOURCE manifest、JSON parse／必須field／identity、文書リンク、privacy pattern、変更path allowlist、既存tracked filesの非変更。

SOURCE上のAPI接続はrte.pyのRTEComponent／make_rte_config／_make_event／iter_rte_eventsとsignatureを静的照合する。Qiskit／mpmathのruntime型／convention／enclosure／compiler依存は検証していない。新source同士の型annotationはruntime type checkerで実行していない。

静的PASSは実行時の正しさ、科学gate、confidence certificate、性能、新規性のPASSではない。sourceを実行しなければ判定できない箇所は[B引継ぎ](INPUT_BINDING_HANDOFF.md)と[blocker](READINESS_AND_BLOCKERS.md)に残す。

## 原本と公開

起点b8d54708から、新package・専用runner・test定義・新技術記録へのA追加だけを許可する。既存SOURCE／入力／結果／grant／ledger／marker／STOP／レビュー／契約／原稿を変更しない。元workspaceはread-only snapshotで観測し、他の変更をreset／cleanしない。

公開後、remote HEAD、変更file集合、UTF-8 bytes／SHA256／Git blob、文書リンク、source dependency mapを再照合する。[PUBLICATION.json](PUBLICATION.json)は検証済みpayload commitを記録する。receipt自身の最終commit/hashを同じreceiptへ埋める自己参照はしない。最終HEADは別途API／Git fetchで確認する。

新科学input、lambda/B/L、event／trajectory、matrix／state、quantum circuit、numerical result、科学grant／ledger／markerは0件。公開・照合後mandatory STOP。

## 公開照合実績

payload commit `ae72c74f8ae15360b51865a749879c6b87feba75` の26fileをGitHubから全件再取得し、内容・Git blobとローカルbytes／SHA256を照合した。独立Git fetchでも同じtreeを確認した。baseからは26件のA追加だけで、既存1,032fileは非変更。内訳はPython14、技術記録12。JSON3、固定資料33record、相対link26件、SOURCE signature／依存graphの管理検査はPASS。

元workspaceのstatusとworktree registry、local branch headsはsnapshotと同一。remote-tracking refsとframework capture refsには作業中の変動を観測した。本作業は元workspaceでfetch／ref更新／reset／cleanを実行しておらず、観測した変動を巻き戻さない。private snapshot／ref名は公開しない。公開receipt更新後の最終HEADと26fileは別途再照合する。
