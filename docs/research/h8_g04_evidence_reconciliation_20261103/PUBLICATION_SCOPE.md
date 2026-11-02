# 公開範囲と非公開保全

2026-11-03 JST。本依頼は既存の公開拒否・機密性・原資料の共有制約を解除しない。

## 今回公開する最小限のmetadata

- REPORT.md、H8_CONFIRMATION_EVIDENCE.csv、PUBLICATION_SCOPE.md、VALIDATION.jsonの4ファイル。
- 既存公開資料の固定commit/path/selector、科学SOURCEとdataset SOURCEの役割、候補/PF/ε、n、保存N/C/G/SE token、主表/Mechanism/Figure 2のcohort alias。
- 公開済みdesignにある科学的Hamiltonian/state/snapshot identity。これらは再現性の入力識別であり、承認証明やprivate入力hash一覧の公開ではない。
- ユーザー指定候補OID 659e555714c8ccf3a6ec7fd1cb0dbf05cfcc342d / 96aa3878a344f479ee064118cb90002a770db627の所在確認というmetadata、公開列との版関係。候補commitの内容全体はuploadしない。
- 既存公開終了監査との差、既存の公開拒否、未確認・未保証とSTOP。今回の公開文面にprivate監査の詳細結果・hash一覧を含めない。

## 非公開で保持するもの

private raw/input、native/terminal/worker証跡、実seed一覧・除外元・内部対応表、内部絶対path/host/IP、運用grant/ledger/marker、runtime/provider/cache、秘密情報、原本回収dossierとlocal-only監査資料は公開しない。原本を移動・上書きせず、今回の限定原本照合の結果は独立出力cloneのGit管理metadata内だけに保存した。Git管理metadataは今回のcommit/treeへ含めない。

公開結果e6eea5387a6e84d1e39a9adb3b61975acc74e200は、自動承認審査による詳細監査・hash一覧・運用来歴の公開拒否を記録する。公開回収索引9f33ad0b11b4febe48b802000c38c5a0e3fd267cも、private由来の監査をpublic packetへ含めていない。匿名化だけで公開許可が成立するとは扱わず、今回もその境界を保持する。既存の公開拒否に反するuploadを試みたり、別branchや転送経路で回避したりしない。

## 残る判断

G04の公開不足を解消するには、ユーザーが必要な監査field、共有相手（LOCAL担当等）、経路、既存公開拒否の対象との関係を具体的に判断する必要がある。今回、その判断を推測で代行せず、詳細補足を自動送付しない。承認証明書・署名・台帳・認証基盤を新設する必要はない。

この判断待ちは、許可された4公開成果物の作成・公開を妨げない。科学計算の実行許可や希少RTE研究再開とは別であり、公開後はMANDATORY STOP。
