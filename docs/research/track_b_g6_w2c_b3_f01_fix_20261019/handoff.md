# 技術ハンドオフ：F01修正完了・未統合

[入口](README.md) / [保存契約](failure_provenance_contract.md) / [人工検証](artificial_regression_report.md) / [将来統合](source_diff_and_integration_handoff.md)。状態：**`F01_FIXED_ARTIFICIAL_REGRESSION_PASS_INDEPENDENT_SOURCE_NOT_INTEGRATED`**。

## Q1. 原因確認と限定修正

旧run_pairのstale resultと、emit前digest/後counterのscope不一致を確認して修正した。対象はrun_pair、Streamの関連失敗情報、executeのfailure転送/cleanupだけ。旧科学SOURCE・law・N/予算・条件を変更していない。

## Q2. 前eventの誤帰属は解消したか

各trial/axisでcurrent stateを初期化し、sample return前ならresult未取得・inflight_event=nullとする。X→Y、bits factory、state guard、連続trialでも前eventを混入しないことを人工回帰で確認した。現在eventとlast_completed_trialは別fieldに保持する。

## Q3. Digestと保存境界

Candidate copy上でdigest/counterを準備し、emit正常return後に完了prefix参照を一度切り替える。部分write・file hash失敗・flush失敗はstream snapshotとtrial完了を区別する。正常時のtrace/chain/counterの意味とbytesは人工referenceで維持した。

write返却・emit正常returnは、disk再読やfsync/電源断耐性の認定ではない。failure record自体が書けない場合は原例外属性や書けたprefixまで。OS kill/OOM/保存先全障害の完全回復を保証せず、resumeも許可しない。

## Q4. テストと実G7科学実行

**新26 PASS、既存driver36 PASS、補助48 PASS、計110 PASS。FAIL/ERROR/SKIP各0。** 正常8ケースは旧SOURCEで先に固定した人工referenceとの比較。各suiteで禁止された実G7/F3/G9科学moduleのimportも0だった。実G7 correctness/sampling/profiling、native compile/synthesis、新科学計算は0。

## Q5. 将来統合する差分

Codex Aの公開成果と、driverの上記3宣言・failure metadata/decoder・新driver SHAを照合する。最終SOURCE release/pins、未採択binding、正式契約digest/外部admissionの参照は統合後の新versionへ固定する。旧candidate/receiptを上書きせず、今回の修正版がAで採択済みとも扱わない。

## Q6. その他の実行前技術課題

今回の限定scopeではF01以外の新しい必須blockerは確認していない。既知の外部atomic admission/claim/source-release binding、出力作成前の失敗保全、τ/Kφ/J等の正式採択、別途科学実行承認は残る。実G7 integration correctnessの成立を人工PASSで代替しない。

## 次の担当・STOP

**ユーザー**が、Codex A完了後の公開SOURCE統合・最終binding確認を依頼するか判断する。別途指示された**Codex**だけがその差分を確認・固定する。今回は統合・科学条件採択・外部admission有効化・31,710,848 calls・新レビューへ進まない。

独立branchとSOURCE commitを公開し、旧成果・契約・consumed ledger/claim・STOP、親worktreeと前の自分の監査checkoutを保全した。Codex Aのprivate worktree/index/lock/processは参照も変更もしていない。新規実authorization/grant/marker/claim発行・消費0件。[管理検査](verification.json)と[公開照合](publication_receipt.json)後、**mandatory STOP**。
