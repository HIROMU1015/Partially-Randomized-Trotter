# 固定SOURCEとの統合差分

直接基点はAdmission最終 `1c15bacab349b8686c09d2981e1c0bc70e9c3ec9`。F01 source commit `a777a2ab54d85a95dce91a06e65d4ec0511403ae`からdriver・26件tests・正常fixtureだけを固定blobで取得した。
F01最終commitともbytes同一。branch全体のmerge/rebaseや旧資料の上書きはしていない。

| 対象 | 統合内容 |
|---|---|
| driver | F01修正版を完全bytes一致で採用。旧版との差はStream/run_pair/executeだけ |
| admission | host必須の外側schema v2、固定host role/API/完全SHA・import path照合を追加 |
| admission context終了 | secondary STOP失敗がprimary driver例外を置換しない。正常returnでSTOP保存失敗なら技術的失敗にする |
| host | 新execute_approvedがwith port内でpreflight→既存executeを一度呼び、外側STOPへ接続 |
| 既存69 tests | 人工fixtureへ現在driverのbytes hashを指定する1行の変更だけ。69の検査方法・件数は保持 |
| F01 tests/fixture | 固定公開版とbytes同一、正常8ケースのreferenceを再生成していない |
| 新統合tests | 66件。private人工authority、mock modules/endpoint、fake clock、owned childだけ |

ClaimLedger/PinnedApproval/TrustAnchorはAST同一。BEGIN IMMEDIATE/DELETE/EXTRA、一意制約、spent/resume拒否、承認内容hash照合を弱めていない。
旧driver拒否のOLD_DRIVER_SHAは旧d7947a...のまま。F01 pinは別定数/manifestで5e6904...として区別した。
統合driverはF01 bytesを再編集していないため、F01 SHAと統合技術driver SHAは同値。旧driver SHAとは異なる。
将来の最終science driver SHAはnullであり、技術driver pinをそのまま最終採択扱いにしていない。

Boundary、LegacyEndpoint、Meter、FunctionObserver、seed_for、candidate_errors、identityと定数・科学call順序は旧版からAST不変。
SOURCE/state guard、sampling/bit/angle/requery、R/C/A、phase/weight/word、N/confidence/error、setup/cache/保存と二passは維持。
新hostは外側coordinatorの接続であり、scientific schedulerやCPU費用の改善は導入していない。

差分patchは[integration_source.patch](integration_source.patch)。管理資料は新ディレクトリへ追加し、R1採択記録/旧contract/inventory/receiptは不変。
