# B-PRE静的・管理検査

実施するのはSOURCEテキスト／AST、JSON構造、固定Git bytes/hash/blob、link、変更範囲の管理検査だけ。binding package／I prototype／NumPy／Qiskit／mpmath等をimportせず、runnerを起動せず、pytest／科学preflight／数値mockを実施しない。

新Python4fileの構文とmodule-level workの不在、stdlib permission gateより後のengine import、固定rte.pyだけをloadする箇所、Qiskit/mean/event/sampling等へのdirect call不在を静的に確認する。既存10file active closureと、未使用I関数が持つdormant importsを区別する。SOURCEを実行したcall graph確認ではない。

B_CALL_INVENTORYは11case、2/4/5 family、exact DAG、subcall件数、hash/output/失敗/重複防止fieldを照合する。SRCのhelper countはRTEConfig／RTEFiniteDistributionの内部再検査も含める。L依存件数はnull／式のまま保ち、実値を取得しない。

schemaはobject/type/required/const/enum等の管理構造とSOURCE上のvalidator対応を確認し、科学input instanceは作らない。canonical fractionとbinary64 hexの仕様、F1 handoff root fieldsとI readerの対応、environment未解決、price/shot未資格、B実行未承認のflagを照合する。

固定資料36recordのbytes／SHA256／Git blobを確認する。うちI/P0必須19資料は固定GitHubから取得したblob identityと照合する。既存I sourceの14fileと元設計／R1／RTE sourceも固定Git objectで確認する。管理environment調査は候補metadataのread-only確認だけで、専用環境の実在・正しさを認証していない。

起点bfaab383から新B package／runner／技術記録のA追加だけ。旧科学SOURCE、P0/I、input/raw、旧レビュー、契約、grant／ledger／marker／STOP、原稿を変更しない。元workspaceはsnapshot観測だけで、parallel変動をreset/cleanしない。

公開後、remote HEAD、file集合、exact UTF-8 bytes／SHA256／Git blob、active SOURCE closure、相対linkを再照合し、可能な範囲で独立Git fetchする。[PUBLICATION.json](PUBLICATION.json)は検証済みpayload commitを記録し、receipt自身を含む最終commitと区別する。自己参照hashを捏造しない。

静的PASSはscientific API・algebra・law・operator・confidence・performanceのPASSではない。今回の科学call、fixture／lambda/B/L／law／input hash生成、event/trajectory、build/compile、grant発行は0件。公開照合後mandatory STOP。

## 公開照合実績

payload commit `a686b0f6037245fe910b5fcef8e65b0bf8f4a801` の21file全件をGitHubから再取得し、内容とGit blobをローカルbytes／SHA256へ照合した。独立Git fetchでも同じtreeを確認。追加はPython4、技術記録17で、既存1,058fileは同一bytes。JSON6、固定資料36record、相対link25件、active SOURCE10file、11case/DAGの管理検査はPASS。

元workspaceはread-only snapshotで観測し、本作業では書換え・fetch・ref更新・reset/cleanを行わない。並行変動を観測した場合もそのまま保全し、非公開snapshot／内部pathを公開しない。receipt更新後の最終HEADと全21fileは別途再照合する。
