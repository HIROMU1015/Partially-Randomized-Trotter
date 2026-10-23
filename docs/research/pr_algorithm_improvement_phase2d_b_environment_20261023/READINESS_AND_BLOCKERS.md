# 状態と残条件

主状態は **READY_FOR_B_EXECUTION_APPROVAL**。指定Python／NumPy、wheel・installed package・実native identity、分離、SOURCE10file対応の技術的証拠が揃ったため、ユーザーが限定B科学実行を承認するか判断できる。

| 状態 | 判定 |
|---|---|
| READY_FOR_B_EXECUTION_APPROVAL | 該当。環境技術準備と実行前packet案が準備済み |
| B_ENVIRONMENT_UNRESOLVED | 解消。実binary／専用venv／package/nativeを固定 |
| ENVIRONMENT_VERSION_UNAVAILABLE | 非該当。3.11.9／1.26.4を実際に用意 |
| ENVIRONMENT_LOCK_INCOMPLETE | 非該当。full private receiptと公開summary・digestを分離して保全 |
| ENVIRONMENT_VALIDATION_FAILED | 現最終結果では非該当。初回のdetached ldd判定問題を管理helper内で解決し、実mapped依存と再照合を確認 |
| B_SOURCE_ENVIRONMENT_INCOMPATIBLE | 静的要求・ABI smokeで不適合を検出せず。科学SOURCEは未importで、科学API runtime正しさは未検証 |

実行前に残るのは、ユーザーによるB11callの**別途明示承認**、fresh run ID/outputの指定、実承認文・final contract・新authorizationのbinding、直前のSOURCE/envと資源・非干渉の再確認。これらはREADYを実行済みへ変える条件であり、今回勝手に補完していない。

RTE／mixed prototypeの科学的runtime正しさ、科学入力とそのSHA、各L/lambda/B、F1/F2環境・価格・interval資格は未実施・未確定。basic Bの環境blockerとは区別し、Bまたは後続の承認済みscopeで扱う。F1/F2への自動進行はしない。

専用実体は非公開領域で保全する。削除・移動・package/native更新が起きた場合、同じenvironment identityとは扱わず再固定する。第二環境での独立再構築を検証済みとは主張しない。

次の担当はユーザー。公開資料とprivate receiptを確認してBを承認するか判断する。Codexは今回の公開照合後mandatory STOPし、承認前にB runnerを起動しない。
