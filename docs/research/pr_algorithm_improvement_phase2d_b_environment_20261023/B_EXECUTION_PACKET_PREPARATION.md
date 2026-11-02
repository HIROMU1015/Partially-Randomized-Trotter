# B実行前packet：承認を含まない案

**EXECUTION_PACKET_DRAFT_NOT_AUTHORIZATION**。[機械可読案](B_EXECUTION_PACKET_DRAFT.json)はSOURCE、環境、仕様・台帳・schema、11case/DAGを参照する。実authorization／grant／ledger／consumed marker／実run ID／科学input hashは作成していない。

| 固定済み項目 | identity／状態 |
|---|---|
| SOURCE commit | `3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c` |
| SOURCE closure | canonical 10file map `4b4fbde323bce42a87e44b2aad5ac0504efbcfabac21ffb536da489466092e12` |
| public environment lock | `32e8201e50ca37b99e58a516ef8b0e85317dd03527a1fd7f3b34dab8cac27c00` |
| private B environment.json | `b69f325f6df3a9c9af186e031919d39577608f5e3a12a215c25b8869388b0e3e`（固定gateのenvironment_sha256はこれ） |
| full private receipt | `5645df7bcfbb624573024fb130973d67f8bbed47845553a1285aee1217c921d8` |
| input specification | B-PRE B_INPUT_SPECIFICATION.mdの既存bytes SHA。科学inputのSHAではない |
| inventory/schema | B_CALL_INVENTORY.json／B_SCHEMAS.jsonの既存bytes SHA |
| call plan | B.INPUT2、B.ARMS4、B.CONTEXTS5、計11。順序・依存DAGそのまま |
| scope | NO_BUILD_NO_EVENTS_F1_LAWS_ONLY_PRICE_UNBOUND |
| permission gate | 固定contract.py。将来USER_EXPLICIT_B／APPROVED_BOUND_ONE_SHOTのみを受付 |
| fresh output | templateだけ。approved_run_idと実namespaceはnull |
| one-shot／STOP | retry=false、mandatory_stop=true。失敗・成功いずれでもSTOP |

台帳の関数内部件数とoutput39JSON／run.log1本の契約は変更しない。価格PRICE_PROVIDER_UNBOUND、interval UNPRICED_FALLBACK、shots SHOT_COUNT_UNRESOLVEDを保持する。今回の環境解決で価格やinterval資格を追加承認したことにはならない。

ユーザーがBの11callを別途明示承認する場合、SOURCE・環境・仕様・call・STOPを確認し、新しいrun IDとfresh outputを指定する。承認後だけ、Codexが実承認文のdigest、USER_APPROVED_B_FINAL契約、B専用authorizationを作り、未消費を照合する。draftのstatusを書き換えるだけで許可を捏造しない。環境・SOURCE・資源非干渉を再照合してからgateへ進む。

現時点で実行可能packetは発行していない。actual full permission gateの実行も未実施。科学runtime APIの初回使用は承認されたB.CONTEXTS内だけであり、事前に独立probeを追加しない。B後に必ずSTOPし、F1、F2はそれぞれ別承認。
