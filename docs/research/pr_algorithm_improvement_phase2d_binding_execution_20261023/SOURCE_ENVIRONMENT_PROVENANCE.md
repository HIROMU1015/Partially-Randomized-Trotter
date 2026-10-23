# SOURCE・環境・承認のprovenance

| 層 | 固定identity |
|---|---|
| Science SOURCE commit | `3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c` |
| Active SOURCE | 10file、B4＋I5＋standalone rte.py1 |
| Canonical SOURCE closure map SHA | `4b4fbde323bce42a87e44b2aad5ac0504efbcfabac21ffb536da489466092e12` |
| B-ENV公開commit | `db432dcc998a8b6dabf387ee84b251518c334a52` |
| Public environment lock SHA | `32e8201e50ca37b99e58a516ef8b0e85317dd03527a1fd7f3b34dab8cac27c00` |
| Private B environment.json SHA | `b69f325f6df3a9c9af186e031919d39577608f5e3a12a215c25b8869388b0e3e` |
| Full private receipt SHA | `5645df7bcfbb624573024fb130973d67f8bbed47845553a1285aee1217c921d8` |

実環境はCPython3.11.9／NumPy1.26.4、system-site-packages=false、-I/-B。旧private receiptと実binary／wheel／RECORD／native／prefixが再照合で一致し、再構築・package導入・別identityへの切替えはしていない。B scalar metadata pathは実際に使われたが、F1用広域environmentの資格ではない。

SOURCE/spec/inventory/schema/recipe policyのfile別hashは[PROVENANCE.json](PROVENANCE.json)。固定GitHub17資料を新規取得してlocal Git objectとbytes/blobを照合した。B-PRE/I/P0の旧契約を変更せず、最新のユーザー実行指示により今回の限定B権限だけを発行した。正式共通指示2026-10-23原本は取得確認できず、読了とせず今回の明示条件を適用した。

承認原文・final contract・authorization・marker・private環境・logsは非公開で保全した。承認は今回のuser添付指示に直接基づく。直前assistant提案をuser承認へ読み替えず、取得できない外部同意原文を補っていない。grantは1件、固定gate照合後に1回だけconsumeした。

## 既存出力field名の注意

固定engineが出力した`manifests/environment_public.json`の`private_environment_receipt_sha256`というfieldの値は **B用7field environment.jsonのSHA（`b69f325f6df3a9c9af186e031919d39577608f5e3a12a215c25b8869388b0e3e`）**である。full private receiptのSHAは **`5645df7bcfbb624573024fb130973d67f8bbed47845553a1285aee1217c921d8`**で別物。原出力のfield名やbytesは変更せず、この対応を明記する。科学input SHAの代用にも使わない。

科学SOURCEの実行HEADをB-ENVや結果branchのHEADへ置換していない。旧SOURCE、既存input/result/grant/STOP、原workspaceと他worktree/jobは変更せず、新規output・新packet・別結果namespaceだけを使用した。
