# H4 future D/P integration contract

現在の固定版は**H4_PRE_D_P_VALIDATION**。legacy DF12、PF測定subset16、新A探索n8、旧n32 fixed anchor。現在の32比較に将来のD/P候補・費用・独立標本を推定／補完しない。

将来の入力schemaにはdataset_version、stage、result_commit、science_SOURCE、Hamiltonian/state/DF順序/compiler/profile、candidate/axis、比較集合ID・登録範囲、PR cohort・n・replica/pairing、固定shot会計・bias/factor/α、exact Gと保存SE、直接/model、原receipt/SHA、旧STOP来歴を持たせる。

D完了でPF比較集合が変わっても、現在のPF候補を上書きせずH4_POST_D_VERSIONを新しいimmutable datasetとして登録する。P完了で独立PR確認が得られても、旧fixed anchorを独立確認へ再ラベルせず、新cohortと固定対照を別rowにする。候補IDが同じでもcohort/result commitをキーから外さない。

接続手順は、ユーザーが採用するD/P完了commitと比較条件を指定→Codexが新しい入力・対象・解析SOURCE・hash・受入条件を固定→ユーザーが別途数値実行を承認、の順。現runnerはH4_stageがH4_PRE_D_P_VALIDATION以外なら拒否するため、将来接続には新SOURCEが必要。

D/Pの未完了を理由にH6/H8・H4暫定版の準備を待たない。別Codexのworktree、SOURCE、入力、grant、ledger、marker、STOP、予定jobは観測凍結・変更・操作の対象にしない。本契約は接続設計だけで、D/P実行・再ランキング・数値解析は未実施。
