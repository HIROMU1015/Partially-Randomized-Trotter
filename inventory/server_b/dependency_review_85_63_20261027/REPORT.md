# SERVER_B 残存85＋63件の依存解消調査

2026-10-27 JST。対象は旧[DEPENDENCY_HOLDS.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50612696d8ef7db322729fd3946b2033d3d39e01/inventory/server_b/bulk83_preservation171_20261027/DEPENDENCY_HOLDS.csv)の85 IDと[OTHER_ENVIRONMENT_63.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50612696d8ef7db322729fd3946b2033d3d39e01/inventory/server_b/bulk83_preservation171_20261027/OTHER_ENVIRONMENT_63.csv)の63 ID。集合と元分類を保持し、新判断を別fieldに記録した。

**148件中、新しい技術候補146件（元85件から84、元63件から62）、KEEP_REQUIRED2件、TARGETED_CONFIRMATION_REQUIRED0件、PRESERVATION_ACTION_REQUIRED0件。実削除・SOURCE/PLAN/科学契約変更・科学計算は0。登録は210件のまま。**

| 元集合 | 新しい技術候補 | 保持 | 限定確認待ち | 追加保全待ち |
| --- | ---: | ---: | ---: | ---: |
| 依存保留85 | 84 | 1 | 0 | 0 |
| 他環境待ち63 | 62 | 1 | 0 | 0 |
| 合計148 | 146 | 2 | 0 | 0 |

全行は[DEPENDENCY_REVIEW_148.csv](DEPENDENCY_REVIEW_148.csv)、候補は[NEW_CLEANUP_CANDIDATES.csv](NEW_CLEANUP_CANDIDATES.csv)、保持理由は[KEEP_AND_DEPENDENCY_REASONS.csv](KEEP_AND_DEPENDENCY_REASONS.csv)。146件の利用終了は未判断であり、今回の分類を実削除許可として扱わない。

## 参照の役割と現行性

以前の一律保留をそのまま継続せず、既存4077件の参照索引、B側4474件のSOURCE/path索引、前回のSOURCE/control差分、保全manifest・非公開対応表を再利用した。対象に結び付く1,844参照元を読取確認し、1,746件は保存file、98件は物理referrer削除後も保持Git blobから取得した。参照先、JSON field/line、source hash、役割、terminal/STOPとの関係、今後の復旧条件を本人専用対応表へ保持した。

旧63件の多くは具体的な他環境依存が判明していたのではなく、新しい/変更されたscopeの未確認による保留だった。この欠けた条件だけを追加し、63 HEADのSOURCE/controlをunique Git blob単位で静的照合した。3,304 blob・146,338,329 bytesを読み、数値input/rawの計算・再解析やSOURCE実行は0。単なる共通prefixの隣接runtime directoryをcheckout内部と誤認しないようpath境界も区別した。

保存されたRUNNINGやactive_contractという名称だけで現行jobを認定していない。例えば旧H6 started観測にはH6_PARALLEL_COMPLETE_MANDATORY_STOP/H6_PARALLEL_STOP等のterminalを照合し、PFのRUNNING_HANDOFFにはCOMPLETED_PARTIAL_MANDATORY_STOP等の後続記録を区別した。H4のsealed D/P specにはCOMPLETE_MANDATORY_STOPを、Non-DFのactive契約には宣言outputのSTOP/ENGINE_STOPを照合した。これらは契約の時間的役割の確認であり、科学的結果の正しさを新規承認したという意味ではない。

seed-origin・preservation/source inventory・失敗tracebackは履歴provenance、終了/未実行の契約・PLANは将来の復旧時参照として保持する。保存SOURCE/input/precision・比較条件・one-shot/retry/STOPは不変。旧pathを将来使うにはexact workspaceの復元や別許可の設定変更が必要となる場合があるため、保存PLANを削除後も即時実行可能とは認定しない。

## 現在保持する2件

- **SERVER_B-WT-0128**：指定されたB3保護対象であり、保持B3設定のsealed SOURCE catalogueにもbindingがあるためKEEP_REQUIREDを維持。
- **SERVER_B-WT-0130**：元63件から新たに役割を特定。保持B3が参照する設定内のsealed SOURCE catalogueに73 path（既存索引では重複を含め146行）が結び付くためKEEP_REQUIRED。workerがそのcheckoutの全fileを直接importしているとは主張せず、現行B3 SOURCE役割を今回の整理判断で解除しない。

2件の保持は単に古いpath文字列があるためではなく、保持B3設定に結び付く現在のSOURCE役割と今回の保護境界による。旧直接保護0003/0123/0124と0128を保持し、0130も削除候補から除外した。B3の現在稼働・停止原因・科学完了、service/controller/worker/schedulerの操作や判定は0。

既存controlが宣言する専用Python2 pathのpointer file4件をtextとして確認し、保持KEEP59側のtracked Git symlinkも限定確認した。今回の146候補を指すeditable/shared pointerは検出せず、実行中processの可視本人UIDによる対象path利用・lockも未検出。未知namespace/全環境の不存在証明ではなく、削除直前のfresh確認を置き換えるものではない。

## 保全と他環境

元85件のうち候補84件は既存byte custodyを再利用して保存leaf・member SHA・stage/working復元の差分条件を確認し、元63件の候補62件は現在のclean/未追跡0/ignored0とGit保持を個別確認した。追加コピーが必要な変更0、既存94 archive・3 bundle・復旧manifestを保持。全146候補のcommit/blobは保持object storeから取得可能だった。同じfilesystem上の保全であり独立バックアップではない。

SERVER_Aの20＋H46 4件、LOCAL131件の公開削除/保全報告12資料と既存LOCAL139/B側対応を照合した。Aの保持PLANがPHASE_A_FIXED_AWAITING_USER_EXECUTION_PERMISSIONで、旧6 locatorを残したまま復旧条件を明示して旧4 checkoutを削除した方法を参考にした。PLAN保存と旧path永久保持を分離する考え方を使い、A/LOCALの実削除結果をSERVER_Bの物理依存不存在や利用終了承認には転用しない。

今回148件と他環境削除結果のexact branch/HEAD一致は0であり、これを安全認定の根拠にはしていない。旧63件の条件はB側SOURCE/control/runtime/保全の個別scopeで閉じた。[CROSS_ENVIRONMENT_CONFIRMATIONS.csv](CROSS_ENVIRONMENT_CONFIRMATIONS.csv)に確認範囲と承認非転用を記録し、現時点で別環境へ追加確認が不可欠な具体的bindingは0。SSH・他環境の変更は0。

## 次期一括計画

候補146 IDと復旧先・保持ref・判断根拠・直前preflight・個別保留/共通STOPは[NEXT_BULK_CLEANUP_PLAN.md](NEXT_BULK_CLEANUP_PLAN.md)。現在93件はCLEAN、53件はLOCAL_ASSETS_REMAIN。保全完了と通常remove可否を分離し、53件の削除方法は後続の具体的許可が必要。旧86件だけの単一force許可は転用しない。

| 研究単位 | 新候補 |
| --- | ---: |
| BASELINE_OR_OTHER_UNRESOLVED | 3 |
| DF_H2 | 16 |
| DF_H246_HIGHER_PF | 1 |
| DF_H4 | 70 |
| DF_H46_PF_PR | 3 |
| DF_H6 | 14 |
| DF_H8 | 4 |
| DF_NATIVE_TRACK_A | 5 |
| PAPER_EVIDENCE_PRESENTATION | 3 |
| PAULI_AND_REPRESENTATION | 7 |
| TRACK_B_G6_HISTORY | 10 |
| TRACK_B_OTHER_HISTORY | 10 |

次の担当は**ユーザー/研究担当**。上表の研究単位についてSERVER_Bでの利用・再開予定の終了を判断し、その後に正確なID集合と削除方法へ明示指示する。保持2件のSOURCE役割やB3の科学条件を無断解除しない。

最終確認で210登録・元ref・対象index・保全file・B3保護資源は不変。書込みは新しい本人専用調査記録と独立公開cloneの新directoryだけ。実削除0、branch操作/merge0、新科学計算/詳細科学レビュー0。匿名化6資料を通常fast-forwardで公開し、GitHub読戻し・local/remote SHA照合後mandatory STOPする。
