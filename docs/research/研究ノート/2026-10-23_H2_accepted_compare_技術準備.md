# 2026-10-23：受入済みH2診断を再利用する比較分岐

元DIAGNOSTICSは4候補の診断を正常完了しているため、旧COMPAREを使うと重複する8 path/4 reference再結合を省く専用分岐を実装した。元SOURCE/結果を変更せず、元契約・completion・prep/operation・元grant/ledger/markerのmetadataとhashを新契約へ結合する。科学vectorのloadは将来の明示承認後の1単位に分離した。

公開storage manifestを追加回収し、402 archive metadataと134 compiled raw SHAの保存単価対応を確認した。原configのcanonical root/full event/旧stream/実runtimeはなお不足し、PR単価再利用をBLOCKEDに保つ。一致hashやpolicy名から完全identityを推定せず、runtimeの宣言差と観測欠測も分離した。

新SOURCE46a3b33e…、人工138件PASS、実metadata preflightはbinding検査後BLOCKED。新しい科学call・u/N/G・順位評価・レビュー・live grant発行は0。元診断とgrantは保全する。[実装・正確な予算・必要資料](../track_a_h2_accepted_compare_20261023.md)。local commit後mandatory STOP。
