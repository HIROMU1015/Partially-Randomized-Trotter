# H4保存評価結果の自動比較

固定10入力から34点（適合27・モデル不適合5・価格待ち2）を比較するconsumerを実装した。重複4件を統合、LD3の旧信号のみ状態1件をhistoryへ保存。Full PF S2は559457386 RZ、p7 S2は395962678 RZが指定集合内の最安適合点。LD3 VP/LD7 V0はQ3適合・費用null、旧PR予算と基準を保全する。

[再実行コマンド・接続条件](../h4_saved_rpe_cost_catalog_20261116/HANDOFF.md)。人工と保存値の検収5件PASS。同じcommandで入力固定commitの追加だけによる更新を確認した。新科学call0。原データ・SOURCE・判定・STOPは不変、ロードマップ本体は統合担当へ。

## 粗幅Y4復旧結果の追加

p7標準Yoshida4・粗幅32h/5 B_PRICEの復旧Q3（8a2f5c46d）をモデル不適合・費用nullとして追加。旧UNKNOWNは34783cd09の実行履歴へ接続。固定11入力35点（適合27/モデル不適合6/価格待ち2）、従来34点と全方式最安値は不変、p7最安395962678 RZ。新科学call0、技術検収5 PASS、公開後STOP。

## Pauli LD3 VP価格の接続

PR-Pauli LD3 VPにA f25a05f8aの26軸モデル参照価格を接続。configuration/total view/binding/配分/m/axis/Nを26/26照合、exact総額173557305044/165 RZ（約1,051,862,454.81）がPR-Pauliの保存集合内最安へ更新。LD2 OWN-Bは適合点として保全。直接は別較正cellの2pathのみ、全26価格MODEL・qualified_for_candidate=false。固定12入力35点（適合28/モデル不適合6/価格待ち1）、他34点・他方式最安は不変。技術検収6 PASS、新科学call0、公開後STOP。

SOURCE・26軸の照合・旧価格待ち履歴は[現行HANDOFF](../h4_saved_rpe_cost_catalog_20261116/HANDOFF.md)とCATALOGのprice_evidenceへ保持。ロードマップ本体は統合担当へ。

## PR-DF LD7価格・Q3の接続

PR-DF LD7 V0に価格を接続し、A_SHAPE/B_PRICEを追加。LD7の3配分は保存Q3適合。保存候補集合内のPR-DF最安はLD7 A_SHAPE、155652932221/126 RZ（約1,235,340,731.912698）、X/Y各[48,45,42,39,35,32,29,25,22,19,16,12,9]。費用の95.945359%はq>64のm7〜m12へのモデル外挿。価格は未資格モデル参照、E[C]/CI/正式Gは未認定。LD1/LD2の再掲6点は既存結果を再利用。固定13入力37点（適合31/モデル不適合6/価格待ち0）、他34点・他方式最安は不変。テスト7 PASS、新科学call0、公開後STOP。

SOURCE・26軸の照合・既存6点の再利用は[現行HANDOFF](../h4_saved_rpe_cost_catalog_20261116/HANDOFF.md)とCATALOGのprice_evidence/saved_cost_breakdownへ保持。

## 粗幅Y4 A_SHAPEの追加と最終入力集合

LD7 V0へ価格接続、A_SHAPE/B_PRICEを追加し、PR-DFの保存集合内最安をLD7 A_SHAPE・155652932221/126 RZ（約1,235,340,731.912698）へ更新。後半m7〜m12へのモデル外挿が費用の95.945359%。再掲LD1/LD2の6点は再利用。粗幅p7 Yoshida4・32h/5のA_SHAPEはQ3適合・価格待ち/null、同configurationのB_PRICEはモデル不適合/nullを維持。現幅価格の転記なし。固定14入力38点（適合31/不適合6/価格待ち1）、回帰7 PASS、新科学call0、公開後STOP。

既存readerの無変更再利用、配分別の判定とSOURCEは[現行HANDOFF](../h4_saved_rpe_cost_catalog_20261116/HANDOFF.md)に記載。
