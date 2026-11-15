# C2 → D1：Full-H κ2 D7独立holdout規則案

**[規則案](PROPOSAL.md)公開・ユーザー採択待ち。取得・開封・比較計算は未実行・未許可。**
第一案は4件の記述的比較であり、Full-Hの数値許容差やモデルPASS/FAILを採択していない。

## 次回D1指示への転記案（採択・別途実行許可後に限る）

1. ユーザーが採択した比較規則の固定commit/pathを参照する。採択だけを科学実行許可と扱わない。
2. holdout観測前にCが公開した予測正本を下欄へ結合する。m5-X/Y・m6-X/Yの4保存予測と
   同じround/軸/physical bindingを確認する。予測・locatorが不足する間は取得・開封へ進まない。
3. 固定D7のm5は同軸fresh 2 path平均、m6は同軸fresh 1 path値。後続6 compileのみ。
   q/R/T・cohort・seedは規則案§2の固定metadataに一致させ、X/Yの標本数を合算しない。
4. 既存の必要gate・SOURCE・receipt対応確認後、4行でP、O、P−O、(P−O)/Oをexactに保存。
   %は表示のみ丸める。O=0の相対誤差は未定義。個別6 receiptを保全する。
5. m5の片方欠測なら比較不能。軸/SOURCE不一致・gate不成立をモデル精度FAILと混同しない。
   全4有効なら記述完了、不足があれば未完了／無効。モデルPASS/FAILは付けない。
6. 部分証拠と理由を保全し、失敗時・報告後にSTOP。追加sampling/compile・m6 n2化・再fit・
   閾値の事後採択による独立検証の主張、E[C]/CI/N/G・長round資格への昇格は行わない。

## Cの公開後に埋める参照欄

| 必要項目 | この引渡し時点 |
|---|---|
| Cの凍結予測commit（完全SHA）とrepository-relative path | 未結合。Cの後続公開を参照する |
| m5-X / m5-Y / m6-X / m6-Yのselectorまたは一意のkey | 未結合。各行に対応付ける |
| 各予測のphysical binding・特徴量順・fit SOURCE・較正入力参照 | C正本から継承し、D7 metadataと照合する |

この欄は後続作業の依存条件であり、C2がCの完了を待つ条件ではない。
架空SHA・予測値は記入せず、Cの凍結原本を書き換えない。

## SOURCE・作業先・ロードマップへの影響

C2専用branch `codex/full-h-d7-holdout-rule-proposal-20261115` はD1公開
`e628c0d69c7da559369d99f323e3656a96c0728a`を基点に作成した。
既存Full-H設計branchはA/D1が使用中のため再利用せず、Cのfit・D1の作業を変更しない。
固定A/D7/D1資料とPauli先例のSHA/pathは規則案§1に記載した。
Pauli LD2の凍結モデル・4予測・規則案・STOPを別branchの固定成果として保全し、今回編集・採択しない。

ロードマップへの追加は「Full-H D7の観測前記述的比較規則案が公開され、ユーザー採択待ち」まで。
科学的証拠・モデル資格・独立holdoutの未評価状態は変わらない。依存は、規則採択とC凍結予測の
公開・結合、そして別途のD1実行許可。Cの予測と規則は観測前に固定する。
ロードマップ本体は編集しない。

確認は固定文書・metadata・関連規則履歴、文書リンクと差分範囲の静的確認のみ。
性能対策・科学SOURCE変更・実行testは不要な文書作業で、今回の数値計算・観測価格開封は0。
規則案のcommit/pushとremote一致確認後、**規則案公開・ユーザー採択待ちとしてmandatory STOP**。
