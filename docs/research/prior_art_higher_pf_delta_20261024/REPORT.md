# 高次PF・PR資源効率：先行研究との差分と論文案A/B

2026-10-24。**研究判断用の差分資料。論文方針・新規性・中心主張は未採択。**
10月21日の[既存調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/501b4232b3e9ef9778e917f0a86f1d6c1d05f189/docs/research/prior_art_novelty_20261021/report.md)を再利用し、
[正式採用09e71910](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/09e71910ee49e0d9d3a2f2f410249f6623022c0d/artifacts/resource_applicability/track_a_h46_standard_y4_adoption/2026-10-24/REPORT.md)以後の証拠に対応させた。

## 判断に直結する点

1. **高次PF＋部分ランダム化、負時間stageの負担、精度による方式crossoverは既知。** GüntherらP01のApp.A3は一般次数のPR、App.D1/Fig.16はS2/S4比較を既に扱う。Hagan–Wiebe P03 §5も高次outer/inner compositeを扱う。[一次資料比較](papers_comparison.md)
2. **今回の追加は具体的な実装・固定構成の条件付き知見。** 異なるqのFull/PR方向差、同一qで逆転しなかった対照、標準Y4の独立費用標本、保存C/N/Gの対応を一緒に提示できる。「高次ほど有利」「次数単独で逆転」という一般則ではない。[Claim–Prior Art差分](claim_prior_art_matrix.md)
3. **独立確認の完了は証拠強度を補強したが、既知構成を新アルゴリズムへ変えない。** H4/H6・ε=.001、S2/q8対標準Y4/q4、独立S2 n25/23と新Y4 n45/32。保存G比はH4 `0.8649598948712324`、H6 `0.8627624094690314`。固定2比較のBonferroni正規近似を保持し、母平均・希少tailを保証しない。[固定証拠と条件](evidence_map.md)
4. **案Aは既存の適用条件研究への増補、案Bは構成依存の費用比較を中心に据える。** 案Bも限定事例研究なら既存結果を使える。一方、一般的な選択則・因果説明・独立した方法論を主張するには不足がある。両案を正式採択せず、どの強さの主張を目指すかを判断材料にする。[構成・比較表](paper_options.md)

## 読む順序

- [既知／実証／未確立の区別とMatrix](claim_prior_art_matrix.md)：旧C12/C15/C16の差分を明示。
- [直接関連10文献の比較](papers_comparison.md)、[12項目のCSV](papers_comparison.csv)、[書誌・確認箇所](references.md)、[再利用BibTeX](references.bib)。
- [論文構成案A/B・図表候補・重複境界](paper_options.md)。図表は提案のみで新規生成0。
- [GPT・ユーザー向け判断資料](decision_packet.md)：採用できる文言候補、過大主張の境界、判断後の担当。
- [旧調査からの変更点](changes_from_20261021.md)、[検索範囲と版差](literature_update_log.md)。
- [技術検査](audit/validation_summary.json)、[固定資料索引](audit/source_catalog.json)、[運用版差](audit/policy_version.json)、[STOP](audit/STOP.json)。

## 残る不足と今回の境界

残る中心課題は、移送可能な説明・公平な候補域の意味・論文間の重複整理である。
同一条件で先行法を再実行した比較はなく、世界初・普遍的優位・全域最適性を認定できない。
P32は次数/step数を含む精度別選択と有限samplingを既に扱うため、それらの一般原理を新規性に置かない。
Yoshida/Suzukiの元出版社本文は未取得。GPT側正式運用ファイルの提示SHA原本も未取得で、今回明示された§10.3基本方針はユーザー指示として適用した。

今回の新科学計算、数値再解析、sampling、compile、図表生成、grant、詳細GPT/Claudeレビューはすべて0。
文献10件/12比較項目、Matrix差分8行、保存引用・参照・版差・保全など16項目の文書検査がPASS。
旧文書checkerのGit読取・digest・check helperを再利用し、旧39文献/cache全監査は繰り返さず今回の差分へ限定した。
既存の採用CEM・台帳・manifest、原稿、主表、Figure 2、SOURCE・結果・STOPは変更しない。
公開対象は今回の文献情報・比較・構成案のみ。**mandatory STOP、旧strict COMPAREはBLOCKED。**

次担当：**GPT・ユーザー**が中心主張と案A/B（またはSupplement限定）を判断する。
**Codex**は指示があるまで停止を維持する。科学実行や最終freezeを自動開始しない。
