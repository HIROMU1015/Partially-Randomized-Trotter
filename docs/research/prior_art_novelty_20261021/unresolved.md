# 未取得文献・未確定事項

調査基準日：2026-10-21 JST。未取得・未確認は不存在の意味ではない。以下は不足の記録で、追加科学処理の実行指示ではない。

## 1. 文献・付録・公開実装

|ID|未取得/未確認|現時点で確認したもの|判断への影響|次に確認する担当（提案、未着手）|
|---|---|---|---|---|
|L01|P09 Pocrnicらの雑誌版全本文のローカルPDF|arXiv v1 PDF、雑誌書誌・DOI、一次配布PDF冒頭のweb抜粋。直接取得は429|雑誌版で追加されたpartition最適化等の全数値/Methodsまでは認定しない|Codex：依頼があれば誌版の関連節を追加取得|
|L02|P20 Mehendaleらの雑誌版本文PDFのローカル取得|arXiv v3、出版社web本文入口/abstract・DOI。直接PDFは403|v1のfirst-order/3著者とv3/誌版のsecond-order/4著者を混ぜない。誌版の定量相関値を本研究へ移さない|Codex：依頼があれば誌版との節単位照合|
|L03|P07 Jin–Liの雑誌本文（subscription preview）|arXiv v1とSpringerの正式書誌・abstract。online2023/issue2025を区別|詳細定理・実験について誌版まで精読済みとはしない|Codex：必要なら合法的に利用できる誌版で照合|
|L04|P27 SPRINTの全51頁の証明・数値付録、P30 MLMCのaugmented-state実装費用の全証明|関連本文/導入/Fig.1、P30 v2 Theorem1と§2.3|これらを本研究へ適用できるとも、既存比較を全て代替するとも認定しない|Codex：新しい方法論主張に必要な箇所だけ別途照合|
|L05|P01の公開figure/code packetと2026年改訂版の完全な対応|Zenodo 15387187のZIP取得成功。308 entryの目録、17 text/code fileを抽出し、関連plot entryを静的確認。arrayのdecode・実行なし|2025年データを2026年誌版/v2の全図再現物と自動認定しない。先行実装に同等native比較がないと断定しない|Codex：必要なら版対応・コードの該当範囲だけ追加確認|
|L06|背景文献の全本文/Methods|39件のPDF取得と読取範囲を [references](references.md) に個別記録|abstract/導入評価の文献について同等対照・手法の不存在を主張しない|Codex：中心主張と直接競合する箇所だけ優先的に補読|
|L07|未確認の雑誌DOI・出版状況|該当arXiv landingに誌版登録がないものは、未確認と記録|「未出版」「査読されていない」と断定しない。arXiv DOIとjournal DOIを別欄に置く|Codex：投稿時点で書誌を更新（今は新規投稿しない）|
|L08|Qiskit1.3公式compiler APIページのweb取得|既存SOURCEと保存compiler条件。公式versioned URL取得はtoolで失敗|特定transpiler passの因果説明を新しく認定しない。現在版の仕様で旧費用を説明しない|Codex：必要なら1.3.0固定codeを静的確認|

## 2. 版を混同してはいけない箇所

|文献|版差の確認|本報告での扱い|
|---|---|---|
|P01 Güntherら|arXiv v1: 2025-03-07、PRX Quantum誌版: 2026-05-19、arXiv v2: 2026-07-10。v1の費用式E5と誌版/v2のE21–E23は番号/定数等が異なる|主な式番号は誌版/v2に対応。v1の数値定数を現在版のものと混ぜない。版間の全変更監査はしていない|
|P21 Moralesら|v2: 2024-07-16 “Greatly improved higher-order product formulae for quantum simulation”。v3: 2025-01-10 “Selection and improvement…”、誌版DOI 10.2478/qic-2025-0001|両版のPDFを取得。BibTeXもv2とv3/誌版を別entry。P33が引用するv2をv3タイトルで置換しない|
|P20 Mehendaleら|v1(2023): Martínez-Martínez/Kamath/Izmaylov、first-order。v3(2025-08-04): Mehendaleが加わりsecond-order。誌版abstractの相関表現も異なる|現在版著者4名と誌版書誌を明示。旧版の主結果を誌版のものとして引用しない|
|P07 Jin–Li|Springerオンライン公開2023-12-27、volume7/pages442–469のissueは2025-04|online年と通常の誌面引用年を両方記録。2025年発明と誤認しない|
|P14 Chenら|誌版2021、arXiv最新v4は2026-03-25|v4を取得した事実と2021誌版の存在を分離。全改訂の同一性は未認定|
|P30 MLMC|v1 2026-04-29、v2 2026-05-19|v2の§2.3を確認し、exact pathwise observableとquantum shot-noiseの条件を区別|
|P25 Paganelliら|arXiv v1履歴と、取得HTML/PDF内部のdate表記に差がある|PDF v1と取得SHAを基準。HTMLのdateを新arXiv versionと推測しない|
|P37 FreeQuantum|arXiv v1初公開2025-06-25、JCTC誌版2026-04-19（issue2026-05-12）|誌版PDFを取得してresource Appendixを確認。arXiv初版と全文同一とは認定しない|
|P28 SCU|DOI文字列に2025を含むが、誌版公開は2026-02-19。arXiv v2は2026-07-06|DOIの年部分から出版年を推測しない|

BibTeXのarXiv entryは使用版のdate/year、参考文献本文では初公開日とversion historyを記録した。雑誌entryはCrossref/出版社の書誌に対応する。online年とissue年が異なる場合の通常引用年は個別確認する。

## 3. 研究証拠・新規性の保留

|ID|未確定事項|現在の支持範囲|新規性評価への影響|
|---|---|---|---|
|E01|希少eventを含むcompiled Cの母平均・分散/順位|有限標本の保存点とSE。未観測費用unknown|優位性を母平均へ置く論文の必須論点。別Codexの既存証拠調査予定を今回の完成結果には数えない|
|E02|bias/uの厳密certificate|経験的signed/noncancellation・倍率感度|“guaranteed matched accuracy”という方法論主張は強過ぎる|
|E03|候補域/最適化自由度の対称性|登録D/A/P、H6/H8 PF q≤8、PR q16個別点あり|普遍的なalgorithm rankingより登録domainの結論に限定|
|E04|最新構造化/高次PF、ordering/representationに対する比較|S2/S4・固定prefix、P27/P21等は文献比較のみ|“strongest deterministic baseline”とは言えない|
|E05|理論/compiled差の因果分離|固定NATIVE_MIXED期待値とcompiled標本|単一compiler最適化の因果効果、P01のFT式の反証ではない|
|E06|H8元compiled回路|原profile回収済みという補足receipt。保存PR回路0/192、旧bit同一性未検証|再現性の限定を残す。profile未取得という古い批判は更新|
|E07|全入力生成手順の一致、一般化・サイズの因果/scaling|共通DF policy、solver/normalization/候補域差は残る|H2–H8の並びを普遍的scaling則としない|
|E08|energy/RPE全体、状態準備、FT合成|固定T=.8、P=0、compiled RZ×analytic shots|既存P01/P26より広い実用総費用研究とは主張できない|
|E09|進行中の別4次PF研究の完成内容|本報告は結果未取得・未介入|高次PRの新規性・総費用改善は予告できない|
|E10|移送可能な非自明な判定則|局所case/反例/感度の蓄積|独立論文のimpactを強めるには説明力が重要。追加点の量だけでは解消しない|

## 4. 次作業の区分

- **ユーザー**：新しい文書の公開範囲を判断する。研究方針の採択や追加科学処理は、望む場合のみ別途指示する。
- **別Codex（ユーザーが予定している担当）**：希少RTE既存証拠調査。今回の担当は介入しない。
- **別4次PF研究の担当**：その独立した既承認scopeで研究を継続する。本報告は停止・条件変更・結果提出を要求しない。
- **今回のCodex**：ローカル成果の構造・SOURCE保全を確認してcommitし、mandatory STOP。新規科学計算や追加レビュー依頼はしない。
