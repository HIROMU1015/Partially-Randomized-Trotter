# Track A 通し原稿v0.1：claim/evidence audit

2026-10-05。対象は[本文](track_a_resource_study_v0_1.md)、[補足](track_a_resource_study_supplement_v0_1.md)、
4主図・図S1・表示CSV。基準はdesign commit d45c4006d440ba053517d6f646844a61b18dfd15と
evidence commit 5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc。

結論：執筆者によるscope・出典・記載値監査では、下記の過大一般化を採用していない。
新規性の十分性・投稿先・採択可能性は未判定である。
これは独立peer review、全先行研究の精査、immutable CI、厳密な数値証明ではない。

## 1 中心・補助claimの追跡

| claim | 本文の所在 | evidenceと比較domain | 同時に保つ限定 |
|---|---|---|---|
| C1 近接discard追加後も低いB2 primary点値 | Abstract、§3、図1、Conclusion | M1-B1＋PM-1、PM-2元精度台帳、development218、ε=.05、P=0 | 登録二次実装classのみ、B0 L5 q1との43.07%点差。全deterministic法・厳密winnerではない |
| C2 精度がB2内部設定を変える | Abstract、§4.1、図2 | PM-2 development218×302点、q4→q2→q1 | method最小は全表示点B2。切替は隣接表示点間、302独立実験ではない |
| C3 元transferと別precisionの適格性は別 | Abstract、§5、図3 | 元M2の結果前固定5構成・ε=.05と、その保存値のPOSTHOC感度 | TRANSFER_SUPPORTEDを維持。元5構成の競争力でありheld-out method optimumではない |
| C4 q帰属とselector損失を限定 | §6、補足S3 | PM-0、同M1 domainのq8部分集合と全体、旧16-cell selector | 両方B2、primary regret0、6指標Pareto一件欠落。S2との差をqだけへ因果帰属しない |
| C5 bias改善とcompiled costの競合 | §4.2、図4 | PM-0既存same-R group、B2 L3 K2 T.8 R8 | 一組の説明例。det/random/basis compiled費用分解、q一般則ではない |
| C6 状態準備感度と比較集合 | §6、補足S4、図S1 | PM-2 development218、PM-0 M1/M2共通5 | 共通仮想P。候補集合差をgeometryだけへ帰属しない、実準備回路は未評価 |

数値の正本は[主張・証拠対応表](../research/track_a_post_pm2_claim_evidence_map.md)が指すE1〜E6と、
各validation文書・machine-readable artifactである。
figure builderはPM2/PM0の明示9入力を元commit blobと一致確認し、表示用整形だけをした。
新しいsignal・shot式再評価・費用標本・crossover探索はない。

## 2 全文で確認した6つの飛躍

Abstract、Methods、各Result、caption、Discussion、Conclusion、Supplementを読んで確認した。
単語検索だけで文章の意味を証明したとはしない。

| 監査対象 | 原稿での扱い | 確認箇所 |
|---|---|---|
| B2一般最適性 | 登録集合のprimary点最小／点推定に限定し、未評価prefix・高次PF・強い合成への優位を否定 | Abstract、§3末尾、§6、Conclusion |
| M2をheld-out再最適化とする | 元5構成を保持。近接L5や未登録q2/q4を描かず、原transferとPOSTHOC感度を分離 | §2.4、§5.1–5.2、図3、補足S1 |
| ε_minを原理的限界とする | 対称軸配分・Hoeffding十分shot規則のstrict適格境界。等号不適格、別配分未検証 | §2.2、§5.2 |
| 302点を独立実験と数える | 同じbias/B/元32cost標本を使う事後感度。線はvisual guide、根を求めない | §2.4、図2・3、補足S2 |
| 点±2SEをformal CIとする | paired covariance、independent-candidate delta method、条件付きengineering intervalと明記 | §2.3、§5.1、図1・2、補足S2 |
| signal studyをchemical-accuracy energy/QPE総costへ拡張 | finite-time保存状態signal、compiled RZ proxy、analytic shotsに限定 | Abstract、§1、§2.1–2.3、§6、Conclusion |

追加確認：C_useのrigorous bound、H12外挿、新ground stateの取得、truth-free selector、
all-random B3、未保存B0 error分解、実準備費用、MC uncertainty込みの厳密winnerを主張しない。
binary64のceilを任意精度の厳密最少shot数と呼ばない。
Methodsでは物理short-step時間Δ_R=T/(qr)と無次元τ=λ_RΔ_R、paired cutoff Kと
通常のdegree K+1を区別した。有限分布のBを実装sourceへ照合し、infinite normalizationと混同しない。
図4の費用増加はfull-wrapperで観測したものとし、未保存の決定論成分だけへ分解・帰属しない。

## 3 図・表示値・scopeの検査

- 図1：元4構成のaxis shots・axis平均RZを保持し、primaryとの算術一致を確認。
  PM1-B0 IDを削らず、B0 L4は本文補助例だけ。random ±2SEをsampling変動として表示。
- 図2：development218×302の全保存domainを使い、method別eligible点最小と全体設定を分ける。
  候補が不適格なら欠測。表示構成のSEをfamily全体の保証にしない。
- 図3：固定5構成×302だけ。現行適格境界と費用順位切替を別に注記。
  元ε=.05の縦線を表示し、未登録構成の曲線を補間しない。
- 図4：既存一組に限定し、τ・B・random action期待値の一致を検査。
  saved bias、N、C_eff、Gを表示する。点は既存cost平均で、厳密winnerの主張はない。
- 図S1：候補domainを3panelで分離。保存affine segmentの式を描画し、
  log軸で両端だけを直線接続して実際のaffine関係を歪めない。
  新たなenvelope・交点探索を行わず、仮想Pと明記。

PNGを視認し、軸、legend、caption、footerの重なりを修正した。
修正したのは表示layoutと保存affine segmentの描画で、科学CSVや閾値ではない。
初回・二回目renderは/tmpへ退避し回収可能。正式科学artifactの削除・上書きは0。
最終PNG/SVG/PDFと表示CSVは専用asset manifestでhash追跡する。

## 4 文献の位置付けと残るreview

本稿はpartial randomization、DF部分保持、discard比較、cost×shotsという原理を新規性にしない。
最接近Güntherらは本文Sec. VII・Appendix D.2を確認し、DF・切断比較の先行例を認めた。
その他6件は一次公開ページのAbstract・metadata確認の範囲で、全定理・実装の比較を完了したとはしない。
各versionを本文Referencesに示す。

文献metadataの整合性確認により、原稿ReferencesではCasaresらのタイトルを
Theory and practice of Trotter product formulas for quantum chemistryとし、
Simon–Loveのv1をHalving the Cost of Controlled Time-Evolution、著者William A. Simon / Peter J. Loveとした。
設計文書の暫定書誌名は歴史として改変せず、science claim・証拠・方針は不変。
[Casares一次資料](https://arxiv.org/abs/2606.30741v1)、
[Simon–Love一次資料](https://arxiv.org/abs/2511.13855v1)。

残る判断は、完成原稿が独立resource/application paperに十分か、
technical/resource noteが適切か、どの未取得情報だけが本当に結論を変えるかである。
追加計算を完成要件へ自動追加しない。
[完成原稿レビュー依頼](track_a_publication_readiness_review_request_v0_1.md)で独立reviewへ渡す。

## 5 Auditの証拠階層と停止

script/test、元入力、出力manifest、表示CSV、原稿のhash・リンク・数値対応検査は
[verification audit](../../artifacts/resource_applicability/track_a_manuscript_audit/2026-10-05/verification.json)へ保存する。
synthetic/local testsの合格を科学結果の再実行、OS sandbox証明、独立再現、投稿可判定と呼ばない。
既存result/status/runner manifest、Track B、global validation manifestは変更しない。

原稿作成時点はlocal uncommitted v0.1。利用者の指示で原稿bundleをcommit/pushするが、
この執筆者監査は独立再現・immutable CI・独立投稿可判定ではない。
科学計算STOP、scientific_next_stage_authorized=false。
PM-3、追加trajectory/geometry、strong synthesis、高次PF、energy/RPE、Track B統合は未認可。
