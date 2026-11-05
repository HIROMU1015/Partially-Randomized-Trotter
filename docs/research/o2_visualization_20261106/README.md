# O2：保存値による暫定可視化

2026-11-06 JST / LOCAL。指定14候補を、10比較図と1定義表の計11点へまとめた。各点をSVGとPNGで保存した。
既存図ファイルの直接流用は0点。保存値から描き直し、既存描画コードの型変換・log軸・panel/marker配置の適合部分だけを再利用した。
これは研究用の暫定可視化であり、完成論文caption/Results本文、主図/補足/枚数/配置の正式確定ではない。

基準：図表計画 `59f7083dacf6fdf25b6b593608d60c1e816846a3`、台帳 `2779710a86d380bb484accc51d3dccba183370b8`、3環境統合 `3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd`。
正式共通指示 `research_project_instructions_unified_2026-10-25.md` §8.1/8.4/10を確認した。
`CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT` と正式科学statusは変更しない。

## 一覧

| 図表 | 元候補ID | 比較内容 | ファイル |
|---|---|---|---|
| V01 | DF03 | DF03：H2/H4/H6の強PF対照 | [SVG](figures/V01.svg) / [PNG](figures/V01.png) |
| V02 | DF04 | DF04：時間・精度依存と確認 | [SVG](figures/V02.svg) / [PNG](figures/V02.png) |
| V03 | DF07 | DF07：B3の負結果 | [SVG](figures/V03.svg) / [PNG](figures/V03.png) |
| V04 | DF09 | DF09：H8 Stage C固定比較 | [SVG](figures/V04.svg) / [PNG](figures/V04.png) |
| V05 | PA01, PA02, PA03 | PA01/PA02/PA03：PauliのN/C/G | [SVG](figures/V05.svg) / [PNG](figures/V05.png) |
| V06 | XP02 | XP02：P3条件付きmatched partition | [SVG](figures/V06.svg) / [PNG](figures/V06.png) |
| V07 | XP04 | XP04：方式定義と比較条件の表 | [SVG](figures/V07.svg) / [PNG](figures/V07.png) |
| V08 | ME01 | ME01：保存N/C_eff/G比の対応 | [SVG](figures/V08.svg) / [PNG](figures/V08.png) |
| V09 | ME03, ME04 | ME03/ME04：H4/H6のR境界 | [SVG](figures/V09.svg) / [PNG](figures/V09.png) |
| V10 | ME05 | ME05：H6 P1の差と分解能 | [SVG](figures/V10.svg) / [PNG](figures/V10.png) |
| V11 | ME06 | ME06：H6 P2の固定program価格 | [SVG](figures/V11.svg) / [PNG](figures/V11.png) |

V05はPauli3候補、V09はR境界2候補を共通形式の別panelへ統合した。他は比較目的を分離した。
対象14候補の作成保留はない。追加分析を要するME09/ME10や、未確認/条件不足のXP05–07は対象外で開始していない。

## 読み方と限界

Nは元十分条件による解析上の量子shot数、nは古典価格trajectory標本数。Cは測定付きfull Hadamard wrapperのcompiled RZ/shot。
G=N_cos C_cos+N_sin C_sinは原保存値。GとG_SEの単位はRZ×shot。今回この式を新しく評価していない。
P=0。native T/CX・物理runtime・状態準備込み費用・RPE総費用への換算はしない。
DFはFull PF=one-body＋全fragment決定論、B0=tail discard、B0I=既知identity scalar復元、B2=one-body/prefix保持＋finite RTE、B3=one-body保持＋全二体tail finite RTE。
PauliはFull PF=全項S2、B0/B0I=constant保持＋prefix外をdiscard、Hybrid=非空prefix保持、純RTE=constant以外の全非identity項をランダム化。

全て元保存H/state/reference/rank・fragment順・精度政策・有限候補域に限定する。H6 DF19はweighted-Hermitian completion込み。
同じ分子名・L_D・q/r/Kだけで一致を認定せず、DF12とDF10や異なるtime/stateをpoolしない。
選んだ価格は元のDIRECT compiled価格・有限標本価格。元MODEL/EXTRAPOLATED系列をDIRECTへ読み替えていない。
探索・固定確認・確認後min・逐次延長の選択policyは元のまま。複数精度・複数panelで同じ価格を再表示しても、独立標本数は増えない。
表示データの行数は実験数/trajectory数でなく、図のmark・field参照の数である。

±1 SEは元価格標本の保存SEの表示。新しいCI・有意差・母平均順位・population tail保証ではない。
保存SEが小さくmarkerに隠れる場合も原文字列と誤差棒の対応を保持する。null/空欄/未取得SEをゼロにしない。
原保存ゼロと未解像差は別。C/N/Gの別単位を同じ軸へ混ぜず、C_effは元のshot加重有効費用として扱う。
数値文字列のFraction/float変換、log軸、tick/セル文字列の丸め、SE棒の幾何学的表示だけを行う。

P3には `CONDITIONAL_MATCHED`、finite-bit law/angle gap=0、compiled residual ρ=0、B engineering候補coverageを図中にも明記した。
表現だけの因果比較・実装全域保証・母平均保証ではない。finite-bit追補で旧N/Gや仮定を更新しない。
XP04はCONTEXTUALの定義表。同一Pauli programのcontrol比較がMATCHEDであっても表現単独効果ではないという原台帳の区分を保持する。

### V01：DF03：H2/H4/H6の強PF対照

原60比較行のS2/Y4/m5/ALL_FORMULASを分離。共通PF588候補域と保存PR探索価格を使用。別の固定高次確認118 trajectoryはpoolせず、文脈参照に留めた。

保存比較ID：`H246_common588 / H246_fixed_high_order_confirmation`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V02：DF04：時間・精度依存と確認

原40セルのALL探索選択と、固定比較CSVにある33件のPR–PF確認を別markerで表示。残るセルの確認値を補完しない。元33/40と拡張33/40・境界Rは別集合。確認側の個別G SEはこのCSVにないので、差のSEを個別誤差棒へ転用しない。

保存比較ID：`H46_original_independent_comparisons`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V03：DF07：B3の負結果

H4/H6の40条件とB2・Full PF・PF envelopeの3保存baselineを全て表示。各baseline行で繰り返される同じB3値は1セル1回だけ表示。one-body保持B3の有限取得域の結果であり、全Hamiltonian純RTE不可能性ではない。

保存比較ID：`B3_fixed_negative`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V04：DF09：H8 Stage C固定比較

PR確認n9の保存Gとpaired価格SE、Stage B由来の固定PF価格を表示。削減率24.58909439%は原reportの表示文字列。探索PR n9をpoolせず、PF SE nullも保持。PR平均Cは未保存なのでGだけで成立する表示とした。

保存比較ID：`H8_M5_P8_Q1_CONFIRMATION_01`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V05：PA01/PA02/PA03：PauliのN/C/G

3分子を列、G・N・Cを別行へ統合。H2の元精度別選択、H4の最新直接比較の選択policy、H6の後続r32 Hybridを固定。元14/15とH4粗精度例外を注記。cos/sinは別marker、Cはゼロ始点の線形軸。

保存比較ID：`Pauli_latest14of15`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V06：XP02：P3条件付きmatched partition

共通Gaussian保持のPauli/DF残差について保存N・C・Gを別panelで表示。CはStage 4入力に固定されたStage3/run04 price_statistics.jsonの保存平均・SEをそのまま引用し、新しい平均を計算していない。

保存比較ID：`P3_Stage4`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V07：XP04：方式定義と比較条件の表

DF B3のone-body保持とPauli純RTEのconstantのみ保持、tail/basis/law/control、固定条件の違いを英語表へ整理。CONTEXTUALであり費用順位図ではない。

保存比較ID：`B3_fixed_negative / Pauli_latest14of15`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V08：ME01：保存N/C_eff/G比の対応

C_N_G_decomposition.csvの160保存行を3つの無次元panelへ配置。S2・高次側の元candidate IDも保存best表へ対応付けた。新しい比は作らず、C_effをC_cos/C_sinと区別。セルの丸めは表示のみ。

保存比較ID：`C_N_G_decomposition.csv`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V09：ME03/ME04：H4/H6のR境界

原台帳のLITERAL_EXTRACT selectorを再利用して確認cohortと保存確認baselineだけを抽出。探索価格を同じmarkerへ重ねない。N・C・Gを別panelにし、H4 HIGHERとH6 point_lower=True/INCONCLUSIVEを保存判定のまま表示。

保存比較ID：`H4_R16_R128_vs_R64 / H6_R11_vs_R16`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V10：ME05：H6 P1の差と分解能

保存absolute差とabsolute_resolutionをそのまま表示。RESOLVED_EMPIRICAL/UNRESOLVEDをmarkerで区別。保存ゼロはlog軸上の正数へ置換せず文字で表示。構造ゼロ・浮動残差・未解像差を区別。

保存比較ID：`P1_bias`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

### V11：ME06：H6 P2の固定program価格

Full/R/E/Emptyを積み上げず別棒で表示。保存Δinteractionを注記。固定confirmation/index0000の価格診断であり、母平均・N/G・原因別寄与率へ拡張しない。

保存比較ID：`P2_parts`。元study・全candidate ID・cohortはmanifest/displayデータへ進む。

## 固定データ・表示の追跡

- [FIGURE_MANIFEST.csv](FIGURE_MANIFEST.csv)：図表→元候補/study/comparison、科学SOURCE役割、固定条件/cohort、selector/field、仮定、作成状態。
- [DISPLAY_DATA.csv](DISPLAY_DATA.csv)：全表示markの原ref_id、固定row/pointer、保存literal、N/n、candidate/cohort/価格種別・価格共有group。新しい数値正本ではない。
- [FIGURE_VALIDATION.json](FIGURE_VALIDATION.json)：ref_id→固定commit/path/blob/bytes/URL、技術確認、保全・限界。

SVG各markのidはDISPLAY_DATAのpoint_id。SVG内metadataに描画artistへ渡した値と原literalの対応を保存した。
図を別panelへ再配置しても、同じ元価格を追加実験と数えない。完全なprivate raw/全seed/全SOURCE依存閉包の新監査は行っていない。
境界Rの完全SOURCEはprivate保全で、公開summaryの記録済みIDと公開保存結果を参照した範囲。

## 再利用・実行・技術確認

既存 `h46_long_time_report.py` はG/SE/比/選択を再評価するため全体を起動しなかった。
旧Pauli event図は別s軸・旧Hybrid anchorであるため、現在の精度比較図へそのまま流用しなかった。
`plot_pauli_rte_event_zoom.py` の保存有理数→float・log軸・marker分離、長時間図のpanel配置、旧H4図の欠測をgapに保つ方針を適用した。
旧2SE表示・SE欠測を0にする処理は流用しない。新規実装は保存field抽出と表示だけに分けた。

1. [抽出script](scripts/extract_saved_display.py)：固定Git blobとselectorで保存fieldを転記。数値型へ変換せず、平均/SE/比/順位を算出しない。
2. [描画script](scripts/render_saved_figures.py)：DISPLAY_DATAだけを読み、許可された表示変換でSVG/PNGへ描画。
3. [検査script](scripts/validate_figures.py)：元literal、cohort、candidate binding、SVG mark/表示値/単位・仮定、PNG形式、保全と静的非科学実行を検査。

```sh
python3 docs/research/o2_visualization_20261106/scripts/extract_saved_display.py
python3 docs/research/o2_visualization_20261106/scripts/render_saved_figures.py
python3 docs/research/o2_visualization_20261106/scripts/validate_figures.py
```

描画には利用可能なMatplotlib/Pillow環境を使用。既存環境を変更せず、font cacheはこのtaskの一時領域へ置いた。
着手時affinity32 CPU、MemAvailable約52.6GBを確認。11個の独立した軽量描画jobを並列化し、共有設定・他ユーザーjobを操作していない。
図表計画の既存validatorは新しい出力directoryを許可する契約ではないため、元commitのclean checkoutで変更せず実行した。
現checkoutの原台帳・計画・統合資料のbyte保全を別途確認し、既存CSV/pointer/literal helperを再利用した。

全PNGを開いてラベル・単位・凡例・余白・欠測・cohortの表示を確認した。
R境界は原LITERAL_EXTRACTの確認cohortに固定し、H6 INCONCLUSIVEを保持。C軸の数値目盛り、R境界annotationの余白、比率図の共通色凡例を調整した。
新科学計算・N/C/G/SE/比/平均/差の算出・数値再解析・再選択・sampling・build/compile・詳細科学レビュー・論文執筆は実施していない。
原SOURCE/raw/QPY/contract/grant/ledger/marker/STOP、既存台帳・図表計画・35行mapは無変更。

## 次の担当

ユーザーが暫定図表を確認し、採用する比較、表示形式・panel分割・言語・配色・注記の修正を選ぶ。
今回の対象を可視化するための新しい科学計算は不要。H8平均Cや因果寄与率を追加する場合も、今回の処理から自動で開始しない。
専用branchへcommit/pushし、remote SHAと公開全成果物を照合した時点でCodexはmandatory STOP。
