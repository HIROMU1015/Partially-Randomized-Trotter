# O2：DF＋Pauli-JW研究データの参照台帳

2026-11-05 JST / LOCAL。基準は3環境統合commit `3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd`。
既存23研究scope、候補表等への24参照、主要28保存レコードのliteral転記、24保存比較への参照、8比較可能性レコードを整理した。
行数は実験数・独立標本数ではない。既存の `CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT` は再審査せず保持する。
O2の正式採択、主要claim・論文構成・結論の採用は行っていない。

## 使い方

| ファイル | 行の意味・用途 |
|---|---|
| [study_catalog.csv](study_catalog.csv) | 元artifact/runnerのscope名を使った研究索引。H/state/reference、rank/項数、SOURCEの役割、結果・精度・compiler・限界へ進む |
| [candidate_results.csv](candidate_results.csv) | `TABLE_REFERENCE` は元表全体またはJSON領域への入口、`LITERAL_EXTRACT` は保存fieldの選択転記。両者は独立観測でない |
| [comparison_catalog.csv](comparison_catalog.csv) | 元研究で比較済みの対象・条件・方向・共有価格・機構説明を参照。C/N/G・r/K・P1/P2/P3もここから探す |
| [df_pauli_matching.csv](df_pauli_matching.csv) | 比較目的ごとの一致・相違・仮定・未確認項目。費用差を参照できることと表現単独の因果効果を区別する |
| [data_dictionary.md](data_dictionary.md) | 記号、価格種別、欠測、selector、転記field、各CSV列の定義 |
| [SOURCE_INDEX.json](SOURCE_INDEX.json) | 各ref_idの固定commit/path/blob/bytes/公開URL。科学SOURCEと結果carrierは別 |
| [VALIDATION.json](VALIDATION.json) | 文書・参照・転記の技術確認。新科学的PASSではない |
| [validate_catalog.py](validate_catalog.py) | stdlibだけの参照・literal検査。科学moduleをimportせず、N/G/SE等を計算しない |

`study_id → result_refs → SOURCE_INDEX.references[ref_id] → commit:path` の順に辿る。
候補は `row_selector`、転記値は `projection_map` を使う。CSV selectorは文字列完全一致、JSONはRFC6901 pointerである。
元表の列名・有理数文字列・空欄・nullを保ち、数値の型変換・単位換算・log_BからBの復元をしない。
科学SOURCEを未転記の研究は `source_resolution=USE_FIXED_BINDING_OR_REPORT` と明示し、固定report/input bindingから遡る。
結果commitを代わりのSOURCEへ付け替えていない。
`source_availability` は記録されたSOURCE IDと手元のcommit object取得を区別する。
境界Rの完全SOURCEはprivate保全で、公開summaryにある実行・会計IDのみ参照する。原本取得・完全照合済みとは扱わない。

検査例（必要な固定Git objectが既にあるcheckoutで実行）：

```sh
python3 docs/research/o2_data_catalog_20261105/validate_catalog.py --repo .
```

## 収録範囲

DFはH4 legacy DF12のM1/PM2・1.30 Å固定transfer・B0I geometry、H6/H8強化対照、H2/H4/H6共通高次PFと固定確認、
H4/H6長時間B2/B3、R境界、旧H8 T=.8確認、H8 T=1.6 Stage A/B・Cを収録した。
Pauli-JWはH2/H6旧family、最新H2確認、H4 B0/PR、H4/H6純RTE event拡張、最新H6 Hybridを収録した。
表現・機構は旧H4共通入力比較、H6 P1/P2、H4 P3 Stage4/finite-bitを収録した。

H4 legacy DF12とtol-only DF10は別H。DF10/19/28を同一データ集合へpoolしない。
H6 DF19は保存weighted-Hermitian completion込み。Pauli14/184/918のstate/referenceも分子別に保持する。
元研究が複数系を扱う索引は `system` / `molecule` / geometry / H・state / cohort で分けて使う。
同じ分子名・精度・q/r/Kだけではmatchedとしない。

最新Pauli内部14/15の方向は元統合・追補の引用として保持し、旧 `latest15.csv` のH6欄を新H32/R64へ転用しない。
H8 Stage Cは同一の既存固定PR確認n9であり、再登録した新実験ではない。
固定PF価格はStage B探索由来、確認PR n9と探索PR n9はpoolしない。
旧H8主表／Mechanism／Figure2B、G04、p8原因説明も同じ保存cohortへの別の参照である。

## DF–Pauli間で比較できる範囲

旧H4共通MO入力/raw rayの固定実装には保存単回価格と条件付きN/Gがある。
H4 P3は共通Gaussian T_eff保持/V_square残差を固定した `CONDITIONAL_MATCHED` case studyで、
Stage4のfinite-bit gap=0、compiled rho=0、B engineering候補coverage等の仮定を保持する。
finite-bit追補はengineering診断で、実装全域の保証や旧N/Gの更新ではない。

`MATCHED` の行は同じ保存Pauli programのgeneric/central制御価格対照であり、DF–Pauliの表現のみを変えた比較ではない。
DF–Pauliの固定比較は条件付きで参照できるが、保持・law・B・basis/control等が同時に異なる。
H6 P1/P2をDFとの一般因果比較へ転用しない。DF B3はone-body保持、Pauli純RTEは全非identityランダム化である。
legacy DF12とP3 DF10のpool、H8 T1.6とPauli別分子T.8の直接費用比較は `NOT_COMPARABLE`。
最新family best同士の共通bindingは未確認のため分類を空欄、`verification_state=UNCONFIRMED` とした。
比較可能性ラベルはデータ整理用であり、元科学statusを変更しない。

## 資源競合データと今後の利用

`comparison_catalog.csv` の `RESOURCE_DECOMPOSITION` / `RESOURCE_SENSITIVITY` / `MECHANISM` 等から、
`C_N_G_decomposition.csv`、`same_q_prefix_R32_K4.csv`、`r_K_sensitivity.csv`、境界R原表、P1信号差、
P2部品・非加法性、P3 bias/shot/headroom表へ進める。finite-RTE/PFの基礎文書はSOURCE_INDEXのSRC_FINITE/SRC_COEFFにある。
保存された観測と元reportの機構解釈を別列にし、未分離の因果要因を残した。

原保存値と図表assetの参照は今後の整理・図表化の材料になる。
本台帳からの新集計、平均/SE/比/G計算、winner選択、統計・因果分析は今回実施しておらず、実施許可も含まない。
新たな科学的数値分析が必要なら、その具体的scopeへの別指示が必要である。
現在のデータ整理を完了するための新規科学計算は不要。

## 未収録・保全

全候補の新しい一括CSV、全119,304観測、全private raw/QPY/seed、全経路のSOURCE依存閉包は複製していない。
M1の大きなcompile_map、長時間の全候補表などは固定selectorによる索引に留める。
転記した28レコードは境界6、H4 PR主表5、H6 H32確認5、P3 Stage4両arm10、H8確認aggregate2であり、完全な候補包含ではない。
欠けたC、実rho、rare tail、厳密実装保証を0で補わない。H8のPR平均Cは原aggregateにないのでG/Nから逆算しない。

Phase2/3と3環境差分の索引を再利用した。新規opaque canonical IDを発行せず、元artifact scope名と候補IDを保持する。
本台帳は既存35行claim-evidence mapを置き換えない。RPE・Track Bは基準統合の派生成果リンクに残し、有限時間signal表へ混合しない。
科学SOURCE・原結果・固定契約・grant/ledger/marker/STOPと、基準統合directoryは無変更。
新科学計算・数値再解析・sampling・build/compile・詳細レビュー・原稿/図作成は実施していない。

専用branchへcommit/pushし、remote SHAと変更全fileの公開blob/本文を照合後mandatory STOP。
次の担当はユーザー：必要な図表化・追加分析の対象を選ぶ。今回、論文構成・主要claimの正式採否は判断しない。
