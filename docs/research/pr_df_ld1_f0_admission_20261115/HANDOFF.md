# C受入：PR-DF LD1 F0保存確率診断

2026-11-15 JST。**診断表14/14件、round別条件付きevent/path上界13/13組、対応先26/26軸を受入。保留はいずれも0、不一致なし。新科学計算0。**

正本は[ADMISSION.json](ADMISSION.json)。`tables`は共通component 1件＋order 13件、`round_bounds`は共有上界13組、`rows`はCのm0〜m12×X/Yの26軸。X/Yの2行を独立な2診断とは数えない。

## 固定参照と追跡

- D2最終引渡し：`d749ef92e8ac3087747481a46aeb28ec4ca04c4b:docs/research/d_pr_df_ld1_f0_law_diagnostic_20261115/C_HANDOFF.json`。
- F0実行SOURCE：`fad0c1e4fe41c0e4c6d0690158056572295ed15a`、同ディレクトリの`INPUTS.json`。元入力のSOURCE/selectorは診断表ごとに保全。
- F0 raw：`ec78fde57e117e9bbf79a26bfaa2e1575c1f5784`、同ディレクトリの`run01/{RESULT,ROUND_BOUNDS,components,order_m0..12}.json`。
- 採択計画：`950577f416c17a20d309a8053a4bb016027b724d:docs/research/d_pr_df_ld1_ec_plan_20261115/NEXT_F0_PLAN.json`。
- C候補/元E_I[S]/価格receipt索引：`c99d511e2422b8f250490b11054f754a81e536c5:docs/research/pr_df_ld1_q3_price_admission_20261115/ADMISSION.json`。案内は指定`8eef0ad7…`の同ディレクトリ`HANDOFF.md`。
- 保全したm4受入：`9edfa5d3201f22c23dd77569d102801be80dacf3:docs/research/pr_df_ld1_m4_holdout_admission_20261115/ADMISSION.json`。

候補・軸→`ADMISSION.rows[j]`→`source`でD2行、`prior_admission`でC既存行、`alignment`/`Q3`で物理/精度側の固定対応へ戻る。`diagnostics`と`conditional_bound`は本JSON内selectorで、共有entryの`diagnostic`/`source`からD2 rawへ戻る。各診断の`input`→F0固定入力→`original_source`が元preparation/law。`structural`は保存E_I[S]への既存locator、価格receiptは`prior_admission`の`finite_paths`から追跡する。m4のreceipt・記述的比較は保全した別受入を参照する。

## 照合した範囲

指定branchの開始HEADは`81596ce7c561d7b75890cfe2758a0d4a12fb6b47`、worktreeはclean、remoteと一致。既存C索引・関連Git履歴/結果path名で同じF0受入を確認できず、元D2照合を再利用して差分consumerだけを追加した。他担当のworktree/実行領域は変更していない。

既存`price_implementation_guard.evidence`の固定Git JSON reader/cacheとCの`at/equal/require`を再利用。価格guard自体は呼び出しも変更もしていない。14表の完了status・原SOURCE・F0固定入力signatureと元保存metadataの一致、共通preparation hash、元component順、roundのphysical ID/q/R/event位置数、Cの保存構造lawの宣言、X/Yの共有locatorを確認した。確率の和・TV・上界・E_I[S]・価格・誤差を再計算していない。

正本のexact TVとevent/path上界はD2保存JSONの分子・分母をそのままcopy。`support_changes`は元recordへのlocatorを保持。目印となる表示値はD2報告から転記：component P→G TV約4.5413×10⁻¹⁵、CDF丸めで25成分・格子化で追加7成分が正→0、最大条件付きpath TV上界はm12の約1.5248×10⁻⁷。これらの値で価格許容差や資格を設定しない。

## 仮定・未解決事項

F0のPは保存reference weightから定義するlaw、Gは一様53-bit格子上の写像分布。P→Gを実PCG64のjoint-stream距離へ読み替えない。以下の元仮定をJSONの`assumptions`と各`saved_bound.assumptions`へそのまま引き継いだ。

- 同じevent labelが同じ保存word/angle/phaseを表す。
- ideal/gridの各モデル内でorder条件付きcomponent drawが独立。
- 全event位置・再開outer streamに独立一様53-bit drawを用いる、又は一致履歴に一様な条件付き距離の根拠がある。

実PCG64同時分布との差、解析的な実数重みへの誤差、C−Sの全support変動幅、欠けるsupportの寄与、価格補正、正式E[C]・CIは未確定。有限cohort/rare/共有誤差と長roundの残差移送資格も未解消。今回の記録は価格provider recordではなく、価格guard・モデル資格を変更しない。

保存E_I[S]、凍結価格モデル、m4独立holdout受入、Q3のideal-law契約は変更なし。`qualified_E_C=null`を全26軸で維持する。N/G/winnerは算出していない。

## 技術確認・終了

[TESTS.txt](TESTS.txt)：新規人工5件＋既存m4受入回帰5件、計10件PASS。人工入力で共有件数、exact値のcopy、source/law/physical不一致時の保留、実PRNG/価格資格への昇格防止を確認。実入口の保存読取も14/13/26件すべて受入。作者による技術確認であり、独立科学レビューではない。

ロードマップ差分：**F0の条件付き分布差をCの26軸へ接続。正式E[C]は引き続き全26軸で未資格。** ロードマップ本体は統合担当が更新する。

**mandatory STOP：commit/push・remote SHA/公開成果物照合で終了。F0再実行、追加数値解析、希少event調査、価格補正、fit、sampling、compile、N/Gへ進まない。**
