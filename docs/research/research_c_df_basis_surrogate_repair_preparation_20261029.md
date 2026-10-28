# 研究C：saved64 basis-key修正・run02実行準備

2026-10-29 JST。起点`52e152e211d1e321b5017eda4c8e8b01cfdda3d7`。ユーザーは技術修正と科学再実行を明示許可した。追加承認・承認証明・新grantなし。旧run01の3-call STOPは保全し、fresh run02を新しい148-call実行とする。途中結果/残枠をresumeしない。

## 最小修正とSOURCE保全

旧geometryは`_rotation(np.kron(U,I2))`のreprをhashしていた。既存general-K生産側は`_matrix`でcomplex化し、`_spin_lift`のoff-spinに明示的0jを入れる。数値が等しくてもsigned zeroのrepr差でkeyが一致しない。

新`c_df_basis_surrogate_v2.geometry`は**既存`_matrix`/`_spin_lift`をそのまま再利用**してkeyを再構成する。旧key・保存U・catalog・sharing条件は変更せず、近似key一致/許容差/fallback/gauge変更を導入しない。

旧99 SOURCEはbyte単位で無変更。新helper/v2 binding/CLI/人工testの4ファイルだけを追加し、旧feature/run/RPE/統計コードを継承する。v2 bindingは単独task process内でfeatures/scope/fresh pathを差し替え、finallyで復元する。共有環境・旧ファイルには変更しない。

[修正契約](research_c_df_basis_surrogate_contract_repair_v2_20261029.json)は旧契約のcohorts/models/folds/feature/numerical/statistical/call/packages条件を照合する。A/B/C/mean、両cohort非paired、4-fold/index mod4、主RZ MAE、rank不足/null、new-DF-fit-GO=falseは不変。新scopeとrun02、修正SOURCEだけを変更した。元の固定条件は[設計](research_c_df_basis_native_cost_surrogate_design_20261029.md)参照。

## 人工・回帰

Reflection/det−1、π/det+1、負成分の回転、random SO/O、複数frame、signed zeroを含む7新テスト＋既存43＝**50 PASS（2.09秒）**。既存生産経路で作った人工catalogにv2 keyが一致し、旧geometryでは失敗すること、U改変の真のkey不一致はv2も拒否することを確認した。Bindingのfinally復元、旧statistics/trajectory関数の同一性、科学条件変更の拒否も確認した。保存H4を人工試験に用いていない。

- [SOURCE103](../../artifacts/research_c_df_basis_surrogate/2026-10-29/technical_repair_v2/SOURCE.json) SHA256：`dc3890d10fdda3c378fd8d35c386a5a709b81634d0988e3c5c8c1d056204bce1`。
- 契約SHA256：`365d52a7df356a4162dbfd65e9810b7820b5418ac3dcb2b83688c789ba3a81d3`。
- Input275件、旧保護1570件、旧契約/SOURCE/run01/STOPをhash照合。原科学input267件は変更していない。
- [人工ログ/verification](../../artifacts/research_c_df_basis_surrogate/2026-10-29/technical_repair_v2/)。Passive preflight PASS。科学開始前にSOURCEをcommit/pushしremote照合する。

## 実行範囲とSTOP

保存H4・1Å/STO-3G、CDF8/LD2、partial S2 δ=h=.1/s1/r128/K0、固定32×2 cohort。静的特徴2、trajectory64、raw QPY census64、記述モデル12、読込/identity/最終判定6＝148。新DF fit/sampling/build/compile/追加候補/新目的採択は0。

Fresh outputは`artifacts/research_c_df_basis_surrogate/2026-10-29/run02`。技術relaunch1、run内retry/resume0。Correctness FAILでは部分証拠とSTOPを保全する。ユーザーが別途許可した意図しない技術中断の修正は、科学条件を維持し旧runを上書きせず、必要なら別SOURCE/契約・fresh outputへ固定して扱う。科学的不適格・負結果を条件変更で回避しない。

32 affinity CPU/空きRAM約52GiB/低負荷を確認。小さい保存I/O・4D行列・4-fold OLSが中心で、重いcompiler workloadなし。Task-local venv/cache、依存read-only、人工CPU/RAM/worker/wall capなし。結果を公開・remote/SOURCE/旧新原本照合後mandatory STOP。新DF/η/scale/詳細科学レビューへ進まない。
