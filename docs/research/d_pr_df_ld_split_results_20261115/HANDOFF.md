# 統合担当／C向けHANDOFF：PR-DF LD1/LD2 finite比較の部分STOP

**状態: PARTIAL_SOURCE_GATE_STOP。LD2較正24 compile・凍結価格・4配分まで完了。新Q3科学call0、4件は未評価。** [REPORT](REPORT.md)／[SUMMARY](SUMMARY.json)／[MENU.csv](MENU.csv)。

保存V0適合だけを再利用した現在の最小確認モデル費用は、LD1 **261,314,830,475.559718 RZ**、LD2 **63,914,076,114.980849 RZ**。各軸の選択列はV0 `[64,60,56,52,47,43,38,34,30,25,21,17,12]`。六件のmenu全評価最小ではない。MSE exact上下界はSUMMARYと固定a82b0f0eの元RESULTへ接続した。

## 固定入口

- LD2 raw公開: `b030ea97a6062a35a73069988c2ed616b6869bc4` / `docs/research/d_pr_df_ld_split_results_20261115/acquisition/RESULT.json`。
- 24行: [COMPILE_ROWS](COMPILE_ROWS.json)。各行にphysical ID・cohort/index/seed・axis・S/C/D・event digest・fingerprint・元axis receiptとPATH_RECEIPTの `{commit,path}`。
- 例: `b030ea97a6062a35a73069988c2ed616b6869bc4` / `docs/research/d_pr_df_ld_split_results_20261115/acquisition/m6/p0/cosine.json`、同path `PATH_RECEIPT.json` / selector `record.records[0]`。Xはcosine、Yはsine。値C551648、S526274、D25374。実行codeは `ea9e66092465d39dfcd4aeb6fea8e9f09808cabb`、結果公開SHAと別。
- 入力固定: `1ef521ba55ab4b128d22eca4970c4eba773f2dc4` / `docs/research/d_pr_df_ld_split_results_20261115/ACQUISITION_INPUTS.json`。LD2 configuration1fa33556…、m0/m3/m6、prefix2、元κ・scalar/normal law・cohort・seedを維持。
- LD2モデル/全52価格: `3ac3e09b73602ed705a00fe5e94858ca1846f48c` / `docs/research/d_pr_df_ld_split_results_20261115/FROZEN_LD2_MODELS.json` と `AXIS_PRICES.json`。LD1価格26件は85dab58dのまま。LD2はoffset0のfull C model、Sの再加算禁止。
- 4配分・実費用・予算残額: `4ef70274261cf18bd9356a15d780eb5b6b61c60c` / `docs/research/d_pr_df_ld_split_results_20261115/ALLOCATION_INPUTS.json`。A/Bは各LDのV0費用×3/4から独立に生成した。
- 科学call前のQ3入力: `4d3aa8a666e153188addfb3a4f2b8ef10733a3dd` / `docs/research/d_pr_df_ld_split_results_20261115/Q3_CALL_INPUTS.json`。source `a6d732fe35ba819f44cdbafd522aa857dd3faa0f`。新Q3API_RETURNは存在しない。
- STOP原本: [q3_calls/COMPLETION](q3_calls/COMPLETION.json) と各4 `WORKER_PERSISTENCE_ERROR.json`。前のPicklingErrorも[q3](q3/COMPLETION.json)に保全。新Q3のclosure/MSEを補完しない。

## 受け取れること・保留

12 path/24 receipt有効、再利用path0・cache hit0。元P2 fresh範囲の入力/preparation/request/lowering/compiler/fingerprint検査済み。full-wrapper statevector/新数値作用gateは未実施。X/Yは同pathの別軸で、価格が一致しても独立path数を水増ししない。eventは全pathでorder0のみで、order2/productは未観測。旧Pauli LD2成果は混ぜていない。

LD2は各軸3点meanで3係数を決めた未検証モデル。m7..12外挿。較正残差0を性能検証としない。LD1は保存E_I[S]＋残差モデル、LD2はfull-wrapper直接3点モデルで根拠が違う。モデル参照・指定finite cohort価格・対象law期待価格を区別し、正式E[C]/CI/実PCG64移送/G/winnerはnullのまま。

新4配分は `HELD_BEFORE_SCIENCE_SOURCE_GATE`。kernel bytesは710aa69bと同じだが、Python3.12のAST空field追加でacceleratorの固定AST hashが不一致。[Q3_SOURCE_STOP_ANALYSIS](Q3_SOURCE_STOP_ANALYSIS.json)。guardを緩めず新規投入停止、科学評価のretry0。actual constructorをpreflightに移す再発修正と人工再現4件を保存した。

## 次に一つ選ぶ研究対象

**現在のV0とモデル寄与に基づく暫定推奨はLD2の長round価格確認。LD3はまだ加えない。** V0のm0..6はLD1約198,979,534／LD2約66,223,445 RZ、m7..12はLD1約261,115,850,941／LD2約63,847,852,670 RZ。モデル差をほぼ長roundが担う。m12の差だけで約−130,823,719,406 RZであり、未検証外挿の検査が今の比較を直接左右する。最初のlong pointはm9等一つとし、元prefix2条件・同path X/Yを保つ独立normal pathの取得案を次回固定する。path数・seed・件数は今回追加採択していない。m9だけでm12を検証済みにしない。

その研究判断の前に、今回の未完了作業としてQ3 SOURCE/runtimeの接続を技術的に解決し、四つの**未開始**callの再開を別指示で確定する。保存価格・signal・列・予算・target/loss/閾値を再利用し、再fitや新配分は要らない。元AST契約と合うruntimeまたは空fieldだけの明示互換処理を、実constructorを含む人工検証で固定する。固定hashをただ置換する案は採らない。現STOPを越えるcallは実施していない。

## ロードマップ・資源・STOP

統合ロードマップ本体は統合担当が更新する。LD2の価格欠測は3較正cell/26モデル参照について解消。LD選択は新配分Q3と長round価格を残す。全L_Dの最適性・単調性、物理的winnerを認定しない。PF参照615,304,502／435,481,090 RZはPR予算に使用していない。

取得wall262.287853秒、path CPU合計1053.660353秒、単一worker最大RSS1,223,216 KiB（tree peakではない）。元rawイベントはprivate保全＋可逆gzip公開。旧SOURCE・raw・m4独立holdout・旧STOPは無変更。新科学callを停止し、公開remote照合後mandatory STOP。
