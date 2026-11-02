# H8証拠回収：固定公開資料のレビュー用索引

2026-10-20 JST。これは既存公開資料だけから作成した索引・取得receiptである。private原本の回収結果・hash・保存先はこの公開packetに含めない。既存の結果・SOURCE・監査資料を変更せず、詳細科学レビューと最終evidence freezeは開始していない。

|役割|固定commit|取得先|
|---|---|---|
|H8独立確認結果|`e6eea5387a6e84d1e39a9adb3b61975acc74e200`|[結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md)|
|H8科学SOURCE|`745a674546f2cb6258526fbfde1d9a00601689f8`|[確認adapter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/h8_pr_confirmation.py)|
|H8公開証拠整理|`1923fce8f7461513d1bc47b5061e87cd70917e87`|[公開証拠14項目](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1923fce8f7461513d1bc47b5061e87cd70917e87/artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/h8_public_evidence_status.csv)|

## 二条件と公開監査の対応

linear H8、隣接1 Å、STO-3G、tol-only 1e-8 DF28、16 spin modes、sector4900、T=.8、P=0。二次partial PF、r4/K4、元DF順序と無条件finite-RTE law、ordinary ancilla、full-Fock scalar一回適用、reference不回転。通常signed bias・allowance倍率1。以下は保存値の文字列転記であり再集計ではない。

|scope|L_D / q / δ|n / 軸数|PR平均G（exact）|paired SE（保存値）|固定PF / G（保存値）|SOURCEとPR実行監査|
|---|---|---|---|---|---|---|
|H8_CONFIRM_P8_Q2|8 / 2 / .4|48 / 96|9840910801/8|3175066.353534034|B0I_p11_q1_2nd / 1267321444|SOURCE 745a6745…、公開報告のローカル完了記述のみ|
|H8_CONFIRM_P7_Q16|7 / 16 / .05|48 / 96|546493203592571/6|144211830223.71762|B0I_p15_q4_4th / 94193016652860|SOURCE 745a6745…、公開報告のローカル完了記述のみ|

結果表・192軸projection、SOURCE閉包11ファイル、過去監査・選択済みPF receipt等、計18公開ファイルをGitHubから全文取得し、固定Git blob・SHA256・bytesを照合した。[取得receipt](../../artifacts/resource_applicability/track_a_h8_evidence_recovery/2026-10-20/public/public_read_receipt.json)に固定URLを記録し、[二条件対応表](../../artifacts/resource_applicability/track_a_h8_evidence_recovery/2026-10-20/public/condition_map.csv)に主評価の保存tokenを収録した。

## GPTレビューへ渡す資料

1. 結果報告、`comparisons.csv`、`paired_metrics.csv`を同じ結果commitから取得する。副評価も元表の別行に保持する。p8副評価の固定PF不適格・比較nullを維持する。
2. SOURCE commitの確認adapter、並列adapter、準備・resume・launcher、契約、native adapter、pool、pure integration、固定design、既存testを併読する。SOURCE中の検査機能は実行時PASSの証明にならない。
3. 公開証拠14項目、過去技術監査、compiler auditを参照する。選択済みPFのnative監査を後日のPR実行監査に代用しない。
4. 原本の実行receipt・native exit/reap、grant/承認receipt、registry/除外元、予約台帳、one-shot marker、旧STOP、input/provider/runtime閉包、原結果と公開projection対応を別途提示する必要がある。private原本は明示承認を得た安全な共有経路で扱う。

[資料一覧](../../artifacts/resource_applicability/track_a_h8_evidence_recovery/2026-10-20/public/review_materials.csv)は18固定資料の役割・取得先を示す。原profile file bytes、元回路からのfingerprint再導出、詳細PR実行監査は、指定された固定公開資料だけでは未確認である。この索引はprivate原本での確認状況を更新しない。

## 科学的な限界とSTOP

保存された結論は二条件の標本点費用の再現である。探索n8、旧確認n32、条件付きpilotを合算しない。Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17の設定一致から旧compiler outputの完全同一性を推定しない。compiled RZを角度分類済み非Clifford数やT数へ読み替えない。

母平均優位性、未観測eventの費用・分散、rigorous bias、最終energy/RPE総費用は追加認定しない。CI・p値は保存nullを維持する。新規sampling・compile・統計解析・費用再計算0、科学的採否判断なし。

**MANDATORY STOP。** 次はユーザーがprivate証拠の共有範囲とレビュー開始を判断する。Codex/GPT/Claudeは、この索引の作成から科学レビューや次stageへ自動進行しない。
