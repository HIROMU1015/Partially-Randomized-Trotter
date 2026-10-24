# H4/H6標準Yoshida4：条件付き独立費用方向の正式採用

2026-10-24。状態：`FORMALLY_ADOPTED_CONDITIONAL_INDEPENDENT_COST_DIRECTION`。
今回のユーザー指示を採用判断の根拠とする。過去の実行承認を採用承認へ転用せず、承認receipt原本も公開しない。

## 採用した結果

固定科学結果は `4d117041469fb8eebf2ad55405c0f7d1b70dea7d`、科学SOURCEは
`a2a53698a72da7280d217adc7931938121ed732d`、元数値SOURCEは
`c028f1c2885afdba65ef8c62c075738aadd7323d`。
原典は[結果JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d117041469fb8eebf2ad55405c0f7d1b70dea7d/artifacts/resource_applicability/track_a_h46_standard_y4_confirmation_recovery/2026-10-24/results.json)。
入力・契約SHA、候補、群ごとの元SOURCE・n・G・paired SE、差・比・近似区間の全保存tokenを
[採用記録](adoption_record.json)へ転記した。数値の再計算は行っていない。

条件はlinear H4/H6、隣接1.0 Å、STO-3G、tol-only DF1e-8、rank10/19、T=.8、ε_signal=.001、
α_axis=.025、P=0、usual signed、保存signal/reference・empirical allowance倍率1・analytic shot政策。
PRはprefix3/5、K4、ABS_TIME_PER_OUTER32、S2/q8対標準Yoshida4/q4。
symmetric-directional control、状態準備を除いたfull cosine/sine Hadamard wrapperのcompiled RZ総費用である。
compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17。

|系|既存独立S2 n|新独立Y4 n|保存G比 Y4/S2|保存判定|
|---|---:|---:|---:|---|
|H4|25|45|0.8649598948712324|REPRODUCED_APPROXIMATE|
|H6|23|32|0.8627624094690314|REPRODUCED_APPROXIMATE|

両系で、固定2比較family・α_family=.05のBonferroni正規近似区間は保存差 `G_S2−G_Y4` の正側。
元の `REPRODUCED_APPROXIMATE` は結果判定、新しい正式採用statusはユーザーの証拠採用判断として分けた。
採用する主張は、固定科学条件・固定shot政策に条件付けた独立PR回路費用方向の再現である。
同じεの保存FullはS2/q8が低費用であり、異なるqの固定構成におけるFull/PR方向の逆転を支持する。

## 探索・独立標本・否定的対照

既存独立S2の取得元は `3eca46148070ba4d4b0f5815ef45943407cc959e`。
今回の4群統計は、そこから再利用したS2単価と新Y4単価に固定shotを適用した、既に保存済みの結果である。
旧PR対PR確認の別候補・m5_best側の確認を標準Y4の独立確認へ転用していない。
旧H6 K2/S2のε=.05/.0005採用も別scopeである。

新Y4の77 trajectoryはH4 n45＋H6 n32という取得総数であり、family数・合算推定量ではない。
S2 n25/23、探索H4 S2 n48/Y4 n33・H6各n33を混ぜず、cos/sinを独立標本として倍増しない。
cohortの対応は[役割索引](evidence_roles.csv)から固定resultの群へ戻る。

既存同一q対照 `eebdbfcbd0d5f5608d50bfd91c9cd526c68f0d23` は、H4/H6とも
q4ではFull/PR双方Y4低費用、q8では双方S2低費用。**同一qでのFull/PR方向逆転は観測されなかった。**
8比較と4方向を固定出典へ結合し、独立確認の新たな標本として数えない。
H2/H4/H6の精度依存方向・C/N/G分解は、それぞれ既存batchと
`d2536c4f959bfb4edd7d2deeceb155585fb53eb0` の保存結果を引用する。

## 限界・採用しない主張

正規近似は厳密coverageや母平均順位certificateではない。未観測strataの費用・分散・tailはunknown/nullを維持する。
経験的allowance・保存reference、有限標本、探索後の候補選択に依存する。
q、signed piece構成、RTE配分、deterministic/random partition、bias、normalizationが関与し、次数単独の因果効果は未確定。
全候補最適性、全分子・任意精度への一般化、PRの一般的優位、全誤差・FT全資源保証、先行研究との差別化は採用しない。
独立外部研究者・別施設による再現、論文中心主張・新規性の最終採択、研究全体の最終GOでもない。

## 証拠体系の採用版

[paper manifest](paper_manifest.json)を今回の追加版の入口とする。
既存採用版 `f75a7c25d584b69d90de454731a7f343a421e059` と
3環境索引 `6dafed86cc82149d55b4e6395382953d5f3631d7` は固定したまま参照する。

- [CEM v3](claim_evidence_matrix_v3.csv)：旧21行のbyte prefixを保持してH4/H6の2主張を追加。CE1等を改訂・格上げしない。
- [Ledger v5](evidence_ledger_v5.csv)：旧391行を保持し、4群・2比較・政策・Full/探索・契約・成功監査/STOP・失敗監査/STOP・同一q対照の14参照を追加。
- [paper data manifest](paper_data_manifest.csv)：旧9行を保持し、公開固定資料への参照を追加。新たな数値解析表ではない。
- [source catalog](source_catalog.json)：固定commit/path/SHA/URLを一意に追跡。旧catalogも参照で再利用。
- [3環境差分](three_environment_overlay.json)：旧29 claim行・49系列は不変。新2主張を今回のCEMの同一IDへ結び、重複証拠に数えない。

旧版と今回版を連結して旧行や標本数を増やさない。paper manifestの状態は証拠採用であり、論文主張の最終採択ではない。

## 実装再利用・技術検査・保全

既存 `scripts/integrate_h468_paper_evidence.py` の保存token読取・JSON/CSV出力・pointer/literal helperを再利用した。
f75版の `adopt_h246_pf_common_results.py`、そのテスト、旧H6採用schemaと3環境索引も確認した。
前者は588候補の採用専用で、今回の独立4群・近似区間の結合には適合しないため実行せず、同じCSV prefix保持方式を最小adapterへ適用した。
新statusは別schema内で定義し、旧statusや元結果の判定は変更しない。

[検査adapter](../../../../scripts/adopt_h46_standard_y4_confirmation.py)は固定Git blobの転記・参照・保全確認だけを行う。
科学runnerをimportせず、旧SOURCEも実行しない。JSON空白やCSV改行の違いを科学的不一致として扱わず、保存token・行・provenanceを検査する。
[技術検査記録](validation_summary.json)と[保全監査](preservation_audit.json)を参照。
新規17件と既存の証拠統合11件がPASS。旧CEM/台帳/manifestの全cell保持と、転記・cohort・保証・参照の不正変更拒否を確認した。
新科学call、sampling、build/compile/simulation、G/SE/CI等の再解析、新grantはすべて0。
作者自身の技術検査であり、別担当による独立科学レビューではない。

主表SHAは `2d910a62451d6a0b3247751625e85c8aebd181b4dea6b493b94a469d4e49d86c`。
固定 `1923fce8f7461513d1bc47b5061e87cd70917e87` の主表・Figure2全34参照、科学SOURCE、固定結果/契約/STOPを照合した。
Git objectのみ確認した資料と、このworktreeの実体byteも照合した資料を監査で区別する。
private raw/runtime/grant/生ledgerは今回再監査せず、原終了監査の保存報告へ参照する。共有環境・他ユーザー変更0。

旧失敗 `4ce444eb5b7cd1dad7cf61f149aa182949ad60a1` は、親context18・旧消費grant・compile/新trajectory0とSTOPを保持した別履歴。
成功recoveryの終了監査が記録する旧停止64 file保全・grant再利用なしを引用し、今回それらを更新・再実行しない。
共通運用指示2026-10-23の§8.1/8.4/10.1/12を確認した。指定§10.3は当該正式版に存在せず、実在する§10.2と§12の実行境界・版管理を適用した。

原結果・同一q対照・旧失敗STOP・3環境入口はGitHub側のblobとも一致した（[選択した公開原典の確認](remote_source_checks.json)）。
新資料の公開は共通AGENTSの利用者指示「コミット許可にプッシュ許可を含む」を適用し、今回の公開可能な差分に限定する。
private raw、seed原本、authority/grant原本、内部絶対path、認証情報を追加していない。

## 停止・次担当

今回の証拠統合は完了。[mandatory STOP](STOP.json)。旧strict COMPAREはBLOCKEDのまま。
新科学計算、詳細GPT/Claudeレビュー、最終evidence freeze、原稿の大幅改稿へ進まない。
次担当：**ユーザー**が、論文の中心主張・新規性として採択するかを必要な時点で判断する。
**Codex**はSTOPを維持する。pushできなかった場合の公開作業だけは最終チャットの確定コマンドで再実行できる。
