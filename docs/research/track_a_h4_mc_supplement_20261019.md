# H4 PR単独MC費用変動：正式採用と論文補足資料への統合

2026-10-19。ユーザーの「H4のPR単独のMC費用変動検証を正式採用し、論文の補足資料へ統合してください」に基づき、完成commit `295b93d50662a0f8af3ec469df43099a3f04af60` を補足資料として正式採用した。新規科学計算・bootstrap・sampling・compile・統計解析・図の再描画は行っていない。

[補足資料packet](../../artifacts/resource_applicability/track_a_h4_mc_supplement/2026-10-19/README.md)に、掲載用caption、Supplementary Methods、更新版paper manifest・CEM・証拠台帳を保存した。既存主表6条件・Figure 2の原図34ファイルとcaption、既存紙面配置・原データは変更していない。

## 採用した3主張

| CEM ID | 採用範囲 | 図・表 |
|---|---|---|
| CE_H4_MC_MEAN | 固定保存経験分布のもとで、MC標本数nによるPR平均費用推定値の揺れを保存平均・分位幅・累積平均で示す | H4_MC_S1、H4_MC_D1/D2 |
| CE_H4_MC_SE | 保存bootstrap SEがn増加とともに概ね1/√nで減少し、既存保存基準列と概ね対応する | H4_MC_S2、H4_MC_D1/D2 |
| CE_H4_MC_EVENTS | 観測S2_SINGLE群はS0_ONLY群より平均費用が高いが、群内変動と範囲の重なりがありS2件数だけではfull PR費用は決まらない | H4_MC_S3/S4、H4_MC_D3/D4 |

これはユーザーによる上記scopeの採用記録であり、新しい詳細科学レビューの合格や一般的な母平均保証ではない。既存CEMの14行を保持した追加版へ3行を加え、旧主張の条件・未解決事項を上書きしていない。

## 費用指標と条件

linear H4、隣接1 Å、STO-3G、tol-only 1e-8 DF rank10、T=.8、P=0、通常signed bias／allowance倍率1。
ε=.05はL_D3/q1/r4/K2/δ.8/N1108、ε=.0005はL_D2/q8/r2/K4/δ.1/N78。
費用は**決定論的prefix・確率的RTE tail・full cosine/sine Hadamard wrapperを含む保存shot×compiled RZ**であり、tail単体やサーバーCPU費用ではない。保存shotはnで変更しない。

図の揺れとSEは平均費用推定値についてのものであり、n本を実行する合計費用の変動と区別する。S0/S2の層別平均もfull PR費用で、S2 event単独の加法的費用と解釈しない。

## SOURCE・結果・表示の来歴

| 役割 | 固定SOURCE | 完成結果 |
|---|---|---|
| 元のH4独立P取得 | 3c664ca4a9f003880b9f6d56f2a0e55ed5b56603 | 821cc0f6ced57168293ea166644024867b064ffb |
| 保存標本のMC費用解析 | 78352b861b3047f6af2d432535e23cc88b439bbe | defa5f6dbc72910c0a46ae2fef304a584ae408e0 |
| PR単独の表示整理 | 295b93d50662a0f8af3ec469df43099a3f04af60の描画SOURCE | 295b93d50662a0f8af3ec469df43099a3f04af60 |

指定結果の12画像、CSV4表、元報告書・表示来歴等を固定Git objectとbyte同一で取り込んだ。対応する元MCの保存3CSVとresult JSONも同じcommit bytesで保持し、新たにraw trajectoryや個別bootstrapを解析していない。SOURCE登録表は固定commitのコードblob/SHAとURLで結合し、コードを実行していない。

本補足は主表のH4_DF10_P_1/2と**同じH4 P cohortの再解析**である。新たな独立実験・確認標本として証拠数を加算しない。H4旧DF12や探索n8/FP n32と混合せず、他分子へ拡張しない。

## 補足図表とmanifestへの接続

[caption集](../../artifacts/resource_applicability/track_a_h4_mc_supplement/2026-10-19/supplementary_captions.md)は英語でそのまま掲載文へ移せる形にした。H4_MC_S1〜S4、H4_MC_D1〜D4は補足用固有IDであり、既存Figure 2や既存のSupplementary Figure番号を改番しない。

- 新しい[paper manifest](../../artifacts/resource_applicability/track_a_h4_mc_supplement/2026-10-19/paper_manifest.json)が既存paper manifest・掲載manifest・採用済み共通プロトコルv2と補足packetを結ぶ。主表とFigure 2のbindingは従前のままコピーして参照する。
- [CEM追加版](../../artifacts/resource_applicability/track_a_h4_mc_supplement/2026-10-19/claim_evidence_matrix.csv)は既存14行＋今回の3主張。元CEMとcurrent overlayは保持する。
- [証拠台帳v4](../../artifacts/resource_applicability/track_a_h4_mc_supplement/2026-10-19/evidence_ledger_v4.csv)は既存316行を保持し、保存mean/SE/分位点/層別値と16図表assetへの出典を追加した。すべて保存値文字列かファイル参照の転記であり、費用や統計を再計算しない。
- [対応表](../../artifacts/resource_applicability/track_a_h4_mc_supplement/2026-10-19/evidence_linkage.csv)は3主張から図表・証拠台帳・caption・Methods・SOURCEを追跡できる。

## 構造検証と保全

`python3 scripts/integrate_h4_mc_supplement.py`は原本SHA、固定commitとのbyte一致、既存行の保存、3主張だけの追加、16図表asset、caption/Methodsのscope、出典文字列一致、相対リンクを検査する。既存`check_h468_paper_display.check`も再利用して主表6行とFigure 2原本34ファイルの不変性を確認する。元の科学runner・統計moduleはimportしない。

これはCodex自身による文書・metadataの構造確認であり、独立科学レビューや科学再実行ではない。[構造監査receipt](../../artifacts/resource_applicability/track_a_h4_mc_supplement/2026-10-19/structural_audit.json)と[保全一覧](../../artifacts/resource_applicability/track_a_h4_mc_supplement/2026-10-19/preservation.json)を保持する。元grant・科学実行ledger・marker・STOPと、既存論文packetの版を変更しない。

## 残る限界と終了

保存標本に条件付けた経験的評価であり、真の母平均・分散、未観測希少event、厳密CI、必要nや逐次停止規則を保証しない。複数S2は未観測、K2のS4は支持外、K4のS4は未観測のまま未知を維持する。SEの約1/√nは保存列の記述であり、新しい回帰や理論的保証ではない。

今回の正式採用・文書統合は最終evidence freezeや詳細科学レビューを開始する承認ではない。新規科学計算・sampling・compile・bootstrapは0。専用commitへ保存し、公開は承認済み範囲のみとする。文書統合完了後はmandatory STOP。

次の作業なし。**Codexはmandatory STOPを維持する。**
