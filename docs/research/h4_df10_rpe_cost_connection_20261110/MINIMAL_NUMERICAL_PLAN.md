# 最小の次回N×C検証案：保存m=3だけ（未許可・未実行）

[REPORT](REPORT.md)。今回の技術作業は実H4のN×C実行許可を含まない。本案はユーザーが具体的scopeを判断するためのもので、plan採択だけを実行許可にしない。

## 入力と計算

固定D1 carrier ffe8a71da、科学SOURCE311255aa、準備e91021ae2、主価格carrier155529e2を使用する。公開PREPARED_INPUTS.jsonと、それを作った元固定blobを参照し、N/価格cohortの同一性を確認する。H4/1Å/STO-3G/DF10、保存DF-H ground条件付き、P=0、εE=.00159360010199040 Ha、αtotal=.05/方式、βtotal/stat=.4/.2、h=.1、全schedule m0…12を維持する。

| case | candidate | N_X=N_Y | 価格SOURCE/cohort |
|---|---|---:|---|
| PF B0/S2/p7/qPF8 | 49709717a6c3d405c60d | 705 | 保存決定論RZ 44411/軸 |
| PR S2/p1/qPF8/R4/K2 | 14bd81a375400c9b0305 | 977 | 保存探索n9の各軸平均 |
| 同PR、独立確認 | 同上 | 977 | 保存確認n33の各軸平均 |

**一括許可の候補範囲**：T=.8/m3について、既存adapter `account_subset`へこの3ケースだけを渡し、G_RZ,m3=N_X C_RZ,X+N_Y C_RZ,Yを各1回計算する。Nは再計算せず、探索/確認を別結果で返す。新しい平均、pool、比、価格感度、モデル残差や成功率は算出しない。主指標RZの小計確認だけとする。

期待する成果は3ケースのRZ小計とSOURCE/価格分類/条件付き資格である。全13roundをexpected_roundsとして保持し、covered_rounds=[3]、他12roundはmissing、full_rpe_resources=nullにする。N/n/Bを二重乗算しない。保存幅は欠測null、total covariance/CIも未取得null。有限cohort平均は条件付きproxyであり、ideal lawとactual sampling/native/groundの保証不足を保持する。PF/PRの全RPE費用順位は判定できない。

## 含めない処理と終了条件

新N/prior/horizon、T1.6の追加scalar、signal/trajectory/build/compile/model fit/bootstrap/残差・感度、候補再最適化、RPE simulation、全G/H6/詳細科学レビューは含めない。既存grantを再使用せず、新たなapproval証明機構も作らない。明示されたscopeは細分化して再承認を求めず一括処理する。

**ユーザー**：この3ケースの保存RZ小計を数値評価するか判断する。**Codex**：明示許可があればSOURCEを固定し、このscopeだけ実行・公開・remote照合してSTOPする。今は未実行。

## その後の全13roundへの接続

最小小計確認の後、他roundのN、候補構成、同law/compiler/full-wrapperのC/Ĉを別列で整える。保存T1.6の候補は再利用可能だがNは未評価。δ=.1のPR q16は価格欠測、別q32/δ=.05には直接価格があるため、採択候補に応じてcompileの必要性を判断する。m0…2とm5…12のsignal/誤差資格と長q価格の不足は自動補完しない。

同条件のCとĈが揃った場合だけ、固定NのΔG_mを別途許可された数値scopeで調べる。Track D/旧DF12の保存残差・共分散・holdoutは再実行せず方法を再利用する。DF10への係数移植、未知外挿bias=0、未取得Nの仮定埋めは行わない。必要な追加signal/model検証/compileの最小集合をユーザーへ提示してから実行範囲を定める。
