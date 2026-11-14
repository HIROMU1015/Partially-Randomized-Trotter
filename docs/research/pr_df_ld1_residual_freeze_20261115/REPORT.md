# PR-DF LD1：保存残差2軸fitとm4価格予測の観測前固定

2026-11-15 JST。利用者の今回の明示許可により、**m0/m3/m6の各4path×X/Y＝24組のC−S、6セル軸平均、2残差モデル、m4予測2件をexact有理数で計算・保存した。入力保留0。両軸rank3、較正算術残差0。m4価格は未取得・未開封・未比較。**

正本：[run01/FROZEN_MODELS.json](run01/FROZEN_MODELS.json)。各pathの整数S/C・残差から平均、係数、構造/残差の予測項、最終Pまでたどれる。[RESIDUALS.csv](run01/RESIDUALS.csv)／[CELL_MEANS.csv](run01/CELL_MEANS.csv)は保存JSONからの転記。[INPUTS.json](INPUTS.json)は実行前の受入済み入力snapshotで、`science_executed=false`は準備時点を示す。実行済みの範囲はFROZEN_MODELSのcall件数を正本とする。

## 固定SOURCEと重複・観測前確認

- 科学実行SOURCE：**`d83737b6d5241962ce39c81d39c56af3506c0cdd`**、`scripts/freeze_pr_df_ld1_residual_models.py`。既存C branchを継続し、固定入力と人工testをcommitしてからfresh run01を一回だけ実行した。
- C受入索引：`c99d511e2422b8f250490b11054f754a81e536c5:docs/research/pr_df_ld1_q3_price_admission_20261115/ADMISSION.json`。採用モデル/配置は`8eef0ad7a9b04e04d766f5e05b3b32eee50a794e:…/NEXT_VALIDATION_PLAN.md`と今回の利用者指示。
- m3入力：`2592d442ae8f36f71e42820a26e847b1fa38b59d:docs/research/d_pr_df_ld1_m3_results_20261115/C_HANDOFF.json`。元科学実行SOURCE `2f423befbba5075bad4ede3057f89edd788d2661`、元価格/receipt公開 `4031dbab5871d8b51513dae5e97b7a4e45704071`。
- 構造値の原正本：C索引のselectorが指す`9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f:docs/research/h4_structural_schedule_results_20261113/run01/COVERAGE.json`。今回E[S]を再計算していない。

共通指示§10.5に従い、開始HEAD/remote `8eef0ad7…`、関連公開ref、Cの関連結果path名、同Linuxユーザーの関連job名、既知D2 private namespace名を確認。同じ入力・式のfit/予測又はm4取得開始を確認範囲で認めなかった。実行直前にも確認し、`PREOPEN_CHECK.json`とFROZEN_MODELSの`preopening`へ保存した。結果内容・cache・event列を探索していない。既知のSOURCE/server metadataに限る確認で、未列挙archive/他serverの不存在を証明したものではない。

## 入力受入・物理条件

Linear H4/1.00 Å/STO-3G/保存tol-only DF10、元H/state/snapshot、LD1、S2、fixed ΔPF=.1、finite K2、S0 exactκ、元fragment/basis順。元scalar c+s_Rを総Tで一回、新ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`、symmetric-directional control、測定付きfull Hadamard wrapper/P0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。

保存P2 fresh m0/m6の各4pathは既存C受入と原result/axis receiptを再利用。元cohort `D_P2_DIRECT_PILOT_V1`、元role DEVELOPMENTを保持し、今回のモデル用途だけをCALIBRATIONと明記した。P1/replayは参照のままでpoolしていない。

m3はcohort `D_PR_DF_LD1_M3_CAL_NORMAL_20261115_V1`の4path。D2の8行から元physical ID・request/law、cohort/index/seed、event digest、axis fingerprint、S/C/STRUCTURAL receipt、scalar binary64、lowering/profile、元gateを照合した。**全24入力でfull-wrapper statevector検証=falseを保持**。D2の元記録を新しい作用検証PASSへ昇格させていない。入力保留0、補完/再compile0。

## 残差集計・exact fit

`r[m,a,i]=C[m,a,i]−S[m,a,i]`、`rbar[m,a]=sum_i r[m,a,i]/4`。X/Yは同pathの別軸として各々4pathを使い、8独立標本とは扱わない。

| m | physical ID | q | R | X平均残差 | Y平均残差 |
|---|---|---:|---:|---:|---:|
| 0 | 53e00d8b07d1547d19a7 | 1 | 2 | 375/2 | 375/2 |
| 3 | 045843b2429bece6f29b | 8 | 16 | 6915/4 | 6915/4 |
| 6 | 2a8c6718220eec59d419 | 64 | 128 | 297555/4 | 297555/4 |

`rhat_a(q,R)=θ0,a+θq,a*q+θL,a*qR`。設計行列は行順m0/m3/m6で`[[1,1,2],[1,8,128],[1,64,8192]]`。既存`price_model_validation._solve`のexact Gauss-JordanをX/Y各一回呼び、3非零pivotでrank3を確認した。signed残差を正のcompiled価格recordへリラベルせず、同じソルバを残差の3×3有理数系へ直接再利用した。平均と係数はすべて分子・分母で保存した。

| 軸 | θ0 | θq | θL |
|---|---:|---:|---:|
| X | 61225/588 | 6495/98 | 10055/1176 |
| Y | 61225/588 | 6495/98 | 10055/1176 |

3点・3係数、lack-of-fit自由度0。較正点の6残差がexact0であることは算術/実装の確認であり、予測性能・誤差・価格資格の判定ではない。独立検証は未実施のm4。

## m4の2予測

m4 T1.6/q16/R32、physical **`3df0cfd3356c358aeda8`**。X/Yとも各項を別保存した。

| 軸 | 保存E_I[S]のexact binary64値 | 残差予測 | 最終P（exact） | 表示用P |
|---|---:|---:|---:|---:|
| X | 4241507055112647/34359738368 | 1086275/196 | 217164876899444503/1683627180032 | 128986.32160079344 |
| Y | 4241507055112647/34359738368 | 1086275/196 | 217164876899444503/1683627180032 | 128986.32160079344 |

構造項の元保存値は123444.10221303834、元数値区間は`[123444.10221287991,123444.10221321041]`。元float値・binary64 hex・exact比・原selector・law・kernel SOURCEを保全し、表示丸め値から再構成していない。この構造数値区間は全compiled価格のCIではない。

予測は`E_I[S]+rhat_Q`という混合モデル。ideal IのE[S]と、元PCG64/CDF sampler Qの有限cohort残差を組み合わせている。I-targetなら`E_Q[C−S]−E_I[C−S]`、Q-targetなら`E_I[S]−E_Q[S]`のlaw systematicが残る。有限cohort/係数不確かさ、law距離、rare、共有誤差、CIは未評価のnull。exact算術はこれらを0にする根拠ではない。全law E[C]を認定しない。

## D2に引き継ぐ比較規則

取得前D2 INPUTSの`future_holdout`をそのまま保持。cohortは`D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1`、index0～3、各16個の元outer seedは科学SOURCE2f423bef…の`INPUTS.json#future_holdout.cell.seeds`参照。seed再生成なし。今回のm4 sampling/compile/価格開封は0。

観測Oは軸別4pathのcompiled RZ平均。後続でP−O、絶対誤差、符号付き/絶対相対誤差（分母O）を報告する。O=0なら相対誤差null、欠測ならO/errorsはnullとしてn4を維持する。**数値許容差・モデルPASS/FAILを追加していない**。決定論PF2%や旧pilot2%/95%を用いない。D2は固定Pを読むだけでよく、追加fit不要。

## 検証・実行範囲・ロードマップ

人工新5＋既存exact fit回帰3＝8 test method PASS。前段受入22 testの証拠はc99d511e2から再利用し、再実行件数へ加算しない。作者自身の技術確認で、独立科学レビューではない。

実科学処理は24残差・6平均・2fit・2予測と6較正算術確認。数値段階wallは約0.0035秒（SOURCE/metadata受入・テスト・公開時間は含まない）。小さいexact系なのでprocess並列化を追加せず、既存保存reader cacheとexact helperを利用した。実測していない速度倍率は主張しない。新sampling/build/compile/E[S]再計算/CI/Q3再評価/N/G/winnerなし。

ロードマップ差分は「m3入力受入・C残差fit待ち」から**「2係数組・m4予測2件の事前固定済み、D2の独立m4取得・記述的比較待ち」**。適格E[C]・法則差・有限標本/rare/長q-R移送・価格精度は未解消。本体は統合担当が更新する。

**mandatory STOP：係数/予測の公開・remote照合とD2への引渡しで終了。m4取得/開封/判定、他round予測、全law E[C]、CI、G/winnerへ自動進行しない。**
