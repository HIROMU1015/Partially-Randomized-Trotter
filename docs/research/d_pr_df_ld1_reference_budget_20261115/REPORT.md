# PR-DF LD1：固定shot参照総額とPF 2基準の順位反転境界

2026-11-15 JST。明示承認された保存数値の有限再解析を、一回実行して完了。

**参照総額 T_ref＝261,314,830,475.559717693 RZ**（表示用丸め）。B0I p7 S2に対する符号付き境界は **-260,721,275,123.559717693 RZ**、Full PF S2には **-260,476,136,895.559717693 RZ**。両方とも負である。参照値の大小はB0I p7 S2 < Full PF S2 < T_refだが、未知のΔを制限していないため、**対象lawの費用順位は未判定**。

| PF基準 | T_PF (RZ) | T_ref (RZ) | 符号付き境界 D_PF＝T_PF−T_ref (RZ) |
|---|---:|---:|---:|
| B0I p7 S2（16h/7） | 593,555,352 | 261,314,830,475.559717693 | -260,721,275,123.559717693 |
| Full PF S2（16h/7） | 838,693,580 | 261,314,830,475.559717693 | -260,476,136,895.559717693 |

表の小数は9桁の表示用で、判定は次のexact値を使う。

```text
T_ref = 329967563350580055747055/1262720385024
D_B0I = -329218068907969559898607/1262720385024
D_Full = -328908527870325298801135/1262720385024
Delta = T_LD1−T_ref  （値も上界・下界も未評価）
Delta < D_PF: LD1が低費用
Delta = D_PF: 同額
Delta > D_PF: PFが低費用
```

[26軸元結果・exact総額・2境界](run01/RESULT.json)／[26行exact CSV](run01/AXIS_TERMS.csv)／[検査](run01/CHECKS.json)／[統合担当HANDOFF](HANDOFF.md)。境界は誤差の許容幅でも、実証されたΔの上界でもない。新path/build/compile・Q3・fit・CI・正式E[C]/G/winnerは0。

## 1. 実行SOURCE・入力・重複照合

- **実行SOURCE：`68a808872842e3325e8a20a9118f19117483b9ad`**。同commitの [evaluate.py](evaluate.py) と [INPUTS.json](INPUTS.json) を数値計算前に固定した。
- **元結果の公開commit：`1fb47bf9291c9cc2328bbfe42a8b3d40f13befa5`**。本ディレクトリの `run01/RESULT.json`、CSV、CHECKS/EXECUTION、資源ログをそのまま保存した。SOURCEと結果公開SHAを区別する。最終REPORT/HANDOFFの公開SHAは最終チャットで示す。
- 計画・26軸join：`44f4e40f24a027346e686d7044d102e707292374:docs/research/d_pr_df_ld1_expectation_construction_20261115/{REPORT.md,AXIS_JOIN.json}`。今回の最新指示はPF 2基準へのD_PFに限定し、旧計画のF0加重予算K_IJや構造数値幅の伝播は計算していない。
- 凍結正本：`3cac53da1a4a8b2228245593c74c1ff9ec1f5212:docs/research/pr_df_ld1_residual_freeze_20261115/run01/FROZEN_MODELS.json`。係数とm4予測の正本を直接読む。既存exact helperは `d83737b6d5241962ce39c81d39c56af3506c0cdd:src/trottertracks/price_model_validation.py` の `_exact`/`predict`。固定Git blobからその関数ASTを無変更で使用し、C branchのmergeやfit/validate、H4 moduleのimportを避けた。
- Q3：`a82b0f0ec7f0f11d9981ddb2343fcd5378a87503:docs/research/pr_df_fullh_q3_fixed_results_20261115/C_RESULT_JOIN.json#candidates[0]`。各軸Nは **[64,60,56,52,47,43,38,34,30,25,21,17,12]**。USER_SUPPLIEDのQ3適合済み固定配分であり、十分N・最適Nではない。
- 保存E_I[S]：8eef0ad7のC HANDOFFが指定するc99d511eのADMISSION → `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f:docs/research/h4_structural_schedule_results_20261113/run01/COVERAGE.json` の各元行。値を再計算せず、その保存binary64値をexact有理数へ変換した。
- PF正本：`1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/RESULT.json#rows[label=...]` の2 totalを再利用。593,555,352は当該PF6候補中の条件付き最小、838,693,580はFull PF3候補中の条件付き最小。PFのK/Uや価格表を再集計・再ランキングしない。

§10.5では44f4e40fまでの旧ID/receipt/26軸の照合を再利用し、現D2/C/B2/統合remote、関連公開履歴、同Linuxユーザーの関連job名と指定出力namespaceを差分確認した。同等の26軸参照総額/2差額の完成結果・進行中jobは確認範囲で0、run01は未使用だった。C/B2/統合refは前回から同じで、他担当を待つ必要はなかった。PF正本・m4保存予測は再利用、24参照価格と26加重項・総額・2境界が新規。アクセス不能なarchive/他serverまでの不存在は主張しない。[PRECHECK](PRECHECK.json)。

## 2. 定義・数値表現・モデル適用範囲

対象は保存H4直線1.00 Å/STO-3G/tol-only DF10、PR-DF LD1/S2/Δ=.1/K2/元exact κ、元H/state/snapshot・fragment/basis順、元physical/canonical対応である。Cは準備費P=0、scalar c+s_Rを総Tで一回、新ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`、symmetric-directional/P0、X/Y別の測定付きfull controlled Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17のcompiled RZ数。今回回路を生成していない。

```text
s_saved[m,a] = 保存E_I[S]のbinary64値のexact比
rhat[m,a] = theta0[a] + thetaq[a]*q[m] + thetaL[a]*q[m]*R[m]
C_ref[m,a] = s_saved[m,a] + rhat[m,a]
weighted_term[m,a] = N[m,a] * C_ref[m,a]
T_ref = 26 weighted_termのexact和
```

SはI＝保存SOURCE ideal finite-RTE lawの構造期待値。rhatは元PCG64/CDF samplerの有限較正cohort由来であり、同じIの母平均と認定された値ではない。従ってC_refはモデル参照価格であり、law Iを目標とするE[C]の認定値ではない。解析的実数重み→保存値、実PCG64 joint-law接続は今回も評価していない。

元JSONのfloatをPythonのbinary64として読み、`Fraction.from_float` とhexを保存した。凍結係数/残差/予測は元の分子・分母から読む。表示小数や整数RZへ丸めてからの合算はしていない。原構造数値区間は元locatorへ残すが、その伝播・新CIは今回未実施。exact算術はモデル誤差を0にしない。

m0/m3/m6の6軸は凍結済み較正残差平均を再利用した。m4は残差予測と**最終C_refの2件を観測前JSONから直接再利用**し、構造項/残差を最終予測へ二重加算しない。観測Cや約−0.544502%で予測を差し替えず、m4は独立holdoutのまま。その他のm1/m2/m5/m7〜m12、計18軸は固定predictを初めて評価した。較正点以外への適用はm1/m2/m4/m5/m7〜m12、長round m7〜m12の移送は未検証のまま。

各軸は別々に扱った。X/Yの参照値は結果として一致したが、同一pathの両応答を独立標本として数えていない。S側のcosine/sine表記とX/Yの固定対応を明示して照合した。normalization B・n・q/Rを価格に再度掛けず、残差を二重加算しない。全26行にphysical ID、m/axis、N、exact C_ref、exact加重項、元S/model/Q3/finite-evidenceへの固定locatorを保存した。

## 3. Δについて次に必要な片側の証拠

- **B0I p7 S2**：LD1の低費用を示すにはshot加重したΔの上限を **D_B0I ≈ -260,721,275,123.559717693 RZ（上記exact比）より小さく**、逆にPFの低費用を示すにはΔの下限を同じ境界より大きく押さえる必要があり、境界と等しければ同額である。
- **Full PF S2**：LD1の低費用を示すにはshot加重したΔの上限を **D_Full ≈ -260,476,136,895.559717693 RZ（上記exact比）より小さく**、逆にPFの低費用を示すにはΔの下限を同じ境界より大きく押さえる必要があり、境界と等しければ同額である。

負の境界を「正の誤差をこれだけ許せる」という予算へ読み替えない。現状、参照総額が両基準より大きいため、LD1側の低費用を主張するには、真の期待総費用が参照値より大幅に小さいことを支える片側根拠が必要になる。一方、PF側を主張するにも、未評価Δの下側を制限する証拠が要る。今回その証拠は得ていない。

必要なのは、**同じ対象I・同じ26軸・固定Nで加重した総差**の片側評価である。全26軸一律の2%等の許容差は設けない。有限cohortの母平均誤差、共有係数/モデル誤差、長roundへの適用、law/保存数値の差を対象範囲に応じて含め、未知項を0としない。m4の記述的差を全体上界にしない。Δの証拠を取得・検証する方法と有限scopeは別指示で決め、今回はsamplingや新しい科学条件へ進まない。

F0/dcf905fcのb_IJは、同じf=C−Sについての **|E_I[f]−E_J[f]|** の条件付き上界。基準lawのE_J[f]そのもの、凍結モデルからの偏差、又はΔ全体の上界ではない。I/Jの仮定、解析実数重みとの差、実運用PCG64とのjoint lawを分けたまま保つ。b_IJをΔへ代入せず、実PCG64へ移送もしない。

## 4. 件数・技術確認・資源

| 数量 | 予定 | 再利用 | 新規 | 有効／保留・欠測 |
|---|---:|---:|---:|---:|
| 軸別C_ref | 26 | m4の2 | 24 | 26／0 |
| 残差参照 | 26 | 較正6＋m4の2 | 18 | 26／0 |
| 加重項 | 26 | 0 | 26 | 26／0 |
| T_ref | 1 | 0 | 1 | 1／0 |
| PF基準総額 | 2 | 2 | 0 | 2／0 |
| D_PF | 2 | 0 | 2 | 2／0 |
| 新path／H4 build／compile | 0 | — | 0 | — |

人工6検査で、Fractionを丸めない和、負の境界、固定predict関数、m4最終予測の再利用/二重残差防止、軸対応/入替え拒否、入力不一致拒否を確認した。[TESTS](TESTS.txt)。準備reader初版は元cosine/sineをX/Yと直接比較してmetadata段階で拒否したため、元の既定対応とrole文字列を明示して修正し、人工検査を追加してからSOURCE固定・科学実行へ進んだ。実入力の変更やguard緩和ではない。

実行後は26項のexact積/和、2差額、m4元予測の一致、CSV、142入力locator、SOURCE/INPUTS固定、旧証拠不変を確認した。科学predictの再実行は0、科学実行は1回、retry0、STOP/欠測0。作者の技術検査であり独立科学レビューではない。保存H4 P0数値再検査やfull-wrapper statevector等のsuiteは起動していない。

[EXECUTION](run01/EXECUTION.json)／[GNU time](RESOURCES.txt)：内部wall **0.075138185秒**、起動を含むGNU wall **0.15秒**、user0.12秒/system0.03秒、max RSS **18,432 KiB**、exit0。小さい26軸exact算術なのでprocess並列化は加えず、PFの既存Git blob cacheパターンで同一入力を再読しない。性能倍率や他条件への速度保証は主張しない。

## 5. 引渡し・ロードマップ差分・mandatory STOP

ロードマップの「PR-DF LD1参照総額・PF差額が未数値化」は本scopeで解消した。参照上は両PF基準より大きく、必要な順位反転の境界は負であることを固定した。次は未知Δのどちら側をどこまで抑えるかという課題へ移る。これは期待残差の基準値/誤差や長round移送を認定した結果ではなく、正式E[C]/CI/G・十分N・winnerは未認定のまま。

ロードマップ本体は統合担当が更新する。元S/凍結モデル/予測・m4独立holdout受入9edfa5d3・F0/range・旧raw/STOP/SCOPE_DEVIATION、他担当worktree/jobを保全した。REPORT/HANDOFFと結果を指定D2 branchへcommit/pushし、remote SHA/公開成果物一致確認で**mandatory STOP**。新path/build/compile、Q3、再fit、F0再実行、実PCG64移送、正式E[C]/CI/G/winnerへ進まない。
