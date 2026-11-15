# 統合担当・C/B向け：PR-DF LD1期待価格の構成と次の有限作業

**保存証拠の結論**：26軸の固定shot配分と価格証拠は同一physical/m/axisへ結合済み。E_I[S]と条件付きI↔IID格子の期待残差差上界があっても、基準lawの期待残差Dが未評価なので、対象lawのE[C]は未確定である。

**推奨1件**：新compileの前に、凍結モデルの26軸参照価格と、PF条件付き最小総額 **593,555,352 RZ** を跨ぐために必要な未評価残差平均誤差の境界を評価する。0 path/0 compile・再fitなし。モデルを真とする採択を前提にせず、未知誤差はnullのまま残す。今回は計画まで。

[REPORT](REPORT.md)／[26軸join](AXIS_JOIN.json)／[次の有限scope](NEXT_FINITE_PLAN.json)／[metadata照合](CHECKS.json)。最終公開commitにこれらのpathを添えて参照する。

## 入力・law・式

正本Q3は `a82b0f0ec7f0f11d9981ddb2343fcd5378a87503:docs/research/pr_df_fullh_q3_fixed_results_20261115/C_RESULT_JOIN.json`。PR-DF LD1のUSER_SUPPLIED配分は各軸 **[64,60,56,52,47,43,38,34,30,25,21,17,12]**。十分N・最適Nとは別。元/total/旧価格configurationとcanonical IDを保持し、X/Yを別記録にした。元26軸表950577f4とC索引c99d511eを再利用し、rawへ新しいIDを代入していない。

対象は保存H4直線1.00 Å/STO-3G/tol-only DF10、LD1/S2/Δ=.1/K2/元exact κ、元H/state/snapshot/basis順、scalar総T一回、new ZZ・symmetric-directional・P0・測定付きfull wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、準備費P=0。

Iは保存SOURCE ideal finite-RTE law、JはF0が仮定する独立一様53-bit格子のpath law、Q_ηは明示したseed測度ηを元PCG64実装へ渡したlaw。Iは既存Q3の対象を維持する。解析的実数law→保存値の誤差はF0に含まない。

```text
f=C−S,  D_L=E_L[f]
E_I[C]=s_I+D_I=s_I+D_J+Δ_IJ,   |Δ_IJ|≤b_IJ
      =s_I+dhat+e_J+Δ_IJ,     e_J=D_J−dhat（未評価）.
```

- s_Iの保存値/数値enclosure：C受入8eef0ad7が指定するc99d511e → 9d95d3eaのCOVERAGE。構造数値幅をcompiled価格のCIにしない。
- dhat：3cac53daの凍結2軸モデル。m0/m3/m6の元PCG64有限cohortが根拠。母平均ではなく参照関数。
- b_IJ：dcf905fcのRANGE_EVIDENCE、各軸のexact `conditional_RZ_shift_upper`。F0受入259a0b2d、元raw/ec78fde5へ戻れる。表を再作成しない。**これは期待残差の差の上界で、D_I/D_Jの値ではない。**
- e_J：有限標本、取得law→J、モデル/長round移送の不足を含む。推定値/区間は未取得。仮の0・2%・95%を入れない。

運用lawへ分けるなら `e_J=(D_Qη−dhat)+(D_J−D_Qη)`。ηの確率測度、hash/index/outerからのseed生成と再利用規則、PCG64の履歴条件付き又はjoint lawの根拠が必要。SOURCEのhash・再現性・周辺CDFだけはIID証明にならない。固定seedの指定cohort平均を全law平均へ格上げしない。Q_η対象へ変える場合は構造期待値と信号/Q3のlaw接続も必要。

**保存Iでの条件付き比較には、実PCG64証明を先行必須にしない。** ただし有限PCG64 cohortからD_I又はD_Jを推定できるとの根拠も、現状のままではない。この二点は両立する。Iの再採択は不要で、運用Q_ηに主張を変えるときだけηと新しいlaw契約のユーザー判断が要る。

## 有限価格の扱い

m0/m6のP2 fresh各4 path、m3のfresh4 pathは凍結モデルの較正。P1/replayは開発・再現参照のまま。m4のfresh4 pathは独立holdoutのまま維持し、fitへ戻さない。m4の−0.544502%は各軸4 path平均との記述的予測差であり、標本誤差・law差・モデル誤差の分離や長round上界ではない。X/Yは同じpathの別応答で、独立標本数を二倍にしない。

既存full-wrapper statevector=false等のgate範囲を維持した。構造/有限価格/作用gate/consumer受入/母平均資格を別々に扱う。最新rangeのCによる正式受入完了は今回確認範囲で未同定で、CのF0受入を新range受入へ読み替えない。

## 次の実行指示に使う有限案

担当案はC（D2はSOURCE/identity支援）。Cの別worktreeを今回変更していない。NEXT_FINITE_PLANに固定入力と件数・出力先・STOPを具体化した。

1. m0〜m12×X/Yの保存構造値/区間と凍結係数、元N、F0/range条件を確認。既存 `price_model_validation.predict` とGit readerを再利用し、再fitするfreeze caller/mainとtolerance必須validateは呼ばない。
2. dhatは較正6＋m4予測2の保存8値を再利用、新評価は18以下。full-price参照はm4保存2＋新24＝26。m4正本をexactに再利用し、係数や観測前予測を変更しない。
3. 次の参照/誤差境界だけをexact算術で評価し、正式E[C]/G・CI・順位判定は行わない。

```text
T_ref=ΣN(s_I_saved+dhat), K=ΣN b_IJ
v_lo/hi=ΣN(s_I_lo/hi−s_I_saved), E_res=ΣN e_J（未評価）
T_I=T_ref+E_res+ξ_S+Δ, ξ_S∈[v_lo,v_hi], |Δ|≤K
M=593555352−T_ref
PR側が低い十分条件：upper(E_res)<M−v_hi−K
PR側が高い十分条件：lower(E_res)>M−v_lo+K.
```

E_resの上下限がなければ判定はUNRESOLVED。この境界が、後続の母平均評価に必要な精度を費用差に沿って決める材料となる。別途ユーザーが全26軸でe_J=0等のモデル仮説を採択した場合だけ、追加の条件付きモデルシナリオが成立する。この計画はその仮説を採択していない。

新path/compile上限0/0、H4 build/fit/F0再実行/rare探索/Q3/CI/正式E[C]/Gも0。後続出力先は `docs/research/d_pr_df_ld1_reference_budget_20261115/run01`（未使用）。SOURCEと入力を数値評価前に固定し、§10.5の差分だけを照合する。重大なidentity/law/input/保存不一致、非有限・負の参照価格、未解決重複では停止し、partialを保全。clamp・guard緩和・retry・追加pathは行わない。公開・remote照合でSTOP。

今回の指示は計画まで。次の有限数値評価は別の実行許可が必要だが、承認証明や追加の管理帳票は不要。

## 統合ロードマップへの差分・終了

ad6ab45bの§0/§7/§8.2/§10で、PR-DF LD1の不足を「law差上界がない」ではなく、**期待残差の基準値/推定誤差と長round移送が未確定**へ具体化する。実PCG64接続は運用lawの課題として分離。初回比較の設計は保存Iのまま進められる。次案は0 compileのモデル参照/誤差予算評価で、PF総額・Full-H/Pauli進行を待つ依存はない。

比較先PF593,555,352 RZは1002f1dfの条件付き総額を転記し、再集計していない。PF2%をPRへ移さず、十分N・正式G・5方式winnerはnull。ロードマップ本体は統合担当が更新する。

今回の新規科学計算/予測/保存価格算術/path/H4 build/compile/fit/CI/正式E[C]/Gは0。26軸metadataとlocatorだけを検査した。固定SOURCE/raw・凍結正本・m4独立比較・旧STOP・他担当worktree/jobを保全。指定branchへcommit/push・remote一致確認後、**mandatory STOP**。
