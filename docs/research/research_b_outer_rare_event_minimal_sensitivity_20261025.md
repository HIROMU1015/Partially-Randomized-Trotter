# 研究B：高次数RTE eventの最小条件付き費用感度

## 範囲とSOURCE

作業起点・入力commitは `bef63edbd8cc564444b44e4e8cf23baa90f05fe5`。[B1結果](research_b_outer_b1_expected_native_cost_20261025.md)の科学SOURCE `11e2532a62d54a65f8fad5489be6b135f826011b` と原本・`INCONCLUSIVE_RARE_EVENT_COVERAGE`を維持する。本解析は別の **CONDITIONAL_SENSITIVITY_ONLY** である。新しいsample、Hamiltonian／operator／signal、回路build／compile、shots／gamma／u_numの計算は行わない。

Synthetic 2-mode one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix、n_D=1、T=±0.18。分子geometry／AO basis／分子DF rank探索は非該当。固定比較はA=B4対B3、B=D1対R4だけ。B4=SABA2/A2/q2/r2K2、B3=Yoshida4/A2/q2/r2K2、D1=Yoshida4/A3/q2/r2K2、R4=Yoshida4/A3/q2/r2K4。費用は既存Qiskit1.3.0 phase-preserving native v5のfull prepared X/Y Hadamard wrapper、主指標RZ count、補助CX count。

実装 [`rare_sensitivity.py`](../../src/trottertracks/b2d_evaluator/rare_sensitivity.py)、入口 [`run_b2d_outer_rare_sensitivity.py`](../../scripts/run_b2d_outer_rare_sensitivity.py)、人工test [`test_b2d_rare_sensitivity.py`](../../tests/management/test_b2d_rare_sensitivity.py)。本実装commitを解析SOURCEとして固定し、一回実行後に本書の結果節へ記録する。科学的INCONCLUSIVEを理由に再実行しない。技術的中断の復旧は原因確認・人工検証・新SOURCE・fresh outputで旧attemptを保全する場合だけ、今回の許可範囲に含まれる。Resume・使用済みoutput上書きは行わない。

## 既存手法の適合性と再利用

旧PR研究の次の固定文書をgit履歴から読んだ。本B1起点には文書が含まれないため、各commitの原本を参照し、旧研究branchは変更していない。

- `pr_rare_rte_stage_b_results_20261021.md`、commit `3d5ab5bb1cf0d78a3157ad7f742cfa5c7b58efbb`。
- `h8_rare_rte_saved_integration_results_20261021.md`、commit `c30a70f354069f584420fdcaea78f6acf6c21714`。
- `h8_rare_rte_closure_policy_20261021.md`、commit `862d732fba0a5f353c522a9219dec78ae2f21a4b`。

段階B SOURCE `e16c3be8a039f6bc468e112f0ed086bde544fef2` の `scripts/rare_rte_stage_b_core.py` の `dec`／`finite_law`（scalar Decimal式）を必要部分だけ移植した。旧処理の共通tau・R乗・PR対固定PF・shot加重G会計は、signed outer列の異なるtau・両候補の未知費用・full-wrapper単価という本条件に適合しないため、全体driverを流用しない。旧の数値・確率・grant・承認・STOPは流用しない。

B1の保存config／probability／normalizationと統計を直接読み、既存 `FollowupLedger`／`registered_scope`、SOURCE/fresh-output確認、`Evidence`の失敗保存・STOPを再利用する。新しいsimulation／compiler／承認frameworkは作らない。

## 確率・統計・逆転境界

各caseの実際のappend順R occurrenceと各r=2 short-stepに対して

`p_S0 = product_occurrence p0_occurrence^r`、`p_rare = 1-p_S0`。

保存binary64のp0を厳密にDecimalへliftし、50桁で積と補数を評価する。保存tau・Kから既存paired finite-Taylor scalar式を照合し、signed duration、lambda/r、order support、normalization、probability sum、全256 provenanceのlaw一致を確認する。元のp0をcanonical値へ置換・再正規化せず、両経路の微小な差を保存する。50桁演算は桁落ち回避のscalar計算で、入力精度・PRNGの有限bit law・数値包含を保証しない。

対象は8 candidate-time、4 comparison-time、X/Y×RZ/CXの16境界。S0 mean／variance／SE／n=256はB1保存値を維持し、今回の条件付きmean推定 `mu0_hat` として扱う。X/Yは同じtrajectoryなので各n=256、独立512本へpoolしない。

正側候補P、負側候補Nについて

`Delta_hat(u) = a + p_P*u_P - p_N*u_N`、`a=(1-p_P)*mu0_hat_P-(1-p_N)*mu0_hat_N`。

真の差は `Delta_true=Delta_hat(u)+pS0_P*(mu0_P-mu0_hat_P)-pS0_N*(mu0_N-mu0_hat_N)`。

比較AはP=B3,N=B4、比較BはP=R4,N=D1。`u_i=E[C_i|rare]`は非負の未知変数。`Delta>0`ならN側が低費用。境界は

`u_N = a/p_N + (p_P/p_N)*u_P`。

同じu_Pに対してu_Nが線の下／上ならN側／P側が低費用となる。ただし経験S0平均固定下の感度であり、真の母平均順位ではない。倍率座標 `lambda_i=u_i/mu0_hat_i` も併記する。軸切片は他方を座標上0とした交点で、未知実費用0の仮定を採用したものではない。p_P,p_Nとも正なので、費用の非負性だけではDeltaの両符号が可能であり無条件順位を保証できない。

有限標本感度は `w=(1-p_P)*SE_P+(1-p_N)*SE_N` とし、個々のmeanを±kSEの範囲で動かす矩形の対角端点 `a±k*w`（k=1,2）を用いる。80行の境界表はshift=-2,-1,0,+1,+2。正shiftではPのmeanを上げNを下げる。倍率の分母は中央mu0_hatを固定する。候補間共分散を0と仮定せず、合成SEやp値・信頼区間は計算しない。これらは矩形感度でcoverage保証ではない。一般にはS0差の分散に `-2*pS0_P*pS0_N*Cov(mean_P,mean_N)` が含まれる。

## 実行・保存契約

標準ライブラリのみ。小規模のCSV／18MB provenanceの一回stream走査と44 occurrenceのscalar照合であり、起動時にCPU affinity／空きRAM／負荷を記録する。Process起動・IPCを追加するほどの数値仕事量がないため逐次走査を選び、CPU/RAM/workerの人工的な上限は設けない。共有環境の変更なし。

Clean detached SOURCEから次の専用fresh outputへ一回実行する。

```bash
/usr/bin/python3.11 -I -B -S \
  /tmp/prt-research-b-outer-rare-science-20261025/scripts/run_b2d_outer_rare_sensitivity.py \
  --execute --source-commit a8cf99156af692f0facaa1885ab9ed52b28c6f4c \
  --output /tmp/prt-research-b-outer-rare-science-20261025/artifacts/research_b_outer_rare_sensitivity/20261025/saved-once
```

入力9fileは固定B1結果commitのblobと照合する。結果JSONに8caseの全occurrence law／8case×X/Y×2metricの既存統計／B0対象代表値／16境界・80 SE cornerを保存し、確率CSV8行・境界CSV80行を添える。旧B1/A0/B0/PRのSOURCE・raw・STOP・各契約は変更しない。実行記録・部分結果・失敗・STOPには既存Evidenceを使う。

## 実行結果

状態 **CONDITIONAL_SENSITIVITY_ONLY／MANDATORY STOP**。固定SOURCE `a8cf99156af692f0facaa1885ab9ed52b28c6f4c` をclean detached checkoutで一回実行した。Read-only preflight、人工8件、8確率／16境界／80 SE cornerの全scopeをPASS。失敗・保存不足・retry/resume・新sampling/build/compileは0件。解析本体wall0.118秒、whole-process wall0.37秒、最大RSS33,512KiB、exit0。資源に人工capを設定せず、元データの小規模走査として逐次実行した。

[原本・STOP](../../artifacts/research_b_outer_rare_sensitivity/20261025/saved-once/)、[確率8行](../../artifacts/research_b_outer_rare_sensitivity/20261025/saved-once/PROBABILITIES.csv)、[境界80行](../../artifacts/research_b_outer_rare_sensitivity/20261025/saved-once/BOUNDARIES.csv)、[全数値・既存統計・B0代表値](../../artifacts/research_b_outer_rare_sensitivity/20261025/saved-once/result.json)、[環境・SOURCE・入力blob](../../artifacts/research_b_outer_rare_sensitivity/20261025/saved-once/provenance.json)、[実行・人工test記録](../../artifacts/research_b_outer_rare_sensitivity/20261025/execution_logs/)。公開用rawと実行原本が一致し、保存p0のexact Fraction積、全80境界の代数照合もPASSした。これは技術照合であり独立科学レビューや保証付き包含ではない。

### S0／rare確率

下表は表示丸め。保存signed duration・r/K・append順を全256 trajectoryで照合し、44 occurrenceのdistribution／normalizationを確認した。JSONには各occurrenceの全order確率、signed tau、normalization、照合差を保存した。元の確率を変更・再正規化していない。

| Candidate | T | Independent short-steps | p_S0 | p_rare |
|---|---|---:|---:|---:|
| B4 | +.18 | 8 | 0.99998603647309 | 1.396352691308E-5 |
| B4 | −.18 | 8 | 0.99998603647309 | 1.396352691308E-5 |
| B3 | +.18 | 12 | 0.99981710424360 | 1.828957564026E-4 |
| B3 | −.18 | 12 | 0.99981710424360 | 1.828957564026E-4 |
| D1 | +.18 | 12 | 0.99981710424360 | 1.828957564026E-4 |
| D1 | −.18 | 12 | 0.99981710424360 | 1.828957564026E-4 |
| R4 | +.18 | 12 | 0.99981710375408 | 1.828962459168E-4 |
| R4 | −.18 | 12 | 0.99981710375408 | 1.828962459168E-4 |

最高次数K4だけの確率ではなく、order2も含む「少なくとも一つ非0 event」の確率である。保存p0とscalar canonical照合から得たp_rareの差は最大約1.03e-15。これは保存binary64とscalar式の整合診断であり、PRNGの厳密bit-lawや費用biasの上界ではない。対象8caseのB1抽出はすべてS0のままで、観測数0を確率0へ置換しない。

### 保存S0統計

各値は経験mean ± SE、各axis n=256。X/Yは別々に記録し、512本へpoolしない。Variance／min／maxは原本result.jsonのS0_statisticsでB1値をそのまま保持する。

| Candidate | T | Axis | RZ mean ± SE | CX mean ± SE |
|---|---|---|---:|---:|
| B4 | +.18 | X | 2087.82422 ± 1.42353 | 1416.09375 ± 1.13304 |
| B4 | +.18 | Y | 2087.82422 ± 1.42353 | 1416.09375 ± 1.13304 |
| B4 | −.18 | X | 2089.28125 ± 1.35072 | 1417.17187 ± 1.06539 |
| B4 | −.18 | Y | 2089.28125 ± 1.35072 | 1417.17187 ± 1.06539 |
| B3 | +.18 | X | 4096.89453 ± 1.59814 | 2788.84375 ± 1.26439 |
| B3 | +.18 | Y | 4096.89453 ± 1.59814 | 2788.84375 ± 1.26439 |
| B3 | −.18 | X | 4098.81641 ± 1.70473 | 2790.48437 ± 1.35475 |
| B3 | −.18 | Y | 4098.81641 ± 1.70473 | 2790.48437 ± 1.35475 |
| D1 | +.18 | X | 27423.63281 ± 1.65513 | 18772.04688 ± 1.29393 |
| D1 | +.18 | Y | 27423.63281 ± 1.65513 | 18772.04688 ± 1.29393 |
| D1 | −.18 | X | 27423.23828 ± 1.67576 | 18771.53125 ± 1.31642 |
| D1 | −.18 | Y | 27423.23828 ± 1.67576 | 18771.53125 ± 1.31642 |
| R4 | +.18 | X | 27424.37109 ± 1.70766 | 18772.65625 ± 1.34921 |
| R4 | +.18 | Y | 27424.37109 ± 1.70766 | 18772.65625 ± 1.34921 |
| R4 | −.18 | X | 27423.45312 ± 1.44781 | 18771.62500 ± 1.15785 |
| R4 | −.18 | Y | 27423.45312 ± 1.44781 | 18771.62500 ± 1.15785 |

### 比較A：B4対B3

未知rare条件付き費用を `u_B3,u_B4 >=0` とする。中央mu0_hat固定下で

`u_B4* = alpha + 13.09810605*u_B3`。

u_B4がこの線より小さければB4低費用、大きければB3低費用、等号は同費用となる。RZのalphaは+Tで約1.43828287e8、−Tで約1.43861553e8。これはu_B3=0軸上の数学的切片であり、実u_B3=0という仮定ではない。
倍率座標では+TのRZが `lambda_B4*=68889.08+25.70214*lambda_B3`、−Tでは `68856.96+25.69627*lambda_B3`（lambda_i=u_i/中央mu0_hat_i）。未知費用を等しいと仮定していない。

中央S0加重差aはRZで約2008.35／2008.81、SE矩形幅wは約3.021／3.055。±2SEでもaの下端は正で、S0推定誤差のこの感度範囲では大きな差が維持された。しかしu_B4の有限上界がないため、非負性だけから無条件のB4順位保証は得られない。極端なrare費用が逆転させる領域を示したのであって、その費用が実在・起こりやすいと推定したものではない。

### 比較B：D1対R4

中央mu0_hat固定下のRZ境界は

- +T：`u_D1*=4035.81150+1.00000267647*u_R4`。
- −T：`u_D1*=1174.39046+1.00000267647*u_R4`。

u_D1が線より小さければD1低費用、大きければR4低費用となる。倍率では+Tが `lambda_D1*=0.14716546+1.00002960*lambda_R4`、−Tが `0.04282465+1.00001051*lambda_R4`。これらは未知rare費用の推定値ではなく、条件付きの等費用線である。
中央S0加重差aはRZで+T約0.73813、−T約0.21479。一方wは約3.36218／3.12301で、±1SE矩形だけですでに符号が変わる。高次数を無視しても費用母平均順位は未確定であり、rare境界を算出したことによってD1またはR4の優位を認定しない。

### SEを含む全境界表

表のIは `u_negative*=I+slope*u_positive` の切片。各meanを±kSE動かす矩形の端点であり、信頼区間・coverage保証ではない。負の切片もそのまま保存し、物理的な負費用や0補完とは解釈しない。RZ／CXとも単位はfull-wrapper gate count。

| Pair | T | Axis | Metric | a | I central | I at ±1SE corners | I at ±2SE corners |
|---|---|---|---|---:|---:|---|---|
| A | +.18 | X | rz_count | 2.008350E+3 | 1.438283E+8 | 1.436119E+8 … 1.440447E+8 | 1.433955E+8 … 1.442610E+8 |
| A | +.18 | X | cx_count | 1.372260E+3 | 9.827458E+7 | 9.810290E+7 … 9.844625E+7 | 9.793123E+7 … 9.861793E+7 |
| A | +.18 | Y | rz_count | 2.008350E+3 | 1.438283E+8 | 1.436119E+8 … 1.440447E+8 | 1.433955E+8 … 1.442610E+8 |
| A | +.18 | Y | cx_count | 1.372260E+3 | 9.827458E+7 | 9.810290E+7 … 9.844625E+7 | 9.793123E+7 … 9.861793E+7 |
| A | −.18 | X | rz_count | 2.008815E+3 | 1.438616E+8 | 1.436428E+8 … 1.440803E+8 | 1.434240E+8 … 1.442991E+8 |
| A | −.18 | X | cx_count | 1.372822E+3 | 9.831484E+7 | 9.814154E+7 … 9.848814E+7 | 9.796824E+7 … 9.866144E+7 |
| A | −.18 | Y | rz_count | 2.008815E+3 | 1.438616E+8 | 1.436428E+8 … 1.440803E+8 | 1.434240E+8 … 1.442991E+8 |
| A | −.18 | Y | cx_count | 1.372822E+3 | 9.831484E+7 | 9.814154E+7 … 9.848814E+7 | 9.796824E+7 … 9.866144E+7 |
| B | +.18 | X | rz_count | 7.381328E-1 | 4.035811E+3 | -1.434721E+4 … 2.241883E+4 | -3.273023E+4 … 4.080186E+4 |
| B | +.18 | X | cx_count | 6.092544E-1 | 3.331156E+3 | -1.111783E+4 … 1.778014E+4 | -2.556681E+4 … 3.222913E+4 |
| B | +.18 | Y | rz_count | 7.381328E-1 | 4.035811E+3 | -1.434721E+4 … 2.241883E+4 | -3.273023E+4 … 4.080186E+4 |
| B | +.18 | Y | cx_count | 6.092544E-1 | 3.331156E+3 | -1.111783E+4 … 1.778014E+4 | -2.556681E+4 … 3.222913E+4 |
| B | −.18 | X | rz_count | 2.147910E-1 | 1.174390E+3 | -1.590094E+4 … 1.824972E+4 | -3.297628E+4 … 3.532506E+4 |
| B | −.18 | X | cx_count | 9.372366E-2 | 5.124431E+2 | -1.301341E+4 … 1.403830E+4 | -2.653926E+4 … 2.756415E+4 |
| B | −.18 | Y | rz_count | 2.147910E-1 | 1.174390E+3 | -1.590094E+4 … 1.824972E+4 | -3.297628E+4 … 3.532506E+4 |
| B | −.18 | Y | cx_count | 9.372366E-2 | 5.124431E+2 | -1.301341E+4 … 1.403830E+4 | -2.653926E+4 … 2.756415E+4 |

AのCX切片は中央で約9.82746e7／9.83148e7。BのCX切片は約3331.16／512.44で、こちらも±1SE感度が符号を跨ぐ。全精度の係数・mean shift・倍率境界は80行CSV／JSONを参照する。候補間共分散は未評価のままであり、独立性を暗黙に採用した合成SEではない。

### B0の代表費用との記述的照合

B0 science SOURCE b190fe28bの代表0/1はorder0、代表2は最初にK最大次数、代表3は指定stageにorder2を入れた決定論的列である。下表は原CSVから対象だけを引用したRZ/CXの代表0,1,2,3順。条件付きrare平均・不偏標本・全support上界として用いていない。

| Candidate | T | Axis | RZ representatives 0/1/2/3 | CX representatives 0/1/2/3 |
|---|---|---|---|---|
| B4 | +.18 | X | 2020 / 2061 / 2061 / 2066 | 1368 / 1392 / 1395 / 1395 |
| B4 | +.18 | Y | 2020 / 2061 / 2061 / 2066 | 1368 / 1392 / 1395 / 1395 |
| B4 | −.18 | X | 2020 / 2061 / 2061 / 2066 | 1368 / 1392 / 1395 / 1395 |
| B4 | −.18 | Y | 2020 / 2061 / 2061 / 2066 | 1368 / 1392 / 1395 / 1395 |
| B3 | +.18 | X | 3994 / 4070 / 4057 / 4057 | 2716 / 2764 / 2755 / 2755 |
| B3 | +.18 | Y | 3994 / 4070 / 4057 / 4057 | 2716 / 2764 / 2755 / 2755 |
| B3 | −.18 | X | 3994 / 4070 / 4057 / 4057 | 2716 / 2764 / 2755 / 2755 |
| B3 | −.18 | Y | 3994 / 4070 / 4057 / 4057 | 2716 / 2764 / 2755 / 2755 |
| D1 | +.18 | X | 27322 / 27398 / 27385 / 27385 | 18700 / 18748 / 18739 / 18739 |
| D1 | +.18 | Y | 27322 / 27398 / 27385 / 27385 | 18700 / 18748 / 18739 / 18739 |
| D1 | −.18 | X | 27322 / 27398 / 27385 / 27385 | 18700 / 18748 / 18739 / 18739 |
| D1 | −.18 | Y | 27322 / 27398 / 27385 / 27385 | 18700 / 18748 / 18739 / 18739 |
| R4 | +.18 | X | 27322 / 27398 / 27398 / 27385 | 18700 / 18748 / 18753 / 18739 |
| R4 | +.18 | Y | 27322 / 27398 / 27398 / 27385 | 18700 / 18748 / 18753 / 18739 |
| R4 | −.18 | X | 27322 / 27398 / 27398 / 27385 | 18700 / 18748 / 18753 / 18739 |
| R4 | −.18 | Y | 27322 / 27398 / 27398 / 27385 | 18700 / 18748 / 18753 / 18739 |

B0の少数代表と、さまざまなcomponent列を含むB1のS0標本は同じ母集団の無作為標本としてpoolできない。たとえばB4/+T/Xの4代表RZ最大2066に対しB1 S0経験平均は約2087.82であり、代表最大値はS0の費用上界ですらない。高次数代表で極端な増大が見えないことも、未観測rare全体の平均や上界を保証しない。

### 結論と次の費用評価へ渡すもの

B4対B3は、観測S0差が大きく、±2SE感度でも差が残り、逆転には上記affine境界を超える未知rare寄与が必要となる。D1対R4はS0推定誤差だけでも順位が未確定で、高次数の影響と区別して扱う。両比較ともp_rareが正で未知費用上界がなく、非負条件だけでは無条件母平均順位を保証できない。

今回取得したのは8caseのprobability係数、S0条件付き経験費用・variance・SE・n、unknown muRareを残すRZ/CX費用関数、±1/2SE感度である。次のshot加重評価ではこれらを単価入力として参照できるが、必要shot数・normalizationやsignalによるshot負担・gamma／certified u_num・その誤差配分は別課題で、今回は新規数値計算していない。Gate単価の差を同精度総費用優位に読み替えない。

旧B1のINCONCLUSIVE_RARE_EVENT_COVERAGEを維持し、新しい解析はCONDITIONAL_SENSITIVITY_ONLYとする。追加高次数sampling／compileが現在の限定主張のために自動的に必須になったとは扱わず、主張を強めるかどうかは研究Bのユーザーが判断する。Codexは結果公開・remote照合後mandatory STOP。Shots／追加MC／A1／他科学実行を開始しない。
