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
  --execute --source-commit <ANALYSIS_SOURCE> \
  --output /tmp/prt-research-b-outer-rare-science-20261025/artifacts/research_b_outer_rare_sensitivity/20261025/saved-once
```

入力9fileは固定B1結果commitのblobと照合する。結果JSONに8caseの全occurrence law／8case×X/Y×2metricの既存統計／B0対象代表値／16境界・80 SE cornerを保存し、確率CSV8行・境界CSV80行を添える。旧B1/A0/B0/PRのSOURCE・raw・STOP・各契約は変更しない。実行記録・部分結果・失敗・STOPには既存Evidenceを使う。

## 実行結果

人工8件PASS（toy signed/K4 law、normalization不一致、未知費用境界、SE矩形、scope拒否、mock失敗保存/fresh/STOP、科学import禁止・明示実行指定）。Metadata inspectでは8確率／16境界／80 cornerを確認した。人工テストと科学的結果を区別し、SOURCE固定・preflight後の一回の保存値解析結果をここへ追記する。全supportのcompiled費用上界は未取得のままで、未知rare平均・無条件期待値・最適性・保証付き同精度総資源優位は認定しない。今回完了後はmandatory STOPし、shots／追加MC／A1／他科学実行へ自動移行しない。
