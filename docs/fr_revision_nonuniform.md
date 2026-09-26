# FR-R1b：非一様4×4の正scalar・位相境界検証

最終更新：2026-09-27 JST  
状態：`COMPLETE_MANDATORY_STOP`  
判定：`MECHANISM_ONLY_NO_PRACTICAL_GO`

## 1. 結論

結果を見る前に固定した
[FR-R1b事前登録](research/fr_revision_nonuniform_preregistration.md)に従い、非一様spectrumを持つ
4×4系の20 matrix conditions、61 state rows、2 semantic controlsを実行した。610 method recordのうち
227が適用可能で、位相上界・物理半径下界のsoundness違反は0だった。負時間、K=4、controlled
relative phaseも数値精度内で意味論を満たした。

非一様条件では正scalar除去後にもHermitian radial widthが正となり、同じ情報を使う比較で
`OPT-SCALAR-FR-I1`が`OPT-SCALAR-NORM-I1`より厳しい主条件を8件確認した。したがって、旧involution
toyの正scalarだけでは説明できないFR固有の改善機構は存在する。

一方、事前固定した位相予算$10^{-2},10^{-3},10^{-4}$ radでは、FRだけが認証する主条件は0件だった。
PF/RTE設定や研究判断を変えるdecision relevanceは確認できないため、R5は不通過である。事前規則どおり
`MECHANISM_ONLY_NO_PRACTICAL_GO`とし、FR-R2、H4/H12、compile、Monte Carlo、長RPE、最終総costへ
進まず必須停止する。

## 2. 固定scopeと完全性

- 事前登録SHA-256：`1bc2a72fa8dec98e2bdbe3504e65366d8daa93f7a7797837622ba7545607515e`
- matrix conditions：20（primary 18、負時間control 1、K4 control 1）
- state rows：61
- semantic controls：2
- method records：610、適用可能227
- 主`fixed_supplied_superposition`：18行、利用可能半径certificateは全て0.8
- 主状態の最小true reference radius：0.8662989773364418
- semantic residual最大値：$3.3423\times10^{-16}$
- spectral実装とmatrix実装のlocal error差：最大$2.1684\times10^{-19}$
- optimizer失敗、境界不確定、適用可能methodの未定義位相：全て0

expected artifactを`--expected-only`で結果計算前に保存し、そのsource hashと完全性を検査した後にだけ
`--run`を実行した。状態は同じ$(\nu,H_D,\sigma)$について全$q$で再利用している。

## 3. 主結果

主条件のうち$q=4,8$では、可換・非可換の両方、$\nu=0,0.5$の両方で、最適正scalarを使うFR境界が
norm境界より厳しかった。代表値を示す。単位はradである。

| $\nu$ | $q$ | `OPT-SCALAR-NORM-I1` | `OPT-SCALAR-FR-I1` | 判定 |
|---:|---:|---:|---:|---|
| 0.0 | 4 | 1.687474e-4 | 1.437545e-4 | FRが厳しい |
| 0.0 | 8 | 2.089763e-5 | 1.520264e-5 | FRが厳しい |
| 0.5 | 4 | 1.586573e-4 | 1.376963e-4 | FRが厳しい |
| 0.5 | 8 | 1.960570e-5 | 1.442742e-5 | FRが厳しい |

各行は可換・非可換の2条件で現れ、strict gain witnessは合計8件である。共通scalar比較でも同じ8条件で
FR側が厳しく、oracle情報を使わずに機構を確認した。$q=2$ではFR境界はnorm境界より厳しくない。

非一様性のradial widthは次のとおりである。

| $\nu$ | $q=2$ | $q=4$ | $q=8$ |
|---:|---:|---:|---:|
| 0.0 | 1.010344e-3 | 6.578000e-5 | 4.152786e-6 |
| 0.5 | 9.445642e-4 | 6.162721e-5 | 3.892587e-6 |
| 1.0 | 0 | 0 | 0 |

$\nu=1$のinvolution controlではwidthが0で、非一様条件だけで正となる。負時間controlでもwidthは正、
K4 controlでは$8.617542\times10^{-8}$であり、R7 control transferを通過した。

## 4. 位相予算とgate判定

最適化比較で認証された固定予算は次のとおりである。

- $q=2$：norm、FRとも$10^{-2}$のみ。
- $q=4$：norm、FRとも$10^{-2},10^{-3}$。
- $q=8$：norm、FRとも$10^{-2},10^{-3},10^{-4}$。

従って、FR側だけが事前固定予算を通るone-sided decision witnessは0件である。bound値の厳密な減少は
存在するが、固定された離散的判断を変えていない。

| gate | 結果 | 根拠 |
|---|---|---|
| R0 completeness/semantics | 通過 | 20条件・61状態・2 controlが完全 |
| R1 input certificate | 通過 | 主18状態のtrue radius最小値が0.8663で0.8以上 |
| R2 soundness | 通過 | 適用可能227 recordで違反0 |
| R3 nonuniform mechanism | 通過 | 非一様width正、involution width 0 |
| R4 same-information gain | 通過 | common/optimized比較のstrict gain各8件 |
| R5 decision relevance | **不通過** | one-sided budget witness 0 |
| R6 oracle independence | 通過 | I1の利用可能情報だけでR4を確認 |
| R7 control transfer | 通過 | 負時間、K4、controlled意味論を確認 |

この優先規則から判定は`MECHANISM_ONLY_NO_PRACTICAL_GO`となる。

## 5. Artifactと再実行

- expected：`artifacts/fr_revision_nonuniform/2026-09-27/fr_revision_nonuniform_expected_v1.json`
  - content fingerprint：`408b5a6cf007ff7cbaec804435ea867ec33fd1d1c08af87509f2316ddd8c2fb4`
  - configuration fingerprint：`b12f477f89e570090e75e57809868308f8b29df10b3177508276ef14a0d1df1c`
  - file SHA-256：`93b462e311a6759fb2a4120abb4272fced87824ad0aa1c99b15bc32ec672b7f4`
- result：`artifacts/fr_revision_nonuniform/2026-09-27/fr_revision_nonuniform_r1b_v1.json`
  - content fingerprint：`affac0ae8132450ccb2de3512b6a463a3f9d7b1ac8a6f12cc38303ac75e891d4`
  - file SHA-256：`e2a6f9326951fe67e979022dc733704e0c036d3342ab3cb317c5f77b885aeebc`
  - runtime：0.153秒

再実行では、既存artifactを上書きしない未使用pathを指定する。

```bash
PYTHONPATH=src .venv311/bin/python scripts/run_fr_revision_nonuniform.py \
  --expected-only --expected /tmp/fr_r1b_expected_check.json

PYTHONPATH=src .venv311/bin/python scripts/run_fr_revision_nonuniform.py \
  --run --expected /tmp/fr_r1b_expected_check.json \
  --output /tmp/fr_r1b_result_check.json
```

専用testは`5 passed`、FR-1/FR-R1a/FR-R1bの関連testは`13 passed`、全suiteは
`628 passed, 2 skipped, 4 warnings`で失敗0だった。4 warningsは既存grouped-UWC経路の
complex-to-real castで今回の変更箇所ではない。実行中にSciPy bounded optimizer内部で`inf`を含む
試行点由来のRuntimeWarningが3件出たが、optimizer失敗・境界不確定は0で全recordが検証を通過した。
`py_compile`と`git diff --check`も通過している。

本結果はdirty worktree上のlocal evidenceであり、immutable CIや外部独立再現とは扱わない。

## 6. 主張範囲と次段

確認したのは固定4×4 toy、固定grid、決定論的密行列評価に限られる。FR境界が非一様spectrumで
norm境界より厳しくなり得る機構は確認したが、実用上の設定選択差、化学Hamiltonian、sampled shot、
compiled circuit、長RPE、H4/H12、最終総costは確認していない。

事前登録の強制停止に従い、FR-R2を自動開始しない。次に進む場合は、今回の連続的な境界改善を
離散的な予算・K/r選択差へ結びつけられる独立RQと比較契約を、追加計算より先に再設計する。
