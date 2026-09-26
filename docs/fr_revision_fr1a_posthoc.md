# FR-R1a：既存FR-1の正scalar事後再解析

最終更新：2026-09-26 JST  
状態：`COMPLETE_POSTHOC_SCALAR_EXPLAINS_OLD_GAIN`  
判定：`POSTHOC_SCALAR_EXPLAINS_OLD_GAIN`

## 1. 結論

完了済みFR-1の33条件・99状態を、凍結済み
[事後計画](research/fr_revision_fr1a_posthoc_plan.md)どおり再構成し、正scalarを除いた強い
norm baselineとFR境界を比較した。全状態fingerprintと旧信号・旧境界は再構成値に一致し、
9手法×99状態の891 method recordでsoundness違反は0だった。

主5行では、正scalar処理後の`SCALAR-FR-COMMON`は
`SCALAR-NORM-COMMON`より厳しくならず、最適化後にもFRだけが通る位相予算はなかった。
旧FR-1で見えた`OLD-FR`の改善は、少なくともこのinvolution toyでは、
Hermitian/anti-Hermitian分離固有の利益ではなく正scalar除去で説明できる。

これは事後説明監査であり、旧FR-1のG2不通過と`GO_FR2_MECHANISM_ONLY`を変更しない。
FR-R1b、H4/H12、回路compile、Monte Carlo、RPE総costは開始しておらず、新しい研究GOも
認めない。

## 2. 固定入力と再構成監査

- 元result：`artifacts/finite_rte_phase_amplitude/2026-09-26/finite_rte_phase_amplitude_fr1_v1.json`
- 元content fingerprint：`d96b200163f3a432652656ae97c65c837e01fe81323cd69528373dff16ce6152`
- 元file SHA-256：`6a81a0ba6e39ba0f5d79ba026a2a45c3c65709e071e9c6fa5606b3e99a89b0f7`
- 凍結事前登録：`artifacts/finite_rte_phase_amplitude/2026-09-26/fr1_preregistration_frozen.md`
- 凍結事前登録SHA-256：`bc8066d7a31a3f46f476d2c591c7f0dad5e6f49023fc261dc518b61e2ec600a3`
- FR-R1a計画SHA-256：`a0aa2e75d2e304e008646f138211ed1e4da4d98def697f8503b07c3e7b56c95f`
- 条件数33、状態数99、主行5。
- deterministic再構成の最大数値差：0。
- state fingerprint一致：99/99。

現行の旧FR-1事前登録文書には実行後status節が追記されているため、そのファイル全体のhashは
凍結時と異なる。再解析では元artifactと同じディレクトリの凍結コピーを検査し、指定hashとの
一致を確認した。元resultと旧artifactは変更していない。

## 3. 正scalar比較

各tail occurrenceで

$$
D=U_{\rm tail}^{\dagger}A_{\rm tail}-I=F+iG,
\qquad
\widehat D=\frac{D-cI}{\gamma}
$$

とした。共通比較では$F$のspectral midpointを$c$に用い、
$\Gamma_c=\gamma^{qr}$を物理半径下界へ残した。主toyは$h^2=I$であり、$F$の二つの固有値が
等しいため、midpoint除去後のHermitian spectral widthは全33条件で数値精度内の0である。

主gridの`analytic_mixture_state`（$T=0.8,K=2,r=1,\underline\rho=0.8$）は次のとおり。
位相上界の単位はradである。

| $q$ | short time | `OLD-FR` | `STRONG-NORM` | `SCALAR-NORM-COMMON` | `SCALAR-FR-COMMON` | actual phase |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8 | 2.341293e-2 | 2.115382e-2 | 1.281285e-2 | 1.807450e-2 | 1.191771e-2 |
| 2 | 0.4 | 2.276553e-3 | 2.662405e-3 | 8.381674e-4 | 1.174024e-3 | 7.783316e-4 |
| 4 | 0.2 | 2.425161e-4 | 3.331889e-4 | 5.308410e-5 | 7.432010e-5 | 4.929282e-5 |
| 8 | 0.1 | 2.766262e-5 | 4.166172e-5 | 3.329384e-6 | 4.661147e-6 | 3.091562e-6 |
| 16 | 0.05 | 3.291551e-6 | 5.208170e-6 | 2.082714e-7 | 2.915800e-7 | 1.933935e-7 |

`SCALAR-NORM-COMMON`は5行全てで`SCALAR-FR-COMMON`より厳しい。位相予算
$10^{-2},10^{-3},10^{-4}$ radと物理半径下限0.2に対して、共通scalar比較・各法最適化比較の
どちらにもFR側だけが認証する行は0だった。I2 `DENSE-ORACLE`だけが生む片側認証も0だった。

## 4. 判定

凍結分類の優先順位に従い、結果を

`POSTHOC_SCALAR_EXPLAINS_OLD_GAIN`

とする。意味は次に限定される。

- この一様involution toyでは、旧FR境界の見かけの改善は正scalar処理を入れたnorm境界で説明できる。
- FR固有の追加利益を示す証拠にはならない。
- 非一様spectrumでのFR-R1bの結果を先取りしない。
- FR-R1bの事前登録grid、位相予算、GO/STOP条件を変更しない。

## 5. Artifactと再実行

- expected：`artifacts/fr_revision_fr1a_posthoc/2026-09-26/fr_revision_fr1a_expected_v1.json`
  - content fingerprint：`83fa10248e71cad40fa0ba87877c49f88f6755c24470c232fb41ccff4205f4a9`
  - file SHA-256：`332f8d5efe53659c5f3ec1c53ea36e116ca1f685df376496089e3ee536e22d3f`
- result：`artifacts/fr_revision_fr1a_posthoc/2026-09-26/fr_revision_fr1a_posthoc_v1.json`
  - content fingerprint：`5cc5c29656b9ceb69a00674cee5987b93c7cc8d9403892c57b341a51c878a18a`
  - file SHA-256：`3fab6acdde3798dc7005101713ecaa708f5cf0b5fc24d8c77f80939dcaa6a610`

再実行コマンドは次である。

```bash
PYTHONPATH=src .venv311/bin/python scripts/run_fr_revision_fr1a_posthoc.py \
  --expected /tmp/fr1a_expected_check.json \
  --output /tmp/fr1a_result_check.json
```

runnerは異なる既存artifactを上書きしないため、再検査では未使用の出力pathを指定する。

専用testは次である。

```bash
TMPDIR=.tmp_pytest PYTHONPATH=src .venv311/bin/pytest -q \
  tests/test_fr_revision_fr1a_posthoc.py --basetemp=.pytest_tmp/fr1a
```

検証時の専用testは`4 passed`、元FR-1との関連testは`8 passed`、全suiteは
`623 passed, 2 skipped, 4 warnings`で失敗0だった。warningsは既存grouped-UWC経路の
complex-to-real castで、今回の変更箇所ではない。Blackは環境に未導入のため未実行だが、
`py_compile`と`git diff --check`は通過した。

本結果はdirty worktree上のlocal evidenceであり、immutable CIや外部再現とは扱わない。
