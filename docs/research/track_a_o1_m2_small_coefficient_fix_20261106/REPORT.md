# O1 M2：微小係数保持の局所修正・H4 operator再検証準備

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**実装・人工テスト完了、実H4 operator再計算は未実行、mandatory STOP。** 基点`a21be7b589bfe43ec7495e9872a961c84858ff91`の旧SOURCE/raw/assessment/input/STOPはそのまま保持した。新branch上の修正は旧commitの変更や旧runのretryを意味しない。

## 最小修正と再利用

前段の保存update005完全残差は、OpenFermionの`EQ_TOLERANCE=1e-8`によって全1056非zero fermion項が失われた。これは前段報告の引用であり、今回は再計算していない。

新しい[局所mapper](../../../src/trottertracks/o1_m2_cutoff_free_pauli.py)は、既存`spin_lift_unthresholded`と`InteractionOperator`のtensor/index/1/2規約をそのまま使う。raw monomialを直接列挙し、**単位係数のmonomialだけ**を既存OpenFermion JWに渡す。quartic以下の単位JW係数はexact binary fractionで最小1/16であり、実入力の微小係数をOpenFermionの加算・乗算へ流さない。

その単位JW辞書に実係数をPython数値として掛け、既存`PauliHamiltonian.from_terms`の`math.fsum`集約とexact-zero除去だけを使う。unit JWはmonomial別にcacheする。nuclear scalarもidentityへ一度だけ加える。これはbinary64の係数をcutoffなしで集約する処理であり、厳密symbolic演算の保証ではない。

[adapter](../../../src/trottertracks/o1_m2_residual_adapter.py)の`pauli_parts()`と`residual_rte()`だけをこのmapperへ接続した。原g由来κ、全8frame、coreのnormal ordering、`R_g=g_original−g_fit`、`R_h=h_original−h_core`、残差核scalar0、sector規約は変更していない。旧LD2 tailは使わない。新fit・rank探索・optimizerはない。共有ライブラリ・global EQ_TOLERANCE・旧venvは変更していない。

## synthetic/unit test結果

**22 passed in 3.06s**：既存17件＋[新しい人工回帰5件](../../../tests/management/test_o1_m2_small_coefficients.py)。既存[独立CAR・Pauli label参照とfixture](../../../tests/management/test_o1_m2_residual_adapter.py)を再利用した。

- 正負J・非自明U・非zero一体補正の人工2空間軌道系全体を`1e-10,1e-16,1e-22`に縮小。
- 残差自身のmax-entry scaleに対し`128 eps * max|R|`でPauli/CARを照合。ゼロmapperを受理できるunit-floor許容値は使わない。
- nonidentity項、電子identity、符号、一体補正、核scalar一度、core＋Rを照合。残差RTE設定も同じ保持済み辞書を使用。
- tiny signed一体/二体packetを旧mapperへ通すnegative controlは差を検出し、新mapperは独立CARへ一致。
- global EQ_TOLERANCEが不変であること、ゼロ/正負scalarを一度だけ保持することを確認。
- `np.load`は禁止した。旧H4入力のdecode・operator再構成はしていない。

```bash
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME=/tmp/o1-m2-tiny-prep-cache-20261106 \
MPLCONFIGDIR=/tmp/o1-m2-tiny-prep-cache-20261106/mpl \
/tmp/track-a-o1-m2-runtime-20261106/bin/python -m pytest -q \
  tests/management/test_o1_m2_residual_adapter.py \
  tests/management/test_o1_m2_small_coefficients.py
```

Python3.12.3、NumPy2.5.4、SciPy1.18.1、OpenFermion1.8.1、Qiskit2.6.2の既存専用runtimeをread-only再利用した。人工テストのPASSは実H4の修正完了を意味しない。

## 別途実行許可を求めるH4 operator-only再検証

[新runner](../../../scripts/validate_o1_m2_retained_residual.py)を用意した。`--help`は入力をロードせず、core/input/outputは必須引数、出力はfresh pathだけを受け付ける。元の[固定driver](../../../scripts/validate_o1_m2_h4_operator.py)から独立sparse CAR、density-product全core、Pauli bit-action、operand scale helperを再利用し、旧driverは変更しない。

| 項目 | 固定条件 |
|---|---|
| 保存core/元H | commit `aced1647b81e8d28091058cdc7b6e3e1ee54896a`のupdate005 J/U全8frameと元h/g/scalar |
| core原path/hash | `artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz`、SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` |
| 元H原path/hash | `artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`、SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| core生成SOURCE | `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`。今回の新検証SOURCEと区別する |
| 系 | linear H4、1Å、STO-3G、4空間軌道/8mode、full256とNalpha=Nbeta2のsector36、途中射影なし |
| 対象 | 新mapperのphysical/core/Rと独立CAR、raw全density core＋R=H、残差自身、identity、一体補正、符号、sector leakage |
| 対象外 | state生成・元state解析、時間発展、sampling、回路、shots/費用、新fit/rank探索 |

**残差判定は総H許容値から独立**。結果を見る前にSOURCEで式を固定した：`L=max(operation_lengths(4,256,8))=33288`、`gamma(L)=L*2^-53/(1−L*2^-53)`、guard8。`S_R=|c_R|+2sum|h_R|+2sum|g_R|`、`tau_R=8 gamma(L) S_R`。`max(1,S_R)`は使わない。core/Hの再構成は別の総operand scaleで検査する。前段と同様、これは工程スケールの数値検査であり、rigorous rounding enclosureではない。

独立CARで得る残差サイズに対し`tau_R < 0.01 max|R_CAR|`を満たさなければ判定分解能不足としてNOT_QUALIFIEDとする。ゼロmapperと反符号のnegative controlも記録する。残差電子identityには同じtau_R、残差nonidentity非空、元g由来κ/一体収縮/符号、核scalar0を別検査する。人工atolの移送、後付けの閾値変更、結果に合わせた切捨てはしない。

実行許可後のcommand（この準備では実行しない）：

```bash
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME=/tmp/o1-m2-retained-run-cache-20261106 \
MPLCONFIGDIR=/tmp/o1-m2-retained-run-cache-20261106/mpl \
/tmp/track-a-o1-m2-runtime-20261106/bin/python scripts/validate_o1_m2_retained_residual.py \
  --core ../track-a-o1-m2-h4-operator-correctness-20261106/artifacts/track_a_o1_m2_h4_operator_correctness/2026-11-06/inputs/update005_candidate.npz \
  --input ../track-a-o1-m2-h4-operator-correctness-20261106/artifacts/track_a_o1_m2_h4_operator_correctness/2026-11-06/inputs/input.npz \
  --output artifacts/track_a_o1_m2_retained_residual/2026-11-06/run01/result.json
```

## 未解決・次の担当・STOP

保存update005に対する新mapperの数値correctnessは**未検証**。原H/元coreを再計算していない。科学SOURCEはこの新branchのadapter/mapperとrunnerの組であり、旧core生成SOURCEを新実行SOURCEとして扱わない。新driverは各SOURCE hash・科学input binding・runtime・判定値を実行時に記録する。

主agentによる技術確認であり、sub-agent委任・重要科学レビューではない。既存結果のstatus、原raw、grant、marker、STOPを変更していない。今回は直接関連の実装・テスト・報告に限定した。

**次の担当：ユーザー**が新SOURCEによる同じH4 operator-only再検証を許可する。許可後に**Codex**が実行し、結果を別保存・公開してmandatory STOPする。今回は実装・人工テスト・独立branch公開・remote内容照合までで**mandatory STOP**。PF/RTE信号・回路・RPE総費用へは進まない。
