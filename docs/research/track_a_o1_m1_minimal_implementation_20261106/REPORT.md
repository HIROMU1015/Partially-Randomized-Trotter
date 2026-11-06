# Track A（O1）M1：H4-A最小adapter実装

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。基点は`8f768ccf616af8ff2c35502ea8548138814f5554`、専用branchは`codex/track-a-o1-m1-minimal-implementation-20261106`。今回H4-A（T_eff/V_square）が指定され、最小adapterと人工unit/synthetic testを実装・検証した。**実H4の数値load、operator再構成、state correctness、回路、科学simulation、shot/総費用は実行していない。** M2とは独立で、merge/統合はしていない。

## 1. 実装と再利用

追加ファイルは次の3つのみ。既存core、SOURCE、raw/result、grant/marker/STOP、前回文書は変更しない。

- [`src/trotterlib/o1_m1_matched_split.py`](../../../src/trotterlib/o1_m1_matched_split.py)：明示入力と遅延loader、共通分割のPauli/DF接続。
- [`tests/test_o1_m1_matched_split.py`](../../../tests/test_o1_m1_matched_split.py)：小さい人工CAR/JW参照、人工scalar/一体補正、mock archive/dispatch。
- 本書。

| API | 実装した境界 |
|---|---|
| `MatchedAInputs` | bare spin一体/二体、保存DF一体/λ/G、核scalar、元H `state`、physical indicesを明示受領。有限値・shape等の入力検査後、独立したread-only copyを保持。raw stateの再正規化・固有状態判定・sector射影はしない |
| `load_h4_a_snapshot(path)` | default pathや探索を持たず、明示呼出し時だけ`np.load(..., allow_pickle=False)`。必要な8 fieldのみ読み、spin8/rank10/sector36を検査。`DF_state`へのfallback、再DF、新state生成はない |
| `prepare_matched_a(inputs)` | 共通T_effを保存`df_one_body`、共通V_squareを保存λ/Gのsquare和として、global collected Pauliと既存`DFHamiltonian`に接続。original-H Pauliと一体補正を別に返す。回路・時間発展・RTE lawを作らない |
| `MatchedASplit.pauli_full()` | `c_N I + T_eff + V_square`を集約。`original_pauli`との差は独立に比較できる。P側だけを`H_original−T_eff`にしてDF差を埋める経路はない |
| `pauli_tail_coefficients(..., identity_policy=...)` | P残差のfaithful identity／identity抽出を明示選択。返すscalarはV_squareからの移管分だけ。T_effの電子identityと核scalarは別の場所に保つ |
| `extract_matched_df_tail(..., identity_policy=...)` | 保存順の全fragmentを既存`extract_df_tail_from_hamiltonian`へ渡す薄い入口。既存入口はGaussian basis回路を生成し得るため、今回この実体呼出しは行わず、mockでdispatchだけ確認した |

再利用元は基点収録の`pauli_signal_adapter.PauliHamiltonian`（label順、global collect、scalar、sparse operator）、`df_hamiltonian.DFHamiltonian`、`df_rte_tail.extract_df_tail_from_hamiltonian`（既存I/Z/ZZ・RTE tail接続）、OpenFermion 1.6.1の`FermionOperator`/JW/`InteractionOperator`と`get_chemist_two_body_coefficients`。新しいDF factorizationやHamiltonian primitive、selector、混在内点探索、RTE sampling法を実装していない。

P/Sは**同じ保存square factors**を元にする。元Hはbare spin tensorから別referenceへ変換し、`df_one_body−spin_one`とSOURCEのnormal-ordering contractionも別fieldとして保持する。既存DF近似差をM1 residualへ補償するM2型処理は追加しない。

adapterに追加の係数cutoffはないが、OpenFermion内部の`EQ_TOLERANCE`等は残る。微小係数の完全保持や厳密operator同値を保証する意味ではない。Hermitian入力の`1e-12`検査もsyntactic input checkであり、次回のH4 scientific comparison toleranceを採択していない。

## 2. 許可範囲内の検証

新testファイルを指定して、**6 passed**。人工2/4 spin-orbital系とmockだけを用いた。

1. complex Hermitian G・負/正λ・非対角一体を独立のCAR行列で照合し、Qiskit little-endian・squareの符号・共通HD/HR・元Hとの差の分離を検査。
2. 人工spin-symmetric tensorで、既存normal-ordering helperと独立CARの一体補正符号・prefactorを照合。故意に補正を落とす入力が不一致になることも検査。
3. 核scalar、保持Tの電子identity、Vのidentity移管を一度ずつ加える両policyを検査。
4. 異なる`state`/`DF_state`を持つmock archiveで、元H fieldだけを読むこと、raw copy、rank10、close、missing-state時にfallbackしないことを検査。
5. complex DF weightの無断実数化と非Hermitian Gを拒否。
6. circuit-producing DFtail入口をmockし、全保存factor・順序・identity policy・zero追加cutoffのdispatchを確認。

人工行列の`assert_allclose`は全て`rtol=0`、絶対許容差`1e-14`以下。これは小toyの実装検査の精度で、H4への精度資格ではない。DFtailのbasis回路、compiled operator、実H4のDF辞書和のPASSは主張しない。

`tests/conftest.py`等がないこととpytest設定を先に確認し、既存の軽量package aliasパターンで必要coreだけをimportした。`trotterlib/__init__.py`の既存eager importは変更・使用していない。module自身のtop-levelには入力I/Oがなく、test collectionでも実H4を読まない。各testでは`np.load`と`qiskit.QuantumCircuit`を禁止し、loader/DFtail入口だけ明示mockへ差し替える。これは人工testの実行境界であり、新しい実行承認システムではない。

使用コマンド（専用worktree内）：

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MPLCONFIGDIR=.venv/mplconfig XDG_CACHE_HOME=.venv/cache \
.venv/bin/python -m pytest -q -p no:cacheprovider --noconftest \
  tests/test_o1_m1_matched_split.py
```

worktree専用`.venv`のみ使用。Python 3.12、pytest 8.4.2、NumPy 1.26.4、SciPy 1.14.1、OpenFermion 1.6.1、OpenFermion-PySCF 0.5、Qiskit 1.3.0、PySCF 2.7.0。必要なhelper importの依存として用意したもので、PySCF計算は呼んでいない。共有package/venv/OS、他jobには変更を加えていない。

## 3. 科学correctnessの準備状態と次の最小範囲

H4-Aは今回の指定に従う。保存inputは前回[M1 setup][SETUP]で固定したcommit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`の`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、rank10。元H保存`state`を用い、`DF_state`へ変更しない。loaderのshape検査だけでは科学入力pinの同一性を認定しない。

実H4 snapshotは今回一度も数値load/prepareしていない。新H/HR生成、元H ground correctness、実H4 T/V別一致、辞書和、scalar移管、sector保存、原targetへの数値DF差のallowance、mean/normalization、no-prep資源、全RPEは未検証。synthetic PASSは実装能力の証拠だけであり、これらの科学的資格を与えない。前回F2のprep込み価格やN=null/STOPも維持する。

次の最小科学実行には、**ユーザー**が固定SOURCE・上記input pinの下で、保存配列のload、H4-AのP/S operator化、既存独立DF operatorとのfull H/T/V・一体補正・scalar比較を明示許可する必要がある。その際、共通scalar/identity方針、operator toleranceと元HへのDF差の扱い・許容差を決める。時間点、sampling、compiler、shotはこの最小operator比較には不要。`extract_matched_df_tail`の実体を使うなら、Gaussian basis回路生成と辞書和検査をその範囲に明示して含める。元H stateの精度資格を検査する場合も、その保存数値再解析を範囲に含める。

**Codex**は許可された範囲だけを実行する。mean、simulation/sampling、回路build/compile/synthesis、shot、PR＋RPE総費用、重要科学レビューは、それぞれ許可された後続範囲なしに開始しない。今回の実装許可・文書公開はそれらの科学実行を代行しない。

## 4. 公開と停止

専用branchには新module・新test・本書だけをcommit/publishし、remote commitと3ファイル再取得内容、変更範囲、clean statusを技術確認する。実際のremote SHA/URLと一致結果は最終チャットで報告する。

**mandatory STOP：実装・許可済み人工test・公開照合をもって終了。実H4 scientific call、旧run再開、M2との統合を自動開始しない。**

[SETUP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8f768ccf616af8ff2c35502ea8548138814f5554/docs/research/track_a_o1_m1_validation_preparation_20261106/M1_COMPARISON_SETUP.md
