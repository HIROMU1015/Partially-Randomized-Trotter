# Track A / O1 M2：全core・完全残差の最小実装

2026-11-06 JST。基点 `da7a6a2a785fcce08dc850b721aac350b81d7ad0`、専用branch `codex/track-a-o1-m2-minimal-implementation-20261106`。新module・人工test・本報告の3ファイルだけを追加する。M1とは独立でmergeしない。通常の技術確認であり、重要科学レビューではない。

**実装と人工17 testsは完了。実H4の科学correctness・性能は未資格。** 本段階は、保存update005の全coreと元物理Hとの差を扱う接続を実装し、人工配列でその代数と既存helperへの入口を検査した。実H4のJ/U・積分・stateの数値load、operator/残差の生成・再解析、simulation、sampling、回路build/compile/synthesis、shot/総cost計算は実行していない。

## 1. 追加APIと保存inputの結合

- [o1_m2_residual_adapter.py](../../../src/trottertracks/o1_m2_residual_adapter.py)：`prepare_residual`、`load_saved_pair`、`SavedCoreBinding`、`WholeCoreResidual`。
- [test_o1_m2_residual_adapter.py](../../../tests/management/test_o1_m2_residual_adapter.py)：人工2空間軌道・4 spin modeの独立CARと実helper検査。

保存coreは[前段core map][core-map]で固定したupdate005を採用する。新fit、rank探索、cost-aware optimizerを呼ばない。研究Cの既存cost-aware fit NO-GOは維持する。

| 項目 | 固定値 |
|---|---|
| 保存coreの科学SOURCE | `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4` |
| 保存結果commit | `aced1647b81e8d28091058cdc7b6e3e1ee54896a` |
| [core NPZ][saved-core] | `artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz`、J/U各float64 `(8,4,4)` |
| core SHA256 | `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` |
| [元H input NPZ][saved-input] | `artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`、h `(4,4)` / g `(4,4,4,4)` / scalar `()` |
| input SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |

`UPDATE005`は上記のcore生成SOURCE・input/core bytes・shapeのbindingである。`source_identity=980b…`は保存coreの生成来歴であり、今後このadapterを動かす新科学実行SOURCEではない。将来の実行は今回の公開commitと適合依存を別途固定し、980b…の過去実行と区別する。Pauli metadataでも`core_generation_source_identity`と明記する。新SOURCE台帳や承認証明は追加しない。

今回これらの保存配列はdecodeしていない。`load_saved_pair(core_path,input_path)`は将来の明示的な入口であり、パスを自動探索しない。両ファイルのbytesを確認した同じpayloadを`allow_pickle=False`でdecodeする。import・test collectionはこの入口を呼ばない。人工loader testは一時directoryの人工NPZと専用bindingだけを使う。

`prepare_residual(h,g,constant,J,U,*,source_identity,input_identity,core_input_identity,expected_frames,sector)`は明示配列を受け取り、input同一性、shape、real/finiteness、J対称性、U直交性、実chemist積分の対称性を検査する。数値補修・projectionを行わない。全frameを元の順序で`GeneralKModel`へ渡す。出力積分は独立copyとread-only flagを持つ。sectorは`(Nalpha,Nbeta)` metadataであり、operatorやeventを途中射影する指定ではない。

## 2. 何をHcoreとRへ置くか

既存`c1_stage1.corrected_kappa`と`c1_h6_initialization.reconstruct`を直接流用する。

\[
\kappa_{\rm orig}=h-\tfrac12\sum_r g_{prrq},\qquad
g_{\rm fit}=\operatorname{reconstruct}(J,U).
\]

\[
H_{\rm core}=c_{\rm nuc}I+\kappa_{\rm orig}E
+\tfrac12\sum_{t=0}^{7}\sum_{ij}J^{(t)}_{ij}N_i(U_t)N_j(U_t).
\]

通常のnormal-ordered積分入口へ渡すため、adapterは

\[
h_{\rm core}=\kappa_{\rm orig}+\tfrac12\sum_r(g_{\rm fit})_{prrq},
\quad R_g=g-g_{\rm fit},\quad R_h=h-h_{\rm core},\quad R_{\rm nuc}=0
\]

を明示する。κは元gに固定し、fit tensorで再補正しない。Rの一体収縮を落とさない。旧保存field `tensor_residual=g_fit−g_original`とR_gは逆符号である。核scalarはcoreへ一度だけ置くが、JW後の**電子identityがゼロとは仮定しない**。

これは全coreをDにするための表現接続である。旧LD2 PRのcandidate内frame2..7 tailをRとして再利用しない。`split_native_model(ld=8)`が空tailを返すことも、物理残差ゼロを意味しない。一般signed Jを平方DF fragmentへ型だけ変換する処理は追加していない。

## 3. 既存処理の再利用と残る境界

| API / 再利用 | 今回の処理 | 残る資格 |
|---|---|---|
| `general_core` / `from_ffsim_number_arrays` | 原κ、核scalar、全signed J/U、interleaved `2p+spin` と元frame順を保持 | 実H4の同一性は未検査 |
| `native_core()` / `c1_stage2_native.prepare_cdf` | base＋全frameの既存`NativeModel`と対角primitiveを準備 | 明示的な固有分解入口。builderは呼ばない。generic basisであり、強対照のspin-adapted precision/fusion政策を採択したとの主張はしない |
| `pauli_parts()` / `PauliHamiltonian.from_spatial_integrals` | physical/core/residualを同じ通常積分・spin/JW規約へ渡す。Qiskit labelの右端がqubit0 | operator/state生成や科学correctness検査は別scope |
| `residual_rte(step_time,r,K)` / `PauliSignalAdapter`、既存`RTEComponent`・finite law | R全体のnonidentity Pauli tail、sign/probability、電子identityの一度だけの保持 | R単独のlaw入口。D-half/R/D-halfの合成kernel、sampling、native event、RPE/priceは未接続 |

Pauli/RTEは既存のprivate-package importパターンを小さく限定し、実SOURCEをロードする。`trotterlib.__init__`の広いeager importや`df_hamiltonian`経由のPySCFを避ける。mapper・RTE・nativeの数式や既存SOURCEは変更しない。nativeの人工testではpackage `__init__`だけを一時的にpassive namespaceへ置き換え、実`prepare_cdf`、実固有分解、実general-K係数・実primitiveを使う。数式helperのmockではない。productionのnative入口には通常の既存library runtimeが必要である。

**normal-ordered積分段の完全残差と、Pauli bridgeの数値資格を区別する。** 当adapterは追加の係数cutoffを置かないが、既存mapperの`get_fermion_operator`とJWのOpenFermion symbolic累積には内部`EQ_TOLERANCE`がある。微小係数がすべて保存されるとは主張しない。今回の通常スケール人工入力のPASSは、この既存処理をexact/certifiedにするものではない。将来の実H4 gateではCAR等の独立referenceとのoperator一致と微小項の影響を実測し、必要ならその科学条件を保つ最小修正を別scopeで判断する。

## 4. 人工検査の結果

**17 passed in 2.14s（最終実行）。** 独立referenceは占有bitとCAR符号から作った4mode・16次元行列で、Qiskit circuit/Operatorからreferenceを得ていない。Pauli labelの解釈もtest内のI/X/Y/Z Kronecker積で独立に行った。

- 非対角・正負の一般J、異なるUを使う全density-product coreと、adapterのnormal-ordered coreが一致する。base＋全frame＋Rが人工physical Hへ戻る。
- 元g由来κ、R_gの符号、非zero一体残差、核scalarの一度の所有、非zero電子identityを検査する。
- 独立full-Fock CARと実Pauli mapper出力を照合し、宣言sector部分行列でも和の一致を検査する。sector途中射影はしない。
- 既存nativeのbase固有分解と全frame primitiveを実体で検査する。全16占有patternでJのnumber-density値とI/Z/ZZ値が一致する。回路constructorは呼ばれたら失敗するようmockし、呼出しなし。
- 実finite lawのnonidentity residual全体、sign、probability sum、λの定義、電子identityを検査する。sampling・time evolutionは呼ばない。
- 完全一致core、核offsetの共有、mismatched identity/rank/shape、非対称J/h/g、非直交U、complex、nonfinite、非scalar、無効sectorを検査する。
- lazy import、人工NPZのbinding、byte不一致でdecode前に失敗すること、出力copyを検査する。

CAR/JWの`atol=1e-14, rtol=0`は小人工fixtureの加算順丸め用の技術許容値である。初回の`2e-15`は差`2.22044605e-15`で失敗し、代数referenceを保ったまま設定した。これは実H4の科学tolerance・保証・誤差予算へ移送しない。

実行command（新testファイルだけ、conftest・自動plugin・pytest cache無効）：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-a-o1-m2-minimal-implementation-20261106
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
XDG_CACHE_HOME=/tmp/track-a-o1-m2-cache-20261106 \
MPLCONFIGDIR=/tmp/track-a-o1-m2-cache-20261106/matplotlib PYTHONPATH=src \
/tmp/track-a-o1-m2-runtime-20261106/bin/python -m pytest \
  --noconftest -p no:cacheprovider -q tests/management/test_o1_m2_residual_adapter.py
```

専用runtime：Python3.12.3、NumPy2.5.4、SciPy1.18.1、Qiskit2.6.2、OpenFermion1.8.1、pytest9.2.0。systemには必要依存がなかったため、共有packageを変更せず専用`/tmp`環境へ導入した。過去実行runtimeの再現やH4科学資格を主張しない。既存test群・旧runnerは実行していない。新しいapproval proof、grant、ledger、receiptは作成していない。

## 5. 準備状態と次の最小科学scope

今回完了したのは配列adapterと人工correctnessの技術準備である。実H4による科学correctness、元H stateでのPF/finite mean/半径、全round成功・費用は未実施。強baselineの再ランキングやM1との統合も行わない。

次の最小段階は、**ユーザーが明示許可した場合にCodexが**、上記の固定update005/coreと元h/g/scalarだけを一度ロードし、全core＋Rのoperator一致を独立CAR referenceと同じspin/JW規約で検査することである。実際のSOURCE、科学tolerance、full-Fock/sector比較範囲、既存mapper微小係数の扱いを先に固定する。新fit・simulation・state生成・回路・shot/価格をこの最初のgateへ自動追加しない。

そのgate後に元H stateの検査へ進む場合は、[保存physical ground][ground]と同じinput・sector basis/indexの対応、normalization、元H eigenstate残差を許可scope内で確認する。旧candidate own-groundを代用しない。保存stateは検証referenceであり、算法への無料E0/gap oracleを意味しない。

科学性能の比較段階はさらに別の判断である。同じ元H target/state・情報・error/price条件で、全core＋完全残差と適用可能な強PF/collected-Pauli/UWC/DF等を比較する。generic native準備や旧LD2/own-DF-groundの価格を強baseline成績として流用しない。D/R合成kernel、mean/半径、実装誤差・有限bit law、full-wrapper価格、全round N×費用が不足している。

公開と全変更fileのremote再取得一致確認の後、**mandatory STOP**。次の担当：ユーザーが上記最小科学gateの実行範囲を判断し、許可された場合にCodexが実施する。今回はその科学実行を開始しない。

[core-map]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da7a6a2a785fcce08dc850b721aac350b81d7ad0/docs/research/track_a_o1_m2_validation_preparation_20261106/M2_EXISTING_CORE_MAP.md
[saved-core]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz
[saved-input]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz
[ground]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/C/ground_physical.npz
