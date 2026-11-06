# Track A（O1）M1：実H4-A operator correctness検証

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**固定adapterによるH4-AのP/DF数値表現・HD/HR・scalar・一体補正は、事前に定めたoperator検査基準を通過した。全比較の最大要素差は2.842170943040401e-14。** これは保存入力一つでのbinary64数値検証であり、厳密同値、ground-state精度資格、PF/RTE/RPE、回路encoding、費用改善の保証ではない。

今回の新しい明示実行許可はoperator reconstruction/correctnessに限定する。旧runのretryやSTOP再開ではない。M2とは独立し、merge/統合していない。成果物は[driver](../../../scripts/run_o1_m1_h4_operator_correctness.py)、[RESULTS.json](../../../artifacts/track_a_o1_m1_h4_operator_correctness/2026-11-06/RESULTS.json)、本書の3つのみ。

## 1. 固定SOURCE・input・範囲

| 項目 | 実行条件 |
|---|---|
| adapter SOURCE | commit `5e4f00487720dced86d53637464c48245a35a557`、`src/trotterlib/o1_m1_matched_split.py`、SHA256 `fa9b775c887437ded5ce3f5b08bcf7a390f20f0174986dc564ec4d0ee8397c20`。一切変更していない |
| 新driver | `scripts/run_o1_m1_h4_operator_correctness.py`、SHA256 `552c874240ea88f4608d7680db452906ecce5ee214b1db85c8cf505d499aa0db`。実行前に判定式を固定し、人工exterior-power/Walsh helper検査を通してからH4をload |
| 保存input | commit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`。Git objectから新作業領域`.venv/m1_input_snapshot.npz`へコピーし、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`一致 |
| 系とsplit | linear H4、隣接1Å、STO-3G、中性singlet、4 spatial/8 interleaved spin orbital。保存DFはtol-only1e-8、rank10。H4-Aの共通HD=T_eff、HR=V_square、全10 fragmentを残差側（fragment L_D0、一体項は保持） |
| register/sector | 全256次元でoperatorを比較。保存Nα=Nβ2の36 indexはmetadataで、各event/途中状態を射影していない |
| state | 元H保存`state`のみ。hash `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`、raw squared norm `1.0000000000000002`。差operatorの作用・期待値にのみ使用し、時間発展せず、再正規化・groundvector再生成・DF_state読取をしていない |
| 時間・PF/RTE条件 | 非該当。新δ/q/r/K、sampling、shot/cost条件を設定・評価していない |

Python 3.12.3、NumPy 1.26.4、SciPy 1.14.1、OpenFermion 1.6.1、Qiskit 1.3.0、PySCF 2.7.0。前回M1専用venvをread-only利用し、cacheは新worktree内だけに置いた。PySCF計算、再DF、共有環境変更はない。実行は14.856秒。保存入力コピーは元Git objectを再利用するlocal作業物であり、重複binaryを新成果物として公開しない。

## 2. 結果を見る前の判定基準

binary64 ε=2.220446049250313e-16、n=8、d=256、rank10として、`m=4*n^4+8*d+16*rank=18592`、`γ=m*ε/(1−m*ε)=4.128253294783225e-12`を使う。

`S=|c_N|+Σ|h_bare|+Σ|W|+Σ|h_eff|+Σ_l |λ_l|(Σ|G_l|)^2`、`tol_entry=γ*max(1,S)`を事前に固定した。実入力のSは103.50446951296075、要素ごとの絶対判定値は**4.272926672916701e-10**。数千項の加算・小行列積と係数スケールを保守的に扱う数値検査用基準であり、厳密forward-error certificateではない。synthetic testの1e-14をH4へ流用せず、結果に合わせて式や閾値を変更していない。

元H−DFの差は、同一演算子の算術誤差と区別する。保存inputで使われた既存operator診断の1e-9を別基準とし、今回はFrobenius normがその値以下であることを確認した。これはspectral normの数値上界による診断で、新しいenergy budget/allowanceを採択したものではない。

## 3. 回路を生成しない独立参照

P側は固定adapterのOpenFermion/JW→既存`PauliHamiltonian`をそのまま使う。参照側は、little-endianのbit占有・CAR符号から作るsparse ladderを使い、T_eff、bare一体、元quartic、保存Gのsquare和を追加cutoffなしで構成した。二体tensorは保存済み1/2込み値をそのまま使った。

さらに既存`df_linear_operator(backend="python")`を全Fock基底256本に作用させ、OpenFermion big-endianから明示bit reversalでlittle-endianへ戻した。matrix-free DFの既存実装とCAR/JWの経路を照合している。

数値S辞書は、保存GのHermitian固有分解だけを行い、軌道Uのminor/determinantによるexterior-power Fock表現でoccupation basisへ接続した。既存`exact_df_diagonal_coefficients`のI/Z/ZZを再構成し、各fragmentと全和を比較した。10 fragmentそれぞれ37個の非zero数値componentを持つ。最大Fock basis unitary誤差3.1086244689504383e-15、軌道eigenpair residual最大5.551115123125783e-16。これは**数値S辞書の演算子検査**であり、Gaussian gate列・encoded basis hash・controlled circuitの検査ではない。`extract_matched_df_tail`/`build_df_blocks`は呼んでいない。

## 4. 一致結果

全項目PASS。最大要素差とFrobeniusを別々に示す。全比較・fragment別・差作用/期待値はJSONに保存した。

| 比較 | 最大要素差 | Frobenius |
|---|---:|---:|
| HD：P対CAR | 1.7763568394002505e-15 | 1.7038054601998547e-14 |
| HR：P対保存factor CAR | 5.329070518200751e-15 | 3.64548916753669e-14 |
| full：P対CAR DF | 3.9968028886505635e-15 | 2.8233835409773773e-14 |
| 既存DF full対CAR | 4.884981308350689e-15 | 2.2373283068220498e-14 |
| 数値S辞書HR対CAR | 2.842170943040401e-14 | 1.2093636153278154e-13 |
| full：P対数値S辞書 | 2.842170943040401e-14 | 1.1740230600506738e-13 |
| 元H：P対bare/quartic CAR | 5.773159728050814e-15 | 3.145588874917658e-14 |
| 一体収縮係数：h_eff−h_bare対既存SOURCE contraction | 1.1102230246251565e-16 | 2.312732826880181e-16 |
| V_normal対D̂＋V_square | 7.105427357601002e-15 | 2.7815742043192566e-14 |

保存`non_DF_matrix`対CARは最大5.773159728050814e-15、保存`DF_matrix`対CARは4.551914400963142e-15。これらも新規operator参照と整合した。

**元H−DFは最大要素差8.881784197001252e-15、Frobenius3.12074066975129e-14。** 元H保存stateへの差作用L2は5.50639759304183e-16、raw normで一度割った差期待値はreal1.1073571117173366e-18（imag6.842277657836019e-49）。この期待値をground energy誤差保証とは呼ばない。PのHRを`H_original−T_eff`へ置換して差を埋める処理はしていない。

## 5. scalar・一体補正・微小項

| scalarの場所 | 値 |
|---|---:|
| 核c_N | 2.29310124732 |
| T_effの電子identity（P） | −7.3042454943550705 |
| V_squareのidentity（P） | 4.679666433618259 |
| V_squareのidentity（数値S） | 4.679666433618258 |
| full identity（P） | −0.3314778134168117 |

T_effのtrace/2、Vの全Fock trace/d、数値S identity、full traceとのscalar対応最大差は8.881784197001252e-16。faithful tailとidentity抽出の両bookkeepingでHRを再構成し、どちらも最大5.329070518200751e-15でCARに一致した。抽出scalarはV由来分だけとし、核scalarや保持Tの電子identityを重ねて移管していない。一体収縮の符号・prefactorと`T_eff=T_bare+D̂`も独立に照合した。

runtimeのOpenFermion `EQ_TOLERANCE`は**1e-8**。adapterの追加cutoff=0から微小項完全保持を推定していない。独立CAR行列の全4^8 Pauli係数をXOR diagonal＋Walsh変換で求め、固定adapter出力と比較した。

| 対象 | 全係数差の最大値 | 出力にない参照係数の最大値 | そのL1和 |
|---|---:|---:|---:|
| HD | 4.440892098500626e-16 | 2.7755575615628914e-16 | 9.824552875186336e-15 |
| HR | 8.881784197001252e-16 | 4.2653459813964467e-16 | 2.6677448150058035e-14 |
| 元H | 7.216449660063518e-16 | 3.7164738047604715e-16 | 2.3049883291831445e-14 |
| DF full | 3.8716771815419604e-16 | 3.8716771815419604e-16 | 2.37492884536175e-14 |

EQ_TOLERANCE以上または算術検査tol以上の欠落は全対象0。欠落nonzero数にはCAR/Walshのbinary64丸め由来も含むため、全てを厳密な物理項の切捨てと認定しない。この独立照合で観測した**cutoff・集約・丸めを合わせた数値影響**は上表およびoperator差の範囲だった。cutoffの一般的無害性、厳密symbolic一致、別入力への保証は主張しない。

## 6. 実行command・未解決と停止

新worktree `track-a-o1-m1-h4-operator-correctness-20261106`内で実行：

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 \
MPLCONFIGDIR=.venv/mplconfig XDG_CACHE_HOME=.venv/cache \
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-a-o1-m1-minimal-implementation-20261106/.venv/bin/python \
scripts/run_o1_m1_h4_operator_correctness.py \
  --snapshot .venv/m1_input_snapshot.npz \
  --output artifacts/track_a_o1_m1_h4_operator_correctness/2026-11-06/RESULTS.json
```

検査基準を超える不一致はなかった。adapterの技術修正、再DF、groundvector再生成、DF_stateへの変更、PF時間発展、simulation、sampling/RNG、shot/cost、回路build/compile/synthesis、fit/rank探索は行っていない。重要科学レビューではない。

残るのは、数値S辞書を実Gaussian/controlled gate列へencodeしたときの一致、共通保持program、finite-RTE実装law/補正mean、元H ground精度・数値差のenergy allowance/転送などである。今回のoperator PASSだけでN/半径や全PR＋RPE改善を認定しない。

**次の判断はユーザー**：目的に応じ、共通保持/identity方針と許容誤差を指定し、例えば補正meanの最小検証を新たに許可する。回路encodingを検証するならbuild範囲も明示する。**Codex**は許可された次scopeだけを実施する。今回は3成果物の独立branch公開・remote SHAと全文照合・clean確認で**mandatory STOP**。旧SOURCE/raw/result/grant/marker/STOPと他作業は変更しない。
