# LOCAL：固定H_physの指定sector基底状態

2026-11-06 JST。**`HPHYS_SECTOR_GROUND_STATE_CORRECTNESS_PASS`：21/21 checks PASS。**
ユーザーが採用したcutoff前のH_physについて、Nα=Nβ=2、36次元sectorの基底状態を取得した。
**基底エネルギーは約 `−2.1663874486347607 Ha`。** full Fock256次元の最低energyを求めた
結果ではない。元HDF5・MO積分・state配列は公開せず、stateはLOCAL private NPZへ保存した。
実分子のstate solveはこのH_physに対する1回だけ。RPE・SERVER 2統合へ進まずmandatory STOP。

## 1. 固定target・入力・SOURCE

H4 linear、1.00 Å、STO-3G、charge0、4 spatial/8 spin orbitals。
spatial MOの保存積分を係数cutoffなしでspin liftし、核scalarを一回加える。
spin順は`2p=α,2p+1=β`。対応する旧DF12・B2 `L_D=3`は接続先の識別情報であり、
今回DF state/DF演算子を評価せず、delta窓・時間発展・回路を用いていない。

| 対象 | identity |
|---|---|
| 元HDF5 SHA256 | `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958`、実行前後一致 |
| 旧DF12 NPZ | `bf5d3b2405b2bec4f88dd0589b75dd737b13e052`の固定snapshot、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| 旧DF Hamiltonian hash | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424` |
| raw H_phys operator hash | `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2`。前段階run03とexact一致 |
| solve用Hermitian表現のhash | `037ea63618bd595a3525f432871de0d7c8429727bb6ea282e313f52b0716c4bd` |
| 今回の実行SOURCE | `b168cbf93d7c319045057e8b67ea7ff014c8d7dc`。結果前PLAN・実装・runnerをcommitしてから実行 |
| 再利用した固定参照 | [元H–DF12結果][PRIOR] `275813d0…`、構築SOURCE `126b9d4480c7937b39c8be3aef7fbecda48cb556`、[Agent B実装][B] `66a41ad9…` |

元H–DF12検証の保存JSON、元積分/MO/AO overlap hash、再利用SOURCEのbytes、旧生成SOURCE
`c644925…`のhashを照合した。旧DF12 NPZから使ったのはfile identity・metadata・sector indices
だけで、DF ground stateを読み出して代用していない。前段階の全operator比較は再実行しなかった。

## 2. 構築・solve・Hermitian化

既存MolecularData readerと`h4_molecular_inputs.spin_lift_unthresholded`、M1のsparse CAR
helperを再利用して、前段階と同じraw H_physを構築した。MO gのOpenFermion添字は
`g[p,q,r,s]=(ps|qr)`、chemistへ戻すtransposeは`(0,3,1,2)`、two-body係数には1/2を含む。
H_srcへのcutoff、DF再分解、別SCFは呼ばない。

solve用表現を **`H_solve=(H_phys+H_phys†)/2`** と明示した。
rawの微小なanti-Hermitian成分を記録し、片側三角だけを黙って採用していない。
targetはH_physのままで、rawに対する残差も検証した。

保存OpenFermion sector basisを既存`reverse_bits`でQiskit indicesへ写し、既存
`h2_non_df_pilot.canonical_ground(H_solve,indices)`を一回呼んだ。内部の
`scipy.linalg.eigh`が36次元行列の最低pairを返す。既存関数が同じ呼出しで返す
sector gapは約`0.23263021512018667 Ha`で、別Hamiltonianの追加solveではない。
独立occupationの同じraw sector参照には取得stateの残差だけを評価した。

## 3. 結果前基準とcorrectness

[PLAN](PLAN.md)でnorm誤差`1e-12`、絶対L2残差`1e-10 Ha`、Hermiticity/symmetrizationの
spectral norm`1e-10 Ha`を結果前に固定した。緩和・別stateへのfallbackは行っていない。

| 検査 | 観測値 |
|---|---:|
| full / sector state norm | 1.0 / 1.0（norm誤差0） |
| H_solveのsector / full残差 | 2.11e-15 / 2.54e-15 Ha |
| raw H_physのsector / full残差 | 2.13e-15 / 2.56e-15 Ha |
| 独立occupation raw sectorに対する残差 | 2.54e-15 Ha |
| sector外へのstate漏れ（L2） | 0 |
| raw Hermiticity defect、max-entry / spectral norm | 3.33e-16 / 1.08e-15 Ha |
| H_solve−raw、max-entry / spectral norm | 1.67e-16 / 5.42e-16 Ha |
| H_solveのHermiticity defect | 0 |
| CAR sector対独立occupation参照、spectral norm差 | 1.33e-15 Ha |
| raw Rayleigh値の実部と取得Eとの差 / 虚部 | 3.11e-15 / 0 Ha |
| OF↔Qiskit・sector埋込・保存/再読込 | array/hash exact一致 |

位相は既存規則に従い、Qiskit full-vectorで最大絶対振幅の最初のindex（今回15）が実正に
なるよう固定した。pivot虚部・phase再適用差は0、同じpivotで`exp(0.371i)`から復元した
L2差は`1.30e-16`。phase検査に再solveは含めていない。

21 checksがすべてPASS。Hermitian化差とraw残差から、この固定binary64入力に対する
数値的correctnessを確認した。厳密なinterval証明、保存積分のSCF精度保証、全Fock最低性や
RPE成功保証は与えない。表示energyの末尾桁を数学的に厳密な値とは扱わない。

## 4. 保存したstateとhash定義

LOCALの元checkout以下：

```text
artifacts/b2_hphys_ground_state/2026-11-06/run01/private/state.npz
artifacts/b2_hphys_ground_state/2026-11-06/run01/private/provenance.json
```

NPZには`state_vector(256)`（Qiskit LSB順）、`sector_state_vector(36)`（保存OF sector順）、
`sector_basis_indices(36)`、`qiskit_full_vector_indices(36)`、`energy`、provenance metadataを
保存した。private directoryはmode700、state.npzはmode600で、局所.gitignoreでGit対象外。
公開branchにこれらの配列ファイルは含めない。

| 識別子 | SHA256 |
|---|---|
| full state vector | `1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db` |
| sector state vector | `c579f9da9d4097667418288630ad7bd601f1c1b9176ac5a2a5d4d704102a8280` |
| private NPZ file bytes | `deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c` |

array hashは既存S0規約：`dtype.str`のUTF8、compact JSONのshape（例`[256]`）のUTF8、
C-order contiguous array bytesを順にSHA256へ渡す。state dtypeは`<c16`、indicesは`<i8`。
NPZ file hashはcontainer全bytesの別識別子であり、array hashと混同しない。
完全な[公開provenance/correctness JSON][RESULT]にSOURCE・基準・全残差・環境を保存した。

人工mock検証4件PASS。人工testでは既知固有pairをmockし、追加の数値eighを行っていない。
実入力は既存sparse CAR＋dense36 eigh、CPU affinity32・既定BLAS設定で処理した。
Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、OpenFermion-PySCF0.5、
h5py3.16.0、Qiskit1.3.0。計算関数内runtimeは約0.261秒、solve＋検証は約0.063秒
（Python起動・import除外）。GPUは使用せず、共有環境・他ユーザーjobは変更していない。

## 5. Agent Bへの入力と残る条件

[agent_b_input_spec.json][SPEC]に、`target=H_phys`、
`state_role=original_H_ground_state`、raw H_phys hash、旧DF12 hash、full-state hash、
sector/basis hash、interleaved spin・Qiskit bit-order、生成SOURCE、private NPZのfield名を示した。
これは入力仕様であり、RPEConnectionContextを生成・実行していない。
記録の`source`はstate生成SOURCE。将来のRPE実行では、新しいdriver SOURCEとこのstateの
SOURCE/hashの双方を結合する。状態準備費用はユーザー指定の研究仮定として除外する。

**ユーザー**：次の科学実行を望む場合、RPE精度・成功確率・energy/alias条件、required roundと
schedule、model/PF/RTE/統計/実装の予算、native correctness・compiler・価格測定の範囲を
事前に採択・明示承認する。今回のstate取得はそれらの承認ではない。

**Codex（次の明示承認後のみ）**：このprivate stateと固定DF12を使うAgent B入力接続を行い、
承認範囲のsignal/半径/normalization等を検証する。SERVER 2統合や価格/shot/総費用まで含める
場合は、その条件・実行範囲も明示する。今回は転送・統合・実行していない。

今回の新SCF・積分生成・DF分解、H_src/DF12の追加state solve、B2探索、PF/RTE、
回路build/compile、sampling・価格MC、RPE round/shot/費用計算、詳細科学レビューは0件。
旧SOURCE/input/raw/result/grant/marker/STOPを保全し、専用branch公開・remote SHA/tree/
全変更内容照合後に**mandatory STOP**する。

[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/275813d0a12a4bca62a8c972312c80386829de22/docs/research/b2_original_df12_correspondence_20261106/REPORT.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/docs/research/track_a_b2_rpe_implementation_20261106/REPORT.md
[RESULT]: ../../../artifacts/b2_hphys_ground_state/2026-11-06/run01/result.json
[SPEC]: ../../../artifacts/b2_hphys_ground_state/2026-11-06/run01/agent_b_input_spec.json
