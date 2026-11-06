# 固定H_phys sector基底状態：結果前条件

2026-11-06 JST。ユーザーが指定したH_physの基底状態取得・correctnessだけを実行する。
旧STOPは保存し、今回の明示指示による新しい限定作業とする。元H–DF12比較は再実行しない。

## 固定する対象・再利用

- HDF5 SHA256 `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958`。
  H4 linear、1 Å、STO-3G、8 spin orbitals、保存MO積分をcutoffなしでspin liftする。
- 元operatorは対応検証`275813d0a12a4bca62a8c972312c80386829de22`のrun03のH_physと同じ
  S0 array hash `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2`を要求する。
  核scalarは一回。元HDF5/DF12・当時のSOURCEと再利用コードのhashを照合する。
- 既存MolecularData reader、`h4_molecular_inputs.spin_lift_unthresholded`、既存sparse CAR
  helperで同じfull256 operatorを構築する。DF12はfile hash・metadata・sector indicesを
  確認するだけで、DF operator/stateを構築・評価しない。
- 保存OF basis順のNα=Nβ=2、sector dimension36。既存`reverse_bits`でQiskit indicesへ写す。
  `h2_non_df_pilot.canonical_ground`を一回呼ぶ。内部の`scipy.linalg.eigh`が同じsector行列を
  解き、最低pairとgapを返す。別targetのsolve、full Fockの最低energy探索は行わない。

## Hermitian化と事前判定

raw H_physを保存規約のtargetとし、solve用に`H_solve=(H_phys+H_phys†)/2`を明示する。
raw/full/sectorのHermiticity defectとH_solve−H_physのmax-entry/spectral normを記録する。
defectと変更のspectral normは既存Hermiticity基準`1e-10 Ha`以下を要求する。
この数値表現はcutoffやH_src/DFへのtarget変更ではない。

結果を見る前に次の適用を固定する（全て絶対誤差）：

- full256/sector36の`abs(||psi||_2-1) <= 1e-12`。
- H_solveのsector/full残差`||H psi-E psi||_2 <= 1e-10 Ha`。
- raw H_physのsector/full残差も同じ`1e-10 Ha`。
- 既存の独立occupation参照で作った同じraw H_phys sectorへの残差も`1e-10 Ha`。
  CAR sector対この独立参照のspectral norm差は`1e-10 Ha`以下。
- 指定sector外stateのL2 normは`1e-12`以下。bit-order/埋込indicesとarray roundtripはexact。
- phaseは既存canonical_groundと同じ、Qiskit full-vectorで最大絶対振幅の最初のindexを
  pivotとし、実正へ回す。pivotの虚部絶対値は`1e-12`以下・実部は正。
  同一pivotでphase再適用とglobal phase `exp(0.371i)`からの復元を行い、L2差`1e-12`以下。
  phaseチェックは再solveを含まず、同振幅tieの別pivotへ乗り換えない。
- rawのnormalized Rayleigh値と取得Eの実部差・虚部は`1e-10 Ha`以下を記録・確認する。

energyは指定sectorの最低固有値である。gapは同じ既存eigh呼出しの戻り値を記録するだけ。
固有値残差とHermitian化差は数値的不確かさの診断で、厳密intervalやSCF誤差保証ではない。
閾値変更、別state/別Hamiltonianへのfallback、追加solveは行わない。

## 保存・性能・終了

private NPZは元LOCAL checkoutの新しい
`artifacts/b2_hphys_ground_state/2026-11-06/run01/private/`へ保存する。
private directoryに`*`の.gitignoreを置き、payloadはmode600、directoryはmode700で新規作成する。
公開はhash・energy・provenance・correctness・入力仕様だけ。stateや積分配列は公開しない。

事前資源：CPU affinity32、available RAM約55 GiB、GPU driver利用不可。
full256の既存sparse CARとdense36 eighを使用し、BLAS既定設定を維持する。
solveは一つなのでworker増設は不要。共有環境・他ユーザーの設定/jobを変更しない。
人工検証後にSOURCEをcommitしてから実入力をsolveし、結果公開・remote照合後mandatory STOP。
SERVER 2統合、signal/round/shot/価格・費用計算は次の明示承認対象。
