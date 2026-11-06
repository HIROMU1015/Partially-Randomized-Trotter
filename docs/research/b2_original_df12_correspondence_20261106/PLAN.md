# 元H–保存DF12：第一段階の結果前固定条件

2026-11-06 JST。対象は利用者が明示承認したLOCALのoperator対応検証だけ。
HDF5 `a3c88f6a…a958`、DF12 NPZ `3bc92e92…f80a`、生成SOURCE
`c644925b50587072784846df09bf02c39e8453e1`を固定する。新SCF/DF、
元H state solve、時間発展、回路、sampling、shot/費用計算は実行しない。

## 規約・再利用

- `MolecularData`のread-only loaderと`get_molecular_hamiltonian`を再利用する。
  H_physは保存MO積分のunthresholded spin lift、H_srcはSOURCEがDFへ渡した
  InteractionOperator tensor（spin liftの`EQ_TOLERANCE=1e-8`を含む）。
  chemist ERIへ戻すtransposeは`(0,3,1,2)`。spin liftは既存
  `h4_molecular_inputs.spin_lift_unthresholded`。このmoduleの生成関数は呼ばない。
- 固定DF12は既存`pr2_s0_s1_validation.load_snapshot`でhash照合して読む。
  返される保存DF stateは使用せず、元H stateを供給しない。
- full Fockは既存`run_o1_m1_h4_operator_correctness`のCAR sparse helperと
  OpenFermionの`jordan_wigner_sparse`を対照にする。SymbolicOperatorの`+=`は
  追加cutoffを含むため、cutoffなしの参照ではInteractionOperatorの全非零termを
  FermionOperator.termsへ直接渡す。packageの閾値・設定は変更しない。
  通常の`get_sparse_operator(InteractionOperator)`との差は別診断として記録する。
- DFは全12 signed factorsを保存生成順で一度ずつ使用する。既存
  `df_linear_operator`のfull/sector作用と、独立`occupation_df_matrix`を再利用。
  後者はG²の中間状態をfull occupation空間で扱う。元H quartic項に限り、
  同じ既存ladder helperを使う短いoccupation拡張を追加する。
- 8 spin orbitals、full dimension 256、Nα=Nβ=2の保存sector 36。
  OpenFermionのmode0=MSBから、既存`reverse_bits`でQiskit LSBへ写す。
  full行列を先に検査し、sector projectionで漏れや非Hermiticityを隠さない。

## 結果前の数値基準

binary64の実装整合性は既存M1 operator-only runnerの規則をそのまま使う：

```text
n=8, d=256, rank=12
m=4*n**4 + 8*d + 16*rank
eps=binary64 epsilon; gamma=m*eps/(1-m*eps)
S=|c|+sum|h_phys|+sum|W_phys|+sum|h_eff|
  +sum_l |lambda_l|*(sum|G_l|)**2
tau_entry=gamma*max(1,S)
```

hash/layout/scalar/orderは指定値とのexact照合。独立構築・一体補正のSOURCE規約・
normal-order再構成・Hermiticity・粒子数/αβ・Sz/S²保存・sector漏れはmax-entryで
`tau_entry`以下を検査する。MOの`CᵀSC=I`は既存H4入力契約の`1e-10`。
full/sectorの差ではmax-entry、Frobenius、induced infinity norm、spectral norm
（最大singular value）を区別する。固有vector・最低energyは計算しない。

独立occupationのfull/sector行列を、元のbinary64値を整数比でexact liftして80/120桁で
構築する。両precisionのentry差は`1e-60*max(1,max_entry(H_120))`以下を要求する。
80桁に対し20桁以上を残した保守的な安定性検査で、interval保証ではない。
既存参照の80/120桁方針を使うが、signal/expm/state関数は呼ばない。
binary64との差、80/120桁安定性、high-precision行列差のノルムを別々に記録する。

H_phys−H_src、H_src−H_DF12、H_phys−H_DF12には**モデル差の合否閾値を置かない**。
非零だけで不適格にせず、近似差を測定する。`tau_entry`はRPE energy/model予算ではなく、
厳密なroundoff enclosureでもない。数値的整合性、近似差、来歴証明の不足を別判定にする。
結果を見て許容値・target・rank・factor順を変更しない。

## 性能と停止

事前確認：CPU affinity 32、RAM約62.5 GiB・available約55 GiB、GPU driver利用不可。
256次元のsparse CARと既存vectorized BLASで十分小さい。独立80/120桁×full/sectorの
4 taskを4 processで並列化する（task数とaffinityから選択、資源上限ではない）。BLAS normは親processで
順次実行しoversubscriptionを避ける。共有環境・他ユーザーjobは変更しない。
同じ経路を人工入力で先に確認し、実行SOURCEをcommitしてから固定入力を数値読込する。

第一段階の結果公開・remote照合後、または入力/実装の不適格判定後にmandatory STOP。
H_phys/H_srcの最終target採用、元H state solve、Agent B実H4実行は次回の明示承認対象。
