# H-chain系サイズ・実行可能delta窓におけるPF係数検証

## 目的

最終的な$C$をsmall-time外挿だけから決めず、各H-chainサイズでQiskit回路を実際に
作用できる$\delta$窓から決める。geometryは全系で既定の原子間距離1.0 Å、basisは
STO-3G、DF rankはproject configの選択値に固定する。各系の代表分割は
$L_D=\lfloor L/2\rfloor$とする。

今回の運用係数は、実行窓$\mathcal D_n$上のpoint coefficientの上包絡として

$$
C_{\mathrm{use}}(H_n,L_D)
=\max_{\delta\in\mathcal D_n}
\frac{|\Delta E_{\mathrm{partial}}(\delta)|}{\delta^2}
$$

と定義する。H2--H5では支配固有位相係数と摂動係数の大きい方を検証用上包絡とし、
PF演算子を対角化しないH6以降では論文Eq. (D6)の摂動係数上包絡を使う。この値は
検証窓内の経験的上包絡であり、厳密上界ではない。

## delta窓とEq. (D6)の悪条件

lower-order PF用の既存Qiskit実行窓を使う。

| 系 | $\delta$窓 |
|---:|---:|
| H2 | 0.730, 0.732, 0.734, 0.736 |
| H3 | 0.750, 0.752, 0.754, 0.756 |
| H4 | 0.370, 0.372, 0.374, 0.376 |
| H5 | 0.360, 0.362, 0.364, 0.366 |
| H6 | 0.250, 0.252, 0.254, 0.256 |
| H12（将来のGPU run） | 0.120, 0.122, 0.124 |

今回の主推定量は、Evaluation側の現行CPU/GPU実装と論文Appendix Dに合わせて、

$$
\Delta E_{\mathrm{pert,D6}}(\delta)
=
\frac{
\operatorname{Re}\langle\psi_0|
[U_{\mathrm{partial}}(\delta)-e^{-iE_0\delta}]|\psi_0\rangle
}{\delta\sin(E_0\delta)}
$$

を使う。$\sin(E_0\delta)$が0に近い点では悪条件になるため、暫定的に
$|\sin(E_0\delta)|<0.1$を不適格点とし、small-$\delta$近似へ置換せず係数fitから除外する。
H2--H6の実行可能窓では最小値が0.718で、不適格点は0だった。

比較診断として

$$
z(\delta)=e^{iE_0\delta}
\langle\psi_0|U_{\mathrm{partial}}(\delta)|\psi_0\rangle,
\qquad
\Delta E_{\mathrm{SI}}=-\frac{\operatorname{Im}z(\delta)}{\delta}
$$

も保存するが、これは主たる$C_{\mathrm{partial}}$には使わない。旧$\cos(E_0\delta)$式の
conditioningも診断として残す。

小さすぎる$\delta$でのstatevector差の桁落ちと、大きすぎる$\delta$での高次項は別の
問題なので、摂動値とsurvival phaseの一致および窓内point coefficientの変化を同時に
検査する。

## 二つの実行経路

H2--H5ではpartial-$S_2$演算子をsector内で構築・Schur分解し、支配固有位相のbiasを
直接求める。H6ではこの方法のdense memoryが大きくなるため、PF演算子を構築せず、

1. full-$H$基底状態にQiskitで決定論forward halfを作用
2. sector内matrix-free $H_R$へ`scipy.sparse.linalg.expm_multiply`を適用
3. Qiskitでreverse halfを作用

してsurvival phase、Eq. (D6)値、shift-invariant診断値を得る。H2--H5ではこのstate-action経路も実行し、dense参照
との差を検査する。

## 結果

### 支配固有位相との直接比較

| 系 | qubit | rank | $L_D$ | $C_{\mathrm{eig,max}}$ | $C_{\mathrm{D6,max}}$ | 相対差 | D6 2%条件 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| H2 | 4 | 3 | 1 | 0.00328942 | 0.00331879 | 0.893% | 通過 |
| H3 | 6 | 5 | 2 | 0.00498346 | 0.00504048 | 1.144% | 通過 |
| H4 | 8 | 7 | 3 | 0.01340628 | 0.01349787 | 0.683% | 通過 |
| H5 | 10 | 9 | 4 | 0.01187331 | 0.01192681 | 0.451% | 通過 |

Eq. (D6)推定器はH2--H5の全系で支配固有位相との2%基準を通過した。一方、H3では
従来のsurvival signal biasと支配分枝biasの差が最大2.076%であり、単一位相signal条件は
引き続き不通過として残す。これはD6推定器の不通過ではなく、QPE/RPE signalを単一分枝で
表す別の受理条件である。

H4 rank 12の全$L_D=0,\ldots,11$でもD6検証を追加した。各分割で6点中、
$|\sin(E_0\delta)|<0.1$の2点を除いた4点をfitに使い、D6係数と同じ4点の支配位相係数の
fit差は最大0.288%、pointごとの差は最大0.882%で、全12分割がD6条件を通過した。

### PF演算子非構築のstate-action経路

| 系 | qubit | rank | $L_D$ | $C_{\mathrm{D6,use}}$ | D6/phase最大相対差 | dense参照との差 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| H2 | 4 | 3 | 1 | 0.00331879 | 0.755% | $2.12\times10^{-15}$ | 通過 |
| H3 | 6 | 5 | 2 | 0.00504048 | 0.912% | $1.33\times10^{-15}$ | 通過 |
| H4 | 8 | 7 | 3 | 0.01349787 | 0.297% | $7.22\times10^{-15}$ | 通過 |
| H5 | 10 | 9 | 4 | 0.01192681 | 0.188% | $7.94\times10^{-15}$ | 通過 |
| H6 | 12 | 11 | 5 | 0.02086663 | 0.105% | 未対角化 | 通過 |

H2--H5でstate-actionとdense参照が数値誤差内で一致したため、H6の係数は同じ演算モデル
の延長として採用できる。係数は系サイズに対して単調ではないので、H4の値をH12へ
流用しない。

## H12での採用規則

GPU経路を拡張した後、H12では$\delta=0.120,0.122,0.124$だけを実際に実行し、各点で

- ground-state residual
- survival radius
- branch tracking／単一位相診断
- Eq. (D6)摂動値、shift-invariant診断値、phaseの整合性
- $|\sin(E_0\delta)|$のD6 conditioningと旧$|\cos(E_0\delta)|$診断

を記録する。通過点が3点そろった場合に

$$
C_{\mathrm{use}}(H12,L_D)
=\max_{\delta\in\{0.120,0.122,0.124\}}
\frac{|\Delta E_{\mathrm{pert}}(\delta)|}{\delta^2}
$$

を採用する。したがって現時点ではH12の$C$は未決定であり、H6の値を外挿して最終入力
にはしない。

## 実行と成果物

```bash
.venv311/bin/python scripts/run_pf_c_system_size_validation.py
```

Eq. (D6)推定器とstate-action実装の検証成否をprocess終了codeへ反映する。H3の別条件である
単一位相signal不通過はartifactに残すが、このscriptの終了codeは1にしない。
indexは`artifacts/pf_c_system_size_validation/h2_h6_paper_d6_c_v1.json`、H2--H5の
dense参照は同directoryの`*_pf_d6_validation_v1.json`である。いずれもdirty local worktree evidence
であり、最終コスト評価またはimmutable CI evidenceではない。

| 確認内容 | 結果 |
|---|---:|
| 系サイズ検証専用テスト | `5 passed` |
| PF delta・系サイズ検証の専用テスト | `8 passed` |
| indexと4個のcore artifactのschema・fingerprint | 全件成功 |
| 全local test | `444 passed, 4 warnings` |

4 warningは既存の`chemistry_hamiltonian.py`の`ComplexWarning`であり、今回の検証の
失敗ではない。
