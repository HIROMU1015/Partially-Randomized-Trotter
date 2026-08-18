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
PF演算子を対角化しないH6以降ではshift-invariant摂動係数の上包絡を使う。この値は
検証窓内の経験的上包絡であり、厳密上界ではない。

## delta窓と旧摂動式の悪条件

lower-order PF用の既存Qiskit実行窓を使う。

| 系 | $\delta$窓 |
|---:|---:|
| H2 | 0.730, 0.732, 0.734, 0.736 |
| H3 | 0.750, 0.752, 0.754, 0.756 |
| H4 | 0.370, 0.372, 0.374, 0.376 |
| H5 | 0.360, 0.362, 0.364, 0.366 |
| H6 | 0.250, 0.252, 0.254, 0.256 |
| H12（将来のGPU run） | 0.120, 0.122, 0.124 |

`Evaluation-of-...`の旧実装には、摂動値を$\cos(E\delta)$で割る式があり、別版には
$\sin(E\delta)$で割る式がある。このため分母が0に近い特定の$\delta$では結果が
悪条件になる。今回使う

$$
z(\delta)=e^{iE\delta}\langle\psi|U_{\mathrm{partial}}(\delta)|\psi\rangle,
\qquad
\Delta E_{\mathrm{pert}}=-\frac{\operatorname{Im}z(\delta)}{\delta}
$$

はどちらの三角関数でも割らないshift-invariant式である。それでもartifactには
$|\cos(E\delta)|$と$|\sin(E\delta)|$を記録し、旧式で同じデータを再解析する場合の
悪条件を検出する。暫定閾値は0.1である。H2--H6の今回の窓では両分母の最小値が
0.60以上で、旧式に対しても悪条件点は0だった。

小さすぎる$\delta$でのstatevector差の桁落ちと、大きすぎる$\delta$での高次項は別の
問題なので、摂動値とsurvival phaseの一致および窓内point coefficientの変化を同時に
検査する。

## 二つの実行経路

H2--H5ではpartial-$S_2$演算子をsector内で構築・Schur分解し、支配固有位相のbiasを
直接求める。H6ではこの方法のdense memoryが大きくなるため、PF演算子を構築せず、

1. full-$H$基底状態にQiskitで決定論forward halfを作用
2. sector内matrix-free $H_R$へ`scipy.sparse.linalg.expm_multiply`を適用
3. Qiskitでreverse halfを作用

してsurvival phaseと摂動値を得る。H2--H5ではこのstate-action経路も実行し、dense参照
との差を検査する。

## 結果

### 支配固有位相との直接比較

| 系 | qubit | rank | $L_D$ | $C_{\mathrm{eig,max}}$ | $C_{\mathrm{pert,max}}$ | 相対差 | 単一位相2%条件 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| H2 | 4 | 3 | 1 | 0.00328942 | 0.00334399 | 1.66% | 通過 |
| H3 | 6 | 5 | 2 | 0.00498346 | 0.00508677 | 2.07% | 不通過 |
| H4 | 8 | 7 | 3 | 0.01340628 | 0.01353805 | 0.98% | 通過 |
| H5 | 10 | 9 | 4 | 0.01187331 | 0.01194929 | 0.64% | 通過 |

H3は摂動値とsurvival phase自体には最大$2.50\times10^{-5}$相対で一致する。しかし
非支配分枝によるsignal biasと支配分枝biasの差が最大2.076%となり、事前に固定した2%
条件をわずかに超えた。この結果は閾値を後から緩めず、不通過として残す。H3で単一
固有位相モデルを使うなら追加検討が必要である。

### PF演算子非構築のstate-action経路

| 系 | qubit | $C_{\mathrm{use}}$ | 摂動/phase最大相対差 | dense参照との差 | 判定 |
|---:|---:|---:|---:|---:|:---:|
| H2 | 4 | 0.00334399 | $9.97\times10^{-6}$ | $2.12\times10^{-15}$ | 通過 |
| H3 | 6 | 0.00508677 | $2.50\times10^{-5}$ | $5.03\times10^{-16}$ | 通過 |
| H4 | 8 | 0.01353805 | $2.09\times10^{-6}$ | $4.65\times10^{-15}$ | 通過 |
| H5 | 10 | 0.01194929 | $1.18\times10^{-6}$ | $7.99\times10^{-15}$ | 通過 |
| H6 | 12 | 0.02088853 | $3.60\times10^{-7}$ | 未対角化 | 通過 |

H2--H5でstate-actionとdense参照が数値誤差内で一致したため、H6の係数は同じ演算モデル
の延長として採用できる。係数は系サイズに対して単調ではないので、H4の値をH12へ
流用しない。

## H12での採用規則

GPU経路を拡張した後、H12では$\delta=0.120,0.122,0.124$だけを実際に実行し、各点で

- ground-state residual
- survival radius
- branch tracking／単一位相診断
- shift-invariant摂動値とphaseの整合性
- $|\cos(E\delta)|,|\sin(E\delta)|$の旧式conditioning

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

H3の単一位相条件をrejectするためprocess終了codeは1になるが、全成果物は保存される。
indexは`artifacts/pf_c_system_size_validation/h2_h5_operational_c_v1.json`、H2--H5の
dense参照は同directoryの`*_pf_v4.json`である。いずれもdirty local worktree evidence
であり、最終コスト評価またはimmutable CI evidenceではない。

| 確認内容 | 結果 |
|---|---:|
| 系サイズ検証専用テスト | `3 passed` |
| PF・finite-RTEを含む関連テスト | `8 passed` |
| indexと4個のcore artifactのschema・fingerprint | 全件成功 |
| 全local test | `442 passed, 4 warnings` |

4 warningは既存の`chemistry_hamiltonian.py`の`ComplexWarning`であり、今回の検証の
失敗ではない。
