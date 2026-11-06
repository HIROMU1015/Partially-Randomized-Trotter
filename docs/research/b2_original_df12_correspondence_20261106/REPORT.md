# LOCAL：Track A・B2 元H–保存DF12数値対応検証

2026-11-06 JST。**第一段階は `OPERATOR_CHECKS_PASS_MODEL_DIFFERENCES_RECORDED`。**
固定元HDF5と保存DF12は、今回のoperator検証で数値的に整合した。保存binary64係数を
120桁へexact liftした参照の元H–DF12差はfullで約`9.27e-15 Ha`、sectorで約`6.23e-15 Ha`
（spectral norm）。厳密な同一operator・完全な生成来歴証明・RPE精度資格を意味しない。
**H_phys/H_srcの採用は未決定、元H state solveは0件。第一段階でmandatory STOP。**

## 1. 固定入力・実行SOURCE

| 対象 | 固定参照・確認 |
|---|---|
| LOCAL HDF5 | `.venv311/lib/python3.11/site-packages/openfermion/testing/data/H4_sto-3g_singlet_df_d100.hdf5`（LOCAL repository以下）。SHA256 `a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958`、実行前後に一致 |
| 保存DF12 | commit `bf5d3b2405b2bec4f88dd0589b75dd737b13e052`、`artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz`。SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、canonical Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`を既存loaderで照合 |
| DF生成SOURCE | `c644925b50587072784846df09bf02c39e8453e1`。`pr2_s0_s1_validation.py` SHA256 `1ccf6505d2933699504db187e3849251d1fef7bc58d8dca0b5f19871896ca6fe`、`df_hamiltonian.py` SHA256 `7d711e1c3c093079ae890ffc1907813d55d211a4fb622e4f36edee05fdea0c52`がsnapshotの記録と一致 |
| 本検証SOURCE | `126b9d4480c7937b39c8be3aef7fbecda48cb556`、Agent B実装`66a41ad9219f12377e7d325e272c1a90d66ab7e5`を基点とする専用worktree/branch。実行時SOURCEと結果前PLANのcommit bytesを照合 |
| 文書参照 | [LOCAL回収][LOCAL]、[SERVER 1調査][SERVER1]、[Agent A仕様][A]、[Agent B実装][B]を読取参照。SERVER 1/2のjob・filesystemへ接続していない |

全hash・再利用SOURCE hash・MO/h/g/AO overlapのarray hash・構築operator hash・環境は
[最終result.json][RESULT]に保存した。公開物はSOURCE、診断結果とhashのみであり、元HDF5、
積分・MO配列、operator行列、state配列は公開していない。元ファイル、旧SOURCE、raw、
grant/marker/STOP、既存dirty checkoutは保全した。

## 2. 構築規約と独立参照

固定条件：linear H4、隣接距離1 Å、z=`[-1.5,-0.5,0.5,1.5]` Å、STO-3G、charge0、
SCF multiplicity1、4 spatial/8 spin orbitals。全Fock256次元と保存`Nα=Nβ=2, Sz=0`の
36次元sectorを用いる。この36次元はtotal-spin singletだけへ射影した空間ではない。
DF rank12・全factor固定、B2の`L_D=3`は接続先の識別情報だけで、今回split/PFを評価しない。
delta窓・時間発展は使用していない。

| operator | 保存入力からの定義 |
|---|---|
| H_phys | HDF5のspatial MO h/gを既存`spin_lift_unthresholded`でspin化。係数cutoffなし |
| H_src | 既存`MolecularData.get_molecular_hamiltonian()`がDF生成へ渡したInteractionOperator。`spinorb_from_spatial`の`abs(coefficient)<1e-8` cutoffを含む |
| H_DF12 | 保存`c I + Σ h_eff[pq] a†p a_q + Σ_l λ_l (Σ G_l[pq] a†p a_q)^2`。全12 signed factorsを保存生成順で一度ずつ使用。index0–9は正、10–11は負。再分解・並替え・符号修正なし |

HDF5の`g_OF[p,q,r,s]=(ps|qr)`はchemist ERIから`transpose(0,2,3,1)`したもの。
既存spin liftへ渡すときは逆transpose`(0,3,1,2)`でchemistへ戻す。
spin順は`2p=α,2p+1=β`、InteractionOperatorのWには1/2を含む。
OpenFermionのmode0=MSBから既存`reverse_bits`を用いてQiskitのmode0=LSBへ写す。
sector行列のrow順は保存OpenFermion basis順を保持する。

核scalarは保存`2.29310124732 Ha`、hex `0x1.25845777d3feap+1`。一回だけ加える。
DFの一体補正は既存`get_chemist_two_body_coefficients`で求め、当時の
`(h+correction + (h+correction)†)/2`を照合した。このhelperは添字変換と収縮だけで、
DF分解関数を呼んでいない。元H側を黙ってHermitian化して差を消していない。

再利用した参照：既存M1のsparse CAR `annihilators/one_body/quartic`、OpenFermion
`jordan_wigner_sparse`、既存`df_linear_operator`、Track Aの独立
`occupation_df_matrix`。G²の中間状態はfull occupation空間で扱う。
新規実装は比較driverと、既存occupation ladderを使う元H quartic項の短い拡張だけ。
M1専用loader/preparationはDF10・stateを要求するため流用していない。

OpenFermionの`get_fermion_operator`/SymbolicOperator加算には追加の小係数削除がある。
cutoffなし参照では全非零tensor termをFermionOperator.termsへ直接渡し、既存sparse JWを使う。
package設定は変更しない。通常symbolic変換とH_srcとの差も別診断として保存した。

## 3. 判定基準・精度と数値差

[結果前PLAN](PLAN.md)を科学入力の数値読込前に固定した。既存M1 operator-only runnerの
入力規模基準は`m=18624`、`S=103.50446476572802`、
`tau_entry=4.2802809114947716e-10 Ha`。MO直交性は既存H4基準`1e-10`。
このtauは実装整合性のmax-entry基準で、モデル差・RPE energy予算ではない。
モデル差には合否閾値を置かず、結果を見た許容値変更は行っていない。

binary64参照に加え、保存binary64係数を整数比でexact liftし、独立occupation行列を
80/120桁で構築した。以下は120桁で行列差を先に求めた結果である。
spectral normだけはその差をbinary64へ丸めた行列の最大singular valueで求めた。
固有値・固有vector・最低energyは計算していない。全てHa単位。

| 空間・差 | max-entry | spectral norm | Frobenius | induced infinity norm |
|---|---:|---:|---:|---:|
| full256：H_phys−H_src | 1.26635e-15 | 2.46995e-15 | 1.55813e-14 | 5.49907e-15 |
| full256：H_src−H_DF12 | 9.27343e-15 | 9.27343e-15 | 5.68050e-14 | 9.27343e-15 |
| full256：H_phys−H_DF12 | 9.27343e-15 | 9.27343e-15 | 5.88339e-14 | 1.18884e-14 |
| sector36：H_phys−H_src | 1.26635e-15 | 2.46995e-15 | 6.73615e-15 | 5.49907e-15 |
| sector36：H_src−H_DF12 | 4.27883e-15 | 4.86713e-15 | 1.81031e-14 | 6.66226e-15 |
| sector36：H_phys−H_DF12 | 4.27883e-15 | 6.23343e-15 | 1.92566e-14 | 1.02878e-14 |

binary64の行列を先に引いた場合のspectral normはH_phys−DF12でfull`1.06942e-14`、
sector`7.02300e-15`だった。120桁との差は小差の減算時のroundoffを含むため、表では
高精度側で先に差を求めた値を主に表示する。80/120桁のoperator entry差は最大
`4.89e-80`で、事前の`1e-60 × max(1,max_entry(H_120))`基準を全6件で通過した。
binary64構築と120桁参照のmax-entry差はfull最大`8.44e-15`、sector最大`2.44e-15`。
これは固定binary64入力に対する算術安定性であり、SCF精度・積分の物理的誤差・
厳密interval保証や完全なforward-error boundではない。

cutoffはspin一体14成分・二体512成分の非零値を0化したが、この入力ではそのoperator差は
上表の規模だった。`1e-8`という係数閾値をoperator normの上界と取り違えない。

## 4. 再構成・対称性の確認

**最終run03は65/65 checks PASS、80/120桁安定性6/6 PASS。** 主なmax-entry residual：

| 検査 | 観測値 |
|---|---:|
| 元HのCAR対sparse JW、full | 最大6.66e-15 |
| DF CAR対既存full excitation matvec | 8.44e-15 |
| DF full対既存sector matvec | 2.00e-15 |
| DF full対独立occupation | 5.33e-15 |
| 保存h_eff対SOURCE補正＋Hermitian化 | 0 |
| signed squares対normal-order再構成、full | 4.44e-15 |
| 核scalarのexact照合／full vacuumへの一回計上 | 完全一致／residual 0 |
| MOのCᵀSC−I | 1.33e-15 |
| H_phys / H_src / H_DF12のfull Hermiticity | 3.33e-16 / 2.50e-16 / 0 |
| 各HのN、Nα、Nβ、Sz保存とsector漏れ | 全て0 |
| S²とのcommutator | 最大2.11e-15 |

一体補正の差を引算から求め直したresidualは`1.25e-16`、normal-orderの一体項と
SOURCE bare hとの差は`4.44e-16`。電子項と核scalarの規約が整合した。
保存metadataの`df_truncation_value=0.0`だけから元Hとの厳密一致を主張していない。

人工テストは7件PASS。実行中の2件のdriver誤りは
[技術修正記録](TECHNICAL_CORRECTIONS.md)に説明し、run01の失敗記録とrun02のFAIL JSON、
各SOURCE commitを保持した。run02のfull precision診断はMSB/LSBの不一致を含むため
科学的な入力不適格の根拠に使わない。修正はscalar配列形状と参照行列のbit-orderだけで、
SOURCE規約・科学入力・factor・precision・許容値は不変更。run03だけを最終資格記録とする。

計算環境：Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、
OpenFermion-PySCF0.5、Qiskit1.3.0、h5py3.16.0、mpmath1.3.0。
CPU affinity32、実行前available RAM約55 GiB、GPU driver利用不可。
既存sparse/BLASを使い、80/120桁×full/sectorの4 taskを4 processで実行した。
最終runは約14.74秒、独立full精度参照は各約2.7秒。speedup倍率は測定していない。

## 5. 判定の分離

| 判定項目 | 今回の結論 |
|---|---|
| file/SOURCE identity | 依頼されたHDF5・DF12・当時の生成SOURCEのhash一致を確認 |
| 数値的operator対応 | full/sectorの独立構築・補正・scalar・符号・順序・対称性が事前基準を通過 |
| DF/model差 | 固定入力では上表の非零差。十分小さい数値差として記録するが、厳密な同一operatorとは認定しない |
| 完全な生成来歴証明 | 未証明。当時の元h/g/MOとDF12を直接結ぶ保存hash receiptはなく、数値的な近さはこれを補わない。9月27日旧pilot原本の回収でもない |
| B2総RPEへの接続 | 元H/stateを指定する次段の準備材料は得られた。target採用・元H ground state・信号/誤差配分/成功条件は未確定であり、科学実行GOではない |

原本回収後の保存stateの状況はLOCAL/SERVER 1調査から変更なし。この元Hに結合できる
保存ground stateは見つかっていない。DF12保存DF groundやM1の新SCF stateを代用しない。
旧研究のSTOP・SOURCE・grantを解除せず、SERVER 2への統合も行っていない。

## 6. 次回の元H state solve：提案する承認範囲

**ユーザー**がH_physまたはH_srcを選び、その固定入力だけのstate solveを明示承認した場合に限る。
現在はどちらも採用していない。次の指定を結果前に固定する。

1. target定義とarray/operator hash、元HDF5・DF12 hash、SOURCE。
   raw operatorの微小anti-Hermitian residualを記録し、solve用の明示的Hermitian化を
   採るならその定義と差を先に固定する。片側三角だけを黙ってHamiltonianにしない。
2. 既存`h2_non_df_pilot.canonical_ground`のsector36 `scipy.linalg.eigh`経路を再利用する。
   固定sectorの最低固有pairだけを取得し、full256・sector36双方の固有状態残差、norm、
   sector漏れ、必要ならgapを検査する。人工参照や既存DF stateからの代用はしない。
3. 既存の候補基準はstate norm`1e-12`・residual`1e-10`。最終の採用基準をsolve前に固定し、
   phaseは最大振幅成分を実正にする既存規則を使う。OpenFermion sector順と
   bit-reversal後Qiskit full-vector順を両方保存・照合する。
4. private NPZには`state_vector(256)`、`sector_state_vector(36)`、保存OF basis indicesと
   Qiskit indicesを保存し、JSONにtarget種別、元integral/operator/state hash、SOURCE、
   精度・残差・位相規則を保存する。既存S0 array-hash規約とAgent B側で使うhash関数を
   区別し、同じ文字列だからと異なるhash定義を混同しない。
5. **Codex（その別承認後）**がAgent Bの`RPEConnectionContext`へ
   `original_hamiltonian/original_state/df_hamiltonian/sector/spin_orbital_order/source`を
   明示的に渡せる入力recordを用意する。state roleは`original_H_ground_state`、
   preparation costは研究仮定として除外。今回はRPE条件やcompiler/cost条件を補完しない。

今回の新SCF/DF分解、元H state solve、PF/RTE・sampling、回路build/compile、shot・費用算出、
GPT詳細科学レビューはいずれも0件。結果・REPORT・参照索引を専用branchへ公開し、
remote SHA・tree・全変更内容の一致を確認して**mandatory STOP**する。

[RESULT]: ../../../artifacts/b2_original_df12_correspondence/2026-11-06/run03/result.json
[LOCAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe933d779d3759c5fa63c6e95c67bbedff3f1652/docs/research/local_track_a_b2_input_recovery_20261106/REPORT.md
[SERVER1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/26aed3e1271c08eb2e938bf4d6e218db5b75dc6b/docs/research/track_a_b2_server1_input_recovery_20261106/REPORT.md
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2796d26270cf39e46b40accc4f898fe29d459819/docs/research/track_a_b2_original_state_spec_20261106/REPORT.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/docs/research/track_a_b2_rpe_implementation_20261106/REPORT.md
