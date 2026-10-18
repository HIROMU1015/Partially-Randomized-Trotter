# Controlled回路・位相・native成立条件

2026-10-18 JST。既存SOURCEと既知の共役恒等式を読む作業に限定した。新しい回路build、compile、synthesis、行列検査は行っていない。

## 1. 単一Qとその回転

unsigned成分を`T=B P B†`、`P=Z_p`または`Z_pZ_q`、`Q=sT, s=±1`とする。controlをcと書くと、

\[
C(T)=(I_c\otimes B)\,C(P)\,(I_c\otimes B^\dagger),
\]
\[
C(e^{-i\phi Q})=(I_c\otimes B)\,
C(e^{-is\phi P})\,(I_c\otimes B^\dagger).
\]

control=0ではbasisの対が相殺し、control=1では目的の共役作用になる。時間順は右から`B† → controlled central → B`。Bを先に適用する規約へ勝手に反転しない。SOURCEはinverse時にgate順も反転する。[S10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_qiskit.py#L448-L502)。

| 操作 | 既存builderの中央primitive | 位相・制約 |
|---|---|---|
| `T=B Z_p B†` | Z、controlledではCZ | Bの両側は無制御 |
| `T=B Z_pZ_q B†` | 2個のZ、controlledでは2個のCZ | `p≠q`。CZ(p,q)単独とは異なる |
| `Q=sT`のproduct role | 上記unsigned作用＋s=-1ならscalar π | control時はancillaのPhase(π) |
| `exp(-iφQ)`、Z support | `RZ(2sφ)`、またはそのcontrolled gate | Qiskitのhalf-angle規約 |
| `exp(-iφQ)`、ZZ support | `RZZ(2sφ)`、またはそのcontrolled gate | productのsign位相を再び加えない |
| identity product | signのscalar位相のみ | s=-1はcontrol相対位相 |
| identity rotation | scalar `exp(-isφ)` | ancillaへPhase(-sφ) |

これはprimitiveを作れる既存SOURCEの確認であり、特定G6角度のnative gate数やT数を新たに取得した結果ではない。未知unitary black-boxだけからcontrolled版を無料で生成することも仮定しない。

## 2. G6 eventの順序とphase

既存G6監査のeventは、even parent word uについて

\[
V_{u,i}=(-i\sigma)^{|u|}e^{-i\sigma\phi_u Q_i}Q(u),
\quad Q(u)=Q_{u_1}\cdots Q_{u_\ell},\quad \tan\phi_u=s_u/a_u.
\]

この語の添字はoperatorの左から右である。回路時間順では`Q_uℓ,…,Q_u1`の後に終端回転を適用する。`digital_parent`の`circuit_word_time_order`もreversed wordであり、終端回転が必要なのは単一Q_iである。語全体をHermitian generatorにはしない。[S01](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py#L214-L239)、[S02 §§1,6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)。

even長のparent phaseは`(-1)^(ℓ/2)`である。符号付きproductのscalar、signed終端回転、抽出identityのphaseとは別に保持する。adjointを取るときは全積の順序を反転し、角度とscalar phaseも共役する。負時間を角度の絶対値だけへ置き換えない。

既存builderはevent phaseとproduct signを集め、controlledならancillaへ`PhaseGate(accumulated_phase)`、uncontrolledならglobal phaseへ戻す。`unsigned_rotation_angle`は歴史的な名前と異なり、実際は`coefficient_sign * rotation_angle`である。[S10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_qiskit.py#L275-L362)、[S15](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/rte.py#L628-L643)。G6 eventのこの経路へのbindingは未確認である。

## 3. 定数shiftとprojectorの位相

Hamiltonianのconstant Eと抽出identity c_Iは、時間hに対してscalar`exp[-ih(E+c_I)]`を与える。controlledではancillaへの`Phase[-h(E+c_I)]`が必要。既存partial-S2の`_append_step_phases`にその経路がある。[S12](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_partial_s2.py#L875-L891)。

`exp(iα n)=Phase(α)=exp(iα/2)RZ(α)`なので、occupation phaseをRZだけへ置き換えるとscalarが変わる。共役basisのglobal phaseは厳密なinverseと対で相殺するが、中央作用や全eventのscalarは別である。`sΓ(M)`を`Γ(sM)`へ置き換えない点も[既存Gaussian監査S07 §§3–4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md)に従う。

## 4. 共有できる範囲

現行`raw_adjacent_equal_basis`は、連続する非identity applicationのbasis IDとcanonical hashが一致すると境界のforward/inverseを除く。event内だけでなくevent間にも働く。identityはactive basisを閉じ、異なるbasisは閉じて開き直す。今回、新しい並べ替えや異基底圧縮を提案しない。[S23](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/df_rte_event_circuit_api.md)。

step境界ではDのsweepやscalar処理などを含む実際の隣接性まで確認しなければならない。同じ基底が後で再出現することだけでは相殺できない。G6ではlawが変わるので、旧ordinary trajectoryのbasis共有頻度・費用をG6に移せない。比較対象にも同じ合法な共有・compiler最適化を与える。

## 5. Workspace、入力取得、誤差、費用

| 論点 | 確認済みの範囲 | 未確立の範囲 |
|---|---|---|
| quantum register | 既存DF builderはsystem外の1 control ancillaを要求 | G6全wrapperのpeak workspaceとFT配置・routing |
| basis取得 | DF model→orbital diagonalization→Gaussian gate列の既存経路 | 今回の分子固有取得時間、G6向けstrict certificate |
| orbital列 | 既存監査にreflection/active-spanの式 | registryからNx r列を得るproduction adapter・thin実装 |
| angle取得 | G6 packetのsymbolic tangentと有限bit算術監査 | 新しいword依存角度ごとのstrict native synthesisとcost oracle |
| coefficient bias | dyadic proposal補正と近似d由来biasを区別する既存理論 | DF float→rational入力の誤差との統合 |
| native error | Q/RZ/RZZ loweringがSOURCEに存在 | 全G6 eventの位相込み誤差、basis・control・synthesisの合成認証 |
| 資源 | 必要なbasis、CZ、controlled-RZ/RZZ、ancilla phaseを特定 | 今回の実T/CX/depth、CPU時間、分子PR全costはNOT_ESTABLISHED |

primitiveに一様な誤差δ_eが別途認証されていれば、weighted meanの実装誤差には`sum_e p_e |w_e| δ_e`が関係する。finite-bit係数bias、入力表現誤差、Taylor truncation、決定論PF bias、QPE/RPE統計誤差を混ぜない。既存G6監査のfinite-bit meanは一般に元のfinite meanと厳密一致せず、近似biasを持つ。[S03 §§3–7](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md)。

`controlled primitiveのSOURCEあり`、`G6でその回路をbuild済み`、`strict native誤差と費用を認証済み`は三つの別の状態である。今回確認したのは最初の状態と残る条件である。
