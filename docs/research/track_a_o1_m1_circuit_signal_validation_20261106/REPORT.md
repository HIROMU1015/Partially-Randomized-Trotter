# O1 M1：実H4 Gaussian／controlled回路・PF/finite-RTE信号

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**固定SOURCEの486数値検査はPASS。実回路の選定eventと解析的finite meanの接続を確認し、mandatory STOP。** 全H4 eventの総列挙、sampling、shot・費用は実行していない。これは下記保存入力・4条件でのlocal numerical correctnessであり、energy精度資格・全RPE性能・総費用改善の判定ではない。

## SOURCE・入力・実行条件

実行SOURCEは`d31413eb5f1d25e9e78df1a5dc37cb522acbf937`。新worktree/branchは`track-a-o1-m1-circuit-signal-validation-20261106` / `codex/track-a-o1-m1-circuit-signal-validation-20261106`。実行中・実行後ともSOURCEを変更していない。準備SOURCEを新結果と区別して保存した。

- [新raw result](../../../artifacts/track_a_o1_m1_circuit_signals/2026-11-06/run01/result.json)：SHA256 `b75f811dbe3ee591bbbb290823ed30720735aa98f661adf750e33ef43239338c`。
- [既存runner](../../../scripts/validate_o1_m1_circuit_signals.py)、[検証module](../../../src/trotterlib/o1_m1_circuit_signal_validation.py)、前段CAR/exterior-power参照、既存Gaussian/DF event builder、finite-RTE momentsとpaired polynomialを再利用。
- 入力commit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`。SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`を実行前とrunner内で照合。
- linear H4、間隔1Å、STO-3G、中性singlet、4空間/8 interleaved spin orbital。保存DF rank10、共通HD=T_eff、HR=V_square全10fragment、fragment LD0。
- 元H保存`state`のみ。hash `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`、raw squared norm `1.0000000000000002`。信号はそのnormで割る。DF_stateへの変更・state再生成はない。
- full256、controlledはsystem8＋ancilla-last、`diag(I,U)`へ直接比較。途中sector射影なし。同じGaussian T_eff half programを両辞書へ使用。
- identity方針は両辞書とも`faithful_identity_in_tail`。核scalarは一度、T_eff電子identityはGaussian保持block内、tail identityはRTE内。
- delta=0.1,0.05、q=2、r=2、K=0,2、計4点。T=q delta=0.2,0.1。新しい辞書探索・再DF・新fit・rank探索はない。

Python3.12.3、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、Qiskit1.3.0、qiskit-nature0.7.2、qiskit-algorithms0.3.1。準備で用意した専用runtimeとwheel directoryをread-only再利用し、cacheは今回専用pathに置いた。共有設定・package・他ユーザーのprocessは変更していない。CPU/RAM/workerの独自上限は設定していない。[timing](../../../artifacts/track_a_o1_m1_circuit_signals/2026-11-06/run01/timing.txt)はwall98.07秒、user781.57秒、sys949.53秒。[console](../../../artifacts/track_a_o1_m1_circuit_signals/2026-11-06/run01/console.log)は空、終了code0。

## 事前の検査基準

準備commitに固定したbinary64・係数scale・program長の式をそのまま用いた。実入力scale `103.50446951296075`、m18592。operator判定値 **4.272926672916701e-10**、basis判定値 **6.548361852765083e-11**（最大basis instruction数36）、implementation判定値 **7.99037287835423e-9**。式・rtol0・詳細はrawの`tolerance`と準備commitの報告にある。

人工テストatolを移送せず、差を見て閾値を変更していない。これらは数値検査用の保守的工程scaleであり、rigorous rounding enclosure・energy allowanceではない。486検査すべてPASSでもPF/RTE近似誤差がゼロという意味ではない。

## Gaussian・controlled operatorとrelative phase

実backendは`qiskit_nature.second_q.circuit.library.bogoliubov_transform`。basisのgate familyは`rz`と`xx_plus_yy`で、dense UnitaryGate fallbackを使っていない。10全fragment/basisを検査した。

| 検査 | 最大entry差 |
|---|---:|
| Gaussian full Fock作用と独立exterior-power（vacuum phaseを明示） | 7.771561172376096e-16 |
| basisが保存Gを対角化する残差 | 2.5288826380884394e-11 |
| Gaussian fragment operatorと独立CAR | 5.329070518200751e-15 |
| DF辞書和とCAR HR | 5.329070518200751e-15 |
| Pauli辞書和とCAR HR | 5.329070518200751e-15 |
| 共通T半step／controlled T半step | 6.753223014464258e-16 / 7.447602459741819e-16 |
| 選定DF event（noncontrolled/controlled全体） | 1.6739892789816728e-15 |
| 選定Pauli event／controlled event | 8.082545620880531e-16 / 9.420554752102651e-16 |
| 核phase＋保持half＋event＋保持halfのcontrolled全step | 1.4133557720257279e-15 |

basis単体のvacuum/global phaseは記録し、Fock作用・保存G対角化・fragmentを別々に確認した。basisのU†/controlled diagonal/Uではこのphaseが相殺する。**evolution/eventのglobal phaseを後から合わせていない。** eventは`(-1)^(order/2)`、係数sign、I/Z/ZZ、異なるfragment/basisを覆う決定的選定である。

各deltaでDF order0が48、order2が48、Pauli order0が3、order2が3。計204eventを各noncontrolled/controlledで検査し、さらに4つのcontrolled全stepを検査した。これは確率標本ではなく、全event検査の主張でもない。basis G対角化の最大2.53e-11と、重みを含むfragment/operator・実回路の差を混同しない。

## 補正mean・複素信号・normalization

P/DF補正meanの最大entry差は **3.5220338128421975e-15**。実装meanと独立CAR R＋exact T halfによるnominal meanの最大差は **4.348688458596125e-15**。各tailの既存momentsと独立paired-polynomial recurrenceとの差は最大7.447602459741819e-16だった。

Pのlambda_Rは **18.658676520210765**、DFは **18.403365082066507**。同じ物理HR・補正meanでも、辞書のL1とfinite normalizationが異なる。以下のBはr short-stepをq回繰り返した全normalization、attenuationは1/Bである。

| delta | K | B(P) | B(DF) | attenuation(P) | attenuation(DF) |
|---|---:|---:|---:|---:|---:|
| 0.1 | 0 | 3.498267192300418 | 3.410336405308171 | 0.2858558094707489 | 0.2932262044423257 |
| 0.1 | 2 | 11.053071892925754 | 10.538683695318209 | 0.09047258623550845 | 0.0948885106442893 |
| 0.05 | 0 | 1.482528770911775 | 1.4681621312790971 | 0.6745231658371033 | 0.6811236842955326 |
| 0.05 | 2 | 2.168843285494364 | 2.128311537482983 | 0.4610752684106731 | 0.4698560254870561 |

| delta | K | 補正複素信号（P、DFも丸め内で一致） | raw半径(P) | raw半径(DF) |
|---|---:|---|---:|---:|
| 0.1 | 0 | 0.9763710647918844 + 0.46410411806158686 i | 0.309027594060359 | 0.3169954413101943 |
| 0.1 | 2 | 0.9073792625655954 + 0.41967353413085956 i | 0.09044829114666462 | 0.0948630297235786 |
| 0.05 | 0 | 0.9958240232783243 + 0.22055847455052463 i | 0.687984374738047 | 0.6947166172977726 |
| 0.05 | 2 | 0.9766098130532765 + 0.2149403380979754 i | 0.461067447561118 | 0.46984805569656907 |

raw複素値、phase、radius、正規化の詳細はrawに保存した。K0の補正半径が1を超えるのは、normalizationを戻した有限Taylor多項式が非unitaryだからであり、確率やunitary振幅と取り違えない。raw半径を1へ丸めたり補正meanを再正規化したりしていない。この4点ではDFのattenuationが大きいが、shot数・総費用の改善は評価していない。

## PF誤差・finite-RTE誤差・実装差の区別

PFはexact central HRとexact T halfのS2をDF exact evolutionと比較した。RTEは独立CAR nominal corrected meanをそのS2と比較した。実装差は上記別検査であり、下表へ混ぜていない。q=2固定なので、deltaを半分にするとTも半分になる。下表の変化を固定Tでの収束次数fitとは呼ばない。

| delta | K | PF operator最大entry差 | PF複素信号差 | RTE operator最大entry差 | RTE複素信号差 | RTE wrapped phase差(rad) |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 0 | 5.702235668706494e-5 | 2.298247265113365e-5 | 1.6367559038170632 | 0.08178848807794067 | 0.010467773563422258 |
| 0.1 | 2 | 5.702235668706494e-5 | 2.298247265113365e-5 | 0.05405150364196331 | 0.0002722254091212927 | 4.469251878661927e-5 |
| 0.05 | 0 | 7.144497527940336e-6 | 2.8702951797027386e-6 | 0.31846742866844563 | 0.02000170637825556 | 0.0013288918875760689 |
| 0.05 | 2 | 7.144497527940336e-6 | 2.8702951797027386e-6 | 0.003451478903275492 | 1.701997105668461e-5 | 1.4016999696884214e-6 |

元H exactとDF exactのoperator差はT0.2で最大1.847779371410367e-15、T0.1で8.082545620880531e-16。元stateでの信号差はbinary64で0と観測されたが、厳密同一やground energy誤差保証と主張しない。PF/RTEの精度受理予算を新たに採択していないため、このgrid全体をenergy-qualifiedとは判定しない。

## 再現command・未解決・STOP

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-a-o1-m1-circuit-signal-preparation-20261106/.venv/gaussian_deps \
XDG_CACHE_HOME=/tmp/o1-m1-signal-run-cache-20261106 \
MPLCONFIGDIR=/tmp/o1-m1-signal-run-cache-20261106/mpl \
NUMBA_CACHE_DIR=/tmp/o1-m1-signal-run-cache-20261106/numba \
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-a-o1-m1-minimal-implementation-20261106/.venv/bin/python \
scripts/validate_o1_m1_circuit_signals.py \
  --snapshot ../track-a-o1-m1-h4-operator-correctness-20261106/.venv/m1_input_snapshot.npz \
  --step-times 0.1 0.05 --q 2 --r 2 --K-values 0 2 \
  --output artifacts/track_a_o1_m1_circuit_signals/2026-11-06/run01/result.json
```

新科学実行は上記1回。固定SOURCE変更・旧runのretry/resumeはなく、準備時14人工テストの後に追加の人工条件探索もしていない。主agentによる検査であり、独立研究レビューではない。

未解決は、選定eventを超えた実回路meanの検査、別time/q/r/Kへの移送、許容signal/energy誤差と比較条件の採択である。今回のmean PASSだけでsampling law実験やcost性能を認定しない。**次の担当はユーザー**：必要な次scope・精度条件を選ぶ。**Codex**は新たに許可された範囲だけを実行する。今回は結果・本報告を独立branchへ公開し、remote SHA・全変更file内容・cleanを照合して**mandatory STOP**。sampling、回路cost compile/transpile、RPE shots/総費用へ自動進行しない。
