# O1 M2：B-relative core-only省略誤差の限定検証

2026-11-06、SERVER 2（GPUA100-gleap02）。ユーザーの限定実行許可後、固定準備SOURCE `840ec8322d21e897fb47f3f12982af5b10bf7640` の既存runnerを**1回だけ**実行した。exit 0、実行時間0.2917秒。**限定係数上界評価を完了しmandatory STOP。物理モデルはUNASSESSED、未指定の省略予算判定はDEFERREDのまま。**

## 固定SOURCE・入力・方法

- [準備報告][P]と同じ `scripts/run_o1_m2_core_only_bound.py`／`src/trottertracks/o1_m2_core_only_bound.py`、既存B adapterと保存update005 loaderを変更せず使用した。起動前にHEAD・clean状態・4 SOURCEのbyte一致を確認し、runnerも再利用B／adapterを固定baseと照合した。実行SOURCE/hashは[新result][R]に記録している。
- 対象は直鎖H4、1 Å、STO-3G、4 spatial orbitals、保存update005 number-CDF rank8／全8 frame、元h/gと一体補正・核scalar。保存sectorは `Nα=Nβ=2`。旧LD2 tailではなく、Bの完全残差を扱う。
- input commit `aced1647b81e8d28091058cdc7b6e3e1ee54896a`。coreは `artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz`、SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。元H入力は `artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`、SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。生成SOURCEは `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`。旧B実行の保全copyを読み取り専用で再利用し、hash一致を確認した。
- Pythonは既存専用 `/tmp/track-a-o1-m2-runtime-20261106/bin/python`。Python 3.12.3、NumPy 2.5.4、SciPy 1.18.1、OpenFermion 1.8.1、Qiskit 2.6.2で旧M2と一致。共有環境は変更せず、cacheだけを `/tmp/o1_m2_core_only_science_cache_20261106` へ分離した。

数学的B参照は、保存済みbinary64 raw H0/C0 packetをexact real係数として解釈し、`P(X)=(X+X†)/2`、`H_H^ref=P(H0_bits)`、`C_H^ref=P(C0_bits)`、`R_H^ref=H_H^ref−C_H^ref` をFractionで形成する。この参照では恒等式を正確に保つ。既存adapterがbinary64で形成したB packetは別の対象として保持し、閉包／形成差を分けた。

事前固定のCAR monomial係数上界 `||O(c,h,g)||₂ ≤ |c|+2Σ|h|+2Σ|g|` を使用した。各spin個別monomialのnormが1以下、hのspin数2、gのspin数4とprefactor1/2による。係数の差と絶対和をFractionで正確に計算し、表示floatはその値を下回らない上向き丸めである。微小項・符号・scalar・一体補正を零化せず、観測結果に合わせたtolerance設定はしていない。これは精密なspectral norm／固有値の算出ではない。

## 数値結果

全exact Fractionと上向きfloatは[result][R]、単回起動のcommand・時刻・exitは[実行ログ][L]に保存した。result SHA256は `138fb7ce14fb9b4c633e521e5fa537df89843c91851334e080c4ae5eb9529f91`。

| 対象 | 保守的CAR係数norm上界 [Ha] |
|---|---:|
| 数学的参照 `R_H^ref=H_H^ref−C_H^ref` | `6.930386640756262e-08` |
| 形成済みBの `H_H−C_H` | `6.930386635118411e-08` |
| 形成済み `R_H` | `6.930386635118411e-08` |
| 形成済み `H_H−C_H−R_H`（閉包差） | `1.0442933021225735e-24` |
| 形成済み `R_H−R_H^ref`（投影／形成差） | `4.501607430616656e-16` |
| 形成済み `H_H−H_H^ref` | `0` |
| 形成済み `C_H−C_H^ref` | `4.501607420194146e-16` |
| 数学的参照 `H_H^ref−raw H0` | `5.919376153131022e-16` |

形成済みRと形成済みH−Cの表示上界が同じでも、exact Fraction上界は異なり、閉包差の上界も零へ切り捨てていない。非零の上界から実operator差の非零性を判定しない。今回の係数閉包上界と、旧Bでfloat CAR行列を形成・加減算したmax-entry閉包値は、丸め段階とnormが異なるため同一指標として数値比較しない。旧[B operator PASS][B]は引用再利用し、今回full256／sector36行列を再構成していない。係数norm上界はfull Fock全体で通用し、sectorへのcompressionにも通用する。各空間spin-lift monomialはNα/Nβを代数的に保存するため、共通Hermitian保存sector Sについて

`|E0_S(H_H^ref)−E0_S(C_H^ref)| ≤ 6.930386640756262e-08 Ha`

というB参照内の最低energy差上界が得られる。形成済みH_H/C_Hに対する別上界は `6.930386635118411e-08 Ha` である。これは実際のenergy差の測定値、元物理Hの誤差、finite-PF／RPEの誤差ではない。

## 成立したこと・保留すること

**成立した限定事項**は、固定保存packetを基点とするB-relative core-only省略誤差の保守的上界を、微小残差と形成誤差を区別して得たことである。元の物理Hamiltonianを再定義せず、`--hh-omission-budget` は指定しなかった。従って `hh_budget_status=DEFERRED`、`physical_model_status=UNASSESSED`、`physical_energy_bound=null` を保持する。C0の0.1 HaをM2の許容予算へ転用していない。

`raw H0→P(H0)` の `5.919376153131022e-16 Ha` は、保存raw非Hermitian operatorからの変更norm上界である。元Hermitian物理target H_physへのenergy boundではない。H_physの定義・保存値との対応、参照／積分生成誤差、model許容上限が不足しており、その評価は保留する。raw core packetの形成後bitsに対する計算なので、正確なJ/U縮約／native coreとの対応も未検証である。

必要な省略予算がこの上界以上で、別途model差・native対応・PF／信号・統計条件も満たせるなら、core-onlyを次の比較候補にできる。予算が上界より小さくても、この保守上界だけでは実誤差超過を証明できず `BOUND_INCONCLUSIVE` となる。より鋭い評価や補正が必要かは未判定で、**残差random化が必須という結論にはならない**。補正が必要でもrandom化は選択肢の一つである。

強決定論PFとの比較には、全8 frame／一体項の決定論core費、native／controlled相対位相、元H stateでのPF信号半径・誤差、全必要RPE roundの同精度／成功条件が残る。元Hの厳密基底状態・状態準備費用除外の仮定を維持するが、それをC_Hの基底状態や単位半径へ置き換えない。core-only／同J/U強D／強DF-PFのfull no-prep費用を同条件で取得していないため、費用優位性は判定しない。既存[研究CのNO-GO][S]と旧FAIL・許容値は維持する。

## 実行範囲と次の条件

今回行った科学処理は、固定2入力のload、既存raw core／B形成と、上記係数上界の計算1回だけである。全H4行列・eigensolve・state solve・PF/RTE simulation・回路build/compile・sampling・shot／価格／総費用計算・新fit・rank探索は行っていない。計算は短い係数集約であり、新たな科学worker分割は不要だった。CPU／RAMの人工上限、共有設定や他ユーザーへの変更はない。準備段階の人工49 PASSは[P]の記録であり、今回再実行していない。

次は**ユーザー**がH_phys／参照精度、model・省略energy予算と全RPE比較条件、次の限定検証の要否を決める。**Codex**は別途指示・必要な科学実行承認がある場合だけnative／PF／費用の次段階を行う。現在の許可範囲では追加科学作業なし。新result／ログ／本報告を専用branchへ公開・照合して**mandatory STOP**する。旧SOURCE・入力・raw・結果・grant・marker・STOPは変更しない。

[P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/840ec8322d21e897fb47f3f12982af5b10bf7640/docs/research/track_a_o1_m2_core_only_preparation_20261106/REPORT.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68ef4feee3835ed7135d254ac879e8d88c69c895/docs/research/track_a_o1_m2_hermitian_b_science_20261106/REPORT.md
[S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f891f6e3265e4f0124c598fddfb7775bc1eddd1b/docs/research/track_a_o1_m2_improvement_survey_20261106/REPORT.md
[R]: ../../../artifacts/track_a_o1_m2_core_only_science_20261106/run01/result.json
[L]: ../../../artifacts/track_a_o1_m2_core_only_science_20261106/run01/execution.log
