# RPE m17：欠損置換だけではguard認定に届かない

2026-11-04 JST。**現行係数を使う独立・積型のノルム伝播では、欠損を理想的にゼロにしても
単位raw norm時のguardを認定できない。完全なnorm-only adapterは見送り、
保存stateに対する最小の証明限界診断SOURCEを準備した。実H4の数値評価は未実行。**

基点：[静的検討](../rpe_m17_guard_bound_static_assessment_20261104/GUARD_BOUND_ASSESSMENT.md)、
commit `ded65600f7e5517166b432587e28c4f4bac9d6bf`。
旧科学SOURCE `c3a5cc387b8962b63782dc71ab175147f2d87e33`と
結果 `81fa6813194e4e14189092626f66dd10793225e1`は変更しない。

対象はlinear H4 / 隣接1.00 Å / STO-3G / 保存legacy DF12 / B raw physical state /
8 qubit・4電子sector70、Partial L_D3 / 単一δ=.02 / m17 q131072 r32 K2。
extract_identity_phase、coefficient_atol=1e-12、保存N411/axis・α・β・εを維持する。
数学的IID uniform53を仮定するModel IIの局所bridgeを対象とし、固定PCG64のIIDを認定しない。

## 欠損ゼロの場合の楽観的な式

[現行kernel係数](../rpe_m17_openblas_kernel_bound_preflight_20261104/kernel_bounds.py)の
70×70 matvec/dotは、u=2^-53、D=156に対し

    κ = κ_mv = κ_dot = 2 D u / (1−D u) = 312u/(1−156u).

前案のnorm-only作用係数は`g_A=sqrt(1+ρ_A)+κ||A64||F`、
`ρ_A≥||A64†A64−I||₂`。欠損をゼロとした厳密unitaryの仮想ケースなら`||A64||F=√70`。
RTE eventをすべてノルム係数1・加法項0へ楽観化し、F/Rだけを2q回数えると、
raw norm²をn²として候補signal上端は少なくとも

    n² (1+κ_dot) (1+κ√70)^(2q).

これは実signalの下限ではなく、**この方式が作る上界の下限**である。
√70≥8とBernoulli不等式から、厳密rationalの弱い下限

    n² Λ,  Λ=(1+κ_dot)(1+2q·8κ)
    Λ=5070602769257503269825519024655/5070602400912741965601345373681
      ≈1.00000007264319545897

を得る。qは固定整数131072であり、保存context・norm・平均・予算を使っていない。
単位normの場合、この下限だけで座標guard `float(1.+1e-12)`と
magnitude guard `float(1.+1e-10)`の両方を超える。
underflowやrotationの丸めを含めれば、同じ非負項を足す方式の上界はさらに増える。
それらを実行誤差の式から取り除いてよい、という意味ではない。

## 非unitary数値行列でも使える下限

上のΛは、欠損ゼロという仮定に限らず、前案の**絶対欠損型係数**のすべての数値Aに適用できる。
一般のd×d行列でc=⌊√d⌋、κ≥0、`(1+cκ)²≤2`とする。この条件は固定係数でexactに確認した。
τをAの最小特異値とすると、Frobenius上界f≥√d·τ≥cτ。

- τ≥1なら、sqrt(1+ρ)≥1、f≥cから`g_A≥1+cκ`。
- 0≤τ≤1なら、ρ≥1−τ²から`g_A≥sqrt(2−τ²)+cκτ`。
  この右辺はτ上の凹関数で、端点値は√2と1+cκ。
  上の係数条件により、区間全体で`g_A≥1+cκ`。

したがってd=70なら、どのF/R64でもこの形式のgは1+8κ以上。
rotationも`g_rot=sqrt(1+ρ_T)+E_rel≥1`、加法項は非負とする前案の範囲では、
全列と最後のdotから`M_candidate≥n²Λ`が従う。
半径を含む上向きの包含評価でgを作っても、この下限は小さくならない。

これは収縮を許す直接spectral上界、F/R間の相関、状態依存の包絡、
別の丸め証明などを否定する定理ではない。
特に`||A†A−I||`という絶対欠損から作るsqrt(1+ρ)は、実際のAが収縮しても1未満にならない。
本結論を「現行kernelでいかなるguard証明も不可能」へ拡張しない。

## raw state条件と判断の限界

実行SOURCEと同じRNE binary64閾値Cをexact rationalで使うと、

    C = 562949953421875/562949953421312
    n² > C/Λ ≈0.99999992735780990689

なら、この積型上界による座標guard認定は不可能。
等号では、この弱い下限だけからは断定しない。
今回、保存ψや保存norm²をロードしてこの条件へ代入していない。
実contextへの条件照合は未実行であり、実abortの有無もUNKNOWN。
単位norm付近では方式自体の限界があるため、全221行列の新欠損評価は準備しない。
上界の数値が旧約2.55e-6から改善する余地と、guardを認定する能力は区別する。

## 実装・再利用・代替方針

[run_norm_proof_diagnostic.py](run_norm_proof_diagnostic.py)は新しい最小診断runner。
`proof_floor`は既存`path_bounds`を再利用し、q/r長列を実行せずBernoulli下限をexact rationalで計算する。
`--describe`/defaultはSOURCEと入力のbyte照合および上記静的定数のみ。
将来の許可済み`--run`だけが保存contextのψを一度ロードし、dyadic norm²を厳密和で求め、
下限が閾値を超えるか判定する。他の220行列memberはdecodeしない。
これは仮想測定をexact算術へ変更するものではなく、既存float64プログラムに関する証明の診断である。

既存の`fixed_bytes`、`path_bounds`、`validate_array`、atomic `write_json`を再利用する。
旧SOURCE内の作用誤差・Growth・rotation・全support有限性・complex abs・座標guard・strict判定は変更しない。
ノルム係数を`signal_error`へ渡す経路も追加しない。
今回のrunnerは中間有限性やguardを認定せず、そのproofフラグをfalse、両Uをnullに固定する。
実BLAS作用を行わない診断であり、現在のdispatch/FP modeを確認したとは主張しない。
将来のbridge認定runnerでは旧runtime/finite/abs/guard条件が引き続き必要。

代替方針の第一候補は、**同じkernelの証明を鋭くする技術検討**。
κ||A||Fを各作用で独立に加算すること、Cauchyでsignalを覆うこと、
abs欠損で収縮を失うことのどれを改善できるか、SOURCE・数式と人工例で調べる。
誤差の相殺や直交性を経験的に仮定せず、全supportに適用できる根拠が必要。
kernel変更、guard緩和、renormalization、sampling増加、α/β/ε変更は自動採用しない。
次の科学計算を実行する前に、この静的限界で不要と分かる欠損評価を避ける。

## 到達点・次の担当・STOP

新人工11＋既存24、計35件PASS。非unitary行列、非Hermitian rotation、作用誤差との区別、
長列・境界・underflow・未判定/中断STOPを確認した。詳細は[VALIDATION.txt](VALIDATION.txt)。
旧ideal包含・旧結果・SOURCE/raw/grant/ledger/marker/STOPは保全し、新H4評価は0。

限定診断のSOURCE・入力・精度・call範囲は[科学実行計画](SCIENTIFIC_EXECUTION_PLAN.md)に固定した。
bridge認定用のnorm-only adapterは未実装で、科学的guard改善は未達成。
`U_virtual_IID_upper=null`、`U_exec_upper=null`、全体`UNDETERMINED`を維持する。

次は**ユーザー**が、この診断を実行して保存stateへの限界適用を記録する必要があるか判断する。
実行する場合は公開固定SOURCEへの別途明示承認が必要で、**Codex**が許可後に単一passを実施する。
診断を省き、同じkernelの証明改善を技術作業として依頼する選択肢もある。
重要な独立科学レビューは今回未実施で、必要になれば独立の許可範囲とする。
専用branchへのcommit/push・remote SHA/公開blob照合後に**mandatory STOP**する。
