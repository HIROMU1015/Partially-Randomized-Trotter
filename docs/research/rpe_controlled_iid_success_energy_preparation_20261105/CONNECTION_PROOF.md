# Controlled IID平均から確率・数学的energyへの接続

2026-11-05 JST / LOCAL。SOURCE調査・数式の接続・人工テストのみ。
**新しい実H4のt・確率・総和・95%判定は未実行。**

対象はlinear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D=3、
δ=binary64 `.02`、K=2、8 system qubit、4電子sector70、保存physical state normalized ray。
m=0..17、q=2^m、保存r∈{1,2,4,8,16,32}、N/α/β/ε/μ₀/E_refを維持する。
`extract_identity_phase`、`coefficient_atol=1e-12`、保存component/basis/sign、
`support_for_source_basis_run_le_1`、uniform53 component/order lawも同一。

## 1. 平均の保証対象とSOURCE identity

ν_refは既存categorical referenceの真の平均、ν_ctrlは指定ideal logical controlled circuitを
独立uniform53で平均したHadamard測定の平均。どちらもcosine=Re(z)、sine=+Im(z)、
bit0→+1、bit1→−1、U(t)=exp(−itH)の規約を用いる。
保存中心μ₀はbinary64値が表すexact rationalであり、真の平均とは区別する。

[最新平均結果1789a8c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1789a8ca731a1de631e80a34d2b8c90657d24458/docs/research/rpe_support_weighted_transfer_execution_20261105/REPORT.md)
はSOURCE `7a39a78eccdbe457786978e14d36b2f0d7abc290`の平均専用証明を使用する。
各trajectoryの条件付き測定平均をlawに沿って平均した結果であり、任意trajectoryへの一様保証ではない。
旧pointwise一様保証18/36と平均保証36/36は別の結果として保全する。

[FIXED_BINDINGS.json](FIXED_BINDINGS.json)はcommit/path/blob/SHA256/bytesを固定する。
[run_controlled_success.py](run_controlled_success.py)のdefaultは、既存`base.static_inputs()`で
Hamiltonian/stateの保存byte、law、r/q/K、signal τ、production SOURCEと測定規約の根拠を照合し、
次に全36軸について旧categorical原上端・保存中心・N/ε/αと新平均上端を照合する。
旧DとA/Cに渡されたround/axis/H/stateファイルのhashも同一である。
NPZ数値のdecode、平均包含の再評価、回路生成は行わない。

入力selectorは以下の通り。m17のU_catは既存全36軸表に含まれる再利用値である。

| 入力 | selector・照合 |
|---|---|
| 保存round | `ROUNDS.rounds[m]`：m/q/r/K/δ/μ₀/axes.N,epsilon,alpha/β |
| U_cat | `CAT_AXES`の一意な(m,axis)。既存raw categoricalと旧Aのaxesにも一致 |
| Δ_mean | `MEAN_RESULT.rounds[m].mean_extra_upper`。同roundの両軸extraとMEAN_AXESに一致 |
| 成功事象G | 旧D入力のFIXED_ROUNDS/AXESと同じhash。εは旧包含と同じexact decimal |
| 共通reference | 旧D/A/Cのconstantsが一致。旧Bのphysical_state.energy/fingerprintとも一致 |

## 2. 新しく必要になる確率上界

旧U_catと新Δ_meanの証明から、三角不等式により

    |ν_ctrl,m,a − μ₀,m,a| ≤ U_cat,m,a + Δ_mean,m = U_ctrl,m,a.

Δ_meanには既存のboundary/lowering移送を含む。新たなπ64項、旧uniform追加上界、
shot half-width、既に拡張したintervalをここへ加算しない。
π64 event位相誤差は旧categorical包含に計上済み。
runnerが足す項は、元の`U_cat_upper`と`mean_extra_upper`の**2つだけ**である。

成功事象は旧Dと同じ開座標事象

    G = ∩_(m,a) {|hatμ_m,a − μ₀,m,a| < ε_m,a}.

各軸のN個のshotを同じν_ctrlを持つ独立な±1変数とする。
t=ε−U_ctrl>0なら、三角不等式とHoeffdingより

    {|hatμ−μ₀| ≥ ε} ⊂ {|hatμ−ν_ctrl| ≥ t},
    Pr(|hatμ−μ₀| ≥ ε) ≤ p = 2 exp(−N t²/2).
    Pr(Gᶜ) ≤ Σ_36 p,   Pr(G) ≥ max(0,1−Σ_36 p).

各軸内では、各shotの全trajectoryにfreshな独立uniform53 component/order選択を使い、
同じ保存rayからideal Born測定するIIDモデルを仮定する。trajectoryを固定して何回も測定する
条件付き平均へ平均専用Δ_meanをそのまま適用しない。
round間・axis間の独立性はunion boundに不要。例えば軸間の相関があっても、
各軸内の独立性と同じ周辺分布が維持されるならこの経路を妨げない。
固定PCG64列、seed、非重複列、有限samplingテストはこの数学的IID仮定の証明にならない。

`alpha_adapter.failure_upper`を無変更で再利用し、Arb160でtとexpを外向き包含する。
U_ctrlは先にexact rationalで一度加算する。tのstrict正値が証明できなければ停止する。
`aggregate`も無変更で、36個のdyadic上端をexact rationalで足し、**≤1/20**を比較する。
保存α/N/εや成功目標は再配分しない。

最新の36軸局所strict条件は、この新しい総和の判定結果ではない。
旧Aの成功下界0.950307030063252182は元categoricalモデルの結果としてのみ残す。
新controlledモデルの値への転記、近似差分による95%推定、実H4のtの試算は今回行っていない。

## 3. Gから数学的decoderへの既存証明

[旧D証明SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6942e5176d2dec115ac3c271e48f84377abda915/docs/research/rpe_branch_energy_preparation_20261104/PROOF.md)
は、G上の**全座標**を包含する矩形・円板の幾何を扱う。
ν_ctrl自体が単一のunit-radius信号exp(iqφ)であるとは仮定しない。
同じμ₀/ε/qと共通φ_ref=−E_ref δへ接続できれば、Gの確率を生むモデルが変わっても決定論的含意は同じ。

[旧D結果89cb94f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89cb94fe4db1d39d574877c3bc2db79a6d7a5f9f/docs/research/rpe_branch_energy_execution_20261104/REPORT.md)
では全18roundの正半径とb_m<保存β_RPE、全17隣接対の

    M_m = π − 2 b_(m−1) − b_m > 0

が成立済み。q₀=1、q_m=2q_(m−1)の数学的nearest-branch decoderでは正解rootがstrictに一意で、
最終円周誤差はe_phi≤b_17/q_17となる。この保存結果を読み、再計算しない。
旧Dの総合判定は保存Σαの僅かな超過によりUNKNOWN/STOPだったことも変更しない。
今回はその総合ラベルをPASSに読み替えず、既存の決定論的含意だけを再利用する。

## 4. Alias・energy・reference

EとE+2πk/δは信号から区別できない。旧Dと同じ外部alias priorと代表元の規約を条件とする。
referenceは保存binary64 `−2.1663874486347616 Ha`、δはbinary64 `.02`のexact値であり、
δをdecimal1/50へ置換しない。旧DのA=π−|φ_ref|>0とe_phi<Aが成立済みなので、
最終円周誤差を同じ実区間へliftし、数学的変換E_hat=−φ_hat/δを使える。
外部priorの物理的正当性とFloat64変換誤差は別問題である。

[旧A/C結果61f569c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/61f569c32a94c2b698df10a62a51e515b12160ed/docs/research/rpe_three_gaps_execution_20261104/REPORT.md)
のCは、保存DF binary64係数をexactとする
H=cI+T(h)+Σλ_l T(g_l)²の完備N=4 sector70の最小固有値E_*を包含した。
70個の正LDL* pivotとRayleigh上界によるreference誤差η_ref、およびsector energyのalias条件は成立済み。
したがってG上で

    |E_hat−E_*| ≤ e_phi/δ + η_ref
                 ≤ 既存Cのreference込みenergy上端
                 ≤ 0.000138689200360119854 Ha（保存表示）.

これは保存目標binary64 `0.00015936001019904 Ha`とのstrict比較が成立済みの同じ条件付き上界。
元exact endpointはC.jsonからそのまま引用し、加算・幾何・固有値・目標比較を再評価しない。
**Δ_meanはGの確率へ一度だけ入り、G上の位相・energyへ再加算しない。**
PF/RTE系統差やη_refも各段階へ重ねて加えない。

この対象は保存DF sector ground energyであり、上流積分/DF構築誤差、他sector、
真の分子energy、実state preparation、compiled/noisy実回路の保証ではない。
Float64 decoderの全域誤差証明は未取得のままで、本作業では再開しない。

## 5. 再利用と今回の差分

旧Aの`alpha_adapter.py`と`common.py`、依存する旧Dの`adapter.py`は
このC系列branchに未存在だったため、固定commitから**byte-identicalに3ファイルだけ復元**した。
旧SOURCEやrawを編集せず、既存commonのimport構造をそのまま使う。
geometry/reference/Float64 runnerは復元・実行しない。
新規部分は36軸のliteral join、U_cat+Δ_meanの入力接続、部分結果を保存して停止するrunnerと人工テスト。
認可証明書、grant機構、承認記録は作らない。

形式状態は`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持する。
