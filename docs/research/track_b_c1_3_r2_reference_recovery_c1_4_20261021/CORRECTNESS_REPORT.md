# C1-3-R2：限定correctness報告

**`C1_3_CORRECTNESS_PASS`、新規1回・retry 0。** SOURCE `2be82f7bff78b7d11ffb11013b03a49e296acd1d`、新authorization `324634a37538eea2bf6153467fa0216ffa8a5c56`。固定後のSOURCE/precision/event/threshold/runtime/model/schedule変更はない。旧2系列の失敗/authorization/STOPは保持し、新しい来歴で実行した。

対象：linear H4 1.00 Å/STO-3G/DF rank12、R1新partition（L_D=3のgeneration-prefix）、8 system qubit、13 basis/324 label、m=5、T=4/5、q_outer=1、r_RTE=4、sigma=+1、outer symmetric S2。H=2048/K_root=4096、eta=rho=tau=10^-12、K_phi=256/J=64。旧PR-2と135係数のbinary64が異なるため旧入力との同一実行とは扱わない。

| Gate | 結果・保存Evidence |
|---|---|
| Phase A・B | 人工bitwise/operator回帰PASS。13 basis全412局所行列が保存bytes/hashと一致。[Phase B](phase_b/summary.json) |
| SOURCE/input/runtime | 新authorizationの直接親SOURCE、file manifest、R1 model/partition/NPZ/basis/label、固定runtime照合PASS。[execution_identity](result/execution_identity.json) |
| State | 正式dtype/shape/C-order array hash actual=固定expected、norm誤差0。無料な物理準備の認証ではない。[state_identity](result/state_identity.json)・[schedule](result/schedule.json) |
| Schedule/short step | deterministic halvesはT/2=2/5、tail 4 microstepsは各1/5、sigma=+1、x=lambda/5<=1。constant+抽出identityは可換scalarとして診断上4分割し合計Tを保持。[schedule](result/schedule.json) |
| P0〜P5 | 固定precisionで全PASS、ready=true。全parent列挙なし。fresh IIDとphysical providerは未認証。[preflight](result/preflight.json) |
| Parent/child・root/angle・weight | 6件のF3合法性/phase/order/positive proposal、rootの既存固定precision guard、angle enclosure guard、実pi*W=alpha_midを通過。各`event_NN_sidecar.json`と`event_NN_parent.json`にsource/input/policy/Evidenceを保存 |
| Controlled operator・actual adjoint | 下表全件で既存1e-10 criterionを満たす。global phaseでalignせずfull controlled matrixを比較。inactive/active block、scalar相対位相も比較結果へ含む |
| X/Y | 固定保存stateによる数値診断値を保存。shots/Monte Carloなし。新rigorous閾値は導入しない |

Referenceは保存hex局所行列をbitwise復元し、time-orderでBを構成し、`sign B P B†`とordered parent・single-child回転・parent/scalar位相を計算した。adapterは元gate constructorとbasis ID/hashを復元し、B†→controlled中央操作→Bを回路へ追加する。一般の奇数語をinvolutionと仮定していない。符号はQに一度だけ入り、parent wordの回路時間順は逆、actual adjointはreverse order + gate inverseを使う。

比較対象は`diag(I,V)`全体。使用normはFrobeniusであり、既存G10のspectral診断基準1e-10に対し十分な上界になる。浮動小数点計算そのものをrigorous norm certificateとは呼ばない。[固定G10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g10_comparison.py#L73)

| index | parent → child | Full controlled Frobenius残差 | X/Y絶対差（診断） | Gate |
|---|---|---:|---|---|
| 0 | [] → 0 | 7.147413750258212e-15 | 1.1102230246251565e-16 / 1.6653345369377348e-16 | PASS |
| 1 | [] → 9 | 7.163304827573408e-15 | 5.551115123125783e-16 / 5.551115123125783e-17 | PASS |
| 2 | [0,2] → 36 | 1.861869944729678e-14 | 3.608224830031759e-16 / 2.0816681711721685e-16 | PASS |
| 3 | [36,0] → 9 | 1.823859660677649e-14 | 4.163336342344337e-17 / 6.938893903907228e-17 | PASS |
| 4 | [0,2,36,9] → 2 | 2.868147051928566e-14 | 9.992007221626409e-16 / 4.926614671774132e-16 | PASS |
| 5 | [9,36,2,0] → 36 | 2.7418337523939178e-14 | 1.3322676295501878e-15 / 5.551115123125783e-16 | PASS |

値はone-shot保存`event_00_comparison.json`〜`event_05_comparison.json`の引用。inactive/active/actual-adjointの個別残差は同JSON参照。最大値もone-shot自身の[summary](result/summary.json)に保存済みで、C1-4で再計算していない。今回のselected labelは0/2/9/36、reference operatorとして動的に構成したbasisは2件。13件全局所bytes照合とはscopeが異なる。

実行wall 37.31578362546861秒、user CPU 49.101103秒、system CPU 61.892935秒、max RSS 375044 KiBは[terminal](result/terminal.json)の観測。性能benchmark/比較ではない。人工資源上限・共有設定変更は0。原stdout/stderr、起動/途中/終了資源観測も保存する。

成功は選択6 eventの数値correctnessとsource-bound有限bit guardまで。全324 labelの動的correctness、全event/path/round、native strict error、rigorous DF binding、物理state準備、十分shot数、資源優位・独立新規性・投稿GOは認定しない。valid zero/technical failureと全trial分母は人工回帰と既存SOURCEで保持し、H4 trajectoryを生成して確認したものではない。

Phase C terminalのmandatory_STOP=falseは、明示承認済みC1-4文書へ進むという意味である。C1-4完了後の最終STOPは[STOP.md](STOP.md)に別記する。
