# Track B／M1-S：有限step複素信号pilotのGPT独立科学レビュー

- **レビュー日**：2026-10-16（JST）
- **レビュー区分**：Level 2 — 既存T1a／T1bとGate 0の枠組みに限定した科学的実験契約・SOURCE意味論のレビュー
- **開始承認**：ユーザー「レビューを開始して」
- **固定対象**：`HIROMU1015/Partially-Randomized-Trotter`、候補契約commit `b48d4cdcd7d8dae7961d7834853c9256f0d1b7fd`、公開HEAD `a398a3fab6f68abc2a47799d7454b312082af292`、SOURCE commit `9758aeaa968d1d0af55723bc6386959f32d680cf`
- **判定**：`CONDITIONALLY_ACCEPT_FINITE_STEP_SIGNAL_PILOT_DESIGN_WITH_PRE_ADOPTION_MODEL_DRIFT_INTERPRETATION_AMENDMENT; NO_FORMAL_ADOPTION; NO_SCIENTIFIC_EXECUTION_AUTHORIZATION; MANDATORY_STOP`
- **実施**：GitHub上の科学契約・SOURCE・監査資料、T1a／T1bとCheckpoint 2、直前のGate 0研究レビューの読解、紙上の演算子恒等式・比較意味論・誤差および研究目的の独立検討。出版社の論文概要の外部照合。
- **実施していない**：NPZの数値decode、新規Hamiltonian・Pauli係数・複素信号・誤差予算の評価、科学数値再解析、模擬／実系の数値correctnessテスト、benchmark、sampling、回路生成／compile、GitHub改変。
- **権限**：GPTによる条件付き科学的推薦であって、ユーザーによる研究方針／契約の正式採択、科学計算承認、authorization／grant／markerではない。旧v4 STOP・v5 one-shot・独立照合結果と消費済みmarkerを維持。

---

## 0. 要旨と結論

**有限step複素信号pilotの比較構造は、目的を「保存H4モデルにおける平均複素信号の有限時間screening」に限定する限り、科学的に妥当である。** 特に shared ±、ordinary、independent ± の平均写像は固定T1bの意味論と一致し、同じ概念的乱択label数を揃えるという明確な比較を構成している。T1bの非CPTPな二側平均をCPTPの量子チャネルへ置換する誤りも、取得した計算経路では確認しなかった。

ただし**正式採択前の小さな契約修正を推奨**する。SOURCEは生の保存binary64 DF Hamiltonianを理想参照 `H_full` としながら、実際の3方式には「独立再構成したDF-prefix HD＋v5保存済みHR Pauli係数」を用いる。これは一般には厳密に同じHamiltonianではない。差をraw exact行列から評価して保存する設計自体は良いが、その差を科学的な「finite-step algorithmic bias」に暗黙に混入させると、*モデル不一致*と*乱択／分割／有限step誤差*が識別できなくなる。**raw-H基準の総biasと、split-targetに対するアルゴリズムbiasを区別し、モデル差の時間依存上界を契約に明示することを必要条件とする。** 新しい科学計算を今実施する必要はない。

**今採択しない主張**：相対的な先導headroom `rho` によるfinite-step visibility保証、shot／RPE・native回路・最終総資源改善、新しい一般DF-native PR法の優位、全空間固有状態／（SP）適用、ハードウェアレベルの誤差certificate。

提案する次の順序：Codexが契約・報告意味論の最小修正と静的整合確認を行う → ユーザーが修正版pilot契約（条件付き誤差モデル、grid、比較、runtime、STOPを含む）を採否判断 → **別途**one-shot科学実行を明示承認 → Codex実行・mandatory STOP → 得られた結果を必要時に科学的判断へ戻す。新しい広範な理論検証はこのpilot採択の条件ではない。

## 1. 取得した一次資料と固定性

必須のGitHub資料を固定commitから実際に取得した。取得不能によるreview STOPはない。

| 役割 | 固定資料 |
|---|---|
| pilot主契約 | [`m1s_finite_step_complex_signal_pilot_contract_candidate_20261016.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b48d4cdcd7d8dae7961d7834853c9256f0d1b7fd/docs/tracks/algorithm_codesign/m1s_finite_step_complex_signal_pilot_contract_candidate_20261016.md) |
| machine候補 | [`finite_signal_contract_candidate.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a398a3fab6f68abc2a47799d7454b312082af292/artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/finite_signal_contract_candidate.json)、SHA256 `3ee7e999eebf1a62b2da9f760b647c78d5e956211b5fc8ce26739eb54501096a` |
| 実装意味論・予算設計 | [`m1s_finite_signal_precision_and_semantics_design_20261016.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a398a3fab6f68abc2a47799d7454b312082af292/docs/tracks/algorithm_codesign/m1s_finite_signal_precision_and_semantics_design_20261016.md) |
| v1 SOURCE | `src/trottertracks/algorithm_codesign/m1s_finite_signal_v1/{control,exact_input,numerics,word_reference,pilot}.py`、SOURCE commit `9758aeaa968d1d0af55723bc6386959f32d680cf` |
| 科学実験選択・保全 | `unresolved_choices.json`、`source_and_evidence_identity.json`、`resource_and_acceleration_plan.json`、`publication_receipt.json` （`artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/`） |
| T1b定理 | [`m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/54336028edd7ef759f8a0c73bd0a96ade13f47f9/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md)、固定commit `54336028…` |
| Checkpoint 2独立受理 | [`track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md) |
| 前回Gate 0科学レビュー | [`track_b_m1s_gate0_v5_independent_result_gpt_scientific_review_20261016.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a398a3fab6f68abc2a47799d7454b312082af292/docs/research/track_b_m1s_gate0_v5_independent_result_gpt_scientific_review_20261016.md)、24,927 bytes、SHA256 `07864ab817cc5fa8f3b8f2a834619c3ea4af090dec3e553cb7be3438a401f6fb` |
| 生入力・旧law | NPZ SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、保存v5 `result.json` SHA256 `8e32925e99b64ce6bc2c939e45ba51485d18b1eb4b62952e22a0debe1d9d4ec5` |
| 旧独立reference | 固定結果SHA256 `32c73f57a1b3fada6bc185311bd56007628849222ad77e4a33e8383a2d0085ac` |

GitHub公開branch `track-b-m1s-finite-signal-preparation-20261016` のHEADは `a398a3fab6f68abc2a47799d7454b312082af292` と一致する。receiptにはSOURCE静的601件PASS、保全・文書139件PASS、候補未採択、数値未実施、mandatory STOPが記録されている。**これは当該数値演算の正しさの実証ではない。**

## 2. 研究目的と観測量

前回Gate 0では、保存H4 1.00 Å／STO-3G／8 spin orbitals／DF rank12に対し、生成順prefix G3/G6/G9の非冗長Pauli-tailで `W`、`v`、`rho` を得た。独立した整数／有理数referenceの照合可能範囲はPASSしたが、旧v5未保存full-H／HD比較の全scopeはUNDETERMINEDのままである。

今回の新pilotは、**small-stepでの先導分散診断から、有限stepの複素平均信号へ移る**ための限定されたdevelopment試験である。中心量は各方式 `m` の

\[
z_m(T)=\frac{\langle\psi|\mathcal F_m^{s}(I)|\psi\rangle}{\langle\psi|\psi\rangle},\qquad s=T/\delta,
\]

ordinaryでは対応する平均operatorのべき、独立±では二側平均のべきを使う。基準信号は

\[
z_{\rm raw}(T)=\frac{\langle\psi|e^{iT H_{\rm full}^{\rm raw}}|\psi\rangle}{\langle\psi|\psi\rangle}.
\]

報告：複素Re/Im、\(|z_m|\)、\(|z_{\rm raw}|-|z_m|\)、主値 `Arg(z_m conj(z_raw))`、\(-\log|z_m|\)（適用域内）、sharedから対照を引いた**絶対**visibility差、各数値誤差球と未解決分類。**shot測定の有限標本平均ではなく、IID lawについての解析的な平均写像を計算する設計**であり、サンプリング分散、RPE復号、native価格は対象外である。

このpilotはGate 0結果を既に見た後で設計している。gridは有限step出力に対しては事前固定候補だが、**H4の独立held-out検証ではない。**

## 3. 3方式の比較意味論の独立検証

### 3.1 共通lawと共有±

\(H_R=\lambda h\)、\(h=\sum_j p_jP_j\)、\(P_j^2=I\)、\(A_j(\theta)=e^{i\theta P_j}\)。同位置で共有されたlabel、位置間・step間IIDとする。\(D=e^{i\delta H_D/4}\)、\(\theta=\lambda\delta/(2R)\)。負枝は**同じword順序の各gateの符号反転**で、full-word adjointではない。

\[
\Phi_\theta(X)=\mathbb E_j[A_j(\theta)XA_j(\theta)],\quad
\Psi_{\rm sh}(X)=D\Phi_\theta^R(DXD)D.
\]

\(c=\cos\theta,s=\sin\theta\)として

\[
\Phi_\theta(X)=c^2X+ics(hX+Xh)-s^2\sum_jp_jP_jXP_j.
\]

`numerics.py`の`phi/shared`、`pilot.py`の`shared=k.shared(shared,D,h,angle/2,R,twirl)`は、この構成に合致する。\(A_jXA_j^\dagger\)ではなく\(A_jXA_j\)であり、**CPTPのconjugation mapとして扱わない。** 一方、Hilbert–Schmidt Frobenius normにおいて、各\(X\mapsto A_jXA_j\)はisometryであるため、凸平均\(\Phi\)は収縮である。この違いをSOURCEのoperator bound処理でも維持している。

### 3.2 ordinaryとindependent ±

\[
M_a=\cos a\,I+i\sin a\,h,\qquad a=\lambda\delta/R,\qquad D_o=e^{i\delta H_D/2}.
\]

ordinaryは\(O=D_oM_a^RD_o\)、信号\(\langle O^s\rangle\)。independentは各枝\(r=R/2\)で\(V=DM_a^{R/2}D\)、二側map\(\Psi_{\rm ind}(X)=VXV\)、信号\(\langle V^sV^s\rangle\)。**`V†XV`にすることは間違い**である。`pilot.py`の生成式はT1bの平均operator構造と一致する。Rが偶数でなければ独立±の各枝へ割れないが、固定R=2,4はこの条件を満たす。

### 3.3 イベント数と回路価格の違い

| 方式 | 一step概念的独立draw | branch Pauli exponentials | HD half exponentials（両枝を別々に数える形式） |
|---|---:|---:|---:|
| shared ± | R | 2R | 4 |
| ordinary | R | R | 2 |
| independent ± | R | R | 4 |

数え方は契約の定義と整合する。これは同一の相対時間、入力、Pauli law、conceptual label数\(sR\)の比較である。**同じ物理gate数・RZ/T/CX/depth・状態準備・shot数の比較ではない。** sharedの同位置label共有がnativeのcontrolled/directional gateで値引き可能かは別の研究課題。

### 3.4 Walshによるtwirl

Pauli語を位相を除いた\((x,z)\)で表し、\(Y_d(u)=X_{u,u\oplus d}\)とすれば

\[
(P_{x,z}XP_{x,z})_{u,u\oplus d}=(-1)^{z\cdot d}Y_d(u\oplus x).
\]

したがってtwirlは\(u\)方向のXOR畳み込みであり、Walsh空間で

\[
\widehat Y'_d(k)=S(d,k)\widehat Y_d(k),\quad S(d,k)=\sum_{x,z}p_{x,z}(-1)^{z\cdot d+x\cdot k}.
\]

`twirl_plan`の`fwht(fwht(weights).T)`が最初にz、続いてxを変換する順序、`twirl`のXOR対角・前後FWHT・256での逆変換は、この恒等式に沿う。Yの位相因子は\(PXP\)で相殺されるため追加のY符号は不要。**この代数の整合性は紙上で確認した。浮動小数点での厳密なerror radiusの検証とは異なる。** 将来one-shotで行うliteral tensor probeは必要な独立correctness gateである。

## 4. 108行gridの適切さと漸近の読み方

固定候補：\(\delta\in\{1/8,1/16\}\,\mathrm{Ha}^{-1}\)、\(R\in\{2,4\}\)、\(T\in\{1,4\}\,\mathrm{Ha}^{-1}\)、split3、method3。非零時間は\(3\times2\times2\times2\times3=72\)信号、zero-time controlは\(3\times2\times2\times3=36\)信号、計108行。\(s=T/\delta\)はすべて整数（最大64）。この程度ならpilotとして過大なパラメータ探索には当たらない。

**ただしT1b主定理の非自明な減衰スケーリングと今回の実験経路は異なる。** T1bはR固定・\(\delta\to0\)で\(u=s\delta^2\)を有限に保つ系列（\(T=s\delta\)が\(\delta^{-1}\)程度に増大し得る）を扱う。今回のpilotではTを1または4に固定してδを二点比較するため、\(u=T\delta\to0\)であり、非自明な一定減衰予算の長時間一様保証そのものを数値的に検証する設計ではない。**有限の実用時間でのscreening**として位置づけるのが正しい。

また二つのδ点では、漸近係数の一意なfit、\(\delta\to0\)の収束域、r増大に関する一様性、RPEのmulti-round taskは判定できない。\(\rho\)によってδ／R／Tを結果依存で選び直さず、G9で差が精度を下回っても予定どおり記録する設計は妥当。

## 5. 先導減衰尺度の解釈

T1bの条件付き理論では対象が適切な全H固有stateで、特に単一pair-sum block（SP）を満たす場合、固定Rの一step二次係数について

\[
\beta=\frac{\lambda_R^2}{4R},\quad u=s\delta^2=T\delta,\quad
\kappa_{\rm sh}=\beta u(W+v),\quad
\kappa_{\rm ord}=2\beta uW,\quad
\rho=\frac{W+v}{2W}.
\]

同じ総label数のindependent ±も先導ordinaryと一致する。ただし保存stateは**sector内の数値oracle**であり、全Fockでの非縮退固有状態や（SP）、pair-sum gapを認定していない。したがって現在の契約が `leading_kappa_*_hypothesis_only` と明記し、実測visibilityと直接等置しないのは適切。

**λ²が重要**である。例えばG9の極小tailはρが小さくても絶対的な減衰差が数値分解能以下となり得る。絶対差を主指標とし、相対visibility比や「○％資源節約」を自動出力しない方針を支持する。

## 6. Hamiltonian targetの差：正式採択前の必須整理

### 6.1 固定SOURCEでの具体的な問題

`exact_input.py`の`targets`は、保存NPZのbinary64 bit値からexactな\(H_{\mathrm{full}}^{\mathrm{raw}}\)を生成する。一方、方法側の

\[
\widetilde H^{(j)}=\widetilde H_D^{(j)}+\widetilde H_R^{(j)}
\]

は、raw DF prefixに**保存v5のHR identity scalar**を追加したHDと、保存v5のnonidentity binary64 Pauli coefficientsをexact rationalとして再構成したHRからなる。全空間の\(H_{\mathrm{full}}^{\mathrm{raw}}\)と厳密一致するとは主張していない。SOURCEはすでに\(\Delta H_j=\widetilde H^{(j)}-H_{\mathrm{full}}^{\mathrm{raw}}\)について、exactな行ごとの絶対和の最大\(B_j\)を保存する設計である（Hermitian性の下でoperator normを覆う）。これは良い保全策であり、旧v5に未保存だったHD／full行列との直接一致を回復したとは誤認しない。

ただし主契約の「この差を同じraw-H理想参照に対する有限stepアルゴリズム誤差に含める」という一括表現は、科学的な誤差原因の識別には不十分。どの方式も同じ\(\widetilde H^{(j)}\)を目標にするので**方式間の直接visibility差**は公正だが、raw-H参照との総biasを、pure algorithmic biasとして扱ってはいけない。

### 6.2 数値計算なしに導けるDuhamel上界

二つのHermitian行列に対するDuhamel恒等式とunitarityから

\[
\|e^{iT\widetilde H^{(j)}}-e^{iT H_{\rm full}^{\rm raw}}\|_2
\le |T|\,\|\Delta H_j\|_2\le |T|B_j.
\tag{M1}
\]

従って保存stateの正規化期待値も

\[
\bigl|z_{\rm split,ideal}^{(j)}(T)-z_{\rm raw}(T)\bigr|\le |T|B_j.
\tag{M2}
\]

この別台帳を導入すれば、数値計算時に「raw-Hからの**総bias**」「split-modelの**model drift**」「split-modelに対するalgorithmic deviation（独立の数値推定があれば）」を混同しないで済む。新しいideal split信号\(z_{\rm split,ideal}\)の評価は本契約には含まれていないので、**M1/M2の上界を保存しただけでalgorithmic deviationを点値として得たと書かない**。

必要なら参照visibility \(|z_{\rm raw}|\) がzeroから十分分離し、\(|T|B_j<|z_{\rm raw}|\) の場合、モデル差によるprincipal phaseの最大変化は\(\arcsin(|T|B_j/|z_{\rm raw}|)\)で上界化できる。ただしこれも理想model間の上界であり、有限step方式の位相誤差と同一視しない。

### 6.3 必要な小修正

**正式採択前にCodexへ依頼する最小修正（新しい科学計算は不要）**：

1. contract/schema/報告欄で \(H_{\rm raw}\)、\(\widetilde H^{(j)}\) を別targetとして明記。現在の「phase bias」は `TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H` として定義し、純粋なPF／randomized biasと呼ばない。
2. 既にSOURCEが生成する\(B_j\)と、固定gridの\(|T|B_j\)を**model driftの専用台帳**として報告する計画を定義。`signal numerical radius`と混ぜて精度保証を水増ししない。\(B_j\)が大きい場合も閾値を結果依存で緩和せず、raw-Hに対する原因帰属を `MODEL_TARGET_CONFOUNDED` などに分類する。
3. 3方式間の`visibility_difference`は共通split-targetでの直接コントラストとして維持する。raw-H基準総biasと区分。旧v5のfull/HD未保存による `UNDETERMINED` は保全し、新再構成がそれを解消したと記載しない。
4. 追加するモデル差の判定が科学的な報告規則に影響するならユーザーが修正版契約として採択する。**採択前に固定SOURCEを書き換えて実行するのではなく、新commit／新contract候補として保全**する。

物理Hを変更する必要も、別のδ/R/T候補を足す必要も、アルゴリズム本体の平均mapを書き換える必要もない。

## 7. 精度設計・SOURCEの数値的正しさ・停止規則

- **条件付き**Frobenius Ball：matrix product `e_AB = b_A e_B+b_B e_A+e_Ae_B+γ_(8d+32)||Â||_F||B̂||_F`、Hermitian exact exponentialをunitary `b=1`として扱う点は一般的な保守的forward propagationに整合。ただし全source経路のrigorous coverageを形式認証したわけではない。
- **行列指数**：正確なHermitian targetの\(e^{itH}\)を、\(\|itH/2^k\|\le1/8\)まで縮小した24次Taylorとrepeated squaringで近似する。Frobeniusで\(\sqrt{256}(8/7)(1/8)^{25}/25!\)というtail上界を別加算する方法は紙上の標準的な展開剰余評価に適合する。`exact exponent`とは数学的ターゲットの意味であり、実装の丸め／打切りはBall内に残る。
- **Walsh twirl**：上記の代数的構造は一致。ただしγ32・γ128などの実装error budget、complex BLAS、norm、`math.sin/cos/atan2/log`に対する絶対 `2^-51`、hypotに対する相対2u、underflow/FTZ排除は、**未検証のengineering仮定**。NumPy配布物hash固定・人工probe PASSが将来出てもハードウェア数値certificateへ格上げしない。
- **信号**：`eps_signal_abs=2^-24`、phase半径`2^-18 rad`、floor`1/4` は、結果が得られた場合の分類を安定させるための事前選択。化学的精度やRPE最終精度の保証ではない。根拠は意味のある**pilot resolution policy**としては十分だが、正式採択時に「scientific necessityとして証明した値ではない」と明記する。
- **位相**：`Arg(z conj(z_ref))`の主値だけ。位相球が0・負の実軸branch cut・visibility floorに近づけば`UNDETERMINED_PHASE`。phase unwrapやground energy偏差を主値から復元したことにしない。
- **対照差**：shared−ordinary、shared−independentの絶対visibility差に両者の誤差半径を加え、差の区間が0を跨げば `UNRESOLVED_ABSOLUTE_CONTRAST`。これは一般の資源negativeでもない。
- **in-run correctness**：非可換HDの2×2 toyの物理word全列挙（R=2,s=2）と解析的Taylor remainder、Walsh/direct twirl、H4各splitでのliteral tensor probeが準備されている。数値未実行。これらは意味論の強い検査になるが、full256実行のすべての数値経路を形式的に証明するものではない。
- **global STOP**：入力・SOURCE・contract/run-identity、再構成、数値予算、resourceに問題があれば、完了行を保全し残行は未評価、one-shot消費、no retryで停止。未解決のphaseやvisibility contrastを科学的FAILへ誤変換しない設計を維持する。

**確信度**：数学的比較意味論＝高、SOURCEの主要平均経路との対応＝高、conditional数値radiusのplatform coverage＝低／未認証、固定gridの実用的識別力＝結果なしゆえ未判定。

## 8. 研究としての情報価値・先行研究・出口

このpilotで判定できるのは、**同じ保存H4・Pauli law・相対時間・独立label数において、実際のfinite-step平均信号の絶対visibility差とprincipal phaseの影響が存在するか**である。低い\(\rho\)だけで実際の差が出ると断定しない設計なので、Gate 0から次へ進む検証として情報価値がある。

- **改善が解像された場合**：コスト／shot評価へ進む前に、当該splitにおけるraw model driftの大きさと位相biasが許容されるかを区別。ここで初めてnative directional wrapper単価、finite-RPE shot・bias budgetの独立契約を検討する価値が生じる。自動GOにしない。
- **改善が位相biasで相殺／解釈不能の場合**：視認可能なattenuation減少でもenergy/phase taskに有利とは限らない。T1c（固定δ・r連成、mode/dressing）などの新理論方向は候補になるが新研究契約が必要。
- **改善が解像されない場合**：G9などの微小tailでは絶対差が無視できる可能性がある。これは当該固定time／law／分解／数値分解能での限定結果であって、他の分子、DF-native PR、回路コスト全般の科学的NO-GOではない。
- **技術的未完了の場合**：数値モデルやSOURCEを原因調査し、元one-shotのmarkerと結果を保全。無断のprecision変更やretryはしない。

学術的新規性について、Güntherほか *Phase Estimation with Partially Randomized Time Evolution*，PRX Quantum **7**, 020332（2026-05-19），DOI `10.1103/ynxb-p2xq` が量子化学の部分乱択と位相推定資源評価を既に扱うことを外部の出版社抄録で確認した。**本pilotを「最初の部分乱択位相推定」や「既知手法に対する一般的優位性」と主張することは不可。** 今回は論文全文の新規性監査までは実施せず、branch-shared coherent mapの精密比較が先行研究と定理レベルで独立であることも新規認証していない。

旧Checkpoint 2の推奨「T1a／T1bを条件付き理論成果として残し、Gate 0から有限時間へ進むか判断」に対して、Gate 0でheadroomが確認されたため限定有限step pilotへ進む判断は整合する。ただし**同じ固定H4 developmentでの事後的延長**であり、別geometry／外部再現への転送、RPE全資源、hardware robustnessは未確認。

## 9. 判定一覧

| 対象 | 独立判断 | 備考 |
|---|---|---|
| 3平均写像と負枝の順序 | **整合** | sharedは非CPTP map、independentはVXV、ordinaryはO^s |
| 同一概念的label数の公正性 | **限定的に妥当** | 物理gate単価・branch数は一致しない |
| Walsh twirlの代数 | **紙上整合** | 数値forward errorの実効coverageは未証明 |
| 108行grid・zero control | **限定pilotとして妥当** | 非自明有限減衰T1b極限やRPE全体は検証しない |
| \(\lambda_R^2\)・\(\kappa\)報告 | **適切な警告付き仮説** | state/(SP)未認証、先導式をfinite-step結果に等置しない |
| raw-Hと実split-Hの差 | **修正要** | 時間依存\(|T|B_j\)の独立ledgerと総biasの名称が必要 |
| signal・phase precision | **条件付き候補として受理可能** | エラー算定／library仮定はplatform非認証 |
| future 2×2・H4 probe | **必要な初回品質gateとして妥当** | 現時点で未実行 |
| Runtime/one-shot/STOP | **設計として妥当** | 実環境での完遂・メモリは未測定 |
| 最終契約・科学計算 | **未採択／未承認** | ユーザーの別々の明示承認が必要 |

## 10. Codexへ戻す最小修正・その後の承認

### 技術担当：Codex（契約採択前の非数値改訂）

1. 固定SOURCE、`exact_input.targets()`、`pilot.phase_metrics/store/run()`における**raw-H vs split-targetの意味論を契約・保存schemaへ明示的に反映**し、モデル差の\(|T|B_j\)台帳を別保存する。未来の科学結果への追加算定は、採択／科学実施承認後に限る。
2. 3方式・固定grid・epsilon・状態・比較方法は変更しない。モデル不一致が比較の原因帰属を妨げるときの `MODEL_TARGET_CONFOUNDED` 表示と、数値precision unresolvedを区別。既存旧v5結果と旧independent結果を変更しない。
3. 静的SOURCE/schema/failure保存の整合を調べ、候補contractの新しい版・evidence・publication receiptをGitHub公開、mandatory STOP。SOURCEの意味を変更する場合の正式採択はユーザーへ戻す。

### 採択担当：ユーザー

- 修正版有限stepのgrid／意味論／条件付きcoverage／phase floor／runtime・one-shotを**正式採択するか判断**する。現段階のレビューは採択の代行ではない。
- 別途、**新規科学計算one-shotを明示承認**するか判断する。source prepの静的PASSやレビュー開始承認は実行許可ではない。

### 実施担当：Codex（双方の承認後）

- 2×2 exact word・全256 twirl probeなどの初回correctnessを含め、108行の有限step mean pilotを**一度だけ**実行して結果保存、retry0、mandatory STOP。phase unresolved・contrast unresolved・global technical stopを混同しない。

### 科学判断の戻し先

- 上記実行後、固定law/timeにおけるvisibility差とraw-H基準のphase bias・model driftを並べて判断する。大きな新規科学判断が必要になった場合だけ、ユーザーの新しいレビュー開始承認を求める。

## 11. 参照文献・review audit

- Günther, J., Witteveen, F., Schmidhuber, A., Miller, M., Christandl, M., Harrow, A. W. (2026), “Phase Estimation with Partially Randomized Time Evolution,” *PRX Quantum* 7, 020332, 2026-05-19, https://doi.org/10.1103/ynxb-p2xq. 確認したのは出版社公開のメタデータ・概要であり、全本文の網羅的な新規性比較ではない。
- T1a：`m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md`、commit `4e7b18dd2aa14c127bce9e248c63996c62a055cc`。
- T1b：`m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md`、commit `54336028edd7ef759f8a0c73bd0a96ade13f47f9`。
- Checkpoint 2 GPT：`track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md`、commit `62554ee127a2bec1206367cf1754df2ff49fa000`。
- Gate 0 v5独立数学review：`track_b_m1s_gate0_v5_gpt_independent_mathematical_review_20261016.md`、固定公開済み原本。
- Gate 0 v5結果：`m1s_gate0_v5_one_shot_results_and_stop_20261016.md`、commit `cdcb7f6f151df217960611f502eb1bac8be63f58`。
- 独立correctness結果：`m1s_independent_one_shot_results_and_stop_20261016.md`、commit `cc2f451324b7292ae50cef94ff794d46f225d892`。
- 本レビューのSOURCEチェック：GitHub上の`control.py`, `exact_input.py`, `numerics.py`, `word_reference.py`, `pilot.py`と固定machine contract, model/precision doc, static receiptを読解。**SOURCEを実行していない。**

**Review complete. No new scientific numerical computation. No GitHub mutation. Mandatory STOP preserved.**
