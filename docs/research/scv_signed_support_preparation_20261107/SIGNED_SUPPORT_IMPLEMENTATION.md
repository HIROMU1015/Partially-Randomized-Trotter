# Signed support scalar SCVの最小実装

2026-11-07 JST / SERVER1。**実装・人工correctnessのみ。実科学へ未接続。**

## 入口と固定資料

- 新SOURCE：[scv_signed_support.py](../../../src/trottertracks/algorithm_codesign/scv_signed_support.py)
- 新テスト：[test_scv_signed_support.py](../../../tests/tracks/algorithm_codesign/test_scv_signed_support.py)
- [correctnessと再現コマンド](SIGNED_SUPPORT_CORRECTNESS.md)、[次回の未決事項](NEXT_SCIENCE_SCOPE_DRAFT.md)、[SOURCE・環境・hash](SOURCE.json)
- 最新方向整理：`4741cae05af662dfc42d0519ff977756a94cb7b7` の `docs/research/scv224_next_directions_20261107/` のREPORT・候補比較・方向選択肢。
- 設計正本：`84e021b6a8decde7fa3ed3fa44d7148cc5428061` の `docs/research/intrastratum_cv_design_20261107/` の候補・層別CV・次回検証草案。
- 既存SCV SOURCE：`560c6092bfb5e71af8eedc4a4be5416922764f98`。下表の5ファイルはこのcommitとbyte一致し、変更していない。
- 既報SCV224：`ba483ab3038693048e58830e39a676493a892053`。完了結果を背景として読むだけで、raw/bit/価格/係数を新特徴へ再投入しない。
- 有限confidence調査：`af47cf9161241c8710383ff8539a4a7f8f67492c`。保証接続の未完了事項を継承し、今回calibrationしない。

`PROJECT_MAP.md`、研究概要・現状、適用AGENTSと正式共通指示の§8.1/8.2/8.4/10/10.3を確認した。旧SOURCE read-onlyという今回の限定に従い、旧索引・科学的状態・manifestは書き換えず、本directoryから新SOURCE・テスト・証拠へリンクする。科学runnerは追加しない。

## 再利用とadapterの境界

| 既存SOURCE | 再利用箇所 | 追加が必要な理由 |
|---|---|---|
| `scv_scalar.py` | Metadataのlaw/catalog/H/preparation binding、cell分類、ConditionalBankの基本API、CellModel/CellModels、paired_evaluate | 旧特徴はfragment indicator。新しいsign/supportと一位置momentが必要。旧fitのindicator range/guardと既定quotaを使わない |
| `a1_control_variates.py` | EventMetadata.from_rte_event、intercept付きfit_and_freeze、FrozenCVの係数固定・中心化・stream分離、raw_mean | FrozenCV.output_rangeは[0,1]専用なのでsigned subclassで置き換える |
| `a1_n3_connection.py` | ConditionalSampler/PathChoice、actual law、Domain、validate_batch、production_mean | sampler・law・層定義・加重平均はそのまま。新規sampling処理なし |
| `scv224_pilot.py` | Envelope、cost_summary、既存STOP境界の人工検証 | 新controlの科学activation/driverとしては使わない。旧purpose・group選択・guardは新契約へ自動移植できない |
| `a1_n3_campaign.py` | 既存journal/bit先行保存/STOP・reconcileの人工回帰テスト | 科学worker/managerを起動せず、旧処理を改変しない |

新しいsampler、OLS、統計会計framework、engine、runnerは実装していない。既存ControlBankのexact finite-law計算方針を再利用し、signed scalarには閉じた一位置和だけを追加した。

## 特徴とmetadata identity

`Component(component_id, coefficient_sign, diagonal_pauli_support)` は同一labelの保存sign/supportをまとめる。signは整数±1、supportはsystem register内の重複なし昇順整数で、サイズ0/1/2に限定する。既存 `df_rte_tail.py` のI/Z/ZZ仕様、`df_rte_circuit.py` の非identity Z/ZZ・identity空supportと静的照合した。よって

\[
f(l)=s_l|S_l|/2\in\{-1,-1/2,0,1/2,1\}\subset[-1,1].
\]

identityの空supportは0。zero-grid labelもcatalog/identityに残すが、そのlabelを実現pathへ投入すると既存positive-support検査で拒否する。

`metadata_from_bytes` はcallerが渡したbytesのSHA256とH/preparation identityを既存binderで照合し、追加sign/supportを同じcomponent行から読む。全行とlabel順の整合・qubit domainも検査する。特徴定義・追加metadata・既存LAW/catalog/H/preparation identityを新identityへ含める。metadataのtuple化とfingerprint再照合で入力の意図しない変更を検出する。

適用順はproducts→rotation。n=0ならL0、n=2ならL2、一般には保存された最後のlabelを参照する。明示RTE eventの接続には既存 `EventMetadata.from_rte_event` を通し、productsとrotationが保存順と矛盾すれば拒否する。係数絶対値、rotation角度、Born、state、bitは特徴へ入れない。Qとbit weightは変更しない。

## Exact conditional moments

`SignedControl` はactual finite-grid label質量ρと固定aを使い、Fractionでk=1,2の

\[
m_k=\sum_l\rho(l)f(l)^k,
\qquad
(m_{k,0},\ldots,m_{k,4})=
\left(f(a)^k,\frac{m_k-\rho(a)f(a)^k}{1-\rho(a)},m_k,m_k,m_k\right)
\]

を求める。νはk=1、二次momentはk=2、varianceはm₂−ν²。全path列挙は不要で、label数に比例する有限和である。理想係数比、実数連続law、global平均によるh1平均の代用はしない。

5層は既存のまま：h0=(n0,L0=a)、h1=(n0,L0≠a)、h2=(n2,L0≠a)、h3=(n2,L0=a,L1≠b)、h4=(n2,L0=a,L1=b)。全正層を残し、質量を変更しない。h2–4のrotationは自由なので同じunconditional label momentになる。

## h1 fit/freezeと負値のrange

`ConditionalBank` はsigned descriptor・ν_hを既存OLS APIへ渡す。`fit_h1` は既存intercept付きOLSを一回だけ呼び、binary64 slopeをFractionへ固定する。rank2・finiteを必要とし、失敗時は例外でSTOPする。clip/retry/refitはない。sinkへrank・actual係数・range・feature counts・identityを渡せる。

その他の層の係数は事前に0。interceptはOLSで学習するが、推定器へ加えない。出力は元のBを条件付き平均で中心化した

\[
Z=B-\lambda(u-\nu_h)
\]

である。B∈[−1,1]、u∈[a,b]なら

\[
[Z_-,Z_+]=[-1+\min\{\lambda(\nu-a),\lambda(\nu-b)\},\;
1+\max\{\lambda(\nu-a),\lambda(\nu-b)\}].
\]

今回はSOURCEの保守的な固定box a=−1,b=1を使い、width=2+2|λ|。実際の凍結Fraction係数からexactに計算し、適用後の全residualがrange内か検査する。[0,1]、sample由来の狭いbox、不正mean/nonfinite係数、旧indicator FrozenCVの混入は拒否する。

**このrangeは数学的enclosureで、数値的な受理guard閾値ではない。** 新しいguardはNOT_ADOPTED。h1以外は補正0なのでrange[−1,1]を維持する。旧D≤4を継承しない。

## 独立評価とpaired会計

fitはA1/train/h1 acquisition domainを検査し、既存conditional validationへ一時的に投影する。保存training recordを書き換えず、law・axis・wrapper・weight1・cellを確認する。

評価はfresh B1-OL/prodの全5層。元scope一致とtrain/evalのstream分離を既存APIで検査する。`paired_evaluate(..., counts=...)` はcallerの明示quotaを必須とし、旧224のquota既定値を継承しない。既存production_meanとscalar paired計算で同じrecordからbaselineとSCVを加重合成し、raw/residual sample variance、paired covariance、差のvariance診断を返す。旧serializerの0/1 feature countsだけを全signed値対応へ置き換える。

物理的測定・価格はpaired両推定器で一回だけ。学習費をSCVへ計上し、baselineの共通評価reference費と区別する。cache hitでも量子価格は課金し、未知価格はUNKNOWNのまま。今回確認した価格は人工値のみ。

実科学のcampaign identity、selected policy/state/engineへのbinding、fresh stream生成、guard/予算、起動・停止は未接続。旧driverをそのまま新signed campaignとして呼ぶ経路は追加していない。
