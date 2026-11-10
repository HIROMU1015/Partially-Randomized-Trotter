# H6 DF19 Affine価格provider：B共通interfaceへの接続準備

2026-11-10 JST。**H6 V2の凍結Affine・保存直接価格を、Bの`c_provider(task, requested_rows)`からlookupできるH6専用providerへ整備した。** コードSOURCEは `0b5c493f2b2728b0c7c6c848d2cbf084db8f7f76`。既存branch `codex/resource-eval-h6-df19-price-design` を継続し、Bのbranch/worktreeは変更していない。B最新環境への取込みは[HANDOFF](HANDOFF.md)を使ってBが行う。

## 再利用と実装

- [H6 V2完了成果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73dcbe23678bfc50d5654f09e600d4cfbe75e697/docs/research/h6_df19_price_v2_20261110/REPORT.md)：`PILOT_INPUTS.json`、`Q2_DIRECT.json`、`Q4_DIRECT.json`、`Q_VALIDATION.json`、`COMPLETION.json`を固定Git readerで読む。
- 係数はq4価格取得前の `b85837960c80d619a421c46235baf5ef2523d217:docs/research/h6_df19_price_v2_20261110/Q4_FROZEN_MODELS.json` から全binary64桁をそのままcopy。`C_hat=a+b*qPF`。PFのa≈870/b≈428048、PR_Bのa≈923.666666667/b≈178659.666666667という表示に合わせて丸め直さず、再fitしない。
- q1/δ1.6の保存directとPR探索9/確認33はV1 carrier `a4d743cb5007d5fc6b6e93c3b43a4701ff956ac4` の`DEVELOPMENT/HOLDOUT.json`から選択。PR_A p5/R32やPF δ.8/.4の行を混ぜない。
- [B固定版のREPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cb2b1d6b4309ca7d705f397e25e27d14e8ba125a/docs/research/h4_price_pipeline_integration_20261110/REPORT.md)とH4 providerのquote/record方式を参照。Bの`SavedArtifacts.read`、`validate_task`、`candidate_id`、`identity`、`measurement`を再利用する。N/G会計器、共通pipeline、H4 providerを複製・変更しない。

新しい[H6専用module](../../../src/trottertracks/h6_df19_price_provider.py)は標準libraryのみ。constructorと`quote`/batchは保存値の選択だけで、新平均・残差・共分散・予測を再計算しない。`predict_affine`は将来の明示数値APIとして分離し、lookup miss・Bの`run_missing`から呼ばない。数値を計算できることと科学的な採用資格を分ける。

## Domain・価格と役割

Linear H6/1.00 Å/STO-3G、保存tol-only1e-8 DF19・weighted-Hermitian completion、Nα=Nβ=3/sector400、保存H/state/snapshot/fragment/scalar、数値state条件付き・P=0。δ=1.6、4次m5_best/11 signed piece、PF Full19またはPR prefix7/R_outer16/K2。元finite RTE law、symmetric_directional、測定付き完成Hadamard wrapperの各X/Y一回のcompiled RZ/gate_count。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17・compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` に限定する。

| 範囲 | 保存direct | 保存Affine lookup | 資格の意味 |
|---|---|---|---|
| qPF1/T1.6 | PF1本、PR探索9/確認33を別scenarioで保持。PR path/cohort meanあり | なし。校正anchorを新しい予測値へ読み替えない | q1は校正。確認33はV1使用済み既知データ |
| qPF2/T3.2 | PF1本、PR独立8 path＋保存cohort mean | PFとPR cohort予測 | q2校正。model値は存在しても独立検証資格なし |
| qPF4/T6.4 | PF1本、PR別独立8 path＋保存cohort mean | PFとPR cohort予測 | 有限の独立q検証。全law・全13roundの正式採用ではない |
| qPF>4 | 未取得としてnull | lookupはnull | `UNVALIDATED_EXTRAPOLATION`。明示数値APIも未資格 |
| 別δ/次数/prefix/R/K/H/状態/compiler | 対象外としてnull | 対象外としてnull | `OUTSIDE_DOMAIN`、代用なし |

直接値と予測値は別kind/scenarioで選び、missing時に相互代用しない。PRは`price_unit`と`cohort`が必須。pathにはindexと**private event fileのSHA256**を要求し、canonical event配列のdigestと混同しない。Affineはpath predictorではないのでpath別model値を返さない。全law平均`law_mean`は常にunavailable。明示AffineのPR `model_mean`は未取得cohort平均でもlaw平均でもなく、Bへは`UNQUALIFIED_MODEL`で渡す。

m0〜3のPR prefix5/R32や異なるδには、direct lookupを含め本providerを自動適用しない。既存の他domainのdirect priceをBが使う場合は、別のsource-bound provider/明示選択が必要。qRPEとqPFを分け、式に使うのはqPFのみ。Normalization Bは価格へ再乗算しない。

## 不確かさとB policy

PR有限cohortは`FINITE_COHORT_ONLY`、pathは`FINITE_PATH_ONLY`。Bの既定`qualified_expectation`はPR有限cohortを期待価格として受け付けない。`finite_cohort_scenario`で明示選択しても、その有限scenarioの計画であり全law保証ではない。

保存q4 modelの`qualified_for_candidate=True`は当該domainの有限独立q観測に限定する。q2のmodelは校正値のためfalse。directのtrueはsource/単位が一致した保存価格としての資格だけを表す。`full_RPE_price_adopted=False`を全返答で維持する。

保存nominal係数共分散、q4予測平均のnominal共分散、対応X/Yの保存誤差共分散、cohort直接価格統計をcopyする。完全な係数不確かさ・joint covariance・長q systematic・全law期待値・order2価格・total CIはnullを維持。保存nominal量を総誤差へ格上げしない。q1探索/q2 directはAffine校正入力を共有するため、modelの`shared_calibration_group`とdirectの`calibration_data_group`を対応付ける。q1確認33へ探索9の共分散を付けない。

q4 V2 pilotの768 event/order0、order2未観測という記録を保持し、order2価格を0としない。PFにtrajectory SEを付けず、X/Yを独立16本と数えず、q間・PF/PR間の共有依存を独立と仮定しない。観測maxを上界、αtotal=.05を価格の95%CIには変換しない。

## 技術検証と実行範囲

[TESTS.txt](TESTS.txt)：**10 test method PASS、保存catalog 138 scalar一致**。主担当の技術チェックであり、独立科学レビューや新しい科学的精度検証ではない。

- B固定SOURCE `cb2b1d6b4309ca7d705f397e25e27d14e8ba125a` の実moduleを含む6ファイルをGitから一時領域へ配置し、変更前bytesとの一致を確認。B作業treeをimport/変更せず、B自身の`plan/run_missing`へ接続した。
- constructor/lookupを使う間はAffine式を禁止して138保存値を照合。保存q2/q4でのみ、PF/PR各2点の式と保存値を照合した。未取得q8は人工係数の算術とmock値7でprotocolだけを試験し、H6係数のa+bqは実行していない。
- mock Nを供給し、実G入口と既存積和を禁止して結合を確認。nullを取得済みへ数えず、未資格/null返答を同じproviderの自動再呼出しで埋めない。q2校正/q4資格、PR有限cohort policy、model/path/law単位、foreign binding、source/order、未検証外挿の拒否を検査した。

[CHECKS.txt](CHECKS.txt)にSOURCE・保全の対応を記録。今回の新compile/sampling/fit/未取得H6価格予測/全law期待推定/実signal・N・N×C/Gは0。保存fixtureの照合以外の数値再解析はしていない。旧V2/V1、SOURCE/input/raw/result/grant/ledger/marker/KEEP/mandatory STOPを保全し、共有環境や他担当のprocessへ介入していない。

## 次の担当とSTOP

- **Codex B**：HANDOFFの固定2ファイルを取り込み、自分の最新branch・環境で技術結合を確認する。共通factoryが必要ならBが薄い入口を追加する。CはBの共通部分を変更していない。
- **ユーザー**：未検証qへの実予測、q8等の追加価格検証、外挿採用、PR全law期待価格、実N×C/Gを進める場合は、対象・policy・科学実行範囲を別途承認する。

commit/push後にremote SHA/tree・公開成果物を照合して**mandatory STOP**。q8取得、全RPE科学計算、他担当の待機を理由とした追加作業へ自動進行しない。
