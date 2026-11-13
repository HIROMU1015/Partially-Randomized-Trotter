# 全RPE・4方式の平均compiled価格モデル：技術準備

2026-11-13。**外部係数による加法モデル、既存E[S]＋単一のfull-wrapper C−Sモデル、全4方式のB入力への受動接続を実装した。全モデルは未資格で、正式Gへ自動採用しない。** 実H4/H6の新S/C/平均・保存数値再解析・sampling/compile/fit/Gは0。

[HANDOFF](HANDOFF.md)／[次の科学検証条件](NEXT_SCIENCE_PLAN.md)／[人工・回帰ログ](TESTS.txt)。基点 `3802bcc0212325091515325a889fcab20eabd909`、リソース評価系列の既存C branch `codex/r-variable-compiled-price-design-20261111`を継続。他担当のworktreeは変更しない。

## 既存資産と再利用の境界

|資産・固定SOURCE|再利用するコード/情報|今回のモデルへの移送資格|
|---|---|---|
|Mean kernel `2a882fc9d17ed5e32f5581d0f220514daad0235a`、現`mean_compiled_price.py`|MeanKernel/MeanPiece/full_wrapper_mean、combine_mean、local_residual_mean|supplied閉状態の合成・一残差結合を再利用。人工状態の閉包を実compiled Cの閉包としない|
|S0a `a11efd130e9dfe7086b88386570594ebe54d46af`|保存4path×XYの8件parity、元Accounting/stream_structure|同イベント列のS実装parityだけ。E[S]、E[C]、law平均やmodel資格を意味しない。再実行0|
|全13round E[S] `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f`|H4 DF10/S2/p1/Δ.1/K2、同prep・ideal finite-Kの26値/数値区間/K0/K1|供給済み構造平均の入口として維持。新モデルへ値を投入・再集計していない。別prefix/lawへの移送不可|
|C1 `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`、科学SOURCE `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`、凍結 `7eeabf0ae482ee610d667e28470692d209f2bb21`|S＋θ0＋θq q＋θL qR、paired応答・校正/holdoutの分離|H4 S2/.1/p1/K2・q8/16,R4/8、H6 m5/1.6/p7/K2・q1/2,R16/32の有限cohort。係数/false資格/旧loweringは不変。長q/Rや新loweringへ自動移送しない|
|旧connected-cluster：`docs/rte_compiled_cost_validation_summary.md` / `docs/rte_connected_cluster_cost_validation.md`、固定SOURCE索引 `f78bbc0d6ab350a2b2534ea6d169565895f0f1dc:docs/research/mean_compiled_price_preparation_20261112/SOURCE_INDEX.json`|1–3局所窓、条件付きorder pattern、_iid_sequence_form、pair/三event相殺|主条件H4/1Å/STO-3G/DF12/LD3/δ.1/r4/K2、同snapshot/compiler。現在DF10と異なる。旧raw κ(C)をSへ追加しない。dirty source hash来歴をclean HEAD証拠へ変更しない|
|旧controlled q-affine、同summary §5.5|q方向の校正/独立q検証の方法|ordinary controlled subcircuit、旧DF12/LD3/K0等。Hadamard/軸/測定なしの係数をfull wrapperへ移植しない|
|H4 Mq `1cf87201ebc6c9e0296fe647354981b6f7f8b0c1`、H6 Affine `73dcbe23678bfc50d5654f09e600d4cfbe75e697`、既存H4/H6 provider|固定Rの保存lookup、direct/model・cohort・外挿・共有校正metadata|元domain/qualificationはそのまま。別R/δ/prefixへの係数移植なし。新bridgeは旧providerを書換えない|
|Full PF receipt `e7dca5f7301c8633b17a59b6a7161331303aef2d`、HANDOFF_C|73候補の両軸exact receipt・compiler/control/full wrapper/sourceの固定対応|同system/rank/FULL/formula/T/q/元loweringとV2 traceを受け入れた場合のdirect資産。再監査・再計数0。B0/B0I、新lowering、全q affineの資格にはしない|
|Full-H/Truncated：既存baseline_signal_pipelineとD Full-H preparation/保存価格SOURCE|one-body込み全non-scalar、head0、q/R grouping、B0/B0I scalar/retained-Hの区別|full-HをPR prefix0/B3に置換しない。Full PF receiptをtruncatedへ流用しない。今回の汎用人工モデルは実価格を取得したことではない|
|D request `748be6875b5007707d864e16aa7a4fc4250f3714`、C接続 `3802bcc02`|元expected_scalar/context bindingと新lowering metadata|request定義・scalar値・guardを変更しない。新providerはD Runtime/Adapterを構築しない。Dのscalar診断は別担当/別scope|
|B `b9b9ffa6233db9dd2b413dc07b27e7039324c666`、現I.compare/price_requests|canonical identity/law、measurement、exact cost/select、N資格/不確かさ検査|元B会計を再利用し、モデルを通すためにqualificationを緩和しない|
|B構造費用 `23d74855135e718554d824a269427de9943213d1`|P0–P4の既存Σ nominal N E[S]、長時間domainの不足という判断材料|十分N×E[C]ではない。保存比率を係数/長時間上界へ採用せず、今回は再集計しない|

Claudeレビュー `Claude_PRリソース評価_RPE総コスト_統合科学レビュー_20261113.md` の§4.3・§7.3 T3をユーザー指定のローカル原本から確認した。加法性/規模比例残差/C/S安定性や比較倍率別精度は**提案・仮説**として扱う。T3のreceipt再現案も、新しい科学的係数評価や旧値のV2再資格に拡張せず、今回は人工値のcopy/会計だけで確認する。レビューの閾値・候補削減・科学案を正式採択していない。2026-11-09版共通運用指示を適用する。

## 最小の追加実装

`src/trottertracks/rpe_mean_price_model.py`を追加。既存PR専用MeanPriceProvider/V2 lookupはPF・Full-HをOUTSIDE_DOMAINにするため、そこを無理に拡大せず、既存I.price_requestsへ全方式の供給recordを合わせる薄いPriceProviderを追加した。B controller/optimizer、D価格計算器、物理kernel、旧科学SOURCEには差分なし。

今回の人工結合は既存IのH4形configurationを対象とした。Full/TruncatedのS2・Yoshida4・M5やFull-Hの合法layout等は元pipelineの対応範囲を継承し、H6の新しい物理price bridgeや未対応公式を追加したとはしない。H6資産は固定domainの元providerを維持し、未対応requestは元validator/OUTSIDE_DOMAINのままとする。

### 加法式

外部の係数とfeatureを明示的に与えた時だけ、Σ coefficient×featureを評価する。PF/Truncatedは固定費＋PF step数、PRはhead step＋event数＋境界、Full-Hはevent数＋固定費等を表現できる。builtinはone、元physical q、PF時のみpf_step_count、元event_count、q−1、q×(signed pieces−1)。q/Rの全展開はしない。Full-Hのqはgroupingで、qPFと呼ばない。

価格項のcoversはPF_BODY/RTE_BODY/PF_RTE_BOUNDARY/SIGNED_PIECE_BOUNDARY/OUTER_BOUNDARY/SCALAR/WRAPPERからmethodに応じて全体を一度だけ覆う。重複/別method寄与を拒否し、wrapper/scalarはone以外の倍率を拒否する。body係数に内部相殺を含めるなら、同じ相殺を追加項へ重ねない。明示的に内包した寄与は同じ項のcoversへ記録する。

このcoversは**係数の定義・仮定を明示するためのラベル**で、compiler加法性の証明ではない。制御操作は全body/境界の回路定義に含め、独立の追加control費として再加算しない。全体はsymmetric-directional等の元profile＋明示lowering・測定付きfull wrapper/P0で固定する。相殺creditとして負の係数は許すが、合計負価格は拒否しclampしない。

外部の期待featureは同context/law・source・quantityを必要とし、欠測なら価格null。未観測order2を0にしない。event平均の線形性にIIDは不要だが、局所組合せ/cluster期待値には適合するjoint law又はkernelが必要。特徴量の新しい物理解析は起動しない。加法式の名目値はBのexact_scalar/exact_valueで供給scalarを有理数会計する。係数の科学的不確かさを0とする意味ではない。

### E[S]＋E[C−S]

旧combine_meanを呼び、一つのfull-wrapper residual modelだけを足す。raw connected κ(C)、RTE内部だけの残差、C1と局所残差の重ね合わせ、加法式とSの重ね合わせは拒否又は欠測にする。元C1 Sは内部/境界相殺込みで、局所補正は同条件の**C−S**に定義する必要がある。scalar/wrapper/outer/signed-piece残差も含まなければfull平均Cにならない。

Sと残差のSOURCE/law/compiler/lowering/physical条件の一致をcontextで検査する。宣言の一致だけで実samplerとideal lawの一致を証明しない。C/S比を使う場合も、残差=(ρ−1)E[S]とする別の仮説・精度評価が必要で、標本の比をlaw平均比に置き換えない。今回この実データ変換は実行していない。旧combine_meanのbinary64演算と数値未資格を保つ。

## B接続・検証・限界

quote_modelはMODEL_PREDICTION/UNQUALIFIED_MODEL/qualified_for_candidate=falseを返す。DIRECTはPriceProviderが外部recordをcopyするだけ。finite cohort、全law宣言、path数、source/uncertaintyは元のままで、Cが資格を作らない。複数cohortは明示選択、欠測/FAILED/外挿を適格価格へ変えず、CI/joint covariance/law距離/長qR systematicの未知nullを保持する。

人工新11＋旧mean kernel/provider15＋統合10＋pinned B metadata2＝**38 PASS**。5つのラベル（Full PF/B0/B0I/PR/Full-H）、X/Y/q/R、boundary一回、S/C区別、残差重複、SOURCE/law/lowering不一致、受動lookup、既存B費用会計を確認した。人工DIRECT・適格N fixtureでは既存G会計へ接続できるが、同Nでも未資格modelはG UNKNOWNのまま。作者自身の技術確認で、科学的な独立レビューではない。

初回は旧人工fixtureのnormalization flag欠落、次回はRが変わらない整数丸め領域のtest入力で停止した。fixtureに明示falseを追加し、人工R対照を元B mapperの異なる整数R領域へ変更した。production guardや科学閾値は緩和していない。最終ログはTESTSに保存した。

加法評価の仕事量はfeature/term数に依存し、供給済みE[S]との結合とlookupはqR列を保持しない。物理featureの取得コストは別で、実m12性能を計測したとはしない。MeanKernelの有限状態成立条件・数値区間契約は元のまま。実H4/H6の構造・平均価格・calibration・receipt再解析は一切実行していない。

モデルを正式費用へ使うには、[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)の同domain直接価格、独立holdout、law/rare/共有誤差と必要精度の根拠、BのN/risk資格が必要。科学的採用は別判断。commit/push・remote照合後 **mandatory STOP**。
