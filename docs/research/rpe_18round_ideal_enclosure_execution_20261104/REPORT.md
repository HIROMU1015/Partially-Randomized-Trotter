# RPE Partial18round：ideal categoricalの全36軸局所十分条件が成立

2026-11-04 JST / LOCAL / 主エージェントのみ。
固定SOURCE `f4905fff592351d33ba07b927892fb9532c99711`への別途明示実行許可を受け、m0〜m16を**96 bit・1 pass**で評価した。
**新34軸と既存m17の2軸を合わせ、全18round・36軸でstrict局所十分条件を確認した。**
判定は`CONDITIONAL_IDEAL_CATEGORICAL_LOCAL_CONDITIONS_CONFIRMED`。
これは指定ideal categoricalモデルに限定した結果であり、branch・energy・実回路の成功保証ではない。

## 固定対象・入力・再利用

Linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D3、単一δ=.02、8 qubit・4電子sector70。
保存physical stateのnormalized ray、`extract_identity_phase`、`coefficient_atol=1e-12`、保存q/r/K/N/α/β/εを維持した。
対象は独立uniform53によるcategorical選択、exact-parameterのideal未compile gate、ideal IID ±1測定。
固定PCG64列や元float64仮想測定、compiled/noise/state preparationの保証へは移送していない。

- [固定科学計画](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f4905fff592351d33ba07b927892fb9532c99711/docs/research/rpe_18round_ideal_enclosure_preparation_20261104/ENCLOSURE_EXTENSION_PLAN.md)、[固定入力一覧](../rpe_18round_ideal_enclosure_preparation_20261104/ROUND_INPUT_AVAILABILITY.md)、[fixed_inputs.json](../rpe_18round_ideal_enclosure_preparation_20261104/fixed_inputs.json)。
- V2の保存18round入力はcommit `78c49d0b5f1d71e63b5959e78813237923bbc648`。保存μ0はphysicalのattenuated event mean。model τとsignal τの保存値を混同せず、包含lawにはsignal側を使用した。
- m17結果はcommit `34840ce9bf81d985b32764d9bc9d77645ed52e59`、科学SOURCE `18fffce48e33136b91eb6ac5a6f9a5f77a08d36f`の[既存包含](../rpe_m17_rigorous_mean_enclosure_execution_20261103/REPORT.md)を再利用。m17のpower/mean/判定を再計算していない。
- m17の保存A/F/R・component law/basis・state・保証付きnorm²区間とr32 stepを再利用。H/DF/state/basisの再構成、新しいψ†ψ計算は0回。
- r1/2/4/8/16の5群で既存finite distribution SOURCEとatan/CDFを各1回replay。得られたθ hex・CDF・exact uniform53 grid massは群別原本へ保存。**新SOURCE replayで定義されたlaw**であり、未保存の過去sampler payload identityを認定しない。

## 結果と精度

各軸の包含Iから`U_cat=upper sup_{x∈I}|x−μ0|`を作り、固定N/α/εについて

    upper(U_cat + sqrt(2 log(2/α)/N)) < lower(ε)

を外向きArb演算で判定した。全36軸が`SUFFICIENT_CONDITION_CONFIRMED`。
operator radius、保存norm²分母の不確かさ、K2 builderのπ64加算phase transferを含む。
同じ包含誤差・PF/RTE予算・attenuationを二重計上していない。

**[全18roundのU_cat上端・数値精度一覧](U_CAT_TABLE.md)**。
この表示は原rational上端を12桁で上向きに丸めただけであり、追加の科学再解析ではない。
正式なμ0・N/α/ε、包含endpoint、U_cat、shot半幅、両軸判定・原本selectorは[AXIS_RESULTS.tsv](AXIS_RESULTS.tsv)の36行に固定した。
m17の2行は旧RESULTの転記で、旧SOURCE・結果commitを区別している。

| 科学call・実行条件 | 実施 |
|---|---:|
| m0〜m16のcomplex power/mean包含 | 17セル／34軸 |
| m17包含・判定の再計算 | 0（旧2軸再利用） |
| 新order law/CDF/θ replay | 5群で各1回 |
| 新component/basis replay、H/DF/state solve、norm²計算 | 0 |
| Sampling、RNG replay、実H4回路build/compile、費用・branch/energy評価 | 0 |
| 数値精度／pass | 96 bit／各セル1 pass |
| retry/resume、精度追加、条件変更 | 0 |

## 原結果・実行資源・検証

[RESULT.json](../../../artifacts/rpe_18round_ideal_enclosure/2026-11-04/run01/RESULT.json)のSHA256は
`410221788791280c7f69ed71ad188c13a7bf0db7167b87aac7a01738fc9ba568`。
SOURCE・全入力hash、群別lawと各roundの区間・判定・global radiusを含む。
同じnamespaceの`group_r1/2/4/8/16/32.json`は各workerの開始・完了checkpoint原本。
原本7 JSONとstdout/stderr/resource記録の10ファイルのbyte identityを[VALIDATION.json](VALIDATION.json)に保存した。
実行後は原RESULT/checkpointを編集していない。別の実行結果へ置換・再生成していない。

Python3.11.0rc1／NumPy1.26.4／python-flint0.9.0／FLINT3.6.0、binary64・round-to-nearestを固定SOURCE条件へ照合。
32 CPU affinity、実行前MemAvailable 54,472,232 kBを確認し、6 spawn process・FLINT各5 threadで実行した。
70×70の保存operatorを再利用し、r群内のstepを再利用、binary powerを使用した。GPU driverは利用不可のため固定CPU経路を使用。
`/usr/bin/time -v`のwallは1.20秒、maximum RSSは77,264 kB。このRSSを全workerの合計使用量とは扱わない。
実測比較がないためspeedup倍率を主張しない。共有環境・他ユーザー・他branchの設定は変更していない。

固定SOURCE準備時の人工18件＋既存25件のPASSを再利用した。科学SOURCEは無変更なので同じ人工テストを再実行していない。
実行後は17セルの一意性・欠測なし、各worker checkpointとRESULTの一致、34新軸と2再利用軸の判定・U_exec null、SOURCE/input hashの保全を技術照合した。
主エージェント自身の技術確認であり、独立科学レビューやimmutable CIの証拠とは呼ばない。

## STOP・全体保証の限界

exit0、`COMPLETED_MANDATORY_STOP`。未判定軸、入力不一致、中断、worker errorはなく、追加精度・retryは不要だった。
ここで科学実行を終了し、branch/energyの新評価、Full L_D12、元float64 guard証明へは進んでいない。

全軸の結果はidealモデル下の保存μ0に対する座標誤差事象を支える。
共通E0への系統位相・正の信号半径、物理stateとPF有効固有state、strict branchとdecoder数値丸め、alias windowと境界margin、保存αのexact和、energy reference誤差の保証は別途必要。
V2の保存中心点のfloat64二項確率を新ν_catへそのまま移送しない。
詳細は[既存の全体保証ギャップ](../rpe_18round_ideal_enclosure_preparation_20261104/RPE_GLOBAL_EVIDENCE_GAPS.md)を維持する。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、`U_virtual_IID_upper=null`、`U_exec_upper=null`を保持した。
旧SOURCE/raw/grant/ledger/marker/STOP、旧INCONSISTENT/u_num=null、V2 CONDITIONAL_SUCCESSは無変更。

**公開SHA・tree・全成果物照合後mandatory STOP。**
次はユーザーがbranch/energy接続の技術準備を指示するか判断する。Codexはその別指示後に必要最小限のSOURCEを準備し、新しい科学再解析はそのSOURCE公開後の別途明示許可を受ける。
