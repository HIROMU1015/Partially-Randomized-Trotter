# H8 Phase C：通常signed biasの5精度費用再会計・一回完了

**登録1,900行・5精度×2倍率の会計を一回で完了した。driver/launcher native exit0・全回収・最終gate PASS。**
強化184候補にモデル推定費用を適用すると、PR確認n32の削減率はε=.01/.005/.001で正、.05/.0005で負となる。
全精度でのPR優位を主張しない。強化集合の最良費用は全条件でモデル推定を含み、全候補実測最良ではない。

## 固定条件と来歴

linear H8、隣接1.00 Å、STO-3G、保存tol-only DF rank28、16 spin modes/sector4900、T=.8。
q=1/2/4/8、ε=.05/.01/.005/.001/.0005、allowance倍率1/10、通常signed bias、既存固定shot/normalization/適格性、P=0。
指標はstate preparationを除いたshot-weighted full-wrapper RZ。
比較集合は旧32決定論候補、追加152を加えた184候補、PR確認union6（各n32）。探索n8・非相殺shot・別分子は使用しない。

- 実行SOURCE：`931755c4c84aa0751dbde7ddf84cb5187b5172ae`
- 公開確認済み準備REVIEW：`b357c77a1a35d70dde011d44516bd8cab3bfcc8a`
- plan：`79d624b9bc1f10232b7cd5e48ce0d4468f7a4375ba6608d6ab832a5f27947232`
- config：`6ce100df4f8e975e9e37347949380cb855e6e664640957eef61f573f8a762238`
- 一回grant：`21775ba66e4dcb2bac84cd0682ff56b079a625ff84ef9fc0c33239e828950561`（消費済み、再使用不可）
- user承認原本SHA：`d016b1533218e686463716f2b1f7cf4880e232f42aeba7079f698756f2f6210a`
- profile：`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`
- compiler：`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`
- B1固定model：`8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0`
- [結果・監査bundle](../../artifacts/resource_applicability/track_a_h8_phase_c_run01/2026-10-17/README.md)
- [入力12件のbytes/SHAと固定commit/path](../../artifacts/resource_applicability/track_a_h8_phase_c_execution_binding/2026-10-17/input_manifest_v3.json)。入力契約SHA `6965f5b04b8764ed12946c36c4fa204bf913871bb8404557d8d2d648c6f53246`。
- [SOURCE固定・人工49件・旧plan拒否](track_a_h8_phase_c_execution_binding_20261017.md)。SOURCEは同一のため人工campaignを反復していない。

## 5精度×2倍率の費用と削減率

費用欄は読みやすく丸めた表示である。[exact rational CSV](../../artifacts/resource_applicability/track_a_h8_phase_c_run01/2026-10-17/comparison_summary_exact_v1.csv)と元比較JSONに正確な分子/分母を保存した。
削減率は100×(1−PR/決定論費用)。負値はPR点費用がモデル条件付きの決定論費用より高いことを表す。
変化は「184基準−旧32基準」のpercentage points。

|ε|倍率|旧32最良G|184最良G（モデル）|PR確認n32 G|旧32削減率%|184削減率%|変化pp|
|---:|---:|---:|---:|---:|---:|---:|---:|
|0.05|1|1.92188318e+09|1.27780264e+09|1.36899826e+09|28.767874|-7.136909|-35.904783|
|0.05|10|1.92188318e+09|1.27780264e+09|1.36899826e+09|28.767874|-7.136909|-35.904783|
|0.01|1|1.28573352e+11|9.08603469e+10|6.07920792e+10|52.717979|33.092838|-19.625141|
|0.01|10|1.28573352e+11|9.08603469e+10|6.07920792e+10|52.717979|33.092838|-19.625141|
|0.005|1|6.27154923e+11|4.31531286e+11|2.96433476e+11|52.733612|31.306608|-21.427004|
|0.005|10|6.27154923e+11|4.31531286e+11|2.96433731e+11|52.733572|31.306549|-21.427022|
|0.001|1|3.50050094e+13|1.99380514e+13|1.53600067e+13|56.120547|22.961345|-33.159202|
|0.001|10|3.50051226e+13|1.99380919e+13|1.53600993e+13|56.120424|22.961037|-33.159388|
|0.0005|1|1.86504990e+14|9.39887100e+13|1.42054819e+14|23.833234|-51.140301|-74.973535|
|0.0005|10|1.86506889e+14|9.39897081e+13|1.42057710e+14|23.832459|-51.141772|-74.974231|

|ε|旧32最良候補|184最良候補|PR確認union6内の点費用最良候補|
|---:|---|---|---|
|0.05|B0I_p14_q1_2nd|B0I_p11_q1_2nd|B2_p7_q2_2nd_r4_K2|
|0.01|B0_p21_q4_2nd|B0_p15_q4_2nd|B2_p7_q4_2nd_r4_K4|
|0.005|B0_p21_q4_2nd|B0_p15_q4_2nd|B2_p7_q4_2nd_r4_K4|
|0.001|B0_p21_q8_2nd|B0_p15_q8_2nd|B2_p7_q8_2nd_r4_K4|
|0.0005|B1_p28_q4_4th|B0I_p15_q4_4th|B2_p7_q8_2nd_r4_K4|

候補IDの `B1` はfull PF対照、`B2` はPRを表す。Phase B1/B2というstage名とは別である。

## 実測・モデル・有限標本の区別

B2最終受入 `fef2db7a79886ab682b652da7266b463c98c7d3a` の登録4軸はすべてPASSとして受け入れた。
B0 p15/q1 S4は両軸168353 RZ、B0+I p21/q8 S4は両軸1852640 RZ。
軸別SOURCE・native exit/reap・最終gateの結合を維持し、過去のSTOPや未settle予約を変更していない。
単価優先順位はB2登録S4実測→B1登録anchor実測→旧full PF保存実測→固定model推定。

旧32の最良は、ε=.05のB0I p14/q1 S2、ε=.001のB0 p21/q8 S2がB1登録実測anchor、ε=.0005のfull S4は旧保存実測でprofile unknownを保持する。
ε=.01/.005のB0 p21/q4 S2は固定model推定である。旧32基準も全候補実測最良という意味ではない。

184集合の最良はすべて固定modelによる条件付き点費用である。
ε=.05のB0I p11/q1 S2は学習prefix14〜21の外挿を含む。
ε=.01/.005のB0 p15/q4 S2、ε=.001のB0 p15/q8 S2、ε=.0005のB0I p15/q4 S4も未測定候補である。
特にS4の仮説は登録4軸でのみ検証済みであり、p15/q4など全格子で2%未満の精度を保証しない。
モデルの再fit、未承認compile、係数や候補の事後変更は0。

PR費用は独立確認n32の有限標本平均を保存値から使用したものである。
paired SEも結果JSON/CSVに残したがCIやPR母平均優位の保証ではない。union6からの選択後winnerにも保証を付けない。
厳密bias保証や未観測Taylor層の上界を主張しない。

## 完全性・不適格・欠測

全190候補（決定論184＋PR6）×5精度×2倍率＝1,900行、重複/欠測/未解決0。
適格980行、不適格920行。不適格の費用はnullで、0扱いしていない。
実測点費用120行、モデル条件付き点費用816行、PR有限標本費用44行。
各条件の旧32/184/PR6について適格数を監査JSONへ保存した。
新しい科学評価の呼出しは一回のみ。保存結果の構造・receipt・exact比率を照合し、evaluateやランキングの再実行は行っていない。

## 起動・終了・保全

公開branchのREVIEW一致を確認してから、新grantを別fileへ発行した。旧INVALID grantは変更していない。
source8 files・入力12件・Pythonを厳密に照合し、CPU・実効メモリ/cgroup・PSI/OOM・容量・inode・quota・未使用output/lockを確認した。
CPU1をこの軽量な算術processに選び、内部thread1。これは本処理の配置であり、将来の資源上限ではない。
新たなCPU/RAM/wall capは設けず、PSIは観測のみ。他processや共有環境/既存venvは変更していない。

- 開始：2026-10-17 19:08:17 JST（launcherをdispatchした時刻）
- 終了：2026-10-17 19:08:18 JST（callerがlauncherのnative終了をwaitした時刻）
- 科学dispatch 1回、会計pass1、保存行1900、比較20、unique model軸推定268（上限352以内）
- new signal/shot/compile/fit/sampling 0、retry/resume 0、実行時技術修正0
- driver wall 0.355289秒、peak RSS40,972,288 bytes（約39.07 MiB）
- launcher全体wall 0.702641秒、driver/launcherのnative exit0・wait4/reap済み、残存0
- OOM counter増分0、stdout/stderr空、SOURCE・入力・config・grantは実行後も一致
- 最終gate `PASS_MANDATORY_STOP`、次stage認可false

結果の軽量JSON・1900行gzip・exact CSV・終了監査のみを公開対象とする。元runtime/control/lock/使用済みgrant/private logはhome下に保全し、一括公開しない。
B2/M1-S/他worktree・既存grant/ledger/marker/STOPへの書込0。無関係なworktreeの資料更新は観測のみで、禁止していない。

## 次の判断とmandatory STOP

今回のモデル条件付き比較ではε=.05/.0005のPR点費用優位が消えており、従来の限定比較と解釈が変わる。
**ユーザーが結果を確認し、必要なGPT科学判断へ戻す。** Codexは新しい科学レビュー・追加計算・モデル修正・次stageを自動開始しない。
公開成否とremote SHAは最終チャット・private publication receiptに記録する。認証失敗の場合はlocal commitとそのまま実行可能なpushコマンドを保全する。
