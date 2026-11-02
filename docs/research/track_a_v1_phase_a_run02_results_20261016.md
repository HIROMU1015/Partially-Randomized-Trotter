# Track A V1 Phase A run02：追加signal計算完了・mandatory STOP

利用者の「修正してもう一度行って」に基づき、修正版SOURCE
`26b9fde60df11ec46dedd97e81b066880014bdf3` を別grant・別output/controlへ固定して一度実行した。
**新104候補のsignal/bias/適格性とPR診断8件を取得し、native exit0・全27 worker回収後にmandatory STOPした。**
[run01の停止記録](track_a_v1_phase_a_run01_stop_20261016.md)・grant・旧結果は保全した。

[全表・終了監査・binding入口](../../artifacts/resource_applicability/track_a_v1_phase_a_run02/2026-10-16/README.md)。
branchは `track-a-v1-h6-deterministic-readiness-20261016`。科学演算provider・数値allowance・比較条件の変更0。
今回の差分はobserverの現在executableのpeakとlifetime peakの区別・値の保存であり、資源上限は増量していない。

## 対象と取得件数

linear H6、1 Å、STO-3G、DF rank19、12 modes、sector400、T=.8、q=1/2/4/8。
保存H6 Hamiltonian/stateとarchive `ad52f9f6…` を再利用し、SCF/DF分解/state solverやGaussian再生成は行わない。

|取得したもの|件数|
|---|---:|
|B0追加signal|52|
|同じsignalからの解析的B0+I overlay|52|
|共通full reference pair / reference matvec|1 / 400|
|prefix exact truncated reference pair / tail scalar|9 / 9|
|登録primitive time / 3probe検証|398 / 1,194|
|PR prefix3/4のexact-tail outer-S2診断|8|
|新compile・full native wrapper・trajectory sampling|0|

追加S2はprefix6〜14のうち旧p10を除く32候補/手法、追加S4はprefix11〜15の20候補/手法。
B0・B0+Iの両方を含む。旧p10-S2の8件は保存再利用の対象で、今回の104件には含めない。
旧104候補を再計算せず、PR診断を正式候補に追加しない。
PR診断のnormalization・shot・費用は未評価であり、formal PR適格性の判定ではない。

## 保存された適格性判定

5精度、factor1/10、alpha_axis=.025の既存規則を保持した。
以下は**今回の追加候補だけ**について、両軸・経験的u・headroom guardを通過した候補数。
factor1と10の判定は全1,040行で一致した。

|ε|B0 S2 /32|B0+I S2 /32|B0 S4 /20|B0+I S4 /20|
|---|---:|---:|---:|---:|
|.05|28|32|20|20|
|.01|12|12|20|20|
|.005|12|12|20|20|
|.001|8|8|15|15|
|.0005|4|4|15|15|

既存の経験的allowanceに条件付けた診断であり、厳密な数値誤差certificate・familywise winner保証ではない。
signed discard/PF分解とscalar/roundoff receiptを全行に保持した。
**厳密な保守的bias保証は未解決**で、相殺のある点biasから保証を新たに認定しない。
compiled費用・affine model・PR/PF比・最良候補は今回算出していない。

PR診断のq=8では、outer-PF signed biasの(real,imag)は
prefix3で約 `(8.730e-5, 1.414e-4)`、prefix4で約 `(8.727e-5, 1.413e-4)`。
exact-tailに限定した診断であり、有限RTE biasやnormalizationを加えた正式PRの性能ではない。

## 数値整合・保存形式

- 全398 time・1,194 probe、負のYoshida timeを含め通過。
- primitiveのnative/expm最大差 `6.43754e-15`、spectral/expm最大差 `5.71403e-14`。
- occupation matrixと独立DF作用の最大差 `3.33117e-15`。
- B0 native/independent signal最大差 `1.52907e-13`、state最大差 `2.36185e-12`。
- 最大u `4.15104e-10`。値は経験的診断でありoperator norm boundではない。

全104 recordを元 `V1_signal_complete.json` とbyte一致するgzipで保存し、展開後1,833,990 bytes・圧縮95,196 bytes。
SHA256と無損失検査を別receiptに保存した。1,040行CSVはsigned bias・u・保存判定の閲覧用。
新たなshot配分・selection・cost推定は行わず、runnerが保存したprecision rowsを集計した。
NPZ・量子state vector・raw config/grant・private絶対path・cacheを公開対象へ含めない。

## 並列実行・資源・修正の確認

primitive16 worker、続いてprefix9＋PR診断2の11 workerを使用し、全27件native exit0・wait/reapを確認した。
観測最大process数はdriver/worker17＋observer1＝18、各thread1。
wall **95.102326秒**、合算RSS peak **6.4083 GiB**、監視間隔最大1.022567秒。
合算RSSは共有頁の重複計上を含む。serial baselineや高速化倍率は測定していない。
各AS32/RSS9 GiB、observer各512 MiB、全体RSS160 GiB、CPU96h、wall12h等の既定上限を保持した。

observerの今回exec peakは52,953,088 bytes、現在RSS37,801,984 bytesに対し、
lifetime peakは647,266,304 bytesだった。旧lifetime判定なら512 MiBを超える一方、
今回executableの実peakは上限内であり、区別が必要なケースを本番値で確認した。
run01の実測値は未取得のままで、その個別原因を遡って断定はしない。

CPUは5秒負荷観測後にdriver31・observer32・worker16枠を新grantへsealした。
run01時のH7 jobは今回fresh gateの保護job一覧に存在せず、run01のCPU割当を流用していない。
SOURCE15/provider230/runtime34,273/input5のbytes/SHAを実行前後で確認した。
有効memory・PSI/OOM・FS/inode・外部監視gateに合格。quota独立証明は未取得として保持する。
旧grant/ledger/carry/one-shot等**646ファイルが全件不変**。共有venv・H5・旧結果・run01を保全した。

## STOPと公開

Phase Aは `COMPLETE_MANDATORY_STOP`。driver/observer/全worker退出、partial再利用・自動retryなし。
費用モデル、最大4軸のcompile検証、費用winner選択は次Phaseであり未実行・未承認。
V2のprimitive生成・理論計算は保留を継続し、指定保存理論結果を変更しない。
主担当の実行・終了監査であり別担当の独立reviewとは称さない。GPT/Claude再review・subagents0。
結果をローカルcommitし、利用者の指示どおりpushは行わずpush commandを提示する。
