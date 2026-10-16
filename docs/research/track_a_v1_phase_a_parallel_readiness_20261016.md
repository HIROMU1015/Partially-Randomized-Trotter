# Track A V1 Phase A：並列signal計算の承認待ち準備

科学計算は未起動。最新の利用者指示に従いV1の準備を継続し、V2は
[既存理論評価との照合報告](track_a_v2_existing_theory_reconciliation_20261016.md)でSTOPする。
旧[科学契約・比較条件](track_a_v1_h6_deterministic_readiness_20261016.md)は保持する。

[実行範囲・静的照合・人工検証の証跡](../../artifacts/resource_applicability/track_a_v1_phase_a_readiness/2026-10-16/README.md)。
branchは `track-a-v1-h6-deterministic-readiness-20261016`。
実行SOURCEはこの準備をcommitした最終HEADを新grantに固定し、旧grantを再利用しない。

## 今回の承認対象

linear H6、1 Å、STO-3G、DF rank19、12 modes/sector400、T=.8。
B0・B0+IともS2追加prefix6〜14、S4追加prefix11〜15、q=1/2/4/8。
旧p10-S2の8候補は保存結果を使うため、新規登録は104候補（B0 52＋解析的B0+I 52）。
PR prefix3/4のq4種類、計8件はexact-tail outer-S2 bias診断だけで、正式PR候補・normalization・shot・費用比較に加えない。

|計算|Phase Aの上限|
|---|---:|
|snapshot load / frozen preparation decode|各1|
|共通full reference pair / reference matvec|1 / 400|
|新B0 signal / B0+I overlay|52 / 52|
|exact truncated reference pair / tail scalar|9 / 9|
|PR outer-S2診断|8|
|primitive correctness probe|20,000以下（登録timeを3probeで照合）|
|SCF・DF分解・state solver・trajectory sampling・新compile・full native wrapper|0|

5精度・factor1/10、元の経験的allowance・同一reference・保存状態・演算順を保持する。
相殺を除いたbias分解は補助診断であり、厳密な保守的bias保証は未解決のまま。
cost fitting、最大4軸のcompile検証、winner選択・費用比較は次Phaseであり今回起動しない。

## 並列化と高速化

共通snapshot/archive/referenceを親で一度だけ準備し、copy-on-writeで継承する。
最大prefixのnative action列と16 blockのoracle spectraを共用し、prefixごとの同一block再準備を避ける。
primitive検証は16 blockタスクを最大16並列。通過後、prefix単位9タスクとPR診断2タスクを最大11並列。
各prefix内でS2/S4・qをまとめ、exact truncated referenceを一度だけ計算する。
B0+Iは同じB0 signalとprefix scalarから解析的overlayを作る。
原providerのnative/independent oracle・expm検査・数値allowanceを保ち、行列近似への置換は行わない。

親がfork前にtask全体の予約をfsyncし、workerはメモリ内で呼出数を検証する。
全体上限は親が管理し、native exit0/wait4・予定呼出数・結果hashの確認後だけ統合する。
worker失敗時は予約を返却せずSTOP。自動retry/resume・失敗partialの再利用はない。
実H6での高速化倍率・総所要時間は未測定。

## 資源案・監視

最大18 process（driver1＋worker16＋stdlib observer1）、各thread1。
driver/worker各AS32 GiB・RSS9 GiB、observer各512 MiB、全体RSS160 GiB。
driver/worker各CPU8時間、全体CPU96時間、wall12時間、output2 GiB、stage保存4 GiB。
これは未承認のSTOP ceilingであり所要時間予測ではない。
CPUはdriver/observer/worker16の計18枠をfresh gateで他jobと競合しないよう選び、grantへ固定する。現時点はCPU[]。

独立observerが1秒間隔でdriver/workerのPID/start/parent/UID、thread、AS/RSS、全体CPU/RSS、
memory headroom/PSI/OOM、disk/inode、出力を確認する。観測間隔5秒超はSTOP。
driverとworkerは親死亡時に停止し、launcherはsubreaperで孤立したown childも回収する。
TERM→KILL/wait4は所有確認済みprocessに限定する。H5等へのsignal・affinity変更は行わない。

## SOURCE・input・profileとgate

科学provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、230 source fileを固定。
既存Python3.11 venvを変更せず使い、runtime profile34,273 fileのbytes/SHAを静的照合した。
H6 snapshotは150,052 bytes / SHA256 `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`。
archiveは5,670,237 bytes / SHA256 `ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010`。
自前SOURCE15 file・provider230・runtime34,273・入力5件が全一致。科学array展開・科学importは0。
compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、rz/sx/x/cx・opt1・seed17を保持するが本Phaseのcompileは0。
旧profileとcurrent profileの差・旧compiler出力の未認定事項はそのまま残す。

承認後にのみ、新grantへ最終SOURCE/config/plan SHA・上限・18 CPU・新output/control/one-shot lockをsealする。
起動直前30秒以内のsource/profile/input/resource/CPU/保護job/外部monitor gateを検証し、
FS空きはstage4 GiB＋reserve2 GiB・inode4096、有効memoryは全体RSS＋headroom16 GiB以上を確認する。
既存quotaの独立証明は未取得であり、FS容量確認と区別する。
launcherは明示grantなしでは拒否する。private configは `approved=false/runtime_authorization=false`、command=null。

## 限定検証・STOP

旧30＋新13＝44人工tests PASS。人工2 workerの重なり・affinity・native wait/reap、失敗時の兄弟回収、
deadline、driver異常終了時の孤立worker回収、task予約、grant/role/fresh gate、scalar/両oracle診断を検査した。
実H6の科学計算・速度実測・新grantは0。主担当の検査であり独立reviewではない。
旧manifest全体checkerの歴史的拡張キー非対応は残るため、全体CI成功とは称さない。

**現段階でmandatory STOPし、V1 Phase Aの明示実行承認を待つ。承認後もPhase A終了時にSTOPし、次Phaseへ自動移行しない。**
