# H5中性開殻：限定pilotと接続修正

利用者の「実際に計算を初めてよい　計算が完了しなかったら修正して」に基づく、中性H5のROHF/DF/state生成と既定6構成pilot。科学条件の追加やH7/full resource-mapへの進行は認可していない。

- SOURCE：`d86a4673d6ada701797d3e035f745d0d4faf5150`
- 実行ID：`h5-neutral-open-shell-pilot-20261016-run04`
- [保存結果・監査入口](../../artifacts/resource_applicability/track_a_h5_neutral_open_shell_pilot_v1/2026-10-16/README.md)
- [元の準備資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/803a24ef7a01d8c7394dde88357c74168795bb34/docs/research/track_a_h5_neutral_open_shell_readiness_20261016.md)

## 固定条件と実装

Linear H5 / 1.00 Å / STO-3G / charge0 / ROHF共通real spatial MO / 5 electrons / Nα3,Nβ2 / 10 spin modes / sector100 / T=.8 / q1。DFはtol-only1e-8・cutoff0・signed generation order・weighted Hermitian projection。実rankは14、prefix ceil(14/4)=4で全returned rankを保持した。

B0p4、B0+Ip4、full PF S2、global Yoshida S4、PRp4/r2/K2、PRp4/r2/K4の6構成。最初に各1groupの12compileを取得し、いずれかの要求精度に適格なPRだけ残7trajectoryを取得する、最大20group/40full-wrapper/16trajectory/32RTE eventsの限定panelである。

要求signal精度 .05/.01/.005/.001/.0005、P=0のshot-weighted RZが主。探索・独立確認・希少次数pilotの標本と混合しない。固定prefix/q/r/Kを変更せず、全候補最良・母平均優位性・厳密numerical allowanceの保証は主張しない。

α/β別occupation eigensystemとfermionic sign、位相LRU、3種preparation共用、evolutionのcos/sin再利用、content-checked streaming fingerprintを使用。通常は最大4group並列・内部thread1、S4は単独。CPU worker61/60/59/58、coordinator57、AS32GiB/worker RSS9GiB/aggregate38GiB、headroom16GiB。own-process affinityだけを設定し、共有環境・venv・他jobは変更しない。

Python3.11.0rc1、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、Numba0.60.0、OpenFermion1.6.1、PySCF2.7.0。既存private venvはinstall/updateなし。環境profile digest `70ebab6aec9b79b21592da2d928f3b68c15ba5b152be52560014596a8dca6992`、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。旧hostとのcompiler出力のbit完全同一性は未検証。

## 失敗からの技術修正

run01はfresh CPU gateで起動前STOPし、科学call/one-shot消費0。run02はROHF/DF後にsolverのcounter interfaceでSTOPした。既存bounded solverが参照する `Calls.limits` を追加し、call予約journalとraw積分/DFの早期保存を追加した。SOURCE `14337d27ea5503e0dd0ad4ac5880eb749a3f5d35`、33人工/mock検査PASS。

run03はinput/state/native primitive/signalとS4の2compileを完了し、K2の回路構築でSTOPした。固定Kに対しrequest APIの既定truncation tolerance1e-10が適用された接続不備である。既存H6/H8と同じ `_explicit_cutoff_tolerance(lambda_R*.8/2,K)` を指定し、既定のK・有限分布・component選択・信号bias・normalization・shot allowanceを保持した。

追加の人工high-tail fixtureで、既定toleranceでは拒否されるK2とK4のevents/legacy explicit回路との同一性を確認した。最終34人工/mock検査PASS、実科学compileをテストのために追加していない。主担当の検査であり、別担当による独立review/CI再現ではない。

run03の全workerをwait/reap、cleanup error0、native exit1を確認してから新SOURCE/run04を起動した。過去のknown actual・未完了予約・unknown actualは保全し、返却/resetしていない。run04のPR seedは新SOURCE/inputに結合し、過去partial結果と混合していない。

## 入力とstateの実検証

保存したHamiltonian actionの独立100×100全column照合、state norm/eigen residual、S²、reference split、full-system native primitiveを検査した。DF rank14は予測値から合わせたものではない。ROHF determinantを評価stateとして代用せず、DF sector solver stateを使用する。

実測値・signal・allowance/shotはbundleの `input_state_signal_v1.json` を参照。S²≈.75、target residual≈4.9e-16。これは固定Ms-sectorのengineering doublet確認であり、全spin/Fock空間の基底状態証明や厳密数値certificateではない。

## 実結果

20group・40compile・16PR trajectory・32RTE eventsが全件完了、native exit0、wall249.438秒。DF/state生成は各1回、solver41matvec、独立action100column、native285probe、6signalを取得した。全worker exit0/wait/reap、cleanup error0、own-run session残存0。実行後もSOURCE264file・環境14817file・snapshot SHAは一致した。

要求精度ε=.05、P0、経験的allowance下ではPR K4の総RZ点推定215,276,664に対し、限定panel内の最良決定論的対照fullS2は388,818,084、比.553669（約44.63%減）。PR K2は217,125,228.25、比.558424。各8trajectoryの有限標本点推定で、未観測次数を含む母平均の上界は得ていない。

PRはε≤.01で不適格、fullS4は.01/.005でも適格。このpanelでは.001/.0005を満たす構成はない。不適格を費用0として順位付けしない。

S4のwall122.802秒・worker HWM1.918GiB、1秒観測の全stage最大同時RSS1.768GiB。最大監視間隔1.003013秒/処理.007409秒、first STOPなし。通常4group並列を使用したが、旧直列版との速度倍率は測っていない。

## 実行終了と次段階

最終状態、完了件数、native exit/reap、費用、資源観測はbundleの監査を参照する。資源観測は別coordinator、原則1秒・最大5秒、own-run pidfd停止、first STOP保持、pressure/OOM/AS/RSS/filesystem/output監視を行う。

新しい計算へ自動進行しない。H7は未実行、H5全resource-mapも未実行。H7には別入力・sector1225のsolver/reference adapter・namespace/契約・実行判断が必要であり、このpilotの結果だけから移行を保証しない。

NPZ/raw runtime/cache・認証情報・内部SSH情報はGitへ収録せず、軽量JSON/CSV/検査logと本資料だけを公開する。private evidenceは `/home/AbeHiromu/projects/h5-handoff-evidence/20261016/pilot-run04/`。
