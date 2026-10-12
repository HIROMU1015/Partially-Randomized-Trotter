# N3幅比較 v2：固定6入力・実行前比較設計

2026-10-12レビュー系列。科学レビュー第11・14節の範囲内で実行詳細を結果前固定する。
[数学・対照契約](n3_width_comparison_v2_contract_20261012.md)。**準備のみ。科学batchは未認可。**

## 固定集合

[input_pack.json](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_pack.json)、[SHA索引・schema監査](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_identity_audit.json)。
3 stress recipe×2 sizeの6H、各HのD_set={0,n−1}、計12候補mode。
full modes5/6、active4/5、ν2/3、正のε_proj=1/100。実対称h、全unordered pairの非負density。
geometry、化学basis、DF rank/rank policy、L_D、PF時間窓は適用なし。unitは係数recipeの共通energy単位。
原12入力、原8入力、S1はdevelopmentとして保持し今回の成績に加えない。

| family | n=5 seed | n=6 seed | 係数生成の特徴 |
|---|---:|---:|---|
| banded | 20261012031 | 20261012032 | 隣接coupling、非負density、正の候補外部対角 |
| dense_density | 20261012041 | 20261012042 | 全pair coupling、やや大きいdensity、控えめな候補外部対角 |
| two_external_stress | 20261012051 | 20261012052 | 同じHに2候補、異なる外部対角 |

Python standard randomの固定seedで有理literalを生成しbytesを保存した。NumPy/solver/U/c/採否/幅の評価はしていない。
成功するように結果で筛選した集合ではなく、事前開示したstress collection。IID分布性能、独立baseへの一般移送、旧2-base成功とは表示しない。
εを結果後に変更、NORM不受理だけ抽出、成功familyを増やす、追加solver level/tiers、原入力再探索を行わない。

## 二層と計測

kernel層：同じ証拠付きU,cを事前取得して各法へ渡す。共通bound取得費用を別recordにし、method kernelへ無料以上の特別情報を渡さない。
cold層：各法のfresh Engineで係数からbound→候補→有理化／修復→exact checker→幅まで計測する。
FULL_H_LOWERは独自の係数lower経路であり、両層の表示でもsame-U/c kernelの費用順位と混ぜない。L_Hは一Hにつき一回取得し各modeへbindする。不要なcは取得しない。
各層3 timing repetitionを結果前固定。反復は同じHであり、新しい独立入力数ではない。
CVXPY/NumPy/SCSのdependency importはworker setupに分離し全method timingから除外。prototypeの係数・証拠acquisitionを測る。
各methodで証拠／数値cacheをfreshにし、同method内の同じmatrixはexact SHAで再利用する。SOSの普遍的(m,ν) CAR templateはそのpath内でreuse。
kernel、cold、warm primitive hits、全factorial harnessのwallを区別する。既存science証拠のwarm taskは今回測定しない。

wall/CPUはbound、numerical proposal、rational repair、independent checker、width/output、SOS assemblyのexclusive timerを記録する。
mode selectorとmethod/endpointのin-memory serializationの実測時間・bytesを別に保存し、Hのmethod path totalにも加える。
共有のdisk IOはrunner harnessの費用に属する。OR coverageの再構成を独立運用測定と呼ばない。
explicit NumPy eigh、solver calls/iterations、PSD probes/checks、有理演算／bit、証拠bytes、失敗理由を保存。
SCS内部eigensolver数は未instrumentedで、managed NumPy callsへ加算／0認定しない。
RSSはEngine開始/終了とprocess lifetime highwater、runner+descendantsの50ms目標sampleを記録。孤立したmethod allocation peakではない。
この限界で厳密なmethod memory優位や短時間の安定speedup倍率を確定しない。

## 実行上限

CPU2 spawn workers、各numeric thread1、worker AS3GiB、aggregate sampled RSS6GiB、開始前available8GiB以上。
solverはCVXPY1.6.5/SCS3.2.8、eps_abs=eps_rel=1e−6、max_iters20000、normalize=True、acceleration_lookback=10、warm_start=False、direct solver。
1 mode/methodでmanaged eigh128、SDP3calls、PSD repair probes256、checker10,000,000 monitored q calls、rational4096bits。
proposal40bits、有限PSD repair12段、root128bits／scaled integer8192bits。Fraction内部temporaryを全監視したとは言わない。
ν/m、block generator、trial pool8、κ score32更新、δ tiers3を固定する。
全science output128MiB、JSON shard8MiB、log8MiB、停止用64KiBを予約。OS FSIZEも8MiBでstdout/stderrを含むper-file出力を制限する。
wall/CPU time capは設けない。ツールpoll timeoutは科学上限とは別。
同一capに達したあとbudgetを増やして再試行しない。attempt1、retry0、run1 fresh、全結果後Mandatory STOP。

計504 mode-method records（6H×2候補×7法×2層×3反復）。managed solver callの上限は同等SDP216＋SOS216=432。
この上限は原S1 one-shot認可とは独立した新scopeである。source/plan/input SHAにbindした新認可だけを受け付ける。

## 集計・判断の境界

全input/family・全methodを残し、未判定を分母から除かない。
同mode採否・有限発見幅、H単位の一mode縮約成立数、rawとoperational ORを分ける。
3反復の幅変動、wall/CPUのmedianとrange、calls/bit/bytes、bound precondition fail／有限未判定／技術・soundness停止を報告する。
同等SDPに対する理想保証優位は成功要件にしない。比較するのは有限budgetで同保証を取得できたか、その費用。
強SOSに対して幅／費用の非支配点が残るかは新結果後の科学的判断事項。大きな改善率thresholdや新2-base gateを追加しない。
H単位で両法が別modeを1個削除できる場合、追加mode候補の受理を追加qubit削減へ読み替えない。
元NO_ADDITIONAL_ACCEPTANCE_OBSERVED／旧gate／N1停止／N2保留／中心テーマ未採択を維持する。

次は準備と技術preflightの確認を経た明示実行認可。通常のsource/test/publicationのみで同内容のGPT詳細レビューを再要求しない。
前提・対照・主要taskを変える問題が生じた場合は、変更前にGPTへ戻す。
