# Track A V1 Phase B：保存費用モデル固定と最大4軸検証

終了追記：[登録S4全4軸PASS・終了STOP](track_a_v1_phase_b_run01_results_20261016.md)。以下はcompile前に固定したモデルと準備の履歴であり、モデル・予測・transfer receiptは実測後も変更していない。

Phase A結果commit `bf865de1efae51ce1663c878ad1e9daa5bb915c2` の追加104候補と既存実測費用を照合し、S2の4独立stratumと、S4の4軸予測をcompile前に固定した。
本資料作成時点では新compile・grantは0、S4は未検証。利用者の後続指示「計算を実際に開始するところまで進めて」により、準備公開・remote SHA確認・fresh gateの後、同Phase Bの検証だけをone-shot起動する。完了または異常でmandatory STOP。次の5精度費用再評価は自動起動しない。

[固定モデル・照合・契約の入口](../../artifacts/resource_applicability/track_a_v1_phase_b_readiness/2026-10-16/README.md)。実装は[固定処理](../../scripts/prepare_track_a_v1_phase_b.py)、[grant限定launcher](../../scripts/launch_track_a_v1_phase_b.py)、[compile adapter](../../scripts/track_a_v1_compile.py)、[契約](../../src/trotterlib/track_a_v1_contract.py)、[人工tests](../../tests/test_track_a_v1_phase_b.py)。サブエージェントは使用していない。

## 比較条件と固定モデル

linear H6、間隔1Å、STO-3G、tol-only DF19、12 modes、sector dimension400、T=.8、q=1/2/4/8、δ=.8/.4/.2/.1。打切りS2追加prefix6–14（旧10は再利用）、S4追加prefix11–15、B0とB0+Iの両方。
P=0・RZ、symmetric_directional、rz/sx/x/cx・Qiskit1.3.0・optimization1・seed17・topologyなしを維持する。compiler hashは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

RZのS2モデルは `a + b*q + d*prefix*q`。cosine/sineを別々にfitし、method間で係数を共有していない。完全な保存実測の全アンカーを使用し、有理数で最小二乗を解いた。選択基準は新compileの結果に依存しない。

|method|各軸のアンカー数|a|b|d|最大アンカー相対残差|
|---|---:|---:|---:|---:|---:|
|B0|8|797|109|9708/5|0|
|B0+I|6|797|269|9613/5|0|

B0は保存exploration raw16軸を元result `df99d6ee28942df8d12aac3c4db7cd202df55d77` のGit blobから読み、D0 provenanceのbytes/SHAへ照合。全16軸の元SOURCEは `54986a9f54515c8c2289e38fa16fd81e2b6008c2`。B0+IはSOURCE `ac6d28e35cf01398f169ce59577182a38da9b959` の新測定12軸だけをfitへ採用し、旧pilotの別profile6軸は除外した。不適格・未測定を0埋めしていない。

Phase Aの104候補それぞれについて、5精度×allowance1/10の保存適格性と費用取得状況を[照合表](../../artifacts/resource_applicability/track_a_v1_phase_b_readiness/2026-10-16/saved_cost_coverage.json)に保存した。追加候補の新測定費用はまだない。PR prefix3/4の8診断は正式比較候補にしない。保守的bias certificateはnull、数値allowanceは経験的であり、元の適格性・shot規則を変更していない。

## SOURCE・profile・phaseの転用条件

native provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` と旧B0 SOURCEの間で、native builder、control schedule、fingerprint、matched port、DF partial builder、diagonal primitive、Hadamard builderの7ファイルがbyte一致する。archive `ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010` のfull prefix viewと旧discard5/10/15のblock参照も一致する。

ただし旧B0 legacy449のbinary profile在庫はなく、旧compiler出力とのbit同一性は未確認。この不足を架空のprofile hashで補完せず、`profile_hash`欄には「legacy profile未在庫」の来歴receiptのdigestを使い、その種類をtransfer receiptへ明記した。B0+Iアンカーのprofile `622ce4a8…`、archive export時profile `4e32bb9b…`、現在runtime closureも別々に保存する。profile一致・旧binary互換性の証明とはしない。

したがって現profileへの転用とprefix方向の線形性は条件付き仮説であり、これからの4軸検証で棄却可能とする。transfer receiptの`verified`は、この区別を含むSOURCE・入力・制御・phase・来歴監査が済んだことを意味し、費用同値の証明を意味しない。matched B0/B0I差によるidentity改善の因果解釈も行わない。

S4は保存S2測定と区別した未検証のschedule仮説 `a + 3*b*q + 3*d*prefix*q`。未mergeの符号付きYoshida3段を合成し、wrapperとscalar ancilla phaseは1回だけとする。B0+Iの追加scalarはPhase Aで保存したfull-Fock tail Cを使用し、新しいscalarを計算しない。

## 登録4軸、即STOP、資源

signal適格候補だけから、B0は辞書順の最小(prefix,q)、B0+Iは最大を選ぶ既存規則に従った。

|順序|構成|軸|固定予測RZ|wrapper命令上界|
|---:|---|---|---:|---:|
|1|B0 p11 q1 S4|cosine|65196.8（325984/5）|25529|
|2|B0 p11 q1 S4|sine|65196.8（325984/5）|25529|
|3|B0+I p15 q8 S4|cosine|699389|275141|
|4|B0+I p15 q8 S4|sine|699389|275141|

命令上界は保存codecのblock構造をstdlibで数えたメモリ入場判定であり、新しいprimitive実測や理論費用計算ではない。既存のnative100万命令・transpiled500万命令cap内。

1軸ごとに `abs(exact_prediction - measured_RZ)/measured_RZ` を有理数で評価し、**2%以上なら即STOP**。最初の不一致の実測・入力fingerprint・固定model SHAを保存して停止し、次軸compile・再fit・追加compile・自動retry・resumeを行わない。
この停止条件を満たすためcompileの同時実行数は1。snapshot/archiveを各1回だけload/decodeし、同構成のevolutionとcontrol probeを両軸で共有する。監視は別processで並行して動く。速度向上の実測はまだ主張しない。

最大予約はsnapshot1、archive decode1、native evolution2、wrapper4、control probe8、compile4。他のsignal/reference/SCF/DF分解/state solver/trajectoryは0。
driver1＋observer1、各thread1・別のidle CPUをfresh gateで固定。driver AS32GiB/RSS9GiB、observer AS/RSS512MiB、総RSS9.5GiB、総CPU8時間、wall12時間、output512MiB、stage2GiB。監視間隔1秒・最大gap5秒、PSI/OOM増分・disk/inode・thread/CPU/lineage違反でSTOPし、owned processだけTERM/KILL/reapする。

起動直前にSOURCE/working bytes、provider230ファイル、runtime34,273ファイル、9入力のbytes/SHA、既存grant/ledger/carry/one-shot、他job、idle CPU、memory/PSI/OOM、disk/inodeを再確認する。実runtime・NPZ・内部絶対path・grant本文は公開しない。H5・V2・他worktreeへ書き込まない。

## 技術検証と次の段階

pinned PythonでV1関連57人工tests PASS。新10件はexact 2%の上下境界、変更grant・重複CPU・stale gate拒否、最初の不一致後の2軸目予約/compileゼロ、完全な検証前の将来費用map拒否、model推定の表示を確認した。primary agentによるlocal検証であり、CI・別担当の独立レビューではない。

[将来の再会計計画](../../artifacts/resource_applicability/track_a_v1_phase_b_readiness/2026-10-16/future_reaccounting_plan.json)はε=.05/.01/.005/.001/.0005、factor1/10、old S2／fine S2／fine S2+S4、既存PR確認union8/n32の点費用比較を定義する。
`validated_additional_cost_rows`は固定model SHAに対する登録全軸のPASSを要求し、実測とmodel推定を区別する。今回のrunnerから呼ぶ経路はない。全prefixで誤差2%未満という保証や保守的bias保証は得られない。5精度の最良費用とPR削減率の再評価は、検証結果を報告してSTOPした後に別途判断する。
