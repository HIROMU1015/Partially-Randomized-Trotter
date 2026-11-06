# A1 N3：採択済み一条件・一campaignの実行仕様

2026-11-06 JST、SERVER1 GPUA100-gleap-01。
ユーザーは固定a167effd4a1192d9806ea0bde8e0f7af71a1ef55のN3契約草案の推奨条件を今回だけの限定条件として採択し、4方式×X/Y・1campaignを明示許可した。
旧草案のDRAFT/未許可という履歴は変更せず、この新しい実行指定を適用する。承認原文・grant・追加承認発行を要件にしない。

## 固定対象・条件

保存linear H4/1.00 Å/STO-3G/legacy DF12、8system qubits、全4電子number sector、L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y。
N1=744e1a09661ebba6841fe5faf456f88c41c47780、N2=5e9bd08068738cdf1070bdc43bee8194b113ae33、engine=fdf00cee3c87d804845497afc89ac6a71f3b3f59。
N1 actual Q/selected policy、N2保証ground、保存gate証明のexact Fraction engineをそのまま再利用。新preparation/CDF/solve/spectral/engine qualificationは0。

ε_X=ε_Y=1/8、同時α=1/20。B0 pilot0、A1は5層×64 conditional train/axis、B1は各正層64 pilot/axis、calは各評価単位512、prod≤8192/method/axis。
固定3control、D_CV=2+Σ|λ|≤4。A1 cal/prodはunconditional Q。直接integer-count conditional取得、独立uniform53-bitモデル、PCG64/SeedSequence seed20261106、path64word/trial、別bit word1。
Confidence・Bernstein/Hoeffding・J8配分は固定N3 moduleを再利用。b_engineは保存ENGINE_ERROR_CERTIFIED.jsonの厳密な有理数値。1/32は採択しない。
価格は同selected full no-prep interrogationのRZ、Qiskit1.3.0/(rz,sx,x,cx)/opt2/seed17/backend等なし。cache hitもshot価格を毎回計上する。

## SOURCE・実行入口

[追加driver](../../../src/trottertracks/algorithm_codesign/a1_n3_campaign.py)、[固定入口](../../../scripts/tracks/algorithm_codesign/run_a1_n3_campaign.py)、[追加人工tests](../../../tests/tracks/algorithm_codesign/test_a1_n3_campaign.py)を追加した。
既存a1_n3_connection/a1_control_variates/a1_engine_connectionとbuilder/compilerのbytesは変更しない。
[PROFILE_LOCK](PROFILE_LOCK.json)は全SOURCE・依存/数値環境を記録、[PREFLIGHT](PREFLIGHT.json)は固定inputの静的bytes/identity照合と実science0を記録。[人工11件PASS](TESTS.json)、旧人工46件の証拠を再利用。

実入力前にこのSOURCEをcommit・公開してremote tree/blob/本文と照合する。そのcleanな専用worktreeで次を**一回だけ**起動する。

```bash
MPLCONFIGDIR=/tmp/a1-n3-science-cache-20261106/mpl \
NUMBA_CACHE_DIR=/tmp/a1-n3-science-cache-20261106/numba \
XDG_CACHE_HOME=/tmp/a1-n3-science-cache-20261106/xdg \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python -B \
scripts/tracks/algorithm_codesign/run_a1_n3_campaign.py
```

fresh出力：`artifacts/o2_a1_new_target/2026-11-06/n3_campaign_run01`。
`/tmp/a1-n3-science-invocation-20261106.json`のexclusive作成は二重起動防止だけで、承認証明ではない。出力衝突・marker衝突時に別namespaceへ逃がさない。retry/resume/精度・seed変更のCLIはない。

## 実行順序と分離

最初にA1 X、次にYの5層trainを固定quotaで取得し、それぞれ既存OLSを一回fit、rank/rangeを保存して判定する。
元n3.freezeと同じfit/rank/range判定をdriverで分けたのは、range失敗時にも実係数・rankを保存するため。新fit方式・再fitではない。
次にB1-O/B1-OLの各pilot。各方式axisで独立cal→J8/十分Nを凍結→prodへ進む。失敗の最初でcampaignを中断し、他方式を救済実行しない。

親が固定domainのpath/bit wordを事前に一回取得し、trial開始・path・word traceをjournalする。word traceは取得audit用で、CVのBatchには含めない。
workerは一回decodeした同じ保存bundleをforkで継承し、既存facade/exact probabilityをtrialごとに呼ぶ。親で固定したpathを再samplingせず渡すだけのPinnedSamplerを使う。測定wordの取得タイミングの移動は独立domain間の順序だけで、wordやlawを変更しない。
workerはbitを専用journalへflush/fsyncしてからprice/cache処理を開始する。戻り値はbit/metadata・価格・実行時間だけで、Born/state/真signal/E0/gapは含まない。

prohibition guardはN1/N2/qualificationのentry pointを拒否する。gateごとの既存exact kernelにはSTOP確認wrapperだけを置き、数値・引数・返り値を変更しない。
Qiskit compiler内部の局所分解と、禁止された全Hamiltonian state/spectral再計算を区別する。compilerの汎用線形代数を一律に無効化しない。

## 資源・費用・中断保存

worker数は実際に観測した空きCPUから選ぶ。CPU/RAM/workerに固定の恣意上限は設けない。worker自身のniceだけを+10にして、共有環境・他ユーザーのprocess/設定を変えない。
保存bundleのfork共有、既存metric SQLite cache、同path価格lockを使用する。確率・真signalの表を作成/cacheしてCVへ渡さない。状態準備は量子価格から除外するが、Hadamard full列の残りは全て含める。
古典時間はgeneration、fit、cal/配分・補正、engine simulation、compile、cache/build/lock待ち、共通setupを分離する。worker時間和とcampaign wall時間を混同しない。
parent/workerのRSS highwaterを記録する。worker highwaterの和は共有pageを重複計上する保守的なprocess-RSS指標で、実同時物理使用量とは呼ばない。

全trial開始/worker call/bit/価格/失敗をjournalsに保存。CALL_COUNTS/ACCOUNTINGは全worker末尾も照合し、未完了groupでも取得済bitと価格を落とさない。unpriced bitはUNKNOWNとして扱い、未取得価格0にしない。
SOURCE/input/rank/range/十分N/word/価格/非干渉不成立では条件を変更せずSTOP。共有STOP eventで新規gate作用・price compileを止め、取得済bitは保全する。既に走っているcallはログを残して回収し、新しいtrial/stageは投入しない。
成功でも失敗でもmandatory STOPし、その後は保存値の整合確認・lossless圧縮・報告・commit/push/公開照合だけを行う。
raw JSONLは必要ならlossless gzipで保全し、raw SHA256と復号一致を保存してから自分のrawコピーだけを除去する。旧データやcacheは削除しない。

期待価格・期待総費用の保証はUNKNOWN。全方式の保証済み精度が揃った場合だけA1対強いB1-OLの実現RZ差を記述し、それを期待改善の保証と呼ばない。
旧SOURCE/結果/STOP・元main/dirty/untracked・C1凍結を維持。N3以外、別round、追加campaign、科学レビューへ進まない。
