# 固定科学実行計画：Controlled Hadamard全18round有限certificate

2026-11-05 JST。技術SOURCEの公開段階。**未実行・ユーザーの別途明示承認待ち。**
前回57b4792728c40acb337d69d0120dd0417b257bbdの
[計画](../rpe_categorical_circuit_connection_20261104/SCIENTIFIC_VALIDATION_PLAN.md)を
今回の実装へ固定した。新しい誤差予算・policy・precisionは導入しない。

## SOURCE・固定入力

科学SOURCEは本計画とrun_connection.py/integration_audit.py/connection_pipeline.py、
test、SOURCE_INPUTS.jsonを含む公開commit（最終報告の完全SHA）とする。
科学起動時のHEADに新SOURCEがcommit済みであることを確認する。
[bindings](SOURCE_INPUTS.json)は親57b479272のFIXED_BINDINGS.jsonをcommit/blob/SHA256で固定し、
そこからH/state/export/保存tuple/law/結果とproduction SOURCEへ接続する。
追加依存SRC・Qiskit関連6ファイルも同bindingsへ固定した。

- H4 linear・隣接1.00 Å・STO-3G・保存legacy DF12。
- Partial L_D=3・単一δ=.02・8 system qubit・4電子sector70・保存physical stateのnormalized ray。
- extract_identity_phase・coefficient_atol=1e-12、m0..17・q=2^m、保存r1/2/4/8/16/32・K2。
- 保存N/α/β/ε、218 component、full11 basis、sign/support policyとtraining fingerprint。
- signal側τ、保存θ・uniform53 order/component CDF/grid mass。model側τへ置換しない。
- C SOURCE7279f700a、C run02結果82b2df6af、categorical SOURCEf4905fff5、
  全18結果d2e7a7db6、m17結果34840ce9b。m17包含は旧96-bit結果をそのまま再利用する。

必要なimmutable Git objectと保存入力のbytesがこのcheckoutに存在することが実行前条件。
missing objectを科学起動中に自動fetchしたり、近いpayloadへ代替したりしない。
科学入口を呼ばず、以下でbyte/schema/literal照合だけを実施できる。

~~~sh
/tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python \
  docs/research/rpe_controlled_connection_runner_20261104/run_connection.py
~~~

このdefault modeではH load・primitive・matrix・上界・strict比較を実行しない。

## 追加科学call：1 passの固定範囲

| 操作 | 範囲 |
|---|---|
| 保存H load / DF preparation / support登録 | 各1。legacy DF再因子化は0 |
| descriptor取得 | full11＋selected最大218。source/hash/wire/class/parameterを照合 |
| half primitive | 4 block各1。保存exportへhex一致 |
| full primitive | block0だけ1 |
| scalar / boundary | 18round各1、生成済みbinary64のFraction差 |
| local controlled RZ/RZZ audit | 最大172 constructor、同parameterのcacheあり |
| Gaussian包含 | full11＋selected最大218のうちunique hashごと各1、8×8 Acb |
| component差 | 最大218、全support coverage必須 |
| transfer / strict比較 | 18round・36軸 |
| precision / pass | 96 bit / 1。自動precision追加・retry/resumeなし |

callsは開始/試行数を記録する。失敗したconstructor等についてprefinite検査で停止した場合は
保守的な試行数となる。workerの異常終了による開始数不明は明示する。
full/selected hashの重複をcacheして実数は上表以下になる。

既存DF preparation/support登録はSOURCE内部で8×8のorbital diagonalizationとGaussian basis合成を行う。
これは上表の既存preparation/登録callに含まれる。
新しいH/state生成、many-body固有値/固有状態solve、DのE0計算は含まない。
今回の技術準備中はこれらのorbital処理も含め、H4数値処理を全く実行していない。

state load・norm²再計算、SCF、新trajectory、sampling/quantum shot、
law/CDF/atan/mean再計算、70×70/256×256 evolution、
長q回路build/transpile、費用・noise・状態準備・Dのα/decoder/E0は全て0。
既存preparation/登録に必要な既存local fingerprint処理を除き、
新export/lowering auditでH4 local Operator/to_matrixを追加評価しない。

## 構成・判定

production SOURCEと同じ演算順序でraw phase a,bとoptimized gを生成し、
d_s=|Q(g)−q(Q(a)+Q(b))|。
4 halfとblock0 fullのprimitiveからd_bを構成する。
Δ_boundary=min(2,d_s+(q−1)d_b)。

runtime descriptorからRZとJW隣接XXPlusYYの一粒子行列をAcb96 bitで包含し、
full/selected両radiusを含むcolumn差に対して
Δ_S=min(2,4Σ_(i∈S)||g_i−g'_i||)。
未使用列の完全一致やfloat residual=0を証明として使わない。

RZ/RZZ controlのclass/definition、closed control、wire/global phase、
exact doubling/halving、全basis class/inverseのSOURCE条件を確認する。
標準logical CCX/CX/CZ/Phaseとfixed templateの範囲で全前提が揃えばΔ_lowering=0。
native gateへの再帰分解・compiled・Statevector算術へは拡張しない。

Δ_extra=min(2,Δ_boundary+3qr max Δ_S+Δ_lowering)。
各軸の保存値だけを使い、exact rationalで
U_cat_upper＋Δ_extra＋shot_half_width.upper < ε
を比較する。等号はPASSにしない。保存I_cat±Δ_extraも保存する。
π64項は既存I_cat/U_catに含まれるため再加算しない。
N/α/β/εとshot半幅・旧包含は変更・再計算しない。

全36軸strictと全前提が成立したときだけ
CONDITIONAL_IDEAL_CONTROLLED_CATEGORICAL_LOCAL_CONDITIONS_CONFIRMED。
不足はUNKNOWN/UNDETERMINEDであり0へ補完しない。
strict不成立は証明が足りない結果であって、真の回路誤差超過の証明とはしない。

成功時の範囲は指定ideal gate・normalized ray・独立uniform53 lawの局所十分条件。
固定PCG64 streamのIID性や実測定成功確率をこの結果だけで認定しない。
全体RPE_tolerance_certified=false、scientific_status=UNDETERMINED、
U_virtual_IID_upper=null、U_exec_upper=nullを成功時も維持する。

## 実行・資源・保全・STOP

**以下は別途承認後にCodexが一度だけ使用する実行command。今回は起動しない。**

~~~sh
/tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python \
  docs/research/rpe_controlled_connection_runner_20261104/run_connection.py \
  --run --output artifacts/rpe_controlled_connection_certificate/2026-11-05/run01
~~~

成果物はfresh run01のRESULT.json / DESCRIPTORS.json / basis_enclosures.jsonl。
前二者はstage checkpoint、後者は各unique basisのcenter/radius/normまたはerror原本。
既存出力なら起動せず、旧C run01/run02、全18結果、grant/ledger/marker/STOPを変更しない。

CPU affinity/idle、MemAvailable、親RSSから現在のworker数を選び、
独立basisを並列化する。各workerのFLINT/BLASはtiny matrixに適した1 thread。
任意の低いCPU/RAM/worker数やwall-time capは設けず、共有環境の変更・他job干渉はしない。
資源不足なら自分の起動を停止する。
人工serial/parallel一致は確認済みだが、H4性能は未測定。

SOURCE/input/tuple/hash/descriptor/primitive不一致、非有限値、未被覆support、
未知gate/nonadjacent XXPlusYY、lowering不足、worker/包含errorは途中UNKNOWN/STOP。
完了済み部分と停止理由を保存し、失敗後の別candidate・sampling・precision追加へ進まない。
strict不成立だけなら事前固定した36比較まで完了し、UNDETERMINEDでSTOP。
成功/失敗を問わず公開照合後mandatory STOP。自動retry/resumeはない。
新grantや承認証明書、承認text保存・認証機構は不要。

次の担当は、ユーザーが公開SHAと本scopeへの科学実行を明示許可し、
Codexがその許可後に固定1 passだけを実行・報告する。
今回の技術成果公開後はmandatory STOPし、承認待ちの間に科学入口を起動しない。
