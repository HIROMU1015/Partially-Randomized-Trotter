# C-S V0限定科学実行契約

2026-11-08 JST／SERVER1 GPUA100-gleap-01。
**条件はユーザーが採択済み。今回の作業は最終activation・人工検証・SOURCE公開まで。科学実行は未許可・未開始。**

## 正本と採択範囲

技術基点は `606c40d7809a5941a76c54d0bc2dd8e1a5bc1818`。その[技術binding][binding]、[coarse/residual/native政策][policy]、[旧実行草案][draft]、NATIVE_PROFILE、TOOL_IDENTITYを再利用する。本契約は旧文書の「未採択」を**下記条件に限り採択済み**へ更新する。旧SOURCE/文書/STOP自体は変更しない。科学実行許可は別である。

| 項目 | 採択済み仕様 |
|---|---|
| 保存モデル | linear H4／1.00 Å／STO-3G／legacy DF12／8 system qubits／全4電子sector参照／L_D3 |
| 原条件 | m0,q1,δ=.02,r1,K2,coefficient_atol=0、N1 actual finite-grid Q／selected full |
| 局所対象 | canonical first eligible D、first positive R/order0、各joint factor0、最大D1/R1。factor1・basis・scalar phaseはFIXED |
| coarse | exact π成分保持、radian成分を最寄り1/256radへ、tieは絶対値増加側。D/R・S0/S1/S2で同一政策 |
| residual | exact差、r_res1/K_res2、既存ordinary degree3、root256/proposal160、η=ρ=10^-12 |
| strict | pygridsynth2.0.0、up_to_phase=false、seed0、dps=interval_dps=100、各RZ precision10^-6。旧profileのepsilon/4 headroom・tool optionsを維持 |
| cap | b_finite≤10^-12、各native≤10^-6、b_local≤3×10^-6 |
| 有限scope | 1campaign、science retry0、追加campaign0。8unique positive keys以下、負角は保存positive列のactual adjoint |

固定入力はN1 `744e1a09661ebba6841fe5faf456f88c41c47780`、N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`、既存認証 `fdf00cee3c87d804845497afc89ac6a71f3b3f59`。13入力のpath/bytes/hashは606c40dのSOURCE_AND_CHECKSをそのまま正本とする。N2 state vectorは読まずsector/state identityを参照するだけ。N1/N2/engine上界の再計算・変更後C-Sへの無条件継承なし。

## 実行入口と一回性

新しい実行SOURCEは**本契約・activationを公開したcommit全tree**。最終チャットのSHAを使う。自己参照のためSOURCE.jsonを再編集せず、CLIの`--source`をHEAD・remote source branch・固定ファイルhashへ照合する。

独立worktreeで、次の操作を**別途明示科学実行許可の後だけ**一回行う。

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/cs-v0-numba-cache PYTHONPATH=/tmp/cs-v0-provider-20261108:src /home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python scripts/tracks/algorithm_codesign/run_cs_v0.py --source <公開された最終科学SOURCEの40桁SHA> --execute-one-campaign
```

既定CLI（実行optionなし）は`SCIENCE_NOT_STARTED`を返し、入力・outputに触れない。今回、この実行optionで実入力を起動していない。フラグは操作選択であり、ユーザーの許可を証明するgrant/packetではない。

outputは `artifacts/cs_v0_science/2026-11-08/campaign_run01/` 固定、cacheはその内部。exist_ok=Falseによるexclusive予約を一回のattemptとして使う。旧cache・rawを読まず、既存outputに上書き・resumeしない。outputを消して再実行することも契約外。SOURCE不一致等のStage A失敗も保全して停止し、自動retryしない。

## Stage A：新SOURCEと既存preflightの接続

1. 新activation source/契約hash、HEAD、clean tracked tree、未登録実行code不在、remote `codex/cs-v0-activation-20261108` のSHAを照合する。
2. **既存cs_v0_preflight.inspectをそのまま呼ぶ**。13入力bytes、旧SOURCE hash、runtime executable・package lock・pygridsynth/mpmath treeを照合する。今回は旧preflightを実データで繰り返していない。新接続は人工root/mockで確認した。
3. output/cacheは専用・未使用。CPU affinity、load、実idle tick、RAM、pressureを保存する。旧CPU tick parserだけ再利用し、N3の待機・worker政策・旧grantは呼ばない。
4. 段階依存・最初の失敗で即STOP・mpmath context保護のため逐次call。worker/CPU/RAMの固定上限・quota・rlimitは追加しない。実観測に1 call分のspare core/RAMがなければ新input取得前にSTOPし、共有設定や他jobへ介入しない。観測window .25秒は資源上限ではない。

このpreflightはCPU/RAMの継続的な占有権や外側sessionの持続性を保証しない。OS/hostの強制消失時は保存prefixを保全し、RETURN/STOPのない末尾はUNKNOWN。出力namespaceの再使用・科学retryはしない。新manager・共有service変更は行わない。

## Stage B/C：既存接口を固定順で呼び出す

`cs_v0_execution.run`が既存Callsとatomic_jsonへ接続する。処理はD→Rの順で固定し、価格や誤差を使うslot差替えをしない。

- `binding`一回で既存bind_slotsと元R event-angle resolverを呼ぶ。選択site、exact target/coarse/residual、joint generator、FIXED partner、selected basis、phase/basis境界を保存する。
- 各specのtarget/coarse/residualを**Pauli ±1 eigenspaceの2×2局所表現**へ記録する。既存strict target式を標準mpmath.ivで表現し、元operator_target callback APIでfinite eventを合成する。新しいH4 operator kernel、Spectrum solve、state engineは作らない。float診断経路や旧G9 providerは使わない。
- 既存residual_tableの2event、actual proposal、exact係数/weight、B_d、range、moment、truncation/係数幅、finite meanを保存する。finite meanはexact residualと一般に異なる。旧N1 Q/CDFの再計算・path samplingなし。
- 全requestを取得前に固定し、unique positive keyだけ一回ずつ既存acquire→checkedへ送る。negativeは実adjoint、exact fast pathはbackend0。backendの返却sequenceは**strict guardより先にraw保存**し、不合格時も失わない。fast-path列も先行保存する。
- `b_local ≤ b_finite+B_d b_coarse+Σ c_n δ_native,n` を既存local_boundsで合成してcapを検査する。不足項を0補完しない。各native gateの数値生成・point roundingは既存strict exact-expression guardに含める。
- 最大2組のlocal_native／native_accountingを呼び、S0通常strict、S1両選択spec、S2 Dだけの局所構成を保存する。T/T†、CX、1Q Clifford、W scalar、serial depthとactual residual proposalによる局所期待countを記録する。局所加法モデルであり、full compile/並列最適化depth/母平均full価格ではない。

zero residualは既存exact I policy、event0・B/range/moment1。重複keyやzero/fast pathは取得数を減らすだけ。8keyは2target+2coarse+2order×2residualのこのscopeに限る。R側代表order0の選択は原Qの変更ではない。

## 有限callと保存

| 群 | 上限 |
|---|---:|
| input binding | 1 |
| selected D/R factor0 | 2spec |
| local operator comparison | 2組 |
| residual finite mean/error | 2組／計4event |
| positive acquisition/strict qualification | 8unique keys、各一attempt |
| local native price | 2組 |
| sampling、state/Born、quantum shot、full build/compile、N/G | 0 |
| science retry／追加campaign | 0 |

CALL_STARTはfunctionより先にexclusive/fsync保存。RETURNと例外STOPを記録する。STARTがあるがRETURNがないcallは未完了／例外であり、無実行・成功へ補完しない。backend startedは開始予約数、returnedは返却確認数。途中で消失した実実行の成否はUNKNOWN。既存Callsのattempt capを使い、新しいgrant/managerを作らない。

完了・未完了のschema：

| 保存物 | 内容 |
|---|---|
| SOURCE.json / PREFLIGHT.json / NATIVE_PROFILE.json | 実SOURCE、採択cap、13入力参照/hash、remote/tree、runtime、resource |
| BINDING.json / LOCAL_PLANS.json | exact site・role・joint/FIXED、3armの共通coarse表 |
| spec*_operators.json / spec*_finite.json | 局所interval表現、finite law/mean/誤差・range/moment |
| REQUESTS.json / cache/*.raw.json / cache/*.json | 取得前key集合、先行raw、qualified strict行・tool・取得wall |
| spec*_errors.json / spec*_price.json | 誤差内訳・cap、局所枝/期待count、S0/S1/S2 |
| journal/*.json / CALL_COUNTS.json | durable START/RETURN/STOP、消費済attempt、backend予約/返却、既知qualified数 |
| RESULT.json / CLASSICAL_COST.json / STOP.json | 成否・未取得UNKNOWN、古典wall/CPU/peak memory/cache、mandatory STOP |

成功はLOCAL_V0_COMPLETE。不適合/例外はSTOPを先行保存し、LOCAL_V0_INCOMPLETEと取得済prefix・既知費用・UNKNOWNを残す。SIGTERMは自分の実行process内で例外化し、元handlerへ戻す。SIGKILL/host消失、永続storage障害はfinalizer未保証で、partialや未返却journalを保全してUNKNOWNとする。旧SOURCEや他jobに対するsignalはない。

## STOPと結論の範囲

SOURCE/input/profile、slot/role/generator/joint factor、exact angle、coherent phase、対応gate/key、strict/finite/local cap、law/weight/normalization、unique/call上限、保存、非干渉に不整合があればSTOP。別slot、coarse/r/K/seed/precision変更、追加合成で救済しない。

Stage Dで結果を専用branchへ保存・公開・SHA/body/blob/tree照合しmandatory STOP。実行関数自身にGitHub公開や次campaign開始を埋め込まない。科学実行担当Codexが取得済ファイルを公開し、UNKNOWNを維持する。

V0は選択最大2局所specの資格確認。旧engine上界をC-S列へ継承せず、局所T削減をfull wrapper、shot、N/G、PR全体の優位性へ格上げしない。全4電子groundは参照のみ。V1、全18round、C-N/C-P、SCV/C1/旧STOP解除、重要科学レビューへ進まない。

[binding]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/606c40d7809a5941a76c54d0bc2dd8e1a5bc1818/docs/research/cs_v0_technical_preparation_20261108/V0_TECHNICAL_BINDING.md
[policy]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/606c40d7809a5941a76c54d0bc2dd8e1a5bc1818/docs/research/cs_v0_technical_preparation_20261108/V0_COARSE_RESIDUAL_NATIVE_POLICY.md
[draft]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/606c40d7809a5941a76c54d0bc2dd8e1a5bc1818/docs/research/cs_v0_technical_preparation_20261108/V0_SCIENCE_EXECUTION_DRAFT.md
