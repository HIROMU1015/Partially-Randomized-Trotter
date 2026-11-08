# Track D O5：保存H4 q4一pathへのcut選択移送結果

2026-11-08。**限定検証は完了。固定規則は中央cut=2を選び、このpathでの追加depth削減は0だった。** 対応する旧raw PASS・4 compiled wrapper/PASS/価格を適格に再利用し、新build・raw correctness・compile・compiled postcheckは全て0。別cutの強制選択や別pathへの探索は行わなかった。

## 対象と原本回収

Linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2／元H／number-CDF8／元frame順L_D2／ordered partial S2/s1/r32/K0／h=δ=.1、q4・4 complete steps各32 event。元H厳密ground利用を仮定し準備費を除外。C6a calibration cohortの保存順先頭`q4_sample00`だけを対象とし、独立holdoutではない。新しいground solve/state/Born/測定shotは使わない。

前回[入力preflight STOP](../track_d_cut_h4_q4_20261108/REPORT.md)後、利用者が指定archiveを提供した。旧STOP・SOURCE・報告はそのまま保全し、固定条件のまま独立branch/outputで続行した。

- 人工基点：`a54d04059bfc61adef43aa2d40fb9c001b62378f`。今回の直接の保全基点：`848ca0deecc08fd09bbb9027fb372905edd4414d`。
- C6a実行SOURCE：`6b6667026ec8b670f610e41141d195883a9eb513`。元science SOURCE：`ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。
- archive：6,950,831 bytes、SHA256 `a631cd394073f7770d2723eb0cfa93fc6a8f703534b9cdce88b7c126381a007b`。
- event：SHA256 `919f2e784920581380cb4090c59fff34741d914e9d340d249bca554eba8cadb1`。20 member全てが公開hash索引とbyte一致し、4×32 event、保存seed/SOURCE/endpoint/input/lawを確認した。

原本はread-only扱い、展開物・QPY・参照配列は独立private `local/`内に保全し、GitHubへ公開していない。

## Compile前の一回decision

既存 `track_d_cut_selection.py` と、固定 `64072ba81bb70db1df1b34e94adb08e78926f24b` のC7-4 `event_features` を変更せず使った。負荷は `l_i=1+basis_runs_i+support_work_i`、中央±1の合法cutだけ、同点は中央への距離→小さいcut。価格やP0 DAGは入力にしていない。

step負荷は **(117,110,118,117)**。

| Cut | Prefix負荷 | Suffix負荷 | Score=max+1 | 選択 |
|---:|---:|---:|---:|---|
| 1 | 117 | 345 | 346 | — |
| 2 | 227 | 235 | 236 | **選択** |
| 3 | 345 | 117 | 346 | — |

[DECISION](../../../artifacts/track_d_cut_h4_q4/2026-11-08/run01/DECISION.json) は個別価格を読む前にSOURCE hash・event hashと結び付けて保存。選択は一回だけ。Cut2を選んだ後に他cutをbuild/compileして事後minを取ることはしていない。

## 全7資源（X/Y各。同価格であり独立2pathではない）

P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1.0/initially_zero=False/topology-free/既存MPFR160宣言global-phase保持。全Reference register/CSWAP/位相補正/actual inverse/cut損失を含むfull wrapper。

| 構成 | RZ count | CX count | RZ depth | CX depth | Total depth | Size | Logical qubits |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong symmetric-directional | 21020 | 8024 | 5013 | 3874 | 11671 | 42839 | 9 |
| Reference中央cut2 | 21239 | 7336 | 2571 | 1590 | 5686 | 42562 | 17 |
| Reference選択cut2 | 21239 | 7336 | 2571 | 1590 | 5686 | 42562 | 17 |
| **選択−中央** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| 選択−Strong | +219 | −688 | −2442 | −2284 | −5985 | −277 | +8 |

主評価ΔDはX/Yとも0。Strongとの差は既存Reference機構の効果であり、候補Bの追加利益ではない。RZ/width増と他指標減を区別する。全lawの期待価格・全RPE費用・旧Nの新しい資格は評価していない。

## Correctnessと再利用

旧SOURCEのbuilder/Strong/gadget/phase/compiler/checkerを保持し、同じrequest fingerprint、event順序・符号・normalization、endpoint/input/law、scalar/identity/χ、cut2、最適化policy、P0・精度、QPY hashとPASS/価格を照合した。既存serializerで保存eventを4stepともround-tripし、raw/compiled X/Yの8 QPYを読み、幅・bound状態・ancilla末尾→clbit0・YのSdg→Hを確認。保存compiled QPYの全7資源を既存counterで確認し、旧価格と一致した。

旧C6aのraw correctness 1件とcompiled wrapper 4件のPASSを再利用する。AB=V、actual A†、half vacuum ray、analytic χ、scalar/identity/signed event、全256 system入力・全reference出力、X/Y observable/coherence、leakage、paired/χ検査を含む。新しい数値simulationとして再実施したという意味ではない。旧値の最大成分差3.867403892829237e−12、compiled observable最大Frobenius1.9532248960493187e−11。既存abs1e−10、observable/signal2e−10、rtol0、phase fitなしを維持。局所binary64 evidenceであり、全law丸め誤差証明ではない。

追加したarchive/record binding・再利用guardの人工9件（5+4）はPASS。旧人工8件や旧科学checkerを不必要に再実行しなかった。新gadget・Hamiltonian builder・sampler・Gaussian compiler・state simulatorはない。技術接続は[実装・再利用記録](IMPLEMENTATION_AND_REUSE.md)を参照。

## 構造と今回分からないこと

Proxy prefix/suffixは227/235で中央が均衡する。保存A†/Bのraw total depthは1231/1145、どちらも8-qubit。これは未compile laneの指標で、P0 standalone価格ではない。Strongのstep融合は3境界、Referenceは内部2境界＋global cutの1損失で不変。二層CSWAPは既存構成のまま。保存P0のDAG/weighted wire finishは記録したが、CSWAP・closure・lane各項の因果的寄与は分離しておらずUNKNOWN。`shell+max(lane)`をfull depthの厳密式にしない。

**この固定規則は本pathで追加利益を生まなかった。cut1/3がP0でも不利であることや、中央が最適であることは検証していない。** 人工不均衡例の30.8%減を実H4へ一般化せず、今回の0も全q4/全lawの無効性へ一般化しない。

## 実call・保全・次の判断

Cut選択1、新sampling/seed/RNG0、新bundle/raw/compile/postcheck0、再利用raw PASS1/compiled wrapper PASS4。保存QPYの再利用照合8 load/compiled metric count4、構造診断5 loadであり、gate-state作用や新回路生成ではない。ground/DF/fit/N/全13round更新/science retryは0。

元mainと利用者提供archiveを含む既存dirty/untracked、旧SOURCE/raw/結果/grant/ledger/marker/STOP、C1凍結を保全。q64のprocess/資源割当/実行中SOURCE/input/private出力に介入せず、共有OS/venv/cache変更なし。

次の担当はユーザー/GPT。今回の1 pathだけでBの一般的な継続/棄却や候補Cへの移行を決める根拠は不足している。Bを続けるなら、次は保存q4 cohortのmetadataだけで不均衡・非中央選択がどれほど存在するか確認する有限scopeの採否を先に判断する案がある。今回その探索・他path選択・追加compileは開始しない。候補Cは別途実装方針/実行scopeの採択が必要で、自動移行しない。

専用branch `codex/track-d-cut-h4-q4-completion-20261108` へ公開し、remote SHA/tree/全公開bytesを照合後**mandatory STOP**。[SOURCE](../../../artifacts/track_d_cut_h4_q4/2026-11-08/run01/SOURCE.json)、[再利用SOURCE](../../../artifacts/track_d_cut_h4_q4/2026-11-08/run01/SOURCE_REUSE.json)、[全比較](../../../artifacts/track_d_cut_h4_q4/2026-11-08/run01/COMPARISONS.json)、[call数](../../../artifacts/track_d_cut_h4_q4/2026-11-08/run01/CALL_COUNTS.json)、[STOP](../../../artifacts/track_d_cut_h4_q4/2026-11-08/run01/STOP.json)を保全する。
