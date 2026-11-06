# Exact Fraction engine：静的hotspot分析

2026-11-07 JST / SERVER1 `GPUA100-gleap-01`。**静的SOURCE調査・保存記録の引用のみ。新規計算・profile・benchmarkなし。**
総括と固定資料一覧は [REPORT](REPORT.md)、候補比較は [OPTIONS](EXACT_ENGINE_OPTIMIZATION_OPTIONS.md)、次回の検証案は [PLAN](EXACT_ENGINE_VALIDATION_PLAN.md)。

## 1. 対象と証拠の読み方

対象は保存linear H4、原子間1.00 Å、STO-3G、legacy DF rank12、8 system qubits＋ancilla、全4電子number sector、`L_D=3,m0/q1,δ=0.02,r=1,K=2,coefficient_atol=0`、X/Y。δ掃引はなく、この固定点だけである。N1 actual finite-grid Qとselected full policy、N2保存保証state、理想gate・保存DFモデルを維持する。元Hamiltonianの厳密groundは理論基準であり、実装の保存binary64 stateと区別する。

以下を区別する。

- **保存事実**：固定commitの既存REPORT/log/certificateの記載。今回の再測定・独立再現ではない。
- **SOURCE事実**：静的に読める呼出し・ループ・cache構造。
- **静的推論**：演算数・bit長依存から導く負担の候補。wall時間の寄与率を意味しない。
- **未検証仮説**：最適化後の速度・memory・hit率。全てUNKNOWN。

コード参照はSCV準備固定版 [560c6092bfb5e71af8eedc4a4be5416922764f98](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/560c6092bfb5e71af8eedc4a4be5416922764f98)。以下 `E` は [a1_engine_connection.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_engine_connection.py)、`N` は [a1_n3_connection.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_connection.py)、`C` は [a1_n3_campaign.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_campaign.py)を指す。line番号はこの版のもの。

## 2. 保存timingが示す範囲

以下は全て上記H4条件の752-shot attempt02の [固定REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3c8e55a2728f86a1eda17a2e0105050a637e7c86/docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md)の表示値を引用する。新しい合算、比率、shot別ランキングは作成していない。

| 保存項目 | 既存記載 | 意味・限界 |
|---|---|---|
| 完了・全体時間 | 752/752、約70分 | 一campaignの経過時間。kernel単体benchmarkではない |
| engine worker区間の和 | 250,491.399秒 | `perf_counter`のwall区間和。CPU timeでもFraction時間でもない |
| compile区間の和 | 261.224秒 | `transpile_and_measure_cost`呼出し区間のwall和 |
| price/cache/build/待機、compileを除く和 | 5,347.421 worker秒 | lock、回路再生成、fingerprint、SQLite等を含み得る |
| setup / 親path生成 | 3.572秒 / 1.952秒 | 記録された範囲のみ。未分離overheadは0としない |
| worker / price cache | 127 / hit496・miss/compile256 | compile metricsのcache。確率やstateのcacheではない |
| peak RSS | 親509,755,392 bytes、単一worker最大471,846,912 bytes | 各processのhigh-water。RSS和から物理peakを推定しない |
| 人工検証 | 52件PASS | 当時のlocal人工検証。今回の実行件数は0 |

`C:146–178` のengine timerは、patch準備、facade作成、pinned pathの検査、`SavedEngine.probability`、threshold生成、保存済みmeasurement wordの適用、Measurement生成までを包む。`observed_probability`等のCALL journal追記/fsyncも含む。`BIT_SAVED`のfsync自体はtimer停止後、価格lock前である。`probability`内にも回路build・matrix取得/hash・exact contraction・Bornが含まれる。

価格timerは `C:182–187`、compile timerは `C:148–155`。親の会計は `C:238–239` でengine/compileを加え、priceからcompileを引いた区間を`cache`へ加える。したがって`cache`は純粋なcache lookup時間ではない。`TrialCost`のclassical欄0はplaceholderである。

保存 `DIAGNOSTIC_TIMING.jsonl`、`RESOURCE_START.json`、`RESOURCE_ADMISSION.jsonl`、`PILOT_ACCOUNTING.json` は同じ固定commitの `artifacts/o2_a1_new_target/2026-11-07/n3_pilot_752_run02/` にある。最初のtimer行は関数名＋secondsの形式、resourceにはworkers127とaffinity_CPUs128が保存されている。稼働中224-shotのjournalやprocessは参照していない。

**確認できるのはengineと呼ばれた広い区間の累計が大きいこと。Fraction乗算、gcd、build、hash、Born等の内訳、CPU利用率、bit長分布、GC/allocator寄与はUNKNOWN。**

## 3. 回路生成・演算子準備

### 一回のservice構築

`N:289–305` が固定bundle/SOURCE/runtimeを照合し、`E:197–240` で保存basis・deterministic blocks・components/specs・policy・raw stateを復元する。`N:261–273` は保存証明の有理数matrixをtableへ戻す。実path列挙、interval再認証はしない。このfactoryは今回呼んでいない。

### 各probability呼出し

`E:248–270,327–338` の静的経路は次のとおり。

1. order0/2、正質量label、X/Y、label数を検査する。保存label順は**products→rotation**。`_make_event`へは末尾rotationを先頭に移し、生成された作用順はproducts→rotationへ戻る。
2. RTE event、sequence request、support-run長1のselected basis planを作る。これは新しいpathの乱数生成ではなく、既に選ばれたpathの回路化である。
3. scalar phaseを加える。`df_partial_s2_repeated._append_aggregated_phases`は `-step_time*(constant+extracted_identity)` の既存binary64評価順を使い、controlled時にはancillaの`p`になる。
4. `self.blocks`順のdeterministic half-stepを作る。各blockはinverse basis→diagonal primitive→forward basis、時間はstep/2。中央RTEの後、逆block順で同じhalf-stepを作る。逆block順は各blockの逆時間発展を意味しない。
5. RTEは既存`df_rte_qiskit` builderを使う。productの制御ZはCZ列、negative product signやorder2の位相はeventの相対ancilla phaseへ入る。rotationはsupport長1/2のcontrolled RZ/RZZ。basis cancellationとselected planは既存規則のまま。
6. Hadamard wrapperはH→evolution→YだけSdg→H。gate列長cap、`global_phase==0`、classical bitなしを検査する。
7. 各gateでkey、再取得matrix hash、certified matrixの対応を検査してからexact stateへ適用する。

`_make_event`はcomponent確率の`fsum`やmetadata/fingerprintも生成する。固定分布からの選択はsampler側で完了しているので、このmetadata生成をsampling lawそのものと混同しない。

| 処理 | 現在の再利用 | 追加再利用の可能性 |
|---|---|---|
| N1/N2復元、registry、certificate table | service構築後再利用、fork継承 | 不変化して共有。profile検査は維持 |
| scalar・D half-step | 各shotでbuild、同じpathでは価格側も再build | 固定gate列template/検証済みdescriptorをprivateに再利用 |
| event/selected plan | 各shotでbuild | order＋ordered labels＋policy別の遅延cache。全path事前列挙は不要 |
| X/Y wrapper | 各shotでbuild | 共通evolution templateは可能。軸はkeyへ必須 |
| exact matrix | tableのFractionを既に再利用 | `matrix_record`の再構築と一時Fraction変換の重複を減らす余地 |
| price/compile | selected_circuitを再buildしてからcache照合 | 同一trial内のprivate descriptor共有。ただし測定前後のpublic境界は維持 |

## 4. Exact state evolutionの負担

`apply_exact` (`E:140–153`) はcomplex値を(real Fraction, imag Fraction)として持つ。`D=2^N`、gate support数k、`d=2^k`とすると、各gateでD個のbaseをscanし、有効なD/d blockごとにd×dの積和を実行する。complex multiplyはFraction乗算4回と加減算2回、complex addは加算2回である。零やidentityに対しても同じ経路を通る。

- dense contractionのcomplex積数は `D*d`、index生成はblockごとに `d*k` 程度、base scanと`out=list(v)`は各O(D)。これらは**ループから導く演算数**で、時間比ではない。
- maskは毎gate計算、`indices`は毎block構築。`qubits`の順序がlocal matrixのbit順を定めるため、集合だけでkeyを作れない。
- `out=list(v)`はFractionをdeep copyせず、参照listを複製する。その後tuple/Fraction結果が新規に作られる。全出力を上書きするので初期参照copy削減の余地があるが、全位置を一回書く保証が必要。
- 同一block内で先に出力へ上書きすると後のrowが元入力を失う。in-place化はd成分のsnapshotか別bufferが必要。block間のindexは互いに素である。
- Fractionは任意長整数を使い、加算・乗算・除算にgcd/交差約分が入る。CPython3.12.3の[実装](https://github.com/python/cpython/blob/v3.12.3/Lib/fractions.py)で確認した機構であり、現runのgcd時間の測定ではない。

### bit長：一般論とこのSOURCE固有の構造

通常の有理数では分母の積や最小公倍数が中間式を拡大し、大整数の乗算・gcd・allocation費がbit長とともに増す。しかしこのengineの保存stateとlocal matrixはfinite binary64からexact変換された値で、**全実部・虚部はdyadic rational（整数/2の冪）**である。`Fraction(float)`がfloatと同じ値を保存することは[Python公式仕様](https://docs.python.org/3.12/library/fractions.html)にも対応する。contractionは加減乗算だけなのでBorn除算前までdyadic性は閉じる。

従って「互いに素な多数の一般分母が無制御に積み重なる」だけを現engineの説明にしない。dyadicでもgate列に伴う分母指数・分子bit長の増加、大整数乗算、冪2相手のgcd、Fraction/tupleの反復生成は残る。**実際の最大・分位bit長、gate別成長、消去量、時間寄与は全て未測定。**

`E:112–120` はgateの各matrix要素をbinary64→Fractionへ変換するが、`probability`はそこで得た`pairs`を捨て、tableのcertified matrixを使う。これはSOURCE上の重複である。dyadic共通分母表現なら一般gcdを減らせる可能性があるが、state全体の最大指数に揃えると零/小さい成分まで大整数化するため、採用・速度は未定。

## 5. Born確率と53-bit測定

`norm_squared`と`born_probability` (`E:137,162–168`) は全stateの平方和Tとancilla=0部分の平方和Pを別に計算し、`p=P/T`をexact divisionする。ancilla=0成分の平方は二度計算される。共通の一回scanでPとancilla=1のMを求め、T=P+Mにする候補はある。各sector内の和の順序を保てても合成順が変わるため、新実装のexact同値性確認を要する。

初期stateをsqrt(norm)で正規化せず、Born比で相殺する。rounded gate列の後はnormが厳密1とは限らず、**Tを1として省略できない**。T≤0や範囲外確率はclipせずSTOPする。

`threshold_count`は `M=2^53, t=ceil(M*p)`。独立一様word `w∈{0,…,M−1}` に対し `bit=int(w>=t)` なので、`Pr(bit=0)=t/M`、`E[(-1)^bit]=2t/M−1` である。p=0/1と格子境界を含む厳密ceilを維持する。grid座標差の保存項は2/M。これは連続理想lawとactual finite-grid Qの同一性やPCG64の数学的独立性を証明しない。

Born比はdyadic同士の比でも一般の有理数になり得る。state kernelをdyadic化しても、P/Tとthresholdには正確な整数比・ceilが必要。同じwordの消費数とdomainを変えずに、同じthresholdを返すことがexact候補の要件である。

## 6. Gate matrix照合の分解

| `probability`内の工程 | 実装 | 費用候補とidentity上の役割 |
|---|---|---|
| key生成 | `gate_key`、gate名＋parameterのexact hex、ordinary closed control検査 | 少数paramsの変換。keyだけでは任意のgate定義変更まで証明しない |
| matrix再取得 | `matrix_record`→Qiskit `Operator(gate).data` | controlled RZZは固定定義の評価が必要。shape/finite検査、全entryのFraction化も実行 |
| hash照合 | complex128/C-order生bytesのSHA256と保存rowを比較 | signed zeroも含むraw identityを保護 |
| certified matrix取得 | `self.table[key]['matrix']` | 復元済Fraction matrix。取得自体と再構築を区別 |
| state contraction | `apply_exact` | 任意長有理数積和。内訳測定なし |

重複を減らすには、**最初の検証を残して不変のdescriptorへbindする**案がある。SOURCE/runtime/certificate/input identity、gate定義・class/control/angle、matrix hash、ordered supportとshapeを確認してから、privateなimmutable matrix参照へ結ぶ。任意のmutable gateに対して `(name,angle)` cacheを引くだけの変更は同じ保証にならない。gate objectのidだけも変異/id再利用へ弱い。

raw hashと有理数の値identityは異なる。保存[CORRECTNESS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdf00cee3c87d804845497afc89ac6a71f3b3f59/docs/research/o2_a1_engine_qualification_20261106/CORRECTNESS.json)には、Fractionでは失われるsigned-zeroを復元する3個の2×2 matrixのwitnessがある。再検証を省く設計ではこの来歴も保つ。tinyな非零要素を理想式の零へ置き換えない。

## 7. 静的hotspot候補とUNKNOWN

| 候補 | 根拠 | 現段階の判断 |
|---|---|---|
| dense Fraction contraction、bit長/gcd/object生成 | denseループとPython Fraction構造 | 最優先の技術検証候補。支配率UNKNOWN |
| gate再取得・捨てるFraction生成・hash | gateごとに実行するSOURCE | 重複削減候補。時間UNKNOWN |
| index生成・state参照copy | gate/blockごとのループ | 小さい変更で検証可能。時間UNKNOWN |
| 回路build・D列再構築・価格側二度目build | probabilityとprice双方のselected_circuit | immutable template候補。build時間UNKNOWN |
| Born平方和の重複 | P部分をTとPで二度処理 | correctness負担は小さいが全体効果UNKNOWN |
| group末尾の待ち・cache lock/SQLite | 保存REPORT＋group SOURCE | engine数学とは独立の実行効率候補 |

Fraction乗算が何%を占めるか、最適化後何倍になるか、127→128で何分減るかは示さない。科学的優位性、最終PR/RPE総コスト、224-shotの状態・成否を本調査から判断しない。
