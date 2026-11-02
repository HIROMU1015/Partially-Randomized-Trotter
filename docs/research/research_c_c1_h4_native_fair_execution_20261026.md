# H4 CDF8／保存順XDF8：共通precision・native・full512限定科学実行

2026-10-26 JST。独立branch `codex/research-c-c1-h4-native-fair-execution-20261026`。

**固定SOURCEを変更せず一度実行し、305/305 logical unitsがCOMPLETE。correctness FAIL・inflight・retry・resumeは0。18局所因子、4 kernel、3 shell、8 actual adjoint、12 X/Y gateがPASSした。52/52 roundは数値pilotの登録位相条件内。単一h-stepのP0 RZ数はCDF8が12,272、XDF8が13,944で、同じsにおいてCDF側が約11.99%少なかった。共通CA・shots・条件付き総費用は今回未評価。公開照合後mandatory STOP。**

- [raw結果・SOURCE・305-call journal・QPY・STOP](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/run01/)
- [機械可読要約](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/execution_summary.json)／[実行前照合](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/execution_preflight.json)／[実行後照合・raw109件のhash](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/execution_postflight.json)
- [native metrics全7回路](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/native_metrics.csv)／[52 round](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/round_profiles.csv)／[同s比較2件](../../artifacts/research_c_c1_h4_native_fair/2026-10-26/native_comparisons.csv)
- [技術準備・再利用・数値伝播式・制約](research_c_c1_h4_native_fair_preparation_20261026.md)／[固定305-call契約](research_c_c1_h4_native_fair_contract_unapproved_20261026.json)

## 固定条件とSOURCE

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、sector36、system full256／ancilla-last controlled full512。通常number CDF rank8と、保存OpenFermion1.6.1 eigh由来のλ/G元順prefix8 XDFを比較した。元g補正κ・核scalar一回・spin/bit/frame順を維持。全frame ordered symmetric S2、s=1/4、δ=.1、m=0..12、PR splitなし。own-DF ground-state oracle、状態準備費用除外、共通CA=.00159 Ha／DF reserveCA/8=.00019875 Ha。

両armへ同じH6 Gaussian精度修正 `nature_jw_exact_zero_lapack_atan2_v1` とP0 phase修正 `qiskit_P0_mpfr160_declared_phase_v1` を適用。Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=false、backend・target・coupling・layout指定なし。既存の隣接完全同一basis共有・fsum角度結合だけを使い、一般相対Gaussian融合・literal追加比較は行わなかった。head／h-step kernel／readoutは別compile。

| 固定対象 | Identity |
|---|---|
| 科学SOURCE commit | `862a97d4213121749448ca716cde820a44b99772` |
| SOURCE tree | `e0ee8c5c7c9fb52eaeabeeac7b9fbde1d932c0d7` |
| SOURCE manifest SHA256 | `fc00dfef12de6e642ae3a5f52b7b495ad161fb2c2bc6c70f187af7802d74a464` |
| 固定契約 SHA256 | `4a99e76ea879226e077376dd69f12b529f9eb1a48e98dd048143892a59b0e578` |
| 元P-H4 SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| CDF元fit SOURCE | `da2edb465d5ae6432439fbd1df8e57512ee59c11` |
| 保存CDF export SHA256 | `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| CDF full CAR／PF SOURCE | `bec41ac7dc7e034bd664817acfdcfed858ddadd0`／`6ea5c6a9c10cc4b5411039a0d61a9f19acc4f34f` |
| XDF8 DF／PF SOURCE | `f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5`／`328f31007ee8069d1f7ba24f91029a3748b59109` |

ユーザーのチャット許可に基づく実行。歴史的な契約名・`UNAPPROVED_SCIENCE_CONTRACT` metadataは変更していない。承認証明書・承認原文保存・新grantは作成していない。

## Gateと全空間検証

| 検証 | 結果・範囲 |
|---|---|
| SOURCE29件・保存input33件・原packet・依存9版 | preflight PASS、実行後protected62件・契約・manifest・原packet不変 |
| 保存DF/PF/ground・κ/scalar/rank/frame来歴 | PASS。新DF norm・ground eigensolve・PF52セルの再計算なし |
| CDF/XDF sector target・XDF保存kernel2との照合 | PASS |
| Gaussian18 block request・共通precision公平性 | PASS、既存cache hitを維持 |
| 局所18 complete controlled因子 | PASS。最大成分差1.55436e-15、unitarity最大成分差3.21965e-15 |
| head/X/Y shell3、raw/P0 ancilla構造6 | PASS、full512 compiled/reference Frobenius最大3.07674e-15 |
| kernel4 raw/reference・P0/raw・P0/reference・unitarity | 全PASS。数値成分allowance1e-10、rtol0、global phase fitなし |
| actual adjoint8 | 全PASS。実inverse回路対dagger・reference・復帰を別々に確認 |
| own-ground m0 complex X/Y gate12 | 全PASS、固定allowance2e-10 |
| 4候補×13 round | 52/52 `POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL` |

全512入力columnと全512出力componentの`Operator`を評価した。有限probeや個々のGaussian gateのsector射影で代用していない。最大成分差をFrobeniusやoperator normと混同しない。下表のXDFのFrobeniusは1e-10より大きいが、登録gateは最大成分差1e-10でありPASS。Frobeniusは測定値のまま伝播に使った。

| 候補 | raw/ref最大成分差 | P0/raw最大成分差 | P0/ref Frobenius | m0 P0/ref X/Y誤差 |
|---|---:|---:|---:|---:|
| CDF s1 | 4.78285e-15 | 2.12565e-12 | 3.28971e-11 | 3.53511e-13 |
| CDF s4 | 4.44228e-15 | 1.86146e-12 | 2.69082e-11 | 1.40478e-12 |
| XDF s1 | 5.44349e-15 | 7.88586e-12 | 1.57664e-10 | 3.58966e-13 |
| XDF s4 | 5.66357e-15 | 7.93936e-12 | 1.58754e-10 | 1.44726e-12 |

compiled actual adjointのdaggerとの差は最大1.75304e-14、referenceとの差は最大7.94383e-12、復帰の最大成分差は3.73480e-13。全4 kernelのraw/P0 unitarityも登録allowance内だった。

局所保存名`CDF_Fi`／`XDF_Fi`はblock indexで、i0がκ＋核scalarのbase、i1..8が元frame F0..F7である。元frame順や科学的分割を変更したものではない。

## 13 roundの数値合成profile

直接測ったfull512 compiled/reference Frobeniusと、両方の数値行列およびshellのnorm guardを既存telescoping式へ入力した。n=s2^m、最大n=16384を維持。保存PF operator誤差をcomplex signal単位で一回だけ接続し、asin後の位相をπ/12と比較した。DF誤差（Ha）はここに足していない。

全候補の最大値はm12だった。詳細52行はCSVとraw候補JSONに保存した。

| 候補 | 合成complex signal上界pilot最大 | 保存PF誤差（m12） | 合成＋PFの位相上界pilot最大 [rad] | 判定 |
|---|---:|---:|---:|---|
| CDF s1 | 3.81121e-7 | .06913693071 | .06919250966 | 13/13 数値潜在適格 |
| CDF s4 | 1.24695e-6 | .004321338252 | .004322598667 | 13/13 数値潜在適格 |
| XDF s1 | 1.82657e-6 | .06842795073 | .06848329542 | 13/13 数値潜在適格 |
| XDF s4 | 7.35680e-6 | .004276939210 | .004284309114 | 13/13 数値潜在適格 |

共通の登録位相roomはπ/12=.2617993878 rad。m0のX/Yは実作用で確認したが、残り12 roundのX/Yを直接再測定したわけではなく、全operatorからの数値伝播profileである。

`B_NUMERICAL_OPERATOR_UPPER_BOUND`は**表現された数値行列に対するpilot**。CAR/expm reference形成、Operator復元、差・norm評価、旧PF基準の丸めenclosureは未資格。`rigorous_operator_bound`／`rigorous_signal_error_bound`／`rigorous_enclosure`はnull、`floating_point_rounding_enclosed=false`を保持した。未知誤差を0とは認定しない。

旧CDFの数値PF kernel自体は保存されていないため、同一hashの保存full CAR9因子・順序・時間・reference実装による同じ解析的PF targetとの対応を使う。旧新kernelのbitwise一致の証明ではない。XDFは保存sector kernel2との数値照合もPASSした。

## 同一条件のnative費用

**測定単位は1 h-stepのcontrolled S2 kernel。** s4のkernelはh=.1/4であり、s1のh=.1とは時間が違う。同じgate supportのため今回のcount/depthは両sで一致した。s4全体や13roundの反復費用が同じ、という意味ではない。literal/shared追加削減も今回測定していない。

| 候補・共通shared route | RZ count | RZ depth | CX count | CX depth | total depth | circuit size |
|---|---:|---:|---:|---:|---:|---:|
| CDF8 s1 | 12,272 | 5,873 | 7,388 | 6,440 | 12,806 | 23,543 |
| CDF8 s4 | 12,272 | 5,873 | 7,388 | 6,440 | 12,806 | 23,543 |
| XDF8 s1 | 13,944 | 6,005 | 7,756 | 6,468 | 13,113 | 26,740 |
| XDF8 s4 | 13,944 | 6,005 | 7,756 | 6,468 | 13,113 | 26,740 |

同sでCDF−XDFはRZ −1,672、RZ depth −132、CX −368、CX depth −28、total depth −307、size −3,197。RZ比CDF/XDF=.8800917957544464、約11.99%削減。headはRZ2/CX0/depth3、X readoutはRZ2/CX0/depth4、YはRZ1/CX0/depth3（readout depthは測定を含む）。

保存DF norm＋guardはCDF 7.6662159163e-5 Ha、XDF 9.5339014551e-5 Haで、両方CA/8内だが近似誤差は異なる。上記の単一kernel削減を同CAの総資源優位とは扱わない。旧CDF8／XDF10の約27%条件付き総RZ結果も、今回のCDF8／XDF8比較へ置換・流用していない。

## 科学call・環境・保全

| 分類 | 実績／上限 |
|---|---:|
| 保存input/export/evidence/行列/ground/kernel load | 27/27 |
| identity・models・CAR・XDF generators・sector/fairness | 16/16 |
| Gaussian18・local reference/build/Operator/gate各18 | 90/90 |
| shell build/structure・kernel reference/sector/build | 19/19 |
| P0 compile7・主要Operator14・segment7 | 28/28 |
| actual adjoint build/Operator/check各8 | 24/24 |
| signal reference/prepare/axis/gate | 40/40 |
| metrics7・round profile52・比較2 | 61/61 |
| **合計** | **305/305 COMPLETE** |

37分類すべてjournal attempt=COMPLETE=固定上限と照合した。correctness FAIL/inflight0。QPY40、operator packet18、raw109ファイル／24,680,598 bytesを保存。SOURCE29＋input33のprotected62件と元packet・契約・manifestが実行後も不変。旧H2/H4/H6のSOURCE・raw・grant・ledger・marker・STOPは変更していない。新しい実装SOURCEへの変更も0。

read-only既存環境のliteral Python `/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python`、Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0ほか固定9版を確認。実行直前の人工20 testsは20 PASS／16.64秒。前段の156 unique人工・回帰testsとmock分離条件は技術報告を参照し、今回の科学的PASSと区別する。

affinity32 CPU、実空きRAM約52.8 GiBから既存resource_planが局所18／候補4 workerを選択。process-local BLAS1で独立jobを協調し、P0/inverse生成のみ既存mutex。runner wall106.2676秒、process CPU450.2239秒、peak RSS916720 KiB。固定CPU/RAM/wall capなし。cache/tmpはtask内、共有venv・設定・cache・他ユーザーjobは変更していない。旧SVD回帰問題は今回解消したとは扱わず、固定経路はfull SVDを使っていない。

新fit/optimizer・DF/PF誤差再計算・literal比較・CA/shots/総費用・retry/resumeは0。fresh run01は消費済み。今回のCSV・要約は保存出力の転記・整形であり、新たなoperator/norm評価を呼んでいない。これは公開されたローカル実行証拠であり、外部再現やCI再現ではない。

## 次の最小作業とmandatory STOP

今回範囲の技術blockerはない。両armのfull512 native correctnessと、登録13roundの数値合成/PF位相条件、単一kernel費用を取得した。厳密rounding enclosure、最終CA、RPE shots、shots加重interrogation総RZ、状態準備込み資源は未資格・未評価のままである。

**ユーザー／研究C**がこの結果を確認し、同CA・RPE shots・条件付き総資源へ進むか判断する。進む場合は**Codex**が保存52 profile・新precisionのmetrics・各armの保存DF boundを使う最小接続と限定契約を準備し、新たな科学実行はその固定範囲への別明示許可後に行う。旧XDF10 shotsや旧precisionのCDF shotsは無条件に流用しない。詳細科学レビューにも別許可が必要。

**commit・push・remote SHA／成果物照合後mandatory STOP。追加の合成誤差計算、CA、shots、総費用へ自動進行しない。**
