# C1：H6 native精度修正後の限定科学実行

2026-10-25 JST。Branch `codex/research-c-c1-h6-native-stable-phase-20261025`。

**固定SOURCE `3d7b56b44832a2a90a8d3e500bf55e856f36da76` のfresh実行は298/298 logical unitsが完了し、correctness FAILは0。局所30因子、kernel8経路、ancilla shell3件、actual adjoint8件、X/Y8件が固定allowanceをPASSした。P0 metrics11件・QPY52件・合成誤差profile104行・費用比較8件を保存した。これは有限probeの数値pilotで、global合成誤差上界・最終CA・shots・総費用は未資格／未評価のまま。公開照合後mandatory STOP。**

**今回の1 h-step kernelでは、CDF14のRZ数はXDF14よりs1で1,010（約1.98%）、s4で1,002（約1.96%）多かった。literal/shared間の費用差は全4候補で0。H4で観測した費用削減をH6へ転用する根拠は得られていない。**

[要約](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/execution_summary.json)／[8経路の全値・metrics](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/candidate_summary.csv)／[kernel/adjoint gate](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/kernel_and_adjoint_gates.csv)／[局所](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/local_summary.csv)／[Gaussian](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/gaussian_summary.csv)／[104 profile](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/round_synthesis.csv)／[費用比較](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/cost_comparison.csv)／[call](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/calls_summary.csv)／[原本・QPY・STOP](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/run01/)／[postflight](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/postflight.json)。

## 条件・来歴・修正

ユーザーの修正・再実行許可と、その後の継続指示に基づき、旧SOURCE／raw結果／STOPを保全して技術修正と同一範囲のfresh評価を行った。承認原文保存・証明書・新grantは作成していない。独立worktree `/tmp/prt-c1-h6-native-stable-phase-20261025`、fresh output `artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/run01`。

直鎖H6・1.00 Å・STO-3G、Nα=Nβ3／sector400、修正codecによる保存最終CDF14／保存selected XDF14、通常number・元g補正κ・scalar・元frame順。全frame ordered symmetric S2、s1/s4、literal/shared8経路、δ0.1・m0..12、own-DF ground-state oracle・状態準備費用除外。PR split非適用、共通CA0.00159 Ha、DF reserve CA/8=0.00019875 Ha。測定対象はh=1/(10s)のcontrolled kernelで、長round回路はbuildしていない。

P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1.0／qubits_initially_zero=falseを維持する。既存NativeBuilderのexact basis共有・境界・角度結合規則を双方へ同等に適用し、一般相対Gaussian融合等の追加最適化は導入していない。

| 固定資料 | identity |
|---|---|
| 科学SOURCE commit | `3d7b56b44832a2a90a8d3e500bf55e856f36da76` |
| SOURCE tree | `01a9584a4e7c07a3d533b40d740f4cfdc218bca8` |
| [契約](research_c_c1_h6_native_stable_phase_contract_20261025.json) SHA256 | `7992823a0c01e022eeeee2cd0ac422c4ad0ec113911c8f5975f0e7cecb82edb5` |
| [SOURCE manifest](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/technical/SOURCE.json) SHA256 | `d34a3816af7af93093c971a4cc915707d41cc0d85d8fd4b07bb6fc85c12b3f37` |
| 元input SHA256 | `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447` |
| 保存CDF14 export SHA256 | `cd4de687f99d32bd6f644ac33100fb52d4f7543e710e6c273723a0908b87dc45` |
| 保存XDF14 export SHA256 | `559dc43bf923a421ccc62d05c4cefa0bdf39a48ba68824abf3c8f770c04642b1` |

元input/XDF SOURCEは33c380d8…、CDF fit・保存DF誤差SOURCEは33432e786…、PF SOURCEは87ea924ae…で、今回の評価SOURCEとは独立に保持した。保存DF norm+guardはCDF1.0803058513925193e-4／XDF1.0361230763094318e-4 Haで異なり、双方CA/8内の保存判定を再利用した。前段一般K・ground/alias・52 PF roundも保存証拠を再利用し、DF誤差・PFは再計算していない。

修正は[Gaussian精度・inverse/P0 mutex](research_c_c1_h6_native_precision_repair_20261025.md)と[declared phase集計](research_c_c1_h6_native_stable_phase_repair_20261025.md)。Natureの既存JW消去順序・RZ/XXPlusYYを保ったH6専用処理で、近似zero省略／arccos小角精度をexact-zero・既存LAPACK・atan2へ置換した。P0は同じpass/profileを使い、declared global_phaseの加算だけ既存SymEngine MPFR160で安定化した。入力に加えた正確なunit offsetは最後に相殺する。出力state/operator/signalを参照するphase fitは行わず、科学allowanceも変更していない。

人工の長回路ではP0 gate本体・qubit・angle列の完全一致を確認した。実H6の過去P0 bodyとのbitwise比較を新たに行ったとは主張しない。小回転を保持する修正で従来省略されたgateが残り得るため、正しさをPASSした今回のnative費用だけを比較する。旧H2/H4の共有builder/compilerとインストール済みpackage・venvには変更を加えていない。

直前関連97 tests PASS、先行のGaussian/既存回帰185 tests PASS、独立state-action診断の人工2 tests PASS。環境はPython3.11.1／NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Aer0.15.1／SymEngine0.13.0等を維持した。SOURCE46 files・保存入力25 hash・依存lock/versionのpreflightはPASS。

## correctnessの到達範囲

| 検証 | PASS | 最大残差・意味 |
|---|---:|---|
| Gaussian単粒子transfer | 30 | 7.78267736583e-16、basis-only真空位相を除く小行列検査 |
| complete局所因子／独立CAR | 30 | max-component 2.01138176123e-16、norm drift 1.11022302463e-15 |
| raw kernel／reference | 8 | max-component 7.94798730346e-15 |
| compiled kernel／raw | 8 | max-component 5.51609549676e-11 |
| compiled kernel／reference | 8 | max-component 5.51617063962e-11 |
| actual raw adjoint | 8 | max-component 7.11757093485e-15 |
| actual compiled adjoint | 8 | max-component 9.8898730789e-13 |
| X/Y raw／reference | 8 | complex error 2.65231792623e-14 |
| X/Y compiled／raw | 8 | complex error 5.76792693468e-12 |
| X/Y compiled／reference | 8 | complex error 5.77524648506e-12 |
| head／X／Y ancilla shell | 3 | ancilla-only、全2×2 raw/reference・compile・unitarity |

component/norm allowance1e-10、complex signal2e-10／rtol0・phase fit禁止を維持した。局所は各因子のcomplete Gaussian逆変換→controlled中央→Gaussianを、control+の固定complex-sector probeでfull8192へ作用させている。Gaussian gate個別のsector射影で代用していない。

kernelはcontrol0 full-Fock random、control1 sector first basis／complex random／own-ground、control+ own-groundの固定5 probeを使う。sector外leakageも含む全8192振幅とnorm driftを比較した。control0 identity・相対位相・順序・Gaussian向きに対する有限probe検証であり、全operatorの厳密同値性・完全なunitarity certificateではない。

actual adjointは実際に生成されたraw／compiled回路のinverseを、それぞれのforward control+ground出力へ作用させた。X/Yは実head→kernelをs回→実readoutのm0・合計時間0.1で独立reference complex signalと比較した。全13 roundの実回路信号や量子samplingを実施したとは扱わない。shellはsystem gateがないこととancilla全2×2を数値確認したが、丸めenclosureはない。

## native費用（1 h-step kernel）

| arm | s | route | RZ数 | RZ depth | CX数 | CX depth | total depth | size |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CDF14 | 1 | literal | 52,056 | 23,133 | 30,670 | 26,174 | 50,764 | 99,444 |
| CDF14 | 1 | shared | 52,056 | 23,133 | 30,670 | 26,174 | 50,764 | 99,444 |
| CDF14 | 4 | literal | 52,048 | 23,125 | 30,666 | 26,170 | 50,752 | 99,428 |
| CDF14 | 4 | shared | 52,048 | 23,125 | 30,666 | 26,170 | 50,752 | 99,428 |
| XDF14 | 1 | literal | 51,046 | 22,697 | 30,134 | 25,766 | 49,876 | 96,890 |
| XDF14 | 1 | shared | 51,046 | 22,697 | 30,134 | 25,766 | 49,876 | 96,890 |
| XDF14 | 4 | literal | 51,046 | 22,697 | 30,134 | 25,766 | 49,876 | 96,890 |
| XDF14 | 4 | shared | 51,046 | 22,697 | 30,134 | 25,766 | 49,876 | 96,890 |

以下のleft/rightは比較名の順序（CDF/XDF、literal/shared）に対応する。ratioは保存されたRZ比で、全round・shots加重費用ではない。

| 比較 | left RZ | right RZ | left/right | left-right |
|---|---:|---:|---:|---:|
| CDF_XDF_s1_literal | 52,056 | 51,046 | 1.019786075 | +1,010 |
| CDF_XDF_s1_shared | 52,056 | 51,046 | 1.019786075 | +1,010 |
| CDF_XDF_s4_literal | 52,048 | 51,046 | 1.019629354 | +1,002 |
| CDF_XDF_s4_shared | 52,048 | 51,046 | 1.019629354 | +1,002 |
| CDF_s1_literal_shared | 52,056 | 52,056 | 1.000000000 | +0 |
| CDF_s4_literal_shared | 52,048 | 52,048 | 1.000000000 | +0 |
| XDF_s1_literal_shared | 51,046 | 51,046 | 1.000000000 | +0 |
| XDF_s4_literal_shared | 51,046 | 51,046 | 1.000000000 | +0 |

同じs・routeの双方へ同じcompiler・合法共有規則を適用した。literal/sharedの差は観測値だけを報告し、未実装の一般相対Gaussian融合や未観測の追加削減は仮定しない。CDF14／XDF14という同rankでもDF近似誤差は異なる。kernel費用の大小を同CA達成・shots加重総費用・無条件の資源優位へ昇格しない。

## 合成誤差の資格

8経路×13 roundの104行を保存し、occurrences=s*2^mを記録した。すべて`UNRESOLVED_FINITE_PROBES`で、kernel global operator norm bound／complex signal error pilot／rigorous signal boundはnull。有限probe最大値をcoherent反復数で乗じて保証付き上界にする処理は行っていない。

局所・kernel・shell・actual adjoint・m0 X/Yの数値pilot PASSは得たが、全入力・全13 roundの合成誤差の厳密enclosureは取得していない。未知誤差を0へ置換せず、CA_achieved／shots／total_costもnullのまま。前段PF52 PASSとの接続で最終CAを認定していない。

## call・資源・保全

科学callは298/298 COMPLETE、FAIL0／inflight0、全call categoryが契約の上限どおり。新fit・optimizer・rank探索・DF/PF再計算・shots・最終CA／総費用算定0。各attempt内retry/resume0。

| 実行 | SOURCE | units | 結果 |
|---|---|---:|---|
| 元native | bf431765… | 162 | raw/reference FAIL、旧証拠保持 |
| Gaussian精度修正後 | 5309006… | 162 | raw/reference PASS、compiled/raw FAIL |
| 保存1経路の独立診断 | 38c8da551… | 5 | simulator一致、compiler側の差を切り分け |
| 今回stable phase | 3d7b56b44832… | 298 | 登録correctness全PASS |
| 累積 | | **627** | native試行622＋診断5 |

fresh修正再実行は2回であり、旧残枠の流用・旧runのresume・失敗履歴の削除は行っていない。最新のユーザー許可に基づき各修正SOURCE・契約・fresh outputを実行前に固定した。

実行前はCPU affinity32、idle約99.5%、MemAvailable約52.6 GiB、CPU/memory pressure0を確認。既存planによる8 worker・各Aer4 threads／BLAS1、P0とinverse回路生成だけtask-local mutexを使用した。恣意的CPU/RAM/worker/wall上限・共有環境変更・他jobへの操作はない。

wall 628.764680 s、process CPU 644.366049 s、peak RSS 3,365,736 KiB。累積compile待ち 869.989172 sは各workerの待ち時間の合計で、wall時間ではない。高精度phase集計とQiskit inverse生成の逐次部分が残る。比較ベンチマークを行っておらず、speedup倍率は主張しない。

raw 122 files／QPY52件を保存した。SOURCE46／保存入力25／旧証拠727 filesの実行後hash照合はPASS。旧H2/H4/H6のSOURCE・結果・grant・ledger・marker・STOP、2回のnative FAILと独立診断を保全した。報告作成は既存JSON/journalの整形・保存scalarの表示だけで、STOP後のNPZ/QPY数値decode・追加評価は行わない。

## 次の判断とSTOP

次は**ユーザー／研究C**が、今回のnative費用と有限probe結果を確認し、共通CA・shots・条件付き総費用へ接続するための追加資格付けを依頼するか判断する。**Codex**の次の提案は、既存PF／RPE資産を使い、必要な合成誤差の検証範囲・厳密enclosureと数値pilotの区別・実行規模を技術設計すること。重要な科学レビューや新しい数値評価は別途明示許可後に行う。

今回はnative正しさ・kernel費用の固定範囲で完了。結果commit・push・remote SHA／SOURCE／input／artifact照合後に**mandatory STOP**し、全operator追加検証・合成誤差上界・CA・shots・総費用へ自動進行しない。
