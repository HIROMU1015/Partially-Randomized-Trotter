# H8長時間・高次PRの第一科学実行案

**提案・実行未承認。今回Stage A/B/Cの実行はゼロ。**
本計画は旧runの再開・既存STOPの解除を含まない。

## 問いと固定条件

長時間化とPF高次化による必要qの削減が、signed piece数、finite-RTE bias、normalization B、必要shot、compiled単価の増加を補うかを調べる。
比較する量は状態準備P=0の `G=N_cos C_cos+N_sin C_sin`。
qが小さいことやwork proxyだけでは優位と判定しない。

Linear H8、隣接距離1.00 Å、STO-3G、保存tol-only1e-8 DF rank28、元state・fragment順序0..27、T=1.6、ε=.0005を第一条件とする。
入力identityは[実装報告](IMPLEMENTATION_REPORT.md)と[VALIDATION](VALIDATION.json)で固定する。
Hamiltonian、snapshot、state、preparation、referenceのidentityは別のものとして扱う。
元T=.8のfull-H信号、複素値の二乗、近似energy位相によるreference代用は禁止。

| 比較群 | prefix | 公式 | q候補 | R_outer | K |
|---|---|---|---|---|---|
| PR S2 | 7,8 | 2nd | 8,16,32 | 4,32 | 最初4、必要時2 |
| PR m5 | 7,8 | 4th(m5_best) | 1,2,4 | 16,32 | 最初4、必要時2 |
| Full PF | 28 | S2,Y4,m5 | 1,2,4,8,16,32 | なし | なし |
| B0/B0I PF | 15,16 | S2,Y4,m5 | 1,2,4,8,16,32 | なし | なし |

これは許可を求めるdomainの案であり、全直積を自動生成・compileする指示ではない。
[configuration](../../../configs/h8_long_time/proposed_scope.json)にもこの区別を保存した。
最初の有限batch案はconfiguration内に57候補を列挙した。PRは両prefix・両公式の主q窓をK4/R32で12点、PFは5つのmethod/prefix×3公式×q1/4/16の45点。いずれも今回未実行。
その後、適格性の境界や比較の情報価値に応じてPF q2/8/32、PR低RまたはK2感度を明示した追加batchで調べる。
このdomain全体の評価・compileを一律には要求しない。PF側も同じ誤差政策で強い公式・prefixを検査し、適格qを絞る。
未評価のqを単調性だけで適格とせず、PF最良を主張する範囲を実測済み候補へ限定する。
許可されたdomain内での追加batchも明示scopeと新保存先を使い、先の結果を消さない。

m5の固定binary64係数とsigned順序は`h46_m5_plan.weights/schedule`のまま。
δ=T/q。R_outerは1 outer全体の総short-step数で、全trajectoryではq R_outer。
ABS_TIME_PER_OUTER配分は各piece最低1step、残りを絶対時間で整数配分し、既存tie-breakを維持する。
Kはeven event cutoffで、平均の多項式は最高K+1次。
同q機構対照は選んだS2のq一つだけを別診断として提案できるが、通常のm5 q窓とは別の`MATCHED_Q_ONLY`候補として実装した。
診断は一つだけ、R32/K4に限定する。実行前にdiagnostic_S2_endpointと元Stage A resultのhashを固定し、同一SOURCE/inputの適格S2候補と一致する場合にだけ有効にする。科学実行許可は別途必要。

## SOURCEと実行前条件

新adapterのSOURCEは、この専用branchへ公開してremote照合した最終commit。
再利用元とcarrierの区別は[SOURCE_REUSE_AND_VALIDATION](SOURCE_REUSE_AND_VALIDATION.md)に記す。
private configと各stage scopeに同じSOURCE・関係するsource file hash・入力bindingを指定する。
旧H8 loaderやproviderの物理入力生成は呼ばない。保存snapshotとFull exportだけをrestoreする。

実行前に必要な確定項目は、Stage Aの57候補案の採用、同時worker数、出力先、Stage Bの探索nと選択規則。
科学実行承認を求める対象はSOURCE、保存入力identity、上のT/ε/domain、以下の科学callと移行規則。
チャットの明示許可で足り、承認packet・署名・grant発行基盤は作らない。
CLIの`--execute`は単なる明示dispatchで、許可そのものではない。

## Stage A：新時間referenceとsignal/shot

入力はprivate configと`stage=A`のscope。scopeはSOURCE、source_files、binding、T、epsilon、state_preparation_cost、workers、candidate関数が生成する各候補rowを持つ。
各rowはsystem/method/formula/prefix/q/T/R_outer/K/idを固定し、許可domain外・重複id・改変idを拒否する。

順序と科学callは次のとおり。

1. H8保存loader、Full-block restore、basis bridge、prefix7/8/15/16とFull28 view。新SCF/DF/state生成や新Gaussian synthesisなし。
2. B0Iで使うexact stored-binary64 scalarを保存H8行列から評価。p16は既存DBAのexact rational receiptと完全照合。p15は新しい計算であることを明記。
3. sparse full-H `reference_pair(T=1.6)`：direct actionと二つのT/2 action、signal/vector差、residual診断。referenceはfull-H作用から求め、energy位相は補助診断だけ。
4. 実際に使用する全signed half-time、必要全blockについてspin作用と独立occupation CSR作用を保存state・先頭/末尾basis vectorの3 probeで比較。差<1e-9。
5. 各候補のsigned PFとPR finite mean。PRのcorrected/raw、forward/Horner、exact-tail PFを計算し、normalization vector closureとrecurrence差<1e-9、exact-tail norm差<1e-9を検査。
6. full-Hからのtotal bias、outer PF/discard bias、finite-RTE bias、log B、数値allowance、axis headroom、N_cos/N_sinを算出。既存`qualify_record`とDecimal整数shot照合を使う。u floor1e-9、log-B margin floor1e-12、u/headroom≤.01を維持し、reference/primitive/recurrence差もuの入力へ入れる。PFにはfinite recurrenceがないため同じvectorを再利用し、重複伝播を避ける。

保存ファイルはscope.json、preparation_binding.json、reference.json、primitive_gate.json、result.json。
結果はELIGIBLE/INELIGIBLE/UNDETERMINEDを明記。科学的error certificateではなくempirical numerical allowance。
この段階はRNG sampling、実回路build/compile、価格取得を含まない。

入力・preparation・DBA scalar不一致、非有限値、reference差>1e-10、primitive/recurrence/norm gate失敗なら影響scopeをHOLD。
stage専用HOLD.jsonを残し、先へ進めない。通常の技術debugは許可scope内で一括処理できるが、科学条件やSOURCEを変える場合は変更理由と新scopeを明示する。

## Stage B：選んだ候補だけの直接compiled価格

Stage AのPASS、同じSOURCE/input、新T reference、固定Nを前提とする。
scopeにStage A resultのpath/hash、候補一覧、cohort=EXPLORATION、n、workersを固定する。
探索n=9は出発案であり未確定。PFは候補当たり両軸1回、PRは候補当たりn trajectory。
Stage Aで情報価値のある候補を選び、両PR公式と強いFull/B0/B0I対照を公平に残す。
適格候補の費用未測定はunknown。0補完、work proxyによるcompiled価格・下界の代用、未測定候補を含む全domain最良の主張は禁止。

実行する科学callは、SOURCE/input/candidate/cohort/index/outer/pieceに結び付く新seed、既存finite lawによるsigned event sampling、DF-native build、full Hadamard wrapper、transpile、測定込みRZ取得。
保存済みの旧seedと新cohort内の重複を拒否する。seed実値・event列・QPYはprivateのまま。
compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/target/coupling mapなし。
scalarを一回だけ加え、actual gate adjointとsymmetric-directional controlを使う。

各候補/cohortの代表index0でlow/high branch、algebraic event vector、cos/sin期待値、compiled vectorを確認する。
native/vector誤差<1e-8、compiled vector差<1e-7、phase alignmentなし。
代表cache hitでもcompiled circuitを検証できるよう必要時だけ再compileし、保存価格と照合する。
signalとnativeのlog BはStage Aのmargin内で一致すること。
fingerprint cacheはこのstage専用保存先に閉じ、compiler hashも照合する。

結果はprices/candidate/index以下のprivate eventsとresult、compiled_cache、seeds_PRIVATE.json、stage result.json。
cos/sinは同じtrajectoryから取得し、固定Nでpaired Gのexact mean、sample variance、SEを保存する。
未観測orderの費用を0とせず、既存rarity診断、finite-bit categorical roundoff、未観測条件費用unknown、population tail boundなしを併記する。
制御・compiler・normalization不一致、sampler/seed/候補不一致、compile失敗はHOLD。通常の一時的PSIや他worktree更新だけではworkerを止めない。

## Stage C：別cohortでの固定候補確認

Stage Bを読んで、確認する候補と比較方向、固定PF対照、精度目標とnを**確認sampling前**に固定する。
現時点でnと精度目標は未定。これは今回の未解決の科学判断である。
Stage C scopeにはStage A/B resultのhash、同じ固定N endpoint、候補一覧、cohort=CONFIRMATION、n、comparison_ids（PR/PFの固定id）、comparison_direction、absolute_SE_G_targetを指定する。
探索と確認で別seed namespaceを使い、探索seedとの実値衝突も拒否する。

探索sampleは `G_i=N_cos C_cos,i+N_sin C_sin,i` とし、軸間共分散込みのsample variance s_G²から精度設計する。
`n≈ceil(s_G²/h_G²)`は目標SE h_Gに対する計画推定に限る。確認後の精度不足を理由に自動top-upしない。
PR対PFではPF価格とNは固定量。PR同士で独立cohortを比べる場合は両群のG分散を含め、候補間pairingを仮定しない。
未観測希少orderによる分散過小評価を検討する。n=9の小SEやゼロsample varianceはpopulation精度保証ではない。
厳密tail保証や別の精度基準が必要なら科学判断として事前に戻す。

実行callはStage Bと同じsampling/build/control/compile/paired会計を別cohortだけで行う。
探索価格は候補選定とn設計にだけ使い、確認標本とpoolしない。
候補差替え・無条件top-up・旧希少RTE runの再開は禁止。
結果はStage Bと同じ構造で、確認n、sample SE、希少orderの限界、事前方向との比較を報告する。
精度未達・方向未解決はそのまま未解決とし、PASSや優位へ読み替えない。

## 資源・新しい条件・許可範囲案

prefix/operatorと70×70のspin eigensystemを再利用し、全sector dense diagonalizationを避ける。
独立candidate/trajectoryをfork workerで並列化し、読み取り専用データを共有する。phase cacheは候補ごとにclearし、歴史的32-time cacheを新研究全体の制限にしない。
compileはfingerprint cacheを使い、同一stageで重複を省く。全直積compileはしない。
worker数は実行時のCPU/RAM/I/Oと所有jobの測定に合わせ、scopeで指定する。CPU/RAM/GPUの恣意的上限や共有設定変更は行わない。
GPU経路・性能・RSS実測は今後の課題であり、今回測定済みとはしない。

一括承認案は「公開SOURCEと固定入力で、列挙Stage A候補、上のdomain内の明示追加batch、適格候補のStage B選定・固定探索n、通常debugまで」をまとめるもの。
確認nと精度政策が未確定なのでStage Cは探索後の具体案で判断する。
T/ε/geometry/basis/DF/state/compiler/domainの実質変更、同q診断の有効化、既存STOP変更、population保証の追加は自動許可しない。

担当：ユーザーが科学範囲と探索nを判断し明示許可する。Codexが許可後にscopeを固定しStage A→必要候補のStage Bを実行、確認設計を提示する。
**現在はmandatory STOP。科学計算を開始しない。**
