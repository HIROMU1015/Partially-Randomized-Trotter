# Track A H4：1 Å nativeの対照・PR局所探索・非相殺診断

利用者の「その方針でよいので、検証を実際に開始するところまで進めてよい」を、以下の
3つの独立scopeの実行承認として結合する。geometry六距離系列は保留し、旧結果を変更しない。

linear H4/1 Å/STO-3G/legacy固定DF rank12/8 modes/sector36/T=.8/P=0。
保存Hamiltonian・state・full-H referenceを再利用する。SCF・DF・state生成・新full-H referenceは0。
元SOURCE217c971d…、固定16結果0f96a2e4…、D0の5精度・倍率1/10の会計条件を継承する。
native symmetric-directional full Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/no topology。
H6/H8のtol-only DF政策との同一性や統制されたサイズscalingは認定しない。

## 実行範囲

- D：prefix2/3/4/6/9/11、q1/2/4/8、S2/global Yoshida S4。
  B0の48 signalとB0+Iの48解析的phase overlay、計96候補。
  full-Fock tail identityを保存binary64係数の有理数で計算し、signal/referenceの経験的allowanceを分離して回転する。
  通常/非相殺×5精度、allowance倍率1の10 endpointで、事前固定screening proxyの最小候補のunionを取る。
  proxyは同q/次数の保存full-PF実測RZ/軸×(prefix+1)/13×shot。係数fitは行わない。
  **これは費用モデル・精度保証ではなく、直接実測する最大10構成の選別規則である。**
  最大20 compile/40 native control probe。構成間を並列化し、各構成でcosine受入後にsineへ進む。
  新しい費用モデルがないため2%モデル検証基準をこのproxyへ適用しない。
  native制御・scalar・X/Yを独立sector signalと1e-9で照合し、技術的不一致ならSTOP。
  費用比較は旧16＋直接実測した追加部分集合に限定し、未実測Dを含む全96最良はnull。
- A：p2/4×q1/4/8×r4×K2/4の12構成、p3/q1/r2/K2、p3/q4/r8/K4、p3/q8/r4/K2/4の4構成。
  計16の新signal/normalization/通常・非相殺適格性。既存8 PRと重複しない。
  q16、新sampling、PR cost compile・順位付けへは自動移行しない。
- N：旧native固定16の保存成分のみ。通常biasの元128会計行と保存選択8費用の再現をgateとし、
  各軸で成分絶対値を足した経験的非相殺biasのshot/費用を5精度×倍率1/10で診断する。
  cosine/sineは同一trajectoryとしてpairedに扱い、各PR n32。新signal/sampling/compileは0。

非相殺診断を厳密bias保証と呼ばず、標本平均・SEを母平均保証としない。
欠測を0補完しない。未完了の新A候補を費用最良集合へ混入しない。
今回の承認は希少層pilot・新PR費用sampling・T/state/DF変更を含まない。

## 実装・並列化・保全

旧MatchedPort、occupation/forward oracle、native lowering、empirical accounting、B0+I bridge、
owned fork worker/same-parent wait4を再利用する。人工17件でscope・相殺・欠測・shot/paired SE・
scalar回転・S4負時間/control・grant拒否・選別・native worker回収を検査した。
これは作者自身の技術検証で、独立reviewではない。

独立clone/branchとD/A/N別grant・one-shot lock・outputを使用する。
fresh physical cores/RAMから並列度を選び、内部thread1で過剰な入れ子並列を避ける。
primitive/native/spectral cacheをwarmし、copy-on-writeで共有する。
production numba/OpenMP operatorはfork後のworker内で初期化する。
一時的PSI・CPU競合や無関係worktree更新だけではSTOPしない。
自分のSOURCE/input/provider/runtime破損、非finite、科学的不一致、worker異常にはfail closedする。
他研究のSOURCE・raw・grant・ledger・marker・STOPは書き換えない。AI subagentは使用しない。

各stage終了後mandatory STOP。自動retry・再fit・追加候補・次stageへの自動拡張は行わない。
[契約・入力・SOURCE・実行証拠](../../artifacts/resource_applicability/track_a_h4_native_validation/2026-10-18/)。
実行開始・終了の実測状態は後続の時点付きreceiptへ記録する。

次の担当：Codexがfresh gate後に承認済みD/A/Nを各一回起動し、安定進行・終了証拠を回収する。
ユーザーは結果を見て、A候補の費用取得等を次に行うか判断する。

## 起動・正常終了snapshot

時点：2026-10-18T15:00:09.753927+09:00。SOURCE `79758d28d89ccaf2f942dc10ad724e70cf513089`。
3つの独立grantを発行し、SOURCE11/provider236/runtime34,273/入力9件のfresh gate後に起動した。
Dは10 worker枠、Aは16 worker、Nは1 worker、内部thread1。fresh空きRAM約468 GiB。
Aは16/16・native exit0・wait4全回収・最終gate PASS、wall11.563秒でmandatory STOP。
倍率1での通常/非相殺適格数はε=.05/.01/.005/.001/.0005の順に16/11/11/11/6（両bias同数）。
新PR sampling/compile・費用順位は0。
Nはnative exit0・wait4回収・最終gate PASS、wall7.408秒でmandatory STOP。
通常128会計行と保存選択8費用を再現し、旧固定16内の非相殺費用比較を保存した。
Dは48 B0＋48解析overlayを完了。固定規則で8構成を選び、最大16軸を直接compile中。
snapshotでは4構成完了・10軸取得・STOP0。全8構成/16軸の正常完了はこの時点では未確認。
[A/N完了receiptと結果gzip、D時点付きsnapshot](../../artifacts/resource_applicability/track_a_h4_native_validation/2026-10-18/)。
次の担当：CodexがDの残り実測・native終了・最終gateを回収する。追加候補やPR cost stageは起動しない。

## 完了追記：D/A/Nすべて正常終了

DもSOURCE79758d28…で8構成16軸の直接実測を完了した。48 signal worker＋8 compile workerがnative exit0・同じ親wait4回収・result SHA一致。driver native exit0、最終gate PASS。
本体wall278.059秒、witness込み278.640秒（約4分39秒）。新compile16、native control32。
通常/非相殺の費用比較は旧16＋直接実測8構成の部分集合に限定。残る88 D候補と新A16の未取得費用を0補完せず、全96最良はnull。
[D完了receipt・保存結果gzip](../../artifacts/resource_applicability/track_a_h4_native_validation/2026-10-18/D_completion_receipt.json)。A/Nの完了証拠は前節の通りで、3stageともmandatory STOP。
### H6/H8で実施した設定との対応

1 Å/STO-3G/T=.8/native symmetric-directional/Qiskit1.3.0 opt1 seed17/P0、S2/S4対照、K2/K4、5精度、通常/非相殺と経験的allowanceを対応させた。候補集合・母平均・厳密保証の同一性は主張しない。
H6固定δ=.2/T=.4/.8/1.6 panelに対応するH4の新T検証、H6/H8探索のr1と境界q16、希少Taylor条件付きpilotは今回のH4scopeにない。H4はlegacy固定DF12のままで、H6/H8 tol-only生成政策には統一していない。別入力状態・別compiler設定はH6/H8でも今回系列の完了範囲ではない。
次の担当：ユーザーが、H6/H8で実施済みの範囲との差を埋めるH4追加scopeを選ぶ。Codexは承認された範囲だけを準備・実行する。
