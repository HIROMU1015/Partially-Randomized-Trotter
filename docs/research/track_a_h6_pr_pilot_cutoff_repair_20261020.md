# H6 PR DESIGN_PILOT：cutoff修正と軽量な並列化・高速化

対象はH6直鎖1 Å、STO-3G、tol-only 1e-8/DF19、保存Hamiltonian/state/reference、prefix5、T=.8、P=0。S2/Yoshida4/m5_best、q1/2/4/8、K4、ABS_TIME_PER_OUTER=32、12群×33本の既存pilotである。

修正SOURCE：`e7595e8ff543a0d0f5e1ea91dec14283b61f9c32`。
今回は技術修正、保存metadata/byte照合、人工テストまで。新しい実H6 sampling・compile・signal・状態作用、pilot統計、shot、会計、ランキング、有効grantは全て0。H4完了結果とH6元runのSTOPは保持した。AD03と正式shot政策は保留。

## 1. 原因と修正

元SOURCE `de5485d4b7c6de25dd9b6137c9a12bce697242c6` のH6 run01は、m5/q2/sample0の4番目のsigned pieceで `finite Taylor cutoff does not meet truncation_tolerance.` によりSTOPした。

外側の許容値生成は `(lambda_R * signed_time) / rte_steps`、factory内部は `symbolic_tail.lambda_r * (signed_time / rte_steps)` を使っていた。固定K4を受け入れるhelperは計算したresidualの直上1ULPを許容値にするため、τの丸め差がresidual側で増幅されると拒否する。

保存contextと停止前3片のlawを照合した結果、λは同じ `0.9182185130575018` で整合した。今回の停止を説明するためにλの差を仮定する必要はない。[保存metadata診断](../../artifacts/resource_applicability/track_a_h6_pr_pilot_cutoff_repair/2026-10-20/saved_metadata_audit.json)は、成功3片と停止片の判定を旧経路で `[PASS,PASS,PASS,FAIL]`、修正経路で全PASSとして再現した。

| 停止片の値 | 旧外側 | factory内部・修正後 |
| --- | ---: | ---: |
| τ | 0.017362852976247525 | 0.01736285297624753 |
| 許容値 | 3.814804273394561e-14 | 3.814804273394575e-14 |
| factoryの残差上界 | 3.814804273394574e-14 | 同じ |

`canonical_tau()` でfactory自身のsymbolic lambdaと同じ演算順序を使う。K4の選択、signed time、整数step配分、seed、確率法則、event順序は変えない。piece receiptにτ・許容値・policyを記録し、factoryのτと完全一致するgateを追加した。

以前のcompiled状態差CT01（1e-7未満）とは別の修正である。CT01、native/oracle/Hadamard等の1e-9未満、compiler Qiskit1.3.0/rz-sx-x-cx/opt1/seed17、scalar-once/control、level1保存方式は変更しない。科学的なbias許容基準を緩和する変更ではない。

## 2. 採用した軽量な高速化

新契約で `H6_PR_PILOT_CANONICAL_CUTOFF_AND_SHARED_CONTEXT_V1` を明示した場合だけ有効になる。元SOURCE・grantへ自動適用しない。

- H6の保存contextを親processでprefixごとに1回構築し、POSIX forkでworkerへ共有する。同じH6/prefix5を各workerで復元・再構築する重複を避ける。書込は各workerのprivate copyになる。
- fork前はPython threadが1本であることを前後2回検査する。監視threadは全worker起動後に開始する。親context構築後にもfresh CPU/RAM/scratch gateを再実行する。forkが使えない場合の自動条件切替は行わない。
- 新policyの並列数は、登録task数・実空きCPU・実効空きRAM/既存の同入力RSS proxyから起動直前に決める。過去のH6 28workerを新作業の恒常的上限へ転用しない。旧policyでは元のadmission条件を保持する。
- seed表とsignal表をworker内で一度だけJSON解析する。毎lookupで全byteのSHAを検査し、同size・同mtimeの改変も拒否する。選んだrecordをcopyして返しcache汚染を防ぐ。既存のSOURCE/input/compiler全検証も保持する。
- 長いsigned scheduleの優先dispatch、trajectoryごとのprocess並列、cosine受入後sine、nested BLAS/Qiskitの並列過剰を避ける既存経路を維持する。共有環境・他job・CPU affinity・OS設定は変更しない。

人工1.05MB入力表・80 lookupでは解析cacheが約14.8倍だった。実H6の全体速度・RAM削減率は未測定であり、この倍率を転用しない。call予算のcache候補は人工測定で約2倍遅くなったため採用せず、元の直接計算を保持した。[測定範囲](../../artifacts/resource_applicability/track_a_h6_pr_pilot_cutoff_repair/2026-10-20/artificial_benchmark.json)。

## 3. 検証と既存結果の保全

新規16 testと既存回帰161 testがPASS。追加したpost-prewarm資源gateの人工launcher検査もPASSした。作者自身の技術検証であり、独立レビューではない。

新規検査には、演算順序・symbolic lambda、3公式×4qの生成law/event不変、K4/seed/step配分、改変検知cache、fork共有・exit/reap、fresh並列数、最適化を有効にした人工native両軸/6archiveのend-to-endを含む。保存H6 stateを使った再compileやnative作用は行っていない。

保存74完了pair/148軸について、8 native gate、元SOURCE/plan、444 archiveの原byte SHAを再照合した。生成4404片のseed・signed time・rte_steps・τも新計算法と一致した。これは再利用の技術的な証拠であり、再利用の承認や元396本全体のPASSを意味しない。

旧SOURCE/runtime/provider/inputと元runのraw・grant・ledger・marker・STOP等、作業前に記録した21,727 file（3,262,947,581 byte）を同じpath集合で再照合し、変更0だった。[保全receipt](../../artifacts/resource_applicability/track_a_h6_pr_pilot_cutoff_repair/2026-10-20/preservation.json)。無効な再開提案はexecute/supervise両CLIで、科学入力を読む前にexit2で拒否された。[検査記録](../../artifacts/resource_applicability/track_a_h6_pr_pilot_cutoff_repair/2026-10-20/draft_rejection.json)、[人工・回帰の範囲](../../artifacts/resource_applicability/track_a_h6_pr_pilot_cutoff_repair/2026-10-20/tests.json)。

## 4. 再開に向けた提案と停止境界

既存33本×12群＝396登録を維持し、完了74本を再利用、未完了の固定322登録を取得する案である。再利用が別契約で採択されれば、新規compileは最大644軸。旧予約172軸の消費・部分出力・STOPは保持し、新予算へ返却しない。

未完了登録は元のseed/trajectory IDを使い、partial cosineと新sineを継ぎ合わせない。再開を承認する場合は未完了trajectoryの両軸を新領域へ取得する。希少eventを見てseedを引き直したり、結果依存で標本を補充したりしない。

[無効な再開提案](../../artifacts/resource_applicability/track_a_h6_pr_pilot_cutoff_repair/2026-10-20/continuation_proposal_DRAFT.json)を保存した。これは実行可能なgrantではない。既存runnerの396登録固定gateとSOURCE/registry検査は維持している。322登録だけの継続には、74本の独立受入・元registryの結合・残余call scopeを扱う専用adapter/契約の作成と、ユーザーの実行承認が必要である。

今回のユーザー指示は修正と高速化であり、H6再sampling/compileの承認として扱っていない。旧契約の自動retry・top-up禁止を維持し、新たな有効grant・科学workerは起動しない。

**mandatory STOP。** 次の担当：ユーザーが74本の再利用と残り322本の取得を採択し再実行するか判断する。承認がある場合、Codexが専用継続adapter・SOURCE/input/registry/resourceのfresh gateを完成し、承認範囲で実行する。統計評価・n_E決定・会計・AD03へ自動移行しない。
