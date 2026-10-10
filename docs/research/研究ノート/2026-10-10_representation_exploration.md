# 2026-10-10 独立Hamiltonian表現探索の判断履歴

この記録は独立系列の当日履歴。現行結果は[初期検証報告](../representation_exploration_initial_validation_20261010.md)、
規範範囲は[scope](../representation_exploration_scope.md)。既存Track A/Bの履歴は書き換えない。

1. 利用者依頼を、PR内部改善の別Trackへ統合せず、A/B/Cの初期探索・小規模検証として受けた。
   当初欠けていたGPTレビューは利用者がrootに追加した原本を読み、独立worktreeの入力namespaceへ保存した。
2. mainを最新結果とは仮定せず、b2e1bf6を既存DF/RTE helperの基点に選び、後続Track A/Bの別commitを監査した。
   元rootのdirty差分と1312 pathsの初期状態を保護し、専用branch/worktreeを作成した。
3. factor恒等式・Gram proxy・block平均を先に確認し、分子計算に進まず判別力のあるtoyと対照を固定した。
   source451bfa5を固定したrun1はcontrolled primitiveのscalar phase mismatchでtechnical STOP。
   failure/auditを保持した。利用者依頼には技術的対処が含まれ、旧Trackのone-shot契約を流用しない。
4. scalarだけの差を密行列認証できる場合に限りglobal phase補償し、追加controlの同値性も検査した。
   入力・角度・比較・資源上限は不変、source25d7135で別run2を実行。
5. run2はA20/B9/C3行と19 compilesを完了。Aのproxyと実lambda/costの不一致、Bの平均とtrajectoryの違い、
   Cのmixed oracle代用による理論利益の喪失が比較可能になったため、科学計算をここで停止した。
6. 保存結果・commit source・イベント集計・Taylor行列を別stdlib verifierで監査し、旧結果を上書きしなかった。
   bounded-support可解core＋正確残差は新しい検討候補として記述するだけで、追加実装や採択はしない。
7. 追跡可能なsource/入力/raw結果/失敗/log/環境/監査を公開し、別取得でGitHubの実branch・commit・blobを確認する。
   中心仮説、新規性、研究方針はGPTへ戻す。next-stage=false、central-hypothesis=null、mandatory STOP。

## 追加履歴：独立review後の限定構成・比較batch

8. 利用者の「こんな感じで進める」と独立review第9節を受け、3934583基点の新branch/worktreeを作成した。
   A/Cの小系構成と同一精度比較、Bのisometry接続だけを扱い、元root1316 pathsとdirty状態を保護した。
9. 固定frameの全factor coreはlabel回転不変なのでO-angle走査を除いた。
   [P,Hbar]=0は第一moment保存の十分条件で、一般の必要条件ではないというreviewの訂正を採用した。
   既存二準位toyだけに限定せず、3-mode connected square入力とdegree2 chainを固定した。
10. source/test/scope/inputをce99b57に固定し、native DF全endpoint、actual finite-RTE、対称S2対照、実isometry fixtureを一度実行した。
    pre-freeze引数名不一致の2 test failureと修正後67/70 passesを保存。科学run1は成功しretry0。
11. Aはlambda最小と資源最小が一致せずall-RとのRZ/CX tradeoff、Cは対称S2よりTHRIFTが高費用、
    Bはisometry接続成立だがdirect小系Pauli RTEより高費用。重要な判別結果が得られたので科学計算を停止した。
12. Qiskit/project importsなしの別保存verifierで全374 IR/176 source blobs/会計/有限平均を照合し、3改変を拒否した。
    local開発証拠として新[結果報告](../representation_construction_comparison_results_20261010.md)とmanifestへ追加する。
    元結果・Track A/Bは保持し、公開後はGitHub別取得でblob照合する。中心仮説null、次段false、mandatory STOP。

## 2026-10-11追加：10-10科学reviewに従うA寄与分解・B′比較

13. 利用者のreview採用指示でc24dced基点の新独立branch/worktreeを作成。比較scopeとsource3975650を先に固定した。
14. Aにwhole-H係数収集all-Pauliと全占有対角core、Bにphysical pair Z/Z/ZZとQを追加。旧library/RTE sampler/source/resultは保持。
15. 初回専用testは14 pass/1 fail（RTEEvent field名誤り）を通常修正、focused85 pass/18既存warnings。1科学batchのみを実行し1558 wrappers/30 meanを保存。
16. 次数0全列挙＋次数2条件付き96（Qは64全列挙）、共通uniform couplingで費用と候補差SEを評価。A core改善は安いwhole-Pauli対照で支持されず、B pairはbasis費用でdirectより高かった。C追加scan0。
17. 別保存verifierが177 blobs/1558 IR/933 draw/会計をPASS、最大位相差3.1303e−14。提供有理数algebraも再照合。
18. [新報告](../representation_attribution_pair_results_20261010.md)とmanifestへ追加し、中心仮説null/次段falseでSTOP。科学run retry0、分子/GPU/ground-state/量子shot0。
19. global manifest checkerは基点でもrelevant_commits[7]のdescription欠測で失敗。既存entriesを修正せず、その同一失敗と新entryだけのschema検査を別auditに残す。
# 2026-10-11追記：新設計の限定feasibility準備

利用者は`hamiltonian_algorithm_design_2026-10-11.md`に沿う進行を指示した。
N1縮退生成と群frame、N2 J-only signed chargeと加算込み費用、N3係数下界取得を
[固定scope](../hamiltonian_algorithm_design_scope.md)で一回の小系batchへまとめる。
旧A-core/B′同型探索は一区切り、旧B/Cおよび他Trackのsource・契約・科学結果・STOPは保持する。
新sourceはf98050e基点の独立branchで固定する。前処理83 tests passed、失敗修正ログも保存。
中心テーマ・新規性・一般優位は未確定。結果公開・別取得の後はGPT判断に戻す。
以下の既存の日付内履歴は変更しない。

## 2026-10-11追記：run1の技術停止とscalar位相修正

source d0be140のrun1は最初の基底compileの絶対位相検査で停止。失敗/auditを保持。
Qiskit native Operatorと別IR作用は一致したがbuilt回路とscalar −1で相違した。
raw residual2、scalar residual約2.16e−15。built action全列と照合し純粋scalarだけを補正する
監査を今回module内へ追加し、全raw誤差・補正を記録する。相対位相・漏れは修正不可。
同一科学条件で別source/別runへ技術再実行する。run1の結果を成功扱いしない。

## 2026-10-11追記：run2の技術停止と入力仮定の修正

source b684ba3のrun2はN1三contextのcheckpointを保存した後、N2でnon-scalar差を検出し停止。
Qiskitの既定qubits_initially_zero=Trueが全入力を真空と仮定する点が原因。
任意system/ancilla入力を要求する本契約ではFalseが必要であり、設定変更により
全clean embedding32列のnative対built差は約4.49e−14となった。
個別gate分解は全て一致し、全回路のancilla借用時の入力仮定が問題だった。
全対照にFalseを適用し、sourceを再固定する。科学fixture・予算・q・selectorは変更せず、
run2の失敗・完了checkpoint・診断とともに技術再実行の履歴を残す。
