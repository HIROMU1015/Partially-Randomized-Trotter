# H8共通資源比較：科学判断前の実験・高速化・予算案

2026-10-13完了追記：限定8ケース20compileが正常完了。[実測結果と更新技術予算](track_a_h8_fast_feasibility_result_v1.md)を最新値とする。SOURCE9bfea8bd、heavywall7126.273秒/RSS6.815GiB、残7wall3012.395秒。core1708/832、carryはH4補完/既存H8failedを加えた既知lineageで3997/1944案、optional tail追加80/40。preflight reuse binding未確定で節約予約0。full-map runtime_authorization=false、grantなし。以下の古い未完了/未測定・H6外挿値は履歴として保持。

2026-10-13最新指示追記：利用者はGPTレビューで採択したresource-mapを主評価と明示した。rank28・relative prefix7/14/21・q1/2/4/8・r1/2/4・K2/4・fullPF S2/S4・finite-time complex-signal accuracy matching・測定込みfull-wrapper RZ総費用を保持し、固定H6設定transferを副評価の別枠にする。本格実行承認は未取得。[限定feasibilityの保存監査](../../artifacts/resource_applicability/track_a_h8_signal_compile_preflight_v1/2026-10-13/README.md)は8signal/native1164完了、20compileは進行中。全件完了後にwall・RSS・容量・carryの技術的見積もりを更新し、未実測をPASSにしない。以下の古い「primary未決」やH6倍率仮定は当時の案として保持する。

2026-10-13事前検証追記：[入力/stateの実受領結果](../../artifacts/resource_applicability/track_a_h48_preflight_result_v1/2026-10-13/README.md)。linearH8/1Å/STO-3G/tol-only actual rank28/16modes/sector4900、weighted N16 decision gate9.9e-11を保持、eta3.6595e-12、snapshot SHA d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955。提案のrelative prefixは7/14/21。H8 sourceのsolver dimension400 guardは別moduleで4900へ対応したが、signal/reference/native16-mode/compile feasibilityは未完了、full comparison未認可。以下は入力未受領時の設計案として保全。

2026-10-13 JST。[machine-readable案](../../artifacts/resource_applicability/track_a_h6_manuscript_h8_plan_v1/2026-10-13/h8_plan_draft.json)。approved=false、grant=null、runtime_authorization=false、allowed_cpus=[]、technical_sealed=false、absolute_launch_command=null。H8実装・input・rank・性能は未確認。

利用者は「計算する場合はできるだけ高速化・並列化して開始」と指示し、後に「準備できたら開始してよい」と許可した。開始対象はH8小規模事前検証とH4比較補完に確定し、本格H8は対象外である。未確定のH8科学契約やCPU/RAMを自動採用したgrantではない。細かな実装判断・metadata検証はCodexで進める。

## 1. 研究目的を二つに分ける

第一の候補は同じpre-result generatorによる系別resource map、第二はH6事前選択設定のH8へのtransferである。前者のprefixは相対rank、後者は絶対prefix5を保つため同じ実験ではない。primaryをどちらにするかをGPTで決める。PR勝利を成功条件にしない。

提案targetはlinearH8/1.00 Å/STO-3G/16 modes/電子4+4/T=.8、tol-only DF1e-8、cutoff0、signed generation order、Hermitian projection。actual rank・Hamiltonian/state bytes・ground-state品質は未取得。H6 rank19やstate/sourceを代用しない。

epsilon=[.05,.01,.005,.001]は複素signalの要求精度である。symmetric_directional measured full wrapper、paired cosine/sine、RZ/CX/depth/size、P=0とcommon P感度、Qiskit1.3.0/basis/opt1/seed17/topologyなし、原RTE law/normalization/shot式を比較軸として維持する。matrix-free検証で数値allowanceが変わる場合は差異を事前に科学契約へ明記する。

## 2. 候補generatorと選択

Lが固定された後、prefix=unique ceil(L*k/4),k=1,2,3、0<p<L。L=19なら5/10/15、24なら6/12/18、32なら8/16/24、48なら12/24/36。これらはrankの仮定を示すscalar例で、H8 DFの測定値ではない。

core：q1/2/4/8、r1/2/4、K2/4。B0 prefix S2、B1 full S2/S4、B2 partial S2。3prefixの場合92信号候補、PR各8探索、各精度上位2のunion最大8を独立32確認する。selection tie rule・namespace・seed/source/input hashを結果前に固定する。precision間で同じ回路の費用が再利用できることと、trajectory標本を複製することを混同しない。

q8がcoverage boundaryなら未解決と記録する。q16を含める別tierは結果前に選択し、B0/B1/B2全てへ公平に追加する。PRだけ結果を見てr/q/pを広げない。絶対p5 transferをcoreと同時実施し、p5がcoreに含まれない場合は、別8信号/最大256trajectory/512compileの予算が必要である。下表にその追加は含めない。

rare costのpre-result診断案は選択max8のs2を各4本、K4 max4のs4を各2本、max40 trajectory/80compile。元法則を層だけに条件付け、component分布と量子重みは保持する。これをmain層別estimatorにするかdiagnostic-onlyにするか、保証の範囲とともにGPTで先に決める。40本でpopulation tail boundを閉じるという意味ではない。

## 3. 計算量とcarry

全deterministic候補が適格な保守上限。cache節約は0予約。

| 案 | 信号候補 | 探索group上限 | 探索PR trajectory | 確認trajectory | optional tail | compile合計 | trajectory合計 | random occurrence上限 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3prefix・q<=8 | 92 | 596 | 576 | 256 | 40 | 1788 | 872 | 4528 |
| 3prefix・q<=16 | 115 | 745 | 720 | 256 | 40 | 2086 | 1016 | 9200 |

compileにはpaired primaryとordinary control2anchor×2axes=4件を含む。tailを採用しなければcoreは1708compile/832trajectoryとなる。H6の592という実適格group数をH8の予算上限へ流用しない。

現carryはknown compile1716/trajectory842、予約20/10、承認cap1736/852。返却・reset0、unknown historical actual=null。core追加1788/872だけでも累積上限案は3524/1724、q16案は3822/1868となる。既存occurrence known3200+予約64にcore4528を加えると7792で、旧6656を超える。control/primitive/referenceも旧cap1024/10000/20000との余裕を、input/validation設計後に再計算する。

SCF/DF/state/input生成、reference matvec、primitive/control/numerical検証の件数はまだ未確定で、上表は全campaign予算ではない。新grantにはこれらを加えた完全なcarry込み枠が必要。科学入力を今回再生成しない。

## 4. 技術的に必要な変更

H8の粒子sectorはC(8,4)^2=4900である。既存H6 portのsector400 guardを超える。complex128のsector行列1枚は384160000 bytes（約366.36 MiB）、H6の150.06倍。dense cubic演算規模は約1838.27倍で、wallの実測予測ではない。16qubit full dense unitaryは64GiB、ancilla付きは256GiBとなるため、その経路は避ける。statevector自体はsystem1MiB/ancilla2MiBで、matrix・compiler object/cacheとは別である。

単にASを上げてguardを外すだけでは準備できない。H8専用input/schema/source closure、sparse/matrix-free sector作用、Krylovなどのreference候補、独立小oracleと登録数値probeの設計、native 16-mode Gaussian lowering、phase/control/nonfinite/signed-zero serializationを実装する。dense400のexpm/eigh確認をmatrix-free4900へ移す数値品質の扱いは科学契約の判断対象である。これらは現在未実装・実H8未検証。

## 5. 小規模事前検証と本格campaign

最初のscience phaseは入力freeze・state品質・reference/primitive/numerical gateだけを別budget/grantで行う案。続くcompile feasibility panelはB1 S2/S4のq1/q8（4det）とquarter-prefix B2の(q,r,K)=(1,4,2),(2,4,2),(4,4,4),(8,2,2)各1本（4PR）、primary16compile＋ordinary2anchor4compile、計20compile/4trajectory/15random occurrenceを提案する。

最初のheavy S4/q8は1workerでwall/AS/RSS/instructions/outputを測り、remaining panelをweighted admission下で並列化する。小規模panel結果だけで精度適格・母平均・全map完走をPASSとしない。結果後のworker数調整は計算資源判断であり、科学候補変更とは分ける。

事前検証20/4を本格coreにcomplete-onlyで再利用できるよう、同じtask/seed/namespace/compilerを最初に固定する。上の4528はfull campaignのPR occurrence上限であり、未受理partial・追加validation・やり直しは別予約で加算する。再利用bindingが不成立ならcoreに20/4/15を加え、3524/1724/7792を最終capと誤認しない。

## 6. 最大限の軽量高速化を先に組み込む

- 既存H6のcanonical hash再利用をportし、input/source/compiler/dtype/shape/phase/control/orderのidentityを保持する。変更を検出するguardを含める。旧H4/H6の結果をH8のcacheとして代用しない。
- immutable DF/Gaussian preparationとprimitive/time lookupをinputごとに一度固定し、workerで再利用する。matrix/cacheは個別RSSとaggregate admissionに含め、上限を設ける。
- 独立groupを最大8 workerへ分配、各内部thread1。seed/taskはworkerや実行順に依存させず、paired axesは同じevent列を使う。完了順が違っても出力順・科学標本が変わらないようにする。
- matrix-free作用と疎のボトルネックを事前profileし、同じ演算・誤差budgetの範囲で処理する。実測前に速度倍率を保証しない。
- complete-only checkpoint/resume、未完了予約保持、新grant/one-shot lock、own-run STOP/cleanupを準備する。再開時に結果hash/source/profile/inputを照合し、old grantを再利用しない。

実装不足を自律的に補うが、compiler設定・回路構成・random law・科学comparisonを高速化の名目で変更しない。GPUは使用しない。

## 7. 資源・時間・容量案

通常worker AS32GiB/RSS9GiB、heavy worker AS32GiB/RSS24GiB、coordinator AS8GiB/RSS2GiB＋余白2GiB、aggregate RSS76GiB、job admission96GiBは未承認の候補。8normalは72+4=76GiB、1heavy+5normalは73GiB、1heavy+6normalは82GiBで不可。8heavyは196GiBを要し、共用サーバーの標準案にしない。H8実RSSが未測定ならこの分類を確定しない。親のAS hard limitはworker用32GiB以上を維持し、fork/exec後にworkerがsoft32GiB、親がsoft8GiBを適用できる順序を検査する。親hard8GiBをworkerへ継承させない。

CPU候補worker20/21/24/25/26/27/28/29、co22/23。allowed_cpusは空のまま。fresh affinity/他run配置を照合後、own-runだけに適用し、他jobを動かさない。親のaffinityをworker CPU確認前に22/23へ狭めない。CPUの占有権をhost affinityから推論しない。

H4等のheadroom152.25GiBを残す候補として、fresh host available248.25GiB/effective96GiB、runtime host168.25GiB/effective16GiB。host full PSI>=5%、cgroup full PSI>0、新OOM、監視5秒超・欠測、resource upper超過はfail-closed案。H4に過去承認したPSI猶予をH8へ自動移植せず、threshold/graceを新契約に明記する。schedulerなしでもユーザーの運用条件とfresh shared resourceを分けて確認する。

保存H6 anchors：S4/q8実worker peak7.929GiB、paired axis classical fields和1653.25秒。確認256groupのclassical field和13646.58秒、median50.95秒、max113.97秒。探索wrapper fields和66996.17秒。これは重複/phase/sourceを含むmetadata anchorsで、campaign elapsed合計ではない。H8費用がこれらの4/8/16倍という仮定だけを置くと、8worker理想下限は約11.20/22.40/44.80時間になる。reference/input・tail・I/O・load imbalance・single heavy critical pathは別で、H8完了時間は未推定である。H8事前検証後に実測で更新する。時間上限なしは利用者の希望に沿う候補だが、資源監視とSTOPは維持する。

output32GiB/storage admission64GiB/inode100000案。core892groups×32MiBは27.875GiB、q16は1041groups×32MiB=32.53125GiBなのでq16には同じ32GiB出力案を採用できず、上限を再見積もる。1fileのobject dumpを避け、compact metadata/metrics/event journalを保存する。最大file数、4KiB等block allocation、manifest/checkpoint/control/monitor/journal/temp/cacheも含める。monitor/journalはbounded rotation＋hash-sealed segments/terminal summaryで、長時間runが無制限log増加や有限journal上限STOPへ陥らない設計にする。quotaはDISABLED/KNOWN/UNKNOWNを区別し、UNKNOWNをPASSにしない。disk/inodeの数値はfresh gateで取得する。今回共有FS設定変更なし。

## 8. 次の研究判断と実行gate

GPTで確定するもの：primaryをresource-mapかtransferにするか、prefix/q16 tier、state/reference/numerical allowance、tail diagnostic/estimator、H4 panelの範囲。研究条件確定後、Codexが専用worktree/source/profile/input/carry/budget/output/plan/authをbindingし、軽量差分review・必要tests・fresh gateを完了する。H8小規模事前検証とH4比較補完の準備完了後の開始許可は受領済みで、実行値の固定と技術的な準備を続ける。

不適格（科学bias/normalization）、数値未解決、資源STOP、coverage boundaryを区別し、結果欠測を0cost・PR敗北・PF不可能へ置換しない。H8科学結果未取得、source未固定、input未受領、technical plan未seal、実行command未生成。新grantやscience runnerは今回作成・起動していない。
