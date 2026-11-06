# Track C：3候補の並列科学検証計画

2026-11-06 JST。**計画作成・静的技術確認のみ完了。科学計算・保存値再解析・simulation/sampling・回路build/compile・algorithm実装は未実施。**

3候補を独立に検証できる有限比較案、再利用SOURCE、必要差分、科学call、縮小/STOP条件を具体化した。元H/stateなどの科学条件は未確定のため、**どのpilotも固定実行契約として発効していない**。新しい優劣・科学GO/NO-GOを判定していない。

## 1. 実施環境・成果物・固定資料

SERVER 2を継続使用。利用者所有の専用worktree・新branch `codex/track-c-o3-pilot-plans-20261106`、基点`4b625e5bef06ac51f2b05cafe66f709df2f8bddb`。LOCALへの切替、共有設定変更、他jobの停止・再起動はない。AI sub-agentは使用していない。「並列」は将来の計算/研究の独立性を指す。

現在の**既定interpreterだけ**のmetadataはPython3.12.3、NumPy1.26.4。SciPy/Qiskit/pygridsynth/python-flintはこのinterpreterに未導入だった。科学packageのimportや環境構築はしていない。別の専用venvの不存在を意味しない。SOURCEを読めることと、旧compiler条件で科学実行可能な環境が準備済みであることを区別する。将来は採用profileに適合する既存専用環境を優先し、必要な変更も自分のproject内だけに限定する。

| 成果物 | 内容 |
|---|---|
| [SHARED_PILOT_CONTRACT_DRAFT.md](SHARED_PILOT_CONTRACT_DRAFT.md) | 元H/state・既存H4条件、energy成功条件、native profile、一般signal/shot/価格会計、未決事項 |
| [C_N_PILOT_PLAN.md](C_N_PILOT_PLAN.md) | 既知PFと強い加法proxy/full価格、有限arm、signed occurrence、同時刻比較 |
| [C_S_PILOT_PLAN.md](C_S_PILOT_PLAN.md) | PF分割を変えない同位置residual、3合成arm、coherent mean、strict合成、weighted N |
| [C_P_PILOT_PLAN.md](C_P_PILOT_PLAN.md) | 既存公開D-context kernel/P一組、P有無、全体共役/局所processing、有限信号と端点費 |
| 本REPORT | 独立性、実行前に残る判断、SOURCE/追加実装、科学callと保全・STOP |

正本は[前回プレフライト3文書][preflight]を実読して引継いだ。補助固定commitは3環境統合`3571477093441ac4852166fe829f8ccc5db5ec4f`、先行調査`e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`。既知反証・STOP・再利用資産を再発行せず参照する。文献調査や過去の科学検証の再実行はしていない。

適用AGENTS、PROJECT_MAP、正式共通指示2026-10-25の§8.1/8.2/8.4/10/10.3を確認した。旧研究の正本/manifest/実行契約/SOURCE/raw/result/grant/ledger/marker/STOPは不変更。今回の5文書以外を公開対象へ追加しない。

## 2. 共通条件の静的確認で分かったこと

### H4は使える候補だが、元Hの厳密groundという前提は未確定

既存H4はlinear1Å/STO-3G/8qubit/4電子sector70、legacy DF12、Partial LD3対Full LD12。保存DF snapshotと後続physical_state.npz、元DFの最低固有状態を得るSOURCEは存在する。

ただし[既存energy接続調査][bridge]と[V1][v1]がいう「元H」は保存DFに結び付く。未近似の分子Hamiltonianへの保証とは明記されていない。[m17包含][enclosure]も保存rayと指定ideal categorical lawに限り、真の分子groundへの距離・compiled全実装へのbridgeは未認定。今回これを解決する数値処理はしていない。

したがって主計画は、既存geometry/basis/sectorとDF/native資産を優先するが、`H_phys`の元積分・scalar/orderingとDF近似、共通stateの対応を確定してから主要科学評価を行う構成とした。DF model上の診断だけを希望するなら別の限定目的として選択し、元H energy task達成へ読み替えない。

### 既存18roundと合成profileを正しく分ける

旧opt2/V2にはδ=.02、m0..17、q1..131072、暫定CA/10、α=.05、β=.4、round別r/K/予算がある。[V2][v2]の条件付き成功と[V3-R1][r1]の総費用UNDETERMINEDを保持する。旧4roundだけの資料と扱わず、条件付き18roundを全実装保証とも呼ばない。

旧Qiskit1.3/opt2/rz-sx-x-cx/seed17/free-topologyはC-N/C-Pの最も近い価格profile候補。一方、C-SのT trade-offにはstrict Clifford+T profileが必要で、旧RZ価格と同じprice tableとして扱えない。D1/D3/SPの1e-6/1e-7等は移植候補であり、長qの全誤差予算に適合した新しい科学条件としては未採択。

## 3. 各pilotがどこまで具体化できたか

| pilot | 有限比較案・初期固定 | 最小検証と新たに得る情報 | 実行前に残る固有条件 |
|---|---|---|---|
| **C-N** | S2/Yoshida4/公開Morales Y8m10b、既存h=.02/.01、LD3/12、共通RPE時刻。processorなし・合成固定 | 元H stateへの全18round signal。q1/2/8のpaired価格で、強いoccurrence加法対照とfull価格の選択差を識別。差を同targetのN/Gと不確かさへ接続 | formula集合・signed整数配分案、profile/主指標、有限再最適化自由度、価格精度 |
| **C-S** | S2/LD3/h=.02・original r/K・processorなし。通常合成、一様同位置residual、D classだけresidualの3arm | 同じPF位置/順序のtargetとcoherent mean。strict合成誤差・finite residual・B/weight/range/N、rareを含むfull価格とG | profile T採否、eligible class/coarse catalogue、全interrogation誤差予算とprecision/residual policy |
| **C-P** | 既存YP8m8のD-context raw kernelとdeterministic P一組、外側D/R列とfinite law/合成固定。Pなし/Pあり | 全kernel共役の固有値保存、元H stateの有限phase/radius/混入、N変化と実際に残る端点価格。全Gとunprocessed強対照 | D-context/kernel/n_D/τ/Pの向き案、端点のordinary controlledか測定等価loweringか、数値/価格精度 |

各計画は入力、algorithm構成、SOURCE/関数/script/test、最小差分、correctness、signal/shot、価格、全G、call範囲、保存物、判断条件、独立性、未決条件を記す。未決値を仮の数値で埋めてlaunch可能と装っていない。ユーザーの条件選択後に、追加の網羅調査なしで対象差分を固定できる程度の所在と手順を示した。

## 4. 3候補を並列に進められる条件

| 作業 | 共有するもの | 独立に固定するもの／依存 |
|---|---|---|
| H/state/basis | 同じ元H/sector/state参照、DF入力、primitive/basisの保存cache | 元H/stateが違う場合は別cohort。結果やstateを無条件にpoolしない |
| C-N | 共通signal/error/N/provider | PFだけ変える。合成policy固定、processorなし。C-S/C-P winnerに依存しない |
| C-S | 同じ元H/state・canonical RTE・phase保持builder | PF/科学split/processor固定。合成/residualだけ変える。C-Nで安かったPFへ事後に差し替えない |
| C-P | 同じ元H/state・original finite mean/価格記録形式 | kernel/合成固定、P有無だけ。C-NのPF/CSのresidualを混ぜない |
| 共通会計 | 実半径、weight/range、十分N、X/Y失敗配分、branch、全資源別Gのschema | 旧S2/unit-radius式は一般化が必要。各armの物理平均/価格scopeが異なることを消さない |

共通input/cacheと会計接口を先に同定した後は独立armのoperator/signal・価格層別を計算processで並列化できる。共通cacheの完成待ちで個別静的設計まで止める必要はない。CPU/RAM/worker数の恣意的な小上限は置かず、利用可能資源と共有環境非干渉に従う。新しい合成profileや実行条件を選ぶ際は各pilotの費用単位を明記する。

組合せ実験は初期call範囲に入れていない。相乗効果が必要と分かった場合だけ、単独対照を保持した別の有限比較として扱う。「共通会計を使う」は「3層を一度に変更する」ことではない。

## 5. SOURCE・追加実装の最小範囲

| 固定SOURCE | 確認した再利用単位 | 実装が将来必要な箇所 |
|---|---|---|
| `f29eafbf4412193e8e8660d9f051889c2cba684b` | `PF-ORDER`、`rte.py`のsigned/heterogeneous finite mean、DF event/control builder、`RPERoundCostProvider` | 一般macroをelementary builderへ接続する小さいadapter。実半径/一般bias/weighted Nを渡す共通会計差分 |
| `b190fe28b98d2f94e1509e19cd3d36a63b2bb99e` | `b2d_adapter/_published.py`・`adapter.py`の列/向き/actual adjoint、`b0_native.py`のphase保持方法 | H4 DContextへの限定接続。toy専用launcherをH4 runnerへ流用しない |
| `4ccc1832f63cd323b23fc05623f7e0b58cf34f37` | Parent訂正版の実proposal/moment/range/誤差/phase/native plan | 同位置residualの記述とprovider。新sampling lawやglobal optimizerは不要 |
| `546bca7ed2cece59b714b986a264d7b0b0871a01` | SP economics/adapter、strict key/native、D1/D3の既存precision条件 | 分子の新angle inventory/strict guard/full-wrapper価格。旧toy inventory・projective guardを無条件適用しない |
| `4591b8d75a2a6a55c74a51225f0c61648023c8ee`と後続固定文書 | H4 snapshot/18round input、RPE decoder、価格の共有較正/長qscope | 新armの科学評価・再会計は別実行。保存値の引用は新G評価ではない |

SOURCEは静的本文/ASTとGit treeで確認し、carrier・元execution SHA・依存版を区別した。各planに既存test/runnerの利用範囲を記した。現行のlibraryや旧runの固定SOURCEを上書きして計画を実装したことにはしていない。

必要となるのは対象macro/residual/processorと既存builder/providerの接続、一般signal/error/Nの受渡し、既存scriptが固定toy/S2に拘束される場合の薄い実行入口だけ。今回これらの実装は0。承認authority、grant発行、承認証明packet/receipt/ledger、汎用研究管理frameworkを新設しない。

## 6. 将来必要な科学callと、今回は行っていないこと

共通では元H/state対応の数値確認が必要なら一度の参照H/state処理、DF/basis preparation、各finite armのideal/finite/native signalと半径/phase評価が必要。価格は回路build/compile・必要な角度合成・登録した標本/rare層別、会計は十分N/branch/資源別全G評価を含む。これらは全て新科学callであり、ユーザーの別の明示許可を要する。

初期案は保存q1/2較正・q8 holdoutと全18round mean評価を使い、価格の直接域と長q移送を分ける。原H4の高q価格不足を全候補の大規模compileで機械的に埋めない。未知幅が結論を支配する場合は条件付きG/UNDETERMINEDまでに留める。標本数n、必要shot N、seed、precision/finite-bit、価格lawを別々に記録する。

これまでに行ったのはテキスト・Git object/metadata・ASTの読取と計画文書作成だけ。NPZ数値配列、元scientific runner/test、synthesis/compiler/providerは起動していない。重要な詳細科学レビューや新しい証明の採択も行っていない。

## 7. ユーザーが決める科学条件

1. **元Hとstate**：H4の元積分・scalar・orderingとlegacyDF/保存stateの対応。主要taskを満たす参照state取得を含めるか、DFモデル限定診断に縮小するか。
2. **energy/成功**：旧CA/10・α=.05・β=.4・18round案の採否。元H近似・state・合成/数値誤差を含めるbudgetと外部alias条件。
3. **native/主資源**：QのRZ/CX/depthとTのClifford+T比較のどれを採るか。特にC-SのT profileと、全round合成precision policy。
4. **有限armと比較自由度**：C-Nの3formula/2h・signed配分、C-Sのsame-position residual class/catalogue、C-PのD-context kernel/P/向きと端点実装。全armへ同じβ/α/precision調整機会を与える範囲。
5. **判別精度**：数値/state/合成誤差の根拠、価格標本/strataと不確かさ、差のmateriality。精度不足を物理仮説の負結果へ読み替えない。

これらを仮採択して実行はしない。正式GO/NO-GOを付けず、各候補の有効/不成立を判別する計画だけを渡す。旧P-D/BF/BM/SP/Parent/研究B/FRのSTOPや負結果は、別の目的での資産再利用によって解除されない。

## 8. 技術検査・公開・mandatory STOP

公開前の検査対象は5文書のUTF-8、相互リンク、表/Markdown、固定commit/pathの存在、変更pathの範囲。旧数値を再計算してチェックするtestは行わない。独立branchへ5文書のみcommit/公開し、固定remote SHA、取得本文とGit blobを照合する。照合の実績は最終チャットで報告する。

文書作成時の静的検査では、5文書のUTF-8・相互リンク・表構造、主repositoryの固定参照13件、6つのrunner path、processor/Parentの指定関数・class名の実在を確認した。opt2のJSON selectorもkey構造だけ確認し、予算/N/価格を再評価していない。これは実装test PASSや科学的実行可能性の認定ではない。

**mandatory STOP：計画作成・5成果物公開・remote照合まで。科学実行/compile/実装へ自動移行しない。**

次の担当は**ユーザー**：上記科学条件と進めるpilot・許可範囲を選ぶ。**Codex**：別途明示許可されたSOURCE/入力/条件の範囲だけ、必要な接続実装と科学検証を担当する。現時点で自動開始する次の作業はない。

[preflight]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/track_c_o3_technical_preflight_20261106
[bridge]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/pr_rpe_energy_bridge_readiness_20261103/REPORT.md
[v1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[v2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[r1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md
[enclosure]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/rpe_m17_rigorous_mean_enclosure_execution_20261103/REPORT.md
