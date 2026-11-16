# H4分散Q3：次runの計画と実入力dry-check

今回の成果は保存結果に基づく次runの計画と、固定入力の乾式検査である。
サーバー2でFull PF・B0I・PR-DFの3候補はdry-check PASS、PR-Pauliは不足確認となった。
**PR-Pauliは本番HOLDであり、4方式すべてが最新の現最安を通常入口へ取り込めた計画、
または3ホストで実行可能になった計画とは認定しない。**
新Q3、信号生成、sampling、価格fit・較正、compile、本番queue操作、既存jobのcollectは0件。
元の科学run・SOURCE・入力・結果・STOP・UNKNOWN状態は変更していない。
実2台・3台と切断・復帰の人工検収、42件の既存テストも今回は再実行していない。

## 固定条件と参照した保存範囲

通常入口は `scripts/run_rpe_schedule_search.py`、固定実装SOURCEは
`28d976b99cdf8329adcc1b79327e329e947ef774`、統合基点は
`5829a7288d28395d2b3e89bfb676c23c690b9ddb`。
[PROBLEM](inputs/PROBLEM.json)と[計画round](inputs/FULL_HISTORY_PLAN_ROUND.json)を用い、
`science_approved=false`、全方式の新Q3予算0で通常入口を実行した。
計画結果は[full_history/ROUND_RESULT.json](checks/full_history/ROUND_RESULT.json.gz)にあり、
全方式 `NEW_SCIENCE_NOT_APPROVED`、新Q3呼出しは各0件である。
最初の段階START〜SAVE_RESULTS終了は254.180秒、結果JSONの `wall_seconds` は保存前の91.057秒、
SAVE_RESULTSは163.130秒である。大量のexact保存値の縮約・出力を含む計画処理時間であり、Q3性能の測定ではない。

物理条件はlinear H4・1 Å・STO-3G、固定snapshot
`12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`、
DF rank policy `tol_only_1e-8`、保存DF10 solverを条件付きtargetとする。
共通の `h` はbinary64の0.1に対応するexact値
`3602879701896397/36028797018963968`、`M=12`、`T_m=2^m h`、`P=0`。
評価は `consecutive_dyadic_nearest_branch_principal_energy_total_v1` と
`ALL_EVENT_SQUARED_ENERGY_ERROR`、閾値は
`15136964828156312001/5960464477539062500000000 Ha²` を保つ。
PFは保存決定論raw、PRは有限K=2の保存ideal rawを用いる条件付きIIDモデルである。
Minkowski参照を合算せず、物理的な総誤差保証や正式な `E[C]`・CI・G・全域最適性は認定しない。

保存根拠は次の3系統を区別した。

- [C catalog inventory](checks/CATALOG_INVENTORY.json)：公開branchの最新remoteを照合した
  `e665e9ee60f35bdc40a69894c9cbbc861efa467a`、CATALOG Git blob
  `6add55fea352cb5e465761942720766f9d233df5`、58保存点。
  C固定版にはR06のPR-DF最安と、その後調査したB0I全履歴の最安が未反映である。
- [PF inventory](checks/PF_INVENTORY.json.gz)：元PF探索のREQUEST 557件、保存OUTCOME 556件
  （Full PF 244件、B0I 312件）とB0I UNKNOWN 1件。
  保存評価のID・評価契約・閾値を照合し、元runを変更せずprior resultとして再利用した。
- [サーバー2 inventory](checks/SERVER2_INVENTORY.json)：R06と先行PR探索の保存履歴、
  configuration・signal・price registry・route・完了結果を参照した。
  R06の保存最安はC固定版と別出典として扱う。

固定locatorとsnapshot digestは[SNAPSHOT_LOCATORS](inputs/SNAPSHOT_LOCATORS.json)へ集約した。
共通指示§10.5は
`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-11-14.md`
に従い、configuration名だけでなく物理schedule、X/Y配分、評価契約、価格の適用条件を照合した。

## 現在の保存集合内最安

価格は状態準備を除くfull controlled Hadamard wrapperのcompiled RZ/shotを用いた条件付き総額。
下表は保存済み適合点の比較であり、今回新しくMSEを評価した結果ではない。

| 方式 | 保存最安の構成 | 条件付き総額 RZ | 根拠と通常入口への取込み |
|---|---|---:|---|
| Full PF | Full10、`4th(m5_best)`、要求幅16h | 480258064 | C固定版とPF履歴が一致。通常入口も同額 |
| B0I | p7、標準Yoshida4、要求幅32h/5 | 299307862 | PF保存適合job `2f8a9c696808106e594f`。既存26単価での通常prior-record再積算もexact一致 |
| PR-Pauli | LD5、S2、要求幅h、κ基準×2 | 410199237 | C固定版の保存最安。固定28d通常入口への価格binding不足あり |
| PR-DF | LD7、S2、要求幅h、R各round=1 | 117575550907/126（933139292.91） | R06保存jobを再利用。通常入口も同額 |

B0I最安のX/Y各配分は `[22,34,32,26,18,24,19,19,16,14,10,7,4]`。
PR-DFのX/Y各配分は `[48,45,42,39,35,32,29,25,22,19,16,12,3]`。
他の配分、configuration/total/binding ID、13-roundのq/R・実幅、signal・価格・Q3の固定locatorは
[CATALOG_INVENTORY](checks/CATALOG_INVENTORY.json)と[PF_INVENTORY](checks/PF_INVENTORY.json.gz)を参照する。
C固定版だけを読んだB0Iの372035410、PR-DFの155652932221/126を、今回の現最安へ流用していない。

PauliのC最安行は保存総額と26軸MODEL coverageの要約を持つが、`price_axes=[]` で数値単価列を持たない。
固定28dの通常入口はこの構成を `PRICE_26AXIS` 不足として候補から外し、内部のPauli bestは
437758513 RZになった。この値は410199237 RZを上回るため、真の最新保存基準の代替にはならない。

既存の受動API `PriceRules.quote(policy="MODEL")` に固定configurationと保存registryを与えると、
新fit・compile・Q3なしで26単価を復元でき、同じ配分の加重和は410199237 RZとexact一致した。
根拠はregistryの `PAULI_LD5`、凍結モデル
`fb0d60b0a1e7c31aee73b2b31f802dda7b2f0125:docs/research/a_pauli_ld5_cal_price_20261116/PRICE_26AXIS.json`、
規則 `A5 + H5*q + E5*q*R`（`A5=311, H5=4215, E5=255/28`）である。
これは[分離して記録した価格復元](checks/CATALOG_INVENTORY.json)であり、通常入口の入力bindingは未完。
既存catalogや固定科学SOURCEを書き換えて不足を解消した扱いにはしていない。

## 通常入口が選んだ次のshot候補

PF全履歴を加えた最終の `pending_request` を採用する。
REQUESTは[requests/](requests/)に保存した。すべて未評価であり、下表の総額は既存単価での配分価格。
`UPDATE` もMSE適合や最安更新を保証する分類ではない。

| 方式 | 次候補 | request ID | 分類 | 条件付き総額 RZ | 現最安との関係 |
|---|---|---|---|---:|---|
| Full PF | Full10、S2、要求幅16h/7 | `9b9b4b5132ff89768c5c` | EXPLORE | 545495300 | 現最安より高い探索候補 |
| B0I | p7、S2、要求幅16h/7 | `72864f6cf968d3b514b1` | EXPLORE | 386082768 | 現最安より高い探索候補 |
| PR-Pauli | LD4、S2 | `805877c2ab6c3e03f7f1` | EXPLORE | 4675964940/7（667994991.43） | C現最安より高い。価格binding・route両面でHOLD |
| PR-DF | LD7、S2、要求幅h | `48494ae31276a64c46ce` | UPDATE | 111229232165/126（882771683.85） | 適合なら現最安から約5.397652%の直接改善候補 |

Full PF/B0IのX/Y各配分は `[59,55,51,47,43,39,35,31,27,23,19,12,2]`。
Pauliは `[48,45,42,39,36,33,29,26,23,19,16,13,7]`、
PR-DFは `[48,45,42,39,35,32,29,25,22,19,16,10,3]`。

現最安−候補費用を改善幅とすると、Full PFは−65,237,236 RZ（−13.583788%）、
B0Iは−86,774,906 RZ（−28.991857%）、Pauliは−1,804,570,281/7 RZ（−62.846473%）、
PR-DFは3,173,159,371/63 RZ（50,367,609.063492 RZ、5.397652%）である。
負の値はその配分のPASSだけでは現最安を下げないことを表す。MSEの適否はすべて未評価のまま。

4候補は方式間で独立な初回headとして並列計画できるが、Pauliはblockingのため実行可能としない。
同じ探索枝の後続 `CHAIN` は先行結果が適合して採用された後に選ぶ。
依存する後続候補まで初回から独立Q3として投入する計画ではない。

初回のcatalog中心の計画では、Full PF 436403114 RZとB0I 329694342 RZを未評価候補として挙げた。
PF原runの全履歴を照合すると、前者は既評価のモデル不適合、後者は既評価のモデル適合だった。
556件の保存OUTCOMEを追加して通常入口をやり直し、上表へ選び直した。
今回の通常取込みはFull PF 243・B0I 312・Pauli 3787・PR-DF 15のresume recordsである。
Full PFの保存244件との差1件はcatalog側との重複であり、新規Q3ではない。
Pauliの999件とDFの1件は今回のready候補・route条件との照合で再利用されなかった。
これを未評価件数や新規Q3の必要件数へ振り替えない。選ばれた4配分については、
有効な取込済みconfiguration＋X/Y配分の完全一致がないことを別に確認した。
再発防止上、最安catalogだけでなく同一評価契約の不適合・適合を含むprior履歴を候補選択前に読み込む。

## 幅探索と未準備prefix

shot配分headとは別に、保存239点から既存幅探索helperで次の幅提案を整理した。
[WIDTH_FRONTIER](checks/WIDTH_FRONTIER.json)に全proposalと物理q/R・実幅を保存した。
方式別に見る先頭はPauli LD5・要求幅h・κ=33の `REFINE_DOWN`、
PR-DF LD6・要求幅h/2・κ=101の `EXPLORE_GAP`。
これらはcandidate-specific configuration・signal・26軸価格・routeの準備を要し、
新しい準備やQ3は今回実行していない。

固定28d `WidthDriver.plan` は方式を混ぜて `point.best` を使う一方、
R06の方式別判定は `confirmed_all_best` も参照する。
今回の方式別frontierはこの差を明示した補足metadataであり、固定28dの実装を変更した結果ではない。
R06の科学STOPを解除せず、frontierをそのまま本番dispatch可能なqueueとして扱わない。

PF自動準備のfirst candidateはB0I p9・標準Yoshida4・要求幅32h/5である。
保存T1/T2 evidenceは準備の優先順位だけに使い、Q3採否や価格下界には使わない。
13-round signal packet、26軸価格、新configuration用Q3 route templateが不足しており、shot配分も未指定。
この準備候補を上表のp7 shot候補と混同しない。

## 実入力dry-checkと残る不足

[サーバー2のREADINESS](dry-check/server2/)は、今回のREQUESTをホスト別pathへ解決し、
固定SOURCE・入力blob hashを照合して既存adapterの `--mode dry-check` を呼んだ記録である。
確率評価・DP・signal・compile・queue操作は行っていない。

| 候補 | サーバー2 | サーバー1 | ローカル（RomuTail） |
|---|---|---|---|
| Full PF | [実機dry-check PASS](dry-check/server2/FULL_PF/READINESS.json) | 今回は実機未確認 | 今回は実機未確認 |
| B0I | [実機dry-check PASS](dry-check/server2/TRUNCATED_PF/READINESS.json) | 今回は実機未確認 | 今回は実機未確認 |
| PR-Pauli | [実機で不足確認](dry-check/server2/PR_PAULI/READINESS.json) | 今回は実機未確認 | 今回は実機未確認 |
| PR-DF | [実機dry-check PASS](dry-check/server2/PR_DF/READINESS.json) | 今回は実機未確認 | 今回は実機未確認 |

Pauliは既存intakeが
`Unsupported saved Pauli assembly SOURCE: entry code differs: src/trottertracks/pauli_signal_entry.py`
で停止した。保存assemblyと固定routeが受理するentry SOURCEの差であり、SOURCE guardを書き換えてPASSにはしていない。
最新Pauli最安の価格binding不足と、このLD4候補のroute受理不足は別々に解消する必要がある。

遠隔8セルは未確認。[LOCAL0_DRY_CHECK.md](LOCAL0_DRY_CHECK.md)と
[LOCAL0_DRY_CHECK.zip](LOCAL0_DRY_CHECK.zip)で、local0のWindows PowerShellから
既存alias `qg1`・`RomuTail` を使う実入力dry-checkを引き継ぐ。
サーバー1→サーバー2の直接接続は使わず、過去のrelay・人工run・spoolも再開しない。
PR-DF routeには内側固定SOURCEの絶対pathとPython 3.11.0rc1のruntime条件が残るため、
外側viewだけの配置や過去の人工PASSから遠隔実行可とは認定できない。

R06の起動記録21件（OUTCOME 17件、UNKNOWN 4件）と、旧B0IのUNKNOWN 1件はそのまま保全した。
観測時点でR06のlive jobと完了済み転送待ちは0件、今回collectも0件。
UNKNOWNを未起動と扱って別ホストへ再投入することはしていない。
[B0I UNKNOWN調査](checks/B0I_UNKNOWN_INSPECTION.json)も参照。

## 次runへの引継ぎとロードマップへの影響

機械可読の結論は[RESULT.json](RESULT.json)、開始・再開のdraftは
[NEXT_RUN_ROUND_DRAFT.json](inputs/NEXT_RUN_ROUND_DRAFT.json)と[START_RESUME.md](START_RESUME.md)に分ける。
今回の許可はplan-onlyと実入力dry-checkまでであり、draftの存在を本番起動の許可とは扱わない。
本番へ進む前に、Pauliの保存最安の価格bindingとassembly SOURCEの受入経路、
遠隔8セル、PR-DFの内側path/runtime条件を具体的に確認する必要がある。

研究ロードマップ上は、全方式の新科学実行の前に必要な入力・route準備が明確になった。
保存履歴の再利用によって既評価点の重複Q3を避け、既存単価で比較する保存基準を更新できた。
部分ランダム化の科学的優位性、未評価候補のMSE適合、実H4分散実行の速度向上は新たに認定していない。

## 利用者へ戻す判断

- **現状で乾式検査を通った候補**：サーバー2のFull PF・B0I・PR-DF。直接の最安低下を狙う初回候補はPR-DFで、適合なら約5.40%減る。3ホスト分散開始を今の状態で可能とは認定しない。
- **追加準備が必要**：遠隔8セルの実機dry-check、Pauliの価格bindingと保存assembly受入、未準備B0I p9と幅・κ提案。UNKNOWN5件は元所有runのまま保持し、科学retryを含めていない。
- **次の判断**：local0のCodexへ専用dry-check一式を渡すか、サーバー2で乾式PASSの初回3Q3だけを別範囲で実行するかを選ぶ。新信号・較正・compileを伴う候補準備は、Q3だけの案とは別の科学範囲で判断する。
