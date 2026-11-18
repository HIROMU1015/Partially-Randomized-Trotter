# H4 PR最適化器：探索戦略と計算本体の高速化（読取レビューと人工例）

想定読者：B2（最適化器の担当）、B（中央controller）、統合担当。
- 対象：SOURCE `5c8ced4b2d407a3828d189e1e0f0a24a4807f94a` の PR-Pauli／PR-DF 探索。
- 作成：2026-11-18、A1109（Claude Code）。作成者自身の読取レビューで、独立レビューではない。計算本体の調査（§4）の一部はサブエージェントに読取専用で任せ、要点は作成者がコードと保存ログで確かめてから使った（§4に明記）。
- 新規の実H4計算・Q3・信号・価格較正・compile は0件。稼働中のrunと採用規則は変更していない。
- 中央レビュー E1〜E4 は配送・実行側の結果として引き継いだ。E5（時間あたり削減の順位付けがほとんど効かない）は前提とした。

証拠の表記：【保存】保存runで確認、【コード】コード上の事実、【人工】人工例で確認、【要計測】追加の計測・Q3が必要。

## 冒頭の答え

| 問い | 答え |
|---|---|
| 最初に試す変更（1つ） | **V2のEXPLOREに「自価格の下限」ゲートを入れる。** 構成自身の単価の和（全roundと両軸に1 shotずつの配分の費用）が、すでに方式最安 M 以上なら、その構成のどの配分も M を下回れない。その構成の候補をEXPLOREから外し、DEFERRED（ほかに候補が無いときだけ試す）に回す。<br>関数：`rpe_priority_search.run_priority` 内の `classify`（5c8ced4b2 の272行目付近）。MSEの単調性は仮定しない。 |
| その根拠（保存例） | 【保存】R08 live-ready 継続run（run02）で、採用されたQ3 149件のうち57件（PR-DF 28、PR-Pauli 29）が、その時点で自価格の下限 ≥ M の構成へのQ3だった。<br>例：PR-DF LD3 `92ad37ce` は下限が M の40.8倍で、24件のQ3（配分の費用は68〜266 G RZ）を使った。<br>この間、方式最安は PR-Pauli で不変（227,521,807.86）、PR-DF で0.04%減っただけだった。 |
| 期待する効果 | 同じQ3件数で、最安を下げられない構成に使う分（run02では約38%）を、最安を下げ得る候補に回せる。【人工】人工例で、24件中16件が無駄になっていたのが0件になり、最安がさらに0.8%下がった（人工値）。 |
| 複数幅の WidthDriver が plan_wide に price_floor を渡していない件 | 保証が立つのは、同じ family（方式・L_D・公式・幅）の中で、価格規則が κ に共通で、R について単調非減少（qRの係数 ≥ 0）の場合に限る。Pauli の登録規則（A + Hq + EqR、E > 0）がこれに当たる。PR-DF は候補ごとに較正した規則で、係数が候補ごとに違うため、証明にはならない。保証が無い場合は削除ではなく後回しにする（§1.3）。<br>【保存】run02の未準備の提案241件のうち54件（PR-DF 22、PR-Pauli 32）は、同じ family の小さいscheduleの見積下限がすでに M 以上だった。 |
| shot配分の近傍（§2） | 新しい構成の初回配分には、近いκからの転用（TRANSFER）が有効だった（run02 PR-Pauli で適合15／25件）。<br>最安の構成で減算の近傍が尽きた（SMALL・DEFERREDだけが残る）ら、opt-in の `allocation_exchange`（高単価roundから安いroundへ振り替える）をその構成に限って有効にする。効果は保存ログでは判定できない【要計測】。 |
| 比較の手順（§3） | 自価格ゲートは、選択を取り除くだけの変更なので、旧run の予算の範囲では保存済みの答えだけで比較できる。浮いた予算で選ぶ候補は保存結果が無く、別途承認の短いroundが要る。 |
| 計算本体の高速化（§4） | **Q3**：R08 の job 時間の70%は、round 間に1コアで行う、打ち切った状態の残差の厳密積算だった。最初の案は、この積算を分母ごとにまとめて並列化すること【人工】合成表で 31.3秒 → 13.4秒、出力は完全一致。<br>**準備**：PR-DF の較正は、最も重い round の1 path の中で、X と Y の compile が直列になっている（m6 p0 で 1,104 + 1,152秒）。最初の案は、X と Y を別 job にすること（算術で 3,453秒 → 約2,090秒、未計測）。<br>**prefix cache**：費用の約50倍の時間を省いているが、境界 [5, 11] は浅すぎる。disk も上限なしで 175 GiB。 |

## 1. 新しい構成の探索と、既知構成のshot改善への配分

### 1.1 実経路

| 段階 | 関数（5c8ced4b2） | 決めること |
|---|---|---|
| κ schedule の提案 | `scripts/run_rpe_schedule_search.py` `WidthDriver.plan`（580行）→ `rpe_schedule_search.plan_wide`（180行） | 方式ごとに、全 family（L_D×幅）の物理schedule を提案する。INIT_FLOOR／INIT_FAR（4倍の梯子）、REFINE／EXTEND（family最安の周辺）、EXPLORE_GAP（最大のlog κ間隙）。1回の決定で最大8件（`per_decision`） |
| 候補の準備 | `WidthDriver.complete_points`（624行） | 方式ごとに1 wave あたり8件を準備（信号・価格）に出す。family の順は、既知費用順と未知の交互 → 準備回数で並べ替え（`work_conserving_search` では幅を交互） |
| 配分の選択 | `rpe_priority_search.run_priority`（V2） | UPDATE（費用 < M）、EXPLORE（4回に1回、またはUPDATEが無いとき）、FINE、SMALL、DEFERRED の順 |

【コード】要点：
- `WidthDriver.plan` は `W.plan_wide(scoped, points, self.wide_for(method))` と呼び、`price_floor` を渡さない（`run_rpe_schedule_search.py` 600行付近）。単一幅の旧Driver（120行）は、準備済み点の `price_floor`（全軸1 shotの費用）から下限関数を作って渡す。
- WidthDriver には、旧Driverの `prune()`（準備済みで下限 ≥ M の点を PRUNED にする）に当たる処理が無い。そのため run02 の設定（`max_kappa` = 65536、`kappa_bounds` なし）では、κ の上限は常に RANGE_LIMIT か経験的停止になる。
- V2 の `classify` は、可行点の無い構成の EXPLORE 順位を `(0, その構成の最安の未評価候補の費用)` とする。このため、M より何倍も高い構成でも、新しければ最優先で EXPLORE に入る。

### 1.2 保存証拠（R08 run01 と、その live-ready 継続 run02）

**読んだ範囲**：
- R08：`h4-pr-candidate-isolation-20261117/.../run01`
- 継続：`h4-pr-ready-slot-refill-20261118/.../run02`
- いずれも稼働中・保存済みの run を読取りだけで扱った。run02 は 2026-11-18 18:47 JST 時点の PROGRESS.jsonl と、17:00 の KAPPA_STATE.json を使った。
- 集計script：[evidence/search_evidence.py](evidence/search_evidence.py)。出力：[evidence/EVIDENCE.json](evidence/EVIDENCE.json)。

| | R08 run01 | run02（継続） |
|---|---|---|
| 採用されたQ3（SEARCH_ADOPT） | 204 | 149 |
| 分類 | UPDATE 38、EXPLORE 165、FINE 1 | すべて EXPLORE |
| 方式最安の改善 | 23回。PR-Pauli 410.20 M → 285.01 M、PR-DF 637.23 M → 613.07 M | 1回。PR-DF 480.77 M → 480.60 M。PR-Pauli は 227.52 M のまま |
| 改善を生んだ選択 | UPDATE 20回（最安構成からの減算 7、その連鎖先読み 13）、EXPLORE中の別構成の減算 2回、FINE 1回。すべて既知構成の減算系の動き | EXPLORE中の減算 1回 |
| 自価格の下限 ≥ その時点の M の構成へのQ3 | 0件（下限が分かる構成では） | **57件**（PR-DF 28、PR-Pauli 29） |

- 自価格の下限は、completion の FINAL_QUOTE（V2 が費用計算に使う単価と同じ）から計算した。M は同じイベントに記録された値である。
- run02 の間、PR-Pauli の M は一定で、PR-DF の M の変化も0.04%なので、選択の時点でも下限 ≥ M だった。
- 先読み・投機で起動して採用されなかったjobも含めると、run02 の jobs（18:5x時点、PR-DF 186件・PR-Pauli 128件）のうち、起動時点で自価格の下限 ≥ M だった構成のjobは次のとおり。Q3本体の時間（JOB_STAGES の SCIENTIFIC_ROUTE_START→RETURNED）で数えた。
  - PR-DF：41件、4,684秒（PR-DF本体時間の21%）
  - PR-Pauli：42件、3,637秒（同35%）
- run01 の大半（PR-DF 121件、PR-Pauli 55件）は、構成が過去roundや保存catalog由来で、この run の completion に見積が無いため、下限を確かめられなかった。

**下限 ≥ M の構成へのQ3の例（run02）**：

| 構成 | 方式・L_D | 下限 / M | Q3件数 |
|---|---|---:|---:|
| `92ad37ce` | PR-DF LD3 | 40.8 | 24 |
| `ee7f3c3a` | PR-Pauli LD3 | 1.08 | 7 |
| `6d6ce402` | PR-Pauli LD2 | 3.95 | 7 |
| `ae4059ad` | PR-Pauli LD3 | 4.03 | 7 |
| `046826c2` | PR-Pauli LD5 | 55.9 | 2 |

**run02 の最後の状態**：
- 方式最安の構成の残り候補は SMALL と DEFERRED だけだった。例：PR-Pauli `fa1c573a` は SMALL 7、DEFERRED 6。PR-DF `8e654c0a` は SMALL 6、DEFERRED 6。
- 方式全体の残りは、PR-Pauli が EXPLORE 9・FINE 49・SMALL 388・DEFERRED 253、PR-DF が EXPLORE 7・FINE 20・SMALL 177・DEFERRED 68。
- つまり、最安構成の減算近傍は尽き、予算は新しい構成の EXPLORE に流れていた。

### 1.3 WidthDriver の price_floor：保証できる範囲

`plan_wide` の PROVEN_COST_FLOOR は、family の κ を倍々に上げ、下限 ≥ M になった最初の κ を上限にする。その κ 以上の schedule は提案しない。保証に必要な条件は次の4つである。

1. **1 shot 以上**：どの配分も各roundの両軸に1 shot以上を持つ（減算・交換で `min(v) >= 1` を守っている）【コード】。したがって、構成の費用は自身の単価の和以上になる。
2. **同じ family**：q（幅で決まる）が同じで、κ を上げると全roundの R が成分ごとに増えない方向には動かない（`R_vector` の単調性）【コード】。
3. **価格規則が κ に共通で、R について単調**：単価が a + b·q + c·q·R で、c ≥ 0 の同じ規則であること。
   - 【保存】Pauli：登録規則 PAULI_AUDIT（A + Hq + EqR）と PAULI_C2 は、L_D ごとに共通。ただし幅ごとに自動較正の step group 規則に替わる例があるので、同じ規則の出所かを照合する必要がある。
   - 【保存】PR-DF：FULL_AFFINE_QR は候補ごとに較正される（completion の `df_price_groups/<group>/FROZEN_MODELS.json`）。qR 係数は正（104〜305）だが、同じ LD3 でも候補により 104 と 109 のように違う。ある候補の下限は、別の候補（大きい κ）の見積下限の証明にならない。
4. **M は同じ価格方針（MODEL）での確認済み最安**であること。価格の再較正や、最安点の保留・取消しがあれば計算し直す。

**提案**：
- 条件1〜4がそろう family（Pauli の共通規則）：旧Driverと同じく、`price_floor` を `plan_wide` に渡し、PROVEN_COST_FLOOR で梯子と間隙の探索を止める。
- 条件3が立たない family（PR-DF の候補別較正、規則の出所が違う Pauli）：削除せず、**準備の順番を後回し**にする。`complete_points` の family 順で、未評価の幅・分割と、下限 < M の提案を先にする。保留した提案も残し、ほかが尽きたら準備する。

**保存例（run02 の未準備の提案、COMPLETION 241件）**：同じ family の小さい schedule の見積下限がすでに M 以上だったものは、次のとおり。
- PR-DF 22件：EXPLORE_GAP 11、INIT_FAR 11
- PR-Pauli 32件：EXPLORE_GAP 12、INIT_FAR 20

準備の所要時間【保存】：

| run | PR-DF の準備（READY） | PR-Pauli の準備 | 備考 |
|---|---|---|---|
| R08 run01 | 中央値 4,702秒、最大 7,155秒 | 中央値 42秒 | — |
| run02 | 中央値 139秒 | 中央値 102秒 | PR-DF の WAITING_INPUTS は 3,849秒 |

PR-DF の後回しは、1件あたり最大で1〜2時間の較正を後ろに回す。

**人工例**（[artificial/price_floor_plan_wide.py](artificial/price_floor_plan_wide.py)。出力：[PRICE_FLOOR_PLAN_WIDE.json](artificial/PRICE_FLOOR_PLAN_WIDE.json)）：
- 設定：単価 a + bq + cqR（c > 0）、κ=2 に適合点。`plan_wide` を12回の決定まで回した。
- price_floor なし：提案25件、うち下限 ≥ M が8件。最大κは 409600/9（約45,511）。
- price_floor あり：提案17件、下限 ≥ M は0件。最大κは 1600/9（約178）。

### 1.4 深掘り・別幅や別L_D・κ近傍のどれを優先するか

【保存】：
- 改善の大半は、既知の安い構成の深掘りから来た。R08 run01 では23回すべてが既知構成からの減算・連鎖・FINE で、うち20回が UPDATE。
- 大きな段差は、別の幅から来た。
  - R08 run01 で PR-Pauli が 410.2 M（幅0.1・LD5・κ=2）から 285.0 M に下がった改善は、すべて新しい幅0.2 の構成（LD5 `b9e75255`、LD6 `97dc72be`、いずれもκ=1）の降下だった。
  - その後、幅0.2・LD4・κ=14（`fa1c573a`）の 227.5 M が見つかった（run02 の開始時点で既知だった過去roundの結果）。
  - つまり、新しい幅・L_D を最初に試すこと（EXPLORE の種）と、見つかった安い構成を深掘りすること（UPDATE）の両方が効いた。
- run02 では、最安構成の近傍が尽きた後の EXPLORE がほとんど成果を出さなかった。

**優先の根拠として提案する順**：
1. 自価格の下限 ≥ M の構成と、family の下限 ≥ M の schedule には、Q3 も準備も使わない（証明できる場合）。証明できない場合は最後に回す。
2. UPDATE（費用 < M）を先に試す（現行どおり）。
3. 次に、まだ試していない幅・L_D の最初の schedule を試す（INIT_FLOOR と、下限 < M の INIT_FAR の1段目）。下限と M の比が小さい family を先にする。
4. 安い κ schedule の近傍（REFINE）は、family 最安が M の `window`（1.5倍）以内のときだけ。
5. 最安構成で減算近傍が尽きたら、交換近傍（§2）を試してから次の構成へ移る。

3 と 4 の割合（現行は2:1）を保存ログから最適化する根拠はまだ無い【要計測】。

## 2. shot配分の近傍と、深掘りを切り上げる条件

【保存】採用されたQ3の、出所の種類ごとの判定（適合／不適合）：

| 出所（ENQUEUE origin） | run01 PR-DF | run01 PR-Pauli | run02 PR-DF | run02 PR-Pauli |
|---|---|---|---|---|
| 最安構成からの減算（DECREMENT） | 24／9 | 25／7 | 44／6 | 19／30 |
| 減算の連鎖先読み（CHAIN） | 75／9 | 26／10 | 9／1 | — |
| 近いκからの転用（TRANSFER） | — | — | 2／0 | 15／10 |
| 生成器の種（GENERATOR_SEED） | 3／0 | 4／5 | 2／0 | 2／0 |
| 同じL_Dの保存最安の種 | 1／0 | 1／5 | 4／0 | 2／3 |

比較：

- **単一roundの減算（現行）**：最安構成の降下では主力だった。
  - run02 では、最安構成の減算候補が SMALL（削減 < 0.1%）か DEFERRED（不適合の配分に成分ごとに支配される）だけになり、尽きた。
  - DEFERRED は、MSE が shot について単調だと仮定した後回しで、削除はしていない。ここは指示どおり維持する。
- **近いκからの転用（TRANSFER）**：可行点の無い構成の初回配分として有効だった。run02 PR-Pauli で15／25が適合し、費用も元の構成の最安に近い。
  - 上限ベクトル（CAP）や生成器の種は、構成によっては費用が M の数百倍になる（例：`92ad37ce` の採用費用 68〜266 G RZ）。
  - 転用先では、その構成自身の単価で費用を付け直している（`transfers()` の `O.cost(v, st.c['prices'])`）【コード】。
- **allocation_exchange（opt-in、run02 では無効）**：最安構成の配分から高単価roundのshotを減らし、安いroundに足す。費用が下がる動きだけを作り、MSE は予測しない。
  - 不適合の配分に成分ごとに支配されないので、DEFERRED にならない。
  - 減算近傍が尽きた最安構成に、新しい UPDATE 候補を供給できる。効果（適合率）は保存ログでは判定できない【要計測】。

**提案**：
- 初回配分：可行点の無い構成では、TRANSFER を最初に使う（現行の順位どおり、ただし §1 の下限ゲートの後）。CAP・生成器の種は、TRANSFER が無いときだけ使う。
- 深掘りの切り上げ：構成の UPDATE 候補（費用 < M、非SMALL、非DEFERRED）が尽きたら、その構成で `allocation_exchange` の候補を試す。
  - 交換の対象は、最安構成と、最安が M の `window` 以内の構成に限る。
  - 交換で費用 < M の適合点が出なければ、次の構成に移る。目安は、構成あたり交換を `max_moves` 件試しても適合が0のとき。経験的な規則であり、構成全体の不適合の証明ではない。
- 個別配分の不適合から、構成全体の不適合や MSE の単調性を結論しない（現行の DEFERRED は後回しだけで、この方針と整合する）。

## 3. 改善案の比較方法

**経過時間で比べるときの前提**【保存】：
- run02 の最初の起動（ROUND_RESULT、17:00時点）の executor 計測は次のとおり（両方式で共有）。
  - 20 slot、経過 2,642秒、slot 使用率 27%
  - 空き slot 秒：候補が無い 38,669、選んだ結果を待つ 37,886、候補があるのに空き 39
- つまり、経過時間を決めていたのは、Q3 の計算量ではなく次の2つである。
  - 候補の供給（準備）
  - V2 が答えを1件ずつ順に消費することによる待ち
- 自価格ゲートは、消費順の先頭に M を下げられない job が来ることを無くすので、待ち時間の面でも効く。
- 準備込みの経過時間を縮めるには、2番目の変更（WidthDriver の price_floor による準備の後回し）が効く。

**自価格ゲート（最初の変更）は、保存証拠だけで旧runの予算の範囲を比べられる**：
- ゲートは選択を取り除くだけの変更で、ほかの構成の UPDATE／EXPLORE の順位を変えない。
- 例外：ゲートで外した構成の適合点は、隣の κ の TRANSFER の種にもなるので、後続の選択が変わり得る。これは再生で「保存結果の無い選択」として数える。

**最小の比較手順**：
1. ゲートを opt-in の探索設定（例：`explore_price_floor_gate`）として入れる。既定は無効で、無効時は現行と同じバイト列の出力になることを、既存テストで確かめる。
2. 保存再生：既存の `scripts/replay_rpe_schedule_search.py` の方式（選択の後に保存結果を開示し、`slots` 個の executor の時刻モデルで job ごとの記録時間を使う）を、WidthDriver 版の run02（と R08 run01）に適用する。
   - 現行の再生器は単一幅の PR_PAULI（round01〜04）が対象なので、複数幅の KAPPA_STATE と completion の READY 時刻（DRIVER_PROGRESS の CANDIDATE_INPUTS_END と LIVE_READY_INSTALLED）を読む拡張が要る。これが追加の作業になる。
   - 比較する値：
     - (a) 同じQ3件数での確認済み最安の推移
     - (b) 準備込みの同じ模擬経過時間での確認済み最安。準備は記録された READY 時刻以降にだけ使える。Q3 は記録された本体時間で slot を占有する
     - (c) 保存結果の無い選択の件数（ここで比較の地平が切れる）
     - (d) 下限 ≥ M の構成へのQ3件数（新規則で0件になることが受入条件）
   - 判断に使える観測は、その時点までに完了した結果だけにする。再生の既存方針と同じ。
3. 保存証拠で判定できる範囲：旧runの選択から、下限 ≥ M の構成への選択を除いた列。run02 では 149件中92件が該当し、M の推移は同じになる見込み。浮いた57件分の予算は、保存結果の無い選択（UNDETERMINED）になる。
4. 追加検証が必要な範囲：浮いた予算で選ばれる候補（多くは最安構成の FINE、下限 < M の構成の EXPLORE、交換を有効にした場合はその候補）の Q3。
   - R08 の状態から、ゲートの有無だけを変えた同じ予算（例：方式あたり32件）の短い round を2本、別途の承認で実行し、(a)(b) で比べる。

**人工例**（[artificial/floor_gate_example.py](artificial/floor_gate_example.py)。出力：[FLOOR_GATE_EXAMPLE.json](artificial/FLOOR_GATE_EXAMPLE.json)）：
- 5c8ced4b2 の `run_priority` をそのまま使い、ゲートだけを `classify` に差し込んだ複製と比べた。
- 設定：4構成（最安A、下限がMの2.2倍と3.0倍のB・B2、より安いκのC）、人工MSE、予算24件。

| | ゲートなし | ゲートあり |
|---|---|---|
| B・B2 へのQ3 | 16件 | 0件 |
| 4・8・12・16・20・24件目での最安 | 10,537,376（8件目以降不変） | 10,537,376 → 10,470,880 → 10,453,536 |

人工値であり、H4での改善幅を表さない。

**反例と注意**：
- 最安点が後で保留・取消しになると M は上がる。ゲートは毎回の分類で現在の M と単価から計算するので、その構成は自動で戻る（削除しないため）。
- 価格の再較正で単価が下がれば、下限も下がる。毎回、現在の `c['prices']` から計算する。
- 外した構成の適合点が、隣の安い κ の TRANSFER の種になる場合がある。ゲートはこの種を減らすので、隣の構成の初回適合が遅れる可能性がある【要計測】。
- 下限は採用中の MODEL 価格に対するもので、compile した実回路の費用の保証ではない（比較の基準と同じ価格を使う限り、矛盾は無い）。

## 最初に試す変更の仕様（探索）

| 項目 | 内容 |
|---|---|
| 固定SOURCE | `5c8ced4b2d407a3828d189e1e0f0a24a4807f94a` |
| 関数 | `src/trottertracks/rpe_priority_search.py` の `run_priority` 内 `classify`（272行 `items = local(st);open_ = []` の直後） |
| 変更（opt-in） | `DEFAULTS` に `explore_price_floor_gate=False` を足す。有効時は、`M is not None` かつ `O.cost((1,)*len(prices), prices) >= M` の構成について、その構成の候補をすべて `DEFERRED`（理由 `OWN_PRICE_FLOOR_GE_METHOD_MIN`）に入れ、EXPLORE・FINE・SMALL に出さない。分類は毎回やり直すので、M や単価が変われば自動で戻る |
| 変えないもの | UPDATE の定義、資格・予算・採用規則・STOP、DEFERRED の「捨てない」扱い。分散実行の standby 経路は SMALL・DEFERRED を既に除外しているので、投機もしなくなる。旧来の prefetch 経路（`order()`）は、ほかが尽きたときに FINE・SMALL・DEFERRED も先読みし得る |
| 保存例 | run02（R08 live-ready継続）：採用149件中57件、起動jobでは PR-DF 41件（本体4,684秒）・PR-Pauli 42件（3,637秒）が対象。`92ad37ce`（PR-DF LD3、下限/M = 40.8、24件） |
| 期待する効果 | 同じQ3件数・同じ経過時間で、M を下げられない構成への Q3 が0件になる。浮いた予算は、下限 < M の構成の EXPLORE と FINE に回る。M がどれだけ下がるかは、浮いた予算で選ばれる候補の Q3 次第【要計測】 |
| 反例・副作用 | 外した構成の適合点が、隣の安い κ の TRANSFER の種にならなくなる。最安点の保留・取消しで M が上がる場合は自動で戻る |
| 最小の比較 | §3 の手順。保存再生で (d) が0になることと、残りの選択列が旧runと一致することを確かめ、浮いた予算の効果は承認済みの短いround 2本（ゲート有無）で比べる |

人工例で使った差込みは [artificial/floor_gate_example.py](artificial/floor_gate_example.py) の `load()` にある。実装の正本ではない。

**2番目に試す変更**：`WidthDriver.plan` で、family ごとの `price_floor` を `plan_wide` に渡す（§1.3の条件つき）。準備（特に PR-DF の較正）の量を減らす。

## 4. 計算本体の高速化

### 4.1 Q3 本体（DP）とprefix cache

**調査の分担**：
- 調査はサブエージェントに読取専用で任せた。人工ベンチマークは scratch で、合成の表を使った。
- 作成者が確かめたこと：
  - 保存job `10a3f975`（PR-DF MISS）の round 時刻：伝播 69.0秒、round 間 198.8秒。
  - 保存job `46f29ddd` の prefix HIT（round 5まで再利用）。
  - 5c8ced4b2 の prefix の鍵：構成ごとの namespace と、round ごとの表の連鎖 hash。
  - 9e294778 の `_accumulate`。

**経路**【コード】：
- 通常の worker から次の順に呼ばれる。
  - `run_rpe_optimizer_q3_job.main` → route の `supervise`
  - → 固定 worker（`run_pr_df_fullh_total_q3_fixed.worker`。PR-Pauli も同じ worker）
  - → `rpe_total_risk.evaluate`（kernel 710aa69b）
  - → `total_state_dp`（加速版 `rpe_total_dp_parallel` 9e294778 の AST 置換）
- 3つの route view で kernel file は byte 一致だった。

**時間の内訳**【保存】：

| 区分 | R08 の全job process時間 35,366秒に占める割合 |
|---|---:|
| DP 合計 | 90% |
| └ round 間の処理（打ち切った状態の残差を厳密有理数で積算する `drop()`／`_accumulate`。1コアで実行） | **70%** |
| └ 状態の伝播（fork pool、3 worker） | 18% |
| 閉包・補題・入出力・metadata | 約10% |

- controller と配送の時間は小さい。LAUNCH→JOB_START は 0.05秒、JOB_END→EXIT は 0.04秒。
- 1 job あたり、DP の外に約10〜12秒の固定費がある。metadata と表の作成を adapter と worker で二重に行い、同じ記録を2つの JSON に書いているためである。

**job の種類別（R08、中央値）**：

| 種類 | 件数 | process wall | DP | 伝播 | round 間 |
|---|---:|---:|---:|---:|---:|
| PR-DF MISS | 4 | 283.6秒 | 271.6秒 | 69.5秒 | 198.8秒 |
| PR-DF HIT@5 | 158 | 133.3秒 | 121.7秒 | 26.2秒 | 92.7秒 |
| PR-DF HIT@11 | 26 | 15.6秒 | 3.7秒 | — | — |
| PR-Pauli MISS | 23 | 144.3秒 | 134.2秒 | 26.0秒 | 104.7秒 |
| PR-Pauli HIT@5 | 82 | 111.8秒 | 100.4秒 | 16.8秒 | 81.6秒 |

- PR-DF HIT@5 の job は、3 worker を確保しても平均 1.35 コアしか使っていない。round 間の処理の間、worker は空いている。

**prefix cache の費用対効果**【保存】：
- R08 の HIT 289件・MISS 27件。cache の費用（hash、保存、復元）は計約600秒だった。
- 一方、保存した job の計算時間と照合できた HIT 218件だけで、21,885秒を省いた。照合できなかった HIT も含めた推定では、全体で約29,000秒。費用の約50倍の効果がある。
- 改善の余地は、checkpoint の位置にある。
  - 同じ構成の過去 job と shot 列が先頭から一致する長さは、PR-DF では 9〜12 round が多い。
  - それでも、境界が [5, 11] のため、round 5 までしか再利用できない job が多い。
  - 保存 job の再生推定では、省ける時間は [5, 11] で 31,300秒、[5, 8, 11] で 47,700秒、[5, 9, 11] で 49,100秒だった（再生推定）。
- disk：R08 の共有 cache dir（run02 も同じ dir に書く）は、19:09 時点で 175 GiB（`du`）・約2,360 file あり、上限が無い。
  - R08 で保存した鍵のうち、R08 と run02 で一度でも使われたのは 285 中 32 だった。
  - 使用履歴の LRU 再生では、40 GB の上限で、観測された HIT 540件をすべて保てた。
- 保存形式（JSON の16進）の読書きは、1 job あたり 1〜2.5秒。二進形式にすると size は半分になるが、時間の効果は小さい（人工）。

**最初に試す高速化（Q3）：round 間の残差積算を、分母ごとにまとめて並列化する**
- 固定SOURCE・関数：
  - 加速版 `src/trottertracks/rpe_total_dp_parallel.py`（9e294778266c9c867a8b1b98e040c2e94b1cb635）の `_accumulate`・`_res_value` と、`Accelerator._kernel` の AST 置換。
  - 参照版 `rpe_total_risk.total_state_dp`（710aa69b）は変えない。
- 案：
  - 厳密和を（質量の分母、係数の分母）ごとに分けて持ち、通分は読み出すときだけにする。
  - 打ち切った状態の `drop()` を、既存の worker 数で chunk に分けて fork pool で処理し、親で厳密な部分和を足す。
- 律速：R08 の job 時間の70%。1 round で約24,000状態を打ち切り、状態あたり約0.7 ms かかる。
- 人工ベンチマーク（合成表、実際と同じ規模の質量、3 worker、出力の sha256 は完全一致、1回ずつ）：

| 合成ケース | 元の実装 | 分母ごとにまとめるだけ | まとめる＋並列 |
|---|---:|---:|---:|
| 13 round、frontier 512 | 31.3秒 | 24.9秒 | 13.4秒 |
| 5 round、frontier 2048 | 35.6秒 | — | 15.1秒 |

  - H4 の倍率ではない。
- 副作用：
  - kernel の hash が変わるので、prefix cache は空から始まる（鍵に加速版の hash が入る）。`rpe_dp_prefix_reuse` の固定 hash の見直しも要る。
  - 保存の前に、分母ごとの値を `Mass` にまとめ直す必要がある。そうしないと `_encode` が失敗し、保存が `WRITE_FAILED` になる。
  - route の固定版が変わると、過去の完了 Q3 の再利用に影響し得る。採否は利用者の判断になる。
- 2番目の案：kernel を変えずにできる checkpoint 方針の変更。
  - `prefix_checkpoint_rounds` を [5, 8, 11] か [5, 9, 11] にする。job の identity を作った後に足す設定なので、identity は変わらない見込み【コード、未試験】。
  - あわせて、cache dir に約40 GB の LRU 上限を付ける。
- 最小ベンチマーク：
  - 段階A（人工、承認不要）
    - 対象と方法：9e294778／710aa69b と候補版を、合成の2形状で、交互に3回ずつ比べる。Python 3.11.0rc1 の環境を使う（3.12 は既存の AST guard が拒否する）。
    - 計測：`/usr/bin/time` の wall・user・sys・最大 RSS と、子プロセスの CPU。
    - 回帰：既存の加速版テスト43件と prefix テスト10件。
  - 段階B（実 H4。新しい科学実行なので別途承認が要る）
    - 対象：保存 request `10a3f975`（PR-DF MISS）、`46f29ddd`（PR-DF HIT@5）、`a320a1b9`（PR-Pauli MISS）。
    - 方法：通常の adapter から、空の専用 cache で、元の実装と候補版を同じ host で続けて実行する。
    - 合格条件（いずれも保存 OUTCOME と照合）：
      - DP 出力の dict が完全一致
      - MSE 上下界の分子・分母が完全一致、判定が同じ
      - `PARTIAL_TOTAL_DP.json`（round ごとの残差確率、打ち切り状態数、遷移対の数）が出所欄を除いて一致
      - 質量保存の gate が PASS

### 4.2 候補準備：PR-DF の価格較正

調査はサブエージェントに読取専用で任せた。人工ベンチマークは scratch で6回（各30秒以下）。下に載せる数値は、作成者が保存ファイルで確かめたもの（`m6_p0_STAGES.jsonl` の各段階時間、`FINAL_QUOTE.json` の判定）。

**律速**【保存】：
- PR-DF の準備時間は、ほぼ価格較正（DF_CALIBRATION）である。
  - R08 run01：DF 準備レーン 26,816秒のうち 25,915秒（96.6%）。
  - run02（R08継続）の m6 較正（候補 `dd80df10`、LD1、κ=488/5、較正round [0, 3, 6]）：3,849秒のうち 3,458秒。
- 較正では12本の path を同時に走らせる。時間は最も重い round の1本で決まり、各 path は1スレッドで動く（m6 p0：CPU 3,447.6秒、wall 3,453.4秒）。

**m6 p0 の内訳**（`.../dd80df103b521892b7b4/df_price_groups/aa4dc3085aaea519ccad/df_prices/calibration/acquisition/m6_p0_STAGES.jsonl`）：
- 合計 3,453秒、peak RSS 9.65 GiB。

| 段階 | 秒 |
|---|---:|
| sampling・event処理 | 79 |
| structural（RZ構造数） | 278 |
| compose | 326 |
| X軸：wrapper 91 → fingerprint 84 → **compile 1,104** → mutation check 85 | 1,364 |
| Y軸：wrapper 91 → fingerprint 81 → **compile 1,152** → mutation check 82 | 1,406 |

- X と Y は、1プロセスの中で順に処理される。compile の中の Qiskit `transpile` と、その後の6指標の計算は、保存ログでは分けられない【要計測】。
- R08 run01 の m11 較正（q = 2048〜4096）では、compose の比率が上がる（28〜30%）。compile は48〜52%。

**既存の高速化は実経路で効いていたか**【コード・保存】：
- `compiled_price_fast`（e827367）は、実際の較正経路で有効だった。STAGES の段階名と `TECHNICAL_TIMING.json` は、この版だけが出す。
- ただし、これが速くするのは構造数と fingerprint の書き出しだけで、compose・wrapper・compile は対象外である（docstring どおり）。
- 準備済み context の共有も有効だった。PREPARATION.json の `preparation_hash` は全 m で同一。path 開始前の親の処理は 2.4〜3.5秒しかなく、律速ではない。
- X/Y を並列に compile する既存手法 `ParallelAxesCompiler`（`docs/PERFORMANCE_METHODS.md`、`codex/d-full-h-p2-caller-prep-20261114`）は Full-H／D7 経路だけに入っていて、PR-DF には入っていない。

**価格が出ない較正**【保存】：
- `dd80` は、3,849秒の較正の後に WAITING_INPUTS で終わった。FINAL_QUOTE は26軸すべて NEEDS_PRICE_EVIDENCE（理由 `MULTIPLE_SCOPED_MODELS_CHOOSE_SOURCE_NO_POOLING`）。
- R08 run01 の同種の候補 `84130a07`（3,840秒）も同じだった。
- この間、PR-DF の準備レーンは1件ずつ順に処理するので、後続の6候補（各30〜144秒）は待たされた。
- どのモデルを価格の出所に選ぶかは、価格方針の判断である。B・統合担当で決める必要がある。決まるまで、同じ理由で止まる較正を新たに始めないようにすると、1時間単位の準備を節約できる。

**最初に試す高速化（準備）：較正 path の X と Y を別の job にする**
- 固定SOURCE・関数：`scripts/run_pr_df_candidate_prices.py` の `_acquire`（5c8ced4b2 で481〜549行）で、`pending` に (cell, index, ['X']) と (cell, index, ['Y']) を別々に積む。
  - 単一軸の要求は、`selected_request`／`selected_acquire_path` がすでに受け付ける。
  - 軸別の receipt を読み、X/Y の event digest が食い違えば保留する `scan_path_axes` もある。
  - STAGES の記録名にも軸の接尾辞を付ける。今は directory だけに付くので、そのままでは同じ file に書き込む。
- 短縮する区間：X と Y の直列（m6 p0 では 1,364 + 1,406秒）。
- 期待値（保存時間からの算術。未計測）：m6 path は 3,453秒から約 2,090秒になる。
- 費用：
  - 共通の前処理（683秒）を両方の job で計算するので、CPU が増える。
  - event file が2つになる（m6 で 986 MB）。
  - 同時に動く重いプロセスが倍になるので、メモリの総量を事前に確かめる【要計測】。
- 人工ベンチマーク（9 qubit、合成40,000 event、2回ずつ）：

| | serial | 分割 job |
|---|---|---|
| wall | 27.1 / 28.2秒 | 14.9 / 15.2秒 |
| CPU | 32.6 / 31.8秒 | 33.5 / 35.2秒 |

  - 全回で、X/Y の fingerprint・6指標・mutation check が一致した。
  - これは H4 の倍率ではない。
- 2番目の案：`structural` と `compose` を重ねる（m6 で最大278秒）。固定版の `compiled_price_fast` を差し替える必要があり、手間が大きい。
- 最小ベンチマーク（H4 回路の compile を伴うので、実行には別途承認が要る）：
  - 方法：`dd80` m3 p0 の保存入力（約59秒の path）を scratch で再実行する。serial と分割 job を交互に3回ずつ。
  - 計測：wall、子を含む user+sys CPU、最大 RSS と子の RSS の合計。
  - 合格条件：次がすべて保存値と一致すること。
    - wrapper の fingerprint（m3 p0：X `245aa09b…`、Y `6791eac2…`）
    - event digest と S（`53cccf4b…`、S = 123,988）
    - 6指標と rz_count（両軸 128,252）
    - compiler hash `1239304c…`
    - `new_compiles == len(compiles)`

### 4.3 サブエージェントの作業記録

- 集計・人工ベンチマークの script は、作成者側の scratch に置いた。再現に必要な要点と数値は本文に記載した。
- 人工ベンチマークはいずれも小規模（各30秒前後）で、H4 の入力・回路・Q3 は使っていない。
- 未確認のまま残した点：
  - 実 H4 での速度倍率
  - process tree の真の最大 RSS（2秒の標本化では短い山を逃す）
  - 取り込んだ完了 Q3 のうち、探索で実際に使われた件数
  - run02 で較正以外の段階が R08 より2〜3倍遅かった理由

## 範囲と保全

- 読取レビューと人工例だけ。稼働中の run、採用規則、共有環境は変更していない。保存runのファイルは読取りだけで、prefix_cache の大きなファイルは開いていない。
- 新規の実H4計算・Q3・信号・価格較正・compile は0件。人工例は 5c8ced4b2 の関数を呼ぶだけで、科学結果ではない。
- mandatory STOP。
