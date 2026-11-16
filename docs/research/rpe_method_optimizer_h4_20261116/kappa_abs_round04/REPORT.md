# H4 PR-Pauli：Cの共通較正入口と、κの物理schedule区間による探索（round04）

2026-11-16 JST、B2（Claude Code）。確認はすべて作成者自身によるもので、独立レビューではない。

対応した指示：comment(B1114) の追記「PR-PauliのΔ_PF・κ探索に、C実装・A検証済みの共通入口を接続」（固定commit `02e7cc6a127954fa7c446e7383d037f0617d50d3`、HANDOFF `docs/research/pauli_multistep_calibration_request_20261116/HANDOFF.md`）。

## 1. 結論

| 区分 | 構成 | 最安適合費用（RZ） | 配分 N_X=N_Y | MSE上界/τ |
|---|---|---:|---|---:|
| κ=2の最安（規則が採択） | LD5 κ=2（`ff6b21ae`） | 501,233,190.43 | 48,45,42,39,36,32,14,21,17,14,7,7,5 | 0.999979 |
| κ=2から上側で最初に異なるschedule | LD5 κ=7（`a620a355`） | 518,960,285.57 | 48,45,42,39,36,33,16,20,17,16,10,12,2 | 0.999993 |
| PR-Pauli全体の保存最安（変化なし） | LD5 κ≈48.61（`cb1aa748`） | 410,199,237.00 | 48,45,42,39,22,27,14,20,16,16,8,7,2 | 0.999912 |

- **LD5**
  - κ=7は、κ=2の1.035倍で、κ=2より高い。
  - 範囲[1/2, 8]の中にあるLD5の物理scheduleは、次の2つだけである。両方を評価した。
    - κ≤6.4401…：全roundでR=1。
    - 6.4401…<κ≤12.8803…：m12だけR=2。
  - 下側には、異なるscheduleがない（全roundがすでに下限R=J=1にある）。
- **LD1・2・3**：上下それぞれの最初の異なるschedule、計6点はすべて不適合だった。
  - shot上限いっぱいの配分（48,45,…,13,9）でも、MSE上界/τは次のとおりで、1を大きく上回る。
    - LD1：198（531/266）、150（683/341）
    - LD2：58.2（両側）
    - LD3：18.7（43/22）、18.5（33/16）
  - κ=2での値は、LD1 198、LD2 58.2、LD3 18.6で、ほとんど変わらない。
- **停止**：2026-11-16 21:37:56 JST、`NEIGHBOURHOOD_RULE_EXHAUSTED_NO_PROMISING_CANDIDATE`。
  - V2の停止理由は `ONLY_SMALL_OR_DEFERRED_LEFT`。
  - 系列ごとの判断は、LD5が `STOP_FAMILY_RULE_EXHAUSTED`、LD1・2・3が `STOP_FAMILY_NO_FEASIBLE_POINT`、LD4・6・7・8・9が `STOP_FAMILY_INPUTS_HELD`。
- **他方式との比較**（変化なし）
  - Full PF 559,457,386に対して：κ=2は0.896倍、κ=7は0.928倍、PR-Pauli全体の最安は0.733倍。
  - p7 372,035,410に対して：PR-Pauli全体の最安は1.103倍で、p7を下回っていない。
- **先読みで得たが、規則が採択していない適合点**（§5参照）
  - LD5 κ=2：500,962,225.57 RZ（MSE上界/τ 0.999986）。規則上の最安より270,964.86（0.054%）安い。
  - 小改善の閾値（0.1%）未満のため、V2は採択せずに停止した。結論の表には、規則が採択した値を載せている。

## 2. 指示の各項目との対応

### 2.1 L_Dと要求幅 → mapperによるq_mと実幅（指示1）

- **探索本体（Δ_PF=h）**：mapperの出力は、q_m=2^m、実幅T_m/q_m=h（binary64の0.1）。
- **要求幅の受け渡し**：`rpe_candidate_preparation.kappa_item` が、Δ_PFをJSONの表現のままAの入口へ渡す。
  - 文字列 `"1/5"` は厳密な1/5として、数値 `0.2` はbinary64の値として扱われる。Aの `rational()` は、どちらも丸めない。
- **Δ_PF≠hの幅**：新規の科学計算は採択されていないため、メタデータだけを作った（§3）。

### 2.2 κ=2を起点とした、R_mが変わる区間（指示2）

- **区間の計算**（`src/trottertracks/rpe_kappa_schedule.py`）
  - mapperの式 R_m = max(J, ceil(ceil(κ·c_m)/q_m))（c_m=λ_R²T_m²、厳密な有理数）で計算する。
  - R_mはκについて非減少の階段関数なので、13 roundのscheduleが変わるκは、厳密に求められる。
  - 生成済みの18構成（幅が異なるものを含む）で、mapperのR_mと全一致した。
- **κ=2からの最初の異なるschedule**（範囲[1/2, 8]。代表点は、区間内で分母が最小の有理数）

| L_D | 下側：代表κ（区間） | 変わるround | 上側：代表κ（区間） | 変わるround |
|---|---|---|---|---|
| 5 | なし | — | 7（6.4401…, 12.8803…] | m12（1→2） |
| 3 | 43/22 | m11・m12 | 33/16（2.0276…, 2.0645…] | m12 |
| 2 | 175/88 | m12 | 343/171 | m11・m12 |
| 1 | 531/266 | m12 | 683/341 | m5〜m12 |

- **同じscheduleのκのまとめ方**
  - 近傍規則は、同じR vectorの点を1点として扱う（`kappa_neighbourhood_plan(points, nb, sched)`）。
  - 例：LD5のκ=4/3・2・3は、1点として扱った。保存評価は、ALIASとして共有している。
  - 規則が提案したκのscheduleが評価済みなら、その方向で次に異なるscheduleへ移す。κの刻み幅のせいで、異なるscheduleを飛ばすことはない。
  - schedule情報を渡さないときは、旧規則と同じ提案になる（回帰検査）。
- **既評価の再利用**：同じjob keyのQ3は、round01〜03の4,499件を読み込んで再利用した。

### 2.3 Cの共通入口・価格と信号の再利用（指示3）

- **共通入口への渡し方**：新しい点は、1cycleにつき1回のbatchで、`prepare_pauli_calibration.py --batch`（view `b2-route-view-a-02e7cc6a1`、h6 20261011 venv）へ渡す。
  - 渡す中身は、canonical構成と、D2の生の `PR_PRICE_APPLICABILITY_V1`（MODEL、B2が拡張した登録表）。
  - 結果の扱い
    - `REUSE_SUPPLIED_PRICES_NO_CALIBRATION`：そのまま次へ進む。
    - それ以外：点を保留にし、roundごとの処置・較正cell・不足を記録する。
  - 較正cellは、入口が候補の間で1件にまとめる。
- **round04の7点**：すべて `REUSE_SUPPLIED_PRICES_NO_CALIBRATION`（26軸を再利用、較正cellは0件）。
  - 記録：`run01/KAPPA_STATE.json` の `price_entry`、`run01/price_batch/cycle02/BATCH.json`。
- **価格・信号・Q3の再利用の判定**：それぞれ別の条件で判定している。

| 対象 | 判定の単位・条件 | round04の結果 |
|---|---|---|
| 価格 | D2のMODEL規則で、round単位 | 7点とも26軸を再利用、較正0件 |
| 信号 | Aの信号入口の計画で、roundの物理key単位 | 新規の平均値16 round（R_mが変わったroundだけ）、再利用75 round |
| Q3 | 構成と配分のjob keyで、経路のidentityを照合 | 4,499件を再利用、新規284件 |

- **信号の内訳**：LD1 683/341はm5〜m12の8 round。LD3 43/22とLD2 343/171は、m11・m12の2 round。他の4点は、m12の1 roundだけを新規に取得した。

### 2.4 scheduleごとのshot配分と比較（指示4）

- **配分の最適化**：入力が揃った各scheduleで、V2がshot配分を最適化した。同じ精度閾値τ（MSE上界 ≤ τ）を満たす条件付きの総RZ費用で比較した。
- **不足のある候補**
  - round04の7点に、価格・信号の不足はなかった。
  - 幅Δ_PF≠hの候補の不足（対象round・実幅・較正cell・不足SOURCE）は、§3に記録した。
- **新規の科学計算の範囲**：採択済みの範囲（S2、K=2、Δ_PF=h、κ∈[1/2, 8]、L_D 1〜9）だけで行った。

## 3. 要求幅Δ_PF≠h（メタデータのみ）

- **方法**：`scripts/prepare_pauli_width_requests.py` で、LD1・2・3・5のκ=2を処理した。
  - Aの入口（mapper）→ D2のMODEL見積り → Cの共通入口のbatch、の順。
  - 信号・path・compile・fit・Q3は0件。
- **記録**：`width_requests/WIDTH_REQUESTS.json`、`BATCH.json`、`BATCH.CALIBRATION_0*.json`。

| 要求幅（JSONの表現） | q_m（m0〜m12） | 実幅 | 共通入口の判定 |
|---|---|---|---|
| `"1/5"`（厳密な1/5） | 1,2,3,5,9,17,…,2049（m≥1で2^(m-1)+1） | m0・m1は0.1（binary64）。m2〜m12はbinary64で表せない（例：m12は3602879701896397/18023194602504192） | `NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION`。m0・m1は価格を再利用。m2〜m12は `EXACT_STEP_OR_T_NOT_BINARY64_DRIVER_REPRESENTABLE` |
| `0.2`（binary64、=2×binary64(0.1)） | 1,1,2,4,…,2048 | m0は0.1、m1〜m12はbinary64の0.2 | `CALIBRATION_REQUEST_READY_NOT_EXECUTED`。m0は価格を再利用し、m1〜m12は較正cellに入る |

- **binary64 0.2の較正cell**：L_Dごとに1件で、計4件。実行はしていない。
  - 4件とも、q=8、T=1.6、K=2、実幅は0.2（binary64）。
  - 同じL_Dのm1〜m12は、1つの較正cellにまとまる。cellのRは、その幅のroundのうちR_mの最大値。
  - 入口が示した上限は、新規path 8本、compile 16回。

| L_D | cell |
|---|---|
| 1 | Q8R1792 |
| 2 | Q8R692 |
| 3 | Q8R109 |
| 5 | Q8R1 |

- **各幅でのκの区間**（κ=2を起点、範囲[1/2, 8]）

| L_D | 幅 | 最初の異なるschedule（上側）：代表κ・変わるround | 下側 |
|---|---|---|---|
| 5 | 厳密1/5 | 4、m12 | なし |
| 5 | 0.2 | 4、m12 | なし |
| 3 | 厳密1/5 | 71/35、m12 | 275/138 |
| 3 | 0.2 | 75/37、m12 | 73/37 |

  - その他（LD1・2）の区間は、JSONに記録した。

## 4. 実装と検証

- **変更**（`24b51fbc3`、`2057650a3`）
  - `src/trottertracks/rpe_kappa_schedule.py`（新規）：区間の計算。
  - `src/trottertracks/rpe_priority_search.py`：schedule単位の近傍規則。
  - `src/trottertracks/rpe_candidate_preparation.py` と `rpe_method_optimizer.py`：要求幅を表現のまま渡す。
  - `scripts/run_rpe_kappa_neighbourhood.py`：R vector、系列の定数、共通入口での価格判定。
  - `scripts/run_rpe_optimizer_price_quotes.py`：`--raw` を追加。
  - `scripts/prepare_pauli_width_requests.py`、`scripts/check_rpe_kappa_schedule.py`：新規。
  - round file：`configs/rpe_method_optimizer/rounds/h4_pr_pauli_kappa_schedule_round04.json`。
- **検査**：`checks/KAPPA_SCHEDULE_CHECKS.json` で7/7 PASS。
  - mapperの再現：18構成。
  - 区間：境界の前後が一致すること。代表点との間に、第3のscheduleがないこと（有理数400点）。
  - 計画：最初の異なるschedule、移動、旧規則との一致。
  - 幅：文字列とfloatが別のq_mになること。
  - 価格段：実際の共通入口で、Δ=hは再利用され、0.2は保留とcell記録になること。
- **既存の受入**：優先探索6、実行器21、準備入口8、生成11、いずれもPASS。

## 5. 実行の記録と速度

- **時系列**

| 段階 | 時刻（JST） | 内容 |
|---|---|---|
| 起動 | 20:19:34 | — |
| cycle 1 | 230秒 | 引き継いだ状態でV2を実行。新規Q3は0件。7点を提案 |
| cycle 2の準備 | 37秒 | 価格batchを含む |
| cycle 2の信号 | 107秒 | 7点の並列取得、取込み、経路登録 |
| cycle 2のV2 | 4,328秒 | — |
| 停止 | 21:37:56 | — |

- **Q3**
  - 新規284件（平均の実行時間145秒）。同時実行は平均10.1件で、所要68分。
  - 規則が採択したのは82件。連鎖の先読みで投じて採択されなかったのは202件（71%）。
  - 構成別では、LD5 κ=7が230件で、適合と不適合が交互に出た。
  - 採択82件を逐次に流すと約198分かかる。実際は72分で、約2.7倍速かった。代わりに、Q3の件数は3.5倍になった。
- **採択されていない先読みの適合点**：いずれも小改善（<0.1%）の候補で、規則は採択前に停止した。結果は `run01/EVALUATIONS.csv` にある。

| 構成 | 先読みの適合点（RZ） | 規則が採択した最安（RZ） |
|---|---:|---:|
| LD5 κ=2 | 500,962,225.57 | 501,233,190.43 |
| `4d6e2c9c`（LD5 base κ≈24.3） | 437,758,513.00 | 437,894,306.43 |
| `39355bcf` | 441,233,001.29 | 441,301,209.00 |

- **生データ**：job一式はサーバー上の `run01/jobs/` だけにある（4.2GB）。次の2件は `best_points/` にコピーした。
  - κ=7の最安：`32ecd1ff57d0b2e6a8c6`
  - κ=2の先読み適合点：`9d10cc9f815edeeefdd7`
- **信号branch**：`codex/b2-pauli-kappa-signals-20261116` の `fa9c4cf17`（push済みで、remoteと一致）。構成は `26c1c4162`、平均値は `21c7c969f`。

## 6. 途中の問題

- **監視の不具合**：作成者が使った監視（Monitor）が、30分間通知を出さなかった。pipe内の `cut` が行単位にflushしないことが原因。計算には影響していない。
- **それ以外**：driver・入口・Q3に失敗はなかった。

## 7. 残課題・判断待ち

- **LD2・LD3の大きいκ**：採択済みの近傍規則は、適合点のない系列を延長しない。
  - shot上限での最小MSE下界/τは、κとともに下がっている。
    - LD3：κ=4/3で54.5、2で18.6、3で1.95。
    - LD2：κ=3で3.29。
  - したがって、κ∈(3, 8]には適合するscheduleがありうる。
  - これらを評価するかどうかは、研究上の判断になる。規則を変える場合は、新しいround fileで行う。
- **Δ_PF≠h**
  - 厳密1/5：binary64で表せない実幅の扱い（driverの規則か実装の拡張）が必要。
  - binary64 0.2：4件の較正cellの実行には、別途承認が必要。
- **引継ぎの課題**
  - LD4・6：Bの取込みの版を更新する必要がある。
  - LD7・8・9：P0の数値gateで停止している。
  - LD1の価格：D2本体の登録表へまだ取り込んでいない。
- **先読みの効率**：適合と不適合が交互に出る下降では、連鎖の先読みが外れやすい。両方の結果を見越して投入する方式なら、改善の余地がある（今回は変更していない）。
