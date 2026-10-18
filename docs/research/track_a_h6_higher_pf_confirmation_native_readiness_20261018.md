# H6高次PF：最小3候補の独立確認・native実行準備（2026-10-18）

**技術準備完了、実計算は未承認。mandatory STOP。** 最小3候補は利用者が選択済み。名目power90%の13/45/45本は第一候補として具体化した未採択案であり、標本数の正式採択・sampling/compileの実行承認ではない。

正本は統合比較commit `1fc58aa3f5568887e0678a9be939551b0ed6523d` と前段の設計commit `87e6ead0c7b0ab77a7a0fbb420c517acc82ef6e1`。今回は実H6のsignal・費用・順位・確率を再計算していない。

## 1. 固定する候補と目的

| 候補 | q | 提案する新独立trajectory数 | cosine/sine compile数 |
|---|---:|---:|---:|
| H6 S2 | 8 | 13 | 26 |
| H6 Yoshida4 | 2 | 45 | 90 |
| H6 Yoshida4 | 4 | 45 | 90 |
| 合計 | | 103 | 206 |

対象はH6直鎖・隣接1 Å・STO-3G・保存DF rank19・prefix5・保存入力状態、T=.8、K4、outer当たり32 short steps。signed時間のabs-largest-remainder配分、generation-order tie、ordinary control、explicit XXPlusYY/number Phase loweringを維持する。ε=.001、各軸ε/√2、α_axis=.025、経験的allowance1e-9、log_B margin1e-12、状態準備P=0。保存したbias・normalization・軸別shot数を結合し、shotは再計算しない。

目的Aは、探索n8の点順位が新しい独立標本でも再現するかの診断である。固定比較は以下の2つ。

1. mean(S2/q8) − mean(Yoshida4/q2)。探索点差は約3.29%。
2. mean(Yoshida4/q4) − mean(Yoshida4/q2)。探索点差は約0.97%。

目的Bである、希少eventを含む母平均費用順位の統計的保証は保留。名目90%は保存n8の分散と点差を使った設計上の近似であり、実際のpowerや母平均順位を保証しない。H4やnew_2、他qへの候補追加はしない。新A_COST候補の独立確認とは別系列で、seed・grant・ledger・結果を混合しない。

## 2. SOURCE・再利用と入力固定

専用SOURCE：`e3ac6fd8494a612d7a1aa1061fa79b3fc50d278c`。

| 要素 | 実装・結合 |
|---|---|
| 専用adapter | `scripts/h46_confirmation_native.py` |
| 認可・候補・seed検査 | `scripts/h46_confirmation_contract.py` |
| one-shot launcher | `scripts/launch_h46_confirmation.py` |
| 確認標本専用集計 | `scripts/h46_confirmation_collect.py` |
| 無効契約の準備CLI | `scripts/prepare_h46_confirmation_native.py` |
| 人工テスト | `tests/test_h46_confirmation_native.py` |
| 費用集計の再利用 | `src/trotterlib/higher_pf_readiness.py` のpaired整数会計 |

検証済み `h46_acquire_native.py` の科学部分はbyte単位で不変。SHA256は `ccc71379739c243e86024e897643220ae67fa0b3409d8d8be4d8193183228308`。専用adapterは、worker内だけでtask登録をsample0〜44へ差し替える。preparation・sampler・native lowering・compile・controlは変更しない。旧kernelが出すraw receiptは新run内でもそのまま保持し、別の `confirmation.json` にCONFIRMATION stage・SOURCE・plan・raw SHAを結合する。旧探索receiptを確認標本に読み替えない。

provider SOURCEは `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。H6 snapshot SHA256は `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`。保存preparation hashは `8f1a392f8ab2bcb6cf59cad513b96f15ff518cb6dc237233410b03bb14567938`。H6 snapshot/DF receipt、scalar、λ、q固有log_B・shot、統合結果を結合した。

Python3.12.3、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1を固定。basis rz/sx/x/cx、optimization_level1、seed_transpiler17、backend/couplingなし。compiler profile hashは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

SOURCE・provider・選定runtime・入力・seedを合わせたprivate closure306 fileを検査した。これは実使用範囲の封印であり、環境全体の全fileを封印したという主張ではない。入力・compiler/preparationの条件は緩和していない。

## 3. seedとone-shotの契約

専用namespaceは `H6_HIGHER_PF_C_CONFIRMATION:{SOURCE}:{seed_scope_digest}`。SOURCE、3候補、提案n、入力hash、選定証拠、CONFIRMATION stageを含むseed-scope digestと、grantが結合する最終plan file SHAを分け、registry hashとplan hashの循環を避ける。

暫定registryには914個のSHA派生uint64 seedを登録した。旧nativeの成功・失敗namespaceと固定reference由来1,505個、B_ACQUIREの1,824個、計3,329個を除外し、重複・衝突は0。対象は既知の固定系列と参照可能なseed metadataであり、未知の全利用者jobを網羅したという主張ではない。異なるseedはPRNGの数学的独立性の証明ではない。

これは準備用のseed整数metadataで、eventを抽選していない。nは未採択、有効grant0。公開planのstatusは `DRAFT_UNAPPROVED`、`sample_size_formally_adopted=false`。無効grantを実CLIに渡すと、SOURCE/入力読込・native import・output/ledger/marker作成より前にexit2で拒否された。

承認後も、採択n・SOURCE・最終plan SHA・approval reference・専用実行pathを結合した新しい有効grantが必要。元draft/無効grantを上書きしない。103本・206軸・追加compile0の登録と非払い戻し予約を守り、消費済みmarkerの再利用は拒否する。nを変更するなら新契約として準備し直す。

## 4. 並列化と起動前gate

候補ごとのsample0を最初に並列実行し、その候補のcontrol PASS後に残りをtrajectory単位で投入する。cosine/sineは同一eventを共有し、同じtask内で順に処理する。各workerは保存preparation/oracleをcacheする。transpilerの内側poolとBLASは1 threadとし、trajectory poolで並列化する。共有環境や既存venvは変更しない。

worker数はfresh affinity・CPU idle・利用可能RAM・保存RSS proxyから起動時に選ぶ。過去の4/32/60 workerなどを恒常的上限にしない。RAM不足なら新taskの投入を待ち、実際に1本も受け入れられない場合はSTOPする。proxyはメモリ上界保証ではない。前段の32 workerで約55分という試算は旧elapsed時間を用いた計画値であり、実行時間の保証ではない。

起動前と完了後にSOURCE/入力/runtime/seed closureを検査する。fresh gateで、実行Python/package profile、未使用の専用output/ledger/marker、seed除外・固定予算を確認する。別worktreeのHEAD/statusや上位PSIは観測であり、通常の競合だけでworkerを停止しない。他jobへsignal・環境変更・lockを行わない。

## 5. control・判定・STOP

各trajectoryでscalar・λ・保存preparation・生成順・normalizationを検査。各候補sample0ではoccupation oracle、control low/high/leakage、Hadamard両軸、compiled statevectorも検査し、許容差1e-9を維持する。予定536 check、実compile206軸に含まれるcontrolであり、追加compile枠はない。

技術PASSには103本/206軸の完全性、両軸の同一event/seed、独立確認receipt、非払い戻しledger、全control、SOURCE/profile、worker正常exitと回収が必要。入力改変・非finite・compiler/control不一致・自分のprocess異常・実際の継続不能な資源不足でSTOP。自動retry・再fit・追加compile・追加sampling・n増加をしない。過去のSTOPをPASSへ変更しない。

費用は各trajectoryで `N_real*RZ_cos + N_imag*RZ_sin` を整数で計算する。CXも併記し、軸単価・保存bias/normalization/shotと対応させる。主判定は固定2比較の有理数平均差の符号で、正なら `REPRODUCED`、0なら `TIE`、負なら `NOT_REPRODUCED`。期待した順位にならないこと自体は科学的な結果であり、技術失敗と混同しない。結果にかかわらず終了後STOP。

補助診断は候補間独立を仮定した `sqrt(s_left²/n_left+s_right²/n_right)` と、family2/α_MC=.05の両側Bonferroni正規近似区間。α_MCとshotのα_axisは別である。探索n8は混ぜず、候補間CRN共分散を仮定しない。近似区間を厳密CI、population winner保証、選択後coverage保証と呼ばない。標本分散0でも母分散0とは扱わない。新たなq選択・全格子最適性認定はしない。

## 6. 希少eventの静的照合

前段の保存n8は、3候補全てs4とs2_multiが未観測。Yoshida4/q2の理想独立step法則上のs4確率約1.74e-5は設計metadataとして再利用した。今回は確率を新たに計算していない。

既存builder・signed schedule・人工の0/2/4次/正負時間controlはコード検証に再利用できる。未観測の分子trajectory費用や母分散上界を与える証拠にはならない。他の条件付きpilotも、law・preparation・control/profileの完全なjoinがないものは今回の無条件標本へ混合しない。未知費用はnullで、0補完・恣意的補完をしない。目的Bは今回の確認結果だけでは解決しない。

## 7. 人工検証・成果物・次担当

新17件と既存回帰を合わせ、固定Python3.12.3で92件PASS。人工のspawn poolで103 task/206予約、初回control失敗STOPと全回収、完全な認可経路と改変拒否、seed再導出・旧seed衝突、CONFIRMATION receipt、536 check/整数集計、tie/逆転/標本追加拒否を検査した。新17件はPython3.11.0rc1でもPASS。広い回帰の初回は固定providerのimport設定が不足し、private test harnessのspawn main guardも修正した。最終clean実行は92件PASS。これは作者自身の検証であり、別担当者による独立レビューではない。

証拠は `artifacts/resource_applicability/track_a_h6_higher_pf_confirmation_native_readiness/2026-10-18/` のplan、無効grant、seed audit、source/input closure、rare reuse、人工testcase台帳、draft refusal、保全監査、summary/inventoryに保存した。NPZ・event列・seed値一覧・実行用private config・有効grant・ledgerは公開しない。旧SOURCE・入力・結果・grant・ledger・他worktreeに書き込んでいない。

- ユーザー：3候補の13/45/45本案を正式採択し、固定206軸の新sampling/compileと登録2比較の集計を一回実行するか判断する。現時点では未承認。
- Codex：明示承認後のみ、正確なSOURCE checkoutから新しい最終plan/grantを作り、fresh gate、one-shot実行、結果保存・公開・mandatory STOPを行う。目的BやA_COSTへ自動移行しない。
