# 研究C M3：JSON catalog修正と同条件fresh run02の準備

2026-10-28 JST。ユーザーは前回の技術STOPに対して「修正して再実行してよい、意図しない中断も修正・再実行してよい」と明示許可した。これを同一M3の技術修正・fresh再実行の許可として使う。科学条件や数値閾値の変更、不適格候補の自動探索はしない。承認証明・原文保存・新grantは不要。

起点：`def3748fc33f923f1a684a5d3ee0608fd7818a44`。独立branch `codex/research-c-M3-s1-r64-repair-20261028`、worktree `/tmp/prt-c-M3-s1-r64-repair-20261028`。旧[run01報告](research_c_pr_aware_df_M3_s1_r64_execution_20261028.md)・[契約](research_c_pr_aware_df_M3_s1_r64_contract_20261028.json)・SOURCE manifest・raw8・STOPはそのまま保全する。

## 最小修正

[adapter](../../src/trottertracks/c_pr_aware_df_m3.py)の`catalog_gate`は、atom_recordと保存catalogの`origins`/`word`だけをtupleへ正規化して比較する。元atomを変更せず、係数の有理数文字列、符号、basis key、frame ID、atom ID、provenance、word、要素数、順序を厳密に保持する。他の差や不正sequence schemaは明示的FAIL。ソート・近似比較・fallbackはない。

人工fixtureをJSONファイルへ保存して再読込するよう修正した。旧直接比較で不一致となるtuple/list境界が、新gateではPASSすること、8種類の内容改変と不正型がFAILすることを確認する。これにより実際の保存evidence経路を人工E2Eへ含める。新しいPR sampler/PF/compiler/RPE/費用集計器を作らず、B1/B2/C1/M2の適合関数は前準備と同じものを使う。

科学SOURCEで変更したのはM3 adapterと同名人工test。VERSION/scope/base/freshを修正版へ切り替えた。旧版SOURCEはcommit `9752f28cf5863e5caab815004a21d65b53c122d4`と旧manifestで保持する。旧run01を再開・上書きしない。

## 科学条件・実行gate

[新契約](research_c_pr_aware_df_M3_s1_r64_repair_contract_20261028.json)へ、旧policy/target/allowance/sampling master・cohort・32標本/2scenario/compiler/phase/call定義を同一値で引き継ぐ。新SOURCE digestでseedが結び直されるが、旧run01はsampling0であり、旧cohort選択やpoolはない。

対象は保存直鎖H4・隣接1Å/STO-3G、通常number CDF8 Proposed、Nα=Nβ2/sector36/full512、元κ・scalar・frame0..7/LD2。Partial symmetric S2 s1/h=.1/r64/K0、δ=.1/m0..12/n=2^m、CA=.00159/DF reserveCA/8、βtotalπ/6・βstatπ/12、candidatefailure.01/axis.01/26、own-DF ground oracle/状態準備除外を維持する。

A61→全13適格時のみB325→32 trajectory correctness完了時のみC84→summary1、**最大471 units**。科学的INFEASIBLEではA結果を保全してB/Cを実行しない。Bの価格が高くても正しく完了したらCへ進む。数値tolerance1e-10/complexsignal2e-10/rtol0/globalphasefitなし。未知全law synthesis/law/supportはnullのまま、旧C1の2仮定付き費用scenarioだけを評価する。

Fresh output：`artifacts/research_c_pr_aware_df_M3/2026-10-28/run02`。技術的relaunch1、run内retry/resume0、旧19callは消去せず累積を別記する。意図しない技術故障があれば部分証拠保全・最小修正・人工検証・新SOURCE/契約/fresh outputで対応できる最新許可がある。科学的FAILを技術故障に置き換えたり、条件/候補/閾値/標本数を変更したりしない。

## 検証・資源・公開

[新人工/回帰log](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/technical_repair/tests_final.log)に新22＋既存79の101 testsを保存する。旧準備の92人工PASSがJSON境界を見逃した事実は取り消さない。人工PASSを科学的適格性へ昇格させない。

新SOURCE/inputを固定し、旧raw133件（従前125＋M3 run01の8件）と旧契約/manifestをhash保全する。Task-local venv/cache/output、read-only依存。実affinity/quota/MemAvailable/pressureを確認し、既存resource plannerとBLAS協調、Qiskit共通build/compile/actual-adjoint-build mutexを維持する。CPU/RAM/worker/wallの人工cap・共有設定変更はしない。

人工検証後は追加承認を求めず、固定SOURCEを公開してA→条件付きB/Cを実行する。別分子・追加arm・s/r/K/γ探索・新fit・追加sampling・全law/K1/CA保証・詳細科学レビューは行わない。結果・call・resources・STOPを公開しremote SHA/tree/SOURCE/input/raw照合後mandatory STOP。
