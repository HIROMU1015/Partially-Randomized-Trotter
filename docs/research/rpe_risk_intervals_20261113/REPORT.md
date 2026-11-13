# RPE risk・RMSE統計区間の技術接続

2026-11-13 JST。B基準`2906a231224184e36c86fbcfa609af647fdf9f8a`、既存branch `codex/rpe-decoder-statistics-20261113`を再利用した。今回の再開は技術実装・人工検証だけであり、risk水準・loss・target・N探索方式を科学的に採択していない。実H4 MC、保存科学数値の再解析、実RMSE/必要N/E[C]/G評価は実行していない。

## 再利用と不足

| 機能 | 対応 |
|---|---|
| 経験的な解決条件付きRMSE・解決分MSE寄与 | 既存`rpe_decoder_statistics.summarize`を再利用 |
| 未解決率のClopper–Pearson | 元`rpe_hadamard_failure_validation._clopper_pearson_interval`を再利用 |
| 全事象bounded-loss/Hoeffding envelope | 既存summarizeの`population_RMSE_CI`を再利用。元decoderのenergy補完は追加しない |
| 条件付き母集団RMSE区間 | 新規：CPによるpとHoeffdingによるvの同時区間からv/pを囲む |
| 複数候補・配分の同時confidence | 新規：外部指定の有限family・各failure budgetの和を検査し、各固定callへ結合する |
| 元energy/Minkowski・十分Nの科学的資格 | 未取得。UNKNOWN/nullのまま。資格producer/optimizerは実装しない |

新module [rpe_risk_intervals.py](../../../src/trottertracks/rpe_risk_intervals.py)と、既存callerの任意`risk_options` hookだけを追加した。evaluation_targetの受入、A signal binding、whole-shot prepare、scalar/R3 decoder、乱数/count/status処理、既存summary、N/C会計・選択器は再実装していない。risk_options省略時の呼出し・出力は従来どおりである。

## 追加出力

1. 元の経験的conditional RMSE。
2. 一様loss supportとIID仮定・同時failure budgetに条件付くconditional MSE/RMSE区間。
3. 未解決率とCP区間。全試行で解決しても母集団の未解決確率を0としない。
4. 外部の全事象loss採択が明示され、bounded-loss contractとsupportが揃う場合だけ、宣言したmodel loss familyのrisk区間。それ以外はUNKNOWN。
5. 元energy RMSE契約の資格は常にUNKNOWN。Minkowski/物理・数値/target/state/actual samplerへの移送を認定しない。

p_L=0又はsupport不足なら条件付き上界もUNKNOWN。解決分寄与Yで未解決を0にする定義と、全事象lossを0とする採択を区別する。誤branch/aliasでもDECODEDなら元の非circular energy二乗誤差に含め、priorでclipしない。全trial/statusと原科学契約はそのまま残る。

## 人工・回帰検証

[TESTS.txt](TESTS.txt)：新規12＋既存47＝**59技術テストPASS**。比率区間、p_L=0、未解決0/有、support欠測・違反、元zero/tie/誤branch、有限人工分布の被覆、failure budget超過、LD/SOURCE/law/target/配分/MC範囲の取り違えを確認した。人工2構成×2配分×13roundでも元診断と一致し、十分N/Gへ昇格しない。

保存Aを扱う既存回帰はSOURCE/metadata照合のみで、MC/実G入口を禁止した。新risk区間を保存H4 trialや旧S3結果へ適用していない。C全runtimeやDの価格実行を起動せず、他担当のworktreeに介入していない。検証はprimary agent自身による技術検査であり、独立した科学レビューではない。

## 限界と次の担当

区間は宣言したIIDモデル・一様support・固定有限familyに条件付く統計結果である。SOURCE付きsupportを渡すだけで、その物理的正当性が証明されるわけではない。SciPy CPとbinary64算術は認証区間演算ではない。旧summary内のmarginal CIは、新familyの同時被覆対象に自動追加しない。

[RISK_INTERVAL_HANDOFF](RISK_INTERVAL_HANDOFF.md)に数式・API・N/C接続、[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)に未採択条件と有限案の差分を記載した。ユーザーが条件と有限scopeを明示採択・実行許可した後にのみ、Bが必要な科学評価を行う。Cは付加されたrisk証拠を保持できるが、元の正式N/会計gateを維持する。Dの価格研究とは独立である。

commit/push・remote SHA/tree/主要成果物照合後、**mandatory STOP**。新しい科学実行へ進まない。
