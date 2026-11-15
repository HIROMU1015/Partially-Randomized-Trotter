# Full-H κ2 D7独立holdout：実行中

利用者が採択したC2の規則に基づき、固定m5の2 pathとm6の1 path、最大6 compileの取得を開始した。
最終receipt・4件の比較・C向け引渡し・結果公開は未完了。現時点でモデル精度やgate PASSを先取りしない。

利用者の追加高速化指示後、完了cellの公開準備を他cellの取得に重ねる処理を起動した。
導入済みpigzの並列圧縮と2 cellのprocess並列公開照合を採用し、人工3件でbytes/hash/比較行の保持を確認。
後処理SOURCE03cc993ee、科学実行SOURCE9cbabc617と稼働workerは不変。[追加高速化](PUBLICATION_ACCELERATION.md)。

実行SOURCE/input：`9cbabc617e95a81810924413c1b2b06ced329514`。指定branchへpushしremote SHA一致を確認した後、
別のdetached checkoutで固定して実行する。元較正raw・旧STOP・C/C2の原本は保全。

観測前のfrozen_model_gateと4予測のphysical/cohort/seed対応検査はPASS。Cの公開前照合以後を、
観測値を開かずに関連ref・保存先名・稼働jobで再確認した。再利用可能な既取得価格0、重複科学job0。
[観測前照合](PREOPEN_CHECK.json)、[固定入力](FROZEN_JOBS.json)、[SOURCE準備](SOURCE_PREPARATION.md)。

H4 linear1Å/STO-3G/保存tol-only DF10、one-body込みFull-H pure RTE/L_D0/K2/κ2、元scalar/event phase、
symmetric-directional、新ZZ lowering、測定付きfull wrapper/P0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を保持する。
各pathは固定seedで一度samplingし、同一pathのX/Yを個別compile。独立cell/pathとX/Yはprocess並列、
既存exact gate cache・fast fingerprintを再利用し、共有設定を変えない。

| 項目 | 状態 |
|---|---|
| 予定 | 3 fresh path、最大6 compile、4比較 |
| 事前保存compile再利用 | 0 |
| M5_ROUND q32/R283/T3.2 | path0/1の取得job実行中 |
| M6_ROUND q64/R566/T6.4 | path0の取得job実行中 |
| 取得/実装gateの最終結果 | 未確定 |
| 比較の有効・欠測件数 | 未確定 |
| 新規compile実績・資源 | 完了後に元recordから報告 |
| モデル判定 | NOT_ASSESSED_NO_ADOPTED_TOLERANCE、tolerance=null |

人工・回帰32 test PASS（caller11/acceleration13/holdout8）、import-only23 module PASS。
これはD1本人による技術検査であり、実H4のgateやモデル精度のPASSではない。

後続はP/O/P−O/(P−O)/Oのexact有理数4行、個別6軸の価格・元receiptとC予測への固定locatorを公開しCへ渡す。
m5片方だけの平均代替、X/Yの標本数合算、retry/seed差替えを行わない。モデルPASS/FAILを付けない。
実行中の観測方法・再投入禁止と残作業は [CONTINUE.md](CONTINUE.md)。

ロードマップへの現差分は「C凍結と規則採択後の指定独立holdout取得中、記述的比較は未完了」。
本体は編集しない。m6 n2・追加round・再fit・事後閾値・全law E[C]/CI/N/G/winnerへ進まない。
失敗時は部分証拠を保全しSTOP、指定成果物のcommit/push・remote/公開照合後mandatory STOP。
