# C・統合担当へ：p7/S2・16h/7のm12=2固定Q3

**2shotもモデル適合、新規Q3は1件だけ。条件付き達成費用を395,962,678 RZへ更新する。** 正本は[REPORT](REPORT.md)、[EXACT_COMPARISON](EXACT_COMPARISON.json)、[SUMMARY](SUMMARY.json)、[C_RESULT_JOIN](C_RESULT_JOIN.json)。科学実行SOURCEは **f60fafeca26f1cc65048c9a4f33e79434ca8fdc8**。元候補はe7d24c219の固定第2案で、旧早期終了runを変更・再開していない。

| m12 X/Y各N | 条件付き費用 RZ | MSE区間 Ha²（表示） | 判定 |
|---:|---:|---|---|
| 4 | 435,481,090 | [1.506660616251764083585358E−6, 1.506660616251764083585362E−6] | 保存適合・b4296b110 |
| 3 | 415,721,884 | [1.951734976728839865403189E−6, 1.951734976728839865403194E−6] | 保存適合・0b33b4afd |
| **2** | **395,962,678** | **[2.415680996446553638790769E−6, 2.415680996446553638790773E−6]** | **新規適合** |

全点のm0〜m11は各軸 `[59,55,51,47,43,39,35,31,27,23,19,13]`。今回allocationは `P7_S2_16H_OVER7_LOCAL_R12_D2_01`。全13round・Z1＋T1/solver proxy/IID raw/全事象平方loss/exact τを保全し、有効上界≤τをexactで判定した。新配分は3shot点より19,759,206 RZ低い。

C_RESULT_JOINは元configuration a6e55777…、total e496358b…、binding d54a41b4…、carrier06a15ee4…の保存packet、allocation、1002f1dfeaの元価格行・元physical/候補IDを対応付ける。価格は16直接軸＋10条件付き閉包軸、元receiptと閉包前提・誤差/CI=nullを維持し、正式G資格を新認定しない。

原API/partials/資源を完全保存し、新閉包・質量・残余・上下界/momentsを検査。caller/記録済み子99件は終了code0。zero/tieはtotal規則で会計し、残余mass約1.19E−31を残した。SOURCE/対象1件を人工検査2件で確認し、探索関数や旧4/3shot Q3は起動していない。

ロードマップ本体は統合担当が更新する差分：p7/S2・16h/7の保存達成点にm12各2shot・395,962,678 RZを追加できる。4/3shotの元値と旧早期終了は保存参照として残す。3点から単調性・未評価N・最小N・大域最適・物理RMSE/十分Nを推定しない。今後の費用比較には、採用するallocationとこの条件付き価格/Q3資格を明記する。

新signal/compile/価格fit/他配分Q3/retryは0。AδEを再加算せず、元Minkowski・ground/実law/数値certificateの未資格を保全。公開・remote照合後 **mandatory STOP**。
