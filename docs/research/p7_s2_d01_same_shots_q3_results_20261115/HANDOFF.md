# C・A・統合担当へ：p7/S2要求幅hの同一shot精度比較

正本は[REPORT](REPORT.md)、exact有理数の[EXACT_COMPARISON](EXACT_COMPARISON.json)、表示[SUMMARY](SUMMARY.json)、[INPUTS](INPUTS.json)、[C_RESULT_JOIN](C_RESULT_JOIN.json)。科学実行SOURCEは **650017fa1c6d3f408f34e5eb391d038e3b3c1475**。結果公開SHAとは別である。

## 引渡し結果

X/Y各軸の固定配分は [59,55,51,47,43,39,35,31,27,23,19,13,4]。Z1＋T1 total decoder、保存solver proxy、全事象平方loss、IID二項モデル、exact τを比較先と揃えた。

| 要求幅 | Q3モデルMSE区間 Ha²（表示） | 判定 |
|---|---|---|
| h=0.1 | [1.113188114132502061033934E−6, 1.113188114132502061033945E−6] | 適合・新1call |
| 16h/7 | [1.506660616251764083585358E−6, 1.506660616251764083585362E−6] | 適合・b4296b110の保存APIを再利用 |

差（h−16h/7）は [−3.934725021192620225514282E−7, −3.934725021192620225514126E−7] Ha²。閾値までの余裕は約1.032900669E−6から約1.426373171E−6 Ha²へ増加した。2幅1配分の条件付きモデル精度比較として利用できる。

元configuration e6754564…、total view b8ceb775…、binding 8e1e602d…、元保存packet 5fb2c32e…/B0I_P7_S2_D01をSOURCE付きで結合した。元specはdelta_pf以外が比較先と一致し、幅h固有の全13round qPF=2^m、実幅0.1を保持した。C_RESULT_JOINは元physical ID、round、X/Y整数配分、元/total ID、packet locatorを保持する**精度側索引**であり、価格receiptを付け替えない。

## 保存状態・主張範囲

新1callは閉包・質量・残余・上下界gateを通過し、原API/3種partials/RESULT/資源記録をworker自身で保存した。caller/記録済み子全て終了code0。非終端8,192状態の部分DPで残余mass約2.90883E−31とloss上界約1.13319E−29を残し、zero/tieはtotal規則で評価した。実行143.316秒、記録詳細はREPORTとRESOURCE_FINALを参照。

旧Minkowski・AδEは別参照で、今回MSEへ再加算しない。solver proxy・idealモデルへの条件付き評価であり、厳密ground/実sampler移送/数値certificateは未認証。十分N・global RMSE・G_qualifiedはnullのまま。

**435,481,090 RZは16h/7の保存費用に限る。hへ流用しない。** hの全価格がないため、幅間の費用差・費用比・winnerは今回未評価。価格担当は既存物理IDへ結合できる証拠がある場合だけ別途受け入れる。

## ロードマップへの差分

ロードマップ本体は統合担当が更新する。

1. B0I p7/S2・要求幅hについて、保存13roundと新固定配分のtotal Q3が1件完了した。readinessをmetadata受入のみから、この1配分の条件付きモデル適合・完全保存へ進められる。
2. 同じshot列の16h/7に対し精度余裕の増加を確認した。粗幅の既存達成費用435,481,090 RZと結果は保全し、精度余裕だけで費用優位を変更しない。
3. 幅hの全価格との結合、別幅・別配分、十分N・物理RMSE保証は今回の成果で補完されない。追加実行や価格取得を自動開始する根拠ではない。

新信号/価格/MC/build/compile/追加配分/retryなし。既存固定成果・他担当worktree・旧STOPは不変。公開・remote照合後 **mandatory STOP**。
