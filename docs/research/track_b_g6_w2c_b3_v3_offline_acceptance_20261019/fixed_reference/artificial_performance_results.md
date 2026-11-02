# 最終SOURCEの人工性能

SOURCE `860fef4a3a8ebb5a46628f4e0364e5934b5af416`。60測定（4構成×3workload×5反復）、別途12 warmup。各中央値、秒。
Pは改善P1＋cached observer＋coalesced IPC、PFは同じものにfusedを追加。S/SFは同じv3でpairを逐次実行する対照。
全値に新Host/Admission/runtime・schema検査、spawn/import、全worker、保存、外側STOPを含む。fixture作成/readback/圧縮は除く。

| 負荷 | S wall / 総CPU | SF wall / 総CPU | P wall / 総CPU | PF wall / 総CPU | P→PF CPU減 | wall減 |
|---|---:|---:|---:|---:|---:|---:|
| light | 10.4874 / 10.8460 | 9.8261 / 10.1781 | 2.5547 / 11.3730 | 2.4537 / 10.7355 | 5.6% | 4.0% |
| mixed | 8.2286 / 8.4279 | 7.8769 / 8.0719 | 2.0241 / 8.8214 | 1.9460 / 8.4712 | 4.0% | 3.9% |
| heavy | 9.2419 / 9.3671 | 9.0515 / 9.1729 | 2.1676 / 9.6755 | 2.1532 / 9.5016 | 1.8% | 0.7% |

P→PFの追加利益は小さく、特にheavyのwall差は0.7%。人工範囲でPFを第一技術候補、Pを比較対照として維持する。
科学的正式採択や実G7の所要時間予測ではない。共有負荷・構成順・起動検査費も影響し、統計的優位を認定していない。

| 負荷/構成 | wall [min–max] | 総CPU [min–max] | 親CPU | worker CPU |
|---|---:|---:|---:|---:|
| light/S | 10.4874 [10.4570–10.5594] | 10.8460 [10.8118–10.9102] | 1.4103 | 9.4372 |
| light/SF | 9.8261 [9.8065–9.8387] | 10.1781 [10.1630–10.1917] | 1.3906 | 8.7928 |
| light/P | 2.5547 [2.4350–2.5887] | 11.3730 [11.2896–11.5516] | 0.9066 | 10.4831 |
| light/PF | 2.4537 [2.4103–2.5492] | 10.7355 [10.7101–10.8281] | 0.8950 | 9.8493 |
| mixed/S | 8.2286 [8.2214–8.2418] | 8.4279 [8.4131–8.4435] | 0.8992 | 7.5310 |
| mixed/SF | 7.8769 [7.8343–7.8940] | 8.0719 [8.0363–8.0965] | 0.8899 | 7.1862 |
| mixed/P | 2.0241 [2.0074–2.0706] | 8.8214 [8.7278–8.8986] | 0.5979 | 8.2311 |
| mixed/PF | 1.9460 [1.9403–2.0563] | 8.4712 [8.4631–8.6107] | 0.5816 | 7.8899 |
| heavy/S | 9.2419 [9.1984–9.2669] | 9.3671 [9.3234–9.3849] | 0.7137 | 8.6549 |
| heavy/SF | 9.0515 [9.0166–9.0790] | 9.1729 [9.1407–9.2012] | 0.7082 | 8.4691 |
| heavy/P | 2.1676 [2.1431–2.1733] | 9.6755 [9.6019–9.7894] | 0.5332 | 9.1453 |
| heavy/PF | 2.1532 [2.1143–2.1771] | 9.5016 [9.4698–9.6236] | 0.5073 | 8.9993 |

独立に取った中央値の親＋worker和は総CPU中央値と丸め程度以上ずれる場合がある。各runの実値は厳密に親＋workerである。

| 負荷/PF | sampled RSS和 MiB | worker最大HWM MiB | pair出力 MiB |
|---|---:|---:|---:|
| light | 287.1 | 38.0 | 2.96 |
| mixed | 287.1 | 38.0 | 2.18 |
| heavy | 287.1 | 38.0 | 4.43 |

CPU affinity 64、測定区間host idle 84.8–93.7%、最低MemAvailable 467.1 GiB、CPU PSI avg10最大0.3、memory PSI最大0.0。
各run前後の観測をresource_observations.jsonへ保存。資源上限や共有設定を変えず、他ユーザーprocessを操作していない。
8 workerはphase内の8独立pairに対応する。逐次対照は比較条件であり、共有serverを理由に課したCPU上限ではない。

親/worker I/O、pair CPU/wall、startup/unattributed、各coarse/fine spanとresidualはruns.json・reports.jsonl.gz・cost_breakdown.jsonに保存。
generator_and_auditは予約RPCを含む。RPC CPUはその一部と重なるため加算しない。Observer単独費用とunknown bit operationを0へ置換しない。
旧77f8667fの測定とは新Host・runtime検査・spawn・観測path・共有負荷が異なり、今回の利益率へ混ぜない。
全60 runで158,720人工trial、全24fileのreadback hash/bit/trace/stateを照合。実科学callは0。
