# p7/S2・16h/7：m12各軸2shotの固定1件

2026-11-16 JST。今回のユーザー指示で、e7d24c2192d4640137bdaf993a4e954a49f7932cが固定した `P7_S2_LOCAL_R12_D2` だけを最大1新規Q3。前回の早期終了や原runを再開・変更せず、新しいnamespaceへ保存する。

## §10.5差分照合

0b33b4afdの局所探索はm12=3の最初の1件で終了しており、m12=2は未評価。関連all-refのp7/配分/Q3履歴、固定PF6配分、現在参照可能な同ユーザーの該当jobを限定確認し、同configuration・当該整数列・decoder・target・law/lossの保存/進行中結果を確認範囲で未同定。d01c780d0のYoshida4準備、別幅h、他方式の結果は対象と異なる。

既存A_SHAPE/B_PRICE（b4296b110）とm12=3（0b33b4afd）の原APIを再利用catalogで照合する。新2shot列と一致せず、旧4/3shotは比較として再利用し、再評価しない。SOURCE/target/law・数値契約の異なるcatalog entryは既存受入関数で拒否する。

## 固定入力と費用

正本はe7d24c219の `docs/research/p7_s2_local_shot_search_results_20261116/INPUTS.json` の `single_cases[1]`。候補、元/total configuration、binding、13round signal locator、26軸price_terms、decoder/target/loss/exact τ・数値optionsをbyte不変の既存callerとともに再利用する。選択・振替探索関数は起動しない。

```text
N_X=N_Y=[59,55,51,47,43,39,35,31,27,23,19,13,2]
allocation_id=P7_S2_16H_OVER7_LOCAL_R12_D2_01
元configuration=a6e557770a2e4d96585ad5ca5903c4626ae47dda9775e9a9babc756149c38d84
条件付き費用=395,962,678 RZ
```

元B_PRICEからm12だけ4→2、他roundは不変。価格は1002f1dfeaの同p7/S2固有行、16直接軸＋10条件付き閉包軸。raw/normalization/scalar・元physical ID・receiptを変更しない。

count-onlyを既存関数で再照合し、outcomes上限3,600、verification38,523,172、transitions116,388,368。scopeはρ=1、outcomes4,225、verification55,043,173、非終端frontier8,192、transitions141,559,036を維持し、新count格子の閉包・質量・残余・上下界をcall内で検査する。

## 実装・保存・停止

新caller `scripts/run_p7_s2_m12_two_fixed.py` は固定case選択と既存1case supervisor/worker、結果検査を使う薄い入口。旧local search・PF worker・decoder/kernel・高速DPは変更していない。新2人工/metadataテストで固定1件のみ・全13round/両軸・価格・scope・旧3shot再利用・SOURCE/law拒否、science/search未起動を確認。既存worker/monitor保存の検証資産をそのまま利用する。

caller・[INPUTS](INPUTS.json)とテストをcommit/pushして実行SOURCEを固定してから、新run01で最大1回のみ実行。Q3内部の既存8 DP worker/batch8・協調stage・worker自身の原API/partials/資源保存を維持し、CPU/RAMの新制限や非同期frame dumpを追加しない。

採択済みZ1＋T1、保存solver energy proxy −2.1663874486347634 Ha、ideal IID二項/EXACT_AFFINE_OF_BINARY64_RAW、全事象平方loss、exact τを維持。2shotが有効上界≤τで適合した場合だけ達成費用を更新し、それ以外は415,721,884 RZを保持する。UNKNOWNでは部分証拠を保存しSTOP、無断retryなし。4/3/2shotを並べ、未評価Nや単調性・最小Nを推定しない。新signal/compile/fit/他配分Q3は0件。公開・remote照合後mandatory STOP。
