# 採択済み局所shot探索：実行前固定

2026-11-16 JST。計画9992c594d4c72d1e8b65b42029d2b231641c0d17と今回ユーザー指示を適用する。p7/S2・要求幅16h/7の1候補、出発点B_PRICEからの単round4案→必要時のみ振替4案、合計最大8新規Q3。最初の厳密に安い適合点で終了する。

## §10.5・保存再利用

関連all-refのp7/配分/Q3履歴、固定PF6結果、幅hの後続結果と局所探索計画、参照可能な同ユーザーの該当Q3 processを限定照合した。確認範囲で今回の粗幅・具体列・同一契約に対応する保存/進行中Q3は未同定。新規対象を既取得か判断不能な記録は確認していない。直前の幅hの結果やPauli配分は別configuration/lawであり代用しない。

主出発点はb4296b110のB_PRICE原APIで、元configuration a6e55777…、total e496358b…、binding d54a41b4…、元packet06a15ee4…を照合。保存適合点はQ3再実行せず、同candidateのA_SHAPE原APIも再利用catalogへ登録する。N0は旧component/部分保存のまま保全。既存API再利用は元target/SOURCE、canonical binding、packet locator、全26N、decoder、law/lossと数値optionsを照合し、science callerが注入した来歴sourceだけを意味比較から分離する。

価格は1002f1dfeaのAXIS_TERMSのp7/S2固有26軸。各行のconfiguration・physical ID・元/total候補IDとsignal側を照合し、16直接軸＋10条件付き閉包軸、元receipt/SOURCEを保つ。新compile/fit/価格予測は行わない。今回の費用はこの価格policyに条件付く。

## 固定した候補・SOURCE

[INPUTS](INPUTS.json)に元seed/cap、初期4案、有限振替pool4案、全軸のN/C/積/価格SOURCE、実費用・削減額、count-only、既存error contractとscopeを固定した。振替poolの算術準備は新しいQ3投入ではない。実際の選択は前段Q3終了後で、未選択案は未評価のまま残す。

| 順 | 元seedからの変更（X/Y各軸） | 条件付き費用 RZ | 削減額 RZ |
|---:|---|---:|---:|
| 1 | m12：4→3 | 415,721,884 | 19,759,206 |
| 2 | m12：4→2 | 395,962,678 | 39,518,412 |
| 3 | m11：13→12、m12は4 | 425,601,180 | 9,879,910 |
| 4 | m11：13→11、m12は4 | 415,721,270 | 19,759,820 |

他roundはすべて元seedのまま。各軸capは元N0、下限1。MSE非単調性を保ち、1shot減失敗で2shot減を省略しない。振替は計画どおりfamily内既評価有効上界の最小値→実費用→変更量→r/d/k/sで優先度を更新する。Q3上界を未評価案の予測値として用いない。

runner `scripts/run_p7_s2_local_shot_search.py` は、既存PF worker/total binding/kernel/高速DP/保存supervisorを再利用し、候補順序と履歴保存だけを追加する。kernel/元decoder/旧DP/旧結果は変更しない。全候補poolを実行前commitのINPUTSへ固定しているため、各stepの元データはそのcommitから復元可能。実行時のSELECTIONは親配分、具体列・費用、選択理由、参照した親結果をQ3前に別途保存し、将来の選択を事前に観測済みとは表示しない。

新8人工/metadata testがPASS。順序・同seedからの変更・非単調な合否・8件上限・早期終了・保存再利用・INCONCLUSIVE/UNKNOWN・振替優先度を確認した。既存worker/monitor故障時保存の回帰も再利用し、人工入力だけで確認する。科学実行前にcaller/INPUTS/testsをcommit/pushして固定する。

## 数値・性能・STOP

元Z1＋T1、solver proxy −2.1663874486347634 Ha、全事象平方loss、保存deterministic raw/ideal IID二項/EXACT_AFFINE_OF_BINARY64_RAW、exact τを保持。AδEを再加算しない。ρ=1、outcomes4,225、verification55,043,173、非終端frontier8,192、transitions141,559,036は不変。具体列のcount-onlyはすべてscope内だが、新count格子の閉包・質量・残余・上下界は各call内で新たに検査する。

後続Q3を投機並列投入せず、1件終了・保存・判定の後だけ次候補を選ぶ。Q3内は既存8 DP worker/batch8・協調stage/segment・子終了codeとworker自身の原API/partials/資源保存を使う。これはCPU/RAMへの新しい制限ではなく、早期終了に必要な逐次選択と既存高速backendの再利用である。

有効上界≤τかつ費用<435,481,090で即終了、下界>τは当該列不適合、閾値をまたぐ区間はINCONCLUSIVE。重大SOURCE/数値/保存gate不成立はUNKNOWNで停止し、原結果を保全。scope/候補域拡張、別幅、他方式、追加compile/fit/MC、無断retryなし。十分N/物理RMSE/正式G/winnerは認定しない。実行終了と公開照合後mandatory STOP。
