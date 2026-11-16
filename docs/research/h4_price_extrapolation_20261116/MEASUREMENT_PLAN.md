# 追加の直接価格確認案（未実行）

候補を二つの独立holdoutに絞る。今回の承認は保存資料分析までであり、以下の取得は実施していない。
元のS2/Yoshida4、H/state/snapshot、scalar一回会計、compiler/control/wrapper、Pauli辞書・sampling lawを変えない。
exact値・round ID・固定予測は[MEASUREMENT_OPTIONS.json](MEASUREMENT_OPTIONS.json)に保存した。

## 1. 推奨順と根拠

|順|対象|固定点|そのroundの費用|参照単価/軸|得られる情報|
|---:|---|---|---:|---:|---|
|1|PR-Pauli LD5|m7、q128/R1、T12.8、実幅h|21,639,868.57 RZ（5.28%）|3,786,977/7 RZ|元q8/R4較正からqを16倍へ、実際のR1へ移した全wrapper・head境界を確認|
|2|p7標準Yoshida4|m10、q160、T102.4、実幅32h/5|58,224,034 RZ（15.65%）|2,646,547 RZ|既測q80を越える最初のroundで長q affine仮説を確認|

LD5のH5は全費用407,152,140 RZ（99.26%）を共有する。同じhead配置・境界規則のq移送を調べる情報価値があり、
cell自身の費用だけで順位をつけていない。尾部E5を0まで減らしても差を解消できないため、Eだけを高精度化する取得は優先しない。

p7 q160は最大q640より小さく、同一実幅の検証前線をq80→160へ延ばす。
q320は95,270,166 RZ（25.61%）、q640は127,023,204 RZ（34.14%）を占めるが、回路取得も大きくなる。
未検証範囲の費用は280,517,404 RZ。q160の観測だけでこの全額を直接測定済みにはしない。

LD5 q128/R1より安いq32/R1は境界/R変更のsmoke案として残るが、qの延長は元較正の4倍に留まる。
q512/R1は69,217,723.43 RZ（16.87%）、q2048/R4は121,901,314 RZ（29.72%）に直接対応する後続候補。
最大q4096/R8は70,253,495.43 RZ（17.13%）であり、最大寄与roundではない。qとRを同時に大きく動かすことも初回の原因切分けには不利である。

誤差の事前分布・新cellのwall実測がないため、「情報利得最大」の数学的解とは呼ばない。
費用寄与、検証したい共有則、既存証拠、取得負荷の代理値から選んだ有限提案である。
新最安点では[HANDOFF](HANDOFF.md)の同じ感度計算を更新し、順序を再判断する。

## 2. 測定済み負荷と未測定負荷

|既存取得|実測wall|観測peak|wrapper compile|
|---|---:|---:|---:|
|LD5 Q8R4、独立normal2path|約41秒|約0.70 GB|4（2path×X/Y）|
|p7 q5/q10較正2cell|172.276秒|約0.705 GB|4|
|p7 q80独立holdout|1,222.373秒|約3.42 GB|2|

SOURCEは[EVIDENCE_AND_PROGRESS.json](EVIDENCE_AND_PROGRESS.json)。別q/Rの時間をこれらから厳密に予測しない。
参照RZ回路サイズの比較では、LD5 q128/R1は約0.541M/軸、p7 q160は約2.647M/軸。
transpile、archive、作用検査等の負荷はRZ数だけでは決まらず、CPU/RAM/workerの新しい恒常上限を置かない。
LD5の独立2pathは計4wrapper、p7は決定論1cellでX/Y各1の計2wrapperを使う既存の最小形を提案する。
有限2path平均を母期待値やCIとみなさない。順番・並列化は共有環境への非干渉を保って実行準備で決める。

## 3. 既存入口の対応状況

**二つとも、現行の専用CLIへround引数を足すだけでは実行できない。今回、その入口は変更していない。**

LD5：`02e7cc6a1:scripts/run_pauli_split_cal.py --request` のdriver検査はQ8R*、q8、T=step×8、m12等のanchor Rを固定している。
q128/R1には、独立HOLDOUT用cell-layout metadataと検査分岐を最小追加する必要がある。
`13ab1acfa:scripts/run_pauli_b1_prices.py:acquire_path` と既存normal2path/X/Y workerは任意q/Rに利用できる構造で、数値emitterを新設する必要はない。
元Q8較正経路を保ち、canonical m7・exact T/step・dict/law・implementationを束縛したfresh cohort/namespaceと、
係数を更新しないholdout結果区分を用意する。audit/helperのq計数は再利用できるが、自動fit/26価格更新へ接続しない。

p7：`015d04a2a:scripts/run_p7_y4_coarse_prices.py` は較正m5/m6・holdout m9/q80固定で、`--round 10`は存在しない。
`c94f290d5:scripts/prepare_pf_candidate.py:acquire(value,out,phase,ms,source)`、native childとC guardを再利用し、
m10のcanonical条件、固定モデル予測、fresh namespace、no-refitを明示した小さなholdout caller差分が必要。
通常のcomplete入口は既成p7を再利用するため、それ自体を追加測定入口と呼ばない。

## 4. 取得するなら保存する量とSTOP

- 同じcanonical/物理条件、exact scheduleとnative値、辞書/descriptor/implementation/quantity/profile/SOURCEを固定する。
- 価格取得前に元係数・予測を固定し、fresh holdoutの元path/seed/cohort、X/Y個別receipt、RZ、作用gate、段階時間/peakを保存する。
- LD5はfull wrapperの予測−観測と既存head/境界・tail body分解を確認する。X/Yを独立pathとして水増ししない。
- p7はfrozen 307+16539qと観測を比較する。既測q80をfitや再取得へ戻さない。
- 入力/辞書/law/SOURCE不一致、数値gate不成立なら部分出力を保全してSTOPする。閾値緩和・自動retry・候補変更なし。
- 差が観測されても、今回のfixed Nに対する寄与を示すところまで。新fit・N再最適化・Q3追加・G/winner認定へ自動移行しない。

追加測定の直前にD/D2の公開・進捗metadataを再照合する。
p7 q80とLD5 Q8R4は取得済みなので新規候補から外す。
D c94f290d5のFull PF/S2/2h q16、D2 0ef9668e9のPauli LD4/DF LD6、no-drop 8478b8200の作業は別条件である。
観測した範囲で対象新cellの保留取得はなかったが、全サーバーでの不存在は主張しない。

公開直前の再照合：Dは`352539e2749989b4ddc3ab74c31f486982bfe265`へ進んだが、c94f290d5からの差分はHANDOFFのAPI SOURCE追記2行だけで、新しい対象価格取得はない。D2の二つのSOURCEは不変。確認した3worktreeはcleanだった。
