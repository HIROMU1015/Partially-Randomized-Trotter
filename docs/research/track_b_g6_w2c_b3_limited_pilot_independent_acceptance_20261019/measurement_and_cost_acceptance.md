# CPU/wall・observer・保存の受入

公開reportを持つ13件すべてで、独立cost adapterがcoordinator total=parent+worker、host parent+全worker総CPU、host outer wall、pair coarse/fine reconciliationとsetup/online/angle/evidence/output/IPC/memory/IOを照合した。reportのない5件の費用はnull/証拠不足。親crashを0補完していない。別の新S projectionでも境界は整合した。

pair CPUはworker total内、launch/IPC/finalizationはparent/outer内。再加算しない。worker wallを加えてglobal wallにしない。coarseとfineは代替partitionとして別reconcile。shared X/Y setup/Full prepareをpairとglobalで二度足さない。generatorとauditはcoarseでは一体であり、fine labelの内訳を超える分離を認定しない。

単独observer overhead、lost-pair CPU、unknown bit operations、receipt self-cost等はnull/unknownのまま。baselineとinstrumented差を純observer costやCPU CIへ自動転用しない。RSS/HWMは観測でありshared-page sumをunique memoryとは呼ばない。保持bytesと累積IO/pwrite量、flushとfsyncを分ける。失敗prefixは正常comparison CSVから除く。

P1/fused=PF、S/fused=SF。両方同じcodec/observerという二条件比較だけではfused単独効果は識別できない。固定portfolioの運用費用であり、機械独立なarm本来の費用ではない。cache/帯域/IO/起動・他job負荷・順序の交絡が残る。各条件1回の同prefix観測とblock記述は独立CPU母平均CIではなく、量子alpha/confidenceも転用しない。

公開人工CPU倍率・今回のsynthetic trace負荷から実G7の所要時間・順位を予測していない。実G7 local coefficient、rational arithmetic、guard、angle/requeryの分布は未取得である。
