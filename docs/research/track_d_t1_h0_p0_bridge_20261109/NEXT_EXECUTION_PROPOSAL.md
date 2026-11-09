# 次の研究判断：局所互換の拡大とnative emitterへの接続

2026-11-09 JST。**構造的評価を続ける価値はあるが、今回の結果だけでq1024〜4096のfull P0価格や全RPE費用へ進む根拠は足りない。** 次の担当はユーザー/GPTで、以下は未実行案である。

## 取得済みdomain

| 対象 | 現在できること | 残条件 |
|---|---|---|
| 確定native列 | T0のcount/depth/size/宣言幅を厳密・streaming計数 | descriptorから同じnative列を作る意味資格 |
| CX局所rule | 両wire隣接の同方向対を相殺し、chunk間の履歴を保持して資源を計数 | 全P0との同値性は未資格、memoryは最悪O(L) |
| 保存人工domain | 70 native record再計数、raw16中10の全指標／15のCX count/depthがP0と一致、指定3 P0も一致 | 有限caseで、未観測角・1Q・Gaussianを覆わない |
| 指定H4 2path | 保存compiled全8の絶対価格照合とraw局所相殺 | compiled答えの再読をpre-P0予測と混同しない |
| 全H4／長q P0 | 未完成 | 1Q run、Gaussian、parity、CSWAP／χ等の全界面が未閉包 |

## 次の最小技術課題

1. **既存の1Q／zero反例を先に使う。** 今回未対応の6 raw native caseを固定して、exact zero RZ、同support signed RZ、X/SXを含む1Q runの合成とP0の挙動を分ける。最初は保存traceとQiskit1.3 SOURCEで検討でき、新しい分子処理は不要。独自の一般optimizerではなく、既存の1Q合成規約・phase・特殊角判定を再利用できるか調べる。
2. **native emitter出口を小さく接続する。** 既存C3／StrongBuilder／Gaussian／tailの命令生成順・relative fusion・control mode・scalarを保持する。新sampler、Gaussian合成器、Reference gadget、H4全builderを再実装しない。未展開gateをnative一命令と数えるshortcutを使わない。
3. **frontierをいつ確定できるか調べる。** 同じwireへ将来届く命令が過去CXを露出できるため、単純な一定長窓は不十分。安全にflushできるinterface／opaque境界を定めるか、可変memoryを受け入れるか判断する。T0 max-plusだけでP0境界のgate情報を代替できない。

追加H4実測や新P0を先に増やす必然性はない。次の処理で必要になれば、目的、対象case、SOURCE・native化規約・precision・call数・STOPを明示して別scopeとして採択する。今回の新P0上限3回はすでに全て使用した。q128や別pathへの自動拡大はしない。

## q1024〜4096とモデルの使い分け

native streamが供給されればT0は長さに比例して計数できる。今回bridgeの履歴storageも長さに比例し得るため、巨大qの経済性は未実測・未保証。full QuantumCircuitを作らないことだけを性能成果とせず、必要な価格精度と実際の負荷を比較する。

全P0へ対応できない資源は主A／比較F/Sを維持してproxyとして扱う。局所CX削減を保存compiled価格やモデル予測から追加で差し引く二重計上はしない。新fit、A→S変更、長q biasを0とする操作、全13-round会計は行わない。

各方式の絶対価格とpaired差を分け、全law期待値にはfeatureのjoint分布、E[max]、finite-bit／law資格、共分散、model biasの情報が必要である。今回の2 path一致や人工case一致率から母平均CIを作らない。N／actual RPE成功は価格bridgeと別の資格である。

## O1/O2/O3・PR全体への接続

共同最適化の候補がevent、basis、PF、control、cutを変更する場合、actual SOURCE／law／native順序／phase／最適化policyを評価器へ反映する。C1 ON/OFF、O2の別baseline等を同一cacheや同じP0観測として混ぜない。現段階の局所指標だけを使って全PR費用の最適候補を断定しない。

次に判断すべきことは、全P0互換への1Q／Gaussian界面研究を優先するか、未保証モデルを使う探索段階を維持するかである。主資源指標・新しい数値精度・成功確率は本書では採択しない。**Codexは今回の公開照合後mandatory STOPし、追加実装・科学計算・詳細レビューへ自動進行しない。**
