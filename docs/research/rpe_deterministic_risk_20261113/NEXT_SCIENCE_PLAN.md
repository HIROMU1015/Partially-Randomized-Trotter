# 次の有限適用：MCを主評価から省いたLD1/LD2計画

2026-11-13 JST。利用者はRPE Monte Carloを主資源評価から省く方針を採択済み。B91d730/B2906/Bc26のMC計画は履歴として保全し、主経路から自動起動しない。今回は新しいMCを一切実行していない。

初回案は、既存A91349のLD1/LD2×共通whole-shot vector1本＝2決定論的評価、又は同2本＝4評価。全13roundを一組で扱い、round別の最良Nを独立に組み合わせない。両案とも今回未実行、具体的N・探索族・範囲は未採択。

実行前にユーザーが採択・明示許可する事項：

1. **energy taskと評価target**：保存full DF-H、元H/state/snapshot・h=.1/M12/各LDのcanonical条件を保持し、evaluation_targetの値・proxyの意味・SOURCE・資格scopeを定める。solverを厳密ground energyと自動認定しない。
2. **risk/loss契約**：元Minkowskiを保持する。今回の直接rawモデルMSEを何の診断又は予算項に使うか、未解決loss上界とSOURCEを決める。直接総risk契約、energy補完、reviewの水準B/Cを自動採択しない。rawに含まれる物理biasを二重加算しない。
3. **count law・数値の意味**：exact-affine又は旧binary64-affineのどちらか、ideal finite-K/fresh IID軸・round条件、raw/target数値allowanceのscopeを明示する。元の未資格なstate/actual samplerへの適用はUNKNOWNのまま。
4. **whole-shotとphase安全集合**：各N_X/N_Yと全roundのρを外部指定する。π/3を既定値にしない。KLY型などを探索する場合も族・有限格子を別途採択し、偶奇による非単調性を無視した二分探索やglobal optimumを主張しない。
5. **有限計算範囲**：count列挙数と隣接検証pair数、対象構成/配分数、SOURCE・出力先を固定する。メモリ・時間はcount supportと整数のbit長にも依存する。必要なら共通geometryの再利用や、より軽い実装margin証明を先に技術検証する。CPU/RAM/workerへ恒常的な人工capは設けない。
6. **Gへの資格接続**：同model入力に結び付くN評価の受入水準と、同identity/law/compiler/lowering/full wrapper/P0の適格E[C]が必要。現adapterは外部N packetを受け取るだけで、新たな科学的N資格producerは作っていない。実G計算は別の許可範囲とする。

初回は有限predicate検査が通る範囲で、モデルMSE上界・round別unsafe/zero・未解決確率範囲を出す。loss未採択、raw欠測、有限検査範囲外、decoder閉包不成立はUNKNOWNであり、物理候補/方式全体の不可能性としない。範囲を無断で拡張しない。より鋭いπ/T driftや一般のbinary64一様marginが必要なら、条件を特定して追加の技術検証を提案する。

STOP：SOURCE/identity/law/target/整数N不一致、数値domain外、未取得資格に遭遇した場合、値を補完してPASS/G/winnerを作らない。MCへfallbackせず、既存科学結果・grant/ledger/marker/KEEP/STOP、他担当の実行を保全する。

次は**ユーザー**が上記条件と有限科学実行範囲を採択し、**Codex B**が明示許可後に必要範囲だけ評価する。**C/D**は既存pipelineと同domainの価格資格を担当する。今回は公開・remote照合後に**mandatory STOP**し、実H4への数値適用・N最適化・C/G計算へ進まない。
