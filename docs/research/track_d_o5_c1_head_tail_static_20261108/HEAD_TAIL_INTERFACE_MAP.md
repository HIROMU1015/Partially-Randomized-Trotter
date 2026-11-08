# Head-tail接口の静的構造表

2026-11-08。参照版は[SOURCE_INDEX](SOURCE_INDEX.md)。以下はSOURCEの読解結果であり、新しい回路・event metadata分類・QPY解析の結果ではない。回路の列は**時間順（左から先にappend）**で表す。行列の積順はその逆。

## 呼出しと表現

`prepare_arm` → `prepare_common` → `prepare_steps`が現行whole-pathの経路。headはNativeModel/NativeBlockのrotationと対角primitive、tailはRTEEvent/applicationとBasisProvider、結合後はflattenされたQuantumCircuitである。`compose(..., inplace=True)`で隣接させ、head/tail境界をまたぐ専用relative変換は呼ばない。フラグだけでなく、この呼出し位置を確認した。

- `StrongBuilder.sequence` / `directional_head` / `mixed_head`は、入口にB_first†、出口にB_lastを持つ閉じたhead。内部は同rotationのgroup化とanalytic gauge補正付きrelative transitionを使用する。
- `build_event_sequence`はapplicationを保存順にflattenし、basis_keyが同じ連続runを一つのB†→centers→Bで包む。異なるrun同士は一度閉じて次を開く。連続中心を全て一つの角度へ合算する実装ではない。
- H4のProviderはatom.frame_idに対応する**rotationの厳密一致**でmodel blockを探し、同じbuilderのbasisをcopyする。singleton=Trueでもfull frame。generic BasisProviderのsupport completion、physical Pauli basisをH4と同じものと仮定しない。
- 固定H4のheadはblock0=one-body、block1=F0、block2=F1。random tailはLD2以後のfragment由来。異なるIDを同じbasisとは扱わず、両端が実際に一致する頻度は今回調べていない。C7-4のrun特徴はheadの入口/出口frameやgaugeを符号化せず、その頻度の証拠にもならない。

## 接口別map

| 接口 / 呼出し元 | 接続前後の表現・basis frame | Control mode | Scalar/phase | 現在の処理・未解決点 |
|---|---|---|---|---|
| 最初のhead→tail：C3 prepare_steps 139–142 / Stage A prepare_path | 閉じたforward headの末尾B_H → tail先頭runのB_T†。間のmatrixはB_T†B_H | head centersはuncontrolled、tail centersはordinary。**basis自体は両方uncontrolled** | forwardのscalar_time=0。tailのevent phaseはancilla P | 横断relative呼出しなし。両basisは末端列として隣接し、同じsystem wiresを使う。異frameのrelative化が追加候補、P0残差UNKNOWN |
| tail→最後のhead：C3 prepare_steps 142–143 | tail最終runのB_T → reverse head入口B_H†（通常block2） | tail ordinary、reverse head directional。basisはuncontrolled | tail末尾のP(event phase)はsystem Gaussianと可換だが、削除・system global化は不可。path scalarは別集約 | 横断relativeなし。phaseを保った配置確認が必要。初回人工提案にはこの向きを追加しない |
| step間：tailᵢ→reverseᵢ→forwardᵢ₊₁→tailᵢ₊₁ | mixed_head(boundary)は2/1/0/1/2のframe列。tail同士はheadを挟む | directional半0＋uncontrolled半0はordinary full0へ。その他は元mode | path scalar/identityを末尾Pへ一回集約 | 中央half-sweep融合とhead内部relativeは**実装済み**。両外端のtail↔headは未処理。既存q−1融合をC1利益として数えない |
| ReferenceのB内部：physical_segment | forward、各control_one(tail)、bridge/backwardを閉じて結合 | すべてphysical。control_oneでancilla=1のbase gateへ制限 | 元P等をglobalへ移す既存規則＋segment scalar。任意のP/RZを落とさない | head/tail同様のsystem接口あり。内部step融合は既存builderで実装済み |
| ReferenceのA内部とA†：physical_segment→reference_gadget | Aを閉じて構成後、**A.inverse()全体**を別laneへ配置。A†内では接口順も反転 | physical、CSWAPの制御は別 | Aのdeclared phaseもactual inverseで反転。χはwhole pathに保持 | Aへの局所同値書換え後にactual inverseを取るなら対象を保存できる。A†専用に新しい「逆っぽい」列を作らない |
| Reference global cut：prepare_steps 146–148 | earlier=Bとlater=Aは別register。gadget内でA†/Bを二CSWAP cascadeが挟む | CSWAP-controlled routing | segment phase、P(χ) | 内部step融合は計q−2、cutで1境界分を失う既存条件。**別laneのB_endとA_startを隣接Gaussianとして融合不可**。cut変更・gadget変更は今回の候補外 |

## 既存機能との境界

`StrongBuilder.transition`はhead内部でB_j†B_iを既に合成し、`mixed_head`はmodeを分けたままGaussianを共有する。tailは同basis runを共有済みである。repeated builderの`_build_boundary_optimized`も、隣接する同block0 halfをfullにしscalarを集約する既存経路。C3のmixed-mode版をこれで置換したり、両方の節約を加算したりしない。

rawはbarrierを新設せずfull-wrapper P0へ渡される。したがって、SOURCEで閉じていることはcompilerで閉じたまま残ることを意味しない。[P0分析](P0_OVERLAP_ANALYSIS.md)の局所重複と全接口UNKNOWNを区別する。
