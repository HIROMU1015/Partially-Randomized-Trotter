# Architecture

bridge は Storage V2 `Coordinator` の versioned subclass を一つだけ生成する。S/P1 は共通の48-task plan、uint8 statefile、RAM shadow、global counter/cursor/ticket、byte budget、ordered trace、control journalを共有する。Sの3 phaseをjoin後、同じcoordinatorのsettingsをP1へ切り替える。条件ごとに新しいclaim・budget・statefileを作る構造ではない。

worker/F01/Stream/fused codec/CachedObserver/coalesced clientは固定SOURCEを再利用し、既存moduleは変更しない。新bridgeはcondition begin/end、global trial identity、global offset、PHASE_ABORT、post-close保存とcaller観測を追加する。S/P1各24pairはfresh spawn child。Sは逐次、P1は既存8-pair構造。固定順S→P1で科学反復1候補、順序反転は含めない。

```mermaid
flowchart TD
  A[単一 fake atomic claim: SPENT] --> B[単一 global coordinator / budget]
  B --> C[S: correctness → baseline → instrumented]
  C --> D{S完了}
  D -->|yes| E[P1: correctness → baseline → instrumented]
  D -->|no| X[STOP / drain / 部分証拠保持]
  E --> F[全worker join / summary / diagnostic]
  X --> F
  F --> G[fake Admission STOP / snapshot]
  G --> H[trace・state・journal close]
  H --> I{全条件と保存証拠が完全}
  I -->|yes| J[global receipt close → marker close]
  I -->|no| K[失敗・不完全の final evidence]
  J --> K
  K --> L[final slot → delivery slot → caller observation]
  L --> M[Producer-free offline join]
```

runtime SOURCE と科学参照/input/runtime bytesを起動時・phase境界・worker setupで照合する。call entryのpermitは親RPCが唯一のgate。consumerはbridge/Producer/scheduler/Admissionをimportせず、旧global/F01/cost primitivesと新trace判定を使う。
