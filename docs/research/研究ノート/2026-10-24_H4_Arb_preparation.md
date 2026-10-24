# 2026-10-24 H4 Pauli central-control Arb準備

基点36e993e698953f24415f4e90c4cac199a693ee16、専用branch
`codex/non-df-h4-central-arb-preparation-20261024`。
今回承認は技術実装・人工検証・実行準備のみ。

既存Cのexact変換/Arb/端点、R0のQPY/fingerprint、R1 sineの人工検証を再利用した。
新native sparse vector kernelはexact center＋unitary L2 radiusを保持し、長い列のwrappingを抑える。
新10＋6＋再利用9＝25人工PASS。3qubit49,000gateの人工恒等列は64/96/128bitすべて既知signal=1を包含、
幅9.42654e-15/2.25642e-24/5.11760e-34。実H4への幅や時間の外挿はしない。

実PF cosine QPY1個を構造decodeし、48,966 native gateのexact binary64 descriptorを保存した。
実H4 Arb評価・状態作用・grant・compile/sampling/N_Gは0。
PF C-targetはR0保存区間全体を引用。PR個別C-targetはまだ経験値であり、包含経路の設計だけを記述した。

無効pilot案はPF cosine1局所、64→96→128bit最大3状態作用、新target/input作用0。
幅1e-8は設計目標であり、旧allowanceや閾値ではない。budgetの関係が決まれば最初の精度でSTOPする。
既存ρ/PR全law/tail/C-L/旧N_G/元nativeなどの未解決状態は保持する。
§10.3は提供された最新版に存在せず、実際の§8.4/10/10.1/10.2と利用者指示を適用した。

次：ユーザーが最小pilotの範囲を明示承認後、Codexがproduction SOURCE/controllerを固定して限定評価・終了監査。
今回の契約は無効、科学grantなし。mandatory STOP。
