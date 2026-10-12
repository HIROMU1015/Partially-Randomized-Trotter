# E-r3 identity-only v2：一回実行authorization

2026-10-12 JST。

最新の利用者指示の原文：

> 次の作業に進んで

直前の完了報告は、observer修正完了・77 focused PASS・READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATIONであり、次の作業をS2への明示one-shot承認と提示していた。この最新指示を、その限定診断を開始する文脈上の指示として記録する。前回の同文指示はobserver修正だけを認可した別turnであり、今回の根拠へ流用しない。

固定source S2：`b9519c29b5f1528ad5c8a6e398b2fa12062db8c6`。公開確認`e3b18413aa56748bb926c0992b2bf83acfc24198`は実行HEADにしない。S2の直接子authorization-only Aから開始する。

diagnostic_execution_authorized=true、science_execution_authorized=false、runs=1、retries=0、mandatory_STOP=true。保存E-r3 uncontrolled、current→legacy各一回、contract/manifest/runtime/expected/caps不変。

登録diagnostic runnerを一回だけ呼ぶ。partial failure・cap・exception後もretryしない。CD、E-r6、controlled、native transpile body、Pauli費用、T合成、sampling/trajectory/LP/DF/分子/NPZ/GPU、旧one-shot再実行は認可しない。

終了後は保存値のread-only整合監査・結果/来歴資料のGitHub公開だけを行い、mandatory STOPする。旧科学結果・source・marker・authorization・terminalは保持し、原因や修正方式の研究判断は実測証拠とともにGPT側へ戻す。

contract SHA256：`12345d68a1d3d50869f2928d9a441073fce8e9e82f66c73856c6956686a5f488`。

manifest SHA256：`359c606b13a04dbba5991baa3415c3a9c762956a77891c170d3d1e019ac99e8c`。
