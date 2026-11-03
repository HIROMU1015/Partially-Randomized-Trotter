# H8長時間・高次PR：Stage A/B実行

2026-11-03。高速化率のbenchmarkは行わず、承認された実計算を進める。
旧rare-RTE、旧独立確認、H46研究のSTOPは維持する。新しい独立scopeであり、Stage Cは確認n・精度目標が未定のため実行しない。

## 固定条件とSOURCE

Linear H8、隣接距離1.00 Å、STO-3G、保存tol-only1e-8 DF rank28、16 qubit、spin sector4900、元state・fragment順序0..27。
T=1.6、ε=0.0005、状態準備P=0、費用はG=N_cos C_cos+N_sin C_sin。
入力identity・先行計画は[adapter準備報告](../h8_long_time_adapter_preparation_20261103/IMPLEMENTATION_REPORT.md)と[科学実行計画](../h8_long_time_adapter_preparation_20261103/SCIENTIFIC_EXECUTION_PLAN.md)。

Stage AのSOURCEは`d1272016592ff06c0567862f770ed796fd61b366`。
元SOURCE/input/raw/QPY/grant/ledger/marker/STOPを変更せず、専用cloneのprivate run領域に原結果を保存する。
内部path、seed実値、event列、private入力とQPYは公開しない。

## 実行範囲

Stage Aの第一batchは[固定提案一覧](../../../configs/h8_long_time/proposed_scope.json)の57候補。
PRはprefix7/8、S2 q8/16/32とm5 q1/2/4、R_outer32、K4の12候補。
PFはFull28、B0/B0I prefix15/16、S2/Y4/m5、q1/4/16の45候補。
新T full-H sparse reference、signed primitive、finite mean、bias/normalization、axis別headroom/Nを検査する。

Stage Bは探索cohort、n=9。各PR formula/prefixとPF method/formula/prefixで、Stage A評価済みの最小の適格qを初期価格対象とする。
直接価格と固定NによるGまたはheadroomから比較が未解決の場合だけ、許可domain内の次q等を検討する。
未取得価格はunknown。work proxyをcompiled価格・下界へ読み替えず、選択した候補だけの比較とする。
compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/target/topology未指定。
paired cos/sin価格、代表control/actual-adjoint/scalar検査、measurement付きcompiled RZ、希少order診断を保存する。

## 実行状態

Stage A第一batchは57候補PASS、適格30・不適格27・UNDETERMINED0。signed primitiveは855 block/time組×3 probeで最大差2.62e-14。新T full-H referenceはdirect/split-time照合をPASS。
PR S2はq8/16不適格、q32適格。PR m5はq1/2/4で適格。q/Nの差だけで資源優位は判定しない。
強いPF対照の適格性境界を補完するため、同じ計画domain内のY4 q8（5 method/prefix）、S2 q32（5）、B0 S2 q8（2）の12候補を追加実行し、計69候補が完了した（適格40、不適格29、判定保留0）。初期結果は[保存結果](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-03/stage_A_initial.json.gz)。
Stage Bはm5のPR prefix7/8（q1、各n9）とPF5群のm5 q1、計7候補・23 trajectory・46 axis価格を8 workerで実行中。まだcompiled価格・順位の結論はない。既存native APIのinstruction boundは正の整数が必要なため、無限大ではなく既存block_instruction_boundの整数を渡す互換修正をSOURCE `df93fbb88761b61c2691bf01f761ed20a7987d7f`へ固定した。人工/mock21件PASS。Stage Aのsource mapは別に保持し、科学入力やsignal値を変更しない。

今回の検査は主担当Codexによるもので、別担当者の独立科学レビューではない。
科学性能・全候補の最適性・population cost保証・最終全RPE costを主張しない。
