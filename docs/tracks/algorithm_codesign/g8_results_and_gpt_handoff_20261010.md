# Track B G8：on-demand経路・条件付きprovider予算の結果とGPT引継ぎ

技術状態：`G8_ON_DEMAND_PATH_AND_CONDITIONAL_PROVIDER_BUDGET_COMPLETE`。
一束の取得は完了し、**mandatory STOP**。新規性・主method・次stageの科学判断はGPT/利用者へ戻す。
G7の既知development条件だけで、非列挙local generatorからlive Rz miss取得へ接続できた。
物理controlled-Q providerは選択・取得していない。条件付き予算と短い実装traceの確認であり、
一般入力でのnative性能、独立validation、immutable CI、PR/QPE総costの結果ではない。

## 1. 固定source・認可・読む順序

- branch：`track-b-g8-on-demand-provider-budget-20261010`
- source S：`cb60a4a1336f1803ad49d890f4e74aafd6ad7c61`
- base/G7結果commit：`a689694080f4b7600fe67cf77d841d1cbbf04503`
- [採用GPT G7 review](../../research/track_b_G7_scientific_review_20261010.md) §11.1–11.3が
  proof/contract/on-demand経路の技術設計・限定取得を委任。別の研究条件・providerの採択ではない。
- [結果前proof/scope](g8_proof_contract_and_on_demand_scope_20261010.md)と
  [contract](../../../artifacts/track_b_g8_on_demand_preparation/2026-10-10/contract_v1.json)。
- [source manifest](../../../artifacts/track_b_g8_on_demand_preparation/2026-10-10/source_manifest_v1.json)、
  [入力review identity](../../../artifacts/track_b_g8_on_demand_preparation/2026-10-10/review_input_identity.json)。
- [raw結果](../../../artifacts/track_b_g8_on_demand_result/2026-10-10/v1/result_v1.json)、
  [保存値監査](../../../artifacts/track_b_g8_on_demand_result/2026-10-10/v1/saved_output_audit.json)、
  [display CSV](../../../artifacts/track_b_g8_on_demand_result/2026-10-10/v1/resource_summary_display_v1.csv)。
  表/CSVの小数は表示用。exact rational・sequence identity・timing詳細はraw結果を正本とする。
- marker SHA256：`8925877915462422258a9f16fe7e72aa190a3211309a428734888b8832b3e1e8`。
  exclusive marker作成後、runner一回、retry0。元G7のmarker/result/authorization/STOPは保持。

入力はP3 control `p=(3/7,4/7), x=2/5, m=3`、P5 general-order
`p=(1/5,3/10,1/2), x=5/7, m=5`。
`ordinary / partial_return_tail / closed_P3_tail / full_return` の同じ4構成を比較。
finite Taylor target `P_m(-i x sum_i p_i Q_i)`、`Q_i²=I`、bit H=160/root K=256、eta=rho=10^-12、
Rz strict error 10^-6と既存backendを維持。分子geometry/basis/DF rank/splitは非該当。
P3/P5の差はdegreeだけの因果比較ではなく、異なる既知p/xの2例。

## 2. on-demand経路で取得した証拠

productionにはevent全表、全angle key表、旧G7 sequence、global `B_new`を渡していない。
local生成 → pre-quantum zero → 符号付きtangent → live missのstrict Rz取得 →
actual adjoint/relative phaseを保持したconditional provider IRへ接続した。
全production完了後にだけ、旧G7保存表をreference診断へ読み込んだ。

8 row × cold128/warm128 = **2048 interface trials**。これは固定SHA256 bitstreamのinterface確認で、
量子測定、Hamiltonian trajectory、信頼区間検証用のsampling campaignではない。
frequencyをm2、N、期待costの推定へ使っていない。
20 unique positive keysを各一回取得し、全strict phase-sensitive error guardが通過。
20 sequence hashはそれぞれ既存G7と一致した。同じbackend/seedによる取得であり独立再現とは扱わない。
rawの`runtime.synthesizer_calls=0`はidentity preflightの値で、実取得数はtop-levelの20。

| input | arm | N/axis（未実行） | accepted cap/2 axes | hard attempts/2 axes | cold accepted/zero | cold row miss | warm row miss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P3_control | ordinary | 707157 | 1414314 | 1414314 | 128/0 | 2 | 0 |
| P3_control | partial_return_tail | 611747 | 1223494 | 1223494 | 128/0 | 2 | 0 |
| P3_control | closed_P3_tail | 610330 | 1220660 | 1220660 | 128/0 | 2 | 0 |
| P3_control | full_return | 657191 | 1225940 | 1314382 | 119/9 | 3 | 0 |
| P5_general_order | ordinary | 1195372 | 2390744 | 2390744 | 128/0 | 3 | 0 |
| P5_general_order | partial_return_tail | 910541 | 1821082 | 1821082 | 128/0 | 3 | 0 |
| P5_general_order | closed_P3_tail | 895933 | 1791866 | 1791866 | 128/0 | 5 | 0 |
| P5_general_order | full_return | 1032322 | 1788076 | 2064644 | 108/20 | 6 | 0 |

row LRUは全構成共通8 entries/128 KiB、全bundle共通acquisition memoは32 entries/1 MiB。
共有memoを空で開始し、実際は20 entries/19,053 serialized bytes、26 requests/6 shared hits。
row cacheは最大6 keys、eviction0。warm digestは全rowでcoldと一致、warm miss0。
LRU eviction/failure-poison/容量拒否はfocused synthetic testsで確認したが、実traceは容量stressではない。
serialized byte capはPython heapサイズとは別で、全体RSS capも適用。

各rowのcoldはrow cache空から始める。共有memoはbundle内で対称に再利用するため、
各armの独立cold取得費用は**観測済みunique keyの取得費用を再集計したcharge**であり、
同じkeyを再合成して測ったcold実験ではない。
全support/allanglesは確認していない。P5 fullで観測した6 keysも旧G7 supportの一部に留まる。

wall **1.860446 s**、CPU **1.860393 s**、peak RSS **180,776 KiB**、一process。
上限wall1200/CPU900 s、RSS512 MiB/AS1536 MiB、per-key wall30/CPU20 s、miss32、
sequence20,000 chars/output16 MiBに到達せず、technical failureなし。
local GF/root/queryとratio算術は合計bucketで記録。独立のGF/root microprofileではない。
row cold/warm、key生成、cache、miss取得、provider descriptionのtimingをrawに保存。
この短いtraceの秒数から大規模classical scalingやwall性能優位は結論しない。

## 3. finite-provider仮定とconfidence/resourceの分離

controlled-Qがunitaryで、strict joint operator error≤delta、actual adjointとrelative phaseを保持すると仮定。
**delta=10^-6は仮想誤差parameter**。物理providerがこの誤差を達成したという証拠ではない。
`T_Q_i / CX_Q_i / 1Q_Q_i`は未指定非負変数で、prep/readoutは旧G7条件を保持。

各eventの誤差は `2 epsilon_Rz + sum_i n_ei delta_i`。
coherent biasはproposal query頻度ではなくtarget係数weightから戻し、
`b_delta = 2[3 rho + 3(1+rho)(2 epsilon_Rz+(m+1)delta)]`、
`epsilon_axis=1/200`、`s_delta=epsilon_axis-b_delta>0`とした。
許容delta ceilingはP3約0.000207833333、P5約0.000138555555。
今回のs_deltaはP3約0.004964、P5約0.004952。

G8の新failure配分は16 estimator axesで0.049、8 resource rowsで0.001、合計0.05。
`alpha_axis=49/16000`、`beta_row=1/8000`。unused resource予算は再配分しない。
旧G7のfailure配分・hard capを結果後変更していない。

`M=2N`、非列挙acceptance upper `z_bar`、`v=M z_bar`から
`min(M, ceil(v + sqrt(2 v * 9) + 2*9/3))`をaccepted-call capとした。
`exp(9)>8000`はexact rational Taylor partial sum、sqrt/logはoutward intervalで確認。
P5 full cap **1,788,076**はclosedの1,791,866より小さいが、
hard attempt **2,064,644**はclosedより大きい。accepted count capはT cost capではない。
量子shotsは実行していない。uniform-bit/provider仮定の下のconfidence/resource契約で、
合成backend失敗は別technical abort。全angleでの合成成功確率やterminationは保証していない。

reviewにあった旧G7のcoarse `z≤0.864,t=7,M=2,028,644`計算は独立式で
**1,757,707**と一致。これは数理self-checkで、旧G7登録resource上限の変更ではない。

## 4. 保存G7費用の条件付き再予算

G7保存per-trial Rz価格/provider query係数にG8のNを掛けた感度確認。
新physical provider費用の取得、量子resource全実行、元G7再分類ではない。
2 axesのRz Tとprovider係数（ラベル順）のexact値はraw結果に保存。
`G_T = T_Rz + sum_i provider_coefficient_i * T_Q_i + K * T_prep`。
`K`は表のexpected accepted calls、`T_prep`は共通の一回あたりprep/readout費用で、総prep費用は構成ごとに異なる。
T_Q_iを0と置くRz-only端点は補助診断で、現実のcontrolled-Qが無料という主張ではない。

| input | arm | Rz T/2 axes | provider coefficients（ラベル順） | expected accepted quantum calls |
| --- | --- | ---: | --- | ---: |
| P3_control | ordinary | 198,003,960.000 | 1,260,560.130, 1,664,649.844 | 1,414,314.000 |
| P3_control | partial_return_tail | 171,289,160.000 | 1,093,629.272, 1,443,198.987 | 1,223,494.000 |
| P3_control | closed_P3_tail | 170,714,098.938 | 1,094,955.447, 1,435,515.084 | 1,220,660.000 |
| P3_control | full_return | 170,754,947.134 | 1,095,217.446, 1,435,858.571 | 1,220,952.078 |
| P5_general_order | ordinary | 327,368,694.046 | 1,099,650.052, 1,621,942.564, 2,613,555.133 | 2,390,744.000 |
| P5_general_order | partial_return_tail | 242,737,873.171 | 853,566.070, 1,256,315.798, 2,015,575.260 | 1,821,082.000 |
| P5_general_order | closed_P3_tail | 251,054,974.049 | 857,916.918, 1,246,043.901, 1,952,290.958 | 1,791,866.000 |
| P5_general_order | full_return | 241,470,947.527 | 840,026.636, 1,222,042.580, 1,919,143.054 | 1,770,981.710 |

P5 fullは3対照よりRz Tも各provider係数も小さい。P3 fullはclosed対照よりRz T/各provider係数が大きい。
G7の符号付きaffine差・K ratio×per-call cost factorizationもexact保存値で照合した。
この条件付き関係をgeneral DF/native総costや新method採択に拡張しない。

## 5. cost-aware補助診断と下界の限定

全production後の**I2列挙oracle**。旧G7 event係数/価格だけを使用し、各representationにつき一つの
`q_e ∝ alpha_tilde_e / sqrt(C_Rz,e)`候補をH160/K256で構成。
同じfinite rational係数をexactに再weightし、実際のrational m2/rangeと共通Bernstein Nを再計算。
production sampling lawは変更していない。oracleの全表access/取得は無料扱いしない。

| input | representation | oracle N/axis | oracle Rz T/2 axes | fixed-policy Cauchy leading lower |
| --- | --- | ---: | ---: | ---: |
| P3_control | ordinary | 707157 | 198,003,960.000 | 197,439,589.650 |
| P3_control | partial_return_tail | 611747 | 171,289,160.000 | 170,764,035.604 |
| P3_control | closed_P3_tail | 610336 | 170,713,270.482 | 170,189,101.018 |
| P3_control | full_return | 610336 | 170,713,270.482 | 170,189,101.018 |
| P5_general_order | ordinary | 1195782 | 327,407,968.637 | 326,612,794.345 |
| P5_general_order | partial_return_tail | 911011 | 242,758,707.143 | 242,073,990.858 |
| P5_general_order | closed_P3_tail | 896218 | 251,094,683.335 | 250,398,296.435 |
| P5_general_order | full_return | 879918 | 239,936,891.036 | 239,306,711.180 |

Cauchyから`m2 * E[C_Rz] ≥ (sum_e alpha_tilde_e sqrt(C_Rz,e))²`。
固定Bernstein十分shot policyの2-axis費用には
`4 ln(2/alpha_axis) (sum_e alpha_tilde_e sqrt(C_Rz,e))² / s_delta²`が下界となる。
表のlowerはsqrt/logのlower endpointを使用。
**同representation・同rational係数・保存Rz価格・共通bias・固定policy**内の下界であり、
物理shot下界、任意precision/辞書/手法の最適性、native provider込みのIS下界ではない。

この候補はleading m2×costの最適化則。range/ceil込み有限Nのglobal最適解とは言わない。
実際、ordinary/partialのoracle候補は元samplingより僅かに高いRz Tとなる場合がある。
P3 full/closedのoracle結果は一致し、両者の残差はproduction予算/取得accessにある。
P5 local fullのRz T **241,470,947.527**は、登録3対照の固定policy lowerより小さい。
最も近いpartial lower **242,073,990.858**との差は約603,043.331。
一方、full自身のI2 oracleは239,936,891.036でlocal fullより小さい。
これらは限定された保存値の原因帰属で、全資源の独立優位や新規性の判定ではない。

保存値監査はlowerとのexact scalar比較とflagsを照合した。
STOP後にoracle全event certificateを再生成していないため、完全な独立再導出済みevent証拠とはしない。
根拠は固定source、focused形式test、保存scalar出力である。

## 6. 検証・保全・残る判断

source固定前のfocused **17 tests PASS**（off-domain、native stub、science取得0）。
一束終了後の保存値監査 **14項目PASS**（source54 hash、旧918 path/prefix、marker、sequence、
独立90-term logによるN照合、accepted tail cap、cache trace、failure配分、scope/resource）。
[source preparation test記録](../../../artifacts/track_b_g8_on_demand_preparation/2026-10-10/focused_tests.json)。
raw結果/marker/contract/sourceを保ったまま本handoffを追加した。
rootの未commit資料・Track A worktree・既存artifactは編集/移動/再生成していない。

新p/x/m/dictionary/provider、LP/v4、DF/molecule/NPZ/GPU、whole-circuit compile、量子測定、
Hamiltonian trajectoryはいずれも0。one-shot後の追加generator/synthesis/solver取得0。
next science unauthorized。新materiality閾値・baseline・methodの採択はない。

GPTへ返す判断事項：

1. known短traceの非列挙local/on-demand経路を、次の研究判断に足る実装証拠とみなせるか。
2. 未指定provider、未観測angles、failure abort、global取得/heapを含む限界の下で、次の実証に情報価値があるか。
3. 同representation IS oracle、固定policy下界、local budgetとの分離から、どの狭いclaimが残るか。
4. 物理providerの指定や拡張input/比較が必要か、または現候補を縮小するか。

**本資料は次stageを認可しない。mandatory STOPを保持し、研究方針・RQ・新規性・論文着地点はGPT側へ戻す。**
