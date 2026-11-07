# Track C：改良PRの資源ボトルネックと追加改善

基点 `b9dacb639eabdd3cf499e89f8807ff50b68e23ef`。専用branch `codex/track-c-resource-bottleneck-20261108`。

**PRの資源は削減できたが、強Full PFには届かなかった。** 保存された折返し非依存の改良PRから、独立価格holdoutで総RZを46.52%、CXを49.44%削減した。同じ統計配分最適化を与えた強Full PFに対しては、依然RZ約1.96倍、CX約1.27倍である。主因は、長roundのRTE basis変換とnormalizationによるshot増加の組合せだった。科学実行を終了しmandatory STOP。詳細GPT科学レビュー・別研究は開始していない。

## 条件・再利用

Linear H4、1.00 Å、STO-3G、保存DF12・generation order、Partial LD3、Full LD12、保存元Hのexact ground、状態準備除外。18round、δ0=.02、energy精度1.5936e-4 Ha、失敗上限.05、総位相予算.4 radを維持した。B2 P1の入力・compilerは使用していない。

Qiskit1.3.0の**Track C compiler A**を使用する。rz/sx/x/cxへdefinition loweringし、微小回転を落とす数値最適化・cutoffを使わない参照profileである。既存profileのseed17を保持した。旧opt2価格やB2のopt1/seed7と混合していない。Python3.12.3、NumPy1.26.4、SciPy1.14.1、mpmath1.3.0。CPU affinity64の独立worker、BLAS1で内部oversubscriptionを避けた。観測MemAvailable約476 GiB。共有設定・他ユーザーのprocess/fileは変更していない。

再利用したものは以下である。

| 既存成果 | 今回への適用 |
|---|---|
| C-N / native価格 | 強加法baselineとfullが一致した履歴を踏まえ、加法式を書き換えただけの改善とはしない |
| C-S / C-P | 大きなrange/shot負担・ほぼ無改善という負結果を保全。sampling/return aggregationを変更しない |
| C2-A / C2-B | round依存PF、誤差配分、強baselineの比較機会を再利用。.51%を強baselineへの優越としない |
| D内部PF study | 11988信号、強Full追加7452信号、D価格410セル、R価格33セル1056標本、219条件のnative資格、BCHで固定したD調整係数を再利用 |
| compiler A / 18round保証 | 全support・全basis contextの局所区間算術、source/finite-bit/phase/alias会計を継承・必要角度のみ拡張 |
| 旧長round価格・WP06 | 未資格な短whole fitを移送しない。既存singleton原理をstable partial-column QRで実装し、実occurrence価格を測定 |

旧[WP06a](../../../artifacts/research_direction_structure_pilot/2026-09-22/wp06a_circuit_structure_pilot_v1.json)、[WP06b](../../../artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json)の保存結果を参照した。Gaussian/Givens・部分isometryは既知の手法であり、新公式の発明とは扱わない。[Kivlichanほか](https://arxiv.org/abs/1711.04789)、[Jiangほか](https://arxiv.org/abs/1711.05395)。今回は選択した列を個別に保持するQRであり、Slater determinantのoccupied-subspace自由度をそのままZ/ZZ演算子へ流用したものではない。

## 保存全18roundの費用分解

以下は今回の新最適化前、各軸α=.05/36の出発点である。RZ/CX/sizeは、実行可能なmodular native circuitの加算価格。D/D隣接融合のsigned補正を**既にDへ含めた**。wrapperには決定論scalar相対位相、Hadamard、YのSdg、readoutを含む。

| policy | 全shots | D RZ | RTE RZ | wrapper RZ | 総RZ | D CX | RTE CX | 総CX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 元PR・同じmenu | 11150 | 2.27239e12 | 2.17157e12 | 61325 | 4.44397e12 | 7.00513e11 | 3.18772e11 | 1.01929e12 |
| 改良PR・出発点 | 11572 | 1.45233e12 | 1.89612e12 | 63646 | 3.34845e12 | 4.37572e11 | 2.84086e11 | 7.21658e11 |
| 強Full PF | 8828 | 1.20943e12 | 0 | 48554 | 1.20943e12 | 3.80934e11 | 0 | 3.80934e11 |

元PRの値は今回と同じmenu対照であり、以前の「登録済みr_m固定・8598shots」の4.7792e12とは別である。

改良PRはD/RTEがRZの43.37%/56.63%。総RZのm17寄与72.2%、m16約19.0%、m15約6.4%で、最後2roundが約91%。Fullのm17は約60.4%、m16約24.0%。全18round・全4指標の値は[ROUND_COMPONENTS.csv](../../../artifacts/track_c_bottleneck/2026-11-08/ROUND_COMPONENTS.csv)。

局所templateを用いた加算整合の内訳は次のとおり。数値は改良PR出発点のRZである。

| group | basis変換 | uncontrolled対角演算 | control追加分 | event相対位相 |
|---|---:|---:|---:|---:|
| D | 8.76919e11 | 3.15659e10 | 5.43842e11 | D対角演算に包含 |
| RTE | 1.73376e12 | 6.77954e9 | 1.55582e11 | 1552程度 |

RTEの約91.44%がbasis変換。D/Rで露出する両側basis変換は合計6.58320e10 RZ、総RZの約1.97%であり、この値はbasis費の**部分集合**である。独立した境界費として足し直していない。D/D融合はRZ −2.71769e10、CX −3.51185e9の補正を既に含む。孤立template和と実価格の差はsigned compiler/context残差として保存した。Depthの残差にはDAG schedulingも含み、ゲート相殺だけとは解釈しない。[全成分CSV](../../../artifacts/track_c_bottleneck/2026-11-08/TEMPLATE_COMPONENTS.csv)。

## 感度と優先順位

固定shotでDを100%無料にしてもRTEだけで1.8961e12 RZ、RTEを100%無料にしてもDだけで1.4523e12であり、出発点のFull 1.2094e12を下回らない。D/R露出basisを全て消してもRZ減少は約1.97%。これは実現可能性や一般的下界を主張するものではなく、**現在のshot・構造を固定した反実仮想**である。

同じPF/statistical条件のままnormalizationだけを理想的に除く感度では、改良PR総RZは1.11853e12へ減る。一方PF位相biasだけを0にする感度は3.23473e12、約3.40%減に留まった。他誤差diskを0としてもこの整数shotでは変化しなかった。これらを実現したアルゴリズムとは扱わない。[SENSITIVITY.json](../../../artifacts/track_c_bottleneck/2026-11-08/SENSITIVITY.json)とrawのDECOMPOSITIONを参照。

実現可能な順序は、(1)両armのcost-sensitive α、(2)RTE演算子を保つbasis構造削減、(3)D stepと既存RTE resolutionの共同選択、とした。RTE sampling-lawの式、component確率、return aggregationは変更していない。resolutionの変更は、同じcanonical K2/P3を何回に分割するかという既存パラメータの変更である。

## 実際に検証した改善・負結果

1. **統計配分**：既存`shot_rows`のKKT配分を再利用。36軸のα和をexact rationalで.05以下に調整し、192-bit区間算術で整数Nを再計算した。出発点の同じsource点のままPR RZは3.34845e12→2.40298e12、Fullも1.20943e12→9.13529e11。総shotsは増えるが、高価な長roundのshotsを減らせる。
2. **always-partial QRは負結果**：Z/ZZに必要な1/2列だけを作る隣接Givens構成は8モードで最大7/13回。単一成分は安くなったが、全applicationを置換すると長い同一basis runの共有を失い、m17代表cellのRZは1.682倍、CXは1.509倍へ悪化した。64 paired標本と全218成分検査を保存し、この方式を棄却した。
3. **singletonだけを置換**：元の最大same-basis runが長さ1の箇所だけ部分QRを使い、長いrunは全basis共有を保持する。元回路と提案回路をそのtrajectoryごとにcompileし、**実RZが小さい方の回路全体**を採用する。CX等を別々のminで組み合わせていない。保存1056標本で比較したm17の元cellではRTE RZ約25.65%、CX約18.60%減。分布・event順序・符号・相対位相・normalizerは保持する。
4. **D step/resolution共同選択**：既検証のh menu・D係数を再利用し、支配的なm17にresolution64を追加。4信号、3occurrence cell・96新標本だけ追加した。新source点と新角度8条件を資格化した。最終m16はD調整0321/h=.32/resolution32、m17は同じD/h=.32/resolution64。m17は8192 D/R step、8388608 RTE event、logB≈.27869543、rho下限≈.756675、βstat下限≈.366218となった。追加RTEを無料にしていない。従来m17のnormalizer/logB≈.55739を減らす一方、RTE回路数は倍になるtradeoffを直接評価した。
5. **Fullにも同じbasis機会**：13 D blockのexact nonzero対角supportを調べ、同じ部分QRを提示した。全blockで8列が必要で、RZ/CX削減は0件だった。Fullを不利な全basisに強制して新手法の利益を作っていない。
6. **D/R境界のwhole確認**：元S2、調整0123、最終調整0321で1/2step・X/Yをwhole compile。全12件でwholeとmodularのRZ/CX/size差0。追加の境界融合効果は未観測なので、そこへの大規模探索は行わなかった。これは全長whole最適性の証明ではない。

既存D調整係数の再fitや、wrapped endpointを狙ったh/係数調整はしていない。全armに`|ΔE_dominant|T+asin((1-p)/p)<.4`を同じく適用した。この折返し診断自体はfloat64で、以下の区間native保証と区別する。選択はsource/native資格のある222点を使う有限menuの交互最適化であり、global最適性は主張しない。

比較機会として、Fullは既存S2/Yoshida4・同じh候補・最初の4 blockの24順序と今回のα配分・部分QRを受ける。PRは既存の元/再順序/再配分/ground-BCH調整を受け、今回resolution64を追加した。全13!順序やFullのground-specificな新係数まで探索した比較ではない。したがって「全PFに対する最適資源差」ではなく、実行可能な強Full対照がなお安いという結論である。

## 独立holdout・全RPE資源

候補・α・Nを固定した後、使用する12 RTE cellで各2層×16、**384独立価格標本**を生成した。holdout値で候補を選び直していない。下表のPRはそのholdout、Fullは決定論価格である。括弧内は1 MC標準誤差。

| RZを目的にしたpolicy | 全shots | 総RZ | 総CX | serial depth上界合計 | size合計 |
|---|---:|---:|---:|---:|---:|
| 追加改善PR | 17054 | 1.79060e12 (1.69635e10) | 3.64850e11 (1.85495e9) | 8.85249e11 (6.21888e9) | 2.68559e12 (2.44002e10) |
| 同じα最適化後の強Full | 14228 | 9.13529e11 | 2.87925e11 | 5.98001e11 | 1.43616e12 |

PRのtraining推定はRZ1.78119e12で、独立holdoutとの差は約.55 SE。固定した同じsource点・同じNで全basis実装をpaired比較すると、singleton選択による削減はRZ4.33865e11（差SE9.06979e9）、CX4.69475e10（差SE9.82432e8）。共有cellを使う全roundの重みをまとめてから共分散を計算し、paired差と各armのSEを混同しない。

CXを目的とする別policyも保存した。PRは総RZ1.79019e12、CX3.63745e11、16534shots。対応FullはRZ9.12635e11、CX2.87625e11。異なる指標を加算せず、都合のよい列だけを別policyから採って勝者とはしていない。

最終RZ policyのholdoutではD=5.25635e11 RZ、RTE=1.26496e12 RZ。RTEが約70.6%となり、残存ボトルネックはさらに明確になった。m17の費用寄与は約62.7%、m16約21.9%、m15約10.5%。[FINAL.json](../../../artifacts/track_c_bottleneck/2026-11-08/FINAL.json)、[全round CSV](../../../artifacts/track_c_bottleneck/2026-11-08/FINAL_ROUNDS.csv)に全4指標を保持した。

各RTE occurrenceを実際の長さ・分布で直接compileし、コンパイル済みD区間と連結する**具体的modular実装**を価格対象とする。RZ/CX/sizeの長round加算にwhole fitの外挿誤差はない。有限MC誤差と科学モデルの条件は残る。Depthは部品depthのserial上界であってm17の実DAG depthではない。短回路のDAG depthはrawのshort_checksで別に保存し、serial上界との差も残した。m17全回路の直接compile、T合成、状態準備、routing/noise、量子shot実行は行っていない。

## Correctness・保証と限界

全218成分のFock involutionは、数値Frobenius誤差最大1.981e-14。192-bit区間算術で元/部分QRのone-particle列を比較し、4電子sector上のinvolution差を最大3.067e-14で囲った。新basis角度の128 context、3394 unique native loweringを検査し、basis pair誤差最大2.110e-14。native中間gateをsector射影して誤差を小さく見積もっていない。

3×RTE event数をapplication上限とする全trajectory一様な置換diskを、既存source/native/finite-bit/normalization予算に保守的に追加した。長いrunやfull回路へのfallbackもこのdiskが覆う。最終PR18roundの最小rho下限≈.572700、最小βstat下限≈.366218で全予算が閉じた。m17追加disk上界約3.684e-6。RTE normalizationは保持し、raw複素信号へのdiskとして半径・位相へ換算した。

1stepのsource旧/新、whole native、modular nativeをfull joint512次元で比較し、X/Y測定差最大2.365e-13、状態projective差最大1.525e-12。許したのはjoint回路全体の共通global phaseであり、control対象Uの相対位相は保持した。

既存decoderへE0を渡さず全18roundのcanonical平均信号を復元し、branchが通過、energy誤差1.28402e-5 Ha。新実装のmeanは一様置換disk内にあり、追加誤差を含むrho/βからshotを決めている。α和≤.05、3×.4<π、`.4/(.02*2^17)=1.52587890625e-4 Ha < epsilon`により、**保存Hermitian-H・exact ground・ideal native・fresh独立finite bitsモデルの十分条件**を満たす。MC価格の標準誤差はこの成功率保証とは別である。元連続積分Hの無条件包含や実機性能の保証には拡張しない。

## 公開物・判断・次の仮説

全raw・負結果・中間・区間証拠を[RAW_DATA.tar.gz](../../../artifacts/track_c_bottleneck/2026-11-08/RAW_DATA.tar.gz)に元byte列で保存し、[RAW_INDEX.json](../../../artifacts/track_c_bottleneck/2026-11-08/RAW_INDEX.json)で照合する。完全SOURCE SHAと入力hashはFINAL.json、実装は`src/trottertracks/track_c_bottleneck*.py`。生成は同名moduleの`python -m`入口、読取検査は`python3 scripts/check_track_c_bottleneck.py`。新規scientific runは今回STOP後に自動再開しない。private common cacheは公開せず、以前の公開DF/state参照・hashを継承した。旧SOURCE/raw/grant/ledger/marker/STOPは変更していない。

人工testは部分QRと既存Dの時間/BCH/modular分割の4件PASS。検証はCodex自身によるlocal検証であり、外部再現・別reviewerの独立レビュー・CIとは呼ばない。

残るTrack Cの有望な仮説は、短いmulti-application same-basis runの**列集合を保つ共同isometry**で、共有を失わずbasis費をさらに下げられるかである。これは今回未検証。旧WP06の一般的interval-DPが有効だったとは解釈しない。またRTE resolutionの連続的な最適値・より大きいhで位相を保つground-specific D公式も未確定だが、今回の数字だけでFull逆転を予告しない。exact ground利用可能の仮定を障害とは扱わない。

normalization自体を改善するsampling-law/return aggregationの変更はTrack Bとの担当調整を要するため実行していない。**次の担当はユーザー**：上記のTrack C共同isometry仮説を次に調べるか、Track Bとの共同課題へ移るかを選択する。**Codex**：次の明示指示があった場合に、その固定範囲を検証する。今回の作業は成果公開・SHA/tree/内容照合まで行ってmandatory STOP。
