# PR＋RPE共同最適化への研究移行：統合調査と最小実行計画

2026-11-10 JST。**元の誤差・shot・価格・総資源pipelineは大部分を再利用できる。Track Dの固定CDF8条件では既に13-roundの条件付き会計まで接続済みであり、作り直さない。** 今回は静的調査・計画のみ。新しい数値最適化・shot再算定・総資源再集計・科学レビューは実施していない。

[統合対応表](INTEGRATION_MATRIX.md)／[次回実行案](NEXT_EXECUTION_PROPOSAL.md)／[固定出典とAPI](SOURCE_INDEX.md)。利用者の今回の指示に従い、構造評価器の追加開発から、アルゴリズム改善と元PR＋RPE資源評価の接続へ重点を移す。これは各科学結果・保証・候補の正式採択とは別である。

## 1. 今回の差分と既存調査の再利用

同種の[11月9日横断調査][CROSS]と、[元Track Aの後続調査][ASURVEY]が既にある。再棚卸しや既存候補の再実装を繰り返さず、次の差分を重点確認した。

- Track Dの価格検査器は、[c002153a][BRIDGE]でq4/q8の固定8 wrapper・全56資源を旧P0と再現した。これは有限価格検査用の技術資産として保存し、今回追加開発しない。
- [Closed P3共同選択][COPT]はh・resolution・D・α・整数shot交換を既に評価した。これを「共同最適化未実施」としない。最終値はrootではなくextended/FINAL.json。
- [新A Q3-C/D][Q3]では、保存canonical PR／B′／Reference-B1に対する適応readout、固定モデル内の340-shot十分保証、全18-round native費用まで後続接続した。旧調査のQ診断案を未着手として再提案しない。
- [元Track Aの新HP報告][HP]では旧5 STOP armを別SOURCEで再資格し、保存旧3 armと合わせ8 armのbranch/shot条件付きPASSを報告した。旧横断表の「5 arm未再資格」は最新状態に使用しない。旧STOPは保持する。
- Dのq64も[保存8 pathの完了結果][D64]がある。8828b298の既存会計がq≥64をproxyで扱った事実は変わらず、新q64値への差替え・再fit・再集計は今回行わない。

## 2. 元PR＋RPEで再利用できるもの

元の規範文書は[研究目的](../研究目的・研究課題.md)と[方法](../研究方法・解析手順.md)。固定DF近似を `H_DF=scalar+H_one+ΣH_l` とし、one-body correctionと既定prefixをD、残りのsigned involutory成分をRへ置く。元HとDF近似の差は別に会計し、両者を無条件に同一視しない。scalar・identityの一回会計、frame順、係数閾値、真のλ_Rとranking proxyの区別が必要である。

| 処理 | 既存資産 | 今回の判断 |
|---|---|---|
| 分割・finite RTE | df_partial_s2、df_rte_tail、rteのnormalize/distribution/attenuation | 再利用可能。対象DF・prefix・law・Kが変われば派生入力も変わる |
| PF・RTE誤差からshot | rpe_resource_accounting、Track D original_h_accuracy、B1/B4保存表 | 一般APIと固定元H adapterを使い分ける。元H groundだけからunit radiusを仮定しない |
| round内r/K探索 | optimize_rpe_short_round(s) | 既存実装あり。ただし直接範囲q≤4とproviderの制約を維持。長q探索へ名前だけで転用しない |
| 価格取得 | DFLevel5RCompiledCostProvider、保存D価格、Cの直接compiler A／MC | providerは実build/compileを呼ぶ。静的調査や保存価格接続で不用意に実行しない |
| 全round集計 | c1_stage3_accuracy.aggregate_totals、D B6b aggregate_registered、bc_total | D固定条件は接続済み。対象ID・scope・価格coverageを維持して再利用 |
| α配分 | rpe_allocation_sensitivity_validation、C bottleneck.allocate／joint_opt | 既存機構を流用する。後者は18round／.05／weighted条件へのhard-codeがあり、CDF8へそのまま呼べない |

short-round optimizerの一般signal model、own-DF用accuracy、元H operator誤差adapterは同じ保証ではない。保存結果を型だけ合わせて候補へ注入しない。再利用の具体的な関数・制約は[SOURCE_INDEX](SOURCE_INDEX.md)に示した。

## 3. Track Dの三つの接続

### 測定の正しさ

[B6b][B6B]のideal blueprintは、whole V=AB、actual A†、真空vの `Vv=e^{iχ}v`、正しいχ/scalar、強symmetricの `W0†W1=V`、共通X/Y規約を条件とする。同じ初期system状態に対し測定分布を保存できる。短いgadgetの単純q回反復や、N=4 sectorだけの一致からwhole vacuum条件を推定しない。

### 十分shot数

[B4-2][B4]は元H ground、固定CDF8 ideal IID/exact-native、B1のoperator誤差＋guard、alias/prior/decoder、13roundと固定ε・failureを結び、保存合計14,738 shotsを得た。これは**十分shot数**であり、最小値・必要十分・実際の成功頻度ではない。B6bは同じlaw・各軸内fresh independent trials・同じtarget/decoderを加えて両方式へ共通利用した。

law、weight、PR/PF、誤差配分、native誤差を変えればNの移送は別途必要。特にClosed P3のweighted trialは元の±1 unweighted平均と同一のshot式ではない。Referenceの同値性だけではB4のNを新lawへ移せない。

### 回路価格と全費用

[8828b298][TOTAL]はB4表＋保存full-wrapper価格＋固定A/F/Sから全13roundを既に集計した。Strongのbasis共有・directional control・relative fusion・step融合と、Referenceの追加register／CSWAP／χ／cut損失が価格に入っている。C1や別builderの節約量を追加控除しない。

固定H4／1Å／STO-3G／Nα=Nβ2・元H、number-CDF8／元順LD2／S2/s1/r32/K0／h=.1、元H ground・準備費除外、ε=.00159 Ha／failure=.01／13round／P0という条件の主A報告は、Referenceで総CX約11.17%、累積total depth約54.65%、RZ約1.12%減、幅9→17。ただし絶対総費用の約99.4〜99.5%はq≥64 proxy、全体CIとactual成功は未資格である。ここでは原報告値を引用しただけで再計算していない。

## 4. 共通資源体系と比較上の規則

資源kごとに `G_k=Σ_(m,b) N_(m,b) μ_(m,b,k)`。必要shot、単価、価格モデル不確かさ、peak幅を分ける。RZ/CX/三depth/sizeを別々に比較し、幅は最大値として記録してshot加算しない。DのP0 DAG depth、Cのmodular serial上界、Q3のshot加重DAG境界は同じdepth価格ではない。RZ→T換算やdepth→実機時間換算は行わない。

attenuation／rangeはsignal・shotへ入るもので、無条件に単価へもう一度掛けない。古典価格MCの標本数も量子shotの追加倍率ではない。DF誤差を元H信号へ一回入れたD-B1に、own-DF評価のWeyl energy項を再加算しない。共同basis・対角融合・Reference・C1の同じ節約を二重計上しない。

### A/F/Sを使える範囲

A＝C7-4 all-q affine、F＝C6a凍結q2/q4 affine、S＝[1,q,sqrt(q)]探索model。固定D条件・P0・cut・method/axis・full-wrapper scopeの条件付きproxyである。q1はevent16 cutの保存価格で別扱い、q2〜32は保存cohort平均、q≥64は8828b298で採択した外挿。母平均・bias上界・独立holdout保証ではない。A/Sはq2〜32をfitに使用しておりq32を独立検証へ戻さない。

同じlawと回路でα配分だけ変えるなら、単価を凍結して新Nと組み合わせる条件付き感度評価は可能。ただし相対model bias、直接平均と係数のcross-covariance等は未知のまま。別LD、DF、frame順、r/K、h/PF、Closed law、compiler A、適応Q readoutへA/F/Sを移送しない。これらは新価格校正・独立検証か別の構造的価格境界が必要である。

## 5. 他改善の接続価値と不足

[対応表](INTEGRATION_MATRIX.md)のとおり、Closed P3＋Cの共同選択は既に一つのend-to-end成功例であり、そのSOURCE/native/weighted誤差・MC分離を再利用する価値が高い。新A Qはshot削減を直接扱い、同じQ機会をStrong/Full/Truncatedへ与える公平な移送が重要である。Q3の固定canonical保証をClosed・CDF8へ自動転用しない。

O1辞書／core-residual、研究Cの表現・Z再配置は、元H state・sector・近似誤差・λ_R・PF予算・価格を改めて結ぶ必要がある。own-DF結果のNや削減率を元Hへ流用しない。O2のClosed P3は接続実績がある一方、P5・IS/CV/LCU・Parentの有限toy利益や負結果は条件別に保持する。

強Full/Truncatedは最後の比較から落とさない。同じDF12／元H ground／ε=1.5936e−4 Ha／failure=.05／18round／compiler A・準備費除外の[C既存報告][COPT]では、最適化Closed PRは強Fullより高価だった。これをCDF8/P0の数値順位に混ぜず、最終的に公平な強対照が必要な根拠とする。

## 6. 推奨する最初の検証

**固定D-CDF8の誤差・回路・A/F/Sを保った、round別α配分×Strong/Referenceの小さい条件付き統合検証**を推奨する。保存B4には各roundのε_coordinateが既にあり、新PF評価・ground・sampling・回路生成なしで、Nと単価の接続を検査できる。既存一般α提案式を再利用し、同じ有限候補集合を両方式へ与える。詳細は[実行案](NEXT_EXECUTION_PROPOSAL.md)。

これは新アルゴリズムの発明でも、既存Cのh/resolution/α再探索の再実施でもない。固定CDF8 routeの小さい接続検査であり、価格model内の改善しか判定しない。大きい研究効果を狙う後続にはQの公平なFull/Truncated/Closed移送や、別LDの条件別資格があるが、同時に全部を最適化しない。

## 7. 実施範囲・保全・STOP

今回行ったのは資料・SOURCE本文／AST・既存JSONのschema・Git ref/固定blobの読取、文書作成と公開照合のみ。研究moduleのimport、科学test、数値再解析、sampling、build/compile/simulation、N再算定、fit、13-round再集計、新実装、詳細科学レビューは0。mockで補う必要がないためmockも追加していない。

既存resource-eval branchとnew-A/C joint branchを比較し、系列横断の文書を独立に保全するため、c002153aから **codex/pr-rpe-joint-integration-plan-20261110** を新設した。既存branchのmerge/削除/rebase/force pushはしない。旧SOURCE・input・raw・結果・grant・ledger・marker・KEEP・STOP、他worktree/process、共有設定を変更しない。公開は本資料と索引・静的照合だけ。rawや内部保存先は追加公開しない。

公開remote SHA/tree/bytes照合後**mandatory STOP**。次担当は**ユーザー／GPT**：第一案の目的・有限menu・理想条件付き主張・実行範囲を採否判断する。**Codex**は新たな明示許可があるまで数値共同最適化や科学的GO判定を開始しない。

[TOTAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8828b298975c88a2c0d1fd57ff6ed905233840f5/docs/research/track_d_bc_total_20261108/REPORT.md
[BRIDGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c002153a98a2f2d209a4a60f035c6bd7e36293c9/docs/research/track_d_bridge_run_reuse_20261110/REPORT.md
[CROSS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/docs/research/pr_rpe_cross_track_20261109/REPORT.md
[ASURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/162f3d3f0f42973048fd14a7a6624e147a1e961a/docs/research/track_a_algo_improvement_survey_20261109/README.md
[HP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e162ef708d7f721cd9c95ae07a833f7301d2b449/docs/research/track_a_b2_p1_hp_branch_shots_20261107/REPORT.md
[COPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md
[Q3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08f9cbe77456ec29d50a992d7165c4d726df7bf7/docs/research/pr_reference_feedback_q3cd_20261110/REPORT.md
[D64]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe361b206a6572142775ac01972545d312e5fb14/docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md
[B6B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ffe8b281b76cef3a6fd06f43413904d6766d0ec/docs/research/track_d_b6b_bc_integration_20261108/MEASUREMENT_EQUIVALENCE.md
[B4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md
