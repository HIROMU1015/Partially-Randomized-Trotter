# 研究C：最終成果総括・全検証結果と証拠索引

2026-11-07 JST。Codex主エージェントによる既存成果の静的整理。Repository: HIROMU1015/Partially-Randomized-Trotter。

## 1. Executive Summary

**研究Cの現在の活動を、既存成果を保全して研究管理上一区切りにする。** これは今回のユーザー指示に沿った記録整理であり、科学的な全問題の解決、独立新規性の否定、正式な論文化GO/NO-GO、旧STOPの解除ではない。

出発点は、DFのtensor近似だけでなくPR tailの係数massもJ/U最適化へ入れ、減衰・RPE shots・compiled価格を含む資源を下げられるか、という問いだった。固定H4ではDF予算を保つtail低減、PR内部の条件付き資源改善、強い決定論対照との限定的crossoverまで接続できた。一方、**元辞書で観測したPR-aware目的の優位は、保存J/Uを線形Z再配置した後には維持されなかった。ENTRY由来DF＋PRの利益と、PR-aware目的固有の利益は別である。**

本書と[EVIDENCE_INDEX.csv](EVIDENCE_INDEX.csv)は、[10月31日Evidence Map][EM]のE01〜E23を保持し、前段のH2/H4/H6・codec・native検証、11月1〜7日の最適化・統計・精度・再配置・強D・重複整理へ接続する。索引の**151行は文書・実行・構成要素・設計の参照record数であり、151個の独立実験ではない**。同じ実行の総括、失敗、修正版、再利用、継続segmentを区別した。

| 保存すべき到達点 | 成立範囲と結論 |
|---|---|
| 肯定的成果 | 同じ初期値のH4/CDF8で物理DF予算を保ってtail massを低減し、PF/RTE・条件付きshots・native価格へ接続。固定回路のfull512検査と一部validated保証、後続B2 reference上界も取得 |
| 主な負結果 | 固定P4・CDF7・surrogate・旧round混合等で登録改善を得なかった。全法の不可能性ではない |
| 重要な結論の更新 | 旧r128では主RZがDより高い一方、別候補/r32で名目モデル内crossover。さらにZ再配置後は目的arm順位が逆転 |
| 最新の限定肯定 | 共通LD2再配置の固定2強D対照に対し、保存ENTRY E2＋PR/r16のideal条件付き総RZ点値利益が残った |
| 継続しない理由 | 再配置後の目的は既知weighted-L1²族に含まれ、表現・辞書・残差補償はO1等の既存課題。11/7調査では別立ての具体的独立課題を特定できなかった |

新科学計算、保存数値の新比較・統計・順位計算、DF fit、科学runner/test、回路build/compile、重要科学レビューは今回すべて0。数値・式・資格は既存固定報告から引用した。

## 2. 研究Cの目的と研究課題

### 2.1 当初の問いと最適化自由度

通常CDFは一般対称signed Jとframe別実直交Uで二電子tensorを近似する。XDFのrank-one的な二次表現と、一般Jのnumber-CDFを混同しない。研究Cは元の分子積分・spin/frame規約・one-body補正κ・scalarを保持し、tensor fitting errorとPR tail massの交換を調べた。DF rankとprefix L_Dは表現／分割の条件であり、λ、PF誤差、shots、価格とは別の量である。

初期実装の既存目的は

`Fγ = f/f* + γ(Mε/M*)²,  f = ½ ||g−g_DF||²_F`

で、Mεは元順L_D以後の非identity Z/ZZ係数の平滑mass、exact λは絶対値和である。初期値で固定したscale、ε=1e−8 Ha、同じqualified x0、有限optimizer recipeが比較の一部を成す。Smooth objectiveの低下をexact massの同値な低下や物理精度保証へ置き換えない。[EM §2–3][EM]

元のHamiltonianと候補ごとの近似DF Hamiltonianは別である。研究Cの下流は主に各候補自身のDF ground oracleを用い、状態準備費用を除外した。元Hの厳密groundアクセスを使うO1-M2の完全残差補償taskへ、その費用結果を無条件移送しない。[重複整理 §2–3][OVER]

### 2.2 当初の改善機構と実際に測ったもの

tail λ低減によりRTE normalization・attenuation負担を軽くし、十分shotsを下げることが出発点だった。ただしJ/U変更はPF誤差、basis、head、event law、compiled単価も変える。研究はproxy低減から下流全費用の確認へ広がり、さらに強D、support/rank/basis、目的の公平比較、one-bodyへのZ移送を調べる形へ進んだ。これらは並行する枝であり、すべてを一つの逐次pipelineとして扱わない。

## 3. 科学的設定・資源評価方法

| 層 | 固定条件・評価単位 | 区別するもの |
|---|---|---|
| H4主系列 | 直鎖・隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36/system256/controlled512、number CDF8、元順LD2 | H2 sector4、H6 sector400、H4別geometryとは別。各fit初期値・候補hashも別 |
| 物理DF精度 | sector operator norm＋engineering guard、CA=.00159 Ha、DF reserve CA/8=.00019875 Ha | tensor Frobenius、optimizer success、最終actual CAと同義でない |
| PF/RTE | ordered partial S2、初期s4/r16/K0/h=.025から後続s1/r128・r32・r gridへ。δ=.1、m0..12 | rやsが違う比較をλだけの因果ablationにしない |
| 決定論対照 | 各候補の全frame ordered S2/s1/h=.1、合法な全sweep相対Gaussian融合 | PRのprefix LD分割・r/Kとは別。同J/U Dと別J/Uの保存CDF/XDF Dも別 |
| 信号・shots | own-DF ground、状態準備除外、βtotal=π/6、βstat=π/12、failure=.01、axis=.01/26 | 予定Hoeffding十分shots、実行済み量子shots、普遍的必要最小数は別 |
| Native/P0 | 主H4系列はQiskit1.3.0/opt1/seed17/rz,sx,x,cx、MPFR160 declared phase、位相fitなし。raw/P0/actual inverse/X-Yの登録範囲を検査 | 完全kernelと単event加算、有限数値残差と厳密包囲、sectorとfull512を区別 |
| 総資源 | `C = W μ_kernel + S_shell`、fenced serial、全round/axisのshots加重、6資源指標中RZ主 | μは有限価格標本または固定D価格。rやcohort数をWへ重ねて掛けない |
| 統計 | 初期N32記述値、後続の独立640/256、selection32/final64などは各固定policyごと | cohortをpoolせず、近似IID/t/Welchの名目被覆を保証へ昇格しない |

同じ価格cohortを13roundで使う既存モデルのSEはW倍であり、√13で縮めない。DF誤差はHa、PF/RTE・合成のsignal誤差は無次元、統計は確率／位相である。単位の違う量を根拠なく加算しない。Ideal laneの合成誤差0は反実仮想の前提であって実測0ではない。[EM §6][EM]、[11/3追補 §6][MAP3]、[H4統合][H4]

## 4. 研究の発展と重要な意思決定

以下の日付は主に原報告の識別日である。scope名が20261101でも公開が11/2の例があり、索引では実行日UNKNOWNと報告日を分ける。

| 時期・枝 | 研究上の役割と固定記録 |
|---|---|
| 10/24 | C4 Rustiq／単純角度結合の比較、一般J/CDF入力接続の準備。特定fixtureの削減は単純結合でも再現し、独立機構としなかった（PRE01/PRE42） |
| 10/25–26 基盤 | C1のAPI非互換・frame不一致・compile競合と修正版、H2/H6、修正codec、H4 XDF prefix／共通精度／spin-relative強D（PRE02〜31/E22） |
| 10/27–28 PR-aware | 共通x0のStage A→B1/B2→C1/M1。並行して固定CAR/κH/polarの保証対象を検証（E01〜05/E23） |
| 10/28 対照・政策 | 同J/U D、s1/r64、r128の誤差とnative。PR内部改善と対D不利を区別（E06〜09） |
| 10/28–30 構造枝 | P4 fit/資源、surrogate修正版、第三DF、shared-basis帰着、CDF7、native構造、round/law調査（E10〜21） |
| 10/31 | E01〜23のEvidence Map。ここでの「強Dへの主RZ不利」を後の全政策へ一般化しない |
| 11/1–2 収束・数値安定性 | RC2対照、共通政策、保存JSON復旧、分離・曲率・整合修正・reduced/admission・小step。技術修正と一次収束の未達を併記 |
| 11/2 新候補 | 5 U更新のupdate005をPF/RTE/nativeへ接続、同J/U Dとも比較（POST43/POST37） |
| 11/3 並行枝A | update005のr再調整、別trajectoryの追加64更新update064、各自の独立価格校正（POST36/40/41/28/42） |
| 11/3 並行枝B | 近接ENTRY/PRの16更新比較、r32下流、保存XDFとの探索的統計（POST19/20/27） |
| 11/3–4 精度枝 | IID-grid Stage A、Whole-P0未成立、B2 reference上界、固定4word pilot、期待bias設計NOT_READY（POST12/32/08、DOC群） |
| 11/5–6 統合 | 18 trajectory・2更新予算・公平strength/r selection/final。server旧STOP、cgfix起動STOP、picklefix後のH4完了、1.15 Å初期化STOPを別に保存 |
| 11/6 Z診断→下流 | 36保存checkpointのM_ZZ診断、全H同値な線形Z再配置後に3候補のPF/RPE/nativeを評価。再fitはしない |
| 11/7 強D→重複整理 | 保存CDF8/XDF8にも同じLD2再配置、点利益維持。既存DF/O1/F2との重複を整理 |
| 今回 | ユーザー指定の案Bを基本に、最終総括と索引だけを公開。新研究やSTOP解除へ進まない |

11/3のupdate064下流は、U更新枝のterminalをpeer commit＋hashで受け取る構成だった。Git祖先だけで数値入力の依存を推測しない。11/6の17完了trajectory再利用も、18本の新fitとして数えない。[MAP3 §1][MAP3]、[H4 保存来歴][H4]

## 5. 全検証の一覧・分類

### 5.1 索引の単位と読み方

[EVIDENCE_INDEX.csv](EVIDENCE_INDEX.csv)の各行は、実行identityまたは文書・設計・実行構成要素を示す。IDの数は実験数ではない。

- **E01〜E23**は旧ID・内容を維持する。E03-OLDは同E03内の旧技術STOP、E23-F3は旧E23の別固定trajectoryを明示したもの。
- **PRE**は前段・旧失敗・補助検証。H2/H6の移送性をPR-aware H4と混ぜない。
- **POST**は11月以降の実行・再解析・静的資料。番号の大小は時系列ではない。`record_kind`を必ず読む。
- **DOC**は主に技術調査・設計・統合。人工testのPASSは科学性能の観測ではない。
- **INT-FIT-***は保存表が指定する18個のtrajectory identityで、16/64checkpointは同じfitの中に保持。**INT-DOWNSTREAM**は同じ統合実行の下流構成要素。
- **POST09-ORIGINAL/POST09**は同じ登録比較の中断前と継続後。288＋512の800標本を別々の独立実験へ水増ししない。

E01〜23の既存整理を基礎にしたため、以下は各詳細の再監査ではなく入口である。SOURCE・raw・STOPの具体リンクはCSVを使う。

### 5.2 既存E01〜E23

| ID・元報告 | 検証目的・条件 | 保存された結果 | 実施種別・資格 |
|---|---|---|---|
| [E01](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md) | PR tail目的で精度を保てるか。H4/A8、γ0/.1、各1 fit。 | λ34.72464491113348%低下、両DF予算PASS。30 calls完了。 | 科学実行。Sector norm＋guardの数値pilot。Tensor残差と独立判定。相対目的減少停止。 |
| [E02](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_stage_b1_execution_20261027.md) | 同じ最終2 armのPF/RTEは改善するか。P4/16・R13。 | 因子/総和CAR・ground/alias PASS、26/26 round潜在適格。ProposedのA/RTEが全13 roundで改善。141 calls。 | 科学実行。数値referenceと登録十分条件。合成/law/RPE実行は別。 |
| [E03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_stage_b2_execution_20261027.md) | 同P4/16のnative単価は下がるか。P0/N32、各arm32。 | RZ mean5,198.3125→4,765.875。64 full512検証PASS、652 calls。 | 科学実行。有限bit標本平均。母平均CI・全law supportなし。旧run01の技術STOPは保全。 |
| [E04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_stage_c1_execution_20261027.md) | A/B/価格がPR内部総資源へつながるか。P4/16・R13・P0/N32再利用。 | Ideal shots53,136→19,574、総RZ83.08598%低下。164 calls。 | 科学実行。Ideal/assumedの反実条件付き。Evidenceは全round null。有限価格校正。 |
| [E05](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_mechanism_execution_20261028.md) | 価格とshotsのどちらが保存差へ寄与するか。E04の2×2置換。 | 対称分解のshots側2,496,665,997,082 RZ、価格側157,377,117,755 RZ。20 calls。 | 既存保存結果の事後分析。保存scalarの代数的会計、因果推定なし。 |
| [E06](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md) | PRに同じJ/Uの強D対照を置く。A8両arm/D1/R13/P0。 | 両kernel RZ11,026。Proposed Dのideal総RZ101,825,929,455、253 calls。 | 科学実行。全matrix・adjoint/X/YとRPE合成伝播は数値pilot。PRとはs・融合機会・誤差資格も異なる。 |
| [E07](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_M3_s1_r64_repair_execution_20261028.md) | Head反復を減らせるか。Proposed P1/64と旧P4/16。 | PF m12は約.004476→.071615へ増加、A同じ。Ideal総RZ約2.95179019e11、旧PR比.546334249、D比2.89885907。471 calls。 | 科学実行。非paired native32、shots再評価、全law未資格。旧技術STOP保全。 |
| [E08](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md) | r128で減衰を軽減できるか。Proposed P1/128/R13。 | A_m12=.716844438、補正RTE=.0609453085、ideal shots16,056。13/13潜在適格、98 calls。 | 科学実行。同sのPFをSOURCE照合して再利用。価格未取得だった段階。 |
| [E09](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_M4_native128_execution_20261028.md) | E08の実native単価を含める。P1/128/R13/P0/N32。 | RZ mean14,076.4375、ideal総234,117,788,135。M3比.793138、D比2.299196。332 calls、32 full512 PASS。 | 科学実行。有限校正、ideal/assumedのみ。Evidence null。 |
| [E10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_df_structural_support_execution_20261028.md) | Tail JをP4 supportへ制限できるか。H4/rank8/LD2、head-full、tail対角＋辺01/12/23、γ.1。 | 1 fit、DF予算PASS、λ1.516165932377233 Ha。係数位置216→144。21 calls。 | 科学実行。Projected初期値が変わるため同x0 ablationではない。Tensor/sector別測定。 |
| [E11](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_df_structural_pr_performance_20261028.md) | E10をP1/128/R13/P0/N32へ接続。 | A_m12=.692254786、ideal shots16,568、RZ mean15,047.46875、総275,422,425,481.5。M4比1.176427。449 calls。 | 科学実行。全13 round・固定32 correctness PASS、点比較のみ。 |
| [E12](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_df_basis_surrogate_repair_execution_20261029.md) | 保存M4/P4各32で価格proxyを検証。固定4-fold OLS。A=switches、B=overlap、C=3特徴、cohort平均対照。 | RZ MAEはM4 A221.4073<C227.7980、P4 B261.5380<C273.5792。148 calls、SURROGATE_NOT_QUALIFIED。 | 保存結果の事後分析。静的DFは2個。Head等はcohort interceptへ吸収。RMSEの別結果で主MAE判定を変更しない。 |
| [E13](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_df_basis_switch_optimization_design_20261029.md) | Switch期待値をDF目的へ組み込めるか。厳密basis_key群のmass比q_b。 | IIDで(r−1)(1−Σq_b²)。固定partitionではJのmass配分を扱い、自由Uの幾何へ直接勾配を与えない。 | 技術・設計／同値性。技術設計・人工検証のみ。非凸性、零tail、smooth biasを整理。 |
| [E14](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_df_basis_third_repair_execution_20261029.md) | 別DFへswitch proxyが移送できるか。第三DF=A8 γ0、P1/128/R13/P0/N32。旧2DFで学習後固定。 | Third RZ MAE A369.78698、B303.32856、平均2892.23438。SWITCH_TRANSFER_NOT_SUPPORTED。493 calls。 | 科学実行。静的3DF中、独立holdoutは1DF。32trajectory≠32DF。再fitなし。 |
| [E15](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_shared_basis_df_design_20261029.md) | Shared-basisは独立した分解classか。一般対称J、厳密同一UをG群で共有。 | 群内ΣJへ集約でき、表現集合は通常CDF outer rank≤G。CDF8のtail2 frame共有はCDF≤7に帰着。 | 技術・設計／同値性。数学条件付き同値と人工61 PASS。rank-one J、別support制約、近似U共有へ拡張しない。 |
| [E16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_cdf7_onefit_execution_20261030.md) | 通常CDF7がDF予算を満たすか。H4/rank7独自Cholesky初期値、γ0、1 fit。 | 初期norm＋guard9.0647035e-4→終了3.022641747295844e-7 Ha、予算PASS。25 calls。 | 科学実行。Tensor Frobenius4.6164582354636235e-7とsector残差を別記。相対目的減少停止、最適rank証明なし。 |
| [E17](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_cdf7_PR_execution_20261030.md) | CDF7をP1/128/R13へ接続。LD2、tail F2..6。 | 13/13潜在適格、A_m12=.460928623、ideal shots21,768。119 calls。 | 科学実行。独立CAR/ground/alias PASS、evidence未資格。後の価格はE18で取得。 |
| [E18](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_cdf7_native32_execution_20261030.md) | CDF7のnative単価/総RZは下がるか。E17＋P0/N32、保存shots再利用。 | Mean RZ17,754.9375、ideal総638,880,821,593.5。γ0 CDF8比1.0188545、M4比2.7288863。332 calls、32 full512 PASS。 | 科学実行。小差の母平均有意性なし。独自fit/law/初期値の差があり、rank単独効果ではない。 |
| [E19](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_native_structure_optimization_design_20261030.md) | Nativeの未利用構造は何か。Head相対融合、tail厳密key共有、境界/中央Z/ZZ融合。人工小系のみ。 | Head/tail基本共有は実装済み。Tail相対融合は人工complete P0でRZ増、中央融合は局所削減例があってもcomplete例で不変。68人工PASS。 | 技術・設計／同値性。実H4新build/compile/再解析0。Head/tail境界案のcomplete性能未取得。 |
| [E20](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_roundwise_det_pr_execution_20261030.md) | 同一DF energyへD/PRをround別選択できるか。Proposed D1対P1/128、R13、固定位相/failure配分。 | 共通raw-signal/aliasとβ<π/3のnearest-branch条件。保存189-callでideal/assumedとも13/13全D、MIXED_ROUND_DOMINATED。 | 保存結果の事後分析（登録科学run）。設計117人工PASS、保存189-call完了。主RZ固定点モデル、evidenceはUNQUALIFIED。 |
| [E21](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_sampling_law_freedom_survey_20261030.md) | Euler/Parent/IS/return/相関に未利用自由度があるか。Cの固定K0平均operatorとTrack B一次資料対応。 | Common-parent内L1最小性等を整理しNO_DISTINCT_FREEDOM。新9＋既存4＝13人工PASS。 | 技術・設計／同値性。限定技術調査、H4性能評価0。Affine辞書依存、全path重み等の適用条件を維持。 |
| [E22](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md) | 通常CDF対XDFも強い実装で比較する。歴史的CDF8/XDF prefix8、D1/R13/P0、B2/B3経路。 | 総RZ102,056,444,569対100,267,421,030、元報告ではXDFが1.7529745883%小さい。86 calls。 | 科学実行。数値CA pilot、DF誤差は別。A8最終J/Uではない。 |
| [E23](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_aware_df_F2_K1_execution_20261027.md) | Baseline/Proposed各1保存trajectoryを独立CARへ接続。P4/16の一h-step、full512全基底、Arb192、κH/right-polar Q target。 | F2-K1/F3-K1各7 calls。位相込みraw/target、compiled/targetのvalidated operator2上界が固定allowance内。 | 科学実行。F2/F3は索引のE23・E23-F3に分離。 |

E12は保存cohortの事後的surrogate分析、E20は保存結果を使うround割当評価であり、新しいDF fitではない。E23の二trajectoryは同じcarrierに載っても別SOURCEである。旧E10の「後続未判定」はfit単体の状態で、E11に性能結果がある。

### 5.3 11月1〜7日の科学実行・保存結果再解析

| ID・元報告 | 保存結果と現在の読み方 | 科学SOURCEの案内 |
|---|---|---|
| [POST03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_common_convergence_execution_20261101.md) | RC2/PR各1fitは計算完了、JSON書出しで技術STOP。両一次条件未達。 | `e2f7dfd443…` |
| [POST04](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_common_convergence_export_repair_20261101.md) | 保存JSONから1-unit集計復旧。追加fit0、旧failed STOP不変、ONE_OR_BOTH_UNQUALIFIED。 | commit UNKNOWN・manifestを使用 |
| [POST06](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_fixed_u_tail_lp_numerics_execution_20261101.md) | 微小係数保持profileのLP1回・50 units、NUMERICAL_LP_IMPROVEMENT。Native未評価。 | `c40f812819…` |
| [POST07](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_fixed_u_tail_preserving_lp_execution_20261101.md) | LP1回・9 unitsでUNRESOLVED_LP。Solver Optimalだけでは独立数値資格を満たさず。 | `56a6acb8f7…` |
| [POST08](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_fixed_word_pilot_execution_20261104.md) | 固定4word・86 units。Full512/actual inverse/X-Y数値PASS。全law/CA/母平均費用ではない。 | `ba0e2bf2d1…` |
| [POST09](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/docs/research/research_c_h4_relocated_downstream_results_20261106.md) | 再配置3保存候補、r集合selection32＋final64。主2比較は名目ENTRY低費用。再fit0。 | `cfc64b7086…` |
| [POST10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d601e0ef4375c37185211aa75685ae0e582f3e64/docs/research/research_c_h4_relocation_execution_20261106.md) | 保存36checkpoint＋中立参照1のmass診断。主M_ZZ順位反転。下流評価は別。 | `052941ab7c…` |
| [POST11](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/91417fb3ef75e7eec0849ad39f0649d170754664/docs/research/research_c_h4_strong_d_relocation_results_20261107.md) | 保存CDF8/XDF8へ共通LD2再配置。D数値laneも通過、ideal点値でE2利益維持。 | `e6652ab841…` |
| [POST12](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_iid_grid_stage_a_execution_20261103.md) | IID53-bit grid Stage A・38 units。13round reference対応と格子categorical TV。PCG64保証なし。 | `ddd8333488…` |
| [POST14](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_integrated_server_cgfix_20261106.md) | 旧serverの17 trajectory完了・1本missing-jacでSTOP。Native未実行。SOURCE hash200418…。 | commit UNKNOWN・manifestを使用 |
| [POST16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_j_u_consistency_repair_execution_20261101.md) | 修正版paired fit328 units。両DF予算PASS、両J/U同時一次未達。診断15はPOST16-DIAG。 | `24a7268359…` |
| [POST17](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_j_u_reduced_execution_20261102.md) | Reduced pilot38 units、両armREDUCED_GRADIENT_UNQUALIFIED。 | `e1582a5620…` |
| [POST18](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_j_u_stable_execution_20261102.md) | Stable pilot480試行。PR inner整合例外で2 FAIL、RC2の64更新もU一次未達。 | `12e0079a26…` |
| [POST19](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_near_objective_practical_comparison_execution_20261103.md) | 近接ENTRY/PR有限recipe645 units、各16 U更新。Tail λ差、同時収束ではない。 | `70a98adfb7…` |
| [POST20](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_near_practical_r32_downstream_execution_20261103.md) | 両16-update endpointのr32下流1231 units。元辞書でPR約50.58%低い名目モデル結果。 | `4df8eca29c…` |
| [POST21](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_coefficient_point_execution_20261102.md) | 保存trial2の10-unit診断。旧同一点比較FAILを再現、新同一配列経路PASS。 | `565de95fc3…` |
| [POST22](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_guarded_step_execution_20261102.md) | Qualified-only Armijo60 units。8trialで受理U更新0。 | `d03351ea00…` |
| [POST23](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_inner_point_execution_20261102.md) | 固定U点のinner1回・8 units。整合/一次資格、outer更新は0。 | `ab5d80bd64…` |
| [POST24](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_local_direction_execution_20261102.md) | 固定局所方向7点・49 units。経験的減少を観測、U更新受理0。 | `e06913a234…` |
| [POST25](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_reduced_limited_execution_20261102.md) | PR reduced18 unitsで勾配未資格、受理U更新0。 | `b2e5ab4d5f…` |
| [POST26](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_small_step_execution_20261102.md) | 小step19 units。独立終端確認後U更新1回受理。同時収束ではない。 | `a426d62f27…` |
| [POST27](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_xdf_cdf_statistical_comparison_execution_20261103.md) | 保存native32の7-unit事後統計。対XDF8の探索的名目支持、prospective確認ではない。 | `e2884a071d…` |
| [POST28](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_r32_price_statistics_execution_20261103.md) | update005/r32独立640価格、5845 units。符号・登録精度は未解決。 | `62fa4c0144…` |
| [POST29](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_rc2_pr_conditional_shots_execution_20261101.md) | 保存RC2/PRの105-unit条件付きshots/W。Evidence lane未資格、総資源は別。 | `e88fa599af…` |
| [POST30](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_rc2_pr_native_execution_20261101.md) | 保存RC2/PR各32のnative655 units。単回価格correctness、総資源は未算定。 | `14843a28f7…` |
| [POST31](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_rc2_pr_stage_b_execution_20261101.md) | 保存RC2/PRのPF/RTE143 units。A改善とPF悪化が併存、両13round潜在適格。 | `d6b17b7946…` |
| [POST32](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_reference_bound_execution_20261103.md) | B2・27 units、Arb384の二つのreference operator上界。全native/CAには未接続。 | `01baf2b5e4…` |
| [POST33](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_regularization_comparator_execution_20261101.md) | Uniform L2型RC2・17 units/1fit、DF予算PASS。十分最適化された原法対照ではない。 | `150f904f69…` |
| [POST34](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_regularization_first_order_execution_20261101.md) | 保存3点14-unit一次診断。初期penalty一致は勾配一致でなく、両終点未収束。 | `0996a5abb4…` |
| [POST36](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_rte_r_retuning_execution_20261103.md) | update005固定のr16/32/64新規・r128保存再利用、1265 units。集合内r32点最小。 | `1cdeaa8bd5…` |
| [POST37](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_same_ju_strong_d_execution_20261102.md) | update005同J/U強D・156 units。旧r128 PRより主RZ低費用。 | `5f27539ae0…` |
| [POST38](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_separated_J_U_curvature_execution_20261101.md) | 曲率pilot210試行、208 COMPLETE/2 FAIL。評価点取り違えを含む整合STOP。 | `ecb2e7dc0e…` |
| [POST39](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_separated_J_U_execution_20261101.md) | 分離J/U258 units。DF予算PASS、同時一次未達。旧残枠は転用しない。 | `cbbb11e7e4…` |
| [POST40](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_u_step_control_execution_20261103.md) | update005から追加64 U更新、1042 units。新update064、J MET/U NOT_MET。 | `025d495166…` |
| [POST41](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_update064_r32_downstream_execution_20261103.md) | update064/r32下流＋同J/U D、617 units/native32。条件付き点crossover。 | `928fda0a6d…` |
| [POST42](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/docs/research/research_c_update064_r32_price_statistics_execution_20261103.md) | update064独立256価格、2374 units。NOMINAL_MODEL_PR_LOWER、約4.48%減。 | `9837eac59b…` |
| [POST43](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_pr_integrated_execution_20261102.md) | 543-unit統合。5 U更新→update005→r128下流/native32。PR内部改善と対強D不利が併存。 | `980b514c81…` |

POST16-DIAGは同じ公開報告内の別15-call保存診断で、SOURCE `a6a2c55aa8400428329165578d54e2f65bff4636`。修正版paired fitは `24a7268359506abd171686132f12b4876f9e62bf`。同じsnapshot・同じ報告だからといって統合SOURCE一つにしない。

### 5.4 統合実行・旧失敗・技術設計の所在

| まとまり | CSVで辿る範囲・重複計上しない境界 |
|---|---|
| C1前段 | PRE02/PRE12〜31：H2/H6、初期API STOP、frame修正、保存fit評価、Gaussian精度・位相、native、shots/CA。PRE03は既存分子比較の統合資料 |
| H4対照の変遷 | PRE04〜11/E22：B3 spin対応、XDF prefix8/9、共通precision、強baseline。旧CDF8/XDF10・prefix8・相対融合版を別条件で保持 |
| 旧失敗と修正版 | PRE39→E01、E03-OLD→E03、PRE38→E07、PRE32→E12、PRE33→E14。旧partial/FAIL/STOPを成功へ変更しない |
| 固定保証への途中段階 | PRE34〜37/PRE40〜41→E23/E23-F3。F1、κ診断、polar target、bridge、κH targetの違いを保持 |
| 18 trajectory | POST14の先行17本＋INT-FIT-*、POST15の起動修正＋新1本、POST13総括。各taskと2checkpoint hashはCSVのinput欄 |
| 統合下流 | INT-DOWNSTREAM：108policy、selection/final、D19。POST13はこの結果をまとめる文書で新実験ではない |
| 1.15 Å | POST13の元INTEGRATED_STOP。凍結recipe後の初期化不一致で移送評価未完了。成功結果なし |
| 精度設計 | DOC01/08/15/20/21/02/03/09・POST01。B2公開未確認の旧統合は、後のB2取得・統合へ接続するが旧記録はそのまま |
| 方法・貢献・区切り | DOC04〜07/18、POST02/05/35、E13/15/19/21。新fitや新方法の採択を意味しない |

## 6. 肯定的成果

### 6.1 精度予算を保った有限PR-aware fit

初期H4/A8では同じ修正codec後x0からγ0/.1各1fitを行い、exact tail λが2.2097618367788057→1.4424298855756243 Ha、既報削減率34.72464491113348%となった。両候補のsector norm＋guardはCA/8内だった。Tensor/sector残差はγ.1側がわずかに大きく、より正確な物理DFを作ったという結果ではない。Optimizer successは相対目的減少による有限停止で、一次／大域最適ではない。[E01、EM §3][EM]

後続の同一配列点評価、reduced精度・refinement、qualified-only Armijoと独立終端検査は、数値的に適格な有限U更新とDF予算保持を可能にした。5更新のupdate005、追加64更新のupdate064、3start×3strength×2armの統合gridまで到達したが、最新36checkpointもJ MET/U NOT_METである。修正の成功を「両armを十分最適化した」と表現しない。[MAP3 §4][MAP3]、[H4][H4]

### 6.2 PF/RTE・RPE・nativeへの具体的な接続

初期s4/r16ではAと補正RTEが全13roundで改善し、ideal条件付き十分shotsは53,136→19,574、総RZは3,194,333,274,946→540,290,160,109、既報83.08598%減だった。N32価格proxy・理想／仮定laneのPR内部比較である。λだけを固定した因果実験でも、対強D勝利でもない。[E02〜05、EM §3][EM]

旧r128では同J/U Dより主RZが高かったが、update064/r32の独立256価格ではPR/D点比.955183617395431、nominal95%比区間[.9465946344829994,.9637726003078625]、登録1%精度目標を満たした。対してupdate005/r32の独立640は区間が1を含み、精度目標未達のままである。二候補の区間から直接候補間の勝敗を作らない。[MAP3 §5][MAP3]

### 6.3 線形Z再配置のoperator同値と順位変化

既存SOURCEの移送は、tail frameの行和からone-body κへの補正と対応scalar shiftを作り、全候補H_DFを保ちながらH_D/H_RとPF splitを変える。Frame I/ZZは保持する。SCDFのtarget-sector symmetry shiftや、固定H_RのF2辞書代替と同一操作ではない。[OVER §5.1][OVER]

保存J/Uを再fitせず適用した再配置後の主比較では、P/r16=87,514,417,699、ENTRY E2/r16=75,621,395,471 RZとなった。共通r32比較もENTRY低費用で、主2差は登録近似Welchモデル内の名目支持を得た。これは目的の一般劣位や線形Z単独の因果効果ではない。[Z下流][Z]

### 6.4 共通再配置後の保存強D比較

以下は**固定H4・2保存強D・共通元順LD2再配置・own-DF oracle・P0/fenced serial・ideal条件付きモデル**の既存点値である。Dは全frame ordered S2/s1/h=.1、m0..12。LD8で再配置をやり直していない。

| 方法 | 条件付き総RZ |
|---|---:|
| 保存ENTRY E2＋PR/r16 | 75,621,395,471 |
| 共通LD2再配置XDF8＋強D | 101,780,388,260 |
| 共通LD2再配置CDF8＋強D | 108,561,283,488 |

保存判定は `E2_POINT_BENEFIT_REMAINS`。Dは数値合成laneも通り、整数費用はidealと同じだったが、合成誤差は非零である。PR側の数値lane・actual CA・全lawへD資格を移さない。旧比較の信頼区間を新D対照へ付け替えない。[最終強D報告][D]

### 6.5 取得済み保証・技術資産

E23のArb192は各1固定trajectory・κH/right-polar targetへのvalidated full512 enclosureである。後続B2/POST32のArb384は、保存数学的atom平均→数学的CARと、数学的CAR→binary64 referenceの**異なる二項**を全13roundについて包囲した。任意正規化sector36状態に対するoperator上界であり、native全wordや状態自体の認証ではない。B2後の統合もStage CはNOT_READYを維持した。[B2実行][B2]、[B2統合][B2I]

固定4word/POST08はcomplete full512・actual inverse・X/Yと価格を測った有限数値pilotで、全law価格平均や厳密per-word包囲ではない。「一部で厳密保証取得済み」と「全taskは未資格」を同時に保持する。

## 7. 負結果・NO-GO・停止済み改善案

| 分類 | 対象・保存結果 | 適用範囲 |
|---|---|---|
| 総資源改善なし | Tail-P4/E10→E11。1fit/21-call適格後、449-call/13round/native32まで完了。総RZ275,422,425,481.5対234,117,788,135、約17.64%増 | 元辞書・r128・当時の候補。216→144は係数位置で、event/basis数同率削減ではない。再配置r16や全supportを否定しない |
| 低rankの点改善なし | CDF7/E16→18。DF予算PASS後、native平均・shotsが増え、固定CDF8より主RZ点値が高い | rank単独ablationではなく、初期値/fit/lawも異なる。小差の母平均有意性なし |
| 登録予測条件不成立 | E12複合surrogate未資格、E14第三DFのswitch移送不支持 | cohort内相関や32trajectoryを独立32DFとしない。η追加の根拠ではない |
| 既知表現への帰着 | E15の厳密shared-basis・一般Jは群内ΣJで通常CDF≤Gへ帰着 | 数学的条件付き同値。近似U共有・rank-one・別supportへ一般化しない |
| Complete価格改善未取得 | E19の人工tail相対融合でRZ増、中央結合はcomplete例で不変。外側境界は未測定 | 局所gate削減からcomplete kernel費用を推測しない |
| 準備候補を選ばず | DOC18の `TECHNICAL_NO_GO_CURRENT_EVIDENCE` | 新cost-aware fit機構の選定をしなかった技術判断。PRや研究C全体の不可能性でない |
| 混合利益なし | E20旧J/U・r128のRZモデルで全13round D選択 | Switchingの合法性と利益は別。最新r16や別指標を再評価した結果ではない |
| 独立自由度未特定 | E21 sampling surveyの `NO_DISTINCT_FREEDOM` | Euler/IS/Parent/return/相関の調査集合内。全sampler不存在ではない |
| 数値資格不足 | LP旧profile、J/U勾配安定性、U一次未達、005独立価格の精度目標未達 | Solver success・低いobjective・近似区間の点差で未知を埋めない |
| 精度条件未達／修正 | 旧H6のCDF rank ladder予算未達。codec原因診断・roundtrip・修正rank14後に予算PASS | 旧STOPは失効しない。H6はPR-aware目的の汎化成功ではない |
| 技術的未完了 | 初期API、frame、Qiskit借用、JSON export、missing-jac、pickle bootstrap、1.15Å初期化 | 科学的に不利という結果と区別。旧failed/partialを後続PASSで上書きしない |
| 目的利益の移送不成立 | 元辞書のPR-aware優位がZ再配置後にENTRY低値へ逆転 | 保存J/U・固定比較。再配置後再最適化した両目的の一般比較ではない |

P4/低rank/proxy/roundを「未実行の新候補」として戻さない。[EM §4–6][EM]、[P4性能][P4]、[技術NO-GO][NG]、[OVER §4][OVER]

## 8. 後続検証による結論の修正

| 当初の観測・判断 | 後続の検証 | 最新の成立する結論 | 固定SOURCE・入口 |
|---|---|---|---|
| 旧H4 CDF8対XDF10で約27%削減 | prefix8、共通precision、spin/relative強baseline | 旧値は旧条件で保全。強baseline E22ではXDF8の方が主RZで小さい。rank/政策/実装の差を分離 | PRE03/E22、EM §6.2 |
| PR-awareでλ低下 | 一次診断、共通政策、reduced/refinement | 有限recipeでの低減。全最新checkpointでもU一次未達、十分収束した最適能力比較ではない | E01、POST34/39/16/40、H4 |
| 初期PR内部総RZ約83%改善 | 同J/U D、s/r変更 | PR内部改善は保全し、旧r128では主RZがDより高い | E04/E06/E09、SOURCEはCSV |
| 旧r128ではD低費用 | update005/064のr32・独立価格 | 005未解決、064は名目価格モデルで低費用支持。旧候補/r128の負結果を消さない | MAP3 §5、POST28/42 |
| 元辞書PR/ENTRY比較でPR低値 | 保存36点Z診断と再配置後下流 | 同じ保存J/UでもD/Rとmassが変わり主順位が逆転。再fitなし | 診断SOURCE052941ab…、下流cfc64b708… |
| ENTRY＋PRに同J/U D利益 | 共通再配置の保存CDF8/XDF8強D | 2対照ideal点比較でも利益維持。目的固有利益・全baseline優越ではない | D SOURCEe6652ab… |
| P4準備ではfit未実行 | 同SOURCE59e186…でfit、4ac0e5…で全下流 | 適格fitと総RZ約17.64%増まで完了した負結果 | E10/E11、P4 |
| Reference誤差はUNKNOWN | IID-grid数値bridge、B2 reference包囲 | 二項のreference保証は取得。ただしWhole-P0/trajectory前提・実PCG64・CAは未資格 | ddd833…/01baf2…、B2I |
| 新しい共同最適化研究へ再構成可能か | 既知目的対応・O1/F2等との重複 | w_ZZ L1²は既知族。別立ての具体的独立課題は現在未特定 | BASE/OVER |

この表は条件差と後続関係を整理したもので、旧結果を誤りとして消す表ではない。

## 9. 最終的に成立する科学的主張

### 9.1 既存証拠に支持される限定的事実

固定H4で、同一初期値・有限recipeの目的変更によりDF予算を保ちながらtail massを減らせた。固定保存候補をPF/RTE・ground/alias・native correctnessへ接続し、P4/CDF7/proxy等の限定負結果も取得した。線形Z再配置は全Hを保ち、splitと下流条件を変えた。固定回路やreferenceについて、数値pilotと特定対象のvalidated証拠がそれぞれ存在する。

### 9.2 明示条件付きで支持される主張

Ideal finite-RTE IID law、exact native synthesisまたは明示assumed budget、有限bit価格proxy、own-DF oracle、指定候補・PF/shot/serial政策の下で、PR内部資源利益、旧064/r32のnominal価格優位、再配置後ENTRY＋PRの固定強D点利益が得られた。それぞれの標本・区間・対照を交換しない。

`λ↓`、`rank↓`、`atom位置数↓`、`μ_kernel↓`のどれも、単独では`W μ+S↓`を保証しないという既存事例が揃った。これは新しい普遍定理や各因子の因果効果の認定ではない。

### 9.3 現時点で支持しない主張

- PR-aware DF目的の一般的優位、独立した新アルゴリズムの確立、全DF/全Hamiltonianでの最適性。
- 全law implemented correctness、全trajectoryの合成誤差、actual CA、保証付き無条件資源優位。
- nominal近似区間のvalidated coverage、全候補・選択後・全6指標の同時優位、母平均費用の無条件優位。
- FT T/Toffoli、実機runtime、noise・状態準備込みの利益、分子サイズscaling。
- 元物理Hのenergy taskへの無条件移送、H2/H6の通常CDF結果からのPR-aware一般化、未完了1.15Åの移送成功。
- λやZ移送だけの因果説明、再配置後目的で再最適化したときの順位。

## 10. RC-DF・SCDF・UWC・O1等との関係

新しい文献調査はせず、[既存比較可能性調査][BASE]と[11/7重複整理][OVER]を再利用する。

再配置後の既存定義は `M_ZZ=Σ_tail(Σ_i<j |J_ij|+¼Σ_i|J_ii|)`。Ordered表現ではdiag1/4・off1/2のweighted entry L1で、その二乗はRC-DF weighted-L1²族に含まれる。旧ENTRYのordered diag5/4・off3/2とは強度の比例変更だけで同じにならない。Mask・smoothing・scale・solverも公平比較の条件で、目的が完全に一致する場合は別名二手法の重複fitを必要としない。[BASE §2–5][BASE]

| 系列 | 重複する主題 | 保持する具体差 |
|---|---|---|
| RC-DF | tensor＋norm正則化、λ/資源との交換 | Cのtail mask・重み・smooth/scale・有限recipe。十分最適化した原法対照への優越は未取得 |
| SCDF/BLISS | symmetryを使う表現・norm改善 | target-sector shiftと、Cの全Fock同値Z再配置は別 |
| UWC | global OO/BLISS、可変Pauli head/tail、head費用＋tail²、RPE資源 | Cの固定CDF frame mask・独立J/U・own-DF taskと異なる |
| O1-M1 | 同じH_RのS/P/mixed辞書、λ/B/価格・shots交換 | CのJ/U fitは近似H自体が変わり得る。Z移送はD/Rを変える |
| O1-M2 | 実装しやすいcore＋完全残差、元H task | 保存C coreは素材になり得るが、Cのown-groundを元H groundへ無料置換できない |
| F1/F2 | typed辞書・operator/mean・full-wrapper単回価格 | 4mode人工系のN未資格。実H4・13round総費用との直接順位比較は不可 |

O1の今回再利用する固定HEADは `aeb9d1847f9d500f2389db12b384bac6087117a9`。11/7調査で公開refを取得した版であり、今回全branchの最新性を再監査したわけではない。独立課題未特定は、この固定調査範囲での管理判断材料である。

## 11. 再利用可能なコード・検証資産

| 機能 | 既存module・関数／入口 | 過去の検証・再利用先・制約 |
|---|---|---|
| 一般J/U fit | `c_pr_aware_df.make_value_and_grad/fit`、`c_near_objective.entry_weights/mass`、既存Newton-CG/HVP・stable/reduced/admission | E01、POST16〜26/40/43、統合。O1 core素材・既知目的比較に利用候補。新optimizer不要、U一次未達は残る |
| Codec/AD | `c1_real_orthogonal_codec`、`c1_ffsim_corrected_codec`、ffsim0.0.83/JAX接続 | H6 roundtrip・修正fit、点整合修正。Decode前後の微小差がgradientへ増幅する条件を維持 |
| Tensor/sector/CAR | `c1_general_k_adapter.from_ffsim_number_arrays`、`c1_stage1.corrected_kappa`、`SectorCAR`、`c1_stage3_reference` | 一般signed J、独立CAR・元g補正・scalar・spin/JW/sector。O1へは元H残差/state条件が別途必要 |
| Hard support | `c_df_structural_support.Support.project/reduce/expand/assess`、`pullback/compact_optimizer/fit` | 固定注入x=Ey・元loss/AD/codec再利用済み。P4結果・消費済みrunを保持し、再起動しない |
| 線形Z再配置 | `c_h4_relocated_downstream.prepare/raw_generators/tail_gate/strong_d` | CAR/full-H同値・再配置下流・強Dで検証。H4/CDF8/LD2/保存input用で、汎用元H importerではない |
| 決定論PF | `StrongBuilder`、`c1_stage2_native`、`c1_stage3_native_bridge` | 同J/U D・CDF/XDF強対照、phase/actual inverse。新core/順序/分子への無変更適合は未認定 |
| Finite RTE | `c1_stage3_connection`、既存`trotterlib.rte.make_rte_config`・law/normalization | 候補固有atom/identity・K0各r・mean/reference。F2 typed辞書への接続は同じtarget/半径の確認が要る |
| RPE/資源 | `c1_stage3_accuracy`、既存shot/radius/phase、`c1_h6_ca_resources`、serial会計 | Ideal/assumed/evidenceの分離、own-ground/alias、W/μ/shell。別taskや無料oracleへの資格移転なし |
| NativeとGaussian | `StrongBuilder`、phase/precision builder、head相対融合・tail exact-key共有、H6 ColumnStream | P0のfull512、H6の全sector入力、actual inverse・leakage。近似U共有やsingleton/cross-boundary性能は別 |
| 固定保証・reference | `c_pr_aware_df_fixed_k1`、Arb/CAR、`c_reference_bound_contract`、`c_b1_b2_reference_bound_integration` | 固定trajectory、sector reference二項を対象別に再利用。全word合成の代替にしない |
| 固定word pilot | `c_fixed_word_pilot`→`c_pr_integrated_performance.native(fixed_words=...)` | 4word/full512・phase/inverse/X-Y・実単価。期待価格やRPE十分条件の証拠とは別 |
| 価格・統計 | `c_r32_price_statistics.price_statistics`と各candidate adapter、既存Welch/final会計 | prospectiveとposthoc、cohort非pool、固定precision/influence政策。新順位を今回算定していない |

SOURCEはCSVの各実行版とmanifest closureを使う。収集mainには同名SOURCEの版差があるため、現在のsrcと任意の古いrawを組み合わせない。関数名は入口であり、全domainの汎用化保証ではない。O1/F2/Track Aで必要な部品の再利用は可能性として残すが、今回はSOURCE変更・adapter実装・pipeline統合・Track Aへの全面統合を行っていない。[EM §11][EM]、[OVER §8][OVER]、[統合SOURCE注意][ROBUST]

## 12. 未解決事項・未資格事項

| 論点 | 保存された状態 |
|---|---|
| 最適化 | 最新有限endpointはJ MET/U NOT_MET。原法RC-DF等を十分収束させた公平な最適能力比較、強度最適性は未成立 |
| Native/law | Whole-P0全word一様誤差、literal-Uとexact-unitary trajectory targetのbridge、sector→full512/shellの資格が不足 |
| 確率モデル | IID53-bit grid、有限bit PCG64価格、理想IID lawは別。Joint stream・occurrence/shot独立性・ideal価格移送を未証明 |
| Reference/CA | B2二項は取得済みだが、元generator形成、ground距離、全PF/RTE数値演算、native全law、物理/DF入力全体を一括保証しない。Stage C/actual CA未取得 |
| 統計 | 希少tail・support、nominal coverage、選択後/全候補/同時指標保証は未資格。005未解決、P/r16個別PR/Dは分布診断未解決 |
| 一般化 | H4 1.15Åは初期化STOPで移送結果なし。H2/H6は通常CDF/XDFの別rank/codec/coverage。PR-awareの分子間scaling証拠にしない |
| 新規性 | 目的の既知族対応は整理済み、独立方法差・論文上の新規性は未認定 |
| 原本coverage | 大規模rawの全byte/private元worktree・T7を未監査。公開原本と報告上の保存先を区別 |

これらは新しい計算を今すぐ始めるbacklogではない。今後強めたい主張をユーザーが選ぶ場合にだけ、必要な検証と別の実行／レビュー許可を判断する。

## 13. 研究管理上の一区切りと今後の位置付け

今回のユーザー指示に基づき、**案B「現成果で一区切り」を基本とし、案C「必要な部品をO1等で再利用」と両立させる**。理由は、(1)元PR-aware目的の優位がZ再配置後に維持されず、(2)残るENTRY＋PR利益は目的固有利益と別、(3)w_ZZ²は既知weighted-L1²族、(4)P4等の固定構造改善も総資源利益を示さず、(5)辞書/split/残差補償がO1の既存課題で、(6)11/7に別立ての具体的独立課題を特定できなかったためである。[OVER §7/9–11][OVER]

過去の肯定・負結果は有効な研究資産として残る。全科学問題の解決、将来の再開禁止、独立新規性の正式否定ではない。停止familyを改名して自動再開しない。

| 担当 | 今回の完了後 |
|---|---|
| Codex | 総括・索引の公開照合を終えてSTOP。自動的な次作業なし |
| ユーザー／研究C | 必要になった時だけ既存O1等で再利用したい対象を指定。今ここで追加判断や計算を要求しない |
| ユーザー | 将来新科学実行または重要レビューを希望する場合、それぞれ対象と範囲を別に明示許可 |

## 14. SOURCE・証拠索引・再現資料への案内

### 14.1 主要な固定原本

| 資料 | 固定Result／閲覧commit |
|---|---|
| [10/31 Evidence Map][EM]・[11/2更新][MAP2]・[11/3更新][MAP3] | `546bca7ed2cece59b714b986a264d7b0b0871a01`。収集carrierであり各科学SOURCEではない |
| [H4 1.00Å統合][H4] | `417fa459b823509be5c2010c5836e58955088282` |
| [Z mass診断][ZD] | `d601e0ef4375c37185211aa75685ae0e582f3e64`、科学SOURCE052941ab7ce680f57b0b5ba9ef27ef8cdd4937c1 |
| [Z下流][Z] | `3015d4fb342d4d98b58e41d57553e75539a07655`、継続科学SOURCEcfc64b70861a6482dddd042da8a110abebd6ad7a |
| [再配置強D][D] | `91417fb3ef75e7eec0849ad39f0649d170754664`、科学SOURCEe6652ab841aae5ae700286bf1ab249ad4fb85461 |
| [既知DF比較][BASE]・[重複整理][OVER] | `77ded5c12db21572933653cb26758436da3d9ecd`／`fd0dd0704ff3172c88c598aab91d63dbe2729c62`。静的資料 |
| [P4 fit][P4F]・[P4性能][P4]・[技術NO-GO][NG] | `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`にも固定公開。元科学SOURCEはCSVで分離 |
| [B2実行][B2]・[後続統合][B2I] | b7a3db09…／収集carrier546bca…、B2科学SOURCE01baf2b5e400773e00a5ae548941265f85b9882b |

### 14.2 CSVのschemaと再現の順序

CSVはUTF-8、27列、ID一意。`record_kind`、科学SOURCE、manifest/hash、入力identity、Result commit、report/raw/STOP、後続、reuse、静的確認範囲を別列にした。`legacy_evidence_group`で旧E番号へ戻れる。各SOURCE欄のcommitと、manifest内の実装基点base_commitを混同しない。SOURCE manifest列には継承元の関連manifestも含み得るため、具体scope・primary_execution_artifactと元報告のidentity表を併読する。

再現資料へは **ID→固定report→科学SOURCE/manifest＋入力hash→そのrunのraw/result/STOP** の順に進む。型が文書統合なら元実行のIDへ戻り、同じ計算を二重に数えない。過去runnerの再実行許可は本索引に含まれない。

統合H4の一部はcommit前のローカルSOURCE closureをhashで固定して実行した。科学SOURCE commitが特定できない行では `UNKNOWN_SOURCE_COMMIT_MANIFEST_PINNED` とし、Result417fa…を実行SOURCEと捏造していない。主closureは保存SHA256 `8113f23e66c5eca3c09a8ed1c91fc58c01eeaa91d0c07850b685a20c816539eb`、元serverは `200418968aa9e60e5ce842425e0373a4d8fafdb4f78c53a22df37b204989afd6`。18trajectoryの2checkpoint hashは既存TSVの文字列転記で、配列はdecodeしていない。

### 14.3 確認範囲・coverage limitation

正式共通指示原本2026-10-25、親/作業treeのAGENTS、PROJECT_MAP、研究概要・現状、研究結果索引、VALIDATION_STATUSと中央manifest関連記録を確認・既読資料として再利用した。中央資料には旧snapshot/収集版の時点差があり、最新固定C報告を優先した。旧manifest全体checkerの既知schema FAILを直したり、科学的PASSへ表示変更したりしていない。

固定文書・SOURCE/STOP metadata・既存索引から辿る範囲を整理した。Main収集版のreport blobは、GitHubの固定docs/research tree取得（truncated=false）とローカルobjectで照合した。Z/強D/重複/独立価格などの主要固定本文はGitHub接続で取得済み原本を再利用・追加取得した。公開metadataのSHA256照合はbyte同一性の技術確認で、旧科学結果の数値再現ではない。

**全remote branch/history、全raw byte、非公開サーバー原本を網羅したとは主張しない。** H4統合報告は約14GiBの大きいrawを元worktreeに保全し、GitにはJSON/TSV/必要入力・来歴等を選択収録したと明記している。今回そのprivate原本を確認していない。ほかのNPZ/QPY/JSONL等にも参照treeで未収録のpathがあり、CSVのunverified_artifact_pathsへ残した。公開pathの存在確認はdirectory内の全byte確認ではない。

未実行設計の将来run pathは、科学結果の欠落や失敗としない。E13/E15/E21等の技術資料に専用STOP.jsonが見つからない場合も、存在を捏造せず文書のSTOP記述・技術状態を保持した。元再配置processのwrapper STOP/completion欠落は、別termination記録と後続SOURCEのSTOPを分けた。

## 15. 保全確認・MANDATORY STOP

今回の変更は、新規 `REPORT.md` と `EVIDENCE_INDEX.csv` の二件だけ。既存Evidence Map、中央索引・manifest、旧報告・原稿、科学SOURCE、入力、raw、結果、固定契約、grant、ledger、marker、旧STOPを編集しない。他worktree/process/cache、共有venv/OS・他利用者に変更を加えない。AI sub-agentの使用0。

CSV schema/ID/参照、Markdown見出しとリンク、固定Git object/公開本文、二ファイル差分を静的に確認する。新しい科学testや数値計算を検査工程として追加しない。今回の確認を独立科学レビュー・外部再現・新しい保証とは呼ばない。

専用branch `codex/research-c-final-evidence-synthesis-20261107`、独立worktree、基点 `546bca7ed2cece59b714b986a264d7b0b0871a01`。GitHub接続でcommit・公開し、remote commit/tree、両本文、変更一覧、clean状態を照合する。実績SHAは最終チャットに記す。

**公開照合後MANDATORY STOP。** 新DF開発・fit・H6/H8移送・PR/RPE/native・O1実装・原稿執筆・重要レビュー・正式新規性/論文化判断・Track A統合・STOP解除へ自動進行しない。今回の次作業なし。

[EM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_evidence_synthesis_and_decision_readiness_20261031.md
[MAP2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_evidence_map_updated_20261102.md
[MAP3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_evidence_map_updated_20261103.md
[H4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_integrated_h4_100_results_20261106.md
[ZD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d601e0ef4375c37185211aa75685ae0e582f3e64/docs/research/research_c_h4_relocation_execution_20261106.md
[Z]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/docs/research/research_c_h4_relocated_downstream_results_20261106.md
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/91417fb3ef75e7eec0849ad39f0649d170754664/docs/research/research_c_h4_strong_d_relocation_results_20261107.md
[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/77ded5c12db21572933653cb26758436da3d9ecd/docs/research/research_c_baseline_comparability_20261106.md
[OVER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fd0dd0704ff3172c88c598aab91d63dbe2729c62/docs/research/research_c_existing_df_improvement_overlap_20261107.md
[P4F]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_df_structural_support_execution_20261028.md
[P4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_df_structural_pr_performance_20261028.md
[NG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md
[B2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7a3db09e395d36965e5b9dc28ae4a299db3b2f0/docs/research/research_c_reference_bound_execution_20261103.md
[B2I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_b1_b2_reference_bound_integration_20261103.md
[ROBUST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_integrated_robustness_preparation_20261105.md
