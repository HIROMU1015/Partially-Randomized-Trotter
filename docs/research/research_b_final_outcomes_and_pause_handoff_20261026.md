# 研究B 最終成果統合・一時保留handoff（2026-10-26）

Repository：HIROMU1015/Partially-Randomized-Trotter。固定作業起点：[266b8f887eccef57e85e0eb9866e90f4f06a0b32][BASE]。
**研究Bの運用状態：一時保留。追加科学計算を停止し、既存成果・SOURCE・STOPを保全する。**
本書はユーザーの一時保留方針を反映する静的統合で、新しい科学結果・保存値再解析・独立科学レビュー・数学的保証の認定ではない。

## 1. Executive Summary

研究Bは、既知の高次PF・PR-RTE・DF native手法に対する誤差診断・実装検証・条件付き資源評価として成果を得た。D内部PF、outer D/R分割、finite-RTEを分けた対照と、相対位相を保つprepared X/Y wrapper、有限IID費用、未知rare費用を残すshot会計を公開した。[旧統合][INTEGRATED]

独立した新算法の具体的機構差は、既存の限定先行調査では特定されなかった。これは新算法の不存在証明ではない。[B-Next 1][PRIOR]
後続のExact-R、Full DF-PF q2、Full Yoshida4 q4によって強い対照が加わった。固定synthetic2mode／N1では、Full Yoshida4 q4が観測bias b=1.21606169604673e-10、各prepared axis RZ2356／CX1636を示し、ε=1e-9の参考予算 b≤5e-10以内だった。[q4結果][Q4]

この決定論的単価は高精度PR候補D1／R2／R4の保存S0費用平均より小さい。ただしq4のbiasはR2/R4より大きく、同じ予算内であることは同一biasの達成ではない。u_num／γ、PRのrare費用、Full PFを含むshot加重G、方法別最適化は未取得・未実施であり、保証付き同精度総資源順位は確定していない。

以上を現在の到達点として、**追加科学計算を停止して一時保留する**。永久終了・既存成果の否定・新科学GOを意味しない。現段階で新しい研究課題の探索・採択・実行計画を起動しない。

## 2. 研究Bの目的と対象

初期の目的は、量子化学時間発展で目標精度を維持しながら量子資源を減らす方法を調べることだった。限定系列では「D内部の高次化の利得が、outer分割とfinite-RTEを通ったphysical coherent signalに残り、native単価とshot負担を含めても有益か」を問うた。[旧統合 §2][INTEGRATED]

この系列の対象は、状態準備込み・単一複素signalの限定評価である。プロジェクト本体の状態準備を除く全RPE interrogationの目的関数や、量子化学エネルギー精度の達成と区別する。単一signalのεをそのままenergy誤差へ置換しない。

既知係数・processing・RTE・DF primitiveの組合せによる評価を、新しいPF次数、新RTE方式、独立した新算法の発明とは呼ばない。

## 3. 適用範囲・固定物理条件

| 項目 | 本統合の数値の範囲 |
|---|---|
| 共通モデル | Synthetic 2-mode、one-body＋3 DF因子。同じphysical Hamiltonian／入力状態 |
| Sector／state | N=1、mode0 occupation。実分子の基底状態・多電子相関の検証ではない |
| Geometry／AO basis／分子DF rank探索 | 非該当。DF因子は固定3本 |
| PR／Exact-R分割 | L_D=2 weight-ranked prefix、D=one-body＋DF0/1、R=DF2のidentity分離後 |
| Full DF-PF | one-body／DF0／DF1／DF2の全4 block。D/R、n_D、RTE r/Kを導入しない |
| 36行 | δ=±0.09、q=2,3、A1〜A4のn_D=1,2、r/K=2/2。q間ではT=qδが異なる |
| 後続の主比較 | T=±0.18、PRは主にq2・n_D1、候補別r/K・refinement。Exact-R q2、Full PF q2／Yoshida4 q4 |
| 数値・correctness | Complex128、operator spectral 2-norm atol1e-12／rtol0、phase fitなし。A0高精度は別の90/128桁補足 |
| Native共通条件 | Ordinary control、mode0準備・X/Y readout・measurement込みfull wrapper、phase-preserving native-v5 |
| Compiler | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、approximation1、qubits_initially_zero=False、free topology |
| 主指標 | Physical complex coherent signal直接絶対誤差。Full-Fock／N1 operator、X/Y／半径、raw／correctedを別に保持 |

入力非可換性・identityは各実行の既存検査を根拠とする。N1ではDF二乗の作用がsector内で一体作用へ還元されるというB-Next 1の説明も、このtoyの特殊性を示す。非可換であっても、多電子・多数tail・実分子への一般化は検証されていない。[PRIOR]

## 4. 全実験系列とSOURCE・公開結果

科学SOURCEと結果公開commitを分けて示す。リンクは固定commit。以下の行数・call数は既報値の引用で、今回合算・再生成していない。SOURCE・raw・STOPへの対応は§18も参照。

| 系列 | 科学／解析SOURCE | 結果公開commit・報告 | 完了範囲・科学的状態 |
|---|---|---|---|
| D内部PF36行 | [758a05ed6][SRC-D36] | [51b138799・報告][D36] | 36行、941 dispatch／448 key。A3/A4優劣はINCONCLUSIVE |
| Outer splitting16行 | [b85dd3284][SRC-O16] | [b1d9f46ee・報告][O16] | 16行、906 dispatch／490 key。固定範囲のcorrected改善 |
| Outer／finite-RTE22行 | [26e8734af][SRC-R22] | [33715848b・報告][R22] | 22行、1239 dispatch／274 key。数値資格未確定 |
| A0高精度 | [1e3921ba1][SRC-HP] | [31662b491・報告][HP] | 8行×90/128桁、EMPIRICAL_CONSISTENCY |
| B0 native最終完了 | [b190fe28b][SRC-NATIVE] | [0db9b3ed1・報告][NATIVE] | 48代表／96 wrapper、全correctness後に単価保存 |
| B1 IID費用 | [11e2532a6][SRC-MC] | [bef63edbd・報告][MC] | 3,072 trajectory／6,144 wrapper、INCONCLUSIVE_RARE_EVENT_COVERAGE |
| Rare条件付き感度 | [a8cf99156][SRC-RARE] | [cae3cd784・報告][RARE] | 8確率／16境界／80 SE corner、CONDITIONAL_SENSITIVITY_ONLY |
| Shot加重24条件 | [9bebdc46e][SRC-SHOT] | [e80cf902c・報告][SHOT] | 48 shot要求／144資源行／24比較、INCONCLUSIVE_QUALIFICATIONS_AND_RARE_COSTS |
| Exact-R | [e76c04748][SRC-EXACT] | [58502f554・報告][EXACT] | 4 case／8 wrapper、36 operator／16 expectation、DESCRIPTIVE_ONLY |
| Full DF-PF q2 | [8811ef089][SRC-Q2] | [3244f2af9・報告][Q2] | 4 case／8 wrapper、72 operator／16 expectation、DESCRIPTIVE_ONLY |
| Full Yoshida4 q4 | [620448b14][SRC-Q4] | [266b8f887・報告][Q4] | FD5/FD6／4 wrapper、44 operator／8 expectation、観測bias予算内 |

旧統合は[3cfbac1d4版][INTEGRATED]を当時の記録として保持する。関連する[静的構造費][STRUCT]、[B-Next 1][PRIOR]、[B-Next 3準備][READY3]、[B-Next 4準備][READY4]、[q4実行][Q4]も相互参照する。

B-Next 2という単独の番号付き公開報告・過去GPT科学レビュー原文の固定参照は、指定起点のdocs/researchと取得資料から特定できなかった。未取得の内容を推定して経緯を補わず、B-Next 1が明記するレビュー原文未取得の制約を引き継ぐ。

## 5. PF内部／outer／finite-RTEの科学的成果

| 診断対象 | 既存証拠で分かったこと | 結論の境界 |
|---|---|---|
| D内部高次化 | A2 Yoshida4、A3 published Y8m10b、A4 published processed YP8m8を対照。A4はA0 partial-S2／A1 D-S2より小さいfull signal誤差を観測 | A0→A1にはnested配置変更もある。全利得をYP8m8固有効果に帰属しない |
| Processing／published列 | A4数学参照4件・published列規約4件を独立照合し、実現Pのactual inverseを使用 | A3/A4のfull差は丸め水準で識別不能。Published法の独自改善を確認したとはしない |
| Processor費 | R障壁を越えてprocessorを無料相殺できず、短いD runの端点費を構造数で整理 | 構造指数数はnative費・同精度費用ではない。A4のnative劣位を実測したわけではない |
| Outer分割 | 同Tの16行でS2約2.27e-7に対しYoshida4約4.53–6.76e-10のcorrected signal誤差を観測 | 固定時間・異なる仕事量の記述比較。一般収束次数・同資源優位ではない |
| Finite-RTE | A3を内部対照としたD1／R2／R4でr/K変更によるfull bias変化を確認 | A2ではD近似が残り、RTE成分の改善とfull複素誤差は単調に対応しない |
| Raw／corrected | Yoshida4でcorrected meanが改善してもraw physical信号は改善せず、normalization負担が増えた | Mean補正と量子shot負担を同一視しない |
| Ideal-tail | Finite tailと理想R指数を対照し、D・outer・tailの残差を分けた | 複素差の相殺があり、normの単純差引きから因果寄与率を作らない |
| SABA2 | 既知の弱摂動向け正係数分割としてsignal・構造を評価 | 近似D／finite-RTEを含む固定toyの結果。一般四次・弱摂動保証を認定しない |

根拠：[36行][D36]、[構造費][STRUCT]、[16行][O16]、[22行][R22]。本統合は新しい誤差分解、改善率、次数fitを算出していない。

A0高精度は、保存binary64入力をexact liftし90桁Taylor／128桁Padéで照合した補足である。S1/F1の旧nominal bound超過は高精度では再現せず、経験的整合が得られた。一方、入力identity・Hermiticity等の微小な前提差も保持し、厳密な包含やcertified u_numにしていない。旧22行をA0の都合のよい値へ差し替えない。[HP]

## 6. Native実装とcorrectness

D/Rのordered列、signed時間、Gaussian basis往復とactual inverse、scalar／identity phase、control0 identityとcontrol1相対位相を、mode0準備込みX/Y Hadamard wrapperへ接続した。System、controlled、raw／compiled wrapperとexpectationを独立数学参照へ照合し、全correctnessの後に性能・費用を測る経路を実装した。[NATIVE][NATIVE]

初期B0のcompiled不一致は保存・停止し、許可された復旧で微小RY/U角と離散scalar phaseを保つnative-v5へ修正した。固定tolを緩めたりglobal phaseでfitしたりせず、最終48代表の288 operator／192 expectation後に96単価を取得した。初期失敗・診断・STOPは残っており、初回から成功した履歴へ書き換えない。

後続B1、Exact-R、Full PFも同じprepared wrapper／phase-preserving compiler条件で独立correctnessを通過した。技術的なPASSは、実現した回路の経験的照合であり、新算法の新規性、全supportの保証、有限bit実装meanのγ保証とは別である。人工toy／mockテストも科学性能の証拠に格上げしない。

## 7. MC／rare-event／shot加重評価

| 費用の種類 | 既存成果 | 現在使えない解釈 |
|---|---|---|
| 決定論的wrapper単価 | Exact-R／Full PFの固定回路RZ・CX・depth・size | Shot加重総費用、方式別最小資源 |
| B0代表trajectory単価 | 各候補4決定論的代表の実価格 | 確率重み付き期待値、rare母平均、全support上界 |
| B1有限IID平均 | 各candidate/Tに256 trajectory、X/Y paired。Mean・variance・SEを保存 | X/Yを独立512標本として扱わない。無条件母平均保証ではない |
| S0条件付き参考費用 | 観測標本が全order0だった範囲の経験費用 | 小さいSEで未知rare費用を覆ったとはしない |
| Rare感度 | 未知のrare条件付き平均を変数に残すaffine境界・±SE感度 | 未取得費用の0補完、母平均順位、矩形を信頼区間にすること |
| Shot加重Gの条件付き会計 | 24条件・保存normalization・十分shot整数とS0費用を接続 | 無条件G／winner、保証付き同精度総資源 |
| 保証付き同精度資源 | 未取得のu_num／γ／rare費用等を明示 | nullをcorrectness atolや高精度一致で埋めない |

B1は全51,200 short-stepでorder0、order2/4は未観測だった。18,432 operator／12,288 expectationは通過したが、科学状態はINCONCLUSIVE_RARE_EVENT_COVERAGEを維持する。B0の高次数代表をB1へ混ぜてcoverageを補完しない。[MC]

Rare解析はB4/B3のS0差が±2SEで残り、D1/R4は±1SEで識別不能という条件付き結果を保存した。未知費用の有限上界がないため、無条件順位は確定しない。状態はCONDITIONAL_SENSITIVITY_ONLY。[RARE]

Shot評価はfresh IID bounded outcome、軸別α=.025、統計予算2ε/5等の固定モデルによる十分数で、各axis約1.096e18／1.096e20（ε=1e-8／1e-9）を記録した。実量子shotは実行していない。Scalar normalization enclosureはsignal全体のu_numや実装γの保証ではなく、全24条件のend-to-end資格はnull。Full PF／Exact-Rへ旧shot整数やGを移送していない。[SHOT]

## 8. Exact-R baselineで追加できた情報

単一DF因子Rを既存Gaussian＋DF二乗対角primitiveで実装した。Centered Rのidentity補正と全体scalarを分け、RTE normalizationは1だが、D内部A2／outerの近似は残る。全Hの無料exact oracleを導入したものではない。

Outer Yoshida4／A2／Exact-Rはb=5.926752262417758e-10、RZ/CX=4114/2800。SABA2／A2／Exact-Rはb=2.9044769538026594e-9、2100/1424（両T・各axis）。旧ideal-tail信号と整合し、RTEを除去してもD／outer誤差が残ることを確認した。4 case・8 wrapperの記述的結果で、finite-RTEの無条件期待費用との総資源優位は認定していない。[EXACT]

## 9. Full deterministic DF-PF q2

元のone-body＋全3因子をS2／Yoshida4で近似し、元c0=.17を一度だけ付けた。D/R extractionのidentityを二重計上せず、Full PFにn_D／RTEを使わない。全72 operator／16 expectation後に4 case／8 wrapperを測定した。

Full S2 q2はb=1.80763525850668e-6、RZ/CX=406/276。Full Yoshida4 q2はb=1.947702126906798e-9、1186/820（両T・各axis）。q2同士でも指数列・仕事量は異なり、同資源比較でも方式別最適化済み比較でもない。[Q2]

## 10. Full DF-Yoshida4 q4

既存のq引数付きbuilderと数学参照を再利用し、FD5／FD6のq4・T±.18だけを一回評価した。全138 call、44 operator／8 expectation後に4 wrapper単価を保存し、科学中断・retry/resumeは0だった。

両Tのb=1.21606169604673e-10は参考点予算b≤5e-10以内、WITHIN_OBSERVED_BIAS_BUDGET。各X/YのRZ2356／CX1636、RZ depth1544／CX depth1636、total depth3609、size4996。u_num／γはnullで、b+u_numによる精度保証ではない。Compiled operator correctness残差と数学的PF biasも別量である。[Q4]

## 11. PR／Exact-R／Full PFの最終比較

共通条件は§3のsynthetic2mode、N1/mode0、T±.18、同じphysical H／state、ordinary controlled prepared X/Y、native-v5。以下は既存公開表の転記で、各方法の新しい最適順位・比率を計算していない。

| 方式 | 観測複素signal誤差b（両T） | 各prepared X/YのRZ／CX単価 | 根拠 |
|---|---:|---:|---|
| Full DF-S2 q2 | 1.80763525850668e-6 | 406／276 | [q2][Q2] |
| Full DF-Yoshida4 q2 | 1.947702126906798e-9 | 1186／820 | [q2][Q2] |
| Full DF-Yoshida4 q4 | 1.21606169604673e-10 | 2356／1636 | [q4][Q4] |
| SABA2／A2／Exact-R q2,n_D1 | 2.9044769538026594e-9 | 2100／1424 | [Exact-R][EXACT] |
| Outer Yoshida4／A2／Exact-R q2,n_D1 | 5.926752262417758e-10 | 4114／2800 | [Exact-R][EXACT] |

| PR候補 | 条件（共通q2,n_D1,LD2,T±.18） | 観測bias | 費用の位置付け |
|---|---|---:|---|
| D1 | Outer Yoshida4／A3、r2K2 | 4.526927441031103e-10 | B1有限S0標本平均 |
| R2 | Outer Yoshida4／A3、r4K2 | 8.337749292598329e-11 | 同上 |
| R4 | Outer Yoshida4／A3、r2K4 | 7.003598416351649e-11 | 同上 |

Biasは[22行][R22]、次のRZ/CX mean・SEは[B1][MC]を引用済みの[q4報告 §8][Q4]からそのまま転記した。各axis n=256、X/Yが同じ保存値のためTごと一行。SEは費用の有限標本不確かさで、量子shotのconfidenceやrare tailの保証ではない。

| 候補 | T | RZ mean | RZ SE | CX mean | CX SE |
|---|---:|---:|---:|---:|---:|
| D1 | 9/50 | 27423.63281249999 | 1.65512926553982 | 18772.046875000004 | 1.2939304808042782 |
| D1 | -9/50 | 27423.23828125001 | 1.675764681519188 | 18771.531250000004 | 1.3164200912389705 |
| R2 | 9/50 | 27548.050781249996 | 2.3266371655368503 | 18868.609374999985 | 1.8070571484276778 |
| R2 | -9/50 | 27549.84765625 | 2.3831614754699606 | 18870.203124999993 | 1.8433332343024578 |
| R4 | 9/50 | 27424.371093749996 | 1.707662527135671 | 18772.656250000015 | 1.3492108600411123 |
| R4 | -9/50 | 27423.453124999993 | 1.4478127387232453 | 18771.62499999999 | 1.1578536176909597 |

Full PF q4は同じ観測bias予算内の高精度PR3候補に対し、保存S0平均より小さい決定論的単価を示した。**少なくとも今回の固定synthetic・有限候補では、PR構成の低資源性を示すために強い決定論的baselineが不可欠であり、現段階でPRの一般的な資源優位は支持できない。** これはq4報告を受けた限定的な研究判断材料で、一般劣位定理ではない。

一方、q4のbiasはR2/R4より大きい。同じ参考予算内と同一biasを区別し、PRのrare費用、u_num／γ、Full PFを含むshot加重G、全方式の最適q／最小総資源を未確定のまま残す。q4の小さい単価を保証付き同精度総資源優位へ格上げしない。

強い対照追加前は、主にPR内部候補間の精度・S0会計を見ていた。追加後は、同じfixtureで既知Full PFが観測予算を満たす低単価点を持つことが分かった。従って、内部改善だけでPR全体の資源上の価値を支える解釈は維持できず、追加実行より先に独立した問いの必要性を整理する段階となった。

## 12. 先行研究・過去研究との関係

以下は[B-Next 1の限定静的調査][PRIOR]の対応を引き継ぐ。今回、原論文の新たな網羅調査・再証明・独立した先行性認定は行っていない。

| 既存研究・方法 | 既知の対応 | 研究Bが追加した評価と境界 |
|---|---|---|
| PR原論文 | 一般高次PFのR stageをRTEで置換、signed時間、stage別step、normalization積 | 固定finite K／DF native／複素signalを詳しく対照。高次PR自体は新機構ではない |
| SPRINT | 支配groupを高次化するnested構成、内部反復、processing、残差のRTE等への接続 | 現BはSPRINT full pipelineの実装・同等性検証ではない |
| Morales higher-order／processed PF | Published Y8m10b／YP8m8とprocessor構成 | D内部へ適用し、R障壁ごとの端点とfull signalを照合した |
| Corrected PF | Corrector・共役・誤差項除去の既知構成 | 現A4はMoralesでCPF実装ではない。Finite-RTEへの保証移送は未確定 |
| SABA系 | 弱摂動向けpositive splitting、既知SABA2 | 近似D／finite tail下の記述的評価。一般四次を主張しない |
| Stage別RTE配分 | PRのr_j、絶対時間配分、旧H246の整数allocation | 現Bの候補別r/K対照は一般stage最適化済みではなく、新算法機構ともいえない |
| 過去H2/H4/H6 | 高次PRとshot×native費用を既に有限探索評価 | 現Bのnested D／outer／tail対照は具体的追加情報だが、費用評価という問いは相当部分重なる |
| 終了済みR2-CS | 低費用構成の事前予測・選択という別の独立研究 | 今回の実測baseline比較と区別。再開・名称変更による復活をしない |

[旧H246原報告][H246]は線形H2/H4/H6、隣接1 Å、STO-3G、tol-only1e-8、保存DF rank3/10/19、prefix1/3/5、T=.8、P=0、K4、各outer合計32 step、q1/2/4/8、symmetric-directionalという別条件だった。保存経験的allowanceによる探索点の高次PF利得は既報だが、現Bのprepared ordinary wrapper／native-v5／ε1e-9へ価格や削減率を転用しない。旧m5決定論部分集合との比較を全強化PFへの一般優位にしない。

R2-CSは[R2_CS_INDEPENDENT_SELECTION_RESEARCH_CLOSED][R2CS]を維持する。P1はREMOVED_FROM_R2_CS_EXECUTION_PLAN_UNEXECUTEDで、未実行のまま予定から除外されている。研究Bの一時保留とR2-CSの正式終了を同じ状態にしない。

過去GPT／Claude科学レビュー原文は本作業では取得していない。指定のGitHub原資料、そこで記録された判断、B-Next 1の確認範囲を根拠とし、レビュー原文を読了・再レビューしたとは記載しない。

## 13. 既存証拠で支持される主張

| 支持される主張 | 根拠・限定 |
|---|---|
| D内部・outer・finite-RTEを分けた誤差診断を実施した | 36／16／22行。同じ実験内の対照で、一般寄与率や収束定理ではない |
| 位相保持controlled native wrapperのcorrectnessを検証した | B0/B1とbaselineの実現回路、固定tol・phase fitなし。γ資格とは別 |
| 指定した有限trajectoryと固定回路のnative単価を取得した | 代表／IID／決定論を区別する |
| Rare未観測が期待費用評価を制限することを確認した | B1全order0と未知費用を残す感度・会計 |
| Exact-RとFull DF-PFを強いbaselineとして接続・検証した | 同一H/state/T、prepared ordinary wrapper／native-v5 |
| Full Yoshida4 q4が観測bias予算内に入った | bだけの参考判定。u_num／γはnull |
| q4の決定論的単価がPR3候補の保存S0平均より小さい | 固定synthetic・有限候補。保証付き総資源順位ではない |

これらは公開commitに記録されたローカル検証の成果で、CI再生成・独立外部再現・実量子backend評価とは区別する。

## 14. 現時点で支持されない主張

| 支持されない主張 | 不足する根拠 |
|---|---|
| 研究Bが独立した新算法を提示した | 近接する既知法と異なる具体的機構差が未特定 |
| PRが強い決定論的PFより一般に低資源 | 固定q4対照があり、一般化・方法別最適化・総資源資格がない |
| Synthetic結果が多電子・実分子でも成立する | N1・単一残差因子という特殊性。移送検証なし |
| B1 meanが無条件期待費用を保証する | Rare条件付き費用・分散等が未取得 |
| 保証付き同精度shot加重総資源順位が確定した | u_num／γ／rare費用、baselineのG、同時保証・最適化が未確定 |
| A1／γの厳密資格を取得した | A0は経験的一致、native atolは証明ではない |
| 全候補・全パラメータの最適PFを特定した | 有限の固定点評価で、未実行域を探索していない |

「未確立」は「数学的に不可能」と同義ではない。実行COMPLETEを科学的GOへ、条件付き評価のINCONCLUSIVEを既存全成果の失敗へ読み替えない。

## 15. 残る科学的課題（現在は追跡しない）

| 未解決事項 | 現在ある証拠 | 保留時の扱い |
|---|---|---|
| Rare込み無条件期待費用 | B1 S0標本と条件付き感度 | 未知のまま保全。追加sampling／compileを起動しない |
| u_num／γ | A0経験的一致、scalar区間、実現回路correctness | 資格nullを維持。A1包含・保証計算を始めない |
| Shot加重総資源 | 既存PR24条件の条件付き会計 | Full PF／Exact-RのGを新計算しない |
| 方法別同精度最適化 | 固定q2／q4と既存r/K対照 | q・n_D・r/K・新係数を探索しない |
| 多電子・実分子への一般化 | 再利用候補のDF／sector／nativeコード | 新しい系・入力・回路を生成しない |
| 独立した改善機構 | 既知方法との対応・未特定の差分 | 未解決という理由だけで新研究計画を作らない |

## 16. 一時保留の理由

1. 独立した新算法の機構差を特定できていない。
2. 高次PRと資源費用という問いは、過去H2/H4/H6研究・近接先行研究と相当部分が重なる。
3. 強いFull DF-PFを追加すると、現synthetic条件でPRの一般的な低資源性を支える根拠は得られなかった。
4. 追加MC、shot、精度保証、分子展開は証拠を増やせても、それだけで算法的新規性は成立しない。
5. 現段階では、追加計算に優先して答えるべき新しい科学的問いが明確でない。

これはユーザーが選んだ研究運用上の一時保留で、Codexによる新しい独立科学NO-GOや永久終了の認定ではない。PF/RTEの層別診断、正しいnative実装、rare問題、強いbaselineの重要性は再利用可能な成果として保持する。保証取得を完了のための追加作業にせず、現在の限定成果と未解決事項を残して停止する。

## 17. 将来の再開条件

次のいずれかが具体化し、研究判断を変え得る理由を説明できる場合に、ユーザーが再開を検討する。

| 判断材料 | 明確にする内容 |
|---|---|
| 独立した改善機構 | 既知PR／SPRINT／PFと異なる数学的構成・計算機構・誤差除去原理。名称変更やパラメータ変更だけでは不十分 |
| 明確に異なる応用課題 | 過去H246と重複せず、強い決定論的baselineに対し何を判断するtaskか |
| 科学的結論を変え得る新証拠 | 単なる点数追加ではなく、既存判断のどこが変わるか |

再開する場合は、研究課題、既存SOURCEの適合性・再利用範囲、入力・精度・比較条件、必要な科学call scope、STOP条件を限定する。新科学実行には別途ユーザーの明示許可が必要で、重要な詳細レビューも実行許可とは別に判断する。チャットでの明示許可で足り、承認証明書・保存承認原文・grant等を新設しない。

本節は将来のhandoff条件であり、新しい提案の採択・科学契約の起動・計算許可ではない。旧runをresume／retryしない。

## 18. SOURCE・結果・grant／ledger／marker／STOPの保全

本書のSOURCE一覧は§4の既存科学commitを指す。本作業の文書commitを新科学SOURCEと呼ばず、新たな科学実行契約も作らない。旧報告の「当時未完了」「将来候補」「STOP」は歴史として残し、最新の状態は本書と索引で追加する。

| 系列 | 既存raw／STOP参照（artifacts以下） |
|---|---|
| D内部36行 | [research_b_b2d_stage1/20261025/technical-r1/STOP.json](../../artifacts/research_b_b2d_stage1/20261025/technical-r1/STOP.json)。初回simple-onceも保全 |
| Outer16行 | [research_b_outer_splitting/20261025/once-16rows/STOP.json](../../artifacts/research_b_outer_splitting/20261025/once-16rows/STOP.json) |
| Outer/RTE22行 | [research_b_outer_rte22/20261025/analytic-once/STOP.json](../../artifacts/research_b_outer_rte22/20261025/analytic-once/STOP.json) |
| A0高精度 | [research_b_outer_a0/20261025/independent-once/STOP.json](../../artifacts/research_b_outer_a0/20261025/independent-once/STOP.json) |
| B0最終完了 | [research_b_outer_b0/20261025/corrected-2/STOP.json](../../artifacts/research_b_outer_b0/20261025/corrected-2/STOP.json)。旧native-pilot・診断・corrected-1も保全 |
| B1 IID | [research_b_outer_b1/20261025/iid256-once/STOP.json](../../artifacts/research_b_outer_b1/20261025/iid256-once/STOP.json) |
| Rare感度 | [research_b_outer_rare_sensitivity/20261025/saved-once/STOP.json](../../artifacts/research_b_outer_rare_sensitivity/20261025/saved-once/STOP.json) |
| Shot24条件 | [research_b_outer_shot_weighted/20261025/saved-once/STOP.json](../../artifacts/research_b_outer_shot_weighted/20261025/saved-once/STOP.json) |
| Exact-R | [research_b_outer_exact_r/20261026/exact-r-once/STOP.json](../../artifacts/research_b_outer_exact_r/20261026/exact-r-once/STOP.json) |
| Full PF q2 | [research_b_outer_full_df_pf/20261026/full-df-once/STOP.json](../../artifacts/research_b_outer_full_df_pf/20261026/full-df-once/STOP.json) |
| Full Yoshida4 q4 | [research_b_outer_full_df_y4_q4/20261026/q4-once/STOP.json](../../artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/STOP.json) |

原本・入力・結果・環境・provenanceは各固定結果commitの報告から追跡できる。旧SOURCE／raw／grant／ledger／marker／STOP、R2-CS終了記録、共通指示、他研究・共有環境は変更しない。新pause marker・承認証明・grant・管理frameworkは作らず、一時保留を本書と既存索引へ記載する。

確認範囲は、指定7資料のGitHub本文取得、旧8一次報告の必要な条件・結論・制限事項、既存SOURCE／結果pathと表の対応である。新たな原論文の全面調査、過去レビュー原文の読了、全コードclosure監査を行ったとはしない。

文書検査はリンク、表の引用、SOURCE対応、変更範囲・旧原本不変、公開内容の照合に限定する。科学module import、runner inspect／preflight／execute、数値テスト、回路生成／compile、保存値の新しい差分・比・統計・G計算はすべて行わない。科学status・既存validation manifestは変更せず、新しい数値証拠を登録しない。

## 19. 引き継ぎ先と次の判断

**ユーザー（研究B）**は成果を保全した一時保留状態を受け取る。現在、追加科学計算や詳細レビューについて新たな判断・操作を行う必要はない。将来§17の条件が成立したときだけ、再開の科学的問いと必要な許可を判断する。

**Codex**は本書と必要最小限の既存索引・研究ノートをcommit・pushし、remote SHA／公開内容を確認してmandatory STOPする。一時保留中の自動計算、追加候補、研究課題探索、重要レビュー、他研究への移行、研究統括への送信を行わない。

研究Bは、既知の高次PF・PR-RTE・DF native手法に対する誤差診断・実装検証・条件付き資源評価として一定の成果を得た。一方、独立した新算法の機構差は特定できず、強い決定論的Full DF-PF baselineは固定synthetic条件で観測bias予算内の低い回路単価を示した。一般的な同精度総資源優位は未確立であるため、現段階では追加科学計算を停止して一時保留する。

旧成果・過去契約を遡及変更せず、研究Bを永久終了としない。**公開確認後mandatory STOP。現在の追加作業なし。**

[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/266b8f887eccef57e85e0eb9866e90f4f06a0b32
[INTEGRATED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/docs/research/research_b_integrated_outcomes_and_future_directions_20261025.md
[PRIOR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4a8ef63081011c890d68bfd1cab78515a3d01f02/docs/research/research_b_bnext1_prior_art_and_strong_baselines_20261026.md
[EXACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58502f5549d350a8da8cbc23a6e674c4670feb9b/docs/research/research_b_bnext3_exact_r_execution_20261026.md
[Q2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3244f2af9c4017c57c180b9057b22e3eb19c62b1/docs/research/research_b_full_deterministic_df_pf_execution_20261026.md
[Q4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/266b8f887eccef57e85e0eb9866e90f4f06a0b32/docs/research/research_b_full_df_yoshida4_q4_execution_20261026.md
[H246]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8613b3ecda2030a677b1690fe7aa3da23dec4873/docs/research/track_a_h246_saved_cost_results_20261021.md
[R2CS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/270078a87c1febe8944f829270e8c2a00dbed7de/docs/research/r2_cs_closure_20261024/REPORT.md
[D36]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51b1387996b1b53feb550efa7cb98aadcf5f9189/docs/research/b2d_stage1_execution_20261025.md
[O16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/docs/research/research_b_outer_splitting_stage1_execution_20261025.md
[R22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_rte_22row_execution_20261025.md
[HP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31662b49157b15f2d1c03eef1c2faa07e2f905e7/docs/research/research_b_outer_a0_execution_20261025.md
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0db9b3ed189eb6cd9d66e070993dbd0e72d29104/docs/research/research_b_outer_b0_recovery_20261025.md
[MC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bef63edbd8cc564444b44e4e8cf23baa90f05fe5/docs/research/research_b_outer_b1_expected_native_cost_20261025.md
[RARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cae3cd784dbdfcf2fe96902345379aca4b9606c7/docs/research/research_b_outer_rare_event_minimal_sensitivity_20261025.md
[SHOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/docs/research/research_b_outer_shot_weighted_execution_20261025.md
[SRC-D36]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/758a05ed6aac58a5945af5a5330c792a842b174d
[SRC-O16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b85dd328423de5b1159f11939835f04db90f3187
[SRC-R22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/26e8734aff3969dc8077cb058826d12394be28e1
[SRC-HP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/1e3921ba16e5d531e9a3010cffa9ff717609a93a
[SRC-NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b190fe28b98d2f94e1509e19cd3d36a63b2bb99e
[SRC-MC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/11e2532a62d54a65f8fad5489be6b135f826011b
[SRC-RARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a8cf99156af692f0facaa1885ab9ed52b28c6f4c
[SRC-SHOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/9bebdc46ebd60039f315da9b6c6e57bd9c004b85
[SRC-EXACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e76c0474806dca42598fa92a6564cded10896f98
[SRC-Q2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/8811ef089d588a529aca001f9e40cef1b2f6e1c7
[SRC-Q4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/620448b14235d2d9e0e76d5d0ba7273ce887a628
[STRUCT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f5d60122f158b3320fadab86498963950f00734/docs/research/research_b_post_b2d_stage1_static_cost_outer_splitting_20261025.md
[READY3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7056267ed6e49b573353c815a89bbba7da9ae3fa/docs/research/research_b_bnext3_exact_r_execution_readiness_20261026.md
[READY4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e9214e1a8f68b8c32ca2b06f69a743311f63e68/docs/research/research_b_full_deterministic_df_pf_readiness_20261026.md
