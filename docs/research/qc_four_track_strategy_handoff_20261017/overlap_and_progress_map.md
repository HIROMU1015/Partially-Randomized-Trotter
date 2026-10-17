# 4研究系列の重複防止・進捗対応表

2026-10-17 JST。[研究戦略判断用ハンドオフ](strategy_handoff.md)の付属資料。
固定資料から確認できる**正式状態・検討済み範囲・未検討の範囲**を並べ、同じ研究を名称だけ変えて再開・重複実施しないための一覧である。
新しい科学レビュー、新規性判定、研究優先順位の決定、新候補の提案ではない。行の並びは順位を意味しない。

- 確認時点：remote-tracking refsを2026-10-17 19:09 JSTに取得した状態（最新commit `b357c77a`、19:03:12）。①は更新が続いており、H8 Phase Cは承認済み・実行待ち。
- 各IDは[固定証拠索引](fixed_evidence_index.json)の `key_evidence[].id` に対応する（repository・branch・commit・path・bytes・SHA-256・GitHub取得照合）。
- 「新しいと数えるのに既存資料が要求する差」の列は、既存資料が明記した要件の転記であり、今回新たに設けた基準ではない。
- 「別名例」は、名称の言い換えによる重複を検出するための例示である。既存資料の検討対象名でも新候補でもない。

## 1. 状態の区分

| 区分 | この資料での意味 | 例 |
| --- | --- | --- |
| ユーザー正式状態 | decision.json、ユーザー指示原本、利用者採択・採用を明記した記録に根拠があるもの | SOS-PF A3、D+E、V2 closure |
| レビュー判断・推奨 | GPT／Claudeの科学レビューの判断。レビュー内で「採択」「採用」と書かれていても、ユーザーの正式採択記録がなければここに入れる | R09/R10の方針、C1現系列終了、M1-Sのノート化 |
| 技術判定・技術STOP | Codex等の技術分類、gate STOP、mandatory STOP | NO_GO_NO_INDEPENDENT_GENERATION_RULE、COMPLETE_MANDATORY_STOP |
| 実行承認 | 個別one-shot計算への承認。結果の科学的採択ではない | Track A各runのauthorization receipt |

技術判定やmandatory STOPを正式なCLOSED/STOPへ格上げしない。限定条件の否定的結果を一般の不可能性へ拡張しない。正式状態には原典の適用範囲（「現」構成・「現」定義・「独立新手法として」等）を付けて読む。

## 2. 正式状態一覧（変更しない）

| 対象 | 系列 | 正式状態（適用範囲） | 根拠 | 同じ対象の技術判定・レビュー判断（正式状態ではない） |
| --- | --- | --- | --- | --- |
| N1 表現selector | ④ | STOP（現候補scan。完了済みrecoveryは保持） | [D08]（初出[D09]） | 固定比較で改善0%（合成JW4・signed DF rank2） |
| N2 signed charge・L1 fit | ④ | HOLD | [D08]（初出[D09]） | 旧charge回路はdirectより高価 |
| N3 occupation certificate | ④ | 現検証系列（追加benchmark・帰属系列）のみCLOSED、主研究テーマ不採択、理論・限定prototypeは条件付き保持 | [D08] | 再開経路A/Bは記録のみで実行承認ではない |
| THRIFT／mixed-access現template | ④ | 現template feasibility系列CLOSED（THRIFT/mixed-access一般は閉じていない） | [D10]、再開trigger [D11] | 登録価格modelで予算不足 |
| 一体項再配分PF（第一候補） | ④ | 現固定構成のみCLOSED | [D07] | 固定同一情報アクセス対照内でREDUCES_TO_KNOWN [D21]、登録10%改善4/4未達 [D20] |
| H-E1 | ④ | 現定義での探索のみCLOSED | [D07] | K0 gate不通過 [D22] |
| M-a／M-b | ④ | 現定義での追加探索STOP（M K1は未実施・未認可） | [D07] | core REDUCES_TO_KNOWN、UNDETERMINED_NOT_READY [D23] |
| PF小改変探索系列 | ④ | PAUSED（この系列に限る） | [D07] | — |
| 旧D（DF因子等価回転） | ④ | HOLD | [D07] | 非不変な目的での研究は未実施 |
| SOS-PF | ④ | 現PF-native分割生成探索のみCLOSED（一般SOS-PFのno-goではない） | [D06] | K0 UNDETERMINED_DATA_REQUIRED [D18] |
| SOS-PF保証理論A2 | ④ | NOT_ADOPTED（CLOSEDではない、可能性として保持、研究未開始） | [D06] | — |
| SOS-PF K1-F | ④ | NOT_AUTHORIZED／NOT_EXECUTED | [D06] | — |
| H1〜H4 | ④ | 現探索PAUSED（当該K0.5/K1は未認可・未実施） | [D04]、[D05] | 静的K0：H1/H3/H4はNO_GO_NO_INDEPENDENT_GENERATION_RULE、H2個別UNDETERMINED_DATA_REQUIRED [D12] |
| ④新アルゴリズム研究全体 | ④ | 研究課題選定段階、新アルゴリズム正式採択なし | [D04]、[D05] | — |
| 旧候補C1〜C4 | ④ | **正式記録なし**（[D07]はC1_C4をUNCHANGEDと記録） | — | GPT再レビュー：C1現系列終了、C2は対照として保持、C3/C4現構成への投資見送り [D14] |
| Parent allocation（同一fixture数値探索） | ② | D+E正式採択（限定成果の整理＋区切り。新計算承認・一般no-goではない） | [B29]、[B30] | — |
| DF三次補正 | ② | generic whole系列の自動拡張をCLOSED（新アルゴリズム不採択、Track B全体は終了しない） | [B12] | — |
| G5 固定toy・同辞書の実用優位主線 | ② | 区切り（利用者採用のGPT G4レビュー） | [B07]、[B08] | — |
| G6〜G10 有限Taylor return-aggregation検証系列 | ② | 区切り、技術ノート完成、独立原著投稿は推奨しない（利用者採用） | [B10]、[B40]、[B11] | — |
| Gaussian積生成候補 | ② | 独立新手法としてCLOSED（manifest上は利用者採用、逐語未記録）。性能はUNDETERMINED（negativeではない） | [B13] | — |
| Track B新規探索 | ② | 一時休止（manifest上は利用者採用、逐語未記録）。**ユーザーによる明示的な解除記録は未確認** | [B14]、[B43] | 同日のGPT broad review [B16]（ユーザー開始承認）は②既存成果の発展を休止の対象外とし研究設計の再開を支持。以後D1〜D3・Parent allocation・M1-Sが実施された |
| 新しい相関sampler／補正／compiler系列 | ② | 休止レビュー（manifest上利用者採用）で不採択 | [B43] | 後のステップ間相関監査はKNOWN_PROCEDURE_MAPPING [B15]、GPTは主線にしないと推奨（正式記録なし） |
| BF現仮説 | ② | closure記録あり（GPT推奨に基づくCodex記録、逐語の採択文言は未確認） | [B01] | 限定negative |
| BM現adapterのnew-method路線 | ② | closure記録あり（post-BM05 GPTレビュー受領に基づくCodex記録、逐語未確認） | [B39] | compact BCHと三次係数一致 [B02] |
| D1〜D3、次期方向4候補群 | ② | **正式記録なし** | — | D1〜D3数値系列の一区切り（GPT推奨）[B21]、4候補群は未採択 [B32] |
| M1（部分乱択内側qDRIFT） | ②/③境界 | **正式記録なし**（M1_adopted=false） | [B35]、[B36] | GPTがM1_NOT_ADOPTED・M1-S再定義を暫定支持 [B37] |
| M1-S研究方向 | ③ | **正式な終了・継続・HOLDの採択なし**（research_direction_formally_adopted=false）。統合記録公開後のmandatory STOP | [C06] | Claude：限定ノート化・PR-2等へ資源回帰。GPT：修正technical note／研究記録として確保、資源改善路線の新規実行は保留。いずれもNOT_ADOPTED [C04]、[C05] |
| M1-S G0-T1限定理論契約と各個別契約 | ③ | 正式採択済み（過去one-shot。現在の実行許可ではない） | [C08]、[C03] | T1a/T1bはGPT Checkpoint 1/2で適用域付き受理 |
| Track A V2現行残差fit設計 | ① | CLOSED_NO_GO_BY_USER_DECISION（decision_owner USER、利用者原文の引用なし） | [A29] | 来歴照合で比較可能PF 0件 [A28]。代替解析3案は未採択 |
| Claude L1レビュー（Track A） | ① | 利用者採択範囲は「V1/V2 technical preparation only; H5 unchanged」のみ（推定値の科学採択false） | [A22] | 着地・H5/H7非推奨等の推奨は採択範囲外 |
| Track Aの各結果 | ① | **ユーザーによる正式な科学採択記録なし**。各runは個別のユーザー実行承認（H7は承認文言未記録・digestのみ）によるone-shotで、技術・契約STOPで終わったattemptや複数attemptで完了した系列（H8 B0 anchor、B2）を含み、いずれも終了後mandatory STOP | 各結果文書 | R09/R10（GPT）はH4 240軸・H8 core・D0等を限定／条件付きbenchmarkとして採用 [A04]、[A09]。R10後の結果について、GPTの採否レビュー（R11等）はない（10-16時点 [A46]）。Claude L1 [A21] はH6/H8 native全12構成・T panel・H5/H7方針を含めて評価したが、利用者の採択はV1/V2技術準備に限られ推定値の科学採択はfalse [A22]。V1、H8 Phase A/B、非相殺診断を対象とする公開レビューは確認範囲では見つかっていない |
| 高次PF第2研究 | 別repository | Direction Cを主方向として固定（ユーザー承認2026-10-05） | [P02] | 今回はGitHub raw取得で記載を確認しただけ |
| 高次PF第1研究 | 別repository | S4 benefit条件不成立によりS5へ進まない（最終統合の記載） | [P01] | 同上 |

## 3. 進捗対応表（確認時点）

| 系列 | 直近に完了したもの | 進行中 | ユーザー判断待ち（未承認） |
| --- | --- | --- | --- |
| ① Track A | H8 Phase C統合・実行準備（公開commit 18:34〜18:45）[A44] とユーザーの一回実行承認・再固定（19:03、科学実行0）[A47]。H8 Phase B2 登録S4 4軸PASS（計算完了17:46、公開commit 18:00）[A33]。非相殺shot診断（公開commit 17:45）[A38] | H8 Phase C（承認済み・実行待ち。確認時点で科学実行0、公開記録上の実行中jobなし） | 非相殺COST_COMPARISON stage、V1追加実測 [A26]、V2代替解析3案、H5/H7の次段階、R10以後の結果に対するGPTの採否レビュー |
| ② Track B（M1-S除く） | D+E整理と次期方向Decision Packet [B32]（10-15）、M1 P0/P0.5 [B35]/[B36] | なし | 次期方向（4候補群I〜IV、M2、III/Track A連携、決定論剰余補正、休止）、Parent allocation C2〜C5、D1〜D3原稿化 |
| ③ M1-S | 統合記録 [C01]〜[C03]、worktree整理 [C07] | なし | 研究方向（終了・ノート化・限定再開R0〜R7） |
| ④ 新アルゴリズム | 研究範囲対応表 [D01]〜[D05]（10-16） | なし | GPT Level 1判断の開始承認 [D03] |

④は 56f1cc55 以後、正式状態を変えたcommitがない（全remote branchを確認）。③は 5f079dbe 以後、管理文書（worktree整理）のみ。

## 4. 重複防止マップ

各行の数値条件は原典に従う。異なる分子・費用指標・価格modelの倍率を比較しない。

| 方向（別名例） | 重なる既存成果・先行研究 | 重なる対象の正式状態 | 実際に検討した範囲 | 未検討／結果未取得／費用未評価／既知法還元の別 | 新しいと数えるのに既存資料が要求する差 |
| --- | --- | --- | --- | --- | --- |
| Hamiltonian分割の変更（一体項再配分、fragment順序・配置、DF因子回転、PF-native分割生成）。別名例：fragment最適化、分割共同設計、split調整 | ④一体再配分PF、H-E1、M-a/M-b、旧D、SOS-PF候補A、静的K0 H1。②BF・BM、次期方向候補群III。先行：FFF、SPRINT/GRADE、compact BCH、CDF/FRO | 一体再配分 現固定構成のみCLOSED、H-E1 現定義の探索CLOSED、M-a/M-b 現定義での追加探索STOP、PF小改変PAUSED、旧D HOLD [D07]。SOS-PF 現分割生成探索CLOSED（一般no-goではない）[D06]。H1 現探索PAUSED [D04]。BF/BMはCodex記録のclosure | 二分割S2の三次BCHと期待値符号、多fragmentの残余三体・四体項（静的K0）。一体再配分v2：H4 1.00 Å/STO-3G/DF rank12、1e-4・1e-5 Ha、10%改善4/4未達 [D20]。BMはcompact BCHと一致 [B02] | 未検討：非不変な目的によるDF因子回転（旧D HOLD）。費用未評価：多fragment一般DFの局所更新生成、情報取得込み総費用。既知還元：外側交換子消去（Hejazi/processed PF）、一体項配置（FFF、SPRINT/GRADE） | 係数変更・再配置・既知corrector・外挿を越える入力→回路規則。既知CDF/FRO/軌道最適化と区別できる低費用生成規則。同じH・state・control・精度で強い既知対照と比較 [D12]、[D14]。係数・配置の小変更を新領域と数えない [D01] |
| 軌道回転・Gaussian frame・interaction picture。別名例：時間依存Gaussian frame、Cartan分解、orbital-adapted PR | ④H3、THRIFT現template、mixed-access constructor、旧表現研究S12〜S14、旧D。②Gaussian積生成。①structured Gaussian lowering（実装） | THRIFT現template CLOSED（THRIFT一般は閉じていない）[D10]、H3 現探索PAUSED [D04]、Gaussian積生成は独立新手法としてCLOSED（性能UNDETERMINED）[B13]、旧D HOLD | 回転平方の指数化は既知Gaussian共役。異frameの平方和の指数を一primitiveとして無料（単一DF-square価格）には数えられない [D12]。THRIFT：H4 1 Å/DF16全保持で理想PFのstep削減は保たれたが登録価格modelで予算不足 | 費用未評価：Gaussian積のconditional full-wrapper費用、THRIFT可逆算術の実単価。既知還元：controlled Gaussian compression等 | H3：既知Gaussian共役を越えるprimitiveは未構成 [D12]。THRIFT現templateの再開trigger（次のいずれか：同機能で登録既知Kを減らす具体設計、有利instance探索前に定義した機構的証拠、結論を変える具体的矛盾・新証拠。新primitiveの発明は必須ではない）[D11]。②のclosureを④H3の否定的結果に読み替えない |
| 低体数化・counterterm・補正PF・決定論剰余の補正。別名例：state-specific補正、corrector、低rank補正挿入 | ④H1、一体再配分、H-E1、M-a/M-b。②DF三次補正、次期方向GPTレビュー [B34] §9表の「N3：決定論剰余の確率的補正」（保留）。先行：Hejazi、CPF、NCC/HNCC、LCU Trotter compensation | 一体再配分 現固定構成CLOSED、H-E1 現定義CLOSED、M-a/M-b 現定義での追加探索STOP [D07]、H1 現探索PAUSED [D04]、DF三次補正 generic whole系列の自動拡張をCLOSED（Track B全体は継続）[B12] | DF三次補正：固定加法計画（gate間cancellationなし）でsynthetic rank2×3の3条件でwhole A下界がPauli B上界を上回る（例A≥576対B 112 T。最適compiler・whole一般の下界ではない）。M K0：coreは既知 [D23] | 費用未評価：補正演算子の実装費用、同信号精度でのPR総費用。結果未取得：H-E1の多fragment移送、M K1 | 同じ情報を使うcounterterm/fit対照に対する独立した配分・生成・保証の優位 [D23]。量子化学構造による安価な補正演算子の具体構造 [D14] |
| SOS/PSDによる誤差保証・有限step certificate。別名例：PSD誤差bound、gap依存保証 | ④SOS-PF K0/K0.5/A2、N3（Schur/Fantope/SDP/SOS）、観点C5。保証理論として②D3（Taylor剰余）と③T1a/T1b（C-sec）が別系統 | SOS-PF 現分割生成探索CLOSED、A2 NOT_ADOPTED、K1-F未認可 [D06]。N3 現検証系列CLOSED（理論・限定prototypeは条件付き保持）[D08] | 紙上導出とKang–Su等との対応のみ（SDP・数値0件）[D18]、[D19]。N3は限定classのrational PSD証拠 | 未着手：A2。結果未取得：有用なfinite-h certificate。既知還元：SOS/DF/BLISS/square completion/Gram | 誤差上界だけの改善と新生成算法を区別し、係数から安く得た証拠で回路・測定条件が変わり強い既知対照を越えること [D02]。A2着手には別途明示承認 |
| PF外挿（Richardson型）。別名例：多幅PF補外、h→0外挿 | ④旧C2、一体再配分監査のEXT2対照 | C2は正式記録なし（GPTは対照として保持）[D14] | 静的監査のみ（二点外挿、gap/branch、誤差伝播）[D15] | 結果未取得：安価な同一branch情報とremainder証明。費用未評価：複数幅controlled費用。外挿一般は未検討ではない [D01] | 安価な同一branch同定・remainder取得・複数幅control費用を含め、同精度・同成功確率で総費用優位を生む具体機構 [D15]。新方式でもC2を強い既知対照に含める [D14] |
| self-energy・downfolding・active space縮約。別名例：有効Hamiltonian、downfolded PR | ④旧C3、N3。先行：Feshbach/Schur、DUCC、Schrieffer–Wolff | 旧C3は正式記録なし（GPTは現構成への投資見送り）[D14]。N3 現検証系列CLOSED [D08] | C3は静的監査（Schur root、単調性、resolvent誤差）[D15]。N3は限定classで元ε_projでの追加縮約なし | 費用未評価：安価なencoding/ground solver、高body項、normalization。既存QSVT/LCU中心の実現はD1（block-encoding除外）では流用できない | D1範囲内の安価な実現と総費用差（C3）。N3は再開経路A（独立に動機づけた狭幅task）またはB（既知原理を越える理論差）[D08]。「N3」は②次期方向レビューの別項目と名称衝突（§6） |
| cost-aware randomization（sampling law、係数配分、importance sampling）。別名例：価格考慮乱択、係数配分、proposal最適化、相関乱択 | ④旧C4。②R0〜R1.5、RA-D0、G1〜G5、SP/PAI、D1〜D3、Parent allocation、ステップ間相関、候補群II。①PR候補格子（既知PRの評価）。先行：Resource-Optimal IS、SPS、SCU/CTS、PAI/TE-PAI、qSHIFT | 旧C4は正式記録なし。G5区切り [B07]。Parent allocation D+E [B29]。休止記録 [B14]。新しい相関sampler系列は不採択 [B43]。D1〜D3・候補群IIは正式記録なし | Parent allocation：3-qubit synthetic、h=0、十分N、加法native Tで頂点＋ISがcommonよりT約14%（ordinary）／19%（P5）低い（N・CX増）[B28]。C4は現構成が既知IS等へ還元 [D16] | 費用未評価：online古典費用、全event価格表の取得費用、多step、分子DF/PR/RPE。既知還元：cost×second-moment IS、q∝\|a\|/√g、rejection/zero-fill、SPS/PAI置換機構 | 既知IS/SPSと異なる具体生成機構、非列挙取得の保証、実PR taskでの新証拠 [B30]。単にsamplingを変えるのは新領域ではない [D01]。休止記録の境界（具体task・手順・利益理由・判別検証・ユーザー承認のGPT判断）[B14] |
| fermion encoding・対称性sector・tapering。別名例：sector圧縮、Lie代数可解化 | ④範囲表の狭い⑤(1)、N1、S12〜S14、THRIFT、観点C1/C3。①H5/H7のROHF・固定sector（既知encodingの実装） | ⑤は採択・新規性認定ではない [D01]。N1 STOP（現候補scan）、THRIFT現template CLOSED（THRIFT一般は閉じていない）[D10] | 旧表現研究S12〜S14で群平均/echo、圧縮空間漏洩、conditional SU(2)、小Lie closureをtoy確認し、Patel/THRIFTとの重複もレビュー済み [D01]。2026-10-16にGSE/tapering・Cartan/Patelの文献箇所を照合 [D02]。代数・対称性全般は⑤に置かない | **検討記録不足の可能性（狭い⑤）**：同じ元H・state access・絶対energy精度で encoded preparation/control/readout まで含めた比較。既知mappingの工学的比較に留まる可能性も明記 | 既知mappingを選ぶ工学的比較を越える回路生成規則と費用機構。対象sectorに元問題のgroundが含まれる契約 [D02]。古典のsector行列縮小から安い量子oracleは導けない |
| coherent共有乱択（±branch間label共有）。別名例：correlated ± randomization、coherent qDRIFT制御 | ③M1-S。②M1（P0/P0.5）、ステップ間相関（step間で軸が異なる）。①PR原論文とTrack Aは独立±RTE。先行：Campbell controlled qDRIFT、MLMC-qDRIFT、COS、GüntherほかPR、Hagan–Wiebe | M1-S方向は正式採択なし・mandatory STOP [C06]。S0理論契約・各個別契約は正式（過去one-shot）[C08]。M1は正式記録なし | T1a/T1b（固定有限次元・固定r・δ→0）。H4 1.00 Å/STO-3G/DF rank12/G3・G6・G9の保存oracleでGate 0、有限step、γ V1、π/3、Stage S [C01] | 費用未評価：whole-wrapper単価、HD実装、状態準備、強い対照。結果未取得：Task B（多round RPE）、T1c、一般DF event | 再開ハンドオフR0〜R7 [C03]。最新GPTは、資源研究を続ける場合は同一推定task・利用情報・whole-wrapper費用・比較対象を一つの判断単位にまとめ、実用最良法へのclaimでは決定論PF・PR型RTEも対照に含めるよう推奨（既定推奨は修正ノート整理。いずれもNOT_ADOPTED）[C04] |
| 強い決定論的baselineの整備（B0+I、prefix細格子、打切りS4、非相殺bias） | ①H4/H6/H8 B0+I、V1、H8 Phase A/B/C、非相殺診断。②matched CTS、P3+tail等。④一体再配分のS2/CS2/S4/EXT2対照 | Track A結果のユーザー正式採択記録なし。R06/R09/R10（GPT）がB0+Iを強化対照として採用 [A09] | H6 V1：対照136候補でPR削減率がε=.05で−13.29%（実測点）[A24]、他4精度は固定model条件付き30.89/34.97/31.17/10.03% [A25]。H8強化格子はB2完了、Phase Cはユーザー実行承認済み・確認時点で結果なし [A33]、[A47] | 結果未取得：H8強化後のPR比較、V1残り4精度の実測。比較未成立：最良PF一般（SPRINT/GRADE等） | 新アルゴリズムではなく既知PR評価の対照整備（R09）。④で新方式を主張する場合も同じ強い対照を同条件で与える |
| 測定・exact ground copies・RDMによる絶対energy情報取得。別名例：joint測定、collective measurement、shadow-based energy | ④H2、H4、狭い⑤(2)、観点C4。③M1-S Task A（単一時刻位相の十分shot診断で絶対energyではない）。先行：fermionic shadows、joint測定、sample-based simulation、QFD | H1〜H4 現探索PAUSED [D04]。H2技術判定UNDETERMINED_DATA_REQUIRED [D12]。⑤は採択ではない。測定研究を範囲に含めるかは未決 | 静的K0の混合恒等式・filter・H→H+cI不変性の整理。L8で通常のfermionic joint測定まで既知と確認し探索停止 [D02] | **検討記録不足の可能性（狭い⑤）**：入力差（copies、coherent preparationと逆、複数copy）を固定した同一absolute-energy費用比較。直接測定・RDM・単一copy joint測定は既知 | Hに依存した実行可能な読出し手順、位相・絶対offsetの扱い、copiesとcoherent preparationを混同しない精度保証、全費用 [D02]。「joint測定」の名称だけでは新課題にしない |
| representation selector・低rank表現の費用込み選択。別名例：adaptive DF rank、cluster表現 | ④N1、N2。①DF prefix選択（既知DF打切りの比較）。②候補群III | N1 STOP（現候補scan）、N2 HOLD [D08]。候補群III未採択 | N1：selectorは全3 contextでexactを選び改善0% | 結果未取得：一般分子、FT、最終energy task | 固定比較不成立を「representation未検討」と言い換えない [D01]。近似許容幅がshot負担を上回る具体機構 |
| macrocycle・異符号PF合成による誤差相殺 | ④旧C1。②BF（高次PF schedule）、BM | C1は正式記録なし（GPTは現系列終了）[D14] | 非分子2-qubit Pauli H、24 macrocycle、通常16 contextで登録5%優位なし [D17] | 結果未取得：catalog外、分子、真値なしselector。既知還元：既知composition | 固定macrocycleの段数増加・既知compositionとの重複・構成機構不足を解消する機構 [D14]。新catalog生成へ進まない [D01] |
| native費用会計（compile実測、加法合成価格、理論gate数、basis費用帰属） | ①Qiskit 1.3.0 compile 6指標、理論model A/B [A17]〜[A19]、V2。②pygridsynth加法T/CX/1Q（D1〜D3、G9/G10、Parent allocation）、basis費用診断 [B38] | V2現行残差設計CLOSED（ユーザー）[A29]。理論評価は scientific_adoption=false | ①H6/H8の208候補で理論832行（ranking・残差なし）。②3-qubit syntheticの加法会計 | 費用未評価：T-count/Toffoli・FT合成（①）、online古典費用、実PRで最大の費用要因（NOT_ESTABLISHED）[B31] | 同じ「native費用」でも①compile実測、②加法合成価格、①理論命令数は別量。倍率・順位を同一指標として扱わない |
| RPE/QPE全roundと元Hの絶対基底energy総費用 | ③M1-S Task B（未採択）、②M1 P0/P0.5、①T=0.8の有限時間signal、④D2 | 4系列とも全round総費用を評価・採択した記録なし。R10はenergy/QPE/RPEをTrack A benchmark完成の必須条件にしない [A09] | M1-S Task A（十分shot診断）、Track Aの有限時間signal会計 | 未検討・費用未評価：overlap/prior/alias、raw位相→E·T bridge、全dyadic round、成功条件、state preparation | Task Aの達成はTask Bの保証にならない [C01]。有限時間signal精度ε（Haではない）をenergy精度と読み替えない |
| 有限step bias上界・経験的allowance・非相殺診断 | ①経験的u（certified=false）、非相殺診断 [A38]。②D3 Taylor剰余。③T1a/T1b。④A2、C2のgap/branch認証 | ①非相殺COST_COMPARISON未承認。A2 NOT_ADOPTED [D06] | ①H8：ε=.05で決定論16候補が非相殺条件で不適格、PR6不変、shot比最大46.2（感度診断）[A38] | 結果未取得：厳密bias certificate（Track A全系列でnull）、有用なfinite-h certificate | 前提（Hermitian contraction、固定有限次元、保存DF model）と対象（operator／signal／固有位相）が異なる保証を互いに移し替えない |
| identity scalar・energy原点・定数位相 | ①B0+I。④H4（copies・phase reference）、観点C4、D2 | ①B0+IはGPTが強化対照として採用。④H4 現探索PAUSED | ①H4 240軸で回路指標差0、B0→B0+Iの費用低下はscalar保持によるshot変化 [A03] | 既知還元：scalar保持と既知identity係数の加算は既知 | scalar保持そのものを新理論にしない（R09）。④は未知原点の同定であり①とは問いが異なる |
| 長時間・多step・T依存でのPR利益の残存 | ①H6固定δ T panel [A14]、Claude L1の時間範囲概算（未検証・未採択）。②Parent allocation C5、次期方向レビューの主眼。③M1-S T1c/R2 | ①H6 T panelはR10（GPT）が限定比較として採択し、実行承認のもとで完了（結果の科学採択記録なし）。長時間・多step域の定量会計、Claude L1の概算、②C5、③T1c/R2は未承認・未採択 | ①T≤1.6のみ。ε≤.001の一部セルでfull S4優位またはPR不適格（T=.4/ε=.001はPR/B0（登録決定論最良）=0.45）。5構成固定・各Tで再最適化なし | 未検討：長時間・多step域の定量会計 | 同一task・同精度・同成功確率での長時間会計 |
| basis変換費用の削減（基底共有、可換block延長、境界相殺） | ②basis費用診断 [B38]、候補群I、次期方向M2（第二候補として保持）、ステップ間相関、D2境界相殺。④H3、旧D | M2・候補群Iは未採択。Gaussian積生成は独立新手法としてCLOSED（性能UNDETERMINED）[B13] | D2：literal境界相殺によるP5対partialの相対T差の変化は約0.02 pp（P5自身のT切片減少約0.41%とは別、3-qubit synthetic）[B42]。H4 order0 36 eventでRZの84.8%がbasis由来（加法比） | 結果未取得：実H_Rでの可換性、HWP/early-FT座標 | 厳密な可換gate並べ替え（既知compilerとの非同値性が論点）と分布を変えるM2を別案として比較 [B34]。84.8%を削減可能量と読まない [B31] |
| return集約・非列挙生成・normalizer取得（closed P5を含む） | ②G6〜G10、D1〜D3、候補群IV。④旧C4 | G6〜G10有限Taylor return-aggregation検証系列は区切り（利用者採用）[B40]。D1〜D3は正式記録なし | 3-qubit syntheticでclosed P5のT切片がordinary比約26.6%低い等（m=5、表示値算術 [B43]）[B09] | 結果未取得：分子DF規模access、他provider。費用未評価：online古典費用 | 独立priority・非自明性は先行文献の未発見だけでは示せない。最適IS・SPS等との比較は、主張を広げる場合に限って必要（一律の必須条件ではない）[B21] |

## 5. 系列間で同じ方法・入力・指標を扱う箇所

- **同じ分子ラベルでも別実験**：「linear H4 1.00 Å/STO-3G/DF rank12」は③M1-S（PR-2 S0/S1開発NPZ、分割G3/G6/G9）[C02]、①H4 native fixed16（legacy DF12）、④一体再配分（36次元sector）で使われ方が異なる。①H4 geometryは0.70〜1.60 Åで1.00 Åを含まない。THRIFTはDF16全保持。②BFもH4 1.00 Å/DF12/L_D=3を使う。
- **Hadamard testの複素信号**：①z(T)（T=0.8）、③有限step信号（T=1,4 Ha⁻¹）、④H2の混合相関は推定対象・状態・時刻・精度定義が異なり、いずれも全roundの絶対基底energy推定ではない。
- **±RTEの扱い**：PR原論文と①は独立乱数の±RTE。①のsymmetric_directional controlは決定論部のloweringで、乱択tailの共有実装の証拠ではない [B36]。①のPR費用と③の十分shotは別protocolの量。
- **三次BCH誤差の補正**：②DF三次補正と④M-a/M-b・H1・H-E1が同じ主題を別入力・別指標で扱う。一方の局所結果を他方の結論に流用しない。
- **Gaussian・軌道回転**：①structured Gaussian lowering（実装）、②Gaussian積生成（独立新手法としてCLOSED、性能UNDETERMINED）、④H3（現探索PAUSED）・THRIFT現template（CLOSED、THRIFT一般は閉じていない）・旧D（HOLD）。
- **候補群III（表現・前処理・native access）**：②次期方向レビューは「Track Aとの連携候補」とし、Track Bからの除外は作業振り分けで科学的negativeではないとした [B34]。④観点C1/C3、N1/N2、旧Dと主題が重なる。
- **研究方針の推奨**：①Claude L1の着地推奨は未採択（利用者採択はV1/V2技術準備のみ、推定値の科学採択false [A22]）。③Claudeレビュー（PR-2へ資源回帰）も未採択。③最新GPTは他研究との定量的優先比較を行っていないと明記 [C04]。

## 6. 名称衝突

| 名称 | 指すもの（別物） |
| --- | --- |
| H4 | 分子（直線水素4原子鎖。①②③④で共通の名前だが条件が異なる、§5）／④観点H4（copies・phase reference、現探索PAUSED） |
| N3 | ④N3 occupation certificate（現検証系列CLOSED）／②次期方向GPTレビュー [B34] §9表の「N3：決定論剰余の確率的補正」（保留、未採択） |
| M1・M2 | ②M1（部分乱択内側qDRIFT、P0/P0.5）／M1-A・M1-S（[B37]の分割）／PR-2系のM1-A・M1-B1／②次期方向のM2（基底共有・可換block延長）／PR-2系のM2（held-out transfer契約） |
| G1〜G10、G3/G6/G9、G2/G9/G10 | ②Track B G系列段階（G5固定辞書、G6〜G10 return集約）／③M1-Sの分割G3/G6/G9／④範囲表・技術基盤表の証拠ID（G2、G9、G10等） |
| Q1〜Q15 | [ハンドオフ](strategy_handoff.md)§7の問いID（上記G系列とは無関係） |
| D0〜D6・D1〜D3・D1/D2・旧D・D+E | ①追加検証設計D0〜D6 [A42]／②finite-RTE開発D1〜D3／④固定条件D1（block-encoding系除外）・D2（元Hの絶対基底energy）／④旧D（DF等価回転）／②D+E（Parent allocation整理） |
| B0/B1/B2/B3/B0+I と H8 B0 anchor・Phase B1/B2 | ①比較方式（B2＝PR）／①H8強化格子のB0 anchor取得、Phase B1（S2費用model）、Phase B2（S4 model検証） |
| S4 | Yoshida四次PF／②Parent allocationのS4段階（KKT訂正・補完）／10-14統合資料のscope card S4／高次PF第1研究の段階S4（別repository）／④provenanceのsource ID S4 |
| V0〜V3・V1/V2 | ①Claude L1の検証項目（V1強化baseline、V2理論費用接続、V3/V4は未採択）／③M1-Sのγ V1とRPE decision packetのV0〜V3／②Parent allocationのrun v1〜v3 |
| A2/A3 | ④SOS-PFの決定（A2保証理論、A3採用方針） |
| C1〜C5 | ④旧候補C1〜C4／④観点C1〜C5／③M1-S decision packetの式ラベルC1〜C3・契約C2／②Parent allocation C2〜C5（追加分析） |
| R0〜R7・R01〜R10・R0〜R1.5 | ③M1-S再開候補／①GPT科学レビュー索引（R06、R09、R10等）／②係数再配分R0〜R1.5 |

## 7. 再開根拠・状態解釈として不十分なもの

**既存資料が再開根拠として不十分と明記するもの（系列ごと）**

- N3 closure：ログ欠落だけ、「より大きな系なら勝つかもしれない」、結果を見てからのε変更・成功例だけ [D08]。
- THRIFT現template closure：時間・文書・branchの変化による自動再開はない。Kを変えずFだけを節約することは再開根拠にならない [D11]。
- ④範囲表：「未実装」「検索語に一致しない」ことを未検討・新規の根拠にしない。係数・fragment配置の小変更、split調整、sampling変更、joint測定の呼称など、停止系列を名称変更で新領域として数えない [D01]、[D02]。

**既存資料の解釈上の注意（再開根拠の規定ではない）**

- 系列間で否定的所見・局所肯定例を相互に流用しない（C1/C4の否定的所見とTrack A/Bの結果を相互に移さない [X01]、名前から研究状態や認可を移送しない [D01]）。
- 技術PASS・実行完了・mandatory STOPを科学的GOや正式CLOSEDと読まない（数値PASSを科学GOとしない [C01]、静的K0の技術判定は正式PAUSE/CLOSEDの追加判断ではない [D12]）。

## 参照ID（固定GitHub本文）

各IDのrepository・branch・commit・bytes・SHA-256・GitHub取得照合は[固定証拠索引](fixed_evidence_index.json)の `key_evidence` にある。

- [A03]：H4 B0+I全240軸実測（MAP240_COMPLETE_STOP）。拡張集合の点最小はB0+I 397/B2 1,415
- [A04]：R09（GPT原本exact copy）。H4を限定benchmarkとして採用。PR初提案・最良PF一般への勝利・scalar保持初提案を主張しない
- [A09]：R10（Codex索引でGPTに帰属。本文に自己記名なし）。条件付きbenchmark採用、B0+I強化対照、H5→H7・T比較を科学採択。実行認可なし
- [A14]：H6固定δ=.2のT=.4/.8/1.6比較。ε≤.001でS4優位またはPR不適格のセルあり
- [A17]：native DF理論gate数モデルA/B（固定分解RZ/CX/SX/X）。T/Toffoli・depth・routing・state prepは対象外
- [A18]：理論モデルとQiskit 1.3.0 builderの人工入力接続検証。比較契約への採否はGPT判断待ちと明記
- [A19]：H6/H8 208候補の理論832行。ranking・残差なし、scientific_adoption=false
- [A21]：Claude Level 1レビュー原本（dd695e65にも同一blob）。新規性なし、V1必須・V2推奨・H5/H7非推奨
- [A22]：Claude L1の利用者採択範囲『V1/V2 technical preparation only; H5 unchanged』、scientific_adoption_of_review_estimates=false
- [A24]：B0I p7/q1 S2両軸実測でε=.05のPR削減率−13.29%（逆転維持）
- [A25]：V1成果整理。ε=.05のみ実測点比較、他4精度は固定model条件付き。追加実測は利用者判断後
- [A26]：V1追加実測候補（未選定・未承認、decision_owner USER）
- [A28]：V2来歴照合。比較可能PF 0件、現行残差設計をNO_GO候補（Codex技術判定）
- [A29]：V2現行残差fit設計 CLOSED_NO_GO_BY_USER_DECISION（decision_owner USER、利用者原文の引用なし）
- [A33]：H8 Phase B2：登録S4 4軸すべて2%未満でPASS（17:46:48完了）、mandatory STOP。Phase C未実施・未承認
- [A38]：非相殺NONCANCELLATION_SHOTS stageのみ実行。ε=.05で決定論16候補が不適格化、PR6不変。費用比較stage未承認
- [A42]：Track A追加検証設計D0〜D6（DESIGN_DRAFT_NOT_AUTHORIZATION）。Track BのD1〜D3とは別ラベル
- [A44]：H8 Phase C統合・実行準備（記載SOURCE 29ed7654、plan 9e42fc76、契約未承認・unsealed、無効grant、mandatory STOP）
- [A46]：R10後続実行の完了記録（新reviewではない、R11・詳細GPTレビューは作成していない）
- [A47]：H8 Phase C：利用者が通常signed biasに基づく一回実行を承認。SOURCE 931755c4・plan 79d624b9へ再固定。確認時点で科学実行0・有効grant0（公開認証待ち）
- [B01]：BF現仮説を限定negativeとしてclosure（GPT推奨に基づくCodex記録、逐語採択未確認）
- [B02]：BM DF multirate adapterはcompact BCHと三次係数が完全一致（new-method gate不通過）
- [B07]：G4レビュー（利用者採用）。固定toy・同辞書の実用優位主線を区切る
- [B08]：G5：class下界が保存CTS lawを上回る（固定辞書class除外）
- [B09]：G10 v3：closed P5等のnative資源地図（3-qubit synthetic provider）
- [B10]：G10 v3レビューの利用者採用記録（一般full-return性能探索を区切る）
- [B11]：論文化・着地点レビュー（技術ノートv1.0、独立原著投稿は推奨しない、利用者採用）
- [B12]：DF三次補正 GENERIC_WHOLE_SERIES_CLOSED_TO_AUTOMATIC_EXTENSION（利用者採択GPTレビュー）
- [B13]：Gaussian積生成：独立新手法としてCLOSED（manifest上『adopted by user』、逐語未記録）、性能UNDETERMINED
- [B14]：Track B新規探索の一時休止（manifest上『adopted by user』）。明示的解除記録は未確認
- [B15]：ステップ間相関：KNOWN_PROCEDURE_MAPPING、追加method delta NOT_ESTABLISHED
- [B16]：GPT broad review（②既存成果の発展を支持、D1〜D3の起点）
- [B21]：D1〜D3 post-literature GPTレビュー（数値系列の一区切り、一体原稿化を次作業として採択・推奨。利用者指示は研究ノートのみで原稿は未着手［B22］）
- [B28]：S4完成版レビュー（Claude原稿をGPT-6改訂）：既知機構の特定構造への応用、独立新規性未認定、D+E推奨
- [B29]：ユーザーがD+E（Parent allocation限定成果の整理＋同一fixture数値探索の区切り）を正式採択。新科学計算の承認ではない
- [B30]：Parent allocation closure（一般no-goではない、再検討条件§6）
- [B31]：Track B資源構造整理。実PRで最大の費用要因はNOT_ESTABLISHED
- [B32]：次期方向の4候補群I〜IV（未採択・順位なし）
- [B34]：次期方向GPTレビュー：M1をread-only P0の第一候補（正式採択はD+Eのみと明記）
- [B35]：M1 P0（M1_adopted=false）。③M1-Sの直接の前段
- [B36]：M1 P0.5（M1採択保留）。直後の41fa1eb4から③
- [B37]：②/③境界：GPTがM1_NOT_ADOPTED・M1-S再定義を暫定支持（レビュー判断）
- [B38]：H4保存値のbasis費用割合診断（加法比、T比率NOT_MEASURED）
- [B39]：BM現adapterのnew-method路線を閉じる（post-BM05 GPTレビュー受領に基づくCodex記録）
- [B40]：G10 v3論文化・着地点レビューの利用者採用（有限Taylor return-aggregation検証系列を区切り、技術ノート完成）
- [B42]：D2境界相殺one-shot：P5対partialの相対T差の変化は約0.02 pp（P5自身のT切片減少約0.41%とは別）
- [B43]：全体休止レビュー（manifest上利用者採用）。closed P5比率の算術、新しい相関sampler系列は不採択
- [C01]：M1-S統合サマリー（研究到達点の優先参照）
- [C02]：M1-S証拠・結果索引（S/C/A固定順序）
- [C03]：M1-S再開ハンドオフ（R0〜R7、承認境界）
- [C04]：最新GPT総合再評価（強い新primitive不支持、priority未確定、資源路線保留推奨＝NOT_ADOPTED）
- [C05]：Claude原本（Stage S前時点。限定ノート化・PR-2回帰推奨＝NOT_ADOPTED）
- [C06]：M1-S統合公開receipt（research_direction_formally_adopted=false）
- [C07]：M1-S worktree 31件整理完了（完了事項。再開しない）
- [C08]：M1-S G0-T1限定理論契約（正式採択）
- [D01]：④既存研究と探索範囲の対応表（分類①〜⑤）
- [D02]：④残る科学的問題（観点C1〜C5、L1〜L8）
- [D03]：④GPT Level 1判断点（未承認・未開始）
- [D04]：H1〜H4現探索PAUSED（ユーザー承認の追記記録）、④全体は課題選定段階・正式採択なし
- [D05]：user_adopted_H1_H4_pause=true、new_algorithm_formally_adopted=false
- [D06]：SOS-PF A3：現分割生成探索CLOSED、A2 NOT_ADOPTED、K1-F NOT_AUTHORIZED/NOT_EXECUTED
- [D07]：一体再配分PF現固定構成のみCLOSED、H-E1現定義の探索CLOSED、M-a/M-b現定義での追加探索STOP（M K1未認可）、PF小改変PAUSED（この系列）、旧D HOLD（C1_C4はUNCHANGED）
- [D08]：N3現検証系列（追加benchmark・帰属系列）CLOSED・主研究不採択（理論・限定prototypeは条件付き保持）、N1 STOP（現候補scan）、N2 HOLDを列挙（利用者採択GPTレビュー）
- [D09]：N1 STOP・N2 HOLDの初出
- [D10]：THRIFT/mixed現template CLOSED（THRIFT一般は閉じていない）
- [D11]：THRIFT現templateの再開trigger
- [D12]：H1〜H4静的K0：総合NO_GO_NO_INDEPENDENT_GENERATION_RULE、H2 UNDETERMINED_DATA_REQUIRED（技術判定）
- [D14]：GPT再レビュー：C1現系列終了、C2対照保持、C3/C4投資見送り（ユーザー正式採択記録なし）
- [D15]：C2/C3最終技術監査（既知手順へ対応、独立低費用手順未特定）
- [D16]：C4技術監査（既知IS等へ還元、Phase B停止）
- [D17]：C1保存oracle診断（通常16 contextで登録5%優位なし）
- [D18]：SOS-PF K0（UNDETERMINED_DATA_REQUIRED）
- [D19]：SOS-PF K0.5技術報告
- [D20]：一体再配分h境界再監査v2（登録10%改善は4/4未達）
- [D21]：一体再配分の生成法 REDUCES_TO_KNOWN（技術判定）
- [D22]：H-E1 K0（gate不通過、技術判定）
- [D23]：M K0（core REDUCES_TO_KNOWN、UNDETERMINED_NOT_READY）
- [P01]：高次PF第1研究の最終統合（別repository、④範囲表G10が参照）
- [P02]：高次PF第2研究の方針（Direction C、別repository、④範囲表S7が参照）
- [X01]：2026-10-14時点の横断統合資料（G2）。①②は当時snapshotで現在の最終状態ではない

[A03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md
[A04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_h4_b0i_v04_scientific_review_20261014.md
[A09]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md
[A14]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md
[A17]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1ea42fbf1bd1c8341ffd410b8067467b4ba4e170/docs/research/track_a_native_df_theoretical_gate_model_20261016.md
[A18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d2431b8c2a385af20a3d996cf8074b7a826a6e6b/docs/research/track_a_native_df_qiskit_connection_validation_20261016.md
[A19]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e41670565d9284da9aded94063be591fdcaa3dd/docs/research/track_a_native_df_theory_evaluation_20261016.md
[A21]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e48a6edaa51bd474b8185bfd0f012bf83b0b43de/docs/research/provenance/track_a_claude_level1_scientific_review_20261016.md
[A22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd695e6584f15b88341d2446ed010c7bdfef1666/artifacts/resource_applicability/track_a_v2_h6_readiness/2026-10-16/review_provenance_v1.json
[A24]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md
[A25]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/63e5773b6b408506f4c39783b80f3025f14481b9/docs/research/track_a_v1_results_synthesis_20261016.md
[A26]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/63e5773b6b408506f4c39783b80f3025f14481b9/artifacts/resource_applicability/track_a_v1_results_synthesis/2026-10-16/additional_measurement_candidates.json
[A28]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68a010b35d35a1500dcc901c480fcce433fac4a8/docs/research/track_a_v2_residual_analysis_readiness_20261016.md
[A29]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/36756714fee37a7a6e1188bb4d765292d877ea18/artifacts/resource_applicability/track_a_h8_strengthened_readiness/2026-10-16/v2_design_closure_v1.json
[A33]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fef2db7a79886ab682b652da7266b463c98c7d3a/docs/research/track_a_h8_b2_completed_results_20261017.md
[A38]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f902229cef9da26cb6feeff09eb35860b6115a6/docs/research/track_a_h8_noncancellation_shots_result_20261017.md
[A42]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ccc79502f043b4885e2337ffa98ff45f1875ae4/docs/research/track_a_resource_evaluation_expansion_design.md
[A44]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7be02fcf705425a0d98e871b19d0c1aebf60c09/docs/research/track_a_h8_phase_c_integration_readiness_20261017.md
[A46]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_scientific_review_decisions.md
[A47]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b357c77a1a35d70dde011d44516bd8cab3bfcc8a/docs/research/track_a_h8_phase_c_execution_binding_20261017.md
[B01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b2d624adde979f7d8f983bc7fdbbec88f90c500/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md
[B02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md
[B07]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/research/track_b_G4_scientific_review_20261010.md
[B08]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md
[B09]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcd3ea6217bc00b667180cec149a70102d75f07e/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[B10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10/review_intake_identity.json
[B11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/research/track_b_G10_v3_publication_and_endpoint_review_20261011.md
[B12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4192077723770f4846cc391ea0da96c1e115f2fa/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md
[B13]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md
[B14]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c3d59139febfc94749b7c2ffc74b4f0a882f539/artifacts/track_b_research_pause_record/2026-10-13/decision_manifest_v1.json
[B15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/792af43ef237c0c5cd9b42ec102a15dfc3c60be4/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[B16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8c9b5dbdf96dcc7b4af0605fc24921f47e0971bd/docs/research/track_b_existing_results_development_broad_scientific_review_20261013.md
[B21]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/79a1ff0a45e84f3e45c09cc7ede9255310449302/docs/research/track_b_d1_d3_post_literature_scientific_review_20261014.md
[B28]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3086f48c5598c8bb31b0587e807ca6c1b3d2754/docs/research/parent_allocation_s4_scientific_review_revision_20261015.md
[B29]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288819e17821762eeaab62edb590483f05ddb444/docs/tracks/algorithm_codesign/inputs/track_b_de_direction_preparation_user_instruction_20261015.txt
[B30]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288819e17821762eeaab62edb590483f05ddb444/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[B31]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288819e17821762eeaab62edb590483f05ddb444/docs/tracks/algorithm_codesign/track_b_resource_structure_and_bottlenecks_20261015.md
[B32]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288819e17821762eeaab62edb590483f05ddb444/docs/tracks/algorithm_codesign/track_b_next_algorithm_research_decision_packet_20261015.md
[B34]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ef2c62b4b6fea5dcef2674a29db821cf91bab6/docs/research/track_b_next_direction_gpt_scientific_review_20261015.md
[B35]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa91ffb3d4880d767e55f0f661e8ade294b4c9fe/docs/tracks/algorithm_codesign/m1_p0_prior_art_control_and_resource_decision_packet_20261015.md
[B36]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md
[B37]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41fa1eb480728cd0691b00bb41dd4e5dc4ba6b03/docs/research/track_b_m1_claude_review_gpt_independent_assessment_20261015.md
[B38]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/artifacts/rte_basis_cost_attribution/2026-10-13/README.md
[B39]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3861e6b941745e43863f6a62cd25fe36f3b3e108/docs/tracks/algorithm_codesign/synthesis_placement_design_review_20261006.md
[B40]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e5a5b0a5973f1725c7a6872a4b9b39b86d66cef/artifacts/track_b_return_aggregation_note_completion/2026-10-11/review_intake_identity.json
[B42]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1c59b28f4370cc7be65cff60d1ec877fd1491f5/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
[B43]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c3d59139febfc94749b7c2ffc74b4f0a882f539/docs/research/track_b_overall_direction_and_research_pause_review_20261013.md
[C01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md
[C02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/evidence_and_results_index.md
[C03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/codex_restart_handoff.md
[C04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integrated_reassessment_claude_prior_reviews_stage_s_20261017.md
[C05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_prior_art_novelty_and_research_value_claude_review_20261017.md
[C06]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/publication_receipt.json
[C07]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ef39e3af4d479a1d3ee1a2792f734d67f208872/docs/maintenance/m1s_worktree_cleanup_20261017_agent_ignore_followup/README.md
[C08]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md
[D01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md
[D02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/open_problem_landscape.md
[D03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/gpt_decision_handoff.md
[D04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/research_state_transition.md
[D05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/provenance.json
[D06]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/artifacts/qc_sos_pf_a3_state_adoption/2026-10-16/decision.json
[D07]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json
[D08]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/afecaf591e62b8ab7ba5fb5c20f53f17a6d7be37/artifacts/n3_achievements_closure/2026-10-13/research_decision.json
[D09]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c2955ba0ef391eaf9102c711c378ab73b766fc9f/artifacts/n3_occupation_redesign_preparation/2026-10-12/MANDATORY_STOP.json
[D10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e54991ef2c80fe463de9cdb865b2c48d5b981607/artifacts/qc_mixed_access_closure/2026-10-13/research_decision.json
[D11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e54991ef2c80fe463de9cdb865b2c48d5b981607/artifacts/qc_mixed_access_closure/2026-10-13/closure_state.json
[D12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d460c54f3bf54ea46d4d8129f6702703c4bb217/docs/research/qc_new_algorithm_static_screen_k0_20261016/technical_report.md
[D14]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6711a709ad13144972474effe1faf3933cd7f7fb/docs/reviews/quantum_chemistry_pf/qc_exact_ground_state_pf_direction_rereview_2026-10-14.md
[D15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d91991038b1a2b97fac52a00d8ed4ebb413e2214/docs/research/qc_c2_c3_final_technical_audit_v1_20261014.md
[D16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/60da5738d48e0faaedd3de125b0350301935dc3a/docs/research/qc_c4_finite_rte_method_efficiency_technical_audit_v1_20261014.md
[D17]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5ff55e749ceb33e54da0b3c70ac6ac08599ebb82/docs/research/qc_c1_saved_oracle_diagnostic_v1_results_20261014.md
[D18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md
[D19]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/technical_report.md
[D20]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bbb611f95a7be5036ea0291655862bf3bb7d619/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md
[D21]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9e288089ca22e2729976ae75d096c4df00e49a7d/artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/method_status.json
[D22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/artifacts/qc_he1_stage0_prevalidation/2026-10-15/method_status.json
[D23]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/artifacts/qc_m_fixed_circuit_k0/2026-10-15/method_status.json
[P01]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/cd81018932b9557b7a55e2e1cd8c5e3988245409/PF_first_study_final_synthesis_20260925.md
[P02]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/1d01e77247364b49dbce2e9ce2fe7c32af0fb937/docs/second_study_v2/budget_safety_mechanism_20261005/final_direction_and_claim_scope.md
[X01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md
