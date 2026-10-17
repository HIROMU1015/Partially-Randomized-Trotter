# 量子化学ground-energy研究：4研究系列の現在地・最新差分・重複防止（GPT研究戦略判断用ハンドオフ）

2026-10-17 JST。Repository `HIROMU1015/Partially-Randomized-Trotter`。作成：Claude Code（ユーザー指示「4研究系列の最新進捗・既存成果・重複防止を踏まえた研究戦略ハンドオフ」による資料整理）。

**この資料は研究戦略の判断ではない。** 新しい科学レビュー、新規性判定、研究優先順位、新アルゴリズムの考案・採択、新規科学計算・保存値の再集計は行っていない。既存の固定資料を読み、研究状態を照合し、①②の2026-10-14以後の差分を確定し、③④は完成済み資料を再利用した。研究の方向・優先順位・中心課題の判断は、ユーザーの開始承認を受けたGPTとユーザーが別途行う。

付属資料：[重複防止・進捗対応表](overlap_and_progress_map.md)／[固定証拠索引](fixed_evidence_index.json)（本文の[A01]等のID→repository・branch・commit・path・bytes・SHA-256・GitHub取得照合）。

## 0. 確認範囲と読み方

- **確認時点**：remote-tracking refs（334本）を2026-10-17 19:09 JSTに取得した状態。最新commitは①の `b357c77a`（19:03:12、H8 Phase Cの実行承認と再固定、科学実行0）。①は当日も更新が続いており、H8 Phase Cは承認済みで実行待ちである。これより後のcommit（Phase C結果を含む）は本資料に含まないため、GPTは利用時に `track-a-h8-phase-c-integration-20261017` 以降のbranchを確認する必要がある。
- **基点**：2026-10-14の横断統合資料[X01]。①②はその後に多数の結果・レビュー・正式状態が加わっており、[X01]を現在の最終状態として扱わない。③は統合記録[C01]〜[C03]、④は研究範囲資料[D01]〜[D05]を一次入口として再利用した。
- **読み取り方法**：①②③④に関係するremote branchの固定commitを `git show` で読んだ（③④は全remote branchについて、基点以後の状態変更の有無を確認）。主要本文は索引の `key_evidence` に登録し、GitHubから独立取得してbytes・SHA-256の一致を確認した。分岐した履歴が多いため、branchの更新日時ではなく、各結果に対応するSOURCE・結果・承認・STOPのcommitで判断した。
- **状態の書き分け**：「ユーザー正式状態」「レビュー判断・推奨（GPT/Claude）」「技術判定・mandatory STOP」「実行承認」を区別する（定義と全一覧は[対応表§1–2](overlap_and_progress_map.md)）。GPTレビューが「採用」「採択」と書いた方針は、ユーザーの正式採択記録がない限りレビュー判断として扱う。正式記録がないものに正式状態を作っていない。
- **数値の扱い**：すべて固定文書からの転記で、分子・basis・DF・T・精度定義・費用指標・対照を併記した。異なる指標・条件の倍率を並べた総合ランキングは作らない。①のεは**有限時間複素signalの精度でありHaではない**。複素signal誤差、固有位相誤差、絶対基底エネルギー誤差は別量で、①③の数値結果はいずれも第1段階（signal）までである。①の費用は特記がない限り「P=0（状態準備除外）のmeasurement込みfull wrapperのcompiled RZ単価×解析的十分shot数（Hoeffding型の十分設計で、最適・必要shot数ではない）」の点推定で、量子backendで実行したshotではない。

## 1. 全体エグゼクティブサマリー

| 系列 | 到達点 | 最新の科学的・技術的状態 | 進行中の作業 | 未解決の研究判断（GPT/ユーザー） |
| --- | --- | --- | --- | --- |
| ① PRリソース評価（Track A） | H4/H6/H8で、既知PR（DF prefix＋finite RTE）の測定込みcompiled RZ費用を決定論PF対照と比較する限定benchmark。R09/R10（GPTレビューの採択方針。ユーザーの正式採択記録はない）は条件付きbenchmarkとし、一般的PR優位・「高精度ほどPR有利」・最良PF一般への勝利・新アルゴリズムを主張しない [A04]、[A09] | H6では、B0+I全12構成で登録決定論最良比29〜56%減 [A13] だったが、対照をV1（136候補）まで強めるとε=.05で**逆転（−13.29%、実測点）**、他4精度は固定model条件付きで約10〜35% [A25]。H8はB0+I全12構成で登録決定論最良比24〜56%減 [A15]。H8の同型強化はPhase A/B1/B2（S4 model 4軸PASS）まで完了し、**再会計（Phase C）はユーザーが一回実行を承認済み、確認時点で科学実行0** [A33]、[A47] | H8 Phase C（承認済み・実行待ち。公開記録上の実行中jobはない） | H6の主張の精度域と強さ、H8 Phase C結果を主張の前提にするか、bias規約（signed／非相殺）、実施済みH5/H7 pilotの位置付け。R10以後の結果についてGPTの採否レビューは公開されていない（Claude L1 [A21] は一部を含めてレビュー、利用者採択は技術準備のみ [A22]） |
| ② PR内部アルゴリズム改善（Track B、M1-S除く） | BF/BM、SP/BS、R/RA、G1〜G10、DF三次補正、Gaussian積生成、ステップ間相関、D1〜D3、Parent allocationを検討。限定的native利益はG6〜G10（closed P5）、D1〜D3、Parent allocation等のsynthetic fixture内に限られ、BFは限定negative、BM・BS・Gaussian・相関は方法差未確立、RA-D0は技術INCONCLUSIVE | **独立した新アルゴリズムの確立を認定した既存レビューはない**。Parent allocationはユーザーがD+E（限定成果の整理＋同一fixture数値探索の区切り）を正式採択 [B29]。次期方向の4候補群は未採択 [B32]。M1はP0/P0.5で採択されないまま③M1-Sへ移行 [B37] | なし | Track Bを新RQで再開するか・note化で区切るか・休止を維持するか（10-13休止記録の明示的解除は未確認 [B14]） |
| ③ M1-S | ±branch間label共有の限定理論（T1a/T1b、適用域付き受理）と、H4保存oracle上の条件付き数値診断・十分shot整数witness [C01] | 総回路費用改善、多round RPE成功、方法の独立新規性、論文化は**未確立**。研究方向の正式採択なし、統合記録公開後mandatory STOP [C06]。5f079dbe以後の変更はworktree整理のみ [C07] | なし | 終了・ノート化・限定再開（R0〜R7）の選択と、他系列との関係 |
| ④ PRに限定しない新アルゴリズム | N1〜N3、THRIFT、旧C1〜C4、一体再配分PF、H-E1、M、SOS-PF、H1〜H4を検討済み。具体的な独立新回路生成規則は得られていない [D03] | 研究課題選定段階、新アルゴリズム正式採択なし。2026-10-16統合資料以後、**正式状態の変更なし** | なし | 測定・情報取得を研究範囲に含めるか、記録不足の狭い2問の扱い、次の調査の必要条件（GPT Level 1判断、未承認） |

横断的に、**全roundの絶対基底エネルギー推定の総費用（state preparation・overlap・RPE/QPE全round・成功確率を含む）は4系列のいずれでも評価されていない**。これは共通の空白であって否定的結果ではなく、どの系列の局所結果もこの空白を埋めない。

## 2. 研究系列別の最新状態表

「最新固定証拠」はIDで示し、repository・branch・commit・pathは索引にある。正式状態の列で「なし」と書いたものは、ユーザーの正式採択記録が存在しないことを意味する。

### 2.1 ① PRリソース評価（Track A）

共通条件（特記以外）：linear H鎖・隣接1.00 Å・STO-3G、Qiskit 1.3.0（rz/sx/x/cx、optimization 1、seed 17、topologyなし）、native symmetric-directional control、T=0.8、q=1/2/4/8、経験的numerical allowance（certificateではない）、α_axis=.025。PRは事前固定した確認union×独立32 trajectoryの点推定（母平均winnerではない）。DF：H6はtol-only rank19・sector400、H8はtol-only rank28・sector4900。H4 geometryの条件は行内に記す。

| サブテーマ | 研究目的 | 最新固定証拠 | 実施済み作業 | 確立した成果（適用範囲） | 否定的・未解像結果 | 既知法との関係（既存レビュー） | 未解決事項 | 正式状態 | 担当・承認 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H4 B0+I強化対照（H4 geometry） | DF tailの既知identity scalarを戻すB0+Iを加え、PR利益の帰属を公平にする | [A03]、[A04] | pilot6・120構成240軸の実compile。条件：H4 0.70〜1.60 Å（1.00 Åを含まない）、legacy DF12、ordinary control・H4 geometry用lowering、PR（B2）は同一32 paired trajectoryで選抜と推定を兼ねる | 240軸でB0+IとB0の回路6指標差0。ε=.005では6距離すべてでPRがmin(B0,B1,B0+I)より約50〜60%低い（点推定） | 拡張集合の点最小はB0+I 397／PR 1,415（旧「PRが全点最小」は不採用）。ε=.05では距離により優劣が入れ替わり、.80/.90 Åでは中間精度でB2→B0+I→B2の再入 | R09：PR初提案・scalar保持初提案・最良PF一般への勝利を主張しない | 原稿v0.4編集は保留（R10に「原稿化後回し」の利用者方針が二次記録） | なし（R06/R09はGPT判断。実行承認のみ） | 完了・STOP |
| D0保存値再解析・H8 D0・R10 | 保存値からε=.0005・6指標・仮想P感度を再会計し、次段階方針をGPTが判断 | [A07]、[A08]、[A09] | H4 native fixed16（legacy DF12、B0/B0+Iなし）/H6、H8の保存値one-shot（旧行の完全一致後） | ε=.0005のPR/full PF比：H4 0.430、H6確認0.496（ε=.001の0.387より悪化）。H8確認でPR/取得済B0-B1最良0.439〜0.762（ε=.05で0.605、ε=.0005で0.762） | 「高精度ほどPR有利」は不支持。確認標本で4次Taylor event未観測（費用0ではない） | R10（GPT）：既存結果を条件付きbenchmarkとして採用、B0+Iを強化対照に、H5→H7・H6固定δのT比較を採択 | 希少次数の厳密compiled上界はBLOCKED_BY_MISSING_EVIDENCE（R10以前の技術状態 [A43]）。R10は全support証明を現時点で必須にしない | なし（R10はGPT判断、実行認可なし） | 完了・STOP |
| H8 resource map core（10-14完了、[X01]未収録） | H8でPRの測定込み費用地図 | [A11] | 探索592・確認192、full wrapper 1,572件 | 確認点最良PRがfull PF比54.29–65.76%、B0比39.48–56.12%低い（ε=.05〜.001） | 母平均・希少次数上界・状態準備込みは未保証。ε=.0005とB0+Iは含まない | — | — | なし | 完了・STOP |
| H6/H8 native B0+I全12構成 | R10採択範囲の強化対照で比較分母を強化 | [A12]、[A13]、[A15]、[A46] | H6：pilot6→全12構成。H8：全12構成（適格8構成16軸）。全条件で不適格な構成の軸は未測定 | 登録決定論最良（B0/B0+I/full S2/S4のうち適格最良）に対するPR点費用低減（ε=.05/.01/.005/.001/.0005）：H6 29.35/49.69/53.27/56.14/50.37%、H8 28.77/52.72/52.73/56.12/23.83%。両系60条件すべてで比<1。分母はH6ではε=.05〜.001がB0+I、ε=.0005がfull S4。H8ではε=.05がB0+I、ε=.01〜.001が旧B0 p21、ε=.0005がfull S4 | H8のε=.05ではB0+I追加で低減率が約39.48%→28.77%に縮小。matched B0は未取得（delta null） | Claude L1 [A21] がこれらの結果を含めてレビュー（新規性なし、評価研究として帰属と範囲）。利用者採択はV1/V2技術準備のみ [A22] | [A13]は採否を「次のGPT科学レビューへ戻す」と記載したが、公開されたGPTの採否レビューはない（[A46]：新しいR11は作成していない） | なし（実行承認のみ） | 完了・STOP |
| H6固定δ=.2のT依存性 | 固定δでbias・normalizationの蓄積とPR優位のT依存を見る | [A14] | T=.4/.8/1.6、5構成、144 compile | 登録決定論最良に対し、ε=.05〜.005ではPRが約50〜55%低い。ε=.001・T=.4ではPR/B0（登録決定論最良）=0.453 | ε=.001：T=.8でPR/full S4=1.757、T=1.6でPR不適格。ε=.0005：T=.4でPR/full S4=6.869、T=.8/1.6でPR不適格。厳しい精度での逆転・不適格化はouter-PF biasによる統計予算縮小が主（Bの増加より強い） | Claude L1は費用比をB²×回路費用比で説明（紙上算術・推定値は未採択） | 各Tで候補を再最適化していない | なし（R10の採択範囲内、実行承認のみ） | 完了・STOP |
| native DF理論gate model・理論評価・V2 | compilerに依存しない固定分解の命令数で理論費用を数え、compile値と接続 | [A17]〜[A20]、[A28]、[A29] | model A/B、Qiskit接続の人工検証、H6/H8 208候補で理論832行、V2来歴照合 | 人工入力で命令列・unitaryが一致。理論832行を保存費用へ接続（ranking・残差なし） | V2：来歴照合で比較可能PF 0件、残差fitに入らず終了。T/Toffoli・depth・routing・state prepは対象外 | Claude L1：固定分解の命令数は標準作業で単独の価値は低い | model A/Bを比較契約に採るかはGPT判断待ちと明記 [A18] | V2現行残差fit設計 **CLOSED_NO_GO_BY_USER_DECISION** [A29]。理論評価はscientific_adoption=false | 代替解析3案は未採択 |
| V1 H6強化決定論baseline | Claude L1が必須とした、prefix細格子と打切りS4を含む強い対照での再比較 | [A23]〜[A27] | 136候補（B0/B0+I S2 p6–14、打切りS4 p11–15、full S2/S4）、Phase A/B、5精度再会計、p7両軸実compile | ε=.05：B0I p7/q1 S2の実測単価×保存shotで決定論356,097,424 RZ対PR 403,423,200 RZ、**PR削減率−13.29%（逆転）** [A24] | 他4精度（.01/.005/.001/.0005）は**固定model条件付き**で30.89/34.97/31.17/10.03%。全domainの非相殺最良は欠測によりnull | Codex成果整理 [A25]：新アルゴリズムや一般的PR優位を主張する結果ではない。レビュー：Claude L1 [A21]（V1必須）、GPT V1強化baselineレビュー（原本未公開、receiptのみ [A27]、確認できる判断はp7両軸実測の優先） | 追加実測（第一候補 B0I_p11_q4_4th @ε=.0005）は未選定 [A26] | なし（実行承認のみ。追加実測はdecision_owner USER） | ユーザー判断待ち |
| H8強化決定論格子（B0 anchor、Phase A/B1/B2/C） | H8でも同型の強い対照へ再比較する | [A30]〜[A34]、[A44]、[A45]、[A47] | Phase A（追加152候補のsignal・bias・適格性）、B0新anchor 8軸、B1 S2費用model固定、B2 S4 model 4軸実compile、Phase C統合・実行準備・実行承認の結合 | 152候補中132が少なくとも1条件で適格。B1でB0独立holdout誤差0%。**B2：登録S4 4軸すべて2%未満でPASS**（B0 p15/q1 0.215%、B0+I p21/q8 0%）[A33] | Phase C（5精度再会計）未実施のため、**強化後のH8でPR優位がどう変わるかは未判定** | — | 全格子への2%保証・厳密bias・PR母平均は未認定（Phase C後も同じ）。Phase Cは通常signed bias・P=0の保存値再会計で、非相殺行・探索n8を混ぜない | なし（各runは実行承認。B0 anchor run02・B2 run01等の技術・契約STOPは保持） | Phase C：ユーザーが一回実行を承認（SOURCE 931755c4・plan 79d624b9に再固定）。確認時点で科学実行0・有効grant0（公開認証の問題の解消待ち）[A47] |
| H8 bias・非相殺診断 | 通常signed biasと非相殺envelopeで適格性がどう変わるかの感度 | [A35]〜[A38] | bias証拠監査、signed成分復元（184候補368軸）、NONCANCELLATION_SHOTS stageのみ実行 | 通常会計1,900行を完全再現。ε=.05で決定論16候補が非相殺条件で不適格化、ε≤.01とPR6の適格性は不変。総shot比 最大46.2・中央値1 | 経験的感度診断であり厳密上界ではない。費用比較・ranking・PR削減率は未算出 | — | 非相殺費用比較はB2結果との結合と別承認が前提。Phase C（[A44]）は非相殺行を入力から除外する設計 | なし | COST_COMPARISON stage未承認 |
| H5/H7中性開殻pilot | 奇数鎖・開殻でのcorrectness/feasibility確認 | [A39]〜[A41] | H5（ROHF、α3β2、sector100、DF14、p4）・H7（ROHF、α4β3、sector1225、DF23、p6）、各q1・6構成・PR各8標本 | ε=.05でPR/full S2：H5 K4 0.553669、H7 K2 0.684・K4 0.706 | 両系ともPRが適格なのはε=.05のみ。H5はε≤.01でPR不適格。H7はPRがε≤.01で不適格（ε=.01で適格なのはfull S4のみ）、ε≤.005は全構成不適格、B0/B0+Iはε=.05でも不適格 | R10（GPT）がH5→H7を採択し、正式共通ノートではSCIENTIFICALLY_ADOPTED / EXECUTION_SEPARATE [A05]。Claude L1は非推奨だが、利用者採択範囲は「V1/V2技術準備のみ、H5 unchanged」[A22] | pilotは実行済み（H5は利用者指示、H7は承認文言未公開・digestのみ [A41]）。結果の科学レビューと主張範囲への位置付けは未決 | なし | 次段階未承認 |
| PR-2・並列化・測定込み会計 | 既存PR-2基盤 | branch `pr2-v4-s2-parallelization-20260928` tip `b2e1bf65`（2026-10-09、AX-1b H4モデル比較の保存） | 2026-10-13以後、PR-2系branchの更新なし | 測定込み会計は上記各結果のfull measured wrapperとして使用 | — | — | 今回は再監査していない | 記録未確認 | 変化なし |

### 2.2 ② PR内部アルゴリズム改善（Track B、M1-Sを除く）

数値条件は系列ごとに異なる。G6〜G10・D1〜D3・Parent allocationは既知development用の3-system-qubit synthetic provider（p=(1/5,3/10,1/2)、τ=5/7、strict-phase合成1e-6、h=0の加法native T、十分予算N。Nは実shot数ではない）。R/RA・G1〜G5は2-qubit distinct-basis finite P3（R=3/4·Q0+1/4·Q1等、x=1/8・1/4）、SPは2-qubit wrapper、DF三次補正はsynthetic rank2×3（Qiskit opt0の局所native）、BFはH4 1.00 Å/DF12/L_D=3のaction proxy。いずれも分子DF規模の総費用ではない。

| サブテーマ | 研究目的（PR内部の改善部位） | 最新固定証拠 | 実施済み作業 | 確立した成果（適用範囲） | 否定的・未解像結果 | 既知法との関係（既存レビュー） | 未解決事項 | 正式状態 | 担当・承認 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF/BM（決定論部のPF schedule・multirate） | 有限task目的で高次PF係数・DF multirateを設計 | [B01]、[B02]、[B39] | BF one-shot・復元（H4 1.00 Å/DF12/L_D=3）、BM同値性の記号監査（抽象word、DF入力なし） | F/Lはいずれも既知Suzuki5を選択。BMの三次係数はcompact BCHと完全一致 | BFは限定negative、BMはnew-method gate不通過 | multirate・hybrid構成・cost-aware設計に先行研究あり（post-BFA GPTレビュー） | — | closure記録（GPT推奨に基づくCodex記録、逐語採択は未確認）：BF [B01]、BM [B39] | STOP |
| SP/BS（合成配置・PAI/LCU） | 回転合成とsampling配置の交換 | [B03] | SP-0.5/SP-1 one-shot（2-qubit wrapper）、BS-0.5設計監査 | DRが短いnで有利（n=16でNONE比約0.103）、n=32で反転 | BS-0.5はgeneric sparse LCUに対する独立method delta未定義 | 既知PAIの実装検証 | — | なし | STOP |
| R0〜R1.5、RA-D0（Taylor次数間の係数再配分・LP） | 有限Taylor meanを保存する再配分 | [B04]、[B05] | R0理論、R1 one-shot、RA-D0 v3/v4 pilot（2-qubit finite P3） | 限定classでのnormalization最適性。x=1/4・精度1e-3でG_T比 約.796（ordinary比） | 精度選択を許すと全座標では優越しない。RA-D0はTECHNICAL_INCONCLUSIVE | Euler・identity再配分・common angleは既知 | — | なし（GPTはv4 production自動進行なしと推奨） | STOP |
| G1〜G5（degree-local class、有限law、matched CTS） | 有限lawでのT・CX・1Q利益 | [B06]〜[B08] | 6頂点class導出、G3有限law、G4条件付き分離、G5 class除外（2-qubit P3、x=1/4中心） | G4-Aで限定B2 classに対する条件付き分離を認証。R0・G1の限定理論はG4レビューで保持 | matched CTSのlawがJ1をT/CX/1Qすべてで下回る。G5でclass下界が保存CTS lawを上回る | CTSが強い対照、ISは既知 | — | **区切り**（利用者採用のG4レビュー） | STOP |
| G6〜G10（return集約・非列挙生成・closed P5） | normalizer未知のreturn集約と局所生成 | [B09]〜[B11]、[B40]、[B43] | 係数恒等式、局所query、native資源地図、技術ノートv1.0 | m=5でclosed P5のT切片がordinary比約26.6%、CTS比約29.8%、partial比約1.16%低い（CXはCTSが小。比率は全体休止レビュー [B43] の表示値算術） | m=7で一般fullが約2.9%不利 | 部品（Euler pairing、Green再帰、zero-fill、IS、CTS）は既知。結合全体のpriority・非自明性は未確立とされ、既知とも新規とも判定されていない | — | **有限Taylor return-aggregation検証系列の区切り、技術ノート完成、独立原著投稿は推奨しない**（利用者採用 [B10]、[B40]） | STOP |
| DF三次補正 | 三重交換子補正の低確率挿入 | [B12] | 独立監査、whole/Pauli比較、E/T v4〜v7、保存値復元 | 固定加法計画（取得済みsequenceを保持、gate間cancellationなし）で、synthetic rank2×3の3条件でwhole A completion下界がPauli B worst-event上界を上回る（例A≥576対B 112 T）。最適compiler・whole一般の下界ではない | 6 mode controlledではA下界も欠測。Aの全T、同精度でのPR総費用は未確定 | sparse correction・first momentは既知、NCC/HNCC・CPFが近接 | — | **GENERIC_WHOLE_SERIES_CLOSED_TO_AUTOMATIC_EXTENSION**（利用者採択、新アルゴリズム不採択、Track B全体は終了しない） | STOP |
| Gaussian積生成 | DF-RTE event積の反射積・active span圧縮 | [B13] | 生成監査、scope review | 理想構成は前提下で成立 | 固有の手順未定義（KNOWN_COMPONENT_PIPELINE）、性能はUNDETERMINED（negativeではない） | 既知の反射積・partial Gaussian合成・controlled Gaussian compressionへ還元 | 応用研究は未採択 | **独立新手法としてCLOSED**（manifest上利用者採用、逐語未記録） | STOP |
| ステップ間相関乱択 | step間label相関によるbasis切替削減 | [B15]、[B43] | 先行研究・式の照合 | — | 追加method delta NOT_ESTABLISHED | PRHS、qSHIFT、Resource-Optimal IS等へKNOWN_PROCEDURE_MAPPING | — | 正式記録なし。休止レビュー [B43]（manifest上利用者採用）は新しい相関sampler系列を不採択。その後の監査の扱いはGPT推奨（主線にしない）のみ | STOP |
| FR-R1b（位相・半径保証） | finite RTE信号の位相・半径の保証 | [B41]、[B16] | 理論・数値（非一様） | — | MECHANISM_ONLY_NO_PRACTICAL_GO、COMPLETE_MANDATORY_STOP | — | GPT broad review [B16] は独立理論候補として保持（数値系列は再開しない） | なし | STOP |
| Track B新規探索の一時休止 | 10-13時点の全closure維持と成果保持 | [B14]、[B43]、[B16] | 休止記録、同日GPT broad review | closed P5等の成果を保持 | — | — | [B16]（GPT、ユーザー開始承認）は②既存成果の発展を休止の対象外とし研究設計の再開を支持。**ユーザーによる明示的な休止解除記録は未確認**（以後D1〜D3等は実施） | 一時休止（manifest上利用者採用、逐語未記録） | — |
| D1〜D3（同じfinite meanの表現×十分予算×native価格） | 同精度・同confidenceでnative資源順位がどう変わるか | [B17]〜[B22]、[B42] | D1 one-shot・解析、D2境界相殺、D3 v2保証×価格、先行13 claim照合、post-literature GPTレビュー、研究ノート | D1：closed P5のTがpartialより低い（r1約1.16%、r2約5.84%）。D2：literal境界相殺によるP5対partialの相対T差の変化は約0.02 pp（P5自身のT切片減少約0.41%とは別）[B42]。D3：保証変更でr1の十分Nが約29.5%減、G1_P1/r1でP3+tailがP5より0.170% T低い（固定rational会計内） | normalization・momentが小さくても総Tは必ずしも小さくない反例（固定モデル内） | Euler pairing、Green再帰、zero-fill、IS、CTS、SPS、Clifford+T合成は既知。局所生成とP5少数群構成（高々L²+1群）は方法としての価値を支持、結合生成法の独立priorityは未確立 [B21] | GPTレビュー [B21] は一体原稿の作成を次作業として推奨したが、利用者指示は研究ノート作成のみで原稿は未着手 [B22]。他provider・classical online・分子PR/QPE総費用は未取得 | なし（数値系列の一区切りはGPT推奨） | STOP |
| Parent allocation | parent係数をchild別に配分し、L1増と引換えに回転合成単価を下げる | [B23]〜[B30] | 方針レビュー、技術監査、v1/v2（技術STOP）、v3、S4（KKT訂正）、70 lawの比較、S4完成版レビュー | r=1・finite P5・h=0・十分N・加法native Tで、頂点＋ISがcommon＋同proposal機会よりT約14%（ordinary）／19%（P5）低い（N・CX増） | best-common対best-noncommonの2 law比較では、h_Tが交点（ordinary約27.313、P5約11.322 T/trial）を超えると逆転。全70 lawの下側包絡線は未取得で、他の頂点lawにはh≈33.5〜38まで低Tの例がある。内部KKT差は角度依存価格変動が支配的 | 既知機構（Euler＋線形分割、Resource-Optimal IS、SPS/PAI置換）の特定構造への応用。family内L1最小性は三角不等式・既知理論の帰結。独立新規性は未認定 [B28] | C2〜C5（NC exact再集計、価格感度、全h包絡線、多step境界）は未承認 | **D+E正式採択**（同一fixture数値探索の区切り。新計算承認・一般no-goではない）[B29] | STOP |
| 次期方向・M1 P0/P0.5（③への境界） | D+E後の次の改善方向 | [B31]、[B32]、[B34]〜[B36] | 資源構造整理、Decision Packet、次期方向GPTレビュー、M1 P0/P0.5文献照合 | 実PRで最大の費用要因はNOT_ESTABLISHED [B31]。完全qDRIFTのRPE解析・Composite構成・一次位相安定性（単純固有値）は既知と確認 | M1の部分乱択・±制御・multi-round RPEへの既知定理の適用は未証明。M1と同等の部分乱択RPE保証は確認箇所で未特定（不存在主張ではない） | 4候補群・M1/M2を凌ぐ検証済み新機構は認定できない（次期方向レビュー [B34]） | 4候補群I〜IV、M2、III/Track A連携、決定論剰余補正、休止の選択 | 次期方向について正式採択はD+Eのみ（[B34]明記）。M1_adopted=false | ユーザー判断待ち |

[B37]（GPTがM1_NOT_ADOPTED・M1-S再定義を暫定支持）は②/③の境界文書で、②ではM1不採択の根拠としてだけ引用し、M1-Sの内容は③で扱う。②の行にM1-Sの成果は計上していない。

### 2.3 ③ M1-S（完成済み統合資料の取り込み）

| 項目 | 内容 | 根拠 |
| --- | --- | --- |
| 研究目的 | Hadamard干渉の正負branchで同位置のPauli labelを共有するjoint lawが、減衰と位相をどう変え、同じ推定taskで資源上の利点を持ち得るか | [C01] §1 |
| 最新固定証拠 | 統合記録 `5f079dbe`（branch `track-b-m1s-integrated-handoff-20261017`）[C01]〜[C06]、worktree整理 `7ef39e3a`（branch `track-b-m1s-worktree-audit-20261017`）[C07]、Stage S `f2bfb877` [C09]、先行研究調査 `4301c0f8` [C10] | 索引 |
| 実施済み作業 | S0理論契約、T1a/T1b証明とGPT Checkpoint 1/2、Gate 0 v4（精度事前gateで技術停止）→v5、独立計算経路照合、有限step pilot、γ V1、π/3 gate、Stage S、先行研究調査、Claude・GPTの総合レビュー | [C01]、[C02] |
| 確立した理論 | T1a（完全乱択）・T1b（部分乱択、exact HD half、固定r、δ→0、u=sδ²有界）で有限減衰C-sec参照への一様O(\|θ\|)/O(\|δ\|)近似（√n・逆pair-sum gap依存は残る）。W>0かつ単一pair-sum条件(SP)の下で等総label数の先導rate比 ρ=(W+v)/(2W)∈[1/2,1]（ρはshotsやnative総費用の比ではない）。非縮退対象でのJensen floor（Checkpoint 2でGPT受理、記述は未採用addendum草稿） | [C01] §3–4、[C08] |
| 確立した数値（条件付き） | linear H4 1.00 Å/STO-3G/保存DF rank12/G3・G6・G9/保存数値oracle限定。Gate 0 v5でW/v/ρ、有限step（δ=1/8,1/16 Ha⁻¹、R=2,4、T=1,4 Ha⁻¹）でG3/G6各16対照のvisibility higher（phase biasの代償あり）。γ V1・π/3・Stage Sは同じ保存信号の事後解析で、G3 T=4の8対照にcap直下の整数十分shot削減witness | [C01] §5–6、[C09] |
| 否定的・未解像 | Gate 0 v4はTECHNICAL_STOP（科学的negativeではない）。独立照合はfull-H/HD未保存のため全scopeUNDETERMINED。G9は絶対差未解像。π/3はG6 T=1で不成立。Stage SはG3 T=1・G6 T=4がcap直下同数（GLOBAL_UNDETERMINED）、G6 T=1・G9は旧判定carry。いずれもM1-S一般のNO-GOではない | [C01] §5.1、§6.3–6.4 |
| 未確立 | 総回路費用改善、多round RPE成功（Task B）、方法の独立新規性、論文化。native・whole-wrapper単価、HD実装、状態準備、強い対照は未評価（否定的結果ではない） | [C01] §1、§7 |
| 新規性の既存判断 | Claude原本（Stage S前の時点）：Z_c⊗HのqDRIFTの自然な特殊形で新規性は弱い。最新GPT：強い新primitiveを支持しない点で一致、ただしpartial全体はHD sandwichを残す混成対称積で、定理statement全体のpriorityは未確定。分散の定義・G6・状態準備費・整数gateの位置付けで相違 | [C04]、[C05]、[C01] §8–9 |
| 正式状態 | G0-T1限定理論契約と、Gate 0 v4/v5・独立照合・有限step・γ V1・π/3・Stage Sの各個別契約はユーザー正式採択（過去one-shot。現在の実行許可ではない）。研究方向・Stage C・Task Bは未採択（research_direction_formally_adopted=false） | [C08]、[C03] §6、[C06] |
| レビュー推奨（未採択） | Claude：限定ノートで区切りPR-2等へ資源を戻す。最新GPT：修正technical note／研究記録として確保し、資源改善路線の新規実行は保留（CORRECTED_TECHNICAL_NOTE_RECOMMENDED）。いずれもNOT_ADOPTED | [C04]、[C05] |
| 現在のSTOP・担当 | 統合記録公開後のmandatory STOP（Codex手続上）。次の判断はユーザー。新しい作業の承認なし | [C06]、[C03] |
| 他系列へ再利用できる知見 | 費用式 G=F+N(C+P)：整数shot削減は総費用改善の必要条件でも十分条件でもない。共通per-shot状態準備Pはshot差があれば相殺されない。Task A（単一時刻位相）とTask B（多round energy）の区別。モデル差会計η=\|T\|B（raw-H biasへ再加算しない、radではない）。十分shot設計と必要shotの区別、PER_METHOD_ROW 95%は同時coverageではない。同一保存信号の事後解析を独立再現と数えない。label数・gate数・native・shots・総資源の尺度区別 | [C01] §2.2、§5.3、§7 |
| 未解決事項と再開に必要な根拠 | 理論：固定δ・r変更の運用極限、一般DF event、HD内部PF、一般guiding state・縮退、先行定理への完全な対応。資源：whole-wrapper、HD、状態準備、同精度・同情報の強いbaseline、Task B全round。再開はR0（既存記録の利用・追記）、R1（prior art対応）、R2（有限δ・r変更のT1c）、R3（一般DF/HD内部PF）、R4（共鳴・state拡張）、R5（Task A論理総費用の限定比較）、R6（Task B多round RPE）、R7（別条件一般化）からユーザーがscopeを選び、契約採択と別途の実施承認を経る。最新GPTは、資源研究を続ける場合は同一task・利用情報・whole-wrapper費用・比較対象を一つの判断単位にまとめるよう推奨（実用最良法へのclaimでは決定論PF・PR型RTEも対照に含める） | [C01] §10、[C03]、[C04] |

②との境界：M1 P0/P0.5 [B35]、[B36] はM1-Sの直接の前段（git祖先関係、G0-pre packet [C11] が資料再利用を明記）だが、統合資料には記載がない。本資料はM1/M1-Aを②、M1-S（[B37]以降）を③として数える。

### 2.4 ④ PRに限定しない新アルゴリズム（既存資料の再利用）

| 項目 | 内容 | 根拠 |
| --- | --- | --- |
| 研究目的 | D1（block-encoding・qubitization・QSVT・量子walkを中核とする構成を新規対象から除外）とD2（元電子Hの絶対基底エネルギー推定。定数項・sectorを失った別taskを達成と扱わない）の範囲で、独立した新アルゴリズムを探す | [D04] |
| 最新固定証拠 | 研究範囲資料 `56f1cc55`（branch `qc-new-algorithm-research-scope-20261016`）[D01]〜[D05]、SOS-PF A3 `dbe51f18` [D06]、PF探索状態 `78b50c51` [D07]、N1〜N3 `afecaf59` [D08] | 索引 |
| 実施済み | N1〜N3、THRIFT/mixed、旧C1〜C4、一体再配分PF、H-E1、M-a/M-b、SOS-PF K0/K0.5、H1〜H4静的K0、既存技術基盤表、8一次文献との限定照合、検討範囲表（分類①〜⑤） | [D01]、[D02]、[D12]、[D13] |
| 確立した成果 | 既知性と検討履歴・入力条件・検討記録の不足を区別した範囲表。具体的な独立新回路生成規則は得られていない | [D03] |
| 否定的・未解像 | 有限条件の否定的所見：N1（synthetic JW4・signed DF rank2、固定3 context）でcertified objective改善0%、THRIFT現template（H4 1 Å/STO-3G/DF16全保持、登録価格model）で予算不足、C1（非分子2-qubit Pauli H、通常16 context）でoracleにも登録5%優位なし、一体再配分（H4 1.00 Å/STO-3G/DF12、1e-4・1e-5 Ha）で登録10%改善が4/4条件未達。いずれも一般の不可能性ではない | [X01]、[D10]、[D17]、[D20] |
| 既知法との関係（技術判定） | 旧C2/C3は既知手順へ対応（低費用の独立手順未特定）、C4は現構成が既知IS等へ還元（独立圧縮法・資源利益はUNDETERMINED）、H1/H3/H4は独立生成規則未特定（NO_GO_NO_INDEPENDENT_GENERATION_RULE）で既知恒等式・既知access条件に帰着、H2は個別UNDETERMINED_DATA_REQUIRED、SOS-PF K0はUNDETERMINED_DATA_REQUIRED | [D12]、[D15]、[D16]、[D18] |
| 未解決（記録上） | 観点C1〜C5（代数構造、PF以外の時間発展、回路実装構造、絶対energyの情報、理論の未閉鎖条件）。検討記録不足の可能性がある狭い問いは2件のみ（encoding/sectorの全費用比較、copies/coherent preparation/複数copyを分けたenergy情報取得）で、既存研究の別名や実装比較に留まる可能性も明記 | [D02]、[D01]、[D03] |
| 正式状態 | 研究課題選定段階、新アルゴリズム正式採択なし。個別状態（[対応表§2](overlap_and_progress_map.md)）：N1 STOP（現候補scan）、N2 HOLD、N3 現検証系列CLOSED、THRIFT現template CLOSED、一体再配分 現固定構成CLOSED、H-E1 現定義CLOSED、M-a/M-b 現定義での追加探索STOP（K1未認可）、PF小改変 PAUSED、旧D HOLD、SOS-PF 現分割生成探索CLOSED（一般no-goではない）、A2 NOT_ADOPTED、K1-F未認可・未実施、H1〜H4 現探索PAUSED（当該K0.5/K1未認可・未実施） | [D04]〜[D08]、[D10] |
| 担当・承認 | GPT Level 1判断は未承認・未開始 | [D03] |

## 3. ①の最新差分（2026-10-14以後）

[X01]が参照したTrack Aの最新はH4 geometry [A01] とH6独立確認 [A02] までで、B0+I、H8、D0、V1/V2、Phase A/B/C、bias診断、H5/H7はすべてその後の差分である。

**3.1 最新に完了した検証。** H8 Phase B2（登録S4 4軸、計算完了17:46:48、公開commit `fef2db7a` 18:00、mandatory STOP）[A33] と、H8非相殺 `NONCANCELLATION_SHOTS` stage（公開commit `3f902229` 17:45）[A38]。その後、H8 Phase C統合・実行準備（人工37件PASS、実H8保存値による再会計は0）が公開され [A44]、続いてユーザーがPhase Cの一回実行（通常signed bias、P=0）を承認し、SOURCE 931755c4・plan 79d624b9へ再固定された（人工49件PASS、確認時点で科学実行0・有効grant0、公開認証の問題の解消待ち）[A47]。その前にH8 B1 [A32]、signed-bias復元 [A36]、V1成果整理 [A25]、H7 pilot [A40]、H6/H8 B0+I全12構成 [A13]、[A15]、T依存性 [A14]、理論評価 [A19]、H8 D0 [A08] が完了した。

**3.2 進行中・準備中・停止中・失敗後保留。** 公開記録上の実行中jobはない（B2は17:22の「RUNNING」記録 [A34] の後に完了 [A33]）。承認済みで実行待ち：H8 Phase C（確認時点で科学実行0）[A47]。未承認のまま止まっているもの：非相殺COST_COMPARISON stage、V1追加実測 [A26]、V2代替解析3案、V2 H6 cost binding草稿 [A20]、H5/H7次段階、matched B0（最大6軸、別scope）。技術・契約STOPの履歴（B0 anchor run01/run02、B2 run01のMEMORY_PRESSURE、CPU gate等）はそのまま保持されており、複数attemptで完了した系列（B0 anchor、B2）を「元の一回で成功」とは扱っていない。共通ノートの最終版 [A16] は「H5/H7・H6 T比較の実行認可未取得」と古い記載を残す（実際は完了済み）。

**3.3 確認できている改善と未確立の主張。** 確認できているのは、登録した決定論対照集合内での**条件付き点費用優位とその縮小の仕方**である。H6：旧32候補比29〜56%減 → 強化136候補でε=.05逆転（実測点）、中間精度約31〜35%・ε=.0005約10%（固定model条件付き）。H8：登録決定論最良比24〜56%減、強化格子は未再会計。未確立：母平均での優劣、全候補の実測最良、厳密bias保証、truth-free候補選択（保存full-H referenceによる事後選択を使用）、P>0、他geometry・時間・系サイズへの一般化、energy/QPE/RPE/FT費用。H4〜H8はDF政策・prefix・候補集合が異なり、統制されたsize scalingとして並べない（R10）。

**3.4 強い決定論baselineとの比較の到達点。** B0+I（R06/R09/R10で採用）を、H4 geometry（6距離・120構成240軸、legacy DF12）と、H6/H8 nativeの全12構成（全条件で不適格な構成の軸は未測定）で取得した。H4 native 1.00 Å（fixed16）にはB0/B0+Iは登録されていない。Claude L1が必須とした強化（prefix細格子＋打切りS4）はH6（V1）で完了し、ε=.05の逆転を実測単価で確認した [A24]。H8の同型強化はB2まで完了し、比較結果（Phase C）は確認時点で出ていない（ユーザーが一回実行を承認済み [A47]）。最良PF一般（SPRINT/GRADE、controlled symmetric Trotter等）との比較は対象外で、主張もしない（R09/R10）。

**3.5 評価済みの費用範囲。** 評価済み：Qiskit opt1でのcompiled RZ（主）とCX・depth・size、解析的十分shot（Hoeffding型、軸ごと）、仮想状態準備Pの感度境界（H6 ε=.05で約8.97e5 RZ/shot等）、経験的bias（signed・非相殺感度）、理論固定分解の命令数。未評価：実state preparation、T/Toffoli・FT合成、noise・routing、希少Taylor層の母平均、online古典費用、RPE/QPE全round。

**3.6 局所比較から全roundの絶対基底エネルギー推定へ進む際の未解決要素。** 単一時刻signal精度εから固有位相誤差、さらに絶対基底エネルギー誤差への変換（Claude L1のV4型読み直しは未承認）、長時間・多round域でのPR利益（T panelはT≤1.6で、ε≤.001の一部セルでS4優位またはPR不適格）、overlap・energy prior・alias、状態準備費用の実値、truth-free選択、希少次数の費用保証。R10はenergy/QPE/RPE・実状態準備・truth-free選抜をTrack A benchmark完成の必須条件にしていない。

**3.7 予定されている次の技術作業と承認状態。** H8 Phase C：ユーザーが一回実行を承認済み、確認時点で未実行（公開認証の問題の解消後に一回実行し、mandatory STOPする設計）[A47]。非相殺費用比較、V1追加実測（論文の主張範囲をユーザーが決めた後）、V2代替解析、H5/H7全面map、Claude L1のV3/V4、matched B0：いずれも未承認。R10（GPT）が採択した順序（H8 D0→H6 B0+I→各系12構成24軸→H5→T比較→H7）はすべて実行済み（各系12構成は、全条件で不適格な構成の軸を除いて取得）。

## 4. ②の最新差分（2026-10-14以後、M1-S除く）

[X01]はTrack BをD1〜D3（[B20]時点）までしか反映していない。追加差分は、post-literature GPTレビュー [B21]、return aggregation研究ノート [B22]、方向レビュー [B23]、Parent allocation全系列とS4完成版レビュー [B24]〜[B28]、D+E正式採択とclosure・資源構造整理・Decision Packet [B29]〜[B33]、次期方向GPTレビュー [B34]、M1 P0/P0.5 [B35]、[B36]。旧系列（BF〜休止記録）の最終状態は[X01]に系統的には整理されていないため、§2.2に併記した。

1. **数学的新規性の証拠と既存レビューの判定**：独立した新アルゴリズムを認定したレビューはない [B21]、[B28]、[B34]。B21は局所生成とP5少数群構成（高々L²+1群）について方法としての価値を支持した（仮定・既知部品・priority未確立を併記）。R0の限定class最適性とG1の6頂点classは、G4レビューで限定理論成果として保持された [B07]。Parent allocationのfamily内L1最小性は、既知のEuler pairingと三角不等式の帰結と整理されている [B30]、[B31]。
2. **既知法の組合せ・再表現・実装最適化に相当する成果**：BMはcompact BCHと同値、Gaussian積生成は既知構成へ還元、ステップ間相関は既知手続へ対応、Parent allocationは既知機構（Euler＋線形分割、Resource-Optimal IS、SPS/PAI）の特定構造への応用、D2は既知inverse cancellationの全方式共通の実装・会計。
3. **理論保証・有限数値・native費用の区別**：理論は理想meanの恒等式・限定class最適性・十分予算の保証（D3のTaylor積分剰余は新定理ではない）。有限数値とnative費用は主に3-qubitまたは2-qubitのsynthetic fixtureでの加法T/CX/1Q会計（strict-phase合成。DF三次補正はQiskit opt0の局所native）で、Track Aのcompile実測とは別量。
4. **改善部位**：決定論部のPF schedule（BF/BM）、乱択部のsampling law・係数再配分・proposal（R/RA/G1〜G5/Parent allocation。既知cost-aware ISと主題が重なり、④旧C4とも重なる。§6参照）、return集約とnormalizer取得（G6〜G10）、回転合成とsampling配置（SP/BS）、補正挿入（DF三次補正）、基底変換費用（Gaussian、相関、D2）、位相・半径保証（FR-R1b）。
5. **適用条件**：G6〜G10・D1〜D3・Parent allocationはexplicit 3-qubit synthetic involution（p=(1/5,3/10,1/2)、τ=5/7）、r=1/2、ε_axis=1/200、primitive 1e-6、h=0の加法native T。R/RA・G1〜G5は2-qubit P3。分子・DF規模・online古典費用は未取得。H4を使ったのはBF（H4 1.00 Å/DF12/L_D=3）とbasis費用診断 [B38] のみ。
6. **独立アルゴリズムか局所改善か（既存レビューの判定）**：独立した新アルゴリズムを認定したレビューはない。既知へ対応・還元と判定されたもの：BM、Gaussian積生成、ステップ間相関、Parent allocation（既知機構の特定構造への応用）。結合全体の独立priority・非自明性が未確立とされたもの（既知と判定されたわけではない）：G6〜G10 [B11]、D1〜D3 [B21]、M1 [B34]。次期方向レビューは「4候補群とM1/M2を凌ぐ検証済みの新機構は認定できない」とした [B34]。
7. **停止・保留・終了**：ユーザー正式（または利用者採用の記録）：D+E、G5区切り、有限Taylor return-aggregation検証系列の区切り（G10）、DF三次補正generic whole、Gaussian積生成（manifest）、休止記録（manifest）。Codex記録のclosure（逐語採択未確認）：BF現仮説、BM現adapterのnew-method路線。レビュー推奨のみ：D1〜D3数値系列の一区切り、相関を主線にしない、RA v4自動進行なし。主な技術STOP：BF1原run、RA-D0、DF E/T v7、D3 v1、Parent allocation v1/v2。
8. **未解決の研究判断と必要な既存証拠**：次期方向の選択（[B32]の4候補群、M2、IIIのTrack A連携、決定論剰余補正、休止）、休止記録の扱い [B14]、Parent allocation C2〜C5の要否、D1〜D3原稿化（[B21]の推奨、未着手）、FRを独立理論候補として扱うか [B16]。判断に使える既存証拠は [B31]（資源構造：実PRで最大の費用要因はNOT_ESTABLISHED）、[B34]、[B28]。

## 5. ③④の再利用と追加差分

- **③**：統合資料 [C01]〜[C03] をそのまま再利用した。5f079dbe以後、M1-S関連43 branch（[C07]の固定42 branch＋worktree監査用の管理branch 1）で研究pathを変更したcommitはなく、追加はworktree整理の記録 [C07] のみ。理論・数値の再検証、Stage C準備、shot再計算は行っていない。
- **④**：研究範囲資料 [D01]〜[D05] を一次入口にした。56f1cc55（2026-10-16 14:19）以後、全remote branchのcommitを確認し、④のpath・decision.json・正式状態を変更したcommitはない（以後の更新は①③のみ）。共有文書（研究概要・現状.md等）にも④の状態変更の追記はない。高次PF第1・第2研究は別repositoryの固定commitをGitHubから取得し、第2研究のDirection C（ユーザー承認2026-10-05）と第1研究のS5不進行の記載を確認した [P01]、[P02]（内容の監査はしていない）。
- **①②との重複確認（④の観点から）**：①の強化対照整備・native費用会計は既知PRの評価であり④の新アルゴリズム候補ではない。②のcost-aware sampling・係数配分・return集約・補正挿入・Gaussian積は④旧C4、M、H3等と主題が重なるが、入力・指標・正式状態が異なり、相互に結論を流用しない。今回の照合では、④の正式なCLOSED/STOP/PAUSED/HOLD対象に①②が名称を変えて再参入した例は見つからなかった（②Parent allocationは④旧C4と主題が重なるが、C4には正式終了記録がなく、Parent allocationは②内でD+Eとして区切られている）。詳細は[対応表§4–5](overlap_and_progress_map.md)。

## 6. 研究間の重要な重複（要約）

全20方向の対応は[重複防止マップ](overlap_and_progress_map.md)。特に混同しやすいもの（第4列は既存資料が「新しいと数える」ために要求する条件であり、未検討の候補の提示ではない）：

| 混同しやすい方向 | 既に重なる既存研究と正式状態 | 未検討・未取得の範囲 | 既存資料が新規と数えるために要求する条件 |
| --- | --- | --- | --- |
| Hamiltonian分割変更 vs 旧PF小改変 | 一体再配分（現固定構成CLOSED）、H-E1（現定義CLOSED）、M-a/M-b（現定義での追加探索STOP）、PF小改変（PAUSED）、旧D（HOLD）、SOS-PF（現分割生成探索CLOSED、一般no-goではない）、②BF/BM | 非不変目的のDF因子回転は未実施（旧D HOLD） | 係数・配置の小変更を越える入力→回路規則 [D12]、[D14]。小変更は新領域と数えない [D01] |
| 軌道回転・Gaussian frame vs THRIFT/interaction picture | THRIFT現template（CLOSED、THRIFT一般は閉じていない）、H3（現探索PAUSED）、②Gaussian積（独立新手法としてCLOSED、性能UNDETERMINED） | Gaussian積のconditional full-wrapper費用、THRIFT可逆算術の実単価 | THRIFTの再開trigger [D11]、H3では既知Gaussian共役を越えるprimitive [D12] |
| 低体数化・counterterm vs 一体再配分/H-E1/M | 同上＋②DF三次補正（generic whole系列の自動拡張をCLOSED） | 補正演算子の実装費用、同精度でのPR総費用 | 同情報counterterm/fit対照に対する独立の優位 [D23]、量子化学構造による安価な補正演算子 [D14] |
| SOS誤差保証 vs SOS-PF | SOS-PF A2（NOT_ADOPTED）、N3（現検証系列CLOSED） | A2は未着手（着手には別承認） | 回路・測定条件を実際に変える安価な証拠 [D02] |
| PF外挿 vs C2 | C2（正式記録なし、GPTは対照として保持） | 安価な同一branch同定・remainder・複数幅control費用 | 同精度・同成功確率で総費用優位を生む具体機構 [D15] |
| self-energy/downfolding vs 旧C3/N3 | 旧C3（現構成への投資見送り推奨）、N3（現検証系列CLOSED） | 安価なencoding・ground solver、高body項 | D1範囲内の安価な実現と総費用差 [D15]、N3再開経路A/B [D08] |
| cost-aware randomization vs C4/Track B | 旧C4（正式記録なし）、R/RA/G、D1〜D3、Parent allocation（D+E）、相関 | online古典費用、多step、分子DF/PR/RPE | 既知IS/SPSと異なる具体生成機構 [B30]。samplingの変更だけでは新領域ではない [D01] |
| fermion encoding・対称性 vs 表現研究 | N1（STOP）、S12〜S14、THRIFT | 狭い⑤(1)：同一absolute-energy条件でのencoded準備・control・readoutまでの比較記録 | 既知mappingを選ぶ工学的比較を越える回路生成規則 [D02] |
| coherent共有乱択 vs M1-S | M1-S（方向未採択・mandatory STOP）、②M1、相関（step間） | Task B、whole-wrapper単価、強い対照 | M1-S再開のR0〜R7とユーザーの契約採択 [C03] |

## 7. GPTが判断すべき実質的な未解決問題

既存資料で答えがある問い（R09/R10・各closure・正式状態で決着済みのもの）は除いた。順序は優先順位ではない。問いID（Q1〜Q15）はTrack BのG1〜G10や④のprovenance IDとは無関係。

| # | 問い | 系列 | なぜ未解決か（既存資料） | 判断で変わること |
| --- | --- | --- | --- | --- |
| Q1 | 研究全体の目的（同精度・成功確率・情報条件での絶対E0推定資源の削減）に対し、finite-time複素signal task（①T=0.8の有限時間signal、②finite P5 mean、③Task A）の結果をどう位置付けるか。系列横断で比べる主資源指標を何にするか | ①②③④ | R10はenergy/QPE/RPE・実状態準備を①のbenchmark完成の必須条件にしない（主比較はT=0.8、T比較はH6固定δのみ）[A09]。②Decision Packetは主taskと主資源を未決とする [B32]。③Task A≠Task B [C01]。④D2は絶対E0を主taskに固定 [D04]。横断判断の記録はない | ①が全体目標の主線か、帰属・適用範囲の知見かが決まり、③Task Bや④D2の相対的重みが変わる |
| Q2 | 系列共通に許す情報アクセスと取得費用の計上規約（exact ground copies、coherent preparationと逆、energy reference、symmetry sector、状態準備P、保存full-H referenceによる事後選択） | ①②③④ | ④は「許す入力情報」をLevel 1判断に残す [D03]。①はP=0・truth-assisted選択 [A25]、③は保存sector-ground oracle、②も未決 [B32] | 「同じ情報条件での優位」を系列間で比較できるかが決まる |
| Q3 | 「資源削減」を主張する際の強い対照の範囲（登録PF族、最良PF一般、PR-RTE、block-encoding系QPEを参照対照に含めるか） | ①②③④ | ①は最良PF一般への勝利を主張しない [A04]。③の強い対照は未取得。④D1はblock-encoding系を新規研究対象から除外し [D04]、block-encoding中核の構成を「D1外の比較資料」と分類する [D02] が、資源削減主張の参照対照に含めるかの判断記録はない（SOS-PF A3でも範囲判断は別扱い [D06]） | 各結果が「PF族内の帰属・改善」か「全体目標での削減」かが決まる |
| Q4 | PR系の資源で中心となる費用要因・削減機構を何と同定するか | ①② | ②資源構造整理は実PRで最大の費用要因をNOT_ESTABLISHEDとする [B31]。①の機構説明はClaude L1の未採択の解釈（B²×回路圧縮比）[A21]、[A22] と、T panelの観測（厳しい精度ではouter-PF biasによる統計予算縮小が主）[A14] に分かれる | ②再開の具体RQと①の主張文の機構記述が決まる |
| Q5 | ①の中心主張を、どの精度域・どの強さで固定するか（ε=.05の実測逆転、中間精度の固定model条件付き約31〜35%、ε=.0005の約10%）。選択対照の直接compileを主張の前提とするか | ① | V1成果整理は利用者判断に残す [A25]。V1再会計は事前登録の報告triggerを満たしたと記録した（ε=.05で符号変化・42.39 pt低下、中間精度で18.80/18.31/24.97 pt低下）[A23]。その後の利用者指定GPT V1レビューは原本未公開で、確認できる判断はp7両軸実測の優先だけ [A27]。中心主張の精度域・強さを判断した公開記録はない | H6を固定model条件付きで着地するか、追加実測の個別契約が要るかが決まる |
| Q6 | H8強化対照の再会計（Phase C）の結果を①の主張の前提にするか。H6単独に限定するか | ① | Phase Cはユーザーが実行承認済みで、確認時点では結果がない [A47]。強化前比較は登録決定論最良比23.83〜56.12% [A15]。H6では強化で符号・大きさが変わった [A25] | 主張する系と精度域、Phase C後の追加作業（非相殺費用比較等）の要否が決まる |
| Q7 | 適格性判定のbias規約（経験的signed＋allowanceで主結論、非相殺は感度として併記／非相殺でも維持される結論だけを主張） | ① | R10は経験的uを診断政策として採用（非相殺診断の前）[A09]。H8でε=.05の決定論16候補が非相殺で不適格化 [A38]。H6の全domain非相殺はnull。Phase Cは通常signed biasのみを入力とする設計 [A44] | ε=.05の結論の扱いと、H6旧B0+I 12候補のsigned分解復元の要否が決まる |
| Q8 | 実施済みのH5/H7開殻pilotを①の主張範囲にどう位置付けるか（correctness/feasibility記録に留める、閉殻H4/H6/H8に限定、全面mapへ拡張） | ① | R10（GPT）はH5→H7を採択し、実行された。Claude L1の非推奨は利用者採択範囲外 [A22]。pilot結果の科学レビューはない [A39]、[A40] | 開殻・奇数鎖を主張に含めるか、全面mapへ資源を割くかが決まる |
| Q9 | ①で観測された厳しい精度でのS4優位・PR不適格（T panel [A14]）とV1でのε=.0005の縮小（10.03%、固定model条件付き [A25]）を、①（V3型比較）、②（外側S4のPR、候補群III、決定論剰余補正）、④観点C3（量子回路の実装構造）のどこで扱うか | ①②④ | Decision Packetは隣接課題との分担をGPTに残す [B32]。次期方向レビューはIIIをTrack A連携候補とした [B34]。Claude L1はV3を任意・外側S4方法開発を別研究とした（未採択）[A21]。①の限界と④観点C3を対応付けた既存記載はない | ②再開の具体RQ候補になるか、主張範囲の端として記述して終えるかが決まる |
| Q10 | ②を新しい具体RQで再開するか、既存成果をnote化して区切るか、休止を維持するか | ② | 4候補群は未採択・順位なし [B32]。次期方向について正式採択はD+Eのみで、[B34]はM1をread-only P0の第一候補（非正式）とし、I/M2/IIIを条件付き代替として保持した（M1はその後③へ移行 [B37]）。休止記録の明示的解除なし [B14]。[B21]の原稿化推奨は未着手 | ②への資源配分の有無と、再開時の最小判別検証の設計 |
| Q11 | ③M1-Sを、ノート化で区切る／Task A論理総費用の限定比較（R5）／Task B多round RPE（R6）／理論のみ（R1・R2・R4）のどれで扱うか。他系列との関係 | ③ | Claude・GPTの推奨はいずれもNOT_ADOPTED、PR-2等との定量的優先比較は未実施 [C04]、[C05] | ③への計算資源の有無、新契約設計の要否 |
| Q12 | ③T1a/T1bの定理statement全体のpriorityと、②の結合生成法（局所生成・P5少数群）の独立性を、どの範囲・どの資料で確定させるか（または確定を求めないか） | ②③ | ③は定理statement全体のpriority未確定 [C04]、②は結合生成法の独立priority未確立 [B21]。網羅的priority調査はどのレビューでも未実施 | 論文化・ノート化の主張範囲が決まる |
| Q13 | ④で測定・情報取得の改良を研究範囲に含めるか、記録不足の狭い2問を独立研究問いとするか、次の調査に入る必要条件（具体的構成・適用クラス・未確認定理）を何とするか。④を選定段階のまま維持するか休止に移すか | ④ | Level 1判断6点が未承認・未開始 [D03]。狭い2問は既存研究の別名や実装比較に留まる可能性が明記 [D01]、[D03] | ④の候補空間（回路生成規則のみか測定を含むか）と実質的な休止の判断 |
| Q14 | 有限step誤差・bias・C-sec等の保証理論を独立主題とするか、回路・情報取得・shot費用を実際に変える場合に限るか（SOS-PF A2、M1-S T1c、①の厳密bias certificate） | ①③④ | ④は保証理論の位置付けを判断点に残す [D03]。A2はNOT_ADOPTEDだがCLOSEDではない [D06] | ③④の理論研究への配分と、①の主張をcertificate付きに強める道の有無 |
| Q15 | ①〜④の間の研究資源配分の判断基準 | ①②③④ | 各系列の到達点は指標・対照・taskが異なる（①H4/H6/H8固定条件の実compile RZ×解析的shot、新アルゴリズムではない／②synthetic fixtureの加法native T、独立新規性未認定／③同一保存信号の十分shot整数witness、総費用未評価／④独立新回路生成規則なし（技術判定））。系列横断の比較基準・配分基準の記録はない。③GPTは他研究との定量比較を未実施と明記 [C04] | 次に承認すべき作業（①追加実測・非相殺費用比較、③R5/R6、④Level 1開始）とnote化・休止の対象 |

**新しい科学判断なしでCodex（技術担当）が取得・照合できる事項**（いずれも未承認。実施にはユーザーの指示が必要）：

- Track A横断の統合状態文書の作成（10-15共通ノート [A05] 以後に分岐したH6/H8 B0+I、T panel、理論評価、V1/V2、H8 Phase A/B/C、bias系列、H5/H7を固定commitで接続し、[A16]の古い記載を解消）。新計算なし。
- 正式記録の欠落照合：GPT V1レビュー原本、H7実行承認文言、[A13]が戻したGPTレビューの有無、H6契約wall 4hと将来wallなし方針の不一致、Track Bの休止記録の解除有無、C1〜C4の正式化の有無。
- 未公開レビュー原本の回収・公開（§8の表。ユーザー保有ファイルの提供が必要）。
- ④・③R1の静的照合（具体的な対象が提示された場合のみ）。

**新しい科学計算・保存値再解析の承認が別途必要な事項**（H8 Phase Cは承認済みのため除く）：非相殺COST_COMPARISON、V1追加実測、H6旧B0+I 12候補のsigned分解復元、Claude L1紙上算術の再計算・V4、Parent allocation C2〜C5、実PRの決定論部／乱択部別native費用内訳の取得、M1-S R5/R6。

## 8. 資料の不足と取得状況

| 資料 | 取得状況 | 影響を受ける判断 |
| --- | --- | --- |
| 本資料が引用する主要本文（索引 key_evidence）と、読取りで引用した補助本文（supplementary） | 全件をGitHubの固定commitから独立取得し、local blobとbytes・SHA-256一致（別repository 2件はGitHub取得のみ）。件数は索引の summary | — |
| SOURCE・raw結果・監査 | 各系列の公開branchに軽量projection・hash・監査JSONあり。大容量NPZ・runtime・grant本体・private入力は設計上非公開（server private領域）。本作業では科学値を再集計していない | 値の独立再現（外部再現・CIは全系列で未実施） |
| GPT V1強化baselineレビュー原本 | **未公開**（bytes・SHAのreceiptのみ [A27]、対象範囲は不明） | Q5 |
| H6/H8 B0+I結果に対するGPT採否レビュー | **公開記録上は未確認**（[A13]は次のレビューへ戻すと記載、[A46]はR11を作成していないと記載）。Claude L1 [A21] は結果を含めてレビュー済み（利用者採択は技術準備のみ） | Q5、Q6 |
| H7本実行のユーザー承認文言 | **未公開**（authorization digestのみ [A41]） | 実行来歴の完全性 |
| R09補助算術script/ZIP、local指示48,223 bytes版、H4 run03 a3248055の公開到達性 | 未取得・未確立。各記録は未解決として保持し、解決済みとしない [A05]。R09本文はexact copy公開済みで補助物の未取得とは区別される。影響度を評価した記録はない | ①の来歴 |
| Track B：S4完成版レビューのClaude原稿「レビュー.md」・GPT「レビュー修正点.md」 | 名称引用のみで未保存 | Q10 |
| Track B：次期方向レビューが参照したClaude「レビュー_inner.md」 | SHA引用のみで未保存 | Q10 |
| Track B：G10 v3のGPT再現script/ZIP | 未提供 [B10] | — |
| ③：旧Claude inner原本3件 | 未取得（二次記録のみ） | Q11、Q12 |
| ③：Davies 1974全文 | 未取得 | Q12 |
| ③：Yi–Crosson出版SI | 取得・関連箇所（Supplementary Note 5）確認済み、byte hashは未記録、出版版とpreprintの全式同一性は未認定 [C10]、[C02] | Q12 |
| ④：H1〜H4の起点GPT Level 1レビュー（main rootのローカル未追跡ファイルのみ、未公開）、Claude「レビューoutter(3).md」等 | 未公開・未取得 | Q13 |
| 文献：Kang–Su ACM出版版（403）、L4 Nature掲載版、PR原論文新版の出版差分 | 未取得 | Q13、Q14 |
| 公開とローカルの区別 | 全引用commitはremote上に存在。main rootのdirty・未追跡ファイルは正としていない。資源方針の正式版は共通運用指示2026-10-17 §8.4 [X02] としてcommit済みで、main rootのAGENTS.mdにある未commitの資源方針diffは正としない | — |

必須資料のうち、①の中心主張判断（Q5）に関わるGPT V1レビュー原本と、[A13]が予定したGPTレビューの有無が欠けている。原本はユーザーが保有している可能性があり、要約から再生成していない。

## 9. 保全・実施しなかったこと・STOP

- 新規科学計算、Hamiltonian生成、固有値・RDM計算、PF/RTE simulation、回路build/compile、T合成、パラメータ探索、benchmark、保存数値の再解析・再ランキング：**0件**。
- 新しい科学レビュー、新規性判定、研究方針・優先順位の正式決定、新アルゴリズムの採択、新しい証明：**0件**。
- 既存SOURCE・raw結果・契約・authorization・grant・ledger・marker・レビュー原本・正式状態：**変更なし**。他branch・worktree・jobへの操作（削除・prune・merge・reset・プロセス操作）：**なし**。
- 本資料は独立した管理branch `qc-four-track-strategy-handoff-20261017`（基点 `56f1cc55`）に新規ディレクトリとして追加しただけで、既存ファイルを変更していない。
- 本資料の公開と取得照合の後、**mandatory STOP**。GPT重要科学レビュー、研究方向の採択、新アルゴリズム探索、数値計算へ自動移行しない。次の担当は、GPTレビューの開始を判断する**ユーザー**である。

## 参照ID（固定GitHub本文）

各IDのrepository・branch・commit・bytes・SHA-256・GitHub取得照合は[固定証拠索引](fixed_evidence_index.json)の `key_evidence` にある。

- [A01]：10-14統合資料が参照したH4 geometry結果（E12）。B0+I導入前の対照集合
- [A02]：10-14統合資料が参照したH6独立確認（E11）。full PF比のみ
- [A03]：H4 B0+I全240軸実測（MAP240_COMPLETE_STOP）。拡張集合の点最小はB0+I 397/B2 1,415
- [A04]：R09（GPT原本exact copy）。H4を限定benchmarkとして採用。PR初提案・最良PF一般への勝利・scalar保持初提案を主張しない
- [A05]：Track A正式共通ノート（情報入口として正式統合）10-15追補。以後の系列を束ねた共通ノートは存在しない
- [A07]：H4 native fixed16/H6のD0保存値one-shot。ε=.0005のPR/full PF比 H4 0.430、H6確認 0.496
- [A08]：H8 D0保存値one-shot。確認32本でPR/取得済B0-B1最良 0.439〜0.762（ε=.05で0.605、ε=.0005で0.762）
- [A09]：R10（Codex索引でGPTに帰属。本文に自己記名なし）。条件付きbenchmark採用、B0+I強化対照、H5→H7・T比較を科学採択。実行認可なし
- [A11]：H8 resource map core。full PF比54.29–65.76%減、B0比39.48–56.12%減（P=0、点推定）
- [A12]：H6 native B0+I代表6軸pilot。3構成のみとの比較（後続全12構成で置換）
- [A13]：H6 native B0+I全12構成。PR点費用低減 29.35/49.69/53.27/56.14/50.37%（ε=.05〜.0005）。採否は次のGPTレビューへ戻すと記載
- [A14]：H6固定δ=.2のT=.4/.8/1.6比較。ε≤.001でS4優位またはPR不適格のセルあり
- [A15]：H8 native B0+I全12構成（適格8構成16軸）。PR点費用低減 28.77/52.72/52.73/56.12/23.83%
- [A16]：共通ノート最終更新版。T panel・H5・H7完了を含まない古い記載（『実行認可未取得』）が残る
- [A17]：native DF理論gate数モデルA/B（固定分解RZ/CX/SX/X）。T/Toffoli・depth・routing・state prepは対象外
- [A18]：理論モデルとQiskit 1.3.0 builderの人工入力接続検証。比較契約への採否はGPT判断待ちと明記
- [A19]：H6/H8 208候補の理論832行。ranking・残差なし、scientific_adoption=false
- [A20]：ユーザー指定参照点。V2 H6 cost binding準備（approved_by_user=false）
- [A21]：Claude Level 1レビュー原本（dd695e65にも同一blob）。新規性なし、V1必須・V2推奨・H5/H7非推奨
- [A22]：Claude L1の利用者採択範囲『V1/V2 technical preparation only; H5 unchanged』、scientific_adoption_of_review_estimates=false
- [A23]：V1 H6強化決定論136候補での5精度再会計（固定model条件付き）
- [A24]：B0I p7/q1 S2両軸実測でε=.05のPR削減率−13.29%（逆転維持）
- [A25]：V1成果整理。ε=.05のみ実測点比較、他4精度は固定model条件付き。追加実測は利用者判断後
- [A26]：V1追加実測候補（未選定・未承認、decision_owner USER）
- [A27]：GPT V1レビュー原本は未公開。bytes/SHAのreceiptのみ
- [A28]：V2来歴照合。比較可能PF 0件、現行残差設計をNO_GO候補（Codex技術判定）
- [A29]：V2現行残差fit設計 CLOSED_NO_GO_BY_USER_DECISION（decision_owner USER、利用者原文の引用なし）
- [A30]：H8強化格子Phase A（追加152候補のsignal・bias・適格性、費用未評価）
- [A31]：H8 B0新anchor 8軸の保存値再利用監査（run02は終了時の技術STOP）
- [A32]：H8 Phase B1：S2費用model固定、B0独立holdout誤差0%
- [A33]：H8 Phase B2：登録S4 4軸すべて2%未満でPASS（17:46:48完了）、mandatory STOP。Phase C未実施・未承認
- [A34]：B2 sine軸RUNNING記録（17:22時点）。A33で完了に更新済み
- [A35]：H8 bias証拠監査。費用比較計画は未承認
- [A36]：signed成分復元・整合性（184候補368軸PASS、経験的整合性、厳密保証null）
- [A37]：非相殺診断の準備・Codexレビュー修正、plan v2をSOURCE cbfd7cacへbind
- [A38]：非相殺NONCANCELLATION_SHOTS stageのみ実行。ε=.05で決定論16候補が不適格化、PR6不変。費用比較stage未承認
- [A39]：H5中性開殻pilot（各q1・PR各8標本）。ε=.05でPR K4/full S2 0.553669、ε≤.01でPR不適格
- [A40]：H7中性開殻pilot。ε=.05でPR/full S2 K2 .684・K4 .706。PRはε≤.01で不適格（ε=.01はfull S4のみ適格）、ε≤.005は全構成不適格
- [A41]：H7実行承認はauthorization digestのみ（承認文言は公開Gitに未記録）
- [A43]：R10以前の技術handoff。希少次数の厳密compiled上界はBLOCKED_BY_MISSING_EVIDENCE（技術状態）
- [A44]：H8 Phase C統合・実行準備（記載SOURCE 29ed7654、plan 9e42fc76、契約未承認・unsealed、無効grant、mandatory STOP）
- [A45]：Phase C承認条件（valid_grant_issued=false）
- [A46]：R10後続実行の完了記録（新reviewではない、R11・詳細GPTレビューは作成していない）
- [A47]：H8 Phase C：利用者が通常signed biasに基づく一回実行を承認。SOURCE 931755c4・plan 79d624b9へ再固定。確認時点で科学実行0・有効grant0（公開認証待ち）
- [B01]：BF現仮説を限定negativeとしてclosure（GPT推奨に基づくCodex記録、逐語採択未確認）
- [B02]：BM DF multirate adapterはcompact BCHと三次係数が完全一致（new-method gate不通過）
- [B03]：BS-0.5：generic sparse LCUに対する独立method delta未定義
- [B04]：R1.5：精度選択を許すと再配分Aは全座標で優越しない
- [B05]：RA-D0再評価（v4 production自動進行なし、GPT推奨）
- [B06]：ユーザー指定参照点。G3有限law診断（固定pool内、post-hoc）
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
- [B17]：D1保存資源解析（10-14統合資料E14）
- [B18]：D2境界相殺の最終統合（相殺による上積みT利益は約0.02 pp）
- [B19]：D3 v2：保証×価格（G1_P1/r1でP3+tailがP5より0.170% T低い、固定rational会計内）
- [B20]：ユーザー指定参照点。D1〜D3先行研究13 claim照合（Codex技術資料）
- [B21]：D1〜D3 post-literature GPTレビュー（数値系列の一区切り、一体原稿化を次作業として採択・推奨。利用者指示は研究ノートのみで原稿は未着手［B22］）
- [B22]：return aggregation G/D研究ノート（新しい科学判断なし）
- [B23]：Track B方向レビュー：parent係数のchild別配分を第一候補（推奨、実行承認ではない）
- [B24]：Parent allocation科学設計レビュー（限定native機構比較を選定）
- [B25]：Parent allocation v3（15構成・70 law前段）。v1/v2は技術STOP
- [B26]：v3レビュー：限定T利益を支持、内部KKT実装と採択目的の不一致を発見
- [B27]：S4訂正を含む正式比較集合70 lawのevidence packet
- [B28]：S4完成版レビュー（Claude原稿をGPT-6改訂）：既知機構の特定構造への応用、独立新規性未認定、D+E推奨
- [B29]：ユーザーがD+E（Parent allocation限定成果の整理＋同一fixture数値探索の区切り）を正式採択。新科学計算の承認ではない
- [B30]：Parent allocation closure（一般no-goではない、再検討条件§6）
- [B31]：Track B資源構造整理。実PRで最大の費用要因はNOT_ESTABLISHED
- [B32]：次期方向の4候補群I〜IV（未採択・順位なし）
- [B33]：ユーザー指定参照点。D+E整理公開receipt
- [B34]：次期方向GPTレビュー：M1をread-only P0の第一候補（正式採択はD+Eのみと明記）
- [B35]：M1 P0（M1_adopted=false）。③M1-Sの直接の前段
- [B36]：M1 P0.5（M1採択保留）。直後の41fa1eb4から③
- [B37]：②/③境界：GPTがM1_NOT_ADOPTED・M1-S再定義を暫定支持（レビュー判断）
- [B38]：H4保存値のbasis費用割合診断（加法比、T比率NOT_MEASURED）
- [B39]：BM現adapterのnew-method路線を閉じる（post-BM05 GPTレビュー受領に基づくCodex記録）
- [B40]：G10 v3論文化・着地点レビューの利用者採用（有限Taylor return-aggregation検証系列を区切り、技術ノート完成）
- [B41]：FR-R1b（位相・半径保証）：COMPLETE_MANDATORY_STOP、MECHANISM_ONLY_NO_PRACTICAL_GO
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
- [C09]：Stage S：G3 T=4の8対照にcap直下整数witness（同一保存信号の事後解析）
- [C10]：M1-S先行研究調査（30レコード、網羅的新規性証明ではない）
- [C11]：M1-S G0-pre packet（M1 P0/P0.5資料の再利用を明記）
- [D01]：④既存研究と探索範囲の対応表（分類①〜⑤）
- [D02]：④残る科学的問題（観点C1〜C5、L1〜L8）
- [D03]：④GPT Level 1判断点（未承認・未開始）
- [D04]：H1〜H4現探索PAUSED（ユーザー承認の追記記録）、④全体は課題選定段階・正式採択なし
- [D05]：user_adopted_H1_H4_pause=true、new_algorithm_formally_adopted=false
- [D06]：SOS-PF A3：現分割生成探索CLOSED、A2 NOT_ADOPTED、K1-F NOT_AUTHORIZED/NOT_EXECUTED
- [D07]：一体再配分PF現固定構成のみCLOSED、H-E1現定義の探索CLOSED、M-a/M-b現定義での追加探索STOP（M K1未認可）、PF小改変PAUSED（この系列）、旧D HOLD（C1_C4はUNCHANGED）
- [D08]：N3現検証系列（追加benchmark・帰属系列）CLOSED・主研究不採択（理論・限定prototypeは条件付き保持）、N1 STOP（現候補scan）、N2 HOLDを列挙（利用者採択GPTレビュー）
- [D10]：THRIFT/mixed現template CLOSED（THRIFT一般は閉じていない）
- [D11]：THRIFT現templateの再開trigger
- [D12]：H1〜H4静的K0：総合NO_GO_NO_INDEPENDENT_GENERATION_RULE、H2 UNDETERMINED_DATA_REQUIRED（技術判定）
- [D13]：既存技術基盤対応表（G9）
- [D14]：GPT再レビュー：C1現系列終了、C2対照保持、C3/C4投資見送り（ユーザー正式採択記録なし）
- [D15]：C2/C3最終技術監査（既知手順へ対応、独立低費用手順未特定）
- [D16]：C4技術監査（既知IS等へ還元、Phase B停止）
- [D17]：C1保存oracle診断（通常16 contextで登録5%優位なし）
- [D18]：SOS-PF K0（UNDETERMINED_DATA_REQUIRED）
- [D20]：一体再配分h境界再監査v2（登録10%改善は4/4未達）
- [D23]：M K0（core REDUCES_TO_KNOWN、UNDETERMINED_NOT_READY）
- [P01]：高次PF第1研究の最終統合（別repository、④範囲表G10が参照）
- [P02]：高次PF第2研究の方針（Direction C、別repository、④範囲表S7が参照）
- [X01]：2026-10-14時点の横断統合資料（G2）。①②は当時snapshotで現在の最終状態ではない
- [X02]：共通運用指示2026-10-17正式版（§8.4 資源方針）。研究の科学条件・実行許可は定めない

[A01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md
[A02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_result_v1.md
[A03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md
[A04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_h4_b0i_v04_scientific_review_20261014.md
[A05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b78e5f39917d92d6c60694d3e0b0d06394d7eb5/docs/research/track_a_shared_research_status.md
[A07]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dbdb68293c988a999645e21a82b53966f9ef6cd/docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md
[A08]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_saved_value_result_20261015.md
[A09]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md
[A11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md
[A12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae3b992f22aead88515ef594cde7e5f867aefa87/docs/research/track_a_h6_native_b0i_pilot6_results_20261015.md
[A13]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h6_native_b0i_full_family_results_20261015.md
[A14]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md
[A15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_h8_native_b0i_full_family_results_20261016.md
[A16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_shared_research_status.md
[A17]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1ea42fbf1bd1c8341ffd410b8067467b4ba4e170/docs/research/track_a_native_df_theoretical_gate_model_20261016.md
[A18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d2431b8c2a385af20a3d996cf8074b7a826a6e6b/docs/research/track_a_native_df_qiskit_connection_validation_20261016.md
[A19]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e41670565d9284da9aded94063be591fdcaa3dd/docs/research/track_a_native_df_theory_evaluation_20261016.md
[A20]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd695e6584f15b88341d2446ed010c7bdfef1666/docs/research/track_a_v2_h6_cost_binding_readiness_20261016.md
[A21]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e48a6edaa51bd474b8185bfd0f012bf83b0b43de/docs/research/provenance/track_a_claude_level1_scientific_review_20261016.md
[A22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd695e6584f15b88341d2446ed010c7bdfef1666/artifacts/resource_applicability/track_a_v2_h6_readiness/2026-10-16/review_provenance_v1.json
[A23]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25bbdd33fe758a0e730138cb91ddac8c41b0a298/docs/research/track_a_v1_reaccount_results_20261016.md
[A24]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md
[A25]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/63e5773b6b408506f4c39783b80f3025f14481b9/docs/research/track_a_v1_results_synthesis_20261016.md
[A26]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/63e5773b6b408506f4c39783b80f3025f14481b9/artifacts/resource_applicability/track_a_v1_results_synthesis/2026-10-16/additional_measurement_candidates.json
[A27]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/84b9ea22bf78c7193f2137ce42a04400292b8e68/artifacts/resource_applicability/track_a_v1_p7_readiness/2026-10-16/metadata_audit.json
[A28]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68a010b35d35a1500dcc901c480fcce433fac4a8/docs/research/track_a_v2_residual_analysis_readiness_20261016.md
[A29]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/36756714fee37a7a6e1188bb4d765292d877ea18/artifacts/resource_applicability/track_a_h8_strengthened_readiness/2026-10-16/v2_design_closure_v1.json
[A30]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01309059636f2432e29b663bcd2eb0e400f5ae0a/docs/research/track_a_h8_phase_a_run01_results_20261016.md
[A31]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b19c337f0b16513d68b59eb3366dc6eff175c313/docs/research/track_a_h8_b0_saved_anchor_audit_20261017.md
[A32]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83a3c4a6688e66576f85471bf41a45e29cbfb705/docs/research/track_a_h8_phase_b1_run01_results_20261017.md
[A33]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fef2db7a79886ab682b652da7266b463c98c7d3a/docs/research/track_a_h8_b2_completed_results_20261017.md
[A34]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a66da0def417dcf52da39c918aed90c023837bc5/docs/research/track_a_h8_b2_sine_psi_running_20261017.md
[A35]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b75039c1d5ead475d8c0ff8b62dba04a9978a6fa/docs/research/track_a_h8_bias_evidence_audit_20261017.md
[A36]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8e54303bfe46bbebbf4d37148cbe26ef4013ae38/docs/research/track_a_h8_signed_bias_result_20261017.md
[A37]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a6227f5b55f6553cf028320a4e450db9d1ca449/docs/research/track_a_h8_noncancellation_readiness_20261017.md
[A38]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f902229cef9da26cb6feeff09eb35860b6115a6/docs/research/track_a_h8_noncancellation_shots_result_20261017.md
[A39]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bde58ba23f1fecdf77ee950795cb80c199c0f86b/docs/research/track_a_h5_neutral_open_shell_pilot_result_20261016.md
[A40]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/162c256ea39b4ab3dadde65a94b3eec73cea4720/docs/research/track_a_h7_neutral_open_shell_pilot_result_20261016.md
[A41]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/162c256ea39b4ab3dadde65a94b3eec73cea4720/artifacts/resource_applicability/track_a_h7_neutral_open_shell_pilot_v1/2026-10-16/launch_binding_v1.json
[A43]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ee211d72a0c8d3c35cb605ccddc0af7e47a46952/docs/research/track_a_pre_review_scientific_decision_handoff_20261015.md
[A44]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7be02fcf705425a0d98e871b19d0c1aebf60c09/docs/research/track_a_h8_phase_c_integration_readiness_20261017.md
[A45]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7be02fcf705425a0d98e871b19d0c1aebf60c09/artifacts/resource_applicability/track_a_h8_phase_c_integration/2026-10-17/approval_conditions_v1.json
[A46]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_scientific_review_decisions.md
[A47]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b357c77a1a35d70dde011d44516bd8cab3bfcc8a/docs/research/track_a_h8_phase_c_execution_binding_20261017.md
[B01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b2d624adde979f7d8f983bc7fdbbec88f90c500/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md
[B02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d55de044b8e956ba6292209a94bb081014dfdae2/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md
[B03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a4ae817ec8d833bb2929c0c0a85e2d4d3064e7d/docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md
[B04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af3d014d0a0cfcbbd25bb544f6544652fec92942/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md
[B05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/86df033cea392664cf365a80e5f068091dc8f5e8/docs/research/track_b_reassessment_and_gpt_checkpoints_20261009.md
[B06]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md
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
[B17]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/docs/tracks/algorithm_codesign/d1_saved_resource_analysis_and_gpt_handoff_20261013.md
[B18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2827c14fec96613a60bb1427287b30766d5da/docs/tracks/algorithm_codesign/d2_final_saved_integration_and_gpt_handoff_20261013.md
[B19]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md
[B20]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[B21]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/79a1ff0a45e84f3e45c09cc7ede9255310449302/docs/research/track_b_d1_d3_post_literature_scientific_review_20261014.md
[B22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66c96f8033bcf8392237d820d4a4615f3cebc75a/docs/research/%E7%A0%94%E7%A9%B6%E3%83%8E%E3%83%BC%E3%83%88/2026-10-14_track_b_return_aggregation_G_D_research_note.md
[B23]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d80b45693a1ea7c14d7cbf9035e7358788953bd/docs/research/track_b_algorithm_direction_review_20261014.md
[B24]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/117000dfbe7e0a8f8166303c8c0a24049dbd19ee/docs/research/track_b_parent_allocation_scientific_design_review_20261014.md
[B25]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/751d6ccd76c387a25beeeffee66fff7ddcbedcc7/docs/tracks/algorithm_codesign/parent_allocation_v3_results_and_gpt_handoff_20261015.md
[B26]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc9a627a60f53704597a59bd377512270687bf95/docs/research/track_b_parent_allocation_v3_scientific_review_20261015.md
[B27]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md
[B28]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3086f48c5598c8bb31b0587e807ca6c1b3d2754/docs/research/parent_allocation_s4_scientific_review_revision_20261015.md
[B29]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288819e17821762eeaab62edb590483f05ddb444/docs/tracks/algorithm_codesign/inputs/track_b_de_direction_preparation_user_instruction_20261015.txt
[B30]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288819e17821762eeaab62edb590483f05ddb444/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[B31]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288819e17821762eeaab62edb590483f05ddb444/docs/tracks/algorithm_codesign/track_b_resource_structure_and_bottlenecks_20261015.md
[B32]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288819e17821762eeaab62edb590483f05ddb444/docs/tracks/algorithm_codesign/track_b_next_algorithm_research_decision_packet_20261015.md
[B33]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/artifacts/track_b_de_direction_preparation/2026-10-15/v1/publication_receipt.json
[B34]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ef2c62b4b6fea5dcef2674a29db821cf91bab6/docs/research/track_b_next_direction_gpt_scientific_review_20261015.md
[B35]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa91ffb3d4880d767e55f0f661e8ade294b4c9fe/docs/tracks/algorithm_codesign/m1_p0_prior_art_control_and_resource_decision_packet_20261015.md
[B36]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md
[B37]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41fa1eb480728cd0691b00bb41dd4e5dc4ba6b03/docs/research/track_b_m1_claude_review_gpt_independent_assessment_20261015.md
[B38]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/artifacts/rte_basis_cost_attribution/2026-10-13/README.md
[B39]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3861e6b941745e43863f6a62cd25fe36f3b3e108/docs/tracks/algorithm_codesign/synthesis_placement_design_review_20261006.md
[B40]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e5a5b0a5973f1725c7a6872a4b9b39b86d66cef/artifacts/track_b_return_aggregation_note_completion/2026-10-11/review_intake_identity.json
[B41]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/fr_revision_nonuniform.md
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
[C09]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2bfb8770e6858de56a63e50e68f07ff2d6464b6/docs/tracks/algorithm_codesign/m1s_stage_s_one_shot_results_and_stop_20261017.md
[C10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/research/m1s_prior_art_survey_20261017/README.md
[C11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md
[D01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md
[D02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/open_problem_landscape.md
[D03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/gpt_decision_handoff.md
[D04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/research_state_transition.md
[D05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/provenance.json
[D06]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/artifacts/qc_sos_pf_a3_state_adoption/2026-10-16/decision.json
[D07]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json
[D08]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/afecaf591e62b8ab7ba5fb5c20f53f17a6d7be37/artifacts/n3_achievements_closure/2026-10-13/research_decision.json
[D10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e54991ef2c80fe463de9cdb865b2c48d5b981607/artifacts/qc_mixed_access_closure/2026-10-13/research_decision.json
[D11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e54991ef2c80fe463de9cdb865b2c48d5b981607/artifacts/qc_mixed_access_closure/2026-10-13/closure_state.json
[D12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d460c54f3bf54ea46d4d8129f6702703c4bb217/docs/research/qc_new_algorithm_static_screen_k0_20261016/technical_report.md
[D13]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/acfa8674d51ede1d5cd50f1b49873e67de5b71b9/docs/research/qc_existing_technical_foundation_map_20261016.md
[D14]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6711a709ad13144972474effe1faf3933cd7f7fb/docs/reviews/quantum_chemistry_pf/qc_exact_ground_state_pf_direction_rereview_2026-10-14.md
[D15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d91991038b1a2b97fac52a00d8ed4ebb413e2214/docs/research/qc_c2_c3_final_technical_audit_v1_20261014.md
[D16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/60da5738d48e0faaedd3de125b0350301935dc3a/docs/research/qc_c4_finite_rte_method_efficiency_technical_audit_v1_20261014.md
[D17]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5ff55e749ceb33e54da0b3c70ac6ac08599ebb82/docs/research/qc_c1_saved_oracle_diagnostic_v1_results_20261014.md
[D18]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md
[D20]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bbb611f95a7be5036ea0291655862bf3bb7d619/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md
[D23]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/artifacts/qc_m_fixed_circuit_k0/2026-10-15/method_status.json
[P01]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/cd81018932b9557b7a55e2e1cd8c5e3988245409/PF_first_study_final_synthesis_20260925.md
[P02]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/1d01e77247364b49dbce2e9ce2fe7c32af0fb937/docs/second_study_v2/budget_safety_mechanism_20261005/final_direction_and_claim_scope.md
[X01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md
[X02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ef39e3af4d479a1d3ee1a2792f734d67f208872/docs/research/research_project_instructions_unified_2026-10-17.md
