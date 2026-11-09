# PR＋RPE総コスト改善：既存成果横断調査

2026-11-09 JST／SERVER 2／Codex主エージェントのみ。
Repository：`HIROMU1015/Partially-Randomized-Trotter`。

**強Fullとの差をさらに縮める問いは残るが、逆転を支持する未確認の改善率は置けない。Closed P3接続とh/resolution/α共同選択は実施済みであり、同じ再探索を新研究として提案しない。** 残る候補を、回路接続、D/R/drop分割、元H groundへの表現接続、confidence保証、追加のground補正制御に分けた。

## 成果物

| 成果 | 内容 |
|---|---|
| [A. 既存成果・重複整理](EXISTING_RESULTS_AND_OVERLAP.md) | 条件の対応、Track A/B/C/Dと関連研究の検証範囲・負結果・SOURCE・STOP、旧調査後の更新 |
| [B. 改善候補の比較](IMPROVEMENT_CANDIDATES.md) | K1–K6の具体的変更、総費用因果経路、既知法との差、相殺要因、証拠・再利用資産。parameter再探索とHOLD事項を分離 |
| [C. 最小検証計画](MINIMUM_VALIDATION_PLANS.md) | 候補別の最小入力・検証・科学的意義、必要な実行許可、GO/STOPと担当判断 |

## 重要な発見

1. **最新Closedの利益は確認済み。** c5a3c102の最終extended結果は、旧固定Closedを同じholdoutで再評価した比較で総RZ6.03267655%減、再最適化Canonical比2.11497621%減。総RZは1.303898609e12、強Fullは5.647934021e11、保存比2.308629。全18roundのD/h/resolutionは両P3で同じだった。[原報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md)
2. **RTE価格とtrialの交換を既に評価している。** m17のevent25%減はrange悪化・trial増を伴うが総費用は減った。最終D RZは増え、RTEは9.599285356e11へ減った。normalization、単価、trialのどれか一つを最小化するだけでは判断できない。同原報告の保存値を引用し、新しい寄与率は算出していない。
3. **比較対象はFullだけでは足りない。** 同じ元H ground・18round条件のH4 Truncated PFは、combined AでRZ3.337772238e11、相殺を弱めた保守Bでも3.864575491e11。H2は登録PRが低RZ、H6はA/BでPRとの順位が変わる。H4の負結果を全PRのno-goへ一般化しない。[最新C統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3c81c2677988c7b9491eeee31fbcc0f9b29b078/docs/research/track_c_o3_integrated_20261108/TRACK_C_O3_INTEGRATED_REPORT.md)
4. **旧調査の次候補には完了済みのものがある。** D C1の人工・実H4/q4は完了。一pathでは両方式RZ−23/CX−30、Referenceの相対CX差は不変、depth差は縮小した。q64も全8 path/32post完了。いずれも全law／全RPE成功資格ではない。[C1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/141fa7624d74aa5deacbd752d09df516cad9206f/docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md)、[q64](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe361b206a6572142775ac01972545d312e5fb14/docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md)
5. **表現・測定・Referenceの既存結果には移送条件がある。** RCはown-DF ground、A M2はoperator-only/model未採択、B C1は別324-label/P5、DはCDF8/K0/P0、現CはDF12/P3/compiler A。新Aのnative CX/depth改善や有限T例も最新ClosedのRZ勝利ではない。O2のIS/CV/LCUとcheap summary未特定の負結果も再利用する。[Aの対応表](EXISTING_RESULTS_AND_OVERLAP.md)

## 残された研究機会と独立性

K1はCの異basis joint completion間を既存relative Gaussianで結ぶ問い。K2は最新Closed＋joint/fusionへ既存Referenceを接続する問い。K3はD/R/dropの中間配分、K4は既存表現・residualを元H groundへ接続する問いである。K5は同lawのconfidence保守性、K6はleading相殺の先の組成成立性を扱う。K5/K6は利益未評価／構成未確立で、即実行可能な算法と称しない。

K1・K3・K5は回路・近似・統計という別軸で判別できる。K2とK1はbasis／boundary費が重なるので単独ablation後に組み合わせる。K4のpromise差を閉じる前に他候補の保証を移さない。削減率は足し合わせない。具体性からK1の固定2-run又はK3の保存retention一組が小さい判別案になるが、これは正式な優先順位・採択ではない。

単なるP5/r/K/h/αの追加探索、depth目的の再選択、D長q価格資格補完は、それぞれ既存familyの探索／目的変更／リソース評価として分けた。whole-path samplingは安価な実conditional取得構造が示されるまでHOLD。RPE以外の推定器は目的family変更としてユーザーへ戻す。

## 実施範囲・指示・branch

正式共通運用指示 **2026-11-09版を実環境の `/home/AbeHiromu/projects/research_project_instructions_unified_2026-11-09.md` から取得し、全体と§10.4を読了した。** 利用者共通AGENTS、projects/AGENTS、作業worktreeのAGENTS／PROJECT_MAP、Git内の研究概要全文と関連statusも確認した。旧snapshotの研究段階を全系列の最新状態と扱わない。

研究階層は「アルゴリズム改善系列の横断調査」。長q価格・T資格は関連するリソース評価系列。既存同目的の **`codex/pr-rpe-cost-refocus-survey-20261108`**（作業前local/remote tip `654b8501c1348cd0694daeefd19391df6b928354`、clean）を継続し、本directoryを追加した。前回3文書と固定科学成果は変更せず、PROJECT_MAPに今回への入口だけを追記する。旧調査の公開後STOPを、科学runの再開許可とは扱っていない。今回の明示依頼に基づく静的後続文書である。

既存三環境台帳・11月8日横断調査を再利用し、指定5成果とその後のC/D/O2/Pure-H結果を差分確認した。GitHubの関連branch検索・主要tip照会と公開固定資料の取得を行った。全remote全文／全raw／他SERVER非公開資産の網羅調査ではない。独立した詳細科学レビューも実施していない。

**今回の新科学計算、保存値の数値再解析、simulation、sampling、回路build/compile、合成、N再算定、価格最適化、新アルゴリズム実装は全て0。** 研究moduleをimportせず、資料・Git object・必要なSOURCE本文／ASTだけを読んだ。作成物は文書4点と既存navigationの小追記のみ。固定SOURCE/input/raw/result/grant/ledger/marker/契約/STOP、他worktree/process、共有環境は変更しない。

文書の検証は参照先・Git object・SOURCE symbolの存在、リンク、差分、数値引用の転記、公開SHA/tree/本文の照合に限定する。科学testsや過去runnerの再実行は不要であり、実施していない。remote確認の具体的SHAは公開完了時のチャットに示し、自己SHAを埋めるためだけの追加commitを作らない。

## 次の担当とmandatory STOP

**ユーザー／GPT**が、主資源、選ぶ候補（複数可）、必要な元H/誤差/統計条件、実装準備・科学検証の許可範囲を判断する。重要な詳細科学レビューを行う場合は、その開始許可と科学実行許可を別に扱う。

**Codex**は今回の文書をcommit/pushし、remote SHA・tree・公開本文を確認した時点で**mandatory STOP**。研究方向の正式採択、新算法実装、科学計算、詳細GPT科学レビューへ自動進行しない。今回の許可内で自動実行する次の科学作業なし。
