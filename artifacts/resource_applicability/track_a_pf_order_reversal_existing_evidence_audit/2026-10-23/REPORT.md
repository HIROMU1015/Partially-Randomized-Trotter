# Partially Randomized Trotter × PF次数：既存証拠の技術監査

2026-10-23 JST。担当：Codex一次担当のみ。**保存順位の逆転候補4条件、同方向11条件。比較Cの科学的成立・因果機構は未認定。作業完了後mandatory STOP。**

対象はlinear H2/H4/H6、隣接1.00 Å、STO-3G、tol-only 1e-8の保存DF rank3/10/19、T=.8、ε=.05/.01/.005/.001/.0005。主比較はS2とYoshida4、m5_bestとnew_2は補助である。保存された選択・適格性・費用を引用し、SOURCEとmetadataを静的照合した。新しいsignal、sampling、simulation、回路build/compile、科学的数値再解析、shot再計算、総費用再会計、再最適化、候補再ランキング、数学的証明、科学的GO、詳細科学レビューは行っていない。

## 1. 判定の範囲と取得状態

**A**＝旧保存候補集合のFull PF内部S2/Yoshida4比較。**B**＝同じouter32探索系列内のPR S2/Yoshida4比較。**C**＝AでS2低費用・BでYoshida4低費用となる組合せを、共通条件を照合して研究上採択できるか、という問いである。

ここでの`REVERSAL_CANDIDATE`は、既存選択点の保存総費用の大小が指定の逆方向になる、という記録である。identity適合性と科学的認定を別列にした。Hamiltonian/state/snapshot/compiler設定の12 hash一致や、本監査の技術照合PASSを科学的証明へ読み替えない。

指定された6固定commitはすべてローカルGit objectから取得した。重要13ファイルはGitHub connectorでも固定refで照合した。6文書は全文取得、7データは先頭行指定でGitHub blob SHAを取得し、全文は同じSHAのローカル固定blobを読んだ。データ全文をGitHubから転送したとは記載しない。[取得台帳](acquisition_manifest.json)、[GitHub取得receipt](github_receipts.json)、[全SOURCE索引](SOURCE_INDEX.md)にfull commit/path/blob SHA1/SHA256/取得方式を保存した。branch HEADや最新版mainへの置換はない。

| 符号 | 固定commit | 主資料・取得内容 | 位置付け |
|---|---|---|---|
| F7 | `5d53698534a1f325c8b7ed7be1bca6fa5192a77d` | [7系列図README][F7]、figure_data.json、notes、input_sources、provenance、group_selection_audit、STOP | Aの採用点とB0/B0+I補助点。旧PR n8はBの対照に流用しない |
| FH | `ee160395e87ed6d46cd4993572457afd0aa18590` | [高次PR図README][FH]、notes、mapping、compatibility_audit、coverage_audit、figure_data、provenance、STOP | outer32の保存選択点。旧PFと直接結合せず別パネルという制約を保持 |
| E | `8613b3ecda2030a677b1690fe7aa3da23dec4873` | [保存費用報告][E]、formula_comparison_30、cost_decomposition_60、candidate_precision_costs_1000、evaluation、input_provenance、end_audit | Bの保存費用・shot・単価・bias・normalizationの正本 |
| IC | `3eca46148070ba4d4b0f5815ef45943407cc959e` | [独立確認報告][IC]、fixed_evaluation/results、入力inventory、再構築end_audit | ε=.0005のPR内部の固定3比較。C全体の独立確認ではない |
| N2 | `495496c4e8e027b6ebd52e6835e5cafe3518502c` | [new_2報告][N2] | H4はlegacy rank12・ordinary control。新H4 rank10や厳密4次と混ぜない |
| TC | `427480e4f1a61385cbffa724930a2221ed9e5e1d` | [理論/compiled報告][TC] | 理論/compiled間のPR/PF方向不一致。PF次数の逆転とは別現象 |

## 2. 比較A/B/C：系×精度の対応

費用は保存compiled RZ総費用。括弧内は保存q。小数は保存表示値を転記し、新しい費用・比率は計算していない。Y4=Yoshida4。全行の科学的比較Cは未認定。

| 系 | ε | A Full S2（q） | A Full Y4（q） | B PR S2（q） | B PR Y4（q） | 保存順位分類 |
|---|---:|---:|---:|---:|---:|---|
| H2 | 0.05 | 5533984.0 (q1) | 14681766.0 (q1) | 15459953.072916666 (q1) | 23003550.166666668 (q1) | SAME_DIRECTION |
| H2 | 0.01 | 188156192.0 (q1) | 386028390.0 (q1) | 525506285.6354167 (q1) | 604863983.8166667 (q1) | SAME_DIRECTION |
| H2 | 0.005 | 1170534707.0 (q2) | 1647882698.0 (q1) | 3312556005.887324 (q2) | 2582019872.883333 (q1) | REVERSAL_CANDIDATE |
| H2 | 0.001 | 59673629128.0 (q4) | 74234356638.0 (q2) | 171015619467.24243 (q4) | 111066028016.32353 (q2) | REVERSAL_CANDIDATE |
| H2 | 0.0005 | 305695076994.0 (q4) | 309561246534.0 (q2) | 875894111002.6364 (q4) | 463148646634.41174 (q2) | REVERSAL_CANDIDATE |
| H4 | 0.05 | 136962465.0 (q1) | 327895754.0 (q1) | 125783311.46666667 (q1) | 245529408.7037037 (q1) | SAME_DIRECTION |
| H4 | 0.01 | 7172627004.0 (q2) | 12308798738.0 (q1) | 6541332950.097902 (q2) | 9191659129.851852 (q1) | SAME_DIRECTION |
| H4 | 0.005 | 46827942770.0 (q4) | 63284490880.0 (q2) | 42505736689.5 (q4) | 41925721153.454544 (q2) | REVERSAL_CANDIDATE |
| H4 | 0.001 | 2430958906210.0 (q8) | 2019789418240.0 (q2) | 2188164571311.0 (q8) | 1336462215872.0 (q2) | SAME_DIRECTION |
| H4 | 0.0005 | 12546400952880.0 (q8) | 12375920238292.0 (q4) | 11285212407470.0 (q8) | 7663967924818.333 (q4) | SAME_DIRECTION |
| H6 | 0.05 | 877049350.0 (q1) | 1790822626.0 (q1) | 577051364.5209581 (q1) | 1484049528.0 (q1) | SAME_DIRECTION |
| H6 | 0.01 | 48918110661.0 (q2) | 81303861484.0 (q2) | 31110568741.636364 (q2) | 47936571158.72727 (q2) | SAME_DIRECTION |
| H6 | 0.005 | 273676251900.0 (q4) | 341213530734.0 (q2) | 171774205601.39395 (q4) | 201145569863.45456 (q2) | SAME_DIRECTION |
| H6 | 0.001 | 14746438809458.0 (q8) | 13298858970625.0 (q2) | 9289932763968.213 (q8) | 7825857757922.728 (q2) | SAME_DIRECTION |
| H6 | 0.0005 | 93588391394613.0 (q8) | 65737570406690.0 (q4) | 58894875680499.97 (q8) | 32773435584223.242 (q4) | SAME_DIRECTION |

全75採用recordのID・prefix・rank・δ・K・outer32・n・shot・bias・u・normalization・単価・exact総費用・元commit/locatorは [evidence_rows_75.csv](evidence_rows_75.csv)。
15条件の順位とidentity適合性を別列にした表は [comparison_ABC_15.csv](comparison_ABC_15.csv)。m5採用15行も75行表に別formulaとして含む。
B0/B0+Iの60枠は [B0_B0I_auxiliary_60.csv](B0_B0I_auxiliary_60.csv) へ分離。H2のε=.01/.005/.001/.0005の16枠は元のnullを保持した。

AのS4ラベルは、実使用providerの`ax2a_control_plan.deterministic_control_plan`が`product_formula._get_w_list('4th')`を三段S2へ合成する経路である。m5_bestやnew_2を意味しない。旧nativeの合成順、負の中央時間、scalar最終一回を静的確認した。Bの`4th`も`h46_m5_plan.weights`の三段Yoshida4である。ただし、実回路のbinary64 word全体とruntimeまでA/B間で一致したとはしていない。

保存最良の探索範囲も限定される。H2はD70、H4はD190のFull PF表示群ごとの保存値。各Full PF公式のqは1/2/4/8/16。H6は旧fine-grid S2＋truncated S4の登録136候補に対するALL_SAVED458単価政策から、保存ordinalを参照した表示点である。Bは系ごとの固定prefix1/3/5、各公式q=1/2/4/8、K4、ABS_TIME_PER_OUTER=32の4q内の保存選択で、pilot・旧n8・確認cohortをpoolしない。探索域が違うことを隠して全候補最適と呼ばず、qやprefixが違うことだけで比較不能ともしていない。

### H2の成立状況

H2はε=.005/.001/.0005で候補となる。特にε=.0005は次の4採用点である。値は元保存行の転記で、以下の比較から新しい比を計算していない。

| 比較 | 保存候補ID | prefix / rank | q / δ | 保存shot cos / sin | 保存RZ単価 cos=sin | 保存総費用 |
|---|---|---|---|---|---:|---:|
| A S2 | `B1_p3_q4_2nd` | 3 / 3 | 4 / 1/5 | 117378315 / 106246467 | 1367 | 305695076994 |
| A Yoshida4 | `B1_p3_q2_4th` | 3 / 3 | 2 / 2/5 | 76747409 / 75520789 | 2033 | 309561246534 |
| B S2 | `H2O32_c7ec590a32a859bc` | 1 / 3 | 4 / 1/5 | 117441887 / 106313566 | 3914.5151515151515（標本平均） | 9634835221029/11 |
| B Yoshida4 | `H2O32_8a2ecdd0294ffb74` | 1 / 3 | 2 / 2/5 | 79644174 / 78371520 | 2931.029411764706（標本平均） | 7873526992785/17 |

Aの元結果は`829d697ae7905b3ea90a61da9d2d4da15d8b757a`の`.../D_completed_run02/result.json`、`/accounting/rows/324`と`/accounting/rows/319`。compile取得SOURCEは`cf99e4452df96291d7e9d37d76d2e91be5cbc413`、signalは元SOURCE `de72e5d4e48177734c0ddd79dffac14222410658`から受け入れており、D再開時にsignalを新評価した結果ではない。Bの取得SOURCEは`4ec26921c6872ed6e8667f167074e1e405cf1a3f`、結果公開commitは`a05f4ed19e87e1eb65f7fc346385b5b4eae69845`、保存会計SOURCEは`9a77d2b0a0b87e246e85f085b37f5923c18e27bb`、保存費用結果commitはEである。各B行のevaluation JSON pointerと元CSV行番号は75行表に残した。

ここではAもBもq4→q2である。したがって、**最良qが減ったという事実だけでは、A/B間で方向が変わる理由を説明しきれない。** 保存単価はAで1367→2033、Bで3914.515…→2931.029…と異なる方向に変わり、shotも各保存bias・normalizationに応じて異なる。この事実と、分割後の回路構成・tail費用の寄与を実験的に切り分けたことは別である。

SOURCEから読める構造上の説明候補は、**outer32では公式を三段へ変えても、1 outerの全pieceに割り振るRTE short stepsの合計は32に固定される**点である。Full PFでは合成段数の増加が全fragmentの決定論的回路へ掛かる。一方PRでは小さいprefixの決定論的段数が増える一方、tail側は全piece合計32という政策を保ち、採用qが減ればouter反復も減る。この構造はH2の保存単価の方向差を調べる具体的な仮説を与える。ただしpieceごとの負時間・normalization・event分布・compilerも関わるため、これだけで原因を正式認定しない。固定outer32政策を離れた一般的なPR高次化の主張にも拡張しない。

Eに保存済みのB Yoshida4/S2分解は、ε=.0005でshot比0.706198、有効単価比0.748759、総費用比0.528772（報告表示）、削減率47.123%である。**これは会計上の説明であり、独立な因果機構の証明ではない。** A/Bを跨ぐPR/Full PF比は本監査では新計算していない。

### H4・H6と独立確認

H4の候補はε=.005だけである。AはS2/q4、BはYoshida4/q2が低費用となる。ただしBの保存差は報告表示1.365%と小さく、保存SE・選択効果を無視して母平均順位を認定できない。ε=.05/.01は両側S2低費用、.001/.0005は両側Yoshida4低費用で、Cの条件を満たさない。

H6では今回の5精度に指定方向の候補がない。.05/.01/.005は両側S2低費用、.001/.0005は両側Yoshida4低費用である。ただし旧Full PFの単価は、全10採用行とも`RETAINED_BASELINE_EVIDENCE`・`old_profile_pending_48=true`・`CONDITIONAL_PROFILE_RUNTIME`を保持する。S4側には`S4_COMPOSITION_HYPOTHESIS`もある。数値の大小が読めることと、その旧native/runtimeの完全同一性を認めることは別である。

m5_bestは別公式として75行表に15採用点を収録した。PR3公式内の最良は、H2が.05/.01でS2、残り3精度でYoshida4。H4は.05/.01でS2、.005でYoshida4、.001/.0005でm5。H6は.05/.01/.005でS2、.001/.0005でm5。H4/H6のm5改善をYoshida4の改善と呼ばない。

ICではε=.0005のH2 S2/q4対Yoshida4/q2、H4/H6 S2/q8対m5/q1の改善方向を、それぞれ46.51%・53.11%・63.13%（保存点表示）として再現している。118 trajectory・236軸はPR内部の別cohort確認で、旧Full PFを含むA/B結合や全15条件を再確認したものではない。H4 .005の近接Yoshida4比較はこの確認に含まれない。H2/H4は再生成した科学配列hashが旧保存値と一致する一方、NPZ容器SHAは異なる。新namespaceを用いたという独立性と、未回収の旧全seedとの完全非衝突は区別する。希少tail・母平均順位・外部再現・別担当レビューの保証も付けない。

## 3. SOURCE・実装・会計条件のidentity監査

`MATCH`は指定欄での静的照合、`METHOD_SPECIFIC`は方式に由来する正当な自由度、`SOURCE_DIFFERENCE`は実装差、`ACCOUNTING_POLICY_DIFFERENCE`は費用政策の差、`METADATA_GAP`は保存証拠の不足を表す。`CONDITION_MISMATCH`を現在の15条件へ一律には付けない。

| 項目 | 確認したA/Bの内容 | 技術判定・残る範囲 |
|---|---|---|
| Hamiltonian/state/snapshot | H2/H4/H6すべてA表示点とB取得bindingで同じSHA256 | `MATCH_METADATA`。全12欄は[hash表](identity_hashes_12.csv)。hash一致だけで完全比較を認定しない |
| 物理系・DF | linear 1Å/STO-3G/tol-only1e-8、rank3/10/19。保存順prefix。rankは観測値で最終rank指定ではない | 対象15条件では条件一致。旧new_2のH4 rank12を混ぜれば真の`CONDITION_MISMATCH`になる |
| 状態・reference | 同じstate hash、T=.8のfull-H複素signal。Aは保存referenceを利用、Bは保存native系列のreference | 基準の意味は一致。A/Bを通したreference実体・数値経路全体の同一bytes認定までは未実施。新reference計算はしない |
| 精度・費用範囲 | εは有限時間複素signal、各軸ε/√2、α_axis=.025、P=0、cos/sin両軸のfull Hadamard wrapper compiled RZ | `MATCH_SEMANTICS`。energy精度、全RPE/QPE、T count、noise、実測量子shotへ拡張しない |
| shot関数 | 実使用旧providerの`u_aware_shots`、`_number`、`axis_headroom`とEの凍結D0実装をAST比較 | 3関数のAST一致。[静的照合](shot_static_checks.json)。単に未使用の同名ファイルを比較したのではない |
| bias | Aはfull PFのsigned total biasの軸絶対値。Bはouter PFとfinite RTEのsigned totalの軸絶対値 | 同じ要求signalに対する意味は対応。誤差構成は`METHOD_SPECIFIC`。H6旧会計行にはbias値・logB自体がなく、headroomから逆算せずnullを保持 |
| allowance・margin | Aはreference/path/roundoff等から適応的に作るempirical uとmargin。Bは保存u=1e-9、logB margin=1e-12 | **`ACCOUNTING_POLICY_DIFFERENCE`を特定。** 倍率1が同じでも同一政策とはいえない。例えばH2 .0005のA S2 u=1e-11、A Y4 u=1.0942358130705557e-11。差が順位に影響するかは未評価 |
| headroom・適格性 | 正のheadroom、u/headroom≤.01、有限安全域の整数ceil、LOG_DOMAIN_ONLYは費用null | 主要意味は対応。75採用点は保存ELIGIBLEを照合。新たな適格性・shotは生成しない |
| normalization | A Full PFの保存logB=0。Bは各signed pieceのfinite-RTE normalizationを累積 | 値の差は`METHOD_SPECIFIC`。負時間はnormalizationで相殺しない。経験的marginを厳密上界とはしない |
| PF係数・合成順 | Aの`4th`はproduct_formula→三段S2、Bはh46_m5_plan→三段S2。どちらも正/負/正。m5は11段 | 数学的familyの誤対応は解消。binary64演算順・native word・compilerまで同じとはしていない |
| scalar・control・wrapper | 元scalar＋必要なtail identityを一回、symmetric_directional、cos/sin wrapper。旧nativeも新合成も最後にscalar位相 | 意図する意味は対応。prefix/r/qの差は正当。旧new_2のordinary control費用はこの監査へ混ぜない |
| native/provider | 旧H2公開bindingはprovider `217c971d...`。private旧specのhashを実読取照合し、取得済み関連provider12ファイルのsealed SHAと固定Git bytesが一致 | 旧H2のprovider出典は実体metadataでも確認。新nativeはS2 pieceごとの合成adapter。A/B全回路のfingerprint・preparation closure・負時間処理を通した完全結合は未認定 |
| compiler設定とSOURCE | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/routingなし、設定hash `1239304c...` | 設定一致と実装一致を分離。新H2は`H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1`。同じ設定hashでも`SOURCE_DIFFERENCE`がある |
| H2 compiler guard | SOURCEはtranslatorのgate/angle/wire streamの同一性を毎pass要求し、global_phase帳簿をguardする。state fittingやpostcompile位相補正ではない | guardはRZ差の原因だと認定しない。Aの全決定論点とB取得軸を跨ぐ完全な受入結合が別途必要 |
| runtime | 旧H2 private specとruntime manifestを既知の公開locatorからread-only取得し、封印hash一致を確認。旧Python SHAも保存 | [private metadata監査](private_metadata_audit.json)。新outer32取得時の完全runtime/provider hash inventoryは公開投影からlocator未取得。package version一致だけでbinary一致を主張しない |
| H6単価・会計SOURCE | 旧元会計とALL_SAVED458置換表の候補ID/ε/保存shot/保存費用を結合。選択会計SOURCEは`19f2c5a9...` | これは実compile SOURCEの一括代替ではない。保持単価はreplacement_SOURCE=null、profile/runtime同一性未認定を継承 |
| 探索・選択 | Aは保存表示group/既存ordinal、Bは各formula4q内の保存best。Bのnは候補ごとに33〜255 | `METHOD_SPECIFIC`かつscope制約。固定採用点の順位照合のみ。新ランキング・全格子母平均最適性・選択後補正はなし |

旧shot kernelの関数一致、Full PF S4のfamily対応、RZ計数の定義、scalar-onceという意味論までは、今回のread-only確認で具体化できた。一方、allowance政策は単なるmetadata欠落ではなく実際のSOURCE上の違いである。異なるアルゴリズムには異なるbias・normalization・必要shotが正当に生じ得るため、数値を一律に揃えればよいわけではない。何を共通の誤差保証・経験的政策として許容するかは研究判断であり、本監査では採択していない。

## 4. 原因候補と既存証拠

| 原因候補 | 保存証拠・SOURCE | 説明できる範囲／未証明事項 |
|---|---|---|
| 1. bias変化 | 75行表のsigned_biasとheadroom、Eの1000行表。H2 .0005のcos biasはA S2 +8.03043881647536e-5→Y4 −1.5628128782974038e-5、B S2 +8.024693260599847e-5→Y4 −1.5624966641980542e-5 | 採用点のbias差は存在。qも変わるため次数単独の因果効果ではない |
| 2. 最良q変化 | A/Bの保存qとE formula_comparison | H2 strictでは両側q4→2。小さいq利用の説明材料だが、逆転の十分な説明ではない。固定qの未保存反事実は生成しない |
| 3. shot変化 | 保存N_cos/N_sin/N_totalとEの分解表 | H2 strictのB保存shot比0.706198。bias・u・normalizationが同時に入る会計結果であり単一機構の証明ではない |
| 4. 有効回路単価変化 | 保存単価と`effective_unit_RZ`、E decomposition | H2ではAの単価上昇とB採用点の単価低下が逆方向。B有効単価比0.748759はshot重み付きで、cos/sin単純平均への置換ではない |
| 5. 負時間とRTE絶対時間 | h46_m5_planのsigned scheduleとallocation、h46_m5_nativeのpieceごとのlogB累積、provider rte.pyの`_paired_order_weight` | 重み生成は`abs(time)`を使い、符号はnative時間発展へ保持。outer32は1 outer全pieceで32、総反復32q。負時間がnormalizationを相殺する説明は不可。係数和や絶対時間を今回新数値評価していない |
| 6. PF/RTE誤差の組合せ | E SOURCEの`signal_bias`、取得SOURCEの`outer_PF_or_discard_bias`/`finite_RTE_bias`/`total_signed_bias`、別modeのNONCANCELLATION | signed closureと別の非相殺診断が保存され、E報告では15条件の最良formula/qは同じ。相殺が一般に無関係である、厳密bias上界がある、という結論にはならない |
| 7. 合成段数・回路構造・compiler | S2一段/Y4三段/m5十一段、native compose、RZ countがtranspile後`rz`数であるSOURCE、H2 guard | 固定qで段数が増える構造と、採用q減少・tailイベント回路の影響が同時に入る。compiler寄与だけを分離した保存実験はない |
| 8. 分割・DF表現 | 固定rank3/10/19、PR prefix1/3/5、Full PF全fragment、signed generation順 | 全fragment PFとprefix＋tailの構造差は明確。DF/prefixを変える因果介入は保存比較の範囲外。H4 rank12/new2からrank10の機構を推定しない |

全30通常signedの保存分解は[saved_cost_decomposition_usual_30.csv](saved_cost_decomposition_usual_30.csv)へそのまま転記した。ε=.0005のH4/H6でm5が有利なこと、H6 Yoshida4/q4の有効単価比は1.0141と増えてもshot比0.5487で改善するというEの記述も、単価短縮が常に必要という説明を支持しない。これらは独立の因果実験ではない。

## 5. 公式・図群・スライドの混同防止

| 名前 | 保存定義とSOURCE | 本報告での扱い |
|---|---|---|
| S2 / `2nd` | 一段の対称2次step | A/Bの基準。旧PR n8と新outer32 S2は別取得系列 |
| Yoshida4 / `4th` | 3個のS2を正/負/正で合成。product_formula.pyとh46_m5_plan.py | 主比較の4次family。旧図S4の対応をSOURCEで確認 |
| m5_best / `4th(m5_best)` | 固定11段binary64列、`PF_M5_BEST_20D64C2_BINARY64_V1`。references/h46_m5_best_formula_v1.json | Yoshida4と別公式。上流PF commit `20d64c2fea1ed2d2188777df4f3d962ee94df486`の関数SOURCEはこの保存referenceに引用済み。上流repository全文を今回取得したとはしない。新たな厳密代数次数証明なし |
| new_2 / `4th(new_2)` | 保存5係数 [.40899193,.42008729,-.65815844,.42008729,.40899193]。N2の既存8桁decimal監査 | 厳密4次条件の立方和0を満たさない。N2の保存値7.346136322761508e-9を引用するだけで再計算しない。H4 legacy12・ordinary control・代表q/n8という別scope |

**高次PR図の「図群A/B/C」と、本監査の「比較A/B/C」は別の名前空間である。** 図群CはH4/H6のsame-native m5決定論部分集合との比較であって、本監査の逆転Cではない。same-nativeの決定論S2対照が保存・結合されておらず、m5比較だけではS2/Yoshida4逆転を認定できない。

会議の`PR資源評価 — bento_slides.pdf`および10/22発表用構成案の原本は今回未取得。取得済みとはしない。ユーザー提示の対応情報に基づき、本編8＝F7 ε=.05、本編9＝F7 ε=.0005、本編10＝FH outer32内部比較、補足S4＝TC、補足S8＝FH same-native m5部分集合と整理した。PDF実ページと図の一致を検査済みとはしない。

TCのH4では、固定PR/PF pairで修正後理論比0.8716321313、保存compiled比1.0393357766が記載される。しかしH4はlegacy rank12、固定7候補の理論/compiledモデル比較であり、今回のrank10・PF次数による方向変化とは別である。ここからCの成立や原因を移送しない。

## 6. 不足証拠と次の最小作業（提案のみ）

今回、既存SOURCEの軽微な照合で解消できる点は実施した。残りは、未取得の原本metadata、共通政策の研究判断、又は新しい数値評価を必要とする点である。

| 未解決事項 | 今回わかったこと | 最小の次作業・担当 | 実行承認 |
|---|---|---|---|
| Cをどの強さで主張するか | 4候補の保存上の方向は確定して読める。完全条件統一の科学的成立は未採択 | **ユーザー**：限定された保存点の観測として扱うか、共通政策に結合したCを必要とするかを判断 | 科学的採択はユーザー判断。今回の技術PASSでは代替しない |
| 新outer32のruntime/provider closure・guard受入receipt | 旧H2の既知private metadataは取得・hash照合済み。新取得の公開summaryは完全manifestの実体locatorを含まない | **ユーザー**：取得SOURCE 4ec26921/5a448d94に対応するprivate config/runtime/provider/guard receiptの正確なlocatorを提供できる場合は提供。**Codex**：提供後に元byte・hash・SOURCEだけをread-only照合 | metadata照合自体に科学実行grantは不要。広域projects/tmp検索や新compileで埋めない |
| allowance政策の整合 | kernelは一致、u/margin設定は違う。数値差の費用順位への影響は未評価 | **ユーザー**：共通の経験的allowance政策又は異なる政策を正当に比較できる解釈を指定。**Codex**：必要ならH2 ε=.0005の固定4採用点だけで再会計する案を作る | 新shot/費用評価は別途明示承認が必要。今回は設計採択・契約/grant発行・評価とも0 |
| H2 old/fullとnew/nativeの受入結合 | scalar/control/wrapper/公式familyは対応。旧PFと新PRの全word/native/runtime受入は未結合 | **Codex**：まず既存fingerprint・preparation・compiler receiptを結合。記録が足りなければH2固定決定論2回路×cos/sinの検証を必要最小候補として提示 | 既存metadata照合は科学grant不要。新build/compile/simulationが必要と確定した場合だけ別途承認。件数予約なし |
| H4 ε=.005の近接順位 | 保存Bの1.365%差はC候補。独立確認対象外 | **ユーザー**：H2結合の後にこの近接例が研究判断を変えるかを判断。必要なら**Codex**がS2/q4対Y4/q2の固定2候補確認を提案 | 新sampling/compile/統計評価は別途承認。nやseedは今回設計・発行していない |
| H6旧profile/runtime留保 | 5条件は同方向で、今回Cの候補ではない。full S4側のcomposition留保は保持 | **Codex**：この逆転の調査だけを目的にH6全格子compileを要求しない。既存単価の条件付き位置付けを維持 | 今回H6追加計算の提案・実行なし。より強い一般化を求める場合のみ別途範囲判断 |
| 原因の因果分離 | shot×有効単価は保存会計分解。PF/tail/normalization/compilerの独立効果は未識別 | **ユーザー**：Cの比較条件を定めた後、どの機構が研究判断に必要かを選ぶ | 新理論導出・非自明証明・因果実験・詳細科学レビューは未承認。自動進行しない |

最小の出発点はH2 ε=.0005であり、全3系×5精度の資源評価体系を作り直す提案ではない。固定4採用点だけを結合しても全候補最良性の新認定にはならない。ここで求めるのが固定点の逆転か、共通政策の下での各方式内bestの逆転かに応じて、その後の範囲をユーザーが判断する。

## 7. 技術検証・原本保護・終了

[verification.json](verification.json)に、75採用recordと元locator/保存exact費用の一致、15条件の保存大小の照合、補助60枠と16 nullの保持、GitHub blob SHA13件、入力manifest由来SHA34件、3 shot関数AST、全取得blob SHAとローカルコピーの一致、private metadata原本の前後hashを記録する。これはCodex一次担当の技術自己確認であり、別担当の独立レビューではない。

既存のF7/FH STOP、H2の元observer STOPとD完了、保存値評価の準備STOP・終了監査、独立確認の科学call0で止まった初回技術起動の記録を歴史として保持する。STOPを解除・書換えず、使用済みgrantを再使用しない。実科学SOURCE・結果・grant・ledger・marker・保全snapshot・旧研究資料は書いていない。監査ディレクトリに作成したコピーと新報告だけが今回の書込みである。すべての他worktreeのHEAD/statusを凍結する要求や、他chat/processへの停止・変更は行っていない。

private旧spec/runtimeはその内容を実行せず、既知のexact locatorだけを読んだ。新config/grant・科学出力域の予約は作成していない。上記のmetadata欠落を、実際には取得していないprivate結果を取得済みと称して埋めていない。全worktreeや全private原本を横断して保全検証したとは主張しない。

今回のcommit・pushは0。追加simulation/sampling/build/compile/shot計算/再会計/新ランキング・新最適化・科学claim採択・詳細科学レビュー・次campaignは0。既存の独立確認で実行済みの数値評価と、今回の静的監査を区別する。

**完了：比較A/B/Cの保存証拠と不足事項を追跡可能に整理した。次の担当はユーザーによる上表の研究上の要否・政策判断、必要なら欠落metadataの正確なlocator提供。Codexはmandatory STOPし、追加科学計算・詳細レビュー・次campaignへ進まない。**

[F7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5d53698534a1f325c8b7ed7be1bca6fa5192a77d/docs/progress/track_a/figures/all_methods_hchain_20261022/README.md
[FH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ee160395e87ed6d46cd4993572457afd0aa18590/docs/progress/track_a/figures/all_methods_hchain_with_higher_pr_20261022/README.md
[E]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8613b3ecda2030a677b1690fe7aa3da23dec4873/docs/research/track_a_h246_saved_cost_results_20261021.md
[IC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eca46148070ba4d4b0f5815ef45943407cc959e/docs/research/track_a_h246_reconstructed_confirmation_results_20261022.md
[N2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/495496c4e8e027b6ebd52e6835e5cafe3518502c/docs/research/track_a_h46_new2_native_validation_20261018.md
[TC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/427480e4f1a61385cbffa724930a2221ed9e5e1d/docs/research/track_a_h46_theory_compiled_paper_evidence_20261018.md
