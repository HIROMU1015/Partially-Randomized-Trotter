# 研究C：H4最新結果と論文claimの対応

2026-11-06 JST。取得済み証拠の統合・作図・原稿準備である。独立新規性、研究方針、論文化GOの正式認定や重要科学レビューは行っていない。既存の[文献・新規性整理](research_c_related_work_novelty_evidence_20261105.md)と[論文構成案](research_c_paper_outline_figure_plan_20261105.md)を引き継ぎ、文献調査をやり直していない。

## 1. 固定資料と成果物への入口

| 役割 | 固定参照・区別 |
|---|---|
| 最新結果の閲覧commit | [`417fa459b823509be5c2010c5836e58955088282`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/417fa459b823509be5c2010c5836e58955088282)。[実行報告](research_c_integrated_h4_100_results_20261106.md)と[geometry完了JSON](../../artifacts/research_c_integrated/2026-11-06-picklefix/development/run01/result.json)を優先する。 |
| 科学SOURCE | [manifest](../../artifacts/research_c_integrated/2026-11-06-picklefix/technical/SOURCE.json) SHA256 `8113f23e66c5eca3c09a8ed1c91fc58c01eeaa91d0c07850b685a20c816539eb`。閲覧commitや今回の作図コードとは別の識別子。 |
| 固定契約 | [contract](../../artifacts/research_c_integrated/2026-11-06-picklefix/technical/contract.json) SHA256 `fd111609d72d2a2d4ded7415b3e622c0b3a8951f55591809794ef3e30c867f11`。ファイル内の歴史的statusを今回の許可・科学的成功に読み替えない。 |
| 分子入力 | SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。候補ごとのJ/UとD分母は別途束縛する。 |
| 保存要約 | [summary.json](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/summary.json)、[comparisons.tsv](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/comparisons.tsv)、[final_costs.tsv](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/final_costs.tsv)、[deterministic_costs.tsv](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/deterministic_costs.tsv)、[optimization_checkpoints.tsv](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/optimization_checkpoints.tsv)、[selection_rows.tsv](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/selection_rows.tsv)。 |
| 原STOP | [INTEGRATED_STOP.json](../../artifacts/research_c_integrated/2026-11-06-picklefix/INTEGRATED_STOP.json)は`FAILED_MANDATORY_STOP`。1.00 Å完了と1.15 Å初期化の不一致による統合停止を分ける。原STOPを改訂しない。 |
| 文献・構成の出発点 | [`1a588f5491bce394ecf039ee4d2de57581e2e0b1`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/1a588f5491bce394ecf039ee4d2de57581e2e0b1)の既存2冊。L1〜L10、C1〜C5、E1〜E5のIDを維持する。 |
| 今回の原稿・図表 | [Results作業稿](research_c_h4_100_results_draft_20261106.md)、[Discussionと判断引渡し](research_c_h4_100_discussion_handoff_20261106.md)、[図表・caption・生成手順](../../artifacts/research_c_h4_paper/2026-11-06/README.md)。 |

GitHub上の要約・必要入力・結果JSONと、実行サーバーに保全された全rawは同じ配布範囲ではない。QPY・dense配列・標本別詳細ログをGitHubから全取得できるとは扱わない。今回参照した数値は公開済みJSON/TSVの保存値であり、NPZ/QPYをdecodeしていない。

## 2. 取得した証拠の範囲

対象は直鎖H4・隣接1.00 Å/STO-3G、number CDF rank8、Nα=Nβ=2/sector36、元frame順LD2。partial ordered symmetric S2、s1、δ=h=.1、K0、m0..12。候補固有own-DF ground oracle・状態準備除外、P0/fenced serial会計である。物理DF予算は.00019875 Ha、目標CA=.00159 Ha、RPE failure=.01。δ掃引や他分子への検証ではない。

3初期値C0/P1/P2、初期正則化寄与.03/.1/.3、PR/ENTRYの18 trajectoryと16/64受理更新の36 checkpointを取得した。17本は既存完了結果を再利用し、1本を新SOURCEで実行した。全36 checkpointのDF予算はPASS、J一次基準はMET、U一次基準はNOT_MET。等しい受理更新予算は古典FLOP・inner iteration・wall timeの等しさを意味しない。到達した物理誤差もarm間で厳密には等しくない。

選択は各適格(candidate,r)のnative32だけで行い、共通r32では3強度から、公平retuningでは双方同じ3強度×3rから選ぶ。選択後の22ペアに独立final64を用いた。108 policy中90が適格、18は不適格、selection2,880件/final1,408件、同J/U D19候補、成立比較10/未成立2である。これらは保存集計の引用であり、今回再集計・再ランキング・新統計推定していない。

C0・64更新の保存値は、共通r32でPR/ENTRY=.4891799803453215（51.08%低い点値）、公平retuningで.5453060241385032（45.47%低い点値）。選択strengthはそれぞれPR/ENTRY=.1/.1、.1/.03、rは32/32、16/64である。元の4主比較familyに基づく98.75%近似Welch差区間を維持し、1.15 Å終了を理由に信頼水準を変更していない。

## 3. C1〜C5の更新対応

| 既存claim候補 | 今回の直接証拠 | 以前より支持が増した範囲 | 成立していない主張・負結果 | 判断として残すこと |
|---|---|---|---|---|
| **C1：写像後tailを用いるDF目的** | 固定SOURCEのmapped nonidentity Z/ZZ tail目的を維持し、近接ENTRYに同等のstrength/r選択機会を与えたfinal比較。Fig.1/2、T1。 | 数理的非同値の説明に、有限手順下で追加の条件付き費用利益が残る観測を接続できる。新たな数学的主張ではない。 | 一般signed J、frame別U、number→Pauli写像、正則化、tail²自体の発明ではない。最適収束後の目的固有能力は未分離。 | 既知の正則化classへの特定tail適用が独立した学術的貢献として十分か。 |
| **C2：物理予算内の下流改善** | C0/64の両regimeの総RZと保存98.75%差区間、全36 DF admission、独立final64。Fig.2/4、T1/2。 | 単一起点・単一強度・native32の以前の比較から、固定grid内の選択を含む検証へ進んだ。fair retuning後も45.47%差が残る。 | common admission thresholdであり同じ到達誤差・同時一次収束ではない。actual CA、全law、実装誤差の保証はない。 | この限定モデルでの改善の重要性と、論文で採る主張の強さ。 |
| **C3：shots/Wとkernel単価の競合** | 共通r32ではPRのkernel単価が高いがWが小さい。retunedではPRのshots/Wが少し大きい一方kernel単価は小さい。Fig.4、T2/3。 | 単価とshotsの交換を両regimeで示し、λだけの評価からshell込み総会計まで追える。 | 新しい因果ablation・λからの普遍的予測則ではない。PF/RTE・Gaussian回転・r・strengthが同時に異なる。 | どの機構説明を観察、モデル上の依存、検証済み因果効果として書き分けるか。 |
| **C4：同J/U上のPR/強D比較** | C0/64のr16 PR/D=.8910179075198119、保存個別95%比区間[.874508163853835,.9075276511857888]。P1/P2も各自のDを分母とする。Fig.5。 | 新しい選定済み候補上で、候補固有の固定強Dに対する低い点推定と副比較区間がある。 | 全決定論PF政策の最適解との比較ではない。旧update005/r128の不利やr32未解決は撤回されない。 | 同J/U固定対照の射程と、主比較・副比較・歴史的観測の配置。 |
| **C5：頑健性・公平retuning・移送** | 公平retuningの3初期値×2予算の全6比較、選択γ/r、未成立2セルの保存理由。Fig.3、S1/2。 | 「未検証」を、**H4/1.00 Å内の3固定初期値・2有限予算・事前grid選択後の観測**へ更新できる。 | 全strengthの独立final検証、初期値分布全体への推論、他geometry/分子への移送はない。1.15 Åの科学的成否は未評価。 | C5の移送部分を本文中心から外し、case studyとしてどの一般性を採るか。 |

中心claimの候補は、**「固定PR-tailの写像後massを用いたDF目的を、近接ENTRY対照・同等の事前retuning機会・独立価格評価に接続し、H4/1.00 Åの登録有限政策で条件付き総RZの追加利益を観測した」**である。目的の既知部分を認めたうえでの方法適用と有限検証という位置付けであり、正式採択・独立新規性認定ではない。

## 4. 先行研究との対応を維持する

文献の詳細・確認箇所は既存整理§2〜4へ集約する。本追補はその調査範囲を広げていない。

| 既存文献ID | 既知として引き継ぐ部分 | 今回の結果が与える追加の対応 | 今回比較していないもの |
|---|---|---|---|
| L5 CDF / L1 RC-DF | 一般対称core、frame別U、正則化、nested最適化、signed行和を含むPauli写像。 | 元順LD2の固定tail辞書に対するmapped mass目的を、共通物理予算の近接ENTRY対照と下流費用で検証した。 | RC-DF原solver・強度tuningを再現した競争、RC-DF class全体への優越。 |
| L2 **Unitary Weight Concentration** | PR向け係数集中、soft deterministic countとsoft tail mass²、資源との接続。 | Cはframe別J/Uの近似fitと固定frame maskを用いる。今回のfair retuningはC内のr/strength選択。 | UWCのglobal OO/BLISS/soft境界との同条件比較。UWCの誤った展開名や旧prose-only表は使わない。 |
| L3 PR原論文、L9 RTE基盤 | PR＋DF、RTE・減衰・shots/gates交換、basis回転費用。 | 具体的なDF目的と近接対照の有限endpointを、候補固有のnative価格まで接続した観測。 | PR原理やshots×回路価格の発明、一般定理から現在の浮動小数点実装全law資格への自動移転。 |
| L4 SCDF / L7 FRO等 | shift、一般係数・回転変数、異なるL1対象、PFと回転資源の交換。 | 固定κ・固定head/tail・mapped atom normという操作対象を明示できる。 | 表現族の一般包含、FT ToffoliとH4 native RZの横断ランキング。 |

ENTRYはこの研究のweighted entrywise L1二乗対照であって、RC-DFやUWCの完全実装ではない。今回の差を「RC-DF/UWCに45%勝った」と書かない。一般signed J・frame別U・写像を発明したという主張へ戻さない。

## 5. 負結果・未成立・未資格を残す

P1/ENTRYの共通r32は両予算・全3strengthでm12の`INFEASIBLE_PHASE`となり、比較が成立しなかった。r64では適格だが、r32の欠測値を0や無限費用に置かず、PR勝利とも数えない。native未実行の18不適格policyについて全標本のnative correctnessを確認したとは主張しない。

過去のE3（update005/r128）のPR/D点比1.97018542062、E5（同update005/r32・独立640）の符号/精度未解決、E4（別update064/r32・独立256）のnominal支持は、既存整理§5.3から別系列として残す。今回の22 finalペアとpoolせず、一本の改善曲線・一般効果へまとめない。tail-P4、CDF7、surrogate/switch移送等の過去の負結果は同整理§5.4を参照し、今回やり直さない。

1.15 Åでは初期復元の整合性FAIL後に停止し、ユーザーが追加作業不要と決めた。修復・保存配列の再解析・retry/resume・代替移送は行わない。**移送成功でも移送失敗でもなく、移送性能を結論できる結果がない。** 本文草稿の中心は1.00 Å内の検証へ限定し、旧構成の移送図は未取得として外す候補にする。

近似Welch/Student-t区間は固定N・IIDモデル内の名目的推論で、validated coverageではない。J/U同時一次収束、大域最適、全law合成/実装誤差、actual chemical accuracy、準備費用込みend-to-end、一般化、独立新規性は未確立のままである。

## 6. 作業範囲と次の判断

再利用したのは2冊の文献・構成案、既存E1〜E5の整理、417fa459bの要約・保存JSON/TSV、固定目的・会計コードの定義である。既存汎用DF作図scriptは最適化比較・比率集計を伴うため呼び出さず、保存欄の選択と表示だけの小さい専用rendererを追加した。図の軸単位換算・丸め・レイアウト以外に、新規統計・比率計算・回帰・再ランキングはない。

図表索引に元commit、SOURCE、candidate/r、データpath、既存区間種別を保全した。Results・Discussionは取得済みの観測と既存数理に限定する。投稿先と最終言語は未固定で、日本語本文・英語図ラベルは今回の作業形式にすぎない。

次は**ユーザー／GPT**が、中心claimの採否、文献との差分の重要性、H4限定case studyとしての十分性、未資格事項をどの強さで扱うかを判断する。追加計算が必須という結論を今回の完成条件にしない。重要科学レビュー・正式GO・投稿は開始せず、commit/push・公開照合後は**MANDATORY STOP**する。
