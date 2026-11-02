> 公開用projection。private内部パス、原入力・回路・運用記録、ローカル補助scriptは含まない。本文に記す照合件数は原調査の報告値であり、未公開原資料を再公開した意味ではない。

# B：保存winnerの登録探索domain境界監査

2026-10-19 JST。**技術監査完了、条件付き・未確定を保持してSTOP。新しいwinner選択、費用計算、再ランキング、scientific code実行は0。**

## 結論

固定commitの保存winner IDは、今回対象とした各比較domainにすべて登録されている。H4/H6/H8の新A16は疎な条件付き集合であり、p/q/r/Kの直積ではない。qだけの最大・最小、あるいは「中間精度だからdomain内部」という一括判定はできない。H4の強化比較は旧決定論8＋実測追加8の部分集合であり、登録打切り96全体の最良とは呼べない。

H6非相殺については、旧Dの保存された有限値・`unresolved=[]`をfull-domain確定と解釈できない。元の12候補は`eligible:null`、`UNRESOLVED_SIGNED_DECOMPOSITION`のまま存在する。最新新A統合とα感度は欠測を明示してfull-domain比較をnullにしている。本監査は旧結果を書き換えず、旧Dのwinnerへ「取得済みsubset内の保存ID」という属性を付けた。

**科学的な大域最適性、domainを広げた場合の改善、母平均winner、選択後保証は認定していない。** これらの限定を付けた既存ID・domain表は、新しい数値処理なしに論文準備へ使用できる。

## 出力と対象範囲

- [winner_boundary_flags.csv](winner_boundary_flags.csv)：762注記行。保存ID・条件・n・p/q/r/K・外側次数・直接実測/model分類・登録条件付きslice・q/r schedule端・欠測を対応付けた。
- [registered_candidate_domains.csv](registered_candidate_domains.csv)：参照した登録集合をIDと座標で明示。費用列なし。
- [candidate_endpoint_status.csv](candidate_endpoint_status.csv)：新Aの960保存endpointの状態とcohortを転記。費用列なし。
- [metadata_verification.json](metadata_verification.json)：ID結合・件数・処理境界の記録。
- [source_manifest.json](source_manifest.json)：実取得したcommit:path、SHA256、bytes、読取receipt。
- `source_verification_receipt.json`（原調査の非公開照合記録）：Git資料36 fileの保存copyと固定blobのbytes一致・manifest SHA一致を確認。
- `output_hashes.json`（原調査の非公開照合記録）：本B成果物のSHA256とbytes。
- `annotate_metadata.py`（原調査の非公開照合記録）：本監査専用の静的metadata結合処理。費用の最小化・費用ソート・費用算術・科学runner importを行わない。

|保存系列|保存比較レコード|注記行|対象条件|
|---|---:|---:|---|
|新A総費用固定64c85f7a…|120|360|H4/H6/H8、5ε、通常/非相殺、factor1/10、旧/強化PF各集合。新PR・旧PR・PFの3 ID|
|既存D固定fc6bd039…|120|240|同5ε、両bias、factor1/10、各2PF集合。旧PR・PFの2 ID|
|α感度固定343e0c59…|66|132|H6 5ε、H8 6ε（.000625を含む）、両bias、factor1、α=.01/.05/.10|
|H8 C_EPS保存panel|12|24|ε=.04/.025/.015/.000875/.00075/.000625、両bias、factor1、旧PR/PF|
|高次PRの別比較固定1fc58aa3…|6選択点|6|H4/H6、ε=.001、S2/Yoshida4/new_2、探索n8、q1/2/4/8の保存選択|

これはこの5系列の保存比較を網羅した表であり、repository全期間・全geometry・全T・全P感度等の全結果を網羅した表ではない。旧H6探索72/n8はdomain inventoryへ保全し、確認8/n32のwinnerへ混ぜていない。H4別rank10・geometry、H5/H7、別RPE系列は本表外。

## 取得・条件・来歴

実際に`git show`で取得した固定資料を読み、監査専用ディレクトリにそのbytesを保存した。新A結果は`64c85f7a833b5950ef2ccae623708b73081edcd6`の`comparisons.csv`と`new_pr_cost_points.csv`、旧D/A登録は`fc6bd039dc7171d5674ac04f0b3ad427f1b6495b`の`a_candidates.json`、`d_saved_comparison_summary.json`、各system A/D plan、H6保存candidate会計、H8明示登録を使った。

H4の元D登録planと実測subset結果は、fc6bd039…のevidence registryが直接指定する`2588bb1ff8a03120050fdaa67b9cd30739131bec`から実取得した。H6の旧/強化domainは固定`src/trotterlib/track_a_v1_contract.py`の登録規則と保存cellを照合した。H8は`references/h8_phase_c_registration_v1.json`の`old/additional/confirmation_cells`そのものを用いた。未取得資料を原本確認済みと扱っていない。

今回の主系列はlinear H-chain、隣接1Å、STO-3G、T=.8、P=0、保存Hamiltonian/state/fragment順、finite complex signal精度、通常signed／経験的非相殺、経験的allowanceである。H4はlegacy固定DF12、H6はtol-only1e-8 DF19、H8はDF28。新AのprefixはH4 p2/3/4、H6 p3/4/5、H8 p6/7/8。native symmetric-directional full Hadamard wrapper、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。H8 profileの`622ce4a8…`は原契約識別子であり、profile原本fileのbyte SHA確認とは表記しない。

新A_COST取得SOURCEはH4/H6が`31ef62a12407010e9c1a07f9b37facedb8dff5fe`、H8残112 trajectoryが`87f1f0d528478ad212ba98159db06eec3c78433e`（元代表16とseed導出は31ef62…）。新総費用保存SOURCEは`8601b921342fc1cc83780a830e1f65bf64fe9e93`。本監査はそれらを再実行していない。

## 実際の登録集合

|系・cohort|比較した登録内容|境界判定上の注意|
|---|---|---|
|H4旧PR8/n32|p3、q1/2/4はr4、q8はr2、K2/4、S2|n32はfixed anchorの選択・推定用。独立確認とは呼ばない。q8/r2はq/rの登録schedule端|
|H6旧PR8/n32|p5、q1/2/4はr4、q8はr2、K2/4、S2|独立確認cohortだが、そのunion内の保存winnerは選択後点比較。単一固定winnerの独立認定ではない|
|H8旧PR6/n32|p7、q2/4/8、r4、K2/4、S2|q1はこの確認unionにない。q2は登録下端で、物理的最小q=1とは異なる|
|H4新A16/n8|核12：p2/4×q1/4/8×r4×K2/4。追加4：p3q1r2K2、p3q4r8K4、p3q8r4K2/4|条件付き疎集合。旧PR8とのpooling・union内winner再選択は行わない|
|H6新A16/n8|核12：p3/4×q2/4/8×r4×K2/4。追加4：p5q4r8K4、p5q8r8K4、p5q16r2K2、p4q8r8K4|q16のr2とq4/8のr8は別登録点。p5を一律「上端q16」と扱わない|
|H8新A16/n8|核12：p6/8×q2/4/8×r4×K2/4。追加4：p7q4r8K4、p7q8r8K4、p7q16r4K4、p6q8r8K4|p7K4の登録scheduleはq4:r8、q8:r8、q16:r4。q16単独のr4固定scanではない|
|H4旧決定論8|full p12、q1/2/4/8、S2/S4|p12は物理的full-rank endpoint。prefix scanの下端/上端とは別|
|H4強化比較16|旧8＋直接実測8。追加はB0I p4q1 S2、p9q1/2/4 S2、p9q2/4 S4、B0 p9q2/4 S2|母登録の打切り96はB0/B0I×p2/3/4/6/9/11×q1/2/4/8×S2/S4。未実測88を含む全96最良はnull|
|H6旧32|B0/B0I p5/10/15 S2×q1/2/4/8、full p19 S2/S4×4q|旧p gridの疎性を維持|
|H6強化136|B0/B0I S2 p5..15×4q、S4 p11..15×4q、full p19 S2/S4×4q|S4 p11はS4の登録下端、S2の内部prefix。次数を混ぜたp境界は不可|
|H8旧32|B0/B0I p7/14/21 S2×4q、full p28 S2/S4×4q|full prefixと打切り上端p21を区別|
|H8強化184|B0/B0I S2 p7..21×4q、S4 p15..21×4q、full p28 S2/S4×4q|S4 p15はS4の登録下端。q16はこのPF登録にない|

## 境界の定義と、保存winnerへの属性付与

`prefix/q/r/K_slice_values`は、同一method・同一外側formulaかつ他の座標をすべて固定したとき、実登録に存在する値だけを列挙する。2値以上なら登録sliceの下端・内部・上端、1値なら`SINGLETON_NOT_AN_EDGE_TEST`。singletonを内部や最適と認定しない。外側次数は順序の違う方式として保存し、同一座標で登録された次数も別列に記す。

qによってrが変わる登録は、別途`fixed_p_K_registered_q_r_panel`を列挙し、`registered_schedule_q_position`でschedule panelの端を示す。これはrを固定したq効果ではない。物理的q1/r1/full-rank prefixは独立したflagであり、探索上限と区別する。H4 measured subsetについては、より大きい登録D96＋full8にも同じ属性を付け、取得subsetの端を親domain全体の端と混同しない。

以下は通常bias・factor1・強化決定論集合における**保存された**主panelのID（短縮表記p/q/r/K）。外側PRは全てS2。全bias・factor・cohort別の詳細はCSVにある。

|系|ε|新A n8 ID座標|旧PR n32 ID座標|保存PF ID・価格区分|
|---|---:|---|---|---|
|H4|.05|3/1/2/2|3/1/4/2|B0I p4q1 S2、直接実測|
|H4|.01|2/4/4/2|3/2/4/2|B0 p9q2 S2、直接実測|
|H4|.005|2/4/4/2|3/4/4/4|B0 p9q4 S2、直接実測|
|H4|.001|2/8/4/2|3/8/2/2|B0I p9q2 S4、直接実測|
|H4|.0005|2/8/4/2|3/8/2/2|B0I p9q4 S4、直接実測|
|H6|.05|4/2/4/2|5/1/4/4|B0I p7q1 S2、直接実測|
|H6|.01|4/4/4/2|5/2/4/2|B0 p11q2 S2、model|
|H6|.005|4/4/4/2|5/4/4/2|B0 p11q4 S2、model|
|H6|.001|4/8/4/2|5/8/2/2|B0I p11q2 S4、model|
|H6|.0005|4/8/4/2|5/8/2/2|B0I p11q4 S4、直接実測|
|H8|.05|8/2/4/4|7/2/4/2|B0I p11q1 S2、直接実測|
|H8|.01|8/4/4/2|7/4/4/4|B0 p15q4 S2、直接実測|
|H8|.005|8/4/4/2|7/4/4/4|B0 p15q4 S2、直接実測|
|H8|.001|6/8/4/4|7/8/4/4|B0 p15q8 S2、直接実測|
|H8|.0005|7/16/4/4|7/8/4/4|B0I p15q4 S4、直接実測|

静的に言える主な例：

- H8 .05の新A p8q2r4K4は、同一r/Kでp6/8の上端、同一p/r/Kでq2/4/8の下端、K2/4の上端。PF B0I p11q1 S2のq1は登録下端かつ物理的最小q。
- H8 .0005の新A p7q16r4K4は、p7K4のq/r schedule上端。一方、全座標を一つだけ動かすsliceではsingletonであり、r4固定のq4/8/16走査が保存されているわけではない。
- H8 .01/.005の新A p8q4r4K2はq方向では内部だが、p6/8の上端・K2/4の下端。レビューpr_2の「中間winnerはdomain内部」という一括記述は、この登録metadataからは成立しない。
- H6 .001/.0005のPF p11 S4は登録S4 prefix下端。S2 prefix p5..15の内部であることを理由にS4側も内部とは呼べない。
- H4 .001/.0005の旧PR p3q8r2は登録q/r schedule上端だが、r2固定のq sliceはsingleton。H4新A p2q8r4はp側下端かつq側上端。
- これらは登録内容の属性であり、境界外に改善点が存在する、あるいは境界が勝敗の原因だという新しい科学的結論ではない。

## 実測・model・cohortの分離

PFの価格区分は各保存比較レコードの`PF_cost_status`／`cost_class`をそのまま保持する。選択点が直接実測でも、比較集合の全候補実測や母平均最良を意味しない。H6主panelでは.01/.005/.001の選択PFにmodelが残る。H8主panelの通常factor1の選択PF5点は直接実測だが、強化184の未実測候補はmodelのまま。

α感度とC_EPSは、それぞれの固定入力時点の単価を保持する。同じH6 p7やH8 p15q8について、その後のD直接測定を遡及的に採用しない。従って同じIDでも、α/C_EPSではmodel、最新D/新Aでは直接実測という場合がある。ID同一性と価格来歴の同一性は別。

H4旧n32は選択・推定用fixed anchor。H6確認union8/n32・H8確認union6/n32は旧探索と別cohortだが、確認union内からのwinner選択があるため、選択後coverageや母平均winnerが認定されたとは呼ばない。新A各16/n8は探索標本の点比較であり、旧n32へのpooling・独立確認への格上げはしない。

固定`1fc58aa3…`の高次PF統合報告はH4/H6の**乱択PR方式**S2/Yoshida4/new_2の別実験で、強化決定論136/184集合とは異なる。p3/5・K4・外側1stepあたり32 short steps・q1/2/4/8、ε=.001、探索n8。保存選択は両系でS2 q8（登録上端）、Yoshida4 q2（q内部）、new_2 q1（登録下端かつ物理下端）。ordinary control/explicit XXPlusYY/number Phaseであり、主n32系列へ混ぜない。new_2は元8桁係数版であり、厳密4次係数版と同一視しない。

## 欠測・不適格・未確定

新A960 endpointの保存状態は有限標本点478、不適格162、会計欠測320。これは保存statusのmetadata件数であり、新しい科学集計ではない。不適格は保存精度会計上の状態、欠測は入力未保存であり、0費用に変換していない。

H6/H8新Aのfactor10は各160、計320 endpointで保存shotがない。winner座標なし/nullのまま保持した。factor1や通常biasから代用しない。H4新Aはfactor1/10の保存範囲がある。

H6非相殺はB0I p5/10/15×q1/2/4/8 S2の12候補にsigned分解がなく、α入力では12×5ε=60欠測（3αの出力180）。元会計には候補自体が登録され`eligible:null`で存在する。旧Dの有限subset値は使用可能な既存引用だが、「全登録32/136最良」「全domain比」へ転記してはいけない。最新新A比較の`PF_best_available_id`もfull-domain winnerではなく利用可能subsetのIDとして注記した。

最新新Aの`new_domain_complete/old_domain_complete`はPRだけの完全性ではなく、比較に必要なPF入力も含む保存フラグである。H6非相殺factor1では新A PRの保存候補点があっても両フラグfalseとなる。input完全性と科学的保証は区別する。

## 確認状態・論文用表記・残作業

|区分|判定|
|---|---|
|確認済み|対象保存winnerのID登録照合、cohort、各座標、条件付きslice、q/r schedule、価格来歴、欠測statusの静的対応|
|条件付き|登録domain内の有限標本点比較。H4は実測subset、H6/H8は候補にmodel残存。新Aは探索n8|
|未確認|探索外最適性、母平均winner、選択後保証、全候補直接実測、厳密bias証明、H8原profile bytes同一性|
|不成立な一般化|中間winnerが全座標でdomain内部／旧D非相殺の空unresolvedからfull-domain確定／高次PR別実験と強化決定論集合の同一視|

論文へ転記できるのは「保存された登録集合と費用来歴の下での有限標本点」「各coordinateまたは登録scheduleでの端」「未確定と不適格を区別した範囲」である。境界flagを一個のglobal-optimum警告に潰さず、p・q/r schedule・K・outer formulaとcohortの注記を併記する。

本Bの成果完成に追加科学承認は不要。未解決を解消する場合だけ、ユーザーが目的に応じて別承認を判断する。候補domain外への拡張、H6欠測signed成分の科学取得、新A独立確認、model点の直接compile等は今回実行していない。表のための単なる再ランキングやn32とのpoolingは必要作業ではない。

## 原本保護とSTOP

共有repositoryは読取専用のGit object参照として扱った。SOURCE、raw結果、grant、ledger、marker、既存STOP、他job・worktreeを編集せず、B専用領域だけに新規出力を作成した。既存削除・staged状態には触れていない。ネット公開・push・review起動・新科学計算は0。

共通AGENTSを明示読取し、後続ユーザーによる今回のsubagent許可を適用した。固定AGENTSのPROJECT_MAPを入口として参照し、研究概要の今回分担1051–1550行を読んだ（他範囲は主担当/他担当が分担）。これは独立科学reviewではなく、主担当の統合監査に対する作業分担である。

**B担当は完了・mandatory STOP。次の担当：Codex主担当がこの既存証拠注記をEと統合報告へ一度取り込む。B担当による次の科学作業なし。**
