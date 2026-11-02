# DF-native・Pauli-JW資源評価：証拠の統合案

2026-10-26。文書・既存資料の静的照合のみ。新規科学計算、保存値の数値再解析、候補選択、順位・比・SE計算、図の再描画、sampling、compile、詳細GPT/Claudeレビューは実施していない。中心claim・投稿方針・一論文／二論文の採択はユーザー判断とする。

推奨する中心課題は、**「ランダム化時間発展の測定込み資源は、保持部分・PF次数と刻み幅・RTEパラメータ・制御実装・比較候補集合によってどう変わるか」**である。Hamiltonian表現をまたぐ結果は、条件を揃えた範囲の実装事例として接続する。新アルゴリズムの発明や、一つの方式の普遍的優位を中心に置かない。

## 1. 成果物と証拠の読み方

- [EVIDENCE_MATRIX.csv](EVIDENCE_MATRIX.csv)：43行・30列。主張、採択状態、入力、T/ε/q/δ、分割/r/K、law/control/compiler、C/N/G、比較集合、選抜、n/SE、SOURCE、結果commit、掲載候補、追加計算の要否を対応付ける。
- [PAPER_OUTLINE.md](PAPER_OUTLINE.md)：一論文案を主案、二論文案を代替案として提示。
- [FIGURE_TABLE_PLAN.md](FIGURE_TABLE_PLAN.md)：原図・保存表の配置、caption、必要なら別許可となる数値作業を区別。

CSVの`claim_id`は既存CEMの対応IDを可能な限り再利用した。新しい行IDは参照用であり、新たな採択台帳ではない。`saved_fields_json`は保存行を文字列として転記したもので、複数の結果を新たな共通政策で再評価したデータセットではない。未記録は`unknown/not_verified`、非該当は`not_applicable`とする。exact Gの分数、保存小数、原文のSEを再計算しない。

`n`は古典的に生成・compileした価格trajectory数、`N`は解析的shot規則の会計値である。cosine/sineは同じtrajectoryのpaired価格で、2個の独立標本ではない。主指標は保存された`G=N_cos C_cos+N_sin C_sin`。状態準備を除くfull Hadamard wrapperのcompiled RZ数と、FT T/Toffoli・実測shot数・全energy/QPE/RPE費用は区別する。

### 原資料と採択状態

|区分|今回の扱い|
|---|---|
|既存正式採択|DFの固定独立P、共通588候補の点結果、標準Yoshida4固定独立比較について、既存CEM v3／ledger v5／paper manifestの限定状態を継承する。最終paper claimやevidence freezeの採択ではない。|
|現在支持される観測|DF長時間、Pauli最新比較、固定H4の実装比較。原報告の有限候補・cohort・仮定の範囲で引用する。今回の掲載・中心claim採択はしない。|
|不支持／負の結果|H4 DF10粗精度の方向非再現、長時間m5の独立比較1件逆転、緩い精度でのS2低費用、H2/H4の大sでの非改善もそのまま掲載対象とする。|
|保留・未保証|全域最適性、無条件母平均G、全law compiled残差、表現一般の優位、系サイズ則。未観測event費用や未取得matched Hybridはunknown/nullで、0ではない。|

旧原稿v0.4は草案である。旧H4 DF12 geometry系列の採択済み実行と、草案の中心claimの正式採択を混同しない。最新結果を根拠に旧主表・Figure 2・CEM・raw・STOPを改訂していない。

## 2. 四つの証拠ブロック

### A. 従来DF-native：強い対照と固定独立確認

[H468統合][A]の主表はlinear1Å/STO-3G、tol-only1e-8のDF10/19/28、T=.8、P=0、通常signed biasと保存allowance条件での6比較である。全行のPRはS2。PF対照は直接compileした固定B0/B0+Iで、一部は高次公式を含む。Qiskit1.3.0・rz/sx/x/cx・opt1・seed17、symmetric_directional、scalar一回を維持する。

|固定比較|ε|PR L_D/q/r/K|確認n|保存方向と低い点費用|
|---|---:|---|---:|---|
|H4 DF10 P_1|.05|3/1/4/2|1108|NOT_REPRODUCED、PF低費用：正式採用された非再現|
|H4 DF10 P_2|.0005|2/8/2/4|78|REPRODUCED、PR低費用|
|H6 DF19 P_1|.05|4/2/4/2|48|REPRODUCED、PF低費用：再現はPR勝利という意味ではない|
|H6 DF19 P_2|.0005|4/8/4/2|104|REPRODUCED、PR低費用|
|H8 DF28 P_1|.05|8/2/4/4|48|POINT_REPRODUCED、PR低費用|
|H8 DF28 P_2|.0005|7/16/4/4|48|POINT_REPRODUCED、PR低費用|

exact G、保存SE、SOURCE、PF IDはCSVの既存CE ID行と[固定主表JSON][A-table]に対応する。H8の2行はMechanism32と同一標本の別名参照で、新しい独立証拠を2組追加しない。H8詳細監査は公開receiptから再検証できない状態を保持する。H4 dense-eighとH6/H8 eigshの状態生成差、共通protocolの科学gate未充足も保持し、COMMON_CORE_READYやサイズ則へ昇格しない。

[共通588候補][PF]は別の比較である。H2/H4/H6、tol-only DF3/10/19、T=.8、S2/Yoshida4/m5_best、q1/2/4/8のFull/B0/B0I対、保存R_outer32/K4のPR探索価格を用いる。CEMで限定採用済みの保存結果は、H2は全5精度PF低費用、H4/H6はε=.001/.0005でPR低費用、他3精度でPF低費用。PRの独立Pや長時間の新確認標本とは異なり、ここでの最良PRは保存探索集合の点最良である。

[旧H4 geometry原稿][GEO]はlegacy固定DF12、6距離、ordinary controlの別系列。B0+I追加後の保存点最小はB0+I397点、B2が1,415点。ε=.05では.70/.80/1.40ÅでB0+I低費用、ε=.005では6距離ともB2低費用。これは同じn32での選抜と推定を含む。DF10の新独立Pへ移送しない。

**追加された知見と役割：** full PFだけでなく、既知scalarを保持する低費用のB0+I、formula/q/prefixを選んだ強い対照が必要である。Aは「ランダム化を有利と仮定しない比較」の基礎を担う。全問題におけるPRの不利／有利を結論しない。

### B. 長時間・高次DF：大きい刻み幅を使う利点と代償

[長時間REPORT][B]はH4/H6、DF10/19、T=.8/1.2/1.6/2、S2/Yoshida4/m5_best。基本qは1/2/4/8/16にT1.2の6、T2の10を追加。S2に限りT1.6/2でq32を追加した。PR prefixはH4=1/2/3/5/7/9、H6=3/4/5/7/11/15、K2/4/6。R_outerはS2=4/8/16/32/64、Y4=8/16/32/64、m5=16/32/64。signed pieceへ整数配分し、trajectory全体のevent数はq R_outerとなる。

11,632構成のsignalと58,160精度endpoint、642候補の価格会計が保存されている。全signal候補の全価格を取得したわけではない。費用の最良は、適格で価格を取得した部分集合に限る。旧signal294・shot1,440・互換価格203候補を再利用しており、T=.8の重複を新たな独立実験として数えない。H6は保存weighted-Hermitian completionのHamiltonianを使用する。

ε=.0005では、H4のT1.2/1.6/2とH6の全Tで、PF・PRともm5 q1が取得域の点最良だった。H4 T.8では両familyともS2 q8が点最良。T2ではδ2を使うm5に対し、保存公式別最良S2はq32であり、REPORTの保存m5/S2 G比はH4 PF=.3439／PR=.3249、H6 PF=.3868／PR=.2475。これは各公式が異なるqやprefix等を選んだ比較で、次数だけの因果効果ではない。

厳しい精度のT2の保存PR構成はH4 prefix3/R32/K6、H6 prefix7/R16/K6。同じm5 q1でも最良r/K・分割点は同じでなく、一律の推奨r/K則を作らない。緩いε=.05では両系・全T・両familyでS2が取得域の点最良であった。固定δ=.2の保存図では、高次化のpiece増による単価・normalization負担も見える。

独立価格確認は探索後に39候補・n計1,295を固定し、確認後の候補再選択やtop-up、探索とのpoolingはしていない。59比較中58で点方向再現。PF対PR33/33、Y4 PR対S2 PR8/8、m5 PR対S2 PR17/18。例外はH4 T1.6・ε=.005で、探索のm5優位が確認では逆転し、Bonferroni感度区間は0を含む。39候補の価格を複数εへ使っているため59独立実験ではない。

**Aからの追加と役割：** T=.8の同じ結果を言い換えるのではなく、時間増大・高次公式・粗いδの使用可能性とC/N/Gの釣合いを加える。signal適格な最小qと価格最適qは別である。理論保証、母平均G、全登録域最適性は未確立。**L_D=0は含まれないので、DFの完全ランダムB3との比較が完了したとは書かない。**

### C. Pauli-JW：Hybrid対純RTEと比較集合依存性

[最新総合資料][C]はlinear1Å/STO-3G、T=.8、S2、Pauli非identity項H2=14/H4=184/H6=918。各分子内で保存raw ray・exact norm・full-H referenceを共有し、central-rotation controlとfull wrappersの価格を比較する。純RTEは非identity項のL_D=0境界であり、constant phaseは保持する。

|保存最新版|観測|候補・確認標本|原結果|
|---|---|---|---|
|H2|5精度すべて純RTE低G|ε=.05/.01はq1/r4/K2、他3精度q2/r2/K4。いずれもs4、n384。保存Hybrid n96。|[H2追加確認][C2]|
|H4|ε=.05だけHybrid低G、他4精度は純RTE低G|RTE s64、n384。Hybrid n48。Kは精度別の保存確認集合minであり、初期nomineeとは異なる。|[event拡張][C46]|
|H6|5精度すべて純RTE低G|q8/r64/K2、s512、n48。保存Hybrid n48。|[event拡張][C46]|

「14/15」は保存された3分子×5精度行の点比較の内訳であり、15独立実験、統計的勝利数、母平均の保証ではない。CSVのC_*行は既存latest15の候補ID、n、exact G、SE、判定を文字列のまま転記した。PF/B0は既存の対照値への参照で、最新RTEを入れて4familyを今回再ランキングしていない。

H2は旧n96のRTE s2対HybridではHybrid低Gだったが、有望だった未確認s4候補の確認集合を充実させると逆転した。s64/128への拡張自体は改善しなかった。H6は旧s≤64の比較ではHybrid低G、新s512の独立n48では純RTE低Gへ逆転。旧計算の誤りではなく、異なる候補集合の比較である。H4ではs128/256のC増がN節約を上回り、s64の保存結果が維持された。

H6の保存K2内訳は、s64：B20.6408／N_total60,852,202,283／C_cos424.1042／G約2.580767e13、s512：B1.475668／N_total305,364,103／C_cos3114.979／G約9.512028e11。s1024はB1.214851でもC_cos6239.5となり、探索G約1.291288e12。隣接点のn8探索とs512のn48確認を同列の独立確認値として扱わない。

**Bとの関係と役割：** Bは非ゼロDF prefix内のPF次数・時間依存性、CはPauli内で完全ランダム境界も含めた保持範囲・event数の依存性である。純RTE一般の優位や、両表現でHybrid不要という主張は支持しない。H2確認は逐次延長、確認後集合minを含む。仮想ρ=0、母平均費用tail、選抜補正、実compiled精度の限界を残す。

### D. 固定H4：表現と実装の交絡を見せる対照

[固定H4整理][D]は共通MO積分と保存raw ray、Pauli184項・prefix27、DF tol-only DF10・one-body＋fragment0、S2/T.8/q8/r4/K4のcase study。入力の保存full-H差8.462237051574415e-15は数値一致診断であり、共通厳密Hamiltonianへの無条件精度転送の証明ではない。

|固定実装の保存価格|PF RZ/shot|PR平均RZ/shot（n256）|
|---|---:|---:|
|Pauli generic|255500|9321.1015625|
|Pauli central|15028|1140.8046875|
|DF-native|135355|23678.44140625|

同じPauli bodyのD2ではPF cosine255500→15028、PR index0は9652→1145 RZ。表現を変えずにcontrol構成とcompiler応答だけでも大きく変わる対照である。index0価格とn256平均は別の値である。Pauli27とDF1、同じ「50%」やr/Kを同等な分割・lawとみなさない。PauliとDFの同じindexをpairedとして扱わない。

仮想ρ=0の保存Gとguard box内の順位安定性は、実ρがguard内に入る証拠とは異なる。6件の局所Arb包含は旧H4のPF両軸と選択PR0/PR1だけで、新探索・H2/H6・PR全law・DFへ転用しない。初回D1 fingerprint FAIL、後続SOURCE基準Bの9構成取得、R2-C PROOF_INCOMPLETEは別の来歴として残す。

**履歴から回収した補足：** [限定scalar結果][D-Cmean]では、SOURCE count前提とIID統計モデルの下で、named Pauli-central PRのuniform C上界12391/12392 RZが、DFの保存n256からの下界包含[19741.473302966152,19741.473302966153]を下回る十分条件が両軸で成立していた。少なくとも95%同時coverageという保存解釈は、compiler正常終了時のcountモデル、全allowed入力の終了、Rust SOURCEと実wheelの対応などの未証明前提を伴う。これは**条件付き母平均Cだけ**の結果で、N/G・全lawρ・表現一般の母平均優位へ拡張しない。「あらゆる母平均C比較が未実施」とも書かない。

**役割：** A～Cの差をHamiltonian表現単独に帰属できない理由を、固定body対照で具体化する。条件付きCの補足は限定理論／統計事例としてSupplementが適切であり、統合論文の主要claimへ自動昇格させない。

## 3. 比較可能性の境界

|横断する比較|共有できるもの|保持すべき差／禁止する解釈|
|---|---|---|
|A固定TとB長時間|対応する保存tol-only H4/H6入力、wrapper費用単位、既存kernel|q/prefix/r/K域、数値allowance、cohort・PF分母が異なる。旧固定結果の置換ではない。|
|旧H4 geometryと新H4|baseline強化という問い|DF12対DF10、geometry、state/control/shot政策の差。n32と独立Pをpoolしない。|
|PF対PRとHybrid対純RTE|C/N/G会計と選抜への注意|前者は決定論対確率的、後者は確率的family内の保持範囲比較。勝敗を同一命題にしない。|
|DFとPauli|同じ条件を持つfixed H4に限った入力対応診断、同じcompiled命令単位|DF fragment／one-body対Pauli項、係数順、tail law、normalization、Gaussian reuse、control、reference転送が交絡。絶対Gから表現一般を順位付けしない。|
|同じsやr/K|登録パラメータ名|DF R_outerはsigned piece配分を含む。Pauliのs=q*rと一律に同じlaw・compiled回路にはならない。Kは最大偶数event次数で、PF次数とは別。|
|数値包含と実compiled保証|reference/meanに対する保存u、局所ρ|仮想ρ=0は実ρ=0ではない。有限PRNGのIIDはモデル仮定。formal certificate未受理を勝手に解消しない。|

全系列の結果を合算した平均改善率、統合winner、分子サイズ則は作らない。適格性・単価・総費用・厳密保証を別に記載する。

## 4. 主張候補と論文化に不足するもの

|候補|現時点の証拠|掲載するなら必要な限定|
|---|---|---|
|強い決定論的対照が費用比較を変える|Aの採用済み固定比較、B0+I、共通588候補|登録域・精度・cohort付き。PR一般の優位／非優位ではない。|
|長時間・厳しい精度ではm5の粗いδが低いGと両立する|Bの探索点最良と固定独立比較|取得価格部分集合、経験的allowance。例外1件と緩精度S2も記載。|
|Pauliでは確認範囲・event上限を変えるとHybrid/RTEの観測順位が変わる|Cの最新15行と旧履歴|14/15は条件付き保存点比較。確認後選抜、逐次観察、未探索Hybridを明記。|
|control合成は同じbodyでもcompiled価格を変える|D2とn256同一Pauli trajectory対照|control＋compiler応答。各passの一意寄与や表現だけの効果ではない。|

**A：限定的な比較研究の報告に必要なもの。** 本4文書によるSOURCE・条件・採択状態の対応付け、図captionでのlegacy／探索／確認の区別、negative resultと未取得値の明示が必要である。更新科学レビュー原本は未取得なので、その正確な採択文言を引用したい場合だけ原本確認が必要。H8の監査公開方針はユーザー判断を残す。これは追加科学計算を必須にする欠陥ではない。現在の限定した主張について、必須の新規科学計算は確認していない。

**B：強い主張の場合のみ必要なもの。** 全候補最適性には比較域の価格coverage等、無条件母平均Gにはlaw全体の費用boundsと適切な選抜・統計処理、実compiled精度保証には該当lawのρ転送、表現一般の優位には交絡を解く設計、DF B3を含む最適性にはL_D=0の別検証、全QPE/RPEには多時間・状態準備・信頼配分・FTモデルが必要。今回は計画の採択・実行をしない。

**C：任意の発展。** S2 q上限、T細分化、geometry、DF B3、rare-event、compiler pass分解など。いずれも限定ベンチマーク論文化の前提には置かない。旧Figure2を新P cohortへ数値更新するなら別の保存値再解析になるが、旧図と独立主表を別配置すれば今回不要である。

## 5. 来歴・取得範囲・運用状態

指定4系列の最終文書、既存CEM v3・ledger v5を指す[Track Aデータ索引][INDEX]、主表、Figure2更新案、原稿v0.4、保存CSV/JSONを参照した。科学runnerはimport／実行していない。CSV生成には既存の文書専用`integrate_pauli_h2_h4_h6_saved_results.py`の`one`（固定キーの一意照合）と`csvsave`を再利用した。科学kernel、元成果物、既存manifest、旧STOP、mainは変更していない。

Pauli履歴はユーザーの指摘に従い、ローカル索引で確認した3セッション`01a1644b-d6e7-7c20-8c11-55d00d674de3`（nonDF）、`01a171f3-ecf8-7940-920e-22c297dd5ab3`（nonDF_B0）、`01a17261-f1bb-7692-ad28-ff7395c27195`（nonDF_B0_H2,6）の最終応答を保存先の探索に利用した。チャット要約を数値の正本にせず、上記D-Cmeanを含む固定Gitの報告へ戻って確認した。無関係なセッションの全件読解は行っていない。

`Pauli_DF_PR_PF_更新科学レビュー_2026-10-26.md`は既知のプロジェクト上位・添付ファイル名検索では取得できなかった。**原本未読**である。添付指示が伝える「現行Pauli有限ベンチマークは追加計算なしでまとめてよい」「実装依存評価として統合を推奨」という要旨は尊重するが、原本読了・正式な主要claim採択とは表現しない。過去の全レビューを再読・再評価していない。

GitHub連携で[A]、[B]、[C]、[INDEX]、[D-Cmean]の指定commit/pathとblob identityを実取得し、ローカル固定Git objectを参照した。その他の原表・CEM・図の参照先は固定Git objectの存在を静的確認する。これをrawの新しい再現実験、外部レビュー、全GitHub内容の再監査とは呼ばない。元のPauli資料が記すmanifest/validator不整合も、本作業で科学statusを変更せず保持する。

文書の技術確認では、CSV43行×30列とID一意性、Pauli15保存行・DF独立6保存行・長時間の8保存セルと例外行のliteral一致、既存CEM採択文言の不変、固定commit/path44組とMarkdown参照を確認した。原Figure2A・H6 event図・固定H4 control図も目視し、PF白抜きmodel、legacy注記、2SE表示をcaption案へ反映した。変更対象はこの4成果物だけで、既存科学ファイルや既存索引は変更していない。これらは文書・構造検査であり、数値性能や科学的保証の新しい検証ではない。

次の担当はユーザー：推奨中心課題、Main/Supplementary配置、一論文か二論文かを判断する。Codexは4成果物の静的確認・専用branch公開とSHA照合後mandatory STOPし、レビュー・科学計算・論文本文執筆へ自動移行しない。

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/docs/research/track_a_h468_paper_evidence_integration_20261019.md
[A-table]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json
[PF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/90f1cd55a0d963953696bd0a27d9d041ff939bb7/docs/research/track_a_h246_pf_common_final_20261022.md
[GEO]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/288d2206d689a9c24516844cba2100c2d9845a32/docs/manuscripts/track_a_resource_study_geometry_native_v0_4.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md
[C2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/docs/h2_pauli_rte_validation_20261026.md
[C46]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03efd057a7e77c40ae7e90b9c77885d6606e793b/docs/non_df_h4_comparison_synthesis_20261025.md
[D-Cmean]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d4f6ff7f1df085f292d36c36e460354df8a740/docs/non_df_h4_pr_mean_cost_scalar_execution_20261025.md
[INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/731b3d6f3572c6aa7950a579b68251fec9a5218b/docs/data_collection/2026-10-24_track_a_research_data/README.md

## 2026-10-27追加：DF-nativeのR_outer境界比較

この節は元の4証拠ブロックへの追加であり、上の歴史的記述・正式採択・論文方針を置換しない。[追加証拠15行と報告](docs/data_collection/2026-10-27_pr_boundary_addendum/REPORT.md)、[EVIDENCE_ADDENDUM.csv](docs/data_collection/2026-10-27_pr_boundary_addendum/EVIDENCE_ADDENDUM.csv)に保存行を対応付け、EVIDENCE_MATRIXの別ID `BND_*` 7項目から参照する。既存43行、CEM v3、ledger v5、paper manifest、主表・原稿・原図は保持する。追加項目の正式claim採択は行っていない。

境界結果は[公開固定結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/202d25eb34be365aaa6638c6353e25f51acebe42/analysis/pr_boundary_execution_20261027/REPORT.md)による。linear H4/H6、隣接1Å、STO-3G、tol-only1e-8 DF rank10/19、H6保存weighted-Hermitian completion、T2/δ2、m5_bestの11 signed piece、ABS_TIME_PER_OUTER。H4 prefix3/q1/K2、H6 prefix7/q1/K6、P0、元finite-RTE・full Hadamard wrapper・compiler・経験的allowanceに条件付く。

主ε=.0005の独立確認では、H4 R16は固定R64よりGが+19.98337%、R128も+11.19767%で、保存±2 combined-SE診断はいずれもHIGHER。R16はC低下でN増を補えず、R128はnormalization/Nの低下でC増を補えなかった。Nは保存bias/u/headroomも含むため、normalizationだけの効果や単価の外挿とはしない。この条件ではRの増減がともに非改善となったが、R64の大域最適性やR32との母平均順位は未確定。R32は探索n9だけで、R64 C800と同じ独立確認水準へ昇格させない。

H6 R11は固定R16より確認点Gが−0.17209%だが、旧R16のSEも含む診断は **INCONCLUSIVE**。母平均優位や確定的改善を主張しない。R11は11-piece allocationでの実装上の下限であり、R8を同一契約で実行可能とはしない。小さな点差を統計的不確かさが制限した下限方向の検証として位置付ける。

元長時間探索33/40、追加7候補の拡張探索集合33/40、今回のT2限定E8/10・C8/10を別証拠として保持する。今回は保存結果の転記のみであり、40セル再ランキングやcohort poolingを行っていない。境界外の固定候補を検証しても全探索域最適性は保証されない。

H8はlinear1Å/STO-3G/tol-only DF28、T.8、S2/prefix7/q16/r4/K4、δ.05/P0の別系列。ε=.001の探索n8とε=.0005の独立n48価格は存在する一方、.001の正確なA_signal原記録は未回収である。n48価格による.001の新G/SEと新しいq8対q16独立比較は成立していない。欠測を費用0・不適格・旧結果の誤りと解釈せず、旧希少RTE研究の正式終了・STOPを維持する。

Pauli-JWのevent数・control比較は既存の別区分を維持し、今回のDF-native数値をその順位へ移さない。未採択B3結果は含めない。Section4のパラメータ感度とDiscussionへ短く参照し、保存C/N/GのSupplementary境界比較表を配置する案に留める。新しい中心claim、章立て、一論文／二論文方針を採択しない。

今回の資料統合では新規科学計算・数値再解析・fit・再描画・詳細科学レビューを実施していない。完全SOURCE/rawはローカル保全という公開記録を引用し、原本の再照合済みとはしない。公開結果を参照する文書差分だけを公開し、静的検査とremote照合後mandatory STOP。
