# H4/H6長時間PF・PR：未取得compiled費用の調査

2026-10-26調査開始、2026-10-27 JST保存。基準commit `155529e20a8fac39e00786c40b57830b71203c4f`。

**最小Gを改善し得る未取得候補は残っている。PF/PRの勝敗ではH4・T=.8・ε=.05のPRを最優先で確認する価値がある。現在の33/40は変更しない。** 保存価格だけでは全登録域の最小性を証明できない。以下は追加取得の選定用解析であり、新しい実測費用・母平均優位・正式な研究方針の採択ではない。

## 固定条件と証拠範囲

Linear H4/H6、隣接1.0 Å、STO-3G、tol-only DF threshold 1e-8、保存rank10/19・生成順prefix。H6は元のweighted-Hermitian completionを含む保存Hamiltonianを維持する。元のstate・full-H reference・scalar、symmetric-directional control、actual adjoint、測定付きfull Hadamard wrapper、P=0を固定。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

T=.8/1.2/1.6/2、ε=.05/.01/.005/.001/.0005。S2、標準Yoshida4、m5_best（S2 signed piece数1/3/11）。H4 prefix=1/2/3/5/7/9、Full10、H6 prefix=3/4/5/7/11/15、Full19。q・R_outer・Kは保存endpointに実在するものだけを読む。δ=T/qであり、S2限定q32追加域を含む保存域のδ窓は.05〜2。Yoshida4/m5のq32等は追加しない。allowance floor1e-9、元margin、u/headroom≤.01、α_axis=.025、元のanalytic shot数をそのまま使い、再適格化しない。数値はこの共通条件と各行のprefix/q/δ/R/Kに限定する。

根拠は[固定REPORT](../../REPORT.md)、[固定成果物](../../artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/)、そのscientific/completion provenanceと価格受入監査。これは単一有限時間signalの費用であり、energy/QPE/RPEの最終総費用ではない。

正式共通運用指示はユーザーが本作業中に追加したルートの `research_project_instructions_unified_2026-10-25.md` を適用した（SHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`）。元研究ファイル・共通索引・科学SOURCE・raw・grant/ledger/marker/STOPは変更していない。今回の解析・提案はこの独立ディレクトリに保存し、元の最小値や正本を更新していない。B3、他branchの整理、入力再生成は対象外。

## Coverageと価格重複

11,632物理候補×5精度＝58,160 endpoint。適格39,157、INELIGIBLE 18,924、UNDETERMINED 79。後二者を区別し、今回の取得候補にはしない。

|family|登録distinct|適格distinct|適格・価格ありdistinct|適格・価格なしdistinct|価格あり適格endpoint|価格なし適格endpoint|
|---|---:|---:|---:|---:|---:|---:|
|Full PF|136|127|73|54|285|241|
|B0|816|305|122|183|414|693|
|B0+I|816|351|125|226|441|712|
|PR|9,864|9,100|310|8,790|1,224|35,147|
|計|11,632|9,883|630|9,253|2,364|36,793|

価格642候補のうち12候補は今回の5精度で適格endpointを持たない。PF側の価格なしは463候補・1,646 endpoint。CSVのB0IがB0+Iに対応する。coverage.csvはsystem/T/ε/method/formula別480行で、精度をまたいでdistinct欄を加算しない。

既存 `saved_assets()` の旧価格430候補のうち、今回の登録域に入る203候補は**全て既存642価格に含まれ、両軸価格和はexact一致**。旧m5の別保存96軸カタログも照合し、登録域と一致する80軸は全て既存価格と一致した。新旧の独立確認価格も既存価格IDの部分集合であり、未取得候補を追加で埋めなかった。既存の探索・確認cohortを混合して価格を更新していない。

したがって、この受入済み公開価格群に関して再取得すべき欠損はない。特に642物理候補は、εごとにcompileし直す必要がない。旧identity auditでnative/preparation/runtimeの互換性未証明とされた価格、H6旧固定δの別Gaussian系列、別DF rankの旧H4は使用しない。同じcompiler hashだけで互換とは認定しない。新しいTやKへ旧価格をそのまま移すこともしない。

ここでの再利用確認は固定commitの公開metadata・受入証跡の照合である。元runnerの固定privateパスは本環境に存在せず、private raw/QPY/入力snapshotの再読込・外部再現は行っていない。別保存資料全てに対する無条件の不存在証明ではない。

## Break-evenと推定の扱い

各未取得の適格endpointに、取得済み最良PFと最良PRを固定参照し、全体最良 `G_best=min(G_PF,G_PR)` と相手familyの最良Gを別に保存した。

`N_cos*C_cos + N_sin*C_sin < G_best` の境界は
`C_sin = G_best/N_sin - (N_cos/N_sin)*C_cos`。
CSVには傾き・両軸切片・exact有理数のGを保存した。**両軸単価が等しいという参考仮定**では `C < G_best/(N_cos+N_sin)`。これは必要単価の条件であって、その価格を達成できるという予測ではない。等価格仮定を外しても、保存した二軸不等式で判定できる。

評価の強さを次のように分けた。

1. **妥当な下界による除外：0候補。** 今回確認できた一般下界は非負性だけで、非自明なcompiled-RZ下界は得ていない。proxyや高い近傍価格で完全除外した候補はない。
2. **実測対照による経験的推定：2,499 distinct。** 同system/method/formula/prefix/q/RでTまたはKが異なる対照は1,254候補。これがなければ、同system/method/formula/prefix/RのC/qを用いる比例移送は1,245候補。両軸別に対照中央値と観測min/maxを示す。T/K変更で回路角度・sampling lawが変わるので価格の互換再利用ではない。
3. **参考proxyのみ：6,754 distinct。** 元の `q*(pieces*(prefix+1)+.25*R_outer)` と近傍価格の倍率を用いる。追加感度として、同system/formula/prefixに2種類以上の実測Rがある場合だけC/qのRに対するsecantを補間・外挿する。どちらも下界・実測価格・確率保証として扱わない。
4. **判定不能。** 全ての未取得候補は実測勝敗・母平均について未確定であり、特にproxyのみの6,754候補は直接の同R/prefix対照が不足する。候補表の見かけ上の高価格は除外証明にならない。

保存価格に対するretrospective診断では、PRの同構造移送263件の絶対相対誤差中央値0.431%、95パーセンタイル3.031%、最大7.649%。q比例移送293件は中央値0.617%、95パーセンタイル4.025%、最大9.006%。PFでは同構造186件の最大5.198%、q移送320件の最大5.615%。これは今回の近傍規則の感度診断であり、独立holdout・信頼区間・未取得域の誤差上限ではない。PRの標本平均・希少event未観測による母平均不確かさは別に残る。

## 40セルで何が残るか

E：経験的中央値で最小Gより低い候補。R：経験的対照の観測範囲が最小Gを跨ぐ候補。X：R補間/外挿proxyだけが最小Gより低い候補。F：PF→PR逆転の最優先確認。Q：同逆転の第二候補。空欄相当の「未確定」は有力候補の直接的根拠が弱いという意味で、除外や最適性証明ではない。

|system/T|ε=.05|.01|.005|.001|.0005|
|---|---|---|---|---|---|
|H4/.8|F|R|未確定|未確定|E|
|H4/1.2|未確定|R|未確定|未確定|未確定|
|H4/1.6|未確定|E|R|未確定|未確定|
|H4/2|未確定|未確定|X|X|X|
|H6/.8|Q|未確定|R|未確定|R|
|H6/1.2|未確定|未確定|未確定|未確定|未確定|
|H6/1.6|E・X|未確定|未確定|未確定|未確定|
|H6/2|未確定|未確定|未確定|未確定|未確定|

E/Rは計8セル、Xによる追加はH4/T2の3セル、F/Qは2セル。選定上の着目は計13/40セル。**全40セルに未取得適格候補が残り、非自明な下界がないため全域最小性は40/40で未認定。** 詳細なfamily別件数・相手G・候補IDはcell_summary.csvと全screening表を参照する。

**PF側の不足によるPR→PF逆転：** 現PR勝利33セルにはいずれも未取得PFがあるので、論理的には否定できない。しかし実測対照の経験的中央値で現PRを下回るPFは0件、観測min価格を移して下回るPFも0件だった。最も近いのはH6/T1.6/ε.05のB0+I/S2/prefix11/q4/δ.4（`af293c9c7685199eeb11`）。N=(9,414,8,058)、現PR G=1.265131522e9、等価格閾値72,409.08 RZ/軸に対し、同構造対照91,014 RZ/軸、推定G比1.25694。逆転には対照より20.44%低い単価が必要で、今回のPF移送誤差の観測幅より大きい。低優先の決定論sentinelとしてCSVに残し、初回取得には含めない。他の32セルもCSVに全件残した。

PF自身の取得済み最小値を下回る経験的点推定も0件。ただし最接近はH4/T1.2/ε.005、B0+I/S2/prefix7/q6/δ.2（`d7fee3ff47c6a9590a0d`）のPF最小比1.02197で、q移送だけの根拠である。PF内の最小値の数%改善まで否定するものではない。この候補の現PR最小比は1.57303で、勝敗逆転への優先度は低い。

**PR側の不足によるPF→PR逆転：** 現PF勝利はH4/T全4点のε.05とH6/T=.8/1.2/2のε.05の7セル。最優先はH4/.8/.05。未取得S2/prefix3/q1/R4/K6の閾値3,648.04に対し、同じT/q/prefix/Rの取得済みK2は3,702.56±102.19（1 SE、n=9）、K4は3,768.56±129.70 RZ/軸。必要低下はK2対比1.47%で0.53 SE相当、K4対比3.20%で0.93 SE相当。これはK6の勝利確率の計算ではないが、33/40の頑健性を確認する価値が高い。R8/K2も別の候補として確認する。第二候補のH6/.8/.05はprefix5/q1/R4/K2、閾値12,339.37対q移送13,264.80で、約6.98%の低下が必要。R8/K2はsecant proxy比1.04306であり、いずれも直接価格がなく第二段階とする。他のPF勝利5セルの逆転は今回の近傍証拠では積極的に支持されなかったが、完全除外していない。

## 推奨する最初の7候補

以下は全てPR。上記固定モデル・保存入力を用い、δ=T/q、prefix=L_D。G比較先と二軸境界はpromising_candidates.csvにexact値を含め保存した。等価格閾値はRZ/軸。単価移送とproxyを分けて示す。

|優先|candidate ID|system/T/ε|formula|prefix/q/δ|R/K|N_cos / N_sin|等価格閾値|選定根拠|
|---:|---|---|---|---|---|---|---:|---|
|1|d748ee85d02193720b34|H4/.8/.05|S2|3/1/.8|4/6|12,306 / 8,259|3,648.04|PF→PR逆転の近接候補。対照の標本変動と比べ必要差が小さい|
|1|205f25b8cb63949b4b18|H4/.8/.05|S2|3/1/.8|8/2|11,662 / 7,827|3,849.45|同逆転セルのR対照。secant参考3,864.63で閾値との差0.394%、元proxy比は1.0721|
|2|53acfe65c49018ff2506|H6/1.6/.05|S2|5/4/.4|4/2|12,287 / 10,528|55,451.74|同構造移送53,293.56。現最小G比.96108。現PR内の最小値が変わり得る|
|2|6eb81b93d048deeeb63f|H4/.8/.0005|m5_best|2/1/.8|32/4|90,493,657 / 90,248,666|28,490.83|q移送28,179.13。比.98906。prefix2の未取得q1を確認|
|2|54510520e99d85af4690|H4/1.6/.01|S2|3/8/.2|4/6|200,135 / 236,195|27,871.49|同構造移送27,652.39。比.99214。同価格をε.005にも再利用可能|
|3|cf094355c78d50ae45f7|H6/1.6/.05|S2|5/4/.4|8/2|10,749 / 9,211|63,383.34|上のR4と直接比較する未取得R対照。元proxy比.95670、secant比.92524。双方ともproxy|
|3|45f9fc0a53c1a37c7345|H4/2/.0005|m5_best|3/1/2|64/2|101,917,842 / 84,750,408|39,967.22|元proxy比1.03990に対しR外挿proxy比.99019。ε.005/.001にも影響し得る未取得R64代表|

最初の2候補は33/40の勝敗確認、次の3候補は経験的点改善が見える3セルを各1構造で確認、最後の2候補は未取得R領域の重要な対照である。この異なる問いを1回ずつ扱うため7候補となり、件数上限から選んでいない。

実測対照範囲で競合する11 distinct候補・12 endpointの全てはempirical_frontier.csvに保持した。同一構造のK2/K4/K6を一律全取得せず、最小推定GのKを代表に選び、同値なら小さいKを選ぶ。K間で実際の費用が同一だとは主張しない。残る4構造の近接K6、H6の逆転候補2構造、PF sentinel1構造、H6 m5の未取得R16/R64代表2構造を含め、promising_candidates.csvには計16 distinctを優先度・初回/後続/任意対照に分けて保存した。これは16候補で全域を閉じられるという主張ではない。

## 最小実行範囲と停止点

将来承認を得た場合の最小初回範囲は、上記**7物理候補のcosine/sine両軸だけ**。元のnative compose・sample_requests・compile_cached・paired統計パイプラインで取得し、価格を保存済み5精度endpointへ参照結合する。追加q/prefix/R/K、signal/Hamiltonian/state/reference/shot/allowanceの再計算は不要。探索と独立確認は別cohortとし、既存価格・独立標本を混合しない。

PRの標本数は本調査で確定していない。元の9 trajectory/候補を形式的に使えば63 paired trajectory・126軸になるが、これは元の粗い取得と同じ規模の算術例であり、1%程度の差や母平均を確定する推奨標本数ではない。次の明示的な実行範囲では、目的を粗い点順位の確認とするか、所望の分解能での独立確認とするかを定め、標本数とSTOPを取得前に固定する。今回、実行用grant・seed・launcher・承認証明の仕組みは作成していない。

**mandatory STOP：新規build/compile/transpile/sampling/Signal・Hamiltonian・reference再計算/新規費用取得はいずれも0。** 次の費用取得はユーザーが候補・標本数等の実行範囲を明示的に許可した場合に限る。元private入力が利用できる実行環境で、保存input bindingを維持して進める必要がある。

## 保存物と検証

- [coverage.csv](coverage.csv)：480条件のcoverage。
- [promising_candidates.csv](promising_candidates.csv)：16 distinctの選定・優先度、両軸N、比較先G、break-even、近傍ID、不確かさ。初回は7。
- [cell_summary.csv](cell_summary.csv)：40セルの両familyの不足・最小推定比・proxy感度。
- [candidate_endpoint_coverage.csv.gz](candidate_endpoint_coverage.csv.gz)：全58,160 endpointのID/条件/N/適格性/価格状態。
- [all_unpriced_screening.csv.gz](all_unpriced_screening.csv.gz)：全36,793未取得適格endpointの二軸閾値と評価。価格を0補完していない。
- [empirical_frontier.csv](empirical_frontier.csv)、[analogue_diagnostics.csv](analogue_diagnostics.csv)、[price_reuse_audit.csv](price_reuse_audit.csv)：判断根拠。
- [summary.json](summary.json)、[verification.json](verification.json)：集計・入力hash・検証範囲。

`python3 analysis/h46_unpriced_screening_20261026/verify.py` で、科学runtime import、subprocess/network起動、NPZ/QPY読込を禁止した状態で再生成できる。既存会計2検査（paired共分散・大整数exact会計）を直接呼び出し、元公開CSV58,160行、80基準最小値、36,793 exact break-even、別m5カタログ80一致軸と22入力hashを照合してPASS。pytestは本hostで未導入のため、既存test関数をstdlib runpyで実行し、環境は変更していない。

再利用：`h46_long_time_signal.read/cid/saved_assets`、`h46_long_time_report.endpoints/costs/cost_for`、`h46_long_time_native.cost_proxy`、既存accounting tests。追加コードは既存にないcoverage・二軸break-even・近傍照合・選定の保存に限定した。書込みを伴う元のinventory/account/selection/run関数は実行していない。検証はlocal保存値解析であり、immutable CI・新規実分子検証ではない。
