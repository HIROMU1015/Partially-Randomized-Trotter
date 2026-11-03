# Codex引継ぎ：PR-2 S2後の研究再設計と必要検証

日付：2026-09-29  
基準branch：`pr2-v4-s2-parallelization-20260928`  
基準commit：`a9171d8bea93441afc9b17c7ffeab79af8dbcc95`  
対応する研究方針：`pr2_research_redesign_a9171d8_20260929.md`

## 0. この文書で許すこと

この文書は研究計画であり、旧S2/S3 authorizationの自動延長ではない。まず既存artifactのread-only集計、新しい比較契約と実行規模の見積りを行う。新しいsignal/compile計算は、新契約を結果前にfreezeしてから行う。held-outはそのさらに後の明示承認まで開かない。

旧S2 status `S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`、旧rank 6 primary、rank 3/9 controlの区分をそのまま保存する。新研究のdevelopment情報としてcontrolを利用することはできるが、旧S2の勝者を遡及的に変更しない。

旧入力hash回収、旧S0 PASS化、canonicalization研究、停止済みP-A～D/R3/FR/PR-3系列の再開は行わない。

## 1. 新しい研究の中心

> 同じDF target・state・物理時間・信号精度のもとで、残差を捨てる／決定論で保持する／ランダムに補完する選択を比較する。splitだけでなく、各方式に必要なq/r/Kを公平に選ぶ機会を与え、1-shot costと必要shotの交換関係を説明する。

「rank 3を勝たせる」「新しいprefix生成法を発明した」「prep費を都合よく選ぶ」を目的にしない。

## 2. 最初に読む固定資料

1. `docs/pr2_v4_s2_development_validation.md`
2. `docs/research/pr2_v4_s2_development_authorization_v5.md`
3. `artifacts/pr2_v4_s2_development/2026-09-29/pr2_s2_development_resource_result_parallel_v1.json`
4. `src/trotterlib/pr2_v4_s2_development_validation.py`
5. `src/trotterlib/pr2_v4_s2_parallel_execution.py`
6. 対応するrunner/test、および保持している既存SQLite cacheのscope metadata

cacheは再利用可否の検査対象であり、新しい証拠の正本ではない。公開artifactと一致する科学条件・source・compiler・seedを持つcellだけを再利用する。

## 3. 作業A：read-onlyの結果再構成

新しいmatrix evolution、sampling、compileなしで行う。

### A1. 全候補台帳

各recordについて以下を一表へ出す。

- old role：primary / control / endpoint。
- input hash、source commit、method、rank、q、delta、r、K。
- raw/corrected mean、exact target、Re/Im bias、finite truncation、outer PF。
- multiplier B、attenuation、axis shots、total shots。
- mean full-wrapper RZ、SE、sample count、engineering interval。
- CX/depth/sizeの既存secondary。
- r/K/qの探索境界、setting uncertainty、未保存・未評価情報。

旧hash・fingerprintは計算して照合してよい。検証に失敗したら資料不整合として止め、再計算で上書きしない。

### A2. 既存の費用差

rank 3/6/B3について、work比をshot比とshot-weighted 1-shot cost比へ分解する。成分costが保存されていなければ「未計測」と書く。full compileの費用を、backbone＋tailの厳密な和と仮定しない。

各固定設定のG(P)=A+NPを作り、pairwise crossingと全評価済み候補のlower envelopeを区別する。Pの有利な値を一つ選んで成功例としない。

### A3. 既存資料だけで分かる不足

最低限、次を未検証として列挙する。

- B0 rank 3/9のS2 task。
- q=8を減らしてよいdeterministic / partial候補。
- B3のr=32境界。
- rank3/9のr/K最適化機会。
- 選択候補のcost推定不確かさ。

この表は新しい実験を無制限に増やすリストではない。作業Bの一回の比較設計へ集約する。

## 4. 作業B：新しい比較契約をfreezeする

新契約案のファイル名例：`docs/research/pr2_matched_accuracy_resource_contract_v1.md`。

### B1. 科学条件

- 同じdevelopment snapshotを読み込む。再生成しない。
- T=0.8、epsilon_complex=0.05、axis alpha=0.025を維持。
- 同じstate、sector、identity policy、coefficient threshold、compiler。
- q={1,2,4,8}、必ずdelta=T/q。
- discard rank={3,6,9}、full deterministic rank=12。
- partial rank={3,6,9}、random-dominant rank=0。
- r={1,2,4,8,16,32}、K={2,4}を基本候補。
- q/r/Kの上限外は最適性未検証と明記。
- r=32の境界候補が結論に関与する場合、同じ規則でr=64を一段だけ確認する条件を結果前固定。r128以降へ自動進行しない。

qは外側反復、rは各tailの細分、Kはpaired cutoffであり、rankとは別記号。sourceの旧DELTA_TIME固定を、新qでもそのまま使うbugを防ぐ。

### B2. estimandと費用

primary mean=normalization-corrected signal。raw meanも保持する。

bias-aware Hoeffding式は既存v5を使い、B_totalをqおよび各occurrenceへ正しく合成する。

primary=total expected compiled RZ, no state preparation。secondary=P>=0の感度。Pは新しい状態のoverlapを変えるparameterではない。

corrected truncation biasとouter PF biasの複素差を保持し、絶対値の和を実測total biasと混同しない。bias floor、非finite、epsilon_stat<=0、候補不適格の扱いを固定する。

### B3. 検索とcompileの分離

基本候補のmean/normalization/analytic shotsを先に計算する。既存と同じ28cellはidentity条件が一致すれば再利用する。

**全192 random cellに128 trajectoryを自動実行しない。** 次を結果前に固定してから計算する。

- 各method・splitに同じ情報のcost-screeningを与える。
- direct-compile候補の選び方、primary予測と別予測が異なる場合のchallengerの残し方。
- 新規random direct-compile cellの上限案16。
- 32 trajectory初回、判断を変えるcellのみ追加96、最大128。
- 16cell以内で比較が閉じなければ、未評価候補を消さずselection-limited/boundary-limitedと報告。
- validなcost下界がない限り、surrogateで除外した候補を真にdominatedと呼ばない。

原稿の主張を「評価した候補／事前固定した候補選択手続きの範囲」へ限定すれば、全gridの厳密最適化を研究課題として追加する必要はない。global最適性を主張したい場合は、未評価候補への保証が別途必要。

### B4. 誤差・費用不確かさ

旧engineering intervalをformal CIにしない。新候補選択後の勝者の最小値だけを報告しない。必要な独立確認streamの対象と上限を事前に決める。

10%差を跨ぐ場合、uncertainとmaterially negativeを区別する。技術的failureは科学的negativeと分ける。

## 5. 作業C：同精度development比較を一巡実行

作業Bのfreeze後に限り、新しい決定論的signal評価と限定compileを許す。

1. same-target discard/full deterministicのq比較。
2. partial/random端点にも同じq/r/K選択機会。
3. accuracy適格性、費用screening、直接compile、必要な境界1段。
4. fixed-q8とmatched-accuracyの比較。
5. prep sensitivityとprimaryの分離。
6. 全候補台帳と除外理由・未評価範囲を保存。
7. 必ず停止して、結果と原稿の中心主張を照合する。

この段階の主判定は「rank3が勝ったか」ではなく、**公平な反復数・discard/endpointを戻した後、部分ランダム化の採用条件について何が新しく分かったか**。

## 6. 作業D：held-outは別authorization

現時点ではheld-out NPZ・signal・cost・rankingを開かない。

作業Cの後、developmentで固定した代表構成とbaselineを選ぶ。rank/q/r/Kを含む候補集合・予測・失敗時報告を結果前固定する。

その後に許可されたH4 1.30 Åへ一度適用する。held-outでrankやqを選び直して同じ評価を成功へ書き換えない。必要shotは同じ定義に従ってその入力のbiasから再計算する。

これは固定構成のtransferである。各geometryでoracle最適化したresource mapを別途作りたい場合は、transferという名称と混ぜない。

## 7. 追加検証を許す理由と許さない理由

許す理由：弱い対照、実装誤り、未解決の探索境界、費用推定誤差などが中心結論を変える。

許さない理由：rank3を勝たせたい、10%を超える点がほしい、まだ他のgridが試せる、H4では小さすぎる気がする。

新しい一般定理の完成、canonical hash、FT全stack、長RPE、全分子への一般化を共通必須条件にしない。一方、実際に比較していない高次deterministic PF・実機費用・量子優位への主張を出さない。

## 8. 最終報告に必須の区別

- 過去のprimary結論 / controlからの探索的仮説 / 新契約下の結果。
- measured circuit cost / cost proxy / analytic sufficient shot count / actual quantum shots。
- point ranking / engineering uncertainty / 10% materiality。
- fixed-q comparison / matched-accuracy comparison。
- pairwise prep crossing / all-evaluated-candidate lower envelope。
- finite candidate setの最良 / 未探索範囲を含むglobal optimum。
- developmentでの選択 / frozen-design transfer。

## 9. 直近の引継ぎ文

> a9171d8の旧S2を変更せず、read-onlyの全候補台帳・費用差の分解・比較不足表を作ってください。次の主RQは、同じtarget signal精度の下でのdiscard／deterministic／partial／random-dominantの使い分けです。rank3を旧primary winnerへ昇格させず、新研究のdevelopment情報として扱ってください。
>
> 固定q8の資源比較だけでは最安方式を判断できないため、同じTでq={1,2,4,8}を全方式へ与え、B0 rank3/6/9とB3の境界を含む新契約案を作ってください。候補meanの評価と高価なcompileを分離し、直接compileする候補の選出規則、未評価候補の扱い、16cell上限案、32→128の追加規則、r64の一段境界確認を結果前に固定してください。
>
> 最初の成果物は契約・計算規模・不足表までです。新計算はそのfreeze後に行い、development比較を一巡した時点で停止してください。held-outは別authorizationまで開かず、未知の結果を有利にするための閾値変更や新しいgeometryの追加はしないでください。
