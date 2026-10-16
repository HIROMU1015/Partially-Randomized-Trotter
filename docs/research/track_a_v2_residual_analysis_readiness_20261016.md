# Track A V2：既存理論・compiled費用の残差解析に向けた再開準備

**既存832理論行と保存compiled接続10,480行を再利用する。旧V2のprimitive生成・A/B基本表作成は重複するため再実行しない。今回は読取専用の対応整理と実行計画までで、残差・回帰係数・新しい費用比は計算していない。**

準備branchは`track-a-v2-residual-analysis-readiness-20261016`、起点はV1成果整理`63e5773b6b408506f4c39783b80f3025f14481b9`。既存V1、V2準備`dd695e6584f15b88341d2446ed010c7bdfef1666`、理論結果`7e41670565d9284da9aded94063be591fdcaa3dd`のbranchを変更しない。新しいGPT/Claude詳細レビュー・サブエージェントは使用していない。

[入力hash・対応政策・未承認計画・技術照合](../../artifacts/resource_applicability/track_a_v2_residual_readiness/2026-10-16/README.md)。[前回の読取専用照合](track_a_v2_existing_theory_reconciliation_20261016.md)、[V1の確定事項と限界](track_a_v1_results_synthesis_20261016.md)。

## 再利用する正本

理論結果の正本は[固定commitの報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e41670565d9284da9aded94063be591fdcaa3dd/docs/research/track_a_native_df_theory_evaluation_20261016.md)と[artifact索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7e41670565d9284da9aded94063be591fdcaa3dd/artifacts/resource_applicability/track_a_native_df_theory_evaluation/2026-10-16/README.md)。理論の実行SOURCEは`7e59df8ac3c356007d85bfd9218ce1a224a40d52`であり、結果commitとは区別する。

|既存成果|保存範囲|今回の扱い|
|---|---|---|
|A/B理論wrapper表|H6/H8各104固有候補、各A/B×cosine/sine、合計832行|Fractionのlower/upperをそのまま再利用。CSVは閲覧用|
|timed primitive値digest|H6 504・H8 708、計1,212件|既存来歴として参照。元の数値producerを再呼出ししない|
|finite-RTE law receipt|H6/H8各72、計144件、K2/K4の全支持次数|再評価・再正規化しない。未観測次数を0にしない|
|preparation identity|H6保存archive7 root再利用、H8補完7 preparationの旧hash一致|新archive/preparation/Gaussian生成なし|
|保存shot/compiled接続|10,480条件・metric行。接続5,736、保存shot不適格4,744|既存値を読み、shot・G・比・winnerを更新しない|
|candidate/SOURCE索引|H6 132、H8 130のscope entries|探索/確認/B0Iと元SOURCEを分離するため再利用|

対象はlinear H6/H8、間隔1.00 Å、STO-3G、保存DF rank19/28、T=.8。H6 prefix5/10/15/full19、H8 prefix7/14/21/full28、q1/2/4/8、旧S2/full S4・PR登録集合。V1追加細格子、p7の実測、T=.4/1.6、H5/H7を832行へ追加しない。

既存接続のgzipは圧縮1,032,973 bytes、展開14,579,546 bytes。元export receiptのSHAへ照合し、JSONの件数・ID・schemaだけを読んだ。保存値の減算・除算・回帰・再会計・順位付けは行っていない。既存結果のterminalは完了STOPで、今回の確認ではV2/既存理論runnerに一致する稼働処理はなかった。

## A/B、実native、compiled費用の対応

|層|構成・意味|compiled値との差をどう解釈するか|
|---|---|---|
|均一モデルA|PF prefix・RTEともGaussian再利用なし。固定明示分解`explicit_rz_sx_x_cx_v1`|実nativeのRTE再利用とcompiler最適化の両方との差を含み得る|
|均一モデルB|宣言済みの同basis隣接境界相殺。identity barrierとclosed occurrenceを保持|PF prefixまでB構成にした仮想比較。実nativeの実装済み高速化とは呼ばない|
|実native混合構成|PF prefix=A相当、RTE tail=B相当。RTE occurrence間、prefix/stepとの境界は閉じる|832行に独立したmixed期待費用列はない。均一A又はBと完全同一ではない|
|保存compiled値|実native wrapperを既存Qiskit1.3.0・rz/sx/x/cx・opt1・seed17でcompileした単価/PR標本平均|構成差・compilerによる融合/消去・有限標本差・来歴の条件を合わせた観測量|

PFのみのnative構造はA相当であり、BモデルはPFでも仮想構成。PRのuniform A/B表とliteral mixed表を取り違えない。A/B全体費用2列だけから、prefix/tailの内訳なしにmixed費用を補間・差分合成することはしない。mixed理論表を追加する場合は、別のcounter評価・binding・実行承認が必要であり、今回も次の最小残差stageも対象にしない。

理論表はstate preparation・measurement命令・routing・FT回転合成を除く。wrapperのH/Sdg等は固定分解で数える。一方、保存値はmeasurement込みfull-wrapper由来。RZ/CXにmeasurementを勝手に換算して足し引きしない。depth、T/Toffoli数への換算は行わない。

一般identity phaseの`0..1 P / closed occurrence`幅、元binary64 lawをFractionへ厳密正規化したという条件は維持する。真の確率・母平均・数値誤差certificateを与える表ではない。

## 来歴に関する実行前条件

既存接続全行のidentity statusは`CONDITIONALLY_ALIGNED_METADATA_NOT_FULL_NUMERIC_ENVELOPE`。元indexの全262 scope entriesもfull envelope未認証であり、832行は数値wrapper fingerprintの全再構築をしていない。この状態をhashの一致だけで完全認証へ昇格させない。

特に以下を独立したexpected値に結合する。

1. system、Hamiltonian/input/state/DF/order、T、candidateのprefix/q/formula/r/K/R、preparation、law/normalizationとscalar phase政策。
2. compiled origin SOURCE、dataset/seed SOURCE、execution SOURCEを区別。接続行のoriginがnullでもindexにSOURCE列挙がある場合は、原receiptへの参照を追加し、nullを単にexecution SOURCEで埋めない。
3. compiler hash `1239304c…`だけで全binary/profile同値とはしない。旧profile`4e32bb9b…`、current`622ce4a8…`、H6旧basis variantを分離する。H6/H8 provider共通6 sourceの一致はprovider全体又は全旧lineageの同一性を意味しない。
4. 探索H6/H8各n8、H6確認union8/n32、H8確認union6/n32、B0IのPF実測を別scopeとする。候補IDだけでjoinせず、両軸・paired group・profile・variant・元receiptを照合する。

未知のcritical fieldは`UNRESOLVED_PROVENANCE`、矛盾は即STOP。SOURCE文字列が異なるものを同一にまとめるには、対象receiptとcompile経路の互換根拠が必要。物理的に同じHという説明だけでは統合しない。
完全数値同一性未認証のまま実施できるのは、別契約で明示的に承認された**metadata条件付きの保存点診断**だけ。旧V2の完全envelope gateを通過したとは称さず、母平均やmodelの正しさを認定しない。必要な来歴が不足していれば較正は不可評価で終了し、新生成へfallbackしない。

## 最小の数値実行計画案（未承認）

最初のstageは**保存RZ単価についてのuniform A/B別・各軸別・system別の条件付き較正/残差診断だけ**とする。CX、shot重み付き総費用、PR/PF費用比の更新、候補ranking、混合理論表、希少次数別の新寄与表、回転角分類は後続の別判断へ分離する。

|順序|承認後に必要な処理|検証・STOP条件|
|---|---|---|
|1|commit/input/gzip/hash・新runner・新grant・資源・保護job・未使用outputをfresh gateへ結合|不一致、承認なし、旧marker使用、監視不備で起動しない|
|2|832理論行・10,480接続行・262 index entryを一度だけ読む。元の値を保持|理論counter/DP/producer・Qiskit・SCF等へ接続しない|
|3|保存axis meansを用いる単価tableを作る。同一scope/candidate/axis/sourceの精度・factor・metric・modelによる反復を照合してdedup|値が食い違えばSTOP。同じ観測を独立10条件や別modelの複数標本として重み付けしない|
|4|上記の来歴stratumを固定。探索と確認を別々に扱い、再利用PF wrapperも二重標本化しない|不明・欠測・零分母を残差0で埋めない。保存shotの不適格と単価の欠測を混同しない|
|5|固定hashで学習/確認groupを結果を見る前に分ける。共通較正`C_compiled=a+b*C_theory`をPR/PF混合の同一stratum内で行う|train/holdout双方にPR/PF各2以上、trainで異なるxが2値以上、rank2が必要。不足ならINCONCLUSIVE、分割の再探索なし|
|6|holdout各行のsigned残差`e=(C_compiled-fit)/C_compiled`、PR/PFの平均差を保存|各system/model/axis/stratumで独立判定。`abs(mean(e_PR)-mean(e_PF)) >= .05`なら即STOP|
|7|比較可能な全登録stratumの状態・欠測理由・部分結果・保全・cleanupを保存|完了も不一致もmandatory STOP。次stageへの自動移行なし|

単価での学習を主とするため、epsilon/factorの繰返しを回帰の重みに使わない。PRはn8/n32の保存標本平均であり、個々のtrajectoryを新たに読み出してSEやbootstrapを再導出しない。得られる差は条件付き点診断であり、有意差や母平均model誤差保証ではない。

holdout groupは`(system,prefix,q,formula)`単位で固定SHA256の`mod 5 == 0`を確認側へ、他を学習側へ割り当てる。groupを順序付きJSON配列にし、`ensure_ascii=True,separators=(',',':'),allow_nan=False`のASCII bytesへencodeし、SHA256 digestをbig-endian整数として判定する。method、r/K、model、axis、epsilon/factor、scopeが違っても同じ物理groupを学習/確認へ分断しない。scope間のPF再利用も同じgroupへ寄せる。分割の可否は件数で判定し、値を見て変更しない。

lower/upperを保持し、両端を事前固定の別scenarioとして較正する。midpointへの無断置換はしない。どちらかが.05以上、又は端点で判定が変わる場合はSTOPして報告する。両端一致は区間内の最悪残差を保証しない。個々のfitはscaled float64のrank2 least squares、`rcond=1e-12`、設計行列condition number上限`1e10`を案とし、非有限値・負の傾き・非正の予測単価・数値境界不明はSTOP。.05の`1e-12`以内はPASSへ丸めず`BOUNDARY_UNRESOLVED`とする。

この.05は**平均signed相対残差差の5 percentage points**で、V1の各軸費用model誤差2%基準とは別。方法別fitや相殺による全体平均で失敗stratumを救済しない。registered順はH6→H8、A→B、cosine→sine、stratum ID辞書順、lower→upper。不一致後の次fit・追加fit・fit再定義・再試行は禁止する。

## 必要な実装検証と資源案

数値runnerを起動する前に、人工metadataだけで、①精度反復とPF再利用のdedup、②SOURCE/profile/variant/axisの混合拒否、③missing/zero/negative/nonfinite、④rank不足・PR/PF不足・固定group分割、⑤.05ちょうどと数値境界、⑥lower/upperと失敗後の次fitゼロ、⑦未承認grantと旧marker拒否、⑧owned cleanupを確認する。実データの残差をテスト代わりに先に算出しない。

今回は実装済み数値runnerや有効grantの完成を主張しない。今回の成果は保存データ対応・未解決箇所・実装要件・未承認契約案である。実装と人工検証、最終SOURCE/input sealを済ませ、別途の明示実行承認が成立した後だけ数値stageを開始できる。

提案上限はdriver1＋observer1、各thread1、freshな別CPU2枠、driver AS4GiB/RSS1GiB、observer AS/RSS512MiB、総RSS1.5GiB、総CPU600秒、wall1800秒、output64MiB、stage128MiB。最大832 theory rows、10,480接続行、262 index entries、1,048単価/model接続行、64 strata・lower/upper計128 fits。上限は実測所要時間ではなく、超過時にSTOPする未承認ceiling。

metadataは一度loadして共有し、個々の小さなfitを逐次実行する。即STOP順序を守り、巨大なtheory DPやcompileを再実行しない。計算workerの追加並列化を必要条件にせず、外部observerを別processにする。旧V2の4時間/6時間・primitive上限や使用済みgrantをこの小さなstageへ流用しない。

NumPy等を使う場合も固定環境で線形代数だけに限定し、科学provider・counter・sampler・compilerをimportしない。起動直前30秒以内のSOURCE/input/profile、CPU idle、既存job非干渉、memory/PSI/OOM、disk/inode、外部監視readyを要求する。旧venv・H5/H7 jobを変更しない。

外部監視は1秒間隔・最大gap5秒、owned PID/start/parent/UIDとthread/AS/RSS/CPUを確認する。入場memoryは総RSS上限＋16GiB headroom、host memory PSI avg10が1%以上・nonroot cgroup PSI正値・OOM増分・disk reserve2GiB/inode4096割れでSTOP。停止対象は所有確認した自分のdriver/observerだけ。gzip展開は16MiBで上限を固定し、既存export receiptのbytes/SHAへ一致しなければparse前に拒否する。

## 未解決事項と保全・公開後STOP

mixed理論の未取得、完全wrapper/basis/seed/profile結合、旧H6 variant、非Clifford角分類、希少次数のcompiled母平均保証、厳密bias保証は未解決。uniform A/Bの残差だけではこれらを分離・認定できない。V1の−13.2901%実測点比較と残る4精度のモデル条件付き結果も変更しない。

新しい科学計算・数値再解析・compile・signal・sampling・DP・primitive/law評価・回帰・費用比更新・新grant発行は0。元SOURCE、理論/compiled/raw、grant/ledger、V1、H5/H7を保全した。既存V2 worktreeの未追跡`track_a_phase_a_parallel.py`も変更しない。検査はprimary agent自身のbytes/hash/schema/metadata照合であり、独立レビューではない。

公開対象は本報告、保存ファイルの公開commit/path/hashを持つ入力台帳、構成対応、未承認計画、metadata照合receiptと関連索引だけ。既存832行・gzip・NPZ・state/角配列・private config/grant・既存job一覧・credentialは新branchへコピーしない。

**準備成果をcommit・pushし、remote SHAとtreeを確認したらmandatory STOP。数値stageは未承認、CPU[]、command=null。追加処理へ自動移行しない。**
