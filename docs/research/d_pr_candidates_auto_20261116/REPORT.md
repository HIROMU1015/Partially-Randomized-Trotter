# PR候補の自動準備・探索復帰・C同期

研究者が候補ごとにseedや較正登録を手で投入せず、**PR-PauliとPR-DFの両方を同じ通常探索入口から費用比較へ戻せた**。B2のV2が選んだPauli LD4とDF LD6について、不足取得→通常intake→候補固有価格→探索器による配分選択→Q3→Cの通常定期同期を実データで確認した。検収は各方式で最初のREADY候補1件・Q3各1件の有限範囲であり、最安探索を完了した主張ではない。

保存H4 linear 1.00 Å / STO-3G / tol-only DF10、元H/state/snapshot・fragment順、one-body別head、S2、K2、κ絶対値2の範囲。単価は元compiler/lowering/P0の測定付きfull controlled Hadamard wrapperのcompiled RZ/shot。価格policyはMODEL。MSEは既存target/decoder/全事象loss/exact τとideal finite-RTE契約に条件付く。

|候補|configuration ID|選択配分のモデル費用 RZ|Q3 MSE [下界,上界] Ha²|判定|
|---|---|---:|---|---|
|PR_DF LD6|`5bebe69183952f0c…`|1452972904.07142854|[4.490321550542e-07, 4.490321550542e-07]|適合|
|PR_PAULI LD4|`8bf6eb88c735daf2…`|725939453.71428573|[1.579495343671e-06, 1.579495343672e-06]|適合|

Pauliのexact費用は `5081576176/7`、DFは `20341620657/14` RZ。PauliのX/Y各配分は `[48,45,42,39,36,33,29,26,23,19,16,13,9]`、DFは `[64,60,56,52,47,43,38,34,30,25,21,17,12]`。D2が配分を指定した結果ではなく、B2の既存規則が各構成の初期候補として選択したもの。次の配分要求を保存して `Q3_BUDGET_EXHAUSTED` で停止した。

## 接続と保存再利用

B2の起点は `d7a976a629a6db68197b24f8f86734e109c8bf5d`。A候補準備・実step較正、D2のPriceRulesとDF信号入口、元P2 fresh取得、元Q3 worker/AST guardを再利用した。新しいL_D・κ・shot chooserは作っていない。通常CLI、直接API、対応条件は[API.md](API.md)。26軸価格・実schedule・SOURCE・配分・MSE exact値・費用は[RESULT.json](RESULT.json)。

Pauli LD4/6へのBの最新指示を反映した。B HANDOFF `13326577297673151fd9735e5ee060c20c6d06f5`、実装 `33e324b43690caeaa353ecdb9e3d42f3139198b6` を固定viewで使用。A entry `6e4b6bd6a491815c2a3cb8f809cbfc0240b60fdb` のplan/execute/assembleすべてへ候補のtail descriptorを渡す。B2 common_metadataは同HANDOFFの `A.ENTRY_SOURCES` membership照合に更新した。LD4は保存configuration/P0を再利用し13 roundだけ取得した。LD6は手動で科学起動していない。

DFではV2がLD6を選んだ。保存contextにないprefix6のλ_R/s_Rを元helperから一度導出し、入力/生成SOURCEを固定してcanonical configuration・13 round信号を取得した。λ_R=`0.022078920369741746`、tail identity=`0.004183531176913642`。Hamiltonian定数cとs_Rの出所を分け、c+s_Rは総Tに一度だけ適用した。全round R=1。元mapper/lawと既存providerを維持し、保存Full-H reference pairを使用した。

|工程|PR-Pauli LD4|PR-DF LD6|
|---|---|---|
|候補選択|実データ：V2 NEW_L_D|実データ：V2 NEW_L_D|
|descriptor/P0/configuration|保存結果：LD4 P0/configuration再利用|実データ：descriptor1・configuration1|
|信号|実データ：不足13 round|実データ：不足13 round|
|通常intake/total view|実データ：B新SOURCE|実データ：既存DF intake|
|候補固有価格|実データ：normal2 path・4 compile・元audit|実データ：較正8 path・16 compile・軸別fit2|
|独立holdout|既存audit規則の範囲。新holdoutなし|実データ：m3独立2 path・4 compile、記述比較|
|探索復帰・配分・Q3・費用|実データ：Q3 1件、適合|実データ：Q3 1件、適合、保存artifact再照合|
|C比較表|通常定期同期で確認|通常定期同期で確認|
|κ提案座標、後続候補継続、物理alias、部分軸再開|人工検査＋保存metadata|人工検査＋保存metadata|

DFモデルは候補自身のm0/m6各4 path平均からX/Y別 `α+βq`。26価格凍結後にm3独立2 pathを取得した。各軸P=`1554005/36`、O=`42975`、P−O=`6905/36`、相対差=`1381/309420`。許容差・model PASS/FAILは採択していない。m7〜m12はq=64を超える外挿。Pauliは自分のS0/r1 auditのA/H/Eを使用し、q/Rとevent stepの外挿欄をquoteへ保存した。LD間の係数移送はない。

## 固定参照・検査・資源

実DF取得SOURCEは `194e00a8a058a7a009e186bcd03b38c7329ec2f7`、Pauli coordinator修正SOURCEは `1691f5f34e43a971ead2f67673788678588106fb`、Pauli信号の実行SOURCEは `243abf38c4375504762ec80c1e945995fa5877c6`。結果公開SHAは実行SHAと別に各recordへ保存した。DF較正rawは `3c7c45e4492d83f4f1acfa831702fdd36e9b0558`、モデル凍結 `d698d93e6216f2335bd2b7b5d011f49cbd3a7bbc`、独立holdout raw `181085fdb9242ceddb40d53d53577d0e8ea477d8`。Pauli信号raw `5c653c406726298ca99499d74e9ecff97e4689be`、packet `2895671ae86d3969dcac1dcc7f538d5911f2c62d`、価格/audit `97413fc1218beb20c70744ca183bfd4f3dfa2437`。正確なpath/selectorはRESULT/quote/STATEへ結んだ。

新規は信号26 round、normal12 path、測定付きwrapper24 compile、DF軸別fit2、参照価格52軸。Pauli auditは追加sampling/build/compile/fit各0、保存QPY読取8件。Q3 worker実時間はPauli約153.3秒・DF約242.3秒、process-tree CPU約252.6秒・427.1秒。DF sampled process-tree peakは5,334,152 KiB。Pauliのparent peak621,020 KiBとmax-single-child438,320 KiBは別指標であり、DF tree peakとの同一定義比較には使わない。各科学leafの開始/終了・所要時間・exit codeは元RETURN/STAGES/RESOURCEへ保存した。

検査は今回の入口に限定した人工fixtureと保存metadata照合。保存H4 P0の独立数値再検査を含む回帰suiteは実行していない。Full-wrapper statevector検証は元P2同様未実施。テスト一覧は[TESTS.txt](TESTS.txt)。各担当による実装検査であり、作者の検査を独立レビューとは扱わない。

## 技術停止と保全

1. run01ではPauli descriptor envelopeのprefix/binding不足でplanが停止し、新規Pauli信号は0件だった。旧plan/STOPを保存し、新SOURCE・別出力で形式だけを修正した。
2. run01の検収範囲が広く、既存ready候補に不要なQ3を3件投入した。いずれも不適合。D2 controllerだけを停止し、すでに走ったworkerは完走・保存した。`fd5c63006c4b8c14c56b5ef7691ce3d73236ad9d` と RESULT.preserved_extra_Q3に残す。B2のrun・worktree・checkpointは変更していない。今回の総Q3実行は5件で、うち新候補検収2件、範囲訂正前3件を区別する。
3. DFの科学workerは適合・終了code0だったが、共通adapterがPauli用PROCESS_EXITSだけを探しUNKNOWNを返した。`69ef01cdc7d53f30e66920f9f05ad5f72d7d7e54`の旧OUTCOME/SUPERVISIONを保全。元DF supervisorのjoined process_exitcodesを同じ厳しさで読む修正後、SOURCE/identity/closure/実終了codeをcollectのみで確認した。別COLLECTED_OUTCOME/COLLECT_SUPERVISIONに結び、科学Q3は再実行していない。終了記録欠落・非zero・signal・identity不一致は受け入れない。

## 再開・適用範囲と残る条件

同じrunのREADY・Q3・価格を再利用し、科学投入を増やさず再開した。[REPLAY_CHECK.json](REPLAY_CHECK.json)を参照。新しい登録証拠が追加された価格HOLDはmetadataだけ再照会できる。既失敗・曖昧な科学leafを無断retryする仕組みにはしていない。

後続候補は元V2のinput_requestsを使い、同じExecutorと累積Q3予算で続行する。κの絶対値を保持し、古いmultiplier提案は明示されたbase/座標SOURCEからだけ変換する。κ宣言だけ異なる物理aliasは元B2登録規則を再利用する。DF descriptorは同じ分割の後続κでも共有する。これらの汎用分岐と部分receipt再利用の新部分は人工検査で、追加の実候補を取得した証拠とは区別する。

未対応公式/piece/K、複数actual stepの係数登録、rank不足、不明なSOURCE/receipt、独立性未解決のcohortは理由付きNEEDS/HELD。元P0の数値停止も無断retryしない。未取得価格は0円やQ3不適合へ置き換えない。条件付きモデル費用は正式E[C]・CI・Gや全L_Dの最適性・winnerを認定しない。

Cの実際の定期同期は[検収記録](C_DAEMON_ACCEPTANCE.json)。公開C commit `1d44d07397031547e402171faae268a640a2807b` のCATALOG/REPORTに両行が入り、26軸・配分・MSE・費用が一致した。ローカルreader検査SOURCE06e…と、独立に進行した実daemon SOURCE76474c3d…を区別して保存した。

ロードマップ：両方式とも未準備候補を理由に人手の取得・転送を要求する段階から、支持されたprovider/価格規則の範囲内で自動取得・探索復帰・C同期する段階へ進んだ。候補選択/探索高速化はB2に残す。次のL_Dや追加価格・Q3の科学範囲は元探索条件に従い、この有限検収の後に無断で広げない。

## 追記：LD7/8/9の登録済み新P0 identity

Aの最新HANDOFF `d3a1a3d359014e7d0bd95e648b4174cfe1598c9e`、準備API/登録表 `9f00ddd86a44b5d1443f29b5f02dbf1db4ef9ab8`、builder `25a08afafa32af2ca52bca4f63233c38b3af0e6e` を反映した。今後の `settings.pauli_workspace` は新登録表を含む所有checkout `d2-pauli-auto-nodrop-evidence-20261116` を使う。LD7/8/9は通常入口へ `--p0-builder no-drop` を渡し、`REUSED_CONFIGURATION` と `P0_NO_INTERMEDIATE_DROP_V1` の保存辞書・基準構成を使う。LD8を含めtail descriptorをplan/execute/assembleへ渡す。

旧辞書のSTOPは旧SOURCE/identityの履歴として残す。新辞書のidentity・λ_P・lawを別物として登録し、旧辞書の価格係数を黙って移送しない。新P0を再生成せず、分割は同じまま公式/幅/κを変える際も辞書を共有する。固定参照とmetadata検収は[NO_DROP_REGISTRY_CHECK.json](NO_DROP_REGISTRY_CHECK.json)。3分割の信号はここでは取得せず、実候補選択と不足round取得は通常B2探索に委ねる。完了済みRUN02のA checkoutとSOURCEは変更していない。
