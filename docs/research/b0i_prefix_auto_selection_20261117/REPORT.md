# B0Iの切り捨て位置を通常探索から選ぶ技術検証

**通常入口から未評価pを自動選択し、Dの既存補完入口へ渡せた。** 保存H4の計画では、pを事前選択せず、p=9・標準Yoshida4・要求幅32h/5（configuration `9a495a223e254747a34f6b31f2457bf7b0498578f1dc49053f915652a92494f2`）を選択した。準備後の同じ探索への復帰、通常generatorによるshot提案、人工Q3、費用比較と再開は人工応答で確認した。

**人手に残るのは、実科学計算を行う場合の対象・有限範囲の判断と許可、固定D provider環境の指定、未解決の旧job所有者への対応である。次のp・公式・幅・shot列を人が書く必要はない。** 今回は新規H4信号・sampling・回路build/compile・価格fit・実Q3をすべて0件とした。p7以外の実Q3適合点や実費用は増えていない。

## 変更前と不足していた接続

高速化版 `e839b0d3f5469c21f5944aec5d978f91d193efc8` はB2 `d8e806265c3ded8e1184181c37c2d7ec71f08ade` を含む。通常入口は `scripts/run_rpe_method_optimizer.py --problem … --round … --run-dir …`。この版ではDのPF補完adapterがまだなく、pのscopeを与えても固定indexのp7候補9件だけを調べた。[変更前run](baseline_native_catalog/ROUND_RESULT.json)は保存最安372,035,410 RZを返し、科学call0件で停止した。`pf_completion`は選択処理に接続されていなかった。

D `36ad751d4ebf0ec160532615da6425b268a0c18a` の補完helperは、**呼出し側が書いた** `pf_completion.selected` を受け取る入口だった。この既存能力と「探索器がselectedを埋める能力」を区別した。今回、候補列挙後・shot探索前に不足候補を一つ選び、そのcanonical configurationを既存helperへ渡す最小の接続を追加した。明示selectedの旧入口は互換用に残し、`CALLER_SUPPLIED_SELECTED_LEGACY`と記録する。

C `fee61d272c3564ef393fecef5b738dbda825b658` の保存索引を無修正e839で読むと、新しいFull PF結果schemaを読めず停止した。[元ログ](BASELINE.log)。新schemaのreaderはCの同commitから再利用し、人工ROUND_RESULTの科学catalogへの混入を拒否する1検査だけを追加した。変更前の動作確認はe839本来のcatalog `417a73e7329fd520381dc88cc0de00a11dd0b641` でも実行しており、両catalogの差を選択能力の改善と混同しない。

## 実装と物理条件

- `pf_candidate_space.py`：既存のcanonical PF mapperを使うmetadata専用の列挙。B0I p0〜9、p10は既存Full PFのcanonical alias。p0は一体項だけを保持する。保存fragment順、signed formula、exact時間・q・actual幅、元Full DF-H target、P0除外、compiler/control/wrapperのdescriptorを保つ。保存Full20構成とT2の42 cellのIDを再現した。
- `pf_candidate_selection.py`：未準備候補を選び、探索器自身が `pf_completion.selected=[{configuration_body:…}]` を構成する。選択と応答を同runに保存し、再開で同じcandidate identityを使う。Dの元mode plan/completeを再利用する。
- `pf_optimizer_connection.py`：Dの既存registry・signal/price/template joinを高速化版へ移植した。対象候補自身の13 round信号・26軸価格・law/physical ID・入力bindingで受理する。異なるpの価格やp7の最良公式・幅を流用しない。scalarはFullのc_H、B0Iのc_H+s_R(p)という既存provider契約を維持し、数値guardは変更していない。今回実preparationの通過は検証していない。
- 通常CLI・O.prepare：prefix列挙、保存registry受入、自動選択・補完後の再prepareを追加した。shot生成・提案・候補ごとの費用比較は元B2実装のまま。固定generatorに対象候補自身の単価を渡す。旧入口にprefix scopeもpf_completion指定もない場合は追加選択を行わない。
- 旧runのLAUNCHを所有権として扱うDの既存修正も保持。出所不明や未完了のB所有jobを新規Q3へ変換せず、読取専用でIN_PROGRESS/STALEを返す。実行中のbranchやrawを変更していない。

選択規則は `UNSEEN_PREFIX_SAVED_EVIDENCE_V1`。未評価p、再利用できそうなT2 step descriptor数、保存T1の|shift|、signed piece数、粗い要求幅、IDの順に**取得要求を並べる**。T1/T2をQ3合否、誤差上界、価格下界にしない。p0〜6も除外しない。descriptor数は保存complex mean数ではない。正確な不足roundはDの元planが返す。これは安価な次候補選択の規則であり、選んだpが最安という科学的主張ではない。

今回の通常列挙は元PF indexにある3公式×各3要求幅をp0〜9へ展開した90候補。scope内の各pで公式・幅・shotを別々に選び直せる。mapper自体は任意の正の有理幅を受け付けるが、通常入口の自動幅探索はこの有限index内であり、連続幅の大域最適化は主張しない。

## 保存H4の計画と再開

[PROBLEM](PROBLEM.json)はe839のproblemにC fee61の索引locatorを結合し、[SAVED_PLAN_ROUND](SAVED_PLAN_ROUND.json)でB0I、p0〜10のscope、3公式、幅範囲、Q3予算0、D mode planを指定した。特定p・公式・幅・shot列の選択結果は入力にない。専用の読取SOURCE view（D36ad）で既存 `prepare_pf_candidate.py --mode plan` を呼び、実科学処理に入っていない。

[SELECTION](saved_plan/run01/pf_auto/SELECTION.json)は評価済みp=[7]、未評価p=[0,1,2,3,4,5,6,8,9]、p9/Yoshida4/32h/5を返した。[D要求](saved_plan/run01/pf_auto/pf_completion/REQUEST.json)、[D計画原本のcopy](saved_plan/run01/pf_auto/D_PLAN.json)、[要約](SAVED_PLAN_SUMMARY.json)から同じIDを追える。

- 保存信号再利用：m0,m1,m2,m4,m5,m12の6 round。
- 不足信号：m3,m6,m7,m8,m9,m10,m11の7 complex mean。
- 同prefixの直接価格は未確認。元D planの新規候補はm0〜m6,m8の8 cell×X/Y、上限16 compile。較正m5/m6・holdout m8を含む。これは既存planが返す未実行の案で、今回採択・許可した計算ではない。価格モデル・許容差も新たに採択していない。
- [通常探索結果](saved_plan/run01/ROUND_RESULT.json)の保存最安はp7/Y4/32h/5/A_SHAPEのまま372,035,410 RZ。未評価p9の不足をp7の信号・価格で埋めていない。
- 同コマンド・同runを再開し、選択・要求・応答・D計画のbytes不変、新科学call0件、provider再呼出しなしを確認した。[再開検査](SAVED_PLAN_RESUME_CHECK.json)。ROUND_RESULT全bytesはwall time等が更新されるので同一とはしていない。

## 人工の未評価p通し検証

[人工受入](artificial_comparison/CHECKS.json)を参照。実際の通常CLIの `main()` にargvを渡し、初期facts/catalogのみ人工入力に差し替える。comparison検査は人工契約用に置換する。一方、canonical列挙・選択・D helperのsubprocess・registry受入・固定shot generator・既存artificial Q3 worker・通常の方式別探索・再開は実コードを通る。保存H4の外部CLI実行とは区別する。

人工providerは候補を選ばれた**後**に、その候補自身の13信号metadata・26軸の人工単価・Q3 templateを返す。期待するpや新しいshot列をrunnerに選択結果として埋め込まない。複数のp・公式・幅から未評価pが選ばれ、復帰時のIDが一致し、shot提案のoriginが `GENERATOR_SEED` となったことを検査する。架空の適合点と単価は科学価格・実Q3の証拠ではない。

mode artificialはQ3 routeがすべてartificialのときだけ許し、registryにartificial印を付ける。Cの通常consumerは人工registryも人工ROUND_RESULTも拒否する。人工出力は当該検証directoryだけに置き、C_RESULT_JOINや本番catalogへ登録していない。legacy/concurrent双方と同runの再開を検査した。

## 回帰・性能・保全

検査一覧は [VALIDATION](VALIDATION.json)。prefix/target・ID・scope、旧job所有権、通常CLI・concurrent queue、既存の価格受入、shot規則を確認した。e839のproposals/method_state/axis_groups/run_methodはAST同一。[既存規則検査](EXISTING_SHOT_RULES_CHECK.json)。既存の並列queue、HYBRID、prefix-cache、DP worker設定を維持し、独立の技術検査を並行実行した。prefix-cacheは元固定worker view・Pythonで人工10件PASS。実H4での速度倍率は今回測定していない。

初期回帰ではPF registryのない旧mock contextへの適用漏れを検出し、旧scopeへの非適用とoptional registry参照を修正した。prefix-cacheテストを他のunittestと同processでimportすると外部固定viewのnamespaceを読めなかったため、元手順どおり専用processで実行しPASS。旧失敗ログも残す。これらを科学的な不適合とは扱わない。

[公開後続・既存job照合](PREEXISTING_WORK.json)：D36ad、C fee61、B2 d8e806、高速化e839の公開headを確認。既知のB p7 runは313 LAUNCH中312完了、未完了1件を所有者に残した。未解決jobを収集・再投入せず、別worktreeで本作業を行った。今回の照合はこれら関連SOURCE・既知runに限定し、他研究の全成果の不在を主張しない。将来の実取得には選択後の物理round/軸について§10.5を差分確認する。

実装とprefix/互換性・人工通し検証は2名のsub-agentと分担した。各実装の作者自身による検査を含み、全検査を独立reviewと呼ばない。既存C/D/B2 branch、固定SOURCE、raw、凍結モデル、旧STOPを変更していない。

## ロードマップへの差分とSTOP

解消したのは「pを呼出し側で選んでselectedへ書く必要がある」という技術接続の欠落である。保存集合内の実最安はp7のみという証拠状態は不変。p9等の信号・価格準備、対象自身の実Q3、費用比較の科学検証は残る。次段階は選択済みの有限不足planを確認し、必要分だけ別途承認後に実行する。ロードマップ本体・本番catalogは統合担当へ渡し、ここでは変更しない。**技術成果の公開後mandatory STOP。実科学計算には進まない。**

実装固定SOURCE：`d6f383715e05cab3018e8b4212816db06b74fb21`。ローカル技術検査時のbase HEAD表示と最終実装SOURCEの対応は [SOURCE.json](SOURCE.json) を参照。独立CI・外部再現の結果とはしていない。
