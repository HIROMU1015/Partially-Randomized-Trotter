> 2026-10-18追補：この準備SOURCEによるrun01はmetadata参照ミスでSTOPした。実sampling/compile0、one-shot消費済み。[停止報告と限定修正](track_a_h8_conditional_s2_pilot_STOP_20261018.md)を現在の状態として参照する。以下は実行前の固定契約。

# H8 PR s2条件付きpilot：実装・実行契約

利用者の2026-10-17指示により、q4とq8の2候補についてs2条件付きtrajectoryを各8本、cosine/sine計32軸以内で一回取得する。これは元n32とは別の費用診断であり、母平均保証・PR削減率の再計算ではない。科学retry/resumeは認可されていない。A(q4)の8本を保存・検証・native回収してからB(q8)へ進む。

## 固定条件と来歴

Linear H8、1.00 Å、STO-3G、保存tol-only DF rank28、16 spin modes、sector4900、T=.8、prefix7、S2、r4、K4、q4/R16とq8/R32。state preparation除外のnative symmetric-directional controlled whole Hadamard wrapperを、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topology/backend/layout/routingなしでcompileする。

元H8 [resource map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md)、元science SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、[直近限定再会計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md)を保持する。[H6先例](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md)の費用をH8へ移送しない。H8 D0/Phase C/非相殺/6軸の既存成果を変更しない。

元compiler SHA `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、profile SHA `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。バージョン番号だけでなく固定provider265件・既存runtime34,273件のbyte hashを照合する。旧hostとのcompiler output完全同一性は認定しない。

## 条件付きlaw

元の次数確率をp0,p2,p4、R=qrとする。s2はno4かつat least one2。pi0=p0^R、pi2=(p0+p2)^R-p0^R、pi4=1-(p0+p2)^Rを使用する。N2をBinomial(R,p2/(p0+p2))のN2>0条件付きから直接抽出し、R位置からN2位置を一様抽出する。任意の次数列の実装確率は `P(N2=m|N2>0)/comb(R,m) = p2^m*p0^(R-m)/pi2`。指数・log1p/expm1・log-sum-expで小さい質量を扱う。無条件棄却・単一2次event強制はしない。

各eventのcomponentは元probability、k+1個、元`_make_event`のrotation/product順・符号・basis・quantum weight/normalizationを使用する。新しいmaster/order/stepの128 seedを事前固定し、元master784件・component step3056件および新stream相互の衝突を拒否する。seedのdomain分離と衝突不在は、統計的独立性の証明ではない。

元native `h8_fast_cost_port_v1.cost_task`をscoped bindingし、sampled_stepsだけ条件付き抽出へ置換する。保存DFの元native表現をmaterializeし、元preparation/partition/lambda/scalar hashと完全一致を要求する。これはSCF・DF・state生成やnormalization/bias/shotの新しい解析ではない。native生成したevolution一つを保持してpaired axisへ使用し、全wrapperを直接compileする。部分費用の和へ置換しない。全軸でevent digest・canonical wrapper fingerprint・compiler hashを保存する。

各構成の先頭trajectoryで元方式の4 logical control probesを行う。証明可能なancilla構造ではjoint-branch最適化、不明なら元のtwo-branch fallbackを使用する。後続trajectoryは元登録方針同様、固定native constructorを用い追加control probeを行わない。全標本を独立に数値control検証したとの主張はしない。

## 科学予算と実行制御

新規最大はcompile32、trajectory16、occurrence96、control_probe8、primitive/reference/signal/normalization0。occurrence ledgerの単位は元contractのouter q callであり、event数Rではない（eventは合計384）。過去known1572compile/768trajectory、held22/11を保持し、新規全予約後1626/795、occurrence3163、control372で元cap1722/843/4219/376以内。primitive2172・normalization72等の既消費も返却しない。[固定証拠](../../references/h8_conditional_s2_evidence_v1.json)のcarry・unknown actual null・heldを保持する。後続の別認可H8 stageのledgerは別namespaceの既存証跡として保全し、このsubtotalを全プロジェクト総消費と呼ばない。

全trajectoryをdispatch前に非返却予約し、worker内のcompile entry/actual counts、未完了予約を区別する。固定ID以外、重複、無効grant、入力/source変更、lock/output衝突を拒否する。科学STOP後は未完了標本を補わない。最初のSTOPを保存しowned PID/start/parent/uidとpidfdでTERM/KILL、wait4/reap、FD終了を記録する。元STOP・元grant・結果・他worktree・M1-Sへ書込/signalしない。

## 資源と人工検証

実空きCPU・有効メモリをfresh gateで確認し、各候補内で最大8 trajectoryを並行処理する。各workerはown affinity1 core・内部thread1で、両軸を同じevent列から順にcompileする。CPU/RAM/wallのエージェント独自capは導入せず、OS制御を維持する。元s0標本の同候補RSSはq4約0.72GiB、q8約1.00GiBであった。s2についてはworker当り4GiBとheadroom16GiBを容量見積りに用い、実測上限とは扱わない。出力2GiB/inode8000の見積り、filesystemとown allocation probeを確認する。数値quotaは不明なら不明と記録し、無制限としない。PSIは観測し、共有資源の一時的競合だけでSTOPしない。

人工検査は確率の列挙同値性、複数2次/位置/小確率、元event constructor、人工16-mode K4/q4/q8のnative接続、wrapper canonical一致、人工2 compileの費用定義、無効grant・改変・予約・one-shot・最小child回収を対象とする。実H8入力を人工fixtureへ使わない。最終SOURCE/test数と結果は実行binding/結果報告に保存する。主担当による検査であり、別担当の独立reviewとは呼ばない。subagent0。

## 実行許可と成果の限定

SOURCE/testsを固定し、無効grant拒否を確認後、保存した利用者指示のbyte hashを一回限りの新grantに結合する。新SOURCEの公開は可能なら実行前に試みるが、今回指示では認証失敗そのものを追加承認待ちgateにしない。認証設定を変更しない。未使用output/control/lock、source/input/runtime/compiler、fresh資源条件がPASSした場合のみ一回実行する。

各qについてs2 n8のaxis別とpaired sumの平均・標本SD・SEを保存する。元n32と混合せずn40にしない。s4費用null、母平均・希少tail・選択後効果・全体PR削減率は未解決。追加sampling/compile、signal、shot、fit、再会計、他分子へ自動移行せずmandatory STOPする。

次の担当：Codexが今回の明示認可scopeを一回実行・保存し、ユーザー/GPTが残る統計的判断を行う。

## 固定SOURCEと検査結果

SOURCE `3db51df7bdb4e28968f0bd4d5af22d8e4553fe56`、closure8（新規7＋既存owned fork pool1）。人工22検査PASS、最終契約16件の限定再確認PASS、人工compile2・実H8 sampling/compile0（grant発行前）。無効grantの実CLIはexit1で科学import/lock前拒否。runtime/provider34,538件・1,508,760,538 bytes照合PASS。過大な初期人工fixtureは科学前に中断し、疎な最終fixtureで接続検査を完了した。共有設定は変更せず、commit著者だけ既存と同じCodex/codex@localhostをコマンド単位で指定した。

[準備bundle](../../artifacts/resource_applicability/track_a_h8_conditional_s2_pilot_readiness/2026-10-17/README.md)。有効grant・fresh gate・実行結果のbindingは次の結果bundleへ記録する。
