# Track D C5：共通whole-path入口と固定q2の公平比較

2026-11-06 JST。利用者のC5指示そのものを今回の実行許可とする。基点はC4 `a5784c4bd3e3fb9779849a1e1fc1759559e2caea`、専用branch `codex/track-d-c5-whole-path-q2-20261106`。新SOURCE・本PLAN・人工testをcommit/pushしremote bytes照合後、固定q2を一度実行する。新grantや承認証明は作らない。

linear H4/隣接1.00Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、各h=δ=.1。元H厳密ground仮定・準備費除外。cohort `near_practical_PR_r32_fixed32_v1` のsample00→01だけ、q2、64 events、V=U01 U00=A B、B=00/A=01、complete step1後cut。endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、元input `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5` を保持する。

## 最小差分と再利用

- C3の`prepare`のbodyを`prepare_ordered(context,saved,ordered_samples,cut,fixture_selection=...)`へ抽出し、固定q4/q8入口は同じmap/cut/metadataで委譲する。primitive演算・最適化規則は変えない。targetのdimension表現は同じ8-mode値を保ち、人工2-mode回帰でも使えるようにする。raw保存/checker呼出しも同じhelperへ抽出する。
- 新`track_d_whole_path_price`はSOURCE/endpoint/cohort/law/inputを持つ明示request、共通bundleからのmethod/axis選択、有限fixtureの価格recordだけを追加する。旧kernel/shell scope、未知q/fixture、未PASS価格を受理せず、law期待値・CI・モデル価格はnull。共通入口は保存cohortの明示列用で、将来のfresh sampling接続を実装済みとはしない。
- Stage A `prepare_common`、StrongBuilder、C3 `mixed_head/physical_segment`、Reference gadget、actual inverse、CAR/checker、C2のP0 compile/postcheckを再利用する。Gaussian合成器・sampler・compiler・state simulatorは新設しない。
- 既存C3/C4のworktree・固定commit・結果・STOPは保全し、新revisionの抽出差分をC5で記録する。q4/q8を分子入力で再構成・再compileしない。C3固定入口の回帰は2-mode人工入力で旧cc554 SOURCEと比較する。

## 人工検証

requestの順序/q/cut/identity拒否、同requestの方式/axis選択、AB、actual adjoint、vacuum/χ、scalar/identity一回、異なるGaussian basisとsigned tail、X/Y coherence、合法な半step融合、cutの1境界損失、full-wrapper費用境界、未検証q/未資格価格、C3固定入口を検査する。既存C3の4人工test、C2の2test、Stage Aの2testも再利用する。分子prepare、sampling、P0 transpileやAer科学実行を人工testへ混入しない。人工fixtureの実装不備は科学実行前に修正・確認し、人工PASSをH4性能結果にしない。

## 一度の科学実行

1. 既存C2 preflightを再利用し、元science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` のclosure/input/law、保存carrier `3649b75af191916daf44abb91e14d34c43b83e8f`の96 cohort binding、旧C1/C2/Stage A、C3完了cc554のSOURCE/STOP/価格を照合する。新C5 SOURCEは現在commitに一致させる。
2. q2だけを共通入口で構成する。strong symmetricはuncontrolled Gaussian、directional control、head relative fusion、tail同basis共有、1境界のdirectional half＋uncontrolled half→ordinary controlled full融合、全path scalar/identity集約を保持する。Referenceは一つのgadget、Aのactual inverse、path全体χ、各step scalar/identity一回、global cutによる1境界融合損失を含む。normalization/attenuationをunitaryへ乗算しない。
3. 既存22 raw check＋paired X/Y、全256入力列のobservable/coherence、AB/adjoint/vacuum/phase/sign/scalarを照合する。両方式とfair baselineがPASS時だけ次へ進む。
4. 生成raw QPYをC2 archiveの同method/axis raw bytesと比較する。完全一致した場合だけ、hash確認済みC2 compiled bytesを再利用しcompile0と記録する。異なれば新SOURCEの別回路として各wrapper一回compileする。rawの名称等を変えて一致を作らない。再利用compiledも今回のraw/CAR・新対照との対応をpostcheckする。
5. 最大4 full-wrapper、P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initial-zero false/topology-free/位相保持。component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fitなし。失敗後の閾値緩和なし。
6. compile後のraw/CAR X/Y、paired observable、vacuum χ、reference leakageを確認し、RZ/CX count、各depth、total depth、size、logical peak qubitsを個別記録する。SOURCE/inputを再照合し、元C2・保存C3との差と条件差を記録する。母平均/fitは算定しない。

## 実行・公開・STOP

実資源は起動時に既存observerで読み、1 raw taskと最大4 compile/postcheck taskへ実空きCPU/RAMに応じて配分する。過去の低いthread/worker数を恒常上限にせず、共有OS/venv/scheduler/他jobを変更しない。起動は独立session・stdin切離し・ローカルlogとし、正常なjobをチャット切断や一時競合のため停止・再起動しない。

出力namespaceは`artifacts/track_d_c5_q2/2026-11-06/run01/`。既存checkerの未加工出力・CPU/RAM観測・cacheは同namespaceの非公開`local/`へ保持する。公開JSONは科学check/phase/metricsをそのまま転記したviewとし、実行時間・peak RSS・thread観測欄を除くことを明示する。元出力は編集しない。raw/compiled QPY・検証NPZはneutral metadataのlossless archiveに保存しmember bytesを照合する。非公開raw入力、内部環境path、資格情報、運用logを公開しない。

SOURCE/input/correctness/fairness/compile後一致の不成立時は依存する科学処理をSTOPし、partial/result/STOPを保全する。追加retry/resume、対象/cut/compiler変更は行わない。許可された範囲の成果を専用branchへ公開し、remote SHA/tree/必要file bytesを照合後mandatory STOP。C6、new sampling、価格校正/fit/holdout、q>2の新科学検証、N/成功率/全RPE総費用、B2/B4-2、ground solve/DF refit、詳細科学レビューへ進まない。次の研究判断はユーザー／GPTへ戻す。
