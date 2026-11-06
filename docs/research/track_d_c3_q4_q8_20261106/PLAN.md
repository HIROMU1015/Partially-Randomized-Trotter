# Track D Stage C3: q=4・8 whole-path固定検証

2026-11-06 JST。利用者のC3指示を実行範囲とする。C2結果commit `c8fe5091c42ec786688b6931105ad64d10c87d18`から専用branchを作成し、本PLANと新SOURCEをcommitした後に一度だけ実行する。過去のSOURCE・結果・grant・ledger・marker・STOPは変更しない。本検証は主agentが実施し、別agentによる独立レビューではない。

## 科学条件と固定fixture

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、各h=δ=1/10。元H厳密groundを仮定し状態準備費は除外する。ground vectorは使わず全256 system入力列でobservableを確認する。

保存cohort `near_practical_PR_r32_fixed32_v1`、endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`を用いる。元science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`の143 SOURCE・16入力と追加law、保存carrier `3649b75af191916daf44abb91e14d34c43b83e8f`の96保存bindingsは既存preflightで照合する。signed係数、event順序、frame、normalization、scalar/identity、λ_Rを維持する。元SOURCEは別の既存読取用tree、Python環境も既存専用環境を再利用する。

|q|時間順sample|global cut|A・B|
|---|---|---|---|
|4|00→01→02→03|complete step2後|B=00..01、A=02..03|
|8|00→01→…→07|complete step4後|B=00..03、A=04..07|

V=U_last … U_first=A B。各全ordered pathを一つのReference gadgetで扱う。cut・fixtureを価格や検証結果によって変更しない。短いgadget反復はしない。構造・価格用固定fixtureでありfresh-IIDの価格標本ではない。

## 再利用と必要な差分

- Stage Aの`track_d_h4_paired_reference.prepare_common`、saved event復元・位相会計、full-frame tailの`build_event_sequence(optimized=True)`を再利用する。
- `StrongBuilder`のGaussian `_basis`、位相保持relative `transition`、`combined_diagonal`と既存ordinary/uncontrolled/directional中心emitterを使う。
- repeated builder `df_partial_s2_repeated._build_boundary_optimized`および人工Track Dの既存規則に従い、step境界の同block half-sweepを融合する。directional halfの後のuncontrolled halfはordinary controlled fullとなり、directional fullではない。
- StrongBuilder.sequenceとdirectional_headは各々一種類の制御しか受け付けないため、新moduleの`mixed_head`だけが制御mode混在に対応する。新しいGaussian合成・対角gate分解・compile・state-action・CAR生成器は実装しない。
- B1/C1のCAR積とAB/adjointのwhole-path構成、Stage Aの22 precheck＋paired X/Y、C2の`compile_one`/`verify_one`/workerを呼ぶ。compile後も同じstate-action harnessを使う。

strong symmetricは初回forward uncontrolled、各tail通常制御、reverse directional、境界ordinary full＋合法なGaussian relative遷移を使う。q−1境界を融合しscalar/identityを一つのancilla Pに集約する。全head Gaussianはuncontrolledで、head相対融合・tail同basis共有を保持する。新たなhead-tail横断融合は導入しない。

ReferenceはA/B内部の各step境界を同様に融合し、各半path内のscalar/identityをglobal phaseへ集約する。Aのactual inverseをgadgetへ使う。内部融合は合計q−2境界で、global cutの1境界は分割して保持する。16 CSWAP、8追加reference qubit、χ補正を含む全wrapperで測る。融合節約値を別実装から足し合わせない。

C2のq2は保存strong coreの単純連結で、新しいcross-step手動融合を含まない。q1/q2は保存価格の再掲のみで再compileしない。q・path・cut・融合条件が異なるため差をq単独の因果効果としない。

## 検証・実行順序

1. 人工入力だけで固定recipe、制御両branch、非零scalar位相の境界融合、結果比較の分離をテストする。SOURCE/PLANをcommitし、実行前のbyte identityを確認する。
2. q4/q8のraw構築・検証を独立processで並行実施する。保存CAR行列を時間順積し、保存ordinary rawを連結した独立対照とも比較する。AB=V、A†、unitarity、真空ray・解析的χ・split位相、strong両branch、Reference全complex coherence、X/Y誘導observable、leakage、χ省略negative controlを既存harnessで確認する。
3. 両raw correctnessと公平な対照がPASSした場合だけ、各q×2方式×X/Yの計8wrapperを同じP0で一度ずつcompileする。失敗が出たら新たなcompile/postcheckを開始せず、既に開始済みの独立処理は結果を保全する。
4. compile後にCAR/rawとのX/Y一致、wrapper位相、reference leakage、paired observable、真空coherence位相を照合する。RZ count/depth、CX count/depth、total depth、qubits、circuit sizeを別記する。
5. SOURCE・入力再照合、raw/compiled QPY、NPZ、検証JSON、全資源比較を保全して結果文書を作成し、公開byte照合後mandatory STOP。

P0はQiskit1.3.0、optimization_level=1、seed_transpiler=17、basis=rz/sx/x/cx、topology-free、位相保持。精度は既存binary64・component abs1e−10、observable Frobenius/signal2e−10、rtol0、global phase fit無し。失敗後の閾値緩和・retry/resumeはしない。全17qubit Operatorは構成せず、全256入力・各131072出力振幅のstate-actionを使う。数値PASSは丸め誤差の厳密包絡ではない。

実空きCPU/affinity/RAMを観測し、raw2taskとcompile/postcheck最大8taskに配分する。C2の実測peak RSSをtask当たりの見積りに使う。小行列BLASは重複並列を避け、重いstate-actionへ観測CPUを配分する。人工的CPU/RAM上限、共有設定変更、他ユーザーjobへの操作は行わない。資源観測にはprivate path/hostname/ユーザー名を含めない。

## STOP・解釈境界

固定SOURCE/入力/phase不整合、correctness失敗、公平なstrong baseline不成立なら該当実行をSTOPし、別fixture・cut・精度へ変更しない。新sampling、別H/endpoint、DF再fit、ground solve、B2、q>8、N/成功確率算定、13-round総費用、詳細科学レビューは対象外。q1/q2の保存結果には手を加えない。

RZをTへ換算せず、topology-free depthを実時間としない。固定fixtureの価格を全law母平均・信頼区間・全RPE費用の保証へ外挿しない。routing、合成精度、全law、実装誤差・Nとの接続には別判断・承認が必要である。

成果物rootは`artifacts/track_d_c3_q4_q8/2026-11-06/run01/`。新SOURCEは`track_d_c3_whole_path.py`、runnerは`run_track_d_c3_q4_q8.py`、人工testは`check_track_d_c3.py`。実行時SOURCE commitと入力hashはSOURCE.jsonに保存する。既存STOPを維持し今回のSTOPを別出力へ記録する。公開後はユーザー／GPTが次の研究判断を行い、Codexは追加計算を実施しない。
