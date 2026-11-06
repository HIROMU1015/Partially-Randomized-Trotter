# Track D Stage C3：固定q=4・8 whole-path検証と中断補完

**両qのraw correctnessと全8 wrapperのcompile後検証、最終paired X/Y・真空位相照合がPASSした。SOURCE/inputを保全し、`TECHNICAL_PASS_MANDATORY_STOP`で科学実行を終了した。**

同じ固定H4・CDF8のq4/q8で、強いsymmetric-directionalとReferenceを比較した。回路構築・8 P0 compileのSOURCEは`52b3cab50d5c5edcbd7e3249637168e9978be535`、保存回路による補完SOURCEは`88f0496349c056d0736f08487889f942c1aa4d37`。[元PLAN](PLAN.md)、[run02 PLAN](RUN02_PLAN.md)、[補完PLAN](COMPLETION_PLAN.md)、[中断記録](RUN02_RESULT.md)を区別する。元run01の失敗とSTOPを上書きしない。

## 固定科学条件・SOURCE・入力

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、各h=δ=.1。元Hamiltonian Hの厳密基底状態を仮定し、状態準備費を除外する。ground vectorは使わず、全256 system入力列に対するobservableと出力振幅を照合する。

cohortは`near_practical_PR_r32_fixed32_v1`、endpoint SHA256は`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。元science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`の143 SOURCE・16入力・追加lawと、carrier `3649b75af191916daf44abb91e14d34c43b83e8f`の96保存bindingsを既存preflightで確認した。SOURCE・input/cohortごとの完全なbindingsは[run02 SOURCE](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run02/SOURCE.json)と[補完SOURCE](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/SOURCE.json)に保存する。

|q|固定した時間順sample|固定global cut|events|path全体χ|
|---|---|---|---:|---:|
|4|00→01→02→03|complete step2後|128|−0.9274769152225486|
|8|00→01→…→07|complete step4後|256|−1.8737054743169421|

V=U_last … U_first=A B、Bが前半・Aが後半。各whole ordered pathに一つのReference gadgetを使った。fixture/cutは価格観測前に固定し、短いgadgetのq回反復へ変更していない。signed係数、event順序、scalar/identity、normalization、attenuationを元SOURCEに一致させ、lawのnormalization/attenuationをunitaryへ重複適用していない。構造・価格用fixtureであり、fresh-IID全round価格標本ではない。

## 再利用した実装と強い対照

Stage Aのsaved-event復元、Reference adapter、StrongBuilder、Gaussian basis/relative遷移、full-frame tail emitter、CAR/checker、位相保持compilerを使用した。whole-path CAR積・actual adjointはC1、compile/postcheckはC2を再利用した。既存のhead関数は単一制御modeのみのため、C3の`mixed_head`だけがmode混在境界を扱う最小拡張である。新しいGaussian合成器・対角分解・CAR生成器・state simulatorは実装していない。

両方式でGaussian basisはuncontrolled、directional control、head relative fusion、tail同basis共有を保持する。既存repeated builderの合法な規則に従い、step境界のdirectional half＋uncontrolled halfをordinary controlled fullへ融合した。directional fullへ置換していない。scalar/identityをpath単位で集約する。

strong symmetricはq−1境界を融合する。ReferenceはA/B内部のq−2境界を同様に融合し、global cutで1境界の融合を失う。Aのactual inverse、各半pathのglobal phase、path全体χ補正を含めた。8追加reference qubits・16 CSWAP・readoutを含むfull-wrapperの実価格で比較する。両方式とも新しいhead-tail横断融合は導入していない。異なる実装の節約値を足し合わせていない。[各qのfusion記録](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/q8/fusion.json)を保存した。

## 中断と補完方法

run02ではraw2 fixtureがPASSし、8 wrapperを各1回compile、7 postcheckがPASSした。最後のq8 Reference Yはcompiled QPYとPASS markerのみが残り、最終result/STOPは未生成だった。該当processが確認できず、終了原因は特定できていない。correctness失敗の結果は記録されていない。

旧runnerはfresh出力専用だったため、SOURCEを変更せずに保存compiled回路を読む補完runnerを追加した。元run02の61科学ファイルを[partial input一覧](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run02_interrupted_inputs.json)に固定し、別の`run03_completion`へbyte一致で複製した。新しい回路生成・compileは0、完了済み7 postcheckの再実行は0。最後のq8 Reference Yのpostcheckを既存`verify_one`で1回実施し、両qの最終paired/χ検査を既存関数で実施した。元run02の未完成状態・旧run01の失敗/STOPを保持した。

独立session・ローカルlog・stdin切離しで補完を起動した。短い人工processでtool呼出し終了後の存続を確認した。この変更はチャットの接続寿命への依存を減らすもので、原因をSIGHUP/OOMと断定したり、外部停止を完全に防ぐとの保証ではない。新しい人工1testは複製時の元ファイル保全、上書き拒否、入力改変拒否を確認した。run02 SOURCEで実施済みの人工4testは別の保存証拠であり、人工PASSをH4の性能結果としない。

実空きCPU/RAM観測によりrun02はstate-action1thread、補完は63threadを使用した。後者は単一の残taskへ観測された空きCPUを配分したもので、旧runの1threadを恒常上限にしていない。共有設定・環境・他ユーザーのprocessやファイルには変更を加えていない。

## Correctness・compile後検証

component abs1e−10、observable Frobenius/signal2e−10、rtol0、global phase fitなしを全段階で維持した。長qのために閾値を緩和していない。P0はQiskit1.3.0・optimization_level1・seed17・rz/sx/x/cx・topology-free・位相保持、approximation_degree1.0、qubits_initially_zero=false。新しいP0 compile条件を導入していない。

rawは各22既存check＋paired X/Y：AB=V、Aのactual adjoint、unitarity、真空ray、解析的χ・split位相、scalar/identity、strong両branch、complex coherence、X/Y、χ省略negative controlを検証した。

|q|raw最大成分差|raw paired X Frobenius|raw paired Y Frobenius|raw leakage成分/L2|
|---|---:|---:|---:|---|
|4|3.197442310920451e−14|2.693982735683193e−13|1.8669118848676906e−13|0 / 0|
|8|6.084022174945858e−14|4.4213650373380394e−13|4.2886750297972857e−13|0 / 0|

|q・wrapper|判定|最大成分差|CAR observable Frobenius|reference leakage L2|
|---|---|---:|---:|---:|
|4 reference_X|PASS|2.04630422e-12|1.8171717e-11|2.52628452e-13|
|4 reference_Y|PASS|3.24377213e-12|7.5771227e-12|2.52628551e-13|
|4 symmetric_directional_X|PASS|1.51606298e-12|4.85447587e-13|非該当|
|4 symmetric_directional_Y|PASS|2.81428726e-12|1.10205879e-12|非該当|
|8 reference_X|PASS|9.22670901e-12|3.13128063e-11|4.74248684e-13|
|8 reference_Y|PASS|3.73355664e-12|2.2508436e-11|4.74248685e-13|
|8 symmetric_directional_X|PASS|8.76990724e-12|2.0281013e-12|非該当|
|8 symmetric_directional_Y|PASS|3.47969239e-12|3.37603397e-12|非該当|

|q|paired X成分 / Frobenius|paired Y成分 / Frobenius|vacuum χ成分差 symmetric / Reference|
|---|---|---|---|
|4|1.29196653e-12 / 1.84577849e-11|1.06548104e-12 / 7.67910857e-12|1.41015807e-14 / 1.24312962e-12|
|8|2.44515519e-12 / 3.17131656e-11|2.424283e-12 / 2.27603335e-11|2.38287719e-14 / 2.4390963e-12|

componentの許容差とobservable Frobeniusの許容差は別である。branch/isometryのFrobenius値をcomponentの1e−10と混同しない。全17-qubit Operatorを構成せず、既存state-actionで全256入力列・各131072出力振幅を扱った。compiled X/Y observableをCAR/raw/両方式間で照合し、vacuum χとreference leakageも検査した。これはlocal binary64照合であり、丸め誤差の厳密包絡・immutable CI・別実装による独立科学レビューではない。

## Full-wrapper資源と保存q1/q2との比較

X/Yで以下の全資源指標は一致した。各軸1 wrapperの価格を示す。

|q・方式|RZ count|CX count|RZ depth|CX depth|total depth|qubits|size|
|---|---:|---:|---:|---:|---:|---:|---:|
|1 symmetric（保存sample00）|5709|2140|1387|1034|3207|9|11594|
|1 reference（保存sample00）|6129|2136|753|495|1679|17|12244|
|2 symmetric（保存）|10773|4100|2624|1996|6096|9|21970|
|2 reference（保存）|11114|3860|1357|860|3014|17|22263|
|4 symmetric|19867|7686|4746|3744|11116|9|40572|
|4 reference|20092|6998|2516|1570|5583|17|40301|
|8 symmetric|39610|15360|9492|7515|22259|9|81015|
|8 reference|39602|13776|4531|2870|10093|17|79615|

Reference−symmetricのpaired差：

|q|ΔRZ|ΔCX|ΔRZ depth|ΔCX depth|Δtotal depth|Δqubits|Δsize|
|---|---:|---:|---:|---:|---:|---:|---:|
|4|+225|-688|-2230|-2174|-5533|+8|-271|
|8|-8|-1584|-4961|-4645|-12166|+8|-1400|

q4とq8の双方でCX count・CX depth・total depth削減が維持された。q4ではRZ countが225増える一方、q8では8だけ少なかった。q8の微小なRZ差はこのfixtureとcompilerの観測値であり、一般的なRZ削減やqによる因果効果を主張しない。両qともRZ depthは減った。RZ count、各depth、回路sizeを単一価格へ合成していない。

q1はStage Aのsample00（sample01の保存値も比較JSONに保持）、q2はC2のsample00→01を再利用し、再compileしなかった。q1のcutはcomplete event16後、q2はcomplete step1後、q4/q8はhalf-path complete step境界である。C2にはC3で追加した手動cross-step融合・全path位相集約は含まれない。各qでpath・cut・融合条件も異なるため、観測差をq単独の因果効果や漸近価格則としない。

q4/q8の全追加register・CSWAP・χ補正・cutによる融合損失は上表のfull-wrapper価格に含まれる。qubitsは9→17で一定の+8幅である。routingはtopology-freeのため未評価で、接続制約を課す場合のSWAP/CSWAP分解・depth増は別検証を要する。RZをT countへ換算していない。任意の合成精度のfault-tolerant価格、noise、実機時間も評価しておらず、topology-free depthを実行時間と同一視しない。

## 保存成果物・成立範囲・次の判断

[最終result](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/result.json)、[STOP](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/STOP.json)、[資源比較](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/resource_comparison.json)、[元出力inventory](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/inventory.json)を保存した。raw/compiled QPY 28本とNPZ 10本は[lossless archive索引](../../../artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/payload_archives.json)からq別tar.gzを取得できる。archiveと全memberのbytes/SHA256を元ファイルと照合し、tarのowner/path/mtimeは公開に不要な内部情報を含めない形とした。元run02のbinaryも同じmember bytesで保存される。科学JSON/回路/STOPは編集せず、非公開raw入力・内部log・PID・cacheは公開対象にしない。

最終SOURCE/input照合は一致し、元run02の61科学ファイルも固定一覧と一致した。8 compileは元run02で各1回、補完では新compile・回路生成0、最後のpostcheck1回、許可された中断補完1回、追加retry0。新sampling・ground solve・DF refit・N評価・量子shot・B2・q>8・全RPE総費用評価はすべて0／未実施。補完実行のwall timeは1841.614秒（約30.7分）、最後のpostcheckは1838.903秒だった。

CPU空き・メモリ等の内部観測を含む両runの`resources.json`はローカルに保全し、GitHub公開から除外した。自動承認レビューがその送信を却下したため、内部環境情報を公開しない条件を適用した。元inventory/input一覧は改変せず、公開上の省略は[公開範囲](../../../artifacts/track_d_c3_q4_q8/2026-11-06/PUBLICATION.md)に明記する。

成立したのは固定q4/q8でのwhole-path correctnessと同一P0のpaired資源比較までである。全lawの期待価格・母平均CIを推定せず、同一エネルギー精度・成功確率での全RPE優位性は判定していない。元H向け信号・N、全sampling lawでの経路分布と価格、長qの数値精度/回路構成、全roundへの移送・routing/合成価格に追加証拠が必要である。主資源指標・新たな精度/成功確率は固定していない。

公開後mandatory STOP。ユーザー／GPTが固定fixture証拠の採否と次の研究範囲を判断する。Codexは新たに明示された範囲だけを担当し、B2、追加q、新sampling、ground solve/DF refit、N/成功確率、13-round総費用、詳細科学レビューへ自動進行しない。
