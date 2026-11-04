# 全18round Controlled Hadamard接続：上界取得、18軸は予算内未保証

2026-11-05 JST / LOCAL / 1 pass。
**全18roundの追加転送上界を構成したが、保存εへのstrict十分条件は18/36軸だけ成立した。
統合判定はUNDETERMINED。追加計算・retry・precision変更は行わずSTOPした。**

runnerはCOMPLETED_MANDATORY_STOP、stage=complete、worker/descriptor/入力エラーなし。
native exit2は全36軸の条件が揃わなかったことを表し、処理途中のcrashではない。
全体RPE_tolerance_certified=false、scientific_status=UNDETERMINED、
U_virtual_IID_upper=null、U_exec_upper=nullを維持する。

## 固定SOURCE・条件・入口

実行SOURCEは5605ea6fdb90fb7de2dac8768d392c81435217a3。
[固定科学計画](../rpe_controlled_connection_runner_20261104/SCIENTIFIC_EXECUTION_PLAN.md)、
[固定bindings](../rpe_controlled_connection_runner_20261104/SOURCE_INPUTS.json)、
[人工147件の技術報告](../rpe_controlled_connection_runner_20261104/IMPLEMENTATION_REPORT.md)を再利用した。
SOURCE・旧入力・旧結果は無変更。専用branchは
codex/rpe-controlled-connection-certificate-execution-20261105。

Linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D=3、単一δ=.02、
8 system qubit・4電子sector70、B保存physical stateのnormalized ray、
extract_identity_phase、coefficient_atol=1e-12、K2。
m0..17・q=2^m、保存r1/2/4/8/16/32、N/α/β/ε・signal側τ・保存θ・uniform53 law、
component/basis/sign/support_for_source_basis_run_le_1とtraining fingerprintを固定した。

C run02（82b2df6af、科学SOURCE7279f700a）のbasis/proof identity、
全18categorical結果d2e7a7db6（SOURCEf4905fff5）、
そのm17再利用元34840ce9bを区別して照合した。
旧36軸のcategorical包含・shot幅・判定は再計算せず、保存値を入力に使った。
今回は、その保存U_catに新しいΔ_extraを加えた36 strict比較だけを新規実行した。

~~~sh
/usr/bin/time -v -o artifacts/rpe_controlled_connection_certificate/2026-11-05/run01.resources.txt \
 /tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python \
 docs/research/rpe_controlled_connection_runner_20261104/run_connection.py \
 --run --output artifacts/rpe_controlled_connection_certificate/2026-11-05/run01 \
 > artifacts/rpe_controlled_connection_certificate/2026-11-05/run01.stdout.log \
 2> artifacts/rpe_controlled_connection_certificate/2026-11-05/run01.stderr.log
~~~

defaultのbyte/schema/literal照合、freeze/runtime確認、fresh directory確認後に上記を1回だけ起動した。
H/stateの再生成、state load/norm²再計算、新mean・law・CDF・atan、trajectory生成、
quantum shot・sampling、many-body evolution、長q回路build/transpile、D・費用評価は0。
既存DF preparationとsupport登録内のorbital処理・local fingerprintは固定計画内に含む。
これまでの旧C run01/run02は再実行していない。

## 得られた上界とstrict判定

全4 half primitiveは保存exportへhex一致。block0 fullとのd_b=0。
18個のscalar差d_sをexact rationalで取得した。
Δ_boundaryの最大はm17で約6.5369932e-13（下表は上向き表示）。
full11＋selected218の全descriptorにclass/wire/parameter/hash・Gaussian/JW隣接条件を確認した。
172 candidate slotを141 unique controlled constructorで監査し、全条件を確認したため
固定logical gateの範囲でΔ_lowering=0となった。

全218成分についてcertified column差からCAR上界Δ_Sを構成した。
最大component上端はdf-fragment-7:Z6Z7の

    94226915901388679050191725869 / 1329227995784915872903807060280344576

で、上向き8桁表示は7.0888453e-8。
これは演算子差の上界であり、実際の回路biasの観測値・下界ではない。
保存preserved-column residual=0を、そのまま合成gateの誤差0として扱っていない。

各mでΔ_extra=min(2,Δ_boundary+3qr max Δ_S+Δ_lowering)を構成し、
保存U_cat_upper＋Δ_extra＋保存shot半幅上端 < 保存εをFractionでstrict比較した。
π64 phase transferは既存I_cat/U_catに既に含まれるため、新たに加算していない。
保存N/α/β/εとshot半幅を変更・再配分していない。

| m（q=2^m） | r | Δ_boundary上端 | Δ_extra上端 | cosine | sine |
|---:|---:|---:|---:|---|---|
| 0 | 1 | 4.9873300E-18 | 2.1266536E-7 | CONFIRMED | CONFIRMED |
| 1 | 1 | 9.9746600E-18 | 4.2533072E-7 | CONFIRMED | CONFIRMED |
| 2 | 1 | 1.9949320E-17 | 8.5066144E-7 | CONFIRMED | CONFIRMED |
| 3 | 1 | 3.9898640E-17 | 0.0000017013229 | CONFIRMED | CONFIRMED |
| 4 | 1 | 7.9797280E-17 | 0.0000034026458 | CONFIRMED | CONFIRMED |
| 5 | 1 | 1.5959456E-16 | 0.0000068052915 | CONFIRMED | CONFIRMED |
| 6 | 1 | 3.1918912E-16 | 0.000013610583 | UNDETERMINED | UNDETERMINED |
| 7 | 1 | 6.3837824E-16 | 0.000027221166 | CONFIRMED | CONFIRMED |
| 8 | 2 | 1.2767565E-15 | 0.00010888467 | CONFIRMED | CONFIRMED |
| 9 | 2 | 2.5535130E-15 | 0.00021776933 | CONFIRMED | CONFIRMED |
| 10 | 4 | 5.1070260E-15 | 0.00087107731 | UNDETERMINED | UNDETERMINED |
| 11 | 8 | 1.0214052E-14 | 0.0034843093 | UNDETERMINED | UNDETERMINED |
| 12 | 8 | 2.0428104E-14 | 0.0069686185 | UNDETERMINED | UNDETERMINED |
| 13 | 16 | 4.0856208E-14 | 0.027874474 | UNDETERMINED | UNDETERMINED |
| 14 | 16 | 8.1712415E-14 | 0.055748948 | UNDETERMINED | UNDETERMINED |
| 15 | 32 | 1.6342483E-13 | 0.22299580 | UNDETERMINED | UNDETERMINED |
| 16 | 32 | 3.2684966E-13 | 0.44599159 | UNDETERMINED | UNDETERMINED |
| 17 | 32 | 6.5369932E-13 | 0.89198317 | UNDETERMINED | UNDETERMINED |

表は原RESULTの正のrational上端を8有効桁へ上向きに表示したもの。
正式な上端・拡張interval・total_upper・ε・判定は
[AXIS_RESULTS.tsv](AXIS_RESULTS.tsv)の36行と原RESULTに保存した。
表作成・TSV転記でstrict判定や科学上界を再計算していない。

m0..5、m7..9の9round・18軸が予算内接続の十分条件を満たした。
m6、m10..17の9round・18軸はUNDETERMINED。
m17のsupport転送上端は上向き表示約0.89198317で、boundary上端より大きい。
この一様worst-component / 最大3qr適用の上界では全18roundの予算内保証に足りない。
真のmean biasがεを超えた、回路が不正である、またはsamplingが失敗する、と示す結果ではない。

## 科学call・資源・原本

| 操作 | 開始/実施数 |
|---|---:|
| H load／DF preparation／support登録 | 各1 |
| full／selected descriptor | 11／218 |
| half／block0 full primitive | 4／1 |
| local controlled constructor | 141（候補slot172、重複31はcache） |
| Gaussian一粒子包含 | 229（全て8×8・96 bit） |
| component bound | 218 |
| scalar/boundary／transfer／axis比較 | 18／18／36 |
| 新mean/CDF/atan、trajectory、sampling/shots、長q build/compile | 全て0 |
| 追加precision／科学retry/resume | 全て0 |

229 basisは全て異なるhashで、basis cacheの重複hitは0。
full basis行列をcomponentごとに再評価せず、得た包含を共有した。
32 spawn worker、各FLINT/BLAS1 thread、CPU affinity32。
起動観測はidle約99.53%、MemAvailable=55,918,891,008 bytes、memory pressure avg10=0。
既存venvのPython3.11・NumPy1.26.4・Qiskit1.3.0・python-flint0.9.0/FLINT3.6.0、
binary64・round-to-nearestを固定SOURCEへ照合した。
GPUは小さなanalytic Acb行列には選択しなかった。

runner elapsed=2.785409894771874秒、timeのwall=3.61秒。
timeのmaximum RSS=399244 KiBは全worker合計RSSとは解釈しない。
速度比較を行っていないためspeedup倍率は主張しない。
stderrのMatplotlib cache通知は一時directoryの使用に関するもので、共有環境の変更や計算失敗はない。

[VALIDATION.json](VALIDATION.json)に6原本のbytes/SHA256とcall・判定のliteral照合を記録した。
RESULT.jsonのSHA256は
00a066cce8acdc08dac5db4dafca404e84b21efa7189270d8c29c227e4ad15d1。
原本は[run01](../../../artifacts/rpe_controlled_connection_certificate/2026-11-05/run01/RESULT.json)、
同directoryのDESCRIPTORS.json・basis_enclosures.jsonlと、隣接stdout/stderr/resourcesの計6件。
229個の包含は全てcomplete、218 component・18round・36軸に欠測なし。
原RESULTを編集せず、保存表への転記とbyte/schema/literalチェックだけを実施した。
SOURCE無変更のため既存人工147 PASSを再利用し、科学結果を再計算するtestは追加していない。
これは実行者自身の技術照合であり、独立科学レビューではない。

## 保証できた範囲・残存事項・次段判断

今回の数学的前提・固定SOURCEの範囲で、全18roundについて
full-referenceからordinary-controlled boundary-optimized ideal circuitへの
任意合法trajectory一様追加誤差上界を得た。
同一の独立uniform53 lawで平均したときもその上界を転送できる。
その上で保存ε内のstrict十分条件を確認できたのは上記9roundだけである。

既存categoricalモデル全36軸の十分条件を撤回・変更した結果ではない。
support差を含むproduction側への接続に追加した上界が残余予算に収まるかを別に判定した。
Qiskitのstandard logical CCX/CX/CZ/Phaseと有限templateまでが対象で、
再帰native lowering、compiled/noisy、Float64仮想測定、固定PCG64の数学的IID性、
状態準備、Dのα/decoder/E0・RPE全体成功・費用は認定していない。
旧INCONSISTENT/u_num=nullや他の条件付き判定も変更しない。

次の提案は、Codexが別指示後にsupport column certificateと3qr max Δ_Sの保守性を
静的に整理し、必要ならより鋭い転送法のSOURCE・人工test・限定科学計画を用意すること。
今回の結果だけから不足の原因を特定したり、phase alignment・probability weighting・
higher precisionなどへ数値評価を追加したりしない。
新しい科学評価は新SOURCEと対象を固定・公開した後のユーザーの別途明示承認が必要。

原本・旧SOURCE・grant/ledger/marker/STOPを保全し、研究概要・状態・manifest・当日ノートへ結果を追加した。
公開SHAと原本hashの照合後mandatory STOP。現在の固定1 passの許可はここで終了する。
