# G10：同構造の次数比較・結果前契約

利用者採用GPT [G9 v2 review](../../research/track_b_G9_v2_scientific_review_20261010.md) §13-A–Dの一束を準備。
基点は受領commit `561cb7508f56e46eece0d6de40e4fdf713422451`、独立branch
`track-b-g10-degree-comparison-preparation-20261010`。旧G9 source/結果/消費済marker/authorization/STOPは保持。
今回の準備は登録science runや実synthesisを実施せず、source-bound別launchの前に止まる。
機械契約は[contract_v1.json](../../../artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json)。

## 固定target・arm

p=(1/5,3/10,1/2)、x=5/7、G9と同じ3-system-qubit provider：
Q0=Z0、V1=R_XX01(pi/4)、V2=R_XX12(pi/4) R_ZZ01(pi/4)、Qi=Vi† Zi Vi、右が先。
R_P(theta)=exp(-i theta P/2)、exact Clifford+T provider delta=0。
これはknown/development synthetic context、Pauli情報もcheap I1。分子/DF/I0 acquisitionの優位条件ではない。
各次数のtargetは同じ有限full first operator moment P_m(-i x sum p_i Qi)。exponential accuracy比較ではない。

| m | direct arms | 処理 |
|---|---|---|
| 3 | ordinary / partial P3 / closed P3 / general full / literal CTS | 5新規row |
| 5 | G9のordinary / partial+tail / closed P3+tail / general full / closed P5 full / literal CTS | 6保存rowの共通policy再会計だけ |
| 7 | ordinary / partial P3+tail / closed P3+tail / general full / closed P5+tail / literal CTS | 6新規row |

計17 rows/34 axes。helperの5診断rowはG9に保持し、新G10 primaryへ追加しない。
同一次数内のT切片+K*共通非負prep Tをprimaryとする。CX/1Q/workspace/古典は別座標。
新しいmaterialityやscalar winnerは定義しない。科学的判断は一束後のGPT。

## P5+tailとgeneralの意味論

closed P5のdigital係数を変更せず、degrees6/7 ordinary pairを追加し、全group lawを同時にdyadic化する。
新global group probability/旧local group probabilityでproposalを置き換え、weight=coefficient/proposalを保つ。
P5部分はfull P5、追加部分は(−ixR)^6/6!+(−ixR)^7/7!なので同P7に戻る。
wordの隣接Q² cancellationは全arm共通、degree6の(−1)^3相対位相は保持。
P3/P5閉形式は低次数standard path、generalはunknown B_newを使わないlocal zero-fill。
P5+tailのroot算術はP5 O(L²)+O(m) tail、digital largest remainderのsortはgroup数に対し別費用。
selectorは線形走査であることをsource/会計に明示する。

literal CTSは同じP_mをQ(sqrt2)でPauli収集し、real correction（identityも）を別event、oddを一共通角へpairする。
元G9 CTSのm5とoff-domainでexact同一の有限式を確認した。channel平均だけの一致ではない。
新G10はcost-aware finite lawの探索を追加しない。全固定辞書へ同じ任意proposal解析下界を保存し、未分離は未分離と報告する。

## confidence・誤差・reuse

Re/Im各epsilon=1/200。34 axes alpha=49/34000で0.049、17 resource rows beta=1/17000で0.001、familywise0.05。
resource tail t=10、exp(10)>17000を有理算術で確認できる。accepted tailとhard2N attemptsを分ける。
root K256 / probability H160、rho=eta=10^-12、同一strict Rz epsilon10^-6。
全mでnorm<8、係数mean error<=8rho、共通bias=2[8rho+8(1+rho)2epsilon_Rz]。
κ=(1+rho)^2/(1-eta)^(m+2)、canonical m2<=κ B²、local m2<=κ B U、range<=(1+rho)B/(1-eta)^(m+2)。
N=ceil(log_upper(2/alpha)[2m2/s²+4range/(3s)])。matrix point signalをN削減へ使わない。
phase/順序用1e-10 matrix診断はconfidenceのcertificateではなく、analytic interval/composition上界を使う。

m5はG9の原event/cost/phase/error資料をそのまま読み、共通alpha/betaのNと費用だけを保存値から再会計する。
旧native合成、matrix guard、source runnerは再実行しない。元G9の22-axis結果/classificationも保持。
19 saved primitive keysはangle/epsilon/tool/strict phase/sequence/count/guard identityを照合して共有する。
他次数にも同じstrict precision/compilerを適用する。負角はactual adjoint、W scalarとcontrolled相対位相を保持。

## 結果前inventory規則と資源上限

新角度はsource固定の式と登録domainから一意に定まるrational tangentsだけ。
run内部で固定armの全必要keyを確定してから合成を開始する。別angle/precision/seed探索を行わない。
静的上界：m3は7 full parents+2 ordinary+4 closed P3+1 partial root+1 CTS=15 angles。
m7は127 parents+4 ordinary+4 closed P3+1 partial root+10 P5 groups+1 CTS=147。
新162 upper＋旧19=181、overlapを無視した上界。cache192 entries/4MiB、新synthesis<=162。
small reference総event上界11319<12000。productionには表を渡さない。
output128MiBはこの有限supportのlong rational bindingsを保存する上限で、勝敗を見て拡張しない。

既存isolated SP05 runtime/pygridsynth2.0.0の同identity/options/seed0。
one process、wall1200s/CPU900s、RSS512MiB/AS1536MiB、per-key wall30s/CPU20s、sequence20000chars、N/axis<=10^8。
capacity/timeout/guard failureはtechnical STOP。prefix比較を最終科学outcomeに使わない。

## 非列挙性と取得費用

new degreeの各production armを64固定bit-interface試行で診断する（9 arms、576 trials）。
seedは契約文字列からm/armを付す一規則で固定。quantum trajectory samplingではなく、統計/速さ/scaling inferenceにも使わない。
productionはp/x/mだけを受け、ref event table、global B_new、native cost table、signalを読まない。
参照表は別層でmatrix意味論/期待native会計に使う。reference列挙からclassical scaling優位を宣言しない。

local kernelのprecomputeはO(Lm²)有理算術、parent queryはO(m³+Lm²)と有限bit root/law cost。
P5 group rootはO(L²)、largest remainders比較sortとlinear selectorは別。
runnerはproduction group/law構築、Pauli acquisition、参照列挙、saved m5再会計、matrix/native参照会計のwall/CPUを
`classical_accounting`へ分けて保存する。angle合成/cache validationは各inventory keyへwall/CPU、取得区分、identityを保存する。
固定interface traceにはbit消費、root/probability幅、係数/weightの整数bit長を保存する。
cheap I1のtoyで得た時間を一般DF取得費用に外挿しない。

## source binding・one-shot・STOP

新source Sを固定・公開後、Sの直接子authorization-only Aから実施する。
変更許可は新authorization JSONと任意receiptだけ。remote execution branch SHA=A、clean、source/contract/runtime/protected hashをgateで確認。
新独立result directoryへexclusive markerを測定前作成する。消費後は結果が何であれretry0、旧G9 marker/authorityの流用不可。
準備authorizationはpending/null/false、runnerはscience import・data/marker accessより前に拒否する。

完了分類は`G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`、不完了は`G10_TECHNICAL_INCONCLUSIVE`。
前者もalgorithm GO/新規性を意味しない。全結果でmandatory STOP、GPTへ必要証拠をcommit/pushして戻す。
m9、新p/x/provider、new precision/backend/seed、solver、DF/分子/NPZ/GPU、実量子shots/trajectory、旧主線救済を禁止する。
