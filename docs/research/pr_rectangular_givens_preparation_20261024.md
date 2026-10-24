# Rectangular Givens候補A：PS48の限定実装・実行前準備

**候補Aのsingleton専用実装と、保存PSの固定48-wrapperを実行する入口を準備した。科学実行は0件。**
技術状態は `READY_FOR_SEPARATE_PS48_SCIENTIFIC_APPROVAL`。同値性・削減量は未検証である。
新しい研究用grantは発行・消費していない。次はユーザー／GPTが、このSOURCEと下記の範囲で一回の科学実行を承認するか判断する。

## 固定対象・再利用

採択した[候補Aの設計](pr_gaussian_singleton_synthesis_design_20261024.md)のcommitは
`70be3c3bae09a8f99fecaa67757da1f8c4887267`。baselineはF2結果
`dc7378ab34015287215bcf4ce52147e46d94ecd9`、測定SOURCE
`9468c7089727f5648d9f59cddbaebe0a42f5f0a0`。
4 mode・2粒子・occupied(0,1)、L_D=1、残差2fragmentの固定合成Hamiltonian。
分子geometry／AO basis／分子DF rankは非該当、rank探索なし。
T=1/5、q=1、δ=1/5の一点、r=2、h=1/10、K=0。ここでF1は**残差fragment F1**。

| 既存資産 | 再利用した範囲 |
|---|---|
| `circuits.BasisProvider / build_event_sequence / build_evolution / hadamard_wrapper` | 基底の取得・actual inverse・controlled中心部・D/scalar/identity・順序・physical prep/readout。default providerの振る舞いは維持 |
| Nature0.7.2 `_prepare_slater_determinant_jw` | 選択列のtransposeをcreation係数とする既存のrectangular Gaussian分解。private helperと数値kernelのhash、依存版を固定 |
| `gates.matrix_gate / wrapper_signal / compiled_wrapper / compile_cache_key` | 全Fock空間の検査、位相を保持するF2修正版compiler、full IRのcache |
| `f2_execution.Worker.compile / build_job / compile_job / _attempt / circuit_metrics` | compile・logical caseごとの照合、journalと失敗時STOP、既存の資源定義 |
| `authorization.claim_packet` | 新scopeの承認を検査した後だけatomic one-shotを消費。旧grantの再利用なし |
| 保存B入力・F2 QPY／reference／case資源 | 元atom/frame、角度、二精度照合済みtarget、baseline費用。再生成しない |

新規実装は `rectangular.py`、`rectangular_preflight.py`、`rectangular_execution.py` と
`scripts/run_pr_rectangular_givens_saved.py`。
既存square-only Gaussian helperへrectangular行列を渡せないため、Gaussian suffix adapterと内部supportの接続が必要だった。
旧F2 executor全体は4arm全列挙・law/reference再生成を含むので呼ばず、適合する狭い処理だけ再利用した。
`BoundK0Rotation` は保存角度と元atomから作るcircuit向けのK0 event viewで、probability／lawの生成器ではない。

## 候補の接続

元frame U、元support S（1または2列）から `V=U[:,S]` を取り、作業用の `V.T.copy()` を固定Nature helperへ渡す。
helperが最初に返す所定のm個のXを、gate名・qubit・順序まで検査して消費し、その後の隣接XXPlusYYだけをGへ加える。
任意のXを一括で除く処理ではない。Slater状態準備のconstructorは使用しない。

内部supportは **T=tuple(range(len(S)))**。時間順に **G† → controlled中心回転(Z_T) → G** を接続する。
G†は同じGの `inverse()`。元atom ID・word・support・frame/basis key・係数は変更しない。
cacheは元frame・basis key・元S・policyで区別し、同じTを持つ別operatorを共有しない。
`BasisProvider.reference_support` のdefaultは元supportで、新providerだけがsingletonのFRAME_ZでTを返す。
同frame共有run、physical Pauli、identity、operator順序、元の符号・scalar/identity位相とsampling lawは従来経路を使う。
wrapperのphysical状態準備X0X1も従来どおり。

この規約を実際のGへ適用した結果が正しいかは未検証。
転置／共役・little-endian・Givens方向は、次回のprojectorと全operator検査で判定する。
Natureの既定`isclose`も固定許容差の保証とは扱わず、不一致ならSTOPする。

## PS48 case・角度・原本binding

[PLAN.json](../../artifacts/pr_rectangular_givens_preparation/2026-10-24/PLAN.json)を単一のcase／SOURCE／入力bindingとする。
48 ID、保存B4ファイル、選択baseline192ファイル（QPY・build JSON・reference NPY・compiled JSONを各48）、
公開F2 law／CSV／summary、SOURCE113ファイル、依存50件、Nature helper2ファイルのhashを固定した。
選択原本は旧raw inventoryのhashと照合済み。原本のQPY／NPYは今回decodeしていない。
非公開raw・承認情報・絶対locatorは公開せず、実行担当が保持する。GitHubだけでrawを再取得できるとは主張しない。

| PS index | 元atom | 元support | 内部T |
|---|---|---|---|
| 20 | S:F1:IIIZ | (3) | (0) |
| 21 | S:F1:IIZI | (2) | (0) |
| 22 | S:F1:IIZZ | (2,3) | (0,1) |
| 23 | S:F1:IZII | (1) | (0) |
| 24 | S:F1:IZIZ | (1,3) | (0,1) |
| 25 | S:F1:IZZI | (1,2) | (0,1) |

partnerは保存PS index0 `P:collect:IIII` とindex2 `P:collect:IXXI`。
6 atom × 2 partner × 2 operator順序 × X/Y = **48 wrapper要求**、support1/2は各24。
24 evolution、6種類のcandidate G、compile要求48、raw/compiled operator・signalはそれぞれ48、
新しい高精度reference評価・baseline build／compileは0。異なる作業単位を総科学callへ単純合算しない。
新しいsampling・simulationによるshot抽選、full-F2期待費用再集計、γ保証、認証N／N加重比較は含めない。

case IDは元と同じ `PS_{i:03}_{j:03}_{axis}`。次回承認後、候補Gを作る前に保存raw QPYをloadし、
既存 `append_central` の連続4命令 `RZ(a,t),CX(anc,t),RZ(-a,t),CX(anc,t)` にだけ角度をbindingする。
Gaussianの他のRZは角度として採らない。元support・case順序・係数符号、identity/scalarのancilla位相を照合し、
同K0の正のθが全48 caseで同じであることを確認する。λからθを再計算しない。
公開lawには独立θ欄がないので、**今回θの数値は未読・未確定**。曖昧なQPY構造、sign/phase/orderの不一致はSTOP対象。

保存 `build/<id>.reference128.npy` は元90/128桁独立referenceの一致を確認したtargetで、
hash・case・raw記録と結び、元X/Y reference信号値も使う。
不足・破損や新たな高精度評価が必要になれば未承認の再計算をせずSTOPする。

## 次回の合格条件と実行経路

1. 6個のGから全16次元operatorと一粒子作用Wを取得し、`W Π_T W† = V V†`、unitarity、actual adjointを1e-10で検査する。
2. 24 evolutionから48 full-wrapperを生成し、保存独立target32×32へ照合する。
   raw operator 1e-10、raw signal 2e-10、compiled operator/signal 5e-10、unitarity 1e-10、rtol=0を維持する。
   固定2粒子sectorだけの検査、global phase alignment、許容差緩和は使わない。
   全32×32のwrapper一致によりcontrol枝の相対位相も検査する。
3. Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree=1、topology-free、
   qubits_initially_zero=FalseとF2位相保持接続を使う。同一full IRのみcompile cacheし、全logical caseを検査する。
4. RZ/CX、total/RZ/CX/quantum-only depth、size、measureを保存baselineとcaseごとに比較する。
   subsetの結果を全F2の期待改善率・一般的優位・独立新規性とは扱わない。

実行時にはGを6個だけ親processで作って検査し、workerへ渡す。
独立build pair／compile groupを並列化し、worker数はCPU affinity・利用可能RAM・既存測定RSSと実job数から選ぶ。
thread設定は当該job内だけ調整。固定の小さいCPU／worker／wall／RAM上限や共有環境変更は行わない。
既存journalに実績・CPU/wall・peak RSSを保存し、失敗後の自動retry／resumeはしない。

今回実行した入口は、次のmetadata専用モードのみ。

```bash
<dedicated-python> scripts/run_pr_rectangular_givens_saved.py \
  --source-root <frozen-checkout> --saved-input-root <saved-B> \
  --baseline-root <saved-F2-raw> --check-only
```

このモードはgrantを要求・消費せず、SOURCE／入力hashとcase記録だけを照合し、科学packageをimportしない。
科学実行モードは別途 `--execute --packet <new-packet> --output <new-output>` を明示する必要がある。
新packetには、別の科学実行承認原文とそのhash、stage `RECTANGULAR_GIVENS_PS48`、
authority `USER_EXPLICIT_RECTANGULAR_PS48`、status `APPROVED_BOUND_ONE_SHOT`、このPLANのhash、
公開SOURCE commit・closure hash、新grant ID `rectangular_...`、新output、retry_allowed=false・mandatory_stop=trueが必要。
旧F1／F2 grantや今回の実装承認では通過しない。新outputはcheckout内 `artifacts/pr_rectangular_givens/` 配下。
新scopeのpreflight→atomic claim→保存角度照合→G／wrapper検査・compileの順で、成功・失敗とも証拠とSTOPを残す。

## 今回の技術確認・未検証事項・引継ぎ

候補用の人工／mock test **8件**と既存F2準備の人工test **6件**がPASS。
adapterのselected transpose/copyとprefix、元identityを保持したcache、canonical T・actual inverse・共有run、
符号・位相の照合、不適切scopeの拒否とone-shot接続を確認した。
[TECHNICAL_CHECKS.json](../../artifacts/pr_rectangular_givens_preparation/2026-10-24/TECHNICAL_CHECKS.json)に証拠範囲を残す。
これはlocal技術確認で、CIや外部再現、科学的同値性・費用削減のPASSではない。

既知の技術blockerはない。ただし実helperの数値convention、実QPYの角度読取り、compiled費用は今回未実行で、
次回の科学gateで失敗し得る。削減量を推測で埋めていない。γ・η・α・intervalの未資格状態も変更していない。
旧SOURCE・raw・grant・ledger・marker・STOPを保全し、原workspaceには変更を加えていない。

**次の担当はユーザー／GPT**。固定48件、6基底projector／actual adjoint、保存reference再利用、上記固定閾値・compilerでの
一回の科学実行を別途承認するか判断する。
承認された場合のみCodexが新packet／outputを用意し、preflight後に実行・結果保全する。
この実装準備を公開した時点で**mandatory STOP**。別候補実装、全F2展開、追加reference、γ・N研究、詳細科学レビューへ進まない。
