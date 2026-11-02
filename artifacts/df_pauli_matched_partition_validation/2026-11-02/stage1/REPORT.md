# P3 Stage 1：H4案Aの独立operator一致検証

**PASS：full256次元の34項目すべてが事前固定した数値整合基準を満たした。Stage 1科学処理は完了し、Stage 2/3/4へ進まない。**

Pauli側を元bare tensorとnormal-ordering補正から、DF側を保存h_effと平方factorから独立に構成し、全HだけでなくT_eff/V_square別の数値的一致を確認した。数学的厳密等号、実装回路、RTE law、bias/B、費用の一致を認定した結果ではない。

## 1. 固定条件・SOURCE・入力

Linear H4、隣接1.00Å、STO-3G、中性singlet、4 spatial/8 interleaved spin orbital、8 system qubits、full Fock256次元。保存Pauli184非identity項＋constant、DF tol-only threshold1e-8の返却rank10を使用した。保持はT_eff、残差は全square和という案Aであり、旧Pauli prefix27/DF prefix1をmatched分割として代用していない。

時間発展・δ窓・q/r/K・trajectoryは今回の計算条件ではない。旧H4比較T=.8は来歴のみ。raw stateは保存配列/hash確認だけに使用し、状態作用・exact normの再評価・reference生成は0。保存Nα=Nβ=2の36 indicesはleakage補助検査だけに使い、主一致比較をsectorへ射影していない。

実operator検証の科学SOURCEは [`da13170e593e6d845b014e3f110691be10e29d43`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/da13170e593e6d845b014e3f110691be10e29d43)。公開してlocal/remote tree一致を確認後に実行した。[固定条件・数値政策・command](../../../../docs/research/h4_p3_stage1_operator_validation_20261102.md)。

入力refは`4591b8d75a2a6a55c74a51225f0c61648023c8ee`、元生成SOURCEは`0c329b30a81accb35c212be8defa66fee0dd0f12`。receipt/snapshot/pauli.json/両Cモデルの5固定SHA、DF one-body/λ/Gとsnapshot、MO→spin tensorのindex/1/2、state/sectorを照合した。新SCF/FCI/DF factorization・入力/ground-state/reference生成は0。

[source_and_inputs.json](run01/source_and_inputs.json)に全path・SHA・git blob・runtime・実行条件を保存した。共通raw state hashは`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。

## 2. 独立構成と閉合の区別

| operator | Pauli側 | DF側 |
|---|---|---|
| H | 保存184項＋constantを既存PauliHamiltonianで行列化 | 保存c_N/h_eff/λ/Gを元native full-Fock actionで256 basis columnへ作用 |
| T_eff | 保存bare一体＋元Wの収縮Dをfermion→JW→Qiskit Pauliへ | 保存DF有効一体をnative excitation signsで行列化 |
| V_square | 元normal-ordered Wと−Dを明示したfermion operatorをJW変換 | 保存factorからnative A_lを作りΣλ_l A_l² |
| D補助 | 元OpenFermionのreordering/contraction helper | 保存DF一体−bareのtensorと、そのnative one-body operator |

Pauli側にDF保存h_effを注入してTを一致させることも、両側のVをH−T−scalarだけで定義することもしていない。独立構成の後でc_N I＋T＋Vの閉合を別検査した。bare tensor全Hと保存Pauli係数の一致も補助検査した。

元OpenFermion1.6.1の`get_chemist_two_body_coefficients`を再利用し、DF factorizationは呼ばない。1/2を含む保存Wをそのまま使い、transpose/spin slice/収縮Dの元規約を維持した。依存low_rank.pyの固定SHAも一致した。

## 3. 主結果と補正・閉合

主指標はspectral norm、SVD driver=gesdd、complex128。Frobenius normと最大要素差もJSONに保存した。tauは比較scale依存であり、PASSはtauそのものではなくtau/4以下という事前設定で判定した。

| 検査 | spectral residual | Frobenius residual | 最大要素差 | tau/4（PASS境界） | 判定 |
|---|---:|---:|---:|---:|---|
| H Pauli/DF | 7.588648017242e-15 | 3.970589399648e-14 | 7.105427357601e-15 | 6.526899926967e-13 | PASS |
| T_eff Pauli/DF | 2.886535547987e-15 | 1.703805460200e-14 | 1.776356839400e-15 | 3.321586288783e-12 | PASS |
| V_square Pauli/DF | 1.284388912573e-14 | 9.258043257553e-14 | 1.243449787580e-14 | 3.452885422713e-12 | PASS |
| D̂ Pauli/DF | 1.319080157716e-15 | 8.057703071547e-15 | 4.440892098501e-16 | 8.032464047280e-13 | PASS |
| D tensor：元W vs 保存差 | 1.177457075639e-16 | 2.312732826880e-16 | 1.110223024625e-16 | 2.273736754432e-13 | PASS |
| Pauli分解閉合 | 6.217269861023e-15 | 4.959874098702e-14 | 6.217248937901e-15 | 6.526899926967e-13 | PASS |
| DF分解閉合 | 4.885919327709e-15 | 2.158650446315e-14 | 4.884981308351e-15 | 6.526899926967e-13 | PASS |

旧receiptのfull-H差8.462237051574415e-15は保存数値診断として引用する。今回の新測定7.588648017242457e-15とbitwise同値を要求していない。今回のDF側はJW経由の元生成診断とは異なるnative fermionic actionを使っており、構成・加算順序の丸め差を含む。いずれも厳密0の証明ではない。

Pauli bare tensor全Hと保存Pauli係数は今回の数値行列で差0。Pauli full matrixと保存snapshotも差0、native DF fullと保存snapshotの差は5.454497040985e-15だった。これらの0はこの実行のfloating-point出力であり、一般の数学的同一性へ拡張しない。

## 4. scalar・identity

各T/V operatorは自分のJW identityを含む物理演算子全体として比較した。核反発c_Nだけを外側に一度加え、Pauli constantをDF核反発へ追加していない。

| identity係数 | Pauli/JW | DF factor/一体tensorから |
|---|---:|---:|
| c_N | 2.2931012473200001 | 2.2931012473200001 |
| T_eff内I | -7.3042454943550705 | -7.3042454943550696 |
| V_square内I | 4.6796664336182632 | 4.6796664336182578 |
| 保存Pauli full constant | -0.33147781341681026 | scalar field同士の単純比較はしない |

補正Dの符号・移管とidentityの整合、両表現のscalar閉合はPASS。DF squareのI成分は各保存Gからλ[(Tr G)²＋Tr(G²)]/4を計算し、native full-Fock traceと照合した。Gaussian対角化やtail law生成は不要であり、実行していない。fragment別値はJSONに保持する。

## 5. Hermiticityとsector

| operator | Hermiticity residual | sector→外 leakage | 外→sector leakage |
|---|---:|---:|---:|
| H_P | 6.189732678672e-17 | 7.424989848260e-18 | 7.102972443794e-18 |
| H_DF | 1.278247572794e-16 | 0.000000000000e+00 | 0.000000000000e+00 |
| T_P | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 |
| T_DF | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 |
| V_P | 1.790516857623e-16 | 1.769533891850e-17 | 1.828309130391e-17 |
| V_DF | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 |

6主operatorのHermiticity、両方向sector leakageの12検査はすべてPASS。和operatorとして保存spin sectorとの整合を数値確認した。個別Pauli string、Gaussian basis途中、RTE event/compiled回路のsector保存性を検証した結果ではない。

## 6. 数値政策・不確かさ・判定

d=256、binary64 machine epsilon e、gamma_d=d e/(1−d e)、scale=max(1,両operandのspectral norm)に対して、tau=16 gamma_d scale。PASS≤tau/4、UNRESOLVEDはtau/4超〜4tau以下、FAIL>4tauという規則を実行前に固定した。leakageには元operatorのscale、D tensor/scalarにも構成元full-Fock次元を使う。

これは独立構成のbinary64 consistencyを判別するengineering resolutionであり、厳密なrounding error上界、DF truncation certificate、統計信頼区間ではない。旧1e-9を無条件継承せず、df_tol1e-8を新しい差の許容値として足してもいない。保存df_truncation_value=0も全誤差0の証明には使わない。

全34項目がPASS band内であり、FAIL/UNRESOLVEDは0。追加精度・別solver・別分割・別入力は実行していない。[全比較・policy・scalar](run01/operator_comparisons.json)。

## 7. 技術preflight停止と修正の来歴

初期SOURCE d57742f182bc4c2ae68f1df5b54dfb074a7ea7e4は、私が追加したcomplex JSON decoderの算術式が虚部−0.0の符号bitを221箇所変えたため、COMMON_RAW_STATEのhash照合で停止した。値と元の保存入力SHAは一致し、成分の直接copyで元hashへ一致することを読取診断した。

この時点のPauli/DF route開始・operator構成・metric件数はすべて0。旧statusのscientific_batches_started=1はpreflight前に立てたrunner起動flagで、科学operator実行を示さない。新SOURCEでは実際のroute開始時にflagを立てる。

許可済みdebugとしてdecoderだけをbit保持のcopyへ修正し、signed-zero人工回帰testを追加した。原本・operator構成経路・dtype・norm・許容差は変更しない。修正SOURCE da13170e…の公開後、run01で初回の実operator batchを実施した。科学operatorの失敗後の再計算ではない。

初期の[source_and_inputs.json](source_and_inputs.json)、[execution_status.json](execution_status.json)、[STOP.json](STOP.json)は元のまま保全した。成功した科学runのSTOPは別の[run01/STOP.json](run01/STOP.json)。初期停止を成功へ書き換えていない。

## 8. 再利用・実行件数・保全

元PauliHamiltonian/JW、OpenFermion収縮helper、df_linear_operator/native excitation table、reverse_bits、spin liftを再利用した。小さい追加は独立経路adapter、native signsを再利用したCSR構成、比較と記録である。full DFは元Python actionで独立に構成し、T/V/Dのcolumn反復はCSRで省いた。両側のoperator中間結果を使い回していない。

最終人工suite10件PASS（初版9件PASS、decoder回帰追加後10件PASS）。2/4qubitのsigned/offdiagonal factor、独立occupation oracle、1/2・D・scalar・bit順・Hermiticity/leakage・境界判定を検査した。作成者による技術確認で、独立科学レビューではない。

| 実行区分 | 件数・状態 |
|---|---|
| H4科学operator batch | 1回、PASS |
| 独立構成経路 | Pauli 1、DF 1、両方完了 |
| 返却・記録対象full-Fock operator | 主H/T/V 6＋D/V_normal/bare全Hの補助4＝10 |
| DF内部の一時A_l CSR | 保存10factor分（別候補・別科学runではない） |
| 元native full Hのbasis column action | 256（raw φの状態作用・時間発展ではない） |
| 数値比較job | 34、全PASS |
| 入力technical preflight停止 | 1回、科学operator開始0で保全 |
| 科学失敗後retry・追加precision/solver | 0 |
| signal/bias/B、law/event/sampling、SCF/FCI/DF再生成、新state/reference、回路build/compile、N/C/G | すべて0 |

科学runは18.84秒、親exit0、全thread join済み。64 CPU affinityと十分な空きRAMを観測し、2独立構成taskと24 metric workerを使った。小行列BLASは1thread/jobとし、metric側の並列度を実際の負荷から選択した。共有OS/venv/他jobへの変更なし。

科学SOURCEの8 file hashは実行後も一致。旧P1/P2のコード・証拠・STOPとP3静的2資料を保持し、main/元研究branchへ書込を行っていない。原本や巨大full行列を重複保存せず、返却対象10 matrixのhash/shapeと8×8補正、必要な数値結果を保存した。

[construction_paths.json](run01/construction_paths.json)、[matrix_fingerprints.json](run01/matrix_fingerprints.json)、[execution_status.json](run01/execution_status.json)、[保全確認](verification.json)。結果公開後のremote SHA/全文照合は最終チャットに報告する。

## 9. Stage 2への条件とmandatory STOP

今回のPASSは、案Aの同じ物理的T_eff/V_squareをPauli/DFで構成できるという数値的基盤である。Gaussian保持blockの回路同一性、RTE辞書/law/B/angle、signal bias、compiler価格、shot/Gの一致・優劣は未検証。主要claimや新しい比較方法の正式採択もしていない。

次にStage 2を行う場合は、ユーザー/GPTが共通target/referenceへの既存H差の転送、raw ν・scalar表記、q/r/K/cutoff、precision/allowanceと実効law評価範囲を決め、別途明示許可する必要がある。Stage 3の共通DF-native Gaussian一体blockという方針は受領済みだが、実装・phase/actual adjoint・control・wrapper/compilerの対応は将来の検証である。

**STOP_STAGE1_PASS_NO_AUTOMATIC_FOLLOWUP。結果公開・remote SHA/全文照合後mandatory STOP。P1/P2の旧STOPと初期technical preflight STOPを保持。Stage 2/3/4、案B、P1N、H6再実行、詳細科学レビュー、論文改訂へ自動移行しない。次の担当はユーザー/GPT：Stage 2の必要性と具体的scopeを判断。Codexの追加科学実行作業なし。**
