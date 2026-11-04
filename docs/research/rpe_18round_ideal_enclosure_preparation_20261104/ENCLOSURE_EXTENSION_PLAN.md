# 全18round ideal categorical包含：固定SOURCEと限定実行計画

2026-11-04 JST。**技術準備完了・科学実行未承認／未実行。**
対象は[固定入力一覧](ROUND_INPUT_AVAILABILITY.md)のH4/1 Å/STO-3G/legacy DF12、Partial L_D3、δ=.02、m0..17。
保存r/K/N/α/β/ε、H/stateを変更しない。m17結果は再計算せず再利用する。
この文書・[round_adapter.py](round_adapter.py)・[runner](run_round_enclosures.py)・[fixed_inputs.json](fixed_inputs.json)・[requirements](requirements.txt)を含む専用branchの公開commitを次の科学SOURCEとする。
実行許可はこの公開commitに対して別途必要。SOURCE照合は数値再現性の条件であり、許可証明の発行・検証ではない。

## 1. 包含対象の定義

独立uniform53 wordによるcategorical選択と、保存gate parameterをexactに扱うideal未compile回路、保存stateのnormalized rayを対象とする。
uniform53はevent/component選択の仮定。最終±1測定はideal Born測定のIID marginalを仮定し、float64のBernoulli比較・clippingへ移送しない。
固定PCG64列が数学的IIDであるとも主張しない。

成分のsigned Hermitian involutionをP_j、保存component lawのexact grid massをp_jとし、A=Σp_j P_j。
order lawのmassをw_n、保存／新SOURCE replayで確定するbinary64角をθ_nとする。

    E_r = Σ_n w_n (-1)^(n/2) [cos(θ_n) I − i sin(θ_n) A] A^n
    S_r = R (E_r)^r F
    ν_logical,m = ψ† (S_r)^q ψ / (ψ†ψ)

独立選択に対する積の期待値の因数分解を使う。Aと各rotationは同じAの多項式だが、F/Rとの可換性は仮定しない。
列vectorの作用順序はF→event products/rotation→R。各event内の符号・θ・builder規約は旧mean_adapterと同じ。
builderのπ64の逐次加算とexact符号との違いは、既存K2専用`builder_phase_bound(q,r)`で一様に転送してν_catを覆う。
F/R・identity/constant位相を移動・再加算しない。
lawにfinite-RTEの減衰が既に含まれるため、Bやattenuationで追加除算しない。

## 2. 再利用と最小拡張

| 処理 | 固定SOURCE・再利用方法 |
|---|---|
| 入力照合 | 固定Git commit/path、blob OID、SHA256と作業tree byte一致。V2 Partial18行をgridの(q,r,K)、physical label、opt2保存N/α/εへ結合。model τとsignal τを区別 |
| 共通operator | m17結果ZIPの96-bit A/F/Rとr32 stepを再利用。`decode()`は既存virtual_bridge_adapterのrecenter方式だけを局所再利用し、bridgeやKernelBoundsをimportしない。保存global radiusとdecode時の丸めを保持 |
| component/basis/state | m17 export/lawを固定入力として利用。export_adapterの再構成は0回。SCF/DF/state solve・basis replayも0回 |
| norm² | 同じexport stateへのm17保証付き区間を分母に使用。新しいψ†ψ計算なし。divisionの包含誤差を保持 |
| r1/2/4/8/16 | 保存signal τ64→既存`rte.finite_rte_distribution(τ,K=2)`→B-1と同じ`atan(τ/(n+1))`→既存`mean.categorical`。order probabilitiesは二重renormalizeしない。5群で各1回 |
| r32 | 保存m17 step・law・θをm15/m16へ再利用。新CDF/θ生成0回、m17 powered/mean評価0回 |
| mean・分類 | `ball.product/power`、`mean.event_average`、`mean.builder_phase_bound`、`ball.classify_interval`を変更せず再利用 |

他5群のlawは**新SOURCE replayで指定されるlaw**である。過去samplerの未保存payload identityはUNKNOWNのまま。
SOURCE replayのscalar/CDFを将来実行時にhex/grid rationalで記録する。異なるlibm結果が過去payloadと同じとは仮定しない。
現在の準備では実H4のτを`finite_rte_distribution`/atan/CDFへ渡していない。

`mean.event_average`自体は一般の非負偶数Kに対応し、人工K0/2/4を検証した。
一方、builder phaseの証明はK2専用なので、科学runnerはK≠2をUNKNOWN/STOPにする。
全保存roundのK2を確認できたため、新しいK依存phase証明は不要。

## 3. 精度・科学call範囲・資源

- Python3.11、binary64 radix2/mantissa53、round-to-nearest、NumPy1.26.4、python-flint0.9.0／FLINT3.6.0。scalar/CDF replayは各workerでも丸めmodeを確認する。
- Acb/Arb **96 bitで1 pass**。保存96-bit operator radiusを再利用し、精度を上げれば元半径が縮むとは扱わない。
- 新規order law/CDFは5回、step組立ては5群、r32 step decodeは1群。m0..m16の新complex power/meanは**最大17セル・34軸**。m17は旧両軸判定を転記するだけ。
- 各群内のstepを共有し、6群を独立processで並列化。workers=min(6,利用可能CPU affinity数)、FLINT threads/worker=floor(CPU数/workers)。自然な6群以外の恣意的CPU/RAM/worker上限やwall-time capを設けない。
- 実行前にそのhostのCPU affinity・利用可能RAM・共有負荷とFLINT/GPU可用性を確認する。小さな70×70の共有入力＋binary powerを使い、基底の再構成・全trajectory列挙・巨大joint decoder arrayを避ける。実行hostで資源不足なら停止し、他jobや共有設定を変更しない。高速化率は未測定。
- Sampling/RNG replay、H/state/DF生成、元V1/V2 runner起動、回路build/compile、費用・branch/energy成功率の新解析は**0回**。Full L_D12、他δ、他state、他N/α/β/εは対象外。

## 4. 各roundの判定と18round統合

各軸a∈{cosine,sine}について、ν_cat,m,aの保証付き包含I_m,aを作る。
保存μ0は元SOURCEと同じbinary64値としてexactに取り込み、Nは整数、α/εは旧m17 classifierと同じ保存decimal literalのexact rationalとする。

    U_cat,m,a = upper sup_{x∈I_m,a} |x − μ0_m,a|
    s_m,a = sqrt(2 log(2/α_m,a) / N_m,a)
    upper(U_cat,m,a + s_m,a) < lower(ε_m,a)

最後はstrict不等号。区間が接する・またぐ場合は確認済みにしない。
旧classifierは十分条件の除外を確定できる場合にEXCLUDEDを出すが、それはRPE自体の失敗証明ではない。
不足・非有限・未対応・UNDETERMINED/EXCLUDEDはいずれも今回の統合成功へ格上げせずSTOPする。

保存norm²分母、operator radius、builder phase transferをそれぞれ一度だけ含める。
U_catは数値包含・指定lawと保存μ0の差を既に覆うため、同じ数値誤差やfinite truncation、β_PF/β_RTEを再加算しない。
PF/finite-RTEの共通E0へのphase差は[全体保証の別条件](RPE_GLOBAL_EVIDENCE_GAPS.md)である。

m17を含む36軸の十分条件が全て確認された場合にだけ、
`CONDITIONAL_IDEAL_CATEGORICAL_LOCAL_CONDITIONS_CONFIRMED`とする。
`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、両bridge U=nullは常に保持する。
結果統合はbranch/energy保証の成立やV2成功率の新しい区間保証を意味しない。

## 5. 出力・STOP

fresh出力ディレクトリのRESULT.jsonと各群のgroup_r*.jsonにSOURCE/入力hash、群ごとのlaw/CDF/grid mass/θ hex、各roundのlogical区間・phase transfer・powered半径・両軸strict判定・資源を記録する。
既存出力を上書きしない。未判定後は群内の次セルを開始せず、未開始のfutureをcancelする。共有STOP eventで他workerの次セル開始も抑止する。実行中セルは回収し、得られた部分結果とworker errorを記録して終了する。各セルの開始・完了とlawは群別checkpointにも保存する。
Ctrl-C等の中断でも結果を未判定・mandatory STOPとして保存する。OSの強制終了時は最後の部分記録を保全し、自動retry/resumeしない。
不足object・hash不一致・入力変更・runtime不一致は科学計算前にUNKNOWN/STOP。
新精度、retry、条件変更、別law/branch/energy解析へ自動で進まない。

科学実行用コマンド（**この準備では未実行**。公開SOURCE確認と別途ユーザー承認後のみ）:

```bash
python docs/research/rpe_18round_ideal_enclosure_preparation_20261104/run_round_enclosures.py \
  --run --output-dir artifacts/rpe_18round_ideal_enclosure/2026-11-04/run01
```

引数なしはbyte/schema照合のみで、数値adapter/FLINTをimportしない。
今回の人工18件＋既存25件はPASS。[test_round_extension.py](test_round_extension.py)は6群のprocess統合・m17再利用、r1..32、coreのK0/2/4、K≠2停止、非可換F/R、半径保持・norm²再利用、strict/UNKNOWN・fresh出力を検証した。
人工PASSをH4の局所条件やenergy保証へ格上げしない。

**公開照合後mandatory STOP。新しい実H4包含はユーザーの別途明示承認待ち。**
