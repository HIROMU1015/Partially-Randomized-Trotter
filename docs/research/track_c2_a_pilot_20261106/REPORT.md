# C2-A：round依存PFの限定pilot

**有限3round診断ではPF族の費用交差を観測しなかった。長roundの期待価格と新構成のcompiler誤差資格が閉じていないため、科学的な優越・一般的no-go・全18round総費用はいずれも未資格。価格不確かさでmandatory STOPした。**

基準SOURCE `6f6c084bdfaf04ba951fd5d8ea0c521313ceb0b3`、科学資産 `720416c5b88b227cf76ff7aa729d91a0a98ed58b`。
SERVER 2 `GPUA100-gleap02`、2026-11-06 JST。条件と科学call範囲は[SCOPE.md](SCOPE.md)、判断の出発点は[科学レビュー](SCIENTIFIC_REVIEW.md)。
専用branch `codex/track-c2-a-round-pf-pilot-20261106`。

Linear H4、1.00 Å、STO-3G、DF rank12、元Hの供給される厳密ground、8 qubit/4 electron。
状態準備費を除く。delta0=.02、18round、t_m=.02*2^m、energy精度1.5936e-4 Ha、alpha総和.05、位相総幅.4 rad。
保存行列と元の連続積分入力の無条件な包含は引き続き未資格。
保存ground/eigenspectra/common preparationを再利用し、SCF・DF fit・ground eigensolveは0件。
E0は小系の信号診断にのみ使い、未知biasの補正やdecoderのalias選択には使っていない。
A/Bは独立SOURCE・独立worktree・独立出力のprocessとして分離した。B計算中にA実装を進めたが、B約54秒の終了後にA科学runが始まり、科学run同士の実時間重複はなかった。内部taskはprocess並列。AIサブエージェントは0。並列指定とのこの実行上の差を保存し、STOP後の再計算は行わない。
共有環境、他ユーザーのprocess/file、旧C-N/C-S/C-P/P-D/BF/BM/PR-3のSOURCE・raw・grant・ledger・marker・STOPを変更していない。

## 有限条件と強対照

LD3をprimary、LD12をtail-free referenceとし、splitごとに独立比較した。split切替の利益は算入していない。
各splitでS2/Yoshida4、h=.02/.01、K2、LD3ではr=1,2,4,8,16,32。
まずm=0,8,17だけを検査し、84信号セルを取得した。高次PFの新係数探索、C2-C、旧processor/residualは実行していない。
合成policyは全候補で理想native固定、RZ/CXを別の目的量とした。T改善とは呼ばない。

固定PF対照にも同じh/r候補と.4 radの残余統計予算を与えた。
既存`kernel`が作るPF/finite-RTE mean、normalization、元H ground expectationから、各セルの位相・半径・必要shotを計算した。
RTEのKは双方とも2固定。統計alphaは初期交差診断で既存一様1/720。
共同alpha最適化は全18round価格に進む段階の条件であり、今回はそこへ到達していない。
厳密な全r compiled optimumや全体の誤差配分最適性を認定していない。

## SOURCEと最小価格接続

- 新adapter：[track_c2_a_pilot.py](../../../src/trottertracks/track_c2_a_pilot.py)。
- 再利用：`track_c_h4.common/kernel`、`track_c_h4_price.trajectory_parts/d_part/wrapper`、`track_c_compiler_a.compile_reference`。
- 保存commonのhashと版：[INPUT_PROVENANCE.json](../../../artifacts/track_c2_a_pilot/2026-11-06/INPUT_PROVENANCE.json)。
- 信号：[signals](../../../artifacts/track_c2_a_pilot/2026-11-06/signals)。短価格：[prices](../../../artifacts/track_c2_a_pilot/2026-11-06/prices)。

全6 rは信号で評価した。primitive数×shotの明示surrogateで初期3roundに選ばれるrだけを価格へ渡し、
q1/q2の診断fitとq4 holdout、order0/rare各stratumの決定的seed代表pathを1本ずつcompileした。
合計60短価格セル、各X/Yをreference Aでfull measured wrapperまでcompileした。
このstratum代表値はMC母平均の推定・信頼区間ではない。
一つのrを全roundへ強制した弱い固定PF対照ではない一方、全rをcompiled価格で最適化した対照にも達していない。
新角度・新構成のSOURCE→native error envelopeは未資格であり、旧correctnessの無条件移送はしていない。

## 得られた診断

各split・各指標について、3つの診断roundの全てでS2、h=.02が選ばれた。
LD3の診断rはm0/8で1、m17で32。LD12はrなし。
Y4の高次数だけでは、この範囲の追加stage費とnormalization負担を埋めなかったという診断である。
各候補の位相・半径・normalization・必要shotと、族内h/r選択は[CROSSOVER_SCREEN.json](../../../artifacts/track_c2_a_pilot/2026-11-06/CROSSOVER_SCREEN.json)に保存した。

q4のRZ/CX最大点誤差は15.3821%。代表pathの母平均不確かさも閉じていないため、
q1/q2から長qへの外挿を価格資格へ格上げしなかった。これらの価格から全18roundの総Gや勝者を認定していない。
診断点間の未実行roundの交差、他のstep/PF/K、他の系を否定する結果ではない。

既知UWCや既存round最適化との違いは、PF族ラベルの切替を追加の自由度として検査した点。
round別h/r・誤差配分・表現選択そのものを新原理とは主張しない。
今回の診断は、固定PFを十分に最適化した後に残る新しい切替利益を確立しなかった。

## 検証・終了

84信号セル・60価格セルの出力coverage、X/Yと4資源のschema、条件付きstratum確率和を検査しPASS。
[TECHNICAL_CHECKS.json](../../../artifacts/track_c2_a_pilot/2026-11-06/TECHNICAL_CHECKS.json)。
これは著者自身の技術検査であり、第三者による独立科学レビューではない。
float64診断に新たな区間保証を付与したものでもない。
[SUMMARY.json](../../../artifacts/track_c2_a_pilot/2026-11-06/SUMMARY.json)、[STOP.json](../../../artifacts/track_c2_a_pilot/2026-11-06/STOP.json)。
科学実行約65.3秒。8 signal taskと最大60 price taskを利用可能64 CPU affinityの下で並列化し、各processの小行列BLASは1 threadとした。

**mandatory STOP。** 18round全探索・追加MC・Y8・新step・旧STOP再開は実施しない。
次の判断担当はユーザー：期待長q価格の資格を閉じる価値があるかを判断する。
必要と判断し別途許可した場合に限り、Codexがそのための有限な取得・保証を行う。現在、次の科学実行予定はない。
