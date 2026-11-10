# r16期待回路価格：実行前固定PLAN

## 条件と研究系列

基点 `e6c6edac084943843b66c87c7899afee327c51c1`。同じTrack D×PR/RPE r16研究の継続として既存branch `codex/track-d-r16-limited-validation-20261110` を再利用し、新namespaceで識別する。旧STOPを変更せず、今回のユーザー指示による有限campaignだけを行う。

Linear H4／1.00Å／STO-3G／Nα=Nβ2、元H exact ground仮定・準備費除外、number-CDF8／元frame順LD2／ordered partial S2/s1/r16/K0／h=δ=.1。X/Y、whole-path一gadget、C1追加融合OFF。元H packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、r16 law hash `6ada41385f82028b45a1d446c52bc4682203a2157c6359f721e73313b0bf7440`。

13roundの既存S1/N表を読取専用で維持する。εE=.00159Ha、αtotal=.01、各軸α=.01/26、βtotal=π/6、βstat=π/12、prior[-10,10]、旧decoder／φ=−hE。今回PF/RTE誤差・N・全round総費用は新計算しない。

## 既存証拠と最小集合の理由

r32保存C6aはq2/q4各16、C6bはq8で32 path。Strong絶対CXのSD/平均はq2で222.61/4296.13、q4で272.97/8152.88。paired CX差の保存SDは両qで0だが全support恒等式を意味しない。paired total depthのSD/|平均|はq2で159.53/3177.25、q4で345.36/6230.06。C7-4の88 pathではcountsとdepthのモデル構造が異なり、長qのdepthには非線形・basis/lane変動が残る。これらは計画の参考で、r16の分布・分散へ移送しない。

q=2/4の2水準だけでは2係数affineを常に通せるため、最小の追加水準q8を含める。q2/4/8の3水準で傾きの変化を診断する。q16/32は長q資格・モデル採択が目的ではない今回には高負荷で、将来の独立検証候補として未実行に残す。q1は今回の候補集合外。

各q4 pathを選ぶ。n2は分散自由度1しかなく、n4でも自由度3／28変量共分散rank≤3に留まるが、探索的な変動とpaired効果の方向を調べる初回pilotとして採用する。必要母平均精度・信頼区間幅はユーザーから固定されておらず、n4を科学的最適値や所定精度の保証とは主張しない。r32より少標本で初期負荷を抑え、弱い・不利な結果や追加校正の必要性もそのまま報告する。結果依存の追加標本は行わない。

前回r16 q4はS2約272秒、raw全256＋4P0/post PASS。実RAMと空きCPUを確認し、独立raw・compile・postcheckを並列化する。大きなReference検証では既存batched Aerを使う。CPU/RAMの固定小上限は設けず、task数・実空き資源・実測peakに基づく。

## 固定campaign

|q|cut（complete step後）|計画path数|保存再利用|新規path|新規compile上限|
|---:|---:|---:|---:|---:|---:|
|2|1|4|0|4|16|
|4|2|4|1|3|12|
|8|4|4|0|4|16|
|計||12|1|11|44|

q4/sample0は旧 `r16_q4_sample00` のraw/compiled/QPY/NPZ hash、SOURCE、cut、compiler、postcheck/paired χが一致することを確認して価格4件をそのまま使用する。再sampling・再構築・再compile・再simulationしない。旧pathは過去に価格が既知であり、新prospective holdoutではない。q4の4本平均と新3本だけの情報を区別できる原記録を保持するが、都合に応じて旧pathを平均から除外しない。

新IDは `r16_cal_q{q}_sample{00..03}`（q4は01..03）。master `2026111017`、domain `track_d_r16_q2_q4_q8_price_pilot_v1`。既存`trajectory_seed((master,domain,base),'proposed',endpoint,log2(q),'R16_PAIRED_PRICE_CALIBRATION_ONLY',sample,occurrence)`でseedを固定する。機械可読 `sampling_plan.json` をsampling前にcommit固定。新規は52 occurrence／832 eventだけ。seed重複を検査し、canonical PCG64 sampling後は原本・hashを回路構築前に保存する。新しいscope外の再sampling・結果によるseed/path/cut変更はしない。

## 共通処理・gate

既存r16 law構成を再利用し、S1のPF/RTE誤差・N関数を呼ばない。旧C6aのsampler guardはr32/16path専用なので変更せず、新adapterで既存canonical samplerとC3 `prepare_steps` を呼ぶ。CAR原子・head exponentialは既存C6a/B2 helperで一度準備して共有する。

Strongはuncontrolled Gaussian、directional control、head relative、tail basis-run共有、合法step境界融合、scalar/identity集約を保持。ReferenceもA/B内部の同等最適化、cutによる融合損失、2×8 CSWAP、χ、追加8qubit、X/Y readoutを含む。同一pathを両方式・両軸に使う。

rawでAB=V、actual adjoint、vacuum eigenray／χ、signed event／scalar／identity、全256入力のcomplex coherenceとX/Y測定演算子を既存harnessで検査。component abs1e−10、observable Frobenius2e−10、rtol0、phase fitなし。未解決の不一致があれば該当pathをcompileせず、新規科学taskの追加投入を止め、実行中の正常taskは安全に完了・保存する。完全campaignでなければcomplete-caseの期待平均を作らない。

raw PASS後のみP0：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initially_zero=False、topology-free、既存SymEngine MPFR160宣言phase保持。保存rawとCARに対するpostcheck、paired X/Yとχを実施。価格はRZ/CX count、RZ/CX depth、total depth、size、logical qubitsの7個を別々に扱う。

## 統計・STOP・公開

12 pathが適格に揃った場合だけ、C6a/C6bのdescribe/ddof1 covariance処理を新集合へ薄く接続し、各q・軸・資源の両方式平均/SD/極値、paired差平均/SD、28変量の方式/軸/資源共分散と14変量paired共分散を記録する。X/Yを独立nとして倍算しない。母平均CI、coverage保証、未観測support boundは作らない。短q傾きの診断は可能だが、正式モデルfit・長q登録・全round価格表・総資源計算はしない。

r32保存mean/model/Nは参考値のまま保全し、r16と別law/pathであることを明記。`208−224q`をr16へ転用しない。N増加と単価低下は別に表示し、全RPE優位を認定しない。

新namespace `artifacts/track_d_r16_price_calibration/2026-11-10/run01`。ordered event、QPY、NPZ、内部log/resource情報はlocalのみ。科学結果・SOURCE・input/hash索引・失敗記録・STOPだけ公開。技術的不具合は最初のSOURCE・入力・失敗を保持した最小修正のみ。科学条件・sampling・比較集合・thresholdの変更や科学的retryはしない。

q≥64、追加q/path、ground solve、DF fit、S1/N再算定、A/F/S変更・転用、全13round再集計、他algorithm、詳細科学レビューは実施しない。計画完了または不成立を公開し、remote SHA/tree/公開bytes照合後mandatory STOP。次の判断はユーザー／GPT。
