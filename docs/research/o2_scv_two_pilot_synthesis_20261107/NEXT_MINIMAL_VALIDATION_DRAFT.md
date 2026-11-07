# 次の最小作業：Product一致の技術・metadata確認草案

2026-11-07 JST。**NOT_ADOPTED。提案だけで、今回の実装・人工実行・科学計算を許可する契約ではない。**
対象候補は保存linear H4／1.00 Å／STO-3G／legacy DF12／8qubits・全4電子sector／L_D3／m0,q1,δ=.02,r1,K2／coefficient_atol0、N1 actual Q・selected full、N2保証ground・認証済みexact Fraction engine、理想gate・X/Y。新しい分子/round/targetは設けない。

## 0. 再開に必要な問い

再開を考える理由は「productsの完全一致がh2/h3の応答差を説明するか」という別の問いにある。ただし希少層の局所削減がfull-target費用に寄与する理由が必要で、未検証であることだけでは足りない。[方向比較](SCV_RESEARCH_DIRECTION_OPTIONS.md)の保留提案を維持する。

## 1. 将来の最小技術scope（新たな実装指示後）

既存`ConditionalSampler.from_record/cells/cell_of`、`Metadata`のbyte/H/preparation binding、`ControlBank.pattern_mass/joint_mean`、indicator `FrozenCV`の算術を優先する。新たなsampler/engine/価格器/汎用frameworkを作らない。

必要差分は次の範囲。

- 完全label一致`1{n=2,L0=L1}`のtyped feature/bankと、[候補比較のexact式](SCV_NEXT_CONTROL_COMPARISON.md)を保存するone-feature adapter。
- label/basis/sign/policy、actual grid・全5層・a≠b・application orderを新feature identityへbindする。
- 将来fresh検証用にはh2/h3別train/モデル・antecedentsを持つ限定modelが必要。既存`CellModels/fit_cells`のh1-only guardを外さず、新しい型として分離する。この接続はmetadata段階では不要。

API名は今後の実装で決める。ここに実在しないrunnerを起動可能なものとして掲載しない。

### 人工correctnessで確認する項目（今回は未実行）

1. 少数labelの人工dyadic lawを列挙し、h2/h3の平方和/条件付き比、m₂=ν、Σ=ν(1−ν)と一致すること。
2. h0/1/4の値0、h2/h3の一致/不一致、a≠b、zero-grid label保持、正層を落とさないこと。
3. 同じsupport文字列でも別basis/labelなら一致にしないこと。products→rotationの位置規約、scalar/event phaseを維持すること。
4. 任意の固定λで条件付き中心化がtargetを保持し、indicator box D=2+|λ_h|が全bit/feature組合せを含むこと。
5. 将来modelではh2/h3を別fit・別stream、他層0、train/eval分離、rank不足・range超過・identity変更でSTOPすること。
6. 保存/STOP/重複禁止と、metadata-only入口からRNG/state/Born/circuit/compile/fit/managerが起動しないこと。

既存`test_scv_scalar.py`のexact moment/中心化/range/独立性と、`test_scv_signed_support.py`のlabel/position/identity拒否パターンを再利用する。既存人工PASSは新Product候補のPASSを意味しない。

## 2. 最初の科学scope案：metadata-only一回

これは**実装・人工確認・SOURCE固定後に別途明示許可が必要な、新しい科学計算**。今回0回。

| 項目 | 提案内容 |
|---|---|
| 入力 | N1 `744e1a09661ebba6841fe5faf456f88c41c47780` のLAW.jsonとPREPARATION.json全bytes。既存H/partition/preparation/tail/policy/a,b/label順のhashを照合 |
| identity参照 | N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`、engine `fdf00cee3c87d804845497afc89ac6a71f3b3f59` は参照hashのみ。stateをロード・作用しない |
| 科学call | 新Product mean/moment evaluatorを**1回**。LAW/catalog各1読込、typed binding1回、全5層のν/m₂/Σを1組。回数は将来の実装SOURCEへ対応付けて固定 |
| 計算方法 | 保存整数grid質量の有限label平方和/条件付き比。全path列挙、sampling、Bornは不要 |
| 出力 | exact5層mean/m₂/Σ、feature/入力identity、非定数性/適格性、call/log・環境・SOURCE・STOP。旧p_hは既存値を引用し再準備しない |
| 明示的に含めない | 旧224/448 raw/bit/係数/価格の投入、N1再準備/CDF、N2 solve/spectral、engine認証、state/RNG/shot、build/compile、fit/paired/cal、費用・varianceの新評価 |
| STOP | identity/式/層/support/rangeの不整合、出力衝突、非有限値、call超過、非干渉不成立で停止。失敗時の条件緩和/retryなし。成功時も公開照合後mandatory STOP |

既存N1の参照byte identity：LAW SHA256=`b77980d198cc9edc620daec9bf87df02b66e96575fd85327aed0793f3e86ecef`、PREPARATION全bytes SHA256=`ca897db7be52ad30dcfab0512d7530188fc13a9d8ca0c78430809492d01f738b`。[既存metadata REPORT][metadata]からの引用。

**これで変わる判断**：既存lawからの非定数性の構造的根拠を前提に、actual Qのexact mean/momentと入力bindingを保存でき、feature出現質量に基づく有限train設計を検討できる状態になる。正のmassや少数train内の両値出現を同一視しない。**これだけでは変わらない判断**：bit相関、variance/RZ改善、SCVの科学的GO。metadata ELIGIBLEなら自動でfresh測定へ進む、という規則は採択しない。

## 3. Fresh train/pairedへ進む前の未確定事項

metadata-only後も必要性が認められる場合だけ、次を一つの新契約に定める。

| 未確定事項 | 固定すべき内容 |
|---|---|
| 活動層 | h2/h3のどちらか、または両方。metadata結果だけで自動選択しない。他層λ0は事前固定 |
| train量 | 各活動層/軸の有限件数とrank2・finite slope・no-refit条件。低出現countと学習費を考慮。224/448のh1 train数を流用しない |
| 評価quota | 全5正層を残す。inactive層も全target推定に必要。少数のh2/h3だけを測って全target改善と呼ばない |
| 比較 | 同じfresh評価をB1-OL/新SCVで共有。旧pilotとの絶対点指標比較で優劣を決めない |
| 独立性 | 新campaign identity・seed/Stream表・train/eval/axis/cellの独立性、評価前に全係数とquota凍結。旧raw/cache混入0 |
| range | indicatorのactual D_h=2+abs(λ_h)。guard値は新契約で採択、旧4/6を自動継承しない |
| 指標 | 既存pairedの平均・層別/全target分散点指標、count/all-equal、variance差の不確かさの扱いを事前定義。精度保証を標榜しない |
| 費用 | train＋共通evalを全額full RZ課金。baseline参照とSCV参照は重複し物理費に加算しない。全5層の価格、古典時間/メモリ/cacheを記録 |
| 固定価格/engine | 同Q/selected policy/exact Fraction/N2保証state、Qiskit1.3.0/rz,sx,x,cx/opt2/seed17、状態準備のみ除外 |
| 実行境界 | 一条件・X/Y・一campaign・有限physical shot/call上限、bit-before-price、consume-once manager、正規STOP、retry禁止、完了/中断で公開保全後STOP |

**今回はfresh shots数・seed・range guard・診断閾値を提案値としても新たに固定しない。** 活動層がh1から変わり、Product平均と学習可能性の数値が未取得だからである。最初の承認対象を上記の「技術実装・人工確認」と「metadata-only一回」に分ければ、fresh予算を無目的に設定せず次担当が進められる。旧224/448 campaignの再開ではない。

既存理論のoracle削減上限Δ_h≤Var(m|h)≤σ_h²≤1、全target重みp_h²/n_hは、希少層の利益を問う記号的な材料になる。[既存設計§2,4,6][design]の再掲であり、実H4の上限数値・break-even・GOを算出していない。

## 4. 保留案を選ぶ場合

新しい相関の根拠と全費用を回収する問いが具体化しない限り、追加scienceを保留する。既存のexact中心化、bit-only境界、独立stream、型付きrange、paired/費用会計、manager/STOPの実績は保存し、将来の別候補や統計接続に再利用できる。SCV全般を無効と記録しない。

再開する場合の担当はSERVER1 Codex。ユーザーの実行指示が承認そのものであり、承認証明書・grant発行手続きは追加しない。共有設定/venv・他job、並行するexact engine O1/O2へ操作せず、実空き資源と非干渉を優先する。

[metadata]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b4512675ae1d650858256547d359210361d77f68/docs/research/scv_signed_support_h4_metadata_science_20261107/REPORT.md
[design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107/STRATIFIED_CV_DESIGN.md
