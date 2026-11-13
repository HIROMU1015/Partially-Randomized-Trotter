# NEXT_OPTIMIZATION_PLAN：有限予備分析と後続の判断

2026-11-13 JST。**今回は計画まで。保存数値の新しい算術も、以下の科学計算も未実行。** 最適化の目的は同じenergy精度/claimを満たすwhole-scheduleのGで、shot数だけではない。[記号式と未実行欄](DECISION_PACKET_V0.md)／[固定SOURCE](SOURCE_SCOPE.json)。

## 1. 承認を求める有限保存再解析：B-PRELIM-V0

これは研究作業名であり、承認証明書/ID/grantではない。チャットでこの範囲の許可を得れば十分。新しい証明手続きを作らない。

### 固定対象

- PRはcanonical `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`の1構成、13round/26軸、保存nominal1配分。h/M/exactκ/δ/L_D/K/公式/元H-state/snapshot/prior/decoder/予算/P0を変更しない。
- Full PFは既存6lane（S2/Y4/M5×要求幅.1/.4）のm3/m4、計12物理セルだけ。SOURCE_SCOPEのIDを使い、新しいschedule/IDを生成しない。
- D pilotは`D_P2_DIRECT_PILOT_V1`のm0/m6 fresh summaryだけ。P1/replayは参考来歴として残し、標本へ加えない。
- A screenはrun02の20状態と4保存m6セルだけ。失敗・未開始・空例外は保存status/nullを維持する。
- S3は保存PRIMARY/HANDOFFと入力mean/nominalの出典確認だけ。43,819 trial archiveを開かず、RMSE/CIを再計算しない。

### 有限な算術と出力

| 項目 | 入力・演算の上限 | 判断目的・出力資格 |
|---|---|---|
| P0：構造費用のround寄与 | A169d `N_HANDOFF.rounds.nominal_N_X/Y`とC9d95 `COVERAGE.values`の26組。m/axis/identity/lawで照合し、13round部分和・全体1個・全13比率、tail開始{6,9,10,12}の4比率。保存数値enclosureを同じ係数で伝播 | 固定nominal×構造量で長時間がどれほど効くか。CONDITIONAL_DERIVATION。正式Gの近似精度/上下界を認定しない |
| P1：時間成長とproxy感度 | 各軸13個のS/T²と12隣接S比。13roundと上の4tail groupの比例感度係数。元q/R/実Tは保持 | 固定scheduleの記述、どの価格誤差がこのproxyに効くか。T²モデルfit/他候補への外挿なし |
| P2：短roundの損益分岐 | D e7dcaの選択12セルのうち直接価格がある8対×2軸だけにθ=f_PF/s_PR。4欠測対はnull。r=N_PR/N_PFとρ=C_PR/S_PRは記号のまま | N比とcompiled補正が結果を変える境界を示す。C_PR/C_PFの実測比やwinnerではない |
| P3：pilot内compiled/構造差 | D39ddd `RESULT.summary`のm0/m6×X/Y、既存C.mean/S.meanとmean_D/S.meanの2比だけ。既存variance/covariance/status/SOURCEは転記 | DEVELOPMENT/FINITE_COHORT_ONLYの局所診断。全law比/CI、E_P[S]+barD_Q、m12価格へ昇格しない |
| P4：signalと欠測の一覧 | PR全13のB2084791 `STATISTICS_INPUTS.rows.mean`・元A出典、選択PF m3/m4の12保存signal、A699178の4保存m6 record。元raw/corrected/logB/既存bias/radius/phase/allowance/sourceを転記。その他screen状態もそのまま | N/riskに不足する情報を明示。新abs/phase/radius/energy bias・N・RMSEの計算なし。方法の順位を付けない |

保存PF signalの原本はSOURCE_SCOPEの各`signal_source`（主に155529e2の`signal_records.json.gz`の固定key）へ戻る。receipt値はe7dcaの`rows[physical_candidate_id].prices`の既存選択carrierを使用する。両者のSOURCE/metadataを確認し、元値を再計数/再compileしない。選択sourceが曖昧、欠落、又は条件が違う場合、そのセルはUNKNOWNで返す。別cohortや似たqの値を代用しない。

Cのsource axisは`cosine/sine`、N側は`X/Y`。SOURCE_SCOPEの対応表でのみ結合し、source recordを書換えない。C valuesは配列順がround順ではないため、位置で推測しない。nominal Nはexact int、binary64値は既存`exact_scalar`でその保存値を正確に有理数へ写して積和/比を計算し、表示丸めを分ける。元C enclosureを認証済みcompiled boundへ読み替えない。

結果は新directoryの表/JSONと短い解釈に保存し、原本のsource/value/statusを保全する。`conditional_structural_cost`、`round_shares`、`break_even_values`、`pilot_sample_ratios`を別quantityにし、十分N/global bound/G/winnerはnullのまま。既存controllerを緩めず、proxy算術は既存exact補助関数で行う。

### 含まないもの・終了

新signal/reference/normalization、Monte Carlo、trial再解析/新CI、phase/T評価、新N配分の生成・最適化、compiled平均の認定/compile、残差/affine fit、新PR候補の生成・ランキング、正式G/4方式winnerを含めない。未知N比/補正/価格誤差の数値gridも勝手に設定しない。

SOURCE/identity/law/profile/単位の不一致、曖昧carrier、保存非有限/overflow/null、分母0はその演算をUNKNOWNとし理由を残す。科学的条件の修正、外挿、再計算、取得追加へ自動fallbackしない。未知値があってもそれ以外の許可済み有限表は完成させる。P0〜P4を各1回集計し、結果の意味を点検・公開照合してSTOP。範囲を変えない通常のコードdebug/人工チェックは可能だが、新しい科学条件や追加解析は別許可へ返す。

## 2. 設計変数：何が分かり、何を追加すべきか

以下は**依存関係と研究判断上の優先案**。保存数値の新ランキングや、どの変数が実際に最も効くかの認定ではない。

| 自由度 | 現在判断できる範囲 | 新たに要る証拠・可能になる判断 |
|---|---|---|
| whole-shot配分 | B adapterで同候補の有限配分を扱える。旧nominalは十分N/最適Nでない | 採択した全事象risk/loss・数値/target/law誤差と、1候補の少数配分の全decoder評価。同じ精度で価格付き配分を選べるか判断 |
| L_D/分割 | 現PRの全13科学値はL_D1固定。Full PF/切断p7は別方式の保存例 | Aが1つの隣接/対照prefixをsource-bound全scheduleとして提案し、全mean/normalizationと対応C domainを得る。決定論head増とtail減の競合を調べる。今回prefix値は採択しない |
| 決定論PF公式・macro幅 | Full PFのS2/Y4/M5の短round保存と4つのm6診断が存在。M5/m12は数値問題で不完 | Aのgate原因/安定性調査を優先し、必要なら代表失敗1セルの独立数値診断を別許可。全schedule meanが揃った後に精度とq/価格の競合を判断。閾値緩和や8次追加は不要 |
| PRのPF構成・幅 | 現S2/.1以外は全体精度・価格がない。Track Aの有限T最適値を移せない | Aの有限menu・B対応formula/signed order・D価格domainを先に揃える。PR-Y4は現B未対応で、正式menuに必要な場合だけ技術接続を別途行う。PF-onlyのY4対応と混同しない |
| 共通κ/R | 全roundが共通κと元λから結合。保存Rをround独立に混ぜられない | Aが共通κ1変更の対照scheduleを提案。全raw/normalization/NとR依存価格が必要。event削減のC利得とattenuation/biasによるN増を比較する |
| finite K | 現PR全roundはK2。Full-H K2/K4案のscreenは未完 | 必要ならKの1対照を他条件固定で供給。新K law/signed grammarの価格資格も必要。K変更のbias/normalizer/回路長の全G競合を判断 |
| h/M/energy budget | 現taskの固定条件。今回のscopeでは探索変数でない | 正式に共通比較menuへ追加する判断がなければ延期。固定13roundの不足を先に解決 |

初期PR menuは、現構成をanchorに**一度に1設計自由度を変える対照を1本だけ**Aが提案する形を推奨する。どの対照にするかはP0〜P4とA/C/Dの適用domainを見てユーザーが判断する。L_D×幅×κ×Kの直積やTrack A winnerの自動採択を行わない。

## 3. 次の科学計算の最小化・担当

| 順序・種類 | 担当 | 最小scope案・それで変わる判断 |
|---|---|---|
| 先に本有限再解析 | ユーザー→B | P0〜P4だけを承認。新取得を始める前に、固定proxyの長round重みと必要N/compiled補正の境界が分かる |
| N資格の研究判断・技術 | ユーザー＋A/B | 全事象loss/target/risk claim・誤差移送・MC/解析法を決める。経験的conditionalだけで十分Nを出せない問題を解く。新grant発行は不要 |
| PF数値安定性の技術調査 | A | 失敗観測の保存と元PF macro-step/powerをSOURCE・人工入力で確認。必要な実診断は代表1失敗セルを別許可で行い、全20再実行を避ける |
| 固定候補のwhole-N評価 | B/A | 別採択した1候補×最大3 whole allocation等の有限案。新MC/解析riskのSOURCE/精度/反復数を固定して許可。N配分と実精度達成を判断できるが、価格欠測ならwinnerは出ない |
| 必要価格domainの供給 | D、必要feature/lawはC | 適格N又は根拠付き許容集合が得られた後、比較を変えるround/domainを選ぶ。m12使用ならその最大domainの妥当性が要る。pilotのm0/m6だけでは認定しない |
| 少数PR対照 | A＋B＋C/D | 初期対照1本の全signal→同risk N→同profile価格。どの自由度が総Gを改善するかを初めて比較。現L_D1から他prefix/Kへ推定しない |

これらの後続runは本P0〜P4承認に含まれない。A/C/Dの現在の作業へ新しい実行指示を送っていない。

## 4. 価格精度を上げる条件と延期できるもの

比較候補に同RMSEの適格Nがあり、全使用domainの価格区間と共同confidenceが供給された場合、`Σ N b`又は同一価格の`Σ(N_left−N_right) C`区間が結論を跨ぐところだけ精密化を依頼する。未知Nの間はセル別誤差又は妥当なN集合に条件付け、nominalを正式なcompile配分重みにしない。

Dの暫定2%/95%は正式採択しない。model/解析式のcalibrationと独立holdoutを分け、最大使用round＋必要な内部challengeは既存B価格検証計画に戻る。数点holdout通過を全13認定とせず、必要domainの解析的囲い/移送根拠/直接CIのいずれかを求める。精度を上げてもlaw/rare/systematicが未資格なら最終Gは依然不足する。

延期可能：全73価格receiptの再監査、m7/8/9の逐次compile、全prefix/K/R格子、新8次PF、目的を変えない環境完全一致の再検査、既取得PR全13signal/構造kernelの再実行、費用差と無関係な一律の価格高精度化。延期の条件は「未取得を0扱いしない」ことで、必要domainの証拠を省略してwinnerを作ることではない。

## 5. 今回の停止と承認事項

準備文書・metadata確認をcommit/pushしremote SHA/tree/主要成果物を照合後STOPする。**ユーザーへは§1のP0〜P4を対象とする有限保存再解析だけの実行承認を求める。** 元科学値の新しい比率/寄与計算はその返答後だけ。新signal/MC/compile/fit/新PR最適化や、科学条件の採択へ自動進行しない。
