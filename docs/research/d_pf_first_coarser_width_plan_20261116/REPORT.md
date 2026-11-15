# 最初に調べる粗幅：B0I p7／標準Yoshida4／32h/5

**推奨候補1件：B0I p7／標準Yoshida4（元formula `4th`）／PF要求幅32h/5、元configuration `dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0`。最初の取得は不足するm6〜m11の6 complex meanだけとする。保存7roundと結合後、現幅Yoshida4の保存B_PRICE配分で最大1 Q3を比較する有限案を推奨する。**

**その結果で、同じ配分のまま粗幅を精度候補として残し、その幅固有の価格検証へ進めるかを決める。** 現段階では費用低下も、shot増での救済可能性も未判定。最初の案は新compile 0件、価格表の完成を待たない。今回は選定・metadata・計画のみで、下記の科学作業を実行する許可ではない。

対象は保存linear H4 / 1.00 Å / STO-3G / tol-only DF10、h_RPE=.1 / m0〜m12 / P0除外。変更するのはPF要求幅。Full PFは一体項＋fragment0..9、B0I p7は一体項＋fragment0..6で、同じq列でも異なる物理候補である。

## 4候補の差分比較

完成済みB `7f27e18f…/WIDTH_SIGNAL_COVERAGE.json` とD `486fa84c…/PRICE_COVERAGE.json` の該当4件を再利用した。12候補・312軸の再棚卸しや元receiptの再監査は行っていない。機械可読の固定locatorとschedule差分は [CANDIDATES.json](CANDIDATES.json)、推奨候補の取得先は [FIRST_FINITE_PLAN.json](FIRST_FINITE_PLAN.json)。

| 方式・公式・PF要求幅（元ID先頭） | 保存信号round | 不足信号 | 同物理点の直接価格証拠 | 不足価格 | 同方式・同公式の現幅B_PRICE適合費用 |
|---|---|---|---|---|---:|
| Full PF/S2・8h/3 (`7754671b…`) | m0〜m3,m12：5 | m4〜m11：8 | m0〜m2：6軸 | m3〜m12：20軸 | 615,304,502 RZ |
| B0I p7/S2・8h/3 (`5576418d…`) | 同上：5 | 同上：8 | 同上：6軸 | 同上：20軸 | 435,481,090 RZ |
| Full PF/Y4・32h/5 (`9e930a47…`) | m0〜m5,m12：7 | m6〜m11：6 | m0〜m4：10軸 | m5〜m12：16軸 | 619,717,524 RZ |
| **B0I p7/Y4・32h/5 (`dd8bfc24…`)** | **同上：7** | **同上：6** | **同上：10軸** | **同上：16軸** | **435,641,568 RZ** |

価格軸はすべてX/Yを別々に数えた。価格quantityは測定付きfull controlled Hadamard wrapper1 shotのcompiled全RZ数。元のfragment順、scalar一回、symmetric_directional、Qiskit1.3.0 / rz,sx,x,cx / opt1 / seed17のdomainに限る。表の「証拠」は旧coverageで同物理点と対応付けられた元trial/receiptの所在であり、今回新configurationへのC guardを実行した数ではない。消費側bindingは未完了。4候補ともsource-boundな完成13round packetは未取得、粗幅固有の採択価格規則もない。

**後続差分**：p7/S2の細幅hは直接10軸＋比較用外挿16軸へ進み、固定配分モデル費用が公開された。その粗幅予算内の別2配分はB `3b5cd49c…` でQ3不適合だった。いずれも今回の4粗幅の追加mean/価格ではなく、上表のcoverageを変えない。B `9992c594…` は現幅p7/S2・16h/7の最大8 Q3局所shot探索の**未実行計画**で、粗幅8h/3や32h/5の実行ではない。435,481,090 RZという保存基準は変更されていない。

## 実際に変わる物理round

以下のq・幅は保存scheduleのmetadataを転記・対照したもの。新しい価格や科学値を計算していない。各公式のq/幅列はFullとp7で共通でも、head・scalar・physical ID・価格は別である。

- **S2：16h/7→8h/3**。m0〜m2はq=[1,1,2]、actual幅=[h,2h,2h]で同一。m3から変わり、q4・actual2h→q3・actual8h/3。m4〜m12はq=[7,14,28,56,112,224,448,896,1792]→[6,12,24,48,96,192,384,768,1536]、actual16h/7→8h/3となる。粗幅固有m3のsignalはあるが、その直接価格はない。
- **標準Y4：16h/3→32h/5**。m0〜m4はq=[1,1,1,2,3]、actual幅=[h,2h,4h,4h,16h/3]で同一。m5〜m12がq=[6,12,24,48,96,192,384,768]→[5,10,20,40,80,160,320,640]、actual16h/3→32h/5となる。新幅の最初m5と最長m12のsignalは既にある。

特にY4の保存10価格軸は、**すべてactual幅がまだ32h/5になっていないearly round**である。現幅の同一wrapperを再利用できる根拠にはなるが、粗幅32h/5の本体係数・長q価格を検証する点にはならない。qが少なくなったことだけでcompiled単価低下を認定しない。

## p7/Y4を最初に選ぶ理由と二つの比較軸

4候補すべてに同方式・同公式の現幅Q3適合点がある。Y4はS2より不足信号が少なく、新幅に固有な最初と最後のsignal、保存step演算子が揃っている。したがって6roundの補完で、全価格取得に先立つ直接的な精度比較へ進める。

その中でp7を選ぶのは、現在のp7最安適合点435,481,090 RZと、p7/Y4の435,641,568 RZが近接し、同公式の幅変更がp7の費用候補を更新するかという判断へ直結するためである。Full/Y4も同じ6meanで調べられる有力な次候補だが、最初から両方式を取得しない。Full/S2・p7/S2は今回より多い8meanを必要とする対照候補として保持する。取得不足の少なさと問いへの接続で優先したのであり、Q3の結果・未取得費用を予測して順位を付けたものではない。

比較は分けて記録する。

1. **同公式内の幅効果**：p7/Y4・現幅16h/3のB_PRICE配分を固定し、粗幅32h/5で精度がどう変わるかをまず評価する。将来の費用challengeは435,641,568 RZ。
2. **p7の保存最安点への挑戦**：新幅の適合配分と固有価格が得られてから、p7/S2の435,481,090 RZを更新できるかを判定する。同公式基準より安くても、この基準未満でなければp7最安点の更新ではない。

旧T2で局所ε_trotを超えたことをQ3不適合の代わりに使わない。逆に粗幅でshotを増やせば必ず救済できるとも仮定しない。PFの系統的な位相差、alias/branch、全round decoderの挙動はshot増だけで消えるとは限らない。細幅hの2列不適合も、粗幅全体の成功を意味しない。

## 最初の有限案：信号を補完して、保存配分を1件比較

**提案上限：保存7 complex mean再利用、新規6 complex mean、13round packet組立1候補、Q3 whole-allocation最大1件。新しいreference算出・演算子build・回路build/compile・価格算出・fit・配分生成・retryは0。** これは後続科学承認の対象を具体化した案であり、今回実行していない。

取得roundはm6〜m11、q=[10,20,40,80,160,320]、T=[6.4,12.8,25.6,51.2,102.4,204.8]、actual幅32h/5。物理IDと保存入力はHANDOFF／JSONを参照。1 complex meanにX/Yの2成分を含み、12個の独立samplingやcompileへ数え替えない。

保存m0〜m4は現幅Y4と同一物理round、m5/m12は元T2の粗幅固有roundである。元role・SOURCE・scalar・gateを保持してpacketへ束縛し、新独立標本や新取得として数えない。m12のsignalがあってもm12のcompiled価格を持つことにはならない。

同幅stepは `5fb2c32e…:…/cell_p7_4th_32o5/OPERATORS.npz#PF_step_without_global_scalars` に保存済み。今回確認したのはlocatorの存在とJSON scopeだけで、NPZや固有系は開いていない。後続取得ではAの `SavedStepContext` / 元 `baseline_signal_numerics.evaluate` / 保存入力・reference reader・packet生成を再利用し、新step構築や新固有値計算を避ける。旧48mean callerの固定契約を直接書き換えず、6点selectorと保存7点の接続だけを人工検査し、新SOURCEと出力先を固定する。

比較に使う保存列は、**p7/Y4 B_PRICE：X/Yとも `[59,55,51,47,43,39,35,31,27,23,19,7,2]`**。元locatorは `b4296b110…/SUMMARY.json#rows[15].allocation`。現幅の保存Q3 MSE区間は `[2.418371981954278179229700e−6,2.418371981954278179229706e−6] Ha²`、適合。p7/S2の末尾13/4の列とは違う。同じ整数列だけを再利用し、元allocation IDを新幅の実行IDへ付け替えない。

新Q3では元Z1＋T1 total decoder、保存DF10 solver proxy、IID deterministic raw、全事象平方loss、exact τを維持する。保存表示τは `0.000002539561285063813282041692 Ha²`。現幅のPASSやbinary64閉包を新signalへ流用せず、元call内の閉包・質量・残余loss・保存gateを改めて適用する。

- **適合（上界≤元exact τ）**：この同配分の粗幅適合点を保存する。費用低下は未認定のまま、粗幅に固有な価格pilotを別途提案・承認する。
- **不適合（下界>τ）**：この配分での粗幅不適合を保存する。粗幅全体を棄却しない。shot増・別の保存配分（例えば元N0）の検討はBの次の有限案へ戻し、今回の枠で追加しない。
- **INCONCLUSIVE/UNKNOWN**：区別して保存、価格取得や配分変更へ自動進行しない。gate不成立・不整合では部分結果でSTOP。

## 価格取得は後段に限定する

価格の不足はm5〜m12の16軸。現幅係数、細幅hの5513q+307、piece数比例式を粗幅へ流用しない。もし信号/Q3結果から候補を残す判断となれば、次の別承認案として、**同じ実幅32h/5のm5(q5)・m6(q10)を開発点、m9(q80)を未使用の長q確認点、X/Y別で最大6 compile** を検討できる。保存early 10軸は再取得しない。

この少数点で粗幅固有の反復・端価格の仮説を検討するが、今ここで係数を決めず、q640までの価格誤差保証にも昇格しない。誤差の扱いと比較用規則の採択が必要で、全26軸の直接compileや厳密上界を初手の前提にしない。この後段案は上記6mean＋1 Q3の第一有限枠には含めない。

## 調査・検査・ロードマップ・STOP

[HANDOFF](HANDOFF.md) に固定SOURCEと§10.5の開始前差分照合を記載。関連D/B branch、Bの11/16更新、固定Aの元保存round/step locator、同ユーザーの関連callerを読取確認した範囲では、対象不足roundの後続取得・同等進行jobは未確認。未参照branch/host全体の不存在証明ではない。次の科学開始直前に再照合し、新しく受入可能な分は上限から減らす。

metadata検査：[CHECKS.txt](CHECKS.txt)。Fullのexact有理数幅とp7のh比文字列というschema差を正規化し、4候補のID・q/幅差分、保存／不足、推奨6点、元配分、全locatorの解決とNPZの存在を確認した。科学moduleはimportせず、元のcached Git/JSON readerだけを再利用。配分生成・新MSE/価格/費用算術は0。

統合担当への差分は「次の粗幅1件と6mean＋1 Q3の未実行有限案が具体化した」。新しい適合点・価格資格・最安点・十分N・G_qualified・winnerは増えていない。ロードマップ本体は編集しない。

原結果・固定SOURCE・raw・grant/ledger/marker/KEEP・旧STOPを保全。commit/push・remote SHA/tree/成果物照合後 **mandatory STOP**。追加科学実行には対象・SOURCE・入力・上限の明示承認が必要。
