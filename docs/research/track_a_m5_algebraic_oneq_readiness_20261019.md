# H6 m5：回路縮減を残す代数的1qubit最適化・実行準備

2026-10-19。**小角度を閾値で捨てず、Euler再合成を使わない縮減方式を実装し、人工検証とH6専用実行準備まで完了した。実H6での縮減量とraw精度は未検証。** 利用者の「続きを行って」を、前報に示した修正設計・技術準備への承認として扱った。新しいH6 compile・作用計算は起動していない。有効grant0、local commit後mandatory STOP、push0。

H6用SOURCE：`0d47f59c351798434f7f29cdcbda993f60b5462c`。先行prototype/人工14条件のSOURCE：`29f5d5f00f3075e60fad17000d8dc76651d845b4`。branch：`track-a-m5-algebraic-oneq-design-20261019`。

[前報のH6改善](track_a_h6_m5_preserve_oneq_run01_results_20261019.md)、[今回summary](../../artifacts/resource_applicability/track_a_m5_algebraic_oneq_readiness/2026-10-19/summary.json)、[人工14条件](../../artifacts/resource_applicability/track_a_m5_algebraic_oneq_readiness/2026-10-19/artificial_results.json)、[未承認実行案](../../artifacts/resource_applicability/track_a_m5_algebraic_oneq_readiness/2026-10-19/plan_DRAFT.json)。

## 1. 今回の修正

前報は1qubit再合成全体を無効にした対照で、H6の保存状態についてcompiled vs変換後の差0、元wrapperとの差2.326360070803798e-11だった。一方、RZは診断値2,855,001であり、回路縮減を残す別方式を用意する必要があった。前報のPASS・旧STOP・費用未受理を変更せず、新SOURCEに分離した。

新方式では次だけを行う。

- 隣接RZのbinary64角度を、値を変えずに有理数へ変換して正確に加算する。
- 合成角がfloatで正確に表せる場合に一つのRZへ縮める。H6案の追加丸め予算は**0**であり、正確に表せない場合は元のRZ列を保持する。
- `SX²=X`、`SX⁴=I`、`X²=I`で隣接SX/Xを整理する。これらの恒等式には追加scalar phaseがない。
- RZを消すのは正確な角度和が0の場合だけとする。非ZEROの小角度を閾値で消さず、2π wrapも行わない。RZ(2π)をIと誤認しない。
- RZ数・総gate数を増やさず、実際に短くなる置換だけを採用する。global phaseのbit列を各passで保持する。

policyは`M5_OPT1_ALGEBRAIC_ONEQ_ZERO_ROUNDING_V1`。既存presetの1qubit pass一箇所を代数的passへ置き換え、他のopt1 taskとflow条件は維持する。共有Qiskit・SymEngineは変更しない。係数・PF公式・control・保存状態・reference・数値allowanceを変更する案ではない。

pure plannerと正確な台帳は検証済みprototypeから再利用した。H6用のrun bodyは、13qubit/terminal measurementを扱うscopeと丸め0 guardを明示した別SOURCEである。prototypeの3qubit以下・人工専用guardは保全した。

## 2. 丸め台帳の意味と限界

一般のprototypeは、各合成角をfloatへ戻す際の差`δ=|正確な角度和−出力floatの正確な値|`を有理数で記録し、合計が明示予算を超える置換を行わない。台帳と予算の比較も有理数で行う。予算に依存する判断をcacheせず、現在の残予算で毎回判定する。

標準RZを数学的に解釈する場合、置換による局所operator norm差は`δ/2`以下である。厳密なSX/X恒等式とunitaryの積を用いる範囲では、これらの変更の理想回路間差は丸め量合計の半分以下として扱える。**これは理想gateに対する代数的丸めの評価であり、Statevector等の浮動小数演算、他のcompiler pass、PF bias、shot統計、PR母平均の保証を含まない。**

今回のH6用契約では予算0をコード・scope・grant案で固定し、正の予算の混入を拒否する。理想gateの追加角度丸めが0でも、gate数と演算順序が変わるため、実H6のraw差0や全体PASSを保証しない。元のfiniteかつ厳密に1e-9未満のraw検査を維持する。

人工実験の`1e-11`は、台帳・fallbackの動作を見るための比較設定だけである。H6案には含めず、主会計のbias allowance・共通プロトコル・正式科学条件として採択していない。今回の推奨案は、追加角度丸めを導入しない予算0である。

## 3. 人工検証

最終のprototype19件とH6 adapter11件、計30 test methodがPASSした。正確なRZ和、SX/X恒等式、相殺後の微小角度保持、2πの制御付き位相、予算の累積と拒否、overflow時の保持、cache後の予算再判定、custom gate lookalike拒否、presetの他taskの実行、one-shot・raw不一致・worker回収を検査した。13qubitの経路テストも人工の数gate回路と人工初期状態だけであり、分子の保存状態を使用していない。作者自身の検査で、独立AIレビューではない。

さらに7人工回路×2設定（0と1e-11）の14条件について、global phaseを含むraw全演算・制御付き全演算を比較した。代表値を以下に転記する。

| 人工例・予算0 | 入力 | 出力 | raw全演算Frobenius差 |
| --- | --- | --- | ---: |
| 正確なdyadic RZ合成 | RZ24/SX24/X12 | RZ1 | 7.043575467574348e-16 |
| floatで正確に表せない`.1+.2` | RZ2 | RZ2を保持 | 0 |
| 2^-40の小角度256 run | RZ256/barrier256 | 全256角度を保持 | 0 |
| entangled blocks | RZ32/SX16/CX16 | RZ24/X8/CX16 | 1.1064000509094923e-15 |

全14条件のraw比較が人工テストの1e-12未満を満たした。この1e-12は人工fixtureの検査値であり、H6の受入基準を変更するものではない。人工回路での縮減を、実H6での縮減率や費用利益へ外挿しない。

14条件は2設定を独立した2processで並列実行し、外部wall約1.38秒、native exit0、全worker回収。高速化は、同じ局所runの記号的token化をcacheし、不要な行列生成・Euler抽出・位相計算を避ける方法で行った。実H6の実行時間は未測定である。

## 4. H6専用の未承認one-shot契約

sealed private plan SHA256：`bce9983698eec25ede2a43730334189d0cf3dccbdcbbac5da032beaab0cf01ce`。公開の[plan DRAFT](../../artifacts/resource_applicability/track_a_m5_algebraic_oneq_readiness/2026-10-19/plan_DRAFT.json)は説明用projectionで、private CLI入力そのものではない。

対象はH6直鎖・隣接1 Å/STO-3G、保存tol-only1e-8・observed DF19、full `PF_M5_BEST_20D64C2_BINARY64_V1` p19/q2、T=.8/P=0、symmetric-directional/full Hadamard cosine、保存low状態一つ。Qiskit1.3.0・rz/sx/x/cx・preset opt1・seed17を用い、新policyを明示する。

SOURCE・入力・compiler/runtime計2,016fileをsealした。準備metadata20項目がPASS、SOURCE・入力hash・compiler seed・正予算・raw閾値・実行範囲等の改変11種を拒否した。無効grantを実際のCLIへ渡し、入力復元前にexit2となることを確認した。有効grantの発行・旧grantの消費・新run root/markerの作成は0。

実行承認後の範囲は以下に限定する案である。

1. 保存lowering QPY/vector/初回PASSをSHA結合して再利用する。前回のno-resynthesis対照PASSの監査も固定証拠へ結合する。
2. cosine一軸を新compile一回・新作用一回で比較する。重複lowering/変換後作用は0。
3. raw3検査（最初は既存PASSのreuse）を順序固定で行い、各finiteかつ厳密に1e-9未満を要求する。不一致なら後続検査を抑止しSTOPする。
4. 作用計算前に全passの追加角度丸め0、global phase bit不変、Euler/小角度閾値/wrapなしを検査する。
5. QPY/vector/JSONとSHA、外部native exit・worker回収・保全を記録し、成功・失敗にかかわらずmandatory STOPする。

新SOURCE・入力・承認・資源・未使用出力のfresh gateを起動直前に再確認する。測定済み旧peakを用いたadmission proxyは約10.29GiBで、runtime capや必要メモリの厳密上界ではない。CPU・RAM・diskの現在値に応じて判断する。1compile→1作用の依存関係のためproductive worker一つと独立監視を使用する。上位PSI、一時競合、無関係worktree更新だけで停止しない。実OOM、継続不能、自己process異常、SOURCE/input破損、数値不一致への保護は維持する。

新しいsampling・signal・SCF/DF/state/reference生成、sine、追加候補、fit、費用再会計、費用受理、自動retryは範囲外である。局所計画の「元gate列を保持するfallback」は固定したcompiler規則であり、科学jobの再実行ではない。

## 5. 保全・採用範囲・次の担当

旧2,720fileを前後SHA照合した。SOURCE68ee03a7のH6保存状態PASS、SOURCE96cd7cd4等の旧STOP、58受理軸、raw結果・grant・ledger・marker、V1/V2、他worktree・共有環境を保全した。準備中のQPY/NPZはbytes/SHAだけを読み、配列・回路を復元していない。実H6の新数値解析・compile・作用・会計は0である。

新policyは標準opt1や旧modelの費用と自動互換にはしない。m5_best主比較のAD03、共通実験プロトコル、他チャットの条件を変更しない。実H6での縮減・raw精度・時間が分かるのは、上記の別実行後である。

**ユーザー**：SOURCE `0d47f59c…`・丸め0・同じcosine一軸の新compile/作用を一回実行するか判断する。前回の一回実行契約は完了・STOP済みであり、今回の準備を新科学計算の承認とは扱っていない。

**Codex（別途実行承認後）**：新しい承認receipt・grantとfresh gateを結合し、固定一軸だけを実行して終了監査する。今回保存した未承認plan・無効grant・旧STOPは上書きしない。追加軸・主比較採用・費用会計へ自動で進まない。

local commit後**MANDATORY STOP**。公開pushなし。
