# M1-S V1 γ依存設計：精度・resource根拠と引継ぎ

2026-10-17 JST。技術準備報告。新しいGPT詳細科学レビューではない。

## 変更と再利用

[改訂契約候補](m1s_v1_gamma_dependent_contract_candidate_20261017.md)が正本候補。旧V0のbounds/controlを**新namespaceへcopyした上で**必要部分を改訂した。新たなsin/cos/atan、境界のcase分岐、時限supervisorを接続した。旧V0 SOURCEは変更していない。SOURCE実装の完全な独立再現を主張しない。独立なmpmathを登録外scalar testsの数値比較にだけ用い、本番SOURCEには依存させない。

単一γ fieldを廃し、`0<gamma<pi/2`全域の条件記述へ変えた。caseは同bias・dominance・direct trade-off・ラベル入替trade-off・domainなし・区間未解像・証拠不足を区別する。全108方式行と48対照をplaceholder付きで保全する結果schemaへ改訂した。旧raw-H参照biasにモデル差を再加算しない。

## 算術精度の根拠と限界

π/log/sinの旧解析的区間設計を保持し、cos交代級数とatanのrange reductionを追加した。`atan x=pi/2-atan(1/x)`（x>1）、`atan x=pi/4-atan((1-x)/(1+x))`（1/2<x<=1）により、交代級数の引数を[0,1/2]へ抑える。πはMachin式。logは2の冪によるrange reductionと正項級数の剰余境界。新しい三角関数libm allowanceは導入しない。

有理数分母の増大を抑えるため、各戻り区間をdyadicへ**外向きに**丸める。採択予定のpに対し、quantum `h=2^-(p+8)`、級数剰余もhを基準とする。pを結果で増やす機構はない。

解析的には、入力domain内で区間幅はπ≤62h、cos≤3h、sin≤4h。0<a,c<=1なら、X区間幅≤8h、Y区間幅≤10h。根拠は交代級数の次項境界と外向きroundingによる最大2hの追加である。これは算術の設計上の幅であって、旧保存値のcoverageを認証しない。

X≈0、(X,Y)≈(0,0)では比／角度のconditioningが悪くなり、**任意の保存入力で境界角の幅が2^-pになるとは保証しない**。係数符号または領域の区間が分離できなければUNRESOLVED、rootが存在しても端に重なれば位置未解像。上限超過はTECHNICAL_STOP。入力尺度や結果に応じたp探索は行わない。

## 登録外テスト

`tests/tracks/algorithm_codesign/test_m1s_gamma_preparation.py`：34 tests PASS。π、sin、cos、log、atan、外向きrounding、精度の区間幅、rational/series/event caps、同bias/dominance/trade-offの直接・逆向き比較、境界区間、係数未解像、候補契約の拒否、schema、hard wall STOPを検査した。

mpmath 150 decimal digitsの独立実装との人工scalar照合は有限テストであり、全入力に対する形式検証ではない。実データへの式の代入や保存H4の再採点は0。Python audit hookで、テストから旧scientific artifactをopenする操作を拒否した。テスト用に生成したtoy値・toy境界をH4結果と混ぜない。

旧IEEE/BLAS/trig/log/underflow仮定は残る。新算術のtoy PASSをplatform certificate、H4のV1 correctness、物理測定の保証に格上げしない。旧v5の未保存full-H/HD直接一致はUNDETERMINEDのまま。

## 時間・メモリ候補の根拠

`benchmark_m1s_gamma_offdomain.py`は保存科学入力を開かず、72組sin/cos＋48組の人工trade-off比較＋π/logという固定scalar workloadを、p=32/64/128/256で各3回実施した。SOURCEは共有π/logとbiasのtrig cacheを用い、数値worker1／GPUなしとする。CPU32論理coreを確認したが、この小さなscalar workloadに並列数値workerは導入しない。新しい速度改善倍率を主張しない。

詳細は`offdomain_resource_probe.json`。最大toy workloadはp=128で約0.351 CPU秒、p=256で約5.078 CPU秒、peak RSSは11520 KiB。**登録H4のruntimeを測定した値ではない**。精度を上げるとFraction演算の費用が大きく変わり、step数だけの真のworst-case保証は得られない。

候補は、実測の最大`CPU/bound_step`に100万stepと明示的なengineering余裕4を掛け、10秒単位で切上げる。wall候補はCPU候補の2倍と同様のwall換算の大きい方を切上げる。係数4は操作予算の余裕であり、確率保証ではない。

|p|CPU候補（秒）|wall候補（秒）|位置付け|
|---|---:|---:|---|
|32|30|60|粗い区間の技術候補|
|64|50|100|より低費用な区間候補|
|128|160|320|技術的な開始候補。未採択|
|256|1290|2580|高精度・高費用候補|

表は`precision_resource_candidates.json`を正本とする。p=128は係数の絶対解像度と人工workload費用の両面から提示する**技術推薦**であり、科学的に必要十分な精度や正式値ではない。ユーザーがp・解像状態の扱い・時間候補を採択する必要がある。machineのp／CPU／wallは今回nullを維持した。

新runはLinuxの数値worker1とmetadata-only親supervisorを使う。親がwall期限でworkerを停止し、`supervisor_result.json`を保全する。CPU soft capとhard cap+1秒、workerのAS/FSIZE上限も適用する。[Python resource公式仕様](https://docs.python.org/3.11/library/resource.html)および[signal公式仕様](https://docs.python.org/3.11/library/signal.html)に基づくUnix機能である。旧「新規契約はtime capなし」の既定に対し、今回のユーザーによる有限候補準備の明示指示を適用した。既存契約のtime capは変更しない。

aggregate virtual memory2 GiB候補から親のVMを予約し、残りを子のRLIMIT_ASへ渡す。GPUなし、BLAS等thread1、入力1 MiB、result/log各16 MiB、rational65536 bits、各series4096項、bound step100万を維持する。親は科学演算をせず、子のcheckpointもwall監視内に含む。OS schedulingやfilesystem停止を含む実時間の数学的保証ではない。

初期108行・48対照placeholderを親で保存してからworkerを起動する。途中kill/OOM等では直前checkpointを維持し、supervisorのTECHNICAL_STOPがworkerの保存statusに優先する。自動retry、precision変更、timeout緩和をしない。今回supervisorの登録外停止テストはPASSだが、登録H4のrun・resource適合は未検証。

## 未決事項と次の担当

ユーザーが採択すべき事項は、α具体値、confidence scope（第一候補PER_METHOD_ROW）、fresh-IID/Hoeffding/等軸配分・円周位相意味論、保存条件付きcoverage、共通pと解像状態の扱い、CPU/wall・構造cap、連続十分因子と整数ceilの限界、descriptive-onlyでGOを出さない規則である。新たな一点γの値は不要で、γ gridも採択しない。

本実装のscopeはγ依存の条件・境界とK区間／記号的なshot式に限る。絶対Nやstrict整数shot削減率の数値化、native費用、Task Bへの拡張を要求する場合は、別のscope選択・契約・実施承認が必要。これは科学的判断として残し、無料のoracle知識や任意の新γ点を導入しない。

今回の改訂契約は未採択、実施未承認。新execution authorization/grant/markerは0。SOURCE固定→候補公開後にmandatory STOP。次はユーザーによる改訂契約の採択判断、その後さらに別途の明示的な科学実行承認。必要時のGPT判断も新しい詳細レビューを自動開始しない。

## 固定SOURCE・静的検査・公開証拠

新SOURCE S：`f2a5611ca9fc7461c64f2deb74a1f624ac8aa7c5`。機械候補は7つのnamespace module、runner、結果schemaの9 fileをcommit／SHA256で結合する。SOURCEと契約payloadは別commitで固定する。

静的監査145 PASS／0 FAIL、登録外tests34 PASS／0 FAIL。保全監査9 PASS／0 FAIL、既存7,628非index file、親1284 tracked file・index・HEAD、全205既存worktree HEADを保全した。これらはローカル技術証拠であり、独立CIや保存H4での新検証ではない。

証拠：`artifacts/track_b_m1s_v1_gamma_preparation/2026-10-17/v1/`の`static_validation.json`、`offdomain_tests.json`、`offdomain_resource_probe.json`、`precision_resource_candidates.json`、`preservation_audit.json`、`evidence_index.json`、`publication_receipt.json`。receiptはpayload公開のbytes／SHA256照合後に別commitへ追加する。

**次の担当はユーザー。改訂契約採択と、その後の別途科学実行承認までmandatory STOP。**
