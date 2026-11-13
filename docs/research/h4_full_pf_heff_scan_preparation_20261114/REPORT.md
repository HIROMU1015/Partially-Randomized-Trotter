# Full PF公式・幅の有限探索準備

2026-11-14 JST。[有限科学計画](SCIENCE_PLAN.md)／[PLAN.json](PLAN.json)／[全round schedule](ROUND_SCHEDULE.csv)／[C HANDOFF](HANDOFF.md)。**実H4計算・保存科学数値の再解析・価格予測は0。科学実行は別途承認待ち。** Bの統計結果を待つ必要のない、PF物理誤差側の準備である。

## 初回menuと根拠

Full PFの既存3公式 **S2（2nd）、標準Yoshida4（4th）、M5 best（4th）** と、要求macro-step幅 **Δ=.05/.1/.2/.4/.8** の全15組を提案する。新しい係数・PF公式・8次PFを追加しない。M5の名称は5次精度を意味しない。

根拠はD固定`e7dca5f7301c8633b17a59b6a7161331303aef2d`の保存Full PF catalogで、H4/T=.8の全3公式にqPF=1/2/4/8/16、actual幅=.8/.4/.2/.1/.05が存在すること。旧B51500の六案（3公式×.1/.4）をseedとして、その同じ保存格子の.05/.2/.8を含めた。安価なH_eff診断で細かい側の精度対照・中間幅・粗い側の費用候補を残し、高次公式を過度に細かい幅へ固定しないためである。旧Track A/PF commonの最良順位を採択していない。

これは有限格子内の初回比較であり、幅の全域最適・最大許容幅・最適公式は未確定。範囲外の幅、係数fit、追加補間、B0/B0I、PR、8次PFは含めない。後で格子端又は予算境界が結論を左右すれば、必要な追加点だけを別の有限案として提示する。

## 共通条件・early round

Linear H4/隣接1.00 Å/STO-3G、保存tol-only 1e−8 DF10、weighted-Hermitian completion、sector(2,2)/dimension36、one-body＋全DF10（prefix10）、同H/state/snapshot・元full DF-H target・scalar。h=.1/M12/P0、prior/decoder/RMSE budget/compiler/profileは元条件を保持する。κ/R/KはFull PFではnull。

既存B/D APIで`qPF=max(1,ceil(T/Δrequest))`、`Δactual=T/qPF`を生成し、T・要求幅は既存binary64値のexact有理数を使用する。新canonical ID規則を実装していない。S2/.1のconfiguration IDも元`f2d58f320ed0ebbeff8e6d4c0fbd5de14d6338b493c92015c966935201fd93ef`と完全一致した。

| 要求幅 | m0…early roundのactual幅 | m12 qPF | m12のS2／Y4／M5 signed S2 piece数 |
|---:|---|---:|---|
| .05 | 全round .05（m0 qPF=2） | 8192 | 8192／24576／90112 |
| .1 | 全round .1 | 4096 | 4096／12288／45056 |
| .2 | m0=.1、m≥1=.2 | 2048 | 2048／6144／22528 |
| .4 | m0=.1、m1=.2、m≥2=.4 | 1024 | 1024／3072／11264 |
| .8 | m0=.1、m1=.2、m2=.4、m≥3=.8 | 512 | 512／1536／5632 |

上のpiece数は構造上の作用回数で、compiled RZ、回路価格、Gではない。負時間・融合・相殺・制御wrapperにより価格は異なる。Y4は3、M5は11の既存signed pieceをそのまま使い、Δはmacro幅であって最大signed substep幅ではない。`q_RPE=2^m`と`qPF`は別欄に保存する。

15 schedule×13round=195行、異なるphysical round-candidateは177件。一方、必要なH_effはformula×actual幅の**15種類だけ**で、early例外もこの中に含まれる。粗い要求幅の全roundをm12のH_eff一個に置き換えない。

## 既存点・元固有系の再利用

A `69ee0ff8898960d5f9959af8d34d89f223caec87`のS2/Δ.1診断は、同じ保存H4 DF10/sector(2,2)/Full PF/prefix10の既存点として参照する。保存されたminimum shift **+0.00013371206319456874 Ha**や二時点診断を新たに計算・再資格していない。この点のPF step/H_effを新計算しない。

同成果の`run01/EIGENSYSTEM.json:reference_comparison`には元full DF-Hの全energy/固有basisが保存されている。これを使う将来の14新診断で同じ元Hを14回対角化しないよう、既存`pf_effective_hamiltonian.evaluate`にoptional `reference_eigensystem`を追加した。元H行列を必須とし、SOURCE/Hamiltonian binding、完全で整列した固有系、Hermitian residual/Gram/state-weight gateを維持する。cache内の旧state weightsは使わず、供給された同stateから計算する。cache不成立時にeighへfallbackしない。

従来の`reference_hamiltonian`だけの呼出しは従来どおりeighを行う。既存2f0/9af1/3a26/69eeのSOURCE・raw・結果・STOPは変更していない。新しいscan callerは既存C step builderとこの評価器を接続する範囲に限り、S2一候補専用の旧runnerを無修正で総走査として実行しない。

## 既定予算と候補を絞る方法

元SOURCEの目標εE=0.00159360010199040 Ha、εqpe=εtrot=0.00079680005099520 Ha、Minkowski契約を保持する。原PF/S2の古い係数Cmaxや旧HP型α/βをH_eff shiftの認定式へ流用しない。B metadata上の旧RMSE policy欄はidentity維持のため保持しているが、新しい十分Nの資格ではない。

承認後は各actual幅の対応energy shiftの符号と絶対値、分枝/残差、元最低部分空間とのoverlap、保存stateの全spectral支持を揃え、εtrotとの**局所spectral screen**として比較する。budget内というだけで全RPE RMSEを保証しない。対応不明や分解不能はUNKNOWNとして残し、最小値同士を無条件に対応させない。実H4での次数則・単調性を仮定せず、格子外へ外挿して最大幅を認定しない。

その後に、actual幅の全round coverageと同条件のC証拠を用いて各公式の有望な幅をshortlistにする。Nを共通又は不変とは仮定せず、piece数又はm12価格だけで勝者を決めない。今回の段階では候補の精度PASS、C予測、N、G、winnerは全て未取得/null。

## 価格証拠への接続

C固定`3ae2e605dd6efcc63ed3742fe3076a8be3e6afae`のcoverage方針と、D e7dcaの正本136候補/73直接receiptを参照した。C文書のA状況欄は当時の2f0時点であり、今回参照する69eeの実行済み点とは区別する。旧六案から今回の15組へ丸めず、同physical ID・完全candidate metadataで照合している。195×XY=390の参照欄には**EXACT_METADATA_MATCH 38軸、catalog内MISSING 18軸、照合catalog外334軸**がある。これはmetadata参照件数で、390個の新回路・独立標本・科学callではない。

既存価格の値・fit・平均を再計算せず、受入を再認定していない。旧native loweringと新ZZ loweringは別domainで、compiler hash一致だけで接続しない。Cの未資格model/予測は呼ばず、長q欠測をq比例又はPRのE[S]で埋めない。full measured controlled X/Y wrapperのcompiled RZ、P=0の単位を維持する。

## 技術検証・実行範囲

cache3人工＋metadata3＋既存19回帰、計**25 PASS**。実入力配列をloadしないmetadata standaloneではNumPy/SciPy/Qiskit/PySCFもimportされないことを確認した。[CACHE_TESTS](CACHE_TESTS.txt)／[PLAN_TESTS](PLAN_TESTS.txt)／[REGRESSION](REGRESSION.txt)。独立科学レビューではない。

承認対象の新科学範囲は**14 PF step＋14 Schur、元Hの新eigh0、直接PF mean最大28＋全spectral mean最大28、新reference0、旧保存signal再解析0**。保存元固有系のSOURCE付き再利用と数値residual確認をこの範囲に含める。N・MC・新compiled価格・正式G・自動追加探索は全て0。詳細は[SCIENCE_PLAN](SCIENCE_PLAN.md)。

**ユーザー**がこの有限範囲を明示承認した後に**Codex A**が実行し、結果を**C**の同candidate価格接続へ渡す。現時点では技術成果公開・remote照合後mandatory STOP。
