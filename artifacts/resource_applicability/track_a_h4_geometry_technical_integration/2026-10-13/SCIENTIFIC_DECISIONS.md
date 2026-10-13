# GPTへ返す科学的判断・未実行の最小実測案

技術調査と保存値解析は完成。以下の採否はCodex側で決めていない。

1. B0+Iを別対照として正式登録するか。元B0は定義通りであり上書きしない。
2. 主要claimを元登録集合内の点推定比較に限定して提示するか。一般的deterministic手法全般、母平均winner、energy精度、系サイズscalingの主張は現証拠から保証しない。
3. B0+I costの実測範囲をどう選ぶか。共通Δは仮定で、候補・axis・geometry別Δが実際には異なり得る。
4. H4に独立確認または希少次数条件付き診断を追加すべきか。B2内近接順位、多重selection、未保存event、未観測成分の母平均寄与は別論点。
5. H4 geometry、native固定anchor、H6、H8を異なる条件のcase studyとして並記する方針と、v0.3の表現を採択するか。

## B0+Iの段階的実測計画（全てproposal、今回実行0）

|案|構成|full-wrapper compile|分かること|残る不足|
|---|---|---:|---|---|
|代表点最小|.70 p5/q1、.80 p5/q1、1.40 p3/q1、各cos/sin|6|epsilon=.05の固定3構成の実Δと費用|397点全体、他補助候補、axis差の共通性を保証しない|
|zero-Δ逆転領域の候補|.70 p4/q1とp5/q1、.80 p5/q1、.90 p5/q1、1.40 p3/q1、各cos/sin|10|397点でzero-Δ最小だった5構成の固定候補費用|実Δで別候補が補助最小に変わり得るので全補助集合winnerは未確定|
|補助対照の全登録q grid|6geometry×20 B0構成×2axis|240|同じ登録domainに対応する補助120構成の費用|研究採択・新source・compiler/profile・数値signal bindingが別途必要；domain外deterministic手法は比較しない|

これは新対照を測る案で、旧B0再compileを含まない最小call数。旧B0 metricは固定参照として保持する。
既存sourceのpreparationを別IDでscalar保持に変更し、旧B0をbyte-identicalに保存する。
phaseの位置・canonical bytes・cos/sin control/action・B=1・signal位相式を、正式実装時に限定人工検査する。
新SOURCEによるFP/seed/carry bindingと旧source結果との隔離を明示する。
compiler options、入力、state/科学条件を変えず、旧grant/cache・failed-call返却・予算resetを使わない。
各wrapperのΔ_c/Δ_sを別保存し、epsilon gridへ再算術投影する。全Δ共通を測定結果へ強制しない。

## 必要な情報・費用の未確定部分

- 新science/source/対照定義の採否と利用者の別認可。今回new grant=false、runner=false。
- 対象input byte identity、compiler/environment profile、old/new numeric FP、final RZ/CX/depth/size、wall、peak RSS/AS、保存bytes、native exit/cleanup。
- 新trajectory・sampling・SCF/DF/state生成は不要という最小案。ただしmodified wrapper build/実compileは新科学処理で、別認可が必要。
- actual compile追加数は6/10/240（予測cache節約0）。旧carry/charge/wallを保持し、新campaign費用の会計を事前固定する。
- 提案は単一worker・内部thread1。AS/RSS capは別認可で設定し、既存32GiB枠を上回らない案からfresh admissionを評価する。CPU番号・affinity・observer roleをこの資料で認可しない。
- wall/保存charge/ピークRSSは**未測定**。pre-transpile q phase数だけから厳密cost予算へ換算できないので数値を捏造しない。6件の段階的capと容量、failure時STOP・自動追加なしを実行planで具体化する必要がある。
- 6件から全240件へ外挿した費用・72h完了を保証しない。実測後に必要な残件を科学判断する。

H6の0.00734%補正やH8 preflightを、このH4対照のcost/probability予測に使わない。
H8への変更・問い合わせ・新launch0。成果報告後mandatory STOP。
