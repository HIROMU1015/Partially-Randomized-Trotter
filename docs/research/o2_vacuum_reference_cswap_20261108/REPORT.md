# O2 真空参照CSWAP：技術調査・人工検証報告

2026-11-08。KNOWN_SE_QPE_APPLICATION ＋ PR_SPECIFIC_APPLICATION_POTENTIAL（人工条件限定）。pathwise信号保存は成立し、独立した新機構は未確認。実PRでの総資源改善はUNKNOWN。

- 単回ではL/R非可換、任意psiでも、両因子が真空を既知位相で保ちP(ell+r)で補正すれば元の複素Hadamard信号とX/Yの±1分布を保存する。Qを変えなければ有限targetも保存。各ランダムUの固有stateは不要。
- 既知SEと同じsplit・位相・routing・fanoutを認めると同じ回路になり、候補固有の増分利益は0。既存O5の未解決だった単回pathwise適用を技術的に接続した。主要一次文献は[Rowe et al. v2](https://arxiv.org/pdf/2604.14921v2)。
- 人工12＋関連2件=14 PASS、compile比較112＋fusion追加16=128行PASS。最適化済みdiagonal-only controlledを対照とし、追加で強いshared-basis融合/可換角度合成にも比較した。全体位相のfitなし。失敗した技術attemptも保全。
- changing-basisが長い人工2-qubit/12blockの全結合Xでは、RZ283→228、CX144→92、depth387→173、qubits3→5（reference reset2込み）。短いpathや強く融合する共通basisでは負ける。共通basis12blockのXはRZ43→86、CX22→44、depth54→75。実H4で同じ利益が出るとは主張しない。
- coherent QPE全体は別条件。共通固有基底の十分条件と、U/U²・厳密固有入力でも非可換splitで最終分布が変わる人工反例を保存。fresh-state RPEには各shotのlaw/独立性と新誤差資格が必要。
- 元PR gateのRZ/RZZ/XXPlusYY・scalar/event/identity・inverseの位相はgate列から取得する。Hの真空energyによる代用は不可。現prototypeは人工1–3 qubits限定で、実保存payloadのbinderではない。
- 追加n referenceと基本2n CSWAP、相対P、reset/readout、routingを含めた人工価格を保存した。cat版は3n qubits。T・物理latency・校正error・実PRのオンライン古典費はUNKNOWN。古いexact engineの認証値を新gadgetに流用しない。

[数理・先行法・SOURCE対応](VACUUM_REFERENCE_THEORY_AND_PRIOR_ART.md)、[費用と全人工表](VACUUM_REFERENCE_COST_COMPARISON.md)、[次の最小科学scope草案](NEXT_SCIENCE_SCOPE_DRAFT.md)。独立新規性や科学的GOは採択していない。

## 再現・保全

新moduleは src/trottertracks/algorithm_codesign/vacuum_reference_cswap.py、runnerは scripts/tracks/algorithm_codesign/run_vacuum_cswap_artificial.py と run_vacuum_cswap_fusion_check.py。SOURCEのcontent hashで実行code・既存helper・環境を固定。人工入力はtest fixtureとARTIFICIAL_INPUTS、全compile profileとcache/時間はBENCHMARK_RESULTSへ保存した。

[artifact正本](../../../artifacts/o2_vacuum_reference_cswap/2026-11-08/SUMMARY.json)、[実行履歴](../../../artifacts/o2_vacuum_reference_cswap/2026-11-08/EXECUTION_HISTORY.json)。最終run05のcorrectnessは約0.23秒、compile約30.6秒、追加fusion約0.8秒、peak RSS約594 MiB。実H4・保存結果再解析・RNG sampling・shotは0。旧科学SOURCE/engine/結果/STOP/grant/ledger/markerは変更しない。元mainと既存dirty/untracked、他worktree/process/cache、C1凍結を維持する。

## 次の判断・mandatory STOP

少数の事前固定実pathで費用資格を確認する価値が残る。新adapter、全scalar/phaseと選択policy、数値誤差方式、固定path/compile profile・call上限を先に確定し、その有限科学実行への明示許可を得る。実state/Bornやshotを要しない段階案から始める。重要な新規性/全RPE優位の認定は本技術作業に含めず、必要なら別途開始許可のある科学レビューへ渡す。

専用branch codex/o2-vacuum-cswap-20261108 で公開し、remote SHA、本文、blob/tree、SOURCE hashを照合する。技術作業と公開照合の完了後mandatory STOP。実H4、全round、M/D/EやC1、別方式へ自動進行しない。
